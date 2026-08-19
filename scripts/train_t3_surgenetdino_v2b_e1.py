#!/usr/bin/env python3
"""Sealed fixed-epoch E1 trainer for SurgeNetDINO DINOv2-B.

This trainer never constructs a validation dataset and never opens a held-out
image or label. It records every scaler attempt and counts 25 *real*, finite,
non-zero optimizer updates per epoch. Candidate metrics are impossible here.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "baseline" / "task3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (  # noqa: E402
    LabeledDataset,
    build_fg_balanced_weights,
    discover_samples,
    discover_unlabeled_images,
)
from model_factory import get_loss_fn  # noqa: E402
from train_t3_bcp import EXCLUDED_FRAMES, make_pinned_split  # noqa: E402

import train_t3_dinov2_unimatch as legacy_train  # noqa: E402
import t3_surgenetdino_v2b as core  # noqa: E402
import t3_surgenetdino_v2b_e1 as contract  # noqa: E402


EPOCHS = 70
WARMUP_EPOCHS = 5
STEPS_PER_EPOCH = 25
MICRO_BATCH = 2
ACCUMULATION = 3
MAX_EXTRA_ATTEMPTS_PER_EPOCH = 100
SEED = 42
STRUCTURALLY_UNUSED_FULL = {
    "backbone.mask_token",
    "head.fuse4.residual_skip.conv1.weight",
    "head.fuse4.residual_skip.conv1.bias",
    "head.fuse4.residual_skip.conv2.weight",
    "head.fuse4.residual_skip.conv2.bias",
}
STRUCTURALLY_UNUSED_HEAD = {
    name for name in STRUCTURALLY_UNUSED_FULL if name.startswith("head.")
}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-video", choices=contract.FOLDS, required=True)
    parser.add_argument("--labeled-root", type=Path, required=True)
    parser.add_argument("--unlabeled-root", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, default=core.DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def seed_before_model() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def discover_sealed_inputs(args: argparse.Namespace):
    core.verify_artifact_file(args.weights_path)
    labeled = discover_samples(args.labeled_root)
    videos = sorted({sample.video_id for sample in labeled})
    if len(labeled) != 180 or len(videos) != 6:
        raise RuntimeError(f"Expected 180 labeled frames/6 videos, got {len(labeled)}/{len(videos)}")
    train, heldout, train_videos, heldout_videos = make_pinned_split(args.holdout_video)(labeled)
    if heldout_videos != [args.holdout_video] or args.holdout_video in train_videos:
        raise RuntimeError("Pinned split failed to isolate the held-out video")
    if any((sample.video_id, int(sample.frame_idx)) in EXCLUDED_FRAMES for sample in train + heldout):
        raise RuntimeError("Registered noisy-frame exclusion drift")
    # Keep only path strings for the sealed fold; never instantiate/read it.
    heldout_paths = [sample.image_path.resolve().as_posix() for sample in heldout]
    unlabeled = sorted(discover_unlabeled_images(args.unlabeled_root))
    unlabeled_dirs = {path.parent.resolve() for path in unlabeled}
    if len(unlabeled) != 1379 or len(unlabeled_dirs) != 46:
        raise RuntimeError(
            f"Expected 1,379 unlabeled frames/46 directories, got {len(unlabeled)}/{len(unlabeled_dirs)}"
        )
    if any(video in path.as_posix() for video in videos for path in unlabeled):
        raise RuntimeError("Labeled video identifier occurs in the official unlabeled pool")
    return train, train_videos, videos, heldout_paths, unlabeled


def grad_evidence(model: torch.nn.Module) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, parameters in (
        ("backbone", model.backbone.parameters()),
        ("head", model.head.parameters()),
    ):
        grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if grads:
            aggregate = torch.zeros((), dtype=torch.float64, device=grads[0].device)
            nonfinite = torch.zeros((), dtype=torch.int64, device=grads[0].device)
            for grad in grads:
                detached = grad.detach()
                nonfinite.add_((~torch.isfinite(detached)).sum())
                aggregate.add_(detached.double().square().sum())
            nonfinite_count = int(nonfinite.item())
            squared_norm_raw = float(aggregate.item())
        else:
            nonfinite_count, squared_norm_raw = 0, 0.0
        finite = nonfinite_count == 0 and math.isfinite(squared_norm_raw)
        result[name] = {
            "n_grad_tensors": len(grads),
            "finite": finite,
            "nonfinite_values": nonfinite_count,
            "fp64_squared_norm": squared_norm_raw if finite else None,
            "nonzero": bool(finite and squared_norm_raw > 0.0),
        }
    return result


def stable_clip(model: torch.nn.Module, max_norm: float) -> dict[str, Any]:
    evidence = grad_evidence(model)
    finite_values = [
        float(group["fp64_squared_norm"])
        for group in evidence.values()
        if group["fp64_squared_norm"] is not None
    ]
    total = math.sqrt(sum(finite_values)) if all(group["finite"] for group in evidence.values()) else math.nan
    if not math.isfinite(total):
        return {"norm_before": None, "coefficient": None, "post": evidence}
    coefficient = min(1.0, float(max_norm) / max(total, 1e-300))
    if coefficient < 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
    post = grad_evidence(model)
    return {"norm_before": total, "coefficient": coefficient, "post": post}


def optimizer_progress(optimizer: torch.optim.Optimizer) -> int:
    progress = 0
    for state in optimizer.state.values():
        step = state.get("step")
        if torch.is_tensor(step):
            progress += int(step.item())
        elif step is not None:
            progress += int(step)
    return progress


def parameter_versions(model: torch.nn.Module) -> tuple[int, ...]:
    return tuple(int(parameter._version) for parameter in model.parameters())


def missing_trainable_gradients(model: torch.nn.Module) -> set[str]:
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    }


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def dice_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = torch.sigmoid(logits.detach()) > 0.5
    truth = target.detach() > 0.5
    intersection = (prediction & truth).flatten(1).sum(1).float()
    denominator = prediction.flatten(1).sum(1) + truth.flatten(1).sum(1)
    score = torch.where(
        denominator > 0,
        2.0 * intersection / denominator.clamp_min(1).float(),
        torch.ones_like(intersection),
    )
    return float(score.mean().item())


@torch.no_grad()
def update_ema(teacher: torch.nn.Module, student: torch.nn.Module, ssl_step: int) -> float:
    decay = min(1.0 - 1.0 / float(ssl_step + 1), 0.996)
    for target, source in zip(teacher.parameters(), student.parameters()):
        target.mul_(decay).add_(source, alpha=1.0 - decay)
    for target, source in zip(teacher.buffers(), student.buffers()):
        target.copy_(source)
    return decay


def make_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    ema: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    bindings: Mapping[str, str],
    ssl_steps: int,
    unlabeled_sampler: legacy_train.ContinuousShuffle,
    attempt_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema": "mvaa-t3-surgenetdino-v2b-e1-checkpoint-v1",
        "epoch": epoch,
        "model_state": model.state_dict(),
        "ema_state": None if ema is None else ema.state_dict(),
        "candidate_state_key": "ema_state" if epoch == EPOCHS else None,
        "optimizer_state": optimizer.state_dict(),
        "ssl_optimizer_steps": ssl_steps,
        "unlabeled_unique_seen": len(unlabeled_sampler.seen),
        "unlabeled_cycles_completed": unlabeled_sampler.cycles,
        "args": dict(config),
        "bindings": dict(bindings),
        "attempt_counts": dict(attempt_counts),
        "heldout_metrics": None,
        "heldout_opened_during_training": False,
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing fold target: {args.output_dir}")
    # The backbone artifact is verified by content in core (SHA-256, tensor/element/byte
    # counts and two tensor shapes), so its location is free; only its identity is fixed.
    if not args.weights_path.is_file():
        raise FileNotFoundError(f"backbone artifact not found: {args.weights_path}")
    seed_before_model()
    train_samples, train_videos, videos, heldout_paths, unlabeled_paths = discover_sealed_inputs(args)
    output = args.output_dir.resolve()
    checkpoint_dir = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir()
    config = contract.frozen_config(
        args.holdout_video
    )
    atomic_json(output / "config.json", config)
    split = {
        "raw_labeled_count": 180,
        "labeled_videos": videos,
        "train_videos": train_videos,
        "val_videos": [args.holdout_video],
        "train_samples": [sample.image_path.resolve().as_posix() for sample in train_samples],
        "heldout_sample_paths_not_opened": heldout_paths,
        "excluded_frames": sorted(f"{video}_{frame:06d}" for video, frame in EXCLUDED_FRAMES),
        "heldout_opened_during_training": False,
    }
    atomic_json(output / "split.json", split)
    bindings = {
        "config_sha256": core.sha256_file(output / "config.json"),
        "split_sha256": core.sha256_file(output / "split.json"),
    }
    atomic_json(output / "unlabeled_manifest.json", {
        "root": args.unlabeled_root.resolve().as_posix(),
        "count": len(unlabeled_paths),
        "paths": [path.resolve().as_posix() for path in unlabeled_paths],
        "unique_seen": 0,
        "all_seen": False,
    })

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("E1 training requires CUDA; use the runner --dry-run on CPU")
    train_ds = LabeledDataset(
        train_samples, core.MODEL_IMAGE_SIZE, 10, True, True, True, SEED
    )
    unlabeled_ds = legacy_train.UnlabeledUniMatchDataset(unlabeled_paths, seed=SEED + 1000)
    unlabeled_sampler = legacy_train.ContinuousShuffle(len(unlabeled_ds), seed=SEED + 2000)
    # Seeded above, before this constructor initializes the DPT head.
    model = core.SurgeNetDinoV2BDPTSegmenter(args.weights_path, freeze_backbone=True).to(device)
    model.set_backbone_trainable(False)
    atomic_json(output / "initialization_preflight.json", {
        "schema": "mvaa-t3-surgenetdino-v2b-e1-initialization-v1",
        "seeded_before_model_construction": True,
        "seed": SEED,
        "artifact_sha256": core.ARTIFACT_SHA256,
        "backbone_state_sha256": state_dict_sha256(model.backbone.state_dict()),
        "head_initial_state_sha256": state_dict_sha256(model.head.state_dict()),
        "intermediate_blocks": list(core.INTERMEDIATE_BLOCKS),
        "backbone_width": core.BACKBONE_CHANNELS,
        "register_tokens": int(getattr(model.backbone, "num_reg_tokens", -1)),
        "mask_token_retained": hasattr(model.backbone, "mask_token"),
        "position_grid": list(model.backbone.pos_embed.shape),
    })
    loss_fn = get_loss_fn(
        loss_type="dice_focal", dice_weight=0.7, focal_weight=0.3,
        focal_gamma=2.0, focal_alpha=0.75, pos_weight=1.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=2e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ema: torch.nn.Module | None = None
    ssl_steps = 0
    attempt_counts = {"real": 0, "overflow": 0, "all_zero": 0}
    ledger_path = output / "optimizer_attempts.jsonl"
    history_path = output / "history.csv"
    fields = [
        "epoch", "stage", "attempts", "real_updates", "overflow_skips", "all_zero_skips",
        "sup_loss", "unsup_loss", "train_dice", "unlabeled_unique_seen", "epoch_seconds",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    total_ssl_steps = (EPOCHS - WARMUP_EPOCHS) * STEPS_PER_EPOCH
    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        stage = "decoder_warmup" if epoch <= WARMUP_EPOCHS else "unimatch_full"
        model.train()
        if stage == "decoder_warmup":
            model.backbone.eval()
        elif epoch == WARMUP_EPOCHS + 1:
            model.set_backbone_trainable(True)
            ema = copy.deepcopy(model).eval().requires_grad_(False)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": 5e-6, "name": "backbone"},
                    {"params": model.head.parameters(), "lr": 2e-4, "name": "head"},
                ],
                betas=(0.9, 0.999), weight_decay=0.01,
            )
        if stage == "unimatch_full" and ema is None:
            raise AssertionError("EMA teacher was not initialized at the frozen transition")

        weights = build_fg_balanced_weights(
            train_ds.sample_fg_ratio, power=0.5, min_weight=0.5, max_weight=4.0
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=(STEPS_PER_EPOCH + MAX_EXTRA_ATTEMPTS_PER_EPOCH)
            * ACCUMULATION * MICRO_BATCH,
            replacement=True,
            generator=torch.Generator().manual_seed(SEED + epoch),
        )
        loader = DataLoader(
            train_ds, batch_size=MICRO_BATCH, sampler=sampler,
            num_workers=0, pin_memory=True, drop_last=True,
        )
        iterator = iter(loader)
        epoch_real = epoch_attempts = overflow_skips = zero_skips = 0
        sums = {"sup": 0.0, "unsup": 0.0, "dice": 0.0}
        while epoch_real < STEPS_PER_EPOCH:
            epoch_attempts += 1
            if epoch_attempts > STEPS_PER_EPOCH + MAX_EXTRA_ATTEMPTS_PER_EPOCH:
                raise RuntimeError(f"Exceeded registered retry budget in epoch {epoch}")
            if stage == "unimatch_full":
                factor = (1.0 - ssl_steps / max(1, total_ssl_steps)) ** 0.9
                optimizer.param_groups[0]["lr"] = 5e-6 * factor
                optimizer.param_groups[1]["lr"] = 2e-4 * factor
            optimizer.zero_grad(set_to_none=True)
            attempt_sup = attempt_unsup = attempt_dice = 0.0
            for _ in range(ACCUMULATION):
                labeled = next(iterator)
                images = labeled["image"].to(device, non_blocking=True)
                labels = labeled["label"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(images)
                    supervised = loss_fn(logits, labels)
                    unsupervised = logits.sum() * 0.0
                    if stage == "unimatch_full":
                        batch = legacy_train.stack_unlabeled(
                            unlabeled_ds, unlabeled_sampler.take(MICRO_BATCH)
                        )
                        weak = batch["weak"].to(device, non_blocking=True)
                        strong1 = batch["strong1"].to(device, non_blocking=True)
                        strong2 = batch["strong2"].to(device, non_blocking=True)
                        with torch.no_grad():
                            weak_logits = ema(weak)
                            pseudo, _, valid = core.make_binary_pseudo_targets(weak_logits, 0.95)
                        mask1 = core.sample_cutmix_mask(
                            MICRO_BATCH, *core.MODEL_IMAGE_SIZE, device=device
                        )
                        mask2 = core.sample_cutmix_mask(
                            MICRO_BATCH, *core.MODEL_IMAGE_SIZE, device=device
                        )
                        mixed1, pseudo1, valid1 = core.apply_aligned_cutmix(
                            strong1, pseudo, valid, mask1
                        )
                        mixed2, pseudo2, valid2 = core.apply_aligned_cutmix(
                            strong2, pseudo, valid, mask2
                        )
                        logits1, logits2 = model.forward_paired_strong(mixed1, mixed2)
                        unsupervised = 0.5 * (
                            core.masked_bce_with_logits(logits1, pseudo1, valid1)
                            + core.masked_bce_with_logits(logits2, pseudo2, valid2)
                        )
                    total = supervised if stage == "decoder_warmup" else 0.5 * (
                        supervised + unsupervised
                    )
                scaler.scale(total / ACCUMULATION).backward()
                attempt_sup += float(supervised.detach())
                attempt_unsup += float(unsupervised.detach())
                attempt_dice += dice_from_logits(logits, labels)

            scaler.unscale_(optimizer)
            evidence = grad_evidence(model)
            missing = missing_trainable_gradients(model)
            expected_missing = (
                STRUCTURALLY_UNUSED_HEAD
                if stage == "decoder_warmup"
                else STRUCTURALLY_UNUSED_FULL
            )
            if missing != expected_missing:
                raise RuntimeError(
                    f"Unexpected structurally-unused gradient set: "
                    f"expected={sorted(expected_missing)}, actual={sorted(missing)}"
                )
            if stage == "decoder_warmup" and evidence["backbone"]["n_grad_tensors"] != 0:
                raise RuntimeError("Frozen backbone received a warm-up gradient")
            finite = all(group["finite"] for group in evidence.values())
            required_groups = ("head",) if stage == "decoder_warmup" else ("backbone", "head")
            nonzero = all(evidence[group]["nonzero"] for group in required_groups)
            old_scale = float(scaler.get_scale())
            versions_before = parameter_versions(model)
            progress_before = optimizer_progress(optimizer)
            if not finite:
                scaler.step(optimizer)
                scaler.update()
                if (
                    float(scaler.get_scale()) >= old_scale
                    or parameter_versions(model) != versions_before
                    or optimizer_progress(optimizer) != progress_before
                ):
                    raise RuntimeError("Overflow attempt was not an exact optimizer/model no-op")
                outcome = "overflow_skip"
                overflow_skips += 1
                attempt_counts["overflow"] += 1
            elif not nonzero:
                optimizer.zero_grad(set_to_none=True)
                scaler.update(new_scale=old_scale)
                if (
                    float(scaler.get_scale()) != old_scale
                    or parameter_versions(model) != versions_before
                    or optimizer_progress(optimizer) != progress_before
                ):
                    raise RuntimeError("All-zero attempt was not an exact no-op")
                outcome = "all_zero_exact_noop"
                zero_skips += 1
                attempt_counts["all_zero"] += 1
            else:
                clip = stable_clip(model, 1.0)
                post = clip["post"]
                if not all(group["finite"] for group in post.values()) or not all(
                    post[group]["nonzero"] for group in required_groups
                ):
                    raise RuntimeError(f"Stable FP64 clip invalidated gradients: {clip}")
                scaler.step(optimizer)
                scaler.update()
                if float(scaler.get_scale()) < old_scale:
                    raise RuntimeError("Finite audited gradients unexpectedly triggered a scaler skip")
                if (
                    parameter_versions(model) == versions_before
                    or optimizer_progress(optimizer) <= progress_before
                ):
                    raise RuntimeError("Audited real update did not advance model and optimizer")
                epoch_real += 1
                attempt_counts["real"] += 1
                outcome = "real_update"
                if stage == "unimatch_full":
                    ssl_steps += 1
                    ema_decay = update_ema(ema, model, ssl_steps)
                else:
                    ema_decay = None
                sums["sup"] += attempt_sup / ACCUMULATION
                sums["unsup"] += attempt_unsup / ACCUMULATION
                sums["dice"] += attempt_dice / ACCUMULATION
            record = {
                "epoch": epoch, "stage": stage, "attempt": epoch_attempts,
                "real_update_index": epoch_real, "outcome": outcome,
                "scale_before": old_scale, "scale_after": float(scaler.get_scale()),
                "gradient_evidence": evidence,
                "stable_clip": clip if outcome == "real_update" else None,
                "structurally_unused_gradients": sorted(missing),
                "optimizer_progress_before": progress_before,
                "optimizer_progress_after": optimizer_progress(optimizer),
                "parameter_versions_changed": parameter_versions(model) != versions_before,
                "ssl_step_after": ssl_steps,
                "ema_decay": ema_decay if outcome == "real_update" else None,
            }
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        row = {
            "epoch": epoch, "stage": stage, "attempts": epoch_attempts,
            "real_updates": epoch_real, "overflow_skips": overflow_skips,
            "all_zero_skips": zero_skips, "sup_loss": sums["sup"] / STEPS_PER_EPOCH,
            "unsup_loss": sums["unsup"] / STEPS_PER_EPOCH,
            "train_dice": sums["dice"] / STEPS_PER_EPOCH,
            "unlabeled_unique_seen": len(unlabeled_sampler.seen),
            "epoch_seconds": time.time() - started,
        }
        if not all(math.isfinite(float(row[key])) for key in ("sup_loss", "unsup_loss", "train_dice")):
            raise RuntimeError(f"Non-finite epoch ledger: {row}")
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        manifest = contract.read_json(output / "unlabeled_manifest.json")
        manifest.update({
            "unique_seen": len(unlabeled_sampler.seen),
            "all_seen": len(unlabeled_sampler.seen) == len(unlabeled_paths),
            "cycles_completed": unlabeled_sampler.cycles,
            "last_completed_epoch": epoch,
        })
        atomic_json(output / "unlabeled_manifest.json", manifest)
        if epoch % 10 == 0 or epoch == EPOCHS:
            payload = make_checkpoint(
                epoch, model, ema, optimizer, config, bindings, ssl_steps,
                unlabeled_sampler, attempt_counts,
            )
            name = contract.CHECKPOINT_NAME if epoch == EPOCHS else f"epoch_{epoch:03d}.pt"
            torch.save(payload, checkpoint_dir / name)

    if len(unlabeled_sampler.seen) != 1379:
        raise RuntimeError("Training did not visit all 1,379 official unlabeled frames")
    final = checkpoint_dir / contract.CHECKPOINT_NAME
    completion = {
        "schema": "mvaa-t3-surgenetdino-v2b-e1-fold-completion-v1",
        "status": "SEALED_TRAINING_COMPLETE_METRICS_UNOPENED",
        "holdout_video": args.holdout_video,
        "checkpoint_sha256": core.sha256_file(final),
        "config_sha256": bindings["config_sha256"],
        "split_sha256": bindings["split_sha256"],
        "history_sha256": core.sha256_file(history_path),
        "optimizer_ledger_sha256": core.sha256_file(ledger_path),
        "unlabeled_manifest_sha256": core.sha256_file(output / "unlabeled_manifest.json"),
        "initialization_preflight_sha256": core.sha256_file(output / "initialization_preflight.json"),
        "attempt_counts": attempt_counts,
        "heldout_opened_during_training": False,
        "heldout_metrics": None,
    }
    atomic_json(output / "TRAINING_COMPLETE.json", completion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
