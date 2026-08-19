#!/usr/bin/env python3
"""Fixed two-fold T3 DINOv2-S/UniMatch-V2-style trainer.

The protocol is pre-registered in
the UniMatch-style consistency recipe described in the project README.
This entry point intentionally exposes only paths, the held-out video, and
runtime plumbing; experiment hyperparameters are constants so the two kill
folds cannot drift.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_T3 = REPO_ROOT / "baseline" / "task3"
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASELINE_T3))
sys.path.insert(0, str(SCRIPTS_DIR))

import train as base_train  # noqa: E402
from dataset import (  # noqa: E402
    LabeledDataset,
    _apply_geom,
    _normalize_if_needed,
    _photo_aug_strong,
    _photo_aug_weak,
    _resize_chw,
    _to_tensor_chw,
    build_fg_balanced_weights,
    discover_samples,
    discover_unlabeled_images,
)
from model_factory import get_loss_fn  # noqa: E402
from train_t3_bcp import EXCLUDED_FRAMES, make_pinned_split  # noqa: E402
from utils import ensure_dir, save_json, seed_everything, setup_logger  # noqa: E402

import t3_dinov2_unimatch as dino_core  # noqa: E402


# Frozen protocol. These are deliberately not command-line tunables.
IMAGE_SIZE = (448, 798)
RAW_LABELED_COUNT = 180
LABELED_VIDEO_COUNT = 6
UNLABELED_COUNT = 1379
UNLABELED_DIRECTORY_COUNT = 46
EPOCHS = 70
WARMUP_EPOCHS = 5
STEPS_PER_EPOCH = 25
MICRO_BATCH = 2
GRAD_ACCUMULATION = 3
SEED = 42
BACKBONE_LR = 5e-6
DECODER_LR = 2e-4
WEIGHT_DECAY = 0.01
POLY_POWER = 0.9
CONFIDENCE_THRESHOLD = 0.95
GRAD_CLIP_NORM = 1.0
SAVE_EVERY = 10
WEIGHTS_SHA256 = "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
ALLOWED_HOLDOUTS = {
    "REC_20250205_102353_979A",
    "REC_20250322_101917_746A",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class UnlabeledUniMatchDataset:
    """One shared geometric view, one weak and two independent strong views."""

    def __init__(self, paths: Sequence[Path], seed: int = SEED):
        self.paths = list(paths)
        self.rng = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.paths[int(index)]
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        hflip = self.rng.random() < 0.5
        vflip = self.rng.random() < 0.3
        rot_k = self.rng.randint(0, 3) if self.rng.random() < 0.4 else 0
        base = _apply_geom(image, hflip=hflip, vflip=vflip, rot_k=rot_k)
        weak = _photo_aug_weak(base.copy(), self.rng)
        strong1 = _photo_aug_strong(base.copy(), self.rng)
        strong2 = _photo_aug_strong(base.copy(), self.rng)

        def convert(value: np.ndarray) -> torch.Tensor:
            tensor = _to_tensor_chw(value)
            if tuple(tensor.shape[1:]) != IMAGE_SIZE:
                tensor = _resize_chw(tensor, IMAGE_SIZE, mode="bilinear")
            return _normalize_if_needed(tensor, True)

        return {
            "weak": convert(weak),
            "strong1": convert(strong1),
            "strong2": convert(strong2),
            "image_path": path.as_posix(),
        }


class ContinuousShuffle:
    """Deterministic shuffled cycles that do not reset at epoch boundaries."""

    def __init__(self, size: int, seed: int):
        if size <= 0:
            raise ValueError("ContinuousShuffle requires at least one sample")
        self.size = int(size)
        self.rng = random.Random(int(seed))
        self.order = list(range(self.size))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.cycles = 0
        self.seen: set[int] = set()

    def take(self, count: int) -> List[int]:
        result: List[int] = []
        while len(result) < int(count):
            if self.cursor == self.size:
                self.cycles += 1
                self.rng.shuffle(self.order)
                self.cursor = 0
            index = self.order[self.cursor]
            self.cursor += 1
            self.seen.add(index)
            result.append(index)
        return result


def stack_unlabeled(dataset: UnlabeledUniMatchDataset, indices: Iterable[int]) -> Dict[str, Any]:
    samples = [dataset[i] for i in indices]
    return {
        key: torch.stack([sample[key] for sample in samples])
        for key in ("weak", "strong1", "strong2")
    } | {"image_path": [sample["image_path"] for sample in samples]}


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(bool(trainable))
    # The core uses this flag both to keep the backbone in eval mode and to
    # select a no_grad context; changing requires_grad alone is insufficient.
    model.freeze_backbone = not bool(trainable)
    if trainable:
        model.backbone.train()
    else:
        model.backbone.eval()


def update_ema(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
            teacher_param.mul_(float(decay)).add_(student_param.detach(), alpha=1.0 - float(decay))
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            teacher_buffer.copy_(student_buffer.detach())


def polynomial_factor(ssl_step: int, total_ssl_steps: int) -> float:
    progress = min(max(float(ssl_step) / float(max(1, total_ssl_steps)), 0.0), 1.0)
    return (1.0 - progress) ** POLY_POWER


def dice_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    return base_train.dice_from_logits(logits.detach(), target.detach(), ignore_empty_gt=False)


def make_model(weights_path: Path) -> nn.Module:
    """Small adapter around the pure core's deliberately narrow constructor."""
    return dino_core.DinoV2DPTSegmenter(weights_path=weights_path, freeze_backbone=True)


def strong_pair_and_targets(
    model: nn.Module,
    strong1: torch.Tensor,
    strong2: torch.Tensor,
    pseudo: torch.Tensor,
    confidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CutMix both views and run their paired complementary-dropout forward."""
    mask1 = dino_core.sample_cutmix_mask(
        batch_size=strong1.shape[0], height=strong1.shape[-2], width=strong1.shape[-1],
        probability=0.5, area_range=(0.02, 0.40), aspect_range=(0.3, 3.33),
        device=strong1.device,
    )
    mask2 = dino_core.sample_cutmix_mask(
        batch_size=strong2.shape[0], height=strong2.shape[-2], width=strong2.shape[-1],
        probability=0.5, area_range=(0.02, 0.40), aspect_range=(0.3, 3.33),
        device=strong2.device,
    )
    mixed1, pseudo1, confidence1 = dino_core.apply_aligned_cutmix(
        strong1, pseudo, confidence, mask1,
    )
    mixed2, pseudo2, confidence2 = dino_core.apply_aligned_cutmix(
        strong2, pseudo, confidence, mask2,
    )
    logits1, logits2 = model.forward_paired_strong(mixed1, mixed2)
    return logits1, logits2, pseudo1, pseudo2, confidence1, confidence2


def checkpoint_payload(
    epoch: int,
    model: nn.Module,
    ema: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    ssl_steps: int,
    unlabeled_sampler: ContinuousShuffle,
    val_metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "ema_state": None if ema is None else ema.state_dict(),
        "model_state_source": "student",
        "candidate_state_key": "ema_state" if int(epoch) == EPOCHS else None,
        "optimizer_state": optimizer.state_dict(),
        "ssl_optimizer_steps": int(ssl_steps),
        "unlabeled_unique_seen": len(unlabeled_sampler.seen),
        "unlabeled_cycles_completed": int(unlabeled_sampler.cycles),
        "args": {
            **vars(args),
            "image_size": list(IMAGE_SIZE),
            "target_label": 10,
            "freeze_backbone": False,
            "threshold": 0.45,
            "arch": "dinov2_dpt_small",
        },
        "val_metrics": {**val_metrics, "val_threshold": 0.45},
        "protocol": frozen_config(args),
    }


def frozen_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "run_name": "task3_dinov2_unimatch_lovo_v1",
        "holdout_video": args.holdout_video,
        "labeled_root": str(args.labeled_root),
        "unlabeled_root": str(args.unlabeled_root),
        "weights_path": str(args.weights_path),
        "weights_sha256": WEIGHTS_SHA256,
        "image_size": list(IMAGE_SIZE),
        "target_label": 10,
        "arch": "dinov2_dpt_small",
        "epochs": EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": GRAD_ACCUMULATION,
        "effective_batch_each_stream": MICRO_BATCH * GRAD_ACCUMULATION,
        "seed": SEED,
        "backbone_lr": BACKBONE_LR,
        "decoder_lr": DECODER_LR,
        "weight_decay": WEIGHT_DECAY,
        "poly_power": POLY_POWER,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "save_every": SAVE_EVERY,
        "amp": not args.no_amp,
        "eligible_checkpoint": "checkpoints/epoch_070_ema.pt:ema_state",
        "pretrained_provenance": {
            "hf_repository": "timm/vit_small_patch14_dinov2.lvd142m",
            "revision": "4610ca143709d58a633b6397a74412c2c3842454",
            "source_url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
            "license": "Apache-2.0",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-video", required=True, choices=sorted(ALLOWED_HOLDOUTS))
    parser.add_argument("--labeled-root", type=Path, required=True)
    parser.add_argument("--unlabeled-root", type=Path, required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> Tuple[
    List[Any], List[Any], List[str], List[str], List[str], List[Path], List[Path]
]:
    for name in ("labeled_root", "unlabeled_root", "weights_path"):
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    actual_hash = sha256_file(args.weights_path)
    if actual_hash != WEIGHTS_SHA256:
        raise ValueError(f"DINOv2 weights SHA-256 mismatch: expected {WEIGHTS_SHA256}, got {actual_hash}")

    all_labeled = discover_samples(args.labeled_root)
    labeled_videos = sorted({sample.video_id for sample in all_labeled})
    if len(all_labeled) != RAW_LABELED_COUNT or len(labeled_videos) != LABELED_VIDEO_COUNT:
        raise ValueError(
            f"Expected {RAW_LABELED_COUNT} labeled frames/{LABELED_VIDEO_COUNT} videos, "
            f"found {len(all_labeled)}/{len(labeled_videos)}"
        )

    unlabeled = sorted(discover_unlabeled_images(args.unlabeled_root))
    unlabeled_dirs = sorted({path.parent.resolve() for path in unlabeled})
    if len(unlabeled) != UNLABELED_COUNT or len(unlabeled_dirs) != UNLABELED_DIRECTORY_COUNT:
        raise ValueError(
            f"Expected {UNLABELED_COUNT} unlabeled images/{UNLABELED_DIRECTORY_COUNT} directories, "
            f"found {len(unlabeled)}/{len(unlabeled_dirs)}"
        )
    overlaps = [video for video in labeled_videos if any(video in path.as_posix() for path in unlabeled)]
    if overlaps:
        raise ValueError(f"Labeled video IDs occur in unlabeled paths: {overlaps}")

    split_fn = make_pinned_split(args.holdout_video)
    train_samples, val_samples, train_videos, val_videos = split_fn(all_labeled)
    if val_videos != [args.holdout_video] or args.holdout_video in train_videos:
        raise AssertionError("Pinned LOVO split did not isolate the held-out video")
    for sample in train_samples + val_samples:
        if (sample.video_id, int(sample.frame_idx)) in EXCLUDED_FRAMES:
            raise AssertionError("Known noisy frame survived the pinned split")
    return train_samples, val_samples, train_videos, val_videos, labeled_videos, unlabeled, unlabeled_dirs


def main() -> int:
    args = parse_args()
    seed_everything(SEED)
    (
        train_samples, val_samples, train_videos, val_videos, labeled_videos,
        unlabeled_paths, unlabeled_dirs,
    ) = validate_inputs(args)

    out_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(out_dir / "checkpoints")
    logger = setup_logger(out_dir, log_name="train.log")
    (out_dir / "command.txt").write_text(
        " ".join(shlex.quote(value) for value in sys.argv) + "\n", encoding="utf-8"
    )
    config = frozen_config(args)
    save_json(out_dir / "config.json", config)
    excluded = sorted(f"{video}_{frame:06d}" for video, frame in EXCLUDED_FRAMES)
    save_json(out_dir / "split.json", {
        "raw_labeled_count": RAW_LABELED_COUNT,
        "labeled_videos": labeled_videos,
        "train_videos": train_videos,
        "val_videos": val_videos,
        "train_samples": [sample.image_path.as_posix() for sample in train_samples],
        "val_samples": [sample.image_path.as_posix() for sample in val_samples],
        "excluded_frames": excluded,
    })

    manifest_path = out_dir / "unlabeled_manifest.json"
    manifest: Dict[str, Any] = {
        "root": args.unlabeled_root.resolve().as_posix(),
        "count": len(unlabeled_paths),
        "directory_count": len(unlabeled_dirs),
        "paths": [path.resolve().as_posix() for path in unlabeled_paths],
        "unique_seen": 0,
        "all_seen": False,
    }
    save_json(manifest_path, manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    logger.info("Frozen config: %s", json.dumps(config, sort_keys=True))
    logger.info("Device=%s AMP=%s", device, use_amp)

    train_ds = LabeledDataset(
        samples=train_samples, image_size=IMAGE_SIZE, target_label=10, train=True,
        cache_masks=True, use_imagenet_norm=True, seed=SEED,
    )
    val_ds = LabeledDataset(
        samples=val_samples, image_size=IMAGE_SIZE, target_label=10, train=False,
        cache_masks=True, use_imagenet_norm=True, seed=SEED,
    )
    val_loader = DataLoader(val_ds, batch_size=MICRO_BATCH, shuffle=False, num_workers=0)
    unlabeled_ds = UnlabeledUniMatchDataset(unlabeled_paths, seed=SEED + 1000)
    unlabeled_sampler = ContinuousShuffle(len(unlabeled_ds), seed=SEED + 2000)

    model = make_model(args.weights_path).to(device)
    set_backbone_trainable(model, False)
    supervised_loss = get_loss_fn(
        loss_type="dice_focal", dice_weight=0.7, bce_weight=0.3,
        focal_weight=0.3, focal_gamma=2.0, focal_alpha=0.75, pos_weight=1.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=DECODER_LR, weight_decay=WEIGHT_DECAY)
    ema: nn.Module | None = None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    total_ssl_steps = (EPOCHS - WARMUP_EPOCHS) * STEPS_PER_EPOCH
    ssl_steps = 0

    history_path = out_dir / "history.csv"
    history_fields = [
        "epoch", "stage", "sup_loss", "unsup_loss", "total_loss", "train_dice",
        "pseudo_fg_ratio", "confident_ratio", "empty_pseudo_microbatches",
        "backbone_lr", "decoder_lr", "ema_decay", "unlabeled_unique_seen",
        "val_loss_grid", "val_dice_grid", "val_hd_grid", "val_asd_grid", "epoch_sec",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=history_fields).writeheader()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        if epoch <= WARMUP_EPOCHS:
            model.backbone.eval()
            stage = "decoder_warmup"
        else:
            stage = "unimatch_full"
            if epoch == WARMUP_EPOCHS + 1:
                set_backbone_trainable(model, True)
                ema = copy.deepcopy(model).to(device).eval()
                for parameter in ema.parameters():
                    parameter.requires_grad_(False)
                optimizer = torch.optim.AdamW(
                    [
                        {"params": model.backbone.parameters(), "lr": BACKBONE_LR, "name": "backbone"},
                        {"params": model.head.parameters(), "lr": DECODER_LR, "name": "decoder"},
                    ],
                    betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY,
                )
            if ema is None:
                raise AssertionError("EMA teacher was not initialized at epoch 6")
            ema.eval()

        weights = build_fg_balanced_weights(
            train_ds.sample_fg_ratio, power=0.5, min_weight=0.5, max_weight=4.0,
        )
        labeled_sampler = WeightedRandomSampler(
            weights, num_samples=STEPS_PER_EPOCH * GRAD_ACCUMULATION * MICRO_BATCH,
            replacement=True, generator=torch.Generator().manual_seed(SEED + epoch),
        )
        labeled_loader = DataLoader(
            train_ds, batch_size=MICRO_BATCH, sampler=labeled_sampler,
            num_workers=int(args.num_workers), pin_memory=device.type == "cuda", drop_last=True,
        )
        labeled_iter = iter(labeled_loader)

        sums = {key: 0.0 for key in ("sup", "unsup", "total", "dice", "pseudo_fg", "confidence")}
        micro_count = 0
        empty_pseudo = 0
        last_ema_decay = float("nan")
        for optimizer_step in range(STEPS_PER_EPOCH):
            if stage == "unimatch_full":
                factor = polynomial_factor(ssl_steps, total_ssl_steps)
                optimizer.param_groups[0]["lr"] = BACKBONE_LR * factor
                optimizer.param_groups[1]["lr"] = DECODER_LR * factor
            optimizer.zero_grad(set_to_none=True)

            for _ in range(GRAD_ACCUMULATION):
                labeled = next(labeled_iter)
                images = labeled["image"].to(device, non_blocking=True)
                labels = labeled["label"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits_sup = model(images)
                    sup_loss = supervised_loss(logits_sup, labels)
                    unsup_loss = logits_sup.sum() * 0.0

                    if stage == "unimatch_full":
                        unlabeled = stack_unlabeled(
                            unlabeled_ds, unlabeled_sampler.take(MICRO_BATCH)
                        )
                        weak = unlabeled["weak"].to(device, non_blocking=True)
                        strong1 = unlabeled["strong1"].to(device, non_blocking=True)
                        strong2 = unlabeled["strong2"].to(device, non_blocking=True)
                        with torch.no_grad():
                            weak_logits = ema(weak)
                            pseudo, confidence, valid_mask = dino_core.make_binary_pseudo_targets(
                                weak_logits, confidence_threshold=CONFIDENCE_THRESHOLD,
                            )
                        logits1, logits2, pseudo1, pseudo2, conf1, conf2 = strong_pair_and_targets(
                            model, strong1, strong2, pseudo, valid_mask,
                        )
                        loss1 = dino_core.masked_bce_with_logits(logits1, pseudo1, conf1)
                        loss2 = dino_core.masked_bce_with_logits(logits2, pseudo2, conf2)
                        unsup_loss = (loss1 + loss2) / 2.0
                        empty_pseudo += int((pseudo.flatten(1).sum(1) == 0).sum().item())
                        sums["pseudo_fg"] += float(pseudo.mean().item())
                        sums["confidence"] += float(valid_mask.mean().item())

                    total_loss = sup_loss if stage == "decoder_warmup" else (sup_loss + unsup_loss) / 2.0

                scaled_loss = total_loss / float(GRAD_ACCUMULATION)
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                sums["sup"] += float(sup_loss.detach().item())
                sums["unsup"] += float(unsup_loss.detach().item())
                sums["total"] += float(total_loss.detach().item())
                sums["dice"] += dice_from_logits(logits_sup, labels)
                micro_count += 1

            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            if stage == "unimatch_full":
                ssl_steps += 1
                last_ema_decay = min(1.0 - 1.0 / float(ssl_steps), 0.996)
                update_ema(ema, model, last_ema_decay)

        if stage == "unimatch_full" and len(unlabeled_sampler.seen) == 0:
            raise AssertionError("UniMatch stage did not consume unlabeled data")

        # Held-out numbers are fixed-threshold/model-grid diagnostics only and
        # never select a checkpoint. Evaluate on save epochs to limit feedback.
        val_metrics = {"val_loss": math.nan, "val_dice": math.nan, "val_hd": math.nan, "val_asd": math.nan}
        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            eval_model = ema if ema is not None else model
            val_metrics = base_train.evaluate(
                model=eval_model, loader=val_loader, loss_fn=supervised_loss,
                device=device, use_amp=use_amp, threshold_candidates=[0.45],
                use_tta=False, fixed_threshold=0.45,
            )
            payload = checkpoint_payload(
                epoch, model, ema, optimizer, args, ssl_steps, unlabeled_sampler, val_metrics,
            )
            checkpoint_name = "epoch_070_ema.pt" if epoch == EPOCHS else f"epoch_{epoch:03d}.pt"
            torch.save(payload, checkpoint_dir / checkpoint_name)

        manifest["unique_seen"] = len(unlabeled_sampler.seen)
        manifest["all_seen"] = len(unlabeled_sampler.seen) == len(unlabeled_paths)
        manifest["cycles_completed"] = unlabeled_sampler.cycles
        manifest["last_completed_epoch"] = epoch
        save_json(manifest_path, manifest)

        denom = max(1, micro_count)
        group_lrs = {group.get("name", "decoder"): float(group["lr"]) for group in optimizer.param_groups}
        row = {
            "epoch": epoch,
            "stage": stage,
            "sup_loss": sums["sup"] / denom,
            "unsup_loss": sums["unsup"] / denom,
            "total_loss": sums["total"] / denom,
            "train_dice": sums["dice"] / denom,
            "pseudo_fg_ratio": sums["pseudo_fg"] / denom,
            "confident_ratio": sums["confidence"] / denom,
            "empty_pseudo_microbatches": empty_pseudo,
            "backbone_lr": group_lrs.get("backbone", 0.0),
            "decoder_lr": group_lrs.get("decoder", DECODER_LR),
            "ema_decay": last_ema_decay,
            "unlabeled_unique_seen": len(unlabeled_sampler.seen),
            "val_loss_grid": val_metrics["val_loss"],
            "val_dice_grid": val_metrics["val_dice"],
            "val_hd_grid": val_metrics["val_hd"],
            "val_asd_grid": val_metrics["val_asd"],
            "epoch_sec": time.time() - epoch_start,
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=history_fields)
            writer.writerow(row)
        logger.info("Epoch %d/%d %s | %s", epoch, EPOCHS, stage, row)

    if len(unlabeled_sampler.seen) != UNLABELED_COUNT:
        raise RuntimeError(
            f"Training ended without visiting the complete unlabeled pool: "
            f"{len(unlabeled_sampler.seen)}/{UNLABELED_COUNT}"
        )
    eligible = checkpoint_dir / "epoch_070_ema.pt"
    if not eligible.exists():
        raise RuntimeError(f"Eligible fixed checkpoint was not written: {eligible}")
    save_json(out_dir / "final_metrics.json", {
        "status": "TRAINING_COMPLETE_EVALUATION_PENDING",
        "eligible_checkpoint": eligible.as_posix(),
        "eligible_state_key": "ema_state",
        "epoch": EPOCHS,
        "unlabeled_unique_seen": len(unlabeled_sampler.seen),
    })
    logger.info("Training complete. Sole eligible candidate: %s:ema_state", eligible)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
