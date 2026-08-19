#!/usr/bin/env python3
"""Task3 Bidirectional Copy-Paste (BCP) Mean-Teacher upgrade — wrapper around
baseline/task3/train.py.

Bidirectional copy-paste mean-teacher variant.
INCLUDING the "Sonnet design-review verdict" section, which LOCKS the design
decisions this file implements (item numbers referenced in comments below
refer to that section). baseline/task3/ is left completely untouched.

Mechanism
---------
train.py is one large, non-decomposed `main()` (baseline/task3/train.py:307-879);
BCP changes what "unsup_loss" means (a mixed-image, mask-aware loss) rather
than swapping a pluggable dataset/model/loss factory, so it cannot be
implemented via the monkey-patch pattern every other T3 lever uses. Per
verdict item 6, option (b): this wrapper's own tiny pre-parser consumes ONLY
the `--bcp-*` flags (`parse_known_args`); the remaining argv is untouched.

  --bcp off (default): the leftover argv is handed to `train.main()`
    UNMODIFIED — literally the same function call as running
    `python baseline/task3/train.py <args>` directly. This is not merely
    "close to" a no-op; there is no BCP code on this path at all.

  --bcp on: a full superset argparse (baseline/task3/train.py's own argument
    list, duplicated verbatim, plus the 4 new --bcp-* flags) parses the
    leftover argv, and a forked copy of train.main()'s setup + epoch loop
    (`main_bcp_on` below) runs, with the per-step unsupervised branch
    (train.py:631-662) replaced by BCP mixing whenever `lambda_u > 0` (the
    exact same gating condition FixMatch uses today — verdict item 9).

Frame exclusion: EXCLUDED_FRAMES below drops the known chamber-included-mask
frame before any split, matching the convention used across all T3 training.

Usage (real GPU run, NOT executed by this task)
------------------------------------------------
    python scripts/train_t3_bcp.py \\
        --labeled-root   data/reference_data/t3_vid/train \\
        --unlabeled-root data/images \\
        --output-dir     runs/task3_bcp_lovo/f-REC_20250205_102353_979A_on \\
        --bcp on --bcp-beta 0.667 --bcp-alpha 0.5 --bcp-crop-placement random \\
        --epochs 90 --save-every 10 --early-stop-patience 0 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_T3 = REPO_ROOT / "baseline" / "task3"
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASELINE_T3))
sys.path.insert(0, str(SCRIPTS_DIR))

import t3_bcp  # noqa: E402  (pure-logic module, scripts/t3_bcp.py)

# Known noisy-label frame (chamber included in the MV mask) — same convention
# Noisy chamber-included mask; excluded from every T3 arm.
EXCLUDED_FRAMES = {("REC_20250322_101917_746A", 130)}


def make_pinned_split(holdout: str):
    """LOVO fold-pinning split, based on scripts/train_t3_temporal.py:50-56
    `pinned_split`: the named video is the ENTIRE val set, every other video
    goes to train. Signature matches dataset.split_train_val_by_video
    (val_video_count/seed accepted but ignored) so it is a drop-in in both
    wrapper paths.

    SINGLE SOURCE OF TRUTH for the noisy-frame exclusion in LOVO mode: this
    function drops EXCLUDED_FRAMES BEFORE partitioning, so that both the
    --bcp on path (which uses this as its split_fn) and the --bcp off path
    (which installs this via monkey-patch onto train.split_train_val_by_video)
    train and evaluate on the identical frame set — the LOVO A/B's only
    variable is then BCP itself. The excluded frame `(746A, 130)` is a NOISY
    LABEL (chamber included in the MV mask), so it is dropped from val as well
    as train: when 746A is the held-out fold, frame 130 is NOT a val target
    (scoring against a known-bad GT would corrupt that fold's metric)."""

    def pinned_split(samples, val_video_count=2, seed=42):
        samples = [s for s in samples if (s.video_id, int(s.frame_idx)) not in EXCLUDED_FRAMES]
        vids = sorted({s.video_id for s in samples})
        if holdout not in vids:
            raise ValueError(f"--holdout-video {holdout!r} not in {vids}")
        trn = [s for s in samples if s.video_id != holdout]
        val = [s for s in samples if s.video_id == holdout]
        return trn, val, sorted({s.video_id for s in trn}), [holdout]

    return pinned_split


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def add_bcp_args(parser: argparse.ArgumentParser) -> None:
    """The LOCKED `--bcp-*` CLI namespace (spec's Hyperparameters section,
    corrected by the verdict). Only 4 new flags — the pseudo-label gate and
    warmup schedule are explicitly INHERITED from train.py's existing flags
    (`--pseudo-pos-thr`, `--pseudo-neg-thr`, `--pseudo-min-area`,
    `--pseudo-min-pos-ratio`, `--semi-warmup-epochs`), not duplicated."""
    parser.add_argument("--bcp", type=str, default="off", choices=["on", "off"])
    parser.add_argument("--bcp-beta", type=float, default=0.667)
    parser.add_argument("--bcp-alpha", type=float, default=0.5)
    parser.add_argument("--bcp-crop-placement", type=str, default="random", choices=["center", "random"])
    # LOVO fold-pinning (wrapper flag, mirrors scripts/train_t3_temporal.py:39,47-59).
    # When set, the named video becomes the ENTIRE val set and every other video
    # goes to train; forces --val-video-count 1. Applied on BOTH --bcp on and off
    # so a paired A/B fold holds out the identical video. Default None => the
    # baseline default val split (unchanged behaviour).
    parser.add_argument("--holdout-video", type=str, default=None)


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Verbatim duplicate of baseline/task3/train.py:46-124's argument list
    (minus the trailing `parser.parse_args()` call). Required because
    train.py's parse_args() is not decomposed into a reusable parser-builder
    (verdict item 6) — this fork must own a full superset parser for the
    --bcp on path. KEEP IN SYNC with train.py if its args change (Risk 5)."""
    parser.add_argument("--labeled-root", type=str, default=str(REPO_ROOT / "data" / "t3_vid" / "train"))
    parser.add_argument("--external-val-root", type=str, default=str(BASELINE_T3 / "data" / "labeled" / "val_external"))
    parser.add_argument("--use-external-val", action="store_true", default=False)
    parser.add_argument("--no-use-external-val", action="store_false", dest="use_external_val")
    parser.add_argument("--unlabeled-root", type=str, default=str(REPO_ROOT / "task3" / "未标记素材-图片" / "images"))
    parser.add_argument("--output-dir", type=str, default=str(BASELINE_T3 / "runs" / "bcp_default"))

    parser.add_argument("--arch", type=str, default="unetplusplus", choices=["unet", "unetplusplus", "fpn", "deeplabv3plus"])
    parser.add_argument("--encoder-name", type=str, default="efficientnet-b4")
    parser.add_argument("--encoder-weights", type=str, default="none", choices=["none", "imagenet"])
    parser.add_argument("--target-label", type=int, default=10)
    parser.add_argument("--image-size", type=int, nargs=2, default=[448, 800], help="H W")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--unlabeled-batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-video-count", type=int, default=2)
    parser.add_argument("--val-only-fg", action="store_true", default=True)
    parser.add_argument("--no-val-only-fg", action="store_false", dest="val_only_fg")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Debug only; 0 means all")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Debug only; 0 means all")
    parser.add_argument("--max-unlabeled-samples", type=int, default=0, help="Debug only; 0 means all")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--loss-type", type=str, default="dice_focal", choices=["dice_bce", "dice_focal"])
    parser.add_argument("--dice-loss-weight", type=float, default=0.7)
    parser.add_argument("--bce-loss-weight", type=float, default=0.3)
    parser.add_argument("--focal-loss-weight", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--pos-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--score-dsc-weight", type=float, default=0.6)
    parser.add_argument("--score-hd-weight", type=float, default=0.2)
    parser.add_argument("--score-asd-weight", type=float, default=0.2)
    parser.add_argument("--score-hd-ref", type=float, default=20.0)
    parser.add_argument("--score-asd-ref", type=float, default=3.0)

    parser.add_argument("--semi-warmup-epochs", type=int, default=20)
    parser.add_argument("--unsup-weight", type=float, default=0.6)
    parser.add_argument("--unsup-ramp-epochs", type=int, default=30)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--pseudo-pos-thr", type=float, default=0.70)
    parser.add_argument("--pseudo-neg-thr", type=float, default=0.10)
    parser.add_argument("--pseudo-min-area", type=float, default=80.0)
    parser.add_argument("--pseudo-min-pos-ratio", type=float, default=0.0005)

    parser.add_argument(
        "--threshold-candidates",
        type=float,
        nargs="+",
        default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
    )
    parser.add_argument("--val-tta", action="store_true", default=True)
    parser.add_argument("--no-val-tta", action="store_false", dest="val_tta")

    parser.add_argument("--use-imagenet-norm", action="store_true", default=True)
    parser.add_argument("--no-imagenet-norm", action="store_false", dest="use_imagenet_norm")
    parser.add_argument("--cache-masks", action="store_true", default=True)
    parser.add_argument("--no-cache-masks", action="store_false", dest="cache_masks")

    parser.add_argument("--use-fg-balanced-sampling", action="store_true", default=True)
    parser.add_argument("--no-fg-balanced-sampling", action="store_false", dest="use_fg_balanced_sampling")
    parser.add_argument("--fg-sampling-power", type=float, default=0.5)
    parser.add_argument("--fg-sampling-min-weight", type=float, default=0.5)
    parser.add_argument("--fg-sampling-max-weight", type=float, default=4.0)

    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--early-stop-patience", type=int, default=40)


# ---------------------------------------------------------------------------
# BCP per-step mixing (the forked replacement for train.py:631-662)
# ---------------------------------------------------------------------------
def bcp_train_step(
    model,
    teacher,
    images: torch.Tensor,
    labels: torch.Tensor,
    weak: torch.Tensor,
    focal_none_fn,
    lambda_u: float,
    device: torch.device,
    use_amp: bool,
    beta: float,
    placement: str,
    alpha: float,
    dice_weight: float,
    focal_weight: float,
    pseudo_pos_thr: float,
    pseudo_neg_thr: float,
    pseudo_min_area: float,
    pseudo_min_pos_ratio: float,
    rng=None,
    min_bs: int = 2,
) -> Tuple[torch.Tensor, float, float, bool]:
    """Standalone, CPU-testable BCP mixing step.

    Returns (unsup_loss, pseudo_pos_ratio, pseudo_conf_ratio, invoked) where
    `invoked` is False (and unsup_loss == 0) whenever BCP mixing does NOT run
    this step — either because `lambda_u <= 0` (warmup no-op, verdict item 9
    / unit test check 8) or because either batch is smaller than `min_bs`
    (verdict item 12 batch-size-1 guard).

    Imported directly from baseline/task3/train.py's `predict_probs` by the
    caller (`main_bcp_on`) — this function takes `teacher`/`model` as plain
    callables so it needs no baseline import itself, only `t3_bcp` (mask,
    compositing, pseudo-labeling, masked-loss primitives) and the caller-
    supplied `predict_probs_fn`-free forward calls under `torch.amp.autocast`.
    """
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    if float(lambda_u) <= 0.0:
        return zero, 0.0, 0.0, False

    labeled_bs = int(images.shape[0])
    unlabeled_bs = int(weak.shape[0])
    if not t3_bcp.can_bcp_mix(labeled_bs, unlabeled_bs, min_bs=min_bs):
        return zero, 0.0, 0.0, False

    # Unequal batch sizes can occur at epoch boundaries (both loaders use
    # drop_last=False and are cycled independently, verdict item 12) — the
    # compositing below requires matching batch dims, so truncate to the
    # common size. Not spelled out verbatim in the spec; documented here as
    # a necessary implementation completion (see task report).
    bs = min(labeled_bs, unlabeled_bs)
    img_a = images[:bs]
    lab_a = labels[:bs]
    uimg_a = weak[:bs]

    img_b = t3_bcp.roll_pair(img_a, 1, 0)
    lab_b = t3_bcp.roll_pair(lab_a, 1, 0)
    uimg_b = t3_bcp.roll_pair(uimg_a, 1, 0)

    h, w = int(img_a.shape[-2]), int(img_a.shape[-1])
    mask_hw, _, _ = t3_bcp.generate_bcp_mask(h, w, batch_size=bs, beta=beta, placement=placement, rng=rng)
    mask_hw = mask_hw.to(device=device, dtype=img_a.dtype)

    device_type = device.type
    with torch.no_grad():
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            t_logits_a = teacher(uimg_a)
        t_probs_a = torch.sigmoid(t_logits_a.float())
        # Teacher is in eval() mode (no batch-dependent ops) so
        # roll(teacher(x)) == teacher(roll(x)) exactly; reuse the single
        # forward pass instead of a second one for uimg_b.
        t_probs_b = t3_bcp.roll_pair(t_probs_a, 1, 0)
        pseudo_a, conf_a, _ = t3_bcp.pseudo_label_from_probs(
            t_probs_a, pseudo_pos_thr, pseudo_neg_thr, pseudo_min_area, pseudo_min_pos_ratio
        )
        pseudo_b, conf_b, _ = t3_bcp.pseudo_label_from_probs(
            t_probs_b, pseudo_pos_thr, pseudo_neg_thr, pseudo_min_area, pseudo_min_pos_ratio
        )

    # Direction "unl": labeled GT patch pasted into an unlabeled background.
    # net_input_unl = uimg_a*mask + img_a*(1-mask)
    net_input_unl = t3_bcp.composite(uimg_a, img_a, mask_hw)
    y_unl = t3_bcp.composite(pseudo_a, lab_a, mask_hw)
    mask_gt_unl = 1.0 - mask_hw
    mask_pseudo_unl_eff = mask_hw * conf_a

    # Direction "l": unlabeled patch pasted into a labeled background.
    # net_input_l = img_b*mask + uimg_b*(1-mask)
    net_input_l = t3_bcp.composite(img_b, uimg_b, mask_hw)
    y_l = t3_bcp.composite(lab_b, pseudo_b, mask_hw)
    mask_gt_l = mask_hw
    mask_pseudo_l_eff = (1.0 - mask_hw) * conf_b

    with torch.amp.autocast(device_type=device_type, enabled=use_amp):
        logits_unl = model(net_input_unl)
        logits_l = model(net_input_l)

    loss_unl, _ = t3_bcp.bcp_direction_loss(
        logits_unl, y_unl, mask_gt_unl, mask_pseudo_unl_eff, focal_none_fn,
        alpha=alpha, dice_weight=dice_weight, focal_weight=focal_weight,
    )
    loss_l, _ = t3_bcp.bcp_direction_loss(
        logits_l, y_l, mask_gt_l, mask_pseudo_l_eff, focal_none_fn,
        alpha=alpha, dice_weight=dice_weight, focal_weight=focal_weight,
    )

    total = loss_unl + loss_l
    pseudo_pos_ratio = float(((pseudo_a.mean() + pseudo_b.mean()) / 2.0).item())
    pseudo_conf_ratio = float(((conf_a.mean() + conf_b.mean()) / 2.0).item())
    return total, pseudo_pos_ratio, pseudo_conf_ratio, True


# ---------------------------------------------------------------------------
# Forked main() for --bcp on
# ---------------------------------------------------------------------------
def main_bcp_on(args: argparse.Namespace) -> int:
    import copy
    import csv
    import math
    import time

    from torch.utils.data import DataLoader, WeightedRandomSampler

    from dataset import (
        LabeledDataset,
        UnlabeledPairDataset,
        build_fg_balanced_weights,
        discover_samples,
        discover_unlabeled_images,
        sample_has_foreground,
        split_train_val_by_video,
    )
    from model_factory import BinaryFocalWithLogitsLoss, get_loss_fn, get_model
    from utils import (
        MetricRefs,
        ensure_dir,
        get_device,
        metric_quality_weighted,
        save_json,
        seed_everything,
        setup_logger,
    )

    import train as base_train  # baseline/task3/train.py helpers, reused verbatim

    seed_everything(int(args.seed))

    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    if image_size[0] % 32 != 0 or image_size[1] % 32 != 0:
        raise ValueError(f"image_size must be divisible by 32, got {image_size}")

    out_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    logger = setup_logger(out_dir, log_name="train.log")
    logger.info("Start BCP training (--bcp on)")
    logger.info("Args: %s", vars(args))

    labeled_root = Path(args.labeled_root)
    external_val_root = Path(args.external_val_root)
    unlabeled_root = Path(args.unlabeled_root)
    if not labeled_root.exists():
        raise FileNotFoundError(f"labeled_root not found: {labeled_root}")

    device = get_device()
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info("Device=%s AMP=%s", device, use_amp)

    all_labeled_raw = discover_samples(labeled_root)
    removed = sorted(
        f"{s.video_id}_{int(s.frame_idx):06d}"
        for s in all_labeled_raw
        if (s.video_id, int(s.frame_idx)) in EXCLUDED_FRAMES
    )

    if args.holdout_video:
        # LOVO: make_pinned_split is the SINGLE source of truth for BOTH the
        # noisy-frame exclusion AND the partition (it is installed identically
        # on the --bcp off path via monkey-patch), so pass the RAW sample list
        # straight through — no pre-exclusion here (single code path, no
        # double-apply).
        logger.info("LOVO holdout=%s | %d noisy frame(s) excluded inside pinned split: %s",
                    args.holdout_video, len(removed), removed)
        split_fn = make_pinned_split(args.holdout_video)
        train_samples, val_samples, train_video_ids, val_video_ids = split_fn(all_labeled_raw)
    else:
        # non-LOVO --bcp on: preserve the original behaviour (exclude, then the
        # baseline video split). This path is not the paired LOVO A/B.
        all_labeled = [s for s in all_labeled_raw if (s.video_id, int(s.frame_idx)) not in EXCLUDED_FRAMES]
        logger.info("Excluded %d noisy frame(s): %s", len(all_labeled_raw) - len(all_labeled), removed)
        train_samples, val_samples, train_video_ids, val_video_ids = split_train_val_by_video(
            all_labeled,
            val_video_count=int(args.val_video_count),
            seed=int(args.seed),
        )

    # Post-exclusion sample set actually used (for the split.json record).
    all_labeled = list(train_samples) + list(val_samples)
    val_samples_all = list(val_samples)
    if int(args.max_train_samples) > 0:
        train_samples = train_samples[: int(args.max_train_samples)]
    if int(args.max_val_samples) > 0:
        val_samples = val_samples[: int(args.max_val_samples)]
        val_samples_all = val_samples_all[: int(args.max_val_samples)]

    if bool(args.val_only_fg):
        val_samples = [s for s in val_samples if sample_has_foreground(s, target_label=int(args.target_label))]
    if len(val_samples) == 0:
        raise RuntimeError("Validation set is empty after foreground filtering.")

    unlabeled_paths = discover_unlabeled_images(unlabeled_root)
    if int(args.max_unlabeled_samples) > 0:
        unlabeled_paths = unlabeled_paths[: int(args.max_unlabeled_samples)]

    train_ds = LabeledDataset(
        samples=train_samples, image_size=image_size, target_label=int(args.target_label),
        train=True, cache_masks=bool(args.cache_masks), use_imagenet_norm=bool(args.use_imagenet_norm),
        seed=int(args.seed),
    )
    val_ds = LabeledDataset(
        samples=val_samples, image_size=image_size, target_label=int(args.target_label),
        train=False, cache_masks=bool(args.cache_masks), use_imagenet_norm=bool(args.use_imagenet_norm),
        seed=int(args.seed),
    )

    train_sampler = None
    if args.use_fg_balanced_sampling and len(train_ds) > 0:
        weights = build_fg_balanced_weights(
            train_ds.sample_fg_ratio, power=float(args.fg_sampling_power),
            min_weight=float(args.fg_sampling_min_weight), max_weight=float(args.fg_sampling_max_weight),
        )
        train_sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True,
            generator=torch.Generator().manual_seed(int(args.seed)),
        )

    train_loader = DataLoader(
        train_ds, batch_size=int(args.batch_size), shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=int(args.num_workers), pin_memory=True, persistent_workers=int(args.num_workers) > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, int(args.batch_size)), shuffle=False,
        num_workers=min(2, int(args.num_workers)), pin_memory=True, persistent_workers=int(args.num_workers) > 0,
    )

    unl_loader = None
    if unlabeled_paths:
        unl_ds = UnlabeledPairDataset(
            image_paths=unlabeled_paths, image_size=image_size,
            use_imagenet_norm=bool(args.use_imagenet_norm), seed=int(args.seed),
        )
        unl_loader = DataLoader(
            unl_ds, batch_size=max(1, int(args.unlabeled_batch_size)), shuffle=True,
            num_workers=min(2, int(args.num_workers)), pin_memory=True,
            persistent_workers=int(args.num_workers) > 0, drop_last=False,
        )

    logger.info(
        "Data: labeled all=%d train=%d val=%d unlabeled=%d | train_videos=%s val_videos=%s | bcp=%s beta=%.3f alpha=%.3f placement=%s",
        len(all_labeled), len(train_samples), len(val_samples), len(unlabeled_paths),
        train_video_ids, val_video_ids, args.bcp, args.bcp_beta, args.bcp_alpha, args.bcp_crop_placement,
    )

    encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights
    model = get_model(
        arch=args.arch, encoder_name=args.encoder_name, encoder_weights=encoder_weights,
        in_channels=3, classes=1,
    ).to(device)

    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    sup_loss_fn = get_loss_fn(
        loss_type=args.loss_type, dice_weight=float(args.dice_loss_weight), bce_weight=float(args.bce_loss_weight),
        focal_weight=float(args.focal_loss_weight), focal_gamma=float(args.focal_gamma),
        focal_alpha=float(args.focal_alpha), pos_weight=float(args.pos_weight),
    )
    if isinstance(sup_loss_fn, torch.nn.Module):
        sup_loss_fn = sup_loss_fn.to(device)

    # Verdict item 2/3: NOT a reuse of DiceFocalLoss "as-is" (smp.losses.DiceLoss
    # has no mask support). Only the focal component piece is reused, as a
    # SECOND instance with reduction="none" (same gamma/alpha/pos_weight as
    # sup_loss_fn's focal term, for loss-family consistency); the masked-Dice
    # primitive is new, in scripts/t3_bcp.py.
    bcp_focal_none = BinaryFocalWithLogitsLoss(
        gamma=float(args.focal_gamma), alpha=float(args.focal_alpha),
        pos_weight=float(args.pos_weight), reduction="none",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    warmup_epochs = max(0, min(int(args.warmup_epochs), max(0, int(args.epochs) - 1)))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(args.epochs) - warmup_epochs), eta_min=float(args.min_lr),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)), eta_min=float(args.min_lr))

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    # Per-fold audit record (mirrors baseline/task3/train.py:540-556 schema).
    # For a 6-fold LOVO run this is exactly the post-frame-exclusion train/val
    # video+sample provenance we want to keep. external_val is unused on the
    # BCP path, so those lists are empty.
    save_json(
        out_dir / "split.json",
        {
            "labeled_root": str(labeled_root),
            "external_val_root": str(external_val_root),
            "unlabeled_root": str(unlabeled_root),
            "val_only_fg": bool(args.val_only_fg),
            "all_labeled_samples": [s.image_path.as_posix() for s in all_labeled],
            "train_samples": [s.image_path.as_posix() for s in train_samples],
            "val_internal_all_samples": [s.image_path.as_posix() for s in val_samples_all],
            "val_internal_samples": [s.image_path.as_posix() for s in val_samples],
            "external_val_all_samples": [],
            "external_val_samples": [],
            "train_videos": train_video_ids,
            "val_videos": val_video_ids,
            "holdout_video": args.holdout_video,
            "excluded_frames": removed,
        },
    )
    save_json(out_dir / "config.json", vars(args))

    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "train_loss", "train_dice", "val_loss", "val_dice", "val_hd", "val_asd",
            "val_threshold", "score", "best_score", "lr", "epoch_sec",
        ])

    refs = MetricRefs(hd_ref=float(args.score_hd_ref), asd_ref=float(args.score_asd_ref))
    best_score = -1.0
    best_epoch = 0
    no_improve_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.time()
        model.train()
        teacher.eval()

        lambda_u = base_train.compute_unsup_weight(
            epoch=epoch, semi_warmup_epochs=int(args.semi_warmup_epochs),
            unsup_weight=float(args.unsup_weight), ramp_epochs=int(args.unsup_ramp_epochs),
        )

        sup_loss_sum = 0.0
        unsup_loss_sum = 0.0
        total_loss_sum = 0.0
        train_dice_sum = 0.0
        steps = 0

        train_iter = iter(train_loader)
        unl_iter = iter(unl_loader) if unl_loader is not None else None
        num_steps = len(train_loader)

        for step in range(1, num_steps + 1):
            sup_batch, train_iter = base_train.cycle_next(train_loader, train_iter)
            images = sup_batch["image"].to(device, non_blocking=True)
            labels = sup_batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits_sup = model(images)
                sup_loss = sup_loss_fn(logits_sup, labels)
            with torch.no_grad():
                batch_train_dice = base_train.dice_from_logits(logits_sup, labels, ignore_empty_gt=True)

            unsup_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

            if lambda_u > 0.0 and unl_loader is not None and unl_iter is not None:
                unl_batch, unl_iter = base_train.cycle_next(unl_loader, unl_iter)
                weak = unl_batch["weak"].to(device, non_blocking=True)

                # BCP REPLACES the FixMatch consistency branch here, gated on
                # the identical `lambda_u > 0` condition (verdict item 9).
                unsup_loss, _pos_ratio, _conf_ratio, _invoked = bcp_train_step(
                    model=model, teacher=teacher, images=images, labels=labels, weak=weak,
                    focal_none_fn=bcp_focal_none, lambda_u=lambda_u, device=device, use_amp=use_amp,
                    beta=float(args.bcp_beta), placement=str(args.bcp_crop_placement), alpha=float(args.bcp_alpha),
                    dice_weight=float(args.dice_loss_weight), focal_weight=float(args.focal_loss_weight),
                    pseudo_pos_thr=float(args.pseudo_pos_thr), pseudo_neg_thr=float(args.pseudo_neg_thr),
                    pseudo_min_area=float(args.pseudo_min_area), pseudo_min_pos_ratio=float(args.pseudo_min_pos_ratio),
                )

            total_loss = sup_loss + float(lambda_u) * unsup_loss

            if scaler is not None:
                scaler.scale(total_loss).backward()
                if float(args.grad_clip_norm) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if float(args.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
                optimizer.step()

            base_train.update_ema(teacher, model, decay=float(args.ema_decay))

            sup_loss_sum += float(sup_loss.item())
            unsup_loss_sum += float(unsup_loss.item())
            total_loss_sum += float(total_loss.item())
            train_dice_sum += float(batch_train_dice)
            steps += 1

            if int(args.print_freq) > 0 and (step % int(args.print_freq) == 0 or step == num_steps):
                logger.info(
                    "Epoch %d Step %d/%d | lambda_u=%.3f | sup=%.4f unsup=%.4f total=%.4f",
                    epoch, step, num_steps, lambda_u,
                    sup_loss_sum / max(1, steps), unsup_loss_sum / max(1, steps), total_loss_sum / max(1, steps),
                )

        eval_model = model
        val_metrics = base_train.evaluate(
            model=eval_model, loader=val_loader, loss_fn=sup_loss_fn, device=device, use_amp=use_amp,
            threshold_candidates=args.threshold_candidates, use_tta=bool(args.val_tta), fixed_threshold=None,
        )

        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_sec = time.time() - epoch_start

        quality = metric_quality_weighted(
            dsc=val_metrics["val_dice"], hd=val_metrics["val_hd"], asd=val_metrics["val_asd"], refs=refs,
            dsc_weight=float(args.score_dsc_weight), hd_weight=float(args.score_hd_weight), asd_weight=float(args.score_asd_weight),
        )
        score = float(quality["score"])

        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "teacher_state": teacher.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                 "args": vars(args), "val_metrics": val_metrics, "score": score},
                ckpt_dir / "best.pt",
            )
        else:
            no_improve_epochs += 1

        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "teacher_state": teacher.state_dict(),
             "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
             "args": vars(args), "val_metrics": val_metrics, "score": score},
            ckpt_dir / "last.pt",
        )
        if int(args.save_every) > 0 and epoch % int(args.save_every) == 0:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "teacher_state": teacher.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                 "args": vars(args), "val_metrics": val_metrics, "score": score},
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

        with history_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, f"{total_loss_sum / max(1, steps):.6f}", f"{train_dice_sum / max(1, steps):.6f}",
                f"{val_metrics['val_loss']:.6f}", f"{val_metrics['val_dice']:.6f}", f"{val_metrics['val_hd']:.6f}",
                f"{val_metrics['val_asd']:.6f}", f"{val_metrics['val_threshold']:.4f}", f"{score:.6f}",
                f"{best_score:.6f}", f"{lr_now:.8f}", f"{epoch_sec:.2f}",
            ])

        logger.info(
            "Epoch %d/%d | lambda_u=%.3f | train(sup/unsup/total)=%.4f/%.4f/%.4f train_dice=%.4f | val_dice=%.4f | score=%.4f best=%.4f(epoch=%d) | %.1fs",
            epoch, int(args.epochs), lambda_u, sup_loss_sum / max(1, steps), unsup_loss_sum / max(1, steps),
            total_loss_sum / max(1, steps), train_dice_sum / max(1, steps), val_metrics["val_dice"],
            score, best_score, best_epoch, epoch_sec,
        )

        scheduler.step()

        if int(args.early_stop_patience) > 0 and no_improve_epochs >= int(args.early_stop_patience):
            logger.info("Early stop at epoch %d", epoch)
            break

    save_json(out_dir / "final_metrics.json", {"best_epoch": best_epoch, "best_score": best_score})
    logger.info("BCP training finished.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    add_bcp_args(pre)
    bcp_args, rest = pre.parse_known_args()

    if bcp_args.bcp == "off":
        # Clean-attribution off-path (verdict item 6 / task's "Run it" smoke
        # test): dispatch the leftover argv straight to baseline/task3/train.py's
        # own main(). No BCP code is imported or executed on this path.
        import train  # baseline/task3/train.py

        if bcp_args.holdout_video:
            # LOVO fold-pinning for the OFF baseline (same monkey-patch as
            # scripts/train_t3_temporal.py:58-59), so the paired A/B fold holds
            # out the identical video.
            train.split_train_val_by_video = make_pinned_split(bcp_args.holdout_video)
            rest = rest + ["--val-video-count", "1"]
            print(f"[train_t3_bcp] BCP OFF | LOVO holdout={bcp_args.holdout_video}")
        else:
            print("[train_t3_bcp] BCP OFF | default val split")

        sys.argv = [sys.argv[0]] + rest
        return train.main()

    full_parser = argparse.ArgumentParser(description="Task3 BCP Mean-Teacher training (--bcp on)")
    add_train_args(full_parser)
    add_bcp_args(full_parser)
    args = full_parser.parse_args(rest)
    return main_bcp_on(args)


if __name__ == "__main__":
    raise SystemExit(main())
