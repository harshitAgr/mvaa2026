"""Pure-logic module for the Task3 Bidirectional Copy-Paste (BCP) Mean-Teacher
upgrade. No baseline/task3 imports here (importable standalone, unit-testable
on CPU with tiny synthetic tensors).

Bidirectional copy-paste mixing utilities.
(the "Sonnet design-review verdict" section holds the LOCKED decisions this
module implements; item numbers referenced in comments below refer to that
section).

Mechanism reference (official BCP repo, code-verified 2026-07-08):
  github.com/DeepMed-Lab-ECNU/BCP, ACDC_BCP_train.py:131-140 (generate_mask),
  utils/losses.py:102-111 (DiceLoss._dice_mask_loss), utils/BCP_utils.py:58-69
  (mix_loss). This module mirrors that logic, adapted to binary (sigmoid,
  1-channel) segmentation instead of multiclass softmax, per verdict item 2/3.

Mask convention (matches the official code exactly): mask == 1 on the OUTER
border region, mask == 0 on the INNER pasted patch. "Zero-centered" in the
paper refers to this value convention, not necessarily a centered position;
default crop placement is RANDOM (verdict item 14 / code-verified).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

DEFAULT_BETA = 0.667
DEFAULT_ALPHA = 0.5


# ---------------------------------------------------------------------------
# 1. Mask generation
# ---------------------------------------------------------------------------
def generate_bcp_mask(
    h: int,
    w: int,
    batch_size: int = 1,
    beta: float = DEFAULT_BETA,
    placement: str = "random",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Build the BCP rectangle mask.

    Mirrors `generate_mask` (ACDC_BCP_train.py:131-140): a single rectangle of
    size `patch_h, patch_w = int(H*beta), int(W*beta)`; mask == 1 outside the
    patch (border), 0 inside (the pasted region).

    placement:
      - "random" (default, code-verified `np.random.randint` top-left) — the
        actual official behaviour and this repo's default (verdict item 14).
      - "center" — deterministic, ablation-only option named in the spec.

    Returns
    -------
    mask_hw : (H, W) float32 tensor, values in {0., 1.}. Broadcasts directly
        against any (..., H, W) tensor (images with a channel dim, labels,
        logits) via standard right-aligned torch broadcasting.
    loss_mask_bhw : (batch_size, H, W) float32 tensor — mask_hw broadcast
        across the batch dim, for callers that want an explicitly batched
        loss mask (mirrors the official `loss_mask` return value).
    geom : dict with patch bbox (top, left, patch_h, patch_w) for testing.
    """
    if placement not in ("random", "center"):
        raise ValueError(f"Unsupported placement: {placement!r}")
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}")

    patch_h = int(h * beta)
    patch_w = int(w * beta)
    patch_h = max(1, min(patch_h, h))
    patch_w = max(1, min(patch_w, w))

    if placement == "center":
        top = (h - patch_h) // 2
        left = (w - patch_w) // 2
    else:
        if rng is None:
            rng = np.random.default_rng()
        top = int(rng.integers(0, h - patch_h + 1)) if h - patch_h > 0 else 0
        left = int(rng.integers(0, w - patch_w + 1)) if w - patch_w > 0 else 0

    mask_hw = torch.ones((h, w), dtype=torch.float32)
    mask_hw[top : top + patch_h, left : left + patch_w] = 0.0

    loss_mask_bhw = mask_hw.unsqueeze(0).expand(int(batch_size), -1, -1).clone()
    geom = {"top": top, "left": left, "patch_h": patch_h, "patch_w": patch_w, "h": h, "w": w}
    return mask_hw, loss_mask_bhw, geom


# ---------------------------------------------------------------------------
# 2. Pairing
# ---------------------------------------------------------------------------
def roll_pair(x: torch.Tensor, shifts: int = 1, dims: int = 0) -> torch.Tensor:
    """`torch.roll` pairing (verdict item 7, overrides the batch-split the
    spec briefly proposed to "match the official code exactly"): keeps the
    full batch as both anchors and partners, trivially guarantees i != j /
    p != q for batch size >= 2, and does not shrink our already-tuned
    labeled/unlabeled batch sizes the way a batch-split would."""
    return torch.roll(x, shifts=shifts, dims=dims)


def can_bcp_mix(labeled_bs: int, unlabeled_bs: int, min_bs: int = 2) -> bool:
    """Verdict item 12: both loaders must be checked independently, every
    step, since drop_last=False lets either yield a short last batch on any
    step (not necessarily synchronized)."""
    return int(labeled_bs) >= int(min_bs) and int(unlabeled_bs) >= int(min_bs)


# ---------------------------------------------------------------------------
# 3. Compositing (image mixing AND label mixing use the SAME mask instance)
# ---------------------------------------------------------------------------
def composite(x_outer: torch.Tensor, x_inner: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
    """`x_outer*mask + x_inner*(1-mask)`. `mask_hw` (H,W) broadcasts against
    any (..., H, W) tensor (images with 3 channels, labels/logits with 1)."""
    return x_outer * mask_hw + x_inner * (1.0 - mask_hw)


# ---------------------------------------------------------------------------
# 4. Pseudo-label + confidence gate (mirrors train.py:639-653 verbatim logic)
# ---------------------------------------------------------------------------
def pseudo_label_from_probs(
    t_probs: torch.Tensor,
    pos_thr: float,
    neg_thr: float,
    min_area: float,
    min_pos_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Reuse of this repo's already-tuned confidence-masked pseudo-labeling
    (verdict item 8, RATIFIED — not the paper's flat 0.5 + largest-CC).

    Returns (pseudo, conf_mask, pseudo_pos_ratio), same semantics as
    baseline/task3/train.py's inline block:
      - pseudo = 1 where t_probs >= pos_thr, else 0.
      - conf_mask = 1 where teacher is confident (>=pos_thr or <=neg_thr).
      - speck filter: pseudo blobs smaller than min_area are zeroed and their
        conf_mask reverts to the plain low-confidence gate.
      - collapse guard: if global pseudo positive ratio < min_pos_ratio,
        conf_mask is zeroed entirely (no unsupervised signal that step).
    """
    pseudo = (t_probs >= float(pos_thr)).float()
    conf_mask = ((t_probs >= float(pos_thr)) | (t_probs <= float(neg_thr))).float()
    pseudo_pos_ratio = float(pseudo.mean().item())

    if float(min_area) > 0:
        area = pseudo.flatten(1).sum(dim=1)
        small = area < float(min_area)
        if small.any():
            pseudo[small] = 0.0
            conf_mask[small] = (t_probs[small] <= float(neg_thr)).float()
        pseudo_pos_ratio = float(pseudo.mean().item())

    if pseudo_pos_ratio < float(min_pos_ratio):
        conf_mask = torch.zeros_like(conf_mask)

    return pseudo, conf_mask, pseudo_pos_ratio


# ---------------------------------------------------------------------------
# 5. Masked binary loss primitives (verdict item 2/3 — smp.losses.DiceLoss
#    has NO mask support, so these are new, not a reuse of DiceFocalLoss)
# ---------------------------------------------------------------------------
def masked_soft_dice(
    probs: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Binary mask-aware soft-Dice, mirrors `DiceLoss._dice_mask_loss`
    (utils/losses.py:102-111) adapted to sigmoid/1-channel:
        1 - (2*sum(p*y*m) + eps) / (sum(p^2*m) + sum(y^2*m) + eps)
    `eps=1e-6` (not the paper's 1e-10 — matches this repo's own
    `dice_from_preds` eps convention, train.py:141, given fp16/autocast
    rounding at 448x800 px). Sums are upcast to float32 (verdict item 5,
    "Autocast/fp16 precision") — callers should already pass float32 tensors
    (sigmoid(logits.float())), this function additionally re-casts for safety.
    Degenerate all-empty-mask case is handled by `eps` alone (dice -> 1,
    loss -> 0); this is the standard Milletari-style softening, NOT a
    `mask.sum()+eps` normalization — those are different mechanisms
    (verdict item 5, do not conflate).
    """
    p = probs.float()
    y = target.float()
    m = mask.float()
    inter = torch.sum(p * y * m, dtype=torch.float32)
    z_sum = torch.sum(p * p * m, dtype=torch.float32)
    y_sum = torch.sum(y * y * m, dtype=torch.float32)
    dice = (2.0 * inter + eps) / (z_sum + y_sum + eps)
    return 1.0 - dice


def masked_mean_skip_empty(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean of a per-pixel loss map, skip-if-empty (verdict item 5:
    "use the skip-if-empty pattern already established at train.py:658-660
    ... not a blind +eps division" — dividing by eps when a region has 0-1
    confident pixels can blow up the gradient of that single pixel).
    Returns a 0-valued scalar tensor (same dtype/device as loss_map) when the
    mask sums to (numerically) zero, with NO division performed."""
    m = mask.float()
    total = m.sum()
    if float(total.item()) <= 0.0:
        return torch.zeros((), dtype=loss_map.dtype, device=loss_map.device)
    return (loss_map.float() * m).sum(dtype=torch.float32) / total


# ---------------------------------------------------------------------------
# 6. Per-direction combined BCP loss
# ---------------------------------------------------------------------------
def bcp_direction_loss(
    logits: torch.Tensor,
    composited_label: torch.Tensor,
    mask_gt_side: torch.Tensor,
    mask_pseudo_side_eff: torch.Tensor,
    focal_none_fn,
    alpha: float = DEFAULT_ALPHA,
    dice_weight: float = 0.7,
    focal_weight: float = 0.3,
    dice_eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One BCP direction's mixed-image loss (verdict item 2):
        dice_dir  = dice_region(p, Y, mask_gt_side)*1.0
                  + dice_region(p, Y, mask_pseudo_side_eff)*alpha
        focal_dir = masked_mean(focal_map, mask_gt_side)*1.0
                  + masked_mean(focal_map, mask_pseudo_side_eff)*alpha
        L_dir = dice_weight*dice_dir + focal_weight*focal_dir

    `composited_label` is the SAME single tensor already composited through
    the geometric mask (GT value in the GT region, pseudo value in the pseudo
    region) — algebraically identical to passing two separate label tensors
    each masked by its own region (since both masked reductions multiply by
    a mask that zeroes out the other region's contribution regardless of
    what value sits there), so a single composited label is used for both
    region terms, matching "composite labels through the SAME mask instance."

    `mask_pseudo_side_eff` MUST already be the geometric-region-mask times
    the confidence mask (verdict item 4) — i.e. low-confidence pseudo pixels
    are excluded from `mask_pseudo_side_eff` before this function is called,
    so they contribute exactly 0 to both terms here (not merely down-weighted
    by alpha).

    `focal_none_fn` must be a `reduction="none"` per-pixel loss callable
    (logits, target) -> per-pixel map of the same shape as `logits`.
    """
    probs = torch.sigmoid(logits.float())
    y = composited_label.float()

    dice_gt = masked_soft_dice(probs, y, mask_gt_side, eps=dice_eps)
    dice_pseudo = masked_soft_dice(probs, y, mask_pseudo_side_eff, eps=dice_eps)
    dice_dir = dice_gt * 1.0 + dice_pseudo * float(alpha)

    loss_map = focal_none_fn(logits, y)
    focal_gt = masked_mean_skip_empty(loss_map, mask_gt_side)
    focal_pseudo = masked_mean_skip_empty(loss_map, mask_pseudo_side_eff)
    focal_dir = focal_gt * 1.0 + focal_pseudo * float(alpha)

    loss_dir = float(dice_weight) * dice_dir + float(focal_weight) * focal_dir
    stats = {
        "dice_gt": dice_gt.detach(),
        "dice_pseudo": dice_pseudo.detach(),
        "focal_gt": focal_gt.detach(),
        "focal_pseudo": focal_pseudo.detach(),
    }
    return loss_dir, stats
