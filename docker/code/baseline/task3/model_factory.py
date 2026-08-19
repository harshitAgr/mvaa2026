#!/usr/bin/env python3
"""Model and loss factory for task3 2D segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
except Exception as e:  # pragma: no cover
    smp = None
    _SMP_IMPORT_ERROR = e
else:
    _SMP_IMPORT_ERROR = None


def get_model(
    arch: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: str | None = None,
    in_channels: int = 3,
    classes: int = 1,
):
    if smp is None:
        raise ImportError(
            "segmentation_models_pytorch is required but not installed. "
            f"Original import error: {_SMP_IMPORT_ERROR!r}"
        )

    arch = arch.lower()
    if arch == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if arch == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if arch == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if arch == "segformer":
        # Pure transformer encoder (MiT) + all-MLP decoder. Handles the
        # 448x800 T3 input natively (patch-based, no fixed img_size grid).
        # Verified: smp.Segformer(encoder_name="mit_b3", encoder_weights=None)
        # forward(1,3,448,800) -> (1,1,448,800).
        return smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if arch == "metaformerfpn":
        # CaFormer-S18 (MetaFormer hybrid: conv token-mixers stages 0-1, self-attention
        # stages 2-3) + FPN decoder, vendored upstream via baseline/task3/surgenet_adapter.py
        # (pinned upstream copy under third_party/surgenet -- see the README).
        # pretrained='SurgeNet' selects the plain-ReLU activation config the SurgeNetXL
        # checkpoint was trained with (stock timm caformer_s18 defaults to StarReLU, which
        # is WRONG for this checkpoint). This branch
        # never downloads weights (pretrained_weights=None); random init only. The trained
        # SurgeNetXL teacher checkpoint is loaded separately into the encoder via
        # surgenet_adapter.load_surgenet_into_model (see scripts/train_t3_caformerxl.py).
        # `encoder_name`/`encoder_weights` are ignored here (upstream MetaFormerFPN has no
        # such knobs); callers should still record them as "caformer_s18"/"surgenetxl" in
        # their ckpt args for provenance only. Upstream's MetaFormerFPN also has no
        # `in_channels` knob (always 3 -- RGB); this pipeline's T3 frames are already RGB,
        # so we just assert that rather than silently ignoring a mismatch.
        if in_channels != 3:
            raise ValueError(
                f"arch=metaformerfpn only supports in_channels=3 (upstream MetaFormerFPN has "
                f"no in_channels parameter), got in_channels={in_channels}"
            )
        from surgenet_adapter import MetaFormerFPN  # local import — needs sys.path set up

        return MetaFormerFPN(
            num_classes=classes,
            pretrained="SurgeNet",
            pretrained_weights=None,
        )
    if arch == "upernet":
        extra_kwargs: dict = {}
        if "swin" in str(encoder_name).lower():
            # timm Swin encoders bake in a fixed img_size grid for their
            # window-based position handling and assert on any other input
            # size. This repo's T3 pipeline always uses a fixed 448x800 input
            # (baseline/task3/train.py --image-size default, enforced
            # divisible-by-32), so pass it through explicitly together with
            # dynamic_img_size=True (timm's mechanism for non-224/non-square
            # inputs) rather than adding a new get_model parameter. Verified:
            # without these two kwargs, tu-swin_tiny_patch4_window7_224 raises
            # "AssertionError: Input height (448) doesn't match model (224)".
            extra_kwargs = {"img_size": (448, 800), "dynamic_img_size": True}
        return smp.UPerNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **extra_kwargs,
        )
    raise ValueError(f"Unsupported architecture: {arch}")


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.7,
        bce_weight: float = 0.3,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        if smp is None:
            raise ImportError(
                "segmentation_models_pytorch is required but not installed. "
                f"Original import error: {_SMP_IMPORT_ERROR!r}"
            )
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        if pos_weight is None:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * self.bce(logits, targets)


class BinaryFocalWithLogitsLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = 0.75,
        pos_weight: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.pos_weight = None if pos_weight is None else float(pos_weight)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = ((1.0 - pt).clamp_min(1e-6) ** self.gamma) * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = loss * alpha_t

        if self.pos_weight is not None and self.pos_weight != 1.0:
            pos_w = 1.0 + (self.pos_weight - 1.0) * targets
            loss = loss * pos_w

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.7,
        focal_weight: float = 0.3,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = 0.75,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        if smp is None:
            raise ImportError(
                "segmentation_models_pytorch is required but not installed. "
                f"Original import error: {_SMP_IMPORT_ERROR!r}"
            )
        self.dice_weight = float(dice_weight)
        self.focal_weight = float(focal_weight)
        self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        self.focal = BinaryFocalWithLogitsLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            pos_weight=pos_weight,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.focal_weight * self.focal(logits, targets)


def get_loss_fn(
    loss_type: str = "dice_focal",
    dice_weight: float = 0.7,
    bce_weight: float = 0.3,
    focal_weight: float = 0.3,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = 0.75,
    pos_weight: float | None = None,
) -> nn.Module:
    loss_type = str(loss_type).lower()
    if loss_type == "dice_bce":
        return DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight, pos_weight=pos_weight)
    if loss_type == "dice_focal":
        return DiceFocalLoss(
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            pos_weight=pos_weight,
        )
    raise ValueError(f"Unsupported loss_type: {loss_type}")
