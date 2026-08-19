#!/usr/bin/env python3
"""Pure model and UniMatch-V2 helpers for the Task-3 DINOv2 kill stage.

This module is intentionally independent of ``baseline/task3``.  It provides
the frozen DINOv2-S/14 + DPT-small binary segmenter and the small, testable
operations needed by the training wrapper.  It never downloads weights.
"""
from __future__ import annotations

import hashlib
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DINOV2_WEIGHTS = REPO_ROOT / "data" / "pretrained" / "dinov2_vits14_lvd142m.safetensors"
BACKBONE_NAME = "vit_small_patch14_dinov2"
PATCH_SIZE = 14
INTERMEDIATE_BLOCKS = (2, 5, 8, 11)
BACKBONE_CHANNELS = 384


def load_dinov2_small_backbone(
    weights_path: str | Path,
    freeze: bool = True,
) -> nn.Module:
    """Build plain timm DINOv2-S and strictly load a local safetensors file.

    ``features_only=True`` is deliberately not used: that wrapper drops the
    final norm parameters and cannot strictly load the official state dict.
    Dynamic image size permits both 448x798 and its 798x448 D4 rotation; this
    module separately requires both dimensions to be divisible by patch 14.
    """

    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"DINOv2 weights not found: {path}")
    if path.suffix != ".safetensors":
        raise ValueError(f"DINOv2 weights must be local safetensors, got: {path}")

    backbone = timm.create_model(
        BACKBONE_NAME,
        pretrained=False,
        dynamic_img_size=True,
    )
    state = load_file(str(path), device="cpu")
    backbone.load_state_dict(state, strict=True)
    if freeze:
        backbone.requires_grad_(False)
        backbone.eval()
    return backbone


class ResidualConvUnit(nn.Module):
    """Residual unit used by the official UniMatch-V2 DPT head."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(x, inplace=False))
        out = self.conv2(F.relu(out, inplace=False))
        return out + x


class FeatureFusionBlock(nn.Module):
    """DPT residual fusion followed by explicit-size bilinear upsampling."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.residual_skip = ResidualConvUnit(features)
        self.residual_out = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if skip is not None:
            x = x + self.residual_skip(skip)
        x = self.residual_out(x)
        if size is None:
            size = (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class DPTSmallBinaryHead(nn.Module):
    """DPT-small decoder for four 384-channel DINOv2 token maps."""

    def __init__(
        self,
        in_channels: int = BACKBONE_CHANNELS,
        features: int = 64,
        projection_channels: Sequence[int] = (48, 96, 192, 384),
    ) -> None:
        super().__init__()
        if len(projection_channels) != 4:
            raise ValueError("DPT head requires exactly four projection widths")
        widths = tuple(int(v) for v in projection_channels)
        self.projects = nn.ModuleList([nn.Conv2d(in_channels, width, 1) for width in widths])
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(widths[0], widths[0], kernel_size=4, stride=4),
                nn.ConvTranspose2d(widths[1], widths[1], kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(widths[3], widths[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.scratch = nn.ModuleList(
            [nn.Conv2d(width, features, 3, padding=1, bias=False) for width in widths]
        )
        self.fuse4 = FeatureFusionBlock(features)
        self.fuse3 = FeatureFusionBlock(features)
        self.fuse2 = FeatureFusionBlock(features)
        self.fuse1 = FeatureFusionBlock(features)
        self.output = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, 1, 1),
        )

    def forward(self, feature_maps: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feature_maps) != 4:
            raise ValueError(f"Expected four DINOv2 feature maps, got {len(feature_maps)}")
        projected = [resize(project(feat)) for feat, project, resize in zip(
            feature_maps, self.projects, self.resize_layers
        )]
        layer1, layer2, layer3, layer4 = [
            scratch(feat) for scratch, feat in zip(self.scratch, projected)
        ]
        path4 = self.fuse4(layer4, size=layer3.shape[-2:])
        path3 = self.fuse3(path4, layer3, size=layer2.shape[-2:])
        path2 = self.fuse2(path3, layer2, size=layer1.shape[-2:])
        path1 = self.fuse1(path2, layer1)
        return self.output(path1)


def complementary_feature_dropout(
    feature_maps: Sequence[torch.Tensor],
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, ...]:
    """Apply paired complementary channel dropout to concatenated 2B views.

    Inputs must be ordered ``[view1 batch, view2 batch]``.  Exactly half of
    the sample pairs are unchanged; the rest receive complementary Bernoulli
    channel masks whose retained activations are scaled by two.  One channel
    mask is shared across all four feature levels, as in UniMatch V2.
    """

    if not feature_maps:
        raise ValueError("feature_maps cannot be empty")
    batch2, channels = feature_maps[0].shape[:2]
    if batch2 % 2 != 0:
        raise ValueError(f"Complementary dropout requires concatenated 2B batch, got {batch2}")
    pairs = batch2 // 2
    if pairs == 0 or pairs % 2 != 0:
        raise ValueError(f"Exactly-half unchanged policy requires a positive even pair batch, got {pairs}")
    for feat in feature_maps:
        if feat.ndim != 4 or feat.shape[0] != batch2 or feat.shape[1] != channels:
            raise ValueError("All feature maps must be NCHW with identical batch/channel dimensions")

    device = feature_maps[0].device
    random_values = torch.rand((pairs, channels), device=device, generator=generator)
    mask1 = (random_values >= 0.5).to(feature_maps[0].dtype) * 2.0
    mask2 = 2.0 - mask1
    unchanged = torch.randperm(pairs, device=device, generator=generator)[: pairs // 2]
    mask1[unchanged] = 1.0
    mask2[unchanged] = 1.0
    mask = torch.cat((mask1, mask2), dim=0).unsqueeze(-1).unsqueeze(-1)
    return tuple(feat * mask.to(dtype=feat.dtype) for feat in feature_maps)


class DinoV2DPTSegmenter(nn.Module):
    """Frozen DINOv2-S/14 backbone with a DPT-small binary head."""

    def __init__(
        self,
        weights_path: str | Path | None = DEFAULT_DINOV2_WEIGHTS,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if weights_path is None:
            self.backbone = timm.create_model(
                BACKBONE_NAME,
                pretrained=False,
                dynamic_img_size=True,
            )
            if freeze_backbone:
                self.backbone.requires_grad_(False)
                self.backbone.eval()
        else:
            self.backbone = load_dinov2_small_backbone(weights_path, freeze=freeze_backbone)
        self.head = DPTSmallBinaryHead()
        self.freeze_backbone = bool(freeze_backbone)

    @property
    def decoder(self) -> DPTSmallBinaryHead:
        """Optimizer-facing alias without duplicate module registration."""
        return self.head

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def _validate_shape(x: torch.Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected RGB NCHW input, got {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if h % PATCH_SIZE != 0 or w % PATCH_SIZE != 0:
            raise ValueError(f"Input H/W must be divisible by patch size {PATCH_SIZE}, got {(h, w)}")

    def _features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # For an unfrozen model, preserve the caller's ambient grad mode.  In
        # particular, EMA teacher/evaluator inference wraps the call in
        # torch.no_grad(); forcing enable_grad here would construct a full ViT
        # graph and can exhaust a 12 GB V100.
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            outputs = self.backbone.forward_intermediates(
                x,
                indices=INTERMEDIATE_BLOCKS,
                norm=True,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        return tuple(outputs)

    def forward(
        self,
        x: torch.Tensor,
        comp_drop: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        self._validate_shape(x)
        features = self._features(x)
        if comp_drop:
            features = complementary_feature_dropout(features, generator=generator)
        logits = self.head(features)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=True)

    def forward_paired_strong(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if view1.shape != view2.shape:
            raise ValueError(f"Strong views must have identical shapes, got {view1.shape}/{view2.shape}")
        logits = self(torch.cat((view1, view2), dim=0), comp_drop=True, generator=generator)
        return logits.chunk(2, dim=0)


def make_binary_pseudo_targets(
    weak_logits: torch.Tensor,
    confidence_threshold: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hard pseudo target, binary confidence, and float validity mask."""

    if not 0.5 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence_threshold must be in [0.5, 1.0]")
    probabilities = torch.sigmoid(weak_logits)
    pseudo = (probabilities >= 0.5).to(weak_logits.dtype)
    confidence = torch.maximum(probabilities, 1.0 - probabilities)
    valid = (confidence >= float(confidence_threshold)).to(weak_logits.dtype)
    return pseudo, confidence, valid


def binary_pseudo_labels(
    probabilities: torch.Tensor,
    threshold: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Probability-input compatibility API for the frozen trainer."""
    if not 0.5 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0.5, 1.0]")
    pseudo = (probabilities >= 0.5).to(probabilities.dtype)
    confidence = torch.maximum(probabilities, 1.0 - probabilities)
    valid = (confidence >= float(threshold)).to(probabilities.dtype)
    return pseudo, confidence, valid


def sample_cutmix_mask(
    batch_size: int,
    height: int,
    width: int,
    probability: float = 0.5,
    area_range: Tuple[float, float] = (0.02, 0.40),
    aspect_range: Tuple[float, float] = (0.3, 10.0 / 3.0),
    device: torch.device | str | None = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample independent rectangular CutMix masks for a non-square batch."""

    if batch_size <= 0 or height <= 0 or width <= 0:
        raise ValueError("batch_size, height, and width must be positive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    amin, amax = map(float, area_range)
    rmin, rmax = map(float, aspect_range)
    if not (0.0 < amin <= amax <= 1.0) or not (0.0 < rmin <= rmax):
        raise ValueError("Invalid CutMix area/aspect range")

    device = torch.device("cpu") if device is None else torch.device(device)
    masks = torch.zeros((batch_size, 1, height, width), dtype=torch.bool, device=device)
    enabled = torch.rand(batch_size, device=device, generator=generator) < probability
    log_min, log_max = math.log(rmin), math.log(rmax)
    for i in torch.nonzero(enabled, as_tuple=False).flatten().tolist():
        target_area = float(torch.empty((), device=device).uniform_(amin, amax, generator=generator))
        aspect = math.exp(float(torch.empty((), device=device).uniform_(log_min, log_max, generator=generator)))
        box_h = max(1, min(height, int(round(math.sqrt(target_area * height * width / aspect)))))
        box_w = max(1, min(width, int(round(math.sqrt(target_area * height * width * aspect)))))
        top = int(torch.randint(0, height - box_h + 1, (), device=device, generator=generator))
        left = int(torch.randint(0, width - box_w + 1, (), device=device, generator=generator))
        masks[i, 0, top : top + box_h, left : left + box_w] = True
    return masks


def apply_aligned_cutmix(
    images: torch.Tensor,
    pseudo: torch.Tensor,
    confidence: torch.Tensor,
    cutmix_mask: torch.Tensor,
    source_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CutMix images, pseudo labels, and confidence with one identical mask."""

    if images.ndim != 4 or pseudo.ndim != 4 or confidence.ndim != 4:
        raise ValueError("images, pseudo, and confidence must all be NCHW")
    batch, _, height, width = images.shape
    if batch < 2:
        raise ValueError("Aligned CutMix requires batch size >= 2")
    if pseudo.shape != confidence.shape or pseudo.shape[0] != batch or pseudo.shape[-2:] != (height, width):
        raise ValueError("Pseudo/confidence dimensions must align with images")
    if cutmix_mask.shape != (batch, 1, height, width):
        raise ValueError(f"Expected CutMix mask {(batch, 1, height, width)}, got {tuple(cutmix_mask.shape)}")
    if source_indices is None:
        source_indices = torch.roll(torch.arange(batch, device=images.device), shifts=1)
    else:
        source_indices = source_indices.to(device=images.device, dtype=torch.long)
    if source_indices.shape != (batch,):
        raise ValueError(f"source_indices must have shape {(batch,)}, got {tuple(source_indices.shape)}")
    if torch.any(source_indices < 0) or torch.any(source_indices >= batch):
        raise ValueError("source_indices out of range")

    mask = cutmix_mask.to(device=images.device, dtype=torch.bool)
    mixed_images = torch.where(mask.expand_as(images), images[source_indices], images)
    mixed_pseudo = torch.where(mask.expand_as(pseudo), pseudo[source_indices], pseudo)
    mixed_confidence = torch.where(mask.expand_as(confidence), confidence[source_indices], confidence)
    return mixed_images, mixed_pseudo, mixed_confidence


def cutmix_batch(
    images: torch.Tensor,
    pseudo: torch.Tensor,
    confidence: torch.Tensor,
    valid_mask: torch.Tensor,
    probability: float = 0.5,
    area_range: Tuple[float, float] = (0.02, 0.40),
    aspect_range: Tuple[float, float] = (0.3, 10.0 / 3.0),
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one CutMix box set and align all four unlabeled tensors."""
    mask = sample_cutmix_mask(
        images.shape[0], images.shape[-2], images.shape[-1],
        probability=probability, area_range=area_range, aspect_range=aspect_range,
        device=images.device, generator=generator,
    )
    source = torch.roll(torch.arange(images.shape[0], device=images.device), shifts=1)
    mixed_images, mixed_pseudo, mixed_confidence = apply_aligned_cutmix(
        images, pseudo, confidence, mask, source,
    )
    _, mixed_valid, _ = apply_aligned_cutmix(
        images, valid_mask, confidence, mask, source,
    )
    return mixed_images, mixed_pseudo, mixed_confidence, mixed_valid


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Confidence-masked binary BCE with a finite differentiable empty case."""

    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, targets, and valid_mask must have identical shapes")
    loss = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
    valid = valid_mask.to(logits.dtype)
    # UniMatch-V2 divides by all non-ignore pixels, not confident pixels.
    # There are no ignore pixels in T3, so this is an ordinary full-map mean;
    # an all-invalid map naturally remains a differentiable exact zero.
    return (loss * valid).mean()


def confidence_masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Frozen trainer-facing name for all-pixel-normalized masked BCE."""
    return masked_bce_with_logits(logits, targets, valid_mask)


def build_model(
    weights_path: str | Path,
    expected_sha256: Optional[str] = None,
    image_size: Tuple[int, int] = (448, 798),
) -> DinoV2DPTSegmenter:
    """Build the frozen model after local artifact and geometry verification."""
    path = Path(weights_path)
    if expected_sha256 is not None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != str(expected_sha256):
            raise ValueError(f"DINOv2 SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    if tuple(image_size) != (448, 798):
        raise ValueError(f"Frozen DINOv2 image_size must be (448, 798), got {tuple(image_size)}")
    return DinoV2DPTSegmenter(weights_path=path, freeze_backbone=True)


def load_dinov2_candidate_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str,
):
    """Load a self-contained candidate checkpoint using the evaluator tuple API."""

    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Candidate checkpoint is not a mapping: {path}")
    if int(checkpoint.get("epoch", -1)) != 70:
        raise ValueError(f"Eligible candidate must be fixed epoch 70: {path}")
    if checkpoint.get("candidate_state_key") != "ema_state" or "ema_state" not in checkpoint:
        raise ValueError(f"Eligible candidate must declare and contain ema_state: {path}")
    args = checkpoint.get("args", {})
    if str(args.get("arch", "dinov2_dpt_small")) != "dinov2_dpt_small":
        raise ValueError(f"Unexpected candidate architecture: {args.get('arch')}")
    protocol = checkpoint.get("protocol", {})
    image_size = tuple(int(v) for v in protocol.get("image_size", args.get("image_size", ())))
    if image_size != (448, 798):
        raise ValueError(f"Eligible candidate image_size must be (448, 798), got {image_size}")
    target_label = int(protocol.get("target_label", args.get("target_label", -1)))
    if target_label != 10:
        raise ValueError(f"Eligible candidate target_label must be 10, got {target_label}")
    threshold = float(checkpoint.get("val_metrics", {}).get("val_threshold", 0.5))
    model = DinoV2DPTSegmenter(weights_path=None, freeze_backbone=bool(args.get("freeze_backbone", True)))
    model.load_state_dict(checkpoint["ema_state"], strict=True)
    model.to(torch.device(device)).eval()
    return model, image_size, threshold, target_label, checkpoint


__all__ = [
    "DEFAULT_DINOV2_WEIGHTS",
    "DPTSmallBinaryHead",
    "DinoV2DPTSegmenter",
    "apply_aligned_cutmix",
    "binary_pseudo_labels",
    "build_model",
    "complementary_feature_dropout",
    "confidence_masked_bce",
    "cutmix_batch",
    "load_dinov2_candidate_checkpoint",
    "load_dinov2_small_backbone",
    "make_binary_pseudo_targets",
    "masked_bce_with_logits",
    "sample_cutmix_mask",
]
