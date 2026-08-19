#!/usr/bin/env python3
"""Target-blind SurgeNetDINO DINOv2-B + DPT model contracts.

This module is additive and intentionally independent of MVAA data discovery.
It strict-loads the verified flat 175-tensor SurgeNetDINO checkpoint into a
plain timm DINOv2-B/14 graph, preserves the released (unused-at-inference)
``mask_token``, and reuses the frozen DPT-small/UniMatch primitives from the
earlier generic DINOv2-S experiment.  It never downloads weights.
"""
from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from t3_dinov2_unimatch import (
    DPTSmallBinaryHead,
    apply_aligned_cutmix,
    make_binary_pseudo_targets,
    masked_bce_with_logits,
    sample_cutmix_mask,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = (
    REPO_ROOT / "data" / "pretrained" / "SurgeNetDINOv2_ViTb14_size336_SurgeNetXL.pth"
)
BACKBONE_NAME = "vit_base_patch14_dinov2"
PRETRAIN_IMAGE_SIZE = 336
PATCH_SIZE = 14
BACKBONE_CHANNELS = 768
INTERMEDIATE_BLOCKS = (2, 5, 8, 11)
MODEL_IMAGE_SIZE = (448, 798)
ROTATED_IMAGE_SIZE = (798, 448)
DPT_FEATURES = 64
DPT_PROJECTION_CHANNELS = (48, 96, 192, 384)

ARTIFACT_SIZE = 343_942_415
ARTIFACT_SHA256 = "cd6e73e692074f1d58d2e4125998818121acc61748ff941849784616b03c2def"
ARTIFACT_TENSOR_COUNT = 175
ARTIFACT_ELEMENTS = 85_971_456
ARTIFACT_TENSOR_BYTES = 343_885_824
ARTIFACT_POS_EMBED_SHAPE = (1, 577, 768)
ARTIFACT_MASK_TOKEN_SHAPE = (1, 768)
OFFICIAL_CODE_REVISION = "19e8325c87b826f5ae637787563891476bbe2b9f"
MODEL_HOST_REVISION = "9b2c75b7d469850b750a715b90cb91d6319f7e30"
MODEL_URL = (
    "https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/"
    "DINOv2_ViTb14_size336_SurgeNetXL.pth"
)
CODE_LICENSE = "MIT"
WEIGHTS_LICENSE = "CC-BY-NC-SA"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_file(path: str | Path) -> dict[str, Any]:
    """Verify immutable file identity without deserializing it."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"SurgeNetDINO weights not found: {artifact}")
    size = artifact.stat().st_size
    if size != ARTIFACT_SIZE:
        raise ValueError(f"Artifact size mismatch: expected {ARTIFACT_SIZE}, got {size}")
    digest = sha256_file(artifact)
    if digest != ARTIFACT_SHA256:
        raise ValueError(f"Artifact SHA-256 mismatch: expected {ARTIFACT_SHA256}, got {digest}")
    return {
        "path": artifact.resolve().as_posix(),
        "size_bytes": size,
        "sha256": digest,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "model_host_revision": MODEL_HOST_REVISION,
        "download_url": MODEL_URL,
        "code_license": CODE_LICENSE,
        "weights_license": WEIGHTS_LICENSE,
    }


def load_verified_flat_state(path: str | Path) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    """Safely load and inventory the verified flat checkpoint.

    The file identity is checked before ``torch.load``. ``weights_only=True``
    prevents arbitrary application classes from being reconstructed.
    """
    receipt = verify_artifact_file(path)
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a flat tensor mapping, got {type(state)!r}")
    if len(state) != ARTIFACT_TENSOR_COUNT:
        raise ValueError(
            f"Artifact tensor count mismatch: expected {ARTIFACT_TENSOR_COUNT}, got {len(state)}"
        )
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise TypeError("Artifact must contain only string-to-tensor entries")
    if not all(value.device.type == "cpu" and value.dtype == torch.float32 for value in state.values()):
        raise ValueError("Every released checkpoint tensor must be CPU FP32 after safe loading")
    elements = sum(int(value.numel()) for value in state.values())
    tensor_bytes = sum(int(value.numel() * value.element_size()) for value in state.values())
    if elements != ARTIFACT_ELEMENTS or tensor_bytes != ARTIFACT_TENSOR_BYTES:
        raise ValueError(
            "Artifact tensor inventory mismatch: "
            f"expected {ARTIFACT_ELEMENTS}/{ARTIFACT_TENSOR_BYTES}, "
            f"got {elements}/{tensor_bytes}"
        )
    if tuple(state["pos_embed"].shape) != ARTIFACT_POS_EMBED_SHAPE:
        raise ValueError(f"Unexpected pos_embed shape: {tuple(state['pos_embed'].shape)}")
    if tuple(state["mask_token"].shape) != ARTIFACT_MASK_TOKEN_SHAPE:
        raise ValueError(f"Unexpected mask_token shape: {tuple(state['mask_token'].shape)}")
    if any(key.startswith("register_tokens") for key in state):
        raise ValueError("Registered candidate is the no-register DINOv2-B graph")
    receipt.update(
        {
            "safe_load": "torch.load(weights_only=True)",
            "flat_mapping": True,
            "tensor_count": len(state),
            "elements": elements,
            "tensor_bytes": tensor_bytes,
            "pos_embed_shape": list(state["pos_embed"].shape),
            "mask_token_shape": list(state["mask_token"].shape),
            "register_tokens": 0,
        }
    )
    return state, receipt


def build_plain_backbone() -> nn.Module:
    """Build the exact timm graph and preserve the released mask token."""
    backbone = timm.create_model(
        BACKBONE_NAME,
        pretrained=False,
        img_size=PRETRAIN_IMAGE_SIZE,
        dynamic_img_size=True,
    )
    if hasattr(backbone, "mask_token"):
        raise ValueError("Unexpected timm graph already defines mask_token")
    backbone.register_parameter(
        "mask_token",
        nn.Parameter(torch.zeros(ARTIFACT_MASK_TOKEN_SHAPE, dtype=torch.float32)),
    )
    if tuple(backbone.pos_embed.shape) != ARTIFACT_POS_EMBED_SHAPE:
        raise ValueError(f"Plain timm pos_embed mismatch: {tuple(backbone.pos_embed.shape)}")
    if int(getattr(backbone, "embed_dim", -1)) != BACKBONE_CHANNELS:
        raise ValueError(f"Plain timm backbone width mismatch: {getattr(backbone, 'embed_dim', None)}")
    if int(getattr(backbone, "num_prefix_tokens", -1)) != 1:
        raise ValueError("Registered graph requires exactly one class prefix token")
    if int(getattr(backbone, "num_reg_tokens", 0)) != 0:
        raise ValueError("Registered graph must not contain register tokens")
    if len(backbone.blocks) != 12:
        raise ValueError(f"Registered graph requires 12 transformer blocks, got {len(backbone.blocks)}")
    if len(backbone.state_dict()) != ARTIFACT_TENSOR_COUNT:
        raise ValueError(
            f"Plain graph tensor count mismatch after mask_token registration: "
            f"{len(backbone.state_dict())}"
        )
    return backbone


def load_strict_backbone(
    weights_path: str | Path = DEFAULT_WEIGHTS,
    freeze: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Strict-load all 175 released tensors into the exact timm graph."""
    state, receipt = load_verified_flat_state(weights_path)
    backbone = build_plain_backbone()
    expected = backbone.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    mismatches = {
        key: {"expected": list(expected[key].shape), "actual": list(state[key].shape)}
        for key in expected
        if expected[key].shape != state[key].shape or expected[key].dtype != state[key].dtype
    }
    if mismatches:
        raise ValueError(f"Checkpoint shape/dtype mismatch: {mismatches}")
    incompatible = backbone.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"Strict load returned incompatibilities: {incompatible}")
    if freeze:
        backbone.requires_grad_(False)
        backbone.eval()
    receipt.update(
        {
            "backbone_name": BACKBONE_NAME,
            "pretrain_image_size": PRETRAIN_IMAGE_SIZE,
            "dynamic_img_size": True,
            "strict_load": True,
            "loaded_tensor_count": len(backbone.state_dict()),
            "backbone_width": BACKBONE_CHANNELS,
            "transformer_blocks": len(backbone.blocks),
        }
    )
    return backbone, receipt


def complementary_feature_dropout_v2b(
    feature_maps: Sequence[torch.Tensor],
    generator: Optional[torch.Generator] = None,
    single_pair_changed: Optional[bool] = None,
) -> Tuple[torch.Tensor, ...]:
    """Complementary dropout with a registered micro-batch-one policy.

    Micro-batch two uses the original exactly-half-within-batch helper logic.
    At micro-batch one, the caller must alternate ``single_pair_changed``
    across accumulation microsteps so exactly half the pairs per optimizer
    update remain unchanged.
    """
    if not feature_maps:
        raise ValueError("feature_maps cannot be empty")
    batch2, channels = feature_maps[0].shape[:2]
    if batch2 % 2:
        raise ValueError(f"Paired strong batch must contain 2B samples, got {batch2}")
    pairs = batch2 // 2
    for feature in feature_maps:
        if feature.ndim != 4 or feature.shape[:2] != (batch2, channels):
            raise ValueError("All feature maps must share N/C dimensions")
    if pairs == 1:
        if single_pair_changed is None:
            raise ValueError("Micro-batch one requires an explicit accumulation dropout policy")
        if not single_pair_changed:
            return tuple(feature for feature in feature_maps)
        random_values = torch.rand(
            (1, channels), device=feature_maps[0].device, generator=generator
        )
        first = (random_values >= 0.5).to(feature_maps[0].dtype) * 2.0
        second = 2.0 - first
        mask = torch.cat((first, second), dim=0).unsqueeze(-1).unsqueeze(-1)
        return tuple(feature * mask.to(dtype=feature.dtype) for feature in feature_maps)
    if pairs % 2:
        raise ValueError(f"Exactly-half policy requires an even pair count, got {pairs}")
    random_values = torch.rand(
        (pairs, channels), device=feature_maps[0].device, generator=generator
    )
    first = (random_values >= 0.5).to(feature_maps[0].dtype) * 2.0
    second = 2.0 - first
    unchanged = torch.randperm(
        pairs, device=feature_maps[0].device, generator=generator
    )[: pairs // 2]
    first[unchanged] = 1.0
    second[unchanged] = 1.0
    mask = torch.cat((first, second), dim=0).unsqueeze(-1).unsqueeze(-1)
    return tuple(feature * mask.to(dtype=feature.dtype) for feature in feature_maps)


class SurgeNetDinoV2BDPTSegmenter(nn.Module):
    """Strict SurgeNetDINO DINOv2-B/14 with unchanged DPT-small capacity."""

    def __init__(
        self,
        weights_path: str | Path | None = DEFAULT_WEIGHTS,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.artifact_receipt: dict[str, Any] | None
        if weights_path is None:
            self.backbone = build_plain_backbone()
            self.artifact_receipt = None
            if freeze_backbone:
                self.backbone.requires_grad_(False)
                self.backbone.eval()
        else:
            self.backbone, self.artifact_receipt = load_strict_backbone(
                weights_path, freeze=freeze_backbone
            )
        self.head = DPTSmallBinaryHead(
            in_channels=BACKBONE_CHANNELS,
            features=DPT_FEATURES,
            projection_channels=DPT_PROJECTION_CHANNELS,
        )
        self.freeze_backbone = bool(freeze_backbone)

    @property
    def decoder(self) -> DPTSmallBinaryHead:
        return self.head

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.freeze_backbone = not bool(trainable)
        self.backbone.requires_grad_(bool(trainable))
        if trainable:
            self.backbone.train(self.training)
        else:
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def validate_shape(x: torch.Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected RGB NCHW input, got {tuple(x.shape)}")
        height, width = x.shape[-2:]
        if height % PATCH_SIZE or width % PATCH_SIZE:
            raise ValueError(
                f"Input H/W must be divisible by patch size {PATCH_SIZE}, got {(height, width)}"
            )

    def features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            outputs = self.backbone.forward_intermediates(
                x,
                indices=INTERMEDIATE_BLOCKS,
                norm=True,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        result = tuple(outputs)
        if len(result) != 4 or any(feature.shape[1] != BACKBONE_CHANNELS for feature in result):
            raise ValueError(
                f"Unexpected intermediate feature contract: {[tuple(v.shape) for v in result]}"
            )
        return result

    def forward(
        self,
        x: torch.Tensor,
        comp_drop: bool = False,
        generator: Optional[torch.Generator] = None,
        single_pair_changed: Optional[bool] = None,
    ) -> torch.Tensor:
        self.validate_shape(x)
        features = self.features(x)
        if comp_drop:
            features = complementary_feature_dropout_v2b(
                features,
                generator=generator,
                single_pair_changed=single_pair_changed,
            )
        logits = self.head(features)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=True)

    def forward_paired_strong(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        single_pair_changed: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if view1.shape != view2.shape:
            raise ValueError(f"Strong views must have identical shapes: {view1.shape}/{view2.shape}")
        logits = self(
            torch.cat((view1, view2), dim=0),
            comp_drop=True,
            generator=generator,
            single_pair_changed=single_pair_changed,
        )
        return logits.chunk(2, dim=0)


def d4_probabilities(
    model: nn.Module,
    image: torch.Tensor,
    use_amp: bool,
) -> torch.Tensor:
    """Eight-way D4 probability average with exact inverse transforms."""
    predictions = []
    for rotations in range(4):
        for horizontal_flip in (False, True):
            transformed = torch.flip(image, dims=(3,)) if horizontal_flip else image
            if rotations:
                transformed = torch.rot90(transformed, k=rotations, dims=(2, 3))
            with torch.amp.autocast(device_type=image.device.type, enabled=use_amp):
                probability = torch.sigmoid(model(transformed))
            if rotations:
                probability = torch.rot90(probability, k=-rotations, dims=(2, 3))
            if horizontal_flip:
                probability = torch.flip(probability, dims=(3,))
            predictions.append(probability)
    if len(predictions) != 8:
        raise AssertionError("D4 inference must produce exactly eight predictions")
    return torch.stack(predictions).mean(dim=0)


__all__ = [
    "ARTIFACT_ELEMENTS",
    "ARTIFACT_SHA256",
    "ARTIFACT_SIZE",
    "ARTIFACT_TENSOR_BYTES",
    "ARTIFACT_TENSOR_COUNT",
    "BACKBONE_CHANNELS",
    "BACKBONE_NAME",
    "DEFAULT_WEIGHTS",
    "DPT_FEATURES",
    "DPT_PROJECTION_CHANNELS",
    "INTERMEDIATE_BLOCKS",
    "MODEL_IMAGE_SIZE",
    "ROTATED_IMAGE_SIZE",
    "SurgeNetDinoV2BDPTSegmenter",
    "apply_aligned_cutmix",
    "build_plain_backbone",
    "complementary_feature_dropout_v2b",
    "d4_probabilities",
    "load_strict_backbone",
    "load_verified_flat_state",
    "make_binary_pseudo_targets",
    "masked_bce_with_logits",
    "sample_cutmix_mask",
    "sha256_file",
    "verify_artifact_file",
]
