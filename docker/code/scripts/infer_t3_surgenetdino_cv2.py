#!/usr/bin/env python3
"""Infer T3 with 0.60*A_DR + 0.40*mean(two surgical-DINOv2 EMA models)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_T3 = REPO_ROOT / "baseline" / "task3"
sys.path.insert(0, str(BASELINE_T3))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dataset import IMAGENET_MEAN, IMAGENET_STD  # type: ignore  # noqa: E402
from infer_t3_d4tta import load_ckpt_config, load_state_dict, predict_probs_d4  # type: ignore  # noqa: E402

SCHEMA = "mvaa-t3-surgenetdino-v2b-inference-state-v1"
MODEL_IMAGE_SIZE = (448, 798)
BACKBONE_NAME = "vit_base_patch14_dinov2"
INTERMEDIATE_BLOCKS = (2, 5, 8, 11)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(x, inplace=False))
        out = self.conv2(F.relu(out, inplace=False))
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.residual_skip = ResidualConvUnit(features)
        self.residual_out = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
        size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if skip is not None:
            x = x + self.residual_skip(skip)
        x = self.residual_out(x)
        if size is None:
            size = (x.shape[-2] * 2, x.shape[-1] * 2)
        return self.out_conv(F.interpolate(x, size=size, mode="bilinear", align_corners=True))


class DPTSmallBinaryHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        widths = (48, 96, 192, 384)
        features = 64
        self.projects = nn.ModuleList([nn.Conv2d(768, width, 1) for width in widths])
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
        projected = [resize(project(feature)) for feature, project, resize in zip(
            feature_maps, self.projects, self.resize_layers
        )]
        layer1, layer2, layer3, layer4 = [
            scratch(feature) for scratch, feature in zip(self.scratch, projected)
        ]
        path4 = self.fuse4(layer4, size=layer3.shape[-2:])
        path3 = self.fuse3(path4, layer3, size=layer2.shape[-2:])
        path2 = self.fuse2(path3, layer2, size=layer1.shape[-2:])
        path1 = self.fuse1(path2, layer1)
        return self.output(path1)


class SurgicalDinoV2B(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=False,
            img_size=336,
            dynamic_img_size=True,
        )
        if hasattr(self.backbone, "mask_token"):
            raise ValueError("Unexpected timm DINOv2 graph already has mask_token")
        self.backbone.register_parameter(
            "mask_token", nn.Parameter(torch.zeros((1, 768), dtype=torch.float32))
        )
        self.head = DPTSmallBinaryHead()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        if height % 14 or width % 14:
            raise ValueError(f"DINOv2 input must be divisible by 14, got {(height, width)}")
        features = self.backbone.forward_intermediates(
            image,
            indices=INTERMEDIATE_BLOCKS,
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        logits = self.head(tuple(features))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=True)


def build_a(checkpoint: Path, device: torch.device):
    from model_factory import get_model  # type: ignore

    payload, train_args = load_ckpt_config(checkpoint)
    encoder_weights = train_args.get("encoder_weights", None)
    if isinstance(encoder_weights, str) and encoder_weights.lower() == "none":
        encoder_weights = None
    model = get_model(
        arch=str(train_args.get("arch", "unetplusplus")),
        encoder_name=str(train_args.get("encoder_name", "efficientnet-b4")),
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)
    load_state_dict(model, payload)
    size = tuple(int(value) for value in train_args.get("image_size", [448, 800]))
    if size != (448, 800) or not bool(train_args.get("use_imagenet_norm", True)):
        raise ValueError(f"Unexpected A_DR inference contract: size={size}, args={train_args}")
    return model.eval()


def build_c(
    checkpoint: Path,
    expected_file_sha256: str,
    expected_source_sha256: str,
    expected_fold: str,
    device: torch.device,
) -> SurgicalDinoV2B:
    actual = sha256_file(checkpoint)
    if actual != expected_file_sha256:
        raise ValueError(f"C state SHA-256 mismatch: expected {expected_file_sha256}, got {actual}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("fold_tag") != expected_fold
        or int(payload.get("epoch", -1)) != 70
        or payload.get("state_key") != "ema_state"
        or payload.get("source_checkpoint_sha256") != expected_source_sha256
        or int(payload.get("tensor_count", -1)) != 237
        or not isinstance(state, dict)
    ):
        raise ValueError(f"Invalid inference-only C state: {checkpoint}")
    model = SurgicalDinoV2B()
    expected = model.state_dict()
    if set(state) != set(expected) or any(state[key].shape != expected[key].shape for key in expected):
        raise ValueError(f"C state graph mismatch: {checkpoint}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def d4_probability(model: nn.Module, image: torch.Tensor, use_amp: bool) -> torch.Tensor:
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
            predictions.append(probability.float())
    return torch.stack(predictions).mean(dim=0)


def discover_images(root: Path) -> list[Path]:
    return [
        path for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not path.name.lower().endswith("_label_bin.png")
        and "_png_label_vis" not in path.name.lower()
    ]


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ckpt-a", type=Path, required=True)
    parser.add_argument("--ckpt-c", type=Path, action="append", required=True)
    parser.add_argument("--expected-ckpt-c-sha256", action="append", required=True)
    parser.add_argument("--source-checkpoint-sha256", action="append", required=True)
    parser.add_argument("--fold-tag", action="append", required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--weight-a", type=float, default=0.60)
    args = parser.parse_args()
    if not (
        len(args.ckpt_c)
        == len(args.expected_ckpt_c_sha256)
        == len(args.source_checkpoint_sha256)
        == len(args.fold_tag)
        == 2
    ):
        raise SystemExit("Exactly two C checkpoints, hashes, source hashes, and fold tags are required")
    if args.fold_tag != ["979A", "675A"]:
        raise SystemExit(f"Frozen C fold order is 979A,675A; got {args.fold_tag}")
    if not 0.0 <= args.weight_a <= 1.0:
        raise SystemExit("--weight-a must be in [0,1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model_a = build_a(args.ckpt_a.resolve(), device)
    models_c = [
        build_c(path.resolve(), file_sha, source_sha, fold, device)
        for path, file_sha, source_sha, fold in zip(
            args.ckpt_c,
            args.expected_ckpt_c_sha256,
            args.source_checkpoint_sha256,
            args.fold_tag,
        )
    ]
    files = discover_images(args.data_dir.resolve())
    if not files:
        raise SystemExit(f"No input frames under {args.data_dir}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    records = []
    per_frame = []
    print(f"Device: {device}")
    print("Fusion: 0.60*A_DR + 0.40*mean(C_979A,C_675A); native soft probability fusion")
    print(f"A checkpoint: {args.ckpt_a}")
    for fold, path, digest in zip(args.fold_tag, args.ckpt_c, args.expected_ckpt_c_sha256):
        print(f"C_{fold}: {path} sha256={digest}")
    for index, path in enumerate(files, 1):
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        native_shape = image.shape[:2]
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        tensor_a = (F.interpolate(tensor, size=(448, 800), mode="bilinear", align_corners=False) - mean) / std
        tensor_c = (F.interpolate(tensor, size=MODEL_IMAGE_SIZE, mode="bilinear", align_corners=False) - mean) / std
        if device.type == "cuda":
            torch.cuda.synchronize()
        frame_start = time.perf_counter()
        prob_a = predict_probs_d4(model_a, tensor_a, use_amp=use_amp).float()
        prob_c = torch.stack([d4_probability(model, tensor_c, use_amp) for model in models_c]).mean(0)
        native_a = F.interpolate(prob_a, size=native_shape, mode="bilinear", align_corners=False)
        native_c = F.interpolate(prob_c, size=native_shape, mode="bilinear", align_corners=False)
        fused = args.weight_a * native_a + (1.0 - args.weight_a) * native_c
        mask = (fused > args.threshold)[0, 0].to(torch.uint8).cpu().numpy() * 255
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_frame.append(time.perf_counter() - frame_start)
        relative = path.relative_to(args.data_dir.resolve())
        save = output / relative.parent / f"{path.stem}_label_bin.png"
        save.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask, mode="L").save(save)
        records.append({"case_id": path.stem, "segmentation": save.relative_to(output).as_posix()})
        if index % 10 == 0 or index == len(files):
            print(f"[predict] {index}/{len(files)} {per_frame[-1]:.3f}s {save.name}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    (output / "task3_predictions.json").write_text(json.dumps({"cases": records}, indent=2) + "\n")
    (output / "_timing.json").write_text(json.dumps({
        "frames": len(files),
        "total_seconds": total,
        "seconds_per_frame_mean": float(np.mean(per_frame)),
        "seconds_per_frame_max": float(np.max(per_frame)),
        "peak_cuda_allocated_bytes": peak,
    }, indent=2) + "\n")
    print(f"Total {total:.3f}s; mean {np.mean(per_frame):.3f}s/frame; peak {peak / 2**30:.3f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
