#!/usr/bin/env python3
"""Infer T3 with A_DR plus the mean of three surgical-DINOv2 models."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import infer_t3_surgenetdino_cv2 as shared


FROZEN_FOLD_ORDER = ["979A", "675A", "ALL6"]


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
        == 3
    ):
        raise SystemExit("Exactly three C checkpoints, hashes, source hashes, and fold tags are required")
    if args.fold_tag != FROZEN_FOLD_ORDER:
        raise SystemExit(f"Frozen C fold order is {','.join(FROZEN_FOLD_ORDER)}; got {args.fold_tag}")
    if not 0.0 <= args.weight_a <= 1.0:
        raise SystemExit("--weight-a must be in [0,1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model_a = shared.build_a(args.ckpt_a.resolve(), device)
    models_c = [
        shared.build_c(path.resolve(), file_sha, source_sha, fold, device)
        for path, file_sha, source_sha, fold in zip(
            args.ckpt_c,
            args.expected_ckpt_c_sha256,
            args.source_checkpoint_sha256,
            args.fold_tag,
        )
    ]
    files = shared.discover_images(args.data_dir.resolve())
    if not files:
        raise SystemExit(f"No input frames under {args.data_dir}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(shared.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(shared.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    records: list[dict[str, str]] = []
    per_frame: list[float] = []
    weight_c = 1.0 - args.weight_a
    print(f"Device: {device}")
    print(
        f"Fusion: {args.weight_a:.2f}*A_DR + {weight_c:.2f}*"
        "mean(C_979A,C_675A,C_ALL6); native soft probability fusion"
    )
    print(f"A checkpoint: {args.ckpt_a}")
    for fold, path, digest in zip(args.fold_tag, args.ckpt_c, args.expected_ckpt_c_sha256):
        print(f"C_{fold}: {path} sha256={digest}")
    for index, path in enumerate(files, 1):
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        native_shape = image.shape[:2]
        tensor = (
            torch.from_numpy(image.astype(np.float32) / 255.0)
            .permute(2, 0, 1)[None]
            .to(device)
        )
        tensor_a = (
            F.interpolate(tensor, size=(448, 800), mode="bilinear", align_corners=False) - mean
        ) / std
        tensor_c = (
            F.interpolate(tensor, size=shared.MODEL_IMAGE_SIZE, mode="bilinear", align_corners=False)
            - mean
        ) / std
        if device.type == "cuda":
            torch.cuda.synchronize()
        frame_start = time.perf_counter()
        prob_a = shared.predict_probs_d4(model_a, tensor_a, use_amp=use_amp).float()
        prob_c = torch.stack(
            [shared.d4_probability(model, tensor_c, use_amp) for model in models_c]
        ).mean(0)
        native_a = F.interpolate(prob_a, size=native_shape, mode="bilinear", align_corners=False)
        native_c = F.interpolate(prob_c, size=native_shape, mode="bilinear", align_corners=False)
        fused = args.weight_a * native_a + weight_c * native_c
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
    (output / "task3_predictions.json").write_text(
        json.dumps({"cases": records}, indent=2) + "\n", encoding="utf-8"
    )
    (output / "_timing.json").write_text(
        json.dumps(
            {
                "frames": len(files),
                "total_seconds": total,
                "seconds_per_frame_mean": float(np.mean(per_frame)),
                "seconds_per_frame_max": float(np.max(per_frame)),
                "peak_cuda_allocated_bytes": peak,
                "threshold": args.threshold,
                "weight_a": args.weight_a,
                "fold_order": args.fold_tag,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Total {total:.3f}s; mean {np.mean(per_frame):.3f}s/frame; "
        f"peak {peak / 2**30:.3f} GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
