#!/usr/bin/env python3
"""Task 3 inference with 8-flip / D4-dihedral TTA.

For each image, runs 8 forward passes covering the D4 dihedral group
(rot90 k ∈ {0,1,2,3} × h-flip ∈ {False, True}), inverse-transforms each
probability map, averages, then thresholds. Same checkpoint, same
preprocessing and same threshold-from-checkpoint as the organizer baseline's
Task 3 prediction script — only the TTA differs (4-flip -> 8-flip).

For 448 × 800 input the rotated passes feed the model 800 × 448 (both
divisible by 32, the encoder stride), so the same model handles both
orientations without resizing.

Usage
-----
    python scripts/infer_t3_d4tta.py \\
        --data-dir data/reference_data/t3_vid/val/images \\
        --output-dir submission/d4tta_v1/t3_vid \\
        [--ckpt runs/task3_imagenet_v1/checkpoints/best.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_T3 = REPO_ROOT / "baseline" / "task3"
sys.path.insert(0, str(BASELINE_T3))

from dataset import IMAGENET_MEAN, IMAGENET_STD  # type: ignore  # noqa: E402
from model_factory import get_model  # type: ignore  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def discover_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"data dir not found: {folder}")
    files = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        nl = p.name.lower()
        if nl.endswith("_label_bin.png") or "_png_label_vis" in nl:
            continue
        files.append(p)
    if not files:
        raise RuntimeError(f"no image files under {folder}")
    return files


def load_ckpt_config(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    if not isinstance(train_args, dict):
        train_args = {}
    return ckpt, train_args


def load_state_dict(model, ckpt_obj):
    state = ckpt_obj.get("model_state", ckpt_obj.get("model_state_dict", ckpt_obj.get("state_dict", ckpt_obj)))
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def predict_probs_d4(model, image_t: torch.Tensor, use_amp: bool) -> torch.Tensor:
    device_type = image_t.device.type
    probs_sum = None
    n = 0
    for k in (0, 1, 2, 3):
        for hflip in (False, True):
            x = image_t
            if hflip:
                x = torch.flip(x, dims=(3,))
            if k > 0:
                x = torch.rot90(x, k=k, dims=(2, 3))
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                logits = model(x)
            probs = torch.sigmoid(logits)
            if k > 0:
                probs = torch.rot90(probs, k=-k, dims=(2, 3))
            if hflip:
                probs = torch.flip(probs, dims=(3,))
            probs_sum = probs if probs_sum is None else probs_sum + probs
            n += 1
    return probs_sum / float(n)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory of input images (per-video subfolders or flat).")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=REPO_ROOT / "runs" / "task3_imagenet_v1" / "checkpoints" / "best.pt",
                    help="Checkpoint .pt with 'args' and 'val_metrics.val_threshold'.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override binarization threshold. Default: ckpt['val_metrics']['val_threshold'].")
    ap.add_argument("--no-amp", action="store_true", default=False)
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt, train_args = load_ckpt_config(args.ckpt)
    arch = str(train_args.get("arch", "unetplusplus"))
    encoder_name = str(train_args.get("encoder_name", "efficientnet-b4"))
    encoder_weights = train_args.get("encoder_weights", None)
    if isinstance(encoder_weights, str) and encoder_weights.lower() == "none":
        encoder_weights = None
    image_size: Tuple[int, int] = tuple(int(v) for v in train_args.get("image_size", [448, 800]))
    use_imagenet_norm = bool(train_args.get("use_imagenet_norm", True))
    threshold_default = float(ckpt.get("val_metrics", {}).get("val_threshold", 0.5))
    threshold = float(args.threshold) if args.threshold is not None else threshold_default

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    model = get_model(
        arch=arch,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)
    load_state_dict(model, ckpt)
    model.eval()

    files = discover_images(data_dir)
    norm_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    print(f"Device:      {device}")
    print(f"Checkpoint:  {args.ckpt}")
    print(f"Arch:        {arch} / {encoder_name}")
    print(f"Image size:  {image_size}")
    print(f"Threshold:   {threshold:.4f}")
    print(f"TTA mode:    8flip (D4)")
    print(f"AMP:         {use_amp}")
    print(f"Frames:      {len(files)}")
    print(f"Out:         {out_dir}")

    records = []
    timings = []
    per_video_first_seen = {}
    per_video_t0 = {}
    per_video_total = {}
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0_total = time.perf_counter()

    for idx, image_path in enumerate(files, start=1):
        video_dir = image_path.parent.name
        if video_dir not in per_video_first_seen:
            per_video_first_seen[video_dir] = idx
            if device.type == "cuda":
                torch.cuda.synchronize()
            per_video_t0[video_dir] = time.perf_counter()

        img_u8 = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        h, w = img_u8.shape[:2]
        img_t = torch.from_numpy(img_u8.astype(np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(device)
        img_t = F.interpolate(img_t, size=image_size, mode="bilinear", align_corners=False)
        if use_imagenet_norm:
            img_t = (img_t - norm_mean) / norm_std

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_pred = time.perf_counter()

        probs = predict_probs_d4(model, img_t, use_amp=use_amp)
        pred_small = (probs > threshold).float()

        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t_pred)

        pred_full = F.interpolate(pred_small, size=(h, w), mode="nearest")
        pred_mask = (pred_full[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)

        rel = image_path.relative_to(data_dir)
        save_path = out_dir / rel.parent / f"{image_path.stem}_label_bin.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L").save(save_path)

        records.append({
            "case_id": image_path.stem,
            "segmentation": save_path.relative_to(out_dir).as_posix(),
        })
        if idx % 10 == 0 or idx == len(files):
            print(f"  [predict] {idx}/{len(files)}  last={timings[-1]*1000:.1f} ms  -> {save_path.name}")

        # Track per-video walltime for budget check.
        per_video_total[video_dir] = time.perf_counter() - per_video_t0[video_dir]

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0_total
    timings = np.asarray(timings)

    # Summary
    out_json = out_dir / "task3_predictions.json"
    out_json.write_text(json.dumps({"cases": records}, ensure_ascii=False, indent=2))
    timing_json = out_dir / "_timing.json"
    timing_json.write_text(json.dumps({
        "n_frames": len(files),
        "total_seconds": float(total_time),
        "model_predict_ms_mean": float(timings.mean() * 1000),
        "model_predict_ms_p50": float(np.median(timings) * 1000),
        "model_predict_ms_p95": float(np.percentile(timings, 95) * 1000),
        "model_predict_ms_max": float(timings.max() * 1000),
        "per_video_seconds": per_video_total,
    }, indent=2))

    print()
    print(f"Saved JSON:    {out_json}")
    print(f"Saved timing:  {timing_json}")
    print(f"Total wall:    {total_time:.2f} s for {len(files)} frames")
    print(f"Per-frame ms:  mean={timings.mean()*1000:.1f}  median={np.median(timings)*1000:.1f}  p95={np.percentile(timings, 95)*1000:.1f}  max={timings.max()*1000:.1f}")
    print(f"Per-video:")
    for v, secs in per_video_total.items():
        budget_ok = "OK ≤10s" if secs <= 10.0 else "OVER BUDGET"
        n_v = sum(1 for f in files if f.parent.name == v)
        print(f"  {v}: {secs:.2f} s for {n_v} frames  [{budget_ok}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
