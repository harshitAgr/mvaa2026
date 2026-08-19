#!/usr/bin/env python3
"""External-val Task 3 inference using a two-checkpoint probability-level
ensemble (imagenet_v1 + surgenetxl_v1) with 8-flip / D4 TTA.

For each input image:
    probs_a = D4-TTA(model_a, image)
    probs_b = D4-TTA(model_b, image)
    pred = (w * probs_a + (1 - w) * probs_b) > threshold        # w = --weight-a

Inference cost: 2 × 8-flip = 16 model forwards per frame.

Usage
-----
    python scripts/infer_t3_ensemble.py \\
        --data-dir   data/reference_data/t3_vid/val/images \\
        --output-dir submission/ensemble_v1/t3_vid \\
        --threshold  0.45
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dataset import IMAGENET_MEAN, IMAGENET_STD  # type: ignore  # noqa: E402
from infer_t3_d4tta import predict_probs_d4, load_ckpt_config, load_state_dict  # type: ignore  # noqa: E402

DEFAULT_RUN_A = REPO_ROOT / "runs" / "task3_imagenet_v1" / "checkpoints" / "best.pt"
DEFAULT_RUN_B = REPO_ROOT / "runs" / "task3_surgenetxl_v1" / "checkpoints" / "best.pt"


def build_model(ckpt_path: Path, device: torch.device):
    from model_factory import get_model  # type: ignore  # local import — needs sys.path

    ckpt, train_args = load_ckpt_config(ckpt_path)
    arch = str(train_args.get("arch", "unetplusplus"))
    encoder_name = str(train_args.get("encoder_name", "efficientnet-b4"))
    encoder_weights = train_args.get("encoder_weights", None)
    if isinstance(encoder_weights, str) and encoder_weights.lower() == "none":
        encoder_weights = None
    image_size: Tuple[int, int] = tuple(int(v) for v in train_args.get("image_size", [448, 800]))
    use_imagenet_norm = bool(train_args.get("use_imagenet_norm", True))
    model = get_model(arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights,
                     in_channels=3, classes=1).to(device)
    load_state_dict(model, ckpt)
    model.eval()
    return model, image_size, use_imagenet_norm


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--ckpt-a", type=Path, default=DEFAULT_RUN_A)
    ap.add_argument("--ckpt-b", type=Path, default=DEFAULT_RUN_B)
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="Sweep on internal val 39-frame chose 0.45 as the best (composite 0.5818).")
    ap.add_argument(
        "--weight-a", type=float, default=0.5,
        help="Probability-level fusion weight for member A (A_DR); member B gets 1 - weight_a. "
             "0.5 reproduces the historical equal-weight path exactly.",
    )
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_a = float(args.weight_a)
    if not 0.0 <= weight_a <= 1.0:
        raise SystemExit(f"--weight-a must lie in [0,1], got {weight_a}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model_a, sz_a, norm_a = build_model(args.ckpt_a, device)
    model_b, sz_b, norm_b = build_model(args.ckpt_b, device)
    if sz_a != sz_b:
        raise SystemExit(f"image_size mismatch: A={sz_a} B={sz_b}")
    if norm_a != norm_b:
        raise SystemExit(f"imagenet-norm mismatch: A={norm_a} B={norm_b}")
    image_size = sz_a

    # Image discovery
    files = []
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            continue
        if p.name.lower().endswith("_label_bin.png") or "_png_label_vis" in p.name.lower():
            continue
        files.append(p)
    if not files:
        raise SystemExit(f"no input frames under {data_dir}")

    norm_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    print(f"Device: {device}")
    print(f"Ckpt A: {args.ckpt_a}")
    print(f"Ckpt B: {args.ckpt_b}")
    print(f"Image size: {image_size}, threshold: {args.threshold}, TTA: 8flip x 2 models = 16 passes/frame")
    print(f"Fusion: {weight_a:.2f}*A + {1.0 - weight_a:.2f}*B")
    print(f"Frames: {len(files)}")

    records, timings = [], []
    per_video_t0, per_video_total = {}, {}
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for idx, p in enumerate(files, 1):
        v = p.parent.name
        if v not in per_video_t0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            per_video_t0[v] = time.perf_counter()

        img_u8 = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        H, W = img_u8.shape[:2]
        img_t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        img_t = F.interpolate(img_t, size=image_size, mode="bilinear", align_corners=False)
        if norm_a:
            img_t = (img_t - norm_mean) / norm_std

        if device.type == "cuda":
            torch.cuda.synchronize()
        tp = time.perf_counter()
        probs_a = predict_probs_d4(model_a, img_t, use_amp=use_amp)
        probs_b = predict_probs_d4(model_b, img_t, use_amp=use_amp)
        probs = weight_a * probs_a + (1.0 - weight_a) * probs_b
        pred_small = (probs > float(args.threshold)).float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - tp)

        pred_full = F.interpolate(pred_small, size=(H, W), mode="nearest")
        mask = (pred_full[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        rel = p.relative_to(data_dir)
        save = out_dir / rel.parent / f"{p.stem}_label_bin.png"
        save.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(save)
        records.append({"case_id": p.stem, "segmentation": save.relative_to(out_dir).as_posix()})
        per_video_total[v] = time.perf_counter() - per_video_t0[v]
        if idx % 10 == 0 or idx == len(files):
            print(f"  [predict] {idx}/{len(files)}  last={timings[-1]*1000:.0f} ms  -> {save.name}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - t0
    timings = np.asarray(timings)
    out_dir.joinpath("task3_predictions.json").write_text(
        json.dumps({"cases": records}, ensure_ascii=False, indent=2)
    )
    out_dir.joinpath("_timing.json").write_text(json.dumps({
        "n_frames": len(files), "total_seconds": float(total),
        "ms_per_frame": {"mean": float(timings.mean()*1000), "p50": float(np.median(timings)*1000),
                          "p95": float(np.percentile(timings, 95)*1000), "max": float(timings.max()*1000)},
        "per_video_seconds": per_video_total,
    }, indent=2))
    print(f"\nTotal: {total:.2f} s for {len(files)} frames")
    print(f"Per-frame ms: mean={timings.mean()*1000:.1f}  median={np.median(timings)*1000:.1f}  max={timings.max()*1000:.1f}")
    for v, s in per_video_total.items():
        ok = "OK" if s <= 10 else "OVER 10s"
        print(f"  {v}: {s:.2f}s  [{ok}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
