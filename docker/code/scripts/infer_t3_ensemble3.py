#!/usr/bin/env python3
"""3-way Task 3 inference: probability-level ensemble of THREE checkpoints
(imagenet_v1 + surgenetxl_v1 + the new architecture-diversity member) with
8-flip / D4 TTA each. Copy-and-extend of scripts/infer_t3_ensemble.py (the
deployed 2-way script) — kept as a SEPARATE file so the deployed 2-way path
is completely untouched (additive only, per
the repository README).

For each input image:
    probs_a = D4-TTA(model_a, image)
    probs_b = D4-TTA(model_b, image)
    probs_c = D4-TTA(model_c, image)
    pred = (weight_a*probs_a + weight_b*probs_b + weight_c*probs_c) > threshold

Inference cost: 3 x 8-flip = 24 model forwards per frame (budget:
deployed 2-way ~= 0.8 s/frame on V100 -> ~1.2 s/frame for 3-way, well under
the 10 s/case budget — CONFIRM ON THE cu124 CONTAINER BEFORE ANY SUBMIT, this
script only runs the math, it does not measure the container's own budget).

Usage
-----
    python scripts/infer_t3_ensemble3.py \\
        --data-dir   data/reference_data/t3_vid/val/images \\
        --output-dir submission/ensemble3_v1/t3_vid \\
        --ckpt-a runs/task3_imagenet_v1/checkpoints/best.pt \\
        --ckpt-b runs/task3_surgenetxl_v1/checkpoints/best.pt \\
        --ckpt-c runs/task3_segformer_allvid/checkpoints/last.pt \\
        --threshold 0.45
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

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

DEFAULT_CKPT_A = REPO_ROOT / "runs" / "task3_imagenet_v1" / "checkpoints" / "best.pt"
DEFAULT_CKPT_B = REPO_ROOT / "runs" / "task3_surgenetxl_v1" / "checkpoints" / "best.pt"
DEFAULT_CKPT_C = REPO_ROOT / "runs" / "task3_segformer_allvid" / "checkpoints" / "last.pt"


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
    return model, image_size, use_imagenet_norm, arch


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--ckpt-a", type=Path, default=DEFAULT_CKPT_A)
    ap.add_argument("--ckpt-b", type=Path, default=DEFAULT_CKPT_B)
    ap.add_argument("--ckpt-c", type=Path, default=DEFAULT_CKPT_C)
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="Decision threshold for the mean-of-3 probability. Re-tune on the "
                         "LOVO OOF sweep (scripts/eval_t3_ensemble_lovo.py) before any submit — "
                         "do NOT assume the deployed 2-way's 0.45 transfers unchanged.")
    ap.add_argument("--weight-a", type=float, default=1.0 / 3.0)
    ap.add_argument("--weight-b", type=float, default=1.0 / 3.0)
    ap.add_argument("--weight-c", type=float, default=1.0 / 3.0)
    args = ap.parse_args()
    weights = (float(args.weight_a), float(args.weight_b), float(args.weight_c))
    if any(weight < 0.0 for weight in weights) or not np.isclose(sum(weights), 1.0, atol=1e-6):
        raise SystemExit(f"ensemble weights must be non-negative and sum to 1, got {weights}")

    data_dir = args.data_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model_a, sz_a, norm_a, arch_a = build_model(args.ckpt_a, device)
    model_b, sz_b, norm_b, arch_b = build_model(args.ckpt_b, device)
    model_c, sz_c, norm_c, arch_c = build_model(args.ckpt_c, device)
    models = [model_a, model_b, model_c]
    sizes = [sz_a, sz_b, sz_c]
    norms = [norm_a, norm_b, norm_c]
    if len(set(sizes)) != 1:
        raise SystemExit(f"image_size mismatch: A={sz_a} B={sz_b} C={sz_c}")
    if len(set(norms)) != 1:
        raise SystemExit(f"imagenet-norm mismatch: A={norm_a} B={norm_b} C={norm_c}")
    image_size = sz_a
    use_imagenet_norm = norm_a

    # Image discovery (identical filtering to infer_t3_ensemble.py)
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
    print(f"Ckpt A ({arch_a}): {args.ckpt_a}")
    print(f"Ckpt B ({arch_b}): {args.ckpt_b}")
    print(f"Ckpt C ({arch_c}): {args.ckpt_c}")
    print(f"Image size: {image_size}, threshold: {args.threshold}, weights={weights}, "
          f"TTA: 8flip x 3 models = 24 passes/frame")
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
        if use_imagenet_norm:
            img_t = (img_t - norm_mean) / norm_std

        if device.type == "cuda":
            torch.cuda.synchronize()
        tp = time.perf_counter()
        probs_list: List[torch.Tensor] = [predict_probs_d4(m, img_t, use_amp=use_amp) for m in models]
        probs = sum(weight * member for weight, member in zip(weights, probs_list))
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
    timings_arr = np.asarray(timings)
    out_dir.joinpath("task3_predictions.json").write_text(
        json.dumps({"cases": records}, ensure_ascii=False, indent=2)
    )
    out_dir.joinpath("_timing.json").write_text(json.dumps({
        "n_frames": len(files), "total_seconds": float(total),
        "ms_per_frame": {"mean": float(timings_arr.mean() * 1000), "p50": float(np.median(timings_arr) * 1000),
                          "p95": float(np.percentile(timings_arr, 95) * 1000), "max": float(timings_arr.max() * 1000)},
        "per_video_seconds": per_video_total,
    }, indent=2))
    print(f"\nTotal: {total:.2f} s for {len(files)} frames")
    print(f"Per-frame ms: mean={timings_arr.mean()*1000:.1f}  median={np.median(timings_arr)*1000:.1f}  "
          f"max={timings_arr.max()*1000:.1f}")
    for v, s in per_video_total.items():
        ok = "OK" if s <= 10 else "OVER 10s"
        print(f"  {v}: {s:.2f}s  [{ok}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
