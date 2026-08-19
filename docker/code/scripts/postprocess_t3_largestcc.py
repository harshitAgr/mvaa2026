#!/usr/bin/env python3
"""Per-frame connected-component cleanup for Task 3 binary 2D predictions.

For each frame in --in-dir (recursing into per-video subfolders), keep only the
connected components whose area is at least --min-area-frac * (largest-CC area).
Optionally apply morphological opening + closing with a small kernel before CC
labelling. Mirrors the T2 largest-CC trick: erases stray far-away FP blobs that
drive HD up without touching the main valve mask.

The default min-area-frac is 0.3, which is permissive enough to keep a real
two-blob frame intact (e.g. anterior + posterior leaflet visible separately)
while still removing typical small far-away false positives.

Usage
-----
    python scripts/postprocess_t3_largestcc.py \\
        --in-dir submission/imagenet_v1/t3_vid \\
        --out-dir submission/largestcc_v1/t3_vid \\
        --min-area-frac 0.3 \\
        --morph-kernel 3
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


def _read_binary_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr > 127).astype(np.uint8)


def _write_binary_mask(path: Path, mask: np.ndarray) -> None:
    out = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    Image.fromarray(out, mode="L").save(path)


def cc_keep_by_area_frac(
    mask: np.ndarray,
    min_area_frac: float = 0.3,
) -> Tuple[np.ndarray, int, int]:
    """Keep CCs with area >= min_area_frac * largest CC area.

    Returns (cleaned_mask, n_cc_before, n_cc_after).
    """
    if not mask.any():
        return mask.copy(), 0, 0
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n <= 1:
        return mask.copy(), int(n), int(n)
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
    largest = float(sizes.max())
    threshold = max(1.0, float(min_area_frac) * largest)
    keep_ids = [i + 1 for i, s in enumerate(sizes) if s >= threshold]
    out = np.isin(labels, keep_ids).astype(np.uint8)
    return out, int(n), int(len(keep_ids))


def morph_clean(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return mask
    k = int(kernel)
    structure = np.ones((k, k), dtype=np.uint8)
    opened = ndimage.binary_opening(mask, structure=structure).astype(np.uint8)
    closed = ndimage.binary_closing(opened, structure=structure).astype(np.uint8)
    return closed


def discover_label_pngs(in_dir: Path) -> List[Path]:
    return sorted(p for p in in_dir.rglob("*_label_bin.png"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="Directory containing T3 predictions (per-video subfolders of *_label_bin.png).")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where to write cleaned predictions. Mirrors in-dir layout.")
    ap.add_argument("--min-area-frac", type=float, default=0.3,
                    help="Keep CCs with area >= this fraction of the largest CC. Default 0.3.")
    ap.add_argument("--morph-kernel", type=int, default=0,
                    help="Side of square structuring element for morphological open+close before CC. "
                         "0 disables morph (default).")
    ap.add_argument("--min-total-fg-frac", type=float, default=0.0,
                    help="Empty-frame gate: after CC cleanup, if the mask's total FG fraction is "
                         "below this, zero the whole mask (emit empty). Targets the deployed model's "
                         "~33%% false-positive rate on held-out no-valve frames; on hidden test frames "
                         "with no valve, empty==perfect. 0 disables (default). Calibrated 0.005 = "
                         "board-safe (no val frame affected) + FN-safe (no real valve >=0.0055 zeroed).")
    ap.add_argument("--copy-json", action="store_true", default=True,
                    help="Copy any task3_predictions.json from in-dir to out-dir verbatim. Default True.")
    ap.add_argument("--no-copy-json", action="store_false", dest="copy_json")
    args = ap.parse_args()

    in_dir = args.in_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = discover_label_pngs(in_dir)
    if not label_files:
        raise SystemExit(f"No *_label_bin.png found under {in_dir}")

    summary = []
    total_pre = 0
    total_post = 0
    n_changed = 0
    n_gated = 0
    for src in label_files:
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        mask = _read_binary_mask(src)
        before_pos = int(mask.sum())

        if args.morph_kernel and args.morph_kernel > 1:
            mask_post = morph_clean(mask, args.morph_kernel)
        else:
            mask_post = mask

        cleaned, n_cc_pre, n_cc_post = cc_keep_by_area_frac(
            mask_post, min_area_frac=float(args.min_area_frac)
        )

        gated = False
        if args.min_total_fg_frac > 0.0 and cleaned.size and \
                (float(cleaned.sum()) / float(cleaned.size)) < float(args.min_total_fg_frac):
            cleaned = np.zeros_like(cleaned)
            gated = True
            n_gated += 1

        after_pos = int(cleaned.sum())
        delta = before_pos - after_pos
        total_pre += before_pos
        total_post += after_pos
        if delta != 0:
            n_changed += 1

        _write_binary_mask(dst, cleaned)
        summary.append({
            "frame": rel.as_posix(),
            "px_before": before_pos,
            "px_after": after_pos,
            "px_removed": delta,
            "cc_before": n_cc_pre,
            "cc_after": n_cc_post,
        })
        print(f"  {rel}: cc {n_cc_pre} -> {n_cc_post}, px {before_pos} -> {after_pos} ({-delta:+d})")

    if args.copy_json:
        for j in sorted(in_dir.glob("*.json")):
            dst_json = out_dir / j.name
            shutil.copy2(j, dst_json)
            print(f"  copied {j.name} -> {dst_json}")

    summary_path = out_dir / "_postproc_summary.json"
    summary_path.write_text(json.dumps({
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "min_area_frac": float(args.min_area_frac),
        "morph_kernel": int(args.morph_kernel),
        "min_total_fg_frac": float(args.min_total_fg_frac),
        "n_frames": len(label_files),
        "n_frames_changed": n_changed,
        "n_frames_empty_gated": n_gated,
        "total_px_before": total_pre,
        "total_px_after": total_post,
        "total_px_removed": total_pre - total_post,
        "per_frame": summary,
    }, indent=2))

    print()
    print(f"Frames processed:        {len(label_files)}")
    print(f"Frames changed:          {n_changed}")
    print(f"Frames empty-gated:      {n_gated}  (min_total_fg_frac={args.min_total_fg_frac})")
    print(f"Total px before -> after: {total_pre} -> {total_post}  ({total_pre - total_post:+d} removed)")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
