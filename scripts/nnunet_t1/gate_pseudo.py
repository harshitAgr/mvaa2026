"""Gate ensemble pseudo-labels for Arm A self-training (anatomy hard filter).

Reads the 5-fold-ensemble argmax masks (nnU-Net predictions over the unlabeled
CT pool) and keeps only anatomically plausible ones:
  - non-empty
  - exactly 1 connected component (all 27 real GT are 1-CC)
  - FG volume in [vol-min, vol-max] mm^3 (measured labeled range 1854-4639)
  - FG fraction in [frac-min, frac-max]
  - source finest-axis spacing < max-fine-spacing mm (drop the ~9% OOD-coarse tail)

The mask header carries the source spacing (nnU-Net resamples predictions back
to source grid), so no separate image read is needed. Writes a yield report
(no silent truncation) + the list of passing case-ids for the Arm-A dataset build.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label

# Repo root: scripts/nnunet_t1/<this file> -> parents[2]. Override with MVAA_ROOT.
ROOT = os.environ.get("MVAA_ROOT", str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo-dir", default=f"{ROOT}/data/nnunet/pseudo_raw")
    ap.add_argument("--out-json", default=f"{ROOT}/data/nnunet/pseudo_gated/gate_report.json")
    ap.add_argument("--vol-min", type=float, default=1500.0)
    ap.add_argument("--vol-max", type=float, default=5000.0)
    ap.add_argument("--frac-min", type=float, default=0.005)
    ap.add_argument("--frac-max", type=float, default=0.04)
    ap.add_argument("--max-fine-spacing", type=float, default=0.55)
    ap.add_argument("--min-dominance", type=float, default=0.90,
                    help="largest CC must be >= this fraction of foreground (replaces strict 1-CC; "
                         "tiny spurious islands are removed by largest-CC at build time)")
    args = ap.parse_args()

    passing, dropped = [], {}
    dom_hist = {"0.80": 0, "0.90": 0, "0.95": 0, "0.99": 0}
    files = sorted(glob.glob(f"{args.pseudo_dir}/*.nii.gz"))
    for p in files:
        cid = os.path.basename(p).replace(".nii.gz", "")
        im = nib.load(p)
        arr = (np.asanyarray(im.dataobj) > 0).astype(np.uint8)
        zooms = im.header.get_zooms()[:3]
        voxvol, fine = float(np.prod(zooms)), float(min(zooms))
        fg = int(arr.sum())
        if fg == 0:
            dropped.setdefault("empty", []).append(cid)
            continue

        # Evaluate the LARGEST-CC-cleaned mask (what we actually train on),
        # plus a dominance score = largest component / total foreground.
        lab, ncc = cc_label(arr)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        lvox = int(sizes.max())
        dominance = lvox / fg
        for k in dom_hist:
            if dominance >= float(k):
                dom_hist[k] += 1
        clean_vol = lvox * voxvol
        clean_frac = lvox / arr.size

        reason = None
        if dominance < args.min_dominance:
            reason = "low_dominance"
        elif not (args.vol_min <= clean_vol <= args.vol_max):
            reason = "vol_oob"
        elif not (args.frac_min <= clean_frac <= args.frac_max):
            reason = "frac_oob"
        elif fine >= args.max_fine_spacing:
            reason = "coarse_spacing"

        if reason:
            dropped.setdefault(reason, []).append(cid)
        else:
            passing.append({"cid": cid, "vol_mm3": round(clean_vol), "fg_frac": round(clean_frac, 4),
                            "fine_spacing": round(fine, 3), "n_cc": int(ncc), "dominance": round(dominance, 3)})

    report = {
        "n_total": len(files),
        "n_pass": len(passing),
        "drop_counts": {k: len(v) for k, v in dropped.items()},
        "dominance_histogram": dom_hist,
        "thresholds": vars(args),
        "passing": passing,
        "dropped_examples": {k: v[:8] for k, v in dropped.items()},
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"PASS {report['n_pass']}/{report['n_total']}  |  drops: {report['drop_counts']}")
    print(f"report -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
