"""Build Dataset512_T1CT_selftrain (Arm A self-training).

27 real labeled + K gated pseudo-labeled cases, plus a `splits_final.json` that
MIRRORS Dataset511 (same real cases held out per fold, for CV comparability) and
adds ALL pseudo cases to every fold's train set, oversampling the real cases so
real:pseudo ≈ 1:target-ratio by draw frequency. Pseudo cases get a `PLBL_` prefix
so they never collide with real `T1CT_` ids and never enter a held-out fold.
"""
import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label

# Repo root: scripts/nnunet_t1/<this file> -> parents[2]. Override with MVAA_ROOT.
ROOT = os.environ.get("MVAA_ROOT", str(Path(__file__).resolve().parents[2]))
# Every root is overridable by environment variable; defaults follow the layout in TRAINING.md.
#   MVAA_T1_LABELED     the 27 labeled CT cases (images/ and labels/)
#   MVAA_T1_UNLABELED   the unlabeled CT pool
#   MVAA_T1_PSEUDO      raw pseudo-label predictions over the unlabeled pool
#   MVAA_T1_GATE        gate_pseudo.py report to read
#   MVAA_T1_SPLITS      Dataset511 splits_final.json, mirrored so folds stay comparable
#   MVAA_T1_DST         output nnU-Net raw dataset to create
LAB = os.environ.get("MVAA_T1_LABELED", f"{ROOT}/data/reference_data/t1_ct/train/labeled")
UNL = os.environ.get("MVAA_T1_UNLABELED", f"{ROOT}/data/reference_data/t1_ct/train/unlabeled")
PSEUDO = os.environ.get("MVAA_T1_PSEUDO", f"{ROOT}/data/nnunet/pseudo_raw")
GATE = os.environ.get("MVAA_T1_GATE", f"{ROOT}/data/nnunet/pseudo_gated/gate_report.json")
SUP_SPLITS = os.environ.get(
    "MVAA_T1_SPLITS", f"{ROOT}/data/nnunet/preprocessed/Dataset511_T1CT/splits_final.json")
DST = os.environ.get("MVAA_T1_DST", f"{ROOT}/data/nnunet/raw/Dataset512_T1CT_selftrain")


def largest_cc(m):
    lab, n = cc_label(m)
    if n <= 1:
        return m
    s = np.bincount(lab.ravel())
    s[0] = 0
    return (lab == s.argmax()).astype(m.dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-ratio", type=float, default=2.5, help="pseudo:real by draw")
    ap.add_argument("--max-pseudo", type=int, default=0, help="0 = all passing")
    args = ap.parse_args()

    img, lbl = f"{DST}/imagesTr", f"{DST}/labelsTr"
    os.makedirs(img, exist_ok=True)
    os.makedirs(lbl, exist_ok=True)

    real_ids = []
    for ip in sorted(glob.glob(f"{LAB}/images/*.nii.gz")):
        cid = os.path.basename(ip).replace(".nii.gz", "")
        shutil.copy(ip, f"{img}/T1CT_{cid}_0000.nii.gz")
        shutil.copy(f"{LAB}/labels/{cid}-seg.nii.gz", f"{lbl}/T1CT_{cid}.nii.gz")
        real_ids.append(f"T1CT_{cid}")

    passing = [c["cid"] for c in json.load(open(GATE))["passing"]]
    if args.max_pseudo > 0:
        passing = passing[: args.max_pseudo]
    pl_ids = []
    for mcid in passing:
        ul = mcid.replace("T1CT_", "")
        src_img = f"{UNL}/{ul}.nii.gz"
        if not os.path.exists(src_img):
            continue
        m = nib.load(f"{PSEUDO}/{mcid}.nii.gz")
        arr = largest_cc((np.asanyarray(m.dataobj) > 0).astype(np.uint8))
        plid = f"PLBL_{ul}"
        shutil.copy(src_img, f"{img}/{plid}_0000.nii.gz")
        nib.save(nib.Nifti1Image(arr, m.affine, m.header), f"{lbl}/{plid}.nii.gz")
        pl_ids.append(plid)

    json.dump({"channel_names": {"0": "CT"}, "labels": {"background": 0, "valve": 1},
               "numTraining": len(real_ids) + len(pl_ids), "file_ending": ".nii.gz"},
              open(f"{DST}/dataset.json", "w"), indent=2)

    sup = json.load(open(SUP_SPLITS))
    splits, F = [], 1
    for f in sup:
        rtr, rval = f["train"], f["val"]
        F = max(1, round((len(pl_ids) / args.target_ratio) / max(1, len(rtr))))
        splits.append({"train": rtr * F + pl_ids, "val": rval})
    json.dump(splits, open(f"{DST}/splits_final.json", "w"), indent=2)
    print(f"Dataset512: {len(real_ids)} real + {len(pl_ids)} pseudo | oversample F~{F} | "
          f"splits written to {DST}/splits_final.json (copy into preprocessed after plan_and_preprocess)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
