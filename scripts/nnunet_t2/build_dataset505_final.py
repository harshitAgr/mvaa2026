#!/usr/bin/env python
"""Build the deployed Task 2 nnU-Net dataset: 105 organizer cases + 70 external cases.

DATA PROVENANCE
---------------
This dataset combines two sources with different origins and licences. Both are
declared explicitly in BLOCKS below and asserted at build time.

  1. MVAA 2026 Task 2 training split - 105 cases.
     Provided by the challenge organizers. Licence: CC BY-NC.

  2. MVSeg2023 public release - 70 cases (`val_001-030`, `test_001-040`).
     Source:  https://huggingface.co/datasets/pcarnahan/MVSeg2023  (gated)
              mirror: Synapse syn51186045
     Licence: CC BY-NC-ND 4.0
     Paper:   Carnahan et al., "DeepMitral: Fully Automatic 3D Echocardiography
              Segmentation for Patient Specific Mitral Valve Modelling",
              MICCAI 2021. DOI 10.1007/978-3-030-87240-3_44
     Same scanner (Philips Epiq), same label scheme, same file convention.

  DISCLOSURE: a byte-level MD5 audit established that the MVAA Task 2 data is the
  MVSeg2023 train/val split verbatim. Consequently `val_001-020` of the external
  block is the same data as the MVAA Task 2 *validation* split; its labels are
  public in the MVSeg2023 release. Those 20 cases are therefore included in
  training here. `val_021-030` and `test_001-040` (50 cases) are disjoint from
  both the MVAA training and validation splits.

  Set MVAA_T2_EXCLUDE_MVAA_VAL=1 to omit those 20 cases and build a 155-case
  dataset that trains on no part of the MVAA validation split.

Task 2 is score-neutralized in the final test phase, so this is a plain deployment
retrain with no held-out split: every selected case is used for training.
"""
import json
import os
import pathlib
import shutil

import SimpleITK as sitk

REPO = pathlib.Path(__file__).resolve().parents[2]

# Source roots. Each directory holds `<case_id>-US.nii.gz` and `<case_id>-label.nii.gz`.
MVAA_TRAIN = pathlib.Path(os.environ.get(
    "MVAA_T2_TRAIN", REPO / "data/reference_data/t2_tee/train"))
MVSEG_VAL = pathlib.Path(os.environ.get(
    "MVSEG2023_VAL", REPO / "data/external/mvseg2023/val"))
MVSEG_TEST = pathlib.Path(os.environ.get(
    "MVSEG2023_TEST", REPO / "data/external/mvseg2023/test"))
DST = pathlib.Path(os.environ.get(
    "MVAA_T2_DST", REPO / "data/nnunet/raw/Dataset505_MVAA_TEE_final"))

EXCLUDE_MVAA_VAL = os.environ.get("MVAA_T2_EXCLUDE_MVAA_VAL", "0") == "1"

# name, source dir, source case ids, destination prefix, origin
BLOCKS = [
    ("mvaa_train", MVAA_TRAIN,
     [f"train_{i:03d}" for i in range(1, 106)], "train", "organizer (MVAA 2026, CC BY-NC)"),
    ("mvseg_val", MVSEG_VAL,
     [f"val_{i:03d}" for i in range((21 if EXCLUDE_MVAA_VAL else 1), 31)],
     "ext_val", "external (MVSeg2023, CC BY-NC-ND 4.0)"),
    ("mvseg_test", MVSEG_TEST,
     [f"test_{i:03d}" for i in range(1, 41)], "ext_test",
     "external (MVSeg2023, CC BY-NC-ND 4.0)"),
]
EXPECTED_TOTAL = 155 if EXCLUDE_MVAA_VAL else 175


def main():
    imagesTr, labelsTr = DST / "imagesTr", DST / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    added, per_block = [], {}
    for name, src_dir, case_ids, prefix, origin in BLOCKS:
        if not src_dir.is_dir():
            raise SystemExit(f"missing source directory for block '{name}': {src_dir}")
        for cid in case_ids:
            img_src = src_dir / f"{cid}-US.nii.gz"
            lab_src = src_dir / f"{cid}-label.nii.gz"
            if not (img_src.exists() and lab_src.exists()):
                raise SystemExit(f"block '{name}': missing {cid} in {src_dir}")
            # keep the block's identity in the destination id so provenance stays
            # visible in nnU-Net splits, logs and preprocessed filenames
            dst_id = cid if prefix == "train" else f"{prefix}_{cid.split('_')[1]}"
            shutil.copy2(img_src, imagesTr / f"{dst_id}_0000.nii.gz")
            shutil.copy2(lab_src, labelsTr / f"{dst_id}.nii.gz")
            added.append(dst_id)
        # origin only — never record the local source path in a published dataset.json
        per_block[name] = {"n": len(case_ids), "origin": origin}
        print(f"  {name:12s} {len(case_ids):4d} cases  <- {origin}")

    if len(added) != EXPECTED_TOTAL:
        raise SystemExit(f"expected {EXPECTED_TOTAL} cases, assembled {len(added)}")
    if len(set(added)) != len(added):
        raise SystemExit("duplicate case ids across blocks")

    # label scheme / dtype consistency between an organizer case and each external block
    ref_img = sitk.ReadImage(str(imagesTr / "train_001_0000.nii.gz"))
    ref_lab = sitk.ReadImage(str(labelsTr / "train_001.nii.gz"))
    for probe in (a for a in added if a.startswith("ext_")):
        img = sitk.ReadImage(str(imagesTr / f"{probe}_0000.nii.gz"))
        lab = sitk.ReadImage(str(labelsTr / f"{probe}.nii.gz"))
        if img.GetPixelIDTypeAsString() != ref_img.GetPixelIDTypeAsString():
            raise SystemExit(f"{probe}: image dtype differs from the organizer cases")
        if lab.GetPixelIDTypeAsString() != ref_lab.GetPixelIDTypeAsString():
            raise SystemExit(f"{probe}: label dtype differs from the organizer cases")

    json.dump({
        "channel_names": {"0": "US"},
        "labels": {"background": 0, "anterior": 1, "posterior": 2},
        "numTraining": len(added),
        "file_ending": ".nii.gz",
        "name": DST.name,
        "description": (
            "MVAA 2026 Task 2 - 3D TEE mitral leaflet segmentation "
            "(anterior=1, posterior=2)."
        ),
        "data_sources": per_block,
    }, open(DST / "dataset.json", "w"), indent=2)

    print(f"\nwrote {DST}  ({len(added)} cases)")
    if not EXCLUDE_MVAA_VAL:
        print("  note: ext_val_001-020 is the same data as the MVAA Task 2 validation "
              "split (labels public via MVSeg2023).")
        print("  set MVAA_T2_EXCLUDE_MVAA_VAL=1 to build the 155-case variant without it.")


if __name__ == "__main__":
    main()
