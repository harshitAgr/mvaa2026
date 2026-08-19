"""Convert the 27 labeled T1 CTs into nnU-Net v2 raw format (Dataset511_T1CT)."""
import json, os, shutil
from pathlib import Path

# Repo root: scripts/nnunet_t1/<this file> -> parents[2]. Override with MVAA_ROOT,
# or point the roots directly with MVAA_T1_SRC / MVAA_T1_RAW_DST.
ROOT = Path(os.environ.get("MVAA_ROOT", Path(__file__).resolve().parents[2]))
SRC = Path(os.environ.get("MVAA_T1_SRC", ROOT / "data/reference_data/t1_ct/train/labeled"))
DST = Path(os.environ.get("MVAA_T1_RAW_DST", ROOT / "data/nnunet/raw/Dataset511_T1CT"))


def main():
    img_dir, lbl_dir = DST / "imagesTr", DST / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted((SRC / "images").glob("*.nii.gz"))
    assert len(images) == 27, f"expected 27 labeled images, found {len(images)}"
    n = 0
    for img in images:
        cid = img.name.replace(".nii.gz", "")  # e.g. "0001"
        seg = SRC / "labels" / f"{cid}-seg.nii.gz"
        assert seg.exists(), f"missing label for {cid}: {seg}"
        shutil.copy(img, img_dir / f"T1CT_{cid}_0000.nii.gz")  # _0000 = channel 0
        shutil.copy(seg, lbl_dir / f"T1CT_{cid}.nii.gz")
        n += 1

    dataset = {
        "channel_names": {"0": "CT"},  # "CT" -> nnU-Net CTNormalization
        "labels": {"background": 0, "valve": 1},  # binary FG (verified unique {0,1})
        "numTraining": n,
        "file_ending": ".nii.gz",
    }
    (DST / "dataset.json").write_text(json.dumps(dataset, indent=2))
    print(f"wrote {n} cases + dataset.json to {DST}")


if __name__ == "__main__":
    main()
