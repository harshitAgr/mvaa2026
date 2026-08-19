#!/usr/bin/env python3
"""MVAA 2026 — unified Docker inference orchestrator (all 3 tasks, one process).

Reads a test input root with per-task subfolders and writes Codabench-layout
predictions to an output root:

  <input>/t1_ct/   *.nii.gz             -> <output>/t1_ct/<id>-pred.nii.gz            + task1_predictions.json
  <input>/t2_tee/  *-US.nii.gz|*.nii.gz -> <output>/t2_tee/<case>-pred.nii.gz         + task2_predictions.json
  <input>/t3_vid/  <video>/*.png        -> <output>/t3_vid/<video>/<frame>_label_bin.png + task3_predictions.json

Models (baked into the image; the active configuration is selected by build args/env):
  T1: nnU-Net v2 3d_fullres. Dataset512 self-train folds, optionally fused at family level
      with the Dataset511 LargePatch folds.
  T2: nnU-Net v2 3d_fullres, Dataset505 (SafeMirror trainer + LargePatch plans), fold_all.
  T3: selectable ensemble path; the deployed configuration is the equal probability mean of
      three surgical-DINOv2 members with D4 TTA and threshold 0.45.

Post-processing:
  default:            T1 largest-CC (binary); T2 per-class largest-CC; T3 small-CC cleanup.
                      The deployed T3 path disables cleanup entirely.
  --extra-postproc:   additionally per-class hole-fill on T1/T2. Off by default.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

T1_DATASET = os.environ.get("MVAA_T1_DATASET", "512")
T1_TRAINER = os.environ.get("MVAA_T1_TRAINER", "nnUNetTrainer_250epochs")
# 5-member Dataset512 self-train ensemble. All five folds share the same trainer,
# plans and dataset. Fall back to the single-model baseline with MVAA_T1_FOLD=0.
T1_FOLD = os.environ.get("MVAA_T1_FOLD", "0 1 2 3 4")
T1_FUSION_LARGEPATCH = os.environ.get("MVAA_T1_FUSION_LARGEPATCH", "0") == "1"
T1_LP_DATASET = os.environ.get("MVAA_T1_LP_DATASET", "511")
T1_LP_TRAINER = os.environ.get("MVAA_T1_LP_TRAINER", "nnUNetTrainer_250epochs")
T1_LP_PLANS = os.environ.get("MVAA_T1_LP_PLANS", "nnUNetPlans_LargePatch")
T1_LP_FOLD = os.environ.get("MVAA_T1_LP_FOLD", "0 1 2 3 4")
T2_DATASET, T2_TRAINER, T2_PLANS = "505", "nnUNetTrainer_SafeMirror", "nnUNetPlans_LargePatch"
MVAA = Path(os.environ.get("MVAA_HOME", "/opt/mvaa"))
T3_CKPT_A = os.environ.get("MVAA_T3_CKPT_A", "/opt/weights/t3/imagenet_dr_epoch070.pt")
T3_CKPT_B = os.environ.get("MVAA_T3_CKPT_B", "/opt/weights/t3/surgenetxl_best.pt")
# Rejected architecture-diversity 3rd member (SegFormer-MiT-B3, all-vid supervised).
# It remains available for reproducibility but must be explicitly opted into: merely baking the
# checkpoint into docker/weights must never change the proven two-way deployment path.
T3_CKPT_C = os.environ.get("MVAA_T3_CKPT_C", "").strip()
T3_PRETRAINED_WEIGHTS = os.environ.get("MVAA_T3_PRETRAINED_WEIGHTS", "").strip()
T3_PRETRAINED_FP32_WEIGHTS = os.environ.get("MVAA_T3_PRETRAINED_FP32_WEIGHTS", "").strip()
T3_SURGENETDINO_CV2 = os.environ.get("MVAA_T3_SURGENETDINO_CV2", "0") == "1"
T3_SURGENETDINO_TRI = os.environ.get("MVAA_T3_SURGENETDINO_TRI", "0") == "1"
T3_SURGENETDINO_PURE_TRI = os.environ.get("MVAA_T3_SURGENETDINO_PURE_TRI", "0") == "1"
T3_SURGENETDINO_C0 = os.environ.get(
    "MVAA_T3_SURGENETDINO_C0", "/opt/weights/t3/surgenetdino_v2b_979A_ema_fp16.pt"
)
T3_SURGENETDINO_C1 = os.environ.get(
    "MVAA_T3_SURGENETDINO_C1", "/opt/weights/t3/surgenetdino_v2b_675A_ema_fp16.pt"
)
T3_SURGENETDINO_C0_SHA256 = "fbc3db9b0c308d834b615507daa42976aafa88c4a70558338163c8df20dc6683"
T3_SURGENETDINO_C1_SHA256 = "8ad5ce272f60147646ee056e7abfe13a5b858184d5919513d264fca31147000d"
T3_SURGENETDINO_C0_SOURCE_SHA256 = "f1c0de37a34fe336dcf13208ffe1bbe60e057ca4f052bed05547bfb6eb10ecbe"
T3_SURGENETDINO_C1_SOURCE_SHA256 = "916df1ac4d372a3c2e55c173666435de64a50039713423ffa1a468066c3185a6"
T3_SURGENETDINO_ALL6_C = os.environ.get(
    "MVAA_T3_SURGENETDINO_ALL6_C", "/opt/weights/t3/surgenetdino_v2b_all6_ema_fp16.pt"
)
T3_SURGENETDINO_ALL6_SHA256 = "c6e9b4cf605b52a55171ab01cb9176d054fdff1acaf2d46e2bc2e3bbcf8fb159"
T3_SURGENETDINO_ALL6_SOURCE_SHA256 = "598e310137e361db829e953f91da2f727af2075341e899e7901a01bd46079351"
T3_THRESHOLD = os.environ.get("MVAA_T3_THRESHOLD", "0.45")
# A:B probability fusion weight for the two-member path. 0.60 was chosen on the 6-video
# leakage-free LOVO out-of-fold set: versus 0.50 it cuts HD 111.09 -> 91.16 px and ASD
# 46.74 -> 29.68 px, halves empty-frame false positives (3 -> 1) and improves 5/6 videos
# (paired HD t(5) = -2.97). Set MVAA_T3_WEIGHT_A=0.5 to revert. Unused by the deployed
# three-member surgical-DINOv2 path, which weights its members equally.
T3_WEIGHT_A = os.environ.get("MVAA_T3_WEIGHT_A", "0.6")
T3_WEIGHT_B = os.environ.get("MVAA_T3_WEIGHT_B", "0.3")
T3_WEIGHT_C = os.environ.get("MVAA_T3_WEIGHT_C", "0.1")
T3_MIN_AREA_FRAC = os.environ.get("MVAA_T3_MIN_AREA_FRAC", "0.1")
# Empty-frame gate: zero a frame's mask if total FG < this fraction. Targets the
# model's false-positive rate on no-valve frames (an empty prediction on an empty
# frame is perfect). 0.005 leaves every observed real valve intact (smallest observed
# = 0.0055). Set MVAA_T3_MIN_FG_FRAC=0 to disable.
T3_MIN_FG_FRAC = os.environ.get("MVAA_T3_MIN_FG_FRAC", "0.005")

# T2 cascade (leaflet-identity refinement, opt-in). Stage-1 = the model above; stage-2 =
# a 2-class nnU-Net on a valve-ROI crop @0.25mm that LEARNS the anterior/posterior split.
# A GT-free conditional merge keeps stage-1's boundary and overrides the A/P split ONLY
# where stage-2 disagrees with stage-1 by > TAU (the leaflet-confusion signal). Fixes the
# ~8% confusion tail (OOF: tail ASSD -29%) with zero downside on the rest. OPT-IN: set
# MVAA_T2_CASCADE=1.
# Stage-2 TTA defaults ON — that is the config TAU=0.08 was calibrated against on OOF
# (fires only on true confusion; 0/20 on the current val set). Disabling it
# (MVAA_T2_CASCADE_STAGE2_NOTTA=1) is ~3x faster (~1.7 vs 5.4 s/case) for the <=10s test
# budget, BUT shifts the disagreement distribution up and over-fires (2/20 on val) —
# RE-CALIBRATE TAU on TTA-off OOF before using it.
T2_STAGE2_DATASET, T2_STAGE2_TRAINER = "502", "nnUNetTrainer_250epochs"
T2_CASCADE = os.environ.get("MVAA_T2_CASCADE", "0") == "1"
T2_CASCADE_STAGE2_NOTTA = os.environ.get("MVAA_T2_CASCADE_STAGE2_NOTTA", "0") == "1"
T2_CASCADE_MARGIN_MM = float(os.environ.get("MVAA_T2_CASCADE_MARGIN_MM", "10.0"))
T2_CASCADE_ISO = float(os.environ.get("MVAA_T2_CASCADE_ISO", "0.25"))
T2_CASCADE_TAU = float(os.environ.get("MVAA_T2_CASCADE_TAU", "0.08"))


def log(msg: str) -> None:
    print(f"[mvaa] {msg}", flush=True)


# ---------------------------------------------------------------------------
# test_cases.json manifest support (mirrors the organizer reference container,
# db0725/mvaa-baseline-infer). If /input/test_cases.json is present it DEFINES the
# scoring case_ids; the private scorer keys its reference labels by those ids. We
# honor it by RE-KEYING each task's output JSON to the manifest ids, matched by the
# resolved input image file — a pure post-processing pass that leaves the models,
# masks and segmentation pointers untouched. Absent manifest OR manifest ids that
# already equal our filename-derived stems => a no-op, so the file-discovery path is
# unchanged. Helpers ported from the organizer reference container so our identity
# semantics match it byte-for-byte.
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TASK_FOLDERS = ("t1_ct", "t2_tee", "t3_vid")
TASK_NUM = {"t1_ct": 1, "t2_tee": 2, "t3_vid": 3}


def is_label_like(path: Path) -> bool:
    n = path.name.lower()
    return (n.endswith("-seg.nii.gz") or n.endswith("-label.nii.gz")
            or n.endswith("-pred.nii.gz") or n.endswith("_label_bin.png")
            or "_png_label_vis" in n or n.endswith("_png_label.tar"))


def _existing_dirs(cands) -> list[Path]:
    out, seen = [], set()
    for p in cands:
        if p.exists() and p.is_dir():
            rp = p.resolve()
            if rp not in seen:
                out.append(p); seen.add(rp)
    return out


def _normalize_task(value):
    s = str(value or "").strip().lower()
    if s in {"t1", "task1", "task_1", "task1_ct", "t1_ct", "ct"}: return "t1_ct"
    if s in {"t2", "task2", "task_2", "task2_tee", "t2_tee", "tee"}: return "t2_tee"
    if s in {"t3", "task3", "task_3", "task3_vid", "t3_vid", "video", "vid"}: return "t3_vid"
    return None


def _infer_task(item: dict, default_task=None):
    for key in ("task", "task_id", "task_name", "folder", "modality"):
        t = _normalize_task(item.get(key))
        if t: return t
    if default_task: return default_task
    text = " ".join(str(item.get(k, "")) for k in
                    ("case_id", "image", "image_path", "image_rel_path", "path", "filename", "file")).lower()
    if "t1_ct" in text or "task1" in text: return "t1_ct"
    if "t2_tee" in text or "task2" in text or "-us.nii" in text: return "t2_tee"
    if "t3_vid" in text or "task3" in text or text.endswith(".png") or "t3_" in text: return "t3_vid"
    return None


def _extract_entries(raw) -> list:
    entries = []
    if isinstance(raw, list):
        return [(None, x) for x in raw if isinstance(x, dict)]
    if not isinstance(raw, dict):
        return entries
    for key in ("cases", "test_cases", "data", "samples", "inputs"):
        v = raw.get(key)
        if isinstance(v, list):
            entries.extend((None, x) for x in v if isinstance(x, dict))
    for key, v in raw.items():
        t = _normalize_task(key)
        if t and isinstance(v, list):
            entries.extend((t, x) for x in v if isinstance(x, dict))
        elif t and isinstance(v, dict):
            entries.extend((t, item) for _, item in _extract_entries(v))
    return entries


def _item_image(item: dict) -> str:
    for key in ("image", "image_path", "image_rel_path", "path", "filename", "file", "input", "input_path"):
        v = item.get(key)
        if v: return str(v)
    return ""


def _task_roots(input_dir: Path, task: str) -> list[Path]:
    return _existing_dirs([input_dir / task / "images", input_dir / task])


def _resolve_image(input_dir: Path, task: str, item: dict):
    val = _item_image(item).strip()
    cands: list[Path] = []
    if val:
        raw = Path(val)
        if raw.is_absolute():
            cands.append(raw)
        else:
            cands += [input_dir / raw, input_dir / task / raw, input_dir / task / "images" / raw]
            for root in _task_roots(input_dir, task):
                cands += [root / raw, root / Path(val).name]
    for p in cands:
        if p.exists() and p.is_file():
            return p.resolve()
    base = Path(val).name if val else ""
    roots = _task_roots(input_dir, task) or [input_dir / task, input_dir]
    if base:
        for root in roots:
            if root.exists():
                hits = sorted(p for p in root.rglob(base) if p.is_file())
                if hits: return hits[0].resolve()
    cid = str(item.get("case_id") or item.get("id") or "").strip()
    if not cid: return None
    for root in roots:
        if not root.exists(): continue
        if task == "t1_ct":
            hits = sorted(p for p in root.rglob(f"{cid}*.nii.gz") if "-US" not in p.name and not is_label_like(p))
        elif task == "t2_tee":
            hits = sorted(p for p in root.rglob(f"{cid}*-US.nii.gz") if not is_label_like(p))
        else:
            hits = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                          and not is_label_like(p) and (p.stem == cid or p.name == cid))
        if hits: return hits[0].resolve()
    return None


def _our_stem(task: str, path: Path) -> str:
    """The case_id our file-discovery path assigns to this input file (must mirror
    task1/task2 case_of and infer_t3_ensemble's p.stem)."""
    n = path.name
    if n.endswith(".nii.gz"):
        s = n[: -len(".nii.gz")]
    elif n.endswith(".nii"):
        s = n[: -len(".nii")]
    else:
        s = path.stem
    if task == "t2_tee" and s.endswith("-US"):
        s = s[: -len("-US")]
    return s


def load_manifest(input_dir: Path) -> dict:
    """Parse /input/test_cases.json -> {task: [{case_id, image_path}]}; empty lists if absent/unparseable."""
    empty = {t: [] for t in TASK_FOLDERS}
    path = input_dir / "test_cases.json"
    if not path.exists():
        return empty
    try:
        raw = json.load(open(path))
    except Exception as e:
        log(f"WARNING: /input/test_cases.json present but unparseable ({e}) — using file discovery")
        return empty
    out = {t: [] for t in TASK_FOLDERS}
    for default_task, item in _extract_entries(raw):
        task = _infer_task(item, default_task)
        if task not in out:
            continue
        cid = str(item.get("case_id") or item.get("id") or "").strip()
        if not cid:
            continue
        out[task].append({"case_id": cid, "image_path": _resolve_image(input_dir, task, item)})
    total = sum(len(v) for v in out.values())
    if total:
        log(f"test_cases.json manifest: {total} cases ("
            + ", ".join(f"{t}={len(out[t])}" for t in TASK_FOLDERS if out[t]) + ")")
    return out


def rekey_to_manifest(out_root: Path, task: str, entries: list) -> None:
    """Rewrite a task JSON's case_ids to the manifest ids, matched by resolved image file.
    Segmentation pointers and mask files are left untouched (the scorer follows the pointer)."""
    if not entries:
        return
    jp = out_root / task / f"task{TASK_NUM[task]}_predictions.json"
    if not jp.exists():
        log(f"{task}: manifest present but {jp.name} missing — cannot re-key (WATCH)")
        return
    data = json.load(open(jp))
    by_stem = {c.get("case_id"): c for c in data.get("cases", [])}
    n, unmatched = 0, []
    for e in entries:
        img, cid = e["image_path"], e["case_id"]
        if img is None:
            unmatched.append(cid); continue
        c = by_stem.get(_our_stem(task, Path(img)))
        if c is None:
            unmatched.append(cid); continue
        if c.get("case_id") != cid:
            c["case_id"] = cid; n += 1
    json.dump(data, open(jp, "w"), indent=2)
    msg = f"{task}: re-keyed {n}/{len(entries)} case_ids to test_cases.json manifest"
    if unmatched:
        msg += f" — WARNING {len(unmatched)} manifest ids had no matching prediction: {unmatched[:5]}"
    log(msg)


def discover_t3_images(base: Path) -> list[Path]:
    """Discover model inputs using the same case-insensitive formats as T3 inference."""
    return sorted(
        path for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not is_label_like(path)
    )


def find_task_dir(in_root: Path, subname: str, exts: tuple[str, ...]) -> tuple[Path, list[Path]]:
    """Locate the dir that holds a task's input files; robust to one level of nesting."""
    base = in_root / subname if (in_root / subname).is_dir() else in_root
    files: list[Path] = []
    for e in exts:
        files += list(base.rglob(f"*{e}"))
    files = [f for f in sorted(set(files)) if not f.name.endswith("-pred.nii.gz")]
    return base, files


def keep_largest_cc(binary: np.ndarray) -> np.ndarray:
    lbl, n = ndimage.label(binary)
    if n <= 1:
        return binary
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    return lbl == int(sizes.argmax())


def stage_symlinks(files: list[Path], stage: Path, case_of) -> dict:
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    idmap = {}
    for f in files:
        cid = case_of(f)
        idmap[cid] = f
        os.symlink(f.resolve(), stage / f"{cid}_0000.nii.gz")
    return idmap


def run_nnunet(stage: Path, raw_out: Path, dataset: str, trainer: str, fold: str = "0",
               plans: str = "nnUNetPlans", disable_tta: bool = False,
               save_probabilities: bool = False, sequential_io: bool = False) -> None:
    shutil.rmtree(raw_out, ignore_errors=True)
    raw_out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["nnUNet_results"] = os.environ.get("nnUNet_results", "/opt/weights/nnunet")
    env.setdefault("nnUNet_raw", "/tmp/nnraw")
    env.setdefault("nnUNet_preprocessed", "/tmp/nnprep")
    folds = fold.split()  # "-f" takes one token per fold; "0 1 2 3 4" must not arrive as a single argv
    if not folds:
        raise SystemExit(f"empty fold specification: {fold!r}")
    cmd = ["nnUNetv2_predict", "-i", str(stage), "-o", str(raw_out),
           "-d", dataset, "-c", "3d_fullres", "-f", *folds,
           "-tr", trainer, "-p", plans]
    if disable_tta:
        cmd.append("--disable_tta")  # budget lever: drops 8x mirroring TTA (~Nx faster, small DSC cost)
    if save_probabilities:
        cmd.append("--save_probabilities")
    if sequential_io:
        # nnU-Net 2.7 switches to predict_from_files_sequential only when both are zero.
        # This bypasses preprocessing_iterator_fromfiles and its Tensor.pin_memory()
        # call, which failed when a second sequential predictor process was started
        # after the first family had completed all its cases.
        cmd += ["-npp", "0", "-nps", "0"]
    device = os.environ.get("MVAA_DEVICE", "").strip()
    if device:
        cmd += ["-device", device]  # MVAA_DEVICE=cpu enables CPU-only validation of the image
    log("run: " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def run_nnunet_ensemble(inputs: list[Path], output: Path) -> None:
    if len(inputs) < 2:
        raise SystemExit("nnU-Net probability ensemble requires at least two input folders")
    shutil.rmtree(output, ignore_errors=True)
    cmd = ["nnUNetv2_ensemble", "-i", *[str(path) for path in inputs], "-o", str(output)]
    log("run: " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=dict(os.environ))


def task1(in_root: Path, out_root: Path, timing: dict, extra_postproc: bool) -> None:
    base, files = find_task_dir(in_root, "t1_ct", (".nii.gz",))
    if not (in_root / "t1_ct").is_dir():
        files = [f for f in files if not f.name.endswith("-US.nii.gz")]  # flat /input: -US is T2/TEE, never a T1 CT
    if not files:
        log("T1: no inputs found, skipping")
        return
    log(f"T1: {len(files)} cases from {base}")
    out = out_root / "t1_ct"
    out.mkdir(parents=True, exist_ok=True)
    stage, raw = Path("/tmp/t1_stage"), Path("/tmp/t1_raw")
    idmap = stage_symlinks(files, stage, lambda f: f.name[: -len(".nii.gz")])
    t0 = time.time()
    disable_tta = os.environ.get("MVAA_T1_DISABLE_TTA", "0") == "1"
    if T1_FUSION_LARGEPATCH:
        raw_a, raw_b = Path("/tmp/t1_raw_family_a"), Path("/tmp/t1_raw_family_b")
        log(
            "T1 fusion: 0.50 * "
            f"Dataset{T1_DATASET}/{T1_TRAINER}/nnUNetPlans/folds={T1_FOLD} + 0.50 * "
            f"Dataset{T1_LP_DATASET}/{T1_LP_TRAINER}/{T1_LP_PLANS}/folds={T1_LP_FOLD}"
        )
        run_nnunet(
            stage, raw_a, T1_DATASET, T1_TRAINER, fold=T1_FOLD,
            disable_tta=disable_tta, save_probabilities=True,
        )
        run_nnunet(
            stage, raw_b, T1_LP_DATASET, T1_LP_TRAINER, fold=T1_LP_FOLD,
            plans=T1_LP_PLANS, disable_tta=disable_tta, save_probabilities=True,
            sequential_io=True,
        )
        run_nnunet_ensemble([raw_a, raw_b], raw)
    else:
        log(f"T1 model: dataset={T1_DATASET} trainer={T1_TRAINER} fold={T1_FOLD}")
        run_nnunet(stage, raw, T1_DATASET, T1_TRAINER, fold=T1_FOLD,
                   disable_tta=disable_tta)
    cases = []
    for cid in sorted(idmap):
        img = nib.load(str(raw / f"{cid}.nii.gz"))
        arr = np.asarray(img.dataobj)
        mask = keep_largest_cc(arr > 0)
        if extra_postproc:
            mask = ndimage.binary_fill_holes(mask)
        m = mask.astype(np.uint8)
        nib.save(nib.Nifti1Image(m, img.affine, img.header), str(out / f"{cid}-pred.nii.gz"))
        cases.append({"case_id": cid, "segmentation": f"{cid}-pred.nii.gz"})
    json.dump({"cases": cases}, open(out / "task1_predictions.json", "w"), indent=2)
    dt = time.time() - t0
    timing["task1"] = {"seconds": round(dt, 2), "cases": len(cases),
                       "per_case_incl_coldstart": round(dt / len(cases), 2)}
    log(f"T1 done: {len(cases)} cases in {dt:.1f}s ({dt/len(cases):.2f}s/case incl one-time cold-start)")


def _stage2_present() -> bool:
    results = Path(os.environ.get("nnUNet_results", "/opt/weights/nnunet"))
    return any(results.glob(f"Dataset{T2_STAGE2_DATASET}_*"))


def cascade_refine_t2(idmap: dict, out: Path) -> int:
    """GT-free refinement of the stage-1 T2 predictions already written to `out`.

    For each case: build a valve-ROI (stage-1 union + margin, @0.25mm) -> stage-2 predict
    (TTA off) -> resample back -> override the A/P split ONLY where stage-2 disagrees with
    stage-1 by > TAU (leaflet-confusion), keeping stage-1's boundary. Returns #cases fired.
    """
    import SimpleITK as sitk

    def resample_iso(im, interp):
        osz = [int(round(s * spc / T2_CASCADE_ISO)) for s, spc in zip(im.GetSize(), im.GetSpacing())]
        r = sitk.ResampleImageFilter(); r.SetOutputSpacing([T2_CASCADE_ISO] * 3); r.SetSize(osz)
        r.SetOutputOrigin(im.GetOrigin()); r.SetOutputDirection(im.GetDirection())
        r.SetInterpolator(interp); r.SetTransform(sitk.Transform()); r.SetDefaultPixelValue(0)
        return r.Execute(im)

    # All arrays here are SimpleITK (z,y,x) order — never mix with nibabel (x,y,z).
    s2_in, s2_raw = Path("/tmp/t2_roi_in"), Path("/tmp/t2_roi_raw")
    shutil.rmtree(s2_in, ignore_errors=True); s2_in.mkdir(parents=True, exist_ok=True)
    for cid, f in idmap.items():
        img = sitk.ReadImage(str(f))
        p1img = sitk.ReadImage(str(out / f"{cid}-pred.nii.gz"))
        p1 = sitk.GetArrayFromImage(p1img).astype(np.uint8)
        if p1.shape != sitk.GetArrayFromImage(img).shape:
            log(f"T2 cascade: {cid} pred/img shape mismatch — skip refine"); continue
        U = p1 > 0
        if U.sum() == 0:
            continue  # no valve -> nothing to refine
        zyx = np.argwhere(U); lo = zyx.min(0); hi = zyx.max(0) + 1
        sp = np.array(img.GetSpacing()); mv = np.ceil(T2_CASCADE_MARGIN_MM / sp[::-1]).astype(int)  # sp=(x,y,z); [::-1]=(z,y,x)
        lo = np.maximum(lo - mv, 0); hi = np.minimum(hi + mv, np.array(p1.shape))
        start = [int(lo[2]), int(lo[1]), int(lo[0])]   # sitk index = (x,y,z)
        size = [int(hi[2] - lo[2]), int(hi[1] - lo[1]), int(hi[0] - lo[0])]
        sitk.WriteImage(resample_iso(sitk.RegionOfInterest(img, size, start), sitk.sitkLinear),
                        str(s2_in / f"{cid}_0000.nii.gz"))
    run_nnunet(s2_in, s2_raw, T2_STAGE2_DATASET, T2_STAGE2_TRAINER, fold="all",
               disable_tta=T2_CASCADE_STAGE2_NOTTA, sequential_io=True)

    fired = 0
    for cid, f in idmap.items():
        s2p = s2_raw / f"{cid}.nii.gz"
        if not s2p.exists():
            continue
        p1img = sitk.ReadImage(str(out / f"{cid}-pred.nii.gz"))
        p1 = sitk.GetArrayFromImage(p1img).astype(np.uint8)
        r = sitk.ResampleImageFilter(); r.SetReferenceImage(p1img)
        r.SetInterpolator(sitk.sitkNearestNeighbor); r.SetTransform(sitk.Transform())
        p2r = sitk.GetArrayFromImage(r.Execute(sitk.ReadImage(str(s2p)))).astype(np.uint8)
        U = p1 > 0; both = U & np.isin(p2r, (1, 2))
        disagree = float((p1[both] != p2r[both]).mean()) if both.sum() else 0.0
        if disagree <= T2_CASCADE_TAU:
            continue  # stage-1 and stage-2 agree -> keep stage-1 (do no harm)
        fired += 1
        merged = np.zeros_like(p1); merged[U] = p1[U]
        ov = U & np.isin(p2r, (1, 2)); merged[ov] = p2r[ov]
        final = np.zeros_like(p1)
        for c in (1, 2):
            final[keep_largest_cc(merged == c)] = c
        out_img = sitk.GetImageFromArray(final); out_img.CopyInformation(p1img)
        sitk.WriteImage(out_img, str(out / f"{cid}-pred.nii.gz"))
    return fired


def task2(in_root: Path, out_root: Path, timing: dict, extra_postproc: bool) -> None:
    # Folder-first, suffix-fallback (matches the organizer reference container's discovery):
    # if t2_tee/ exists it is authoritative; on a flat /input, T2 files are the -US-suffixed ones.
    if (in_root / "t2_tee").is_dir():
        base, files = find_task_dir(in_root, "t2_tee", (".nii.gz",))
    else:
        base, files = find_task_dir(in_root, "t2_tee", ("-US.nii.gz",))
    if not files:
        log("T2: no inputs found, skipping")
        return
    log(f"T2: {len(files)} cases from {base}")
    out = out_root / "t2_tee"
    out.mkdir(parents=True, exist_ok=True)

    def case_of(f: Path) -> str:
        s = f.name[: -len(".nii.gz")]
        return s[: -len("-US")] if s.endswith("-US") else s

    stage, raw = Path("/tmp/t2_stage"), Path("/tmp/t2_raw")
    idmap = stage_symlinks(files, stage, case_of)
    t0 = time.time()
    run_nnunet(stage, raw, T2_DATASET, T2_TRAINER, fold="all", plans=T2_PLANS,
               disable_tta=os.environ.get("MVAA_T2_DISABLE_TTA", "0") == "1",
               sequential_io=True)
    cases = []
    for cid in sorted(idmap):
        img = nib.load(str(raw / f"{cid}.nii.gz"))
        arr = np.asarray(img.dataobj)
        out_arr = np.zeros_like(arr, dtype=np.uint8)
        for c in (1, 2):
            cc = keep_largest_cc(arr == c)
            if extra_postproc:
                cc = ndimage.binary_fill_holes(cc)
            out_arr[cc] = c
        nib.save(nib.Nifti1Image(out_arr, img.affine, img.header), str(out / f"{cid}-pred.nii.gz"))
        cases.append({"case_id": cid, "segmentation": f"{cid}-pred.nii.gz"})
    if T2_CASCADE and _stage2_present():
        log("T2 cascade: refining stage-1 with the stage-2 leaflet-identity split (TTA off)")
        fired = cascade_refine_t2(idmap, out)
        log(f"T2 cascade: conditional merge fired on {fired}/{len(cases)} cases")
    elif T2_CASCADE:
        log("T2 cascade requested but Dataset502 model not in nnUNet_results — using stage-1 only")
    json.dump({"cases": cases}, open(out / "task2_predictions.json", "w"), indent=2)
    dt = time.time() - t0
    timing["task2"] = {"seconds": round(dt, 2), "cases": len(cases),
                       "per_case_incl_coldstart": round(dt / len(cases), 2)}
    log(f"T2 done: {len(cases)} cases in {dt:.1f}s ({dt/len(cases):.2f}s/case incl one-time cold-start)")


def task3(in_root: Path, out_root: Path, timing: dict) -> None:
    base = in_root / "t3_vid" if (in_root / "t3_vid").is_dir() else in_root
    input_frames = discover_t3_images(base)
    n_input_frames = len(input_frames)
    if n_input_frames == 0:
        log("T3: no inputs found, skipping")
        return
    log(f"T3: {n_input_frames} frames from {base}")
    tmp = Path("/tmp/t3_out")
    shutil.rmtree(tmp, ignore_errors=True)
    out = out_root / "t3_vid"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    apply_postprocess = True
    if T3_PRETRAINED_FP32_WEIGHTS:
        if not Path(T3_PRETRAINED_FP32_WEIGHTS).is_file():
            raise SystemExit(f"T3 standalone FP32 weights missing: {T3_PRETRAINED_FP32_WEIGHTS}")
        log("T3: standalone FP32 pre-trained candidate")
        cmd = [sys.executable, str(MVAA / "scripts" / "infer_t3_pretrained_fp32.py"),
               "--input", str(base), "--output", str(tmp),
               "--weights", T3_PRETRAINED_FP32_WEIGHTS]
        apply_postprocess = False
    elif T3_SURGENETDINO_PURE_TRI:
        required = [
            Path(T3_SURGENETDINO_C0),
            Path(T3_SURGENETDINO_C1),
            Path(T3_SURGENETDINO_ALL6_C),
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise SystemExit(f"T3 pure surgical-DINO tri weights missing: {missing}")
        log(
            "T3: pure surgical-DINOv2 three-member ensemble "
            f"mean(C_979A,C_675A,C_ALL6) thr={T3_THRESHOLD}; "
            "native probability fusion; no A_DR; no spatial cleanup"
        )
        cmd = [
            sys.executable, str(MVAA / "scripts" / "infer_t3_surgenetdino_pure_tri.py"),
            "--data-dir", str(base), "--output-dir", str(tmp),
            "--ckpt-c", T3_SURGENETDINO_C0,
            "--expected-ckpt-c-sha256", T3_SURGENETDINO_C0_SHA256,
            "--source-checkpoint-sha256", T3_SURGENETDINO_C0_SOURCE_SHA256,
            "--fold-tag", "979A",
            "--ckpt-c", T3_SURGENETDINO_C1,
            "--expected-ckpt-c-sha256", T3_SURGENETDINO_C1_SHA256,
            "--source-checkpoint-sha256", T3_SURGENETDINO_C1_SOURCE_SHA256,
            "--fold-tag", "675A",
            "--ckpt-c", T3_SURGENETDINO_ALL6_C,
            "--expected-ckpt-c-sha256", T3_SURGENETDINO_ALL6_SHA256,
            "--source-checkpoint-sha256", T3_SURGENETDINO_ALL6_SOURCE_SHA256,
            "--fold-tag", "ALL6",
            "--threshold", T3_THRESHOLD,
        ]
        apply_postprocess = False
    elif T3_SURGENETDINO_TRI:
        required = [
            Path(T3_SURGENETDINO_C0),
            Path(T3_SURGENETDINO_C1),
            Path(T3_SURGENETDINO_ALL6_C),
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise SystemExit(f"T3 surgical-DINO tri weights missing: {missing}")
        log(
            "T3: surgical-DINOv2 three-member replacement "
            f"{T3_WEIGHT_A}*A_DR + {1.0 - float(T3_WEIGHT_A):.2f}*"
            "mean(C_979A,C_675A,C_ALL6) "
            f"thr={T3_THRESHOLD}; native probability fusion; no spatial cleanup"
        )
        cmd = [
            sys.executable, str(MVAA / "scripts" / "infer_t3_surgenetdino_tri.py"),
            "--data-dir", str(base), "--output-dir", str(tmp),
            "--ckpt-a", T3_CKPT_A,
            "--ckpt-c", T3_SURGENETDINO_C0,
            "--expected-ckpt-c-sha256", T3_SURGENETDINO_C0_SHA256,
            "--source-checkpoint-sha256", T3_SURGENETDINO_C0_SOURCE_SHA256,
            "--fold-tag", "979A",
            "--ckpt-c", T3_SURGENETDINO_C1,
            "--expected-ckpt-c-sha256", T3_SURGENETDINO_C1_SHA256,
            "--source-checkpoint-sha256", T3_SURGENETDINO_C1_SOURCE_SHA256,
            "--fold-tag", "675A",
            "--ckpt-c", T3_SURGENETDINO_ALL6_C,
            "--expected-ckpt-c-sha256", T3_SURGENETDINO_ALL6_SHA256,
            "--source-checkpoint-sha256", T3_SURGENETDINO_ALL6_SOURCE_SHA256,
            "--fold-tag", "ALL6",
            "--threshold", T3_THRESHOLD, "--weight-a", T3_WEIGHT_A,
        ]
        apply_postprocess = False
    elif T3_SURGENETDINO_CV2:
        required = [Path(T3_SURGENETDINO_C0), Path(T3_SURGENETDINO_C1)]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise SystemExit(f"T3 surgical-DINO CV2 weights missing: {missing}")
        log(
            "T3: surgical-DINOv2 CV2 replacement "
            f"{T3_WEIGHT_A}*A_DR + {1.0 - float(T3_WEIGHT_A):.2f}*mean(C_979A,C_675A) "
            f"thr={T3_THRESHOLD}; native probability fusion"
        )
        cmd = [
            sys.executable, str(MVAA / "scripts" / "infer_t3_surgenetdino_cv2.py"),
            "--data-dir", str(base), "--output-dir", str(tmp),
            "--ckpt-a", T3_CKPT_A,
            "--ckpt-c", T3_SURGENETDINO_C0,
            "--expected-ckpt-c-sha256", T3_SURGENETDINO_C0_SHA256,
            "--source-checkpoint-sha256", T3_SURGENETDINO_C0_SOURCE_SHA256,
            "--fold-tag", "979A",
            "--ckpt-c", T3_SURGENETDINO_C1,
            "--expected-ckpt-c-sha256", T3_SURGENETDINO_C1_SHA256,
            "--source-checkpoint-sha256", T3_SURGENETDINO_C1_SOURCE_SHA256,
            "--fold-tag", "675A",
            "--threshold", T3_THRESHOLD, "--weight-a", T3_WEIGHT_A,
        ]
    elif T3_PRETRAINED_WEIGHTS and os.path.exists(T3_PRETRAINED_WEIGHTS):
        log("T3: mask-safe spatial refinement with 0.60 imagenet DR + 0.30 surgenetxl + 0.10 pre-trained model")
        cmd = [sys.executable, str(MVAA / "scripts" / "infer_t3_ensemble_pretrained_spatial.py"),
               "--input", str(base), "--output", str(tmp),
               "--ckpt-a", T3_CKPT_A, "--ckpt-b", T3_CKPT_B,
               "--pretrained-weights", T3_PRETRAINED_WEIGHTS]
    elif T3_CKPT_C and os.path.exists(T3_CKPT_C):
        log(
            "T3: 3-way ensemble (imagenet + surgenetxl + segformer) "
            f"thr={T3_THRESHOLD} weights={T3_WEIGHT_A},{T3_WEIGHT_B},{T3_WEIGHT_C}"
        )
        cmd = [sys.executable, str(MVAA / "scripts" / "infer_t3_ensemble3.py"),
               "--data-dir", str(base), "--output-dir", str(tmp),
               "--ckpt-a", T3_CKPT_A, "--ckpt-b", T3_CKPT_B, "--ckpt-c", T3_CKPT_C,
               "--threshold", T3_THRESHOLD,
               "--weight-a", T3_WEIGHT_A, "--weight-b", T3_WEIGHT_B,
               "--weight-c", T3_WEIGHT_C]
    else:
        log(f"T3: 2-way ensemble (imagenet + surgenetxl) thr={T3_THRESHOLD} wA={T3_WEIGHT_A}")
        cmd = [sys.executable, str(MVAA / "scripts" / "infer_t3_ensemble.py"),
               "--data-dir", str(base), "--output-dir", str(tmp),
               "--ckpt-a", T3_CKPT_A, "--ckpt-b", T3_CKPT_B,
               "--threshold", T3_THRESHOLD, "--weight-a", T3_WEIGHT_A]
    subprocess.run(cmd, check=True)
    if apply_postprocess:
        subprocess.run([sys.executable, str(MVAA / "scripts" / "postprocess_t3_largestcc.py"),
                        "--in-dir", str(tmp), "--out-dir", str(out),
                        "--min-area-frac", T3_MIN_AREA_FRAC,
                        "--min-total-fg-frac", T3_MIN_FG_FRAC, "--morph-kernel", "0"], check=True)
    else:
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(tmp, out)
    dt = time.time() - t0
    # count cases from the emitted json if present
    n = n_input_frames
    j = out / "task3_predictions.json"
    if j.exists():
        try:
            n = len(json.load(open(j)).get("cases", [])) or n_input_frames
        except Exception:
            pass
    timing["task3"] = {"seconds": round(dt, 2), "frames": n,
                       "per_frame_incl_coldstart": round(dt / max(n, 1), 2)}
    log(f"T3 done: {n} frames in {dt:.1f}s ({dt/max(n,1):.2f}s/frame incl one-time cold-start)")


def main() -> int:
    # Match the organizer reference container: permissive umask so the eval runner (runs under
    # --cap-drop ALL with a bind-mounted /output) can always collect our predictions.
    os.umask(0o000)
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/input"))
    ap.add_argument("--output", type=Path, default=Path("/output"))
    ap.add_argument("--tasks", type=str, default="1,2,3")
    ap.add_argument("--extra-postproc", action="store_true",
                    help="per-class hole-fill on T1/T2 (off by default)")
    args = ap.parse_args()

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    log(f"torch {torch.__version__} | cuda available={torch.cuda.is_available()} "
        f"| device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    log(f"input={args.input} output={args.output} tasks={sorted(tasks)} extra_postproc={args.extra_postproc}")

    timing: dict = {}
    wall0 = time.time()
    if "1" in tasks:
        task1(args.input, args.output, timing, args.extra_postproc)
    if "2" in tasks:
        task2(args.input, args.output, timing, args.extra_postproc)
    if "3" in tasks:
        task3(args.input, args.output, timing)

    # If the hidden test ships /input/test_cases.json, its case_ids are authoritative
    # for scoring — re-key our output JSON to them. No-op when the manifest is absent
    # or already agrees with our filename-derived stems (the file-discovery path).
    manifest = load_manifest(args.input)
    if any(manifest[t] for t in TASK_FOLDERS):
        log("test_cases.json detected — re-keying output case_ids to the manifest")
        for t in TASK_FOLDERS:
            if str(TASK_NUM[t]) in tasks:
                rekey_to_manifest(args.output, t, manifest[t])

    timing["wall_seconds_total"] = round(time.time() - wall0, 2)
    json.dump(timing, open(args.output / "timing.json", "w"), indent=2)
    log(f"ALL DONE in {timing['wall_seconds_total']}s — timing.json written")
    log(json.dumps(timing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
