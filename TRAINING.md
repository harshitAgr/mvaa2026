# Training on the MVAA 2026 data

How to obtain the challenge data, lay it out, and retrain every model that the deployed
container ships. Inference-only users do not need any of this — see the README.

## 1. Get the data

The MVAA 2026 data is **CC BY-NC** and is not redistributable, so it is not in this repository.
Register for the challenge and download it from the organizers:

- Challenge / baseline: <https://github.com/db0725/MVAA>
- Codabench competition page (registration, data links, terms)

You will receive, per task, labelled training data plus — for Tasks 1 and 3 — an unlabelled
pool. **Semi-supervised use of those pools is required by the challenge rules**, and both of our
ranked models depend on them.

### Data sources and licences

| Source | Used by | Cases | Licence |
|---|---|---|---|
| **MVAA 2026** (organizer) | T1, T2, T3 | T1 27 labelled + 1,040 unlabelled · T2 105 · T3 180 frames + 1,379 unlabelled | CC BY-NC |
| **MVSeg2023** (external) | T2 only | 70 (`val_001-030`, `test_001-040`) | **CC BY-NC-ND 4.0**, gated |
| **SurgeNetDINOv2** (external weights) | T3 backbone init | — | CC BY-NC-SA 4.0 |

- **MVSeg2023** — <https://huggingface.co/datasets/pcarnahan/MVSeg2023> (gated; Synapse mirror
  `syn51186045`). Cite Carnahan et al., *DeepMitral*, MICCAI 2021,
  DOI [10.1007/978-3-030-87240-3_44](https://doi.org/10.1007/978-3-030-87240-3_44).
- **SurgeNetDINOv2** — <https://github.com/timjaspers0801/SurgeNetDINO>, file
  `SurgeNetDINOv2_ViTb14_size336_SurgeNetXL.pth`. Not challenge data; download separately.

**Disclosure.** An MD5 audit showed MVAA Task 2 is the MVSeg2023 train/val split verbatim, so the
70 external cases overlap it. We therefore assume the Task 2 model has seen its own test data and
report Task 2 for completeness only — the organizers exclude it from the ranking for the same
reason. `MVAA_T2_EXCLUDE_MVAA_VAL=1` builds a 155-case variant that avoids the MVAA validation
split. Tasks 1 and 3, the ranked tasks, use organizer data only.

## 2. Lay it out

Every script derives its paths from the repository root and accepts environment overrides, so you
can either match this layout or point the variables elsewhere.

```
data/
├── reference_data/
│   ├── t1_ct/
│   │   ├── train/labeled/images/*.nii.gz        labelled CT volumes
│   │   ├── train/labeled/labels/*-seg.nii.gz    binary valve masks
│   │   ├── train/unlabeled/*.nii.gz             unlabelled CT pool
│   │   └── val/images/*.nii.gz
│   └── t2_tee/
│       └── train/*-US.nii.gz, *-label.nii.gz    105 organizer Task 2 cases
│   └── t3_vid/
│       └── train/<video>/...                    labelled surgical videos
├── external/
│   └── mvseg2023/{val,test}/                    MVSeg2023 release (Task 2 only)
├── images/                                      unlabelled Task 3 frames
├── pretrained/
│   └── SurgeNetDINOv2_ViTb14_size336_SurgeNetXL.pth
└── nnunet/                                      created by the scripts
    ├── raw/  preprocessed/  results/
```

Override with `MVAA_ROOT` (repo root), `MVAA_NNUNET_BASE` (the `data/nnunet` tree), or the
per-script variables documented at the top of each file.

## 3. Environments

Two environments, deliberately kept apart:

| Env | Used by | Key pins |
|---|---|---|
| nnU-Net | Tasks 1 and 2 | `nnunetv2==2.7.0`, `dynamic-network-architectures==0.4.3`, `SimpleITK==2.5.5` |
| PyTorch | Task 3 | `torch==2.11.0+cu128`, `timm==1.0.27`, `segmentation-models-pytorch==0.5.0` |

The inference container pins a different, older stack (`torch==2.5.1+cu124`, `timm==1.0.26`)
because the evaluation server is a V100 (sm_70), which the cu128 wheels no longer support.

Pinned requirement files for both, plus the exact version table and the two pitfalls
(the `timm` / nnU-Net conflict, and offline container builds), are in
**[env/README.md](env/README.md)**:

```bash
python -m venv .venv-nnunet && .venv-nnunet/bin/pip install -r env/requirements-nnunet.txt
python -m venv .venv        && .venv/bin/pip install -r env/requirements-t3.txt
```

Point the nnU-Net scripts at the first with `NNUNET_VENV=/path/to/.venv-nnunet`.

## 4. Task 1 — cardiac CT

Two families are trained, then fused at inference time.

```bash
export nnUNet_raw=$PWD/data/nnunet/raw
export nnUNet_preprocessed=$PWD/data/nnunet/preprocessed
export nnUNet_results=$PWD/data/nnunet/results
NNUNET_VENV=/path/to/nnunet-venv bash scripts/nnunet_t1/install_trainers.sh
```

**Family B — supervised LargePatch (`Dataset511_T1CT`)**

```bash
python scripts/nnunet_t1/convert_dataset.py            # labelled cases -> nnU-Net raw
nnUNetv2_plan_and_preprocess -d 511 --verify_dataset_integrity

bash scripts/nnunet_t1/run_largepatch_experiment.sh    # writes the LargePatch plan, trains fold 0
bash scripts/nnunet_t1/run_largepatch_folds14.sh       # folds 1-4 (two at a time on one GPU)
```

The LargePatch plan is derived from the default plan by widening the 3D patch to
`[112,160,192]`; the script writes `nnUNetPlans_LargePatch.json` itself.

**Family A — anatomy-gated self-training (`Dataset512_T1CT_selftrain`)**

This is the semi-supervised arm. Predict the unlabelled pool with the supervised model, keep only
anatomically plausible predictions, and retrain on labelled + accepted pseudo-labels.

```bash
# 1. predict the unlabelled CT pool with the 5-fold Dataset511 ensemble
nnUNetv2_predict -i data/reference_data/t1_ct/train/unlabeled \
                 -o data/nnunet/pseudo_raw \
                 -d 511 -c 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainer_250epochs

# 2. gate them
python scripts/nnunet_t1/gate_pseudo.py \
    --pseudo-dir data/nnunet/pseudo_raw \
    --out-json  data/nnunet/pseudo_gated/gate_report.json

# 3. build the self-train dataset, then preprocess and train
python scripts/nnunet_t1/build_selftrain_dataset.py
nnUNetv2_plan_and_preprocess -d 512 --verify_dataset_integrity
bash scripts/nnunet_t1/run_selftrain_folds14.sh        # folds 1-4; train fold 0 the same way
```

The gate keeps a case only if it is non-empty, at least 90 % dominated by one connected
component, within the measured labelled volume and foreground-fraction range, and not from the
coarse-spacing tail. Thresholds are flags on `gate_pseudo.py`. Pseudo cases are written with a
`PLBL_` prefix and `build_selftrain_dataset.py` mirrors Dataset511's `splits_final.json`, so the
same real cases stay held out per fold and pseudo cases never enter a validation fold.

**Deployment note.** The container ships all five Dataset512 folds but only LargePatch folds
0, 1 and 4. Folds 2 and 3 were dropped because they scored below their matched plain-plan
baseline on the leakage-free held-out set.

## 5. Task 2 — 3D TEE

```bash
NNUNET_VENV=/path/to/nnunet-venv bash scripts/nnunet_t2/install_trainers.sh

# 105 organizer cases + 70 MVSeg2023 cases = 175. Point the roots at your copies:
MVAA_T2_TRAIN=data/reference_data/t2_tee/train \
MVSEG2023_VAL=data/external/mvseg2023/val \
MVSEG2023_TEST=data/external/mvseg2023/test \
python scripts/nnunet_t2/build_dataset505_final.py

bash scripts/nnunet_t2/run_final_175_foldall.sh
```

`build_dataset505_final.py` declares its composition in a `BLOCKS` table, asserts the per-block
counts, records the origin and licence of each block into the generated `dataset.json`, and
prints the breakdown as it runs. Add `MVAA_T2_EXCLUDE_MVAA_VAL=1` for the 155-case variant.

`nnUNetTrainer_SafeMirror` restricts mirroring to axes `(0,2)`. The anterior–posterior axis is
excluded because mirroring it swaps the anterior and posterior leaflet labels, which is
anatomically invalid for this task.

Task 2 is score-neutralized in the final test phase, so this is a plain deployment retrain with
no held-out split.

## 6. Task 3 — surgical video

Place the backbone at `data/pretrained/SurgeNetDINOv2_ViTb14_size336_SurgeNetXL.pth`, then:

```bash
# the two leave-one-video-out members
SURGENET_E1_GPUS='0 1' bash scripts/run_t3_surgenetdino_v2b_e1.sh

# the all-videos member
python scripts/train_t3_surgenetdino_alldata.py \
    --labeled-root data/reference_data/t3_vid/train \
    --unlabeled-root data/images \
    --output-dir runs/t3_all6
```

Each member is a DINOv2-B/DPT segmenter initialized from SurgeNetDINOv2, trained for a fixed 70
epochs on a UniMatch-V2-style recipe with an EMA teacher over the unlabelled frames.

The backbone is **not frozen throughout**. Epochs 1-5 are decoder-only warm-up with the backbone
in eval mode; at epoch 6 the backbone is unfrozen and the optimizer is rebuilt with two parameter
groups — backbone at LR 5e-6, head at 2e-4 — and the EMA teacher is initialized at that transition
(`train_t3_surgenetdino_v2b_e1.py`, `WARMUP_EPOCHS = 5`).

The candidate is always the epoch-70 EMA state, fixed in advance and never selected on a score.

Convert each trained checkpoint into the fp16 state the container loads:

```bash
python scripts/export_t3_surgenetdino_inference.py \
    --checkpoint runs/<run>/checkpoints/epoch_070_ema.pt \
    --expected-sha256 <sha256 of that file> \
    --fold-tag 979A \
    --output weights/t3/surgenetdino_v2b_979A_ema_fp16.pt

python scripts/export_t3_surgenetdino_alldata_inference.py ...   # same, for the ALL6 member
```

The exporters verify the source checkpoint hash and that it really is a fixed epoch-70 EMA state
with no held-out metrics attached, then write a half-precision inference-only payload. The
container re-checks these hashes at startup, so regenerating a member means updating
`weights/SHA256SUMS` and the constants near the top of `docker/run_inference.py`.

One Task 3 frame — video `REC_20250322_101917_746A`, frame `130` — carries a noisy
chamber-included mask and is dropped before any split (`EXCLUDED_FRAMES` in
`scripts/train_t3_bcp.py`). Keep that exclusion symmetric across every arm you compare.

## 7. Running your own experiments

The evaluation protocol matters more than the metric here: both ranked tasks have very few
independent units (27 CT cases, 6 videos), so it is easy to produce a number that will not
transfer to the hidden test set.

**Split by the independent unit, never by the sample.** For Task 1 that is the case; for Task 3
it is the *video*. Frame count is not sample count — six videos of a few hundred frames each give
you six independent observations, not two thousand.

- **Task 1 / Task 2:** use nnU-Net's own case-level cross-validation as the ruler. Train folds
  0–4 and read the out-of-fold results nnU-Net writes under
  `<nnUNet_results>/<Dataset>/<trainer>__<plans>__3d_fullres/`.
  Pseudo-labelled cases belong only in training folds — never in a validation fold, never in
  checkpoint selection or threshold tuning. A self-train model cannot be honestly scored against
  the labelled cases that produced its own teacher.
- **Task 3:** use six-video leave-one-video-out. `train_t3_surgenetdino_v2b_e1.py --holdout-video`
  pins the held-out video; the trainer never constructs a validation set or opens a held-out
  frame, so the checkpoint cannot be chosen on the target.

**Use spacing-aware distance metrics.** Task 1 HD/ASD are in millimetres and require the voxel
spacing; Task 3 distances are pixels at native frame resolution. Scoring resampled masks on the
model grid is not comparable to the official numbers.

**Include empty frames in Task 3 checks.** An empty prediction on an empty ground-truth frame is
perfect, whereas a small false-positive mask is catastrophic for HD and ASD. Evaluating only
foreground frames hides the error mode that dominates the score — that is the single biggest
reason a local Task 3 result can look good and still regress on the hidden test set.

**Report per-unit deltas, not just the mean.** With n=6 videos or n=27 cases, one case can move
an aggregate. Prefer paired comparisons on matched folds, seeds, epoch budgets, TTA and
post-processing, and check sign consistency across units before believing a gain.

## 8. Runtime

Rough figures on the training machine, a single NVIDIA RTX PRO 6000:

| Stage | Cost |
|---|---|
| T1 nnU-Net fold (250 epochs) | ~2.5–3 h; ~11.8 GB peak, so several fit concurrently |
| T1 unlabelled-pool prediction | the long pole of the self-training loop; scales with pool size |
| T2 fold_all | comparable to a T1 fold, larger patch |
| T3 member (70 epochs: 5 decoder-only + 65 full-backbone) | the two LOVO members run in parallel on two GPUs |

Inference for the full deployed ensemble is far cheaper than training: the scored container
completed the entire hidden test set in ~2,150 s.
