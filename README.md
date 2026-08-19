# MVAA 2026 — challenge entry

Training and inference source code for our final submission to the **MVAA 2026** challenge
(Mitral Valve Anatomy Analysis Using Multimodal Imaging Data, MICCAI 2026 MWM workshop).
Hidden-test result: **5th of 35 overall, 3rd of 35 on Task 1.**

Model weights and challenge data are not included. See [TRAINING.md](TRAINING.md) to obtain the
data and retrain, and [weights/README.md](weights/README.md) for the checkpoint layout the
container expects.

## Method

**T1 — cardiac CT.** Equal family-level fusion of two nnU-Net v2 families at threshold 0.50, then
largest connected component. `Dataset512_T1CT_selftrain` folds 0–4 (27 labeled volumes plus
anatomy-gated pseudo-labels from the 1,040 unlabeled CTs) fused with `Dataset511_T1CT` LargePatch
folds 0, 1, 4 (supervised, patch `[112,160,192]`; folds 2 and 3 pruned as below-baseline).

**T2 — 3D TEE.** One supervised nnU-Net: `Dataset505_MVAA_TEE_final`, `fold_all`,
`nnUNetTrainer_SafeMirror` (mirrors axes `(0,2)` only — the anterior–posterior flip swaps the
leaflet labels) with `nnUNetPlans_LargePatch`, 175 cases.

**T3 — surgical video.** Equal probability mean of three DINOv2-B/DPT members — two
leave-one-video-out folds (`979A`, `675A`) and one all-videos member (`ALL6`) — with D4 TTA,
threshold `> 0.45`, no cleanup. Each initializes from SurgeNetDINOv2 and runs a fixed
UniMatch-V2-style recipe: 5 decoder-only warm-up epochs, then full-backbone training with an EMA
teacher over the 1,379 unlabeled frames, to exactly epoch 70. The checkpoint is fixed in advance
at epoch 70, EMA only, never selected on a score.

Semi-supervised use of the unlabelled pools is rules-mandatory, and both ranked models depend on
it: T1 mines gated pseudo-labels from the unlabelled CT pool, T3 applies consistency
regularization over the unlabelled frames.

## Layout

```
scripts/      training code for every model in the submission
baseline/     organizer baseline modules the T3 trainers import
docker/       the inference container (build context of the scored image)
env/          pinned dependencies for both training environments
weights/      expected checkpoint layout + SHA-256 provenance record
TRAINING.md   data sources, environments, training, evaluation protocol
NOTICE        licence scope: what is ours vs vendored third-party
```

## Training

Data, layout, environments, the semi-supervised loops and the evaluation protocol are in
**[TRAINING.md](TRAINING.md)**. In short:

```bash
# Task 1 — supervised LargePatch family, then the anatomy-gated self-train family
python scripts/nnunet_t1/convert_dataset.py
bash   scripts/nnunet_t1/run_largepatch_experiment.sh
bash   scripts/nnunet_t1/run_largepatch_folds14.sh
python scripts/nnunet_t1/gate_pseudo.py --pseudo-dir data/nnunet/pseudo_raw \
                                        --out-json  data/nnunet/pseudo_gated/gate_report.json
python scripts/nnunet_t1/build_selftrain_dataset.py
bash   scripts/nnunet_t1/run_selftrain_folds14.sh

# Task 2
python scripts/nnunet_t2/build_dataset505_final.py
bash   scripts/nnunet_t2/run_final_175_foldall.sh

# Task 3 — two leave-one-video-out members plus one all-videos member
SURGENET_E1_GPUS='0 1' bash scripts/run_t3_surgenetdino_v2b_e1.sh
python scripts/train_t3_surgenetdino_alldata.py \
    --labeled-root data/reference_data/t3_vid/train \
    --unlabeled-root data/images --output-dir runs/t3_all6
```

## Inference

Place the trained checkpoints under `docker/weights/`, then build with all three args — the bare
defaults select a superseded Task 3 path and will fail looking for checkpoints this layout does not
contain:

```bash
docker build \
  --build-arg MVAA_T1_FUSION_LARGEPATCH=1 \
  --build-arg MVAA_T1_LP_FOLD="0 1 4" \
  --build-arg MVAA_T3_SURGENETDINO_PURE_TRI=1 \
  -t mvaa-submit:final docker

docker run --rm --gpus all --shm-size=8g \
  -v /path/to/input:/input:ro -v /path/to/output:/output \
  mvaa-submit:final --input /input --output /output --tasks 1,2,3
```

Those three args are the only deviation from the Dockerfile defaults, and are the configuration
that was scored. Torch is pinned to **cu124** because the evaluation server is a V100 (sm_70),
which the cu128 wheels no longer support. `docker/run_inference.py` documents the input discovery
and output contract.

## License and attribution

Code here is **Apache-2.0** (`LICENSE`), *except* the vendored third-party files below, which keep
their own terms — see [`NOTICE`](NOTICE) for the exact scope. Weights trained by this code are
non-commercial (**CC BY-NC-SA 4.0**): the challenge data is CC BY-NC and the T3 encoder initializer
is CC BY-NC-SA 4.0.

- Organizer baseline (`baseline/task3/`): <https://github.com/db0725/MVAA>, commit `25274d6`.
  `dataset.py`, `train.py`, `utils.py` are unmodified upstream; `model_factory.py` is modified;
  `surgenet_adapter.py` is ours. Upstream publishes no LICENSE file.
- SurgeNet (`third_party/surgenet/metaformer.py`): <https://github.com/timjaspers0801/surgenet>,
  commit `54831cf`, MIT © 2025 Cris Claessens; the file keeps its own Apache-2.0 header, being
  adapted from `sail-sg/metaformer`.
- SurgeNetDINOv2 backbone: <https://github.com/timjaspers0801/SurgeNetDINO>, weights CC BY-NC-SA 4.0.
- **Data.** MVAA 2026 challenge data (CC BY-NC), from the organizers. Task 2 additionally uses 70
  cases from the public **MVSeg2023** release (CC BY-NC-ND 4.0, gated) — Carnahan et al., MICCAI
  2021, DOI 10.1007/978-3-030-87240-3_44. No data is redistributed here; see the Task 2 source
  table and composition disclosure in [TRAINING.md](TRAINING.md).
- Built on nnU-Net v2 (Apache-2.0), DINOv2 (Apache-2.0), segmentation_models.pytorch (MIT),
  MONAI (Apache-2.0), timm (Apache-2.0).

## Citation

```bibtex
@inproceedings{agrawal2026mitralvalve,
  title     = {Mitral-Valve Segmentation Across {CT}, {TEE}, and Surgical Video},
  author    = {Harshit Agrawal},
  booktitle = {The 1st MICCAI Workshop on Medical World Models},
  year      = {2026},
  url       = {https://openreview.net/forum?id=XUOT3HIXlq}
}
```
