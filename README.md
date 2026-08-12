# MVAA 2026 — Mitral-Valve Segmentation Across CT, TEE, and Surgical Video

Code release accompanying our method paper for the **MVAA 2026** challenge
(Mitral Valve Anatomy Analysis Using Multimodal Imaging Data, MICCAI 2026 / MWM 2026).

This repository covers **only the methods used in the final hidden-test submission**.

> **Status: release in preparation.** The inference code, training recipes, and the
> offline container definition for the deployed system are being prepared and will be
> published here shortly.

## Final hidden-test result

Codabench competition 17301, "Final Test Docker Evaluation" (submission 873932):

| Task | DSC ↑ | HD ↓ | ASD ↓ |
|---|---|---|---|
| Task 1 — Cardiac CT | **0.8443** | **4.944 mm** | 0.4163 mm |
| Task 2 — 3D TEE *(not ranked)* | 0.9464 | 3.152 mm | 0.1367 mm |
| Task 3 — Surgical video | 0.8151 | 243.3 px | 105.2 px |

Task 1: **3rd of 36** teams, **1st on Hausdorff distance**. Overall: **6th of 36**.
Task 2 is excluded from the competition ranking by the organizers, because its
underlying data is publicly available.

## What will be released

- **Task 1 (cardiac CT)** — eight-member probability fusion of two nnU-Net v2 families,
  plus the anatomy-gated offline self-training pipeline over the 1,040 unlabeled volumes.
- **Task 2 (3D TEE)** — supervised nnU-Net v2 `3d_fullres` with the anterior–posterior
  mirroring axis disabled.
- **Task 3 (surgical video)** — ensemble of three DINOv2 ViT-B/14 + DPT networks with
  surgical self-supervised initialization, trained with consistency regularization over
  the 1,379 unlabeled frames.
- The **offline inference container** submitted for hidden-test evaluation.

## Contact

Harshit Agrawal — <harshit@aiatella.com> · AIATELLA Oy, Helsinki, Finland
