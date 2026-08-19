# Model weights

**Weights are not distributed in this release.** This directory documents what the deployed
container expects, and records the exact artifacts that produced the submitted results so they
can be identified unambiguously.

Train them with [`../TRAINING.md`](../TRAINING.md); the Task 3 exporters write directly into the
`t3/` layout below.

## Expected layout

Mirrors `nnUNet_results`. The container mounts this tree at `/opt/weights` and sets
`nnUNet_results=/opt/weights/nnunet`. The `*.json` files here are the plan and dataset
definitions, which the nnU-Net predictor loads alongside each checkpoint; they are tracked
because they are part of the model definition, not weights.

```
weights/
├── nnunet/
│   ├── Dataset505_MVAA_TEE_final/nnUNetTrainer_SafeMirror__nnUNetPlans_LargePatch__3d_fullres/
│   │     dataset.json  plans.json                          [tracked]
│   │     fold_all/checkpoint_final.pth                     T2
│   ├── Dataset511_T1CT/nnUNetTrainer_250epochs__nnUNetPlans_LargePatch__3d_fullres/
│   │     dataset.json  plans.json                          [tracked]
│   │     fold_{0,1,4}/checkpoint_final.pth                 T1 supervised
│   └── Dataset512_T1CT_selftrain/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/
│         dataset.json  plans.json                          [tracked]
│         fold_{0,1,2,3,4}/checkpoint_final.pth             T1 self-trained
└── t3/
      surgenetdino_v2b_979A_ema_fp16.pt                     LOVO fold A
      surgenetdino_v2b_675A_ema_fp16.pt                     LOVO fold B
      surgenetdino_v2b_all6_ema_fp16.pt                     all six videos
```

> **Always pass the three build args** from the README. The Dockerfile's bare defaults select an
> older two-member Task 3 path that expects `t3/imagenet_dr_epoch070.pt` and
> `t3/surgenetxl_best.pt` — checkpoints from superseded variants that are deliberately not part of
> this layout. A no-args build will fail at Task 3 looking for them.

Note the T1 asymmetry: **all five** Dataset512 folds ship, but only LargePatch folds **0, 1 and 4**.
Folds 2 and 3 were pruned because they scored below their matched plain-plan baseline on the
leakage-free held-out set.

T1 fusion is family-level and equal-weight — the five Dataset512 folds average into one
probability map, the three LargePatch folds into another, and those two are averaged. So a
Dataset512 fold carries 1/10 of the final vote and a LargePatch fold 1/6.

## Provenance of the submitted artifacts

[`SHA256SUMS`](SHA256SUMS) records the twelve checkpoints baked into the scored image
(`sha256:a4b91b975dcd…485f4`). It identifies them; it is not a download manifest.

The three Task 3 hashes are additionally asserted **at inference time**: `docker/run_inference.py`
passes `--expected-ckpt-c-sha256` to the Task 3 entry point, which refuses to run on a mismatch.
If you retrain Task 3, update both `SHA256SUMS` and the constants near the top of
`docker/run_inference.py`, or the container will stop rather than silently run your weights.

## Licensing, if you later distribute weights

Any model trained on this data is **non-commercial**: the MVAA data is CC BY-NC, and the Task 3
backbone initializer (SurgeNetDINOv2) is CC BY-NC-SA 4.0, whose ShareAlike term propagates —
CC BY-NC-SA 4.0 is the appropriate licence for Task 1 and Task 3 weights.

Task 2 needs separate thought. Its training set includes 70 cases from the MVSeg2023 release,
which is **CC BY-NC-ND 4.0**, and the NoDerivatives term restricts distributing adaptations rather
than merely constraining their licence. See [`../TRAINING.md`](../TRAINING.md) for the full source
table and disclosure.
