"""
nnUNetTrainer_SafeMirror
========================
Custom nnU-Net v2 trainer for MVAA Task 2 (3D TEE mitral-leaflet seg) that
keeps mirror augmentation on anatomically safe axes (0=L-R, 2=S-I) while
removing it only on axis 1 (A-P), which is chirally unsafe for this task.

Motivation: Task 2 has chirally distinct labels — anterior (1) vs posterior (2)
leaflet.  The anterior and posterior leaflets are separated along the
anterior-posterior axis (axis 1 in the RAS-oriented dataset), so flipping along
that axis maps anterior label voxels into the anatomical territory of the
posterior leaflet, directly poisoning the training signal and widening the
leaflet-swap confusion tail.

Axes 0 (L-R) and 2 (S-I) are safe: the mitral valve annulus has approximate
left-right symmetry, and superior-inferior flips do not exchange the two leaflet
identities.  Keeping TTA on those two axes recovers most of the regularisation
benefit of mirroring while eliminating the label-swap artefact.

The previous nnUNetTrainer_NoMirror experiment disabled ALL mirrors, which was a
valid chirality fix but cost too much TTA signal on the safe axes and failed the
fold-0 gate.  This trainer is the minimal targeted fix.

Usage:
    nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_SafeMirror -p nnUNetPlans_LargePatch

Env knobs (all optional):
    SAFEMIRROR_NUM_EPOCHS  (default 250, same as plain baseline)
    SAFEMIRROR_GIT_SHA     (recorded in repro_config.json for provenance)
"""
import json
import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_SafeMirror(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get('SAFEMIRROR_NUM_EPOCHS', 250))

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # Call super() to get normal geometry decisions, then selectively drop axis 1
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, _ = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        # Keep axes 0 (L-R) and 2 (S-I); drop axis 1 (A-P) — swaps anterior/posterior labels
        self.mirror_axes = (0, 2)
        # TTA uses the same safe-axis set so inference is consistent with training
        self.inference_allowed_mirroring_axes = (0, 2)
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, (0, 2)

    def initialize(self):
        super().initialize()
        self.print_to_log_file(
            "[SafeMirror] mirror_axes=(0,2) inference_allowed_mirroring_axes=(0,2) — "
            "axis 1 (A→P) removed; anterior/posterior leaflet labels are chirally distinct "
            "along that axis.  Axes 0 (L-R) and 2 (S-I) are anatomically safe flips for "
            "the mitral valve."
        )
        cfg = {
            "trainer": "nnUNetTrainer_SafeMirror",
            "num_epochs": self.num_epochs,
            "fold": self.fold,
            "mirror_axes": [0, 2],                       # axis 1 excluded — chiral
            "inference_allowed_mirroring_axes": [0, 2],  # same restriction at TTA time
            "git_sha": os.environ.get("SAFEMIRROR_GIT_SHA", ""),
        }
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            with open(os.path.join(self.output_folder, "repro_config.json"), "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            self.print_to_log_file(f"[repro] WARN could not write repro_config.json: {e}")
