"""Plain all-data Task-1 self-training candidate.

This is the deployed Dataset512 250-epoch PlainConvUNet recipe with one change:
all 27 real labels are used directly instead of the 21 real labels in fold 0.
Real cases are duplicated so the total number of real draws remains 378, exactly
matching fold 0 (21 * 18), while all 921 gated pseudo-labels remain single-copy.

There is deliberately no pretrained-weight logic. Run without ``-pretrained_weights``.
The validation set contains the 27 training cases for monitoring only and is not a
generalization estimate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import torch

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs,
)


REAL_PREFIX = "T1CT_"
PSEUDO_PREFIX = "PLBL_"
DEFAULT_CONFIG_OUT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "task1_alldata_v1"
    / "config.json"
)


def build_all_data_split(
    all_identifiers: List[str], target_ratio: float = 2.5
) -> Tuple[List[str], List[str], dict]:
    """Return the frozen all-data oversampled train and monitoring splits."""
    real_ids = sorted(k for k in all_identifiers if k.startswith(REAL_PREFIX))
    pseudo_ids = sorted(k for k in all_identifiers if k.startswith(PSEUDO_PREFIX))
    unknown = sorted(set(all_identifiers) - set(real_ids) - set(pseudo_ids))
    if unknown:
        raise ValueError(f"Unexpected Dataset512 identifiers: {unknown[:10]}")
    if not real_ids or not pseudo_ids:
        raise ValueError(
            f"Expected both real and pseudo cases, found real={len(real_ids)}, "
            f"pseudo={len(pseudo_ids)}"
        )

    factor = max(1, round((len(pseudo_ids) / target_ratio) / len(real_ids)))
    train_keys = real_ids * factor + pseudo_ids
    monitor_keys = list(real_ids)
    report = {
        "run": "task1_alldata_v1",
        "trainer": "nnUNetTrainer_T1AllData250",
        "n_real": len(real_ids),
        "n_pseudo": len(pseudo_ids),
        "target_pseudo_real_draw_ratio": target_ratio,
        "real_oversample_factor": factor,
        "n_real_draws": len(real_ids) * factor,
        "n_pseudo_draws": len(pseudo_ids),
        "n_total_train_draws": len(train_keys),
        "achieved_pseudo_real_draw_ratio": len(pseudo_ids) / (len(real_ids) * factor),
        "real_ids": real_ids,
        "monitor_ids": monitor_keys,
        "monitor_is_held_out": False,
        "pretrained_weights": None,
    }
    return train_keys, monitor_keys, report


class nnUNetTrainer_T1AllData250(nnUNetTrainer_250epochs):
    """Stock 250-epoch trainer with the frozen all-27 Dataset512 split."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.target_ratio = float(os.environ.get("T1_ALLDATA_TARGET_RATIO", "2.5"))
        self._all_data_report = None
        self._iterations_override = None
        if os.environ.get("MVAA_SMOKE") == "1":
            self.num_epochs = 1
            self._iterations_override = int(os.environ.get("T1_ALLDATA_ITERS_PER_EPOCH", "5"))

    def initialize(self):
        super().initialize()
        if self._iterations_override is not None:
            self.num_iterations_per_epoch = self._iterations_override
            self.num_val_iterations_per_epoch = 1
        self.print_to_log_file(
            f"[T1AllData] epochs={self.num_epochs}, "
            f"iters_per_epoch={self.num_iterations_per_epoch}, fold={self.fold}, "
            f"target_ratio={self.target_ratio}, pretrained_weights=None"
        )

    def do_split(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
        train_keys, monitor_keys, report = build_all_data_split(
            identifiers, self.target_ratio
        )
        report["fold_argument"] = self.fold

        expected = {
            "n_real": 27,
            "n_pseudo": 921,
            "real_oversample_factor": 14,
            "n_real_draws": 378,
            "n_total_train_draws": 1299,
        }
        mismatches = {
            key: (report[key], value)
            for key, value in expected.items()
            if report[key] != value
        }
        if mismatches:
            raise RuntimeError(f"Frozen Dataset512 split mismatch: {mismatches}")

        self._all_data_report = report
        self.print_to_log_file(
            "[T1AllData] split: 27 unique real x14 = 378 real draws; "
            "921 pseudo draws; total=1299; monitor=27 train-seen real cases"
        )
        self._write_config()
        return train_keys, monitor_keys

    def _write_config(self):
        if self.local_rank != 0 or self._all_data_report is None:
            return
        out = Path(os.environ.get("T1_ALLDATA_CONFIG_OUT", str(DEFAULT_CONFIG_OUT)))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self._all_data_report, indent=2) + "\n")
        self.print_to_log_file(f"[T1AllData] wrote frozen config: {out}")
