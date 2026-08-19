#!/usr/bin/env python3
"""Shared sealed-E1 contracts for the SurgeNetDINO DINOv2-B candidate."""
from __future__ import annotations

import json
import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from t3_surgenetdino_v2b import (
    ARTIFACT_SHA256,
    DEFAULT_WEIGHTS,
    MODEL_IMAGE_SIZE,
    SurgeNetDinoV2BDPTSegmenter,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FOLD_979 = "REC_20250205_102353_979A"
FOLD_675 = "REC_20250418_104439_675A"
FOLDS = (FOLD_979, FOLD_675)
FOLD_TAGS = {FOLD_979: "979A", FOLD_675: "675A"}
CHECKPOINT_NAME = "epoch_070_ema.pt"
RUN_NAME = "task3_surgenetdino_v2b_unimatch_v1"
E0_SCHEMA = "mvaa-t3-surgenetdino-v2b-e0-plan-v1"
FROZEN_CONFIG: dict[str, Any] = {
    "run_name": RUN_NAME,
    "image_size": list(MODEL_IMAGE_SIZE),
    "target_label": 10,
    "arch": "surgenetdinov2_vitb14_dpt_small",
    "epochs": 70,
    "warmup_epochs": 5,
    "steps_per_epoch": 25,
    "micro_batch": 2,
    "gradient_accumulation": 3,
    "effective_batch_each_stream": 6,
    "seed": 42,
    "backbone_lr": 5e-6,
    "decoder_lr": 2e-4,
    "weight_decay": 0.01,
    "poly_power": 0.9,
    "confidence_threshold": 0.95,
    "grad_clip_norm": 1.0,
    "save_every": 10,
    "amp": True,
    "heldout_access_during_training": "forbidden",
    "checkpoint_selection": "fixed_epoch_70_ema_only",
}
def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def frozen_config(holdout: str) -> dict[str, Any]:
    if holdout not in FOLDS:
        raise ValueError(f"Unregistered E1 holdout: {holdout}")
    return {
        **FROZEN_CONFIG,
        "holdout_video": holdout,
        "weights_path": str(DEFAULT_WEIGHTS.resolve()),
        "weights_sha256": ARTIFACT_SHA256,
        "pretrained_provenance": {
            "official_code_revision": "19e8325c87b826f5ae637787563891476bbe2b9f",
            "model_host_revision": "9b2c75b7d469850b750a715b90cb91d6319f7e30",
            "license": "CC-BY-NC-SA",
        },
    }


__all__ = [
    "CHECKPOINT_NAME",
    "DEFAULT_WEIGHTS",
    "FOLDS",
    "FOLD_675",
    "FOLD_979",
    "FOLD_TAGS",
    "FROZEN_CONFIG",
    "RUN_NAME",
    "frozen_config",
    "read_json",
    "sha256_file",
]
