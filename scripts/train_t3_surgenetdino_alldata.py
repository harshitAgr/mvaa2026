#!/usr/bin/env python3
"""Train the frozen SurgeNetDINO E1 recipe on all six labeled videos.

This additive wrapper deliberately reuses the sealed E1 trainer rather than copying its training
loop. It changes only the labeled split and run metadata: all eligible labeled samples train, no
validation dataset is constructed, and the sole candidate is the fixed epoch-70 EMA state.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import train_t3_surgenetdino_v2b_e1 as sealed_train  # noqa: E402


RUN_NAME = "task3_surgenetdino_alldata_v1"
ALL_DATA_SENTINEL = "ALL_DATA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-root", type=Path, required=True)
    parser.add_argument("--unlabeled-root", type=Path, required=True)
    parser.add_argument(
        "--weights-path", type=Path, default=sealed_train.core.DEFAULT_WEIGHTS
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # The sealed trainer records and validates this field. The sentinel cannot collide with a
    # real video identifier and is never interpreted as a filesystem path.
    args.holdout_video = ALL_DATA_SENTINEL
    return args


def make_all_data_split(_sentinel: str):
    if _sentinel != ALL_DATA_SENTINEL:
        raise ValueError(f"Unexpected all-data split sentinel: {_sentinel}")

    def split(samples, val_video_count=2, seed=42):
        del val_video_count, seed
        eligible = [
            sample
            for sample in samples
            if (sample.video_id, int(sample.frame_idx)) not in sealed_train.EXCLUDED_FRAMES
        ]
        videos = sorted({sample.video_id for sample in eligible})
        if len(eligible) != 179 or len(videos) != 6:
            raise RuntimeError(
                f"Expected 179 eligible labeled frames/6 videos, got "
                f"{len(eligible)}/{len(videos)}"
            )
        return eligible, [], videos, [ALL_DATA_SENTINEL]

    return split


def frozen_all_data_config(sentinel: str) -> dict[str, Any]:
    if sentinel != ALL_DATA_SENTINEL:
        raise ValueError(f"Unexpected all-data config sentinel: {sentinel}")
    return {
        **sealed_train.contract.FROZEN_CONFIG,
        "run_name": RUN_NAME,
        "holdout_video": ALL_DATA_SENTINEL,
        "training_scope": "all_six_labeled_videos_no_validation",
        "eligible_labeled_frames": 179,
        "weights_path": str(sealed_train.core.DEFAULT_WEIGHTS.resolve()),
        "weights_sha256": sealed_train.core.ARTIFACT_SHA256,
        "pretrained_provenance": {
            "official_code_revision": "19e8325c87b826f5ae637787563891476bbe2b9f",
            "model_host_revision": "9b2c75b7d469850b750a715b90cb91d6319f7e30",
            "license": "CC-BY-NC-SA",
        },
    }


def main() -> int:
    # Patch only the three seams that encode E1's two-fold scope. The model, losses, optimizer,
    # schedules, augmentation, unlabeled stream, audit ledger, and checkpoint writer remain the
    # sealed implementation.
    sealed_train.parse_args = parse_args
    sealed_train.make_pinned_split = make_all_data_split
    sealed_train.contract.frozen_config = frozen_all_data_config
    return sealed_train.main()


if __name__ == "__main__":
    raise SystemExit(main())

