#!/usr/bin/env python3
"""Export the registered all-data SurgeNetDINO epoch-70 EMA state."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch

from export_t3_surgenetdino_inference import SCHEMA_IN, SCHEMA_OUT, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    actual = sha256_file(source)
    if actual != args.expected_sha256:
        raise ValueError(f"Source SHA-256 mismatch: expected {args.expected_sha256}, got {actual}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    state = payload.get("ema_state")
    config = payload.get("args", {})
    if (
        payload.get("schema") != SCHEMA_IN
        or int(payload.get("epoch", -1)) != 70
        or payload.get("candidate_state_key") != "ema_state"
        or payload.get("heldout_opened_during_training") is not False
        or payload.get("heldout_metrics") is not None
        or config.get("run_name") != "task3_surgenetdino_alldata_v1"
        or config.get("holdout_video") != "ALL_DATA"
        or config.get("training_scope") != "all_six_labeled_videos_no_validation"
        or int(config.get("eligible_labeled_frames", -1)) != 179
        or not isinstance(state, Mapping)
        or len(state) != 237
    ):
        raise ValueError("Source is not the eligible all-data epoch-70 EMA checkpoint")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise TypeError("EMA state must be a string-to-tensor mapping")

    compact = {
        key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
        for key, value in state.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA_OUT,
            "fold_tag": "ALL6",
            "epoch": 70,
            "state_key": "ema_state",
            "source_checkpoint_sha256": actual,
            "tensor_count": len(compact),
            "state": compact,
        },
        output,
    )
    print(f"wrote {output}")
    print(f"sha256={sha256_file(output)}")
    print(f"bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

