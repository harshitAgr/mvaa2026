#!/usr/bin/env python3
"""Export a registered SurgeNetDINO E1 EMA state for inference only."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Mapping

import torch


SCHEMA_IN = "mvaa-t3-surgenetdino-v2b-e1-checkpoint-v1"
SCHEMA_OUT = "mvaa-t3-surgenetdino-v2b-inference-state-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--fold-tag", choices=("979A", "675A"), required=True)
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
    if (
        payload.get("schema") != SCHEMA_IN
        or int(payload.get("epoch", -1)) != 70
        or payload.get("candidate_state_key") != "ema_state"
        or payload.get("heldout_opened_during_training") is not False
        or payload.get("heldout_metrics") is not None
        or not isinstance(state, Mapping)
        or len(state) != 237
    ):
        raise ValueError("Source is not an eligible registered epoch-70 EMA checkpoint")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise TypeError("EMA state must be a string-to-tensor mapping")

    compact = {
        key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
        for key, value in state.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA_OUT,
            "fold_tag": args.fold_tag,
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
