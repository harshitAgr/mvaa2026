#!/usr/bin/env bash
# Train the two leave-one-video-out Task 3 members of the deployed ensemble.
#
#   SURGENET_E1_GPUS='0 1' bash scripts/run_t3_surgenetdino_v2b_e1.sh   # parallel
#   SURGENET_E1_GPUS='0'   bash scripts/run_t3_surgenetdino_v2b_e1.sh   # sequential
#
# The third member (C_ALL6) is trained separately with train_t3_surgenetdino_alldata.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-$ROOT/.venv/bin/python}"
DATA="${T3_LABELED_ROOT:-$ROOT/data/reference_data/t3_vid/train}"
UNLABELED="${T3_UNLABELED_ROOT:-$ROOT/data/images}"
WEIGHTS="${T3_BACKBONE:-$ROOT/data/pretrained/SurgeNetDINOv2_ViTb14_size336_SurgeNetXL.pth}"
RUN_ROOT="${T3_RUN_ROOT:-$ROOT/runs/t3_surgenetdino_e1}"

read -r -a GPU_LIST <<<"${SURGENET_E1_GPUS:-0}"
if [[ "${#GPU_LIST[@]}" -lt 1 || "${#GPU_LIST[@]}" -gt 2 ]]; then
  echo "SURGENET_E1_GPUS must contain one GPU (sequential) or two (parallel)" >&2
  exit 2
fi
GPU_979="${GPU_LIST[0]}"
GPU_675="${GPU_LIST[${#GPU_LIST[@]}-1]}"
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing existing target: $RUN_ROOT" >&2
  exit 3
fi
mkdir -p "$RUN_ROOT"

train_fold() {
  local gpu="$1" fold="$2" tag="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train_t3_surgenetdino_v2b_e1.py \
    --holdout-video "$fold" --labeled-root "$DATA" --unlabeled-root "$UNLABELED" \
    --weights-path "$WEIGHTS" --output-dir "$RUN_ROOT/f-$tag" \
    >"$RUN_ROOT/train-$tag.log" 2>&1
}

status=0
if [[ "${#GPU_LIST[@]}" -eq 1 ]]; then
  train_fold "$GPU_979" REC_20250205_102353_979A 979A || status=1
  [[ "$status" -eq 0 ]] && { train_fold "$GPU_675" REC_20250418_104439_675A 675A || status=1; }
else
  train_fold "$GPU_979" REC_20250205_102353_979A 979A & PID_979=$!
  train_fold "$GPU_675" REC_20250418_104439_675A 675A & PID_675=$!
  wait "$PID_979" || status=1
  wait "$PID_675" || status=1
fi
[[ "$status" -ne 0 ]] && { echo "at least one fold failed" >&2; exit 4; }

echo "done: $RUN_ROOT/f-979A  $RUN_ROOT/f-675A"
echo "next: export each with scripts/export_t3_surgenetdino_inference.py"
