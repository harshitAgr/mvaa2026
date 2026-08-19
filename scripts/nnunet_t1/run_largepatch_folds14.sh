#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Dataset511 LargePatch folds 1-4, completing the 27-case out-of-fold set
# (fold 0 is trained by run_largepatch_experiment.sh).
#
# Runs TWO folds at a time on one GPU: each fold peaks at ~11.8 GB, so a pair fits
# on a 24 GB card; ~33 s/epoch solo, ~2.3 h solo wall-clock per fold.
#
# Isolation: `-p nnUNetPlans_LargePatch` keeps results in their own directory, so
# the plain-plan Dataset512 weights are never touched.
#
# Usage: CUDA_VISIBLE_DEVICES=1 scripts/nnunet_t1/run_largepatch_folds14.sh
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$REPO/.venv-nnunet/bin"
BASE="${MVAA_NNUNET_BASE:-$REPO/data/nnunet}"
OUT="$REPO/runs/task1_largepatch_v1"
PLANS="nnUNetPlans_LargePatch"

export nnUNet_raw="$BASE/raw"
export nnUNet_preprocessed="$BASE/preprocessed"
export nnUNet_results="$BASE/results"

mkdir -p "$OUT"
echo "=== T1-L folds 1-4 start $(date -u) (GPU ${CUDA_VISIBLE_DEVICES:-<all>}) ==="

train_fold () {
    local f="$1"
    echo "[fold $f] start $(date -u)"
    "$BIN/nnUNetv2_train" 511 3d_fullres "$f" \
        -tr nnUNetTrainer_250epochs -p "$PLANS" \
        > "$OUT/train_fold${f}.log" 2>&1
    local rc=$?
    echo "[fold $f] exit=$rc $(date -u)"
    return $rc
}

FAILED=""
for pair in "1 2" "3 4"; do
    set -- $pair
    a="$1"; b="$2"
    train_fold "$a" & pid_a=$!
    train_fold "$b" & pid_b=$!
    wait $pid_a || FAILED="$FAILED $a"
    wait $pid_b || FAILED="$FAILED $b"
done

if [[ -n "$FAILED" ]]; then
    echo "FAILED folds:$FAILED — inspect $OUT/train_fold*.log"
    exit 1
fi
echo "all folds done $(date -u)"

# ---- n=27 OOF assembly + gate v2 -------------------------------------------
echo "--- n=27 OOF + gate v2 ---"
"$REPO/.venv/bin/python" "$REPO/scripts/nnunet_t1/eval_largepatch_oof.py" \
    --out-json "$OUT/oof_gate_v2.json" 2>&1 | tee "$OUT/oof_gate_v2.log"

echo "=== T1-L folds 1-4 complete $(date -u) ==="
