#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Train Dataset512 self-train folds 1-4 with the deployed recipe, completing the
# 5-member T1 self-train ensemble (fold 0 is trained separately).
#
# Rationale: on the 27-case out-of-fold set, ASD is tightly coupled to DSC
# (ASD = 2.007*(1-DSC) - 0.0332, r = 0.934). Measured ASD sat well above what
# that line predicts for our DSC, which points at a small number of catastrophic
# far-field cases rather than a uniformly harder cohort. Averaging same-recipe
# members is the canonical suppressor for exactly that failure mode.
#
# This script never writes fold_0; it only adds folds 1-4 under
#   <nnUNet_results>/Dataset512_T1CT_selftrain/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/
#
# Note: a self-train ensemble cannot be scored against the 27 labeled cases without
# leakage, since those cases are teacher-leaked into the pseudo-label pool.
#
# Usage: CUDA_VISIBLE_DEVICES=1 scripts/nnunet_t1/run_selftrain_folds14.sh
# Est: ~2.5-3 h per fold, two concurrent -> ~6-7 h total.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$REPO/.venv-nnunet/bin"
BASE="${MVAA_NNUNET_BASE:-$REPO/data/nnunet}"
OUT="$REPO/runs/task1_selftrain_ens_v1"
DEPLOYED="$BASE/results/Dataset512_T1CT_selftrain/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

export nnUNet_raw="$BASE/raw"
export nnUNet_preprocessed="$BASE/preprocessed"
export nnUNet_results="$BASE/results"

mkdir -p "$OUT"

# Fingerprint the deployed checkpoint before and after — it must not change.
DEPLOYED_SHA_BEFORE="$(sha256sum "$DEPLOYED" | cut -d' ' -f1)"
echo "=== T1-E2 self-train folds 1-4 start $(date -u) (GPU ${CUDA_VISIBLE_DEVICES:-<all>}) ==="
echo "deployed fold_0 sha256 (before): $DEPLOYED_SHA_BEFORE"

train_fold () {
    local f="$1"
    echo "[fold $f] start $(date -u)"
    "$BIN/nnUNetv2_train" 512 3d_fullres "$f" -tr nnUNetTrainer_250epochs \
        > "$OUT/train_fold${f}.log" 2>&1
    local rc=$?
    echo "[fold $f] exit=$rc $(date -u)"
    return $rc
}

FAILED=""
for pair in "1 2" "3 4"; do
    set -- $pair
    train_fold "$1" & pid_a=$!
    train_fold "$2" & pid_b=$!
    wait $pid_a || FAILED="$FAILED $1"
    wait $pid_b || FAILED="$FAILED $2"
done

DEPLOYED_SHA_AFTER="$(sha256sum "$DEPLOYED" | cut -d' ' -f1)"
echo "deployed fold_0 sha256 (after):  $DEPLOYED_SHA_AFTER"
if [[ "$DEPLOYED_SHA_BEFORE" != "$DEPLOYED_SHA_AFTER" ]]; then
    echo "FATAL: the deployed fold_0 checkpoint CHANGED during this run"
    exit 2
fi
echo "deployed checkpoint verified byte-identical"

if [[ -n "$FAILED" ]]; then
    echo "FAILED folds:$FAILED — inspect $OUT/train_fold*.log"
    exit 1
fi
echo "=== T1-E2 complete $(date -u) — members ready, shipping is a separate decision ==="
