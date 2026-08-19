#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# T1-L: LargePatch experiment for Task 1 (Dataset511_T1CT, supervised/clean CV).
#
# Plans: nnUNetPlans_LargePatch  — patch [112,160,192] vs default [112,128,160].
#   The default patch tiles BOTH in-plane axes against the median volume
#   [105,141,161], so every prediction carries sliding-window seams through the
#   valve and no tile sees the whole anatomy. [112,160,192] covers the median
#   volume on all three axes (divisibility: axis0 /16 = 7, axes1-2 /32 = 5,6).
#   Only patch_size and plans_name differ from nnUNetPlans.json; data_identifier
#   is unchanged, so NO re-preprocessing is needed. Mirrors the Task 2 deployed
#   nnUNetPlans_LargePatch.
#
# Trainer: nnUNetTrainer_250epochs (stock) — single-variable change vs the
#   supervised comparator.
#
# Isolation: `-p nnUNetPlans_LargePatch` routes results to
#   results/Dataset511_T1CT/nnUNetTrainer_250epochs__nnUNetPlans_LargePatch__3d_fullres/
#   so deployed Dataset512 fold-0 weights and all existing runs are untouched.
#
# Steps:
#   1. Smoke gate — 2 epochs via Seeded250 + LargePatch plans (catches OOM/NaN
#      from the 1.5x larger patch) into a throwaway results dir.
#   2. Fold-0 training — 250 epochs. nnU-Net runs its own held-out validation at
#      the end and writes preds to <results>/fold_0/validation/T1CT_<id>.nii.gz.
#   3. Fold-0 kill check — scripts/nnunet_t1/eval_largepatch_fold0.py compares the
#      6 fold-0 held-out cases against the staged supervised OOF baseline.
#      This is a KILL CHECK, not a promotion gate: promotion needs full n=27 OOF
#      through runs/task1_ceilingbreak_v1/eval_gate/gate.py (gate v2).
#
#
# Usage:
#   scripts/nnunet_t1/run_largepatch_experiment.sh            # smoke only
#   scripts/nnunet_t1/run_largepatch_experiment.sh --full     # smoke + fold-0 + eval
#
# Est. time: smoke ~5 min; fold-0 250 ep ~4-5 h (1.5x patch volume vs the
#   ~2.5-3 h baseline). Set CUDA_VISIBLE_DEVICES before calling.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$REPO/.venv-nnunet/bin"
BASE="${MVAA_NNUNET_BASE:-$REPO/data/nnunet}"
OUT="$REPO/runs/task1_largepatch_v1"
PLANS="nnUNetPlans_LargePatch"
DATASET=511

export nnUNet_raw="$BASE/raw"
export nnUNet_preprocessed="$BASE/preprocessed"

mkdir -p "$OUT"

if [[ ! -f "$nnUNet_preprocessed/Dataset511_T1CT/${PLANS}.json" ]]; then
    echo "FATAL: ${PLANS}.json missing under $nnUNet_preprocessed/Dataset511_T1CT/"
    exit 1
fi

echo "=== T1-L LargePatch experiment $(date -u) ==="
echo "GPU: ${CUDA_VISIBLE_DEVICES:-<all>}"
"$BIN/python" - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ["nnUNet_preprocessed"]) / "Dataset511_T1CT" / "nnUNetPlans_LargePatch.json"
c = json.loads(p.read_text())["configurations"]["3d_fullres"]
print(f"[plans] patch_size={c['patch_size']} batch_size={c['batch_size']} "
      f"data_identifier={c['data_identifier']} median_image={c['median_image_size_in_voxels']}")
PY

# ---- 1. Smoke gate ---------------------------------------------------------
SMOKE_RESULTS="$OUT/results_smoke"
SMOKE_LOG="$OUT/smoke.log"
rm -rf "$SMOKE_RESULTS"
echo "--- smoke gate (fold 0, 2 epochs, Seeded250 + $PLANS) ---"
if ! SEED250_NUM_EPOCHS=2 nnUNet_results="$SMOKE_RESULTS" \
     "$BIN/nnUNetv2_train" "$DATASET" 3d_fullres 0 \
       -tr nnUNetTrainer_Seeded250 \
       -p  "$PLANS" \
       > "$SMOKE_LOG" 2>&1; then
    echo "SMOKE FAILED — training crashed (OOM or other error); tail:"
    tail -30 "$SMOKE_LOG"
    exit 1
fi
if grep -qiE "nan,? *nan|loss.*nan|nan.*loss" "$SMOKE_LOG"; then
    echo "SMOKE FAILED — NaN loss detected"; tail -20 "$SMOKE_LOG"; exit 1
fi
grep -E "^ *(Epoch|train_loss|val_loss)" "$SMOKE_LOG" | tail -8
echo "SMOKE PASSED"
rm -rf "$SMOKE_RESULTS"

if [[ "${1:-}" != "--full" ]]; then
    echo "smoke-only mode; stopping. Re-run with --full to train fold-0."
    exit 0
fi

# ---- 2. Fold-0 training ----------------------------------------------------
export nnUNet_results="$BASE/results"
echo "--- fold-0 training (nnUNetTrainer_250epochs + $PLANS) ---"
echo "    started $(date -u)"
"$BIN/nnUNetv2_train" "$DATASET" 3d_fullres 0 \
    -tr nnUNetTrainer_250epochs \
    -p  "$PLANS" 2>&1 | tee "$OUT/train_fold0.log"
RC=${PIPESTATUS[0]}
echo "    fold-0 exit=$RC  $(date -u)"
if [[ "$RC" -ne 0 ]]; then echo "TRAINING FAILED"; exit 1; fi

# ---- 3. Fold-0 kill check --------------------------------------------------
echo "--- fold-0 kill check vs staged supervised OOF ---"
"$REPO/.venv/bin/python" "$REPO/scripts/nnunet_t1/eval_largepatch_fold0.py" \
    --out-json "$OUT/fold0_eval.json" 2>&1 | tee "$OUT/fold0_eval.log"

echo "=== T1-L complete $(date -u) ==="
echo "If the kill check passes, train folds 1-4 for the gate-v2 read:"
echo "  for f in 1 2 3 4; do"
echo "    nnUNet_results=$BASE/results $BIN/nnUNetv2_train $DATASET 3d_fullres \$f \\"
echo "      -tr nnUNetTrainer_250epochs -p $PLANS"
echo "  done"
