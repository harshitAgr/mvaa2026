#!/usr/bin/env bash
# Deployed Task 2 training run: SafeMirror trainer + LargePatch plans, fold_all
# on Dataset505 (175 cases by default, 155 if built with MVAA_T2_EXCLUDE_MVAA_VAL=1).
# Preprocesses with the LargePatch plan (the base
# default-plan preprocess is run separately) and then trains fold_all.
# There is no held-out split and therefore no local evaluation step: Task 2 is
# score-neutralized in the final test phase and every labelled case is trained on.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BASE="${MVAA_NNUNET_BASE:-$PWD/data/nnunet}"
export nnUNet_raw="$BASE/raw"
export nnUNet_preprocessed="$BASE/preprocessed"
export nnUNet_results="$BASE/results"
export PYTHONUNBUFFERED=1

PY="$PWD/.venv-nnunet/bin"
OUTROOT="${MVAA_OUT:-$PWD/runs/task2_t2}"
LOG="$OUTROOT/final_175_foldall.log"

exec >> "$LOG" 2>&1

echo "[launch] $(date -u +%FT%TZ) creating nnUNetPlans_LargePatch for Dataset505"
"$PY/python" - <<'PYEOF'
import json, os, pathlib
PP = pathlib.Path(os.environ["nnUNet_preprocessed"]) / "Dataset505_MVAA_TEE_final"
base = json.load(open(PP / "nnUNetPlans.json"))
lp = json.loads(json.dumps(base))
lp["plans_name"] = "nnUNetPlans_LargePatch"
lp["configurations"]["3d_fullres"]["patch_size"] = [128, 192, 224]
json.dump(lp, open(PP / "nnUNetPlans_LargePatch.json", "w"), indent=4)
print("wrote", PP / "nnUNetPlans_LargePatch.json", "patch_size:", lp["configurations"]["3d_fullres"]["patch_size"])
PYEOF

echo "[preprocess] $(date -u +%FT%TZ) starting LargePatch preprocess"
"$PY/nnUNetv2_preprocess" -d 505 -plans_name nnUNetPlans_LargePatch -c 3d_fullres -np 6 \
  || { echo "FATAL: LargePatch preprocess failed"; exit 1; }
echo "[preprocess] $(date -u +%FT%TZ) done"

echo "[train] $(date -u +%FT%TZ) starting fold_all"
"$PY/nnUNetv2_train" 505 3d_fullres all -tr nnUNetTrainer_SafeMirror -p nnUNetPlans_LargePatch \
  || { echo "FATAL: fold_all training failed"; exit 1; }
echo "[train] $(date -u +%FT%TZ) fold_all done"

echo "FINAL_175_FOLDALL_DONE $(date -u +%FT%TZ)"
