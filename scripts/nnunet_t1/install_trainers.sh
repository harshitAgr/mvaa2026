#!/usr/bin/env bash
# install_trainers.sh (Task 1)
# ---------------------------------------------------------------------------- #
# Idempotent installer: symlinks Task-1 custom nnU-Net trainers into the
# nnunetv2 variants dirs and import-checks each. Re-running is safe (ln -sfn).
# Mirrors scripts/nnunet_t2/install_trainers.sh (same venv-resolution + symlink
# + import-check pattern), scoped to the Task-1 trainers in this directory.
#
# NOTE: nnUNetTrainer_FGConsistency is ALREADY installed (symlinked into
# variants/semi_supervised/ by an earlier session) and has existing trained
# checkpoints under it — this script does not touch it. Manages
# nnUNetTrainer_SizeStrat250, nnUNetTrainer_TotalSegInit (fold_all
# all-data + TotalSeg encoder-init deployment trainer), and the plain
# nnUNetTrainer_T1AllData250 deployment candidate, and the isolated all-27 x18
# trusted-real-weight candidate nnUNetTrainer_T1AllData18.
#
# VENV RESOLUTION (worktree-aware):
#   Priority 1: NNUNET_VENV env var (explicit override)
#   Priority 2: $REPO/.venv-nnunet (exists in the main repo; absent in worktrees)
#   Otherwise: fail loudly and ask for an explicit NNUNET_VENV
#
# The symlink TARGETS resolve to trainer files inside $REPO/scripts/nnunet_t1/,
# so when run from a worktree they point at worktree files — intentional for
# experiment isolation.
# ---------------------------------------------------------------------------- #
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -n "${NNUNET_VENV:-}" ]]; then
    PY_VENV="$NNUNET_VENV"
elif [[ -d "$REPO/.venv-nnunet" ]]; then
    PY_VENV="$REPO/.venv-nnunet"
else
    echo "error: no nnU-Net venv found. Set NNUNET_VENV=/path/to/venv, or create $REPO/.venv-nnunet" >&2
    exit 1
fi

PY="$PY_VENV/bin/python"

# Detect python minor version dynamically so this survives a future venv upgrade
PYVER="$("$PY" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
VAR="$PY_VENV/lib/$PYVER/site-packages/nnunetv2/training/nnUNetTrainer/variants"

echo "install_trainers.sh (t1): REPO=$REPO"
echo "install_trainers.sh (t1): venv=$PY_VENV  python=$PY"
echo "install_trainers.sh (t1): variants dir=$VAR"

# Ensure target subdir exists (nnunetv2 ships it but guard anyway)
mkdir -p "$VAR/sampling"

ln -sfn "$REPO/scripts/nnunet_t1/nnUNetTrainer_SizeStrat250.py"  "$VAR/sampling/nnUNetTrainer_SizeStrat250.py"
ln -sfn "$REPO/scripts/nnunet_t1/nnUNetTrainer_TotalSegInit.py"  "$VAR/sampling/nnUNetTrainer_TotalSegInit.py"
ln -sfn "$REPO/scripts/nnunet_t1/nnUNetTrainer_T1AllData250.py"  "$VAR/sampling/nnUNetTrainer_T1AllData250.py"
ln -sfn "$REPO/scripts/nnunet_t1/nnUNetTrainer_T1AllData18.py"  "$VAR/sampling/nnUNetTrainer_T1AllData18.py"

echo "install_trainers.sh (t1): symlinks created"

# Import-sanity check (uses sys.path insert so no install needed)
for T in nnUNetTrainer_SizeStrat250 nnUNetTrainer_TotalSegInit nnUNetTrainer_T1AllData250 nnUNetTrainer_T1AllData18; do
    "$PY" -c "import sys; sys.path.insert(0,'$REPO/scripts/nnunet_t1'); import $T; print('import OK:', '$T')"
done

echo "install_trainers.sh (t1): done"
