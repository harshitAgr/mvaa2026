#!/usr/bin/env bash
# install_trainers.sh
# ---------------------------------------------------------------------------- #
# Idempotent installer: symlinks all three custom nnU-Net trainers into the
# nnunetv2 variants dirs and import-checks each.  Re-running is safe (ln -sfn).
#
# VENV RESOLUTION (worktree-aware):
#   Priority 1: NNUNET_VENV env var (explicit override)
#   Priority 2: $REPO/.venv-nnunet (exists in the main repo; absent in worktrees)
#   Otherwise: fail loudly and ask for an explicit NNUNET_VENV
#
# The symlink TARGETS resolve to trainer files inside $REPO/scripts/nnunet_t2/,
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

echo "install_trainers.sh: REPO=$REPO"
echo "install_trainers.sh: venv=$PY_VENV  python=$PY"
echo "install_trainers.sh: variants dir=$VAR"

# Ensure target subdirs exist (nnunetv2 ships them but guard anyway)
mkdir -p "$VAR/loss" "$VAR/data_augmentation" "$VAR/training_length"

ln -sfn "$REPO/scripts/nnunet_t2/nnUNetTrainer_clDice.py"    "$VAR/loss/nnUNetTrainer_clDice.py"
ln -sfn "$REPO/scripts/nnunet_t2/nnUNetTrainer_Boundary.py"  "$VAR/loss/nnUNetTrainer_Boundary.py"
ln -sfn "$REPO/scripts/nnunet_t2/nnUNetTrainer_USAug.py"     "$VAR/data_augmentation/nnUNetTrainer_USAug.py"
ln -sfn "$REPO/scripts/nnunet_t2/nnUNetTrainer_NoMirror.py"   "$VAR/data_augmentation/nnUNetTrainer_NoMirror.py"
ln -sfn "$REPO/scripts/nnunet_t2/nnUNetTrainer_SafeMirror.py" "$VAR/data_augmentation/nnUNetTrainer_SafeMirror.py"
ln -sfn "$REPO/scripts/nnunet_t2/nnUNetTrainer_Seeded250.py" "$VAR/training_length/nnUNetTrainer_Seeded250.py"

echo "install_trainers.sh: symlinks created"

# Import-sanity check each trainer (uses sys.path insert so no install needed)
for T in nnUNetTrainer_clDice nnUNetTrainer_Boundary nnUNetTrainer_USAug nnUNetTrainer_Seeded250 nnUNetTrainer_NoMirror nnUNetTrainer_SafeMirror; do
    "$PY" -c "import sys; sys.path.insert(0,'$REPO/scripts/nnunet_t2'); import $T; print('import OK:', '$T')"
done

echo "install_trainers.sh: done"
