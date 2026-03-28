#!/usr/bin/env bash
# ============================================================================
# setup.sh — Install LatentMask and register trainers with nnUNet
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

echo "=== Setting up LatentMask ==="

# 1. Install nnUNet in editable mode
echo "--- Installing nnUNet ---"
pip install -e "${PROJECT_ROOT}/nnUNet"

# 2. Install LatentMask in editable mode
echo "--- Installing LatentMask ---"
cd "${PROJECT_ROOT}/latentmask"
pip install -e .

# 3. Register LatentMask trainers with nnUNet by creating bridge file
echo "--- Registering trainers with nnUNet ---"
NNUNET_TRAINER_DIR="${PROJECT_ROOT}/nnUNet/nnunetv2/training/nnUNetTrainer"
cp "${PROJECT_ROOT}/latentmask/trainer/nnunet_bridge.py" \
   "${NNUNET_TRAINER_DIR}/latentmask_trainers.py"
echo "  Copied bridge to ${NNUNET_TRAINER_DIR}/latentmask_trainers.py"

# 4. Set up default environment variables
echo "--- Setting environment variables ---"
export nnUNet_raw="${nnUNet_raw:-${HOME}/nnUNet_data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${HOME}/nnUNet_data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${HOME}/nnUNet_data/nnUNet_results}"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Environment variables (add to your shell profile):"
echo "  export nnUNet_raw=${nnUNet_raw}"
echo "  export nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "  export nnUNet_results=${nnUNet_results}"
echo ""
echo "Next steps:"
echo "  1. bash latentmask/scripts/00_convert_datasets.sh"
echo "  2. bash latentmask/scripts/01_precompute_vesselness.sh"
echo "  3. bash latentmask/scripts/02_train_latentmask.sh"
