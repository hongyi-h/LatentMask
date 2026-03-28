#!/usr/bin/env bash
# ============================================================================
# 02_train_latentmask.sh — Train LatentMask (full method A4) with 5-fold CV
# ============================================================================
set -euo pipefail

# ─── Environment setup ────────────────────────────────────────────────
export nnUNet_raw="${nnUNet_raw:-$HOME/nnUNet_data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$HOME/nnUNet_data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$HOME/nnUNet_data/nnUNet_results}"

# LatentMask-specific
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"

# Dataset to train on (100=READ-PE, 102=CAD-PE)
DATASET_ID="${1:-100}"
CONFIG="${2:-3d_fullres}"
TRAINER="${3:-LatentMaskTrainer}"
GPU="${4:-0}"

echo "=== Training LatentMask ==="
echo "Dataset: ${DATASET_ID}, Config: ${CONFIG}, Trainer: ${TRAINER}"
echo "RSPECT manifest: ${LATENTMASK_RSPECT_MANIFEST}"
echo "Aug-RSPECT manifest: ${LATENTMASK_AUG_RSPECT_MANIFEST}"

# ─── 5-fold cross-validation ──────────────────────────────────────────
for FOLD in 0 1 2 3 4; do
    echo ""
    echo "===== Fold ${FOLD}/4 ====="
    CUDA_VISIBLE_DEVICES=${GPU} nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} \
        -tr ${TRAINER} \
        --npz
done

echo ""
echo "=== Training complete for all 5 folds ==="
