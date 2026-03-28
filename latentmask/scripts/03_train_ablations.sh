#!/usr/bin/env bash
# ============================================================================
# 03_train_ablations.sh — Train all ablation variants (A0-A3) + baselines
# ============================================================================
set -euo pipefail

export nnUNet_raw="${nnUNet_raw:-$HOME/nnUNet_data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$HOME/nnUNet_data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$HOME/nnUNet_data/nnUNet_results}"
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"

DATASET_ID="${1:-100}"
CONFIG="${2:-3d_fullres}"
GPU="${3:-0}"

echo "=== Training all ablation variants ==="

# ─── Baseline trainers ─────────────────────────────────────────────────
BASELINES=(
    "nnUNetTrainer"                      # Baseline: pixel-only nnUNet
    "MixedNaiveTrainer"                  # Baseline: naive tri-granularity
    "nnPUSegTrainer"                     # Baseline: nnPU-Seg
    "MeanTeacher3DTrainer"               # Baseline: Mean Teacher
    "CPSTrainer"                         # Baseline: Cross Pseudo Supervision
    "BoxSup3DTrainer"                    # Baseline: 3D BoxSup
)

# ─── Ablation trainers ────────────────────────────────────────────────
ABLATIONS=(
    "LatentMaskTrainer_A1_UniformPU"     # A1: PU with uniform e=0.5
    "LatentMaskTrainer_A2_PropNetOnly"   # A2: PU + PropNet
    "LatentMaskTrainer_A3_NoRefine"      # A3: PU + PropNet + smoothness
    "LatentMaskTrainer"                  # A4: Full LatentMask
)

TRAINERS=("${BASELINES[@]}" "${ABLATIONS[@]}")

for TRAINER in "${TRAINERS[@]}"; do
    echo ""
    echo "========================================"
    echo "Training: ${TRAINER}"
    echo "========================================"

    for FOLD in 0 1 2 3 4; do
        echo "--- Fold ${FOLD} ---"
        CUDA_VISIBLE_DEVICES=${GPU} nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} \
            -tr ${TRAINER} \
            --npz
    done
done

echo ""
echo "=== All ablation training complete ==="
