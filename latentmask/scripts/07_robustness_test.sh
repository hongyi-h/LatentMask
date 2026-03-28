#!/usr/bin/env bash
# ============================================================================
# 07_robustness_test.sh — Robustness to incomplete weak labels (Table 6)
#
# Synthetically drops weak annotations at increasing rates during training.
# ============================================================================
set -euo pipefail

export nnUNet_raw="${nnUNet_raw:-$HOME/nnUNet_data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$HOME/nnUNet_data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$HOME/nnUNet_data/nnUNet_results}"
export LATENTMASK_RSPECT_MANIFEST="${nnUNet_raw}/manifests/rspect_manifest.csv"
export LATENTMASK_AUG_RSPECT_MANIFEST="${nnUNet_raw}/manifests/aug_rspect_manifest.csv"

DATASET_ID="${1:-100}"
CONFIG="3d_fullres"
GPU="${2:-0}"

echo "=== Robustness experiment (Table 6) ==="

# Drop rates to test
DROP_RATES=(0.0 0.1 0.2 0.3 0.4)

# Trainers to compare
TRAINERS=("LatentMaskTrainer" "nnPUSegTrainer")

for DROP_RATE in "${DROP_RATES[@]}"; do
    echo ""
    echo "========================================"
    echo "Drop rate: ${DROP_RATE}"
    echo "========================================"

    # Set drop rate via environment variable
    export LATENTMASK_WEAK_DROP_RATE="${DROP_RATE}"

    for TRAINER in "${TRAINERS[@]}"; do
        echo "--- ${TRAINER} ---"
        # Train fold 0 only for speed (full paper: all 5 folds)
        CUDA_VISIBLE_DEVICES=${GPU} nnUNetv2_train ${DATASET_ID} ${CONFIG} 0 \
            -tr ${TRAINER} \
            --npz

        # Evaluate
        python -m latentmask.scripts.05_compute_metrics \
            --results_root "${nnUNet_results}" \
            --dataset_id ${DATASET_ID} \
            --trainer ${TRAINER} \
            --mode cv
    done
done

echo ""
echo "=== Robustness experiment complete ==="
