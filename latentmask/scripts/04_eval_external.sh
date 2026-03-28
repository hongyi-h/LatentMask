#!/usr/bin/env bash
# ============================================================================
# 04_eval_external.sh — Evaluate trained models on FUMPE and CAD-PE
# ============================================================================
set -euo pipefail

export nnUNet_raw="${nnUNet_raw:-$HOME/nnUNet_data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$HOME/nnUNet_data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$HOME/nnUNet_data/nnUNet_results}"

# Training dataset
TRAIN_DATASET="${1:-100}"
CONFIG="${2:-3d_fullres}"
TRAINER="${3:-LatentMaskTrainer}"

# External datasets
EXTERNAL_DATASETS=(101 102)  # FUMPE, CAD-PE

echo "=== External Evaluation ==="
echo "Trained on: Dataset ${TRAIN_DATASET}, Trainer: ${TRAINER}"

for EXT_DATASET in "${EXTERNAL_DATASETS[@]}"; do
    echo ""
    echo "--- Evaluating on Dataset ${EXT_DATASET} ---"

    INPUT_FOLDER="${nnUNet_raw}/Dataset${EXT_DATASET}_*/imagesTr"
    # Expand glob
    INPUT_FOLDER=$(ls -d ${nnUNet_raw}/Dataset${EXT_DATASET}_*/imagesTr 2>/dev/null | head -1)

    if [[ -z "$INPUT_FOLDER" ]]; then
        echo "WARNING: No imagesTr found for dataset ${EXT_DATASET}, skipping."
        continue
    fi

    OUTPUT_FOLDER="${nnUNet_results}/Dataset${TRAIN_DATASET}_*/cross_eval_${EXT_DATASET}"
    OUTPUT_FOLDER="${nnUNet_results}/external_eval/${TRAINER}/dataset${EXT_DATASET}"
    mkdir -p "${OUTPUT_FOLDER}"

    # Run inference (ensemble all 5 folds)
    nnUNetv2_predict \
        -i "${INPUT_FOLDER}" \
        -o "${OUTPUT_FOLDER}" \
        -d ${TRAIN_DATASET} \
        -c ${CONFIG} \
        -tr ${TRAINER} \
        -f 0 1 2 3 4 \
        --save_probabilities

    echo "Predictions saved to: ${OUTPUT_FOLDER}"
done

echo ""
echo "=== External evaluation inference complete ==="
echo "Run 05_compute_metrics.sh to compute metrics."
