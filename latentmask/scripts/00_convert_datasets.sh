#!/usr/bin/env bash
# ============================================================================
# 00_convert_datasets.sh — Convert all raw datasets to nnUNet format
# ============================================================================
set -euo pipefail

# Root paths
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/nnUNet_data/nnUNet_raw}"

echo "=== Converting datasets to nnUNet format ==="
echo "Data root: ${DATA_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"

python -c "
from latentmask.data.preprocessing.convert_datasets import (
    convert_read_pe, convert_fumpe, convert_cadpe,
    build_rspect_manifest, build_augmented_rspect_manifest,
)

# Pixel-labeled datasets → nnUNet format
convert_read_pe('${DATA_ROOT}/READ', '${OUTPUT_ROOT}', dataset_id=100)
convert_fumpe('${DATA_ROOT}/FUMPE', '${OUTPUT_ROOT}', dataset_id=101)
convert_cadpe('${DATA_ROOT}/CAD-PE', '${OUTPUT_ROOT}', dataset_id=102)

# Weak-labeled datasets → manifests
build_rspect_manifest('${DATA_ROOT}/RSPECT', '${OUTPUT_ROOT}/manifests/rspect_manifest.csv')
build_augmented_rspect_manifest('${DATA_ROOT}/RSPECT', '${OUTPUT_ROOT}/manifests/aug_rspect_manifest.csv')
print('Done!')
"

echo "=== Running nnUNet preprocessing ==="
export nnUNet_raw="${OUTPUT_ROOT}"
export nnUNet_preprocessed="${OUTPUT_ROOT}/../nnUNet_preprocessed"
export nnUNet_results="${OUTPUT_ROOT}/../nnUNet_results"

mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"

# Plan and preprocess each dataset
for DATASET_ID in 100 101 102; do
    echo "--- Dataset ${DATASET_ID} ---"
    nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity -c 3d_fullres
done

echo "=== All datasets converted and preprocessed ==="
