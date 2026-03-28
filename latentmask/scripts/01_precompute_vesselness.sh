#!/usr/bin/env bash
# ============================================================================
# 01_precompute_vesselness.sh — Precompute vesselness maps for all datasets
# ============================================================================
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/nnUNet_data/nnUNet_raw}"

echo "=== Precomputing vesselness maps ==="

python -c "
import os, sys, glob
import numpy as np
import nibabel as nib
from pathlib import Path
from latentmask.utils.vesselness import frangi_vesselness_3d

output_root = '${OUTPUT_ROOT}'

for dataset_name in ['Dataset100_READPE', 'Dataset101_FUMPE', 'Dataset102_CADPE']:
    images_dir = Path(output_root) / dataset_name / 'imagesTr'
    vesselness_dir = Path(output_root) / dataset_name / 'vesselness'
    vesselness_dir.mkdir(exist_ok=True)

    if not images_dir.exists():
        print(f'Skipping {dataset_name}: no imagesTr')
        continue

    nii_files = sorted(images_dir.glob('*_0000.nii.gz'))
    print(f'{dataset_name}: {len(nii_files)} images')

    for nii_path in nii_files:
        case_id = nii_path.name.replace('_0000.nii.gz', '')
        out_path = vesselness_dir / f'{case_id}_vesselness.npy'
        if out_path.exists():
            print(f'  {case_id}: already exists, skipping')
            continue

        print(f'  {case_id}: computing vesselness...', end=' ', flush=True)
        nii = nib.load(str(nii_path))
        vol = nii.get_fdata().astype(np.float32)
        ves = frangi_vesselness_3d(vol, sigmas=(0.5, 1.0, 2.0, 4.0))
        np.save(str(out_path), ves)
        print(f'done (shape={ves.shape})')

print('=== Vesselness precomputation complete ===')
"
