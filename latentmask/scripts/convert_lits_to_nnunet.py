"""Convert LiTS data to nnUNet format (Dataset501_LiTS).

Usage:
    python -m latentmask.scripts.convert_lits_to_nnunet \
        --input_dir data/LiTS_Tiny \
        --output_dir /path/to/nnUNet_raw/Dataset501_LiTS

LiTS directory structure expected:
    Training Batch/
        volume-0.nii      # CT volumes
        volume-1.nii
        segmentation-0.nii  # Labels (0=bg, 1=liver, 2=tumor)
        segmentation-1.nii

nnUNet output:
    Dataset501_LiTS/
        imagesTr/   case_XXXXX_0000.nii.gz
        labelsTr/   case_XXXXX.nii.gz
        dataset.json
"""
import argparse
import glob
import json
import os
import shutil

import nibabel as nib
import numpy as np


def find_lits_pairs(input_dir):
    """Find matching volume-segmentation pairs in LiTS directory."""
    train_dir = os.path.join(input_dir, 'Training Batch')
    if not os.path.isdir(train_dir):
        # Try flat structure
        train_dir = input_dir

    # Find volume files
    vol_patterns = [
        os.path.join(train_dir, 'volume-*.nii*'),
    ]
    vol_files = []
    for pat in vol_patterns:
        vol_files.extend(sorted(glob.glob(pat)))

    # Find segmentation files
    seg_patterns = [
        os.path.join(train_dir, 'segmentation-*.nii*'),
    ]
    seg_files = []
    for pat in seg_patterns:
        seg_files.extend(sorted(glob.glob(pat)))

    # Match by index
    pairs = []
    vol_dict = {}
    for vf in vol_files:
        base = os.path.basename(vf)
        idx = base.replace('volume-', '').split('.')[0]
        vol_dict[idx] = vf

    seg_dict = {}
    for sf in seg_files:
        base = os.path.basename(sf)
        idx = base.replace('segmentation-', '').split('.')[0]
        seg_dict[idx] = sf

    matched_indices = sorted(set(vol_dict.keys()) & set(seg_dict.keys()))

    for idx in matched_indices:
        pairs.append((vol_dict[idx], seg_dict[idx], idx))

    return pairs, seg_dict, vol_dict


def convert(input_dir, output_dir):
    """Convert LiTS to nnUNet format."""
    pairs, seg_dict, vol_dict = find_lits_pairs(input_dir)

    # Check for missing volumes
    seg_only = set(seg_dict.keys()) - set(vol_dict.keys())
    if seg_only:
        print(f"WARNING: Found {len(seg_only)} segmentation(s) without "
              f"matching volumes: {sorted(seg_only)}")
        print("  Training volumes are required. Download them from the "
              "LiTS challenge website.")

    if len(pairs) == 0:
        print(f"ERROR: No matched volume-segmentation pairs found in "
              f"{input_dir}")
        print(f"  Volumes found: {len(vol_dict)}")
        print(f"  Segmentations found: {len(seg_dict)}")
        if len(seg_dict) > 0 and len(vol_dict) == 0:
            print("\n  HINT: Only segmentation files found. You need to "
                  "download the CT volumes from the LiTS challenge.")
            print("  Place volume-X.nii files alongside segmentation-X.nii "
                  "in the 'Training Batch' directory.\n")
        return False

    # Create output directories
    images_dir = os.path.join(output_dir, 'imagesTr')
    labels_dir = os.path.join(output_dir, 'labelsTr')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    training_list = []

    for vol_path, seg_path, idx in pairs:
        case_id = f'case_{int(idx):05d}'
        print(f"  Converting {case_id}: {os.path.basename(vol_path)}")

        # Load and save volume
        vol_nii = nib.load(vol_path)
        out_vol = os.path.join(images_dir, f'{case_id}_0000.nii.gz')
        nib.save(vol_nii, out_vol)

        # Load segmentation and align spatial metadata to volume
        seg_nii = nib.load(seg_path)
        seg_data = seg_nii.get_fdata().astype(np.uint8)
        # Copy affine and header from the CT volume so spacing/origin/direction match
        seg_aligned = nib.Nifti1Image(seg_data, vol_nii.affine, vol_nii.header)
        out_seg = os.path.join(labels_dir, f'{case_id}.nii.gz')
        nib.save(seg_aligned, out_seg)

        training_list.append({
            'image': f'./imagesTr/{case_id}_0000.nii.gz',
            'label': f'./labelsTr/{case_id}.nii.gz',
        })

    # Write dataset.json
    dataset_json = {
        'channel_names': {'0': 'CT'},
        'labels': {
            'background': 0,
            'liver': 1,
            'tumor': 2,
        },
        'numTraining': len(training_list),
        'file_ending': '.nii.gz',
    }

    json_path = os.path.join(output_dir, 'dataset.json')
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nConverted {len(pairs)} cases to {output_dir}")
    print(f"Next step: nnUNetv2_plan_and_preprocess -d 501 "
          f"--verify_dataset_integrity")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert LiTS to nnUNet format'
    )
    parser.add_argument('--input_dir', required=True,
                        help='Path to LiTS data directory')
    parser.add_argument('--output_dir', required=True,
                        help='Path to nnUNet raw data output')
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir)
