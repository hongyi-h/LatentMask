"""Post-training evaluation with size-stratified metrics.

Usage:
    python -m latentmask.scripts.evaluate \
        --model_dir /path/to/nnUNet_results/Dataset501_LiTS/LatentMaskTrainer__nnUNetPlans__3d_fullres/fold_0 \
        --output results/m3_main/r032_eval.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import nibabel as nib

from latentmask.utils.metrics import (
    compute_dice,
    compute_size_stratified_metrics,
    compute_delta_area,
    save_results,
)


def evaluate_predictions(pred_dir, gt_dir, output_path):
    """Evaluate predictions against ground truth with size stratification.

    Args:
        pred_dir: directory with prediction .nii.gz files.
        gt_dir: directory with ground truth .nii.gz files.
        output_path: path to save results JSON.
    """
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.nii.gz')))
    if not pred_files:
        pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.nii')))

    if not pred_files:
        print(f"ERROR: No prediction files found in {pred_dir}")
        return None

    all_results = []
    all_q1_recalls = []
    all_dices = []

    for pf in pred_files:
        case_id = os.path.basename(pf).replace('.nii.gz', '').replace('.nii', '')
        gt_path = os.path.join(gt_dir, os.path.basename(pf))
        if not os.path.exists(gt_path):
            # Try without extension differences
            for ext in ['.nii.gz', '.nii']:
                candidate = os.path.join(gt_dir, case_id + ext)
                if os.path.exists(candidate):
                    gt_path = candidate
                    break

        if not os.path.exists(gt_path):
            print(f"  Skipping {case_id}: no matching GT")
            continue

        pred = nib.load(pf).get_fdata().astype(np.int32)
        gt = nib.load(gt_path).get_fdata().astype(np.int32)

        # Binary evaluation (foreground vs background)
        result = compute_size_stratified_metrics(pred > 0, gt > 0)
        result['case_id'] = case_id
        all_results.append(result)
        all_dices.append(result['overall_dice'])

        if result['per_quintile']:
            q1 = result['per_quintile'][0]
            all_q1_recalls.append(q1['recall'])

        print(f"  {case_id}: Dice={result['overall_dice']:.4f}")

    if not all_results:
        print("No cases evaluated.")
        return None

    # Aggregate
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_cases': len(all_results),
        'dice_mean': float(np.mean(all_dices)),
        'dice_std': float(np.std(all_dices)),
        'q1_recall_mean': float(np.mean(all_q1_recalls)) if all_q1_recalls else None,
        'q1_recall_std': float(np.std(all_q1_recalls)) if all_q1_recalls else None,
        'per_case': all_results,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_results(summary, output_path)
    print(f"\nResults saved to {output_path}")
    print(f"Dice: {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
    if summary['q1_recall_mean'] is not None:
        print(f"Q1 recall: {summary['q1_recall_mean']:.4f} ± "
              f"{summary['q1_recall_std']:.4f}")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate predictions')
    parser.add_argument('--pred_dir', required=True,
                        help='Directory with prediction .nii.gz files')
    parser.add_argument('--gt_dir', required=True,
                        help='Directory with ground truth .nii.gz files')
    parser.add_argument('--output', default='results/evaluation.json',
                        help='Output JSON path')
    args = parser.parse_args()

    evaluate_predictions(args.pred_dir, args.gt_dir, args.output)
