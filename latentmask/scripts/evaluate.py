"""Post-training evaluation with v5 metrics: Dice, HD95, per-lesion Dice by quintile.

Usage:
    python -m latentmask.scripts.evaluate \
        --model_dir /path/to/fold_0/channel_medium_seed42 \
        --output results/eval_C4_seed42.json
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
    compute_hd95,
    compute_per_lesion_metrics,
    aggregate_lesion_metrics_by_quintile,
    save_results,
)


def evaluate_predictions(pred_dir, gt_dir, output_path, fg_label=None):
    """Comprehensive v5 evaluation: Dice, HD95, per-lesion Dice by quintile.

    Args:
        pred_dir: directory with prediction .nii.gz files.
        gt_dir: directory with ground truth .nii.gz files.
        output_path: path to save results JSON.
        fg_label: foreground label in GT (None=any >0).
    """
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.nii.gz')))
    if not pred_files:
        pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.nii')))

    if not pred_files:
        print(f"ERROR: No prediction files found in {pred_dir}")
        return None

    all_results = []
    all_dices = []
    all_hd95s = []
    all_lesions = []

    for pf in pred_files:
        case_id = os.path.basename(pf).replace('.nii.gz', '').replace('.nii', '')
        gt_path = os.path.join(gt_dir, os.path.basename(pf))
        if not os.path.exists(gt_path):
            for ext in ['.nii.gz', '.nii']:
                candidate = os.path.join(gt_dir, case_id + ext)
                if os.path.exists(candidate):
                    gt_path = candidate
                    break

        if not os.path.exists(gt_path):
            print(f"  Skipping {case_id}: no matching GT")
            continue

        pred_nii = nib.load(pf)
        gt_nii = nib.load(gt_path)
        pred = pred_nii.get_fdata().astype(np.int32)
        gt = gt_nii.get_fdata().astype(np.int32)

        # Binary masks
        if fg_label is not None:
            pred_bin = (pred == fg_label).astype(np.int32)
            gt_bin = (gt == fg_label).astype(np.int32)
        else:
            pred_bin = (pred > 0).astype(np.int32)
            gt_bin = (gt > 0).astype(np.int32)

        # Overall Dice
        dice = compute_dice(pred_bin, gt_bin)
        all_dices.append(dice)

        # HD95
        voxel_spacing = gt_nii.header.get_zooms()[:3] if hasattr(gt_nii, 'header') else None
        hd95 = compute_hd95(pred_bin, gt_bin, voxel_spacing=voxel_spacing)
        all_hd95s.append(hd95)

        # Per-lesion metrics
        lesions = compute_per_lesion_metrics(pred_bin, gt_bin)
        all_lesions.extend(lesions)

        # Size-stratified recall (backward compat)
        strat = compute_size_stratified_metrics(pred_bin, gt_bin)

        result = {
            'case_id': case_id,
            'dice': dice,
            'hd95': hd95,
            'n_gt_lesions': len(lesions),
            'n_detected': sum(1 for l in lesions if l['detected']),
            'mean_lesion_dice': float(np.mean([l['dice'] for l in lesions])) if lesions else None,
            'per_quintile_recall': strat.get('per_quintile', []),
        }
        all_results.append(result)
        print(f"  {case_id}: Dice={dice:.4f}, HD95={hd95:.2f}, "
              f"lesions={len(lesions)}")

    if not all_results:
        print("No cases evaluated.")
        return None

    # Quintile aggregation across all lesions
    quintile_summary = aggregate_lesion_metrics_by_quintile(all_lesions)

    # Overall summary
    finite_hd = [h for h in all_hd95s if np.isfinite(h)]
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_cases': len(all_results),
        'dice_mean': float(np.mean(all_dices)),
        'dice_std': float(np.std(all_dices)),
        'hd95_mean': float(np.mean(finite_hd)) if finite_hd else None,
        'hd95_std': float(np.std(finite_hd)) if finite_hd else None,
        'total_lesions': len(all_lesions),
        'overall_detection_rate': (
            float(np.mean([l['detected'] for l in all_lesions]))
            if all_lesions else None),
        'overall_lesion_dice': (
            float(np.mean([l['dice'] for l in all_lesions]))
            if all_lesions else None),
        'per_quintile': quintile_summary,
        'q1_dice': (quintile_summary[0]['mean_dice']
                    if quintile_summary else None),
        'per_case': all_results,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_results(summary, output_path)

    print(f"\n{'='*50}")
    print(f"Results saved to {output_path}")
    print(f"  Dice:    {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
    if summary['hd95_mean'] is not None:
        print(f"  HD95:    {summary['hd95_mean']:.2f} ± {summary['hd95_std']:.2f}")
    if summary['q1_dice'] is not None:
        print(f"  Q1 Dice: {summary['q1_dice']:.4f} (smallest 20% tumors)")
    if quintile_summary:
        for q in quintile_summary:
            print(f"    {q['quintile']}: n={q['n']}, "
                  f"dice={q.get('mean_dice', 'N/A')}, "
                  f"det={q.get('detection_rate', 'N/A')}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate LatentMask v5')
    parser.add_argument('--pred_dir', required=True,
                        help='Directory with prediction NIfTIs')
    parser.add_argument('--gt_dir', required=True,
                        help='Directory with ground truth NIfTIs')
    parser.add_argument('--output', required=True,
                        help='Path to save JSON results')
    parser.add_argument('--fg_label', type=int, default=None,
                        help='Foreground label in GT (default: any >0)')
    args = parser.parse_args()

    evaluate_predictions(args.pred_dir, args.gt_dir, args.output,
                         fg_label=args.fg_label)


if __name__ == '__main__':
    main()
