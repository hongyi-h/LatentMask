"""M1 Calibration: Fit g_theta via Hungarian matching with 3-fold scan-level CV.

v5 protocol: GT CCs -> simulated box annotations -> Hungarian matching
            -> annotation status -> isotonic / linear fit -> ECE evaluation

Usage:
    python -m latentmask.scripts.run_calibration_cv \
        --dataset_id 501 --fold 0 --steepness medium \
        --output results/m1_calibration_v5/
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import nibabel as nib

from batchgenerators.utilities.file_and_folder_operations import load_json, join

from nnunetv2.paths import nnUNet_preprocessed
from latentmask.calibration.isotonic_fit import (
    fit_g_theta_hungarian,
    cross_validate_g_theta_hungarian,
    predict_propensity,
    compute_ece,
)
from latentmask.calibration.channel_simulator import make_channel_func
from latentmask.utils.cc_extraction import extract_connected_components


def load_pixel_scans(preprocessed_folder, pixel_keys, fg_label=2):
    """Load GT segmentations for pixel-labeled scans."""
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    dataset_class = infer_dataset_class(preprocessed_folder)
    ds = dataset_class(preprocessed_folder, pixel_keys)

    seg_list = []
    for key in pixel_keys:
        data, seg, _, props = ds.load_case(key)
        seg_list.append(seg[0])
    return seg_list


def main():
    parser = argparse.ArgumentParser(
        description='M1: g_theta calibration via Hungarian matching (v5)')
    parser.add_argument('--dataset_id', type=int, default=501)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--steepness', default='medium',
                        choices=['shallow', 'medium', 'steep'])
    parser.add_argument('--n_folds', type=int, default=3)
    parser.add_argument('--pixel_fraction', type=float, default=0.3)
    parser.add_argument('--fg_label', type=int, default=2)
    parser.add_argument('--min_cc_size', type=int, default=10)
    parser.add_argument('--iou_threshold', type=float, default=0.3)
    parser.add_argument('--output', default='results/m1_calibration_v5/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load dataset
    dataset_name = f'Dataset{args.dataset_id:03d}_LiTS'
    preprocessed_folder = join(nnUNet_preprocessed, dataset_name)

    if not os.path.isdir(preprocessed_folder):
        print(f"ERROR: Preprocessed data not found at {preprocessed_folder}")
        sys.exit(1)

    dataset_json = load_json(join(preprocessed_folder, 'dataset.json'))

    # Get pixel-labeled subset (consistent with trainer split)
    all_keys = sorted(dataset_json.get('training', {}).keys())
    if not all_keys:
        # nnUNet format: numeric keys
        all_keys = sorted([
            f for f in os.listdir(preprocessed_folder)
            if f.endswith('.npz') or f.endswith('.npy')
        ])
        all_keys = [os.path.splitext(k)[0] for k in all_keys]
        # Filter to actual npz files
        all_keys = sorted(set(all_keys))

    # Use same split as trainer
    rng = np.random.default_rng(seed=12345 + args.fold)
    perm = rng.permutation(len(all_keys)).tolist()
    n_pixel = max(1, int(args.pixel_fraction * len(all_keys)))
    pixel_keys = [all_keys[i] for i in perm[:n_pixel]]

    print(f"=== M1 Calibration (v5 Hungarian) ===")
    print(f"  Dataset: {dataset_name}")
    print(f"  Pixel scans: {len(pixel_keys)}")
    print(f"  Steepness: {args.steepness}")
    print(f"  N-folds: {args.n_folds}")

    # Load segmentations
    print("\nLoading pixel-labeled segmentations...")
    seg_list = load_pixel_scans(preprocessed_folder, pixel_keys,
                                fg_label=args.fg_label)

    # Compute mu for channel simulator
    all_ccs = []
    for seg_np in seg_list:
        ccs = extract_connected_components(
            seg_np, min_size=args.min_cc_size, fg_label=args.fg_label)
        all_ccs.extend(ccs)

    if not all_ccs:
        print("ERROR: No CCs found in pixel scans")
        sys.exit(1)

    all_log_sizes = np.array([cc['log_size'] for cc in all_ccs])
    mu = float(np.median(all_log_sizes))
    print(f"  Median log-size: {mu:.2f}")
    print(f"  Total CCs: {len(all_ccs)}")

    # Create simulated drop function
    g_true = make_channel_func(args.steepness, mu)

    def drop_fn(log_sizes):
        return g_true(log_sizes)

    rng_fit = np.random.default_rng(seed=args.seed)

    # Full fit
    print("\nFitting g_theta (full dataset)...")
    ir, s0, linear_a, linear_b, stats = fit_g_theta_hungarian(
        seg_list, fg_label=args.fg_label, min_cc_size=args.min_cc_size,
        iou_threshold=args.iou_threshold, drop_fn=drop_fn, rng=rng_fit)

    # Evaluate on full data
    from latentmask.calibration.isotonic_fit import generate_annotation_pairs
    log_sizes, annotations, scan_idx, _ = generate_annotation_pairs(
        seg_list, fg_label=args.fg_label, min_cc_size=args.min_cc_size,
        iou_threshold=args.iou_threshold, drop_fn=drop_fn, rng=rng_fit)

    pred_isotonic = predict_propensity(ir, log_sizes, s0)
    ece_isotonic = compute_ece(pred_isotonic, annotations)

    # Linear predictions
    pred_linear = np.clip(linear_a + linear_b * log_sizes, 0.01, 0.99)
    ece_linear = compute_ece(pred_linear, annotations)

    print(f"\n  Isotonic ECE: {ece_isotonic:.4f}")
    print(f"  Linear ECE:   {ece_linear:.4f}")
    print(f"  Linear: alpha = {linear_a:.4f} + {linear_b:.4f} * log_size")
    print(f"  Unmatched rate: {stats['unmatched_rate']:.3f}")

    # Cross-validation
    print(f"\nRunning {args.n_folds}-fold CV...")
    rng_cv = np.random.default_rng(seed=args.seed)
    cv_results = cross_validate_g_theta_hungarian(
        seg_list, fg_label=args.fg_label, n_folds=args.n_folds,
        min_cc_size=args.min_cc_size, iou_threshold=args.iou_threshold,
        drop_fn=drop_fn, rng=rng_cv)

    print(f"  CV ECE: {cv_results['mean_ece']:.4f} ± {cv_results['std_ece']:.4f}")
    for i, ece in enumerate(cv_results['per_fold_ece']):
        print(f"    Fold {i}: ECE = {ece:.4f}")

    # Save results
    results = {
        'version': 'v5',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'steepness': args.steepness,
        'n_pixel_scans': len(pixel_keys),
        'n_total_ccs': len(all_ccs),
        'mu': mu,
        'full_fit': {
            's0': s0,
            'linear_a': linear_a,
            'linear_b': linear_b,
            'ece_isotonic': ece_isotonic,
            'ece_linear': ece_linear,
            **stats,
        },
        'cross_validation': cv_results,
    }

    out_path = os.path.join(args.output,
                            f'calibration_{args.steepness}_v5.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
