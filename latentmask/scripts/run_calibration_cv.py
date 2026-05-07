"""M1 Calibration: Fit g_theta from offline box annotations with 3-fold CV.

v6.1 protocol: GT CCs + offline box annotations -> Hungarian matching
              -> (log_size, was_retained) -> isotonic / linear fit -> ECE

Produces TWO artifacts:
  1. {output}/calibration_{protocol}_v6.json — human-readable report
  2. {box_annotations_dir}/_calibration_fold{fold}.pkl — the pickled
     calibration object LOADED BY THE TRAINER. This is the contract:
     the method ships exactly what was evaluated here. No in-training
     re-fit, no protocol-function-form access.

Usage:
    python -m latentmask.scripts.run_calibration_cv \
        --dataset_dir data_preprocessed/Dataset501_LiTS \
        --box_annotations_dir data/box_annotations/P-steep \
        --protocol P-steep \
        --output results/m1_calibration_v6/
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from latentmask.calibration.isotonic_fit import (
    fit_g_theta_hungarian,
    cross_validate_g_theta_hungarian,
    predict_propensity,
    compute_ece,
    generate_annotation_pairs,
    isotonic_to_dict,
)
from latentmask.utils.cc_extraction import extract_connected_components


ARTIFACT_FORMAT_VERSION = '1.0'


def load_seg(gt_dir, key):
    """Load GT segmentation from gt_segmentations/."""
    npy_path = os.path.join(gt_dir, f'{key}.npy')
    if os.path.exists(npy_path):
        seg = np.load(npy_path)
        return seg[0] if seg.ndim == 4 else seg

    nii_path = os.path.join(gt_dir, f'{key}.nii.gz')
    if os.path.exists(nii_path):
        import nibabel as nib
        return nib.load(nii_path).get_fdata(dtype=np.float32).astype(np.int32)

    raise FileNotFoundError(f'No seg for {key} in {gt_dir}')


def load_offline_boxes(box_dir, key):
    """Load offline box annotations for one scan."""
    path = os.path.join(box_dir, f'{key}.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get('boxes', [])


def main():
    parser = argparse.ArgumentParser(
        description='M1: g_theta calibration from offline box annotations (v6.1)')
    parser.add_argument('--dataset_dir', required=True,
                        help='nnUNet preprocessed dataset root')
    parser.add_argument('--box_annotations_dir', required=True,
                        help='Directory with offline box JSONs for one protocol')
    parser.add_argument('--protocol', required=True,
                        help='Protocol name (for logging)')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--pixel_fraction', type=float, default=0.3)
    parser.add_argument('--n_folds', type=int, default=3)
    parser.add_argument('--fg_label', type=int, default=2)
    parser.add_argument('--min_cc_size', type=int, default=10)
    parser.add_argument('--iou_threshold', type=float, default=0.3)
    parser.add_argument('--output', default='results/m1_calibration_v6/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ece_gate', type=float, default=0.10,
                        help='Max per-fold CV ECE allowed (strict gate)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    gt_dir = os.path.join(args.dataset_dir, 'gt_segmentations')
    splits_path = os.path.join(args.dataset_dir, 'splits_final.json')

    # Load splits and get pixel keys (same logic as trainer + box gen)
    with open(splits_path) as f:
        splits = json.load(f)
    train_keys = sorted(splits[args.fold]['train'])
    rng_split = np.random.default_rng(seed=12345 + args.fold)
    perm = rng_split.permutation(len(train_keys)).tolist()
    n_pixel = max(1, int(args.pixel_fraction * len(train_keys)))
    pixel_keys = [train_keys[i] for i in perm[:n_pixel]]

    print(f"=== M1 Calibration (v6.1 offline boxes) ===")
    print(f"  Protocol: {args.protocol}")
    print(f"  Pixel scans: {len(pixel_keys)}")
    print(f"  Box annotations: {args.box_annotations_dir}")

    # Load segmentations and offline boxes for pixel scans
    seg_list = []
    offline_boxes = []
    for key in pixel_keys:
        seg_list.append(load_seg(gt_dir, key))
        offline_boxes.append(load_offline_boxes(args.box_annotations_dir, key))

    # Stats + rho bounds (computed here so trainer doesn't need GT access)
    all_ccs = []
    fill_ratios = []
    for seg_np in seg_list:
        ccs = extract_connected_components(
            seg_np, min_size=args.min_cc_size, fg_label=args.fg_label)
        all_ccs.extend(ccs)
        for cc in ccs:
            bbox = cc['bbox']
            bbox_vol = 1
            for (s, e) in bbox:
                bbox_vol *= max(e - s, 1)
            if bbox_vol > 0:
                fill_ratios.append(cc['size'] / bbox_vol)

    print(f"  Total CCs: {len(all_ccs)}")
    if not all_ccs:
        print("ERROR: No CCs found")
        sys.exit(1)

    mu = float(np.median([cc['log_size'] for cc in all_ccs]))
    fill_ratios = np.array(fill_ratios) if fill_ratios else np.array([0.3])
    rho_min = float(np.percentile(fill_ratios, 10))
    rho_max = float(np.percentile(fill_ratios, 95))
    print(f"  Median log-size: {mu:.2f}")
    print(f"  rho bounds: [{rho_min:.3f}, {rho_max:.3f}]")

    rng_fit = np.random.default_rng(seed=args.seed)

    # Full fit using offline boxes
    print("\nFitting g_theta (full dataset, offline boxes)...")
    ir, s0, linear_a, linear_b, stats = fit_g_theta_hungarian(
        seg_list, fg_label=args.fg_label, min_cc_size=args.min_cc_size,
        iou_threshold=args.iou_threshold, rng=rng_fit,
        offline_boxes_per_scan=offline_boxes)

    # Evaluate on full data
    log_sizes, annotations, _, _ = generate_annotation_pairs(
        seg_list, fg_label=args.fg_label, min_cc_size=args.min_cc_size,
        iou_threshold=args.iou_threshold, rng=rng_fit,
        offline_boxes_per_scan=offline_boxes)

    pred_isotonic = predict_propensity(ir, log_sizes, s0)
    ece_isotonic = compute_ece(pred_isotonic, annotations)

    pred_linear = np.clip(linear_a + linear_b * log_sizes, 0.01, 0.99)
    ece_linear = compute_ece(pred_linear, annotations)

    print(f"\n  Isotonic ECE: {ece_isotonic:.4f}")
    print(f"  Linear ECE:   {ece_linear:.4f}")
    print(f"  Linear: alpha = {linear_a:.4f} + {linear_b:.4f} * log_size")
    print(f"  Unmatched rate: {stats['unmatched_rate']:.3f}")
    print(f"  Ambiguous rate: {stats['ambiguous_rate']:.3f}")

    # Cross-validation
    print(f"\nRunning {args.n_folds}-fold CV...")
    rng_cv = np.random.default_rng(seed=args.seed)
    cv_results = cross_validate_g_theta_hungarian(
        seg_list, fg_label=args.fg_label, n_folds=args.n_folds,
        min_cc_size=args.min_cc_size, iou_threshold=args.iou_threshold,
        rng=rng_cv, offline_boxes_per_scan=offline_boxes)

    print(f"  CV ECE: {cv_results['mean_ece']:.4f} +/- {cv_results['std_ece']:.4f}")
    for i, ece in enumerate(cv_results['per_fold_ece']):
        print(f"    Fold {i}: ECE = {ece:.4f}")

    # Gate check
    max_fold_ece = (max(cv_results['per_fold_ece'])
                    if cv_results['per_fold_ece'] else float('inf'))
    gate_pass = max_fold_ece < args.ece_gate
    print(f"\n  Gate (max fold ECE < {args.ece_gate:.2f}): "
          f"{'PASS' if gate_pass else 'FAIL'} (max = {max_fold_ece:.4f})")

    # ── Artifact: the calibration the trainer will load ─────────────
    # Why a dict + np.interp instead of pickling the sklearn object:
    # sklearn IsotonicRegression pickles are not stable across sklearn
    # versions (Codex finding, 2026-05-06). We persist only the step
    # function (x_thresholds_, y_thresholds_) and reconstruct predictions
    # in the trainer via np.interp. This eliminates sklearn as a
    # load-time dependency.
    g_theta_dict = isotonic_to_dict(ir)

    # Identity pins so the trainer refuses mis-pairings.
    pixel_keys_hash = hashlib.sha256(
        '\n'.join(sorted(pixel_keys)).encode()).hexdigest()[:16]
    match_params = {
        'iou_threshold': args.iou_threshold,
        'min_cc_size': args.min_cc_size,
    }
    match_params_hash = hashlib.sha256(
        json.dumps(match_params, sort_keys=True).encode()).hexdigest()[:16]

    artifact = {
        'artifact_format_version': ARTIFACT_FORMAT_VERSION,
        'version': 'v6.1',
        'protocol': args.protocol,
        'fold': args.fold,
        'pixel_fraction': args.pixel_fraction,
        'fg_label': args.fg_label,
        'min_cc_size': args.min_cc_size,
        'iou_threshold': args.iou_threshold,
        'dataset_dir': os.path.abspath(args.dataset_dir),
        'box_annotations_dir': os.path.abspath(args.box_annotations_dir),
        'pixel_keys': pixel_keys,
        'pixel_keys_hash': pixel_keys_hash,
        'match_params_hash': match_params_hash,
        'n_total_ccs': len(all_ccs),
        'mu': mu,
        'g_theta': g_theta_dict,     # version-stable dict, not sklearn obj
        's0': s0,
        'linear_a': linear_a,
        'linear_b': linear_b,
        'rho_min': rho_min,
        'rho_max': rho_max,
        'ece_isotonic_full': ece_isotonic,
        'ece_linear_full': ece_linear,
        'cv_mean_ece': cv_results['mean_ece'],
        'cv_max_ece': max_fold_ece,
        'gate_pass': gate_pass,
        'gate_threshold': args.ece_gate,
        'seed': args.seed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'numpy_version': np.__version__,
    }

    artifact_path = os.path.join(
        args.box_annotations_dir, f'_calibration_fold{args.fold}.pkl')
    with open(artifact_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"\n  Calibration artifact → {artifact_path}")

    # Also drop a shadow copy next to the report (for archival/debugging)
    shadow_path = os.path.join(
        args.output, f'calibration_{args.protocol}_fold{args.fold}.pkl')
    with open(shadow_path, 'wb') as f:
        pickle.dump(artifact, f)

    # ── Human-readable report ──
    results = {
        'version': 'v6.1',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'protocol': args.protocol,
        'fold': args.fold,
        'fg_label': args.fg_label,
        'n_pixel_scans': len(pixel_keys),
        'n_total_ccs': len(all_ccs),
        'mu': mu,
        'rho_min': rho_min,
        'rho_max': rho_max,
        'full_fit': {
            's0': s0,
            'linear_a': linear_a,
            'linear_b': linear_b,
            'ece_isotonic': ece_isotonic,
            'ece_linear': ece_linear,
            **stats,
        },
        'cross_validation': cv_results,
        'gate_threshold': args.ece_gate,
        'cv_max_ece': max_fold_ece,
        'gate_pass': gate_pass,
        'artifact_path': artifact_path,
    }

    out_path = os.path.join(
        args.output, f'calibration_{args.protocol}_fold{args.fold}_v6.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Report → {out_path}")


if __name__ == '__main__':
    main()
