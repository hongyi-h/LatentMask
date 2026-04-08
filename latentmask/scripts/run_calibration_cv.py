"""M1: Cross-validated calibration fidelity experiment (CPU-only).

Runs Block B1 from EXPERIMENT_PLAN.md:
  - 5-fold × 3-steepness isotonic calibration
  - Within-steepness ECE (target < 0.05)
  - Cross-steepness transfer ECE (target < 0.10)
  - Bootstrap 95% CI

Usage:
    python -m latentmask.scripts.run_calibration_cv \
        --preprocessed_dir $nnUNet_preprocessed/Dataset501_LiTS/nnUNetPlans_3d_fullres \
        --output_dir results/m1_calibration

Input: preprocessed nnUNet segmentations (or raw .nii segmentations).
Output: results/m1_calibration/calibration_cv_results.json
"""
import argparse
import json
import os
import time

import numpy as np

from latentmask.utils.cc_extraction import extract_connected_components
from latentmask.calibration.isotonic_fit import (
    fit_isotonic, predict_propensity, compute_ece, bootstrap_ece_ci,
    cross_validate_calibration,
)
from latentmask.calibration.channel_simulator import make_channel_func, simulate_channel


def load_all_ccs_from_preprocessed(preprocessed_dir, min_cc_size=10):
    """Load all CCs from preprocessed nnUNet segmentations."""
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    dataset_class = infer_dataset_class(preprocessed_dir)
    # Find all case identifiers
    import glob
    npz_files = glob.glob(os.path.join(preprocessed_dir, '*.npz'))
    keys = [os.path.basename(f).replace('.npz', '') for f in npz_files]

    if not keys:
        # Try blosc2 format
        b2_files = glob.glob(os.path.join(preprocessed_dir, '*.b2nd'))
        keys = list(set(
            os.path.basename(f).split('.')[0] for f in b2_files
        ))

    if not keys:
        raise FileNotFoundError(
            f"No preprocessed cases found in {preprocessed_dir}"
        )

    print(f"Found {len(keys)} cases. Extracting CCs...")
    all_ccs = []
    for key in sorted(keys):
        ds = dataset_class(preprocessed_dir, [key])
        data, seg, _, props = ds.load_case(key)
        ccs = extract_connected_components(seg[0], min_size=min_cc_size)
        all_ccs.extend(ccs)
        print(f"  {key}: {len(ccs)} CCs")

    print(f"Total CCs: {len(all_ccs)}")
    return all_ccs


def load_all_ccs_from_nii(seg_dir, min_cc_size=10):
    """Load CCs from raw .nii segmentation files (fallback)."""
    import nibabel as nib
    import glob

    seg_files = sorted(glob.glob(os.path.join(seg_dir, 'segmentation-*.nii*')))
    if not seg_files:
        raise FileNotFoundError(f"No segmentation files in {seg_dir}")

    print(f"Found {len(seg_files)} segmentation files. Extracting CCs...")
    all_ccs = []
    for sf in seg_files:
        seg = nib.load(sf).get_fdata().astype(np.int32)
        ccs = extract_connected_components(seg, min_size=min_cc_size)
        all_ccs.extend(ccs)
        basename = os.path.basename(sf)
        print(f"  {basename}: {len(ccs)} CCs")

    print(f"Total CCs: {len(all_ccs)}")
    return all_ccs


def run_calibration_cv(all_ccs, output_dir, n_folds=5, n_bootstrap=1000):
    """Run full calibration cross-validation for all steepness levels."""
    os.makedirs(output_dir, exist_ok=True)

    log_sizes = np.array([cc['log_size'] for cc in all_ccs])
    mu = float(np.median(log_sizes))
    print(f"μ (median log-CC-size) = {mu:.2f}")

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_ccs': len(all_ccs),
        'mu': mu,
        'n_folds': n_folds,
        'within_steepness': {},
        'cross_steepness': {},
    }

    steepness_levels = ['shallow', 'medium', 'steep']

    # ── Within-steepness CV (R004–R018) ────────────────────────────────
    for steepness in steepness_levels:
        print(f"\n=== {steepness} steepness ===")
        g_true = make_channel_func(steepness, mu)

        cv_result = cross_validate_calibration(
            all_ccs, g_true, n_folds=n_folds,
            rng=np.random.default_rng(42),
        )

        # Bootstrap CI on each fold
        rng = np.random.default_rng(42)
        _, selection_flags = simulate_channel(all_ccs, g_true, rng=rng)
        ir_full, s0_full = fit_isotonic(log_sizes, selection_flags)
        boot = bootstrap_ece_ci(
            log_sizes, selection_flags, ir_full, s0_full,
            n_bootstrap=n_bootstrap, rng=np.random.default_rng(123),
        )

        results['within_steepness'][steepness] = {
            **cv_result,
            'bootstrap_ci': boot,
            'pass': cv_result['mean_ece'] < 0.05,
        }

        status = '✓ PASS' if cv_result['mean_ece'] < 0.05 else '✗ FAIL'
        print(f"  ECE: {cv_result['mean_ece']:.4f} ± {cv_result['std_ece']:.4f} {status}")
        print(f"  Bootstrap 95% CI: [{boot['ece_ci_lo']:.4f}, {boot['ece_ci_hi']:.4f}]")

    # ── Cross-steepness transfer (R019–R024) ───────────────────────────
    # Fit on medium, evaluate on shallow and steep
    g_medium = make_channel_func('medium', mu)
    rng = np.random.default_rng(42)
    _, sel_medium = simulate_channel(all_ccs, g_medium, rng=rng)
    ir_medium, s0_medium = fit_isotonic(log_sizes, sel_medium)

    for target_steepness in ['shallow', 'steep']:
        print(f"\n=== Cross-steepness: medium → {target_steepness} ===")
        g_target = make_channel_func(target_steepness, mu)
        rng_target = np.random.default_rng(99)
        _, sel_target = simulate_channel(all_ccs, g_target, rng=rng_target)

        pred = predict_propensity(ir_medium, log_sizes, s0_medium)
        ece = compute_ece(pred, sel_target)

        results['cross_steepness'][f'medium_to_{target_steepness}'] = {
            'ece': ece,
            'pass': ece < 0.10,
        }

        status = '✓ PASS' if ece < 0.10 else '✗ FAIL'
        print(f"  ECE: {ece:.4f} {status}")

    # ── Summary ────────────────────────────────────────────────────────
    within_pass = all(
        v['pass'] for v in results['within_steepness'].values()
    )
    cross_pass = all(
        v['pass'] for v in results['cross_steepness'].values()
    )
    results['summary'] = {
        'within_steepness_all_pass': within_pass,
        'cross_steepness_all_pass': cross_pass,
        'gate_g1': 'PASS' if within_pass else 'FAIL',
    }

    # Save
    out_path = os.path.join(output_dir, 'calibration_cv_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))

    print(f"\nResults saved to {out_path}")
    print(f"Gate G1: {results['summary']['gate_g1']}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='M1: Cross-validated calibration fidelity'
    )
    parser.add_argument('--preprocessed_dir', default=None,
                        help='nnUNet preprocessed directory')
    parser.add_argument('--seg_dir', default=None,
                        help='Directory with raw segmentation .nii files '
                             '(fallback if no preprocessed data)')
    parser.add_argument('--output_dir', default='results/m1_calibration')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--min_cc_size', type=int, default=10)
    args = parser.parse_args()

    if args.preprocessed_dir:
        all_ccs = load_all_ccs_from_preprocessed(
            args.preprocessed_dir, min_cc_size=args.min_cc_size
        )
    elif args.seg_dir:
        all_ccs = load_all_ccs_from_nii(
            args.seg_dir, min_cc_size=args.min_cc_size
        )
    else:
        parser.error("Provide --preprocessed_dir or --seg_dir")

    run_calibration_cv(all_ccs, args.output_dir, n_folds=args.n_folds)
