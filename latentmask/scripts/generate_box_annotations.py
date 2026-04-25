"""Offline box annotation generation with retention-rate matching.

Generates synthetic incomplete-box annotations from GT masks for each
missingness protocol (P-uniform, P-mild, P-steep), matched to the same
expected marginal retention rate R.

Output: one JSON per scan per protocol in data/box_annotations/{protocol}/
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from latentmask.utils.cc_extraction import extract_connected_components
from latentmask.calibration.channel_simulator import make_channel_func


def compute_retention_scale(steepness, mu, cc_sizes, target_R):
    """Find a scalar multiplier so E[keep] over the CC size distribution = R.

    For size-dependent protocols, the raw keep probability is
        p_raw(s) = g(log s; steepness, mu)
    We find scale factor c such that
        E[min(c * p_raw(s), 1.0)] = target_R
    over the empirical CC size distribution.

    For P-uniform, returns (1 - target_R) as the constant drop probability.
    """
    if steepness == 'uniform':
        return None  # handled separately

    g_func = make_channel_func(steepness, mu)
    log_sizes = np.log(np.maximum(cc_sizes, 1))
    raw_probs = g_func(log_sizes)

    # Binary search for scale factor c
    lo, hi = 0.01, 10.0
    for _ in range(100):
        c = (lo + hi) / 2
        scaled = np.minimum(c * raw_probs, 1.0)
        mean_keep = scaled.mean()
        if mean_keep < target_R:
            lo = c
        else:
            hi = c
        if abs(mean_keep - target_R) < 1e-6:
            break

    return c


def generate_boxes_for_scan(seg, protocol, mu, scale_factor, target_R,
                            min_cc_size=10, fg_label=2, rng=None):
    """Generate box annotations for one scan under a given protocol."""
    if rng is None:
        rng = np.random.default_rng()

    ccs = extract_connected_components(seg, min_size=min_cc_size,
                                       fg_label=fg_label)
    if len(ccs) == 0:
        return {'boxes': [], 'n_total': 0, 'n_retained': 0, 'n_dropped': 0}

    log_sizes = np.array([cc['log_size'] for cc in ccs])

    if protocol == 'P-uniform':
        keep_probs = np.full(len(ccs), target_R)
    else:
        steepness = {'P-mild': 'shallow', 'P-steep': 'steep'}[protocol]
        g_func = make_channel_func(steepness, mu)
        raw_probs = g_func(log_sizes)
        keep_probs = np.minimum(scale_factor * raw_probs, 1.0)

    draws = rng.random(len(ccs))
    keep_flags = draws < keep_probs

    boxes = []
    for i, cc in enumerate(ccs):
        if keep_flags[i]:
            boxes.append({
                'bbox': [list(pair) for pair in cc['bbox']],
            })

    return {
        'boxes': boxes,
        'n_total': len(ccs),
        'n_retained': int(keep_flags.sum()),
        'n_dropped': int((~keep_flags).sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate offline box annotations with retention-rate matching')
    parser.add_argument('--preprocessed_dir', required=True,
                        help='Path to nnUNet preprocessed dataset folder')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for box annotations')
    parser.add_argument('--scan_keys_file', required=True,
                        help='JSON file listing box-only scan keys')
    parser.add_argument('--calib_keys_file', required=True,
                        help='JSON file listing pixel-labeled scan keys (for mu/scale)')
    parser.add_argument('--target_R', type=float, default=0.70,
                        help='Target marginal retention rate (default: 0.70)')
    parser.add_argument('--fg_label', type=int, default=2,
                        help='Foreground label (default: 2 for LiTS tumor)')
    parser.add_argument('--min_cc_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load calibration scans to compute mu and scale factors
    with open(args.calib_keys_file) as f:
        calib_keys = json.load(f)
    with open(args.scan_keys_file) as f:
        box_keys = json.load(f)

    print(f"Calibration scans: {len(calib_keys)}, Box scans: {len(box_keys)}")
    print(f"Target retention rate: {args.target_R}")

    # Collect all CC sizes from calibration scans for mu and scale computation
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    dataset_class = infer_dataset_class(args.preprocessed_dir)
    ds = dataset_class(args.preprocessed_dir, calib_keys)

    all_cc_sizes = []
    for key in calib_keys:
        _, seg, _, _ = ds.load_case(key)
        seg_np = seg[0]
        ccs = extract_connected_components(seg_np, min_size=args.min_cc_size,
                                           fg_label=args.fg_label)
        for cc in ccs:
            all_cc_sizes.append(cc['size'])

    all_cc_sizes = np.array(all_cc_sizes)
    mu = float(np.median(np.log(np.maximum(all_cc_sizes, 1))))
    print(f"Median log-CC-size (mu): {mu:.2f}, total CCs: {len(all_cc_sizes)}")

    # Compute scale factors for each protocol
    protocols = {
        'P-uniform': None,
        'P-mild': compute_retention_scale('shallow', mu, all_cc_sizes, args.target_R),
        'P-steep': compute_retention_scale('steep', mu, all_cc_sizes, args.target_R),
    }

    for name, scale in protocols.items():
        if scale is not None:
            print(f"  {name}: scale_factor = {scale:.4f}")
        else:
            print(f"  {name}: constant drop_prob = {1 - args.target_R:.2f}")

    # Generate box annotations for each protocol and scan
    ds_all = dataset_class(args.preprocessed_dir, box_keys + calib_keys)

    for protocol_name, scale_factor in protocols.items():
        protocol_dir = os.path.join(args.output_dir, protocol_name)
        os.makedirs(protocol_dir, exist_ok=True)

        stats = {'n_scans': 0, 'total_ccs': 0, 'total_retained': 0}

        for key in box_keys + calib_keys:
            _, seg, _, _ = ds_all.load_case(key)
            seg_np = seg[0]

            result = generate_boxes_for_scan(
                seg_np, protocol_name, mu, scale_factor, args.target_R,
                min_cc_size=args.min_cc_size, fg_label=args.fg_label,
                rng=rng,
            )

            output = {
                'scan_id': key,
                'protocol': protocol_name,
                'target_R': args.target_R,
                'mu': mu,
                **result,
            }

            out_path = os.path.join(protocol_dir, f'{key}.json')
            with open(out_path, 'w') as f:
                json.dump(output, f, indent=2)

            stats['n_scans'] += 1
            stats['total_ccs'] += result['n_total']
            stats['total_retained'] += result['n_retained']

        actual_R = stats['total_retained'] / max(stats['total_ccs'], 1)
        print(f"  {protocol_name}: {stats['n_scans']} scans, "
              f"actual R = {actual_R:.3f} (target {args.target_R:.2f})")

        # Save protocol summary
        summary = {
            'protocol': protocol_name,
            'target_R': args.target_R,
            'actual_R': actual_R,
            'mu': mu,
            'scale_factor': scale_factor,
            'seed': args.seed,
            **stats,
        }
        with open(os.path.join(protocol_dir, '_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    print("Done.")


if __name__ == '__main__':
    main()
