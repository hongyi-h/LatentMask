"""Offline box annotation generation with retention-rate matching.

Generates synthetic incomplete-box annotations from GT masks for each
missingness protocol (P-uniform, P-mild, P-steep), matched to the same
expected marginal retention rate R.

Reads GT segmentations directly from nnUNet preprocessed gt_segmentations/.
Splits pixel/box keys from splits_final.json automatically.

Output: one JSON per scan per protocol in {output_dir}/{protocol}/
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from latentmask.utils.cc_extraction import extract_connected_components
from latentmask.calibration.channel_simulator import make_channel_func


def compute_retention_scale(steepness, mu, cc_sizes, target_R):
    """Find scalar c so E[min(c * g(log s), 1.0)] = target_R."""
    g_func = make_channel_func(steepness, mu)
    log_sizes = np.log(np.maximum(cc_sizes, 1))
    raw_probs = g_func(log_sizes)

    lo, hi = 0.01, 10.0
    for _ in range(100):
        c = (lo + hi) / 2
        mean_keep = np.minimum(c * raw_probs, 1.0).mean()
        if mean_keep < target_R:
            lo = c
        else:
            hi = c
        if abs(mean_keep - target_R) < 1e-6:
            break
    return c


def load_seg(gt_dir, key):
    """Load GT segmentation from gt_segmentations/ (.npy or .nii.gz)."""
    npy_path = os.path.join(gt_dir, f'{key}.npy')
    if os.path.exists(npy_path):
        seg = np.load(npy_path)
        if seg.ndim == 4:
            seg = seg[0]
        return seg

    nii_path = os.path.join(gt_dir, f'{key}.nii.gz')
    if os.path.exists(nii_path):
        import nibabel as nib
        seg = nib.load(nii_path).get_fdata(dtype=np.float32)
        return seg.astype(np.int32)

    raise FileNotFoundError(
        f'No segmentation found for {key} in {gt_dir} '
        f'(tried .npy and .nii.gz)'
    )


def generate_boxes_for_scan(seg, protocol, mu, scale_factor, target_R,
                            min_cc_size=10, fg_label=2, rng=None):
    """Generate box annotations and box_seg volume for one scan.

    Returns dict with 'boxes', 'box_seg', and stats.
    box_seg: same shape as seg, each retained box's bounding box region
    filled with a unique integer ID (1, 2, ...). Background = 0.
    """
    if rng is None:
        rng = np.random.default_rng()

    ccs = extract_connected_components(seg, min_size=min_cc_size,
                                       fg_label=fg_label)
    if len(ccs) == 0:
        return {'boxes': [], 'box_seg': np.zeros_like(seg, dtype=np.int16),
                'n_total': 0, 'n_retained': 0, 'n_dropped': 0}

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
    box_seg = np.zeros(seg.shape, dtype=np.int16)
    box_id = 1
    for i, cc in enumerate(ccs):
        if keep_flags[i]:
            bbox = cc['bbox']
            boxes.append({'bbox': [list(pair) for pair in bbox]})
            (z1, z2), (y1, y2), (x1, x2) = bbox
            box_seg[z1:z2, y1:y2, x1:x2] = box_id
            box_id += 1

    return {
        'boxes': boxes,
        'box_seg': box_seg,
        'n_total': len(ccs),
        'n_retained': int(keep_flags.sum()),
        'n_dropped': int((~keep_flags).sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate offline box annotations with retention-rate matching')
    parser.add_argument('--dataset_dir', required=True,
                        help='Path to nnUNet preprocessed dataset root '
                             '(e.g. data_preprocessed/Dataset501_LiTS)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for box annotations')
    parser.add_argument('--pixel_fraction', type=float, default=0.3,
                        help='Fraction of training scans used as pixel-labeled (default: 0.3)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Which fold from splits_final.json to use for pixel/box split')
    parser.add_argument('--target_R', type=float, default=0.70,
                        help='Target marginal retention rate (default: 0.70)')
    parser.add_argument('--fg_label', type=int, default=2,
                        help='Foreground label (default: 2 for LiTS tumor)')
    parser.add_argument('--min_cc_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    gt_dir = os.path.join(args.dataset_dir, 'gt_segmentations')
    splits_path = os.path.join(args.dataset_dir, 'splits_final.json')

    assert os.path.isdir(gt_dir), f"gt_segmentations not found: {gt_dir}"
    assert os.path.isfile(splits_path), f"splits_final.json not found: {splits_path}"

    # Load splits
    with open(splits_path) as f:
        splits = json.load(f)

    fold_split = splits[args.fold]
    train_keys = sorted(fold_split['train'])
    val_keys = sorted(fold_split['val'])

    # Split train into pixel / box (same logic as trainer)
    rng_split = np.random.default_rng(seed=12345 + args.fold)
    perm = rng_split.permutation(len(train_keys)).tolist()
    n_pixel = max(1, int(args.pixel_fraction * len(train_keys)))
    pixel_keys = [train_keys[i] for i in perm[:n_pixel]]
    box_keys = [train_keys[i] for i in perm[n_pixel:]]

    # We generate boxes for ALL scans (pixel + box + val) so calibration
    # and evaluation can use them. Training only uses box_keys' annotations.
    all_keys = train_keys + val_keys

    print(f"Fold {args.fold}: {len(train_keys)} train "
          f"({len(pixel_keys)} pixel, {len(box_keys)} box), "
          f"{len(val_keys)} val")
    print(f"Generating boxes for all {len(all_keys)} scans")
    print(f"Target retention rate: {args.target_R}")

    # Save key lists for reference
    os.makedirs(args.output_dir, exist_ok=True)
    meta = {
        'fold': args.fold,
        'pixel_fraction': args.pixel_fraction,
        'pixel_keys': pixel_keys,
        'box_keys': box_keys,
        'val_keys': val_keys,
        'target_R': args.target_R,
        'seed': args.seed,
    }
    with open(os.path.join(args.output_dir, '_split_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    rng = np.random.default_rng(args.seed)

    # Collect CC sizes from ALL scans for protocol-level retention-rate matching.
    # Rationale: the scale factor defines the PROTOCOL (benchmark drop rule),
    # not the METHOD. Using only pixel scans (31/131) gives a biased size
    # distribution, causing actual R to drift from target (observed: P-steep
    # 0.640 vs target 0.70). The method's g_θ is still estimated only from
    # pixel scans — that boundary is unchanged.
    all_cc_sizes = []
    for key in all_keys:
        seg = load_seg(gt_dir, key)
        ccs = extract_connected_components(seg, min_size=args.min_cc_size,
                                           fg_label=args.fg_label)
        for cc in ccs:
            all_cc_sizes.append(cc['size'])

    if len(all_cc_sizes) == 0:
        print("ERROR: no CCs found. Check fg_label.")
        sys.exit(1)

    all_cc_sizes = np.array(all_cc_sizes)
    mu = float(np.median(np.log(np.maximum(all_cc_sizes, 1))))
    print(f"Median log-CC-size (mu): {mu:.2f}, total CCs: {len(all_cc_sizes)}")

    # Compute scale factors
    protocols = {
        'P-uniform': None,
        'P-mild': compute_retention_scale('shallow', mu, all_cc_sizes, args.target_R),
        'P-steep': compute_retention_scale('steep', mu, all_cc_sizes, args.target_R),
    }

    for name, scale in protocols.items():
        if scale is not None:
            print(f"  {name}: scale_factor = {scale:.4f}")
        else:
            print(f"  {name}: constant keep_prob = {args.target_R:.2f}")

    # Generate box annotations and box_seg volumes
    for protocol_name, scale_factor in protocols.items():
        protocol_dir = os.path.join(args.output_dir, protocol_name)
        box_seg_dir = os.path.join(protocol_dir, 'box_segmentations')
        os.makedirs(protocol_dir, exist_ok=True)
        os.makedirs(box_seg_dir, exist_ok=True)

        stats = {'n_scans': 0, 'total_ccs': 0, 'total_retained': 0}

        for key in all_keys:
            seg = load_seg(gt_dir, key)

            result = generate_boxes_for_scan(
                seg, protocol_name, mu, scale_factor, args.target_R,
                min_cc_size=args.min_cc_size, fg_label=args.fg_label,
                rng=rng,
            )

            # Save box_seg volume (same shape as GT seg, int16)
            box_seg = result.pop('box_seg')
            np.save(os.path.join(box_seg_dir, f'{key}.npy'),
                    box_seg[np.newaxis].astype(np.int16))

            # Save JSON metadata (without box_seg)
            output = {
                'scan_id': key,
                'protocol': protocol_name,
                'target_R': args.target_R,
                'mu': mu,
                **result,
            }
            with open(os.path.join(protocol_dir, f'{key}.json'), 'w') as f:
                json.dump(output, f, indent=2)

            stats['n_scans'] += 1
            stats['total_ccs'] += result['n_total']
            stats['total_retained'] += result['n_retained']

        actual_R = stats['total_retained'] / max(stats['total_ccs'], 1)
        print(f"  {protocol_name}: {stats['n_scans']} scans, "
              f"actual R = {actual_R:.3f} (target {args.target_R:.2f})")

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
