"""Size-stratified evaluation metrics for box-supervised segmentation."""
import json
import numpy as np
from scipy import ndimage


def compute_dice(pred, gt, eps=1e-7):
    """Dice coefficient between two binary masks."""
    intersection = (pred & gt).sum()
    return float(2 * intersection / (pred.sum() + gt.sum() + eps))


def compute_size_stratified_metrics(pred, gt, n_quintiles=5):
    """Compute per-CC recall stratified by object size.

    Args:
        pred: binary prediction mask (3D numpy array).
        gt: binary ground truth mask (3D numpy array).
        n_quintiles: number of size groups.

    Returns:
        dict with 'overall_dice', 'per_quintile' (list of dicts),
        'quintile_boundaries'.
    """
    labeled_gt, num_gt = ndimage.label(gt > 0)
    if num_gt == 0:
        return {'overall_dice': compute_dice(pred > 0, gt > 0),
                'per_quintile': [], 'quintile_boundaries': []}

    # Collect CC sizes
    cc_sizes = []
    cc_labels = []
    for i in range(1, num_gt + 1):
        size = int((labeled_gt == i).sum())
        cc_sizes.append(size)
        cc_labels.append(i)

    cc_sizes = np.array(cc_sizes)
    cc_labels = np.array(cc_labels)
    sort_idx = np.argsort(cc_sizes)
    cc_sizes = cc_sizes[sort_idx]
    cc_labels = cc_labels[sort_idx]

    # Split into quintiles
    boundaries = np.quantile(cc_sizes, np.linspace(0, 1, n_quintiles + 1))
    per_quintile = []
    for q in range(n_quintiles):
        lo = boundaries[q]
        hi = boundaries[q + 1] if q < n_quintiles - 1 else cc_sizes.max() + 1
        mask = (cc_sizes >= lo) & (cc_sizes < hi) if q < n_quintiles - 1 \
            else (cc_sizes >= lo)
        q_labels = cc_labels[mask]
        if len(q_labels) == 0:
            per_quintile.append({'quintile': f'Q{q+1}', 'n_ccs': 0,
                                 'recall': 0.0, 'mean_size': 0.0})
            continue

        # Per-CC recall: fraction of GT CC voxels recovered
        recalls = []
        for lab in q_labels:
            gt_cc = labeled_gt == lab
            rec = float((pred[gt_cc] > 0).sum() / gt_cc.sum())
            recalls.append(rec)

        per_quintile.append({
            'quintile': f'Q{q+1}',
            'n_ccs': int(len(q_labels)),
            'recall': float(np.mean(recalls)),
            'mean_size': float(cc_sizes[mask].mean()),
        })

    return {
        'overall_dice': compute_dice(pred > 0, gt > 0),
        'per_quintile': per_quintile,
        'quintile_boundaries': boundaries.tolist(),
    }


def compute_delta_area(pred_masses, true_sizes, g_theta_func):
    """Compute Δ_area: mean |g_θ(log(mass)) - g_θ(log(true_size))|.

    Args:
        pred_masses: array of predicted soft masses per box.
        true_sizes: array of true CC sizes per box.
        g_theta_func: callable, isotonic calibrator.

    Returns:
        float: mean absolute gap.
    """
    log_mass = np.log(np.maximum(pred_masses, 1))
    log_true = np.log(np.maximum(true_sizes, 1))
    gap = np.abs(g_theta_func(log_mass) - g_theta_func(log_true))
    return float(gap.mean())


def save_results(results, path):
    """Save results dict to JSON."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
