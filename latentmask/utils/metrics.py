"""Size-stratified evaluation metrics for box-supervised segmentation."""
import json
import numpy as np
from scipy import ndimage


def compute_dice(pred, gt, eps=1e-7):
    """Dice coefficient between two binary masks."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
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


# ── v5 additions ───────────────────────────────────────────────────────

def compute_hd95(pred, gt, voxel_spacing=None):
    """Hausdorff Distance 95th percentile between two binary masks."""
    from scipy.ndimage import distance_transform_edt, binary_erosion
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    if not pred.any() or not gt.any():
        return float('inf')

    # Extract surface voxels
    pred_border = pred ^ binary_erosion(pred)
    gt_border = gt ^ binary_erosion(gt)

    if not pred_border.any() or not gt_border.any():
        return float('inf')

    # Distance from pred surface to nearest GT surface
    dt_gt = distance_transform_edt(~gt_border, sampling=voxel_spacing)
    dist_pred_to_gt = dt_gt[pred_border]

    # Distance from GT surface to nearest pred surface
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)
    dist_gt_to_pred = dt_pred[gt_border]

    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def compute_per_lesion_metrics(pred, gt):
    """Compute per-GT-lesion Dice.

    For each GT lesion, find the best-overlapping predicted CC and
    compute their Dice coefficient.

    Returns list of dicts with size, log_size, dice per GT lesion.
    """
    pred_bin = (np.asarray(pred) > 0).astype(np.int32)
    gt_bin = (np.asarray(gt) > 0).astype(np.int32)
    labeled_gt, num_gt = ndimage.label(gt_bin)
    labeled_pred, num_pred = ndimage.label(pred_bin)

    lesions = []
    for i in range(1, num_gt + 1):
        gt_cc = labeled_gt == i
        size = int(gt_cc.sum())

        # Find overlapping predicted CCs
        pred_labels_in_gt = labeled_pred[gt_cc]
        unique_preds = np.unique(pred_labels_in_gt)
        unique_preds = unique_preds[unique_preds > 0]

        best_dice = 0.0
        for pl in unique_preds:
            pred_cc = labeled_pred == pl
            best_dice = max(best_dice, compute_dice(pred_cc, gt_cc))

        lesions.append({
            'size': size,
            'log_size': float(np.log(max(size, 1))),
            'dice': best_dice,
            'detected': best_dice > 0.1,
        })
    return lesions


def aggregate_lesion_metrics_by_quintile(all_lesions, n_quintiles=5):
    """Group per-lesion metrics into quintile summaries.

    Returns list of dicts per quintile with n, mean_dice, mean_size.
    """
    if not all_lesions:
        return []

    sizes = np.array([l['size'] for l in all_lesions])
    sort_idx = np.argsort(sizes)
    sorted_lesions = [all_lesions[i] for i in sort_idx]
    sorted_sizes = sizes[sort_idx]

    boundaries = np.quantile(sorted_sizes, np.linspace(0, 1, n_quintiles + 1))

    results = []
    for q in range(n_quintiles):
        lo = boundaries[q]
        hi = boundaries[q + 1]
        if q < n_quintiles - 1:
            mask = (sorted_sizes >= lo) & (sorted_sizes < hi)
        else:
            mask = sorted_sizes >= lo
        q_lesions = [sorted_lesions[j] for j in range(len(sorted_lesions)) if mask[j]]

        if not q_lesions:
            results.append({'quintile': f'Q{q+1}', 'n': 0,
                            'mean_dice': None, 'mean_size': None})
            continue

        results.append({
            'quintile': f'Q{q+1}',
            'n': len(q_lesions),
            'mean_dice': float(np.mean([l['dice'] for l in q_lesions])),
            'detection_rate': float(np.mean([l['detected'] for l in q_lesions])),
            'mean_size': float(np.mean([l['size'] for l in q_lesions])),
        })
    return results
