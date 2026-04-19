"""Isotonic regression calibration for the annotation channel.

Fits g_θ: log(CC_size) -> selection probability via isotonic regression.
Computes ECE and supports cross-validation.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(log_sizes, selected_flags):
    """Fit isotonic regression on (log_size, selected) pairs.

    Args:
        log_sizes: 1D array of log CC sizes.
        selected_flags: 1D binary array (1=annotated, 0=not).

    Returns:
        ir: fitted IsotonicRegression object.
        s0: minimum log-size in the support (for clamping).
    """
    ir = IsotonicRegression(y_min=0.01, y_max=1.0,
                            increasing=True, out_of_bounds='clip')
    ir.fit(log_sizes, selected_flags)
    s0 = float(log_sizes.min())
    return ir, s0


def predict_propensity(ir, log_sizes, s0):
    """Predict propensity scores, clamping below support minimum.

    Args:
        ir: fitted IsotonicRegression.
        log_sizes: array of log sizes to query.
        s0: minimum support value.

    Returns:
        Array of predicted propensities in [0.01, 1.0].
    """
    log_sizes = np.maximum(np.asarray(log_sizes, dtype=np.float64), s0)
    return ir.predict(log_sizes)


def compute_ece(predicted_probs, true_labels, n_bins=10):
    """Expected Calibration Error.

    Args:
        predicted_probs: predicted probabilities.
        true_labels: binary ground truth.
        n_bins: number of calibration bins.

    Returns:
        float: ECE value.
    """
    predicted_probs = np.asarray(predicted_probs, dtype=np.float64)
    true_labels = np.asarray(true_labels, dtype=np.float64)
    n = len(predicted_probs)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (predicted_probs >= lo) & (predicted_probs < hi)
        if hi == 1.0:
            mask = mask | (predicted_probs == 1.0)
        count = mask.sum()
        if count == 0:
            continue
        avg_pred = predicted_probs[mask].mean()
        avg_true = true_labels[mask].mean()
        ece += (count / n) * abs(avg_pred - avg_true)
    return float(ece)


def cross_validate_calibration(all_ccs, g_true, n_folds=5, rng=None,
                               stratified=False, n_repeats=1,
                               group_by_scan=False):
    """K-fold cross-validation of isotonic calibration.

    Args:
        all_ccs: list of CC dicts from the full dataset.
                 If group_by_scan=True, each dict must have 'scan_id'.
        g_true: channel function to simulate selection.
        n_folds: number of CV folds.
        rng: numpy random generator (used for channel simulation).
        stratified: if True, use stratified K-fold by log-size bins.
                    Ignored when group_by_scan=True (GroupKFold is used).
        n_repeats: number of times to repeat the CV with different splits.
        group_by_scan: if True, use GroupKFold so CCs from the same scan
                       stay in the same fold. This is the correct approach
                       since CCs within a scan are not independent.

    Returns:
        dict with 'per_fold_ece', 'mean_ece', 'std_ece', 'oof_ece'.
        'oof_ece' is the ECE computed on the pooled out-of-fold predictions
        (more stable than mean of per-fold ECEs).
        When n_repeats > 1, also includes 'per_repeat_mean_ece' and
        'per_repeat_oof_ece'.
    """
    from .channel_simulator import simulate_channel

    if rng is None:
        rng = np.random.default_rng(42)

    log_sizes = np.array([cc['log_size'] for cc in all_ccs])
    _, selection_flags = simulate_channel(all_ccs, g_true, rng=rng)

    n = len(all_ccs)

    # Extract scan groups if needed
    if group_by_scan:
        scan_ids = np.array([cc.get('scan_id', 'unknown') for cc in all_ccs])
        unique_scans = np.unique(scan_ids)
        # Map scan_id to integer group label
        scan_to_group = {s: i for i, s in enumerate(unique_scans)}
        groups = np.array([scan_to_group[s] for s in scan_ids])

    all_fold_eces = []
    per_repeat_mean_ece = []
    per_repeat_oof_ece = []

    for rep in range(n_repeats):
        rep_seed = 42 + rep

        if group_by_scan:
            from sklearn.model_selection import GroupKFold
            gkf = GroupKFold(n_splits=min(n_folds, len(unique_scans)))
            fold_iter = list(gkf.split(log_sizes, groups=groups))
        elif stratified:
            from sklearn.model_selection import StratifiedKFold
            n_strat_bins = min(10, n // n_folds)
            bin_edges = np.quantile(log_sizes,
                                    np.linspace(0, 1, n_strat_bins + 1))
            bin_edges[-1] += 1e-6
            size_bins = np.digitize(log_sizes, bin_edges[1:])
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                  random_state=rep_seed)
            fold_iter = list(skf.split(log_sizes, size_bins))
        else:
            rep_rng = np.random.default_rng(rep_seed)
            indices = rep_rng.permutation(n)
            fold_size = n // n_folds
            fold_iter = []
            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = (val_start + fold_size
                           if fold < n_folds - 1 else n)
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate(
                    [indices[:val_start], indices[val_end:]])
                fold_iter.append((train_idx, val_idx))

        # OOF predictions for pooled ECE
        oof_preds = np.full(n, np.nan)
        rep_fold_eces = []

        for train_idx, val_idx in fold_iter:
            ir, s0 = fit_isotonic(log_sizes[train_idx],
                                  selection_flags[train_idx])
            pred = predict_propensity(ir, log_sizes[val_idx], s0)
            ece = compute_ece(pred, selection_flags[val_idx])
            rep_fold_eces.append(ece)
            oof_preds[val_idx] = pred

        # OOF pooled ECE (all out-of-fold predictions combined)
        valid_mask = ~np.isnan(oof_preds)
        oof_ece = compute_ece(oof_preds[valid_mask],
                              selection_flags[valid_mask])

        all_fold_eces.extend(rep_fold_eces)
        per_repeat_mean_ece.append(float(np.mean(rep_fold_eces)))
        per_repeat_oof_ece.append(float(oof_ece))

    result = {
        'per_fold_ece': [float(e) for e in all_fold_eces],
        'mean_ece': float(np.mean(all_fold_eces)),
        'std_ece': float(np.std(all_fold_eces)),
        'oof_ece': float(np.mean(per_repeat_oof_ece)),
    }
    if n_repeats > 1:
        result['per_repeat_mean_ece'] = per_repeat_mean_ece
        result['per_repeat_oof_ece'] = per_repeat_oof_ece
        result['repeat_mean_of_means'] = float(
            np.mean(per_repeat_mean_ece))
        result['repeat_std_of_means'] = float(
            np.std(per_repeat_mean_ece))
        result['repeat_mean_oof_ece'] = float(
            np.mean(per_repeat_oof_ece))
        result['repeat_std_oof_ece'] = float(
            np.std(per_repeat_oof_ece))
    return result


def bootstrap_ece_ci(log_sizes, selection_flags, ir, s0,
                      n_bootstrap=1000, ci=0.95, rng=None):
    """Bootstrap 95% CI on ECE.

    Returns:
        dict with 'ece_mean', 'ece_ci_lo', 'ece_ci_hi'.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(log_sizes)
    eces = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        pred = predict_propensity(ir, log_sizes[idx], s0)
        ece = compute_ece(pred, selection_flags[idx])
        eces.append(ece)

    eces = np.array(eces)
    alpha = (1 - ci) / 2
    return {
        'ece_mean': float(eces.mean()),
        'ece_ci_lo': float(np.quantile(eces, alpha)),
        'ece_ci_hi': float(np.quantile(eces, 1 - alpha)),
    }


# ── v5 Hungarian matching protocol ────────────────────────────────────

def _compute_box_iou(box1, box2):
    """IoU between two 3D bounding boxes: ((z1,z2),(y1,y2),(x1,x2))."""
    inter_vol = 1.0
    for (s1, e1), (s2, e2) in zip(box1, box2):
        inter_s = max(s1, s2)
        inter_e = min(e1, e2)
        if inter_s >= inter_e:
            return 0.0
        inter_vol *= (inter_e - inter_s)

    vol1 = 1.0
    vol2 = 1.0
    for (s, e) in box1:
        vol1 *= max(e - s, 0)
    for (s, e) in box2:
        vol2 *= max(e - s, 0)
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / max(union_vol, 1e-12)


def generate_annotation_pairs(seg_list, fg_label=2, min_cc_size=10,
                               iou_threshold=0.3, drop_fn=None, rng=None):
    """Generate (log_size, annotation_status) from pixel-labeled scans.

    v5 protocol (FINAL_PROPOSAL §4.3):
      1. Extract GT CCs per scan
      2. Generate bounding boxes from GT
      3. Simulate annotation miss via drop_fn (size-dependent)
      4. Hungarian matching: GT CCs <-> kept boxes via IoU
      5. Matched (IoU > threshold) -> annotated (1), else missed (0)

    Args:
        seg_list: list of 3D numpy arrays (GT segmentations)
        fg_label: foreground label for CC extraction
        drop_fn: callable(log_sizes) -> keep_probabilities. If None, all kept.
        rng: numpy random generator

    Returns:
        log_sizes, annotations, scan_indices (all numpy arrays)
    """
    from scipy.optimize import linear_sum_assignment
    from latentmask.utils.cc_extraction import extract_connected_components

    if rng is None:
        rng = np.random.default_rng(42)

    all_log_sizes = []
    all_annotations = []
    all_scan_indices = []
    stats = {'n_ambiguous': 0, 'n_total_ccs': 0}

    for scan_idx, seg in enumerate(seg_list):
        ccs = extract_connected_components(seg, min_size=min_cc_size,
                                           fg_label=fg_label)
        if len(ccs) == 0:
            continue

        stats['n_total_ccs'] += len(ccs)
        all_bboxes = [cc['bbox'] for cc in ccs]

        # Simulate annotation (drop small CCs)
        if drop_fn is not None:
            log_sizes_arr = np.array([cc['log_size'] for cc in ccs])
            keep_probs = drop_fn(log_sizes_arr)
            keep_flags = rng.random(len(ccs)) < keep_probs
            annotation_boxes = [bb for bb, k in zip(all_bboxes, keep_flags) if k]
        else:
            annotation_boxes = list(all_bboxes)

        if len(annotation_boxes) == 0:
            for cc in ccs:
                all_log_sizes.append(cc['log_size'])
                all_annotations.append(0)
                all_scan_indices.append(scan_idx)
            continue

        # Hungarian matching
        n_gt = len(ccs)
        n_box = len(annotation_boxes)
        iou_matrix = np.zeros((n_gt, n_box), dtype=np.float64)
        for k, cc in enumerate(ccs):
            for j, abox in enumerate(annotation_boxes):
                iou_matrix[k, j] = _compute_box_iou(cc['bbox'], abox)

        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = set()
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                matched.add(r)

        # Count ambiguous
        for k in range(n_gt):
            n_high = (iou_matrix[k, :] >= iou_threshold).sum()
            if n_high > 1:
                stats['n_ambiguous'] += 1

        for k, cc in enumerate(ccs):
            all_log_sizes.append(cc['log_size'])
            all_annotations.append(1 if k in matched else 0)
            all_scan_indices.append(scan_idx)

    log_sizes = np.array(all_log_sizes, dtype=np.float64)
    annotations = np.array(all_annotations, dtype=np.float64)
    scan_indices = np.array(all_scan_indices, dtype=np.int64)
    stats['unmatched_rate'] = float(1.0 - annotations.mean()) if len(annotations) > 0 else 0.0
    stats['ambiguous_rate'] = stats['n_ambiguous'] / max(stats['n_total_ccs'], 1)
    stats['n_pairs'] = len(annotations)
    return log_sizes, annotations, scan_indices, stats


def fit_g_theta_hungarian(seg_list, fg_label=2, min_cc_size=10,
                           iou_threshold=0.3, drop_fn=None, rng=None):
    """v5 full protocol: generate pairs + fit isotonic + fit linear.

    Returns:
        ir: fitted IsotonicRegression
        s0: minimum support
        linear_a: linear intercept
        linear_b: linear slope
        stats: dict with fitting statistics
    """
    log_sizes, annotations, scan_indices, stats = generate_annotation_pairs(
        seg_list, fg_label, min_cc_size, iou_threshold, drop_fn, rng)

    if len(log_sizes) == 0:
        raise ValueError("No CCs found in any scan — cannot fit g_theta")

    # Isotonic fit
    ir = IsotonicRegression(y_min=0.01, y_max=0.99,
                            increasing=True, out_of_bounds='clip')
    ir.fit(log_sizes, annotations)
    s0 = float(log_sizes.min())

    # Linear fit (OLS: annotation_status ~ a + b * log_size)
    coeffs = np.polyfit(log_sizes, annotations, 1)  # [slope, intercept]
    linear_b = float(coeffs[0])
    linear_a = float(coeffs[1])

    stats['s0'] = s0
    stats['linear_a'] = linear_a
    stats['linear_b'] = linear_b
    stats['n_annotated'] = int(annotations.sum())
    stats['n_missed'] = int((1 - annotations).sum())
    return ir, s0, linear_a, linear_b, stats


def cross_validate_g_theta_hungarian(seg_list, fg_label=2, n_folds=3,
                                      min_cc_size=10, iou_threshold=0.3,
                                      drop_fn=None, rng=None):
    """3-fold CV of g_theta fitting with scan-level splits.

    Returns dict with per_fold_ece, mean_ece, std_ece.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Generate ALL pairs once (consistent simulation)
    log_sizes, annotations, scan_indices, stats = generate_annotation_pairs(
        seg_list, fg_label, min_cc_size, iou_threshold, drop_fn, rng)

    if len(log_sizes) == 0:
        return {'per_fold_ece': [], 'mean_ece': float('inf'), 'std_ece': 0.0}

    # Scan-level fold split
    unique_scans = np.unique(scan_indices)
    n_scans = len(unique_scans)
    scan_perm = rng.permutation(n_scans)
    fold_size = n_scans // n_folds

    fold_eces = []
    for fold in range(n_folds):
        vs = fold * fold_size
        ve = vs + fold_size if fold < n_folds - 1 else n_scans
        val_scan_set = set(unique_scans[scan_perm[vs:ve]])

        train_mask = np.array([s not in val_scan_set for s in scan_indices])
        val_mask = ~train_mask

        if not train_mask.any() or not val_mask.any():
            continue

        ir, s0 = fit_isotonic(log_sizes[train_mask], annotations[train_mask])
        pred = predict_propensity(ir, log_sizes[val_mask], s0)
        ece = compute_ece(pred, annotations[val_mask])
        fold_eces.append(float(ece))

    return {
        'per_fold_ece': fold_eces,
        'mean_ece': float(np.mean(fold_eces)) if fold_eces else float('inf'),
        'std_ece': float(np.std(fold_eces)) if fold_eces else 0.0,
        'pairing_stats': stats,
    }
