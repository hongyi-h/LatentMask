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
