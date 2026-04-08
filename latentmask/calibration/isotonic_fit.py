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


def cross_validate_calibration(all_ccs, g_true, n_folds=5, rng=None):
    """5-fold cross-validation of isotonic calibration.

    Args:
        all_ccs: list of CC dicts from the full dataset.
        g_true: channel function to simulate selection.
        n_folds: number of CV folds.
        rng: numpy random generator.

    Returns:
        dict with 'per_fold_ece', 'mean_ece', 'std_ece'.
    """
    from .channel_simulator import simulate_channel

    if rng is None:
        rng = np.random.default_rng(42)

    log_sizes = np.array([cc['log_size'] for cc in all_ccs])
    _, selection_flags = simulate_channel(all_ccs, g_true, rng=rng)

    n = len(all_ccs)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    per_fold_ece = []
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        ir, s0 = fit_isotonic(log_sizes[train_idx], selection_flags[train_idx])
        pred = predict_propensity(ir, log_sizes[val_idx], s0)
        ece = compute_ece(pred, selection_flags[val_idx])
        per_fold_ece.append(ece)

    return {
        'per_fold_ece': per_fold_ece,
        'mean_ece': float(np.mean(per_fold_ece)),
        'std_ece': float(np.std(per_fold_ece)),
    }


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
