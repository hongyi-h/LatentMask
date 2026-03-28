"""
Evaluation metrics for LatentMask.

Includes:
  - Dice coefficient
  - Hausdorff distance 95th percentile
  - Lesion-level F1 (connected component matching)
  - Small-lesion recall
  - False positives per scan
  - Propensity calibration ECE
  - Bootstrap significance test
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff


# ─── Voxel-level metrics ──────────────────────────────────────────────

def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice coefficient."""
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    if pred.sum() + gt.sum() == 0:
        return 1.0
    return 2.0 * intersection / (pred.sum() + gt.sum())


def hausdorff_95(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1, 1, 1)) -> float:
    """Hausdorff distance at 95th percentile."""
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf

    pred_coords = np.argwhere(pred) * np.array(spacing)
    gt_coords = np.argwhere(gt) * np.array(spacing)

    # Forward distances (pred → gt)
    from scipy.spatial import cKDTree
    tree_gt = cKDTree(gt_coords)
    dists_pred, _ = tree_gt.query(pred_coords)

    tree_pred = cKDTree(pred_coords)
    dists_gt, _ = tree_pred.query(gt_coords)

    return max(np.percentile(dists_pred, 95), np.percentile(dists_gt, 95))


# ─── Lesion-level metrics ─────────────────────────────────────────────

def lesion_level_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_threshold: float = 0.1,
    small_vol_threshold: int = 100,
) -> dict:
    """
    Compute lesion-level TP, FP, FN, F1, and small-lesion recall.

    A predicted lesion is TP if it overlaps any GT lesion with IoU ≥ threshold.
    A GT lesion is detected if any predicted lesion overlaps it with IoU ≥ threshold.

    Args:
        pred: Binary prediction mask.
        gt: Binary ground truth mask.
        iou_threshold: Minimum IoU for a match.
        small_vol_threshold: Max voxels for a "small" lesion.

    Returns:
        Dict with keys: tp, fp, fn, f1, recall, precision,
        recall_small, num_gt, num_pred, num_gt_small
    """
    pred_labels, num_pred = ndimage.label(pred > 0)
    gt_labels, num_gt = ndimage.label(gt > 0)

    # GT lesion sizes
    gt_sizes = ndimage.sum(gt > 0, gt_labels, range(1, num_gt + 1))

    # Match predicted → GT
    pred_matched = set()
    gt_matched = set()

    for p_id in range(1, num_pred + 1):
        pred_mask = pred_labels == p_id
        for g_id in range(1, num_gt + 1):
            gt_mask = gt_labels == g_id
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            if union > 0 and intersection / union >= iou_threshold:
                pred_matched.add(p_id)
                gt_matched.add(g_id)

    tp = len(pred_matched)
    fp = num_pred - tp
    fn = num_gt - len(gt_matched)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Small-lesion recall
    num_gt_small = 0
    gt_small_matched = 0
    for g_id in range(1, num_gt + 1):
        if gt_sizes[g_id - 1] <= small_vol_threshold:
            num_gt_small += 1
            if g_id in gt_matched:
                gt_small_matched += 1

    recall_small = gt_small_matched / max(num_gt_small, 1)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "recall_small": recall_small,
        "num_gt": num_gt,
        "num_pred": num_pred,
        "num_gt_small": num_gt_small,
        "fp_per_scan": fp,
    }


# ─── Propensity calibration ───────────────────────────────────────────

def propensity_calibration_ece(
    predicted_propensity: np.ndarray,
    true_propensity: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error for propensity estimation.

    Bins predicted propensity and measures mean absolute error within each bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = predicted_propensity.size

    for i in range(n_bins):
        mask = (predicted_propensity >= bins[i]) & (predicted_propensity < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_pred = predicted_propensity[mask].mean()
        bin_true = true_propensity[mask].mean()
        ece += mask.sum() / total * abs(bin_pred - bin_true)

    return ece


def propensity_by_branch_level(
    predicted_propensity: np.ndarray,
    vesselness: np.ndarray,
    thresholds: tuple = (0.6, 0.3, 0.1),
) -> dict:
    """
    Report mean propensity stratified by vessel branch level.

    Uses vesselness as proxy for branch level:
        vesselness ≥ 0.6 → proximal
        0.3 ≤ vesselness < 0.6 → segmental
        0.1 ≤ vesselness < 0.3 → subsegmental
        vesselness < 0.1 → background
    """
    t_prox, t_seg, t_sub = thresholds
    levels = {
        "proximal": vesselness >= t_prox,
        "segmental": (vesselness >= t_seg) & (vesselness < t_prox),
        "subsegmental": (vesselness >= t_sub) & (vesselness < t_seg),
        "background": vesselness < t_sub,
    }

    results = {}
    for name, mask in levels.items():
        if mask.sum() > 0:
            results[name] = {
                "mean_propensity": float(predicted_propensity[mask].mean()),
                "std_propensity": float(predicted_propensity[mask].std()),
                "n_voxels": int(mask.sum()),
            }
        else:
            results[name] = {"mean_propensity": 0.0, "std_propensity": 0.0, "n_voxels": 0}

    return results


# ─── Statistical tests ────────────────────────────────────────────────

def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap significance test.

    Tests H0: mean(scores_a) ≤ mean(scores_b).

    Args:
        scores_a: Per-case metric for method A (expected to be better).
        scores_b: Per-case metric for method B.
        n_resamples: Number of bootstrap iterations.

    Returns:
        Dict with p_value, mean_diff, ci_lower, ci_upper.
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = len(a)
    assert len(b) == n

    diffs = a - b
    observed_diff = diffs.mean()

    boot_diffs = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        boot_diffs.append(diffs[idx].mean())

    boot_diffs = np.array(boot_diffs)
    p_value = (boot_diffs <= 0).mean()

    return {
        "p_value": float(p_value),
        "mean_diff": float(observed_diff),
        "ci_lower": float(np.percentile(boot_diffs, 2.5)),
        "ci_upper": float(np.percentile(boot_diffs, 97.5)),
    }


# ─── Aggregate evaluation ─────────────────────────────────────────────

def evaluate_case(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple = (1, 1, 1),
    small_vol_threshold: int = 100,
) -> dict:
    """Run all metrics on a single case. Returns a flat dict."""
    result = {}
    result["dice"] = dice_coefficient(pred, gt)
    result["hd95"] = hausdorff_95(pred, gt, spacing)

    lesion = lesion_level_metrics(pred, gt, small_vol_threshold=small_vol_threshold)
    result.update({f"lesion_{k}": v for k, v in lesion.items()})

    return result


def aggregate_results(case_results: list[dict]) -> dict:
    """Aggregate per-case results into mean ± std."""
    if not case_results:
        return {}

    keys = case_results[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in case_results if np.isfinite(r[k])]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        else:
            agg[f"{k}_mean"] = float("nan")
            agg[f"{k}_std"] = float("nan")

    return agg
