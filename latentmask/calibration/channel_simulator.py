"""Annotation channel simulation for box-supervised segmentation.

Three synthetic channels (shallow/medium/steep) model size-dependent
selection probability: larger CCs are more likely to be annotated.
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def make_channel_func(steepness, mu):
    """Create a channel function g_true(log_size) -> selection probability.

    Args:
        steepness: 'shallow', 'medium', or 'steep'.
        mu: center point (typically median log-CC-size in calibration set).

    Returns:
        callable: g_true(log_sizes) -> probabilities.
    """
    params = {
        'shallow': (0.5, 0.5, 2.0),  # g_min=0.5, range=0.5, slope=2
        'medium':  (0.3, 0.7, 3.0),  # g_min=0.3, range=0.7, slope=3
        'steep':   (0.1, 0.9, 5.0),  # g_min=0.1, range=0.9, slope=5
    }
    g_min, g_range, slope = params[steepness]

    def g_true(log_sizes):
        log_sizes = np.asarray(log_sizes, dtype=np.float64)
        return g_min + g_range * _sigmoid(slope * (log_sizes - mu))

    return g_true


def simulate_channel(ccs, g_true, rng=None):
    """Apply channel model to decide which CCs are annotated.

    Args:
        ccs: list of CC dicts from extract_connected_components.
        g_true: channel function g_true(log_size) -> probability.
        rng: numpy random generator for reproducibility.

    Returns:
        selected_indices: list of indices into ccs that were selected.
        selection_flags: binary array (1=selected, 0=not).
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(ccs) == 0:
        return [], np.array([], dtype=np.float64)

    log_sizes = np.array([cc['log_size'] for cc in ccs])
    probs = g_true(log_sizes)
    draws = rng.random(len(ccs))
    selection_flags = (draws < probs).astype(np.float64)
    selected_indices = list(np.where(selection_flags > 0)[0])
    return selected_indices, selection_flags


def generate_box_annotations(seg, g_true, d_margin=5, rng=None, min_cc_size=10,
                              fg_label=None):
    """Full pipeline: mask -> CCs -> channel -> box annotations.

    Args:
        seg: 3D segmentation mask (foreground > 0).
        g_true: channel function.
        d_margin: safe zone margin in voxels.
        rng: numpy random generator.
        min_cc_size: minimum CC size to consider.
        fg_label: if set, only this label value is treated as foreground.

    Returns:
        dict with keys:
            'boxes': list of bbox tuples for selected CCs.
            'cc_sizes': true sizes of selected CCs.
            'all_ccs': list of all CC dicts.
            'selected_indices': which CCs were selected.
            'observed_seg': modified segmentation (unselected CCs zeroed).
    """
    from latentmask.utils.cc_extraction import extract_connected_components
    from scipy import ndimage

    ccs = extract_connected_components(seg, min_size=min_cc_size,
                                       fg_label=fg_label)
    selected_idx, _ = simulate_channel(ccs, g_true, rng=rng)

    # Build observed segmentation: only selected CCs visible
    if fg_label is not None:
        binary = (seg == fg_label).astype(np.int32)
    else:
        binary = (seg > 0).astype(np.int32)
    labeled, _ = ndimage.label(binary)
    observed_seg = np.zeros_like(seg)
    boxes = []
    cc_sizes = []
    for idx in selected_idx:
        cc = ccs[idx]
        cc_mask = labeled == cc['label_id']
        observed_seg[cc_mask] = seg[cc_mask]
        boxes.append(cc['bbox'])
        cc_sizes.append(cc['size'])

    return {
        'boxes': boxes,
        'cc_sizes': cc_sizes,
        'all_ccs': ccs,
        'selected_indices': selected_idx,
        'observed_seg': observed_seg,
    }
