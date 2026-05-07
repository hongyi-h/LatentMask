"""Unit tests verifying no label leakage in the v6 box loss pipeline.

Tests that:
- compute_batch_box_loss_v6 never accesses a target tensor
- Safe zone is computed from box coordinates only
- IPW weights use predicted mass, not true_size
- Box metadata format is correct
"""
import numpy as np
import torch
import pytest
import inspect


def test_box_loss_v6_signature_has_no_target():
    """v6 loss function must not accept a 'target' parameter."""
    from latentmask.losses.bag_pu_loss import compute_batch_box_loss_v6
    sig = inspect.signature(compute_batch_box_loss_v6)
    param_names = list(sig.parameters.keys())
    assert 'target' not in param_names, (
        f"compute_batch_box_loss_v6 still has 'target' parameter: {param_names}")


def test_box_loss_v6_runs_without_target():
    """v6 loss computes correctly from box metadata alone."""
    from latentmask.losses.bag_pu_loss import compute_batch_box_loss_v6

    B, C, D, H, W = 2, 3, 16, 16, 16
    output = torch.randn(B, C, D, H, W, requires_grad=True)

    box_metadata_list = [
        [{'bbox': ((2, 10), (3, 12), (4, 14))}],
        [{'bbox': ((1, 8), (2, 10), (3, 11))},
         {'bbox': ((5, 14), (6, 15), (7, 15))}],
    ]

    loss, diag = compute_batch_box_loss_v6(
        output, box_metadata_list,
        neg_mode='uniform', d_margin=2, fg_label=2,
        enable_neg=True,
    )

    assert loss.requires_grad, "Loss should be differentiable"
    assert diag['n_samples_with_boxes'] == 2
    loss.backward()


def test_box_loss_v6_empty_boxes():
    """v6 handles samples with no boxes gracefully."""
    from latentmask.losses.bag_pu_loss import compute_batch_box_loss_v6

    B, C, D, H, W = 1, 3, 16, 16, 16
    output = torch.randn(B, C, D, H, W, requires_grad=True)
    box_metadata_list = [[]]

    loss, diag = compute_batch_box_loss_v6(
        output, box_metadata_list,
        neg_mode='channel', d_margin=3, fg_label=2,
    )
    assert diag['n_samples_with_boxes'] == 0


def test_no_true_size_in_sample_loss():
    """_compute_sample_loss_v6 must not read true_size from box dicts."""
    from latentmask.losses.bag_pu_loss import _compute_sample_loss_v6

    src = inspect.getsource(_compute_sample_loss_v6)
    assert 'true_size' not in src.split('# IPW')[0], (
        "_compute_sample_loss_v6 references true_size before the comment")

    # Functional test: passing true_size should have no effect
    fg_probs = torch.sigmoid(torch.randn(16, 16, 16))
    safe_mask = torch.ones(16, 16, 16)
    boxes_without = [{'bbox': ((2, 10), (3, 12), (4, 14))}]
    boxes_with = [{'bbox': ((2, 10), (3, 12), (4, 14)), 'true_size': 999999}]

    def g_func(log_sizes):
        return np.clip(log_sizes / 10.0, 0.01, 0.99)

    loss1, _ = _compute_sample_loss_v6(
        fg_probs, boxes_without, safe_mask,
        neg_mode='channel', g_theta_func=g_func, s0=0.0,
        ipw_mode='channel', enable_neg=True)
    loss2, _ = _compute_sample_loss_v6(
        fg_probs, boxes_with, safe_mask,
        neg_mode='channel', g_theta_func=g_func, s0=0.0,
        ipw_mode='channel', enable_neg=True)

    assert torch.allclose(loss1, loss2), (
        f"true_size in box dict changed the loss: {loss1.item()} vs {loss2.item()}")


def test_safe_zone_from_boxes_not_gt():
    """v6 batch loss uses compute_safe_zone_from_boxes, not compute_safe_zone_mask."""
    from latentmask.losses.bag_pu_loss import compute_batch_box_loss_v6
    src = inspect.getsource(compute_batch_box_loss_v6)
    assert 'compute_safe_zone_from_boxes' in src, (
        "v6 should use compute_safe_zone_from_boxes")
    assert 'compute_safe_zone_mask' not in src, (
        "v6 should NOT use compute_safe_zone_mask (GT-based)")


def test_safe_zone_from_boxes_correctness():
    """Safe zone from boxes excludes box regions + margin."""
    from latentmask.utils.cc_extraction import compute_safe_zone_from_boxes

    shape = (20, 20, 20)
    boxes = [((5, 10), (5, 10), (5, 10))]
    d_safe = 3

    safe = compute_safe_zone_from_boxes(shape, boxes, d_safe)

    # Inside box should NOT be safe
    assert safe[7, 7, 7] == 0.0, "Inside box should not be safe zone"
    # Right at margin boundary
    assert safe[5, 5, 2] == 0.0, "Within d_safe of box should not be safe"
    # Far from box should be safe
    assert safe[0, 0, 0] == 1.0, "Far from box should be safe zone"


def test_no_leakage_imports_in_v6_loss():
    """v6 batch loss should not import GT-dependent functions."""
    from latentmask.losses.bag_pu_loss import compute_batch_box_loss_v6
    src = inspect.getsource(compute_batch_box_loss_v6)
    assert 'extract_ccs_from_patch' not in src, (
        "v6 should not extract CCs from GT patches")
    assert 'target[' not in src and 'target,' not in src, (
        "v6 should not reference target tensor")


def test_channel_neg_modes():
    """All neg_modes (uniform, constant, linear, channel, inverted) work."""
    from latentmask.losses.bag_pu_loss import compute_batch_box_loss_v6

    B, C, D, H, W = 1, 3, 16, 16, 16
    torch.manual_seed(0)
    output = torch.randn(B, C, D, H, W, requires_grad=True)
    box_metadata_list = [[{'bbox': ((2, 10), (3, 12), (4, 14))}]]

    def g_func(log_sizes):
        return np.clip(log_sizes / 10.0, 0.01, 0.99)

    for mode in ('uniform', 'constant', 'linear', 'channel', 'inverted'):
        loss, diag = compute_batch_box_loss_v6(
            output, box_metadata_list,
            neg_mode=mode, d_margin=2, fg_label=2,
            g_theta_func=g_func, s0=0.0,
            linear_a=0.3, linear_b=0.05,
            constant_alpha=0.5,
            enable_neg=True,
        )
        assert not torch.isnan(loss), f"NaN loss for neg_mode={mode}"
        assert diag['n_samples_with_boxes'] == 1


_HAS_SKLEARN = True
try:
    import sklearn  # noqa: F401
except Exception:
    _HAS_SKLEARN = False


@pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not importable")
def test_isotonic_dict_roundtrip_matches_sklearn():
    """Version-stable g_θ serialization (isotonic_to_dict + np.interp)
    must numerically match sklearn IsotonicRegression.predict."""
    from sklearn.isotonic import IsotonicRegression
    from latentmask.calibration.isotonic_fit import (
        isotonic_to_dict, predict_propensity,
    )

    rng = np.random.default_rng(7)
    log_sizes = np.sort(rng.uniform(2.0, 10.0, 80))
    # Retention probability increasing in log_size, Bernoulli-drawn
    true_p = 0.1 + 0.8 / (1 + np.exp(-1.5 * (log_sizes - 6.0)))
    labels = (rng.random(80) < true_p).astype(float)

    ir = IsotonicRegression(y_min=0.01, y_max=0.99,
                            increasing=True, out_of_bounds='clip')
    ir.fit(log_sizes, labels)
    s0 = float(log_sizes.min())

    query = np.linspace(0.0, 12.0, 50)
    sklearn_pred = predict_propensity(ir, query, s0)
    dict_pred = predict_propensity(isotonic_to_dict(ir), query, s0)

    # Both predictions are clamped below s0, so only the above-s0 region
    # needs strict equality. np.interp vs sklearn's step function may
    # differ by up to one step width at exact threshold points, but on
    # generic query points they must be numerically identical.
    assert np.allclose(sklearn_pred, dict_pred, atol=1e-6), (
        f"Max gap sklearn vs dict: "
        f"{np.abs(sklearn_pred - dict_pred).max()}")
    """C2.5: loss must scale linearly with constant_alpha (all else equal)."""
    from latentmask.losses.bag_pu_loss import compute_neg_loss_v5

    torch.manual_seed(1)
    fg = torch.sigmoid(torch.randn(12, 12, 12))
    safe = torch.ones(12, 12, 12)

    l_low, _ = compute_neg_loss_v5(fg, safe, neg_mode='constant',
                                    constant_alpha=0.2)
    l_mid, _ = compute_neg_loss_v5(fg, safe, neg_mode='constant',
                                    constant_alpha=0.5)
    l_hi, _ = compute_neg_loss_v5(fg, safe, neg_mode='constant',
                                   constant_alpha=1.0)
    # α=0.2 should give 0.4× the α=0.5 loss; α=1.0 should give 2.0× the α=0.5 loss
    ratio_low = (l_low / l_mid).item()
    ratio_hi = (l_hi / l_mid).item()
    assert abs(ratio_low - 0.4) < 1e-4, f"constant α=0.2 / α=0.5 = {ratio_low}"
    assert abs(ratio_hi - 2.0) < 1e-4, f"constant α=1.0 / α=0.5 = {ratio_hi}"


def test_inverted_opposes_channel():
    """C4-inv: α_channel + α_inverted ≈ 1 + α_min (reflection around 0.5)."""
    from latentmask.losses.bag_pu_loss import compute_neg_loss_v5

    torch.manual_seed(2)
    # One mature interior CC: 5x5x5 at 0.8 prob -> log_mass ≈ log(100) ≈ 4.6
    fg = torch.full((24, 24, 24), 0.05)
    fg[10:15, 10:15, 10:15] = 0.8
    safe = torch.ones(24, 24, 24)

    # Monotone increasing g_θ, mapped so mid-range gives ~0.5
    def g_func(log_sizes):
        return np.clip(np.asarray(log_sizes) / 6.0, 0.01, 0.99)

    _, diag_c4 = compute_neg_loss_v5(
        fg, safe, neg_mode='channel', g_theta_func=g_func, s0=0.0,
        tau_low=0.3, tau_high=0.5, alpha_min=0.05, min_cc_size=10)
    _, diag_inv = compute_neg_loss_v5(
        fg, safe, neg_mode='inverted', g_theta_func=g_func, s0=0.0,
        tau_low=0.3, tau_high=0.5, alpha_min=0.05, min_cc_size=10)

    # Sanity: both modes should have discovered at least one CC.
    assert diag_c4['n_ccs'] > 0 and diag_inv['n_ccs'] > 0

    # Core relation: α_channel + α_inverted = 1 + α_min (before clamping).
    # With g ≈ 0.77 (log_mass ≈ 4.6 / 6), both stay inside [α_min, 1], so
    # clamping is a no-op.
    s = diag_c4['mean_alpha'] + diag_inv['mean_alpha']
    assert abs(s - 1.05) < 1e-3, (
        f"channel + inverted = {s:.4f}, expected ~1.05 (=1+α_min)")
    # And they must be on opposite sides of 0.5 + α_min/2 = 0.525
    assert diag_c4['mean_alpha'] > 0.525
    assert diag_inv['mean_alpha'] < 0.525


# ── Trainer-dependent tests (require nnUNet install on this machine) ──
# On dev machines without nnunetv2 installed we skip; they run on the
# GPU server where nnunetv2 is present.
_HAS_NNUNET = True
try:
    import nnunetv2  # noqa: F401
except ImportError:
    _HAS_NNUNET = False


@pytest.mark.skipif(not _HAS_NNUNET, reason="nnunetv2 not installed")
def test_trainer_requires_calibration_artifact(tmp_path):
    """Trainer must refuse to run channel/linear/inverted without artifact."""
    import pickle
    from latentmask.trainer import latentmask_trainer as lm_mod

    class _Stub:
        fold = 0
        fg_label = 2
        neg_mode = 'channel'
        box_annotations_dir = str(tmp_path)
        output_folder = str(tmp_path)
        def print_to_log_file(self, *a, **k): pass

    stub = _Stub()
    try:
        lm_mod.LatentMaskTrainer._load_calibration_artifact(stub)
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised, "Trainer must fail loudly when artifact is missing"

    art = {
        'fold': 99, 'fg_label': 2, 'protocol': 'P-steep',
        'g_theta': None, 's0': 0.0,
        'linear_a': 0.0, 'linear_b': 0.0, 'mu': 7.0,
        'rho_min': 0.1, 'rho_max': 0.9,
    }
    with open(tmp_path / '_calibration_fold0.pkl', 'wb') as f:
        pickle.dump(art, f)
    try:
        lm_mod.LatentMaskTrainer._load_calibration_artifact(stub)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "Trainer must reject artifact with mismatched fold"


@pytest.mark.skipif(not _HAS_NNUNET, reason="nnunetv2 not installed")
def test_trainer_no_channel_simulator_import():
    """v6.1 trainer must not import the protocol simulator (protocol unknown)."""
    import inspect
    from latentmask.trainer import latentmask_trainer as lm_mod
    src = inspect.getsource(lm_mod)
    assert 'make_channel_func' not in src, (
        "Trainer still imports make_channel_func — protocol is supposed "
        "to be unknown to the method")
    assert 'channel_simulator' not in src, (
        "Trainer still imports channel_simulator")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
