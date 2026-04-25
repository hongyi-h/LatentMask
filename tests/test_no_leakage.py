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
    """All neg_modes (uniform, linear, channel) work with v6."""
    from latentmask.losses.bag_pu_loss import compute_batch_box_loss_v6

    B, C, D, H, W = 1, 3, 16, 16, 16
    output = torch.randn(B, C, D, H, W, requires_grad=True)
    box_metadata_list = [[{'bbox': ((2, 10), (3, 12), (4, 14))}]]

    def g_func(log_sizes):
        return np.clip(log_sizes / 10.0, 0.01, 0.99)

    for mode in ('uniform', 'linear', 'channel'):
        loss, diag = compute_batch_box_loss_v6(
            output, box_metadata_list,
            neg_mode=mode, d_margin=2, fg_label=2,
            g_theta_func=g_func, s0=0.0,
            linear_a=0.3, linear_b=0.05,
            enable_neg=True,
        )
        assert not torch.isnan(loss), f"NaN loss for neg_mode={mode}"
        assert diag['n_samples_with_boxes'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
