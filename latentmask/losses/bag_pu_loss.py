"""Bag-level biased PU loss with inverse-propensity weighting.

Implements L_box from FINAL_PROPOSAL §4.4:
  L_box = π̂ · Σ_j w_j·(-log p_j)
        + max(0, L_safe − π̂ · Σ_j w_j·(-log(1−p_j)))

where p_j = 1 − Π_{i∈B_j}(1 − f_θ(x_i)) is the bag probability,
w_j = 1/g_θ(log(mass_j)) is the IPW weight, and
L_safe = mean of −log(1−f_θ) over the safe zone.
"""
import torch
import numpy as np


def compute_bag_pu_loss(fg_probs, boxes, safe_mask,
                        pi_hat, g_theta_func, s0,
                        w_max=10.0, ipw_mode='channel'):
    """Compute the bag-level biased PU loss for one sample.

    Args:
        fg_probs: foreground probability map, shape (D, H, W). Detached for
                  mass computation; gradients flow through bag probability.
        boxes: list of bbox tuples ((z1,z2),(y1,y2),(x1,x2)).
        safe_mask: binary tensor (D,H,W), 1 = safe zone.
        pi_hat: class prior estimate (scalar).
        g_theta_func: callable(log_mass_array) -> propensity_array.
                      For 'uniform' mode, this is ignored.
        s0: minimum log-size support of isotonic fit.
        w_max: maximum IPW weight.
        ipw_mode: 'channel' | 'uniform' | 'oracle'.
                  'uniform': w=1 for all boxes.
                  'channel': w=1/g_θ(log(mass)).
                  'oracle': expects boxes to carry 'true_size' key.

    Returns:
        loss: scalar tensor with gradient.
        diagnostics: dict with weight stats, bag probs, etc.
    """
    device = fg_probs.device
    eps = 1e-7

    if len(boxes) == 0:
        # No boxes in this patch -> only safe zone loss
        if safe_mask.any():
            f_safe = fg_probs[safe_mask > 0]
            safe_loss = -torch.log(1 - f_safe.clamp(max=1 - eps)).mean()
            return safe_loss, {'n_boxes': 0, 'safe_loss': safe_loss.item()}
        return torch.tensor(0.0, device=device, requires_grad=True), {'n_boxes': 0}

    pos_terms = []
    neg_terms = []
    weights_list = []
    masses_list = []

    for box_info in boxes:
        if isinstance(box_info, dict):
            bbox = box_info['bbox']
        else:
            bbox = box_info

        (z1, z2), (y1, y2), (x1, x2) = bbox
        f_box = fg_probs[z1:z2, y1:y2, x1:x2]

        if f_box.numel() == 0:
            continue

        # Bag probability in log space
        log_1mf = torch.log((1 - f_box).clamp(min=eps))
        sum_log_1mf = log_1mf.sum()

        # -log(p_j) = -log(1 - exp(sum_log_1mf))
        # Use log1p(-exp(x)) for numerical stability
        neg_log_pj = -torch.log1p(-torch.exp(sum_log_1mf.clamp(max=-eps)))

        # -log(1 - p_j) = -sum_log_1mf
        neg_log_1mpj = -sum_log_1mf

        # Soft positive mass (detached, no gradient)
        with torch.no_grad():
            mass = f_box.detach().sum().clamp(min=np.exp(s0))
            log_mass = float(np.log(mass.item()))

        # Compute weight based on mode
        if ipw_mode == 'uniform':
            w = 1.0
        elif ipw_mode == 'channel':
            propensity = g_theta_func(np.array([log_mass]))[0]
            w = min(1.0 / max(propensity, eps), w_max)
        elif ipw_mode == 'oracle':
            true_size = box_info.get('true_size', mass.item())
            log_true = np.log(max(true_size, 1))
            propensity = g_theta_func(np.array([log_true]))[0]
            w = min(1.0 / max(propensity, eps), w_max)
        else:
            w = 1.0

        pos_terms.append(w * neg_log_pj)
        neg_terms.append(w * neg_log_1mpj)
        weights_list.append(w)
        masses_list.append(mass.item())

    if len(pos_terms) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {'n_boxes': 0}

    pos_loss = torch.stack(pos_terms).sum()
    neg_loss = torch.stack(neg_terms).sum()

    # Safe zone loss
    safe_loss = torch.tensor(0.0, device=device)
    if safe_mask.any():
        f_safe = fg_probs[safe_mask > 0]
        if f_safe.numel() > 0:
            safe_loss = -torch.log((1 - f_safe).clamp(min=eps)).mean()

    # nnPU clipping
    raw_neg = safe_loss - pi_hat * neg_loss
    nnpu_neg = torch.clamp(raw_neg, min=0)
    total = pi_hat * pos_loss + nnpu_neg

    diagnostics = {
        'n_boxes': len(pos_terms),
        'w_mean': float(np.mean(weights_list)),
        'w_std': float(np.std(weights_list)),
        'w_max_actual': float(max(weights_list)),
        'w_min_actual': float(min(weights_list)),
        'mass_mean': float(np.mean(masses_list)),
        'safe_loss': safe_loss.item(),
        'pos_loss': (pi_hat * pos_loss).item(),
        'neg_loss': neg_loss.item(),
        'nnpu_clipped': bool(raw_neg.item() < 0),
        'nnpu_raw_neg': raw_neg.item(),
    }

    return total, diagnostics


def compute_batch_box_loss(output, target, pi_hat, g_theta_func, s0,
                           d_margin=5, w_max=10.0, ipw_mode='channel',
                           min_cc_size=10, fg_label=None):
    """Compute box loss over a batch by extracting CCs from the target.

    This is the simplified M0-ready version: CCs are extracted from the
    actual target segmentation in each patch. For M2+ with proper 30/70
    splits, use precomputed box annotations instead.

    Args:
        output: network output, shape (B, C, D, H, W). Raw logits.
        target: segmentation target, shape (B, 1, D, H, W). Integer labels.
        pi_hat: class prior.
        g_theta_func: isotonic calibrator function.
        s0: minimum log-size support.
        d_margin: safe zone margin.
        w_max: max IPW weight.
        ipw_mode: 'channel' | 'uniform' | 'oracle'.
        min_cc_size: minimum CC size to form a box.
        fg_label: if set, only this label is treated as foreground for CC
                  extraction and safe zone computation.

    Returns:
        loss: scalar tensor.
        batch_diag: aggregated diagnostics.
    """
    from latentmask.utils.cc_extraction import extract_ccs_from_patch, compute_safe_zone_mask

    B = output.shape[0]
    device = output.device

    # Foreground probabilities from logits
    probs = torch.softmax(output.float(), dim=1)
    if fg_label is not None and fg_label < probs.shape[1]:
        # Use only the target class probability (e.g. tumor = class 2)
        fg_probs = probs[:, fg_label]  # shape (B, D, H, W)
    else:
        # Binary case: foreground = 1 - P(background)
        fg_probs = 1 - probs[:, 0]  # shape (B, D, H, W)

    total_loss = (output * 0).sum()  # differentiable zero tied to output
    all_diag = []
    n_valid = 0

    for b in range(B):
        seg_np = target[b, 0].cpu().numpy().astype(np.int32)

        # Extract CCs from this patch
        ccs = extract_ccs_from_patch(seg_np, min_size=min_cc_size,
                                     fg_label=fg_label)
        if len(ccs) == 0:
            continue

        # Build box list
        boxes = []
        for cc in ccs:
            boxes.append({
                'bbox': cc['bbox'],
                'true_size': cc['size'],
            })

        # Safe zone
        safe_np = compute_safe_zone_mask(seg_np, d_margin, fg_label=fg_label)
        safe_mask = torch.from_numpy(safe_np).to(device)

        # Compute loss for this sample
        sample_loss, diag = compute_bag_pu_loss(
            fg_probs[b], boxes, safe_mask,
            pi_hat, g_theta_func, s0,
            w_max=w_max, ipw_mode=ipw_mode,
        )

        total_loss = total_loss + sample_loss
        all_diag.append(diag)
        n_valid += 1

    if n_valid > 0:
        total_loss = total_loss / n_valid

    # Aggregate diagnostics
    batch_diag = {
        'n_samples_with_boxes': n_valid,
        'total_boxes': sum(d.get('n_boxes', 0) for d in all_diag),
    }
    if all_diag:
        ws = [d['w_mean'] for d in all_diag if 'w_mean' in d]
        if ws:
            batch_diag['w_mean'] = float(np.mean(ws))
            batch_diag['w_std'] = float(np.mean([d.get('w_std', 0) for d in all_diag]))
        clipped = [d.get('nnpu_clipped', False) for d in all_diag]
        batch_diag['nnpu_clip_rate'] = float(np.mean(clipped))

    return total_loss, batch_diag
