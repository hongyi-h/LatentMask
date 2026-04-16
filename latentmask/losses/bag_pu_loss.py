"""IPW-corrected box supervision loss (Horvitz-Thompson / Hájek estimator).

Implements (FINAL_PROPOSAL §4.4):
  L_box = R̂_Hájek + L_safe
        = mean_j[ w̃_j · (-log p_j) ] + mean_{safe}[-log(1−f_θ)]

where p_j is the bag probability (Noisy-OR, LSE, or top-k mean),
w̃_j is the self-normalised (Hájek) IPW weight, and
safe_loss = mean of −log(1−f_θ) over the safe zone (low-contamination negatives).

Theoretical basis: Under size-conditional ignorability (A2) and positivity (A0),
the Hájek estimator is consistent for the population positive risk.
See Horvitz & Thompson (1952), Hájek (1971).
"""
import torch
import numpy as np


# ── Bag probability aggregation functions ───────────────────────────────

def _bag_prob_noisy_or(f_box, eps=1e-7):
    """Noisy-OR: p = 1 − Π(1 − f_i).  Witness-seeking (standard MIL)."""
    log_1mf = torch.log((1 - f_box).clamp(min=eps))
    sum_log = log_1mf.sum()
    neg_log_p = -torch.log1p(-torch.exp(sum_log.clamp(max=-eps)))
    return neg_log_p  # returns −log(p_j)


def _bag_prob_lse(f_box, r=5.0, eps=1e-7):
    """Log-Sum-Exp pooling: smooth approximation to max.

    p = σ(r⁻¹ · log(mean(exp(r · logit_i))))
    Interpolates between mean (r→0) and max (r→∞).
    Default r=5 is moderately shape-seeking.
    """
    # Convert probs to logits
    logits = torch.log(f_box.clamp(min=eps) / (1 - f_box).clamp(min=eps))
    lse = torch.logsumexp(r * logits, dim=0) if logits.dim() == 1 else \
          torch.logsumexp(r * logits.reshape(-1), dim=0)
    pooled_logit = (lse - np.log(logits.numel())) / r
    p = torch.sigmoid(pooled_logit)
    neg_log_p = -torch.log(p.clamp(min=eps))
    return neg_log_p


def _bag_prob_topk(f_box, k_frac=0.1, k_min=5, eps=1e-7):
    """Top-k mean: p = mean(top k predictions).

    Shape-seeking: encourages a fraction of voxels to predict positive.
    """
    flat = f_box.reshape(-1)
    k = max(k_min, int(k_frac * flat.numel()))
    k = min(k, flat.numel())
    topk_vals = torch.topk(flat, k).values
    p = topk_vals.mean()
    neg_log_p = -torch.log(p.clamp(min=eps))
    return neg_log_p


BAG_PROB_FN = {
    'noisy_or': _bag_prob_noisy_or,
    'lse': _bag_prob_lse,
    'topk': _bag_prob_topk,
}


def compute_bag_pu_loss(fg_probs, boxes, safe_mask,
                        pi_hat, g_theta_func, s0,
                        w_max=10.0, ipw_mode='channel',
                        bag_pooling='noisy_or'):
    """Compute the bag-level box loss for one sample.

    Args:
        fg_probs: foreground probability map, shape (D, H, W). Detached for
                  mass computation; gradients flow through bag probability.
        boxes: list of bbox tuples ((z1,z2),(y1,y2),(x1,x2)).
        safe_mask: binary tensor (D,H,W), 1 = safe zone.
        pi_hat: class prior estimate (scalar).  Used only for diagnostics now.
        g_theta_func: callable(log_mass_array) -> propensity_array.
                      For 'uniform' mode, this is ignored.
        s0: minimum log-size support of isotonic fit.
        w_max: maximum IPW weight.
        ipw_mode: 'channel' | 'uniform' | 'oracle'.
                  'uniform': w=1 for all boxes.
                  'channel': w=1/g_θ(log(mass)).
                  'oracle': expects boxes to carry 'true_size' key.
        bag_pooling: 'noisy_or' | 'lse' | 'topk'.  Bag probability function.

    Returns:
        loss: scalar tensor with gradient.
        diagnostics: dict with weight stats, bag probs, etc.
    """
    device = fg_probs.device
    eps = 1e-7
    bag_fn = BAG_PROB_FN.get(bag_pooling, _bag_prob_noisy_or)

    if len(boxes) == 0:
        # No boxes in this patch -> only safe zone loss
        if safe_mask.any():
            f_safe = fg_probs[safe_mask > 0]
            safe_loss = -torch.log(1 - f_safe.clamp(max=1 - eps)).mean()
            return safe_loss, {'n_boxes': 0, 'safe_loss': safe_loss.item()}
        return torch.tensor(0.0, device=device, requires_grad=True), {'n_boxes': 0}

    pos_terms = []
    raw_weights = []
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

        # Bag positive loss: −log(p_j)
        neg_log_pj = bag_fn(f_box)

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

        pos_terms.append(neg_log_pj)  # weight applied after self-norm
        raw_weights.append(w)
        masses_list.append(mass.item())

    if len(pos_terms) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {'n_boxes': 0}

    n_boxes = len(pos_terms)

    # Self-normalise IPW weights so they sum to n_boxes
    raw_w = np.array(raw_weights)
    norm_w = raw_w * (n_boxes / raw_w.sum()) if raw_w.sum() > 0 else raw_w
    weights_t = torch.tensor(norm_w, dtype=torch.float32, device=device)

    # Weighted positive loss (IPW-corrected, self-normalised mean)
    pos_stack = torch.stack(pos_terms)
    pos_loss = (weights_t * pos_stack).sum() / n_boxes

    # Safe zone loss: reliable-negative supervision (mean over safe voxels)
    safe_loss = torch.tensor(0.0, device=device)
    if safe_mask.any():
        f_safe = fg_probs[safe_mask > 0]
        if f_safe.numel() > 0:
            safe_loss = -torch.log((1 - f_safe).clamp(min=eps)).mean()

    # Direct supervision: positive push + negative push
    total = pos_loss + safe_loss

    diagnostics = {
        'n_boxes': n_boxes,
        'w_mean': float(np.mean(raw_weights)),
        'w_std': float(np.std(raw_weights)),
        'w_max_actual': float(max(raw_weights)),
        'w_min_actual': float(min(raw_weights)),
        'w_norm_mean': float(np.mean(norm_w)),
        'mass_mean': float(np.mean(masses_list)),
        'safe_loss': safe_loss.item(),
        'pos_loss': pos_loss.item(),
        'bag_pooling': bag_pooling,
    }

    return total, diagnostics


def compute_batch_box_loss(output, target, pi_hat, g_theta_func, s0,
                           d_margin=5, w_max=10.0, ipw_mode='channel',
                           min_cc_size=10, fg_label=None,
                           bag_pooling='noisy_or'):
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
        bag_pooling: 'noisy_or' | 'lse' | 'topk'.

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
            bag_pooling=bag_pooling,
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
        pos = [d['pos_loss'] for d in all_diag if 'pos_loss' in d]
        safe = [d['safe_loss'] for d in all_diag if 'safe_loss' in d]
        if pos:
            batch_diag['pos_loss_mean'] = float(np.mean(pos))
        if safe:
            batch_diag['safe_loss_mean'] = float(np.mean(safe))

    return total_loss, batch_diag
