"""Channel-Modulated Box Supervision Loss (LatentMask v5).

v5 mechanism: per-connected-component negative supervision in the safe zone.
Replaces v3's avg_pool3d per-voxel approach with CC-level α assignment.

Three loss terms:
  L_tight:       3D tightness constraint (IPW-weighted).
  L_fill:        Filling-rate interval constraint (IPW-weighted).
  L_channel_neg: Per-CC channel-modulated negative supervision (v5).
"""
import torch
import torch.nn.functional as F
import numpy as np


# ── Scaffold losses (unchanged from v3) ────────────────────────────────

def _tightness_loss(fg_probs, bbox, kappa=1.0):
    """3D tightness: every axis-aligned slice must have mass >= κ."""
    (z1, z2), (y1, y2), (x1, x2) = bbox
    sub = fg_probs[z1:z2, y1:y2, x1:x2]
    if sub.numel() == 0:
        return torch.tensor(0.0, device=fg_probs.device)
    mass_z = sub.sum(dim=(1, 2))
    mass_y = sub.sum(dim=(0, 2))
    mass_x = sub.sum(dim=(0, 1))
    all_mass = torch.cat([mass_z, mass_y, mass_x])
    return F.softplus(kappa - all_mass).mean()


def _fill_loss(fg_probs, bbox, rho_min, rho_max, alpha_upper=1.0):
    """Filling-rate interval: m_j ∈ [ρ_min·V, ρ_max·V]."""
    (z1, z2), (y1, y2), (x1, x2) = bbox
    sub = fg_probs[z1:z2, y1:y2, x1:x2]
    vol = max(sub.numel(), 1)
    mass = sub.sum()
    lo = F.softplus(rho_min * vol - mass) / vol
    hi = F.softplus(mass - rho_max * vol) / vol
    return lo + alpha_upper * hi


# ── v5 channel-modulated negative loss ─────────────────────────────────

def compute_neg_loss_v5(fg_probs, safe_mask, neg_mode,
                        g_theta_func=None, s0=0.0,
                        linear_a=0.0, linear_b=0.1,
                        tau_low=0.3, tau_high=0.5, alpha_min=0.05,
                        min_cc_size=10, eps=1e-7):
    """Per-component channel-modulated negative supervision (v5 protocol).

    Args:
        fg_probs: (D, H, W) tensor of foreground probabilities.
        safe_mask: (D, H, W) tensor, 1 = safe zone.
        neg_mode: 'uniform' (C2), 'linear' (C3), or 'channel' (C4).
        g_theta_func: callable for isotonic propensity (C4).
        s0: minimum support for g_theta.
        linear_a, linear_b: coefficients for linear baseline (C3).
        tau_low, tau_high: CC extraction thresholds.
        alpha_min: minimum alpha for nascent CCs.
        min_cc_size: minimum CC voxel count.

    Returns:
        loss: scalar tensor.
        diag: dict with coverage_ratio, nascent_ratio, mean_alpha, n_ccs.
    """
    from latentmask.utils.cc_extraction import extract_prediction_ccs_v5

    device = fg_probs.device
    safe_idx = safe_mask > 0

    if not safe_idx.any():
        return torch.tensor(0.0, device=device), {
            'coverage_ratio': 0.0, 'nascent_ratio': 0.0,
            'mean_alpha': 1.0, 'n_ccs': 0}

    f_safe = fg_probs[safe_idx]
    neg_ce = -torch.log((1 - f_safe).clamp(min=eps))

    # C2: uniform α = 1.0, no CC extraction needed
    if neg_mode == 'uniform':
        loss = neg_ce.mean()
        return loss, {'coverage_ratio': 0.0, 'nascent_ratio': 0.0,
                      'mean_alpha': 1.0, 'n_ccs': 0}

    # C3/C4: CC-level modulation
    with torch.no_grad():
        fg_np = fg_probs.detach().cpu().numpy()
        safe_np = safe_mask.cpu().numpy()

    ccs, labeled = extract_prediction_ccs_v5(
        fg_np, safe_np, tau_low, tau_high, min_cc_size)

    # Build per-voxel alpha map (default α=1.0 for unstructured safe zone)
    alpha_np = np.ones(fg_np.shape, dtype=np.float32)

    n_nascent = 0
    n_mature = 0
    n_boundary = 0
    alpha_values = []

    for cc in ccs:
        cc_mask = labeled == cc['label_id']

        if cc['touches_boundary']:
            alpha_np[cc_mask] = 1.0
            n_boundary += 1
        elif not cc['is_mature']:
            alpha_np[cc_mask] = alpha_min
            n_nascent += 1
            alpha_values.append(alpha_min)
        else:
            # Mature interior CC: query alpha function
            if neg_mode == 'linear':
                a_val = linear_a + linear_b * cc['log_mass']
            else:  # channel
                g_val = g_theta_func(np.array([cc['log_mass']]))[0]
                a_val = float(g_val)
            a_val = max(min(a_val, 1.0), alpha_min)
            alpha_np[cc_mask] = a_val
            n_mature += 1
            alpha_values.append(a_val)

    # Convert alpha to tensor (safe-zone voxels only)
    alpha_safe = torch.from_numpy(alpha_np[safe_np > 0]).float().to(device)
    loss = (alpha_safe * neg_ce).mean()

    # Diagnostics
    n_total = n_nascent + n_mature + n_boundary
    with torch.no_grad():
        mask_non_uniform = alpha_safe < 1.0
        if mask_non_uniform.any():
            coverage_mass = f_safe[mask_non_uniform].sum()
        else:
            coverage_mass = torch.tensor(0.0)
        total_mass = f_safe.sum()
        coverage_ratio = float(coverage_mass / total_mass) if total_mass > 0 else 0.0

    diag = {
        'coverage_ratio': coverage_ratio,
        'nascent_ratio': n_nascent / max(n_nascent + n_mature, 1),
        'mean_alpha': float(np.mean(alpha_values)) if alpha_values else 1.0,
        'n_ccs': n_total,
        'n_nascent': n_nascent,
        'n_mature': n_mature,
        'n_boundary': n_boundary,
    }
    return loss, diag


# ── Sample-level loss ──────────────────────────────────────────────────

def _compute_sample_loss_v5(fg_probs, boxes, safe_mask,
                             neg_mode='channel',
                             g_theta_func=None, s0=0.0,
                             linear_a=0.0, linear_b=0.1,
                             ipw_mode='channel', w_max=10.0,
                             kappa=1.0, rho_min=0.15, rho_max=0.85,
                             alpha_upper=1.0,
                             beta_fill=1.0, gamma_neg=0.1,
                             tau_low=0.3, tau_high=0.5, alpha_min=0.05,
                             min_cc_size=10, enable_neg=True):
    """Compute 3-term box loss for one sample (v5)."""
    device = fg_probs.device
    eps = 1e-7

    # ── Scaffold (tightness + fill) with IPW ───────────────────────
    tight_terms = []
    fill_terms = []
    raw_weights = []

    for box_info in boxes:
        bbox = box_info['bbox']
        true_size = box_info.get('true_size', None)

        # IPW weight from g_theta
        if ipw_mode == 'uniform' or neg_mode == 'uniform':
            w = 1.0
        elif g_theta_func is not None:
            if true_size is not None:
                log_s = np.log(max(true_size, 1))
            else:
                with torch.no_grad():
                    (z1, z2), (y1, y2), (x1, x2) = bbox
                    mass = fg_probs[z1:z2, y1:y2, x1:x2].detach().sum()
                    log_s = float(np.log(max(mass.item(), np.exp(s0))))
            propensity = g_theta_func(np.array([log_s]))[0]
            w = min(1.0 / max(propensity, eps), w_max)
        else:
            w = 1.0

        tight_terms.append(_tightness_loss(fg_probs, bbox, kappa))
        fill_terms.append(_fill_loss(fg_probs, bbox, rho_min, rho_max,
                                     alpha_upper))
        raw_weights.append(w)

    n_boxes = len(tight_terms)

    # Scaffold loss
    if n_boxes > 0:
        raw_w = np.array(raw_weights)
        norm_w = raw_w * (n_boxes / raw_w.sum()) if raw_w.sum() > 0 else raw_w
        w_t = torch.tensor(norm_w, dtype=torch.float32, device=device)
        tight_stack = torch.stack(tight_terms)
        fill_stack = torch.stack(fill_terms)
        L_tight = (w_t * tight_stack).sum() / n_boxes
        L_fill = (w_t * fill_stack).sum() / n_boxes
    else:
        L_tight = torch.tensor(0.0, device=device)
        L_fill = torch.tensor(0.0, device=device)

    # ── Channel-neg (v5 CC-level) ──────────────────────────────────
    L_neg = torch.tensor(0.0, device=device)
    neg_diag = {'coverage_ratio': 0.0, 'nascent_ratio': 0.0,
                'mean_alpha': 1.0, 'n_ccs': 0}

    if enable_neg and safe_mask.any():
        L_neg, neg_diag = compute_neg_loss_v5(
            fg_probs, safe_mask, neg_mode,
            g_theta_func=g_theta_func, s0=s0,
            linear_a=linear_a, linear_b=linear_b,
            tau_low=tau_low, tau_high=tau_high,
            alpha_min=alpha_min, min_cc_size=min_cc_size)

    total = L_tight + beta_fill * L_fill + gamma_neg * L_neg

    diagnostics = {
        'n_boxes': n_boxes,
        'tight_loss': L_tight.item(),
        'fill_loss': L_fill.item(),
        'neg_loss': L_neg.item(),
        **neg_diag,
    }
    if raw_weights:
        diagnostics['w_mean'] = float(np.mean(raw_weights))

    return total, diagnostics


# ── Batch-level loss ───────────────────────────────────────────────────

def compute_batch_box_loss_v5(output, target, neg_mode='channel',
                               g_theta_func=None, s0=0.0,
                               linear_a=0.0, linear_b=0.1,
                               d_margin=5, w_max=10.0, ipw_mode='channel',
                               min_cc_size=10, fg_label=None,
                               kappa=1.0, rho_min=0.15, rho_max=0.85,
                               alpha_upper=1.0,
                               beta_fill=1.0, gamma_neg=0.1,
                               tau_low=0.3, tau_high=0.5, alpha_min=0.05,
                               enable_neg=True):
    """Compute v5 channel-modulated box loss over a batch.

    Args:
        output: (B, C, D, H, W) network output logits.
        target: (B, 1, D, H, W) ground truth segmentation.
        neg_mode: 'uniform' | 'linear' | 'channel'.
        All other args: see _compute_sample_loss_v5.

    Returns:
        loss: scalar tensor.
        batch_diag: dict with aggregated diagnostics.
    """
    from latentmask.utils.cc_extraction import (
        extract_ccs_from_patch, compute_safe_zone_mask,
    )

    B = output.shape[0]
    device = output.device

    probs = torch.softmax(output.float(), dim=1)
    if fg_label is not None and fg_label < probs.shape[1]:
        fg_probs = probs[:, fg_label]
    else:
        fg_probs = 1 - probs[:, 0]

    total_loss = (output * 0).sum()
    all_diag = []
    n_valid = 0

    for b in range(B):
        seg_np = target[b, 0].cpu().numpy().astype(np.int32)

        # Extract GT CCs -> bounding boxes for scaffold
        ccs = extract_ccs_from_patch(seg_np, min_size=min_cc_size,
                                     fg_label=fg_label)
        if len(ccs) == 0:
            continue

        boxes = [{'bbox': cc['bbox'], 'true_size': cc['size']} for cc in ccs]

        # Safe zone from GT foreground
        safe_np = compute_safe_zone_mask(seg_np, d_margin, fg_label=fg_label)
        safe_mask = torch.from_numpy(safe_np).to(device)

        sample_loss, diag = _compute_sample_loss_v5(
            fg_probs[b], boxes, safe_mask,
            neg_mode=neg_mode,
            g_theta_func=g_theta_func, s0=s0,
            linear_a=linear_a, linear_b=linear_b,
            ipw_mode=ipw_mode, w_max=w_max,
            kappa=kappa, rho_min=rho_min, rho_max=rho_max,
            alpha_upper=alpha_upper,
            beta_fill=beta_fill, gamma_neg=gamma_neg,
            tau_low=tau_low, tau_high=tau_high, alpha_min=alpha_min,
            min_cc_size=min_cc_size, enable_neg=enable_neg,
        )

        total_loss = total_loss + sample_loss
        all_diag.append(diag)
        n_valid += 1

    if n_valid > 0:
        total_loss = total_loss / n_valid

    # Aggregate batch diagnostics
    batch_diag = {
        'n_samples_with_boxes': n_valid,
        'total_boxes': sum(d.get('n_boxes', 0) for d in all_diag),
    }
    if all_diag:
        for key in ('tight_loss', 'fill_loss', 'neg_loss',
                     'coverage_ratio', 'nascent_ratio', 'mean_alpha'):
            vals = [d[key] for d in all_diag if key in d]
            if vals:
                batch_diag[f'{key}_mean'] = float(np.mean(vals))
        batch_diag['n_ccs_total'] = sum(d.get('n_ccs', 0) for d in all_diag)

    return total_loss, batch_diag
