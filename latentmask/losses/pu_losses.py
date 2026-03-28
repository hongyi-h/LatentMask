"""
Propensity-corrected Positive-Unlabeled (PU) losses for multi-granularity
segmentation.

Based on Kiryo et al. (NeurIPS 2017): non-negative PU risk.

Domain-agnostic losses:
  - PixelPULoss:  standard supervised (propensity = 1)
  - BoxPULoss:    PU loss for box-annotated data
  - ImagePULoss:  PU loss for image-level annotated data
  - SpatialSmoothnessLoss: TV regularisation on propensity map
  - PropNetTrainingLoss: BCE on synthetic propensity (no domain priors)
  - RefinementLoss: EMA teacher pseudo-labels weighted by propensity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelPULoss(nn.Module):
    """
    Standard supervised loss for pixel-labeled data.

    L_pix = Dice(p, y) + CE(p, y)

    Since these voxels have propensity = 1, no PU correction needed.
    """

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask
            target = target * mask

        # Dice loss
        smooth = 1e-5
        intersection = (pred * target).sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (
            pred.sum() + target.sum() + smooth
        )

        # Binary CE
        ce = F.binary_cross_entropy(pred, target.float(), reduction="mean")

        return self.dice_weight * dice + self.ce_weight * ce


class BoxPULoss(nn.Module):
    """
    PU-corrected loss for box-annotated data.

    Inside box: labeled positives (P).
    Outside box: unlabeled (U) — may contain unlabeled positives.

    L_box = π · L_pos + max(0, L_unlabeled - π · L_neg_on_pos)

    The propensity map from PropNet modulates how much we trust the
    negative label outside boxes: low propensity → less negative push.
    """

    def __init__(self, pi_base: float = 0.15):
        super().__init__()
        self.pi_base = pi_base

    def forward(
        self,
        pred: torch.Tensor,
        box_mask: torch.Tensor,
        target_inside_box: torch.Tensor,
        propensity: torch.Tensor,
    ) -> torch.Tensor:
        # --- Positive risk (inside box, positive voxels) ---
        pos_mask = box_mask * target_inside_box
        if pos_mask.sum() > 0:
            l_pos = F.binary_cross_entropy(
                pred[pos_mask.bool()], torch.ones_like(pred[pos_mask.bool()])
            )
        else:
            l_pos = torch.tensor(0.0, device=pred.device)

        # --- Negative-on-positive (inside box, used for PU correction) ---
        if pos_mask.sum() > 0:
            l_neg_on_pos = F.binary_cross_entropy(
                pred[pos_mask.bool()], torch.zeros_like(pred[pos_mask.bool()])
            )
        else:
            l_neg_on_pos = torch.tensor(0.0, device=pred.device)

        # --- Unlabeled risk (outside box) ---
        outside_mask = 1.0 - box_mask
        if outside_mask.sum() > 0:
            weights = propensity * outside_mask
            w_sum = weights.sum().clamp(min=1e-8)
            l_unlabeled = (
                weights * F.binary_cross_entropy(
                    pred, torch.zeros_like(pred), reduction="none"
                )
            ).sum() / w_sum
        else:
            l_unlabeled = torch.tensor(0.0, device=pred.device)

        # --- Non-negative PU risk (Kiryo et al.) ---
        pu_risk = l_unlabeled - self.pi_base * l_neg_on_pos
        pu_risk = torch.clamp(pu_risk, min=0.0)

        loss = self.pi_base * l_pos + pu_risk
        return loss


class ImagePULoss(nn.Module):
    """
    PU-corrected loss for image-level annotated data.

    Positive images: object present but no pixel localization.
    Negative images: confirmed absent → all pixels are TN.
    """

    def __init__(self, pi_base: float = 0.15):
        super().__init__()
        self.pi_base = pi_base

    def forward(
        self,
        pred: torch.Tensor,
        image_label: torch.Tensor,
        propensity: torch.Tensor,
    ) -> torch.Tensor:
        B = pred.shape[0]
        total_loss = torch.tensor(0.0, device=pred.device)
        count = 0

        for b in range(B):
            p = pred[b]  # (1, *spatial)
            e = propensity[b]

            if image_label[b] == 0:
                # Negative case: all pixels confirmed negative
                total_loss = total_loss + F.binary_cross_entropy(
                    p, torch.zeros_like(p)
                )
                count += 1
            else:
                # Positive case: object present somewhere
                # Case-level classification via noisy-OR
                s = 1.0 - (1.0 - p).prod()
                s = s.clamp(1e-7, 1.0 - 1e-7)
                l_cls = F.binary_cross_entropy(
                    s, torch.ones(1, device=pred.device)
                )

                # Voxel-level PU: suppress high-propensity, low-activation
                l_voxel_neg = (
                    e * F.binary_cross_entropy(
                        p, torch.zeros_like(p), reduction="none"
                    )
                ).mean()

                total_loss = total_loss + l_cls + self.pi_base * l_voxel_neg
                count += 1

        return total_loss / max(count, 1)


class SpatialSmoothnessLoss(nn.Module):
    """
    Total variation regularisation on the propensity map.

    L_smooth = TV(e) = mean(|∇e|_1)

    Encourages spatial coherence: nearby pixels should have similar propensity.
    """

    def forward(self, propensity: torch.Tensor) -> torch.Tensor:
        ndim = propensity.ndim - 2  # exclude B, C dims
        tv = torch.tensor(0.0, device=propensity.device)
        for d in range(2, 2 + ndim):
            slices_a = [slice(None)] * propensity.ndim
            slices_b = [slice(None)] * propensity.ndim
            slices_a[d] = slice(1, None)
            slices_b[d] = slice(None, -1)
            diff = propensity[tuple(slices_a)] - propensity[tuple(slices_b)]
            tv = tv + diff.abs().mean()
        return tv


class PropNetTrainingLoss(nn.Module):
    """
    Training loss for PropNet using synthetic missingness.

    L_prop = BCE(predicted_propensity, true_synthetic_propensity)

    Domain-agnostic: no vesselness ordering or anatomy constraints.
    """

    def forward(
        self,
        pred_propensity: torch.Tensor,
        true_propensity: torch.Tensor,
        positive_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positive_mask is not None:
            mask = positive_mask.bool()
            if mask.sum() == 0:
                return torch.tensor(0.0, device=pred_propensity.device)
            return F.binary_cross_entropy(
                pred_propensity[mask], true_propensity[mask]
            )
        return F.binary_cross_entropy(pred_propensity, true_propensity)


class RefinementLoss(nn.Module):
    """
    Propensity-weighted pseudo-label loss from EMA teacher (Stage 3).

    Only select confident pseudo-labels in high-propensity regions:
      - high propensity + high teacher activation → confident positive
      - high propensity + low teacher activation → confident negative
      - low propensity → skip (unreliable)
    """

    def __init__(
        self,
        propensity_threshold: float = 0.5,
        confidence_threshold: float = 0.7,
    ):
        super().__init__()
        self.propensity_threshold = propensity_threshold
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        propensity: torch.Tensor,
    ) -> torch.Tensor:
        # High propensity mask: regions where annotation status is reliable
        reliable = propensity > self.propensity_threshold

        # Confident teacher predictions
        confident_pos = (teacher_pred > self.confidence_threshold) & reliable
        confident_neg = (teacher_pred < (1 - self.confidence_threshold)) & reliable

        mask = confident_pos | confident_neg
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pseudo_label = (teacher_pred > 0.5).float()
        loss = F.binary_cross_entropy(pred[mask], pseudo_label[mask])
        return loss


class RefinementLoss(nn.Module):
    """
    Propensity-weighted pseudo-label refinement loss (Stage 3).

    Uses EMA teacher predictions + APN propensity to generate reliable
    pseudo-labels:
      - High propensity + high activation → confident positive
      - High propensity + low activation → confident negative
      - Low propensity → uncertain → skip (weight = 0)
    """

    def __init__(
        self,
        propensity_threshold: float = 0.5,
        confidence_threshold: float = 0.7,
    ):
        super().__init__()
        self.propensity_threshold = propensity_threshold
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        propensity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_pred: (B, 1, D, H, W) current model output.
            teacher_pred: (B, 1, D, H, W) EMA teacher output (detached).
            propensity: (B, 1, D, H, W) APN propensity.
        """
        # Generate pseudo-labels from teacher
        pseudo_pos = (teacher_pred > self.confidence_threshold).float()
        pseudo_neg = (teacher_pred < (1.0 - self.confidence_threshold)).float()

        # Only trust pseudo-labels in high-propensity regions
        high_prop = (propensity > self.propensity_threshold).float()

        # Confident positive: high propensity, teacher says positive
        conf_pos_mask = high_prop * pseudo_pos
        # Confident negative: high propensity, teacher says negative
        conf_neg_mask = high_prop * pseudo_neg

        # Build pseudo target
        weight_mask = conf_pos_mask + conf_neg_mask
        pseudo_target = conf_pos_mask  # 1 where confident positive, 0 elsewhere

        if weight_mask.sum() < 1:
            return torch.tensor(0.0, device=student_pred.device)

        loss = (
            weight_mask
            * F.binary_cross_entropy(
                student_pred, pseudo_target, reduction="none"
            )
        ).sum() / weight_mask.sum()

        return loss
