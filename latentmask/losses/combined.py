"""
Combined LatentMask loss that wraps all component losses.

Handles the stage-dependent weighting:
  Stage 1: L_pix + L_prop + L_smooth (warm-up + PropNet pre-training)
  Stage 2: L_pix + L_box + L_img + L_prop + L_smooth (joint PU)
  Stage 3: L_pix + L_box + L_img + L_prop + L_smooth + L_ref (refinement)
"""

import torch
import torch.nn as nn

from latentmask.losses.pu_losses import (
    BoxPULoss,
    ImagePULoss,
    PixelPULoss,
    PropNetTrainingLoss,
    RefinementLoss,
    SpatialSmoothnessLoss,
)


class LatentMaskLoss(nn.Module):
    """Unified loss for all training stages."""

    def __init__(
        self,
        pi_base: float = 0.15,
        lambda_pix: float = 1.0,
        lambda_box: float = 1.0,
        lambda_img: float = 0.5,
        lambda_prop: float = 0.5,
        lambda_smooth: float = 0.1,
        lambda_ref: float = 0.4,
        refinement_propensity_thresh: float = 0.5,
        refinement_confidence_thresh: float = 0.7,
    ):
        super().__init__()

        self.pixel_loss = PixelPULoss()
        self.box_loss = BoxPULoss(pi_base=pi_base)
        self.image_loss = ImagePULoss(pi_base=pi_base)
        self.smoothness_loss = SpatialSmoothnessLoss()
        self.prop_loss = PropNetTrainingLoss()
        self.refinement_loss = RefinementLoss(
            propensity_threshold=refinement_propensity_thresh,
            confidence_threshold=refinement_confidence_thresh,
        )

        self.lambda_pix = lambda_pix
        self.lambda_box = lambda_box
        self.lambda_img = lambda_img
        self.lambda_prop = lambda_prop
        self.lambda_smooth = lambda_smooth
        self.lambda_ref = lambda_ref

    def forward(
        self,
        pred: torch.Tensor,
        propensity: torch.Tensor,
        batch_type: str,
        stage: int,
        pixel_target: torch.Tensor | None = None,
        box_mask: torch.Tensor | None = None,
        box_target: torch.Tensor | None = None,
        image_label: torch.Tensor | None = None,
        synthetic_propensity: torch.Tensor | None = None,
        positive_mask: torch.Tensor | None = None,
        teacher_pred: torch.Tensor | None = None,
        use_smoothness: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred: (B, 1, *spatial) model prediction (sigmoid output).
            propensity: (B, 1, *spatial) PropNet output.
            batch_type: One of 'pixel', 'box', 'image'.
            stage: Training stage (1, 2, or 3).
            pixel_target: GT for pixel-labeled data.
            box_mask: Binary box region mask.
            box_target: GT inside boxes.
            image_label: (B,) image-level label.
            synthetic_propensity: GT propensity from synthetic missingness.
            positive_mask: Mask of positive pixels (for PropNet training).
            teacher_pred: EMA teacher predictions (detached).
            use_smoothness: Whether to apply spatial smoothness regularisation.

        Returns:
            Dict of {'total': scalar, 'l_pix': scalar, ...}
        """
        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        if batch_type == "pixel" and pixel_target is not None:
            l_pix = self.pixel_loss(pred, pixel_target)
            losses["l_pix"] = l_pix
            total = total + self.lambda_pix * l_pix

            # PropNet training on pixel data (synthetic missingness)
            if synthetic_propensity is not None:
                l_prop = self.prop_loss(
                    propensity, synthetic_propensity, positive_mask
                )
                losses["l_prop"] = l_prop
                total = total + self.lambda_prop * l_prop

        elif batch_type == "box" and stage >= 2:
            if box_mask is not None and box_target is not None:
                l_box = self.box_loss(pred, box_mask, box_target, propensity)
                losses["l_box"] = l_box
                total = total + self.lambda_box * l_box

        elif batch_type == "image" and stage >= 2:
            if image_label is not None:
                l_img = self.image_loss(pred, image_label, propensity)
                losses["l_img"] = l_img
                total = total + self.lambda_img * l_img

        # Spatial smoothness regularisation (all stages)
        if use_smoothness:
            l_smooth = self.smoothness_loss(propensity)
            losses["l_smooth"] = l_smooth
            total = total + self.lambda_smooth * l_smooth

        # Refinement (stage 3)
        if stage >= 3 and teacher_pred is not None:
            l_ref = self.refinement_loss(pred, teacher_pred, propensity)
            losses["l_ref"] = l_ref
            total = total + self.lambda_ref * l_ref

        losses["total"] = total
        return losses
