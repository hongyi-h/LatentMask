"""
Anatomical Propensity Network (APN).

Lightweight 3D CNN that estimates per-voxel annotation propensity
conditioned on vascular anatomy features and encoder features.

Architecture:
    Input: concat(encoder_features_stage3, vesselness_map)
    → 3×3×3 Conv, BN, ReLU (64 channels)
    → 3×3×3 Conv, BN, ReLU (32 channels)
    → 1×1×1 Conv → Sigmoid → e(v) ∈ (ε, 1−ε)

Parameters: ~0.2M (negligible compared to ~31M backbone).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomicalPropensityNetwork(nn.Module):
    """Estimates voxel-wise annotation propensity from encoder features + vesselness."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: tuple[int, int] = (64, 32),
        epsilon: float = 0.01,
    ):
        """
        Args:
            in_channels: Number of channels from encoder stage 3 + 1 (vesselness).
            hidden_channels: Intermediate channel dims.
            epsilon: Clamp range for propensity output (ε, 1−ε).
        """
        super().__init__()
        self.epsilon = epsilon

        h1, h2 = hidden_channels
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, h1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(h1),
            nn.ReLU(inplace=True),
            nn.Conv3d(h1, h2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(h2),
            nn.ReLU(inplace=True),
            nn.Conv3d(h2, 1, kernel_size=1, bias=True),
        )

    def forward(
        self,
        encoder_features: torch.Tensor,
        vesselness: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features: (B, C_enc, D, H, W) from encoder stage 3.
            vesselness: (B, 1, D', H', W') vesselness map (may need resizing).

        Returns:
            propensity: (B, 1, D, H, W) in (ε, 1−ε).
        """
        spatial = encoder_features.shape[2:]
        if vesselness.shape[2:] != spatial:
            vesselness = F.interpolate(
                vesselness, size=spatial, mode="trilinear", align_corners=False
            )

        x = torch.cat([encoder_features, vesselness], dim=1)
        logits = self.block(x)
        propensity = torch.sigmoid(logits)
        propensity = propensity.clamp(self.epsilon, 1.0 - self.epsilon)
        return propensity
