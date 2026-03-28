"""
Propensity Network (PropNet).

Domain-agnostic lightweight CNN that estimates per-pixel annotation propensity
from encoder features alone — no task-specific priors (e.g. vesselness) required.

Architecture:
    Input: encoder_features (from stage k, e.g. k=3)
    → Conv (64) → BN → ReLU
    → Conv (32) → BN → ReLU
    → Conv (1) → Sigmoid → clamp(ε, 1−ε)

Supports both 2D (Conv2d for natural images) and 3D (Conv3d for volumetric data).
Parameters: ~0.2M (3D) or ~0.05M (2D), negligible vs backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PropensityNetwork(nn.Module):
    """Estimates pixel/voxel-wise annotation propensity from encoder features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: tuple[int, int] = (64, 32),
        epsilon: float = 0.01,
        spatial_dims: int = 3,
    ):
        """
        Args:
            in_channels: Number of channels from the encoder stage.
            hidden_channels: Intermediate channel dims.
            epsilon: Clamp range for propensity output (ε, 1−ε).
            spatial_dims: 2 for 2D images, 3 for 3D volumes.
        """
        super().__init__()
        self.epsilon = epsilon
        self.spatial_dims = spatial_dims

        conv_cls = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        bn_cls = nn.BatchNorm3d if spatial_dims == 3 else nn.BatchNorm2d

        h1, h2 = hidden_channels
        self.block = nn.Sequential(
            conv_cls(in_channels, h1, kernel_size=3, padding=1, bias=False),
            bn_cls(h1),
            nn.ReLU(inplace=True),
            conv_cls(h1, h2, kernel_size=3, padding=1, bias=False),
            bn_cls(h2),
            nn.ReLU(inplace=True),
            conv_cls(h2, 1, kernel_size=1, bias=True),
        )

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_features: (B, C_enc, *spatial) from encoder stage k.

        Returns:
            propensity: (B, 1, *spatial) in (ε, 1−ε).
        """
        logits = self.block(encoder_features)
        propensity = torch.sigmoid(logits)
        return propensity.clamp(self.epsilon, 1.0 - self.epsilon)


class ConstantPropensity(nn.Module):
    """Fixed propensity for ablation (uniform e baseline)."""

    def __init__(self, value: float = 0.5, spatial_dims: int = 3):
        super().__init__()
        self.value = value
        self.spatial_dims = spatial_dims

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (encoder_features.shape[0], 1, *encoder_features.shape[2:]),
            self.value,
            device=encoder_features.device,
            dtype=encoder_features.dtype,
        )
