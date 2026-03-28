"""
Synthetic missingness generator for PropNet training.

Domain-agnostic: uses three corruption patterns based on object geometry,
not task-specific priors like vesselness.

Patterns:
  1. Scale-dependent drop: p_drop(C_i) = α / sqrt(|C_i|)
  2. Boundary erosion: random morphological erosion r ~ U(1, R_max)
  3. Component drop: p_comp ∝ 1/|C_i|

These are applied in random combination to pixel-labeled data to generate
ground-truth propensity maps for PropNet training.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import binary_erosion, generate_binary_structure


class SyntheticMissingnessGenerator:
    """
    Domain-agnostic synthetic label corruption for PropNet training.

    Generates corrupted labels and ground-truth propensity maps from
    pixel-labeled data, simulating realistic annotation biases.
    """

    def __init__(
        self,
        alpha: float = 5.0,
        r_max: int = 5,
        component_drop_scale: float = 1.0,
        pattern: str = "all",
        spatial_dims: int = 3,
        seed: int | None = None,
    ):
        """
        Args:
            alpha: Scale-dependent drop severity. Higher = more small-object drop.
            r_max: Maximum erosion radius for boundary erosion.
            component_drop_scale: Scaling factor for component drop probability.
            pattern: Which patterns to apply. One of:
                     'all', 'scale_only', 'boundary_only', 'component_only',
                     'scale_boundary', 'uniform' (no corruption, propensity=1).
            spatial_dims: 2 for images, 3 for volumes.
            seed: Random seed for reproducibility.
        """
        self.alpha = alpha
        self.r_max = r_max
        self.component_drop_scale = component_drop_scale
        self.pattern = pattern
        self.spatial_dims = spatial_dims
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        label: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply synthetic missingness to a single label array.

        Args:
            label: (*spatial) binary segmentation mask (2D or 3D).

        Returns:
            corrupted_label: label with some positive pixels dropped.
            true_propensity: (*spatial) retention probability per pixel.
                             This is the PropNet training target.
        """
        if self.pattern == "uniform":
            return label.copy(), np.ones_like(label, dtype=np.float32)

        corrupted = label.copy().astype(np.float32)
        propensity = np.ones_like(label, dtype=np.float32)

        # Find connected components
        labeled_arr, num_components = ndimage.label(label > 0)
        if num_components == 0:
            return corrupted, propensity

        component_sizes = ndimage.sum(
            label > 0, labeled_arr, range(1, num_components + 1)
        )

        use_scale = self.pattern in ("all", "scale_only", "scale_boundary")
        use_boundary = self.pattern in ("all", "boundary_only", "scale_boundary")
        use_component = self.pattern in ("all", "component_only")

        # --- Pattern 1: Scale-dependent drop ---
        if use_scale:
            for comp_id in range(1, num_components + 1):
                comp_mask = labeled_arr == comp_id
                size = component_sizes[comp_id - 1]
                if size < 1:
                    continue
                p_drop = min(self.alpha / np.sqrt(size), 0.95)
                # Per-pixel random drop within this component
                rand_vals = self.rng.rand(*label.shape)
                drop_pixels = comp_mask & (rand_vals < p_drop)
                corrupted[drop_pixels] = 0
                propensity[comp_mask] *= (1.0 - p_drop)

        # --- Pattern 2: Boundary erosion ---
        if use_boundary:
            r = self.rng.randint(1, self.r_max + 1)
            struct = generate_binary_structure(self.spatial_dims, 1)
            eroded = binary_erosion(
                label > 0, structure=struct, iterations=r
            ).astype(np.float32)
            # Pixels in original label but not in eroded = boundary zone
            boundary_zone = (label > 0) & (~(eroded > 0))
            # Drop boundary pixels with probability proportional to erosion depth
            boundary_drop_prob = 0.7  # high drop for boundary
            rand_vals = self.rng.rand(*label.shape)
            drop_boundary = boundary_zone & (rand_vals < boundary_drop_prob)
            corrupted[drop_boundary] = 0
            propensity[boundary_zone] *= (1.0 - boundary_drop_prob)

        # --- Pattern 3: Component drop ---
        if use_component:
            for comp_id in range(1, num_components + 1):
                comp_mask = labeled_arr == comp_id
                size = component_sizes[comp_id - 1]
                if size < 1:
                    continue
                p_comp_drop = min(
                    self.component_drop_scale / (size + 1.0), 0.9
                )
                if self.rng.rand() < p_comp_drop:
                    corrupted[comp_mask] = 0
                    propensity[comp_mask] = 0.0

        # Ensure propensity is in valid range
        propensity = np.clip(propensity, 0.01, 1.0)

        return corrupted, propensity

    def generate_batch(
        self,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply synthetic missingness to a batch of labels.

        Args:
            labels: (B, 1, *spatial) binary segmentation masks.

        Returns:
            corrupted_labels: (B, 1, *spatial) corrupted masks.
            true_propensity: (B, 1, *spatial) retention probability maps.
        """
        B = labels.shape[0]
        device = labels.device

        corrupted_list = []
        propensity_list = []

        for b in range(B):
            lbl = labels[b, 0].cpu().numpy()
            c, p = self.generate(lbl)
            corrupted_list.append(
                torch.from_numpy(c).unsqueeze(0)
            )
            propensity_list.append(
                torch.from_numpy(p).unsqueeze(0)
            )

        corrupted = torch.stack(corrupted_list, dim=0).to(device)
        propensity = torch.stack(propensity_list, dim=0).to(device)
        return corrupted, propensity
                    # Create bounding box around component
                    slices = ndimage.find_objects(labeled_arr == comp_id)
                    if slices and slices[0] is not None:
                        s = slices[0]
                        bm[s[0], s[1], s[2]] = 1
                        tp[comp_mask] = 1.0 - drop_prob

            box_mask[b, 0] = torch.from_numpy(bm).to(device)
            true_propensity[b, 0] = torch.from_numpy(tp).to(device)

        return box_mask, true_propensity
