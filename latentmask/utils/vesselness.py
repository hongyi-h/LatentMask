"""
Vesselness filter (Frangi) for 3D CTPA volumes.

Provides the vascular anatomy prior used as APN input.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def _hessian_eigenvalues_3d(
    volume: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sorted eigenvalues of the 3D Hessian at scale sigma."""
    smoothed = gaussian_filter(volume.astype(np.float64), sigma=sigma)

    # Second-order partial derivatives via finite differences
    gz, gy, gx = np.gradient(smoothed)
    gzz, gzy, gzx = np.gradient(gz)
    _, gyy, gyx = np.gradient(gy)
    _, _, gxx = np.gradient(gx)

    # For each voxel, build the 3x3 Hessian and find eigenvalues
    shape = volume.shape
    eig = np.zeros((*shape, 3), dtype=np.float64)

    # Vectorised: build upper triangle entries
    H = np.stack(
        [gxx, gyx, gzx, gyx, gyy, gzy, gzx, gzy, gzz], axis=-1
    ).reshape(*shape, 3, 3)

    # np.linalg.eigvalsh is for symmetric matrices, returns sorted ascending
    eig = np.linalg.eigvalsh(H)  # (..., 3), sorted |λ1| ≤ |λ2| ≤ |λ3|
    return eig[..., 0], eig[..., 1], eig[..., 2]


def frangi_vesselness_3d(
    volume: np.ndarray,
    sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    alpha: float = 0.5,
    beta: float = 0.5,
    c_fraction: float = 0.5,
    bright_on_dark: bool = True,
) -> np.ndarray:
    """
    Multi-scale Frangi vesselness filter for 3D volumes.

    Args:
        volume: 3D numpy array (D, H, W).
        sigmas: Gaussian scales for multi-scale analysis.
        alpha: Sensitivity to plate-like structures (Ra).
        beta: Sensitivity to blob-like structures (Rb).
        c_fraction: Frobenius norm sensitivity, as fraction of max Frobenius.
        bright_on_dark: If True, detect bright tubular structures on dark bg.

    Returns:
        vesselness: Same shape as volume, in [0, 1].
    """
    vesselness = np.zeros_like(volume, dtype=np.float64)

    for sigma in sigmas:
        lam1, lam2, lam3 = _hessian_eigenvalues_3d(volume, sigma)

        # Sort by absolute value: |λ1| ≤ |λ2| ≤ |λ3|
        abslam = np.stack([np.abs(lam1), np.abs(lam2), np.abs(lam3)], axis=-1)
        idx = np.argsort(abslam, axis=-1)
        lams = np.stack([lam1, lam2, lam3], axis=-1)
        sorted_lam = np.take_along_axis(lams, idx, axis=-1)
        l1, l2, l3 = sorted_lam[..., 0], sorted_lam[..., 1], sorted_lam[..., 2]

        # Vesselness conditions: for bright vessels, λ2 < 0 and λ3 < 0
        if bright_on_dark:
            mask = (l2 < 0) & (l3 < 0)
        else:
            mask = (l2 > 0) & (l3 > 0)

        al2, al3 = np.abs(l2), np.abs(l3)
        al2 = np.clip(al2, 1e-10, None)
        al3 = np.clip(al3, 1e-10, None)

        Ra = al2 / al3  # plate vs line
        Rb = np.abs(l1) / np.sqrt(al2 * al3)  # blob vs line
        S = np.sqrt(l1**2 + l2**2 + l3**2)  # Frobenius (structuredness)

        c = c_fraction * S.max() if S.max() > 0 else 1.0

        v = (
            (1.0 - np.exp(-Ra**2 / (2 * alpha**2)))
            * np.exp(-Rb**2 / (2 * beta**2))
            * (1.0 - np.exp(-S**2 / (2 * c**2)))
        )
        v[~mask] = 0.0
        vesselness = np.maximum(vesselness, v)

    # Normalise to [0, 1]
    vmax = vesselness.max()
    if vmax > 0:
        vesselness /= vmax

    return vesselness.astype(np.float32)


def compute_vesselness_tensor(
    ct_volume: torch.Tensor,
    sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """
    Convenience: compute vesselness from a (B, 1, D, H, W) CT tensor.

    Returns:
        vesselness: (B, 1, D, H, W) tensor in [0, 1].
    """
    device = ct_volume.device
    B = ct_volume.shape[0]
    results = []
    for b in range(B):
        vol_np = ct_volume[b, 0].cpu().numpy()
        v = frangi_vesselness_3d(vol_np, sigmas=sigmas)
        results.append(torch.from_numpy(v).unsqueeze(0))  # (1, D, H, W)

    return torch.stack(results, dim=0).to(device)  # (B, 1, D, H, W)
