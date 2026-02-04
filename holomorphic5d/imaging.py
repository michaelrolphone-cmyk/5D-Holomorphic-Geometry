from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .manifold import ModularSurface


@dataclass(frozen=True)
class PixelManifoldEmbedding:
    """Container for pixel-derived 5D manifold embedding fields."""

    points: np.ndarray
    tau: np.ndarray
    depth_field: np.ndarray
    spectral_shift: np.ndarray
    phase_field: np.ndarray
    temporal_field: np.ndarray


def _neighbor_sum(field: np.ndarray) -> np.ndarray:
    total = np.zeros_like(field, dtype=float)
    for dx in (-1, 0, 1):
        shifted_x = np.roll(field, dx, axis=0)
        for dy in (-1, 0, 1):
            total += np.roll(shifted_x, dy, axis=1)
    return total


def _normalize_field(field: np.ndarray, eps: float) -> np.ndarray:
    min_val = np.min(field)
    max_val = np.max(field)
    scale = max(max_val - min_val, eps)
    return (field - min_val) / scale


def decode_pixels_to_manifold(
    image: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    radius_y: float = 1.0,
    depth_scale: float = 1.0,
    spectral_scale: float = 1.0,
    phase_scale: float = 1.0,
    temporal_scale: float = 1.0,
    eps: float = 1e-8,
    project_tau: bool = True,
) -> PixelManifoldEmbedding:
    """Decode RGB pixels into a 5D manifold embedding with modular parameters.

    The decoder treats neighborhood luminance variance as a depth proxy, hue
    imbalance as a spectral-shift proxy, gradients as phase angles, and the
    Laplacian magnitude as a temporal-field proxy.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must be an array with shape (H, W, 3)")

    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    local_mean = _neighbor_sum(luminance) / 9.0
    local_mean_sq = _neighbor_sum(luminance**2) / 9.0
    variance = np.maximum(local_mean_sq - local_mean**2, 0.0)
    depth_field = _normalize_field(variance, eps)

    spectral_shift = (r - b) / (r + b + eps)

    grad_x = 0.5 * (np.roll(luminance, -1, axis=1) - np.roll(luminance, 1, axis=1))
    grad_y = 0.5 * (np.roll(luminance, -1, axis=0) - np.roll(luminance, 1, axis=0))
    phase_field = (np.arctan2(grad_y, grad_x) + 2 * np.pi) % (2 * np.pi)

    laplacian = (
        np.roll(luminance, 1, axis=0)
        + np.roll(luminance, -1, axis=0)
        + np.roll(luminance, 1, axis=1)
        + np.roll(luminance, -1, axis=1)
        - 4 * luminance
    )
    temporal_field = _normalize_field(np.abs(laplacian), eps)

    height, width = luminance.shape
    dx, dy, dz = spacing
    x1 = np.arange(height) * dx
    x2 = np.arange(width) * dy
    x1_grid, x2_grid = np.meshgrid(x1, x2, indexing="ij")
    x3_grid = depth_scale * depth_field * dz
    y_grid = (phase_scale * phase_field / (2 * np.pi)) * (2 * np.pi * radius_y)
    t_grid = temporal_scale * temporal_field

    points = np.stack([x1_grid, x2_grid, x3_grid, y_grid, t_grid], axis=-1)

    tau_real = spectral_scale * spectral_shift
    tau_imag = 0.1 + depth_scale * depth_field + temporal_scale * temporal_field
    tau = tau_real + 1j * tau_imag
    if project_tau:
        tau = ModularSurface().project_to_fundamental_domain(tau)

    return PixelManifoldEmbedding(
        points=points,
        tau=tau,
        depth_field=depth_field,
        spectral_shift=spectral_shift,
        phase_field=phase_field,
        temporal_field=temporal_field,
    )
