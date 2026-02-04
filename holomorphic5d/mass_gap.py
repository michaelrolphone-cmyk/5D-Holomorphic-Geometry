from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .fundamental import FundamentalGeometry5D


@dataclass(frozen=True)
class FiberGeometry:
    """Geometry of the compact fiber S^1_y."""

    radius_y: float

    @property
    def circumference(self) -> float:
        return 2.0 * np.pi * self.radius_y


def kk_mode_masses(radius_y: float, modes: np.ndarray) -> np.ndarray:
    """Compute KK masses m_n = |n|/R_y."""
    modes = np.asarray(modes, dtype=float)
    return np.abs(modes) / radius_y


def poincare_constant(radius_y: float) -> float:
    """Return the Poincare constant L^2/(4pi^2) for S^1 of radius R_y."""
    length = 2.0 * np.pi * radius_y
    return (length**2) / (4.0 * np.pi**2)


def poincare_lower_bound(radius_y: float) -> float:
    """Return the spectral lower bound 4pi^2/L^2 = 1/R_y^2."""
    length = 2.0 * np.pi * radius_y
    return (4.0 * np.pi**2) / (length**2)


def zero_mode_removed(field: np.ndarray, axis: int = -1) -> np.ndarray:
    """Remove the zero mode by subtracting the mean along the fiber axis."""
    mean = np.mean(field, axis=axis, keepdims=True)
    return field - mean


def check_zero_mode(field: np.ndarray, axis: int = -1, atol: float = 1e-8) -> bool:
    """Return True if the field has near-zero mean along the fiber axis."""
    mean = np.mean(field, axis=axis)
    return np.all(np.abs(mean) <= atol)


def mass_gap_bound(radius_y: float, coupling_scale: float = 1.0) -> float:
    """Compute Delta >= sqrt(coupling_scale)/R_y (coupling_scale defaults to 1)."""
    return np.sqrt(coupling_scale) / radius_y


def mass_gap_from_geometry(
    geometry: FundamentalGeometry5D, coupling_scale: float = 1.0
) -> float:
    """Compute the mass gap using the fundamental geometry."""
    return mass_gap_bound(geometry.radius_y, coupling_scale=coupling_scale)


def mass_gap_mev(radius_y: float, hbar_c_mev_fm: float = 197.3269804) -> float:
    """Compute Delta (MeV) using hbar c in MeV*fm and radius in meters."""
    radius_fm = radius_y * 1e15
    return hbar_c_mev_fm / radius_fm
