from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .fundamental import FundamentalGeometry5D


@dataclass(frozen=True)
class ModularHilbertSpace:
    """Discrete Hilbert space on the modular surface fundamental domain."""

    x_grid: np.ndarray
    y_grid: np.ndarray

    def measure(self) -> np.ndarray:
        """Return hyperbolic measure dmu = dx dy / y^2 on the grid."""
        return 1.0 / (self.y_grid**2)

    def inner_product(self, f: np.ndarray, g: np.ndarray) -> complex:
        """Discrete approximation of the modular-surface L2 inner product."""
        weight = self.measure()
        return np.sum(f * np.conjugate(g) * weight)


def hilbert_space_from_geometry(
    geometry: "FundamentalGeometry5D", tau_grid: np.ndarray
) -> ModularHilbertSpace:
    """Build a modular Hilbert space by projecting tau onto the fundamental domain."""
    projected = geometry.project_tau(tau_grid)
    return ModularHilbertSpace(
        x_grid=np.real(projected),
        y_grid=np.imag(projected),
    )


def hyperbolic_measure(y: np.ndarray) -> np.ndarray:
    """Compute the hyperbolic measure density 1/y^2."""
    y = np.asarray(y, dtype=float)
    return 1.0 / (y**2)


def modular_operator_h(field: np.ndarray, y_grid: np.ndarray, dy: float) -> np.ndarray:
    """Compute H = -i y d/dy using central differences along y-axis."""
    field = np.asarray(field, dtype=np.complex128)
    y_grid = np.asarray(y_grid, dtype=float)
    d_field = np.gradient(field, dy, axis=-1, edge_order=2)
    return -1.0j * y_grid * d_field


def theta_function(t: np.ndarray, n_terms: int = 50) -> np.ndarray:
    """Compute Jacobi theta function sum_{n in Z} exp(-pi n^2 t)."""
    t = np.asarray(t, dtype=float)
    n = np.arange(-n_terms, n_terms + 1, dtype=float)
    return np.sum(np.exp(-np.pi * (n**2) * t[..., None]), axis=-1)


def theta_functional_equation(t: np.ndarray, n_terms: int = 50) -> np.ndarray:
    """Evaluate theta(t) - t^{-1/2} theta(1/t) as a symmetry check."""
    t = np.asarray(t, dtype=float)
    theta_t = theta_function(t, n_terms=n_terms)
    theta_inv = theta_function(1.0 / t, n_terms=n_terms)
    return theta_t - theta_inv / np.sqrt(t)


def mellin_zeta(s: complex, t: np.ndarray, n_terms: int = 50) -> complex:
    """Approximate Mellin transform integral for zeta(s)."""
    t = np.asarray(t, dtype=float)
    theta = theta_function(t, n_terms=n_terms)
    integrand = (theta - 1.0) * t ** (s / 2.0 - 1.0) / 2.0
    return np.trapezoid(integrand, t)


def zeta_regularized_determinant(eigenvalues: np.ndarray) -> complex:
    """Compute determinant via zeta regularization for finite spectrum."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
    eigenvalues = eigenvalues[eigenvalues != 0]
    if eigenvalues.size == 0:
        return 1.0
    return np.exp(np.sum(np.log(eigenvalues)))


def holomorphic_spectrum_fourier(
    k_coords: np.ndarray,
    sigma: float = 0.6,
    center: np.ndarray | None = None,
    coupling: float = 1.0,
    twist: float = 0.35,
) -> np.ndarray:
    """Evaluate a 5D holomorphic spectrum via an analytic Fourier transform.

    Args:
        k_coords: Array of frequency-domain coordinates with trailing dimension size 5.
        sigma: Gaussian width controlling the Fourier envelope.
        center: Optional 5D center shift for the spectrum.
        coupling: Phase coupling between the first four axes.
        twist: Phase twist applied along the fifth axis.
    """
    k_coords = np.asarray(k_coords, dtype=float)
    if k_coords.shape[-1] != 5:
        raise ValueError("k_coords must have shape (..., 5)")
    center = np.zeros(5, dtype=float) if center is None else np.asarray(center, dtype=float)
    if center.shape != (5,):
        raise ValueError("center must have shape (5,)")

    shifted = k_coords - center
    norm_sq = np.sum(shifted**2, axis=-1)
    envelope = np.exp(-2.0 * (np.pi**2) * (sigma**2) * norm_sq)
    phase = 2.0 * np.pi * (
        coupling * (shifted[..., 0] * shifted[..., 1] - shifted[..., 2] * shifted[..., 3])
        + twist * shifted[..., 4]
    )
    return envelope * np.exp(1.0j * phase)


def discrete_five_d_fourier(
    points: np.ndarray, weights: np.ndarray, k_coords: np.ndarray
) -> np.ndarray:
    """Compute a discrete 5D Fourier transform over weighted points."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    k_coords = np.asarray(k_coords, dtype=float)
    if points.ndim != 2 or points.shape[1] != 5:
        raise ValueError("points must have shape (N, 5)")
    if weights.shape != (points.shape[0],):
        raise ValueError("weights must have shape (N,)")
    if k_coords.shape[-1] != 5:
        raise ValueError("k_coords must have shape (..., 5)")
    phase = -2.0j * np.pi * np.tensordot(k_coords, points, axes=([k_coords.ndim - 1], [1]))
    return np.tensordot(np.exp(phase), weights, axes=([phase.ndim - 1], [0]))
