from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def hyperbolic_measure(y: np.ndarray) -> np.ndarray:
    """Compute the hyperbolic measure density 1/y^2."""
    y = np.asarray(y, dtype=float)
    return 1.0 / (y**2)


def modular_operator_h(field: np.ndarray, y_grid: np.ndarray, dy: float) -> np.ndarray:
    """Compute H = -i y d/dy using central differences along y-axis."""
    field = np.asarray(field, dtype=np.complex128)
    y_grid = np.asarray(y_grid, dtype=float)
    d_field = (np.roll(field, -1, axis=-1) - np.roll(field, 1, axis=-1)) / (2.0 * dy)
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
    return np.trapz(integrand, t)


def zeta_regularized_determinant(eigenvalues: np.ndarray) -> complex:
    """Compute determinant via zeta regularization for finite spectrum."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
    eigenvalues = eigenvalues[eigenvalues != 0]
    if eigenvalues.size == 0:
        return 1.0
    return np.exp(np.sum(np.log(eigenvalues)))
