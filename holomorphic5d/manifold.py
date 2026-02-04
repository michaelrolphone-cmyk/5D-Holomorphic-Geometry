from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModularSurface:
    """Utilities for the modular surface H/PSL(2,Z)."""

    def project_to_fundamental_domain(self, tau: np.ndarray) -> np.ndarray:
        """Project complex points to a standard fundamental domain.

        The algorithm iteratively applies tau -> tau + n and tau -> -1/tau
        until |tau| >= 1 and |Re(tau)| <= 1/2.
        """
        tau = np.asarray(tau, dtype=np.complex128)
        projected = tau.copy()
        for _ in range(20):
            projected = projected - np.round(projected.real)
            mask = np.abs(projected) < 1
            if not np.any(mask):
                break
            projected[mask] = -1.0 / projected[mask]
        return projected

    def in_fundamental_domain(self, tau: np.ndarray) -> np.ndarray:
        """Return boolean mask for points in the standard fundamental domain."""
        tau = np.asarray(tau, dtype=np.complex128)
        return (np.abs(tau) >= 1) & (np.abs(tau.real) <= 0.5) & (tau.imag > 0)


@dataclass(frozen=True)
class HolomorphicManifold5D:
    """Discrete representation of R^3 x S^1_y with a modular parameter."""

    grid_shape: tuple[int, int, int, int]
    spacing: tuple[float, float, float, float]
    radius_y: float
    modular_surface: ModularSurface = ModularSurface()

    @classmethod
    def from_fundamental_geometry(
        cls,
        geometry: "FundamentalGeometry5D",
        grid_shape: tuple[int, int, int, int],
        spacing_base: tuple[float, float, float],
    ) -> "HolomorphicManifold5D":
        """Construct a manifold using the fundamental geometry for the fiber spacing."""
        dx, dy, dz = spacing_base
        _, _, _, ny_fiber = grid_shape
        dy_fiber = geometry.fiber_circumference() / ny_fiber
        return cls(
            grid_shape=grid_shape,
            spacing=(dx, dy, dz, dy_fiber),
            radius_y=geometry.radius_y,
            modular_surface=geometry.modular_surface,
        )

    def grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a meshgrid for x1, x2, x3, and y coordinates."""
        nx, ny, nz, ny_fiber = self.grid_shape
        dx, dy, dz, dy_fiber = self.spacing
        x1 = np.arange(nx) * dx
        x2 = np.arange(ny) * dy
        x3 = np.arange(nz) * dz
        y = np.linspace(0.0, 2 * np.pi * self.radius_y, ny_fiber, endpoint=False)
        return np.meshgrid(x1, x2, x3, y, indexing="ij")

    def laplacian5(self, field: np.ndarray) -> np.ndarray:
        """Compute a 5D Laplacian with periodicity along the fiber axis."""
        dx, dy, dz, dy_fiber = self.spacing
        lap = np.zeros_like(field)
        for axis, step in enumerate([dx, dy, dz]):
            lap += (
                np.roll(field, 1, axis=axis)
                - 2.0 * field
                + np.roll(field, -1, axis=axis)
            ) / (step**2)
        lap += (
            np.roll(field, 1, axis=3)
            - 2.0 * field
            + np.roll(field, -1, axis=3)
        ) / (dy_fiber**2)
        return lap
