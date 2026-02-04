from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .manifold import HolomorphicManifold5D


@dataclass(frozen=True)
class ImpedanceBoundary:
    """Simple impedance boundary model with a scalar attenuation factor."""

    factor: float


def apply_impedance_boundary(field: np.ndarray, impedance: ImpedanceBoundary) -> np.ndarray:
    """Apply impedance attenuation on the boundary slices of the 3D base."""
    updated = field.copy()
    updated[0, :, :, :] *= impedance.factor
    updated[-1, :, :, :] *= impedance.factor
    updated[:, 0, :, :] *= impedance.factor
    updated[:, -1, :, :] *= impedance.factor
    updated[:, :, 0, :] *= impedance.factor
    updated[:, :, -1, :] *= impedance.factor
    return updated


def modular_transform(tau: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a PSL(2,Z) transformation to complex tau values."""
    tau = np.asarray(tau, dtype=np.complex128)
    matrix = np.asarray(matrix, dtype=float)
    a, b, c, d = matrix.ravel()
    denom = c * tau + d
    return (a * tau + b) / denom


def sample_modular_orbit(tau: complex, steps: int) -> np.ndarray:
    """Generate a modular orbit using the S and T generators."""
    orbit = np.zeros(steps, dtype=np.complex128)
    current = complex(tau)
    for idx in range(steps):
        orbit[idx] = current
        if idx % 2 == 0:
            current = current + 1
        else:
            current = -1.0 / current
    return orbit


def simulate_diffusion(
    manifold: HolomorphicManifold5D,
    field: np.ndarray,
    dt: float,
    steps: int,
    diffusivity: float = 1.0,
    impedance: ImpedanceBoundary | None = None,
) -> np.ndarray:
    """Simulate diffusion on the 5D manifold using explicit Euler steps."""
    state = np.array(field, dtype=float)
    history = np.zeros((steps + 1,) + state.shape, dtype=float)
    history[0] = state
    for step in range(1, steps + 1):
        lap = manifold.laplacian5(state)
        state = state + dt * diffusivity * lap
        if impedance is not None:
            state = apply_impedance_boundary(state, impedance)
        history[step] = state
    return history
