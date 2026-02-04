from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HolomorphicFormulas:
    """Shared formula definitions used across the library."""

    @staticmethod
    def kk_mode_masses(radius_y: float, modes: np.ndarray) -> np.ndarray:
        """Compute KK masses m_n = |n|/R_y."""
        modes = np.asarray(modes, dtype=float)
        return np.abs(modes) / radius_y

    @staticmethod
    def poincare_constant(radius_y: float) -> float:
        """Return the Poincare constant L^2/(4pi^2) for S^1 of radius R_y."""
        length = 2.0 * np.pi * radius_y
        return (length**2) / (4.0 * np.pi**2)

    @staticmethod
    def poincare_lower_bound(radius_y: float) -> float:
        """Return the spectral lower bound 4pi^2/L^2 = 1/R_y^2."""
        length = 2.0 * np.pi * radius_y
        return (4.0 * np.pi**2) / (length**2)

    @staticmethod
    def electron_capacitance(charge_q: float, electron_mass: float, c: float) -> float:
        """Return the electron capacitance q^2/(2 m c^2)."""
        return charge_q**2 / (2.0 * electron_mass * c**2)

    @staticmethod
    def spherical_capacitance(radius: float, epsilon0: float) -> float:
        """Return the spherical capacitance 2*pi*epsilon0*radius^2."""
        return 2.0 * np.pi * epsilon0 * radius**2

    @staticmethod
    def universe_capacitance(universe_radius: float, epsilon0: float) -> float:
        """Return the universe capacitance 4*pi*epsilon0*R."""
        return 4.0 * np.pi * epsilon0 * universe_radius

    @staticmethod
    def bridge_capacitance(
        universe_radius: float, epsilon0: float, force_ratio: float
    ) -> float:
        """Return the bridge capacitance C_u/(4*pi^2*force_ratio)."""
        return HolomorphicFormulas.universe_capacitance(
            universe_radius, epsilon0
        ) / (4.0 * np.pi**2 * force_ratio)
