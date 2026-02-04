from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .formulas import HolomorphicFormulas

if TYPE_CHECKING:
    from .fundamental import FundamentalGeometry5D


@dataclass(frozen=True)
class CapacitanceModel:
    """Capacitance dictionary and ladder definitions."""

    epsilon0: float
    charge_q: float
    speed_c: float
    proton_radius: float
    universe_radius: float
    bridge_ratio: float = 2.27e39

    def proton_capacitance(self) -> float:
        return HolomorphicFormulas.spherical_capacitance(
            self.proton_radius, self.epsilon0
        )

    def universe_capacitance(self) -> float:
        return HolomorphicFormulas.universe_capacitance(
            self.universe_radius, self.epsilon0
        )

    def bridge_capacitance(self) -> float:
        return HolomorphicFormulas.bridge_capacitance(
            self.universe_radius, self.epsilon0, self.bridge_ratio
        )

    def electron_capacitance(self, electron_mass: float) -> float:
        return HolomorphicFormulas.electron_capacitance(
            self.charge_q, electron_mass, self.speed_c
        )

    def mode_capacitances(self, modes: np.ndarray, kappa: float = 0.0) -> np.ndarray:
        modes = np.asarray(modes, dtype=float)
        base = self.proton_capacitance() * modes**2
        if kappa == 0.0:
            return base
        return base * np.exp(-kappa * modes)

    def dirichlet_series(self, s: complex, n_terms: int, kappa: float = 0.0) -> complex:
        modes = np.arange(1, n_terms + 1, dtype=float)
        caps = self.mode_capacitances(modes, kappa=kappa)
        return np.sum(caps ** (-s))


def dedekind_eta(tau: np.ndarray, terms: int = 50) -> np.ndarray:
    """Compute Dedekind eta via its q-product expansion."""
    tau = np.asarray(tau, dtype=np.complex128)
    q = np.exp(2.0j * np.pi * tau)
    product = np.ones_like(q)
    for n in range(1, terms + 1):
        product *= 1.0 - q**n
    return q ** (1.0 / 24.0) * product


def kahler_potential(
    z1: np.ndarray,
    z2: np.ndarray,
    tau: np.ndarray,
    phi: float,
    kappa: float,
) -> np.ndarray:
    """Compute the Kahler potential K = phi(|z1|^2+|z2|^2)+kappa log|eta|^{24}."""
    eta = dedekind_eta(tau)
    return phi * (np.abs(z1) ** 2 + np.abs(z2) ** 2) + kappa * np.log(np.abs(eta) ** 24)


def kahler_metric(phi: float) -> np.ndarray:
    """Return the Kahler metric for the quadratic part of K.

    The modular correction term is independent of z1,z2 and does not
    contribute to the metric in these coordinates.
    """
    return phi * np.eye(2)


def kk_mode_masses(radius_y: float, modes: np.ndarray) -> np.ndarray:
    """Compute Kaluza-Klein masses m_n = n / R_y."""
    return HolomorphicFormulas.kk_mode_masses(radius_y, modes)


def kk_mode_masses_from_geometry(
    geometry: FundamentalGeometry5D, modes: np.ndarray
) -> np.ndarray:
    """Compute Kaluza-Klein masses using the fundamental geometry."""
    return kk_mode_masses(geometry.radius_y, modes)


def capacitance_level(z1: np.ndarray, z2: np.ndarray, epsilon0: float) -> np.ndarray:
    """Compute the capacitance level set 2*pi*epsilon0*(|z1|^2+|z2|^2)."""
    return 2.0 * np.pi * epsilon0 * (np.abs(z1) ** 2 + np.abs(z2) ** 2)


def hodge_normalization(universe_capacitance: float, proton_capacitance: float) -> float:
    """Return the normalization factor C_u / (4 pi^2 C_p)."""
    return universe_capacitance / (4.0 * np.pi**2 * proton_capacitance)
