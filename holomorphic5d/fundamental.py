from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .manifold import ModularSurface
from .mass_gap import mass_gap_bound, poincare_constant, poincare_lower_bound


@dataclass(frozen=True)
class FundamentalGeometry5D:
    """Fundamental geometry and algebraic invariants for the 5D manifold."""

    radius_y: float
    modular_surface: ModularSurface = ModularSurface()

    def fiber_circumference(self) -> float:
        """Return the circumference of the compact S^1_y fiber."""
        return 2.0 * np.pi * self.radius_y

    def normalize_fiber_coordinate(self, y: np.ndarray) -> np.ndarray:
        """Normalize fiber coordinates into the [0, 2πR) interval."""
        y = np.asarray(y, dtype=float)
        return np.mod(y, self.fiber_circumference())

    def project_tau(self, tau: np.ndarray) -> np.ndarray:
        """Project modular parameters onto the fundamental domain."""
        return self.modular_surface.project_to_fundamental_domain(tau)

    def algebraic_invariants(self, coupling_scale: float = 1.0) -> dict[str, float]:
        """Return fundamental algebraic invariants for the 5D manifold."""
        return {
            "radius_y": self.radius_y,
            "circumference": self.fiber_circumference(),
            "poincare_constant": poincare_constant(self.radius_y),
            "poincare_lower_bound": poincare_lower_bound(self.radius_y),
            "mass_gap_bound": mass_gap_bound(
                self.radius_y, coupling_scale=coupling_scale
            ),
        }
