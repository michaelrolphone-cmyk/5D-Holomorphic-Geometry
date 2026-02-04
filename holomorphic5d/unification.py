from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .formulas import HolomorphicFormulas


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants used by the unification calculations."""

    electron_radius: float
    proton_radius: float
    universe_radius: float
    electron_mass: float
    charge_q: float
    epsilon0: float
    c: float
    g_const: float
    hbar: float
    k_b: float
    force_ratio: float = 2.27e39


def electron_capacitance(charge_q: float, electron_mass: float, c: float) -> float:
    return HolomorphicFormulas.electron_capacitance(charge_q, electron_mass, c)


def spherical_capacitance(radius: float, epsilon0: float) -> float:
    return HolomorphicFormulas.spherical_capacitance(radius, epsilon0)


def universe_capacitance(universe_radius: float, epsilon0: float) -> float:
    return HolomorphicFormulas.universe_capacitance(universe_radius, epsilon0)


def bridge_capacitance(universe_radius: float, epsilon0: float, force_ratio: float) -> float:
    return HolomorphicFormulas.bridge_capacitance(
        universe_radius, epsilon0, force_ratio
    )


def coulomb_force(charge_q: float, epsilon0: float, radius: float) -> float:
    return charge_q**2 / (4.0 * np.pi * epsilon0 * radius**2)


def gravitational_mass_from_force(force: float, proton_radius: float, c: float) -> float:
    return (2.0 * proton_radius / c**2) * force


def spring_constant(force: float, displacement: float) -> float:
    return force / displacement


def harmonic_frequency(k_spring: float, mass: float) -> float:
    return (1.0 / (2.0 * np.pi)) * np.sqrt(k_spring / mass)


def unruh_temperature(acceleration: float, hbar: float, c: float, k_b: float) -> float:
    return hbar * acceleration / (4.0 * np.pi * c * k_b)


def hawking_temperature(mass: float, hbar: float, c: float, g_const: float, k_b: float) -> float:
    return hbar * c**3 / (8.0 * np.pi * k_b * g_const * mass)


def simulate_electron_binding(constants: PhysicalConstants) -> dict[str, float]:
    """Compute derived quantities from the unification framework."""
    c_e = electron_capacitance(constants.charge_q, constants.electron_mass, constants.c)
    c_u = universe_capacitance(constants.universe_radius, constants.epsilon0)
    c_bridge = bridge_capacitance(
        constants.universe_radius, constants.epsilon0, constants.force_ratio
    )
    c_sphere = spherical_capacitance(constants.electron_radius, constants.epsilon0)
    force = coulomb_force(constants.charge_q, constants.epsilon0, constants.electron_radius)
    mass_from_force = gravitational_mass_from_force(force, constants.proton_radius, constants.c)
    k_spring = spring_constant(force, 2.0 * constants.proton_radius)
    frequency = harmonic_frequency(k_spring, constants.electron_mass)
    acceleration = force / constants.electron_mass
    unruh = unruh_temperature(acceleration, constants.hbar, constants.c, constants.k_b)
    hawking = hawking_temperature(
        constants.electron_mass, constants.hbar, constants.c, constants.g_const, constants.k_b
    )
    return {
        "electron_capacitance": c_e,
        "universe_capacitance": c_u,
        "bridge_capacitance": c_bridge,
        "spherical_capacitance": c_sphere,
        "force_coulomb": force,
        "mass_from_force": mass_from_force,
        "spring_constant": k_spring,
        "harmonic_frequency": frequency,
        "acceleration": acceleration,
        "unruh_temperature": unruh,
        "hawking_temperature": hawking,
    }
