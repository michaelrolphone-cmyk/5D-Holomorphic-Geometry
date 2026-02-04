import numpy as np

from holomorphic5d.unification import (
    PhysicalConstants,
    bridge_capacitance,
    coulomb_force,
    electron_capacitance,
    gravitational_mass_from_force,
    harmonic_frequency,
    hawking_temperature,
    simulate_electron_binding,
    spherical_capacitance,
    spring_constant,
    unruh_temperature,
    universe_capacitance,
)


def test_capacitance_and_force_relations():
    charge = 2.0
    mass = 4.0
    c = 2.0
    capacitance = electron_capacitance(charge, mass, c)
    assert np.isclose(capacitance, charge**2 / (2.0 * mass * c**2))
    sphere = spherical_capacitance(3.0, 1.0)
    assert np.isclose(sphere, 2.0 * np.pi * 9.0)
    universe = universe_capacitance(2.0, 1.0)
    bridge = bridge_capacitance(2.0, 1.0, 10.0)
    assert np.isclose(bridge, universe / (4.0 * np.pi**2 * 10.0))


def test_harmonic_and_thermal_helpers():
    force = 10.0
    displacement = 2.0
    k_spring = spring_constant(force, displacement)
    frequency = harmonic_frequency(k_spring, 4.0)
    assert np.isclose(k_spring, 5.0)
    assert np.isclose(frequency, (1.0 / (2.0 * np.pi)) * np.sqrt(5.0 / 4.0))
    accel = force / 4.0
    unruh = unruh_temperature(accel, hbar=2.0, c=2.0, k_b=2.0)
    assert np.isclose(unruh, 2.0 * accel / (4.0 * np.pi * 2.0 * 2.0))
    hawking = hawking_temperature(4.0, hbar=2.0, c=2.0, g_const=2.0, k_b=2.0)
    assert np.isclose(hawking, 2.0 * 8.0 / (8.0 * np.pi * 2.0 * 2.0 * 4.0))


def test_simulate_electron_binding_outputs():
    constants = PhysicalConstants(
        electron_radius=1.0,
        proton_radius=0.5,
        universe_radius=2.0,
        electron_mass=1.0,
        charge_q=1.0,
        epsilon0=1.0,
        c=2.0,
        g_const=1.0,
        hbar=1.0,
        k_b=1.0,
    )
    results = simulate_electron_binding(constants)
    force = coulomb_force(1.0, 1.0, 1.0)
    mass_from_force = gravitational_mass_from_force(force, 0.5, 2.0)
    assert np.isclose(results["force_coulomb"], force)
    assert np.isclose(results["mass_from_force"], mass_from_force)
