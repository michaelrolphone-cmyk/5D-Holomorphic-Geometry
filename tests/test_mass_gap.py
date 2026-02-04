import numpy as np

from holomorphic5d.mass_gap import (
    FiberGeometry,
    check_zero_mode,
    kk_mode_masses,
    mass_gap_bound,
    mass_gap_mev,
    poincare_constant,
    poincare_lower_bound,
    zero_mode_removed,
)


def test_poincare_constants():
    radius = 2.0
    constant = poincare_constant(radius)
    lower = poincare_lower_bound(radius)
    assert np.isclose(constant, radius**2)
    assert np.isclose(lower, 1.0 / radius**2)


def test_zero_mode_removal_and_check():
    field = np.array([[1.0, 2.0, 3.0]])
    removed = zero_mode_removed(field, axis=1)
    assert check_zero_mode(removed, axis=1)
    assert np.isclose(np.mean(removed), 0.0)


def test_kk_mode_masses_and_gap():
    modes = np.array([-2, -1, 0, 1, 2])
    masses = kk_mode_masses(2.0, modes)
    assert np.allclose(masses, [1.0, 0.5, 0.0, 0.5, 1.0])
    assert np.isclose(mass_gap_bound(2.0), 0.5)
    assert np.isclose(mass_gap_mev(2e-15), 197.3269804 / 2.0)


def test_fiber_geometry():
    geom = FiberGeometry(radius_y=3.0)
    assert np.isclose(geom.circumference, 2.0 * np.pi * 3.0)
