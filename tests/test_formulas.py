import numpy as np

from holomorphic5d.formulas import HolomorphicFormulas


def test_kk_masses_and_poincare_bounds():
    modes = np.array([-2, -1, 0, 1, 2])
    masses = HolomorphicFormulas.kk_mode_masses(2.0, modes)
    assert np.allclose(masses, [1.0, 0.5, 0.0, 0.5, 1.0])
    assert np.isclose(HolomorphicFormulas.poincare_constant(3.0), 9.0)
    assert np.isclose(HolomorphicFormulas.poincare_lower_bound(3.0), 1.0 / 9.0)


def test_capacitance_helpers():
    assert np.isclose(HolomorphicFormulas.electron_capacitance(2.0, 4.0, 1.0), 0.5)
    assert np.isclose(HolomorphicFormulas.spherical_capacitance(2.0, 1.5), 12.0 * np.pi)
    assert np.isclose(HolomorphicFormulas.universe_capacitance(2.0, 1.5), 12.0 * np.pi)
    assert np.isclose(
        HolomorphicFormulas.bridge_capacitance(2.0, 1.5, 3.0),
        (12.0 * np.pi) / (4.0 * np.pi**2 * 3.0),
    )
