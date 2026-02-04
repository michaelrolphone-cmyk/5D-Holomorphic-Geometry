import numpy as np

from holomorphic5d.physics import (
    CapacitanceModel,
    capacitance_level,
    dedekind_eta,
    hodge_normalization,
    kahler_metric,
    kahler_potential,
    kk_mode_masses,
)


def test_capacitance_ladder_and_dirichlet_series():
    model = CapacitanceModel(
        epsilon0=1.0,
        charge_q=1.0,
        speed_c=1.0,
        proton_radius=1.0,
        universe_radius=2.0,
    )
    modes = np.array([1.0, 2.0, 3.0])
    caps = model.mode_capacitances(modes)
    assert np.allclose(caps, 2.0 * np.pi * modes**2)
    series = model.dirichlet_series(1.0, n_terms=3)
    expected = np.sum((2.0 * np.pi * modes**2) ** (-1))
    assert np.isclose(series, expected)


def test_kahler_potential_metric_and_eta():
    tau = np.array([0.5 + 1.0j])
    eta = dedekind_eta(tau)
    assert np.all(np.abs(eta) > 0)
    z1 = np.array([1.0 + 0.0j])
    z2 = np.array([0.0 + 2.0j])
    potential = kahler_potential(z1, z2, tau, phi=2.0, kappa=0.1)
    assert np.isfinite(potential).all()
    metric = kahler_metric(phi=2.0)
    assert np.allclose(metric, 2.0 * np.eye(2))


def test_kk_masses_and_hodge_normalization():
    masses = kk_mode_masses(radius_y=2.0, modes=np.array([0, 1, -2]))
    assert np.allclose(masses, [0.0, 0.5, 1.0])
    normalization = hodge_normalization(8.0, 2.0)
    assert np.isclose(normalization, 8.0 / (4.0 * np.pi**2 * 2.0))


def test_capacitance_level():
    z1 = np.array([1.0 + 0.0j])
    z2 = np.array([0.0 + 2.0j])
    level = capacitance_level(z1, z2, epsilon0=1.0)
    assert np.isclose(level, 2.0 * np.pi * (1.0 + 4.0))
