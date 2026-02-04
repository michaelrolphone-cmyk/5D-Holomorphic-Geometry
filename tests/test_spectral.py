import numpy as np

from holomorphic5d.spectral import (
    ModularHilbertSpace,
    hyperbolic_measure,
    mellin_zeta,
    modular_operator_h,
    theta_function,
    theta_functional_equation,
    zeta_regularized_determinant,
)


def test_hyperbolic_measure_and_inner_product():
    x = np.linspace(-0.5, 0.5, 3)
    y = np.linspace(1.0, 2.0, 3)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    space = ModularHilbertSpace(x_grid=x_grid, y_grid=y_grid)
    measure = hyperbolic_measure(y_grid)
    assert np.allclose(measure, space.measure())
    f = np.ones_like(x_grid)
    g = 2.0 * np.ones_like(x_grid)
    inner = space.inner_product(f, g)
    assert np.isclose(inner, 2.0 * np.sum(measure))


def test_modular_operator_h_linear_field():
    y = np.linspace(1.0, 2.0, 5)
    dy = y[1] - y[0]
    field = np.tile(y, (2, 1))
    output = modular_operator_h(field, y, dy)
    assert np.allclose(output.imag, -y)


def test_theta_functional_equation_near_zero():
    t = np.array([0.8, 1.2])
    residual = theta_functional_equation(t, n_terms=80)
    assert np.all(np.abs(residual) < 5e-3)


def test_mellin_zeta_positive():
    t = np.linspace(0.1, 5.0, 2000)
    estimate = mellin_zeta(2.0, t, n_terms=80)
    assert estimate.real > 0


def test_zeta_regularized_determinant():
    eigenvalues = np.array([2.0, 3.0])
    det = zeta_regularized_determinant(eigenvalues)
    assert np.isclose(det, 6.0)
