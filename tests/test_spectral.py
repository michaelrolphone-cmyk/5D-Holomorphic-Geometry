import numpy as np

from holomorphic5d.fundamental import FundamentalGeometry5D
from holomorphic5d.spectral import (
    ModularHilbertSpace,
    discrete_five_d_fourier,
    hilbert_space_from_geometry,
    holomorphic_spectrum_fourier,
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


def test_hilbert_space_from_geometry():
    geometry = FundamentalGeometry5D(radius_y=1.0)
    tau_grid = np.array([[2.2 + 0.3j, -1.5 + 1.8j]])
    space = hilbert_space_from_geometry(geometry, tau_grid)
    projected = geometry.project_tau(tau_grid)
    assert np.allclose(space.x_grid, projected.real)
    assert np.allclose(space.y_grid, projected.imag)


def test_holomorphic_spectrum_fourier_origin():
    value = holomorphic_spectrum_fourier(np.zeros(5))
    assert np.isclose(value.real, 1.0)
    assert np.isclose(value.imag, 0.0)


def test_holomorphic_spectrum_fourier_decay():
    origin = holomorphic_spectrum_fourier(np.zeros(5))
    shifted = holomorphic_spectrum_fourier(np.ones(5))
    assert np.abs(shifted) < np.abs(origin)


def test_holomorphic_spectrum_fourier_vectorized_shape():
    k_coords = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.5, -0.5, 0.2, -0.2, 0.1]])
    values = holomorphic_spectrum_fourier(k_coords, sigma=0.4, coupling=0.8, twist=0.2)
    assert values.shape == (2,)


def test_discrete_five_d_fourier_origin_point():
    points = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    weights = np.array([1.0])
    k_coords = np.array([[0.1, -0.2, 0.3, -0.4, 0.5]])
    value = discrete_five_d_fourier(points, weights, k_coords)
    assert np.allclose(value, np.array([1.0 + 0.0j]))


def test_discrete_five_d_fourier_half_period_cancels():
    points = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    weights = np.array([1.0, 1.0])
    k_coords = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    value = discrete_five_d_fourier(points, weights, k_coords)
    assert np.allclose(value, np.array([0.0 + 0.0j]))
