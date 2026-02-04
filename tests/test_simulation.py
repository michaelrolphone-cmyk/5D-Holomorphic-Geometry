import numpy as np

from holomorphic5d.fundamental import FundamentalGeometry5D
from holomorphic5d.manifold import HolomorphicManifold5D
from holomorphic5d.simulation import (
    ImpedanceBoundary,
    simulate_diffusion,
    simulate_diffusion_from_geometry,
)


def test_simulate_diffusion_decays_peak():
    manifold = HolomorphicManifold5D(
        grid_shape=(6, 6, 6, 6),
        spacing=(1.0, 1.0, 1.0, 1.0),
        radius_y=1.0,
    )
    field = np.zeros((6, 6, 6, 6))
    field[3, 3, 3, :] = 1.0
    history = simulate_diffusion(manifold, field, dt=0.1, steps=5, diffusivity=0.25)
    assert history[-1].max() < history[0].max()


def test_simulate_diffusion_impedance():
    manifold = HolomorphicManifold5D(
        grid_shape=(4, 4, 4, 4),
        spacing=(1.0, 1.0, 1.0, 1.0),
        radius_y=1.0,
    )
    field = np.zeros((4, 4, 4, 4))
    field[0, :, :, :] = 1.0
    history = simulate_diffusion(
        manifold,
        field,
        dt=0.1,
        steps=1,
        diffusivity=0.1,
        impedance=ImpedanceBoundary(0.5),
    )
    assert np.all(history[-1][0, :, :, :] <= 0.5)


def test_simulate_diffusion_from_geometry():
    geometry = FundamentalGeometry5D(radius_y=2.0)
    field = np.zeros((4, 4, 4, 4))
    field[2, 2, 2, :] = 1.0
    history = simulate_diffusion_from_geometry(
        geometry,
        grid_shape=(4, 4, 4, 4),
        spacing_base=(1.0, 1.0, 1.0),
        field=field,
        dt=0.1,
        steps=2,
        diffusivity=0.2,
    )
    assert history.shape == (3, 4, 4, 4, 4)
