import numpy as np
import pytest

from holomorphic5d.fundamental import FundamentalGeometry5D
from holomorphic5d.manifold import HolomorphicManifold5D, ModularSurface


def test_project_to_fundamental_domain():
    surface = ModularSurface()
    tau = np.array([2.3 + 0.4j, -0.9 + 1.5j])
    projected = surface.project_to_fundamental_domain(tau)
    mask = surface.in_fundamental_domain(projected)
    assert mask.all()


def test_laplacian_constant_field_zero():
    manifold = HolomorphicManifold5D(
        grid_shape=(4, 4, 4, 4),
        spacing=(1.0, 1.0, 1.0, 1.0),
        radius_y=1.0,
    )
    field = np.ones((4, 4, 4, 4))
    lap = manifold.laplacian5(field)
    assert np.allclose(lap, 0.0)


def test_manifold_from_fundamental_geometry():
    geometry = FundamentalGeometry5D(radius_y=2.0)
    manifold = HolomorphicManifold5D.from_fundamental_geometry(
        geometry, grid_shape=(2, 2, 2, 4), spacing_base=(1.0, 2.0, 3.0)
    )
    assert manifold.radius_y == 2.0
    assert manifold.spacing[3] == pytest.approx(geometry.fiber_circumference() / 4)
