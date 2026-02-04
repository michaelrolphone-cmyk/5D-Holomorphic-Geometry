import numpy as np

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
