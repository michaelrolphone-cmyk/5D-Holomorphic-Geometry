import numpy as np

from holomorphic5d.imaging import decode_pixels_to_manifold
from holomorphic5d.manifold import ModularSurface


def test_decode_pixels_shapes_and_ranges() -> None:
    image = np.linspace(0.0, 1.0, 27).reshape(3, 3, 3)
    embedding = decode_pixels_to_manifold(image, radius_y=2.0)

    assert embedding.points.shape == (3, 3, 5)
    assert embedding.tau.shape == (3, 3)
    assert embedding.depth_field.shape == (3, 3)
    assert embedding.spectral_shift.shape == (3, 3)
    assert embedding.phase_field.shape == (3, 3)
    assert embedding.temporal_field.shape == (3, 3)

    assert np.all(embedding.depth_field >= 0.0)
    assert np.all(embedding.depth_field <= 1.0)
    assert np.all(embedding.temporal_field >= 0.0)
    assert np.all(embedding.temporal_field <= 1.0)

    y_values = embedding.points[..., 3]
    assert np.all(y_values >= 0.0)
    assert np.all(y_values < 2 * np.pi * 2.0 + 1e-8)

    assert np.all(np.imag(embedding.tau) > 0.0)


def test_decode_pixels_projects_tau_to_fundamental_domain() -> None:
    image = np.zeros((2, 2, 3), dtype=float)
    image[..., 0] = 1.0
    embedding = decode_pixels_to_manifold(image, spectral_scale=10.0)

    surface = ModularSurface()
    assert np.all(surface.in_fundamental_domain(embedding.tau))
