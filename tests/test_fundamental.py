import json

import numpy as np

from holomorphic5d.cli import main
from holomorphic5d.fundamental import FundamentalGeometry5D
from holomorphic5d.manifold import ModularSurface
from holomorphic5d.mass_gap import (
    mass_gap_bound,
    mass_gap_from_geometry,
    poincare_constant,
    poincare_lower_bound,
)
from holomorphic5d.physics import kk_mode_masses_from_geometry


def test_fundamental_algebraic_invariants():
    geometry = FundamentalGeometry5D(radius_y=2.0)
    invariants = geometry.algebraic_invariants(coupling_scale=1.5)
    assert invariants["radius_y"] == 2.0
    assert np.isclose(invariants["circumference"], 4.0 * np.pi)
    assert np.isclose(invariants["poincare_constant"], poincare_constant(2.0))
    assert np.isclose(invariants["poincare_lower_bound"], poincare_lower_bound(2.0))
    assert np.isclose(invariants["mass_gap_bound"], mass_gap_bound(2.0, coupling_scale=1.5))


def test_fundamental_geometry_helpers():
    geometry = FundamentalGeometry5D(radius_y=1.0)
    length = geometry.fiber_circumference()
    y = np.array([-0.5 * length, 0.25 * length, 1.5 * length])
    normalized = geometry.normalize_fiber_coordinate(y)
    assert np.all(normalized >= 0.0)
    assert np.all(normalized < length)

    surface = ModularSurface()
    tau = np.array([2.2 + 0.3j, -1.5 + 1.8j])
    assert np.allclose(geometry.project_tau(tau), surface.project_to_fundamental_domain(tau))


def test_cli_fundamental_geometry(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["holomorphic5d.cli", "fundamental-geometry", "2.0", "--coupling", "1.25"],
    )
    main()
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["radius_y"] == 2.0
    assert np.isclose(payload["mass_gap_bound"], mass_gap_bound(2.0, coupling_scale=1.25))


def test_fundamental_geometry_math_bridges():
    geometry = FundamentalGeometry5D(radius_y=3.0)
    masses = kk_mode_masses_from_geometry(geometry, np.array([0, 1, -2]))
    assert np.allclose(masses, np.array([0.0, 1.0 / 3.0, 2.0 / 3.0]))
    assert np.isclose(mass_gap_from_geometry(geometry, coupling_scale=2.0), np.sqrt(2.0) / 3.0)
