# 5D Holomorphic Geometry (NumPy)

This repository provides a NumPy-based library to model and simulate a discrete version of the 5D holomorphic modular manifold
\(\mathcal{M}^5 = \mathbb{R}^3 \times S^1_y\), along with modular-surface, spectral, and capacitance-dictionary utilities.

## White paper overview

This library is a computational companion to a set of white papers that propose a 5D holomorphic modular manifold as a unifying
framework for electromagnetism, gravitation, and arithmetic spectral phenomena. In that narrative, a compactified fifth dimension
with modular symmetry supports:

- A capacitance-based dictionary that ties rest energy, force scales, and cosmic/proton radii through impedance-matched boundaries.
- A spectral operator on the modular surface whose eigenvalues are linked to zeta-function structure via theta/Mellin machinery.
- Kähler-geometric structure that encodes gauge fields and holomorphic quantization conditions as period constraints.
- A Yang-Mills mass-gap bound derived from fiber impedance selection and a Poincaré inequality on the compact circle.

The library does **not** attempt to prove physical claims; instead it implements the explicit equations and derived quantities
introduced in the manuscripts so that they can be explored numerically. The unification helpers compute the capacitance ladder,
bridge constants, forces, harmonic scales, and Unruh/Hawking temperatures as described in the text, while the spectral utilities
support theta-function checks and Mellin transforms used in the RH-inspired construction. The mass-gap utilities encode the
zero-mode exclusion, KK spectrum, and Poincaré lower bound used in the Yang-Mills manuscript. The goal is to make the framework
reproducible, configurable, and testable for experimentation.

## Installation

```bash
python -m pip install -e .
```

## API

### `holomorphic5d.ModularSurface`
- `project_to_fundamental_domain(tau: np.ndarray) -> np.ndarray`: project complex points to the standard fundamental domain.
- `in_fundamental_domain(tau: np.ndarray) -> np.ndarray`: boolean mask for points in the fundamental domain.

### `holomorphic5d.HolomorphicManifold5D`
- `grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`: build the 5D meshgrid.
- `laplacian5(field: np.ndarray) -> np.ndarray`: compute the 5D Laplacian with periodicity along the fiber.

### `holomorphic5d.mass_gap` helpers
- `FiberGeometry`: container for fiber radius/circumference.
- `kk_mode_masses(radius_y: float, modes: np.ndarray) -> np.ndarray`.
- `poincare_constant(radius_y: float) -> float`.
- `poincare_lower_bound(radius_y: float) -> float`.
- `zero_mode_removed(field: np.ndarray, axis: int = -1) -> np.ndarray`.
- `check_zero_mode(field: np.ndarray, axis: int = -1, atol: float = 1e-8) -> bool`.
- `mass_gap_bound(radius_y: float, coupling_scale: float = 1.0) -> float`.
- `mass_gap_mev(radius_y: float, hbar_c_mev_fm: float = 197.3269804) -> float`.

### `holomorphic5d.physics` helpers
- `CapacitanceModel`: capacitance dictionary utilities, ladder spectrum, and Dirichlet series.
- `dedekind_eta(tau: np.ndarray, terms: int = 50) -> np.ndarray`.
- `kahler_potential(z1, z2, tau, phi, kappa) -> np.ndarray`.
- `kahler_metric(phi: float) -> np.ndarray`.
- `kk_mode_masses(radius_y: float, modes: np.ndarray) -> np.ndarray`.
- `capacitance_level(z1, z2, epsilon0: float) -> np.ndarray`.
- `hodge_normalization(universe_capacitance: float, proton_capacitance: float) -> float`.

### `holomorphic5d.spectral` helpers
- `ModularHilbertSpace(x_grid, y_grid)`: discrete Hilbert-space utility with hyperbolic measure.
- `hyperbolic_measure(y: np.ndarray) -> np.ndarray`.
- `modular_operator_h(field, y_grid, dy) -> np.ndarray`.
- `theta_function(t: np.ndarray, n_terms: int = 50) -> np.ndarray`.
- `theta_functional_equation(t: np.ndarray, n_terms: int = 50) -> np.ndarray`.
- `mellin_zeta(s: complex, t: np.ndarray, n_terms: int = 50) -> complex`.
- `zeta_regularized_determinant(eigenvalues: np.ndarray) -> complex`.

### `holomorphic5d.unification` helpers
- `PhysicalConstants`: container for the physical constants and scale parameters.
- `electron_capacitance(charge_q, electron_mass, c) -> float`.
- `spherical_capacitance(radius, epsilon0) -> float`.
- `universe_capacitance(universe_radius, epsilon0) -> float`.
- `bridge_capacitance(universe_radius, epsilon0, force_ratio) -> float`.
- `coulomb_force(charge_q, epsilon0, radius) -> float`.
- `gravitational_mass_from_force(force, proton_radius, c) -> float`.
- `spring_constant(force, displacement) -> float`.
- `harmonic_frequency(k_spring, mass) -> float`.
- `unruh_temperature(acceleration, hbar, c, k_b) -> float`.
- `hawking_temperature(mass, hbar, c, g_const, k_b) -> float`.
- `simulate_electron_binding(constants: PhysicalConstants) -> dict[str, float]`.

### `holomorphic5d.simulation` helpers
- `apply_impedance_boundary(field: np.ndarray, impedance: ImpedanceBoundary) -> np.ndarray`.
- `modular_transform(tau: np.ndarray, matrix: np.ndarray) -> np.ndarray`.
- `sample_modular_orbit(tau: complex, steps: int) -> np.ndarray`.
- `simulate_diffusion(manifold, field, dt, steps, diffusivity=1.0, impedance=None) -> np.ndarray`.

## CLI

Use the module as a CLI entry-point:

```bash
python -m holomorphic5d.cli sample-orbit 0.2 1.1 6
```

```bash
python -m holomorphic5d.cli simulate-diffusion 8 8 8 16 0.01 5 --diffusivity 0.5 --radius-y 1.0 --impedance 0.9
```

```bash
python -m holomorphic5d.cli theta 1.0 --terms 80
```

```bash
python -m holomorphic5d.cli mellin-zeta 2.0 0.1 5.0 --steps 2000 --terms 80
```

```bash
python -m holomorphic5d.cli simulate-physics
```

```bash
python -m holomorphic5d.cli mass-gap 2.818e-15 --mev
```

The diffusion command prints the mean of the final field state.
