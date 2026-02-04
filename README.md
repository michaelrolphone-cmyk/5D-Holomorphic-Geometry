# 5D Holomorphic Geometry (NumPy)

This repository provides a NumPy-based library to model and simulate a discrete version of the 5D holomorphic modular manifold
\(\mathcal{M}^5 = \mathbb{R}^3 \times S^1_y\), along with modular-surface, spectral, and capacitance-dictionary utilities.

## White paper overview

This library is a computational companion to a white paper (see [`WHITE_PAPER.md`](WHITE_PAPER.md)) that proposes a 5D holomorphic
modular manifold as a unifying framework for electromagnetism, gravitation, and arithmetic spectral phenomena. In that narrative,
a compactified fifth dimension with modular symmetry supports:

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

### `holomorphic5d.HolomorphicFormulas`
- `kk_mode_masses(radius_y: float, modes: np.ndarray) -> np.ndarray`.
- `poincare_constant(radius_y: float) -> float`.
- `poincare_lower_bound(radius_y: float) -> float`.
- `electron_capacitance(charge_q: float, electron_mass: float, c: float) -> float`.
- `spherical_capacitance(radius: float, epsilon0: float) -> float`.
- `universe_capacitance(universe_radius: float, epsilon0: float) -> float`.
- `bridge_capacitance(universe_radius: float, epsilon0: float, force_ratio: float) -> float`.

### `holomorphic5d.ModularSurface`
- `project_to_fundamental_domain(tau: np.ndarray) -> np.ndarray`: project complex points to the standard fundamental domain.
- `in_fundamental_domain(tau: np.ndarray) -> np.ndarray`: boolean mask for points in the fundamental domain.

### `holomorphic5d.HolomorphicManifold5D`
- `grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`: build the 5D meshgrid.
- `laplacian5(field: np.ndarray) -> np.ndarray`: compute the 5D Laplacian with periodicity along the fiber.
- `from_fundamental_geometry(geometry, grid_shape, spacing_base) -> HolomorphicManifold5D`: construct from the fundamental geometry.

### `holomorphic5d.imaging` helpers
- `PixelManifoldEmbedding`: container for pixel-derived 5D embeddings.
- `decode_pixels_to_manifold(image, spacing=(1.0, 1.0, 1.0), radius_y=1.0, depth_scale=1.0, spectral_scale=1.0, phase_scale=1.0, temporal_scale=1.0, eps=1e-8, project_tau=True) -> PixelManifoldEmbedding`.

### `holomorphic5d.FundamentalGeometry5D`
- `fiber_circumference() -> float`: return the S^1_y circumference.
- `normalize_fiber_coordinate(y: np.ndarray) -> np.ndarray`: normalize fiber coordinates into `[0, 2πR)`.
- `project_tau(tau: np.ndarray) -> np.ndarray`: project modular parameters onto the fundamental domain.
- `algebraic_invariants(coupling_scale: float = 1.0) -> dict[str, float]`: basic invariants (radius, Poincaré bounds, mass gap).

### `holomorphic5d.mass_gap` helpers
- `FiberGeometry`: container for fiber radius/circumference.
- `kk_mode_masses(radius_y: float, modes: np.ndarray) -> np.ndarray`.
- `poincare_constant(radius_y: float) -> float`.
- `poincare_lower_bound(radius_y: float) -> float`.
- `zero_mode_removed(field: np.ndarray, axis: int = -1) -> np.ndarray`.
- `check_zero_mode(field: np.ndarray, axis: int = -1, atol: float = 1e-8) -> bool`.
- `mass_gap_bound(radius_y: float, coupling_scale: float = 1.0) -> float`.
- `mass_gap_from_geometry(geometry, coupling_scale: float = 1.0) -> float`.
- `mass_gap_mev(radius_y: float, hbar_c_mev_fm: float = 197.3269804) -> float`.

### `holomorphic5d.physics` helpers
- `CapacitanceModel`: capacitance dictionary utilities, ladder spectrum, and Dirichlet series.
- `dedekind_eta(tau: np.ndarray, terms: int = 50) -> np.ndarray`.
- `kahler_potential(z1, z2, tau, phi, kappa) -> np.ndarray`.
- `kahler_metric(phi: float) -> np.ndarray`.
- `kk_mode_masses(radius_y: float, modes: np.ndarray) -> np.ndarray`.
- `kk_mode_masses_from_geometry(geometry, modes: np.ndarray) -> np.ndarray`.
- `capacitance_level(z1, z2, epsilon0: float) -> np.ndarray`.
- `hodge_normalization(universe_capacitance: float, proton_capacitance: float) -> float`.

### `holomorphic5d.spectral` helpers
- `ModularHilbertSpace(x_grid, y_grid)`: discrete Hilbert-space utility with hyperbolic measure.
- `hilbert_space_from_geometry(geometry, tau_grid) -> ModularHilbertSpace`.
- `hyperbolic_measure(y: np.ndarray) -> np.ndarray`.
- `modular_operator_h(field, y_grid, dy) -> np.ndarray`.
- `theta_function(t: np.ndarray, n_terms: int = 50) -> np.ndarray`.
- `theta_functional_equation(t: np.ndarray, n_terms: int = 50) -> np.ndarray`.
- `mellin_zeta(s: complex, t: np.ndarray, n_terms: int = 50) -> complex`.
- `zeta_regularized_determinant(eigenvalues: np.ndarray) -> complex`.
- `holomorphic_spectrum_fourier(k_coords, sigma=0.6, center=None, coupling=1.0, twist=0.35) -> np.ndarray`.

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
- `ImpedanceBoundary(factor: float)`: attenuation model for boundary slices.
- `apply_impedance_boundary(field: np.ndarray, impedance: ImpedanceBoundary) -> np.ndarray`.
- `modular_transform(tau: np.ndarray, matrix: np.ndarray) -> np.ndarray`.
- `sample_modular_orbit(tau: complex, steps: int) -> np.ndarray`.
- `simulate_diffusion(manifold, field, dt, steps, diffusivity=1.0, impedance=None) -> np.ndarray`.
- `simulate_diffusion_from_geometry(geometry, grid_shape, spacing_base, field, dt, steps, diffusivity=1.0, impedance=None) -> np.ndarray`.

## CLI

Use the module as a CLI entry-point. Commands and options reflect the callable helpers:

| Command | Purpose | Required args | Key options |
| --- | --- | --- | --- |
| `sample-orbit` | Sample a modular orbit using S/T generators. | `tau_real tau_imag steps` | None |
| `simulate-diffusion` | Run 5D diffusion with optional impedance. | `nx ny nz ny_fiber dt steps` | `--diffusivity`, `--radius-y`, `--impedance` |
| `theta` | Compute theta(t). | `t` | `--terms` |
| `mellin-zeta` | Approximate Mellin-zeta integral. | `s t-min t-max` | `--steps`, `--terms` |
| `simulate-physics` | Compute unification constants. | None | `--electron-radius`, `--proton-radius`, `--universe-radius`, `--electron-mass`, `--charge`, `--epsilon0`, `--c`, `--g`, `--hbar`, `--k-b` |
| `mass-gap` | Compute Yang-Mills gap bound. | `radius_y` | `--coupling`, `--mev` |
| `fundamental-geometry` | Report invariants and optional tau projection. | `radius_y` | `--coupling`, `--tau-real`, `--tau-imag` |
| `decode-pixels` | Decode RGB pixels into a 5D embedding and tau field. | `input output` | `--radius-y`, `--depth-scale`, `--spectral-scale`, `--phase-scale`, `--temporal-scale`, `--spacing-x`, `--spacing-y`, `--spacing-z` |
| `spectrum-fourier` | Evaluate the 5D holomorphic Fourier spectrum at a single frequency coordinate. | `k0 k1 k2 k3 k4` | `--sigma`, `--coupling`, `--twist`, `--center` |

Examples:

```bash
python -m holomorphic5d.cli sample-orbit 0.2 1.1 6
```

```bash
python -m holomorphic5d.cli simulate-diffusion 8 8 8 16 0.01 5 --diffusivity 0.5 --radius-y 1.0 --impedance 0.9
```

```bash
python -m holomorphic5d.cli simulate-physics
```

```bash
python -m holomorphic5d.cli theta 1.0 --terms 80
```

```bash
python -m holomorphic5d.cli mellin-zeta 2.0 0.1 5.0 --steps 2000 --terms 80
```

```bash
python -m holomorphic5d.cli mass-gap 2.818e-15 --mev
```

```bash
python -m holomorphic5d.cli fundamental-geometry 2.0 --coupling 1.25 --tau-real 0.2 --tau-imag 1.1
```

```bash
python -m holomorphic5d.cli decode-pixels image.npy embedding.npz --radius-y 1.5 --depth-scale 0.8 --spectral-scale 1.2
```

```bash
python -m holomorphic5d.cli spectrum-fourier 0.2 -0.4 0.1 0.0 0.6 --sigma 0.7 --coupling 1.1 --twist 0.25
```

The diffusion command prints the mean of the final field state.

## Standalone WebGL temporal-field viewer

Open [`temporal_field_viewer.html`](temporal_field_viewer.html) in a browser to switch between a 5D holomorphic Fourier spectrum
render and the temporal-field viewer. The spectrum mode renders a 5D Fourier transform (phase as hue, amplitude as brightness),
while the temporal-field mode mirrors the luminance/gradient/Laplacian math in `decode_pixels_to_manifold`.
