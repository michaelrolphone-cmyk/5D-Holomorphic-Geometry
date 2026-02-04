# 5D Holomorphic Geometry White Paper (Library State)

## Purpose and scope

This document summarizes the scope of the 5D Holomorphic Geometry library as implemented in this repository. It is a companion
to the conceptual manuscripts, but it intentionally limits itself to what is implemented in code. The library provides
NumPy-based numerical utilities for:

- A discrete 5D manifold \( \mathbb{R}^3 \times S^1_y \) with periodic fiber coordinate.
- Modular-surface utilities (projection to the standard fundamental domain and orbit sampling).
- Capacitance-dictionary formulas and unification-style derived quantities.
- Spectral/theta-function utilities and Mellin integral approximations.
- A Yang–Mills mass-gap bound based on the compact fiber radius.

The library is intended for exploration and reproducibility of the formulas described in the manuscripts; it does not establish
physical claims or proofs.

## Mathematical building blocks implemented

### 5D geometry and modular surface

- The manifold is represented by a discrete grid over \(x_1, x_2, x_3, y\), with \(y\) periodic on a circle of radius \(R_y\).
- A 5D Laplacian uses centered finite differences in all four spatial axes, with periodicity along the fiber.
- The modular surface utilities implement:
  - Projection to a standard fundamental domain via \( \tau \mapsto \tau + n \) and \( \tau \mapsto -1/\tau \).
  - Fundamental-domain membership checks for \( |\tau| \ge 1 \), \( |\Re(\tau)| \le 1/2 \), and \( \Im(\tau) > 0 \).

### Kaluza–Klein spectrum and mass-gap helpers

- Kaluza–Klein masses \( m_n = |n| / R_y \) and Poincaré constants for the compact \(S^1_y\) are implemented directly.
- The mass-gap bound is modeled as \( \Delta \ge \sqrt{g} / R_y \) with a configurable coupling scale.
- Zero-mode removal and zero-mean checks are included to model the exclusion of the \(n=0\) mode in numerical experiments.

### Capacitance dictionary and unification utilities

- Capacitance formulas for electron, spherical, and cosmic scales are implemented.
- Derived quantities include Coulomb force, spring constants, harmonic frequencies, and Unruh/Hawking temperatures.
- A `CapacitanceModel` generates ladder levels and Dirichlet series values for experimentation.

### Spectral and Mellin utilities

- Discrete modular Hilbert space objects approximate the hyperbolic measure on the fundamental domain.
- Jacobi theta-function sums are provided along with a functional-equation check.
- A Mellin transform integral estimates zeta-function values for given complex inputs.
- Zeta-regularized determinants for finite spectra are provided as simple products.

### Simulation utilities

- Diffusion is simulated on the discrete 5D manifold via explicit Euler steps.
- Optional impedance boundaries attenuate the outer slices of the 3D spatial base.
- Modular orbits can be sampled for a given \(\tau\) using the S and T generators.

## CLI coverage

The CLI provides direct access to the key numerical routines:

- `sample-orbit`: sample modular orbits.
- `simulate-diffusion`: run diffusion on the 5D grid with optional impedance attenuation.
- `theta` and `mellin-zeta`: evaluate the theta function and Mellin-based zeta estimates.
- `simulate-physics`: compute the unification-style derived quantities.
- `mass-gap`: evaluate the Yang–Mills mass-gap bound in either base units or MeV.
- `fundamental-geometry`: report the algebraic invariants of the 5D manifold and optionally project \(\tau\).

For exact arguments and options, see the CLI section of `README.md` or run:

```bash
python -m holomorphic5d.cli --help
```

## Implementation notes

- All routines are implemented with NumPy; no GPU acceleration is provided.
- The library favors explicit formulas over numerical optimization.
- Defaults (such as physical constants) are maintained in the CLI for reproducibility.
