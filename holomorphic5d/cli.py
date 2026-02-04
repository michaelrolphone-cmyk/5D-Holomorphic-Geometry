from __future__ import annotations

import argparse
import json

import numpy as np

from .manifold import HolomorphicManifold5D
from .mass_gap import mass_gap_bound, mass_gap_mev
from .simulation import ImpedanceBoundary, sample_modular_orbit, simulate_diffusion
from .spectral import mellin_zeta, theta_function
from .unification import PhysicalConstants, simulate_electron_binding


DEFAULT_CONSTANTS = PhysicalConstants(
    electron_radius=2.81794032e-15,
    proton_radius=1.40897016e-15,
    universe_radius=1.26e26,
    electron_mass=9.109383e-31,
    charge_q=1.6021766e-19,
    epsilon0=8.854e-12,
    c=299792458.0,
    g_const=6.674e-11,
    hbar=1.054571817e-34,
    k_b=1.380649e-23,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="5D holomorphic manifold utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    orbit_parser = subparsers.add_parser("sample-orbit", help="Sample modular orbit")
    orbit_parser.add_argument("tau_real", type=float)
    orbit_parser.add_argument("tau_imag", type=float)
    orbit_parser.add_argument("steps", type=int)

    sim_parser = subparsers.add_parser("simulate-diffusion", help="Run diffusion")
    sim_parser.add_argument("nx", type=int)
    sim_parser.add_argument("ny", type=int)
    sim_parser.add_argument("nz", type=int)
    sim_parser.add_argument("ny_fiber", type=int)
    sim_parser.add_argument("dt", type=float)
    sim_parser.add_argument("steps", type=int)
    sim_parser.add_argument("--diffusivity", type=float, default=1.0)
    sim_parser.add_argument("--radius-y", type=float, default=1.0)
    sim_parser.add_argument("--impedance", type=float, default=None)

    theta_parser = subparsers.add_parser("theta", help="Compute theta(t)")
    theta_parser.add_argument("t", type=float)
    theta_parser.add_argument("--terms", type=int, default=50)

    mellin_parser = subparsers.add_parser("mellin-zeta", help="Approximate Mellin zeta")
    mellin_parser.add_argument("s", type=float)
    mellin_parser.add_argument("t-min", type=float)
    mellin_parser.add_argument("t-max", type=float)
    mellin_parser.add_argument("--steps", type=int, default=2000)
    mellin_parser.add_argument("--terms", type=int, default=80)

    unification_parser = subparsers.add_parser(
        "simulate-physics", help="Simulate unification constants"
    )
    unification_parser.add_argument("--electron-radius", type=float, default=None)
    unification_parser.add_argument("--proton-radius", type=float, default=None)
    unification_parser.add_argument("--universe-radius", type=float, default=None)
    unification_parser.add_argument("--electron-mass", type=float, default=None)
    unification_parser.add_argument("--charge", type=float, default=None)
    unification_parser.add_argument("--epsilon0", type=float, default=None)
    unification_parser.add_argument("--c", type=float, default=None)
    unification_parser.add_argument("--g", type=float, default=None)
    unification_parser.add_argument("--hbar", type=float, default=None)
    unification_parser.add_argument("--k-b", type=float, default=None)

    gap_parser = subparsers.add_parser("mass-gap", help="Compute Yang-Mills gap bound")
    gap_parser.add_argument("radius_y", type=float)
    gap_parser.add_argument("--coupling", type=float, default=1.0)
    gap_parser.add_argument("--mev", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sample-orbit":
        tau = args.tau_real + 1j * args.tau_imag
        orbit = sample_modular_orbit(tau, args.steps)
        for value in orbit:
            print(f"{value.real:.6f}+{value.imag:.6f}j")
        return

    if args.command == "simulate-diffusion":
        manifold = HolomorphicManifold5D(
            grid_shape=(args.nx, args.ny, args.nz, args.ny_fiber),
            spacing=(1.0, 1.0, 1.0, 2 * np.pi * args.radius_y / args.ny_fiber),
            radius_y=args.radius_y,
        )
        field = np.zeros((args.nx, args.ny, args.nz, args.ny_fiber), dtype=float)
        field[args.nx // 2, args.ny // 2, args.nz // 2, :] = 1.0
        impedance = (
            ImpedanceBoundary(args.impedance)
            if args.impedance is not None
            else None
        )
        history = simulate_diffusion(
            manifold,
            field,
            dt=args.dt,
            steps=args.steps,
            diffusivity=args.diffusivity,
            impedance=impedance,
        )
        print(history[-1].mean())
        return

    if args.command == "theta":
        value = theta_function(args.t, n_terms=args.terms)
        print(float(value))
        return

    if args.command == "mellin-zeta":
        t = np.linspace(args.t_min, args.t_max, args.steps)
        estimate = mellin_zeta(args.s, t, n_terms=args.terms)
        print(float(np.real(estimate)))
        return

    if args.command == "simulate-physics":
        constants = PhysicalConstants(
            electron_radius=args.electron_radius or DEFAULT_CONSTANTS.electron_radius,
            proton_radius=args.proton_radius or DEFAULT_CONSTANTS.proton_radius,
            universe_radius=args.universe_radius or DEFAULT_CONSTANTS.universe_radius,
            electron_mass=args.electron_mass or DEFAULT_CONSTANTS.electron_mass,
            charge_q=args.charge or DEFAULT_CONSTANTS.charge_q,
            epsilon0=args.epsilon0 or DEFAULT_CONSTANTS.epsilon0,
            c=args.c or DEFAULT_CONSTANTS.c,
            g_const=args.g or DEFAULT_CONSTANTS.g_const,
            hbar=args.hbar or DEFAULT_CONSTANTS.hbar,
            k_b=args.k_b or DEFAULT_CONSTANTS.k_b,
        )
        results = simulate_electron_binding(constants)
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    if args.command == "mass-gap":
        if args.mev:
            print(float(mass_gap_mev(args.radius_y)))
            return
        print(float(mass_gap_bound(args.radius_y, coupling_scale=args.coupling)))
        return


if __name__ == "__main__":
    main()
