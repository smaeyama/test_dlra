#!/usr/bin/env python3

"""Run the existing scripts for a two-stream example."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the two-stream example with the existing scripts")
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "examples" / "output" / "two_stream")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--nv", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--nt", type=int, default=4000)
    parser.add_argument("--nskip", type=int, default=10)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lx", type=float, default=100.0)
    parser.add_argument("--lv", type=float, default=9.0)
    parser.add_argument("--reference-solver", choices=["semi-Lagrangian", "finite-difference"], default="finite-difference")
    parser.add_argument("--times", nargs="*", type=float, default=[0.0, 10.0, 20.0, 30.0])
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=_command_env())


def _command_env() -> dict[str, str]:
    env = os.environ.copy()
    user_site = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    existing_pythonpath = env.get("PYTHONPATH")
    if user_site.exists():
        env["PYTHONPATH"] = f"{user_site}:{existing_pythonpath}" if existing_pythonpath else str(user_site)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-temp_test_dlra")
    env.setdefault("MPLBACKEND", "Agg")
    return env


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    initial_path = args.outdir / "initial_state.nc"
    reference_path = args.outdir / "reference_result.nc"
    sim_path = args.outdir / "simulation_result.nc"
    figure_path = args.outdir / "two_stream_summary.png"

    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "reference_Vlasov_sim.py"),
            "--out",
            str(initial_path),
            "--reference-out",
            str(reference_path),
            "--flag-init",
            "two-stream",
            "--solver",
            args.reference_solver,
            "--nx",
            str(args.nx),
            "--nv",
            str(args.nv),
            "--lx",
            str(args.lx),
            "--lv",
            str(args.lv),
            "--dt",
            str(args.dt),
            "--nt",
            str(args.nt),
            "--nskip",
            str(args.nskip),
            "--seed",
            str(args.seed),
        ]
    )

    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "dlra_Vlasov_sim.py"),
            "--initial",
            str(initial_path),
            "--out",
            str(sim_path),
            "--rank",
            str(args.rank),
            "--dt",
            str(args.dt),
            "--nt",
            str(args.nt),
            "--nskip",
            str(args.nskip),
        ]
    )

    plot_command = [
        sys.executable,
        str(REPO_ROOT / "plot_figure.py"),
        "--reference",
        str(reference_path),
        "--sim",
        str(sim_path),
        "--plot-mode",
        "full",
        "--save",
        str(figure_path),
        "--no-show",
    ]
    if args.times:
        plot_command.extend(["--times", *[str(time) for time in args.times]])
    run_command(plot_command)

    print(f"Saved initial condition to {initial_path}")
    print(f"Saved reference simulation to {reference_path}")
    print(f"Saved DLRA simulation to {sim_path}")
    print(f"Saved figure to {figure_path}")
    print(f"Configured simulation end time: {(args.nt - 1) * args.dt:.3f} (nt={args.nt}, dt={args.dt})")


if __name__ == "__main__":
    main()
