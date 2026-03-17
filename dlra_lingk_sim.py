#!/usr/bin/env python
# coding: utf-8

"""Run a DLRA linear gyrokinetic simulation and save factors to NetCDF."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr
from tqdm import tqdm

from linear_gyrokinetic import (
    GKParameters,
    build_geometry,
    complex_to_parts,
    compute_time_step_control,
    compute_density_moment_from_h_factors,
    flatten_vm,
    init_state,
    projector_splitting_step,
    solve_fields_from_h_factors,
    weighted_gram,
)
from low_rank_approx import LowRankApprox


EPS = 1.0e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DLRA linear gyrokinetic solver")
    parser.add_argument("--out", default="lingk_dlra.nc")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dt-out", type=float, default=None)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--nskip", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--nz", type=int, default=24 * 5)
    parser.add_argument("--nv", type=int, default=32)
    parser.add_argument("--nm", type=int, default=31)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ky", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--disable-dtc", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def _resolve_time_controls(args: argparse.Namespace, step_dt: float | None = None) -> tuple[float, float]:
    dt_base = args.dt if step_dt is None else float(step_dt)

    if args.dt_out is not None or args.time_limit is not None:
        dt_out = 0.1 if args.dt_out is None else args.dt_out
        time_limit = 10.0 if args.time_limit is None else args.time_limit
        return float(dt_out), float(time_limit)

    if args.nt is not None or args.nskip is not None:
        nt = 41 if args.nt is None else args.nt
        nskip = 10 if args.nskip is None else args.nskip
        if nt < 1:
            raise ValueError("--nt must be >= 1")
        if nskip < 1:
            raise ValueError("--nskip must be >= 1")
        return float(dt_base * nskip), float(dt_base * (nt - 1))

    return 0.1, 10.0


def _build_parameters(args: argparse.Namespace, dt_out: float) -> GKParameters:
    return GKParameters(
        nz=args.nz,
        nv=args.nv,
        nm=args.nm,
        dt=args.dt,
        dt_out=dt_out,
        seed=args.seed,
        ky=args.ky,
        beta=args.beta,
    )


def main() -> None:
    total_start = perf_counter()
    init_elapsed = 0.0
    dlra_elapsed = 0.0
    sample_elapsed = 0.0
    output_elapsed = 0.0

    init_start = perf_counter()
    args = parse_args()
    dt_out, time_limit = _resolve_time_controls(args)
    params = _build_parameters(args, dt_out)
    geom = build_geometry(params)
    dt_control = compute_time_step_control(geom)
    if not args.disable_dtc:
        params.dt = dt_control["dt"]
        if args.dt_out is None and args.time_limit is None and (args.nt is not None or args.nskip is not None):
            params.dt_out, time_limit = _resolve_time_controls(args, step_dt=params.dt)

    if args.rank > min(geom.nz_tot, geom.nv_tot * geom.nm_tot * params.ns):
        raise ValueError(
            f"rank must be <= min(nz_tot, nv_tot*nm_tot*ns) = {min(geom.nz_tot, geom.nv_tot * geom.nm_tot * params.ns)} "
            f"(got {args.rank})"
        )

    print(" # Time step size control")
    print("")
    print(f" # courant num. = {dt_control['courant_num']:20.15f}")
    print(f" # dt_perp      = {dt_control['dt_perp']:23.15E}")
    print(f" # dt_zz        = {dt_control['dt_zz']:23.15E}")
    print(f" # dt_vl        = {dt_control['dt_vl']:23.15E}")
    print(f" # dt_col       = {dt_control['dt_col']:23.15E}")
    print(f" # dt           = {dt_control['dt']:23.15E}")
    print("")

    hk, fk, pk, ak = init_state(params, geom)
    h_lra = LowRankApprox(geom.nz_tot, geom.nv_tot * geom.nm_tot * params.ns, args.rank, dtype=np.complex128)
    h_lra.init_from_full(np.asarray(flatten_vm(hk)), dx=geom.dz, dv=geom.vm_weight)
    init_elapsed += perf_counter() - init_start

    times: list[float] = []
    X_series: list[np.ndarray] = []
    S_series: list[np.ndarray] = []
    V_series: list[np.ndarray] = []
    X_gram_series: list[np.ndarray] = []
    V_gram_series: list[np.ndarray] = []
    phi_series: list[np.ndarray] = []
    a_series: list[np.ndarray] = []
    dens_series: list[np.ndarray] = []

    time = 0.0
    time_out = time + params.dt_out - EPS

    def record_state(current_time: float) -> None:
        left_factor = h_lra.X @ h_lra.S
        pk_full, ak_full = solve_fields_from_h_factors(left_factor, h_lra.V, geom)
        density = compute_density_moment_from_h_factors(left_factor, h_lra.V, ak_full, geom)

        times.append(float(current_time))
        X_series.append(np.asarray(h_lra.X))
        S_series.append(np.asarray(h_lra.S))
        V_series.append(np.asarray(h_lra.V))
        X_gram_series.append(weighted_gram(h_lra.X, geom.dz))
        V_gram_series.append(weighted_gram(h_lra.V, geom.vm_weight))
        phi_series.append(np.asarray(pk_full))
        a_series.append(np.asarray(ak_full))
        dens_series.append(np.asarray(density))

    sample_start = perf_counter()
    record_state(time)
    sample_elapsed += perf_counter() - sample_start

    iterator = range(args.max_steps + 1)
    if not args.disable_progress:
        iterator = tqdm(iterator, desc="lingk-dlra", unit="step")

    for istep in iterator:
        if time > time_limit:
            break

        dlra_start = perf_counter()
        h_lra = projector_splitting_step(h_lra, params.dt, geom)
        dlra_elapsed += perf_counter() - dlra_start
        time += params.dt

        if not args.disable_progress:
            assert isinstance(iterator, tqdm)
            iterator.set_postfix(step=istep + 1, time=f"{time:.3f}", next_out=f"{time_out + EPS:.3f}")

        if time > time_out:
            sample_start = perf_counter()
            record_state(time)
            sample_elapsed += perf_counter() - sample_start
            time_out += params.dt_out

    times_arr = np.asarray(times)
    X_arr = np.asarray(X_series)
    S_arr = np.asarray(S_series)
    V_arr = np.asarray(V_series)
    X_gram_arr = np.asarray(X_gram_series)
    V_gram_arr = np.asarray(V_gram_series)
    phi_arr = np.asarray(phi_series)
    a_arr = np.asarray(a_series)
    dens_arr = np.asarray(dens_series)

    X_real, X_imag = complex_to_parts(X_arr)
    S_real, S_imag = complex_to_parts(S_arr)
    V_real, V_imag = complex_to_parts(V_arr)
    X_gram_real, X_gram_imag = complex_to_parts(X_gram_arr)
    V_gram_real, V_gram_imag = complex_to_parts(V_gram_arr)
    phi_real, phi_imag = complex_to_parts(phi_arr)
    a_real, a_imag = complex_to_parts(a_arr)
    dens_real, dens_imag = complex_to_parts(dens_arr)

    out_start = perf_counter()
    ds = xr.Dataset(
        data_vars={
            "X_real": (("time", "z", "rankx"), X_real),
            "X_imag": (("time", "z", "rankx"), X_imag),
            "S_real": (("time", "rankx", "rankv"), S_real),
            "S_imag": (("time", "rankx", "rankv"), S_imag),
            "V_real": (("time", "vm", "rankv"), V_real),
            "V_imag": (("time", "vm", "rankv"), V_imag),
            "X_gram_real": (("time", "rankx", "rankv"), X_gram_real),
            "X_gram_imag": (("time", "rankx", "rankv"), X_gram_imag),
            "V_gram_real": (("time", "rankx", "rankv"), V_gram_real),
            "V_gram_imag": (("time", "rankx", "rankv"), V_gram_imag),
            "phi_real": (("time", "z"), phi_real),
            "phi_imag": (("time", "z"), phi_imag),
            "A_real": (("time", "z"), a_real),
            "A_imag": (("time", "z"), a_imag),
            "dens_real": (("time", "z", "species"), dens_real),
            "dens_imag": (("time", "z", "species"), dens_imag),
        },
        coords={
            "time": times_arr,
            "z": np.asarray(geom.zz),
            "vl": np.asarray(geom.vl),
            "mu": np.asarray(geom.mu),
            "vm": np.arange(geom.nv_tot * geom.nm_tot * params.ns),
            "rank": np.arange(args.rank),
            "rankx": np.arange(args.rank),
            "rankv": np.arange(args.rank),
            "species": np.arange(params.ns),
        },
        attrs={
            "source": "dlra_lingk_sim.py",
            "rank": args.rank,
            "dt": params.dt,
            "dt_out": params.dt_out,
            "time_limit": time_limit,
            "seed": params.seed,
            "ky": params.ky,
            "beta": params.beta,
            "nz": params.nz,
            "nv": params.nv,
            "nm": params.nm,
        },
    )
    out_path = Path(args.out)
    ds.to_netcdf(out_path)
    output_elapsed += perf_counter() - out_start

    total_elapsed = perf_counter() - total_start
    other_elapsed = total_elapsed - (init_elapsed + dlra_elapsed + sample_elapsed + output_elapsed)
    nsteps_done = int(np.floor(time / params.dt + 1.0e-12))

    print(f"[lingk-dlra] wrote dataset: {out_path}")
    print(
        f"[lingk-dlra] parameters: rank={args.rank}, nz={params.nz}, nv={params.nv}, nm={params.nm}, "
        f"dt={params.dt}, dt_out={params.dt_out}"
    )
    print("")
    print(" ### Elapsed time ###")
    print(f" # Time steps = {nsteps_done:12d}")
    print(" #")
    print(f" #      Total = {total_elapsed:18.15f}")
    print(f" #       Init = {init_elapsed:18.15E}")
    print(f" #       DLRA = {dlra_elapsed:18.15f}")
    print(f" #     Sample = {sample_elapsed:18.15f}")
    print(f" #     Output = {output_elapsed:18.15f}")
    print(f" #      Other = {other_elapsed:18.15E}")
    print(" End program.")


if __name__ == "__main__":
    main()
