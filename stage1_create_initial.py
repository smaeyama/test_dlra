#!/usr/bin/env python
# coding: utf-8

"""Stage (i): initial value creation and NetCDF output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.fft import fft, fftfreq, ifft


@dataclass
class Grid:
    x: np.ndarray
    v: np.ndarray
    dx: float
    dv: float
    nx: int
    nv: int
    lx: float
    lv: float


def build_initial_distribution(flag_init: str = "two-stream", seed: int = 0):
    if flag_init in {"linear-Landau", "nonlinear-Landau", "bump-on-tail"}:
        nx, nv = 128, 256
        lx, lv = 10 * np.pi, 5.0
    elif flag_init == "two-stream":
        nx, nv = 128, 256
        lx, lv = 100.0, 9.0
    else:
        raise ValueError(f"Unknown flag_init: {flag_init}")

    dx, dv = lx / nx, 2 * lv / (nv - 1)
    x = np.linspace(0.0, lx, nx, endpoint=False)
    v = np.linspace(-lv, lv, nv)
    fmx = np.exp(-0.5 * v**2) / np.sqrt(2 * np.pi)

    np.random.seed(seed)
    if flag_init == "linear-Landau":
        ampl = 1e-3
        f0_v = fmx.copy()
        f = f0_v[:, None] + ampl * np.cos(4 * np.pi * x / lx)[None, :] * fmx[:, None]
    elif flag_init == "nonlinear-Landau":
        ampl = 0.2
        f0_v = fmx.copy()
        f = f0_v[:, None] + ampl * np.cos(4 * np.pi * x / lx)[None, :] * fmx[:, None]
    elif flag_init == "bump-on-tail":
        ampl = 1e-3
        nb, vb, vtb = 0.2, 2.0, 0.3
        bump = np.exp(-0.5 * ((v - vb) / vtb) ** 2) / np.sqrt(2 * np.pi * vtb**2)
        f0_v = (1.0 - nb) * fmx + nb * bump
        f = np.zeros((nv, nx))
        rand_phases = np.random.rand(nx // 4)
        for ik in range(1, nx // 4):
            phase = 2 * np.pi * rand_phases[ik]
            f += ampl * np.cos(2 * np.pi * (ik * x / lx + phase))[None, :] * fmx[:, None]
        f += f0_v[:, None]
    else:
        ampl = 2e-3
        nb, vb, vtb = 0.5, 3.0, 1.0
        f0b_pos = np.exp(-0.5 * ((v - vb) / vtb) ** 2) / np.sqrt(2 * np.pi * vtb**2)
        f0b_neg = np.exp(-0.5 * ((v + vb) / vtb) ** 2) / np.sqrt(2 * np.pi * vtb**2)
        f0_v = (1.0 - nb) * f0b_pos + nb * f0b_neg
        f = np.zeros((nv, nx))
        phase1 = 2 * np.pi * np.random.rand(nx // 4)
        phase2 = 2 * np.pi * np.random.rand(nx // 4)
        for ik in range(1, nx // 4):
            f += (
                ampl * np.cos(2 * np.pi * (ik * x / lx + phase1[ik]))[None, :] * f0b_pos[:, None]
                + ampl * np.cos(2 * np.pi * (ik * x / lx + phase2[ik]))[None, :] * f0b_neg[:, None]
            )
        f += f0_v[:, None]

    return f, Grid(x=x, v=v, dx=dx, dv=dv, nx=nx, nv=nv, lx=lx, lv=lv)


def solve_poisson_full(f_vx: np.ndarray, grid: Grid):
    kx = fftfreq(grid.nx, d=grid.dx) * 2 * np.pi
    ksq_inv = np.divide(1.0, kx**2, out=np.zeros_like(kx), where=(kx != 0.0))
    ne = np.sum(f_vx, axis=0) * grid.dv
    rho = 1.0 - ne
    rho_k = fft(rho)
    phi_k = rho_k * ksq_inv
    phi = np.real(ifft(phi_k))
    efield = np.real(ifft(-1j * kx * phi_k))
    return rho, phi, efield


def time_advance_in_x(f_vx: np.ndarray, dt: float, grid: Grid):
    kx = fftfreq(grid.nx, d=grid.dx) * 2 * np.pi
    f_hat = fft(f_vx, axis=1)
    phase = np.exp(-1j * kx[None, :] * grid.v[:, None] * dt)
    f_new = np.real(ifft(f_hat * phase, axis=1))
    rho, phi, efield = solve_poisson_full(f_new, grid)
    return f_new, rho, phi, efield


def time_advance_in_v(f_vx: np.ndarray, efield: np.ndarray, dt: float, grid: Grid):
    kv = fftfreq(grid.nv, d=grid.dv) * 2 * np.pi
    f_hat = fft(f_vx, axis=0)
    phase = np.exp(1j * kv[:, None] * efield[None, :] * dt)
    return np.real(ifft(f_hat * phase, axis=0))


def run_reference_simulation(f_init_vx: np.ndarray, grid: Grid, dt: float, nt: int, nskip: int):
    nsave = (nt - 1) // nskip + 1
    t_all = np.zeros(nsave)
    f_all = np.zeros((nsave, grid.nv, grid.nx))
    rho_all = np.zeros((nsave, grid.nx))
    phi_all = np.zeros((nsave, grid.nx))
    e_all = np.zeros((nsave, grid.nx))

    f = f_init_vx.copy()
    rho, phi, efield = solve_poisson_full(f, grid)
    t = 0.0

    for i_save in range(nsave):
        t_all[i_save] = t
        f_all[i_save] = f
        rho_all[i_save] = rho
        phi_all[i_save] = phi
        e_all[i_save] = efield

        for _ in range(nskip):
            f, rho, phi, efield = time_advance_in_x(f, dt / 2.0, grid)
            f = time_advance_in_v(f, efield, dt, grid)
            f, rho, phi, efield = time_advance_in_x(f, dt / 2.0, grid)
            t += dt

    return t_all, f_all, rho_all, phi_all, e_all


def main():
    parser = argparse.ArgumentParser(description="Stage (i): create initial NetCDF")
    parser.add_argument("--out", default="initial_state.nc")
    parser.add_argument("--reference-out", default="reference_result.nc")
    parser.add_argument("--flag-init", default="two-stream")
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--nskip", type=int, default=20)
    args = parser.parse_args()

    f_vx, grid = build_initial_distribution(flag_init=args.flag_init)
    rho, phi, efield = solve_poisson_full(f_vx, grid)

    ds = xr.Dataset(
        data_vars={
            "f_init": (("x", "v"), f_vx.T),
            "rho_init": (("x",), rho),
            "phi_init": (("x",), phi),
            "E_init": (("x",), efield),
        },
        coords={"x": grid.x, "v": grid.v},
        attrs={"nx": grid.nx, "nv": grid.nv, "lx": grid.lx, "lv": grid.lv, "flag_init": args.flag_init},
    )
    ds.to_netcdf(args.out)
    print(f"[stage i] wrote initial dataset: {args.out}")

    t_all, f_all, rho_all, phi_all, e_all = run_reference_simulation(
        f_init_vx=f_vx,
        grid=grid,
        dt=args.dt,
        nt=args.nt,
        nskip=args.nskip,
    )
    kinetic = 0.5 * np.sum(f_all * (grid.v[:, None] ** 2)[None, :, :], axis=(1, 2)) * grid.dx * grid.dv
    field = 0.5 * np.sum(e_all**2, axis=1) * grid.dx
    ds_ref = xr.Dataset(
        data_vars={
            "f": (("time", "x", "v"), np.transpose(f_all, (0, 2, 1))),
            "rho": (("time", "x"), rho_all),
            "phi": (("time", "x"), phi_all),
            "E": (("time", "x"), e_all),
            "kinetic_energy": (("time",), kinetic),
            "field_energy": (("time",), field),
            "total_energy": (("time",), kinetic + field),
        },
        coords={"time": t_all, "x": grid.x, "v": grid.v},
        attrs={
            "dt": args.dt,
            "nt": args.nt,
            "nskip": args.nskip,
            "flag_init": args.flag_init,
            "source_initial_file": args.out,
        },
    )
    ds_ref.to_netcdf(args.reference_out)
    print(f"[stage i] wrote reference simulation dataset: {args.reference_out}")


if __name__ == "__main__":
    main()
