#!/usr/bin/env python
# coding: utf-8

"""DLRA workflow split into three stages with NetCDF handoff via xarray.

Stages
------
(i)   create-initial : generate initial condition dataset
(ii)  simulate       : run dynamic low-rank approximation simulation
(iii) visualize      : visualize saved simulation results
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.fft import fft, fftfreq, ifft
from tqdm import tqdm


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
    """Return f(v,x) initial distribution and grid parameters."""
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
    else:  # two-stream
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

    grid = Grid(x=x, v=v, dx=dx, dv=dv, nx=nx, nv=nv, lx=lx, lv=lv)
    return f, grid


def solve_poisson_full(f_vx: np.ndarray, grid: Grid):
    """Poisson solve from full distribution f(v,x)."""
    kx = fftfreq(grid.nx, d=grid.dx) * 2 * np.pi
    ksq_inv = np.divide(1.0, kx**2, out=np.zeros_like(kx), where=(kx != 0.0))
    ne = np.sum(f_vx, axis=0) * grid.dv
    rho = 1.0 - ne
    rho_k = fft(rho)
    phi_k = rho_k * ksq_inv
    phi = np.real(ifft(phi_k))
    E = np.real(ifft(-1j * kx * phi_k))
    return rho, phi, E


def stage_create_initial(out_nc: str, flag_init: str = "two-stream"):
    """Stage (i): create initial condition and save NetCDF."""
    f_vx, grid = build_initial_distribution(flag_init=flag_init)
    rho, phi, efield = solve_poisson_full(f_vx, grid)

    ds = xr.Dataset(
        data_vars={
            "f_init": (("x", "v"), f_vx.T),
            "rho_init": (("x",), rho),
            "phi_init": (("x",), phi),
            "E_init": (("x",), efield),
        },
        coords={"x": grid.x, "v": grid.v},
        attrs={"nx": grid.nx, "nv": grid.nv, "lx": grid.lx, "lv": grid.lv, "flag_init": flag_init},
    )
    ds.to_netcdf(out_nc)
    print(f"[stage i] wrote initial dataset: {out_nc}")


class LowRankApprox:
    def __init__(self, nx: int, nv: int, nr: int):
        self.nx = nx
        self.nv = nv
        self.nr = nr
        self.X = np.zeros((nx, nr))
        self.S = np.zeros((nr, nr))
        self.V = np.zeros((nv, nr))

    def init_from_full(self, f_xv: np.ndarray, dx: float, dv: float):
        U, s, Vt = np.linalg.svd(f_xv, full_matrices=False)
        r = min(self.nr, s.size)
        self.X[:, :r] = U[:, :r] / np.sqrt(dx)
        self.S[:r, :r] = np.sqrt(dx) * np.diag(s[:r]) * np.sqrt(dv)
        self.V[:, :r] = Vt[:r, :].T / np.sqrt(dv)

    def to_full(self, nr_truncate: int | None = None):
        r = self.nr if nr_truncate is None else nr_truncate
        return self.X[:, :r] @ self.S[:r, :r] @ self.V[:, :r].T

    def copy(self):
        new = LowRankApprox(self.nx, self.nv, self.nr)
        new.X = self.X.copy()
        new.S = self.S.copy()
        new.V = self.V.copy()
        return new


def solve_poisson_lra(f_lra: LowRankApprox, dv: float, dx: float):
    f = f_lra.to_full()
    nx = f.shape[0]
    kx = fftfreq(nx, d=dx) * 2 * np.pi
    ksq_inv = np.divide(1.0, kx**2, out=np.zeros_like(kx), where=(kx != 0.0))
    ne = np.sum(f, axis=1) * dv
    rho = 1.0 - ne
    rho_k = fft(rho)
    phi_k = rho_k * ksq_inv
    phi = np.real(ifft(phi_k))
    E = np.real(ifft(-1j * kx * phi_k))
    return rho, phi, E


def first_derivative_periodic(arr: np.ndarray, h: float, axis: int):
    return (
        -np.roll(arr, 2, axis=axis)
        + 8 * np.roll(arr, 1, axis=axis)
        - 8 * np.roll(arr, -1, axis=axis)
        + np.roll(arr, -2, axis=axis)
    ) / (12.0 * h)


def K_step_RK4(f_lra: LowRankApprox, efield: np.ndarray, dt: float, x: np.ndarray, v: np.ndarray, dx: float):
    X, S, V = f_lra.X, f_lra.S, f_lra.V
    K = X @ S
    C1 = (V.T @ (v[:, None] * V)) * (v[1] - v[0])
    dVdv = first_derivative_periodic(V, v[1] - v[0], axis=0)
    C2 = 0.5 * (((V.T @ dVdv) * (v[1] - v[0])) - ((V.T @ dVdv) * (v[1] - v[0])).T)

    def rhs(Km):
        dKdx = first_derivative_periodic(Km, dx, axis=0)
        return -dKdx @ C1.T + (efield[:, None] * Km) @ C2.T

    k1 = rhs(K)
    k2 = rhs(K + 0.5 * dt * k1)
    k3 = rhs(K + 0.5 * dt * k2)
    k4 = rhs(K + dt * k3)
    K += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    Q, R = np.linalg.qr(K, mode="reduced")
    f_lra.X = Q / np.sqrt(dx)
    f_lra.S = R * np.sqrt(dx)
    return f_lra


def S_step_RK4(f_lra: LowRankApprox, efield: np.ndarray, dt: float, x: np.ndarray, v: np.ndarray, dx: float, dv: float):
    X, S, V = f_lra.X, f_lra.S, f_lra.V
    C1 = (V.T @ (v[:, None] * V)) * dv
    C2 = 0.5 * ((V.T @ first_derivative_periodic(V, dv, axis=0)) * dv)
    C2 = C2 - C2.T
    D1 = (X.T @ (X * efield[:, None])) * dx
    D2 = 0.5 * ((X.T @ first_derivative_periodic(X, dx, axis=0)) * dx)
    D2 = D2 - D2.T

    def rhs(Sm):
        return D2 @ Sm @ C1.T - D1 @ Sm @ C2.T

    k1 = rhs(S)
    k2 = rhs(S + 0.5 * dt * k1)
    k3 = rhs(S + 0.5 * dt * k2)
    k4 = rhs(S + dt * k3)
    f_lra.S = S + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return f_lra


def L_step_RK4(f_lra: LowRankApprox, efield: np.ndarray, dt: float, x: np.ndarray, v: np.ndarray, dx: float, dv: float):
    X, S, V = f_lra.X, f_lra.S, f_lra.V
    L = V @ S.T
    D1 = (X.T @ (X * efield[:, None])) * dx
    D2 = 0.5 * ((X.T @ first_derivative_periodic(X, dx, axis=0)) * dx)
    D2 = D2 - D2.T

    def rhs(Lm):
        dLdv = first_derivative_periodic(Lm, dv, axis=0)
        return dLdv @ D1.T - (v[:, None] * Lm) @ D2.T

    k1 = rhs(L)
    k2 = rhs(L + 0.5 * dt * k1)
    k3 = rhs(L + 0.5 * dt * k2)
    k4 = rhs(L + dt * k3)
    L += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    Q, R = np.linalg.qr(L, mode="reduced")
    f_lra.V = Q / np.sqrt(dv)
    f_lra.S = R.T * np.sqrt(dv)
    return f_lra


def time_step_dlra_1st_split(f_lra: LowRankApprox, efield: np.ndarray, dt: float, x: np.ndarray, v: np.ndarray, dx: float, dv: float):
    f_lra = K_step_RK4(f_lra, efield, dt, x, v, dx)
    _, _, efield = solve_poisson_lra(f_lra, dv=dv, dx=dx)
    f_lra = S_step_RK4(f_lra, efield, dt, x, v, dx, dv)
    _, _, efield = solve_poisson_lra(f_lra, dv=dv, dx=dx)
    f_lra = L_step_RK4(f_lra, efield, dt, x, v, dx, dv)
    _, _, efield = solve_poisson_lra(f_lra, dv=dv, dx=dx)
    return f_lra, efield


def stage_simulate(initial_nc: str, out_nc: str, rank: int = 64, dt: float = 0.025, nt: int = 1000, nskip: int = 20):
    """Stage (ii): run DLRA simulation and save output NetCDF."""
    ds0 = xr.load_dataset(initial_nc)
    x = ds0["x"].values
    v = ds0["v"].values
    f_init = ds0["f_init"].values
    nx, nv = f_init.shape
    dx = x[1] - x[0]
    dv = v[1] - v[0]

    f_lra = LowRankApprox(nx, nv, rank)
    f_lra.init_from_full(f_init, dx=dx, dv=dv)

    nsave = (nt - 1) // nskip + 1
    t_all = np.zeros(nsave)
    f_all = np.zeros((nsave, nx, nv))
    rho_all = np.zeros((nsave, nx))
    phi_all = np.zeros((nsave, nx))
    e_all = np.zeros((nsave, nx))

    t = 0.0
    i_save = 0
    for _ in tqdm(range(nsave), desc="DLRA"):
        rho, phi, efield = solve_poisson_lra(f_lra, dv=dv, dx=dx)
        t_all[i_save] = t
        f_all[i_save] = f_lra.to_full()
        rho_all[i_save] = rho
        phi_all[i_save] = phi
        e_all[i_save] = efield
        i_save += 1
        for _ in range(nskip):
            f_lra, efield = time_step_dlra_1st_split(f_lra, efield, dt, x, v, dx, dv)
            t += dt

    kinetic = 0.5 * np.sum(f_all * (v[None, None, :] ** 2), axis=(1, 2)) * dx * dv
    field = 0.5 * np.sum(e_all**2, axis=1) * dx

    ds = xr.Dataset(
        data_vars={
            "f": (("time", "x", "v"), f_all),
            "rho": (("time", "x"), rho_all),
            "phi": (("time", "x"), phi_all),
            "E": (("time", "x"), e_all),
            "kinetic_energy": (("time",), kinetic),
            "field_energy": (("time",), field),
            "total_energy": (("time",), kinetic + field),
        },
        coords={"time": t_all, "x": x, "v": v},
        attrs={"rank": rank, "dt": dt, "nt": nt, "nskip": nskip, "source_initial_file": initial_nc},
    )
    ds.to_netcdf(out_nc)
    print(f"[stage ii] wrote simulation dataset: {out_nc}")


def plot_phase_and_phi(f_xv: np.ndarray, phi: np.ndarray, t: float, x: np.ndarray, v: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    plt.subplots_adjust(right=0.85)
    pcm = ax1.pcolormesh(x, v, f_xv.T, shading="auto", cmap="viridis")
    ax1.set_ylabel("v")
    ax1.set_title(f"f(x,v) at t={t:.2f}")
    cbar_ax = fig.add_axes([0.87, 0.42, 0.02, 0.46])
    fig.colorbar(pcm, cax=cbar_ax)
    ax2.plot(x, phi, "r")
    ax2.set_xlabel("x")
    ax2.set_ylabel("phi(x)")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    plt.show()


def stage_visualize(sim_nc: str, sample_times=(0.0, 10.0, 15.0, 20.0)):
    """Stage (iii): visualize simulation output from NetCDF."""
    ds = xr.load_dataset(sim_nc)
    t = ds["time"].values
    x = ds["x"].values
    v = ds["v"].values

    for t_plot in sample_times:
        idx = int(np.argmin(np.abs(t - t_plot)))
        print(f"t ≈ {t[idx]:.2f}, index={idx}")
        plot_phase_and_phi(ds["f"].values[idx], ds["phi"].values[idx], t[idx], x, v)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot()
    ax.plot(t, ds["kinetic_energy"].values, label="Kinetic")
    ax.plot(t, ds["field_energy"].values, label="Field")
    ax.plot(t, ds["total_energy"].values, label="Total")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title("Energy conservation")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="DLRA split workflow with xarray/NetCDF handoff.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("create-initial", help="(i) create initial condition NetCDF")
    p_init.add_argument("--out", default="initial_state.nc")
    p_init.add_argument("--flag-init", default="two-stream")

    p_sim = sub.add_parser("simulate", help="(ii) run DLRA simulation using initial NetCDF")
    p_sim.add_argument("--initial", default="initial_state.nc")
    p_sim.add_argument("--out", default="simulation_result.nc")
    p_sim.add_argument("--rank", type=int, default=64)
    p_sim.add_argument("--dt", type=float, default=0.025)
    p_sim.add_argument("--nt", type=int, default=1000)
    p_sim.add_argument("--nskip", type=int, default=20)

    p_viz = sub.add_parser("visualize", help="(iii) visualize simulation NetCDF")
    p_viz.add_argument("--sim", default="simulation_result.nc")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "create-initial":
        stage_create_initial(args.out, flag_init=args.flag_init)
    elif args.cmd == "simulate":
        stage_simulate(args.initial, args.out, rank=args.rank, dt=args.dt, nt=args.nt, nskip=args.nskip)
    elif args.cmd == "visualize":
        stage_visualize(args.sim)


if __name__ == "__main__":
    main()
