#!/usr/bin/env python
# coding: utf-8

"""Stage (ii): dynamic low-rank approximation simulation and NetCDF output."""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr
from scipy.fft import fft, fftfreq, ifft
from tqdm import tqdm

from low_rank_approx import LowRankApprox


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
    efield = np.real(ifft(-1j * kx * phi_k))
    return rho, phi, efield


def first_derivative_periodic(arr: np.ndarray, h: float, axis: int):
    return (
        -np.roll(arr, 2, axis=axis)
        + 8 * np.roll(arr, 1, axis=axis)
        - 8 * np.roll(arr, -1, axis=axis)
        + np.roll(arr, -2, axis=axis)
    ) / (12.0 * h)


def k_step_rk4(f_lra: LowRankApprox, efield: np.ndarray, dt: float, v: np.ndarray, dx: float, dv: float):
    X, S, V = f_lra.X, f_lra.S, f_lra.V
    K = X @ S
    C1 = (V.T @ (v[:, None] * V)) * dv
    dVdv = first_derivative_periodic(V, dv, axis=0)
    VV = (V.T @ dVdv) * dv
    C2 = 0.5 * (VV - VV.T)

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


def s_step_rk4(f_lra: LowRankApprox, efield: np.ndarray, dt: float, v: np.ndarray, dx: float, dv: float):
    X, S, V = f_lra.X, f_lra.S, f_lra.V
    C1 = (V.T @ (v[:, None] * V)) * dv
    VV = (V.T @ first_derivative_periodic(V, dv, axis=0)) * dv
    C2 = 0.5 * (VV - VV.T)
    D1 = (X.T @ (X * efield[:, None])) * dx
    XX = (X.T @ first_derivative_periodic(X, dx, axis=0)) * dx
    D2 = 0.5 * (XX - XX.T)

    def rhs(Sm):
        return D2 @ Sm @ C1.T - D1 @ Sm @ C2.T

    k1 = rhs(S)
    k2 = rhs(S + 0.5 * dt * k1)
    k3 = rhs(S + 0.5 * dt * k2)
    k4 = rhs(S + dt * k3)
    f_lra.S = S + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return f_lra


def l_step_rk4(f_lra: LowRankApprox, efield: np.ndarray, dt: float, v: np.ndarray, dx: float, dv: float):
    X, S, V = f_lra.X, f_lra.S, f_lra.V
    L = V @ S.T
    D1 = (X.T @ (X * efield[:, None])) * dx
    XX = (X.T @ first_derivative_periodic(X, dx, axis=0)) * dx
    D2 = 0.5 * (XX - XX.T)

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


def step_split(f_lra: LowRankApprox, efield: np.ndarray, dt: float, v: np.ndarray, dx: float, dv: float):
    f_lra = k_step_rk4(f_lra, efield, dt, v, dx, dv)
    _, _, efield = solve_poisson_lra(f_lra, dv=dv, dx=dx)
    f_lra = s_step_rk4(f_lra, efield, dt, v, dx, dv)
    _, _, efield = solve_poisson_lra(f_lra, dv=dv, dx=dx)
    f_lra = l_step_rk4(f_lra, efield, dt, v, dx, dv)
    _, _, efield = solve_poisson_lra(f_lra, dv=dv, dx=dx)
    return f_lra, efield


def main():
    parser = argparse.ArgumentParser(description="Stage (ii): simulate DLRA from initial NetCDF")
    parser.add_argument("--initial", default="initial_state.nc")
    parser.add_argument("--out", default="simulation_result.nc")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--nskip", type=int, default=20)
    args = parser.parse_args()

    ds0 = xr.load_dataset(args.initial)
    x = ds0["x"].values
    v = ds0["v"].values
    f_init = ds0["f_init"].values

    nx, nv = f_init.shape
    dx = float(x[1] - x[0])
    dv = float(v[1] - v[0])

    f_lra = LowRankApprox(nx, nv, args.rank)
    f_lra.init_from_full(f_init, dx=dx, dv=dv)

    nsave = (args.nt - 1) // args.nskip + 1
    t_all = np.zeros(nsave)
    X_all = np.zeros((nsave, nx, args.rank))
    S_all = np.zeros((nsave, args.rank, args.rank))
    V_all = np.zeros((nsave, nv, args.rank))
    rho_all = np.zeros((nsave, nx))
    phi_all = np.zeros((nsave, nx))
    e_all = np.zeros((nsave, nx))

    t = 0.0
    for i_save in tqdm(range(nsave), desc="DLRA"):
        rho, phi, efield = solve_poisson_lra(f_lra, dv=dv, dx=dx)
        t_all[i_save] = t
        X_all[i_save] = f_lra.X
        S_all[i_save] = f_lra.S
        V_all[i_save] = f_lra.V
        rho_all[i_save] = rho
        phi_all[i_save] = phi
        e_all[i_save] = efield

        for _ in range(args.nskip):
            f_lra, efield = step_split(f_lra, efield, args.dt, v, dx, dv)
            t += args.dt

    x_weight = np.sum(X_all, axis=1) * dx
    v2_weight = (v**2) @ V_all * dv
    kinetic = 0.5 * np.einsum("ti,tij,tj->t", x_weight, S_all, v2_weight)
    field = 0.5 * np.sum(e_all**2, axis=1) * dx

    ds = xr.Dataset(
        data_vars={
            "X": (("time", "x", "rankx"), X_all),
            "S": (("time", "rankx", "rankv"), S_all),
            "V": (("time", "v", "rankv"), V_all),
            "rho": (("time", "x"), rho_all),
            "phi": (("time", "x"), phi_all),
            "E": (("time", "x"), e_all),
            "kinetic_energy": (("time",), kinetic),
            "field_energy": (("time",), field),
            "total_energy": (("time",), kinetic + field),
        },
        coords={"time": t_all, "x": x, "v": v, "rank": np.arange(args.rank)},
        attrs={
            "rank": args.rank,
            "dt": args.dt,
            "nt": args.nt,
            "nskip": args.nskip,
            "source_initial_file": args.initial,
        },
    )
    ds.to_netcdf(args.out)
    print(f"[stage ii] wrote simulation dataset: {args.out}")


if __name__ == "__main__":
    main()
