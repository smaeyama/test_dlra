#!/usr/bin/env python
# coding: utf-8

"""Stage (iii): visualization from simulation NetCDF."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


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


def main():
    parser = argparse.ArgumentParser(description="Stage (iii): visualize simulation results")
    parser.add_argument("--sim", default="simulation_result.nc")
    parser.add_argument("--times", nargs="*", type=float, default=[0.0, 10.0, 15.0, 20.0])
    args = parser.parse_args()

    ds = xr.load_dataset(args.sim)
    t = ds["time"].values
    x = ds["x"].values
    v = ds["v"].values

    for t_plot in args.times:
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


if __name__ == "__main__":
    main()
