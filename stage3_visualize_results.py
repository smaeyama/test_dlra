#!/usr/bin/env python
# coding: utf-8

"""Stage (iii): visualization from simulation NetCDF."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import xarray as xr

from low_rank_approx import LowRankApprox


def plot_summary(ds: xr.Dataset, times: list[float]):
    t = ds["time"].values
    x = ds["x"].values
    v = ds["v"].values

    selected_indices = [int(np.argmin(np.abs(t - t_plot))) for t_plot in times]
    selected_times = t[selected_indices]

    selected_f = []
    for idx in selected_indices:
        f_lra = LowRankApprox(nx=x.size, nv=v.size, nr=ds["rank"].size)
        f_lra.init_from_tensors(
            X=ds["X"].values[idx],
            S=ds["S"].values[idx],
            V=ds["V"].values[idx],
        )
        selected_f.append(f_lra.to_full())

    selected_f = np.asarray(selected_f)
    vmin = float(np.min(selected_f))
    vmax = float(np.max(selected_f))

    n_panels = len(times)
    fig = plt.figure(figsize=(4 * n_panels + 2.5, 8), constrained_layout=True)
    gs = GridSpec(
        nrows=3,
        ncols=n_panels + 1,
        figure=fig,
        height_ratios=[2.0, 1.1, 1.4],
        width_ratios=[1] * n_panels + [0.06],
    )

    phase_axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    phi_axes = [fig.add_subplot(gs[1, i], sharex=phase_axes[i]) for i in range(n_panels)]
    cbar_ax = fig.add_subplot(gs[:2, n_panels])
    energy_ax = fig.add_subplot(gs[2, :])

    pcm = None
    for i, (idx, t_snap, ax_f, ax_phi) in enumerate(zip(selected_indices, selected_times, phase_axes, phi_axes)):
        pcm = ax_f.pcolormesh(x, v, selected_f[i].T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax_f.set_title(f"t = {t_snap:.1f}")
        if i == 0:
            ax_f.set_ylabel("v")
        else:
            ax_f.tick_params(labelleft=False)
        ax_f.tick_params(labelbottom=False)

        ax_phi.plot(x, ds["phi"].values[idx], "r")
        ax_phi.set_xlabel("x")
        if i == 0:
            ax_phi.set_ylabel("phi(x)")
        else:
            ax_phi.tick_params(labelleft=False)
        ax_phi.grid(True, alpha=0.3)

    if pcm is not None:
        fig.colorbar(pcm, cax=cbar_ax, label="f(x,v)")

    energy_ax.plot(t, ds["kinetic_energy"].values, label="Kinetic")
    energy_ax.plot(t, ds["field_energy"].values, label="Field")
    energy_ax.plot(t, ds["total_energy"].values, label="Total")
    energy_ax.set_xlabel("Time")
    energy_ax.set_ylabel("Energy")
    energy_ax.set_title("Energy evolution")
    energy_ax.grid(True, alpha=0.3)
    energy_ax.legend()

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Stage (iii): visualize simulation results")
    parser.add_argument("--sim", default="simulation_result.nc")
    parser.add_argument("--times", nargs="*", type=float, default=[0.0, 10.0, 15.0, 20.0])
    args = parser.parse_args()

    ds = xr.load_dataset(args.sim)
    t = ds["time"].values

    for t_plot in args.times:
        idx = int(np.argmin(np.abs(t - t_plot)))
        print(f"t ≈ {t[idx]:.2f}, index={idx}")

    plot_summary(ds, args.times)


if __name__ == "__main__":
    main()
