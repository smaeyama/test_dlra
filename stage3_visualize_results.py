#!/usr/bin/env python
# coding: utf-8

"""Stage (iii): visualization from simulation NetCDF."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import xarray as xr


def plot_summary(ds: xr.Dataset, times: list[float]):
    t = ds["time"].values
    x = ds["x"].values
    v = ds["v"].values

    selected_indices = [int(np.argmin(np.abs(t - t_plot))) for t_plot in times]
    selected_times = t[selected_indices]
    selected_f = ds["f"].values[selected_indices]

    vmin = float(np.min(selected_f))
    vmax = float(np.max(selected_f))

    n_panels = len(times)
    fig = plt.figure(figsize=(4 * n_panels + 2.5, 7), constrained_layout=True)
    gs = GridSpec(
        nrows=2,
        ncols=n_panels + 1,
        figure=fig,
        height_ratios=[2.2, 1],
        width_ratios=[1] * n_panels + [0.06],
    )

    phase_axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
    cbar_ax = fig.add_subplot(gs[0, n_panels])
    energy_ax = fig.add_subplot(gs[1, :])

    pcm = None
    for i, (idx, t_snap, ax) in enumerate(zip(selected_indices, selected_times, phase_axes)):
        pcm = ax.pcolormesh(x, v, ds["f"].values[idx].T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {t_snap:.1f}")
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("v")
        else:
            ax.tick_params(labelleft=False)

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
