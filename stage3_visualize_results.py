#!/usr/bin/env python
# coding: utf-8

"""Stage (iii): visualize reference and DLRA simulation results side-by-side."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import xarray as xr

from low_rank_approx import LowRankApprox


def _reconstruct_dlra_f(ds: xr.Dataset, indices: list[int]) -> np.ndarray:
    x = ds["x"].values
    v = ds["v"].values
    selected_f = []
    for idx in indices:
        f_lra = LowRankApprox(nx=x.size, nv=v.size, nr=ds["rank"].size)
        f_lra.init_from_tensors(
            X=ds["X"].values[idx],
            S=ds["S"].values[idx],
            V=ds["V"].values[idx],
        )
        selected_f.append(f_lra.to_full())
    return np.asarray(selected_f)


def _select_times(ds: xr.Dataset, times: list[float]):
    t = ds["time"].values
    indices = [int(np.argmin(np.abs(t - t_plot))) for t_plot in times]
    return indices, t[indices]


def _plot_section(
    fig: plt.Figure,
    gs: GridSpec,
    row_offset: int,
    heading: str,
    ds: xr.Dataset,
    selected_indices: list[int],
    selected_times: np.ndarray,
    selected_f: np.ndarray,
):
    x = ds["x"].values
    v = ds["v"].values
    t = ds["time"].values

    n_panels = len(selected_indices)
    phase_axes = [fig.add_subplot(gs[row_offset, i]) for i in range(n_panels)]
    phi_axes = [fig.add_subplot(gs[row_offset + 1, i], sharex=phase_axes[i]) for i in range(n_panels)]
    cbar_ax = fig.add_subplot(gs[row_offset : row_offset + 2, n_panels])
    energy_ax = fig.add_subplot(gs[row_offset + 2, :])

    vmin = float(np.min(selected_f))
    vmax = float(np.max(selected_f))
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
    energy_ax.set_title(f"Energy evolution from {heading}")
    energy_ax.grid(True, alpha=0.3)
    energy_ax.legend()

    phase_axes[0].text(
        0.0,
        1.25,
        heading,
        transform=phase_axes[0].transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_summary(ds_ref: xr.Dataset, ds_dlra: xr.Dataset, times: list[float]):
    ref_indices, ref_times = _select_times(ds_ref, times)
    dlra_indices, dlra_times = _select_times(ds_dlra, times)

    f_ref = ds_ref["f"].values[ref_indices]
    f_dlra = _reconstruct_dlra_f(ds_dlra, dlra_indices)

    n_panels = len(times)
    fig = plt.figure(figsize=(4 * n_panels + 2.5, 16), constrained_layout=True)
    gs = GridSpec(
        nrows=6,
        ncols=n_panels + 1,
        figure=fig,
        height_ratios=[2.0, 1.1, 1.4, 2.0, 1.1, 1.4],
        width_ratios=[1] * n_panels + [0.06],
    )

    _plot_section(
        fig=fig,
        gs=gs,
        row_offset=0,
        heading="Grid-based Vlasov simulation",
        ds=ds_ref,
        selected_indices=ref_indices,
        selected_times=ref_times,
        selected_f=f_ref,
    )
    _plot_section(
        fig=fig,
        gs=gs,
        row_offset=3,
        heading="Dynamic Low-Rank Approximation",
        ds=ds_dlra,
        selected_indices=dlra_indices,
        selected_times=dlra_times,
        selected_f=f_dlra,
    )

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Stage (iii): visualize reference and DLRA results")
    parser.add_argument("--reference", default="reference_result.nc")
    parser.add_argument("--sim", default="simulation_result.nc")
    parser.add_argument("--times", nargs="*", type=float, default=[0.0, 10.0, 15.0, 20.0])
    args = parser.parse_args()

    ds_ref = xr.load_dataset(args.reference)
    ds_dlra = xr.load_dataset(args.sim)

    t_ref = ds_ref["time"].values
    t_dlra = ds_dlra["time"].values
    for t_plot in args.times:
        idx_ref = int(np.argmin(np.abs(t_ref - t_plot)))
        idx_dlra = int(np.argmin(np.abs(t_dlra - t_plot)))
        print(f"reference: t ≈ {t_ref[idx_ref]:.2f}, index={idx_ref}")
        print(f"dlra:      t ≈ {t_dlra[idx_dlra]:.2f}, index={idx_dlra}")

    plot_summary(ds_ref, ds_dlra, args.times)


if __name__ == "__main__":
    main()
