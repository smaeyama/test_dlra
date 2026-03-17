#!/usr/bin/env python
# coding: utf-8

"""Animate the time evolution of |f(z,v)| from lingk_dlra.nc."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-temp_test_dlra")
import matplotlib
if os.environ.get("MPLBACKEND"):
    matplotlib.use(os.environ["MPLBACKEND"])
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import xarray as xr

from linear_gyrokinetic import GKParameters, build_geometry, hh_to_ff, unflatten_vm


def _scalar_attr(value, default: float) -> float:
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate |f(z,v)| from lingk_dlra.nc")
    parser.add_argument("--input", default="lingk_dlra.nc")
    parser.add_argument("--mu-index", type=int, default=None, help="Select a mu entry from the NetCDF file")
    parser.add_argument("--species", type=int, default=0)
    parser.add_argument("--interval", type=int, default=100, help="Animation interval in ms")
    parser.add_argument("--every", type=int, default=1, help="Use every N-th saved time slice")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--save", default=None, help="Optional output animation path (.gif/.mp4)")
    parser.add_argument("--frames-dir", default=None, help="Optional directory for PNG frame dumps")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def load_dlra_fkinzv(
    path: str | Path,
    mu_index: int | None,
    species: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    try:
        with h5py.File(path, "r") as f:
            time = f["time"][()]
            z = f["z"][()]
            vl = f["vl"][()]
            mu = f["mu"][()]
            species_values = f["species"][()]
            ky = _scalar_attr(f.attrs.get("ky"), 0.2)
            beta = _scalar_attr(f.attrs.get("beta"), 0.0)

            if "f_real" in f and "f_imag" in f:
                f_real = f["f_real"][()]
                f_imag = f["f_imag"][()]
                if species < 0 or species >= f_real.shape[-1]:
                    raise ValueError(f"species must satisfy 0 <= species < {f_real.shape[-1]}")
                if mu_index is None:
                    mu_pos = 0
                else:
                    if mu_index < 0 or mu_index >= mu.size:
                        raise ValueError(f"mu_index must satisfy 0 <= mu_index < {mu.size}")
                    mu_pos = int(mu_index)
                f_abs = np.sqrt(f_real[:, :, :, mu_pos, species] ** 2 + f_imag[:, :, :, mu_pos, species] ** 2)
                return time, z, vl, f_abs, mu_pos

            x_real = f["X_real"][()]
            x_imag = f["X_imag"][()]
            s_real = f["S_real"][()]
            s_imag = f["S_imag"][()]
            v_real = f["V_real"][()]
            v_imag = f["V_imag"][()]
            a_real = f["A_real"][()]
            a_imag = f["A_imag"][()]
    except OSError:
        ds = xr.load_dataset(path)
        time = ds["time"].values
        z = ds["z"].values
        vl = ds["vl"].values
        mu = ds["mu"].values
        species_values = ds["species"].values
        ky = _scalar_attr(ds.attrs.get("ky"), 0.2)
        beta = _scalar_attr(ds.attrs.get("beta"), 0.0)

        if "f_real" in ds and "f_imag" in ds:
            f_real = ds["f_real"].values
            f_imag = ds["f_imag"].values
            ds.close()
            if species < 0 or species >= f_real.shape[-1]:
                raise ValueError(f"species must satisfy 0 <= species < {f_real.shape[-1]}")
            if mu_index is None:
                mu_pos = 0
            else:
                if mu_index < 0 or mu_index >= mu.size:
                    raise ValueError(f"mu_index must satisfy 0 <= mu_index < {mu.size}")
                mu_pos = int(mu_index)
            f_abs = np.sqrt(f_real[:, :, :, mu_pos, species] ** 2 + f_imag[:, :, :, mu_pos, species] ** 2)
            return time, z, vl, f_abs, mu_pos

        x_real = ds["X_real"].values
        x_imag = ds["X_imag"].values
        s_real = ds["S_real"].values
        s_imag = ds["S_imag"].values
        v_real = ds["V_real"].values
        v_imag = ds["V_imag"].values
        a_real = ds["A_real"].values
        a_imag = ds["A_imag"].values
        ds.close()

    ns = int(species_values.size)
    if species < 0 or species >= ns:
        raise ValueError(f"species must satisfy 0 <= species < {ns}")

    if mu_index is None:
        mu_pos = 0
    else:
        if mu_index < 0 or mu_index >= mu.size:
            raise ValueError(f"mu_index must satisfy 0 <= mu_index < {mu.size}")
        mu_pos = int(mu_index)

    params = GKParameters(nz=z.size // 2, nv=vl.size // 2, nm=mu.size - 1, ky=ky, beta=beta)
    geom = build_geometry(params)

    x = x_real + 1j * x_imag
    s = s_real + 1j * s_imag
    v = v_real + 1j * v_imag
    ak = a_real + 1j * a_imag

    ntime = time.size
    f_abs = np.empty((ntime, z.size, vl.size), dtype=float)
    for it in range(ntime):
        h_matrix = x[it] @ s[it] @ v[it].T
        hk = unflatten_vm(h_matrix, geom)
        fk = hh_to_ff(hk, ak[it], geom)
        f_abs[it] = np.abs(np.asarray(fk[:, :, mu_pos, species]))
    return time, z, vl, f_abs, mu_pos


def save_frames(
    selected_times: np.ndarray,
    z: np.ndarray,
    vl: np.ndarray,
    selected_f_abs: np.ndarray,
    mu_index: int,
    frames_dir: Path,
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for iframe, (time_value, frame) in enumerate(zip(selected_times, selected_f_abs)):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        vmax = float(np.max(frame)) if np.max(frame) > 0.0 else 1.0
        mesh = ax.pcolormesh(z, vl, frame.T, shading="auto", cmap="plasma", vmin=0.0, vmax=vmax)
        ax.set_xlabel("z")
        ax.set_ylabel("v_parallel")
        ax.text(
            0.02,
            0.96,
            f"t = {float(time_value):.8f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        fig.colorbar(mesh, ax=ax, label="|f(z,v)|").ax.set_title(f"im={mu_index}")
        fig.tight_layout()
        fig.savefig(frames_dir / f"dlra_fkinzv_{iframe:04d}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    time, z, vl, f_abs, selected_mu_index = load_dlra_fkinzv(args.input, args.mu_index, args.species)

    frame_indices = np.arange(0, len(time), args.every, dtype=int)
    if frame_indices.size == 0:
        raise ValueError("No frames selected. Check --every.")

    selected_times = np.asarray(time[frame_indices], dtype=float)
    selected_f_abs = np.asarray(f_abs[frame_indices], dtype=float)
    first_vmax = float(np.max(selected_f_abs[0])) if np.max(selected_f_abs[0]) > 0.0 else 1.0

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    mesh = ax.pcolormesh(z, vl, selected_f_abs[0].T, shading="auto", cmap="plasma", vmin=0.0, vmax=first_vmax)
    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set_xlabel("z")
    ax.set_ylabel("v_parallel")
    cbar = fig.colorbar(mesh, ax=ax, label="|f(z,v)|")
    cbar.ax.set_title(f"im={selected_mu_index}")

    def init():
        frame = selected_f_abs[0]
        mesh.set_array(frame.T.ravel())
        mesh.set_clim(0.0, float(np.max(frame)) if np.max(frame) > 0.0 else 1.0)
        time_text.set_text(f"t = {selected_times[0]:.8f}")
        return mesh, time_text

    def update(frame_number: int):
        frame = selected_f_abs[frame_number]
        mesh.set_array(frame.T.ravel())
        mesh.set_clim(0.0, float(np.max(frame)) if np.max(frame) > 0.0 else 1.0)
        time_text.set_text(f"t = {selected_times[frame_number]:.8f}")
        return mesh, time_text

    if args.frames_dir is not None:
        save_frames(selected_times, z, vl, selected_f_abs, selected_mu_index, Path(args.frames_dir))
        print(f"[plot_dlra_fkinzv] wrote frames: {args.frames_dir}")

    anim = None
    if (not args.no_show) or (args.save is not None):
        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=len(frame_indices),
            interval=args.interval,
            blit=False,
            repeat=False,
        )
    fig.tight_layout()

    if args.save is not None:
        save_path = Path(args.save)
        if anim is None:
            raise RuntimeError("Animation object was not created before save.")
        if save_path.suffix.lower() == ".gif":
            anim.save(save_path, writer=PillowWriter(fps=args.fps))
        else:
            anim.save(save_path, fps=args.fps)
        print(f"[plot_dlra_fkinzv] wrote animation: {save_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
