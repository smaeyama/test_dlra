from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr


INITIAL_FILE = Path("initial_state.nc")
SIM_FILE = Path("simulation_result.nc")
REL_L2_THRESHOLD = 0.1


def _read_first_available(ds: xr.Dataset, names: tuple[str, ...]) -> xr.DataArray:
    for name in names:
        if name in ds:
            return ds[name]
    raise KeyError(f"None of the variables {names!r} exist in dataset. Found: {list(ds.data_vars)}")


def _relative_l2(reference: np.ndarray, test: np.ndarray) -> float:
    denom = np.linalg.norm(reference)
    if np.isclose(denom, 0.0):
        return float(np.linalg.norm(reference - test))
    return float(np.linalg.norm(reference - test) / denom)


def _rms(reference: np.ndarray, test: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - test) ** 2)))


def _max_abs(reference: np.ndarray, test: np.ndarray) -> float:
    return float(np.max(np.abs(reference - test)))


def test_stage2_low_rank_reconstructs_initial_fields() -> None:
    if not INITIAL_FILE.exists() or not SIM_FILE.exists():
        pytest.skip("Requires existing 'initial_state.nc' and 'simulation_result.nc' in repo root.")

    ds_initial = xr.load_dataset(INITIAL_FILE)
    ds_sim = xr.load_dataset(SIM_FILE)

    f_ref = _read_first_available(ds_initial, ("f", "f_init")).values
    rho_ref = _read_first_available(ds_initial, ("rho", "rho_init")).values
    phi_ref = _read_first_available(ds_initial, ("phi", "phi_init")).values

    X = ds_sim["X"].isel(time=0).values
    S = ds_sim["S"].isel(time=0).values
    V = ds_sim["V"].isel(time=0).values
    rho_lr = ds_sim["rho"].isel(time=0).values
    phi_lr = ds_sim["phi"].isel(time=0).values

    f_lr = X @ S @ V.T

    metrics = {
        "f": {
            "relative_l2": _relative_l2(f_ref, f_lr),
            "rms": _rms(f_ref, f_lr),
            "max_abs": _max_abs(f_ref, f_lr),
        },
        "rho": {
            "relative_l2": _relative_l2(rho_ref, rho_lr),
            "rms": _rms(rho_ref, rho_lr),
            "max_abs": _max_abs(rho_ref, rho_lr),
        },
        "phi": {
            "relative_l2": _relative_l2(phi_ref, phi_lr),
            "rms": _rms(phi_ref, phi_lr),
            "max_abs": _max_abs(phi_ref, phi_lr),
        },
    }

    for name in ("f", "rho", "phi"):
        rel_l2 = metrics[name]["relative_l2"]
        assert rel_l2 <= REL_L2_THRESHOLD, (
            f"{name} relative L2 error {rel_l2:.6e} exceeds threshold {REL_L2_THRESHOLD}. "
            f"RMS={metrics[name]['rms']:.6e}, max_abs={metrics[name]['max_abs']:.6e}"
        )
