from __future__ import annotations

from pathlib import Path

import numpy as np
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


def _first_time_slice(arr: xr.DataArray) -> xr.DataArray:
    return arr.isel(time=0) if "time" in arr.dims else arr


def test_stage2_low_rank_reconstructs_initial_fields() -> None:
    assert INITIAL_FILE.exists(), f"Missing required dataset: {INITIAL_FILE}"
    assert SIM_FILE.exists(), f"Missing required dataset: {SIM_FILE}"

    ds_initial = xr.load_dataset(INITIAL_FILE)
    ds_sim = xr.load_dataset(SIM_FILE)

    f_ref = _read_first_available(ds_initial, ("f", "f_init")).values
    rho_ref = _read_first_available(ds_initial, ("rho", "rho_init")).values
    phi_ref = _read_first_available(ds_initial, ("phi", "phi_init")).values

    X = _first_time_slice(ds_sim["X"]).values
    S = _first_time_slice(ds_sim["S"]).values
    V = _first_time_slice(ds_sim["V"]).values
    rho_lr = _first_time_slice(ds_sim["rho"]).values
    phi_lr = _first_time_slice(ds_sim["phi"]).values

    # Supports both legacy S(rank, rank) and renamed S(rankx, rankv) conventions.
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

    for name, values in metrics.items():
        assert values["relative_l2"] <= REL_L2_THRESHOLD, (
            f"{name} relative L2 error {values['relative_l2']:.6e} exceeds threshold {REL_L2_THRESHOLD}. "
            f"RMS={values['rms']:.6e}, max_abs={values['max_abs']:.6e}"
        )
        print(
            f"passed: {name} rel_l2={values['relative_l2']:.6e}, "
            f"rms={values['rms']:.6e}, max_abs={values['max_abs']:.6e}"
        )


def test_dlra_orthogonality() -> None:
    assert SIM_FILE.exists(), f"Missing required dataset: {SIM_FILE}"

    ds_sim = xr.load_dataset(SIM_FILE)

    X = _first_time_slice(ds_sim["X"]).values
    V = _first_time_slice(ds_sim["V"]).values

    x_coords = ds_sim["x"].values
    v_coords = ds_sim["v"].values
    dx = float(x_coords[1] - x_coords[0])
    dv = float(v_coords[1] - v_coords[0])

    x_gram = X.T @ X * dx
    v_gram = V.T @ V * dv

    x_max_dev = float(np.max(np.abs(x_gram - np.eye(X.shape[1]))))
    v_max_dev = float(np.max(np.abs(v_gram - np.eye(V.shape[1]))))

    assert np.allclose(x_gram, np.eye(X.shape[1]), atol=1e-10), (
        f"X basis is not orthonormal under dx inner product. max_dev={x_max_dev:.6e}"
    )
    assert np.allclose(v_gram, np.eye(V.shape[1]), atol=1e-10), (
        f"V basis is not orthonormal under dv inner product. max_dev={v_max_dev:.6e}"
    )

    print(f"passed: X orthogonality max_dev={x_max_dev:.6e} (weighted by dx)")
    print(f"passed: V orthogonality max_dev={v_max_dev:.6e} (weighted by dv)")
