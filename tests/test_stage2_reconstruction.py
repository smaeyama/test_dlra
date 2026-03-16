from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


REFERENCE_FILE = Path("reference_result.nc")
SIM_FILE = Path("simulation_result.nc")
REL_L2_THRESHOLD = 0.1
TIME_POINTS = (0.0, 5.0)


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


def _time_slice(arr: xr.DataArray, target_time: float) -> tuple[xr.DataArray, float | None]:
    if "time" not in arr.dims:
        return arr, None

    time_values = arr["time"].values.astype(float)
    idx = int(np.argmin(np.abs(time_values - target_time)))
    return arr.isel(time=idx), float(time_values[idx])


def _shared_time_value(ds_reference: xr.Dataset, ds_sim: xr.Dataset, target_time: float) -> tuple[float, float, float]:
    ref_times = _read_first_available(ds_reference, ("f", "f_ref"))["time"].values.astype(float)
    sim_times = ds_sim["X"]["time"].values.astype(float)

    ref_idx = int(np.argmin(np.abs(ref_times - target_time)))
    shared_time = float(ref_times[ref_idx])
    sim_idx = int(np.argmin(np.abs(sim_times - shared_time)))
    sim_time = float(sim_times[sim_idx])

    return shared_time, sim_time, abs(shared_time - sim_time)


def _compare_reference_and_simulation_at_time(
    ds_reference: xr.Dataset, ds_sim: xr.Dataset, target_time: float
) -> dict[str, dict[str, float]]:
    shared_time, sim_time, time_gap = _shared_time_value(ds_reference, ds_sim, target_time)

    f_ref_da, actual_time = _time_slice(_read_first_available(ds_reference, ("f", "f_ref")), shared_time)
    rho_ref_da, _ = _time_slice(_read_first_available(ds_reference, ("rho", "rho_ref")), shared_time)
    phi_ref_da, _ = _time_slice(_read_first_available(ds_reference, ("phi", "phi_ref")), shared_time)

    X_da, _ = _time_slice(ds_sim["X"], shared_time)
    S_da, _ = _time_slice(ds_sim["S"], shared_time)
    V_da, _ = _time_slice(ds_sim["V"], shared_time)
    rho_lr_da, _ = _time_slice(ds_sim["rho"], shared_time)
    phi_lr_da, _ = _time_slice(ds_sim["phi"], shared_time)

    f_ref = f_ref_da.values
    rho_ref = rho_ref_da.values
    phi_ref = phi_ref_da.values

    X = X_da.values
    S = S_da.values
    V = V_da.values
    rho_lr = rho_lr_da.values
    phi_lr = phi_lr_da.values

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

    time_label = f"target_time={target_time:g}"
    if actual_time is not None:
        time_label += (
            f" (selected_ref_time={actual_time:g}, selected_sim_time={sim_time:g}, "
            f"|Δt|={time_gap:.3e})"
        )

    for name, values in metrics.items():
        assert values["relative_l2"] <= REL_L2_THRESHOLD, (
            f"{name} relative L2 error {values['relative_l2']:.6e} exceeds threshold {REL_L2_THRESHOLD} "
            f"at {time_label}. RMS={values['rms']:.6e}, max_abs={values['max_abs']:.6e}"
        )

    return metrics


def test_stage2_low_rank_reconstructs_reference_fields_multiple_times() -> None:
    assert REFERENCE_FILE.exists(), f"Missing required dataset: {REFERENCE_FILE}"
    assert SIM_FILE.exists(), f"Missing required dataset: {SIM_FILE}"

    ds_reference = xr.load_dataset(REFERENCE_FILE)
    ds_sim = xr.load_dataset(SIM_FILE)

    for target_time in TIME_POINTS:
        metrics = _compare_reference_and_simulation_at_time(ds_reference, ds_sim, target_time)
        for name, values in metrics.items():
            print(
                f"passed: time={target_time:g}, {name} rel_l2={values['relative_l2']:.6e}, "
                f"rms={values['rms']:.6e}, max_abs={values['max_abs']:.6e}"
            )


def test_dlra_orthogonality() -> None:
    assert REFERENCE_FILE.exists(), f"Missing required dataset: {REFERENCE_FILE}"
    assert SIM_FILE.exists(), f"Missing required dataset: {SIM_FILE}"

    ds_reference = xr.load_dataset(REFERENCE_FILE)
    ds_sim = xr.load_dataset(SIM_FILE)

    x_coords = ds_sim["x"].values
    v_coords = ds_sim["v"].values
    dx = float(x_coords[1] - x_coords[0])
    dv = float(v_coords[1] - v_coords[0])

    for target_time in TIME_POINTS:
        shared_time, sim_time, time_gap = _shared_time_value(ds_reference, ds_sim, target_time)
        X = _time_slice(ds_sim["X"], shared_time)[0].values
        V = _time_slice(ds_sim["V"], shared_time)[0].values

        x_gram = X.T @ X * dx
        v_gram = V.T @ V * dv

        x_max_dev = float(np.max(np.abs(x_gram - np.eye(X.shape[1]))))
        v_max_dev = float(np.max(np.abs(v_gram - np.eye(V.shape[1]))))

        assert np.allclose(x_gram, np.eye(X.shape[1]), atol=1e-10), (
            f"X basis is not orthonormal under dx inner product at target_time={target_time:g} "
            f"(selected_sim_time={sim_time:g}, |Δt|={time_gap:.3e}). max_dev={x_max_dev:.6e}"
        )
        assert np.allclose(v_gram, np.eye(V.shape[1]), atol=1e-10), (
            f"V basis is not orthonormal under dv inner product at target_time={target_time:g} "
            f"(selected_sim_time={sim_time:g}, |Δt|={time_gap:.3e}). max_dev={v_max_dev:.6e}"
        )

        print(
            f"passed: time={target_time:g}, X orthogonality max_dev={x_max_dev:.6e}, "
            f"V orthogonality max_dev={v_max_dev:.6e}"
        )
