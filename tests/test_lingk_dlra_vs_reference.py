from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from linear_gyrokinetic import parts_to_complex

TEST_NZ = 8
TEST_NV = 4
TEST_NM = 3
TEST_DT = 0.0025
TEST_NT = 5
TEST_NSKIP = 2
TEST_RANK = 4
REL_L2_THRESHOLD = 0.40


@pytest.fixture(scope="session")
def generated_lingk_datasets(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    run_dir = tmp_path_factory.mktemp("generated_lingk_data")
    fkinzv_file = run_dir / "fkinzv.nc"
    mominz_file = run_dir / "mominzt.nc"
    frq_file = run_dir / "frq.txt"
    simulation_file = run_dir / "lingk_dlra.nc"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "reference_lingk_sim.py"),
            "--fkinzv-out",
            str(fkinzv_file),
            "--mominz-out",
            str(mominz_file),
            "--frq-out",
            str(frq_file),
            "--nz",
            str(TEST_NZ),
            "--nv",
            str(TEST_NV),
            "--nm",
            str(TEST_NM),
            "--dt",
            str(TEST_DT),
            "--dt-out",
            str(TEST_DT * TEST_NSKIP),
            "--time-limit",
            str(TEST_DT * (TEST_NT - 1)),
            "--disable-dtc",
            "--disable-progress",
            "--seed",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "dlra_lingk_sim.py"),
            "--out",
            str(simulation_file),
            "--rank",
            str(TEST_RANK),
            "--nz",
            str(TEST_NZ),
            "--nv",
            str(TEST_NV),
            "--nm",
            str(TEST_NM),
            "--dt",
            str(TEST_DT),
            "--nt",
            str(TEST_NT),
            "--nskip",
            str(TEST_NSKIP),
            "--seed",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    return fkinzv_file, mominz_file, simulation_file


def _relative_l2(reference: np.ndarray, test: np.ndarray) -> float:
    denom = np.linalg.norm(reference)
    if np.isclose(denom, 0.0):
        return float(np.linalg.norm(reference - test))
    return float(np.linalg.norm(reference - test) / denom)


def test_lingk_dlra_tracks_reference_distribution(generated_lingk_datasets: tuple[Path, Path, Path]) -> None:
    fkinzv_file, mominz_file, simulation_file = generated_lingk_datasets
    ds_fkinzv = xr.load_dataset(fkinzv_file)
    ds_mominz = xr.load_dataset(mominz_file)
    ds_lr = xr.load_dataset(simulation_file)

    f_ref = parts_to_complex(ds_fkinzv["f_real"].values, ds_fkinzv["f_imag"].values)
    f_lr_full = parts_to_complex(ds_lr["f_real"].values, ds_lr["f_imag"].values)
    mu_index = int(ds_fkinzv["mu_index"].values[0])
    f_lr = f_lr_full[:, :, :, mu_index, :][:, None, ...]

    phi_ref = parts_to_complex(ds_mominz["phi_real"].values, ds_mominz["phi_imag"].values)
    phi_lr = parts_to_complex(ds_lr["phi_real"].values, ds_lr["phi_imag"].values)

    assert _relative_l2(f_ref, f_lr) < REL_L2_THRESHOLD
    assert _relative_l2(phi_ref, phi_lr) < REL_L2_THRESHOLD


def test_lingk_dlra_basis_orthogonality(generated_lingk_datasets: tuple[Path, Path, Path]) -> None:
    _, _, simulation_file = generated_lingk_datasets
    ds_lr = xr.load_dataset(simulation_file)

    x_gram = parts_to_complex(ds_lr["X_gram_real"].values, ds_lr["X_gram_imag"].values)
    v_gram = parts_to_complex(ds_lr["V_gram_real"].values, ds_lr["V_gram_imag"].values)
    eye = np.eye(TEST_RANK)

    assert np.allclose(x_gram, eye[None, :, :], atol=1e-10)
    assert np.allclose(v_gram, eye[None, :, :], atol=1e-10)
