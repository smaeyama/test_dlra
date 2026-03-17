# test_dlra

Dynamic Low-Rank Approximation (DLRA) test project for 1D electrostatic Vlasov-Poisson simulations.

## Files

- `reference_Vlasov_sim.py` - creates the initial condition and a reference Vlasov simulation in NetCDF format
- `dlra_Vlasov_sim.py` - runs the DLRA simulation from the initial NetCDF file
- `linear_gyrokinetic.py` - shared geometry, operators, field solves, and DLRA utilities for the linear gyrokinetic example
- `linear_gyrokinetic_numpy.py` - backup of the previous NumPy-focused linear gyrokinetic implementation for side-by-side comparison
- `reference_lingk_sim.py` - full-grid reference solver mirroring the `test_lingk` linear gyrokinetic setup and writing `fkinzv`/`mominzt`/`frq`-style outputs
- `dlra_lingk_sim.py` - DLRA solver for the linear gyrokinetic problem using `h(z,vm) = X(z) S V(vm)^T`
- `plot_mominz.py` - animates the time evolution of `|phi(z)|` from `mominzt.nc`
- `plot_fkinzv.py` - animates the time evolution of `|f(z,v)|` from `fkinzv.nc`
- `check_fortran2python.py` - compares Python and Fortran `frq`, `mominzt`, and `fkinzv` outputs
- `plot_figure.py` - visualizes the reference and DLRA results side by side
- `low_rank_approx.py` - low-rank factor class (`X`, `S`, `V`) shared by the DLRA and plotting scripts
- `tests/test_dlra_vs_reference.py` - runs short reference/DLRA simulations and validates reconstruction accuracy
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10+
- `jax[cpu]` is recommended for the linear gyrokinetic solver; `linear_gyrokinetic.py` falls back to NumPy if JAX is unavailable

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. Create the initial condition and reference simulation

```bash
python reference_Vlasov_sim.py --out initial_state.nc --reference-out reference_result.nc --flag-init two-stream
```

`reference_Vlasov_sim.py` writes the initial-condition file `initial_state.nc` and a reference dataset `reference_result.nc`.
You can reduce runtime for quick experiments or tests with options such as `--nx`, `--nv`, `--dt`, and `--nt`.

### 2. Run the DLRA simulation

```bash
python dlra_Vlasov_sim.py --initial initial_state.nc --out simulation_result.nc --rank 64 --dt 0.025 --nt 1000 --nskip 20
```

### 3. Plot the results

```bash
python plot_figure.py --reference reference_result.nc --sim simulation_result.nc
```

You can switch to visualize the perturbed distribution function subtracted the equilibrium.

```bash
python plot_figure.py --reference reference_result.nc --sim simulation_result.nc --plot-mode fluctuation
```

## Linear Gyrokinetic Example

The repository also includes a Python port of the `test_lingk/src` linear gyrokinetic solver.
The DLRA version uses a matrix factorization over parallel position and a flattened velocity-magnetic coordinate:

`h(t, z, vm) = X(t, z) S(t) V(t, vm)^T`, where `vm = (v_parallel, mu, species)`.

Small reference and DLRA runs can be generated with:

```bash
python reference_lingk_sim.py --output-dir lingk_output --nz 120 --nv 32 --nm 31 --dt 0.01 --dt-out 0.1 --time-limit 10.0
python dlra_lingk_sim.py --out lingk_dlra.nc --rank 8 --nz 24 --nv 8 --nm 7 --dt 0.01 --nt 41 --nskip 10
```

The current `linear_gyrokinetic.py` is JAX-first and uses `jax.jit` when available.
For reference and code comparison, the previous NumPy-based implementation is kept as `linear_gyrokinetic_numpy.py`.

`reference_lingk_sim.py` writes:

- `fkinzv.nc` - snapshots of `f(z, v_parallel)` at selected `mu` indices
- `mominzt.nc` - `phi(z)`, `A_parallel(z)`, and density moments
- `frq.txt` - growth-rate/frequency estimates in the style of the original Fortran output

`reference_lingk_sim.py` also prints a Fortran-style time-step control summary, progress updates, and a coarse elapsed-time breakdown (`Init`, `RKG`, `Sample`, `Output`, `Other`).

### Comparing Against Fortran

If you have matching Fortran outputs under `data_fortran/`, you can compare them against the Python outputs with:

```bash
python3 check_fortran2python.py
```

This compares:

- `lingk_output/frq.txt` vs `data_fortran/frq.001`
- `lingk_output/mominzt.nc` vs `data_fortran/mominzt.001`
- `lingk_output/fkinzv.nc` vs `data_fortran/fkinzv_imXXXX_tXXXXXXXX.dat`

and prints `max`, `rms`, and `relative_rms` errors for each compared column.

### Animations

`plot_mominz.py` animates the time evolution of `|phi(z)|`:

```bash
python3 plot_mominz.py --input lingk_output/mominzt.nc
```

`plot_fkinzv.py` animates the time evolution of `|f(z,v)|`:

```bash
python3 plot_fkinzv.py --input lingk_output/fkinzv.nc
```

Both scripts support:

- interactive display
- `--save out.gif` or `--save out.mp4`
- `--frames-dir out_frames --no-show` for numbered PNG output

The displayed time is always read from the NetCDF `time` coordinate.

Each stage exchanges data through NetCDF (`.nc`) files that can be read with `xarray`.
The DLRA simulation stores the low-rank factors `X(time,x,rank)`, `S(time,rank,rank)`, and `V(time,v,rank)` instead of the full `f(x,v)` field.
`plot_figure.py` reconstructs the full field for visualization using `LowRankApprox.to_full()` when needed.

## Example

`examples/linear_landau_damping.py` runs a full linear Landau damping workflow:

- create the linear-Landau initial condition
- compute a grid-based reference solution
- compute the DLRA solution
- save NetCDF outputs and a summary PNG figure

```bash
python3 examples/linear_landau_damping.py
```

By default the script writes its outputs under `examples/output/linear_landau/`.

Additional wrapper examples are also available:

```bash
python3 examples/bump_on_tail.py
python3 examples/two_stream.py
```

## Tests

```bash
pytest
```

The pytest suite first runs short reference and DLRA simulations with reduced grid/time settings, then compares the reconstructed DLRA fields against the reference solution.
