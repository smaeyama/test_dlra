# test_dlra

Dynamic Low-Rank Approximation (DLRA) test project for 1D electrostatic Vlasov-Poisson simulations.

## Files

- `reference_Vlasov_sim.py` - creates the initial condition and a reference Vlasov simulation in NetCDF format
- `dlra_Vlasov_sim.py` - runs the DLRA simulation from the initial NetCDF file
- `plot_figure.py` - visualizes the reference and DLRA results side by side
- `low_rank_approx.py` - low-rank factor class (`X`, `S`, `V`) shared by the DLRA and plotting scripts
- `tests/test_dlra_vs_reference.py` - runs short reference/DLRA simulations and validates reconstruction accuracy
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
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

Each stage exchanges data through NetCDF (`.nc`) files that can be read with `xarray`.
The DLRA simulation stores the low-rank factors `X(time,x,rank)`, `S(time,rank,rank)`, and `V(time,v,rank)` instead of the full `f(x,v)` field.
`plot_figure.py` reconstructs the full field for visualization using `LowRankApprox.to_full()` when needed.

## Tests

```bash
pytest
```

The pytest suite first runs short reference and DLRA simulations with reduced grid/time settings, then compares the reconstructed DLRA fields against the reference solution.
