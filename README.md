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

You can switch to visualize the perturbed distribution function subtracted the equilibrium.

```bash
python plot_figure.py --reference reference_result.nc --sim simulation_result.nc --plot-mode fluctuation
```

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
