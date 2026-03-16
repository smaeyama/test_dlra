# test_dlra

Dynamic Low-Rank Approximation (DLRA) test project for 1D electrostatic Vlasov–Poisson simulations.

## Files

- `test_dlra_implementation_v03.py` — split workflow script
  - (i) initial value generation
  - (ii) DLRA simulation
  - (iii) visualization
- `test_dlra_implementation_v03.ipynb` — notebook version.
- `requirements.txt` — Python dependencies.

## Requirements

- Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage (3-stage workflow with NetCDF handoff)

### (i) 初期値作成

```bash
python test_dlra_implementation_v03.py create-initial --out initial_state.nc --flag-init two-stream
```

### (ii) 動的低ランク近似シミュレーション

```bash
python test_dlra_implementation_v03.py simulate --initial initial_state.nc --out simulation_result.nc --rank 64 --dt 0.025 --nt 1000 --nskip 20
```

### (iii) 結果可視化

```bash
python test_dlra_implementation_v03.py visualize --sim simulation_result.nc
```

各ステージ間のデータ受け渡しは `xarray` を使った NetCDF (`.nc`) です。

Or use Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smaeyama/test_dlra/blob/main/test_dlra_implementation_v03.ipynb)
