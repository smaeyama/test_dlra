# test_dlra

Dynamic Low-Rank Approximation (DLRA) test project for 1D electrostatic Vlasov–Poisson simulations.

## Files

- `stage1_create_initial.py` — (i) initial value generation and NetCDF output
- `stage2_simulate_dlra.py` — (ii) DLRA simulation from initial NetCDF
- `stage3_visualize_results.py` — (iii) visualization from simulation NetCDF
- `low_rank_approx.py` — low-rank factor class (`X`, `S`, `V`) shared by stage2/stage3
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

## Usage (3 Python files + NetCDF handoff)

### (i) 初期値作成

```bash
python stage1_create_initial.py --out initial_state.nc --reference-out reference_result.nc --flag-init two-stream
```


`stage1_create_initial.py` は初期値 `initial_state.nc` に加えて、フル分布関数で時間発展した参照データ `reference_result.nc` も同時に保存します。

### (ii) 動的低ランク近似シミュレーション

```bash
python stage2_simulate_dlra.py --initial initial_state.nc --out simulation_result.nc --rank 64 --dt 0.025 --nt 1000 --nskip 20
```

### (iii) 結果可視化

```bash
python stage3_visualize_results.py --reference reference_result.nc --sim simulation_result.nc
```

各ステージ間のデータ受け渡しは `xarray` を使った NetCDF (`.nc`) です。

Stage (ii) は `f(x,v)` 全体ではなく低ランク因子 `X(time,x,rank)`, `S(time,rank,rank)`, `V(time,v,rank)` を保存します。
Stage (iii) では可視化に必要な時刻だけ `LowRankApprox.to_full()` で再構築します。

Or use Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smaeyama/test_dlra/blob/main/test_dlra_implementation_v03.ipynb)
