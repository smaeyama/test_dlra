# test_dlra

Dynamic Low-Rank Approximation (DLRA) test project for 1D electrostatic Vlasov–Poisson simulations.

## Files

- `test_dlra_implementation_v03.py` — standalone Python implementation.
- `test_dlra_implementation_v03.ipynb` — notebook version of the same workflow.
- `requirements.txt` — Python dependencies for running the script/notebook.

## Requirements

- Python 3.10+

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run the script:

```bash
python test_dlra_implementation_v03.py
```

Run the notebook locally:

```bash
jupyter notebook test_dlra_implementation_v03.ipynb
```

Or use Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/smaeyama/test_dlra/blob/main/test_dlra_implementation_v03.ipynb)
