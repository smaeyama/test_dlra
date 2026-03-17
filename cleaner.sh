#/bin/sh

find . -name .ipynb_checkpoints | xargs rm -rf
find . -name __pycache__ | xargs rm -rf
#rm -rf build/
#rm -rf src/bzx.egg-info/
rm -rf examples/output/
#rm -f tests/log.dat tests/temp.nc
rm -rf .pytest_cache/
rm -rf .venv/
rm -rf .codex/

#jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace examples/*.ipynb
#jupyter nbconvert --to python examples/*.ipynb
