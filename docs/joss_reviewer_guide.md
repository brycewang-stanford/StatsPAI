# JOSS Reviewer Guide

This page gives JOSS reviewers a short path for installing StatsPAI and
checking representative functionality without running the full test suite.

## Scope

StatsPAI is a Python package for causal inference and applied econometrics. The
core package is pip-installable, ships small teaching datasets, and can be
tested offline. Optional extras cover plotting, pyfixest integration, neural
causal estimators, JAX-backed workloads, Bayesian workflows, Optuna tuning,
and exact Calonico-Cattaneo-Titiunik RD parity via `rdrobust`.

## Install

```bash
python -m venv .venv-joss
source .venv-joss/bin/activate
python -m pip install --upgrade pip
python -m pip install statspai
```

For review from a repository checkout:

```bash
python -m pip install -e ".[dev,plotting]"
```

The core install declares the runtime dependencies in `pyproject.toml`:
NumPy, SciPy, Pandas, statsmodels, scikit-learn, linearmodels, formulaic,
numba, patsy, openpyxl, xlsxwriter, python-docx, and tabulate. Optional extras
are named `plotting`, `fixest`, `deepiv`, `neural`, `performance`, `bayes`,
`tune`, `rd-cct`, `text`, and `docs`.

## Smoke Test

```bash
python - <<'PY'
import statspai as sp

print("version:", sp.__version__)
print("registered functions:", len(sp.list_functions()))

card = sp.datasets.card_1995()
iv = sp.ivreg("lwage ~ (educ ~ nearc4) + exper + expersq + black + south + smsa",
              data=card)
print(iv.summary())
print(sp.citation("plain"))
PY
```

The smoke test should import the package, load a bundled dataset, fit a
published-reference IV example, print a result summary, and return estimator
software citation metadata.

## Scripted Examples

The repository includes standalone scripts under `examples/` for four canonical
designs:

```bash
python examples/card_iv.py
python examples/did_mpdta.py
python examples/rd_lee.py
python examples/synth_prop99.py
```

Each script uses bundled data and is intended to run offline.

## Representative Offline Examples

These examples use bundled datasets under `sp.datasets` and do not require
network access.

```python
import statspai as sp

# Difference-in-differences: Callaway-Sant'Anna style staggered adoption.
mpdta = sp.datasets.mpdta()
did = sp.callaway_santanna(data=mpdta, y="lemp", t="year",
                           i="countyreal", g="first_treat")
print(sp.aggte(did, type="simple").summary())

# Regression discontinuity: Lee-style close-election design.
lee = sp.datasets.lee_2008_senate()
rd = sp.rdrobust(data=lee, y="voteshare_next", x="margin", c=0)
print(rd.summary())

# Synthetic control: California Proposition 99 teaching dataset.
prop99 = sp.datasets.california_prop99()
scm = sp.synth(prop99, outcome="cigsale", unit="state", time="year",
               treated_unit="California", treatment_time=1989)
print(scm.summary())

# Registry and schema surface for agent-native workflows.
print(sp.describe_function("rdrobust"))
print(sp.function_schema("rdrobust")["name"])
```

## Targeted Tests

For a quick local check from a repository checkout:

```bash
python -m pytest \
  tests/test_ols.py \
  tests/test_did.py \
  tests/test_registry.py \
  tests/iv/test_iv_diag.py \
  tests/spatial/test_models_base.py \
  -q --no-cov
```

The repository also includes larger parity and validation suites:

```bash
python -m pytest tests/r_parity tests/stata_parity tests/reference_parity -q --no-cov
python scripts/registry_stats.py --check
python scripts/schema_quality.py
python tools/audit_bib_coverage.py --strict-dangling --hide-orphans
python tools/audit_bib_duplicates.py --strict
```

The full default suite is larger and can take tens of minutes on a laptop.
Several optional-dependency tests skip unless the relevant extras are installed.

## Build Check

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

The source distribution and wheel include the MIT license, citation metadata,
bundled datasets, and the package source.
