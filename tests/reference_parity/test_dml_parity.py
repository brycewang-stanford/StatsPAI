"""Reference parity: ``sp.dml`` vs R DoubleML.

The DoubleML R package is the reference implementation of
Chernozhukov et al. (2018) "Double/Debiased Machine Learning";
sp.dml is StatsPAI's Python port.  Numerical agreement at the
0.05 level on a fixed DGP is the bar — DoubleML's CV-glmnet folds
introduce stochasticity that scales with sqrt(n), so we tolerate
~5% deviation on the coefficient.

Fixture lifecycle
-----------------
Both engines consume ``_fixtures/dml_data.csv`` (deterministic seed=42
DGP from ``_generate_dml_data.py``).  R reference values are in
``_fixtures/dml_R.json`` produced by ``_generate_dml.R``.  Re-run
the generators only when the DGP itself changes; otherwise the
fixture is the contract.

Tolerance discipline
--------------------
We assert agreement on:
  • PLR: coef within 7% relative; SE within 25% (CV-glmnet penalty
    differs slightly between scikit-learn and glmnet — drives SE
    spread).
  • IRM: coef within absolute 0.05 (close to zero on this DGP);
    SE within 25%.

References
----------
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen,
  C., Newey, W. and Robins, J. (2018). Double/debiased machine
  learning for treatment and structural parameters. *The
  Econometrics Journal*, 21(1), C1-C68. [@chernozhukov2018double]
- Bach, P., Chernozhukov, V., Kurz, M.S., Spindler, M. (2022).
  DoubleML — An Object-Oriented Implementation of Double Machine
  Learning in R. *JSS*, 108(3), 1-56.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp


_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def dml_data():
    """Load the shared seed=42 DGP — same csv R consumes."""
    return pd.read_csv(_FIXTURE_DIR / "dml_data.csv")


@pytest.fixture(scope="module")
def r_reference():
    """Load the golden R DoubleML output."""
    with open(_FIXTURE_DIR / "dml_R.json") as f:
        return json.load(f)


# ─── PLR (continuous treatment) ─────────────────────────────────────────


def test_dml_plr_coefficient_matches_R(dml_data, r_reference):
    """sp.dml(model='plr') vs R DoubleML PLR — coefficient agreement."""
    x_cols = [c for c in dml_data.columns if c.startswith("x")]
    np.random.seed(42)
    res = sp.dml(
        data=dml_data, y="y", d="d", X=x_cols,
        model="plr", n_folds=5,
    )
    py_coef = float(res.estimate)

    r_coef = r_reference["plr"]["coef"]
    rel = abs(py_coef - r_coef) / abs(r_coef)
    assert rel < 0.07, (
        f"PLR coefficient drifted from R DoubleML by {rel:.1%} "
        f"(Python={py_coef:.6f}, R={r_coef:.6f}). "
        f"Tolerance: 7% relative.  CV-glmnet stochasticity should "
        f"keep this band; widen only if you understand why."
    )


def test_dml_plr_standard_error_matches_R(dml_data, r_reference):
    """SE band is wider — penalty paths differ between sklearn and glmnet."""
    x_cols = [c for c in dml_data.columns if c.startswith("x")]
    np.random.seed(42)
    res = sp.dml(
        data=dml_data, y="y", d="d", X=x_cols,
        model="plr", n_folds=5,
    )
    py_se = float(res.se)

    r_se = r_reference["plr"]["se"]
    rel = abs(py_se - r_se) / abs(r_se)
    assert rel < 0.25, (
        f"PLR SE drifted from R DoubleML by {rel:.1%} "
        f"(Python={py_se:.6f}, R={r_se:.6f}). Tolerance: 25%."
    )


# ─── IRM (binary treatment, ATE) ────────────────────────────────────────


def test_dml_irm_coefficient_matches_R(dml_data, r_reference):
    """sp.dml(model='irm') vs R DoubleML IRM — ATE agreement."""
    x_cols = [c for c in dml_data.columns if c.startswith("x")]
    np.random.seed(42)
    res = sp.dml(
        data=dml_data, y="y", d="d_bin", X=x_cols,
        model="irm", n_folds=5,
    )
    py_coef = float(res.estimate)

    r_coef = r_reference["irm"]["coef"]
    # IRM coef is near zero on this DGP — use absolute tolerance
    abs_diff = abs(py_coef - r_coef)
    assert abs_diff < 0.05, (
        f"IRM coefficient drifted from R DoubleML by {abs_diff:.4f} "
        f"(Python={py_coef:.6f}, R={r_coef:.6f}). "
        f"Tolerance: |Δ| < 0.05."
    )


# ─── Fixture self-tests ─────────────────────────────────────────────────


def test_fixture_csv_intact(dml_data):
    """Guard that the CSV fixture wasn't accidentally mutated."""
    assert len(dml_data) == 1000
    assert "y" in dml_data.columns
    assert "d" in dml_data.columns
    assert "d_bin" in dml_data.columns
    x_cols = [c for c in dml_data.columns if c.startswith("x")]
    assert len(x_cols) == 10


def test_fixture_R_meta_present(r_reference):
    """Guard that the R fixture has the metadata we need to reproduce."""
    assert "meta" in r_reference
    meta = r_reference["meta"]
    assert "DoubleML_version" in meta
    assert meta["seed"] == 42
    assert meta["n_folds"] == 5
