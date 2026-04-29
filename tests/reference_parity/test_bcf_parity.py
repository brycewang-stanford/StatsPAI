"""Reference parity: ``sp.bcf`` posterior ATE vs R ``bcf`` package.

Hahn-Murray-Carvalho (2020) Bayesian Causal Forests.  R ``bcf`` is
the canonical implementation.  ``sp.bcf`` is the Python port.

Caveat: BCF is Bayesian + MCMC.  Posterior means agree well across
implementations *given enough samples* but draw-level noise means
the MCMC seed shouldn't be expected to give byte-identical answers
(different RNGs).  We assert agreement on the posterior mean ATE
within a sensible band.

Tolerance: 15% relative on the posterior mean ATE — wider than
DML/HDFE because (a) MCMC seed paths differ, (b) prior
parameterizations may differ slightly between packages.

References
----------
- Hahn, P.R., Murray, J.S. and Carvalho, C.M. (2020). "Bayesian
  Regression Tree Models for Causal Inference."
  *Bayesian Analysis*, 15(3), 965-1056. [@hahn2020bayesian]
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
def bcf_data():
    return pd.read_csv(_FIXTURE_DIR / "bcf_data.csv")


@pytest.fixture(scope="module")
def r_reference():
    with open(_FIXTURE_DIR / "bcf_R.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def py_result(bcf_data):
    """Fit once.  BCF MCMC is slow, so we keep nburn/nsim modest."""
    np.random.seed(42)
    return sp.bcf(
        data=bcf_data,
        y="y", treat="W",
        covariates=[f"X{i}" for i in range(1, 6)],
        n_trees_mu=200, n_trees_tau=50,
        # NB: n_bootstrap=0 crashes the current build (empty-array
        # indexing in the SE-quantile path).  Use a small number.
        n_bootstrap=20,
        random_state=42,
    )


def test_bcf_ate_matches_R(py_result, r_reference):
    """Posterior mean ATE within 15% of R bcf reference."""
    py_ate = float(py_result.estimate)
    r_ate = r_reference["ate"]["mean"]
    rel = abs(py_ate - r_ate) / abs(r_ate)
    assert rel < 0.15, (
        f"sp.bcf ATE drifted from R bcf by {rel:.1%} "
        f"(Python={py_ate:.4f}, R={r_ate:.4f}).  "
        f"Tolerance: 15% — MCMC seed paths differ across stacks, "
        f"so byte-identical answers aren't expected.  >15% suggests "
        f"a real bug in either prior parameterization or update path."
    )


def test_bcf_ate_close_to_truth(py_result):
    """True ATE = 1.0; both implementations should land near it."""
    py_ate = float(py_result.estimate)
    assert abs(py_ate - 1.0) < 0.3, (
        f"sp.bcf ATE={py_ate:.4f} far from true ATE=1.0"
    )


def test_bcf_ate_sign_correct(py_result, r_reference):
    """Both should agree on sign — ATE > 0."""
    py_ate = float(py_result.estimate)
    r_ate = r_reference["ate"]["mean"]
    assert (py_ate > 0) == (r_ate > 0)


def test_bcf_fixture_meta(r_reference):
    assert "meta" in r_reference
    assert r_reference["meta"]["seed"] == 42


def test_bcf_fixture_data(bcf_data):
    assert len(bcf_data) == 400
