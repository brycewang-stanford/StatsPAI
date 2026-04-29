"""Reference parity: ``sp.causal_forest`` ATE vs R ``grf::causal_forest``.

Caveat — these are not byte-identical algorithms.  R ``grf`` is the
canonical Athey-Tibshirani-Wager 2019 implementation with honest
splitting + R-learner orthogonalisation.  ``sp.causal_forest`` is
built on EconML's ``CausalForestDML`` which uses the same family
of ideas (honest splitting + DML residualisation) but a different
software stack (scikit-learn vs the bespoke C++ in grf).

ATE drift between the two on a fixed seed is dominated by:
  • bootstrap sampling differences (different RNG paths)
  • internal nuisance learners (grf uses regression forests for both;
    EconML defaults to LassoCV for binary W)
  • honest split / leaf size choices

Empirically: on this DGP (true mean τ ≈ 0.91) we see grf ~ 0.98 and
EconML CF ~ 1.14.  Tolerance is therefore wide (25% relative),
intended to **catch order-of-magnitude bugs**, not to certify
implementation equivalence.

References
----------
- Athey, S., Tibshirani, J. and Wager, S. (2019). Generalized random
  forests. *Annals of Statistics*, 47(2), 1148-1178.
  [@athey2019generalized]
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

import statspai as sp


_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def grf_data():
    return pd.read_csv(_FIXTURE_DIR / "grf_data.csv")


@pytest.fixture(scope="module")
def r_reference():
    with open(_FIXTURE_DIR / "grf_R.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fitted_cf(grf_data):
    """Fit once per module — causal forests are slow."""
    return sp.causal_forest(
        "y ~ W | X1 + X2 + X3 + X4 + X5",
        data=grf_data,
        n_estimators=2000,
        random_state=42,
        discrete_treatment=True,
    )


def test_grf_ate_within_tolerance(fitted_cf, r_reference):
    """ATE point estimate should be within 25% of R grf reference."""
    py_ate = float(fitted_cf.ate())
    r_ate = r_reference["ate"]["estimate"]
    rel = abs(py_ate - r_ate) / abs(r_ate)
    assert rel < 0.25, (
        f"sp.causal_forest ATE drifted from R grf by {rel:.1%} "
        f"(Python={py_ate:.4f}, R={r_ate:.4f}). "
        f"Tolerance: 25% — intended to catch order-of-magnitude bugs, "
        f"not certify byte-identical agreement.  See module docstring "
        f"for why exact agreement is not expected."
    )


def test_grf_ate_sign_agreement(fitted_cf, r_reference):
    """Both engines should agree on the sign — true τ ≈ 0.91 > 0."""
    py_ate = float(fitted_cf.ate())
    r_ate = r_reference["ate"]["estimate"]
    assert (py_ate > 0) == (r_ate > 0), (
        f"Sign disagreement is a serious red flag: "
        f"Python ATE={py_ate:.4f}, R ATE={r_ate:.4f}"
    )


def test_grf_ate_close_to_truth(fitted_cf):
    """Both engines should land close to the true mean τ ≈ 0.91."""
    py_ate = float(fitted_cf.ate())
    # Truth is mean(1 + 2*X1) ≈ 0.91 (sample mean of standard normal)
    # Tolerance: ±0.5 absolute (roughly 1 SE on the ATE in this DGP)
    assert abs(py_ate - 0.91) < 0.5, (
        f"sp.causal_forest ATE={py_ate:.4f} far from true τ̄=0.91"
    )


def test_grf_fixture_meta(r_reference):
    assert "meta" in r_reference
    assert r_reference["meta"]["seed"] == 42
    assert r_reference["meta"]["num_trees"] == 2000


def test_grf_fixture_n(grf_data):
    assert len(grf_data) == 1000
