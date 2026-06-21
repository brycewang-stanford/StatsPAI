"""Reference parity: ``sp.causal_forest`` ATE vs R ``grf::causal_forest``.

Both engines report the **AIPW doubly-robust** average treatment effect
(``grf::average_treatment_effect`` on the R side;
``sp.causal_forest.average_treatment_effect`` on the Python side), so
the comparison is like-for-like.  The two implementations use different
RNGs and different nuisance learners, so the estimates are not
bit-identical; the principled parity criterion is therefore that the
two AIPW point estimates agree **within combined Monte Carlo error**
(``|sp - grf| < 3 * sqrt(se_sp^2 + se_grf^2)``), the same combined-SE
standard the package uses for cross-estimator parity, rather than an
arbitrary fixed relative band.

This replaces the previous test, which compared the *plug-in* mean of
the CATE predictions (``cf.ate()``) against grf's AIPW estimate with a
25% tolerance.  The plug-in average is a regularised CATE-summary
convenience path, not grf's doubly-robust AIPW estimand, so that
comparison both used the wrong estimator and needed a band too wide to
be called validation.  The plug-in path is still exercised below as a
documented sanity check, not as the parity claim.

References
----------
- Athey, S., Tibshirani, J. and Wager, S. (2019). Generalized random
  forests. *Annals of Statistics*, 47(2), 1148-1178.
  [@athey2019generalized]
"""

from __future__ import annotations

import json
import math
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
    """Fit once per module -- causal forests are slow."""
    return sp.causal_forest(
        "y ~ W | X1 + X2 + X3 + X4 + X5",
        data=grf_data,
        n_estimators=2000,
        random_state=42,
        discrete_treatment=True,
    )


def test_grf_ate_aipw_within_combined_se(fitted_cf, r_reference):
    """sp AIPW ATE must agree with grf AIPW ATE within 3 combined SE.

    This is the headline causal-forest parity claim: same estimand
    (doubly-robust AIPW ATE), agreement within combined Monte Carlo
    error.
    """
    aipw = fitted_cf.average_treatment_effect(target_sample="all")
    assert aipw["method"] == "aipw", "headline ATE must use the AIPW score"
    py_ate, py_se = float(aipw["estimate"]), float(aipw["se"])
    r_ate, r_se = r_reference["ate"]["estimate"], r_reference["ate"]["se"]
    combined_se = math.sqrt(py_se**2 + r_se**2)
    z = abs(py_ate - r_ate) / combined_se
    assert z < 3.0, (
        f"sp AIPW ATE={py_ate:.4f} (SE {py_se:.4f}) vs grf AIPW "
        f"ATE={r_ate:.4f} (SE {r_se:.4f}): {z:.2f} combined SE apart "
        f"(threshold 3). The two doubly-robust estimates are not within "
        f"combined Monte Carlo error -- investigate the AIPW score or "
        f"the nuisance cross-fitting."
    )


def test_grf_ate_sign_agreement(fitted_cf, r_reference):
    """Both engines agree on the sign of the AIPW ATE."""
    py_ate = float(fitted_cf.average_treatment_effect(target_sample="all")["estimate"])
    r_ate = r_reference["ate"]["estimate"]
    assert (py_ate > 0) == (r_ate > 0), (
        f"Sign disagreement is a serious red flag: "
        f"Python AIPW ATE={py_ate:.4f}, R AIPW ATE={r_ate:.4f}"
    )


def test_grf_aipw_recovers_grf_ci(fitted_cf, r_reference):
    """sp AIPW point estimate lies inside grf's 95% CI (and vice versa).

    A weaker, asymmetric cross-check that does not depend on the sp SE.
    """
    py_ate = float(fitted_cf.average_treatment_effect(target_sample="all")["estimate"])
    r_ate, r_se = r_reference["ate"]["estimate"], r_reference["ate"]["se"]
    lo, hi = r_ate - 1.96 * r_se, r_ate + 1.96 * r_se
    assert (
        lo <= py_ate <= hi
    ), f"sp AIPW ATE={py_ate:.4f} outside grf 95% CI [{lo:.4f}, {hi:.4f}]"


def test_grf_plugin_is_documented_convenience_estimand(fitted_cf, r_reference):
    """Documents (does not validate) the plug-in CATE average path.

    ``cf.ate()`` is the mean of the CATE predictions; it is retained as
    a convenience but is NOT the parity estimand. This test asserts that
    it remains a distinct, finite CATE-summary path so a regression that
    silently aliases it to the AIPW estimand is caught, without treating
    the plug-in mean as validated.
    """
    plug_in = float(fitted_cf.ate())
    aipw = float(fitted_cf.average_treatment_effect(target_sample="all")["estimate"])
    assert math.isfinite(plug_in)
    assert 0.01 < abs(plug_in - aipw) < 0.25, (
        f"plug-in mean ({plug_in:.4f}) should remain a distinct CATE-summary "
        f"path from the AIPW estimate ({aipw:.4f}); semantics may have changed."
    )


def test_grf_fixture_meta(r_reference):
    assert "meta" in r_reference
    assert r_reference["meta"]["seed"] == 42
    assert r_reference["meta"]["num_trees"] == 2000


def test_grf_fixture_n(grf_data):
    assert len(grf_data) == 1000
