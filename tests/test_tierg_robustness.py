"""Tier G — numerical robustness and failure-mode guards.

The project rule is "fail loudly": a pathological input must raise, warn, or
flag diagnostics — never silently return wrong numbers (NaN, or finite
garbage). This suite pins that contract for the core estimators so a future
refactor cannot quietly reintroduce a silent failure.

Three groups:

1. **Newly loud** — cases that previously returned silent garbage and now
   raise / warn (perfect collinearity in OLS, perfect separation in logit).
2. **Already loud** — cases that correctly raise / warn today, locked in.
3. **No false positives** — clean and legitimately ill-conditioned-but-full-
   rank designs (NIST-style) must fit without tripping the new guards. This is
   the safety net: a collinearity detector that flagged ill-conditioned full-
   rank data would be a correctness hazard, not a feature.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    ConvergenceWarning,
    MethodIncompatibility,
    NumericalInstability,
)

# ===========================================================================
# Group 1 — newly loud (were silent garbage before the Tier G fixes)
# ===========================================================================


def test_ols_duplicate_columns_raises():
    """Two identical regressors are rank-deficient -> NumericalInstability."""
    rng = np.random.default_rng(0)
    n = 100
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": x + rng.normal(size=n), "x1": x, "x2": x})
    with pytest.raises(NumericalInstability, match="collinear"):
        sp.regress("y ~ x1 + x2", data=df)


def test_ols_proportional_columns_raises():
    """A scalar multiple of another regressor is still perfect collinearity."""
    rng = np.random.default_rng(1)
    n = 100
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": x + rng.normal(size=n), "x1": x, "x2": 2.5 * x})
    with pytest.raises(NumericalInstability):
        sp.regress("y ~ x1 + x2", data=df)


def test_ols_constant_regressor_raises():
    """A constant non-intercept regressor is collinear with the intercept."""
    rng = np.random.default_rng(2)
    n = 100
    df = pd.DataFrame({"y": rng.normal(size=n), "x": np.full(n, 3.0)})
    with pytest.raises(NumericalInstability, match="constant"):
        sp.regress("y ~ x", data=df)


def test_ols_complementary_dummies_raises():
    """The dummy-variable trap (male + female + intercept) is caught."""
    rng = np.random.default_rng(3)
    n = 200
    male = rng.binomial(1, 0.5, n).astype(float)
    df = pd.DataFrame({"y": rng.normal(size=n), "male": male, "female": 1.0 - male})
    with pytest.raises(NumericalInstability):
        sp.regress("y ~ male + female", data=df)


def test_ols_collinearity_error_names_the_columns():
    """The error payload must identify the offending columns for the user."""
    rng = np.random.default_rng(4)
    n = 80
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": rng.normal(size=n), "a": x, "b": x})
    with pytest.raises(NumericalInstability) as ei:
        sp.regress("y ~ a + b", data=df)
    diag = ei.value.diagnostics
    assert "collinear_pair" in diag
    assert set(diag["collinear_pair"]) == {"a", "b"}


def test_logit_perfect_separation_warns():
    """Perfectly separable data -> ConvergenceWarning (MLE does not exist)."""
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    df = pd.DataFrame({"y": (x > 0).astype(int), "x": x})
    with pytest.warns(ConvergenceWarning, match="separation"):
        sp.logit("y ~ x", data=df)


# ===========================================================================
# Group 2 — already loud, locked in
# ===========================================================================


def test_did_all_treated_raises():
    """No control group -> the 2x2 estimator must refuse, not invent an ATT."""
    rng = np.random.default_rng(5)
    rows = []
    for i in range(100):
        for t in (0, 1):
            rows.append({"i": i, "t": t, "treated": 1, "post": t, "y": rng.normal()})
    with pytest.raises(MethodIncompatibility):
        sp.did(pd.DataFrame(rows), y="y", treat="treated", time="t", post="post")


def test_panel_singleton_entities_raises():
    """One observation per entity fully absorbs the design -> must raise."""
    rng = np.random.default_rng(6)
    n = 50
    df = pd.DataFrame(
        {
            "i": range(n),
            "t": [0] * n,
            "d": rng.binomial(1, 0.5, n),
            "y": rng.normal(size=n),
        }
    )
    with pytest.raises(Exception):  # AbsorbingEffectError from the FE backend
        sp.panel(df, formula="y ~ d", entity="i", time="t", method="fe")


def test_ivreg_irrelevant_instrument_warns():
    """A zero-first-stage instrument must surface a warning, not pass silently."""
    rng = np.random.default_rng(7)
    n = 300
    z = rng.normal(size=n)
    d = rng.normal(size=n)  # independent of z
    df = pd.DataFrame({"y": d + rng.normal(size=n), "d": d, "z": z})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sp.ivreg("y ~ (d ~ z)", data=df)
    assert len(w) >= 1, "irrelevant instrument produced no warning"


def test_ols_zero_variance_outcome_warns():
    """A constant outcome (no variance to explain) must warn."""
    rng = np.random.default_rng(8)
    n = 100
    df = pd.DataFrame({"y": np.ones(n), "x": rng.normal(size=n)})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sp.regress("y ~ x", data=df)
    assert len(w) >= 1, "zero-variance outcome produced no warning"


def test_regress_empty_data_raises():
    """An empty frame has nothing to estimate -> ValueError."""
    with pytest.raises(ValueError):
        sp.regress("y ~ x", data=pd.DataFrame({"y": [], "x": []}))


# ===========================================================================
# Group 3 — no false positives (the guard must not break good fits)
# ===========================================================================


def test_clean_regress_does_not_raise_or_warn():
    """A well-posed regression must fit silently — no spurious collinearity."""
    rng = np.random.default_rng(9)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)  # independent of x1
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        res = sp.regress("y ~ x1 + x2", data=df)
    assert np.all(np.isfinite(np.asarray(res.params)))


def test_correlated_but_full_rank_does_not_raise():
    """Highly correlated (r~0.99) but full-rank regressors must still fit.

    Guards the collinearity threshold from being too aggressive — the worst
    NIST ill-conditioned design sits at |corr| ~0.999, so legitimate strong
    correlation must pass.
    """
    rng = np.random.default_rng(10)
    n = 500
    x1 = rng.normal(size=n)
    x2 = x1 + 0.05 * rng.normal(size=n)  # corr ~0.998, full rank
    y = x1 + x2 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    res = sp.regress("y ~ x1 + x2", data=df)  # must not raise
    assert np.all(np.isfinite(np.asarray(res.params)))


def test_clean_logit_does_not_warn_separation():
    """Overlapping (non-separable) logit data must not trip the separation warning."""
    rng = np.random.default_rng(11)
    n = 400
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.5 * x)))
    y = rng.binomial(1, p)
    df = pd.DataFrame({"y": y, "x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        sp.logit("y ~ x", data=df)  # must not raise a ConvergenceWarning
