"""Tier G — robustness tests for ``sp.dml``.

DML's failure contract (StatsPAI 1.17.0) is exemplary and locked here:

* **model='irm' + non-binary treatment** -> ``MethodIncompatibility``.
* **missing covariate column** -> ``ValueError`` naming the column.
* **n_folds > n** -> ``ValueError`` from the splitter.
* **no treatment variation** (constant d) -> ``RuntimeError`` ("PLR
  denominator ≈ 0") rather than a silent ``inf``/``NaN`` — textbook §7.
* **NaN rows** -> dropped listwise, finite estimate (documented).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

from ._helpers import assert_no_silent_wrong, assert_raises_clean

pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402

_P = 4
_XCOLS = [f"x{i}" for i in range(_P)]


def _make(n=400, seed=0, binary=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, _P))
    lin = X @ np.array([0.5, 0.2, 0.0, -0.3])
    d = (
        (lin + rng.normal(size=n) > 0).astype(float)
        if binary
        else lin + rng.normal(size=n)
    )
    y = 2.0 * d + X @ np.array([1.0, -1.0, 0.5, 0.3]) + rng.normal(size=n)
    df = pd.DataFrame(X, columns=_XCOLS)
    df["d"] = d
    df["y"] = y
    return df


def _plr(df, **kw):
    return sp.dml(
        data=df,
        y="y",
        treat="d",
        covariates=_XCOLS,
        model="plr",
        ml_g=LinearRegression(),
        ml_m=LinearRegression(),
        n_folds=kw.pop("n_folds", 4),
        random_state=1,
        **kw,
    )


# --------------------------------------------------------------------------- #
# G — method/schema errors                                                    #
# --------------------------------------------------------------------------- #
def test_dml_irm_requires_binary_treatment():
    df = _make(binary=False, seed=2)
    assert_raises_clean(
        lambda: sp.dml(
            data=df,
            y="y",
            treat="d",
            covariates=_XCOLS,
            model="irm",
            ml_g=LinearRegression(),
            ml_m=LogisticRegression(),
            n_folds=4,
            random_state=1,
        ),
        Exception,
        match="binary",
    )


def test_dml_missing_covariate_raises():
    df = _make(seed=1)
    assert_raises_clean(
        lambda: sp.dml(
            data=df,
            y="y",
            treat="d",
            covariates=_XCOLS + ["zzz"],
            model="plr",
            ml_g=LinearRegression(),
            ml_m=LinearRegression(),
            n_folds=4,
            random_state=1,
        ),
        ValueError,
        KeyError,
        match="zzz|not found",
    )


def test_dml_too_many_folds_raises():
    df = _make(n=20, seed=1)
    assert_raises_clean(lambda: _plr(df, n_folds=500), ValueError)


# --------------------------------------------------------------------------- #
# G10 — no treatment variation -> loud RuntimeError, never silent inf/NaN     #
# --------------------------------------------------------------------------- #
def test_dml_constant_treatment_raises_not_silent():
    df = _make(seed=1).assign(d=1.0)
    out = None
    try:
        r = _plr(df)
        out = float(r.estimate)
    except Exception as e:  # RuntimeError: PLR denominator ≈ 0
        assert str(e).strip()
        return
    raise AssertionError(f"constant treatment should raise, got silent θ̂={out}")


# --------------------------------------------------------------------------- #
# G1 — NaN handling: listwise drop, finite (documented)                       #
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore")
def test_dml_nan_rows_finite_estimate():
    df = _make(n=500, seed=4)
    clean = _plr(df)
    dirty = df.copy()
    dirty.loc[:4, "y"] = np.nan
    got = _plr(dirty)
    assert np.isfinite(got.estimate), "NaN rows produced a non-finite θ̂"
    np.testing.assert_allclose(got.estimate, clean.estimate, rtol=0.1)


@pytest.mark.filterwarnings("ignore")
def test_dml_inf_outcome_not_silent_finite():
    df = _make(seed=4)
    dirty = df.copy()
    dirty.loc[0, "y"] = np.inf
    assert_no_silent_wrong(lambda: _plr(dirty))
