"""Cohort 0 when 0 is also an observed period (one rule, said out loud).

CS, Sun-Abraham and the imputation estimator read cohort 0 as never
treated (R did / csdid). With periods 0..T the same value could mean
"first treated in period 0"; the estimators keep the convention and warn.
Relabelling the periods 1..T+1 (cohorts shifted with them) removes the
ambiguity, gives the same estimate and no warning.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_MSG = "also an observed value"


def _panel():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(120):
        g = [0, 3, 5][u % 3]
        for t in range(8):
            rows.append(
                dict(id=u, t=t, g=g, y=rng.normal() + (1.0 if g and t >= g else 0.0))
            )
    return pd.DataFrame(rows)


def _shifted(df):
    return df.assign(t=df.t + 1, g=df.g.where(df.g == 0, df.g + 1))


_FITS = {
    "callaway_santanna": lambda d: sp.callaway_santanna(d, y="y", g="g", t="t", i="id"),
    "sun_abraham": lambda d: sp.sun_abraham(d, y="y", g="g", t="t", i="id"),
    "did_imputation": lambda d: sp.did_imputation(
        d, y="y", group="id", time="t", first_treat="g"
    ),
}


def _fit(name, d):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = _FITS[name](d)
    return res, [w for w in rec if _MSG in str(w.message)]


@pytest.mark.parametrize("name", sorted(_FITS))
def test_zero_cohort_with_a_period_zero_warns_once(name):
    _, hits = _fit(name, _panel())
    assert len(hits) == 1
    assert issubclass(hits[0].category, sp.exceptions.AssumptionWarning)
    assert "40 unit(s)" in str(hits[0].message)


@pytest.mark.parametrize("name", sorted(_FITS))
def test_unambiguous_coding_is_silent_and_gives_the_same_estimate(name):
    a, _ = _fit(name, _panel())
    b, hits = _fit(name, _shifted(_panel()))
    assert hits == []
    assert float(a.estimate) == pytest.approx(float(b.estimate), rel=1e-10)


def test_nan_for_never_treated_is_silent():
    df = _panel()
    df["g"] = df["g"].where(df["g"] != 0)
    _, hits = _fit("callaway_santanna", df)
    assert hits == []
