"""``sp.iv.many_weak_ar``: the Anderson-Rubin set, inverted exactly.

Three identities pin it down. The statistic at an end point of the set
equals its critical value, and it is the F statistic for the joint
significance of the instruments in the regression of ``y - beta * d`` on a
constant, the exogenous regressors and the instruments. The value that
minimises the statistic is the LIML estimate. And the set does not depend
on the location of the instruments, which it did before the constant was
partialled out.

Found by the SE calibration screen (``tests/reliability/se_screen/iv.py``,
row ``many_weak_ar/many-weak``): the old default grid, 101 points within 0.5
of the OLS estimate, left the true value outside in 30% of the draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

K = 12


def _data(seed=0, n=600, pi=0.12, shift=0.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, K)) + shift
    x = rng.normal(size=n)
    v = rng.normal(size=n)
    u = 0.6 * v + 0.8 * rng.normal(size=n)
    d = pi * (z - shift).sum(axis=1) + 0.3 * x + v
    y = 2.0 + 0.5 * d + 0.3 * x + u
    df = pd.DataFrame(z, columns=[f"z{j}" for j in range(K)])
    df["x"], df["d"], df["y"] = x, d, y
    return df


ZS = [f"z{j}" for j in range(K)]


def _f_of_instruments(df, beta):
    """F statistic of the instruments in OLS of y - beta d on (1, x, Z)."""
    n = len(df)
    e = (df.y - beta * df.d).to_numpy()
    X0 = np.column_stack([np.ones(n), df.x])
    X1 = np.column_stack([X0, df[ZS].to_numpy()])
    rss0 = np.sum((e - X0 @ np.linalg.lstsq(X0, e, rcond=None)[0]) ** 2)
    rss1 = np.sum((e - X1 @ np.linalg.lstsq(X1, e, rcond=None)[0]) ** 2)
    return (rss0 - rss1) / K / (rss1 / (n - K - 2))


def test_end_points_sit_at_the_critical_value_of_the_f_test():
    df = _data()
    r = sp.iv.many_weak_ar(df, y="y", endog="d", instruments=ZS, exog=["x"])
    assert r.detail["shape"] == "interval"
    crit = stats.f.ppf(0.95, K, len(df) - K - 2)
    assert r.detail["critical_value"] == pytest.approx(crit, rel=1e-12)
    for end in r.ci:
        assert _f_of_instruments(df, end) == pytest.approx(crit, rel=1e-8)
    inside = 0.5 * (r.ci[0] + r.ci[1])
    assert _f_of_instruments(df, inside) < crit
    assert _f_of_instruments(df, r.ci[1] + 0.05) > crit


def test_point_estimate_is_liml():
    df = _data(seed=1)
    r = sp.iv.many_weak_ar(df, y="y", endog="d", instruments=ZS, exog=["x"])
    liml = sp.liml(data=df, y="y", x_endog=["d"], x_exog=["x"], z=ZS)
    assert r.estimate == pytest.approx(float(liml.params["d"]), rel=1e-8)
    assert r.ci[0] < r.estimate < r.ci[1]


def test_set_does_not_depend_on_the_location_of_the_instruments():
    """A constant is partialled out: shifting every instrument changes nothing."""
    a = sp.iv.many_weak_ar(_data(seed=2), y="y", endog="d", instruments=ZS, exog=["x"])
    b = sp.iv.many_weak_ar(
        _data(seed=2, shift=5.0), y="y", endog="d", instruments=ZS, exog=["x"]
    )
    assert b.ci == pytest.approx(a.ci, rel=1e-8)
    assert b.estimate == pytest.approx(a.estimate, rel=1e-8)


def test_irrelevant_instruments_give_an_unbounded_set_not_a_short_interval():
    df = _data(seed=3, pi=0.0)
    r = sp.iv.many_weak_ar(df, y="y", endog="d", instruments=ZS, exog=["x"])
    assert r.detail["shape"] in ("unbounded", "disjoint")
    assert r.ci == (-np.inf, np.inf)
    assert np.isnan(r.se)
    if r.detail["shape"] == "disjoint":
        lo, hi = r.detail["roots"]
        crit = r.detail["critical_value"]
        # Between the roots the instruments are significant: those values
        # are the ones excluded.
        assert _f_of_instruments(df, 0.5 * (lo + hi)) > crit
        assert _f_of_instruments(df, lo - 1.0) < crit


def test_grid_reports_when_the_set_reaches_the_end_of_the_grid():
    df = _data(seed=4)
    exact = sp.iv.many_weak_ar(df, y="y", endog="d", instruments=ZS, exog=["x"])
    wide = np.linspace(exact.ci[0] - 1, exact.ci[1] + 1, 4001)
    g = sp.iv.many_weak_ar(
        df, y="y", endog="d", instruments=ZS, exog=["x"], beta_grid=wide
    )
    step = wide[1] - wide[0]
    assert g.detail["touches_grid_end"] is False
    assert g.ci[0] == pytest.approx(exact.ci[0], abs=step)
    assert g.ci[1] == pytest.approx(exact.ci[1], abs=step)
    narrow = np.linspace(exact.ci[0] + 0.05, exact.ci[1] - 0.05, 51)
    cut = sp.iv.many_weak_ar(
        df, y="y", endog="d", instruments=ZS, exog=["x"], beta_grid=narrow
    )
    assert cut.detail["touches_grid_end"] is True


def test_too_many_instruments_for_the_sample_is_refused():
    df = _data(seed=5, n=K + 2)
    with pytest.raises(sp.exceptions.DataInsufficient):
        sp.iv.many_weak_ar(df, y="y", endog="d", instruments=ZS, exog=["x"])
