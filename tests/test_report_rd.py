"""``result.report()`` for RD, and the density test's own violations."""

import warnings

import numpy as np
import pandas as pd

import statspai as sp


def _rd(seed, sort_share, n=6000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    # a share of units just below the cutoff move just above it
    move = (x < 0) & (x > -0.1) & (rng.random(n) < sort_share)
    x = np.where(move, -x, x)
    y = 0.6 * (x >= 0) + 0.5 * x + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "y": y})


def _fit(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.rdrobust(df, y="y", x="x", c=0)


def test_clean_design_reports_the_robust_interval():
    r = _fit(_rd(1, 0.0))
    rep = r.report()
    assert rep["density_pvalue"] >= 0.05
    assert rep["status"] == "conventional"
    assert rep["ci"] == [float(r.ci[0]), float(r.ci[1])]


def test_sorting_across_the_cutoff_is_a_refusal():
    r = _fit(_rd(1, 0.5))
    rep = r.report()
    assert rep["density_pvalue"] < 0.01
    assert rep["status"] == "refuse" and rep["ci"] is None
    assert "manipulated" in rep["refuse"]
    assert np.isfinite(rep["conventional"]["estimate"])


def test_density_test_has_no_standard_error_violation():
    # the headline of the density test is a t statistic: no SE by design
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = sp.rddensity(_rd(1, 0.0), x="x", c=0)
    assert [v for v in d.violations() if v["test"] == "se_positive"] == []
