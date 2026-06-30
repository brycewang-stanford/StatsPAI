"""Reference parity: ``sp.ivreg(vce="wild")`` vs Stata ``ivreg2`` + ``boottest``.

The IV wild bootstrap is the **WRE** (wild restricted efficient) bootstrap of
Davidson-MacKinnon (2010) — it imposes the null, resamples the structural and
reduced-form residuals with the same wild weight, regenerates the endogenous
regressor and outcome, and refits 2SLS. This pins ``sp.ivreg(vce="wild")`` to
``boottest`` after ``ivreg2`` (David Roodman).

Reference values were produced on two fixed panels (Stata 18 MP; ``ivreg2``
04.1.12, ``boottest`` 4.5.3)::

    ivreg2 y w (d = z1 z2), cluster(firm)
    boottest d, reps(99999) weighttype(rademacher) nograph   // r(p) below

The weak-instrument panel is the decisive test: there the *efficient* reduced
form (which boottest uses, and which is StatsPAI's default) and the naive
reduced form diverge sharply (StatsPAI efficient p = 0.3415 vs naive 0.4256),
and boottest's 0.3412 selects the efficient one.

See ``REFERENCES.md`` and ``docs/guides/grammar.md``.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.inference.iv_wild import iv_wild_bootstrap
from statspai.regression.iv import ivreg

# --- Stata ivreg2 + boottest WRE reference values (frozen) ---------------
# Strong instruments (F ~ 284), 600 obs / 20 clusters.
STATA_STRONG_COEF = 0.07075778262722557
STATA_STRONG_P = 0.20155202  # boottest rademacher WRE
# Weak instruments (F ~ 4), 400 obs / 16 clusters — discriminates efficient RF.
STATA_WEAK_COEF = 0.19650046019529108
STATA_WEAK_P = 0.34120178


def _strong_panel() -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    n, g = 600, 20
    firm = rng.integers(0, g, n)
    clu_u = rng.normal(0, 0.6, size=g)[firm]
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = clu_u + rng.normal(size=n)
    d = 0.7 * z1 + 0.5 * z2 + 0.8 * u + rng.normal(size=n)
    w = rng.normal(size=n)
    y = 1.0 + 0.10 * d + 0.3 * w + u
    return pd.DataFrame({"y": y, "d": d, "w": w, "z1": z1, "z2": z2, "firm": firm})


def _weak_panel() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n, g = 400, 16
    firm = rng.integers(0, g, n)
    clu = rng.normal(0, 0.7, size=g)[firm]
    z1 = rng.normal(size=n)
    u = clu + rng.normal(size=n)
    d = 0.18 * z1 + 1.2 * u + rng.normal(size=n)
    y = 1.0 + 0.15 * d + u
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "firm": firm})


def test_strong_iv_wild_matches_boottest() -> None:
    df = _strong_panel()
    r = ivreg(
        "y ~ w + (d ~ z1 + z2)",
        data=df,
        vce="wild",
        cluster="firm",
        wild_reps=99999,
        seed=12345,
    )
    assert np.isclose(float(r.params["d"]), STATA_STRONG_COEF, atol=1e-5)
    assert np.isclose(float(r.pvalues["d"]), STATA_STRONG_P, atol=4e-3), (
        float(r.pvalues["d"]),
        STATA_STRONG_P,
    )


def test_weak_iv_wild_matches_boottest_and_uses_efficient_rf() -> None:
    df = _weak_panel()
    r = ivreg(
        "y ~ (d ~ z1)", data=df, vce="wild", cluster="firm", wild_reps=99999, seed=777
    )
    assert np.isclose(float(r.params["d"]), STATA_WEAK_COEF, atol=1e-5)
    assert np.isclose(float(r.pvalues["d"]), STATA_WEAK_P, atol=4e-3), (
        float(r.pvalues["d"]),
        STATA_WEAK_P,
    )
    # The naive (non-efficient) reduced form would give ~0.426 here — far from
    # boottest's 0.341. This guards that the default efficient RF is in force.
    base = ivreg("y ~ (d ~ z1)", data=df, cluster="firm")
    naive = iv_wild_bootstrap(
        base, df, "firm", "d", n_boot=99999, seed=777, efficient=False
    )
    assert abs(naive["p_boot"] - STATA_WEAK_P) > 0.05
    eff = iv_wild_bootstrap(
        base, df, "firm", "d", n_boot=99999, seed=777, efficient=True
    )
    assert abs(eff["p_boot"] - STATA_WEAK_P) < 4e-3


def test_iv_wild_requires_cluster_and_only_endog() -> None:
    df = _weak_panel()
    with pytest.raises(Exception):
        ivreg("y ~ (d ~ z1)", data=df, vce="wild")  # no cluster
    base = ivreg("y ~ (d ~ z1)", data=df, cluster="firm")
    # observed cluster SE used by the bootstrap matches sp.ivreg's clustered SE
    out = iv_wild_bootstrap(base, df, "firm", "d", n_boot=49, seed=1)
    assert np.isclose(out["se_cluster"], float(base.std_errors["d"]), atol=1e-9)
