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


# --- multi-endogenous WRE + two-way IV cluster -----------------------------
# Two endogenous regressors, 700 obs / 22 clusters.
STATA_2E_COEF_D1 = 0.09126668
STATA_2E_COEF_D2 = -0.16573432
STATA_2E_P_D1 = 0.21079211  # boottest WRE
STATA_2E_P_D2 = 0.01408014
# Two-way clustering, 800 obs / 25 x 18 clusters; ivreg2 cluster(a b) small.
STATA_TW_COEF_D = 0.31606801
STATA_TW_SE_D = 0.0519819


def _multiendog_panel() -> pd.DataFrame:
    rng = np.random.default_rng(303)
    n, g = 700, 22
    firm = rng.integers(0, g, n)
    cu = rng.normal(0, 0.6, g)[firm]
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    z3 = rng.normal(size=n)
    u = cu + rng.normal(size=n)
    d1 = 0.6 * z1 + 0.4 * z3 + 0.7 * u + rng.normal(size=n)
    d2 = 0.5 * z2 + 0.3 * z3 + 0.5 * u + rng.normal(size=n)
    w = rng.normal(size=n)
    y = 1.0 + 0.12 * d1 - 0.20 * d2 + 0.3 * w + u
    return pd.DataFrame(
        {"y": y, "d1": d1, "d2": d2, "w": w, "z1": z1, "z2": z2, "z3": z3, "firm": firm}
    )


def _twoway_panel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n, g1, g2 = 800, 25, 18
    firm = rng.integers(0, g1, n)
    year = rng.integers(0, g2, n)
    cu = rng.normal(0, 0.5, g1)[firm] + rng.normal(0, 0.4, g2)[year]
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = cu + rng.normal(size=n)
    d = 0.6 * z1 + 0.5 * z2 + 0.7 * u + rng.normal(size=n)
    w = rng.normal(size=n)
    y = 1.0 + 0.3 * d + 0.4 * w + u
    return pd.DataFrame(
        {"y": y, "d": d, "w": w, "z1": z1, "z2": z2, "firm": firm, "year": year}
    )


def test_multiendogenous_wild_matches_boottest() -> None:
    df = _multiendog_panel()
    base = ivreg("y ~ w + (d1 + d2 ~ z1 + z2 + z3)", data=df, cluster="firm")
    assert np.isclose(float(base.params["d1"]), STATA_2E_COEF_D1, atol=1e-5)
    assert np.isclose(float(base.params["d2"]), STATA_2E_COEF_D2, atol=1e-5)
    # Fewer reps here keep the test CI-fast; the wider tolerance still rejects a
    # wrong implementation (the naive reduced form is off by >0.05). The pinned
    # 99999-rep agreement (0.2101 vs 0.21079; 0.0151 vs 0.01408) is in
    # REFERENCES.md.
    o1 = iv_wild_bootstrap(base, df, "firm", "d1", n_boot=9999, seed=42)
    o2 = iv_wild_bootstrap(base, df, "firm", "d2", n_boot=9999, seed=42)
    assert np.isclose(o1["p_boot"], STATA_2E_P_D1, atol=1.5e-2), o1["p_boot"]
    assert np.isclose(o2["p_boot"], STATA_2E_P_D2, atol=1.5e-2), o2["p_boot"]


def test_iv_twoway_cluster_matches_ivreg2() -> None:
    df = _twoway_panel()
    r = ivreg("y ~ w + (d ~ z1 + z2)", data=df, cluster=["firm", "year"])
    assert np.isclose(float(r.params["d"]), STATA_TW_COEF_D, atol=1e-6)
    assert np.isclose(float(r.std_errors["d"]), STATA_TW_SE_D, atol=1e-5)


# --- IV CR2 / CR3 vs R clubSandwich ----------------------------------------
# clubSandwich::vcovCR(ivreg(...), type=...) SE of the endogenous coefficient.
CLUB_STRONG_CR2 = 0.05302589
CLUB_STRONG_CR3 = 0.05482302
CLUB_WEAK_CR2 = 0.31924312
CLUB_WEAK_CR3 = 0.33575834


def test_iv_cr2_cr3_match_clubsandwich_strong() -> None:
    df = _strong_panel()
    r2 = ivreg("y ~ w + (d ~ z1 + z2)", data=df, vce="CR2", cluster="firm")
    r3 = ivreg("y ~ w + (d ~ z1 + z2)", data=df, vce="CR3", cluster="firm")
    assert np.isclose(float(r2.std_errors["d"]), CLUB_STRONG_CR2, atol=1e-7)
    assert np.isclose(float(r3.std_errors["d"]), CLUB_STRONG_CR3, atol=1e-7)
    # vce="jackknife" is an alias for CR3
    rj = ivreg("y ~ w + (d ~ z1 + z2)", data=df, vce="jackknife", cluster="firm")
    assert np.isclose(float(rj.std_errors["d"]), CLUB_STRONG_CR3, atol=1e-7)


def test_iv_cr2_cr3_match_clubsandwich_weak() -> None:
    df = _weak_panel()
    r2 = ivreg("y ~ (d ~ z1)", data=df, vce="CR2", cluster="firm")
    r3 = ivreg("y ~ (d ~ z1)", data=df, vce="CR3", cluster="firm")
    assert np.isclose(float(r2.std_errors["d"]), CLUB_WEAK_CR2, atol=1e-7)
    assert np.isclose(float(r3.std_errors["d"]), CLUB_WEAK_CR3, atol=1e-7)
