"""``sp.wild_cluster_boot``: the bootstrap the estimator defines, multi-way
clustering and joint restrictions, against Stata ``boottest``.

Before, the function refit least squares on whatever design a result
stored: after ``sp.ivreg`` it bootstrapped the OLS coefficient, after
``sp.logit`` a linear probability model, after ``sp.did`` it failed, and
``sp.feols`` IV with ``vce='wild'`` reported OLS cluster SEs of the
structural design (0.024 against the 2SLS 0.059 on the fixture below).

References
----------
* ``_fixtures/wild_multiway_joint_Stata.json`` --
  ``_generate_wild_multiway_joint_Stata.do`` (Stata 18 MP, boottest 4.5.3)
  on ``fixest_postestimation_data.csv`` read as doubles: one coefficient
  with one- and two-way clustering (bootstrap clusters a or b), and two
  restrictions jointly. Rademacher weights, enumerated.
* The score bootstrap after logit / Poisson and the WRE after 2SLS: the
  frozen boottest values of ``test_feglm_wild_boottest_parity.py`` and
  ``test_iv_wild_boottest_parity.py`` (same designs rebuilt here), reached
  now through ``sp.wild_cluster_boot`` on ``sp.logit`` / ``sp.poisson``
  fits; after ``sp.ivreg`` the routing to that WRE engine is checked draw
  by draw.

Tolerances: enumerated p-values are ratios of integers and agree exactly
(``abs=1e-12``); statistics to ``rtol=1e-9`` (Stata reads the CSV as
doubles; the remaining gap is summation order).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures"
S = json.loads((FIX / "wild_multiway_joint_Stata.json").read_text(encoding="utf-8"))
D = pd.read_csv(FIX / "fixest_postestimation_data.csv").assign(
    a=lambda d: d["cl"], b=lambda d: d["f3"]
)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def ols():
    return _quiet(sp.regress, "y ~ x1 + x2", data=D)


@pytest.mark.parametrize(
    "key,kw",
    [
        ("oneway", dict(cluster="a")),
        ("twoway_a", dict(cluster=["a", "b"], bootcluster="a")),
        ("twoway_b", dict(cluster=["a", "b"], bootcluster="b")),
    ],
)
def test_one_coefficient_matches_boottest(ols, key, kw):
    out = _quiet(
        sp.wild_cluster_boot, ols, D, variable="x1", h0=0.75, n_boot=9999, **kw
    )
    ref = S[key]
    assert out["n_boot"] == ref["reps"]
    assert out["p_boot"] == pytest.approx(ref["p"], abs=1e-12)
    np.testing.assert_allclose(out["t_stat"], ref["t"], rtol=1e-9)


def test_two_way_drops_draws_without_a_variance(ols):
    """96/143 = 2496/3718: 378 of the 4096 draws have a negative two-way
    variance and no statistic."""
    out = _quiet(
        sp.wild_cluster_boot,
        ols,
        D,
        cluster=["a", "b"],
        variable="x1",
        h0=0.75,
        bootcluster="a",
        n_boot=9999,
    )
    assert out["n_boot_valid"] == 3718
    assert out["p_boot"] * out["n_boot_valid"] == pytest.approx(2496, abs=1e-9)


@pytest.mark.parametrize(
    "key,kw",
    [
        ("joint_oneway", dict(cluster="a")),
        ("joint_twoway_a", dict(cluster=["a", "b"], bootcluster="a")),
        ("joint_twoway_b", dict(cluster=["a", "b"], bootcluster="b")),
    ],
)
def test_joint_restrictions_match_boottest(ols, key, kw):
    out = _quiet(
        sp.wild_cluster_boot,
        ols,
        D,
        variable=["x1", "x2"],
        h0=[0.75, -0.25],
        n_boot=9999,
        **kw,
    )
    ref = S[key]
    assert out["n_boot"] == ref["reps"]
    assert out["p_boot"] == pytest.approx(ref["p"], abs=1e-12)
    np.testing.assert_allclose(out["statistic"], ref["F"], rtol=1e-9)


def test_one_restriction_through_the_general_path_is_the_one_way_test(ols):
    """Identity: the joint engine with one restriction and one cluster
    variable reproduces the established one-way WCR (|t| ordering)."""
    single = _quiet(sp.wild_cluster_boot, ols, D, "a", "x1", n_boot=9999, h0=0.75)
    general = _quiet(
        sp.wild_cluster_boot, ols, D, ["a"], "x1", n_boot=9999, h0=0.75, bootcluster="a"
    )
    assert general["p_boot"] == single["p_boot"]
    np.testing.assert_allclose(general["t_stat"], single["t_stat"], rtol=1e-12)


# --------------------------------------------------------------------------
# The bootstrap the estimator defines
# --------------------------------------------------------------------------


def _logit_panel() -> pd.DataFrame:
    rng = np.random.default_rng(77)
    n, g = 300, 9
    firm = rng.integers(0, 4, n)
    clu = rng.integers(0, g, n)
    x1, x2, x3 = rng.normal(size=(3, n))
    p = 1.0 / (1.0 + np.exp(-(0.2 + 0.6 * x1 - 0.3 * x2 + 0.0 * x3 + 0.25 * firm)))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "x3": x3, "firm": firm, "clu": clu}
    )


def _pois_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n, g = 240, 10
    firm = rng.integers(0, 5, n)
    clu = rng.integers(0, g, n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    mu = np.exp(0.3 + 0.35 * x1 - 0.2 * x2 + 0.0 * x3 + 0.1 * firm)
    y = rng.poisson(mu)
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "x3": x3, "firm": firm, "clu": clu}
    )


def test_logit_uses_the_score_bootstrap():
    """boottest after ``logit y x1 x2 x3 i.firm``: x3 p = 490/512, x1 2/512."""
    df = _logit_panel()
    fit = _quiet(sp.logit, "y ~ x1 + x2 + x3 + C(firm)", data=df)
    x3 = sp.wild_cluster_boot(fit, df, "clu", "x3", n_boot=2000)
    x1 = sp.wild_cluster_boot(fit, df, "clu", "x1", n_boot=2000)
    assert x3["p_boot"] == 0.95703125
    assert x1["p_boot"] == 0.00390625
    assert "score" in x3["method"]


def test_poisson_uses_the_score_bootstrap():
    """boottest after ``poisson y x1 x2 x3 i.firm``: x3 p = 324/1024."""
    df = _pois_panel()
    fit = _quiet(sp.poisson, "y ~ x1 + x2 + x3 + C(firm)", data=df)
    out = sp.wild_cluster_boot(fit, df, "clu", "x3", n_boot=2000)
    assert out["p_boot"] == 0.31640625
    assert out["t_stat"] == pytest.approx(-1.0999434, abs=1e-5)


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


def test_ivreg_uses_wre():
    """Routed to the WRE bootstrap whose boottest parity (22361 / 65536 on
    this design, ``test_iv_wild_boottest_parity.py``) is established; the
    same draws give the same numbers (a short run keeps the test fast)."""
    from statspai.inference.iv_wild import iv_wild_bootstrap

    df = _weak_panel()
    fit = _quiet(sp.ivreg, "y ~ (d ~ z1)", data=df, cluster="firm")
    out = sp.wild_cluster_boot(fit, df, "firm", "d", n_boot=199, seed=3)
    ref = iv_wild_bootstrap(fit, df, "firm", "d", n_boot=199, seed=3)
    assert out["method"] == "WRE"
    assert out["p_boot"] == ref["p_boot"]
    np.testing.assert_array_equal(out["t_distribution"], ref["t_distribution"])


def test_feols_iv_with_fixed_effects_uses_wre():
    """Identity (Frisch-Waugh-Lovell): WRE on the absorbed design equals
    WRE with the effects entered as dummies (p-values are invariant to the
    small-sample factor, which differs by the absorbed parameters)."""
    fml = "yiv ~ x1 | f3 | d1 ~ z1 + z2"
    fe = _quiet(sp.feols, fml, D)
    dummies = _quiet(
        sp.ivreg, "yiv ~ x1 + C(f3) + (d1 ~ z1 + z2)", data=D, cluster="cl"
    )
    a = sp.wild_cluster_boot(fe, D, "cl", "d1", n_boot=199, seed=5)
    b = sp.wild_cluster_boot(dummies, D, "cl", "d1", n_boot=199, seed=5)
    assert a["p_boot"] == b["p_boot"]
    # the draws agree up to the constant small-sample factor, (n - k) of
    # the dummy model against that of the absorbed one
    ratio = np.asarray(a["t_distribution"]) / np.asarray(b["t_distribution"])
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-8)
    n, k_fe, k_d = len(D), 2, 2 + D["f3"].nunique()
    np.testing.assert_allclose(ratio[0], np.sqrt((n - k_fe) / (n - k_d)), rtol=1e-10)
    np.testing.assert_allclose(a["beta_hat"], b["beta_hat"], rtol=1e-10)
    # sp.feols(vce='wild') on the IV form: WRE on d1, CRV1 kept on x1
    w = _quiet(sp.feols, fml, D, vce="wild", cluster="cl", wild_reps=199, seed=5)
    crv = _quiet(sp.feols, fml, D, vcov={"CRV1": "cl"})
    assert float(w.pvalues["d1"]) == a["p_boot"]
    np.testing.assert_allclose(w.std_errors["x1"], crv.std_errors["x1"], rtol=1e-12)
    np.testing.assert_allclose(w.std_errors["d1"], crv.std_errors["d1"], rtol=1e-12)


def test_did_replays_its_own_wild_bootstrap():
    dd = D.assign(tr=(D.f1 < 15).astype(int), post=(D.t >= 6).astype(int))
    fit = _quiet(sp.did, dd, y="y", treat="tr", time="post", cluster="cl")
    out = sp.wild_cluster_boot(fit, dd, "cl", n_boot=9999)
    direct = _quiet(
        sp.did,
        dd,
        y="y",
        treat="tr",
        time="post",
        cluster="cl",
        vce="wild",
        wild_reps=9999,
    )
    assert out["p_boot"] == float(direct.pvalue)
    assert out["beta_hat"] == pytest.approx(float(fit.estimate), rel=1e-12)
    with pytest.raises(MethodIncompatibility, match="ATT"):
        sp.wild_cluster_boot(fit, dd, "cl", "post")


@pytest.mark.parametrize(
    "fit",
    [
        lambda: sp.nbreg("yp ~ x1", data=D),
        lambda: sp.fepois("yp ~ x1 | f1", D),
    ],
)
def test_models_without_a_wild_bootstrap_are_refused(fit):
    with pytest.raises(MethodIncompatibility):
        sp.wild_cluster_boot(_quiet(fit), D, "cl", "x1")


def test_multiway_after_a_non_least_squares_fit_is_refused():
    fit = _quiet(sp.logit, "yb ~ x1 + x2", data=D)
    with pytest.raises(MethodIncompatibility, match="least-squares"):
        sp.wild_cluster_boot(fit, D, ["a", "b"], "x1")


# --------------------------------------------------------------------------
# Absorbed effects not nested in the clusters (boottest after reghdfe)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,fe", [("absorbed_1", "f3"), ("absorbed_2", "f1"), ("absorbed_3", "f1 + f3")]
)
def test_absorbed_effects_are_reabsorbed_in_each_draw(key, fe):
    """Before, the draws were not residualised on the effects again, so with
    effects not nested in the clusters the p-value was another bootstrap's
    (0.1304 against boottest's 0.1460 for f1)."""
    d = D.assign(yy=D["y"] + 0.33 * D["x2"])
    fit = _quiet(sp.feols, f"yy ~ x1 + x2 | {fe}", d)
    out = sp.wild_cluster_boot(fit, d, "cl", "x2", n_boot=9999)
    assert out["p_boot"] == pytest.approx(S[key]["p"], abs=1e-12)
    w = _quiet(
        sp.feols, f"yy ~ x1 + x2 | {fe}", d, vce="wild", cluster="cl", wild_reps=9999
    )
    assert float(w.pvalues["x2"]) == pytest.approx(S[key]["p"], abs=1e-12)


def test_absorbed_effects_joint():
    d = D.assign(yy=D["y"] + 0.33 * D["x2"])
    fit = _quiet(sp.feols, "yy ~ x1 + x2 | f1", d)
    out = sp.wild_cluster_boot(
        fit, d, "cl", ["x1", "x2"], h0=[0.45, 0.1], n_boot=9999, bootcluster="cl"
    )
    assert out["p_boot"] == pytest.approx(S["absorbed_joint"]["p"], abs=1e-12)
