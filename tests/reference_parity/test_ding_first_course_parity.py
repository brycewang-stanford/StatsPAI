"""R parity for four computations of Ding's *A First Course in Causal
Inference* (2024) that StatsPAI did not reproduce before the October 2026
pass over the book.

The reference numbers are in ``_fixtures/ding_first_course_R.json``, written
by ``_fixtures/_generate_ding_first_course.R`` on data simulated there (the
book's own data are not redistributed). Both sides read the same CSV bytes.

* Rosenbaum's sensitivity analysis of the mean pair difference,
  ``sp.rosenbaum_bounds(method="t")`` against ``sensitivitymw::senmw``.
* The Anderson-Rubin test with a heteroskedasticity-robust variance,
  ``sp.anderson_rubin_test(ar_vcov=...)`` against ``sandwich::vcovHC`` on the
  reduced form.
* The Horvitz-Thompson weighting estimator of the effect on the treated and
  on the controls, ``sp.ipw(normalize=False)``, against its definition. It
  was scaled by the treated (control) share through 1.38.0.
* The Welch, rank-sum and Lin statistics of ``sp.ri_test`` against
  ``t.test``, ``wilcox.test`` and ``lm`` + ``vcovHC``; Abadie-Imbens matching
  against ``Matching::Match`` with both of its variance estimators.

Everything here is deterministic, so the tolerance is the strict parity
budget of ``CLAUDE.md`` section 5.1 (relative 1e-6); the realised gaps are
1e-9 or smaller and the tighter bounds below say so.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "ding_first_course_R.json").read_text(encoding="utf-8"))


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / f"ding_first_course_{name}.csv")


def test_rosenbaum_t_matches_senmw(ref):
    pairs = _csv("pairs")
    res = sp.rosenbaum_bounds(
        pairs["treated"], pairs["control"], method="t", gamma_grid=ref["pairs_gamma"]
    )
    np.testing.assert_allclose(res.pvalue_upper, ref["pairs_p_upper"], rtol=1e-10)
    # the statistic is the sum of the positive differences
    d = (pairs["treated"] - pairs["control"]).to_numpy()
    assert res.statistic == pytest.approx(d[d > 0].sum(), rel=1e-12)
    # the bias the study can absorb grows with Gamma
    assert np.all(np.diff(res.pvalue_upper) > 0)
    assert np.all(res.pvalue_lower <= res.pvalue_upper)


def test_rosenbaum_t_reduces_to_the_sign_flip_test_at_gamma_one():
    # At Gamma = 1 the bound is the normal approximation to the paired
    # randomization test of the mean: z = sum(d) / sqrt(sum(d^2)).
    rng = np.random.default_rng(0)
    d = rng.normal(0.3, 1.0, 60)
    res = sp.rosenbaum_bounds(d, np.zeros(60), method="t", gamma_grid=[1.0])
    from scipy import stats

    z = d.sum() / np.sqrt(np.sum(d**2))
    assert res.pvalue_upper[0] == pytest.approx(stats.norm.sf(z), rel=1e-12)
    assert res.pvalue_lower[0] == pytest.approx(res.pvalue_upper[0], rel=1e-12)


@pytest.mark.parametrize(
    "instruments,key", [(["z1"], "iv_one"), (["z1", "z2"], "iv_two")]
)
def test_robust_anderson_rubin_matches_sandwich(ref, instruments, key):
    iv = _csv("iv")
    for ty, block in zip(ref["iv_types"], ref[key]):
        for b0, stat, pval in zip(ref["iv_b0"], block[0], block[1]):
            res = sp.anderson_rubin_test(
                iv, "y", "d", instruments, exog=["x1", "x2"], h0=b0, ar_vcov=ty
            )
            assert res["ar_stat"] == pytest.approx(stat, rel=1e-9), (ty, b0)
            assert res["ar_pvalue"] == pytest.approx(pval, rel=1e-8), (ty, b0)
            assert res["ar_vcov"] == ty


def test_robust_anderson_rubin_interval_and_what_it_changes(ref):
    iv = _csv("iv")
    robust = sp.anderson_rubin_test(
        iv, "y", "d", ["z1"], exog=["x1", "x2"], ar_vcov="HC3"
    )
    np.testing.assert_allclose(robust["ar_ci"], ref["iv_ci_hc3_one"], rtol=1e-6)
    # The error variance grows with |z1| in this design. The homoskedastic
    # statistic is then about twice the robust one, and its interval is
    # shorter than the interval that has the stated coverage.
    classic = sp.anderson_rubin_test(iv, "y", "d", ["z1"], exog=["x1", "x2"])
    assert classic["ar_vcov"] == "classic"
    assert classic["ar_stat"] > 1.8 * robust["ar_stat"]
    width = lambda r: r["ar_ci"][1] - r["ar_ci"][0]  # noqa: E731
    assert width(classic) < width(robust)


def test_robust_anderson_rubin_refusals():
    iv = _csv("iv")
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.anderson_rubin_test(iv, "y", "d", ["z1"], ar_vcov="HC4")
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.anderson_rubin_test(
            iv.assign(g=np.arange(len(iv)) % 30),
            "y",
            "d",
            ["z1"],
            ar_vcov="HC1",
            cluster="g",
        )
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.anderson_rubin_test(
            iv.assign(g=np.arange(len(iv)) % 30),
            "y",
            "d",
            ["z1"],
            ar_vcov="HC3",
            absorb="g",
        )


def test_horvitz_thompson_weighting_on_the_treated_and_the_controls(ref):
    obs = _csv("obs")
    cov = ["x1", "x2", "x3"]

    def est(estimand, normalize):
        return sp.ipw(
            obs,
            "y",
            "z",
            cov,
            estimand=estimand,
            normalize=normalize,
            n_bootstrap=5,
            seed=0,
        ).estimate

    assert est("ATT", False) == pytest.approx(ref["obs_att_ht"], rel=1e-8)
    assert est("ATT", True) == pytest.approx(ref["obs_att_hajek"], rel=1e-8)
    assert est("ATC", False) == pytest.approx(ref["obs_atc_ht"], rel=1e-8)
    assert est("ATE", False) == pytest.approx(ref["obs_ate_ht"], rel=1e-8)
    # sampling weights of one must not change anything
    w = sp.ipw(
        obs.assign(w=1.0),
        "y",
        "z",
        cov,
        estimand="ATT",
        normalize=False,
        weights="w",
        n_bootstrap=5,
        seed=0,
    ).estimate
    assert w == pytest.approx(ref["obs_att_ht"], rel=1e-6)


def test_horvitz_thompson_att_recovers_a_known_effect():
    # Effect on the treated is 2 by construction; the outcome level is 50, so
    # an estimate scaled by the treated share (the old behaviour) lands near
    # 2 * P(T = 1) and one that mis-scales the control term near 50.
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.normal(size=n)
    t = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x)))
    y = 50 + 2.0 * t + x + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "t": t, "x": x})
    r = sp.ipw(df, "y", "t", ["x"], estimand="ATT", normalize=False, n_bootstrap=5)
    assert abs(r.estimate - 2.0) < 0.5


def test_matching_matches_r_matching_with_both_variances(ref):
    obs = _csv("obs")
    est, se_iid, se_nn2 = ref["obs_match"]
    kw = dict(
        method="nnmatch",
        estimand="ATT",
        bias_adjust=True,
        metric="ivariance",
    )
    r = sp.match(obs, "y", "z", ["x1", "x2", "x3"], vce="iid", **kw)
    assert r.estimate == pytest.approx(est, rel=1e-9)
    assert r.se == pytest.approx(se_iid, rel=1e-9)
    r = sp.match(obs, "y", "z", ["x1", "x2", "x3"], vce="robust", vce_nn=2, **kw)
    assert r.se == pytest.approx(se_nn2, rel=1e-9)


def test_randomization_statistics_match_r(ref):
    rct = _csv("rct")
    n, n1 = len(rct), int(rct["z"].sum())
    welch = sp.ri_test(rct, "y", "z", stat="t", n_perms=50, seed=0)
    assert welch["observed"] == pytest.approx(ref["rct_welch_t"], rel=1e-10)
    rank = sp.ri_test(rct, "y", "z", stat="rank_sum", n_perms=50, seed=0)
    # R's W is the rank sum minus n1 (n1 + 1) / 2; ours is centred at the null
    w = rank["observed"] + n1 * (n + 1) / 2 - n1 * (n1 + 1) / 2
    assert w == pytest.approx(ref["rct_wilcox_W"], rel=1e-12)
    est, se = ref["rct_lin"]
    lin = sp.ri_test(rct, "y", "z", stat="lin", covariates=["x1", "x2"], n_perms=50)
    lin_t = sp.ri_test(rct, "y", "z", stat="lin_t", covariates=["x1", "x2"], n_perms=50)
    assert lin["observed"] == pytest.approx(est, rel=1e-10)
    assert lin_t["observed"] == pytest.approx(est / se, rel=1e-9)
    fit = sp.lm_lin(rct, "y", "z", ["x1", "x2"])
    assert fit.estimate / fit.se == pytest.approx(lin_t["observed"], rel=1e-9)
    assert sp.fisher_exact(
        rct, "y", "z", statistic="t", n_perm=50, seed=0
    ).statistic == pytest.approx(ref["rct_welch_t"], rel=1e-10)
