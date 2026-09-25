"""sp.pretrends_test must use the joint pre-period covariance by default.

Before 1.31, ``sp.event_study`` withheld ``model_info['vcv_pre']`` unless
called with ``expose_pre_vcov=True``, so ``sp.pretrends_test(es)`` treated
the pre-treatment coefficients as independent. On the castle-doctrine panel
that returned p = 0.603 while the event study's own ``pretrend_test`` and
Stata (``reghdfe ..., vce(cluster sid)`` + ``test``) both give
F(4, 49) = 1.27, p = 0.293. A third-party Stata/StatsPAI walkthrough
reported the gap.
"""

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def castle():
    df = sp.datasets.castle_doctrine()
    df["_g"] = df["effyear"]
    return df


@pytest.fixture(scope="module")
def castle_es(castle):
    return sp.event_study(
        castle,
        y="l_homicide",
        treat_time="_g",
        time="year",
        unit="sid",
        window=(-5, 5),
        ref_period=-1,
        cluster="sid",
    )


def test_default_f_test_reproduces_stata_and_own_pretrend_test(castle_es):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no diagonal fallback
        out = sp.pretrends_test(castle_es, type="f")
    own = castle_es.model_info["pretrend_test"]
    assert out["stat_label"] == "F(4, 49)"
    assert out["statistic"] == pytest.approx(own["statistic"], abs=1e-4)
    assert out["pvalue"] == pytest.approx(own["pvalue"], abs=1e-4)
    # Stata 18 MP, run 2026-09-25 on the same castle panel (endpoints
    # binned: lead5 = rel <= -5, lag5 = rel >= 5):
    #   reghdfe l_homicide lead5 lead4 lead3 lead2 lag0-lag5,
    #       absorb(sid year) vce(cluster sid)
    #   test lead5 lead4 lead3 lead2
    # r(F) = 1.2746247780, r(df_r) = 49, r(p) = .29272676.
    # Residual ~3e-9 relative: the CSV stores l_homicide at float precision.
    assert out["statistic"] == pytest.approx(1.2746247780, rel=1e-6)
    assert out["pvalue"] == pytest.approx(0.29272676, rel=1e-6)


def test_default_wald_uses_joint_covariance(castle_es):
    out = sp.pretrends_test(castle_es)
    # chi2(4) on the same quadratic form: 4 * 1.2746 = 5.098.
    assert out["statistic"] == pytest.approx(4 * 1.2746247822, rel=1e-6)
    assert out["pvalue"] == pytest.approx(0.2773, abs=5e-4)


def test_opt_out_restores_diagonal_and_warns(castle):
    es = sp.event_study(
        castle,
        y="l_homicide",
        treat_time="_g",
        time="year",
        unit="sid",
        window=(-5, 5),
        ref_period=-1,
        cluster="sid",
        expose_pre_vcov=False,
    )
    assert es.model_info["vcv_pre"] is None
    with pytest.warns(UserWarning, match="MUTUALLY INDEPENDENT"):
        out = sp.pretrends_test(es)
    # The pre-1.31 number the walkthrough reported.
    assert out["pvalue"] == pytest.approx(0.6035, abs=5e-4)


def test_other_estimators_fall_back_to_event_study_vcov(castle):
    """Estimators that do not write ``vcv_pre`` use sp.event_study_vcov."""
    sa = sp.sun_abraham(castle, y="l_homicide", g="effyear", t="year", i="sid")
    evc = sp.event_study_vcov(sa, allow_diagonal=False)
    pre = evc.times < 0
    b, V = evc.beta[pre], evc.vcov[np.ix_(pre, pre)]
    expected = float(b @ np.linalg.solve(V, b))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = sp.pretrends_test(sa)
    assert out["df"] == int(pre.sum())
    assert out["statistic"] == pytest.approx(expected, rel=1e-10)


def test_fallback_rejects_covariance_that_disagrees_with_table(castle_es):
    """A joint covariance whose diagonal is not the table's SEs is not used."""
    r = copy.deepcopy(castle_es)
    r.model_info.pop("vcv_pre")
    es = r.model_info["event_study"].copy()
    es["se"] = es["se"] * 1.5
    r.model_info["event_study"] = es
    r.detail = es
    with pytest.warns(UserWarning, match="MUTUALLY INDEPENDENT"):
        sp.pretrends_test(r)


def test_f_denominator_without_clusters_is_residual_df():
    rows = pd.DataFrame(
        {
            "relative_time": [-3, -2, -1, 0, 1],
            "att": [0.1, -0.05, 0.0, 0.3, 0.4],
            "se": [0.1, 0.1, 0.0, 0.1, 0.1],
        }
    )
    res = sp.CausalResult(
        method="toy",
        estimand="ATT",
        estimate=0.35,
        se=0.1,
        pvalue=0.01,
        ci=(0.15, 0.55),
        alpha=0.05,
        n_obs=200,
        detail=rows,
        model_info={"event_study": rows, "vcv_pre": np.diag([0.01, 0.01, 0.0])},
    )
    out = sp.pretrends_test(res, type="f")
    assert out["stat_label"] == "F(2, 198)"
