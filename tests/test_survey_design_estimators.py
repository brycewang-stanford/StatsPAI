"""``design=`` on the estimators: identities with the existing variances and
typed refusals. Numbers against R survey are in
``tests/reference_parity/test_survey_causal_parity.py`` /
``test_survey_did_parity.py``.

Identities used
---------------
* A design with clusters only is the cluster-robust variance times
  ``G / (G - 1)`` (one stratum: the PSU totals are centred at their mean,
  which is zero for an estimating equation solved exactly, and scaled by
  ``n_h / (n_h - 1)``). Likewise a weights-only design is the robust
  variance times ``n / (n - 1)``.
* An fpc with sampling fraction ``f`` in every stratum scales the variance
  by ``1 - f``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    G, m = 40, 12
    g = np.repeat(np.arange(G), m)
    u = rng.normal(size=G)[g]
    x1 = rng.normal(size=G * m) + 0.5 * u
    x2 = rng.normal(size=G * m)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2))))
    y = 1 + 2 * d + x1 + 0.5 * x2 + u + rng.normal(size=G * m)
    return pd.DataFrame(
        {
            "g": g,
            "stratum": g % 4,
            "x1": x1,
            "x2": x2,
            "d": d,
            "y": y,
            "w": rng.uniform(1, 3, G * m),
            "f": 0.25,
        }
    )


def test_ipw_cluster_design_is_cluster_sandwich_times_G_over_G_minus_1(data):
    G = data["g"].nunique()
    base = sp.ipw(
        data,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        weights="w",
        cluster="g",
        se_method="sandwich",
    )
    des = sp.svydesign(data, weights="w", cluster="g")
    r = sp.ipw(data, y="y", treat="d", covariates=["x1", "x2"], design=des)
    assert r.estimate == pytest.approx(base.estimate, rel=1e-12)
    assert r.se == pytest.approx(base.se * np.sqrt(G / (G - 1)), rel=1e-10)


def test_aipw_weights_only_design_is_robust_times_n_over_n_minus_1(data):
    n = len(data)
    base = sp.aipw(
        data,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        weights="w",
        cross_fit=False,
        trim=0,
    )
    des = sp.svydesign(data, weights="w")
    r = sp.aipw(
        data,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        cross_fit=False,
        trim=0,
        design=des,
    )
    assert r.estimate == pytest.approx(base.estimate, rel=1e-12)
    assert r.se == pytest.approx(base.se * np.sqrt(n / (n - 1)), rel=1e-10)


def test_fpc_scales_the_variance_by_one_minus_f(data):
    des = sp.svydesign(data, weights="w", strata="stratum", cluster="g")
    des_f = sp.svydesign(data, weights="w", strata="stratum", cluster="g", fpc="f")
    a = sp.regress("y ~ d + x1", data, design=des)
    b = sp.regress("y ~ d + x1", data, design=des_f)
    np.testing.assert_allclose(
        b.std_errors.values, a.std_errors.values * np.sqrt(0.75), rtol=1e-12
    )


def test_tmle_design_detail_uses_design_df(data):
    from sklearn.linear_model import LinearRegression, LogisticRegression

    des = sp.svydesign(data, weights="w", strata="stratum", cluster="g")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.tmle(
            data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            outcome_library=[LinearRegression()],
            propensity_library=[LogisticRegression()],
            design=des,
        )
    df = r.model_info["survey_design"]["df"]
    assert df == data["g"].nunique() - 4
    from scipy import stats

    crit = stats.t.ppf(0.975, df)
    assert r.ci[1] - r.estimate == pytest.approx(crit * r.se, rel=1e-12)


def test_rows_dropped_for_missing_values_keep_their_psu(data):
    """A missing outcome drops the row, not its PSU (domain estimation)."""
    d = data.copy()
    d.loc[d.index[:5], "y"] = np.nan  # first rows of PSU 0
    des = sp.svydesign(d, weights="w", strata="stratum", cluster="g")
    r = sp.ipw(d, y="y", treat="d", covariates=["x1", "x2"], design=des)
    assert r.n_obs == len(d) - 5
    assert r.model_info["survey_design"]["df"] == d["g"].nunique() - 4


@pytest.mark.parametrize(
    "call",
    [
        lambda d, des: sp.ipw(
            d, y="y", treat="d", covariates=["x1"], design=des, weights="w"
        ),
        lambda d, des: sp.aipw(
            d, y="y", treat="d", covariates=["x1"], design=des, cluster="g"
        ),
        lambda d, des: sp.regress("y ~ d", d, design=des, robust="hc1"),
        lambda d, des: sp.logit("d ~ x1", d, design=des, cluster="g"),
        lambda d, des: sp.dml(
            d, y="y", treat="d", covariates=["x1"], design=des, sample_weight="w"
        ),
        lambda d, des: sp.tmle(
            d, y="y", treat="d", covariates=["x1"], design=des, se_method="bootstrap"
        ),
        lambda d, des: sp.dml(
            d, y="y", treat=["d", "x2"], covariates=["x1"], design=des
        ),
        lambda d, des: sp.kaplan_meier(
            d, duration="y", event="d", design=des, se_method="ij", conf_type="log"
        ),
    ],
)
def test_design_conflicts_are_typed_refusals(data, call):
    des = sp.svydesign(data, weights="w", cluster="g")
    with pytest.raises(MethodIncompatibility):
        call(data, des)


def test_design_on_other_rows_is_refused(data):
    des = sp.svydesign(data, weights="w", cluster="g")
    with pytest.raises(MethodIncompatibility, match="rows"):
        sp.ipw(data.iloc[10:], y="y", treat="d", covariates=["x1"], design=des)
    with pytest.raises(MethodIncompatibility, match="SurveyDesign"):
        sp.ipw(data, y="y", treat="d", covariates=["x1"], design="w")


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n_u, T = 120, 4
    uid = np.repeat(np.arange(n_u), T)
    t = np.tile(np.arange(1, T + 1), n_u)
    g = np.repeat(rng.choice([0, 3], n_u), T)
    y = rng.normal(size=n_u * T) + (g > 0) * (t >= g)
    return pd.DataFrame(
        {
            "id": uid,
            "t": t,
            "g": g,
            "y": y,
            "psu": uid // 6,
            "w": np.repeat(rng.uniform(1, 2, n_u), T),
        }
    )


def test_callaway_santanna_design_refusals_and_downstream_guard():
    p = _panel()
    des = sp.svydesign(p, weights="w", cluster="psu")
    with pytest.raises(MethodIncompatibility):
        sp.callaway_santanna(p, y="y", g="g", t="t", i="id", design=des, panel=False)
    with pytest.raises(MethodIncompatibility):
        sp.callaway_santanna(p, y="y", g="g", t="t", i="id", design=des, bstrap=True)
    # weights varying within a unit: the unit is not the sampled element
    q = p.assign(w2=p["w"] + 0.01 * p["t"])
    with pytest.raises(MethodIncompatibility, match="vary within"):
        sp.callaway_santanna(
            q,
            y="y",
            g="g",
            t="t",
            i="id",
            design=sp.svydesign(q, weights="w2", cluster="psu"),
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.callaway_santanna(p, y="y", g="g", t="t", i="id", design=des)
    with pytest.raises(MethodIncompatibility, match="design"):
        sp.aggte(r, type="dynamic")


def test_callaway_santanna_cluster_design_is_cluster_se_times_G_over_G_minus_1():
    """Clusters-only design == csdid cluster() analytic SE x sqrt(G/(G-1))."""
    p = _panel()
    G = p["psu"].nunique()
    des = sp.svydesign(p, weights="w", cluster="psu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = sp.callaway_santanna(
            p, y="y", g="g", t="t", i="id", weights="w", clustervars=["id", "psu"]
        )
        r = sp.callaway_santanna(p, y="y", g="g", t="t", i="id", design=des)
    np.testing.assert_allclose(r.detail["att"], base.detail["att"], rtol=1e-12)
    np.testing.assert_allclose(
        r.detail["se"], base.detail["se"] * np.sqrt(G / (G - 1)), rtol=1e-10
    )
