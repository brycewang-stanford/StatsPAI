"""Reference parity: ``sp.stteffects`` vs Stata 18 ``stteffects``, and
``sp.rmst`` / IPTW-adjusted ``sp.kaplan_meier`` vs R ``survRM2``,
``survival`` and ``adjustedCurves``.

Frozen references, from committed CSVs:

* ``_fixtures/stteffects_Stata.csv`` by ``_generate_stteffects_stata.do``
  (12 configurations on ``_fixtures/stteffects_data.csv``);
* ``_fixtures/survival_rmst_R.json`` by ``_generate_survival_rmst_R.R``
  (survRM2 1.0.4, survival 3.8.3, adjustedCurves 0.12.0, on
  ``_fixtures/survival_weighted_data.csv``).

Tolerances (observed worst gaps in parentheses):

* stteffects point estimates 1e-9 relative (~2e-15); standard errors and
  the effect / POmean covariance 1e-7 relative (~1e-10): our Jacobian of
  the stacked moments is taken by central differences, Stata's
  analytically, which is the whole gap.
* RMST, its SEs, the survRM2 contrasts and the Xie-Liu curve SEs 1e-12
  relative (~3e-15).

One documented departure: for ``method='ipwra', estimand='ATET'`` Stata
averages the treated-weighted outcome models' predictions over the *whole*
sample (its equations 15e-15f carry no treatment indicator), which under a
correctly specified outcome model estimates the ATE, not the ATET.
``sp.stteffects`` averages over the treated, as its RA estimator and
Stata's own RA / IPW ATET do. The tests show (i) that the outcome models
are Stata's (averaging them over all rows reproduces Stata's number to
1e-9) and (ii) on a DGP with heterogeneous effects, that our estimate
recovers the sample ATET while Stata's average lands on the ATE.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.survival import _param_surv as ps

_FIX = pathlib.Path(__file__).parent / "_fixtures"
XC = ["x1", "x2"]

CASES = {
    "ra_ate_weibull": dict(method="ra", covariates=XC),
    "ra_atet_weibull": dict(method="ra", covariates=XC, estimand="ATET"),
    "ra_ate_exponential": dict(method="ra", covariates=XC, dist="exponential"),
    "ra_ate_lognormal": dict(method="ra", covariates=XC, dist="lognormal"),
    "wra_ate_weibull": dict(method="wra", covariates=XC, censor_covariates=["x1"]),
    "ipw_ate_weibull": dict(
        method="ipw", treat_covariates=XC, censor_covariates=["x1"]
    ),
    "ipw_atet_weibull": dict(
        method="ipw", treat_covariates=XC, censor_covariates=["x1"], estimand="ATET"
    ),
    "ipw_ate_probit_lognormal": dict(
        method="ipw",
        treat_covariates=XC,
        censor_covariates=["x1"],
        treat_model="probit",
        censor_dist="lognormal",
    ),
    "ipwra_lac_ate_weibull": dict(method="ipwra", covariates=XC),
    "ipwra_wac_ate_weibull": dict(
        method="ipwra", covariates=XC, censor_covariates=["x1"]
    ),
}


@pytest.fixture(scope="module")
def te_data():
    return pd.read_csv(_FIX / "stteffects_data.csv")


@pytest.fixture(scope="module")
def te_ref():
    return pd.read_csv(_FIX / "stteffects_Stata.csv").set_index("tag")


@pytest.mark.parametrize("tag", list(CASES))
def test_stteffects_matches_stata(tag, te_data, te_ref):
    r = sp.stteffects(te_data, "time", "event", "a", **CASES[tag])
    ref = te_ref.loc[tag]
    det = r.detail.set_index("term")
    np.testing.assert_allclose(r.estimate, ref["b1"], rtol=1e-9)
    np.testing.assert_allclose(det.loc["POmean0", "estimate"], ref["b2"], rtol=1e-9)
    np.testing.assert_allclose(r.se, ref["se1"], rtol=1e-7)
    np.testing.assert_allclose(det.loc["POmean0", "se"], ref["se2"], rtol=1e-7)
    np.testing.assert_allclose(
        r.model_info["vcov"].values[0, 1], ref["cov12"], rtol=1e-7
    )
    assert r.n_obs == int(ref["N"])


@pytest.mark.parametrize("cens", [None, ["x1"]])
def test_ipwra_atet_models_are_statas_and_only_the_average_differs(
    cens, te_data, te_ref
):
    tag = "ipwra_wac_atet_weibull" if cens else "ipwra_lac_atet_weibull"
    r = sp.stteffects(
        te_data,
        "time",
        "event",
        "a",
        covariates=XC,
        method="ipwra",
        estimand="ATET",
        censor_covariates=cens,
    )
    X = np.column_stack([te_data["x1"], te_data["x2"], np.ones(len(te_data))])
    c = r.model_info["coefficients"]
    m0 = ps.mean_time(np.array(list(c["outcome0"].values())), X, "weibull")
    m1 = ps.mean_time(np.array(list(c["outcome1"].values())), X, "weibull")
    ref = te_ref.loc[tag]
    # Stata's number: the same two models averaged over every row
    np.testing.assert_allclose(m0.mean(), ref["b2"], rtol=1e-9)
    np.testing.assert_allclose(m1.mean() - m0.mean(), ref["b1"], rtol=1e-9)
    # ours: averaged over the treated
    tr = te_data["a"].to_numpy() == 1
    np.testing.assert_allclose(r.estimate, (m1 - m0)[tr].mean(), rtol=1e-12)


def test_ipwra_atet_recovers_the_atet_on_a_known_dgp():
    """Heterogeneous effects, treatment selected on x1: the sample ATET and
    ATE differ by ~9 standard errors. Our IPWRA ATET is within 3 SE of the
    ATET; the all-rows average (Stata's) is within 3 SE of the ATE."""
    rng = np.random.default_rng(7)
    n = 4000
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.5, n)
    a = rng.binomial(1, 1 / (1 + np.exp(-1.2 * x1)))
    X = np.column_stack([x1, x2, np.ones(n)])
    b0, b1, s = np.array([0.0, -0.2, 1.0]), np.array([0.6, -0.2, 1.5]), 1.5
    m0 = np.exp(X @ b0 + gammaln(1 + 1 / s))
    m1 = np.exp(X @ b1 + gammaln(1 + 1 / s))
    att, ate = (m1 - m0)[a == 1].mean(), (m1 - m0).mean()
    T = np.exp(np.where(a == 1, X @ b1, X @ b0)) * (-np.log(rng.uniform(size=n))) ** (
        1 / s
    )
    C = np.exp(3.0) * (-np.log(rng.uniform(size=n))) ** (1 / 1.5)
    df = pd.DataFrame(
        {"t": np.minimum(T, C), "d": (T <= C).astype(int), "a": a, "x1": x1, "x2": x2}
    )
    r = sp.stteffects(
        df, "t", "d", "a", covariates=["x1", "x2"], method="ipwra", estimand="ATET"
    )
    assert abs(r.estimate - att) < 3 * r.se
    c = r.model_info["coefficients"]
    all_rows = (
        ps.mean_time(np.array(list(c["outcome1"].values())), X, "weibull")
        - ps.mean_time(np.array(list(c["outcome0"].values())), X, "weibull")
    ).mean()
    assert abs(all_rows - ate) < 3 * r.se
    assert abs(all_rows - att) > 5 * r.se


def test_stteffects_inputs_fail_loudly(te_data):
    with pytest.raises(MethodIncompatibility, match="censoring model"):
        sp.stteffects(te_data, "time", "event", "a", treat_covariates=XC, method="ipw")
    with pytest.raises(MethodIncompatibility, match="no censoring model"):
        sp.stteffects(
            te_data, "time", "event", "a", covariates=XC, censor_covariates=["x1"]
        )
    with pytest.raises(MethodIncompatibility, match="binary"):
        sp.stteffects(
            te_data.assign(a=te_data["a"] * 2 + 1), "time", "event", "a", covariates=XC
        )
    with pytest.raises(MethodIncompatibility):
        sp.stteffects(te_data, "time", "event", "a", covariates=XC, dist="gamma")


def test_uncensored_ipw_is_the_normalised_ipw_mean():
    """Without censoring, stteffects ipw is the Hajek IPW mean of the time
    (Stata's eq. 12a-12b); check the identity directly."""
    rng = np.random.default_rng(11)
    n = 500
    x = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-x)))
    t = np.exp(1 + 0.3 * a + 0.2 * x) * rng.weibull(1.3, n)
    df = pd.DataFrame({"t": t, "d": 1, "a": a, "x": x})
    r = sp.stteffects(df, "t", "d", "a", treat_covariates=["x"], method="ipw")
    Z = np.column_stack([x, np.ones(n)])
    al = ps.fit_binary(Z, a.astype(float), "logit")
    pr = 1 / (1 + np.exp(-(Z @ al)))
    w1, w0 = a / pr, (1 - a) / (1 - pr)
    hajek = (w1 * t).sum() / w1.sum() - (w0 * t).sum() / w0.sum()
    np.testing.assert_allclose(r.estimate, hajek, rtol=1e-12)


# -- RMST ------------------------------------------------------------------


@pytest.fixture(scope="module")
def wt_data():
    return pd.read_csv(_FIX / "survival_weighted_data.csv")


@pytest.fixture(scope="module")
def rmst_ref():
    with open(_FIX / "survival_rmst_R.json", encoding="utf-8") as fh:
        return json.load(fh)


def test_rmst_matches_survrm2(wt_data, rmst_ref):
    r = sp.rmst(wt_data, "time", "event", group="a", tau=rmst_ref["tau"])
    assert r.se_method == "greenwood"
    t = r.table.set_index("group")
    for arm in ("0", "1"):
        ref = rmst_ref["survrm2"][f"arm{arm}"]
        got = t.loc[arm, ["rmst", "se", "ci_lower", "ci_upper"]].to_numpy(float)
        np.testing.assert_allclose(got, ref, rtol=1e-12)
    np.testing.assert_allclose(
        r.contrasts[["estimate", "ci_lower", "ci_upper", "p_value"]].to_numpy(float),
        np.array(rmst_ref["survrm2"]["unadjusted"]),
        rtol=1e-12,
    )


def test_weighted_rmst_matches_survival_ij(wt_data, rmst_ref):
    r = sp.rmst(wt_data, "time", "event", group="a", tau=rmst_ref["tau"], weights="w")
    assert r.se_method == "ij"
    t = r.table.set_index("group")
    for arm, key in (("1", "w1"), ("0", "w0")):
        ref = rmst_ref["ij"][key]
        np.testing.assert_allclose(t.loc[arm, "rmst"], ref["rmst"], rtol=1e-12)
        np.testing.assert_allclose(t.loc[arm, "se"], ref["se"], rtol=1e-12)
    # rows are independent across the arms: the variances add
    np.testing.assert_allclose(
        r.contrasts.loc[0, "se"],
        np.hypot(rmst_ref["ij"]["w1"]["se"], rmst_ref["ij"]["w0"]["se"]),
        rtol=1e-12,
    )
    d1 = wt_data[wt_data["a"] == 1]
    rc = sp.rmst(d1, "time", "event", tau=rmst_ref["tau"], weights="w", cluster="cl")
    np.testing.assert_allclose(
        rc.table.loc[0, "se"], rmst_ref["ij"]["w1_cluster"]["se"], rtol=1e-12
    )


def test_unweighted_ij_rmst_variance_equals_greenwood(wt_data, rmst_ref):
    """For the Kaplan-Meier area the jackknife and Greenwood-type variances
    coincide (R gives the same number both ways)."""
    d1 = wt_data[wt_data["a"] == 1]
    a = sp.rmst(d1, "time", "event", tau=8, se_method="ij").table.loc[0, "se"]
    b = sp.rmst(d1, "time", "event", tau=8, se_method="greenwood").table.loc[0, "se"]
    np.testing.assert_allclose(a, rmst_ref["ij"]["u1"]["se"], rtol=1e-12)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_iptw_curves_match_adjustedcurves_xie_liu(wt_data, rmst_ref):
    km = sp.kaplan_meier(
        wt_data,
        "time",
        "event",
        group="a",
        weights="w",
        se_method="xie_liu",
        conf_type="plain",
    )
    ref = pd.DataFrame(rmst_ref["xie_liu"])
    for g in ("0", "1"):
        tab = km._tables[g].iloc[1:]
        r = ref[ref["group"] == g].set_index("time").loc[tab["time"].to_numpy()]
        np.testing.assert_allclose(tab["survival"], r["surv"], rtol=1e-12)
        np.testing.assert_allclose(tab["std_err"], r["se"], rtol=1e-12)


def test_rmst_boundaries(wt_data):
    with pytest.raises(DataInsufficient, match="beyond the last observed time"):
        sp.rmst(wt_data, "time", "event", group="a", tau=1e6)
    with pytest.raises(MethodIncompatibility):
        sp.rmst(wt_data, "time", "event", group="a", se_method="xie_liu")
    # tau defaults to the smallest of the groups' largest times (survRM2)
    r = sp.rmst(wt_data, "time", "event", group="a")
    assert r.tau == min(wt_data.groupby("a")["time"].max())
    # a constant curve S = 1 (no events before tau) has RMST tau
    late = wt_data.assign(time=wt_data["time"] + 100)
    r = sp.rmst(late, "time", "event", tau=50)
    assert r.table.loc[0, "rmst"] == 50
