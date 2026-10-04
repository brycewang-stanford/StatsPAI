"""Cross-language parity: ``sp.survreg`` against Stata 18 ``streg ..., time``.

Fixture: ``_fixtures/_generate_streg_stata.do`` (official ``streg``). The
durations are generated in Stata, a Weibull with gamma heterogeneity, and
exported at ``%21.16e``, so both sides read the same bytes.

Every block is held to 1e-6. Observed: coefficients 3e-10 or better,
standard errors 3e-10 or better, log-likelihoods 5e-12, for the four
distributions under ``vce(oim)``, ``vce(robust)`` and ``vce(cluster)`` and
for the Weibull with gamma frailty. As in the ``ivprobit`` fixture the
do-file tightens ``ml``'s stopping rule, so the comparison is between two
optima and not between two stopping points.

This file exists because of a silent defect. ``sp.survreg`` accepted
``robust=`` and ``cluster=``, wrote them into ``model_info`` and returned
the observed-information standard errors whatever was asked. The test
asserts each option changes the standard errors and lands on Stata's.

Conventions that differ and are mapped here, not tolerated:

* order: Stata lists the regressors, then ``_cons``, then the ancillary
  parameter; ``sp.survreg`` lists ``_cons`` first;
* Weibull: Stata reports ``ln_p``, which is minus ``log(sigma)``;
* log-likelihood: Stata's is the likelihood of ``log(t)``,
  ``model_info['ll_log_time']``; the R convention (density of ``t``) stays
  in ``diagnostics['Log-likelihood']``.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
DISTS = ["weibull", "exponential", "lognormal", "loglogistic"]
BLOCKS = [f"{d}{s}" for d in DISTS for s in ("", "_robust", "_cluster")] + [
    "weibull_frailty",
    "weibull_frailty_robust",
]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "streg_data.csv")


@pytest.fixture(scope="module")
def stata() -> dict:
    return json.loads((_FIX / "streg_stata.json").read_text(encoding="utf-8"))


def _fit(block: str, data: pd.DataFrame):
    parts = block.split("_")
    kw = {}
    if "robust" in parts:
        kw["robust"] = "robust"
    if "cluster" in parts:
        kw["cluster"] = "clust"
    if "frailty" in parts:
        kw["frailty"] = "gamma"
    return sp.survreg(
        data=data, duration="time", event="event", x=["x1", "x2"], dist=parts[0], **kw
    )


def _in_stata_order(res, dist: str, frailty: bool):
    b, se = res.params, res.std_errors
    names = ["x1", "x2", "_cons"]
    est, err = [b[n] for n in names], [se[n] for n in names]
    if dist != "exponential":
        sign = -1.0 if dist == "weibull" else 1.0  # ln_p = -log(sigma)
        est.append(sign * b["log(sigma)"])
        err.append(se["log(sigma)"])
    if frailty:
        est.append(b["log(theta)"])
        err.append(se["log(theta)"])
    return np.array(est), np.array(err)


@pytest.mark.parametrize("block", BLOCKS)
def test_matches_stata(block, data, stata):
    res, ref = _fit(block, data), stata[block]
    est, err = _in_stata_order(res, block.split("_")[0], "frailty" in block)
    np.testing.assert_allclose(est, ref["b"], rtol=1e-6)
    np.testing.assert_allclose(err, ref["se"], rtol=1e-6)
    assert res.model_info["ll_log_time"] == pytest.approx(ref["ll"], abs=1e-8)
    assert res.model_info["converged"]


@pytest.mark.parametrize("dist", DISTS)
def test_robust_and_cluster_are_used(dist, data):
    plain = _fit(dist, data)
    for suffix in ("_robust", "_cluster"):
        other = _fit(dist + suffix, data)
        np.testing.assert_allclose(other.params.values, plain.params.values)
        assert not np.allclose(other.std_errors.values, plain.std_errors.values)
    assert _fit(dist + "_cluster", data).model_info["n_clusters"] == 75
    assert _fit(dist + "_robust", data).model_info["vce"] == "robust"


def test_frailty_variance_and_its_test(data, stata):
    info, ref = _fit("weibull_frailty", data).model_info, stata["weibull_frailty"]
    assert info["theta"] == pytest.approx(ref["theta"], rel=1e-6)
    assert info["lr_theta_chi2"] == pytest.approx(ref["chi2_c"], rel=1e-6)
    assert info["lr_theta_pvalue"] == pytest.approx(ref["p_c"], rel=1e-5)
    assert not info["theta_at_boundary"]


def test_ignoring_frailty_biases_duration_dependence(data):
    """The do-file's DGP: sigma = 0.7 with gamma heterogeneity of variance 0.6.

    Without frailty the Weibull reads the heterogeneity as a hazard that
    falls faster over time (a larger sigma).
    """
    plain = _fit("weibull", data)
    frail = _fit("weibull_frailty", data)
    s_plain = np.exp(plain.params["log(sigma)"])
    s_frail, se = np.exp(frail.params["log(sigma)"]), frail.std_errors["log(sigma)"]
    assert abs(np.log(s_frail / 0.7)) < 2.5 * se
    assert s_plain > 0.85
    assert abs(frail.model_info["theta"] - 0.6) < 2.5 * frail.model_info["theta_se"]


@pytest.mark.parametrize("dist", ["lognormal", "loglogistic"])
def test_frailty_at_the_boundary(dist, data, stata):
    """On these data theta goes to zero for the two other distributions.

    Stata stops at theta ~ 3e-8 with a standard error of 600 on its log.
    ``sp.survreg`` returns the model without frailty, says so, and reports
    the same likelihood-ratio statistic (0) and p-value (1).
    """
    ref = stata[f"{dist}_frailty_boundary"]
    with pytest.warns(sp.exceptions.ConvergenceWarning, match="boundary"):
        res = sp.survreg(
            data=data,
            duration="time",
            event="event",
            x=["x1", "x2"],
            dist=dist,
            frailty="gamma",
        )
    info = res.model_info
    assert info["theta"] == 0.0 and info["theta_at_boundary"]
    assert info["lr_theta_chi2"] == pytest.approx(ref["chi2_c"], abs=1e-6)
    assert info["lr_theta_pvalue"] == pytest.approx(ref["p_c"])
    assert info["ll_log_time"] == pytest.approx(ref["ll"], abs=1e-5)
    assert info["ll_log_time"] >= ref["ll"] - 1e-9
    plain = _fit(dist, data)
    np.testing.assert_allclose(res.params.values[:4], plain.params.values)
    assert np.isnan(res.std_errors["log(theta)"])
    assert np.all(np.isfinite(res.std_errors.values[:4]))


def test_refusals(data):
    bad = sp.exceptions.MethodIncompatibility
    with pytest.raises(bad, match="zero or negative"):
        sp.survreg(
            data=data.assign(time=data["time"] - 1.0),
            duration="time",
            event="event",
            x=["x1"],
        )
    with pytest.raises(bad, match="frailty"):
        sp.survreg(
            data=data, duration="time", event="event", x=["x1"], frailty="lognormal"
        )
    with pytest.raises(bad):
        sp.survreg(data=data, duration="time", event="event", x=["x1"], robust="hc3")


def test_per_observation_loglik_feeds_vuong(data):
    """Weibull against log-logistic: non-nested, same observations."""
    out = sp.vuong(_fit("loglogistic", data), _fit("weibull", data))
    assert out["k1"] == out["k2"] == 4
    assert out["loglik1"] > out["loglik2"]
