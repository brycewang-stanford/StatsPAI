"""Joint Wald tests against Stata ``test`` / ``testparm``, by variance option.

A coefficient and its standard error can each match a reference while a
joint test built from them does not: the test uses the off-diagonal of
the covariance and its own reference distribution. Before this file the
only joint-test reference in the evidence inventory was ``sp.regress``
under three variances.

``_fixtures/_generate_joint_test_stata.do`` (Stata 18 MP, double
precision, on the translation-holdout data) writes the statistic, both
degrees of freedom and the p-value of 33 tests. ``sp.test`` is held to
each of them:

=====================  ==========================================  ==========
fit                    Stata                                       observed
=====================  ==========================================  ==========
``sp.regress``         ``regress`` + ``testparm`` / ``test``,      2e-15
                       classical / robust / hc2 / hc3 / cluster
``sp.ivreg``           ``ivregress 2sls, small`` + ``test``,       2e-15
                       classical / robust / cluster
``sp.logit``           ``logit`` + ``testparm`` (chi2)             1e-11
``sp.poisson``         ``poisson`` + ``test`` (chi2)               1e-10
``sp.panel`` (FE)      ``xtreg, fe`` + ``testparm``                2e-15
=====================  ==========================================  ==========

Two conventions the numbers depend on, each asserted below:

* ``sp.ivreg`` reports the degrees-of-freedom-adjusted variance and an F
  statistic, which is Stata's ``ivregress, small``. Stata's default
  ``ivregress`` reports a chi-squared with no adjustment; the two are
  related by ``chi2 = q * F * n / (n - k)`` for the classical variance.
* ``sp.panel(cluster=)`` reproduces ``xtreg, fe vce(cluster)`` with
  ``ssc='stata'`` (or ``'fixest'``): F on ``(q, G - 1)``. With the default
  small-sample convention the statistic is 1.7% larger and is referred to
  ``F(q, N - K)``, giving p = 0.0017 where Stata gives 0.0040 on 60
  clusters. That is a documented difference of convention, recorded here
  so it is not mistaken for agreement.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

import statspai as sp

_TESTS = pathlib.Path(__file__).resolve().parents[1]
_GOLD = json.loads(
    (pathlib.Path(__file__).parent / "_fixtures" / "joint_test_stata.json").read_text(
        encoding="utf-8"
    )
)
_HOLDOUT = _TESTS / "stata_translation_holdout"

RTOL_EXACT = 1e-10  # closed-form functionals of (b, V); observed 2e-15
RTOL_ML = 1e-8  # logit / poisson at Stata's default convergence; observed 1e-10


@pytest.fixture(scope="module")
def cross():
    return pd.read_csv(_HOLDOUT / "holdout_cross.csv")


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(_HOLDOUT / "holdout_panel.csv")


# Few-cluster and separation notices are not what is under test.
pytestmark = pytest.mark.filterwarnings("ignore")


def _check(key: str, wald: dict, rtol: float) -> None:
    gold = _GOLD[key]
    assert wald["distribution"].lower().startswith(gold["kind"].lower()[0]), key
    assert wald["statistic"] == pytest.approx(gold["stat"], rel=rtol), key
    assert wald["pvalue"] == pytest.approx(gold["p"], rel=max(rtol, 1e-8)), key
    q, denom = wald["df"]
    assert q == gold["df"], key
    if gold["df_r"] > 0:
        assert denom == gold["df_r"], key
    else:
        assert denom is None, key


def test_fixture_was_generated_in_double_precision():
    assert _GOLD["_meta"]["precision"] == "double"
    assert len([k for k in _GOLD if k != "_meta"]) == 33


_REG_VCE = {
    "ols": {},
    "robust": {"robust": "hc1"},
    "hc2": {"robust": "hc2"},
    "hc3": {"robust": "hc3"},
    "cluster": {"cluster": "g"},
}
_REG_TESTS = {
    "factor": "C(k)[T.2] = C(k)[T.3] = 0",
    "equal": "x1 = x2",
    "both": "x1 = x2 = 0",
}


@pytest.mark.parametrize("vce", sorted(_REG_VCE))
@pytest.mark.parametrize("which", sorted(_REG_TESTS))
def test_regress_joint_test_matches_stata(cross, vce, which):
    fit = sp.regress("y ~ x1 + x2 + C(k)", data=cross, **_REG_VCE[vce])
    _check(f"regress_{vce}_{which}", sp.test(fit, _REG_TESTS[which]), RTOL_EXACT)


_IV_VCE = {"ols": {}, "robust": {"robust": "hc1"}, "cluster": {"cluster": "g"}}


@pytest.mark.parametrize("vce", sorted(_IV_VCE))
@pytest.mark.parametrize("which", ["both", "equal"])
def test_ivreg_joint_test_matches_stata_small(cross, vce, which):
    fit = sp.ivreg("y ~ x1 + (x2 ~ z + z2)", data=cross, **_IV_VCE[vce])
    hypothesis = "x1 = x2 = 0" if which == "both" else "x1 = x2"
    _check(f"iv_{vce}_small_{which}", sp.test(fit, hypothesis), RTOL_EXACT)


def test_iv_and_ivreg_give_the_same_joint_test(cross):
    a = sp.ivreg("y ~ x1 + (x2 ~ z + z2)", data=cross)
    b = sp.iv("y ~ x1 + (x2 ~ z + z2)", data=cross)
    assert (
        sp.test(a, "x1 = x2 = 0")["statistic"] == sp.test(b, "x1 = x2 = 0")["statistic"]
    )


def test_ivreg_is_stata_s_small_convention_not_its_default(cross):
    """Default ``ivregress`` is a chi2 without the df adjustment."""
    fit = sp.ivreg("y ~ x1 + (x2 ~ z + z2)", data=cross)
    wald = sp.test(fit, "x1 = x2 = 0")
    default = _GOLD["iv_ols_both"]
    assert default["kind"] == "chi2" and wald["distribution"] == "F"
    n, k = len(cross), 3
    q = wald["df"][0]
    assert default["stat"] == pytest.approx(
        q * wald["statistic"] * n / (n - k), rel=1e-10
    )
    # Read as a match to the default it would be off by half.
    assert abs(wald["statistic"] - default["stat"]) / default["stat"] > 0.4


@pytest.mark.parametrize("vce,kwargs", [("ols", {}), ("robust", {"robust": "robust"})])
def test_logit_and_poisson_joint_tests_match_stata(cross, vce, kwargs):
    logit = sp.logit("yb ~ x1 + x2 + C(k)", data=cross, **kwargs)
    _check(f"logit_{vce}_factor", sp.test(logit, "C(k)[T.2] = C(k)[T.3] = 0"), RTOL_ML)
    poisson = sp.poisson("cnt ~ x1 + x2 + C(k)", data=cross, **kwargs)
    _check(f"poisson_{vce}_both", sp.test(poisson, "x1 = x2 = 0"), RTOL_ML)


def _time_effects(fit) -> str:
    names = [n for n in fit.params.index if n not in ("x", "const", "Intercept")]
    assert len(names) == 5
    return " = ".join(names) + " = 0"


@pytest.mark.parametrize("ssc", [None, "stata", "fixest"])
def test_panel_fe_joint_test_matches_xtreg(panel, ssc):
    extra = {} if ssc is None else {"ssc": ssc}
    fit = sp.panel(panel, "y ~ x + C(t)", entity="id", time="t", method="fe", **extra)
    _check("xtfe_ols_time", sp.test(fit, _time_effects(fit)), RTOL_EXACT)


@pytest.mark.parametrize("ssc", ["stata", "fixest"])
def test_panel_fe_clustered_joint_test_matches_xtreg(panel, ssc):
    fit = sp.panel(
        panel,
        "y ~ x + C(t)",
        entity="id",
        time="t",
        method="fe",
        cluster="id",
        ssc=ssc,
    )
    _check("xtfe_cluster_time", sp.test(fit, _time_effects(fit)), RTOL_EXACT)


def test_panel_default_convention_differs_from_xtreg_under_clustering(panel):
    """Not agreement: a different small-sample convention and reference law."""
    fit = sp.panel(
        panel,
        "y ~ x + C(t)",
        entity="id",
        time="t",
        method="fe",
        cluster="id",
    )
    wald = sp.test(fit, _time_effects(fit))
    gold = _GOLD["xtfe_cluster_time"]
    assert wald["df"] == (5, 294) and gold["df_r"] == 59
    assert wald["statistic"] == pytest.approx(3.967874, rel=1e-6)
    assert wald["statistic"] / gold["stat"] == pytest.approx(1.017, abs=2e-3)
    # The p-value is less than half of xtreg's on 60 clusters.
    assert wald["pvalue"] < 0.5 * gold["p"]
