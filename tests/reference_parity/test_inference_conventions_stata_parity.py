"""t / F / p conventions against Stata 18, on an external audit's data.

An audit of StatsPAI 1.40.1 against Stata 18 (2026-10-23) put the
complaint "coefficients and standard errors match, test statistics and
p-values do not" to 60 specifications on one 240-row dataset. Every
mismatch it found was one of two kinds:

* a **gap**: ``sp.iv(method='gmm')`` under the unadjusted weight matrix
  carried no finite-sample factor under ``small=True``, so its variance was
  Stata's ``wmatrix(unadjusted) ... small`` divided by N / (N - K) = 240 /
  236. Fixed; the rows ``gmm_unadj*`` below pin it.
* a **convention** that is legitimate but was invisible in the output
  (``hc1`` vs ``robust`` after ML models, the linearmodels small-sample
  scaling of ``sp.panel``, ``small=True`` vs ``ivregress``'s large-sample
  default, the between estimator counting groups). Each is matched here
  with the option that reproduces Stata, and ``result.inference_rule()``
  is asserted to state the rule the default uses.

Reference: ``_fixtures/_generate_inference_conventions_Stata.do`` on
``inference_conventions.csv`` (the auditor's DGP, seed 20261025), with the
ML fits run to convergence on the Stata side. Every comparison below is
within 4.4e-14 relative (measured 2026-10-23, worst: a p-value of the
clustered GMM row); 1e-10 leaves room for BLAS differences across platforms.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-10
IV = "y ~ x1 + x2 + (xd ~ z1 + z2)"

#: key in the reference file -> (callable on the data, Stata distribution)
FITS = {
    # wmatrix(unadjusted) is named since 1.42 made 'robust' the default
    "gmm_unadj_sandwich_small": lambda d: sp.iv(
        IV, data=d, method="gmm", wmatrix="unadjusted"
    ),
    "gmm_unadj_sandwich": lambda d: sp.iv(
        IV, data=d, method="gmm", wmatrix="unadjusted", small=False
    ),
    "gmm_unadj_small": lambda d: sp.iv(
        IV, data=d, method="gmm", wmatrix="unadjusted", gmm_vcov="efficient"
    ),
    "gmm_unadj": lambda d: sp.iv(
        IV,
        data=d,
        method="gmm",
        wmatrix="unadjusted",
        gmm_vcov="efficient",
        small=False,
    ),
    "gmm_robust_small": lambda d: sp.iv(IV, data=d, method="gmm", robust="hc1"),
    "gmm_robust": lambda d: sp.iv(IV, data=d, method="gmm", robust="hc1", small=False),
    "gmm_cluster_small": lambda d: sp.iv(IV, data=d, method="gmm", cluster="g"),
    "gmm_cluster": lambda d: sp.iv(IV, data=d, method="gmm", cluster="g", small=False),
    # wmatrix= picks the weight matrix, cluster= the variance (1.41)
    "gmm_robust_vce_cluster_small": lambda d: sp.iv(
        IV, data=d, method="gmm", wmatrix="robust", cluster="g"
    ),
    "gmm_robust_vce_cluster": lambda d: sp.iv(
        IV, data=d, method="gmm", wmatrix="robust", cluster="g", small=False
    ),
    "gmm_unadj_vce_cluster_small": lambda d: sp.iv(
        IV, data=d, method="gmm", wmatrix="unadjusted", cluster="g"
    ),
    "gmm_unadj_vce_cluster": lambda d: sp.iv(
        IV, data=d, method="gmm", wmatrix="unadjusted", cluster="g", small=False
    ),
    "tsls_small": lambda d: sp.iv(IV, data=d),
    "tsls": lambda d: sp.iv(IV, data=d, small=False),
    "liml_robust_small": lambda d: sp.iv(IV, data=d, method="liml", robust="hc1"),
    "liml_robust": lambda d: sp.iv(
        IV, data=d, method="liml", robust="robust", small=False
    ),
    "logit_robust": lambda d: sp.logit("yb ~ x1 + x2", data=d, robust="robust"),
    "probit_robust": lambda d: sp.probit("yb ~ x1 + x2", data=d, robust="robust"),
    "poisson_robust": lambda d: sp.poisson("yc ~ x1 + x2", data=d, robust="robust"),
    "xtfe_cluster": lambda d: sp.panel(
        d,
        "yp ~ x1 + x2",
        entity="g",
        time="t",
        method="fe",
        cluster="g",
        ssc="stata",
    ),
    "xtre": lambda d: sp.panel(
        d, "yp ~ x1 + x2", entity="g", time="t", method="re", ssc="stata"
    ),
    "xtbe": lambda d: sp.panel(
        d, "yp ~ x1 + x2", entity="g", time="t", method="be", ssc="stata"
    ),
}


@pytest.fixture(scope="module")
def ref():
    return json.loads(
        (_FIX / "inference_conventions_Stata.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "inference_conventions.csv")


def _fit(key, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FITS[key](data)


@pytest.mark.parametrize("key", sorted(FITS))
def test_estimates_standard_errors_and_joint_test_match_stata(ref, data, key):
    want = ref[key]
    res = _fit(key, data)
    for name in ("x1", "x2"):
        assert float(res.params[name]) == pytest.approx(want[f"b_{name}"], rel=RTOL)
        assert float(res.std_errors[name]) == pytest.approx(
            want[f"se_{name}"], rel=RTOL
        )
    joint = sp.test(res, "x1 = x2 = 0")
    assert joint["statistic"] == pytest.approx(want["joint"], rel=RTOL)
    assert joint["pvalue"] == pytest.approx(want["joint_p"], rel=RTOL)
    assert joint["df"][0] == want["df"]
    rule = res.inference_rule()
    if want["df_r"] < 0:
        assert joint["distribution"] == "chi2"
        assert rule["distribution"] == "z" and rule["joint_test"] == "chi2(q)"
    else:
        assert joint["distribution"] == "F"
        assert joint["df"][1] == want["df_r"]
        assert rule["df"] == want["df_r"]


def test_gmm_small_is_the_large_sample_variance_times_n_over_n_minus_k(ref):
    """The gap the audit found, as the identity Stata satisfies."""
    n, k = 240, 4
    for stem in ("gmm_unadj_sandwich", "gmm_unadj"):
        ratio = (ref[stem + "_small"]["se_x1"] / ref[stem]["se_x1"]) ** 2
        assert ratio == pytest.approx(n / (n - k), rel=1e-12)


def test_gmm_small_leaves_estimates_and_hansen_j_alone(data):
    small = _fit("gmm_unadj_small", data)
    large = _fit("gmm_unadj", data)
    assert np.array_equal(small.params.to_numpy(), large.params.to_numpy())
    assert small.diagnostics["Hansen J statistic"] == pytest.approx(
        large.diagnostics["Hansen J statistic"], rel=1e-14
    )


def test_default_gmm_reports_its_variance_not_its_weight_matrix(data):
    """``robust='nonrobust'`` picks the weight matrix; the SE is a sandwich."""
    rule = _fit("gmm_unadj_sandwich_small", data).inference_rule()
    assert rule["vcov"] == "GMM sandwich variance, unadjusted weight matrix"
    assert rule["small"] is True


def test_between_estimator_reports_rows_and_groups(ref, data):
    """xtreg, be: e(N) counts rows; the regression has one row per group."""
    res = _fit("xtbe", data)
    rule = res.inference_rule()
    assert rule["n_rows"] == ref["xtbe"]["N"] == 240
    assert rule["n_groups"] == 30
    assert rule["n_obs"] == 30  # the N the degrees of freedom use
    assert "240 observations in 30 groups" in res.summary()


def test_collinear_column_identified_combination(ref, data):
    """Stata omits x1 (keeps dup = 2*x1); StatsPAI omits dup. The coefficient
    on x1 here is Stata's x1 + 2*dup, and a test of the omitted name is
    refused instead of being answered."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.regress("y ~ x1 + x2 + dup", data=data)
    want = ref["collinear"]
    assert float(res.params["x1"]) == pytest.approx(want["effect"], rel=RTOL)
    assert float(res.std_errors["x1"]) == pytest.approx(want["se"], rel=RTOL)
    joint = sp.test(res, "x1 = x2 = 0")
    assert joint["statistic"] == pytest.approx(want["joint"], rel=RTOL)
    assert joint["df"] == (want["df"], want["df_r"])
    with pytest.raises(Exception, match="omitted from the model"):
        sp.test(res, "dup = 0")


@pytest.mark.parametrize(
    "fit,distribution,df,note",
    [
        # sp.iv defaults to ivregress ..., small
        (lambda d: sp.iv(IV, data=d), "t", 236, "small=True"),
        # linearmodels scaling: t(N - K) where xtreg, fe vce(cluster) uses
        # t(G-1); named since 1.42 made ssc='stata' the default
        (
            lambda d: sp.panel(
                d,
                "yp ~ x1 + x2",
                entity="g",
                time="t",
                method="fe",
                cluster="g",
                ssc="linearmodels",
            ),
            "t",
            208,
            "linearmodels scaling",
        ),
        # ... and t / F after RE, where xtreg, re reports z / chi2
        (
            lambda d: sp.panel(
                d,
                "yp ~ x1 + x2",
                entity="g",
                time="t",
                method="re",
                ssc="linearmodels",
            ),
            "t",
            237,
            "z / chi2 after xtreg, re",
        ),
        # HC1 after logit is N/(N-K); Stata's vce(robust) is N/(N-1)
        (
            lambda d: sp.logit("yb ~ x1 + x2", data=d, robust="hc1"),
            "z",
            None,
            "robust='robust'",
        ),
    ],
)
def test_defaults_that_differ_from_stata_say_so(data, fit, distribution, df, note):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit(data)
    rule = res.inference_rule()
    assert rule["distribution"] == distribution
    assert rule["df"] == df
    assert any(note in n for n in rule["notes"]), rule["notes"]
    assert note.split()[0] in res.summary()
    assert res.to_agent_summary()["inference"] == rule


@pytest.mark.parametrize(
    "key,stat,p",
    [
        ("xtfe_cluster", "F-statistic", "F p-value"),
        ("xtre", "Wald chi2", "Prob > chi2"),
    ],
)
def test_panel_model_test_uses_the_reported_variance(ref, data, key, stat, p):
    """The header model test is Stata's: F(2, G-1) under vce(cluster), Wald
    chi2 after xtreg, re. linearmodels' homoskedastic F (12.29 here) used
    to be printed whatever the variance."""
    res = _fit(key, data)
    assert res.diagnostics[stat] == pytest.approx(ref[key]["joint"], rel=RTOL)
    assert res.diagnostics[p] == pytest.approx(ref[key]["joint_p"], rel=RTOL)


def test_panel_model_test_agrees_with_sp_test_under_the_default_convention(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.panel(
            data, "yp ~ x1 + x2", entity="g", time="t", method="fe", cluster="g"
        )
    joint = sp.test(res, "x1 = x2 = 0")
    assert res.diagnostics["F-statistic"] == pytest.approx(
        joint["statistic"], rel=1e-14
    )
    assert res.diagnostics["F p-value"] == pytest.approx(joint["pvalue"], rel=1e-12)


# --------------------------------------------------------------------- #
# Defaults that took effect in 1.42 (owner decision 2026-10-24)
# --------------------------------------------------------------------- #


def _future(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fn()
    return res, [w for w in caught if issubclass(w.category, FutureWarning)]


def test_gmm_default_weight_matrix_is_stata_robust(ref, data):
    """Since 1.42 sp.iv(method='gmm') is Stata's ivregress gmm: wmatrix(robust)."""
    res, fw = _future(lambda: sp.iv(IV, data=data, method="gmm", small=False))
    assert fw == []
    assert float(res.params["x1"]) == pytest.approx(ref["gmm_robust"]["b_x1"], rel=RTOL)
    assert float(res.std_errors["x1"]) == pytest.approx(
        ref["gmm_robust"]["se_x1"], rel=RTOL
    )
    named = sp.iv(IV, data=data, method="gmm", wmatrix="robust", small=False)
    assert np.array_equal(res.params.to_numpy(), named.params.to_numpy())
    # just-identified: the weight matrix changes nothing
    just = "y ~ x1 + x2 + (xd ~ z1)"
    a = sp.iv(just, data=data, method="gmm")
    b = sp.iv(just, data=data, method="gmm", wmatrix="unadjusted")
    np.testing.assert_allclose(a.params, b.params, rtol=1e-12)


def test_wmatrix_is_validated(data):
    with pytest.raises(Exception, match="wmatrix"):
        sp.iv(IV, data=data, method="gmm", wmatrix="hac")
    with pytest.raises(Exception, match="method='gmm'"):
        sp.iv(IV, data=data, wmatrix="robust")
    with pytest.raises(Exception, match="cluster="):
        sp.iv(IV, data=data, method="gmm", wmatrix="cluster")


@pytest.mark.parametrize(
    "method,kw,is_stata",
    [
        ("fe", {"cluster": "g"}, True),
        ("fe", {"robust": "robust"}, True),
        ("re", {}, True),
        ("pooled", {"cluster": "g"}, True),
        ("fe", {}, False),
        ("pooled", {"robust": "robust"}, False),
        ("be", {}, False),
        ("fe", {"robust": "kernel"}, False),
    ],
)
def test_panel_ssc_default_is_stata_exactly_where_it_moves_numbers(
    data, method, kw, is_stata
):
    """1.42: ssc=None is 'stata' where that convention is defined and changes
    the inference; elsewhere linearmodels' scaling, which there is the same
    number or has no Stata counterpart. No FutureWarning any more."""

    def fit(**extra):
        return sp.panel(
            data, "yp ~ x1 + x2", entity="g", time="t", method=method, **kw, **extra
        )

    res, fw = _future(fit)
    assert fw == []
    assert res.model_info["ssc"] == ("stata" if is_stata else "linearmodels")
    other = fit(ssc="stata" if is_stata else "linearmodels")
    np.testing.assert_allclose(res.std_errors, other.std_errors, rtol=1e-14)
    np.testing.assert_allclose(res.pvalues, other.pvalues, rtol=1e-14)
    if is_stata:
        old = fit(ssc="linearmodels")
        assert not np.allclose(old.pvalues, res.pvalues, rtol=1e-12)
