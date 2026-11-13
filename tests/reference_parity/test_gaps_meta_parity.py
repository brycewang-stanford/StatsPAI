"""Reference parity for the meta-analysis items of the 2026-11 gap list.

New in this round: ``sp.meta_analysis(method='PM' | 'ML')``,
``sp.meta_regression`` and ``sp.trim_and_fill``.

References, on the same CSV bytes (``gaps_meta.csv`` from
``_generate_gaps_meta_data.py`` and the committed ``probe_epi_meta.csv``):

* ``_fixtures/gaps_meta_R.json`` from ``_generate_gaps_meta_R.R``: R 4.5.2,
  ``metafor`` 5.0.1 ``rma(method=)``, ``rma(mods=)``, ``trimfill``.
* ``_fixtures/gaps_meta_Stata.csv`` from ``_generate_gaps_meta_Stata.do``:
  Stata 18 ``meta regress``.

Tolerances. Every quantity is closed form given ``tau2``. ``metafor`` and
Stata stop the iterative estimators (REML, ML, Paule-Mandel) at a relative
change of about 1e-5 by default; both generators tighten the tolerances, and
the comparisons are then at 1e-8 relative (observed 1e-12 or better).

Two documented departures from ``metafor``, each pinned by a second
reference:

* ``trimfill`` refits with ``metafor``'s default stopping rule whatever the
  control of the model passed in, so for the iterative estimators its pooled
  effect carries that 1e-5 to 1e-4 error. The number of imputed studies, the
  side and the imputed studies themselves are compared exactly; the pooled
  effect is compared at 1e-8 with ``metafor``'s own ``rma`` on the augmented
  studies run to convergence, and at 1e-3 with ``trimfill``'s output.
* Without an intercept ``metafor`` centres the effects at their unweighted
  mean before computing ``QE`` and ``tau2``. That leaves both unchanged when
  the model has an intercept and changes them when it has none: its ``QE``
  is then not the weighted residual sum of squares of the fit (``lm`` on the
  same weights gives ours). The reference for that case is Stata
  ``meta regress, noconstant``; a test rebuilds ``metafor``'s ``QE`` from
  ours on centred effects to show that this is the whole difference.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.inference import _meta_core

FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-8


@pytest.fixture(scope="module")
def data() -> dict:
    return {
        "gaps": pd.read_csv(FIX / "gaps_meta.csv"),
        "probe": pd.read_csv(FIX / "probe_epi_meta.csv"),
    }


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads((FIX / "gaps_meta_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stata() -> dict:
    df = pd.read_csv(FIX / "gaps_meta_Stata.csv")
    return {
        case: dict(zip(g["name"], g["value"].astype(float)))
        for case, g in df.groupby("case")
    }


# ---------------------------------------------------------------------------
# sp.meta_analysis(method='PM' | 'ML')
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", ["gaps", "probe"])
@pytest.mark.parametrize("method", ["PM", "ML"])
@pytest.mark.parametrize("hksj", [False, True])
def test_meta_analysis_pm_ml_match_metafor(data, ref, dataset, method, hksj):
    d = data[dataset]
    want = ref["estimators"][f"{dataset}_{method}" + ("_knha" if hksj else "")]
    got = sp.meta_analysis(d["yi"], d["sei"], method=method, hksj=hksj)
    assert got.method == method
    assert got.tau2 == pytest.approx(want["tau2"], rel=RTOL)
    assert got.estimate == pytest.approx(want["estimate"], rel=RTOL)
    assert got.se == pytest.approx(want["se"], rel=RTOL)
    assert got.p_value == pytest.approx(want["pval"], rel=RTOL)
    assert got.ci[0] == pytest.approx(want["ci_lb"], rel=RTOL)
    assert got.ci[1] == pytest.approx(want["ci_ub"], rel=RTOL)
    assert got.q == pytest.approx(want["QE"], rel=RTOL)


@pytest.mark.parametrize("dataset", ["gaps", "probe"])
def test_tau2_estimators_satisfy_their_estimating_equations(data, dataset):
    """PM: generalised Q equals its df. ML: the likelihood score is zero."""
    d = data[dataset]
    y, v = d["yi"].to_numpy(), d["sei"].to_numpy() ** 2
    k = len(y)
    pm = sp.meta_analysis(y, np.sqrt(v), method="PM").tau2
    w = 1 / (v + pm)
    mu = np.sum(w * y) / np.sum(w)
    assert np.sum(w * (y - mu) ** 2) == pytest.approx(k - 1, rel=1e-12)
    ml = sp.meta_analysis(y, np.sqrt(v), method="ML").tau2
    w = 1 / (v + ml)
    mu = np.sum(w * y) / np.sum(w)
    score = np.sum(w**2 * (y - mu) ** 2) - np.sum(w)
    assert abs(score) <= 1e-9 * np.sum(w)
    # ML ignores the degree of freedom spent on the mean: below REML
    assert ml < sp.meta_analysis(y, np.sqrt(v), method="REML").tau2


def test_homogeneous_studies_give_zero_tau2_under_every_estimator():
    y = [0.20, 0.21, 0.19, 0.20, 0.22]
    s = [0.10, 0.12, 0.09, 0.15, 0.11]
    fixed = sp.meta_analysis(y, s, method="fixed")
    for method in ("DL", "REML", "ML", "PM"):
        got = sp.meta_analysis(y, s, method=method)
        assert got.tau2 == 0.0
        assert got.estimate == pytest.approx(fixed.estimate, rel=1e-14)


# ---------------------------------------------------------------------------
# sp.meta_regression against metafor::rma(mods=)
# ---------------------------------------------------------------------------

MODS = {"two": ["dose", "blind"], "one": ["dose"]}


@pytest.mark.parametrize("method", ["REML", "DL", "ML", "PM", "fixed"])
@pytest.mark.parametrize("spec", ["two", "two_knha", "one"])
def test_meta_regression_matches_metafor(data, ref, method, spec):
    if method == "fixed" and spec == "two_knha":
        pytest.skip("Knapp-Hartung is refused for the fixed-effects model")
    d = data["gaps"]
    want = ref["meta_regression"][f"{method}_{spec}"]
    hksj = spec.endswith("knha")
    got = sp.meta_regression(
        d["yi"], d["sei"], d[MODS[spec.split("_")[0]]], method=method, hksj=hksj
    )
    assert list(got.params.index) == ["intercept"] + MODS[spec.split("_")[0]]
    np.testing.assert_allclose(got.params, want["beta"], rtol=RTOL)
    np.testing.assert_allclose(got.std_errors, want["se"], rtol=RTOL)
    np.testing.assert_allclose(got.statistics, want["zval"], rtol=RTOL)
    np.testing.assert_allclose(got.p_values, want["pval"], rtol=RTOL)
    np.testing.assert_allclose(got.conf_int["lower"], want["ci_lb"], rtol=RTOL)
    np.testing.assert_allclose(got.conf_int["upper"], want["ci_ub"], rtol=RTOL)
    np.testing.assert_allclose(got.vcov.to_numpy().ravel(), want["vb"], rtol=RTOL)
    assert got.tau2 == pytest.approx(want["tau2"], rel=RTOL, abs=1e-15)
    assert 100 * got.i2 == pytest.approx(want["I2"], rel=RTOL)
    assert got.h2 == pytest.approx(want["H2"], rel=RTOL)
    assert 100 * got.r2 == pytest.approx(want["R2"], rel=RTOL)
    assert got.qe == pytest.approx(want["QE"], rel=RTOL)
    assert got.qe_pvalue == pytest.approx(want["QEp"], rel=RTOL)
    assert got.qm == pytest.approx(want["QM"], rel=RTOL)
    assert got.qm_pvalue == pytest.approx(want["QMp"], rel=RTOL)
    if hksj:
        assert list(got.qm_df) == want["QMdf"]
    else:
        assert got.qm_df == want["QMdf"][0]
    assert got.k == want["k"]


def test_meta_regression_predict_matches_metafor(data, ref):
    d = data["gaps"]
    want = ref["meta_regression"]["REML_two_predict"]
    fit = sp.meta_regression(d["yi"], d["sei"], d[["dose", "blind"]])
    for new in (
        np.array([[2, 0], [5, 1], [9, 1]]),
        {"blind": [0, 1, 1], "dose": [2, 5, 9]},  # by name, any order
    ):
        got = fit.predict(new)
        np.testing.assert_allclose(got["prediction"], want["pred"], rtol=RTOL)
        np.testing.assert_allclose(got["se"], want["se"], rtol=RTOL)
        np.testing.assert_allclose(got["conf_low"], want["ci_lb"], rtol=RTOL)
        np.testing.assert_allclose(got["conf_high"], want["ci_ub"], rtol=RTOL)


STATA_METHOD = {"REML": "REML", "ML": "ML", "DL": "DL", "fixed": "fixed"}


@pytest.mark.parametrize(
    "case",
    [
        "REML_two",
        "ML_two",
        "DL_two",
        "REML_two_knha",
        "fixed_two",
        "REML_noint",
        "DL_noint",
        "ML_noint",
        "DL_noint_knha",
    ],
)
def test_meta_regression_matches_stata(data, stata, case):
    """Stata ``meta regress``, with and without a constant."""
    d = data["gaps"]
    want = stata[case]
    method, kind, *rest = case.split("_")
    hksj = bool(rest)
    got = sp.meta_regression(
        d["yi"],
        d["sei"],
        d[["dose", "blind"]],
        method=method,
        hksj=hksj,
        intercept=kind == "two",
    )
    for name in got.params.index:
        key = "_cons" if name == "intercept" else name
        assert got.params[name] == pytest.approx(want[f"b_{key}"], rel=RTOL)
        assert got.std_errors[name] == pytest.approx(want[f"se_{key}"], rel=RTOL)
    assert got.qe == pytest.approx(want["Q_res"], rel=RTOL)
    assert got.qe_df == want["df_Q_res"]
    assert got.qe_pvalue == pytest.approx(want["p_Q_res"], rel=RTOL)
    if method != "fixed":
        assert got.tau2 == pytest.approx(want["tau2"], rel=RTOL)
        assert 100 * got.i2 == pytest.approx(want["I2_res"], rel=RTOL)
        assert got.h2 == pytest.approx(want["H2_res"], rel=RTOL)
    if hksj:
        assert got.qm == pytest.approx(want["F"], rel=RTOL)
        assert list(got.qm_df) == [want["df_m"], want["df_r"]]
    else:
        assert got.qm == pytest.approx(want["chi2"], rel=RTOL)
        assert got.qm_df == want["df_m"]
    assert got.qm_pvalue == pytest.approx(want["p"], rel=RTOL)


def test_no_intercept_qe_is_the_weighted_rss_and_metafor_centres(data, ref):
    """Why ``metafor`` and Stata disagree without an intercept."""
    d = data["gaps"]
    want = ref["meta_regression"]["DL_noint"]
    y, v = d["yi"].to_numpy(), d["sei"].to_numpy() ** 2
    X = d[["dose", "blind"]].to_numpy(dtype=float)
    got = sp.meta_regression(y, d["sei"], X, method="DL", intercept=False)
    # ours is the weighted residual sum of squares of the fit (R lm)
    assert got.qe == pytest.approx(want["wrss_lm"], rel=1e-10)
    # metafor's is the same quadratic form in the centred effects
    P0 = _meta_core.projection(v, X, 0.0)
    yc = y - want["ybar"]
    assert float(yc @ P0 @ yc) == pytest.approx(want["QE"], rel=1e-10)
    # ... and so is its DerSimonian-Laird tau2
    assert (float(yc @ P0 @ yc) - (len(y) - 2)) / np.trace(P0) == pytest.approx(
        want["tau2"], rel=1e-10
    )
    assert abs(want["QE"] - got.qe) > 1e-3 * got.qe  # the two are different numbers


def test_meta_regression_reduces_to_meta_analysis_and_to_wls(data):
    d = data["gaps"]
    y, s = d["yi"].to_numpy(), d["sei"].to_numpy()
    # a moderator that is zero for everyone except one study leaves k - 1
    # studies pooled by the intercept, exactly as meta_analysis pools them
    # (for the estimators built on residuals; the ML score also counts the
    # weight of the study that the dummy fits exactly)
    z = np.zeros(len(y))
    z[0] = 1.0
    for method in ("DL", "REML", "PM"):
        mr = sp.meta_regression(y, s, z, method=method)
        ma = sp.meta_analysis(y[1:], s[1:], method=method)
        assert mr.tau2 == pytest.approx(ma.tau2, rel=1e-8)
        assert mr.params["intercept"] == pytest.approx(ma.estimate, rel=1e-8)
        assert mr.std_errors["intercept"] == pytest.approx(ma.se, rel=1e-8)
    # fixed effects: weighted least squares with known variances
    X = np.column_stack([np.ones(len(y)), d["dose"]])
    w = 1 / s**2
    beta = np.linalg.solve((X.T * w) @ X, (X.T * w) @ y)
    fe = sp.meta_regression(y, s, d["dose"], method="fixed")
    np.testing.assert_allclose(fe.params, beta, rtol=1e-12)
    np.testing.assert_allclose(
        fe.std_errors, np.sqrt(np.diag(np.linalg.inv((X.T * w) @ X))), rtol=1e-12
    )
    # the result method of a meta-analysis uses its own settings
    ma = sp.meta_analysis(y, s, method="REML", hksj=True)
    via = ma.meta_regression(d[["dose"]])
    direct = sp.meta_regression(y, s, d[["dose"]], method="REML", hksj=True)
    np.testing.assert_array_equal(via.params, direct.params)
    np.testing.assert_array_equal(via.std_errors, direct.std_errors)


def test_meta_regression_refuses_what_it_cannot_fit(data):
    d = data["gaps"]
    y, s = d["yi"], d["sei"]
    with pytest.raises(MethodIncompatibility, match="collinear"):
        sp.meta_regression(y, s, {"a": d["dose"], "b": 2 * d["dose"]})
    with pytest.raises(MethodIncompatibility, match="collinear"):
        sp.meta_regression(y, s, np.ones(len(y)))
    with pytest.raises(DataInsufficient, match="cannot identify"):
        sp.meta_regression(y[:3], s[:3], d[["dose", "blind"]][:3])
    with pytest.raises(MethodIncompatibility, match="not numeric"):
        sp.meta_regression(y, s, {"g": ["a", "b"] * 11})
    with pytest.raises(MethodIncompatibility, match="rows of moderators"):
        sp.meta_regression(y, s, d["dose"][:5])
    with pytest.raises(MethodIncompatibility, match="method must be"):
        sp.meta_regression(y, s, d["dose"], method="HS")
    with pytest.raises(MethodIncompatibility, match="Knapp-Hartung"):
        sp.meta_regression(y, s, d["dose"], method="fixed", hksj=True)
    with pytest.raises(MethodIncompatibility, match="finite"):
        sp.meta_regression(y, s, d["dose"].where(d["dose"] > 2))


# ---------------------------------------------------------------------------
# sp.trim_and_fill against metafor::trimfill
# ---------------------------------------------------------------------------

TF_CASES = [
    (dataset, method, estimator, side)
    for dataset in ("gaps", "probe")
    for method in ("DL", "REML", "PM", "ML", "fixed")
    for estimator in ("L0", "R0", "Q0")
    for side in ("auto", "left", "right")
]


@pytest.mark.parametrize("dataset, method, estimator, side", TF_CASES)
def test_trim_and_fill_matches_metafor(data, ref, dataset, method, estimator, side):
    d = data[dataset]
    want = ref["trimfill"][f"{dataset}_{method}_{estimator}_{side}"]
    kwargs = dict(
        method=method, estimator=estimator, side=None if side == "auto" else side
    )
    if want.get("error"):
        # Q0 is the square root of a negative number here; metafor stops too
        with pytest.raises(MethodIncompatibility, match="Q0 estimator is undefined"):
            sp.trim_and_fill(d["yi"], d["sei"], **kwargs)
        return
    got = sp.trim_and_fill(d["yi"], d["sei"], **kwargs)
    assert got.k0 == want["k0"]
    assert got.side == want["side"]
    assert len(got.filled.effects) == want["k"]
    # the imputed studies (metafor lists them in its own order; a single
    # one arrives from JSON as a scalar)
    np.testing.assert_allclose(
        np.sort(got.filled_effects), np.sort(np.atleast_1d(want["yi_fill"])), rtol=1e-10
    )
    np.testing.assert_allclose(
        np.sort(got.filled_se), np.sort(np.atleast_1d(want["sei_fill"])), rtol=1e-10
    )
    # pooled effect of the augmented set, model run to convergence
    assert got.estimate == pytest.approx(want["estimate_tight"], rel=RTOL)
    assert got.se == pytest.approx(want["se_tight"], rel=RTOL)
    if method != "fixed":
        assert got.tau2 == pytest.approx(want["tau2_tight"], rel=RTOL, abs=1e-15)
    # trimfill's own output: exact for the closed-form models, within its
    # stopping rule for the iterative ones
    loose = 1e-10 if method in ("DL", "fixed") else 1e-3
    assert got.estimate == pytest.approx(want["estimate"], rel=loose)
    assert got.se == pytest.approx(want["se"], rel=loose)
    assert got.ci[0] == pytest.approx(want["ci_lb"], rel=loose)
    assert got.ci[1] == pytest.approx(want["ci_ub"], rel=loose)
    assert got.p_value == pytest.approx(want["pval"], rel=100 * loose)


def test_trim_and_fill_is_a_mirror_image_and_flips_with_the_data(data):
    d = data["gaps"]
    y, s = d["yi"].to_numpy(), d["sei"].to_numpy()
    tf = sp.trim_and_fill(y, s)
    assert tf.side == "left" and tf.k0 > 0
    # the imputed studies mirror the k0 largest effects around the centre of
    # the trimmed fit, with their standard errors
    order = np.argsort(y)
    centre = sp.meta_analysis(y[order][: -tf.k0], s[order][: -tf.k0]).estimate
    np.testing.assert_allclose(
        tf.filled_effects, 2 * centre - y[order][-tf.k0 :], rtol=1e-12
    )
    np.testing.assert_allclose(tf.filled_se, s[order][-tf.k0 :], rtol=1e-12)
    # filling pulls the pooled effect towards the missing side
    assert tf.estimate < tf.original.estimate
    # negating the effects negates everything and swaps the side
    neg = sp.trim_and_fill(-y, s)
    assert neg.side == "right" and neg.k0 == tf.k0
    assert neg.estimate == pytest.approx(-tf.estimate, rel=1e-12)
    assert neg.se == pytest.approx(tf.se, rel=1e-12)
    # result method and function agree, and carry the settings of the fit
    ma = sp.meta_analysis(y, s, method="REML", hksj=True)
    via = ma.trim_and_fill(estimator="R0")
    direct = sp.trim_and_fill(y, s, method="REML", hksj=True, estimator="R0")
    assert (via.k0, via.estimate, via.se) == (direct.k0, direct.estimate, direct.se)
    assert via.filled.hksj and via.filled.method == "REML"


def test_trim_and_fill_symmetric_funnel_imputes_nothing():
    y = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]) + 0.5
    s = np.array([0.3, 0.2, 0.1, 0.05, 0.1, 0.2, 0.3])
    for side in ("left", "right"):
        tf = sp.trim_and_fill(y, s, side=side)
        assert tf.k0 == 0 and len(tf.filled_effects) == 0
        assert tf.estimate == tf.original.estimate
    with pytest.raises(MethodIncompatibility, match="cannot choose a side"):
        sp.trim_and_fill(y, np.full(7, 0.1))
    with pytest.raises(DataInsufficient, match="at least 3"):
        sp.trim_and_fill([0.1, 0.2], [0.1, 0.1])
    with pytest.raises(MethodIncompatibility, match="estimator must be"):
        sp.trim_and_fill(y, s, estimator="S0")
    with pytest.raises(MethodIncompatibility, match="side must be"):
        sp.trim_and_fill(y, s, side="up")
