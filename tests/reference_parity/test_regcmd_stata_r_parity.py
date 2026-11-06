"""Regression and inference commands vs Stata 18 MP and R, on the same bytes.

References
----------
``_fixtures/regcmd_Stata.json`` from ``_fixtures/_generate_regcmd_Stata.do``
(Stata 18 MP; gologit2 3.2.8 from SSC in a private ado tree; boottest 4.5.3;
ivreg2 4.1.12) and ``_fixtures/regcmd_R.json`` from
``_fixtures/_generate_regcmd_R.R`` (R 4.5.2; survival 3.8.3,
sampleSelection 1.2.14, VGAM 1.1.14, sandwich 3.1.1). The data are
``regcmd_cs.csv`` / ``regcmd_wild.csv`` (``_generate_regcmd_data.py``) and
the committed ``textbook_cs.csv`` / ``textbook_ts.csv`` / ``ivreport_fx.csv``.
Every ML fit in Stata runs to ``tolerance(1e-13) ltolerance(1e-15)
nrtolerance(1e-12)`` and ``prais`` / ``nl`` to ``tolerance(1e-14)`` /
``eps(1e-14)``, so the comparisons are not limited by where Stata stops.
All the packages were run as black boxes; none of their source was read.

What is pinned, and how closely
-------------------------------
* ``sp.intreg`` -- Stata ``intreg``: oim, robust, cluster, ``[pw=]``,
  ``[pw=]`` + cluster. Coefficients and SEs 1e-9 relative (observed 5e-11).
  R ``survreg(type = "interval2")``: coefficients, observed-information SEs
  and log likelihood 1e-9. survreg's ``robust = TRUE`` variance (built from
  its dfbeta residuals) does not reproduce the closed-form score sandwich on
  interval-censored rows (x1: 0.0538 against 0.0497); Stata does, and so do
  we (test below), so survreg's robust variance is not used.
* ``sp.heckprobit`` -- Stata ``heckprobit``: the same five variants, 1e-9
  (observed 4e-11). R ``sampleSelection`` with ``maxMethod = "NR"``: 1e-8
  (observed 2e-9, the reference's convergence). sampleSelection's default
  maximiser is BHHH, whose covariance is the inverse outer product of the
  scores; that quantity is rebuilt here from our own observed-information
  and robust fits and agrees to 1e-7 (observed 1e-8).
* With ``[pw=]`` both report Stata's pseudo log likelihood, the sum of
  ``w_i * ll_i`` with the weights as given (fixed in this change; it used to
  be the sum with the weights rescaled to sum to n).
* ``sp.gologit`` -- gologit2: all / no / partial parallel lines, logit and
  probit, robust and cluster; VGAM ``cumulative(reverse = TRUE)``
  coefficients. 1e-6 relative (observed 4.4e-7 against gologit2, 1.7e-7
  against VGAM): both references stop where their own derivatives stop --
  the score of the closed-form likelihood is 4e-5 at gologit2's estimate and
  3e-5 at VGAM's, 6e-8 at ours, and our log likelihood is the highest of the
  three (asserted below). VGAM's covariance is the expected information, so
  only its coefficients are compared.
* ``sp.prais`` -- Stata ``prais``: Prais-Winsten and Cochrane-Orcutt, six
  rhotypes, iterated and two-step, and robust / hc2 / hc3: coefficients,
  SEs, rho, R-squared, F, root MSE and both Durbin-Watson statistics 1e-9
  (observed 2.5e-15).
* ``sp.nls`` -- R ``nls`` with the analytic gradient from ``deriv()`` and
  ``sandwich``: gnr / HC1 / cluster, 1e-8 (observed 3e-9) at
  ``tol=1e-12``. At the default ``tol=1e-10`` the search stops with a score
  of 1e-6 and the estimates are 2e-7 from R's (asserted at 1e-6). Stata
  ``nl`` differentiates by forward differences (``delta(4e-7)``), so its SEs
  carry about seven digits: 1e-6 (observed 4.2e-7), gnr / robust / cluster /
  hc2 / hc3.
* ``sp.mprobit`` -- Stata ``mprobit`` (independent errors of variance one,
  ``intpoints(120)``): base outcome 1 and 3, no constants. Coefficients
  1e-8 (observed 1.2e-9), log likelihood 1e-10, SEs 1e-6 (observed 2.8e-7):
  our Hessian is a finite difference of finite-difference scores.
* ``sp.standardize_coefs(scale_y=False)`` and ``variables=`` -- Stata
  regress / ivreg2 on x-only z-scores, 1e-9 (observed 2e-14).
* ``sp.iv_wild_bootstrap`` -- boottest after ``ivregress 2sls, small``,
  Rademacher weights; with 12 clusters both sides enumerate all 4,096 sign
  vectors, so the p-values are exact rationals and are compared exactly. One
  endogenous regressor (over- and just-identified, with and without an
  exogenous control, ``beta0`` = 0 and 0.5): p exact, t 1e-9 (observed
  2e-15). The confidence set: boottest does not report the exact set
  ``{beta0 : p(beta0) > alpha}`` (its bounds are where p is still above
  alpha, also at levels where alpha * 4096 is an integer); our bounds are
  the exact jumps of the p-value function, which is identical to
  boottest's at boottest's own bounds (asserted exactly). The gap is at most
  0.006 SE.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import statspai as sp

FX = pathlib.Path(__file__).parent / "_fixtures"
S = json.loads((FX / "regcmd_Stata.json").read_text(encoding="utf-8"))
R = json.loads((FX / "regcmd_R.json").read_text(encoding="utf-8"))
X3 = ["x1", "x2", "x3"]


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(FX / name, float_precision="round_trip")


@pytest.fixture(scope="module")
def cs():
    return _read("regcmd_cs.csv")


@pytest.fixture(scope="module")
def wild():
    return _read("regcmd_wild.csv")


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _rel(ours, ref):
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    return float(np.max(np.abs(ours - ref) / np.abs(ref)))


def _stata_by_name(key):
    ref = S[key]
    return {n: (b, se) for n, b, se in zip(ref["names"], ref["b"], ref["se"])}


# ================================================================ intreg
# ours: const, x1, x2, x3, lnsigma; Stata: model:x1 x2 x3 _cons, lnsigma:_cons
_INTREG_ORDER = [1, 2, 3, 0, 4]
_INTREG = [
    ("intreg_oim", {}),
    ("intreg_robust", {"vce": "robust"}),
    ("intreg_cluster", {"cluster": "g"}),
    ("intreg_pw", {"weights": "pw"}),
    ("intreg_pw_cluster", {"weights": "pw", "cluster": "g"}),
]


@pytest.mark.parametrize("key, kw", _INTREG)
def test_intreg_matches_stata(cs, key, kw):
    fit = sp.intreg(cs, "ilo", "ihi", x=X3, **kw)
    theta = np.asarray(fit._ml_state["theta"])[_INTREG_ORDER]
    se = np.sqrt(np.diag(fit._ml_state["vcov"]))[_INTREG_ORDER]
    assert _rel(theta, S[key]["b"]) < 1e-9
    assert _rel(se, S[key]["se"]) < 1e-9
    # Stata's e(ll); with [pw=] the pseudo log likelihood sum w_i ll_i
    assert fit.model_info["log_likelihood"] == pytest.approx(S[key]["ll"], rel=1e-12)
    assert fit._ml_state["nobs"] == S[key]["N"] == 795


def test_intreg_censoring_counts_match_stata(cs):
    mi = sp.intreg(cs, "ilo", "ihi", x=X3).model_info
    counts = (
        mi["n_uncensored"],
        mi["n_left_censored"],
        mi["n_right_censored"],
        mi["n_interval"],
    )
    stata = tuple(int(S[f"intreg_oim_{k}"]) for k in ("N_unc", "N_lc", "N_rc", "N_int"))
    assert counts == stata


def test_intreg_matches_survreg(cs):
    fit = sp.intreg(cs, "ilo", "ihi", x=X3)
    ref = R["intreg_oim"]  # (Intercept) x1 x2 x3 Log(scale): our order
    assert _rel(fit._ml_state["theta"], ref["b"]) < 1e-9
    assert _rel(np.sqrt(np.diag(fit._ml_state["vcov"])), ref["se"]) < 1e-9
    assert fit.model_info["log_likelihood"] == pytest.approx(ref["ll"], rel=1e-12)


def test_intreg_robust_is_the_score_sandwich(cs):
    """The robust variance Stata and StatsPAI report is n/(n-1) V S'S V with
    V the observed-information variance and S the per-row scores of the
    interval likelihood, here written out from the normal CDF and
    differentiated by central differences (h = 1e-6: ~1e-9 relative)."""
    d = cs[~(cs.ilo.isna() & cs.ihi.isna())]
    X = np.column_stack([np.ones(len(d)), d[X3].to_numpy()])
    lo, hi = d.ilo.to_numpy(), d.ihi.to_numpy()
    exact, left, right = lo == hi, np.isnan(lo), np.isnan(hi)
    band = ~(exact | left | right)

    def loglik(t):
        m, s = X @ t[:4], np.exp(t[4])
        out = np.empty(len(m))
        out[exact] = stats.norm.logpdf((lo[exact] - m[exact]) / s) - t[4]
        out[left] = stats.norm.logcdf((hi[left] - m[left]) / s)
        out[right] = stats.norm.logsf((lo[right] - m[right]) / s)
        out[band] = np.log(
            stats.norm.cdf((hi[band] - m[band]) / s)
            - stats.norm.cdf((lo[band] - m[band]) / s)
        )
        return out

    oim = sp.intreg(cs, "ilo", "ihi", x=X3)
    theta = np.asarray(oim._ml_state["theta"])
    h = 1e-6
    scores = np.column_stack(
        [(loglik(theta + h * e) - loglik(theta - h * e)) / (2 * h) for e in np.eye(5)]
    )
    V = oim._ml_state["vcov"]
    n = len(d)
    sandwich = n / (n - 1) * V @ (scores.T @ scores) @ V
    rob = sp.intreg(cs, "ilo", "ihi", x=X3, vce="robust")
    np.testing.assert_allclose(rob._ml_state["vcov"], sandwich, rtol=1e-6)


# ============================================================ heckprobit
# ours: const x1 x2 x3 | select:const z x1 | athrho
_HP_STATA_ORDER = [1, 2, 3, 0, 5, 6, 4, 7]  # -> ybh:x1 x2 x3 _cons s:z x1 _cons athrho
_HP_R = ["select:const", "select:z", "select:x1", "const", "x1", "x2", "x3", "rho"]
_HECK = dict(y="ybh", x=X3, select="s", z=["z", "x1"])


@pytest.mark.parametrize(
    "key, kw",
    [
        ("heckprobit_oim", {}),
        ("heckprobit_robust", {"vce": "robust"}),
        ("heckprobit_cluster", {"cluster": "g"}),
        ("heckprobit_pw", {"weights": "pw"}),
        ("heckprobit_pw_cluster", {"weights": "pw", "cluster": "g"}),
    ],
)
def test_heckprobit_matches_stata(cs, key, kw):
    fit = sp.heckprobit(cs, **_HECK, **kw)
    theta = np.asarray(fit._ml_state["theta"])[_HP_STATA_ORDER]
    se = np.sqrt(np.diag(fit._ml_state["vcov"]))[_HP_STATA_ORDER]
    assert _rel(theta, S[key]["b"]) < 1e-9
    assert _rel(se, S[key]["se"]) < 1e-9
    assert fit.model_info["log_likelihood"] == pytest.approx(S[key]["ll"], rel=1e-12)


def test_heckprobit_rho_and_counts_match_stata(cs):
    fit = sp.heckprobit(cs, **_HECK)
    assert fit.model_info["rho"] == pytest.approx(S["heckprobit_oim_rho"], rel=1e-9)
    assert fit.model_info["n_selected"] == S["heckprobit_oim_N_selected"]


def test_heckprobit_matches_sampleselection_newton_raphson(cs):
    fit = sp.heckprobit(cs, **_HECK)
    ref = R["heckprobit_nr"]
    assert ref["method"] == "Newton-Raphson maximisation"
    # 1e-8: where sampleSelection's Newton-Raphson stops (observed 2e-9)
    assert _rel([fit.params[k] for k in _HP_R], ref["b"]) < 1e-8
    assert _rel([fit.std_errors[k] for k in _HP_R], ref["se"]) < 1e-8
    assert fit.model_info["log_likelihood"] == pytest.approx(ref["ll"], rel=1e-12)


def test_heckprobit_reproduces_sampleselection_bhhh_covariance(cs):
    """sampleSelection's default (BHHH) covariance is (S'S)^-1. From our two
    fits, S'S = (n-1)/n H V_rob H with H = V_oim^-1, so (S'S)^-1 =
    n/(n-1) V_oim V_rob^-1 V_oim; rho's SE is (1 - rho^2) times athrho's."""
    oim = sp.heckprobit(cs, **_HECK)
    rob = sp.heckprobit(cs, **_HECK, vce="robust")
    V, Vr = oim._ml_state["vcov"], rob._ml_state["vcov"]
    n = oim._ml_state["nobs"]
    opg_inv = n / (n - 1) * V @ np.linalg.solve(Vr, V)
    names = list(oim.params.index)
    se = np.sqrt(np.diag(opg_inv))
    rho = oim.params["rho"]
    ours = [se[names.index(k)] for k in _HP_R[:-1]] + [
        (1 - rho**2) * se[names.index("athrho")]
    ]
    ref = R["heckprobit_bhhh"]
    assert ref["method"] == "BHHH maximisation"
    # 1e-7: R's BHHH estimate is 6e-8 from the maximum (observed 1e-8)
    assert _rel(ours, ref["se"]) < 1e-7


# =============================================================== gologit
_GOLOGIT = [
    ("gologit_npl", {}),
    ("gologit_pl", {"pl": True}),
    ("gologit_npl_x1", {"npl": ["x1"]}),
    ("gologit_pl_x2x3", {"pl": ["x2", "x3"]}),
    ("gologit_probit_npl", {"link": "probit"}),
    ("gologit_probit_npl_x1", {"link": "probit", "npl": ["x1"]}),
    ("gologit_npl_x1_robust", {"npl": ["x1"], "vce": "robust"}),
    ("gologit_npl_x1_cluster", {"npl": ["x1"], "cluster": "g"}),
    ("gologit_probit_pl_cluster", {"link": "probit", "pl": True, "cluster": "g"}),
]


@pytest.mark.parametrize("key, kw", _GOLOGIT)
def test_gologit_matches_gologit2(cs, key, kw):
    fit = sp.gologit(cs, y="yo", x=X3, **kw)
    ref = _stata_by_name(key)
    names = list(fit.params.index)
    assert set(names) == set(ref)  # gologit2's e(b) layout, constrained terms repeated
    # 1e-6: gologit2 stops where ml's numerical derivatives stop (observed 4.4e-7)
    assert _rel(fit.params, [ref[k][0] for k in names]) < 1e-6
    assert _rel(fit.std_errors, [ref[k][1] for k in names]) < 1e-6
    assert fit.model_info["log_likelihood"] == pytest.approx(S[key]["ll"], rel=1e-12)


def _vgam_name(name):
    eq, var = name.split(":")
    return ("(Intercept)" if var == "_cons" else var), int(eq)


@pytest.mark.parametrize(
    "key, kw",
    [g for g in _GOLOGIT if g[0] in R],
    ids=[g[0] for g in _GOLOGIT if g[0] in R],
)
def test_gologit_coefficients_match_vgam(cs, key, kw):
    fit = sp.gologit(cs, y="yo", x=X3, **kw)
    ref = dict(zip(R[key]["names"], R[key]["b"]))
    want = []
    for name in fit.params.index:
        var, j = _vgam_name(name)
        want.append(ref.get(f"{var}:{j}", ref.get(var)))  # parallel terms: one name
    # 1e-6: VGAM's Fisher scoring stops at a score of ~3e-5 (observed 1.7e-7)
    assert _rel(fit.params, want) < 1e-6
    assert fit.model_info["log_likelihood"] == pytest.approx(R[key]["ll"], rel=1e-12)


def test_gologit_estimate_is_the_stationary_point(cs):
    """Why the references are only 1e-7 away: written out from the logistic
    CDF, the likelihood's score is 6e-8 at our estimate and 4e-5 / 3e-5 at
    gologit2's / VGAM's, and ours has the highest log likelihood."""
    Xm, y = cs[X3].to_numpy(), cs.yo.to_numpy()
    rows = np.arange(len(y))

    def loglik(theta):  # three equations of (x1, x2, x3, _cons)
        B = theta.reshape(3, 4)
        above = special.expit(Xm @ B[:, :3].T + B[:, 3])  # P(Y > j), j = 1..3
        cum = np.column_stack([np.ones(len(y)), above, np.zeros(len(y))])
        return float(np.sum(np.log(cum[rows, y - 1] - cum[rows, y])))

    def score(theta, h=1e-6):
        eye = np.eye(theta.size)
        return np.array(
            [(loglik(theta + h * e) - loglik(theta - h * e)) / (2 * h) for e in eye]
        )

    ours = sp.gologit(cs, y="yo", x=X3).params.to_numpy()
    stata = np.array(S["gologit_npl"]["b"])  # same 1:x1 ... 3:_cons layout
    vg = dict(zip(R["gologit_npl"]["names"], R["gologit_npl"]["b"]))
    vgam = np.array(
        [vg[f"{v}:{j}"] for j in (1, 2, 3) for v in ("x1", "x2", "x3", "(Intercept)")]
    )
    assert np.abs(score(ours)).max() < 1e-6
    assert np.abs(score(stata)).max() > 1e-5
    assert np.abs(score(vgam)).max() > 1e-5
    assert loglik(ours) >= max(loglik(stata), loglik(vgam))


# ================================================================= prais
@pytest.fixture(scope="module")
def ts():
    return pd.read_csv(FX / "textbook_ts.csv")


def _names(key):
    return ["Intercept" if n == "_cons" else n for n in S[key]["names"]]


@pytest.mark.parametrize("method", ["prais", "corc"])
@pytest.mark.parametrize(
    "rhotype", ["regress", "freg", "tscorr", "dw", "theil", "nagar"]
)
@pytest.mark.parametrize("step", ["iter", "two"])
def test_prais_matches_stata(ts, method, rhotype, step):
    fit = sp.prais(
        "y ~ x1 + x2 + d", data=ts, method=method, rhotype=rhotype,
        twostep=(step == "two"), tol=1e-14, maxiter=5000,
    )  # fmt: skip
    key = f"prais_{method}_{rhotype}_{step}"
    names = _names(key)
    assert _rel(fit.params[names], S[key]["b"]) < 1e-9
    assert _rel(fit.std_errors[names], S[key]["se"]) < 1e-9
    assert fit.model_info["rho"] == pytest.approx(S[key + "_rho"], rel=1e-9)
    assert fit.diagnostics["R-squared"] == pytest.approx(S[key + "_r2"], rel=1e-9)
    assert fit.diagnostics["F-statistic"] == pytest.approx(S[key + "_F"], rel=1e-9)
    assert fit.diagnostics["Root MSE"] == pytest.approx(S[key + "_rmse"], rel=1e-9)
    assert fit.model_info["dw_transformed"] == pytest.approx(S[key + "_dw"], rel=1e-9)
    assert fit.model_info["dw_original"] == pytest.approx(S[key + "_dw0"], rel=1e-9)
    assert fit.data_info["nobs"] == S[key]["N"]


@pytest.mark.parametrize("method", ["prais", "corc"])
@pytest.mark.parametrize("vce", ["robust", "hc2", "hc3"])
def test_prais_robust_covariances_match_stata(ts, method, vce):
    fit = sp.prais("y ~ x1 + x2 + d", data=ts, method=method, vce=vce, tol=1e-14)
    key = f"prais_{method}_{vce}"
    names = _names(key)
    assert _rel(fit.params[names], S[key]["b"]) < 1e-9
    assert _rel(fit.std_errors[names], S[key]["se"]) < 1e-9
    assert fit.model_info["rho"] == pytest.approx(S[key + "_rho"], rel=1e-9)
    assert fit.diagnostics["F-statistic"] == pytest.approx(S[key + "_F"], rel=1e-9)


# =================================================================== nls
@pytest.fixture(scope="module")
def nl_data():
    data = pd.read_csv(FX / "textbook_cs.csv")
    data["xp"] = np.exp(data["x1"] / 2)
    data["ynl"] = 2 + 3 * data["xp"] ** 0.5 + data["z2"] / 4
    return data


_NL = "ynl ~ {a} + {b} * xp^{c}"
_NL_START = {"a": 1, "b": 1, "c": 1}
_NL_VCE = {
    "gnr": {},
    "robust": {"vce": "robust"},
    "cluster": {"cluster": "g"},
    "hc2": {"vce": "hc2"},
    "hc3": {"vce": "hc3"},
}


@pytest.mark.parametrize("tag", ["gnr", "robust", "cluster"])
def test_nls_matches_r_nls_with_analytic_gradient(nl_data, tag):
    fit = sp.nls(_NL, nl_data, start=_NL_START, tol=1e-12, **_NL_VCE[tag])
    ref = R[f"nl_{tag}"]
    assert _rel(fit.params[["a", "b", "c"]], ref["b"]) < 1e-8
    assert _rel(fit.std_errors[["a", "b", "c"]], ref["se"]) < 1e-8
    assert fit.diagnostics["Residual SS"] == pytest.approx(ref["rss"], rel=1e-12)


def test_nls_default_tolerance_is_within_1e6_of_r(nl_data):
    """At tol=1e-10 (the default) the search stops with a score of ~1e-6;
    the estimates are 2e-7 from R's, the SEs 3e-7."""
    fit = sp.nls(_NL, nl_data, start=_NL_START)
    ref = R["nl_gnr"]
    assert _rel(fit.params[["a", "b", "c"]], ref["b"]) < 1e-6
    assert _rel(fit.std_errors[["a", "b", "c"]], ref["se"]) < 1e-6


@pytest.mark.parametrize("tag", ["gnr", "robust", "cluster", "hc2", "hc3"])
def test_nls_matches_stata_nl(nl_data, tag):
    fit = sp.nls(_NL, nl_data, start=_NL_START, tol=1e-12, **_NL_VCE[tag])
    ref = S[f"nl_{tag}"]
    assert _rel(fit.params[["a", "b", "c"]], ref["b"]) < 1e-7
    # 1e-6: nl's forward-difference derivatives (observed 4.2e-7)
    assert _rel(fit.std_errors[["a", "b", "c"]], ref["se"]) < 1e-6
    assert fit.diagnostics["Residual SS"] == pytest.approx(
        S[f"nl_{tag}_rss"], rel=1e-12
    )


# =============================================================== mprobit
@pytest.fixture(scope="module")
def modes():
    data = pd.read_csv(FX / "textbook_cs.csv")
    data["mode"] = (
        1
        + (data.x1 + data.z1 > 0).astype(int)
        + (data.x1 - data.x2 + data.z2 > 0.5).astype(int)
    )
    data["case"] = np.arange(len(data))
    long = data.loc[data.index.repeat(3), ["case", "x1", "x2", "mode"]]
    long = long.reset_index(drop=True)
    long["alt"] = np.tile([1, 2, 3], len(data))
    long["choice"] = (long["mode"] == long["alt"]).astype(int)
    return long


@pytest.mark.parametrize(
    "key, kw",
    [
        ("mprobit_base1", {"base": 1}),
        ("mprobit_base3", {"base": 3}),
        ("mprobit_base1_nocons", {"base": 1, "constants": False}),
    ],
)
def test_mprobit_matches_stata(modes, key, kw):
    fit = _quiet(
        sp.mprobit, modes, y="choice", case_vars=["x1", "x2"], chid="case",
        alt="alt", correlation="independent", stddev="homoskedastic", **kw,
    )  # fmt: skip
    # the base outcome's equation is in e(b) as omitted (o.) terms
    ref = {k: v for k, v in _stata_by_name(key).items() if ":o." not in k}
    names = list(fit.params.index)
    assert set(names) == set(ref)
    assert _rel(fit.params, [ref[k][0] for k in names]) < 1e-8
    # 1e-6: finite difference of finite-difference scores (observed 2.8e-7)
    assert _rel(fit.std_errors, [ref[k][1] for k in names]) < 1e-6
    assert fit.model_info["ll"] == pytest.approx(S[key]["ll"], rel=1e-10)


# ===================================================== standardize_coefs
@pytest.fixture(scope="module")
def fx():
    return _read("ivreport_fx.csv")


_STD_NAMES = {"s_x": "x", "s_w1": "w1", "s_w2": "w2", "s_d": "d", "_cons": "Intercept"}
_OLS = "y ~ x + w1 + w2 + d"


@pytest.mark.parametrize(
    "key, fit, kw",
    [
        ("stdx_regress_robust", lambda d: sp.regress(_OLS, data=d, vce="robust"), {}),
        (
            "stdx_only_x_regress_cluster",
            lambda d: sp.regress(_OLS, data=d, cluster="region"),
            {"variables": ["x"]},
        ),
        (
            "stdx_ivreg2_cluster_small",
            lambda d: sp.iv("y ~ (x ~ z) + w1 + w2 + d", data=d, cluster="region"),
            {},
        ),
    ],
)
def test_standardize_x_only_equals_stata_on_x_zscores(fx, key, fit, kw):
    out = _quiet(sp.standardize_coefs, _quiet(fit, fx), scale_y=False, **kw)
    ref = S[key]
    for i, name in enumerate(ref["names"]):
        ours = _STD_NAMES.get(name, name)
        assert out.params[ours] == pytest.approx(ref["b"][i], abs=1e-10), name
        assert out.std_errors[ours] == pytest.approx(ref["se"][i], rel=1e-9), name


# ===================================================== iv_wild_bootstrap
_OVER = "y ~ (d1 ~ z1 + z2) + w"
_TWO = "y ~ (d1 + d2 ~ z1 + z2 + z3) + w"
# key: (formula, tested coefficient, beta0, alpha)
_WRE_ONE = {
    "wre_over_b0": (_OVER, "d1", 0.0, 0.05),
    "wre_over_b05": (_OVER, "d1", 0.5, 0.05),
    "wre_over_l90": (_OVER, "d1", 0.0, 0.10),
    "wre_over_l9375": (_OVER, "d1", 0.0, 0.0625),
    "wre_just_b0": ("y ~ (d1 ~ z1) + w", "d1", 0.0, 0.05),
    "wre_nocontrol_b0": ("y ~ (d1 ~ z1 + z2 + z3)", "d1", 0.0, 0.05),
}
_WRE_TWO = {
    "wre_two_d1": (_TWO, "d1", 0.0, 0.05),
    "wre_two_d2": (_TWO, "d2", 0.0, 0.05),
}


def _wre(wild, spec, **kw):
    formula, var, beta0, alpha = spec
    fit = _quiet(sp.ivreg, formula, data=wild, cluster="cl")
    out = sp.iv_wild_bootstrap(
        fit, wild, "cl", var, n_boot=9999, beta0=beta0, alpha=alpha, **kw
    )
    return fit, out


@pytest.fixture(scope="module")
def wre_one(wild):
    return {k: _wre(wild, spec) for k, spec in _WRE_ONE.items()}


@pytest.fixture(scope="module")
def wre_two(wild):
    return {k: _wre(wild, spec) for k, spec in _WRE_TWO.items()}


def _check_wre(key, fit, out, spec):
    ref = S[key]
    # 2^12 = 4,096 sign vectors on both sides: p is a count over 4,096
    assert out["enumerated"] and out["n_boot"] == ref["reps"] == 4096
    assert out["p_boot"] == pytest.approx(ref["p"], abs=1e-12)
    # boottest after ivregress, small: the CRV1 t with G/(G-1) (n-1)/(n-k)
    assert out["t_stat"] == pytest.approx(ref["t"], rel=1e-9)
    if spec[0] == _OVER:  # and the ivregress, small fit itself
        stata = _stata_by_name("iv_over")
        for name in ("d1", "w"):
            assert fit.params[name] == pytest.approx(stata[name][0], rel=1e-9)
            assert fit.std_errors[name] == pytest.approx(stata[name][1], rel=1e-9)


@pytest.mark.parametrize("key", list(_WRE_ONE))
def test_wre_one_endogenous_matches_boottest_exactly(wre_one, key):
    fit, out = wre_one[key]
    _check_wre(key, fit, out, _WRE_ONE[key])


@pytest.mark.parametrize("key", list(_WRE_TWO))
def test_wre_two_endogenous_matches_boottest_exactly(wre_two, key):
    """The other endogenous regressor is re-estimated by LIML under the null
    (with 2SLS there, the d2 p-value was 82 / 4,096 against boottest's 98)."""
    fit, out = wre_two[key]
    _check_wre(key, fit, out, _WRE_TWO[key])
    b, se = _stata_by_name("iv_two")[_WRE_TWO[key][1]]
    assert out["beta_hat"] == pytest.approx(b, rel=1e-9)
    assert out["se_cluster"] == pytest.approx(se, rel=1e-9)


def _p_at(wild, spec, beta0):
    formula, var, _, alpha = spec
    # only p is needed: skip the scan for further confidence-set pieces
    return _wre(wild, (formula, var, beta0, alpha), ci_range=0.0)[1]["p_boot"]


@pytest.mark.parametrize("key", ["wre_over_b0", "wre_over_l9375", "wre_just_b0"])
def test_wre_p_value_function_is_boottests_at_its_bounds(wild, key):
    """boottest's reported bounds are not the jumps of p(beta0); at those
    bounds the two p-value functions are still the same number."""
    spec = _WRE_ONE[key]
    for (lo, hi), (p_lo, p_hi) in zip(S[key]["ci"], S[key]["p_at_ci"]):
        assert _p_at(wild, spec, lo) == pytest.approx(p_lo, abs=1e-12)
        assert _p_at(wild, spec, hi) == pytest.approx(p_hi, abs=1e-12)


@pytest.mark.parametrize("key", ["wre_over_b0", "wre_over_l9375", "wre_just_b0"])
def test_wre_bounds_are_the_jumps_of_the_p_value(wild, wre_one, key):
    """Our bounds are the edges of {beta0 : p(beta0) > alpha}: p is above
    alpha just inside and at or below it just outside (the bisection stops
    at 1e-6 SE)."""
    spec = _WRE_ONE[key]
    _, out = wre_one[key]
    lo, hi = out["ci_boot"]
    d = 1e-5 * out["se_cluster"]
    alpha = spec[3]
    assert _p_at(wild, spec, lo + d) > alpha >= _p_at(wild, spec, lo - d)
    assert _p_at(wild, spec, hi - d) > alpha >= _p_at(wild, spec, hi + d)


def test_wre_confidence_bounds_are_close_to_boottests(wre_one, wre_two):
    """boottest interpolates where we take the exact set: the two sets of
    bounds are within 0.02 SE (observed 0.006 SE with one endogenous
    regressor, 0.018 with two). ``ci_boot`` is the piece that contains the
    estimate; the second piece of the d1 set is checked below."""
    fits = {**wre_one, **wre_two}
    for key, (_, out) in fits.items():
        first = S[key]["ci"][0]
        gap = np.abs(np.subtract(out["ci_boot"], first)) / out["se_cluster"]
        assert gap.max() < 0.02, key
    assert len(S["wre_two_d1"]["ci"]) == 2


WRE_DISJOINT = json.loads((FX / "wre_disjoint_Stata.json").read_text(encoding="utf-8"))


def test_wre_disjoint_confidence_set_matches_boottest(wre_two):
    """Under weak identification the d1 set is two intervals. Our scan finds
    both. boottest's default grid stops at 1.6644, where its own p-value is
    still 0.063 > alpha (p_at_ci), so its default second piece is cut at the
    end of its grid; with gridmin(-5) gridmax(10) gridpoints(301) it is
    [1.5975, 2.05] (2.05 a grid point). Our edges are exact: boottest's own
    p-value crosses alpha within 1e-6 of each of them (p_probe). The wide
    grid's bounds lie within 0.02 SE of ours (observed 0.0019 SE)."""
    _, out = wre_two["wre_two_d1"]
    alpha = 0.05
    assert out["ci_disjoint"] and len(out["ci_set"]) == 2
    assert out["ci_set"][0] == out["ci_boot"]
    se = out["se_cluster"]
    wide = WRE_DISJOINT["d1_wide"]["ci"]
    gap = np.abs(np.subtract(out["ci_set"], wide)) / se
    assert gap.max() < 0.02
    # the default boottest piece ends inside ours, where p > alpha
    default = WRE_DISJOINT["d1_default"]["ci"][1]
    assert default[1] < out["ci_set"][1][1]
    assert S["wre_two_d1"]["p_at_ci"][1][1] > alpha
    # each of our four edges is a jump of boottest's p-value function
    probes = WRE_DISJOINT["p_probe"]
    edges = [e for piece in out["ci_set"] for e in piece]
    for i, edge in enumerate(edges):
        (b_lo, p_lo), (b_hi, p_hi) = probes[2 * i], probes[2 * i + 1]
        assert b_lo < edge < b_hi
        if i % 2 == 0:  # lower edge: outside below, inside above
            assert p_lo <= alpha < p_hi
        else:
            assert p_hi <= alpha < p_lo
    _, out2 = wre_two["wre_two_d2"]
    assert not out2["ci_disjoint"] and out2["ci_set"] == [out2["ci_boot"]]


def test_wre_scan_options(wild):
    fit = _quiet(sp.ivreg, _OVER, data=wild, cluster="cl")
    kw = dict(n_boot=999, seed=1)
    a = sp.iv_wild_bootstrap(fit, wild, "cl", "d1", ci_range=0.0, **kw)
    b = sp.iv_wild_bootstrap(fit, wild, "cl", "d1", **kw)
    assert a["ci_set"] == [a["ci_boot"]] and a["ci_boot"] == b["ci_boot"]
    with pytest.raises(sp.MethodIncompatibility, match="ci_step"):
        sp.iv_wild_bootstrap(fit, wild, "cl", "d1", ci_step=0.0, **kw)
