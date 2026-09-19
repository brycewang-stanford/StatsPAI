"""Time-series family vs R and Stata on identical bytes.

References
----------
R: ``_fixtures/timeseries_R.json`` written by ``_generate_timeseries_R.R``
(urca 1.3.4, aTSA 3.1.2.1, vars 1.6.1, strucchange 1.5.4, sandwich 3.1.1,
rugarch 1.5.6, plm 2.6.7; versions are recorded in the fixture).
Stata: ``_fixtures/timeseries_Stata.json`` written by
``_fixtures/_generate_timeseries_stata.do`` (Stata 18 vecrank, var,
vargranger, irf, estat sbcusum, estat sbsingle, newey, arch, xtunitroot;
SSC egranger 1.0.6 and itsa 1.0.0 in a private ado directory).
Data: ``ts_coint.csv``, ``ts_var.csv``, ``ts_garch.csv``, ``ts_break.csv``,
``ts_its.csv``, ``ts_panel.csv`` from ``_generate_timeseries_data.py``.

Conventions pinned
------------------
* ``sp.johansen(lags=L)`` has L lagged differences = ``ca.jo(K = L + 1)`` =
  ``vecrank, lags(L + 1)``. trend 'c' = ca.jo ecdet "none" = vecrank
  trend(constant); 'rc' = "const" = rconstant; 'rt' = "trend" = rtrend;
  'n' / 'ct' exist in Stata only. Critical values are the Osterwald-Lenum
  table Stata uses (``_vecgetcv``); ca.jo ships a different table (its
  "none" row starts 8.18 / 17.95 / 31.52) -- statistics agree, tables differ.
* ``sp.engle_granger``: residual ADF without deterministic terms,
  ``egranger, lags(L)`` = ``ur.df(resid, type = "none", lags = L)``;
  critical values MacKinnon (2010) response surface at T = n - 1.
* ``sp.irf``: orthogonalised responses use the residual covariance with
  divisor T (Stata ``var``) or T - m (``vars::Psi``; Stata ``var, dfk``),
  chosen by ``sigma_df``.
* ``sp.granger_causality``: chi2 = Stata ``vargranger`` after ``var``;
  F = after ``var, small`` (and ``small dfk`` with ``se_df='r'``). vars'
  ``causality`` uses the same F but the system residual df K (T - m).
* CUSUM: ``efp(type = "Rec-CUSUM")``, statistic of ``sctest`` =
  ``estat sbcusum``; boundary coefficient from the Brownian-motion crossing
  probability.
* sup-F: ``Fstats(from = 0.15)`` grid (first-regime sizes floor(.15 n) ..
  n - floor(.15 n)); ``estat sbsingle`` sup-Wald = k x sup-F.
* ``method='global'``: ``breakpoints(h = 0.15, breaks = 5)``.
* ITS: Newey-West Bartlett(L = 4) without prewhitening;
  ``hac_small_sample=True`` = Stata ``newey`` (n/(n - k)).
* GARCH(1,1): presample ``m = mean(eps^2)``; 'stata' presample = ``arch``
  default arch0(xb); 'rugarch' = rugarch sGARCH recursion start.
* Panel unit roots: see ``sp.panel_unitroot(convention=...)``.

Tolerances: deterministic linear algebra at rel 1e-9 or tighter (observed
1e-12 .. 1e-15). GARCH is an MLE: the objective is asserted at 1e-10
(our log-likelihood evaluated at the reference optimum), parameters at the
reference optimiser's tolerance (stated per test).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.timeseries._critvals import JOHANSEN_CV, MACKINNON_2010, mackinnon_cv
from statspai.timeseries.garch import garch_loglik

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "timeseries_R.json").read_text(encoding="utf-8"))
S = json.loads((_FIX / "timeseries_Stata.json").read_text(encoding="utf-8"))

COINT = pd.read_csv(_FIX / "ts_coint.csv")
VARD = pd.read_csv(_FIX / "ts_var.csv")
BREAK = pd.read_csv(_FIX / "ts_break.csv")
ITS = pd.read_csv(_FIX / "ts_its.csv")
GARCH = pd.read_csv(_FIX / "ts_garch.csv")["r"].to_numpy()
PANEL = pd.read_csv(_FIX / "ts_panel.csv")


def _close(ours, ref, rtol=1e-10, atol=0.0):
    np.testing.assert_allclose(
        np.asarray(ours, float), np.asarray(ref, float), rtol=rtol, atol=atol
    )


# ===========================================================================
# Johansen
# ===========================================================================

_TREND_R = {"c": "none", "rc": "const", "rt": "trend"}
_TREND_STATA = {
    "n": "none",
    "rc": "rconstant",
    "c": "constant",
    "rt": "rtrend",
    "ct": "trend",
}


@pytest.mark.parametrize("lags", [1, 2])
@pytest.mark.parametrize("trend", ["c", "rc", "rt"])
def test_johansen_matches_ca_jo(trend, lags):
    ref = R["johansen"][f"{_TREND_R[trend]}_K{lags + 1}"]
    Y = COINT[["y1", "y2", "y3"]]
    tr = sp.johansen(Y, lags=lags, trend=trend, test="trace")
    mx = sp.johansen(Y, lags=lags, trend=trend, test="maxeig")
    _close(tr.eigenvalues, ref["lambda"][:3], rtol=1e-11)
    _close(tr.test_stats, ref["trace"], rtol=1e-11)
    _close(mx.test_stats, ref["maxeig"], rtol=1e-11)
    V = np.asarray(ref["V"], float)
    # cointegrating vectors for the non-degenerate eigenvalues
    _close(tr.eigenvectors[:, :3], V[:, :3], rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("lags", [1, 2])
@pytest.mark.parametrize("trend", ["n", "rc", "c", "rt", "ct"])
def test_johansen_matches_vecrank(trend, lags):
    key = f"vecrank_{_TREND_STATA[trend]}_p{lags + 1}"
    Y = COINT[["y1", "y2", "y3"]]
    tr = sp.johansen(Y, lags=lags, trend=trend, test="trace")
    mx = sp.johansen(Y, lags=lags, trend=trend, test="maxeig")
    _close(tr.eigenvalues, S[key + "_lambda"], rtol=1e-10)
    _close(tr.test_stats, S[key + "_trace"], rtol=1e-10)
    _close(mx.test_stats, S[key + "_max"], rtol=1e-10)
    assert int(S[key + "_N"]) == len(Y) - lags - 1


def test_johansen_critical_values_are_statas_table():
    """Our Osterwald-Lenum table equals Stata's _vecgetcv cell by cell."""
    cases = ("n", "rc", "c", "rt", "ct")
    for test, key in (("trace", "trace"), ("maxeig", "max")):
        for lev, tag in ((0.05, "95"), (0.01, "99")):
            ref = np.asarray(S[f"vecgetcv_{key}_{tag}"], float).reshape(11, 5)
            ours = np.column_stack([JOHANSEN_CV[test][lev][c] for c in cases])
            _close(ours, ref, rtol=0, atol=0)


def test_johansen_rank_and_trend_names_accept_stata_aliases():
    Y = COINT[["y1", "y2", "y3"]]
    a = sp.johansen(Y, lags=1, trend="rconstant")
    b = sp.johansen(Y, lags=1, trend="rc")
    _close(a.test_stats, b.test_stats, rtol=0)
    # data built with one common stochastic trend among three series
    assert a.rank == 2
    with pytest.raises(ValueError):
        sp.johansen(Y, lags=1, alpha=0.10)
    with pytest.raises(ValueError):
        sp.johansen(Y, lags=1, test="max")


def test_johansen_trace_is_sum_of_maxeig_identity():
    Y = COINT[["y1", "y2", "y3"]]
    tr = sp.johansen(Y, lags=1, trend="c", test="trace")
    mx = sp.johansen(Y, lags=1, trend="c", test="maxeig")
    _close(tr.test_stats, np.cumsum(mx.test_stats[::-1])[::-1], rtol=1e-13)


# ===========================================================================
# Engle-Granger
# ===========================================================================

_EG_CASES = [
    # (R key, Stata key, variables, lags, trend)
    ("y1_y2_L0", "y1_y2_L0", ["y1", "y2"], 0, "c"),
    ("y1_y2_L2", "y1_y2_L2", ["y1", "y2"], 2, "c"),
    ("y1_y23_L1", "y1_y2_y3_L1", ["y1", "y2", "y3"], 1, "c"),
    ("w_y1_L1", "w_y1_L1", ["w", "y1"], 1, "c"),
    ("y1_y2_L1_ct", "y1_y2_L1_trend", ["y1", "y2"], 1, "ct"),
    ("y1_y2_L1_ctt", "y1_y2_L1_qtrend", ["y1", "y2"], 1, "ctt"),
]


@pytest.mark.parametrize("rkey,skey,variables,lags,trend", _EG_CASES)
def test_engle_granger_matches_egranger_and_ur_df(rkey, skey, variables, lags, trend):
    r = sp.engle_granger(COINT, variables=variables, lags=lags, trend=trend)
    _close(r.test_stats, R["engle_granger"][rkey]["stat"], rtol=1e-10)
    _close(r.eigenvectors, R["engle_granger"][rkey]["coef"], rtol=1e-10)
    _close(r.test_stats, S[f"eg_{skey}_Zt"], rtol=1e-10)
    _close(
        r.critical_values,
        [S[f"eg_{skey}_cv1"], S[f"eg_{skey}_cv5"], S[f"eg_{skey}_cv10"]],
        rtol=1e-12,
    )


def test_engle_granger_matches_atsa_coint_test():
    r = sp.engle_granger(COINT, variables=["y1", "y2"], lags=2)
    _close(r.test_stats, R["engle_granger"]["aTSA_y1_y2_L2_stat"], rtol=1e-10)


def test_engle_granger_decisions():
    assert sp.engle_granger(COINT, variables=["y1", "y2"], lags=1).rank == 1
    for a, j in ((0.01, 0), (0.05, 1), (0.10, 2)):
        r = sp.engle_granger(COINT, variables=["w", "y1"], lags=1, alpha=a)
        assert r.rank == int(r.test_stats < r.critical_values[j])
    with pytest.raises(ValueError):
        sp.engle_granger(COINT, variables=["y1", "y2"], alpha=0.2)
    with pytest.raises(ValueError):
        sp.engle_granger(COINT, variables=["y1", "y2"], trend="n")


def test_mackinnon_2010_worked_example():
    """MacKinnon (2010, p. 12): tau_ct, N = 5, T = 100, 5% -> -4.89111."""
    assert mackinnon_cv("ct", 5, 5, 100) == pytest.approx(-4.89111, abs=5e-6)
    # statsmodels' copy has two typos against the paper; ours matches the
    # paper and egranger (asserted cell by cell via the Stata CVs above).
    assert MACKINNON_2010["c"][(2, 1)][2] == -22.527
    assert MACKINNON_2010["c"][(3, 5)][1] == -8.5631


# ===========================================================================
# VAR: irf / granger_causality
# ===========================================================================

_RESP = ["gdp", "infl", "rate"]


@pytest.fixture(scope="module")
def var3():
    return sp.var(VARD, lags=2)


@pytest.fixture(scope="module")
def var3_r():
    return sp.var(VARD, lags=2, se_df="r")


def test_irf_matches_stata_irf_create(var3):
    for resp in _RESP:
        o = sp.irf(var3, periods=8, impulse="infl", response=resp)["irf"]
        s = sp.irf(var3, periods=8, impulse="infl", response=resp, orthogonal=False)
        c = sp.irf(var3, periods=8, impulse="infl", response=resp, cumulative=True)
        key = f"infl -> {resp}"
        _close(o[key], S[f"oirf_m1_infl_{resp}"], rtol=1e-9, atol=1e-15)
        _close(s["irf"][key], S[f"irf_m1_infl_{resp}"], rtol=1e-9, atol=1e-15)
        _close(c["irf"][key], S[f"coirf_m1_infl_{resp}"], rtol=1e-9, atol=1e-15)


def test_irf_unbiased_sigma_matches_vars_and_stata_dfk(var3, var3_r):
    ref_o = R["var"]["irf_ortho_infl"]
    ref_s = R["var"]["irf_simple_infl"]
    ref_c = R["var"]["irf_ortho_cum_infl"]
    for resp in _RESP:
        key = f"infl -> {resp}"
        o = sp.irf(var3_r, periods=8, impulse="infl", response=resp)["irf"][key]
        o2 = sp.irf(
            var3, periods=8, impulse="infl", response=resp, sigma_df="unbiased"
        )["irf"][key]
        s = sp.irf(var3, periods=8, impulse="infl", response=resp, orthogonal=False)
        c = sp.irf(var3_r, periods=8, impulse="infl", response=resp, cumulative=True)
        _close(o, ref_o[resp], rtol=1e-10, atol=1e-15)
        _close(o2, ref_o[resp], rtol=1e-10, atol=1e-15)
        _close(s["irf"][key], ref_s[resp], rtol=1e-10, atol=1e-15)
        _close(c["irf"][key], ref_c[resp], rtol=1e-10, atol=1e-15)
        _close(o, S[f"oirf_m2_infl_{resp}"], rtol=1e-9, atol=1e-15)


def _gstats(key, ncol):
    return np.asarray(S[key], float).reshape(9, ncol)


# Stata r(gstats) rows: gdp:infl gdp:rate gdp:ALL infl:gdp infl:rate
# infl:ALL rate:gdp rate:infl rate:ALL
_GROWS = [
    ("gdp", "infl"),
    ("gdp", "rate"),
    ("gdp", ["infl", "rate"]),
    ("infl", "gdp"),
    ("infl", "rate"),
    ("infl", ["gdp", "rate"]),
    ("rate", "gdp"),
    ("rate", "infl"),
    ("rate", ["gdp", "infl"]),
]


def test_granger_matches_vargranger_chi2_and_small(var3, var3_r):
    chi2 = _gstats("vargranger_chi2", 3)
    small = _gstats("vargranger_small", 4)
    dfk = _gstats("vargranger_small_dfk", 4)
    for i, (caused, causing) in enumerate(_GROWS):
        g = sp.granger_causality(var3, caused=caused, causing=causing)
        _close(g["chi2"], chi2[i, 0], rtol=1e-10)
        assert g["df1"] == chi2[i, 1]
        _close(g["chi2_p_value"], chi2[i, 2], rtol=1e-8)
        _close(g["F_stat"], small[i, 0], rtol=1e-10)
        assert g["df2"] == small[i, 2]
        _close(g["p_value"], small[i, 3], rtol=1e-8)
        g2 = sp.granger_causality(var3_r, caused=caused, causing=causing)
        _close(g2["F_stat"], dfk[i, 0], rtol=1e-10)
        _close(g2["p_value"], dfk[i, 3], rtol=1e-8)


def test_granger_bivariate_matches_vars_causality():
    v2 = sp.var(VARD[["gdp", "infl"]], lags=2, se_df="r")
    g = sp.granger_causality(v2, caused="gdp", causing="infl")
    _close(g["F_stat"], R["var"]["granger_biv_F"], rtol=1e-10)
    df1, df2 = R["var"]["granger_biv_df"]
    assert g["df1"] == df1
    # vars uses the system residual df K (T - m); reproduce its p-value
    assert df2 == 2 * g["df2"]
    from scipy import stats

    _close(stats.f.sf(g["F_stat"], df1, df2), R["var"]["granger_biv_p"], rtol=1e-8)


def test_irf_impact_identity(var3):
    """Orthogonalised impact response = Cholesky column of Sigma_u."""
    P = np.linalg.cholesky(var3.sigma_u.to_numpy())
    o = sp.irf(var3, periods=0)["irf"]
    for j, imp in enumerate(_RESP):
        for i, resp in enumerate(_RESP):
            _close(o[f"{imp} -> {resp}"][0], P[i, j], rtol=1e-14, atol=1e-16)


# ===========================================================================
# CUSUM / sup-F / Bai-Perron
# ===========================================================================


def test_cusum_matches_strucchange_and_sbcusum():
    c = sp.cusum_test(BREAK, y="y", x=["x"])
    sc = R["strucchange"]
    _close(c["cusum"], sc["cusum_process"][1:], rtol=1e-10, atol=1e-13)
    _close(c["statistic"], sc["cusum_stat"], rtol=1e-10)
    _close(c["p_value"], sc["cusum_p"], rtol=1e-10)
    _close(c["statistic"], S["sbcusum_y_x"], rtol=1e-10)
    # Stata's generate() path is the same process
    _close(c["cusum"], S["sbcusum_path"], rtol=1e-8, atol=1e-12)
    s1 = sp.cusum_test(BREAK, y="ys")
    _close(s1["statistic"], sc["cusum_stable_stat"], rtol=1e-10)
    _close(s1["p_value"], sc["cusum_stable_p"], rtol=1e-10)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_cusum_boundary_matches_strucchange_and_stata(alpha):
    c = sp.cusum_test(BREAK, y="y", x=["x"], alpha=alpha)
    ref = R["strucchange"]["cusum_bound"][[0.01, 0.05, 0.10].index(alpha)]
    # strucchange's boundary() root is found by uniroot at its default
    # tolerance (~6e-5); Stata prints the constants to 10 digits.
    _close(c["boundary_coef"], ref, rtol=1e-4)
    # Stata's printed constants (1.142972691 / .9479006054 / .8499248005)
    # sit <= 2.5e-6 relative from the root of strucchange's closed-form
    # crossing probability (which we reproduce to strucchange's own uniroot
    # tolerance). Mechanism not verified (plausibly a longer reflection
    # series on Stata's side); the statistic itself agrees at 1e-10.
    stata = dict(zip((0.01, 0.05, 0.10), S["sbcusum_cv"]))
    _close(c["boundary_coef"], stata[alpha], rtol=3e-6)
    assert c["reject"] == (c["p_value"] < alpha)


def test_supf_matches_fstats_and_sbsingle():
    r = sp.structural_break(BREAK, y="y", x=["x"], method="sup-f")
    sc = R["strucchange"]
    _close(r.f_path * 2, sc["Fstats_x"], rtol=1e-10)
    assert r.candidate_breaks[0] == sc["Fstats_x_from"]
    assert r.sup_break == sc["Fstats_x_breakpoint"]
    _close(r.sup_wald, S["sbsingle_y_x_swald"], rtol=1e-10)
    # Stata dates a break by the first observation of the new regime
    assert r.sup_break + 1 == S["sbsingle_y_x_breakdate"]
    r1 = sp.structural_break(BREAK, y="y", method="sup-f")
    _close(r1.f_path, sc["Fstats_1"], rtol=1e-10)
    assert r1.sup_break == sc["Fstats_1_breakpoint"]
    _close(r1.sup_wald, S["sbsingle_y_swald"], rtol=1e-10)
    assert r1.sup_break + 1 == S["sbsingle_y_breakdate"]


@pytest.mark.parametrize("col,x", [("ym", None), ("y", ["x"])])
def test_global_breakpoints_match_strucchange(col, x):
    tag = "ym" if col == "ym" else "yx"
    sc = R["strucchange"]
    r = sp.structural_break(BREAK, y=col, x=x, method="global", max_breaks=5)
    _close(r.rss_by_breaks, sc[f"bp_{tag}_RSS"], rtol=1e-10)
    _close(r.bic_by_breaks, sc[f"bp_{tag}_BIC"], rtol=1e-10)
    assert r.break_dates == list(np.atleast_1d(sc[f"bp_{tag}_breaks"]))
    for m, bps in enumerate(sc[f"bp_{tag}_by_m"], start=1):
        assert r.breaks_by_m[m] == list(np.atleast_1d(bps))


def test_global_breakpoints_identity_rss_decreasing():
    r = sp.structural_break(BREAK, y="ym", method="global", max_breaks=5)
    assert np.all(np.diff(r.rss_by_breaks) <= 1e-9)
    # planted mean shifts after observations 80 and 170
    assert len(r.break_dates) == 2
    assert abs(r.break_dates[0] - 80) <= 3 and abs(r.break_dates[1] - 170) <= 3


# ===========================================================================
# Interrupted time series
# ===========================================================================


def test_its_matches_lm_neweywest():
    r = sp.its(ITS, y="y", time="month", intervention=40, hac_lag=4)
    ref = R["its"]
    _close(r.coefficients["coef"], ref["coef"], rtol=1e-10)
    _close(r.coefficients["se"], ref["se_nw"], rtol=1e-10)
    r2 = sp.its(
        ITS, y="y", time="month", intervention=40, hac_lag=4, hac_small_sample=True
    )
    _close(r2.coefficients["se"], ref["se_nw_adj"], rtol=1e-10)


def test_its_matches_stata_newey_and_itsa():
    r = sp.its(
        ITS, y="y", time="month", intervention=40, hac_lag=4, hac_small_sample=True
    )
    # Stata order: month D tpost _cons
    b = np.asarray(S["newey_b"], float)
    V = np.asarray(S["newey_V"], float).reshape(4, 4)
    coef = r.coefficients["coef"].to_numpy()
    se = r.coefficients["se"].to_numpy()
    _close(coef[[1, 2, 3, 0]], b, rtol=1e-10)
    _close(se[[1, 2, 3, 0]], np.sqrt(np.diag(V)), rtol=1e-10)
    # itsa (1.0.0) fits by glm2 IRLS with vce(hac nwest) and vfactor(N/df):
    # same estimator, agreement at the IRLS convergence tolerance.
    bi = np.asarray(S["itsa_b"], float)
    Vi = np.asarray(S["itsa_V"], float).reshape(4, 4)
    _close(coef[[1, 2, 3]], bi[:3], rtol=1e-6)
    _close(se[[1, 2, 3]], np.sqrt(np.diag(Vi))[:3], rtol=1e-6)


# ===========================================================================
# GARCH(1,1)
# ===========================================================================


def _stata_arch(v):
    b = np.asarray(S[f"arch_{v}_b"], float)  # _cons, arch, garch, omega
    V = np.asarray(S[f"arch_{v}_V"], float).reshape(4, 4)
    order = [0, 3, 1, 2]  # -> mu, omega, alpha, beta
    return b[order], np.sqrt(np.diag(V))[order]


def test_garch_objective_matches_stata_arch_at_their_optimum():
    b, _ = _stata_arch("oim")
    _close(garch_loglik(GARCH, b, presample="stata"), S["arch_oim_ll"], rtol=1e-12)


def test_garch_objective_matches_rugarch_at_their_optimum():
    g = R["garch"]
    _close(garch_loglik(GARCH, g["coef"], presample="rugarch"), g["loglik"], rtol=1e-12)


@pytest.mark.parametrize("vce", ["oim", "opg", "robust"])
def test_garch_matches_stata_arch(vce):
    r = sp.garch(GARCH, presample="stata", vce=vce)
    b, se = _stata_arch(vce)
    ll_stata = S[f"arch_{vce}_ll"]
    # Same objective (asserted above at 1e-12); ours is at least as high:
    # Stata stops 2.6e-10 below our optimum in log-likelihood, and along the
    # flat omega/alpha direction that moves the parameters by <= 1e-5
    # relative. Our score at the optimum is < 1e-6.
    assert r.log_likelihood >= ll_stata - 1e-12
    _close(r.log_likelihood, ll_stata, rtol=1e-12)
    _close(r.params.to_numpy(), b, rtol=1e-5)
    se_ours = r.std_errors.to_numpy()
    if vce == "robust":
        # Stata's vce(robust) carries the N/(N-1) small-sample factor;
        # ours (as rugarch) is the plain Bollerslev-Wooldridge sandwich.
        se_ours = se_ours * np.sqrt(len(GARCH) / (len(GARCH) - 1))
    _close(se_ours, se, rtol=2e-5)


@pytest.mark.parametrize("vce", ["oim", "opg"])
def test_garch_se_formula_matches_stata_at_their_parameters(vce):
    """Covariance evaluated at Stata's own b: OPG agrees to 5e-7 (our scores
    are analytic, Stata's ml derivatives numerical); OIM to 1e-5 (Stata's
    numerical Hessian vs our difference of the analytic score)."""
    from statspai.timeseries.garch import _garch_filter

    b, se = _stata_arch(vce)
    sc = _garch_filter(b, GARCH, 1, 1, True, "stata", True)[3]
    if vce == "opg":
        V = np.linalg.inv(sc.T @ sc)
        _close(np.sqrt(np.diag(V)), se, rtol=1e-6)
    else:
        V = np.linalg.inv(_neg_hessian(b, "stata"))
        _close(np.sqrt(np.diag(V)), se, rtol=2e-5)


def _neg_hessian(th, presample):
    from statspai.timeseries.garch import _garch_filter

    def g(t):
        return _garch_filter(t, GARCH, 1, 1, True, presample, True)[3].sum(0)

    K = th.size
    M = np.empty((K, K))
    for i in range(K):
        h = 1e-5 * max(abs(th[i]), 1e-2)
        e = np.zeros(K)
        e[i] = h
        M[:, i] = (g(th + e) - g(th - e)) / (2 * h)
    return -(M + M.T) / 2


def test_garch_matches_rugarch():
    r = sp.garch(GARCH, presample="rugarch")
    g = R["garch"]
    # rugarch's solver stops ~2e-7 below the optimum in log-likelihood; ours
    # is the higher (converged) value on the SAME objective, and the
    # parameters differ by rugarch's convergence error (<= 2e-4 relative).
    assert r.log_likelihood >= g["loglik"]
    assert r.log_likelihood - g["loglik"] < 1e-5
    _close(r.params.to_numpy(), g["coef"], rtol=5e-4)
    # rugarch's vcov inverts .hessian2sided: a second difference of the
    # log-likelihood itself with step eps^(1/3)|x| (~6e-6|x|), whose
    # rounding noise is ~0.5% of the beta curvature here. At rugarch's own
    # parameters our Hessian (difference of the analytic score) differs from
    # theirs by <= 0.8%, while it agrees with Stata's to 4e-6 (test above):
    # the gap is rugarch's numerical Hessian, not the model.
    th = np.asarray(g["coef"], float)
    se_ours_at_theirs = np.sqrt(np.diag(np.linalg.inv(_neg_hessian(th, "rugarch"))))
    _close(se_ours_at_theirs, g["se"], rtol=1e-2)


def test_garch_score_is_zero_at_optimum_and_forecast_identity():
    r = sp.garch(GARCH)
    assert r.gradient_norm < 1e-6
    fc = r.forecast(3)
    om, a, b = r.omega, r.alpha[0], r.beta[0]
    s1 = om + a * r.residuals[-1] ** 2 + b * r.sigma2[-1]
    _close(fc[0], s1, rtol=1e-14)
    _close(fc[1], om + (a + b) * s1, rtol=1e-14)


# ===========================================================================
# Panel unit roots
# ===========================================================================


def _pu(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.panel_unitroot(PANEL, "y", id="id", time="time", **kw)


@pytest.mark.parametrize("trend,exo", [("c", "intercept"), ("ct", "trend")])
@pytest.mark.parametrize("dfcor", [False, True])
def test_panel_unitroot_matches_plm(trend, exo, dfcor):
    ref = R["purtest"][f"{exo}_dfcor{int(dfcor)}"]
    kw = dict(lags=1, trend=trend, convention="plm", dfcor=dfcor)
    llc = _pu(test="llc", **kw)
    _close(llc.statistic, ref["levinlin"][0], rtol=1e-10)
    _close(llc.p_value, ref["levinlin"][1], rtol=1e-8)
    ips = _pu(test="ips", **kw)
    _close(ips.statistic, ref["ips"][0], rtol=1e-10)
    _close(ips.individual_stats["t_stat"], ref["ips_trho"], rtol=1e-10)
    fi = _pu(test="fisher", **kw)
    _close(fi.statistic, ref["madwu"][0], rtol=1e-10)
    _close(fi.details["Z"], ref["invnormal"][0], rtol=1e-10)
    _close(fi.details["L"], ref["logit"][0], rtol=1e-10)
    _close(fi.details["Pm"], ref["Pm"][0], rtol=1e-10)
    for robust in (True, False):
        h = sp.panel_unitroot(
            PANEL,
            "y",
            test="hadri",
            trend=trend,
            convention="plm",
            dfcor=dfcor,
            robust=robust,
        )
        _close(h.statistic, ref[f"hadri_H{int(robust)}"][0], rtol=1e-10)


def test_panel_unitroot_llc_none_and_rw_match_plm():
    r = _pu(test="llc", lags=1, trend="n", convention="plm")
    _close(r.statistic, R["purtest"]["none_llc"][0], rtol=1e-10)
    rw = sp.panel_unitroot(PANEL, "yrw", test="ips", lags=1, convention="plm")
    _close(rw.statistic, R["purtest"]["rw_ips"][0], rtol=1e-10)


@pytest.mark.parametrize("trend", ["c", "ct"])
def test_panel_unitroot_matches_xtunitroot(trend):
    k = trend
    llc = _pu(test="llc", lags=1, trend=trend)
    d = llc.details
    _close(d["delta"], S[f"xtur_llc_{k}_delta"], rtol=1e-10)
    _close(d["se_delta"], S[f"xtur_llc_{k}_se_delta"], rtol=1e-10)
    _close(d["t_delta"], S[f"xtur_llc_{k}_td"], rtol=1e-10)
    _close(d["sbar"], S[f"xtur_llc_{k}_sbar"], rtol=1e-10)
    _close(d["sigma_eps2"], S[f"xtur_llc_{k}_Var_ep"], rtol=1e-10)
    _close(d["mu_adj"], S[f"xtur_llc_{k}_mu_adj"], rtol=1e-12)
    if trend == "c":
        _close(d["sigma_adj"], S[f"xtur_llc_{k}_sig_adj"], rtol=1e-12)
        _close(llc.statistic, S[f"xtur_llc_{k}_tds"], rtol=1e-10)
    else:
        # Stata's _xturllc trend table carries sigma* = .971 at T = 40
        # (plm: .871; the column is monotone .906, .871, .842).
        # Rebuild Stata's t* from our quantities with its interpolated value.
        w = (d["T_adj"] - 35) / 5
        sig_stata = (1 - w) * 0.906 + w * 0.971
        _close(sig_stata, S[f"xtur_llc_{k}_sig_adj"], rtol=1e-12)
        tds = (
            d["t_delta"]
            - 20
            * d["T_adj"]
            * d["sbar"]
            / d["sigma_eps2"]
            * d["se_delta"]
            * d["mu_adj"]
        ) / sig_stata
        _close(tds, S[f"xtur_llc_{k}_tds"], rtol=1e-10)
    ips = _pu(test="ips", lags=1, trend=trend)
    _close(ips.statistic, S[f"xtur_ips_{k}_wtbar"], rtol=1e-10)
    fi = _pu(test="fisher", lags=1, trend=trend)
    for key in ("P", "Z", "Pm"):
        _close(fi.details[key], S[f"xtur_fisher_{k}_{key}"], rtol=1e-10)
    _close(fi.details["p_P"], S[f"xtur_fisher_{k}_p_P"], rtol=1e-8)
    h = sp.panel_unitroot(PANEL, "y", test="hadri", trend=trend, robust=False)
    _close(h.statistic, S[f"xtur_hadri_{k}_z"], rtol=1e-10)
    hr = sp.panel_unitroot(PANEL, "y", test="hadri", trend=trend, robust=True)
    _close(hr.statistic, S[f"xtur_hadri_robust_{k}_z"], rtol=1e-10)


def test_panel_unitroot_llc_noconstant_matches_xtunitroot():
    r = _pu(test="llc", lags=1, trend="n")
    _close(r.details["t_delta"], S["xtur_llc_n_td"], rtol=1e-10)
    _close(r.statistic, S["xtur_llc_n_tds"], rtol=1e-10)


@pytest.mark.parametrize("trend", ["c", "ct"])
def test_fisher_logit_stata_constant_differs(trend):
    """Stata's L* uses k = 3(5N+3)/(pi^2 N (5N+2)); plm uses 5N+4 (t df 5N+4
    in both). Rebuild Stata's L* from our p-values: the constant is the only
    difference."""
    fi = _pu(test="fisher", lags=1, trend=trend)
    p = fi.individual_stats["p_value"].to_numpy()
    n = len(p)
    k_stata = 3 * (5 * n + 3) / (np.pi**2 * n * (5 * n + 2))
    L_stata = np.sqrt(k_stata) * np.sum(np.log(p / (1 - p)))
    _close(L_stata, S[f"xtur_fisher_{trend}_L"], rtol=1e-10)


def test_panel_unitroot_tables_equal_plm():
    from statspai.panel import unit_root as ur

    tab = R["purtest"]["tables"]
    assert list(ur._LLC_T) == tab["llc_T"]
    for key, name in (("n", "none"), ("c", "intercept"), ("ct", "trend")):
        ref = np.asarray(tab[f"llc_{name}"], float)
        _close(np.asarray(ur._LLC_ADJ[key]), ref, rtol=0, atol=0)
    assert list(ur._IPS_T) == tab["ips_T"]
    for key, name in (("c", "intercept"), ("ct", "trend")):
        for mom in ("mean", "var"):
            ref = np.asarray(
                [
                    [np.nan if v is None else v for v in row]
                    for row in tab[f"ips_{mom}_{name}"]
                ],
                float,
            )
            np.testing.assert_array_equal(np.asarray(ur._IPS[(key, mom)]), ref)


def test_mackinnon1994_pvalue_matches_plm_and_statsmodels():
    from statsmodels.tsa.adfvalues import mackinnonp

    from statspai.panel.unit_root import mackinnon1994_pvalue

    pad = R["purtest"]["padf1994"]
    for trend, name, smreg in (
        ("n", "none", "n"),
        ("c", "intercept", "c"),
        ("ct", "trend", "ct"),
    ):
        ours = [mackinnon1994_pvalue(t, trend) for t in pad["t"]]
        _close(ours, pad[name], rtol=1e-12)
        for t, p in zip(pad["t"], ours):
            sm = mackinnonp(t, regression=smreg)
            if 0.0 < sm < 1.0:  # statsmodels clips outside its fitted range
                _close(p, sm, rtol=1e-12)


def test_panel_unitroot_input_validation():
    with pytest.raises(ValueError):
        _pu(test="ips", trend="n")
    with pytest.raises(ValueError):
        _pu(test="hadri", trend="n")
    with pytest.raises(ValueError):
        _pu(test="ips", convention="eviews")


# ===========================================================================
# Bayesian VAR (original Minnesota prior, fixed covariance) -- T3 vs Stata
# ===========================================================================
#
# Stata's ``bayes, minnfixedcovprior: var`` samples the (exactly normal)
# coefficient posterior by Gibbs; ours is its closed form. The posterior
# means are compared within 4 Monte-Carlo standard errors (Stata's MCSE,
# 100,000 draws), the posterior SDs within 5 x sd / sqrt(2 N). The fixed
# covariance Sigma0 is deterministic and compared at 1e-10.


def _stata_bvar(tag):
    M = np.asarray(S[f"bvar_{tag}_summary"], float).reshape(21, 6)
    K, p = 3, 2
    mean = np.empty((2 * K + 1, K))
    sd = np.empty_like(mean)
    mcse = np.empty_like(mean)
    for i in range(K):  # equation
        for j in range(K):  # variable
            for lag in range(1, p + 1):
                row = i * 7 + j * p + (lag - 1)
                ours_row = (lag - 1) * K + j
                mean[ours_row, i], sd[ours_row, i], mcse[ours_row, i] = M[row, :3]
        mean[-1, i], sd[-1, i], mcse[-1, i] = M[i * 7 + 6, :3]
    return mean, sd, mcse


@pytest.mark.parametrize(
    "tag,kw",
    [
        ("arcov", {}),
        ("varcov", {"sigma": "var"}),
        (
            "arcov_custom",
            {"lambda1": 0.2, "lambda2": 0.3, "lambda3": 2.0, "lambda4": 10.0},
        ),
    ],
)
def test_bvar_matches_stata_bayes_var_minnfixedcov(tag, kw):
    r = sp.bvar(VARD, lags=2, **kw)
    cov = "varcov" if tag == "varcov" else "arcov"
    _close(
        r.sigma0, np.asarray(S[f"bvar_{tag}_sigma0"], float).reshape(3, 3), rtol=1e-10
    )
    mean, sd, mcse = _stata_bvar(tag)
    assert np.all(np.abs(r.coef - mean) <= 4 * mcse), cov
    _close(r.coef_sd, sd, rtol=5 / np.sqrt(2 * 100_000))


def test_bvar_identities():
    a = sp.bvar(VARD[["gdp", "infl", "rate"]], lags=2)
    b = sp.bvar(VARD[["rate", "infl", "gdp"]], lags=2)
    perm = [2, 1, 0]
    rows = [(lag * 3) + j for lag in range(2) for j in perm] + [6]
    # permuting the variables permutes the posterior (defect before 1.28.x)
    _close(b.coef[rows][:, perm], a.coef, rtol=1e-10, atol=1e-14)
    # diffuse prior -> OLS; dogmatic prior -> random-walk prior mean
    flat = sp.bvar(VARD, lags=2, lambda1=1e8, lambda2=1.0, lambda4=1e8)
    v = sp.var(VARD, lags=2)
    _close(flat.coef[:, 0], v.coefs["gdp"]["coef"].to_numpy(), rtol=1e-7, atol=1e-9)
    tight = sp.bvar(VARD, lags=2, lambda1=1e-8)
    prior = np.zeros((7, 3))
    prior[[0, 1, 2], [0, 1, 2]] = 1.0
    _close(tight.coef, prior, atol=1e-8)
