"""Round-2 time-series items: sup-F p-values, the Bai-Perron sequential
procedure, the CUSUM boundary constant and the Fisher L* constant.

References
----------
R: ``_fixtures/r2_ts_R.json`` written by ``_generate_r2_ts_R.R``
(strucchange 1.5.4 ``Fstats`` / ``sctest`` / ``pvalue.Fstats``; mbreaks
1.0.1 ``dosequa`` with ``pftest`` traced; versions stored in the fixture).
Stata: ``_fixtures/r2_ts_Stata.json`` written by
``_fixtures/_generate_r2_ts_stata.do`` (Stata 18 ``estat sbsingle`` and its
Mata ``pvalsup()``; SSC xtbreak 2.2 in a private ado directory).
Data: ``ts_break.csv`` (round 1) and ``r2_ts_break.csv``
(``_fixtures/_generate_r2_ts_data.py``).

Conventions pinned
------------------
* sup-F p-value: Hansen (1997) response surface p = 1 - chi2(theta_0 +
  theta_1 x; eta), interpolated linearly in pi_0, evaluated at
  strucchange's ``lambda = ((n - from) to) / (from (n - to))``. R and Stata
  use Hansen's own coefficient table; StatsPAI's table was recomputed by
  simulation from the paper (``scripts/generate_hansen_supf_table.py``,
  2026-10-20 clean-room rewrite), so the two are independent Monte Carlo
  approximations of the same asymptotic p-value function and agree to
  approximation accuracy, not to rounding. The budget is twice the maximum
  absolute error Hansen's Table 1 reports for his own approximation:
  0.0025 for p <= .10 (bound 0.005), 0.0058 over all p (bound 0.012). The
  interpolation rules themselves are checked exactly on the coefficients
  (``test_hansen_pvalue_identities``).
* Bai-Perron sequential: ``mbreaks::dosequa(prewhit = 0, robust = 0,
  hetdat = 1, hetvar = 0, eps1 = 0.15, m = 5, signif = 1..4)``; statistics
  on mbreaks' scale ``(T_j - 2q)(SSR_0 - SSR_1) / SSR_1``; critical values
  = mbreaks' ``supF_next_cv*`` tables. Dates are the size of the preceding
  regimes (Stata's ``breakdate`` is that + 1).
* xtbreak's F(l + 1 | l) is the Bai-Perron *test* (breaks estimated
  globally, one pooled sigma with T - (l + 2)q df), not the statistic the
  sequential *procedure* computes segment by segment: a documented
  convention difference, rebuilt here from our own SSRs.
* CUSUM boundary: ``a`` solves P(sup |W(t)| / (1 + 2t) > a) = alpha. The
  closed form strucchange evaluates is checked against an independent
  density-propagation computation of the crossing probability; Stata's
  hard-coded constants miss the root by up to 2.5e-6 (class 4).
* Fisher L*: ``k = 3(5N + 4) / (pi^2 N (5N + 2))`` with t(5N + 4) is the
  only constant consistent with matching the variance and the kurtosis of
  the t law to the sum of N standard logistic variables; Stata's ado uses
  5N + 3 in ``k`` (its own manual prints 5N + 4).

Tolerances: deterministic statistics 1e-10 rel (observed <= 1e-14);
Hansen sup-F p-values: the Monte Carlo budget above (observed 0.0021 for
p <= .10 and 0.0076 above, vs R and Stata).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import statspai as sp
from statspai.timeseries._break_tables import BP_SUPF_NEXT_CV
from statspai.timeseries.structural_break import (
    _bde_boundary,
    _bm_linear_crossing_pvalue,
    hansen_supf_pvalue,
)

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "r2_ts_R.json").read_text(encoding="utf-8"))
S = json.loads((_FIX / "r2_ts_Stata.json").read_text(encoding="utf-8"))
BR = pd.read_csv(_FIX / "ts_break.csv")
R2 = pd.read_csv(_FIX / "r2_ts_break.csv")

CASES = {
    "ts_ym": (BR, "ym", None),
    "ts_yx": (BR, "y", ["x"]),
    "r2_y3": (R2, "y3", None),
    "r2_yx": (R2, "yx", ["x"]),
    "r2_yx2": (R2, "yx2", ["x"]),
    "r2_wn": (R2, "wn", None),
}
LEVELS = (0.10, 0.05, 0.025, 0.01)  # mbreaks signif = 1, 2, 3, 4


# Twice the maximum absolute error of Hansen's (1997) Table 1: our table
# and his are independent simulations of the same asymptotic p-value.
HANSEN_P_BUDGET_SMALL = 0.005  # reference p <= .10
HANSEN_P_BUDGET = 0.012


def _p_close(ours, ref):
    ours, ref = float(ours), float(ref)
    budget = HANSEN_P_BUDGET_SMALL if ref <= 0.10 else HANSEN_P_BUDGET
    assert abs(ours - ref) <= budget, (ours, ref)


# ===========================================================================
# sup-F p-value (Hansen 1997)
# ===========================================================================


@pytest.mark.parametrize("case", list(CASES))
def test_supf_pvalue_matches_sctest_and_sbsingle(case):
    d, y, x = CASES[case]
    r = sp.structural_break(d, y=y, x=x, method="sup-f")
    ref = R["supf"][case]
    np.testing.assert_allclose(r.sup_wald, ref["stat"], rtol=1e-10)
    assert r.sup_break == ref["breakpoint"]
    _p_close(r.p_values, ref["p"])
    np.testing.assert_allclose(r.sup_wald, S[f"sbsingle_{case}_swald"], rtol=1e-10)
    _p_close(r.p_values, S[f"sbsingle_{case}_p"])


def test_hansen_pvalue_function_matches_strucchange_grid():
    for g in R["supf_grid"]:
        for x, p in zip(g["x"], g["p"]):
            _p_close(hansen_supf_pvalue(x, int(g["k"]), g["pi0"]), p)


def test_hansen_pvalue_function_matches_stata_pvalsup_grid():
    P = np.asarray(S["pvalsup_grid"], float).reshape(-1, 4)
    for k, pi0, x, p in P:
        _p_close(hansen_supf_pvalue(x, int(k), pi0), p)


def test_hansen_pvalue_identities():
    # pi0 = 0.5 is a single candidate: chi2(k) exactly
    from scipy.stats import chi2

    assert hansen_supf_pvalue(7.3, 3, 0.5) == pytest.approx(chi2.sf(7.3, 3), rel=1e-15)
    # lambda form and pi0 form agree: pi0 = 1 / (1 + sqrt(lambda))
    lam = (0.85 / 0.15) ** 2
    assert hansen_supf_pvalue(9.1, 2, lam) == pytest.approx(
        hansen_supf_pvalue(9.1, 2, 0.15), rel=1e-12
    )
    # monotone decreasing in the statistic
    ps = [hansen_supf_pvalue(x, 2, 0.15) for x in np.linspace(2, 30, 30)]
    assert np.all(np.diff(ps) < 0)


def test_simulate_pvalue_method_still_available():
    d, y, x = CASES["r2_wn"]
    a = sp.structural_break(d, y=y, x=x, method="sup-f", pvalue_method="simulate")
    b = sp.structural_break(d, y=y, x=x, method="sup-f")
    # two approximations of the same limit law: close, not equal
    assert abs(a.p_values - b.p_values) < 0.03
    with pytest.raises(ValueError):
        sp.structural_break(d, y=y, method="sup-f", pvalue_method="exact")


# ===========================================================================
# Bai-Perron sequential procedure (mbreaks::dosequa)
# ===========================================================================


def test_bp_critical_value_tables_equal_mbreaks():
    for eps, tab in R["bp_cv"].items():
        tab = np.asarray(tab, float)
        ours = np.vstack(
            [np.asarray(BP_SUPF_NEXT_CV[float(eps)][lev], float) for lev in LEVELS]
        )
        np.testing.assert_array_equal(ours, tab)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("sig", [1, 2, 3, 4])
def test_bai_perron_sequential_matches_dosequa(case, sig):
    d, y, x = CASES[case]
    ref = R["sequential"][f"{case}_s{sig}"]
    r = sp.structural_break(
        d, y=y, x=x, method="bai-perron", alpha=LEVELS[sig - 1], max_breaks=5
    )
    assert r.break_dates == [int(v) for v in np.atleast_1d(ref["dates"])]
    assert r.n_breaks == ref["nbreak"]
    # every statistic we compute is one mbreaks computed (pftest trace),
    # at the same date on the same segment length
    calls = {
        (int(T), int(dt)): st
        for st, T, dt in zip(
            np.atleast_1d(ref["calls_stat"]),
            np.atleast_1d(ref["calls_T"]),
            np.atleast_1d(ref["calls_date"]),
        )
    }
    edges_seen = [0, len(d)]
    for t in r.sequential_tests:
        edges = sorted(edges_seen)
        a, b = edges[t["segment"]], edges[t["segment"] + 1]
        key = (b - a, t["date"] - a)
        assert key in calls, key
        np.testing.assert_allclose(t["statistic"], calls[key], rtol=1e-10)
        if t["reject"]:
            edges_seen.append(t["date"])


def test_bai_perron_statistic_identity_and_stopping_rule():
    d, y, x = CASES["r2_y3"]
    r = sp.structural_break(d, y=y, method="bai-perron")
    # first step equals the full-sample sup-F on mbreaks' (Wald) scale
    s = sp.structural_break(d, y=y, method="sup-f")
    np.testing.assert_allclose(
        r.sequential_tests[0]["statistic"], s.sup_wald, rtol=1e-12
    )
    for t in r.sequential_tests:
        assert t["reject"] == (t["statistic"] >= t["critical_value"])
    # the three planted mean shifts (75, 150, 225) are found within 5
    assert len(r.break_dates) == 3
    assert np.max(np.abs(np.array(r.break_dates) - [75, 150, 225])) <= 5


def test_bai_perron_rejects_untabulated_settings():
    d, y, _ = CASES["r2_y3"]
    with pytest.raises(ValueError, match="min_segment"):
        sp.structural_break(d, y=y, method="bai-perron", min_segment=0.12)
    with pytest.raises(ValueError, match="alpha"):
        sp.structural_break(d, y=y, method="bai-perron", alpha=0.07)


def test_xtbreak_f_next_is_the_bp_test_convention():
    """xtbreak's F(2|1) uses the globally estimated break and one pooled
    sigma with T - 3 df; rebuilt from our SSRs it equals Stata's number,
    while the sequential procedure's segment statistic is mbreaks'."""
    y = BR["ym"].to_numpy()
    T = len(y)

    def ss(a, b):
        return float(np.sum((y[a:b] - y[a:b].mean()) ** 2))

    f = np.asarray(S["xtbreak_ts_ym_f"], float)
    g = sp.structural_break(BR, y="ym", method="global", max_breaks=2)
    b1 = g.breaks_by_m[1][0]
    b2 = sorted(g.breaks_by_m[2])
    s1 = ss(0, b1) + ss(b1, T)
    s2 = ss(0, b2[0]) + ss(b2[0], b2[1]) + ss(b2[1], T)
    np.testing.assert_allclose((T - 3) * (s1 - s2) / s2, f[1], rtol=1e-10)
    r = sp.structural_break(BR, y="ym", method="bai-perron")
    np.testing.assert_allclose(r.sequential_tests[0]["statistic"], f[0], rtol=1e-10)
    assert abs(r.sequential_tests[1]["statistic"] - f[1]) > 1.0


# ===========================================================================
# CUSUM boundary constant: exact root vs Stata's constants
# ===========================================================================


def _crossing_prob_numeric(a, M=50, G=400):
    """P(|W(t)| >= a (1 + 2t) for some t in [0, 1]) by density propagation.

    Independent of the series strucchange evaluates: Gauss-Legendre
    quadrature of the Gaussian transition density on the moving interval,
    with the exact Brownian-bridge non-crossing factor for each linear
    boundary between steps (double crossing within one step of 1/M is
    neglected).
    """
    dt = 1.0 / M
    u, w = np.polynomial.legendre.leggauss(G)

    def c(t):
        return a * (1.0 + 2.0 * t)

    c0, c1 = c(0.0), c(dt)
    x, wx = c1 * u, c1 * w
    f = (
        norm.pdf(x, scale=np.sqrt(dt))
        * (1 - np.exp(-2 * c0 * (c1 - x) / dt))
        * (1 - np.exp(-2 * c0 * (c1 + x) / dt))
    )
    for k in range(1, M):
        c0, c1 = c(k * dt), c((k + 1) * dt)
        yv, wy = c1 * u, c1 * w
        X, Y = x[:, None], yv[None, :]
        K = np.exp(-((Y - X) ** 2) / (2 * dt)) / np.sqrt(2 * np.pi * dt)
        K *= (1 - np.exp(-2 * (c0 - X) * (c1 - Y) / dt)) * (
            1 - np.exp(-2 * (c0 + X) * (c1 + Y) / dt)
        )
        f = (f * wx) @ K
        x, wx = yv, wy
    return 1.0 - float(np.sum(f * wx))


def test_cusum_boundary_is_exact_root_and_stata_constant_is_not():
    a = _bde_boundary(0.05)
    # the closed form (strucchange's pvalue.efp) is the crossing probability
    assert abs(_crossing_prob_numeric(a) - 0.05) < 2e-11
    assert abs(_bm_linear_crossing_pvalue(a) - 0.05) < 1e-14
    # Stata's printed 5% constant misses the root: its crossing
    # probability is 0.0499991, 8.9e-7 below alpha (class 4)
    stata95 = 0.9479006054
    p_stata = _crossing_prob_numeric(stata95)
    assert abs(p_stata - _bm_linear_crossing_pvalue(stata95)) < 2e-11
    assert 5e-7 < 0.05 - p_stata < 1.5e-6
    assert (stata95 - a) / a == pytest.approx(2.5e-6, rel=0.01)


# ===========================================================================
# Fisher L*: the 5N + 4 constant
# ===========================================================================


@pytest.mark.parametrize("N", [1, 2, 5, 12, 50])
def test_fisher_logit_constant_matches_t_moments(N):
    nu = 5 * N + 4
    k_plm = 3 * (5 * N + 4) / (np.pi**2 * N * (5 * N + 2))
    k_stata = 3 * (5 * N + 3) / (np.pi**2 * N * (5 * N + 2))
    var_L = N * np.pi**2 / 3  # sum of N standard logistic variables
    exkurt_L = 1.2 / N  # excess kurtosis of that sum
    # kurtosis matching fixes nu = 5N + 4 ...
    assert exkurt_L == pytest.approx(6.0 / (nu - 4), rel=1e-14)
    # ... and variance matching then fixes k (5N + 4, not 5N + 3)
    assert k_plm * var_L == pytest.approx(nu / (nu - 2), rel=1e-14)
    assert abs(k_stata * var_L - nu / (nu - 2)) > 1e-3 / N


def test_panel_unitroot_uses_5n_plus_4():
    panel = pd.read_csv(_FIX / "ts_panel.csv")
    fi = sp.panel_unitroot(
        panel, "y", id="id", time="time", test="fisher", lags=1, trend="c"
    )
    p = fi.individual_stats["p_value"].to_numpy()
    n = len(p)
    k = 3 * (5 * n + 4) / (np.pi**2 * n * (5 * n + 2))
    np.testing.assert_allclose(
        fi.details["L"], np.sqrt(k) * np.sum(np.log(p / (1 - p))), rtol=1e-12
    )
    assert fi.details["df_L"] == 5 * n + 4
