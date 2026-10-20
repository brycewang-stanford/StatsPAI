"""Known-truth and identity checks of ``sp.lee_bounds(trimming='exact')``.

``trimming='exact'`` keeps exactly ``(1 - q) n`` units of mass of the more
retained arm: the observations strictly beyond the trimming quantile in
full, the one(s) at the quantile with the fractional share still needed
(Lee 2009, estimator (6) and footnote 28; Tauchmann's tie rule for
``leebounds``). Until 2026-10-21 this was checked against Stata
``leebounds`` run from a copy whose source text had been patched so that its
thresholds were held exactly; that patch depended on the ado's literal
text (no licence stated) and was removed. The evidence is now:

* exact rational bounds on small designs, by hand;
* the identity ``trimmed mean = (1 / (1 - q)) * integral_0^{1-q} F_n^{-1}(u) du``
  (the mean of the empirical distribution below its ``1 - q`` quantile,
  Lee's estimand with the empirical distribution plugged in) on the two
  committed continuous-outcome fixtures that the retired rows used;
* Lee's Proposition 3 variance evaluated by hand in exact rational
  arithmetic on the small design, and its asymptotic validity on a
  simulated design with known population bounds;
* the shipped ``leebounds`` upper bound on ``misc_sens_attr.csv``, which
  takes the fractional tie branch, still pins the variance convention (in
  ``test_misc_sens_stata_parity.py``).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"


def _tiny(flip: bool = False) -> pd.DataFrame:
    # treated: 5 units, all retained, y = 1..5 (p1 = 1)
    # control: 10 units, 7 retained with mean 0 (p0 = 0.7) -> q = 0.3,
    # kept mass (1 - q) * 5 = 3.5
    y = [1, 2, 3, 4, 5] + [0, 0, 0, 1, -1, 2, -2] + [np.nan] * 3
    d = [1] * 5 + [0] * 10
    s = [1] * 12 + [0] * 3
    df = pd.DataFrame({"y": y, "d": d, "s": s})
    if flip:
        df["d"] = 1 - df["d"]
    return df


def _info(df, **kw):
    r = sp.lee_bounds(df, "y", "d", "s", trimming="exact", **kw)
    return r.model_info


def test_exact_bounds_on_rational_design():
    """Lower keeps 1, 2, 3 and half of 4: 8 / 3.5 = 16/7; upper keeps 5, 4,
    3 and half of 2: 13 / 3.5 = 26/7; the control mean is 0."""
    mi = _info(_tiny(), se_method="analytic")
    assert mi["trimming_fraction"] == pytest.approx(0.3, rel=1e-14)
    assert mi["lower_bound"] == pytest.approx(16 / 7, rel=1e-14)
    assert mi["upper_bound"] == pytest.approx(26 / 7, rel=1e-14)


def test_exact_bounds_control_trimmed_mirror():
    """Relabelling the arms trims the control arm; bounds become (-26/7, -16/7)."""
    mi = _info(_tiny(flip=True), se_method="analytic")
    assert mi["lower_bound"] == pytest.approx(-26 / 7, rel=1e-14)
    assert mi["upper_bound"] == pytest.approx(-16 / 7, rel=1e-14)


def test_exact_bounds_split_tied_mass_on_binary_outcome():
    """Binary outcome, 6 zeros and 4 ones retained in the treated arm
    (p1 = 1), control retention 0.75 with mean 0.5: q = 0.25, kept mass 7.5.
    Lower keeps the six zeros and 1.5 of the ones: 1.5 / 7.5 = 0.2; upper
    keeps the four ones and 3.5 of the zeros: 4 / 7.5 = 8/15."""
    y = [0] * 6 + [1] * 4 + [0, 1, 0, 1, 0, 1] + [np.nan] * 2
    d = [1] * 10 + [0] * 8
    s = [1] * 16 + [0] * 2
    mi = _info(pd.DataFrame({"y": y, "d": d, "s": s}), n_bootstrap=5)
    assert mi["lower_bound"] == pytest.approx(0.2 - 0.5, rel=1e-14)
    assert mi["upper_bound"] == pytest.approx(8 / 15 - 0.5, rel=1e-14)


def test_exact_analytic_variance_is_lee_prop3_by_hand():
    """Lee (2009, Proposition 3) with sample analogues, in exact rational
    arithmetic on the tiny design. Trimmed (treated) arm T: n = 15,
    E[T] = 1/3, E[ST] = 1/3, E[S | T=0] = 7/10, q = 3/10. The kept set
    carries a fractional weight, so its variance divides by the kept mass
    W = 7/2; the control variance divides by its count minus one."""
    F = Fraction
    n, e_t, e_st, s0, q = 15, F(1, 3), F(1, 3), F(7, 10), F(3, 10)
    y0 = [0, 0, 0, 1, -1, 2, -2]
    m0 = F(sum(y0), len(y0))
    v_other = sum((F(v) - m0) ** 2 for v in y0) / (len(y0) - 1) / F(7, 15)

    def v_trim(vals, wts, y_q):
        W = sum(wts)
        mu = sum(w * v for v, w in zip(vals, wts)) / W
        s2 = sum(w * (v - mu) ** 2 for v, w in zip(vals, wts)) / W
        gap2 = (y_q - mu) ** 2
        return (
            s2 * n / W
            + gap2 * q / (e_st * (1 - q))
            + gap2 * ((1 - s0) - q * (1 - e_t)) / (e_t * s0 * (1 - e_t))
        )

    half = F(1, 2)
    var_lo = (v_trim([1, 2, 3, 4], [1, 1, 1, half], F(4)) + v_other) / n
    var_up = (v_trim([5, 4, 3, 2], [1, 1, 1, half], F(2)) + v_other) / n
    mi = _info(_tiny(), se_method="analytic")
    assert mi["se_lower"] == pytest.approx(float(var_lo) ** 0.5, rel=1e-13)
    assert mi["se_upper"] == pytest.approx(float(var_up) ** 0.5, rel=1e-13)


def _quantile_integral_mean(y: np.ndarray, keep: float, top: bool) -> float:
    """(1 / keep) * integral over the kept share of the empirical quantile
    function, i.e. each order statistic y_(i) weighted by the length of
    [(i-1)/n, i/n] inside [0, keep] (or [1 - keep, 1] for the upper tail)."""
    y = np.sort(np.asarray(y, dtype=float))
    n = len(y)
    lo_edge, hi_edge = (1.0 - keep, 1.0) if top else (0.0, keep)
    i = np.arange(n)
    seg = np.clip(
        np.minimum((i + 1) / n, hi_edge) - np.maximum(i / n, lo_edge), 0, None
    )
    return float(np.sum(seg * y) / keep)


def _check_quantile_identity(df, y, treat, sel):
    sub = df[df[sel] == 1]
    p1 = df.loc[df[treat] == 1, sel].mean()
    p0 = df.loc[df[treat] == 0, sel].mean()
    assert p1 > p0  # the treated arm is trimmed in both fixtures
    q = (p1 - p0) / p1
    y1 = sub.loc[sub[treat] == 1, y].to_numpy()
    m0 = sub.loc[sub[treat] == 0, y].mean()
    lo = _quantile_integral_mean(y1, 1 - q, top=False) - m0
    up = _quantile_integral_mean(y1, 1 - q, top=True) - m0
    mi = sp.lee_bounds(
        df, y, treat, sel, trimming="exact", se_method="analytic"
    ).model_info
    np.testing.assert_allclose(
        [mi["lower_bound"], mi["upper_bound"]], [lo, up], rtol=1e-12
    )
    return mi


def test_exact_equals_quantile_function_integral_teffects_fixture():
    df = pd.read_csv(FIX / "teffects_cs.csv")
    _check_quantile_identity(df, "ys", "d", "s")


def test_exact_equals_quantile_function_integral_attrition_fixture():
    df = pd.read_csv(FIX / "misc_sens_attr.csv")
    mi = _check_quantile_identity(df, "y", "treat", "obs")
    r = sp.attrition_bounds(df, "y", "treat", "obs", trimming="exact")
    assert r["lower_bound"] == mi["lower_bound"]
    assert r["upper_bound"] == mi["upper_bound"]


def test_exact_bounds_and_analytic_se_recover_known_population_bounds():
    """S independent of Y within arm: P(S|D=1) = 0.9, P(S|D=0) = 0.6, so
    q = 1/3; Y | D=1 ~ N(1, 1), Y | D=0 ~ N(0, 1). Population Lee bounds are
    1 -+ phi(z) / (2/3) with z = Phi^{-1}(2/3) (truncated-normal means).
    1000 replications at n = 2000 (fixed seed): the mean estimate is within
    4 Monte Carlo SEs of the truth and the mean analytic SE is within 10% of
    the Monte Carlo SD (the SD's own relative sampling error is about
    1/sqrt(2 * 999) = 2.2%, so 10% is about 4.5 of those). Observed at this
    seed: 2.3 / 2.6 Monte Carlo SEs and ratios 1.008 / 1.041."""
    z = norm.ppf(2 / 3)
    truth = np.array([1 - 1.5 * norm.pdf(z), 1 + 1.5 * norm.pdf(z)])
    rng = np.random.default_rng(20261021)
    n, R = 2000, 1000
    est, se = np.empty((R, 2)), np.empty((R, 2))
    for r in range(R):
        D = rng.integers(0, 2, n)
        S = (rng.random(n) < np.where(D == 1, 0.9, 0.6)).astype(int)
        Y = rng.normal(D.astype(float), 1.0)
        Y[S == 0] = np.nan
        mi = sp.lee_bounds(
            pd.DataFrame({"y": Y, "d": D, "s": S}),
            "y",
            "d",
            "s",
            trimming="exact",
            se_method="analytic",
        ).model_info
        est[r] = mi["lower_bound"], mi["upper_bound"]
        se[r] = mi["se_lower"], mi["se_upper"]
    sd = est.std(axis=0, ddof=1)
    assert np.all(np.abs(est.mean(axis=0) - truth) < 4 * sd / np.sqrt(R))
    np.testing.assert_allclose(se.mean(axis=0) / sd, 1.0, atol=0.10)
