"""Varying-slope HDFE absorption — Stata ``reghdfe`` parity and unit tests.

Covers ``sp.hdfe_ols`` (``statspai.panel.feols.feols``) and the native
absorber kernel for Stata's ``absorb(i.g#c.x)`` / ``absorb(i.g##c.x)`` and
fixest's ``g[[x]]`` / ``g[x]``.

Reference numbers
-----------------
Every ``STATA[...]`` entry below is a literal transcription of
``reghdfe``'s output (Stata 18 MP, ``reghdfe`` installed) run on the exact
fixture built by :func:`_bal`, :func:`_unb` and :func:`_deg`. The fixtures
are pure NumPy with a fixed seed, exported to CSV and read by Stata, so
both sides see identical data (round-trip error < 1e-15).

The alternating-projection absorber converges to ~1e-8, not bit-exactly,
so parity is asserted at ``rtol=1e-6``.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from statspai.panel.feols import (
    _factorize_multi,
    _materialize_rhs,
    _parse_formula,
    feols,
)
from statspai.panel.hdfe import Absorber, SlopeSpec, absorb_ols

RTOL = 1e-6


# ======================================================================
# Fixtures (identical to what Stata was fed)
# ======================================================================


def _build_panel() -> tuple:
    """Build the balanced and unbalanced panels from one rng stream.

    The draw order here is what Stata's CSV was generated from, so it must
    not be reordered: FE draws, then x1/x2/d/w, then the error term, then
    the unbalanced keep mask.
    """
    rng = np.random.default_rng(0)
    n_c, n_t = 30, 20
    county = np.repeat(np.arange(n_c), n_t)
    year = np.tile(np.arange(n_t), n_c)
    prov = county // 10
    pref = county // 5
    n = n_c * n_t

    cf = rng.normal(0, 1, n_c)
    yf = rng.normal(0, 1, n_t)
    slope = rng.normal(0, 0.1, 6)
    pslope = rng.normal(0, 0.2, 3)

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    d = (rng.random(n) < 0.45).astype(float)
    w = rng.uniform(0.5, 3.0, n)

    y = (
        0.30 * d
        + 0.50 * x1
        - 0.25 * x2
        + cf[county]
        + yf[year]
        + slope[pref] * year
        + pslope[prov] * x1
        + rng.normal(0, 0.7, n)
    )
    bal = pd.DataFrame(
        dict(
            county=county, prov=prov, pref=pref, year=year, y=y, d=d, x1=x1, x2=x2, w=w
        )
    )
    keep = rng.random(n) > 0.18
    unb = bal.loc[keep].reset_index(drop=True)
    return bal, unb


def _bal() -> pd.DataFrame:
    """Balanced 30-county x 20-year panel with prefecture-varying trends."""
    return _build_panel()[0].copy()


def _unb() -> pd.DataFrame:
    """The same panel with ~18% of rows dropped (unbalanced)."""
    return _build_panel()[1].copy()


def _deg() -> pd.DataFrame:
    """Degenerate fixture: 40 singleton groups + a zero-variance slope var."""
    rng = np.random.default_rng(7)
    m = 240
    g = np.concatenate([np.repeat(np.arange(40), 5), np.arange(40, 80)])
    big = np.repeat(np.arange(12), 20)
    z = rng.normal(0, 1, m)
    zc = g.astype(float)  # constant within every level of g
    d = (rng.random(m) < 0.5).astype(float)
    y = 0.4 * d + rng.normal(0, 1, m) + 0.1 * g
    return pd.DataFrame(dict(g=g, big=big, z=z, zc=zc, d=d, y=y))


# ======================================================================
# Stata reghdfe reference values: (b, se, N, df_r, df_a)
# ======================================================================

STATA = {
    # reghdfe y d, absorb(county i.pref#c.year)
    ("c1", "d"): (0.323136775845525, 0.09724858976430574, 600, 563, 36),
    # reghdfe y d, absorb(i.pref#c.year)               -- slope only, no FE
    ("c2", "d"): (0.00862862820604397, 0.10122862103126914, 600, 593, 6),
    # reghdfe y d, absorb(i.pref##c.year)              -- intercepts + slope
    ("c3", "d"): (0.23952113888844315, 0.11045883843065628, 600, 587, 12),
    # reghdfe y d x2, absorb(county year i.pref#c.year)
    ("c4", "d"): (0.30208638463884957, 0.07524651815644012, 600, 543, 55),
    ("c4", "x2"): (-0.21654123163268332, 0.039312264574422, 600, 543, 55),
    # reghdfe y d x2, absorb(county i.pref#c.year i.prov#c.x1)
    ("c5", "d"): (0.24998040780656489, 0.08645870293655925, 600, 559, 39),
    ("c5", "x2"): (-0.1932151766762099, 0.04493780897879949, 600, 559, 39),
    # reghdfe y d [aw=w], absorb(county i.pref#c.year)
    ("c6", "d"): (0.30048255562600906, 0.09798806923417776, 600, 563, 36),
    # reghdfe y d, absorb(county i.pref#c.year) vce(cluster county)
    ("c7", "d"): (0.323136775845525, 0.10802818595081916, 600, 29, 6),
    # reghdfe y d x1 x2, absorb(county i.pref#c.year)
    ("c11", "d"): (0.24948228134046704, 0.08783146791695513, 600, 561, 36),
    ("c11", "x1"): (0.47427102657785614, 0.04282009735296857, 600, 561, 36),
    ("c11", "x2"): (-0.17633956746387178, 0.04548057736903782, 600, 561, 36),
    # reghdfe y d, absorb(county i.prov#c.year i.pref#c.x1 i.prov#c.x2)
    ("c12", "d"): (0.2801969392553254, 0.09159272508660533, 600, 557, 42),
    # unbalanced: reghdfe y d, absorb(county i.pref#c.year)
    ("c8", "d"): (0.2787949249610524, 0.11015657588118802, 495, 458, 36),
    # unbalanced: reghdfe y d x2, absorb(county year i.pref#c.year i.prov#c.x1)
    ("c8b", "d"): (0.24959771458124452, 0.06625322566940176, 495, 435, 58),
    ("c8b", "x2"): (-0.22507606047027728, 0.03401495902532782, 495, 435, 58),
    # degenerate: reghdfe y d, absorb(i.g#c.z)   -- 40 singletons dropped
    ("c9", "d"): (2.040274765523068, 0.24656219776791968, 200, 159, 40),
    # degenerate: reghdfe y d, absorb(big i.g#c.zc) -- zc constant within g
    ("c10", "d"): (0.1725147096991063, 0.14840738645613744, 200, 150, 49),
    # degenerate: reghdfe y d, absorb(big i.g#c.z)
    ("c10b", "d"): (0.2190273011292541, 0.14754593147085168, 200, 149, 50),
    # reghdfe y d, absorb(prov#year)
    ("f4", "d"): (0.3496560234263733, 0.12101125288349734, 600, 539, 60),
    # reghdfe y d c.x1#c.x2, absorb(county)
    ("f5", "d"): (0.3680132749959259, 0.11053454050010415, 600, 568, 30),
    ("f5", "x1:x2"): (-0.11015938746479664, 0.0549364457245208, 600, 568, 30),
    # reghdfe y d x1 c.d#c.x1, absorb(county)
    ("f6", "d"): (0.3399144254110433, 0.1036927332995579, 600, 567, 30),
    ("f6", "x1"): (0.4892893677870562, 0.06652047238694347, 600, 567, 30),
    ("f6", "d:x1"): (-0.07454178787225153, 0.10487891680746364, 600, 567, 30),
    # reghdfe y d i.pref, absorb(year)
    ("f7", "d"): (0.2972344796060324, 0.10126830318084844, 600, 574, 20),
    ("f7", "pref::1"): (-1.2651716056320368, 0.17233274791211364, 600, 574, 20),
    ("f7", "pref::5"): (-0.1412229727783948, 0.17261815254352522, 600, 574, 20),
    # reghdfe y d i.pref#c.year, absorb(county)  -- slopes as REGRESSORS
    ("f8", "d"): (0.3231367758455251, 0.09724858976430574, 600, 563, 30),
    ("f8", "pref::0#year"): (0.12930206037534608, 0.01990635029174161, 600, 563, 30),
    ("f8", "pref::3#year"): (0.16041096783662645, 0.01991075449656352, 600, 563, 30),
}

# case -> (formula, fixture name, feols kwargs)
CASES = {
    "c1": ("y ~ d | county + i.pref#c.year", "bal", {}),
    "c2": ("y ~ d | i.pref#c.year", "bal", {}),
    "c3": ("y ~ d | i.pref##c.year", "bal", {}),
    "c4": ("y ~ d + x2 | county + year + i.pref#c.year", "bal", {}),
    "c5": ("y ~ d + x2 | county + i.pref#c.year + i.prov#c.x1", "bal", {}),
    "c6": ("y ~ d | county + i.pref#c.year", "bal", {"weights": "w"}),
    "c7": ("y ~ d | county + i.pref#c.year", "bal", {"cluster": "county"}),
    "c11": ("y ~ d + x1 + x2 | county + i.pref#c.year", "bal", {}),
    "c12": (
        "y ~ d | county + i.prov#c.year + i.pref#c.x1 + i.prov#c.x2",
        "bal",
        {},
    ),
    "c8": ("y ~ d | county + i.pref#c.year", "unb", {}),
    "c8b": (
        "y ~ d + x2 | county + year + i.pref#c.year + i.prov#c.x1",
        "unb",
        {},
    ),
    "c9": ("y ~ d | i.g#c.z", "deg", {}),
    "c10": ("y ~ d | big + i.g#c.zc", "deg", {}),
    "c10b": ("y ~ d | big + i.g#c.z", "deg", {}),
    "f4": ("y ~ d | prov^year", "bal", {}),
    "f5": ("y ~ d + x1:x2 | county", "bal", {}),
    "f6": ("y ~ d*x1 | county", "bal", {}),
    "f7": ("y ~ d + i.pref | year", "bal", {}),
    "f8": ("y ~ d + i.pref#c.year | county", "bal", {}),
}

_FIXTURES = {"bal": _bal, "unb": _unb, "deg": _deg}


def _fit(case: str):
    formula, fixture, kw = CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return feols(formula, data=_FIXTURES[fixture](), **kw)


@pytest.mark.parametrize("case,term", sorted(STATA))
def test_reghdfe_parity(case, term):
    """Coefficients and SEs match Stata reghdfe to 1e-6 relative."""
    b_ref, se_ref, n_ref, _df_r, _df_a = STATA[(case, term)]
    res = _fit(case)
    assert res.n_obs == n_ref
    assert float(res.params[term]) == pytest.approx(b_ref, rel=RTOL)
    assert float(res.std_errors[term]) == pytest.approx(se_ref, rel=RTOL)


@pytest.mark.parametrize(
    "case",
    # Cases whose reghdfe run reports the OLS residual dof. Clustered runs
    # (c7) report N_clusters-1 instead, a different convention.
    [c for c in CASES if c != "c7"],
)
def test_reghdfe_df_resid_parity(case):
    """``df_resid`` matches ``e(df_r)``, which pins down the df_a accounting."""
    ref_df_r = next(v[3] for (c, _t), v in STATA.items() if c == case)
    assert _fit(case).df_resid == ref_df_r


def test_slope_absorbs_n_levels_not_n_minus_one():
    """A slope term costs G parameters; an ordinary FE costs G-1 after the first.

    reghdfe reports e(df_a)=36 for absorb(county i.pref#c.year) with 30
    counties and 6 prefectures — 30 + 6, not 30 + 5.
    """
    res = _fit("c1")
    assert res.dof_fe == 36
    assert res.n_fe == [30]
    # Same group count absorbed as an ORDINARY FE instead costs one less.
    ordinary = feols("y ~ d | county + pref", data=_bal())
    assert ordinary.dof_fe == 30 + 6 - 1 == 35


def test_intercept_bearing_slope_costs_two_g():
    """``i.g##c.x`` absorbs G intercepts + G slopes (reghdfe e(df_a)=12)."""
    assert _fit("c3").dof_fe == 12
    assert _fit("c2").dof_fe == 6  # slope-only: no intercepts absorbed


def test_weighted_clustered_multi_slope_matches_reghdfe():
    """The full combination: aweights + cluster + two slope terms.

    reghdfe y d x2 [aw=w], absorb(county i.pref#c.year i.prov#c.x1)
                           vce(cluster county)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = feols(
            "y ~ d + x2 | county + i.pref#c.year + i.prov#c.x1",
            data=_bal(),
            weights="w",
            cluster="county",
        )
    assert float(res.params["d"]) == pytest.approx(0.2371327317925, rel=RTOL)
    assert float(res.std_errors["d"]) == pytest.approx(0.107835707140443, rel=RTOL)
    assert float(res.params["x2"]) == pytest.approx(-0.227377772916382, rel=RTOL)
    assert float(res.std_errors["x2"]) == pytest.approx(0.045976279926143, rel=RTOL)
    # reghdfe reports e(df_a)=9 (county dropped as cluster-nested, 6+3 slopes).
    # StatsPAI's dof_fe_cluster additionally charges the constant, a
    # pre-existing convention: absorb(county) vce(cluster county) gives
    # reghdfe df_a=0 vs StatsPAI 1. The SE, which is what the convention
    # feeds, matches above.
    assert res.cluster_info["dof_fe_cluster"] == 9 + 1


def test_known_gap_nested_intercept_bearing_slope_dof():
    """KNOWN DIVERGENCE: no nested-FE redundancy detection in dof_fe.

    ``reghdfe y d [aw=w], absorb(county i.pref##c.year)`` on the unbalanced
    panel reports ``e(df_a)=36``: because ``pref == county // 5``, the 6
    ``pref`` intercepts introduced by ``##`` are *entirely* redundant given
    ``county``, and reghdfe's dof table charges them 0. StatsPAI charges
    ``30 + 6 - 1 + 6 = 41``, so ``df_resid`` is 453 vs 458 and the reported
    SE is 0.111100999576 vs reghdfe's 0.110492888826627 (5.5e-3 relative).

    This is NOT specific to varying slopes — StatsPAI has never done
    nested-FE redundancy detection, and plain ``y ~ d | county + pref``
    already reports dof_fe=35 against reghdfe's df_a=30 (asserted below).
    The projection itself is exact: the coefficient matches to 2e-15.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = feols("y ~ d | county + i.pref##c.year", data=_unb(), weights="w")

    # The absorbed projection is correct — only the dof bookkeeping differs.
    assert float(res.params["d"]) == pytest.approx(0.247730742816157, rel=RTOL)
    assert res.n_obs == 495

    # Documented divergence, asserted so it cannot drift unnoticed.
    assert res.dof_fe == 41  # reghdfe e(df_a) == 36
    assert res.df_resid == 453  # reghdfe e(df_r) == 458
    assert float(res.std_errors["d"]) == pytest.approx(0.111100999576, rel=1e-9)

    # Same gap with ordinary FEs only, i.e. pre-existing and unrelated to slopes.
    assert feols("y ~ d | county + pref", data=_bal()).dof_fe == 35  # reghdfe: 30


# ======================================================================
# Kernel correctness against dense linear algebra
# ======================================================================


def _dense_resid(col, design, weights=None):
    """Residual of ``col`` on ``design`` via least squares (ground truth)."""
    if weights is None:
        beta, *_ = np.linalg.lstsq(design, col, rcond=None)
        return col - design @ beta
    sw = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(design * sw[:, None], col * sw, rcond=None)
    return col - design @ beta


def _slope_design(group, x, with_intercept):
    levels = np.unique(group)
    blocks = [((group == lv) * x).astype(float) for lv in levels]
    if with_intercept:
        blocks += [(group == lv).astype(float) for lv in levels]
    return np.column_stack(blocks)


@pytest.mark.parametrize("with_intercept", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_slope_projection_matches_dense_ols(with_intercept, weighted):
    """One slope term is an exact projection — compare to lstsq residuals."""
    rng = np.random.default_rng(3)
    n = 400
    g = rng.integers(0, 12, n)
    x = rng.normal(size=n)
    col = rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, n) if weighted else None

    ab = Absorber(
        None,
        weights=w,
        drop_singletons=False,
        slopes=[SlopeSpec(g, x, with_intercept, "g#x")],
        n_obs=n,
    )
    got = ab.demean(col)
    want = _dense_resid(col, _slope_design(g, x, with_intercept), w)
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_slope_plus_fe_matches_dense_ols():
    """FE + slope alternating projection converges to the joint projection."""
    rng = np.random.default_rng(4)
    n = 600
    firm = rng.integers(0, 25, n)
    g = rng.integers(0, 5, n)
    x = rng.normal(size=n)
    col = rng.normal(size=n)

    ab = Absorber(
        firm.reshape(-1, 1),
        drop_singletons=False,
        slopes=[SlopeSpec(g, x, False, "g#x")],
        tol=1e-12,
    )
    got = ab.demean(col)
    design = np.column_stack(
        [
            np.column_stack([(firm == f).astype(float) for f in np.unique(firm)]),
            _slope_design(g, x, False),
        ]
    )
    np.testing.assert_allclose(got, _dense_resid(col, design), atol=1e-7)


def test_slope_projection_is_idempotent():
    """Projections are idempotent: sweeping twice changes nothing."""
    rng = np.random.default_rng(5)
    n = 300
    g = rng.integers(0, 8, n)
    x = rng.normal(size=n)
    col = rng.normal(size=n)
    ab = Absorber(
        None,
        drop_singletons=False,
        n_obs=n,
        slopes=[SlopeSpec(g, x, True, "g#x")],
    )
    once = ab.demean(col)
    np.testing.assert_allclose(ab.demean(once), once, atol=1e-12)


def test_absorbed_slope_columns_are_annihilated():
    """The absorbed slope columns themselves project to zero."""
    rng = np.random.default_rng(6)
    n = 250
    g = rng.integers(0, 6, n)
    x = rng.normal(size=n)
    ab = Absorber(
        None,
        drop_singletons=False,
        n_obs=n,
        slopes=[SlopeSpec(g, x, False, "g#x")],
    )
    for lv in np.unique(g):
        np.testing.assert_allclose(
            ab.demean(((g == lv) * x).astype(float)), 0.0, atol=1e-10
        )


def test_lsmr_solver_matches_map_solver():
    """The Krylov path builds the same absorbed span as alternating projections."""
    rng = np.random.default_rng(11)
    n = 400
    firm = rng.integers(0, 15, n)
    g = rng.integers(0, 6, n)
    x = rng.normal(size=n)
    col = rng.normal(size=n)
    slopes = [SlopeSpec(g, x, True, "g#x")]
    a_map = Absorber(
        firm.reshape(-1, 1), drop_singletons=False, slopes=slopes, tol=1e-12
    )
    a_lsmr = Absorber(
        firm.reshape(-1, 1),
        drop_singletons=False,
        slopes=slopes,
        solver="lsmr",
        tol=1e-12,
    )
    np.testing.assert_allclose(a_map.demean(col), a_lsmr.demean(col), atol=1e-6)


# ======================================================================
# Degenerate levels — must not divide by ~0
# ======================================================================


def test_singleton_level_annihilates_without_nan():
    """A one-observation level has no identified slope; residual is finite."""
    g = np.array([0, 0, 0, 1, 2, 2, 2])  # levels 1 is a singleton
    x = np.array([1.0, 2.0, 3.0, 5.0, 1.0, 4.0, 2.0])
    col = np.array([1.0, 3.0, 2.0, 9.0, 0.5, 2.0, 1.0])
    ab = Absorber(
        None,
        drop_singletons=False,
        n_obs=7,
        slopes=[SlopeSpec(g, x, True, "g#x")],
    )
    out = ab.demean(col)
    assert np.all(np.isfinite(out))
    # [1, x] fits a single point exactly -> zero residual there.
    assert out[3] == pytest.approx(0.0, abs=1e-12)


def test_zero_within_level_variance_does_not_divide_by_zero():
    """x constant within a level: the slope is collinear with the intercept."""
    g = np.array([0, 0, 0, 1, 1, 1])
    x = np.array([2.0, 2.0, 2.0, 5.0, 5.0, 5.0])  # no within-level variation
    col = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    with pytest.warns(UserWarning, match="rank-deficient"):
        ab = Absorber(
            None,
            drop_singletons=False,
            n_obs=6,
            slopes=[SlopeSpec(g, x, True, "g#x")],
        )
    out = ab.demean(col)
    assert np.all(np.isfinite(out))
    # Degrades to a plain group demean.
    np.testing.assert_allclose(out, [-1.0, 0.0, 1.0, -10.0, 0.0, 10.0], atol=1e-12)


def test_all_zero_slope_column_leaves_column_untouched():
    """A level where x is identically 0 absorbs nothing (slope-only)."""
    g = np.array([0, 0, 0, 1, 1, 1])
    x = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    col = np.array([4.0, 5.0, 6.0, 1.0, 2.0, 3.0])
    with pytest.warns(UserWarning, match="rank-deficient"):
        ab = Absorber(
            None,
            drop_singletons=False,
            n_obs=6,
            slopes=[SlopeSpec(g, x, False, "g#x")],
        )
    out = ab.demean(col)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[:3], col[:3], atol=1e-12)


def test_degenerate_levels_excluded_from_dof():
    """reghdfe reports 40 categories - 1 redundant for the zc case (df_a=49)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = feols("y ~ d | big + i.g#c.zc", data=_deg())
    assert res.dof_fe == 49
    assert res.absorber.slope_ops[0].n_degenerate == 1


def test_slope_group_participates_in_singleton_pruning():
    """reghdfe drops the 40 one-observation g levels (N goes 240 -> 200)."""
    res = feols("y ~ d | i.g#c.z", data=_deg())
    assert res.n_obs == 200
    assert res.n_singletons_dropped == 40


# ======================================================================
# Formula parser
# ======================================================================


@pytest.mark.parametrize(
    "formula,fe_labels",
    [
        ("y ~ d | f", ["f"]),
        ("y ~ d | f + g", ["f", "g"]),
        ("y ~ d | i.f#c.x", ["i.f#c.x"]),
        ("y ~ d | c.x#i.f", ["i.f#c.x"]),
        ("y ~ d | i.f##c.x", ["i.f##c.x"]),
        ("y ~ d | c.x##i.f", ["i.f##c.x"]),
        ("y ~ d | f[[x]]", ["i.f#c.x"]),
        ("y ~ d | f[x]", ["i.f##c.x"]),
        ("y ~ d | f^g", ["i.f:i.g"]),
        ("y ~ d | f:g", ["f:g"]),
        ("y ~ d | f + i.g#c.x", ["f", "i.g#c.x"]),
    ],
)
def test_parser_fe_side(formula, fe_labels):
    _lhs, _x, fe = _parse_formula(formula)
    assert [t.label for t in fe] == fe_labels


@pytest.mark.parametrize(
    "formula,x_labels",
    [
        ("y ~ a + b", ["a", "b"]),
        ("y ~ a:b", ["a:b"]),
        ("y ~ a*b", ["a", "b", "a:b"]),
        ("y ~ a*b*c", ["a", "b", "c", "a:b", "a:c", "b:c", "a:b:c"]),
        ("y ~ i.f", ["i.f"]),
        ("y ~ c.x", ["x"]),
        ("y ~ f^g", ["i.f:i.g"]),
        ("y ~ i.f#c.x", ["i.f#c.x"]),
    ],
)
def test_parser_rhs(formula, x_labels):
    _lhs, x, _fe = _parse_formula(formula)
    assert [t.label for t in x] == x_labels


def test_stata_and_fixest_slope_syntax_agree():
    """``i.f#c.x`` == ``f[[x]]`` and ``i.f##c.x`` == ``f[x]``."""
    df = _bal()
    a = feols("y ~ d | county + i.pref#c.year", data=df)
    b = feols("y ~ d | county + pref[[year]]", data=df)
    assert float(a.params["d"]) == pytest.approx(float(b.params["d"]), rel=1e-12)

    c = feols("y ~ d | i.pref##c.year", data=df)
    e = feols("y ~ d | pref[year]", data=df)
    assert float(c.params["d"]) == pytest.approx(float(e.params["d"]), rel=1e-12)


def test_absorbing_slope_equals_slope_as_regressor():
    """FWL: absorbing i.pref#c.year gives the same d as including it on the RHS."""
    df = _bal()
    absorbed = feols("y ~ d | county + i.pref#c.year", data=df)
    explicit = feols("y ~ d + i.pref#c.year | county", data=df)
    assert float(absorbed.params["d"]) == pytest.approx(
        float(explicit.params["d"]), rel=1e-7
    )


@pytest.mark.parametrize(
    "formula,message",
    [
        ("y ~ d | i.pref#year", "i.pref#year"),
        ("y ~ d @ x", "d @ x"),
        ("y ~ d | f[x][y]", "f[x][y]"),
        ("y ~ d | f^", "f^"),
        ("y ~ d + | f", "dangling '+'"),
        ("y ~ d | 0", "not supported"),
        ("y ~ d | i.f#c.x:g", "cannot combine varying-slope"),
        ("y ~ d | i.f#c.x*g", "cannot combine varying-slope"),
    ],
)
def test_parser_rejects_unsupported_syntax(formula, message):
    """Unsupported syntax fails loudly, naming what it saw (CLAUDE.md §7)."""
    with pytest.raises(ValueError, match=re.escape(message)):
        _parse_formula(formula)


def test_error_message_lists_supported_syntax():
    with pytest.raises(ValueError) as exc:
        _parse_formula("y ~ d | i.pref#year")
    text = str(exc.value)
    for token in ("i.f#c.x", "i.f##c.x", "f[x]", "f[[x]]", "f1^f2", "a:b", "a*b"):
        assert token in text


def test_bad_formula_reports_expected_shape():
    with pytest.raises(ValueError, match="could not parse formula"):
        _parse_formula("this is not a formula")


# ======================================================================
# Regression guards: the pre-existing bare-name path must not move
# ======================================================================


def test_bare_name_path_is_unchanged():
    """Plain formulas take the legacy fast path and produce identical arrays."""
    df = _bal()
    _lhs, x_terms, fe_terms = _parse_formula("y ~ d + x1 + x2 | county + year")
    assert all(t.is_plain_name for t in x_terms)
    X, names = _materialize_rhs(df, x_terms)
    np.testing.assert_array_equal(X, df[["d", "x1", "x2"]].to_numpy(np.float64))
    assert names == ["d", "x1", "x2"]


def test_no_slope_results_match_plain_absorb_ols():
    """Adding the slopes= plumbing must not perturb the no-slope path."""
    df = _bal()
    res = feols("y ~ d + x1 | county + year", data=df)
    direct = absorb_ols(
        df.y.to_numpy(), df[["d", "x1"]].to_numpy(), df[["county", "year"]].to_numpy()
    )
    np.testing.assert_allclose(res.params.to_numpy(), direct["coef"], rtol=0, atol=0)
    np.testing.assert_allclose(res.std_errors.to_numpy(), direct["se"], rtol=0, atol=0)


def test_absorber_still_rejects_empty_specification():
    with pytest.raises(ValueError, match="at least one fixed-effect column"):
        Absorber(np.empty((10, 0)))


def test_slope_length_mismatch_raises():
    with pytest.raises(ValueError, match="was built on"):
        Absorber(
            np.arange(10).reshape(-1, 1),
            slopes=[SlopeSpec(np.arange(5), np.arange(5.0), False, "bad")],
        )


def test_non_finite_slope_variable_raises():
    with pytest.raises(ValueError, match="non-finite"):
        Absorber(
            None,
            n_obs=4,
            drop_singletons=False,
            slopes=[
                SlopeSpec(
                    np.array([0, 0, 1, 1]),
                    np.array([1.0, np.nan, 2.0, 3.0]),
                    False,
                    "g#x",
                )
            ],
        )


# ======================================================================
# _factorize_multi: NUL-byte truncation regression (⚠ correctness fix)
# ======================================================================


def test_factorize_multi_does_not_collapse_on_nul_truncation():
    """pandas truncates object strings at NUL, so string-joining is unsafe.

    ``pd.factorize(np.array(['0\\x000', '0\\x001'], dtype=object))`` returns a
    SINGLE group. Combining integer codes instead keeps the groups distinct.
    """
    a = np.array([0, 0, 0, 1, 1, 1])
    b = np.array([0, 1, 2, 0, 1, 2])
    codes, g = _factorize_multi([a, b])
    assert g == 6
    assert len(set(map(int, codes))) == 6

    # The old string-join approach really does collapse these.
    joined = pd.DataFrame({"a": a, "b": b}).astype(str).agg("\0".join, axis=1)
    assert len(pd.factorize(joined.values, sort=False)[1]) < 6


def test_interacted_fe_has_full_level_count():
    """``prov^year`` must absorb 3x20=60 groups, not 3 (regression guard)."""
    res = feols("y ~ d | prov^year", data=_bal())
    assert res.n_fe == [60]
    assert res.dof_fe == 60


# ======================================================================
# NumPy fallback kernels (the path taken when Numba is not installed)
# ======================================================================


@pytest.mark.parametrize("with_intercept", [False, True])
def test_numpy_slope_kernel_matches_numba_kernel(with_intercept):
    """The pure-NumPy fallback must agree with the Numba kernel bit-for-bit."""
    from statspai.panel import _hdfe_kernels as K
    from statspai.panel.hdfe import _slope_stats

    rng = np.random.default_rng(21)
    n, n_g = 500, 9
    codes = rng.integers(0, n_g, n).astype(np.int64)
    x = np.ascontiguousarray(rng.normal(size=n))
    col = np.ascontiguousarray(rng.normal(size=n))

    gsum, xsum, inv_denom, _ = _slope_stats(codes, x, n_g, with_intercept, None)

    a = col.copy()
    K._sweep_slope_numpy(a, x, codes, gsum, xsum, inv_denom, with_intercept)
    b = col.copy()
    K.sweep_slope(b, x, codes, gsum, xsum, inv_denom, with_intercept)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.parametrize("with_intercept", [False, True])
def test_numpy_weighted_slope_kernel_matches_numba_kernel(with_intercept):
    from statspai.panel import _hdfe_kernels as K
    from statspai.panel.hdfe import _slope_stats

    rng = np.random.default_rng(22)
    n, n_g = 500, 9
    codes = rng.integers(0, n_g, n).astype(np.int64)
    x = np.ascontiguousarray(rng.normal(size=n))
    col = np.ascontiguousarray(rng.normal(size=n))
    w = np.ascontiguousarray(rng.uniform(0.5, 2.0, n))

    gsum, xsum, inv_denom, _ = _slope_stats(codes, x, n_g, with_intercept, w)

    a = col.copy()
    K._sweep_slope_weighted_numpy(a, x, w, codes, gsum, xsum, inv_denom, with_intercept)
    b = col.copy()
    K.sweep_slope_weighted(b, x, w, codes, gsum, xsum, inv_denom, with_intercept)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_numpy_slope_kernel_handles_degenerate_levels():
    """The fallback must respect the inv_denom=0 sentinel too (no 0/0)."""
    from statspai.panel import _hdfe_kernels as K
    from statspai.panel.hdfe import _slope_stats

    codes = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    x = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])  # level 0 is all-zero
    col = np.array([4.0, 5.0, 6.0, 1.0, 2.0, 3.0])
    gsum, xsum, inv_denom, n_deg = _slope_stats(codes, x, 2, False, None)
    assert n_deg == 1 and inv_denom[0] == 0.0

    out = col.copy()
    K._sweep_slope_numpy(out, x, codes, gsum, xsum, inv_denom, False)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[:3], col[:3], atol=1e-12)
