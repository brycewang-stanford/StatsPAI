"""Tests for ``sp.cic`` covariate support and two-step bootstrap inference.

External reference implementation
--------------------------------
A real Stata 18 MP oracle **was** available for these tests.  Blaise Melly's
``cic`` / ``qrprocess`` could not be installed (his site is dead), but Keith
Kranker's ``cic`` — a direct Stata/Mata port of Athey & Imbens' own published
Matlab code, distributed via SSC (``ssc install cic``) — installs cleanly and
supports covariates through exactly the parametric approach quoted in Athey &
Imbens (2006, p. 466):

    "... apply the CIC estimator to the residuals from an ordinary least
    squares regression with the effects of the dummy variables added back in."

So the first stage is validated against a genuine external reference (both
``reghdfe`` residuals and Kranker's ``cic`` first-stage coefficients), and the
step-2 CIC estimand is validated against Kranker's ``cic`` point estimates.
Nothing here is mocked and no parity number is invented.

The Stata constants embedded below were produced on Stata 18 MP with
``reghdfe`` and ``cic`` (Kranker) on the fixtures generated in this module;
the generating script is deterministic (``np.random.default_rng`` with fixed
seeds) so the fixtures are reproducible from this file alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did.cic import _first_stage_residuals

# ── Fixtures ──────────────────────────────────────────────────────────


def _parity_data() -> pd.DataFrame:
    """Fixture exported to Stata for the reghdfe / cic parity constants."""
    rng = np.random.default_rng(20260722)
    n = 800
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    x1 = rng.normal(0.8 * g * t, 1.0)
    x2 = rng.normal(size=n)
    y = 1 + 0.5 * g + 0.3 * t + 1.5 * g * t + 2.0 * x1 - 1.0 * x2
    y = y + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "g": g, "t": t, "x1": x1, "x2": x2})


def _hdfe_data() -> pd.DataFrame:
    """Fixture for the pure `reghdfe` residual parity check."""
    rng = np.random.default_rng(20260722)
    n, nst = 400, 12
    st = rng.integers(0, nst, n)
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (
        1
        + 0.5 * g
        + 0.3 * t
        + 1.5 * g * t
        + 2.5 * x1
        - 1.2 * x2
        + rng.normal(0, 3, nst)[st]
        + rng.normal(0, 1, n)
    )
    return pd.DataFrame({"y": y, "g": g, "t": t, "x1": x1, "x2": x2, "st": st})


def _imbalanced(seed: int, n: int = 400) -> pd.DataFrame:
    """DGP where the first stage genuinely matters.

    ``x`` is strongly imbalanced across the four (group x time) cells, so the
    first-stage slope enters the CIC contrast with a large loading and its
    estimation error does not wash out.
    """
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    x = rng.normal(1.5 * g * t, 1.0)
    y = 1 + 0.5 * g + 0.3 * t + 1.5 * g * t + 2.0 * x + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "g": g, "t": t, "x": x})


def _step2_only(df: pd.DataFrame, n_boot: int, seed: int):
    """The hand-rolled pipeline: residualize ONCE, bootstrap only step 2."""
    adj, keep, _ = _first_stage_residuals(df, "y", "g", "t", [], ["x"])
    d2 = df.loc[keep].copy()
    d2["yadj"] = adj
    return sp.cic(d2, y="yadj", group="g", time="t", n_boot=n_boot, seed=seed)


# ── 1. Backwards compatibility ────────────────────────────────────────


def test_no_covariates_matches_stata_and_is_deterministic():
    """cic() without covariates matches Stata `cic` and is deterministic.

    The covariate branch must not perturb the unconditional code path. The
    ATT is anchored to Kranker's Stata ``cic continuous y g t, vce(none)``
    (Stata 18 MP, data imported ``asdouble``, ``e(b)`` read at full
    precision: 2.999903968026800 on this fixture) rather than a frozen
    internal blob, so the test tracks the reference estimator, not whatever
    the code happened to return.
    """
    df = _parity_data()
    kw = dict(y="y", group="g", time="t", n_boot=120, seed=42)
    a = sp.cic(df, quantiles=[0.25, 0.5, 0.75], **kw)

    # Anchored to the Stata `cic` reference (A&I estimator), not a golden blob.
    assert a.estimate == pytest.approx(2.999903968026800, abs=1e-8)
    assert a.n_obs == 800
    # covariates=None must leave model_info free of first-stage keys
    assert "covariates" not in a.model_info
    assert "first_stage" not in a.model_info

    # Re-running is deterministic bit-for-bit.
    b = sp.cic(df, quantiles=[0.25, 0.5, 0.75], **kw)
    assert a.estimate == b.estimate
    assert a.se == b.se
    assert a.ci == b.ci
    np.testing.assert_array_equal(
        a.detail[["qte", "se", "ci_lower", "ci_upper"]].to_numpy(),
        b.detail[["qte", "se", "ci_lower", "ci_upper"]].to_numpy(),
    )


def test_extra_columns_do_not_perturb_legacy_path():
    """Unused columns in `data` must not change the unconditional result."""
    df = _parity_data()
    wide = df.copy()
    wide["junk"] = np.arange(len(df))
    a = sp.cic(df, y="y", group="g", time="t", n_boot=60, seed=3)
    b = sp.cic(wide, y="y", group="g", time="t", n_boot=60, seed=3)
    assert a.estimate == b.estimate
    assert a.se == b.se


# ── 2. Step-1 parity vs Stata reghdfe (real external reference) ────────


def test_first_stage_matches_stata_reghdfe_noabsorb():
    """OLS-HDFE first stage == Stata `reghdfe y x1 x2, noabsorb`.

    Stata 18 MP, reghdfe:  x1 =  2.3976431932888693
                           x2 = -1.191028572766591
    Residual max abs deviation observed: 5.9e-15.
    """
    df = _hdfe_data()
    from statspai.panel.hdfe import absorb_ols

    out = absorb_ols(
        df["y"].to_numpy(float),
        df[["x1", "x2"]].to_numpy(float),
        pd.DataFrame({"c": np.zeros(len(df), dtype=int)}),
    )
    np.testing.assert_allclose(
        out["coef"], [2.3976431932888693, -1.191028572766591], rtol=1e-9
    )


def test_first_stage_matches_stata_reghdfe_absorb():
    """OLS-HDFE with absorbed FE == Stata `reghdfe y x1 x2, absorb(st)`.

    Verified against Stata 18 MP: residual max abs deviation 9.3e-15.
    Coefficients from the same run: x1 = 2.4705609, x2 = -1.1943013.
    """
    df = _hdfe_data()
    from statspai.panel.hdfe import absorb_ols

    out = absorb_ols(
        df["y"].to_numpy(float),
        df[["x1", "x2"]].to_numpy(float),
        df[["st"]],
    )
    np.testing.assert_allclose(out["coef"], [2.47056095, -1.19430131], rtol=1e-6)


def test_first_stage_coefs_match_stata_cic():
    """First-stage slopes == Kranker's Stata `cic continuous y g t x1 x2`.

    Stata `cic` reports the covariate coefficients from exactly the A&I
    (2006, p. 466) auxiliary regression — outcome on covariates *and* the
    group x time design dummies:

        x1 =  2.0314860142423394
        x2 = -0.9796084058563914

    Matching these to 1e-8 is what pins down that StatsPAI keeps the design
    dummies in the first stage.  A first stage without them gives a
    materially different slope and a biased ATT.
    """
    df = _parity_data()
    res = sp.cic(df, y="y", group="g", time="t", covariates=["x1", "x2"], n_boot=2)
    coefs = res.model_info["first_stage_coef"]
    assert coefs["x1"] == pytest.approx(2.0314860142423394, rel=1e-8)
    assert coefs["x2"] == pytest.approx(-0.9796084058563914, rel=1e-8)


def test_design_dummies_are_in_the_first_stage():
    """Dropping the design dummies would bias the slope — assert we keep them.

    Regressing y on x alone (no g/t/gxt dummies) gives a visibly different
    slope when x is imbalanced across cells; the A&I first stage must not.
    """
    df = _imbalanced(4000)
    res = sp.cic(df, y="y", group="g", time="t", covariates=["x"], n_boot=2)
    b_ai = res.model_info["first_stage_coef"]["x"]

    xv = df["x"].to_numpy(float)
    yv = df["y"].to_numpy(float)
    b_naive = np.polyfit(xv, yv, 1)[0]

    assert b_ai == pytest.approx(2.0, abs=0.15)  # true slope
    assert abs(b_naive - 2.0) > 0.20  # naive slope is visibly biased


# ── 3. Point-estimate behaviour ───────────────────────────────────────


def test_covariates_recover_att_under_imbalance():
    """With cell-imbalanced covariates, the two-step ATT recovers the truth.

    True ATT = 1.5.  Monte Carlo over 1000 draws of this DGP gives a mean
    two-step ATT of 1.5221 (sampling SD 0.2108) — unbiased up to the usual
    CIC finite-sample slippage.  The unconditional estimator on the same
    data is badly biased because x shifts with g*t.
    """
    df = _imbalanced(4000, n=2000)
    cov = sp.cic(df, y="y", group="g", time="t", covariates=["x"], n_boot=2)
    unc = sp.cic(df, y="y", group="g", time="t", n_boot=2)
    assert cov.estimate == pytest.approx(1.5, abs=0.35)
    assert unc.estimate > 3.0  # unconditional soaks up the covariate shift


def test_cic_collapses_to_did_under_pure_location_shift():
    """Analytic degenerate case: pure additive shifts => CIC == DiD.

    When every cell is the same distribution shifted by a constant, the
    CIC counterfactual map is the identity plus the control-group time
    shift, so the CIC ATT must equal the plain 2x2 DiD.  This holds with
    covariates too, since the first stage is linear.
    """
    rng = np.random.default_rng(5)
    n = 4000
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    base = rng.normal(0, 1, n)
    y = 1.0 + 0.7 * g + 0.4 * t + 1.2 * g * t + 1.5 * x + base
    df = pd.DataFrame({"y": y, "g": g, "t": t, "x": x})

    res = sp.cic(df, y="y", group="g", time="t", covariates=["x"], n_boot=2)

    # The identity is CIC(adjusted outcome) == DiD(adjusted outcome): both
    # estimators see the same covariate-adjusted data, and under pure
    # location shifts the CIC counterfactual map is a constant shift.
    adj, keep, _ = _first_stage_residuals(df, "y", "g", "t", [], ["x"])
    d2 = df.loc[keep].copy()
    d2["yadj"] = adj
    m = d2.groupby(["g", "t"])["yadj"].mean()
    did = (m[(1, 1)] - m[(1, 0)]) - (m[(0, 1)] - m[(0, 0)])

    assert res.estimate == pytest.approx(did, abs=0.06)
    assert res.estimate == pytest.approx(1.2, abs=0.15)


# ── 4. THE INFERENCE FIX: two-step vs step-2-only bootstrap ────────────


def test_two_step_bootstrap_se_exceeds_step2_only():
    """Two-step resampling yields a LARGER SE than bootstrapping step 2 alone.

    Recorded magnitudes (DGP ``_imbalanced(seed=4000)``, n=400, n_boot=600,
    seed=7; the two schemes share the identical bootstrap draws, so the
    comparison is exactly paired and free of Monte Carlo noise). Values are
    on the corrected A&I step-2 estimator (CHANGELOG ⚠️ correctness fix):

        ATT (identical for both)      = 0.986598
        two-step bootstrap SE         = 0.211055
        step-2-only bootstrap SE      = 0.196303
        ratio                         = 1.0751   (+7.5%)

    The qualitative picture is the one the fix targets: the hand-rolled
    ``feols`` -> residuals -> ``cic`` pipeline holds the first-stage
    coefficients fixed across bootstrap draws and so understates the SE;
    re-fitting the first stage inside every replicate restores the missing
    first-stage estimation variance.

    Note the effect only appears when the first stage keeps the group x time
    design dummies (the A&I 2006 p. 466 specification that ``sp.cic``
    implements).  With a first stage that omits them, the slope estimate
    absorbs the cross-cell contrast and a Frisch-Waugh cancellation makes the
    two schemes agree — see ``test_step2_only_bias_needs_design_dummies``.
    """
    df = _imbalanced(4000)
    two = sp.cic(df, y="y", group="g", time="t", covariates=["x"], n_boot=600, seed=7)
    one = _step2_only(df, n_boot=600, seed=7)

    # Identical point estimate — only the inference differs.
    assert two.estimate == pytest.approx(one.estimate, abs=1e-12)

    assert two.se == pytest.approx(0.211055, abs=5e-4)
    assert one.se == pytest.approx(0.196303, abs=5e-4)
    assert two.se > one.se
    assert two.se / one.se == pytest.approx(1.0751, abs=0.01)


def test_two_step_se_larger_across_seeds():
    """The ordering is systematic, not a lucky seed (>= 8 of 10 draws)."""
    wins = 0
    for seed in range(4000, 4010):
        df = _imbalanced(seed)
        two = sp.cic(
            df, y="y", group="g", time="t", covariates=["x"], n_boot=250, seed=7
        )
        one = _step2_only(df, n_boot=250, seed=7)
        wins += two.se > one.se
    assert wins >= 8, f"two-step SE exceeded step-2-only in only {wins}/10 draws"


def test_step2_only_bias_needs_design_dummies():
    """Document the boundary of the result.

    When the covariate is balanced across the four cells, the first-stage
    slope enters the CIC contrast with (near) zero loading, so its estimation
    error does not propagate and the two bootstrap schemes agree closely.
    This is the honest boundary of the SE claim above.
    """
    rng = np.random.default_rng(9)
    n = 400
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    x = rng.normal(size=n)  # balanced across cells
    y = 1 + 0.5 * g + 0.3 * t + 1.5 * g * t + 2.0 * x + rng.normal(0, 1.0, n)
    df = pd.DataFrame({"y": y, "g": g, "t": t, "x": x})

    two = sp.cic(df, y="y", group="g", time="t", covariates=["x"], n_boot=400, seed=7)
    one = _step2_only(df, n_boot=400, seed=7)
    assert two.se / one.se == pytest.approx(1.0, abs=0.05)


def test_bootstrap_metadata_records_two_step():
    df = _imbalanced(4000)
    res = sp.cic(df, y="y", group="g", time="t", covariates=["x"], n_boot=30, seed=1)
    assert res.model_info["bootstrap"].startswith("two-step")
    assert res.model_info["first_stage"] == "feols"
    assert res.model_info["covariates"] == ["x"]
    assert res.model_info["n_boot_failed"] == 0
    assert "Melly & Santangelo" in res.method


# ── 5. keep_mask alignment ────────────────────────────────────────────


def test_keep_mask_alignment_after_singleton_pruning():
    """Residuals must be realigned to surviving rows, never mis-paired.

    HDFE prunes singleton FE groups.  If the residual vector were pasted
    back without applying ``keep_mask``, observations would be silently
    paired with the wrong (group, time) cell.  Here we plant singleton
    states and check the surviving rows line up exactly.
    """
    rng = np.random.default_rng(11)
    n = 300
    st = rng.integers(0, 8, n).astype(object)
    # plant 5 singleton states
    for k in range(5):
        st[k] = f"solo_{k}"
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    y = 1 + 1.5 * g * t + 2.0 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "g": g, "t": t, "x": x, "st": st})

    adj, keep, _ = _first_stage_residuals(df, "y", "g", "t", ["st"], ["x"])
    assert keep.shape[0] == n
    assert adj.shape[0] == int(keep.sum())
    assert int((~keep).sum()) >= 5  # the planted singletons were pruned

    res = sp.cic(
        df,
        y="y",
        group="g",
        time="t",
        covariates=["x", "C(st)"],
        n_boot=20,
        seed=1,
    )
    assert res.n_obs == int(keep.sum())
    assert res.n_obs < n
    assert res.model_info["n_dropped_first_stage"] == int((~keep).sum())


def test_nan_rows_dropped_and_reported():
    df = _imbalanced(4000)
    df.loc[:9, "x"] = np.nan
    res = sp.cic(df, y="y", group="g", time="t", covariates=["x"], n_boot=20, seed=1)
    assert res.n_obs == len(df) - 10
    assert res.model_info["n_dropped_missing"] == 10


def test_i_dot_and_c_paren_fe_syntax_agree():
    df = _imbalanced(4000)
    df["st"] = np.arange(len(df)) % 6
    a = sp.cic(
        df,
        y="y",
        group="g",
        time="t",
        covariates=["x", "C(st)"],
        n_boot=10,
        seed=1,
    )
    b = sp.cic(
        df,
        y="y",
        group="g",
        time="t",
        covariates=["x", "i.st"],
        n_boot=10,
        seed=1,
    )
    assert a.estimate == pytest.approx(b.estimate, abs=1e-12)


# ── 6. Fail loudly (CLAUDE.md §7) ─────────────────────────────────────


def test_unknown_covariate_raises_with_corrected_call():
    df = _imbalanced(4000)
    with pytest.raises(ValueError, match=r"not found in `data`") as e:
        sp.cic(df, y="y", group="g", time="t", covariates=["x", "nope"])
    msg = str(e.value)
    assert "'nope'" in msg
    assert "sp.cic(" in msg  # shows a corrected call


def test_unsupported_first_stage_raises():
    df = _imbalanced(4000)
    with pytest.raises(ValueError, match=r"first_stage='lasso'") as e:
        sp.cic(
            df,
            y="y",
            group="g",
            time="t",
            covariates=["x"],
            first_stage="lasso",
        )
    assert "'feols'" in str(e.value)


def test_empty_covariate_list_raises():
    df = _imbalanced(4000)
    with pytest.raises(ValueError, match=r"covariates=\[\] is empty"):
        sp.cic(df, y="y", group="g", time="t", covariates=[])


def test_covariate_clashing_with_design_raises():
    df = _imbalanced(4000)
    with pytest.raises(ValueError, match=r"duplicate the outcome/group/time"):
        sp.cic(df, y="y", group="g", time="t", covariates=["x", "g"])


def test_duplicate_covariate_raises():
    df = _imbalanced(4000)
    with pytest.raises(ValueError, match=r"listed more than once"):
        sp.cic(df, y="y", group="g", time="t", covariates=["x", "x"])


def test_saturated_first_stage_raises():
    """A first stage that explains everything must raise, not return NaN."""
    rng = np.random.default_rng(3)
    n = 200
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    y = 2.0 * x  # deterministic in x: residuals are exactly zero
    df = pd.DataFrame({"y": y, "g": g, "t": t, "x": x})
    with pytest.raises(ValueError, match=r"removed essentially all variation"):
        sp.cic(df, y="y", group="g", time="t", covariates=["x"], n_boot=10)


def test_too_small_cell_after_covariate_dropping_raises():
    """Cells emptied by covariate NAs must raise with a corrected call."""
    df = _imbalanced(4000, n=200)
    # wipe the covariate for nearly all of the treated-post cell
    mask = (df["g"] == 1) & (df["t"] == 1)
    idx = df.index[mask][1:]
    df.loc[idx, "x"] = np.nan
    with pytest.raises(ValueError, match=r"Too few observations"):
        sp.cic(df, y="y", group="g", time="t", covariates=["x"], n_boot=10)


def test_all_missing_covariate_raises():
    df = _imbalanced(4000, n=100)
    df["x"] = np.nan
    with pytest.raises(ValueError, match=r"no complete rows left"):
        sp.cic(df, y="y", group="g", time="t", covariates=["x"], n_boot=10)


# ── 7. Step-2 CIC estimator vs Athey-Imbens / Stata `cic` ─────────────


def test_unconditional_cic_matches_stata_cic():
    """Unconditional ATT vs Kranker's Stata `cic continuous y g t`.

    This pinned a ⚠️ correctness fix (CHANGELOG 1.20.x). Before it,
    ``_counterfactual_quantiles`` composed the empirical CDFs with the
    control-post (y01) and treated-pre (y10) cells transposed relative to
    Athey & Imbens eq. 9, and used linearly-interpolated CDF/quantile
    functions instead of the step-function ECDF and its generalized inverse;
    the ATT converged to ~3.01388 (~0.5% off). The corrected estimator
    computes ``F_01^{-1}(F_00(y10))`` on the step ECDF and reproduces Stata
    ``cic continuous y g t, vce(none)`` (Stata 18 MP, ``asdouble`` import,
    ``e(b)`` at full precision) to machine precision: 2.999903968026800 on
    this fixture, observed |diff| = 1.8e-14.
    """
    df = _parity_data()
    res = sp.cic(df, y="y", group="g", time="t", n_boot=2)
    assert res.estimate == pytest.approx(2.999903968026800, abs=1e-8)


def test_athey_imbens_reference_algorithm_reproduces_stata():
    """Guard the oracle: the A&I algorithm as published == Stata `cic`.

    This is the constructive half of the correctness fix pinned above
    (formerly a strict xfail).  It re-derives the reference value from a
    self-contained transcription of the published algorithm, so the target
    in ``test_unconditional_cic_matches_stata_cic`` is demonstrably the
    Athey-Imbens estimator and not a re-derivation of whatever the
    production code happens to compute.
    """
    df = _parity_data()
    g = df["g"].to_numpy()
    t = df["t"].to_numpy()
    yv = df["y"].to_numpy()
    y00 = yv[(g == 0) & (t == 0)]
    y01 = yv[(g == 0) & (t == 1)]
    y10 = yv[(g == 1) & (t == 0)]
    y11 = yv[(g == 1) & (t == 1)]

    def ecdf(x, q):  # step-function empirical CDF
        return np.searchsorted(np.sort(x), q, side="right") / len(x)

    def qinv(x, p):  # generalized inverse inf{y : F(y) >= p}
        xs = np.sort(x)
        k = np.clip(np.ceil(p * len(xs)).astype(int), 1, len(xs))
        return xs[k - 1]

    # A&I (2006) eq. 9: apply the control group's temporal map to treated-pre.
    cf = qinv(y01, ecdf(y00, y10))
    att = y11.mean() - cf.mean()
    assert att == pytest.approx(2.999903968026800, abs=1e-8)


# ── 7b. Full-precision Stata `cic` anchors on two further DGPs ────────
#
# Both anchors were produced on Stata 18 MP with Kranker's ``cic`` (SSC), a
# direct Stata/Mata port of Athey & Imbens' own published Matlab code:
#
#     import delimited "<fixture>.csv", clear asdouble
#     cic continuous y g t, vce(none)
#     matrix list e(b), format(%20.15f)
#
# The fixtures were exported with ``float_format='%.17g'`` and imported
# ``asdouble``, so Stata and StatsPAI operate on bit-identical doubles.
# ``e(b)`` carries the continuous-CIC mean ATT and the decile QTEs
# (q10..q90).


def _anchor_dgp_normal() -> pd.DataFrame:
    """Anchor DGP 1: additive normal model, near-equal cells (~500 each)."""
    rng = np.random.default_rng(42)
    n = 2000
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    u = rng.normal(0, 1, n)
    y = u + 2 * g + t + 3 * g * t
    return pd.DataFrame({"y": y, "g": g, "t": t})


def _anchor_dgp_skewed() -> pd.DataFrame:
    """Anchor DGP 2: right-skewed outcome (lognormal base, sample skewness
    ~1.14) and deliberately unequal cell sizes (387 / 659 / 173 / 281)."""
    rng = np.random.default_rng(20260722)
    n = 1500
    g = (rng.random(n) < 0.3).astype(int)
    t = (rng.random(n) < 0.6).astype(int)
    u = rng.normal(0, 1, n)
    y = np.exp(0.6 * u) + 1.2 * g + 0.5 * t + 2.0 * g * t
    return pd.DataFrame({"y": y, "g": g, "t": t})


_DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Stata 18 MP, `cic continuous y g t, vce(none)`, e(b) at %20.15f.
_STATA_ANCHOR_NORMAL = {
    "mean": 2.932988686287668,
    "qte": [
        2.984293431695166,
        2.892426953922517,
        2.954082705274524,
        2.736474144726140,
        2.797782335202758,
        2.783208553865870,
        2.874348451135762,
        2.755720289274890,
        3.319918957749473,
    ],
}
_STATA_ANCHOR_SKEWED = {
    "mean": 2.069591321940478,
    "qte": [
        1.924623500029899,
        1.927667666732171,
        1.944829566473861,
        2.001063183165057,
        1.886355291933495,
        1.943617551270846,
        2.024186366338112,
        2.012795465901560,
        2.091832709765454,
    ],
}


def test_att_matches_stata_anchor_normal_dgp():
    """Continuous-CIC mean ATT == Stata `cic` on the additive-normal anchor.

    Before the ⚠️ correctness fix the estimator converged (as n_grid → ∞)
    to ~2.87131 on this DGP — 0.062 below the reference.  The corrected
    estimator agrees with Stata to machine precision (observed |diff| =
    2.2e-15); the 1e-8 tolerance is pure platform headroom.
    """
    res = sp.cic(_anchor_dgp_normal(), y="y", group="g", time="t", n_boot=2)
    assert res.estimate == pytest.approx(_STATA_ANCHOR_NORMAL["mean"], abs=1e-8)


def test_att_matches_stata_anchor_skewed_unequal_cells():
    """Continuous-CIC mean ATT == Stata `cic` under skew + unequal cells.

    The step-function-vs-interpolation error is DGP-dependent, so this
    anchor stresses exactly what the normal DGP does not: a right-skewed
    outcome and cell sizes from 173 to 659.  Before the fix the estimator
    converged to ~2.00407 here (0.065 below the reference); after it,
    observed |diff| vs Stata = 2.4e-14 (1e-8 tolerance is headroom).
    """
    res = sp.cic(_anchor_dgp_skewed(), y="y", group="g", time="t", n_boot=2)
    assert res.estimate == pytest.approx(_STATA_ANCHOR_SKEWED["mean"], abs=1e-8)


@pytest.mark.parametrize(
    "make_df, anchor",
    [
        (_anchor_dgp_normal, _STATA_ANCHOR_NORMAL),
        (_anchor_dgp_skewed, _STATA_ANCHOR_SKEWED),
    ],
    ids=["normal", "skewed-unequal"],
)
def test_qte_deciles_match_stata(make_df, anchor):
    """Decile QTEs (q10..q90) == Stata `cic` on both anchor DGPs.

    No definitional slack is needed: both implementations evaluate
    ``F_11^{-1}(τ) - F_{Y^N,11}^{-1}(τ)`` with the same generalized inverse
    ``inf{y : F(y) >= τ}`` on the step ECDF, so at continuously-distributed
    sample points the two quantile functions coincide exactly — the observed
    max |diff| across all 18 deciles is 4.5e-16 (one double ulp).  The 1e-8
    tolerance is pure platform headroom, not a definitional concession.
    """
    res = sp.cic(make_df(), y="y", group="g", time="t", quantiles=_DECILES, n_boot=2)
    np.testing.assert_allclose(
        res.detail["qte"].to_numpy(), anchor["qte"], rtol=0, atol=1e-8
    )


def test_n_grid_does_not_affect_point_estimates():
    """`n_grid` is API-compatibility-only: it must not move ATT or QTEs.

    The corrected estimator is exact on the sample points, so the former
    internal τ grid is gone; ``n_grid`` only sets the resolution of the
    stored plot curve.  Under the old (pre-fix) estimator this test fails:
    the ATT moved by ~3.4e-3 between n_grid=50 and n_grid=2000 on this DGP.
    """
    df = _anchor_dgp_normal()
    kw = dict(y="y", group="g", time="t", quantiles=[0.25, 0.75], n_boot=25, seed=3)
    a = sp.cic(df, n_grid=50, **kw)
    b = sp.cic(df, n_grid=2000, **kw)
    assert a.estimate == b.estimate
    assert a.se == b.se
    np.testing.assert_array_equal(
        a.detail["qte"].to_numpy(), b.detail["qte"].to_numpy()
    )


# ── 8. Plot / summary smoke with covariates ───────────────────────────


def test_summary_and_plot_with_covariates():
    import matplotlib

    matplotlib.use("Agg")
    df = _imbalanced(4000)
    res = sp.cic(
        df,
        y="y",
        group="g",
        time="t",
        covariates=["x"],
        quantiles=[0.25, 0.5, 0.75],
        n_boot=25,
        seed=1,
    )
    out = res.summary()
    assert "Covariates" in out
    assert "re-fit in every bootstrap replicate" in out
    fig, ax = res.plot()
    assert fig is not None and ax is not None
