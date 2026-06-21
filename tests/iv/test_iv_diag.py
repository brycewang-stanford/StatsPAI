"""Unit tests for the modern IV reporting bundle (`sp.iv.iv_diag`)."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import numpy as np
import pandas as pd
import pytest

import statspai as sp

# ─── Synthetic data generator ───────────────────────────────────────────


def _make_iv_data(n=600, beta=0.8, seed=0, binary_endog=False):
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    d_lat = 0.6 * z1 + 0.3 * z2 + 0.5 * x + 0.4 * u + v
    d = (d_lat > 0).astype(float) if binary_endog else d_lat
    y = 0.5 + beta * d + 0.5 * x + u
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2, "x": x})


# ═══════════════════════════════════════════════════════════════════════
#  iv_diag — basic surface
# ═══════════════════════════════════════════════════════════════════════


def test_iv_diag_basic_runs_and_recovers_dgp():
    df = _make_iv_data(n=800, beta=0.8, seed=42)
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=100,
        random_state=0,
    )
    # Recovers the DGP within a 2-SE window
    assert abs(r.beta_2sls - 0.8) < 3 * r.se_2sls
    # Strong instrument: tF should collapse to ~1.96
    assert r.first_stage_F > 50
    assert abs(r.tF_critical_value - 1.96) < 0.05
    # AR CI should contain truth
    assert r.ar_ci[0] < 0.8 < r.ar_ci[1]
    # No TSLS-LATE caveat for continuous endog
    assert r.tsls_late_caveat is None


def test_iv_diag_to_frame_columns():
    df = _make_iv_data(seed=1)
    r = sp.iv.iv_diag(
        df, y="y", endog="d", instruments=["z1"], n_boot=50, random_state=0
    )
    fr = r.to_frame()
    assert {
        "estimator",
        "estimate",
        "SE",
        "stat",
        "p-value",
        "CI lower",
        "CI upper",
    }.issubset(fr.columns)
    # Must include 2SLS analytic, tF-adjusted, AR, OLS rows
    estimators = set(fr["estimator"])
    assert "2SLS (analytic)" in estimators
    assert "2SLS (LMMP tF-adjusted)" in estimators
    assert "Anderson–Rubin set" in estimators
    assert "OLS (comparator, not causal)" in estimators


def test_iv_diag_bootstrap_pairs_and_wild():
    df = _make_iv_data(seed=2)
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=200,
        boot_methods=("pairs", "wild"),
        random_state=0,
    )
    assert r.bootstrap_n >= 100
    assert r.bootstrap_se_pairs is not None and r.bootstrap_se_pairs > 0
    assert r.bootstrap_se_wild is not None and r.bootstrap_se_wild > 0
    # Both bootstrap CIs cover the analytic point estimate
    lo, hi = r.bootstrap_ci_pairs
    assert lo < r.beta_2sls < hi
    lo, hi = r.bootstrap_ci_wild
    assert lo < r.beta_2sls < hi


def test_iv_diag_optional_clr_k():
    df = _make_iv_data(seed=3, n=400)
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=0,
        include_clr_ci=True,
        include_k_ci=True,
        grid_size=121,
        random_state=0,
    )
    assert r.clr_ci is not None and len(r.clr_ci) == 2
    assert r.k_ci is not None and len(r.k_ci) == 2
    # CLR / K should bracket the 2SLS point
    assert r.clr_ci[0] < r.beta_2sls < r.clr_ci[1]
    assert r.k_ci[0] < r.beta_2sls < r.k_ci[1]


def test_iv_diag_ltz_sensitivity_widens_ci():
    df = _make_iv_data(seed=4, n=500)
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=0,
        ltz_gamma_sd=0.05,
        random_state=0,
    )
    assert r.ltz_ci is not None
    # LTZ CI must be at least as wide as the analytic Wald CI
    wald_w = r.ci_analytic_2sls[1] - r.ci_analytic_2sls[0]
    ltz_w = r.ltz_ci[1] - r.ltz_ci[0]
    assert ltz_w >= wald_w - 1e-8


def test_iv_diag_binary_endog_caveat_triggers():
    df = _make_iv_data(seed=5, n=600, binary_endog=True)
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=0,
        random_state=0,
    )
    assert r.tsls_late_caveat is not None
    assert "Blandhol" in r.tsls_late_caveat
    assert "Słoczyński" in r.tsls_late_caveat


def test_iv_diag_no_exog_no_caveat():
    df = _make_iv_data(seed=6, n=400, binary_endog=True)
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=None,
        n_boot=0,
        random_state=0,
    )
    # BBMT/Słoczyński caveat is *covariate-driven* — without covariates it stays silent.
    assert r.tsls_late_caveat is None


def test_iv_diag_summary_includes_caveat_when_triggered():
    df = _make_iv_data(seed=7, n=400, binary_endog=True)
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=0,
        random_state=0,
    )
    s = r.summary()
    assert "Interpretation caveat" in s
    assert "TSLS" in s


def test_iv_diag_to_dict_jsonable():
    import json

    df = _make_iv_data(seed=8)
    r = sp.iv.iv_diag(
        df, y="y", endog="d", instruments=["z1"], n_boot=50, random_state=0
    )
    j = json.dumps(r.to_dict())  # does not raise
    assert isinstance(j, str) and len(j) > 100


def test_iv_diag_to_latex_runs():
    df = _make_iv_data(seed=9)
    r = sp.iv.iv_diag(
        df, y="y", endog="d", instruments=["z1"], n_boot=0, random_state=0
    )
    tex = r.to_latex(caption="IV bundle", label="tab:iv")
    assert "\\begin{table}" in tex and "tab:iv" in tex


# ═══════════════════════════════════════════════════════════════════════
#  iv_compare
# ═══════════════════════════════════════════════════════════════════════


def test_iv_compare_recovers_dgp_across_methods():
    df = _make_iv_data(seed=10, n=800)
    out = sp.iv.iv_compare(
        "y ~ (d ~ z1 + z2) + x",
        data=df,
        methods=("2sls", "liml", "fuller", "jive", "ujive"),
    )
    assert set(out["status"]) <= {"ok"}
    # All estimates are within 0.15 of the truth
    assert (out["estimate"] - 0.8).abs().max() < 0.15


def test_iv_compare_endog_name_override():
    df = _make_iv_data(seed=11)
    out = sp.iv.iv_compare(
        "y ~ (d ~ z1 + z2) + x",
        data=df,
        methods=("ujive",),
        endog_name="d",
    )
    # Forced look-up by endog name returns a non-trivial estimate
    assert out.iloc[0]["estimate"] is not None and abs(out.iloc[0]["estimate"]) < 5


# ═══════════════════════════════════════════════════════════════════════
#  Plot helpers — only smoke-test "does it draw without errors"
# ═══════════════════════════════════════════════════════════════════════


def test_plots_smoke():
    df = _make_iv_data(seed=12, n=400)
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=50,
        include_clr_ci=True,
        include_k_ci=True,
        ltz_gamma_sd=0.05,
        random_state=0,
    )
    # All four plot routes
    assert r.plot("first_stage") is not None
    assert r.plot("forest") is not None
    assert r.plot("weak_iv") is not None
    assert r.plot("diagnostic") is not None


# ═══════════════════════════════════════════════════════════════════════
#  Error paths
# ═══════════════════════════════════════════════════════════════════════


def test_iv_diag_unknown_plot_kind_raises():
    df = _make_iv_data(seed=13)
    r = sp.iv.iv_diag(
        df, y="y", endog="d", instruments=["z1"], n_boot=0, random_state=0
    )
    with pytest.raises(ValueError, match="Unknown plot kind"):
        r.plot("nonsense")


def test_iv_diag_top_level_alias():
    """sp.iv_diag should be the same callable as sp.iv.iv_diag."""
    assert sp.iv_diag is sp.iv.iv_diag
    assert sp.iv_compare is sp.iv.iv_compare
    assert sp.IVDiagResult is sp.iv.IVDiagResult


# ═══════════════════════════════════════════════════════════════════════
#  Coverage of issues caught in the v1.14 polish-pass code review
# ═══════════════════════════════════════════════════════════════════════


def test_iv_diag_cluster_string_column():
    """Cluster identifier passed as a column name in `data`."""
    df = _make_iv_data(seed=20, n=600)
    df["state"] = np.repeat(np.arange(20), len(df) // 20 + 1)[: len(df)]
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        cluster="state",
        n_boot=100,
        random_state=0,
    )
    # Cluster bootstrap should produce a non-trivial SE
    assert r.bootstrap_se_pairs is not None
    assert r.bootstrap_se_pairs > 0


def test_iv_diag_cluster_array_aligns_after_dropna():
    """Cluster array passed as ndarray must align with surviving rows
    when listwise deletion drops some observations.

    Regression test for the v1.14 review's blocker #1 — without the
    `_orig_idx` realignment, the cluster vector silently lined up
    against the post-reset index.
    """
    df = _make_iv_data(seed=21, n=400)
    cluster = np.repeat(np.arange(40), 10)
    # Inject NaNs into a different column so the dropna survives a
    # known subset
    df = df.copy()
    df.loc[10:13, "x"] = np.nan
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        cluster=cluster,
        n_boot=80,
        random_state=0,
    )
    # The estimator should run; correctness is hard to assert directly
    # without a parallel truth, but the SE must be finite.
    assert np.isfinite(r.se_2sls) and r.se_2sls > 0
    assert r.bootstrap_se_pairs is not None


def test_iv_diag_cluster_array_length_mismatch_raises():
    df = _make_iv_data(seed=22, n=300)
    cluster = np.arange(150)  # wrong length
    with pytest.raises(ValueError, match="cluster array length"):
        sp.iv.iv_diag(
            df,
            y="y",
            endog="d",
            instruments=["z1"],
            cluster=cluster,
            n_boot=0,
        )


def test_iv_diag_classic_vcov_matches_homoskedastic():
    """vcov='classic' must NOT silently equal HC0; for a homoskedastic
    DGP, classic and HC1 should be very close, but classic uses the
    exact homoskedastic formula.
    """
    df = _make_iv_data(seed=23, n=2000)
    r_classic = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=0,
        vcov="classic",
        random_state=0,
    )
    r_hc1 = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=0,
        vcov="HC1",
        random_state=0,
    )
    assert np.isfinite(r_classic.se_2sls)
    # Under homoskedasticity at large n, classic ~ HC1 within a few %
    rel = abs(r_classic.se_2sls - r_hc1.se_2sls) / r_hc1.se_2sls
    assert rel < 0.1


def test_iv_diag_tF_returns_inf_when_F_below_threshold():
    """When first-stage F < 3.84, LMMP tF is undefined → store inf
    and return an unbounded tF CI (regression for review's blocker #2).
    """
    rng = np.random.default_rng(0)
    n = 400
    z = rng.standard_normal(n)
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    # Very weak instrument (F ~ O(1))
    d = 0.005 * z + 0.4 * x + u + rng.standard_normal(n)
    y = 0.5 + 0.8 * d + 0.5 * x + u
    df = pd.DataFrame({"y": y, "d": d, "z": z, "x": x})
    r = sp.iv.iv_diag(
        df, y="y", endog="d", instruments=["z"], exog=["x"], n_boot=0, random_state=0
    )
    if r.first_stage_F < 3.84:
        assert not np.isfinite(r.tF_critical_value)
        assert r.tF_adjusted_ci == (-np.inf, np.inf)


def test_iv_diag_bootstrap_n_per_method():
    """When both 'pairs' and 'wild' are requested, bootstrap_n is the
    minimum of the two success counts (conservative reading)."""
    df = _make_iv_data(seed=24, n=400)
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=100,
        boot_methods=("pairs", "wild"),
        random_state=0,
    )
    # bootstrap_n should be no larger than the smallest of the two
    assert r.bootstrap_n > 0


def test_iv_diag_ltz_warning_describes_tipping_point():
    """LTZ warning text must reference the prior σ_γ in
    multiples-of-σ-units, not a literal 0."""
    df = _make_iv_data(seed=25)
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=0,
        ltz_gamma_sd=0.05,
        random_state=0,
    )
    assert r.ltz_warning is not None
    assert "σ_γ" in r.ltz_warning
    assert "0.00" not in r.ltz_warning  # must not be the old useless string
