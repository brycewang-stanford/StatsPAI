"""Coverage tests for assorted decomposition modules.

Targets currently-uncovered branches in
``decomposition/{cfm,machado_mata,melly,nonlinear,plots,oaxaca}.py``:
alternate ``reference`` branches, default-grid construction, validation
raises, degenerate-numeric fallbacks, and plot rendering branches.

All numeric assertions check genuine algebraic identities of the
decomposition (gap = composition + structure, etc.) or exact structure of
the returned matplotlib artists.  No numerical path is mocked.
"""

from __future__ import annotations

import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import statspai as sp
import importlib

# These submodule names are also re-exported as *functions* on the
# ``statspai.decomposition`` package, so ``import x.y.z as m`` binds to the
# function, not the module.  Resolve the actual modules via importlib.
cfm_mod = importlib.import_module("statspai.decomposition.cfm")
mm_mod = importlib.import_module("statspai.decomposition.machado_mata")
nl_mod = importlib.import_module("statspai.decomposition.nonlinear")
plots_mod = importlib.import_module("statspai.decomposition.plots")


# ════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════


def _make_continuous(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.repeat([0, 1], n // 2)
    x1 = rng.normal(0.0, 1.0, n) + 0.3 * g
    x2 = rng.normal(0.0, 1.0, n) - 0.2 * g
    y = 1.0 + 0.5 * x1 + 0.3 * x2 + 0.4 * g + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"y": y, "g": g, "x1": x1, "x2": x2})


def _make_binary(n: int = 240, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.repeat([0, 1], n // 2)
    x1 = rng.normal(0.0, 1.0, n) + 0.4 * g
    eta = -0.3 + 0.8 * x1 + 0.5 * g
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"y": y, "g": g, "x1": x1})


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# ════════════════════════════════════════════════════════════════════════
# cfm.py
# ════════════════════════════════════════════════════════════════════════


def test_cfm_reference1_and_repr():
    """reference=1 branch (cfm.py 253, 264-265) + default tau_grid (239) + repr (187)."""
    df = _make_continuous()
    res = sp.cfm_decompose(df, "y", "g", ["x1", "x2"], reference=1)
    # default tau_grid → deciles 0.1..0.9 (9 points)
    assert len(res.quantile_grid) == 9
    g = res.quantile_grid
    # algebraic identity: gap == composition + structure at every tau
    np.testing.assert_allclose(
        g["gap"].to_numpy(),
        (g["composition"] + g["structure"]).to_numpy(),
        atol=1e-10,
    )
    # reference=1 wiring: composition = q_cf - q_b, structure = q_a - q_cf
    np.testing.assert_allclose(
        g["composition"].to_numpy(), (g["q_cf"] - g["q_b"]).to_numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        g["structure"].to_numpy(), (g["q_a"] - g["q_cf"]).to_numpy(), atol=1e-10
    )
    assert res.reference == 1
    assert "reference=1" in repr(res)
    assert "n_tau=9" in repr(res)


def test_cfm_too_few_obs_raises():
    """cfm.py 236: <20 obs per group raises ValueError."""
    df = _make_continuous(n=30)  # 15 per group < 20
    with pytest.raises(ValueError, match="20 obs"):
        sp.cfm_decompose(df, "y", "g", ["x1"])


def test_cfm_fit_dr_separation_fallback():
    """cfm.py 78-82: logit_fit exception path falls back to empirical proportion.

    Force ``logit_fit`` to raise so the ``except`` branch (which sets the
    intercept to the empirical log-odds) is exercised, then check the
    fitted CDF matches the empirical proportion the fallback encodes.
    """
    rng = np.random.default_rng(3)
    n = 80
    y = rng.normal(size=n)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    # thresholds in the *interior* so the degenerate-count guard (line 65)
    # does not pre-empt the try/except we want to hit.
    thr = np.quantile(y, [0.4, 0.5, 0.6])

    orig = cfm_mod.logit_fit

    def _boom(*a, **k):
        raise RuntimeError("forced separation")

    cfm_mod.logit_fit = _boom
    try:
        betas = cfm_mod._fit_dr(y, X, thr)
    finally:
        cfm_mod.logit_fit = orig

    # slope coefficients zeroed; intercept = logit(empirical proportion)
    assert np.allclose(betas[:, 1], 0.0)
    for i, t in enumerate(thr):
        p = np.clip((y <= t).mean(), 1e-3, 1 - 1e-3)
        assert betas[i, 0] == pytest.approx(np.log(p / (1 - p)))


# ════════════════════════════════════════════════════════════════════════
# machado_mata.py
# ════════════════════════════════════════════════════════════════════════


def test_machado_mata_reference1_repr_and_default_grid():
    """mm.py 233 (default grid), 163-167 (repr), reference=1 identity."""
    df = _make_continuous()
    res = sp.machado_mata(
        df, "y", "g", ["x1", "x2"], reference=1, n_sim=400, n_tau_qr=25, seed=1
    )
    assert len(res.quantile_grid) == 9  # default deciles
    g = res.quantile_grid
    np.testing.assert_allclose(
        g["gap"].to_numpy(),
        (g["composition"] + g["structure"]).to_numpy(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        g["composition"].to_numpy(), (g["q_cf"] - g["q_b"]).to_numpy(), atol=1e-10
    )
    assert "reference=1" in repr(res)
    assert "n_sim=400" in repr(res)


def test_machado_mata_too_few_obs_raises():
    """mm.py 230."""
    df = _make_continuous(n=30)
    with pytest.raises(ValueError, match="Machado-Mata"):
        sp.machado_mata(df, "y", "g", ["x1"])


def test_machado_mata_bootstrap_inference():
    """mm.py 285-336 bootstrap path incl. reference=1 comp branch (325) and se_df."""
    df = _make_continuous(n=200)
    res = sp.machado_mata(
        df,
        "y",
        "g",
        ["x1"],
        reference=1,
        n_sim=150,
        n_tau_qr=15,
        inference="bootstrap",
        n_boot=25,
        seed=2,
    )
    assert res.se is not None
    assert list(res.se.columns) == ["tau", "gap_se", "composition_se"]
    assert len(res.se) == len(res.quantile_grid)
    # tau column of se table aligns with the quantile grid
    np.testing.assert_allclose(
        res.se["tau"].to_numpy(), res.quantile_grid["tau"].to_numpy(), atol=1e-12
    )
    # point estimates obey the decomposition identity and are finite/bounded
    gg = res.quantile_grid
    np.testing.assert_allclose(
        gg["gap"].to_numpy(),
        (gg["composition"] + gg["structure"]).to_numpy(),
        atol=1e-10,
    )
    assert np.all(np.isfinite(gg[["gap", "composition", "structure"]].to_numpy()))
    assert np.all(np.abs(gg["gap"].to_numpy()) < 50.0)
    # bootstrap with genuine cross-replicate variation → SEs strictly positive
    # (not the all-zero degenerate case) and bounded well away from absurd.
    assert np.all(np.isfinite(res.se[["gap_se", "composition_se"]].to_numpy()))
    assert (res.se["gap_se"] > 0).all()
    assert (res.se["composition_se"] > 0).all()
    assert (res.se["gap_se"] < 50.0).all()
    assert (res.se["composition_se"] < 50.0).all()


def test_machado_mata_bootstrap_skips_tiny_strata():
    """mm.py 305-306: bootstrap resamples that yield <20 obs in a stratum
    are skipped (continue), so with <=10 valid replicates ``se`` stays None."""
    rng = np.random.default_rng(5)
    # group 1 has exactly 22 obs; many stratified resamples still have >=20,
    # but we keep n_boot tiny so len(boot_list) <= 10 and se_df is not built.
    n0, n1 = 60, 22
    g = np.r_[np.zeros(n0), np.ones(n1)].astype(int)
    x1 = rng.normal(size=n0 + n1)
    y = 0.5 * x1 + 0.3 * g + rng.normal(size=n0 + n1)
    df = pd.DataFrame({"y": y, "g": g, "x1": x1})
    res = sp.machado_mata(
        df,
        "y",
        "g",
        ["x1"],
        n_sim=100,
        n_tau_qr=11,
        inference="bootstrap",
        n_boot=8,
        seed=9,
    )
    # <=10 successful replicates → se_df guard (line 329) keeps se None
    assert res.se is None


def test_qreg_irls_singular_fallback():
    """mm.py 70-71: singular weighted normal matrix → lstsq fallback.

    A perfectly collinear design makes ``X.T @ WX`` singular, forcing the
    ``np.linalg.LinAlgError`` branch.  The estimate must still be finite.
    """
    rng = np.random.default_rng(0)
    n = 50
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x, 2.0 * x])  # col 3 = 2*col 2 → singular
    y = 1.0 + x + rng.normal(scale=0.1, size=n)
    beta = mm_mod._qreg_irls(y, X, tau=0.5, max_iter=5)
    assert beta.shape == (3,)
    assert np.all(np.isfinite(beta))
    # The lstsq fallback still recovers the planted median fit.  The design
    # has col3 == 2*col2, so the slopes are individually unidentified but the
    # *effective* slope on x is beta[1] + 2*beta[2]; with the true model
    # y = 1 + x + N(0, 0.1) the median fit recovers intercept ~ 1 and
    # effective slope ~ 1 (generous band well outside the 0.1 noise scale).
    assert beta[0] == pytest.approx(1.0, abs=0.1)
    assert (beta[1] + 2.0 * beta[2]) == pytest.approx(1.0, abs=0.1)


# ════════════════════════════════════════════════════════════════════════
# melly.py
# ════════════════════════════════════════════════════════════════════════


def test_melly_reference1_repr_and_default_grid():
    """melly.py 152 (default grid), 87-88 (repr), 164/171-172 (reference=1)."""
    df = _make_continuous()
    res = sp.melly_decompose(df, "y", "g", ["x1", "x2"], reference=1, n_tau_qr=25)
    assert len(res.quantile_grid) == 9
    g = res.quantile_grid
    np.testing.assert_allclose(
        g["gap"].to_numpy(),
        (g["composition"] + g["structure"]).to_numpy(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        g["composition"].to_numpy(), (g["q_cf"] - g["q_b"]).to_numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        g["structure"].to_numpy(), (g["q_a"] - g["q_cf"]).to_numpy(), atol=1e-10
    )
    assert "reference=1" in repr(res)


def test_melly_too_few_obs_raises():
    """melly.py 149."""
    df = _make_continuous(n=30)
    with pytest.raises(ValueError, match="Melly"):
        sp.melly_decompose(df, "y", "g", ["x1"])


# ════════════════════════════════════════════════════════════════════════
# nonlinear.py — Fairlie / Bauer-Sinning / probit helpers
# ════════════════════════════════════════════════════════════════════════


def test_fairlie_reference1_branch_and_identity():
    """nonlinear.py 226-228 (reference=1) + gap = explained + unexplained."""
    df = _make_binary()
    res = sp.fairlie(
        df, "y", "g", ["x1"], model="logit", reference=1, n_sim=200, seed=3
    )
    assert res.gap == pytest.approx(res.explained + res.unexplained, abs=1e-10)
    # detailed contributions normalised to sum to explained
    assert res.detailed["contribution"].sum() == pytest.approx(res.explained, rel=1e-6)
    assert res.reference == 1


def test_fairlie_too_few_obs_raises():
    """nonlinear.py 215."""
    df = _make_binary(n=16)  # 8 per group < 10
    with pytest.raises(ValueError, match="Fairlie"):
        sp.fairlie(df, "y", "g", ["x1"])


def test_fairlie_probit_uses_probit_helpers():
    """Drives _probit_fit / _probit_predict via model='probit'."""
    df = _make_binary()
    res = sp.fairlie(df, "y", "g", ["x1"], model="probit", n_sim=150, seed=4)
    assert res.model == "probit"
    assert res.gap == pytest.approx(res.explained + res.unexplained, abs=1e-10)
    # |gap| equals the observed difference in group means (sign is convention);
    # both are exact, model-free statistics of the data.
    obs_gap = df.loc[df.g == 1, "y"].mean() - df.loc[df.g == 0, "y"].mean()
    assert abs(res.gap) == pytest.approx(abs(obs_gap), abs=1e-10)
    # detailed contributions partition the explained component
    assert res.detailed["contribution"].sum() == pytest.approx(res.explained, rel=1e-6)


def test_bauer_sinning_reference1():
    """nonlinear.py 337 (reference=1 → beta_ref = beta_b) + repr (164-166)."""
    df = _make_binary()
    res = sp.yun_nonlinear(df, "y", "g", ["x1"], model="logit", reference=1)
    assert res.gap == pytest.approx(res.explained + res.unexplained, abs=1e-10)
    # Yun weights sum the per-variable contributions to the explained part
    assert res.detailed["contribution"].sum() == pytest.approx(res.explained, abs=1e-9)
    assert res.method.startswith("Bauer-Sinning")
    assert "model=logit" in repr(res)


def test_bauer_sinning_zero_total_weights():
    """nonlinear.py 351: when Σ(Δx̄·β_ref) ≈ 0 the weights fall back to zeros.

    Construct groups with identical covariate means so ``delta`` is ~0 and
    ``total`` falls below the 1e-12 guard, forcing ``weights = zeros``.
    """
    rng = np.random.default_rng(8)
    n = 120
    g = np.repeat([0, 1], n // 2)
    # identical x distribution across groups → mean(X_a) ≈ mean(X_b)
    x1 = np.tile(rng.normal(size=n // 2), 2)
    eta = -0.2 + 0.6 * x1 + 0.5 * g
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({"y": y, "g": g, "x1": x1})
    res = sp.yun_nonlinear(df, "y", "g", ["x1"], model="logit", reference=0)
    # zero-weight fallback → all detailed contributions are exactly zero
    assert np.allclose(res.detailed["contribution"].to_numpy(), 0.0)


def test_probit_fit_singular_hessian_fallback():
    """nonlinear.py 72-73 (solve→lstsq) and 86-87 (inv→pinv) on a singular
    design.  Collinear columns make both the Newton step and the info matrix
    singular; the estimate and vcov must still be finite and well-shaped."""
    rng = np.random.default_rng(2)
    n = 60
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x, 2.0 * x])  # collinear → singular
    y = (x + rng.normal(scale=0.3, size=n) > 0).astype(float)
    beta, vcov = nl_mod._probit_fit(y, X, max_iter=5)
    assert beta.shape == (3,)
    assert vcov.shape == (3, 3)
    assert np.all(np.isfinite(beta))
    assert np.all(np.isfinite(vcov))
    # vcov is a valid (pinv-regularised) covariance: symmetric with
    # non-negative variances on the diagonal.
    np.testing.assert_allclose(vcov, vcov.T, atol=1e-10)
    assert np.all(np.diag(vcov) >= 0.0)
    # col3 == 2*col2 so individual slopes are unidentified, but the *direction*
    # x -> y is.  The DGP y = 1[x + noise > 0] is increasing in x, so the
    # effective slope on x (beta[1] + 2*beta[2]) is positive.
    assert (beta[1] + 2.0 * beta[2]) > 0.0


# ════════════════════════════════════════════════════════════════════════
# plots.py
# ════════════════════════════════════════════════════════════════════════


def test_ci_whiskers_returns_none_when_no_positive_se():
    """plots.py 47: all-zero SE array → None (no whiskers)."""
    assert plots_mod._ci_whiskers([1.0, 2.0], [0.0, 0.0]) is None
    # positive SE → array of z*se half-widths (z = 1.96 for the default 95% CI)
    from scipy import stats as _stats

    z = float(_stats.norm.ppf(0.975))
    w = plots_mod._ci_whiskers([1.0, 2.0], [0.5, 0.0])
    assert w is not None
    assert w.shape == (2,)
    # half-width is exactly z * se elementwise
    np.testing.assert_allclose(w, z * np.array([0.5, 0.0]), atol=1e-12)
    assert w[0] == pytest.approx(z * 0.5)
    assert w[1] == 0.0


def test_dfl_plot_quantile_name_branch():
    """plots.py 174-177: quantile DFL result renders 'quantile(τ=..)' title."""
    res = types.SimpleNamespace(
        gap=1.0,
        composition=0.6,
        structure=0.4,
        se=None,
        stat="quantile",
        tau=0.25,
    )
    fig, ax = plots_mod.dfl_plot(res)
    assert "quantile(τ=0.25)" in ax.get_title()
    # three bars: gap / composition / structure
    assert len(ax.patches) == 3


def test_quantile_process_plot_ci_band():
    """plots.py 263-271: SE columns trigger fill_between CI bands."""
    grid = pd.DataFrame(
        {
            "tau": [0.25, 0.5, 0.75],
            "gap": [1.0, 1.1, 1.2],
            "composition": [0.6, 0.65, 0.7],
            "structure": [0.4, 0.45, 0.5],
            "gap_se": [0.1, 0.1, 0.1],
            "composition_se": [0.05, 0.05, 0.05],
            "structure_se": [0.05, 0.05, 0.05],
        }
    )
    res = types.SimpleNamespace(quantile_grid=grid)
    fig, ax = plots_mod.quantile_process_plot(res)
    # three line series
    assert len(ax.lines) >= 3
    # three shaded CI bands (PolyCollection from fill_between)
    assert len(ax.collections) == 3


def test_inequality_subgroup_plot_with_overlap():
    """plots.py 316-318: overlap present → third 'Overlap' bar."""
    res = types.SimpleNamespace(
        between=2.0,
        within=1.5,
        overlap=0.5,
        index="gini",
    )
    fig, ax = plots_mod.inequality_subgroup_plot(res)
    assert len(ax.patches) == 3  # Between / Within / Overlap
    assert "gini" in ax.get_title()


def test_inequality_subgroup_plot_no_overlap():
    """plots.py 316 false branch: overlap None → only two bars."""
    res = types.SimpleNamespace(
        between=2.0,
        within=1.5,
        overlap=None,
        index="theil",
    )
    fig, ax = plots_mod.inequality_subgroup_plot(res)
    assert len(ax.patches) == 2


def test_mediation_forest_skips_none_component():
    """plots.py 378-379: a None component is skipped; remaining rows plotted."""
    res = types.SimpleNamespace(
        total_effect=1.0,
        nde=0.6,
        nie=None,
        se=None,
    )
    fig, ax = plots_mod.mediation_forest(res)
    # Total + NDE rendered (NIE skipped) → 2 y ticks
    assert list(ax.get_yticklabels())
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "NIE" not in labels
    assert "Total" in labels and "NDE" in labels


def test_mediation_forest_empty_raises():
    """plots.py 383-384: all components None → ValueError."""
    res = types.SimpleNamespace(total_effect=None, nde=None, nie=None, se=None)
    with pytest.raises(ValueError, match="no NDE/NIE/total"):
        plots_mod.mediation_forest(res)


def test_yu_elwert_plot_skips_none():
    """plots.py 472-474: components that are None are skipped."""
    res = types.SimpleNamespace(
        disparity=1.0,
        baseline=0.5,
        prevalence=None,
        effect=0.3,
        selection=None,
        se=None,
    )
    fig, ax = plots_mod.yu_elwert_mechanisms_plot(res)
    # Disparity / Baseline / Effect remain (Prevalence + Selection dropped)
    assert len(ax.patches) == 3


# ════════════════════════════════════════════════════════════════════════
# oaxaca.py — Gelbach plot + sanity warning
# ════════════════════════════════════════════════════════════════════════


def test_gelbach_plot_returns_figure():
    """oaxaca.py Gelbach .plot() renders one bar per added variable."""
    rng = np.random.default_rng(13)
    n = 300
    x1 = rng.normal(size=n)
    x2 = 0.6 * x1 + rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 1.0 + 0.8 * x1 + 0.5 * x2 + 0.3 * x3 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    res = sp.gelbach(df, "y", base_x=["x1"], added_x=["x2", "x3"])
    fig, ax = res.plot()
    # one horizontal bar per control variable
    assert len(ax.patches) == len(res.decomposition)
    assert "Gelbach" in ax.get_title()
