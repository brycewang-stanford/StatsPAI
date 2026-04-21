"""Tests for v0.9.17 additions: weakrobust, sbw, gformula_mc, enhanced causal()."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def iv_dataset():
    rng = np.random.default_rng(0)
    n = 300
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = rng.normal(size=n)
    d = 0.6 * z1 + 0.4 * z2 + u + rng.normal(size=n)
    y = 0.5 * d + u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2})


@pytest.fixture
def obs_dataset():
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.3 * x1 - 0.2 * x2)))
    T = rng.binomial(1, p)
    Y = 0.5 + 0.8 * T + 0.5 * x1 + 0.3 * x2 + rng.normal(size=n)
    return pd.DataFrame({"Y": Y, "T": T, "x1": x1, "x2": x2})


@pytest.fixture
def panel_gformula_data():
    """Two-period longitudinal with time-varying confounding."""
    rng = np.random.default_rng(0)
    n = 600
    L0 = rng.normal(size=n)
    A0 = (rng.random(n) < 1 / (1 + np.exp(-0.5 * L0))).astype(float)
    L1 = 0.3 * L0 + 0.4 * A0 + rng.normal(size=n)
    A1 = (rng.random(n) < 1 / (1 + np.exp(-(0.3 * L0 - 0.2 * A0 + 0.4 * L1)))).astype(float)
    Y = 0.2 + 0.4 * A0 + 0.6 * A1 + 0.3 * L0 + 0.2 * L1 + rng.normal(size=n) * 0.3
    return pd.DataFrame({"L0": L0, "A0": A0, "L1": L1, "A1": A1, "Y": Y})


# ═══════════════════════════════════════════════════════════════════════
#  sp.weakrobust — unified weak-IV panel
# ═══════════════════════════════════════════════════════════════════════

class TestWeakRobust:

    def test_basic_runs(self, iv_dataset):
        import statspai as sp
        panel = sp.weakrobust(
            iv_dataset, y="y", endog="d", instruments=["z1", "z2"],
            h0=0.0, clr_simulations=2000, grid_size=101, random_state=1,
        )
        assert panel.n == 300
        assert len(panel.instruments) == 2

    def test_contains_all_statistics(self, iv_dataset):
        import statspai as sp
        panel = sp.weakrobust(
            iv_dataset, y="y", endog="d", instruments=["z1", "z2"],
            clr_simulations=2000, grid_size=101, random_state=1,
        )
        frame = panel.to_frame()
        labels = frame["statistic"].tolist()
        assert any("First-stage" in l for l in labels)
        assert any("effective F" in l for l in labels)
        assert any("Kleibergen–Paap" in l for l in labels)
        assert any("Anderson–Rubin" in l for l in labels)
        assert any("CLR" in l for l in labels)
        assert any("Kleibergen K" in l for l in labels)

    def test_ar_ci_covers_true_beta(self, iv_dataset):
        import statspai as sp
        panel = sp.weakrobust(
            iv_dataset, y="y", endog="d", instruments=["z1", "z2"],
            clr_simulations=3000, grid_size=201, random_state=1,
        )
        lo, hi = panel["ar_ci"]
        assert lo <= 0.5 <= hi, f"true beta=0.5 outside AR CI [{lo},{hi}]"

    def test_clr_ci_tighter_than_ar_with_strong_iv(self, iv_dataset):
        # CLR should be similar-or-tighter than AR for strong IVs
        import statspai as sp
        panel = sp.weakrobust(
            iv_dataset, y="y", endog="d", instruments=["z1", "z2"],
            clr_simulations=3000, grid_size=201, random_state=1,
        )
        ar_width = panel["ar_ci"][1] - panel["ar_ci"][0]
        clr_width = panel["clr_ci"][1] - panel["clr_ci"][0]
        # CLR CI shouldn't be dramatically wider than AR
        assert clr_width < 2.0 * ar_width

    def test_exposed_at_top_level(self):
        import statspai as sp
        assert callable(sp.weakrobust)
        assert hasattr(sp, "WeakRobustResult")

    def test_minimal_panel_when_disabled(self, iv_dataset):
        """include_clr=False and include_k=False must still return AR + KP."""
        import statspai as sp
        panel = sp.weakrobust(
            iv_dataset, y="y", endog="d", instruments=["z1", "z2"],
            include_clr=False, include_k=False, random_state=1,
        )
        data = panel.to_dict()
        assert "ar_stat" in data
        assert "first_stage_F" in data
        assert "kp_rk_lm" in data
        assert data.get("clr_stat") is None
        assert data.get("k_stat") is None


# ═══════════════════════════════════════════════════════════════════════
#  sp.sbw — Stable Balancing Weights
# ═══════════════════════════════════════════════════════════════════════

class TestSBW:

    def test_balance_achieved(self, obs_dataset):
        import statspai as sp
        res = sp.sbw(obs_dataset, treat="T", covariates=["x1", "x2"],
                     delta=0.02)
        # Every covariate's post-weighting |SMD| should be <= delta + tol
        smd_after = res.balance["SMD_after"].abs().max()
        assert smd_after <= 0.025, f"max |SMD_after| = {smd_after}"

    def test_att_estimate_near_truth(self, obs_dataset):
        import statspai as sp
        res = sp.sbw(obs_dataset, treat="T", covariates=["x1", "x2"],
                     y="Y", delta=0.02, estimand="att")
        # True ATT ≈ 0.80
        assert 0.5 < res.estimate < 1.2

    def test_ess_reasonable(self, obs_dataset):
        import statspai as sp
        res = sp.sbw(obs_dataset, treat="T", covariates=["x1", "x2"],
                     delta=0.02)
        n_c = int((obs_dataset["T"] == 0).sum())
        # ESS should be a decent fraction of the control sample
        assert res.effective_sample_size > 0.5 * n_c

    def test_weights_normalised(self, obs_dataset):
        import statspai as sp
        res = sp.sbw(obs_dataset, treat="T", covariates=["x1", "x2"],
                     delta=0.02, estimand="att")
        T = obs_dataset["T"].values
        w_c_sum = res.weights[T == 0].sum()
        assert abs(w_c_sum - 1.0) < 1e-6

    def test_entropy_objective_also_works(self, obs_dataset):
        import statspai as sp
        res = sp.sbw(obs_dataset, treat="T", covariates=["x1", "x2"],
                     delta=0.02, objective="entropy")
        assert res.effective_sample_size > 0

    def test_tight_delta_raises_or_converges(self, obs_dataset):
        import statspai as sp
        # Very tight delta with only 2 covariates should still be feasible
        res = sp.sbw(obs_dataset, treat="T", covariates=["x1", "x2"],
                     delta=0.001)
        smd_after = res.balance["SMD_after"].abs().max()
        assert smd_after <= 0.002

    def test_ate_estimand_runs(self, obs_dataset):
        import statspai as sp
        res = sp.sbw(obs_dataset, treat="T", covariates=["x1", "x2"],
                     y="Y", delta=0.02, estimand="ate")
        # Both arms reweighted, ATE should be finite and in a reasonable band
        assert np.isfinite(res.estimate)
        assert 0.3 < res.estimate < 1.3

    def test_balance_columns_present(self, obs_dataset):
        import statspai as sp
        res = sp.sbw(obs_dataset, treat="T", covariates=["x1", "x2"],
                     delta=0.02)
        for col in ("mean_treated", "mean_control",
                    "SMD_before", "SMD_after"):
            assert col in res.balance.columns
        assert set(res.balance.index) == {"x1", "x2"}

    def test_citation_key_attached(self, obs_dataset):
        import statspai as sp
        res = sp.sbw(obs_dataset, treat="T", covariates=["x1", "x2"],
                     delta=0.02)
        # SBWResult hooks into the CausalResult citation registry
        assert getattr(res, "_citation_key", None) == "zubizarreta_2015_sbw"


# ═══════════════════════════════════════════════════════════════════════
#  sp.gformula_mc — Monte Carlo parametric g-formula
# ═══════════════════════════════════════════════════════════════════════

class TestGFormulaMC:

    def test_runs_on_two_period_data(self, panel_gformula_data):
        import statspai as sp
        res = sp.gformula_mc(
            panel_gformula_data,
            treatment_cols=["A0", "A1"],
            confounder_cols=[["L0"], ["L1"]],
            outcome_col="Y",
            strategy=[1, 1],
            control_strategy=[0, 0],
            n_simulations=3000,
            bootstrap=50,
            seed=1,
        )
        assert res.n_simulations == 3000

    def test_ate_matches_truth(self, panel_gformula_data):
        import statspai as sp
        res = sp.gformula_mc(
            panel_gformula_data,
            treatment_cols=["A0", "A1"],
            confounder_cols=[["L0"], ["L1"]],
            outcome_col="Y",
            strategy=[1, 1],
            control_strategy=[0, 0],
            n_simulations=5000,
            bootstrap=100,
            seed=1,
        )
        # True ATE = direct 0.4+0.6=1.0 + mediated 0.4*0.2=0.08 ≈ 1.08
        assert 0.9 < res.contrast_value < 1.3

    def test_bootstrap_ci_contains_contrast(self, panel_gformula_data):
        import statspai as sp
        res = sp.gformula_mc(
            panel_gformula_data,
            treatment_cols=["A0", "A1"],
            confounder_cols=[["L0"], ["L1"]],
            outcome_col="Y",
            strategy=[1, 1],
            control_strategy=[0, 0],
            n_simulations=3000,
            bootstrap=80,
            seed=1,
        )
        lo, hi = res.contrast_ci
        assert lo <= res.contrast_value <= hi

    def test_dynamic_regime(self, panel_gformula_data):
        """A callable strategy that depends on simulated L0/L1 must work."""
        import statspai as sp

        def dynamic(t, hist):
            key = f"L{t}"
            return (hist[key] > 0).astype(float) if key in hist else np.zeros(len(next(iter(hist.values()))))

        res = sp.gformula_mc(
            panel_gformula_data,
            treatment_cols=["A0", "A1"],
            confounder_cols=[["L0"], ["L1"]],
            outcome_col="Y",
            strategy=dynamic,
            n_simulations=2000,
            bootstrap=0,
            seed=2,
        )
        assert np.isfinite(res.value)

    def test_exposed_at_top_level(self):
        import statspai as sp
        assert callable(sp.gformula_mc)
        assert hasattr(sp, "MCGFormulaResult")

    def test_bootstrap_zero_returns_nan_se(self, panel_gformula_data):
        import statspai as sp
        res = sp.gformula_mc(
            panel_gformula_data,
            treatment_cols=["A0", "A1"],
            confounder_cols=[["L0"], ["L1"]],
            outcome_col="Y",
            strategy=[1, 1],
            control_strategy=[0, 0],
            n_simulations=1000,
            bootstrap=0,
            seed=3,
        )
        assert np.isnan(res.se)
        assert np.isnan(res.ci[0]) and np.isnan(res.ci[1])
        # Contrast was requested, so contrast_se should be NaN (not None)
        assert np.isnan(res.contrast_se)

    def test_degenerate_binary_column_does_not_crash(self):
        """A confounder that is identically 0 must not blow up the
        logistic Newton-Raphson loop."""
        import statspai as sp
        rng = np.random.default_rng(42)
        n = 200
        L0 = np.zeros(n)  # degenerate
        A0 = (rng.random(n) < 0.5).astype(float)
        Y = 0.5 + 0.6 * A0 + rng.normal(size=n) * 0.3
        df = pd.DataFrame({"L0": L0, "A0": A0, "Y": Y})
        res = sp.gformula_mc(
            df, treatment_cols=["A0"], confounder_cols=[["L0"]],
            outcome_col="Y", strategy=[1], control_strategy=[0],
            n_simulations=500, bootstrap=0, seed=0,
        )
        assert np.isfinite(res.value)


# ═══════════════════════════════════════════════════════════════════════
#  sp.causal() enhanced workflow
# ═══════════════════════════════════════════════════════════════════════

class TestCausalWorkflow:

    def test_auto_run_triggers_all_stages(self, obs_dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import statspai as sp
            w = sp.causal(obs_dataset, y="Y", treatment="T",
                          covariates=["x1", "x2"],
                          design="observational", auto_run=True)
        stages = set(w.stages_completed)
        assert "diagnose" in stages
        assert "estimate" in stages
        assert "compare_estimators" in stages
        assert "sensitivity_panel" in stages
        assert "cate" in stages

    def test_multi_estimator_comparison_has_rows(self, obs_dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import statspai as sp
            w = sp.causal(obs_dataset, y="Y", treatment="T",
                          covariates=["x1", "x2"],
                          design="observational", auto_run=True)
        cmp = w.estimator_comparison
        assert isinstance(cmp, pd.DataFrame)
        # At least OLS + one weighting + DML should be there
        assert len(cmp) >= 3
        # Point estimates should be finite for the main methods
        numeric = cmp["estimate"].dropna().values
        assert len(numeric) >= 3
        # And they should cluster reasonably near the true ATT (0.8)
        assert np.median(numeric) > 0.4
        assert np.median(numeric) < 1.3

    def test_sensitivity_panel_populated(self, obs_dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import statspai as sp
            w = sp.causal(obs_dataset, y="Y", treatment="T",
                          covariates=["x1", "x2"],
                          design="observational", auto_run=True)
        panel = w.sensitivity_panel_result
        assert isinstance(panel, pd.DataFrame)
        # Expect E-value and Oster δ* at minimum
        methods = panel["method"].tolist()
        assert any("E-value" in m or "Oster" in m for m in methods)

    def test_cate_runs_when_covariates_present(self, obs_dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import statspai as sp
            w = sp.causal(obs_dataset, y="Y", treatment="T",
                          covariates=["x1", "x2"],
                          design="observational", auto_run=True)
        cate = w.cate_summary_table
        assert isinstance(cate, pd.DataFrame)
        assert len(cate) >= 1  # At least one learner succeeded

    def test_report_contains_new_sections(self, obs_dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import statspai as sp
            w = sp.causal(obs_dataset, y="Y", treatment="T",
                          covariates=["x1", "x2"],
                          design="observational", auto_run=True)
        md = w.report(fmt="markdown")
        assert "## 4b. Multi-estimator comparison" in md
        assert "## 4c. Sensitivity triad" in md
        assert "## 4d. Heterogeneity (CATE)" in md

    def test_compare_estimators_records_error_row(self, obs_dataset):
        """A genuinely-failing estimator should produce an ERROR note,
        not a silent NaN without context."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import statspai as sp
            # Build a dataset where DML won't converge cleanly (tiny n)
            tiny = obs_dataset.sample(20, random_state=0)
            w = sp.causal(tiny, y="Y", treatment="T",
                          covariates=["x1", "x2"],
                          design="observational", auto_run=False)
            w.run(full=True)
        cmp = w.estimator_comparison
        # The panel must at least exist
        assert isinstance(cmp, pd.DataFrame)
        # Every row with a NaN estimate must carry a non-empty note
        bad = cmp[cmp["estimate"].isna()]
        if len(bad):
            assert bad["note"].str.contains("ERROR").any(), \
                "NaN rows must explain why"

    def test_run_full_false_skips_extended_stages(self, obs_dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import statspai as sp
            w = sp.causal(obs_dataset, y="Y", treatment="T",
                          covariates=["x1", "x2"],
                          design="observational", auto_run=False)
            w.run(full=False)
        stages = set(w.stages_completed)
        assert "estimate" in stages
        # New extended stages not triggered
        assert "compare_estimators" not in stages
        assert "sensitivity_panel" not in stages
        assert "cate" not in stages
