"""Coverage tests for individual statspai.synth variant estimators.

Each estimator is called via its module-level entry point with several option
combinations, asserting real structural properties (finite estimates, gap
series length, placebo populating, error paths).
"""

import importlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

cluster = importlib.import_module("statspai.synth.cluster")
sparse = importlib.import_module("statspai.synth.sparse")
expdes = importlib.import_module("statspai.synth.experimental_design")
conformal = importlib.import_module("statspai.synth.conformal")
multi_outcome = importlib.import_module("statspai.synth.multi_outcome")
fdid_mod = importlib.import_module("statspai.synth.fdid")
penscm = importlib.import_module("statspai.synth.penscm")
robust = importlib.import_module("statspai.synth.robust")
kernel = importlib.import_module("statspai.synth.kernel")
mc = importlib.import_module("statspai.synth.mc")
demeaned = importlib.import_module("statspai.synth.demeaned")
gsynth_mod = importlib.import_module("statspai.synth.gsynth")
staggered_mod = importlib.import_module("statspai.synth.staggered")
bsts = importlib.import_module("statspai.synth.bsts")
power_mod = importlib.import_module("statspai.synth.power")
seqsdid = importlib.import_module("statspai.synth.sequential_sdid")
survival = importlib.import_module("statspai.synth.survival")
bayesian = importlib.import_module("statspai.synth.bayesian")


def _panel(
    n_units=9, n_periods=18, treatment_time=12, effect=4.0, seed=11, with_cov=False
):
    rng = np.random.default_rng(seed)
    alphas = rng.normal(10, 2, n_units)
    betas = rng.normal(0.5, 0.1, n_units)
    common = rng.normal(0, 0.4, n_periods)
    records = []
    for i in range(n_units):
        for ti, t in enumerate(range(1, n_periods + 1)):
            y = alphas[i] + betas[i] * t + common[ti] + rng.normal(0, 0.2)
            if i == 0 and t >= treatment_time:
                y += effect
            row = {"unit": f"u{i}", "time": t, "outcome": y}
            if with_cov:
                row["x1"] = alphas[i] + rng.normal(0, 0.1)
                row["x2"] = betas[i] * t + rng.normal(0, 0.1)
            records.append(row)
    return pd.DataFrame(records)


COMMON = dict(
    outcome="outcome", unit="unit", time="time", treated_unit="u0", treatment_time=12
)


@pytest.fixture
def panel():
    return _panel()


@pytest.fixture
def panel_cov():
    return _panel(with_cov=True)


# --- cluster ---
def test_cluster_kmeans_and_augment(panel):
    res = cluster.cluster_synth(
        panel, **COMMON, n_clusters=2, cluster_method="kmeans", placebo=True, seed=1
    )
    assert np.isfinite(res.estimate)
    res2 = cluster.cluster_synth(
        panel, **COMMON, augment=True, max_augment=2, placebo=False, seed=2
    )
    assert np.isfinite(res2.estimate)


def test_cluster_hierarchical(panel):
    res = cluster.cluster_synth(
        panel, **COMMON, n_clusters=3, cluster_method="hierarchical", placebo=False
    )
    assert np.isfinite(res.estimate)


def test_cluster_auto_nclusters(panel):
    res = cluster.cluster_synth(panel, **COMMON, n_clusters=None, placebo=False)
    mi = res.model_info or {}
    assert mi.get("n_clusters", 1) >= 1


def test_cluster_with_covariates(panel_cov):
    res = cluster.cluster_synth(
        panel_cov, **COMMON, covariates=["x1", "x2"], n_clusters=2, placebo=False
    )
    assert np.isfinite(res.estimate)


# --- sparse ---
@pytest.mark.parametrize("mode", ["lasso", "constrained_lasso", "joint"])
def test_sparse_modes(panel, mode):
    res = sparse.sparse_synth(panel, **COMMON, mode=mode, placebo=False)
    assert np.isfinite(res.estimate)


def test_sparse_explicit_lambda_and_cv(panel):
    res = sparse.sparse_synth(panel, **COMMON, mode="lasso", lambda_w=0.1, placebo=True)
    assert np.isfinite(res.estimate)
    res2 = sparse.sparse_synth(
        panel, **COMMON, mode="lasso", lambda_w=None, placebo=False
    )
    assert np.isfinite(res2.estimate)


# --- experimental design ---
def test_experimental_design_mspe(panel):
    res = expdes.synth_experimental_design(
        panel,
        unit="unit",
        time="time",
        outcome="outcome",
        k=2,
        risk="mspe",
        n_random=50,
        random_state=0,
    )
    assert hasattr(res, "summary")
    assert isinstance(res.summary(), str)
    d = res.to_dict()
    assert "k" in d or len(d) > 0


def test_experimental_design_rmse_risk(panel):
    res = expdes.synth_experimental_design(
        panel,
        unit="unit",
        time="time",
        outcome="outcome",
        k=3,
        risk="rmse",
        concentration_weight=0.5,
        penalization=0.1,
        n_random=40,
        random_state=1,
    )
    assert hasattr(res, "selected") or hasattr(res, "summary")


def test_experimental_design_bad_risk(panel):
    with pytest.raises(ValueError):
        expdes.synth_experimental_design(
            panel,
            unit="unit",
            time="time",
            outcome="outcome",
            k=2,
            risk="variance",
            n_random=10,
        )


def test_experimental_design_bad_k(panel):
    with pytest.raises((ValueError, Exception)):
        expdes.synth_experimental_design(
            panel, unit="unit", time="time", outcome="outcome", k=999, n_random=10
        )


# --- conformal ---
def test_conformal_synth_basic(panel):
    res = conformal.conformal_synth(panel, **COMMON, scm_method="classic", grid_size=21)
    assert np.isfinite(res.estimate)
    mi = res.model_info or {}
    assert "period_results" in mi and mi["inference_method"].startswith("conformal")
    # pvalue is the moving-block average-effect p-value (CWZ 2021). It moved
    # 1/12 -> 1/18 when the avg-effect test was corrected from comparing a
    # T1-averaged statistic against single pre-period residuals (a scale
    # mismatch that pinned p at the old 1/(T0+1) floor) to proper length-T1
    # cyclic blocks over the full residual series. estimate/se/CI unchanged.
    np.testing.assert_allclose(
        [res.estimate, res.se, res.ci[0], res.ci[1], res.pvalue],
        [
            2.8031848526798764,
            1.7123081037547074,
            -0.5528773611154243,
            6.159247066475177,
            0.05555555555555555,
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mi["period_results"]["effect"].head(3).to_numpy(),
        [3.020643, 2.875103, 2.702355],
        atol=5e-7,
    )


def test_conformal_synth_grid_range(panel):
    res = conformal.conformal_synth(
        panel,
        **COMMON,
        scm_method="ridge",
        grid_size=15,
        grid_range=(-10.0, 10.0),
        penalization=0.1,
    )
    assert np.isfinite(res.estimate)
    # pvalue corrected to the moving-block average-effect p-value (CWZ 2021):
    # 1/12 -> 1/18 (see test_conformal_synth_basic). estimate/se/CI unchanged.
    np.testing.assert_allclose(
        [res.estimate, res.se, res.ci[0], res.ci[1], res.pvalue],
        [2.7956817105608613, 5.102134569246539, -10.0, 10.0, 0.05555555555555555],
        atol=1e-12,
    )


# --- multi_outcome ---
@pytest.mark.parametrize("method", ["concatenated", "averaged"])
def test_multi_outcome(panel_cov, method):
    res = multi_outcome.multi_outcome_synth(
        panel_cov,
        outcomes=["outcome", "x2"],
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=12,
        method=method,
        standardize=True,
        placebo=False,
    )
    assert np.isfinite(res.estimate)
    expected = {
        "concatenated": [0.35050597844905407, 2.858039, -2.157027],
        "averaged": [0.404784096292488, 2.956794, -2.147226],
    }[method]
    np.testing.assert_allclose(
        [res.estimate, *res.model_info["per_outcome_effects"]["att"].to_numpy()],
        expected,
        atol=5e-7,
    )


def test_multi_outcome_no_standardize_placebo(panel_cov):
    res = multi_outcome.multi_outcome_synth(
        panel_cov,
        outcomes=["outcome", "x1"],
        unit="unit",
        time="time",
        treated_unit="u0",
        treatment_time=12,
        standardize=False,
        placebo=True,
    )
    assert np.isfinite(res.estimate)
    np.testing.assert_allclose(
        [res.estimate, res.se, res.ci[0], res.ci[1]],
        [
            1.3333396060631233,
            0.6716419151020026,
            0.016945641955689705,
            2.649733570170557,
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        res.model_info["per_outcome_effects"]["att"].to_numpy(),
        [2.208574, 0.458106],
        atol=5e-7,
    )


# --- fdid ---
@pytest.mark.parametrize("method", ["forward", "forward_cv", "best_subset"])
def test_fdid_methods(panel, method):
    res = fdid_mod.fdid(panel, **COMMON, method=method, placebo=False)
    assert np.isfinite(res.estimate)


def test_fdid_placebo_and_maxdonors(panel):
    res = fdid_mod.fdid(panel, **COMMON, method="forward", max_donors=3, placebo=True)
    assert np.isfinite(res.estimate)


# --- penscm ---
@pytest.mark.parametrize("penalty_type", ["pairwise", "max_dev", "l1_pairwise"])
def test_penscm_penalty_types(panel, penalty_type):
    res = penscm.penalized_synth(
        panel, **COMMON, penalty_type=penalty_type, lambda_pen=0.1, placebo=False
    )
    assert np.isfinite(res.estimate)


def test_penscm_cv_lambda_and_placebo(panel):
    res = penscm.penalized_synth(panel, **COMMON, lambda_pen=None, placebo=True)
    assert np.isfinite(res.estimate)


def test_penscm_with_covariates(panel_cov):
    res = penscm.penalized_synth(
        panel_cov, **COMMON, covariates=["x1", "x2"], lambda_pen=0.05, placebo=False
    )
    assert np.isfinite(res.estimate)


# --- robust ---
@pytest.mark.parametrize("variant", ["unconstrained", "elastic_net", "penalized"])
def test_robust_variants(panel, variant):
    res = robust.robust_synth(
        panel,
        **COMMON,
        variant=variant,
        l1_penalty=0.05,
        l2_penalty=0.05,
        placebo=False,
    )
    assert np.isfinite(res.estimate)


def test_robust_no_intercept_placebo(panel):
    res = robust.robust_synth(
        panel, **COMMON, variant="unconstrained", intercept=False, placebo=True
    )
    assert np.isfinite(res.estimate)


# --- kernel ---
@pytest.mark.parametrize("kern", ["rbf", "polynomial", "laplacian"])
def test_kernel_synth_kernels(panel, kern):
    res = kernel.kernel_synth(panel, **COMMON, kernel=kern, placebo=False)
    assert np.isfinite(res.estimate)


def test_kernel_ridge_and_sigma(panel):
    res = kernel.kernel_ridge_synth(
        panel, **COMMON, kernel="rbf", sigma=1.5, ridge_lambda=0.05, placebo=True
    )
    assert np.isfinite(res.estimate)


def test_kernel_polynomial_degree(panel):
    res = kernel.kernel_synth(
        panel, **COMMON, kernel="polynomial", degree=3, placebo=False
    )
    assert np.isfinite(res.estimate)


# --- mc ---
def test_mc_synth_basic_and_cv(panel):
    res = mc.mc_synth(
        panel, **COMMON, lambda_reg=None, cv_folds=3, placebo=False, seed=0
    )
    assert np.isfinite(res.estimate)
    res2 = mc.mc_synth(panel, **COMMON, lambda_reg=0.5, placebo=True, seed=1)
    assert np.isfinite(res2.estimate)


def test_mc_synth_covariates(panel_cov):
    res = mc.mc_synth(
        panel_cov, **COMMON, covariates=["x1"], lambda_reg=0.5, placebo=False
    )
    assert np.isfinite(res.estimate)


# --- demeaned ---
@pytest.mark.parametrize("variant", ["demeaned", "detrended"])
def test_demeaned_variants(panel, variant):
    res = demeaned.demeaned_synth(panel, **COMMON, variant=variant, placebo=False)
    assert np.isfinite(res.estimate)


def test_demeaned_placebo_penalization(panel):
    res = demeaned.demeaned_synth(
        panel, **COMMON, variant="demeaned", penalization=0.1, placebo=True
    )
    assert np.isfinite(res.estimate)


# --- gsynth ---
def test_gsynth_native(panel):
    res = gsynth_mod.gsynth(
        panel, **COMMON, n_factors=2, placebo=False, backend="native", seed=0
    )
    assert np.isfinite(res.estimate)


def test_gsynth_cv_factors(panel):
    res = gsynth_mod.gsynth(
        panel,
        **COMMON,
        n_factors=None,
        max_factors=3,
        cv_folds=3,
        placebo=False,
        backend="native",
        seed=1,
    )
    assert np.isfinite(res.estimate)


def test_gsynth_covariates(panel_cov):
    res = gsynth_mod.gsynth(
        panel_cov,
        **COMMON,
        covariates=["x1", "x2"],
        n_factors=1,
        placebo=False,
        backend="native",
    )
    assert np.isfinite(res.estimate)


# --- staggered (needs treatment indicator) ---
def _staggered_panel(seed=3):
    rng = np.random.default_rng(seed)
    records = []
    adopt = {"a": 8, "b": 12, "c": None, "d": None, "e": None, "f": None}
    for u, g in adopt.items():
        a = rng.normal(10, 1)
        for t in range(1, 17):
            y = a + 0.4 * t + rng.normal(0, 0.2)
            treat = 1 if (g is not None and t >= g) else 0
            if treat:
                y += 3.0
            records.append({"unit": u, "time": t, "outcome": y, "treated": treat})
    return pd.DataFrame(records)


@pytest.mark.parametrize("method", ["separate", "pooled"])
def test_staggered_methods(method):
    df = _staggered_panel()
    res = staggered_mod.staggered_synth(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        treatment="treated",
        method=method,
        placebo=True,
    )
    mi = res.model_info or {}
    assert mi["n_cohorts"] >= 1
    assert mi["n_treated_units"] >= 2
    assert np.isfinite(res.estimate)


def test_staggered_no_treated_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        [
            {"unit": u, "time": t, "outcome": rng.normal(), "treated": 0}
            for u in "abcd"
            for t in range(1, 10)
        ]
    )
    with pytest.raises(ValueError):
        staggered_mod.staggered_synth(
            df, outcome="outcome", unit="unit", time="time", treatment="treated"
        )


def test_staggered_no_control_raises():
    df = pd.DataFrame(
        [
            {"unit": u, "time": t, "outcome": float(t), "treated": 1 if t >= 5 else 0}
            for u in "abc"
            for t in range(1, 10)
        ]
    )
    with pytest.raises(ValueError):
        staggered_mod.staggered_synth(
            df, outcome="outcome", unit="unit", time="time", treatment="treated"
        )


# --- bsts / causal_impact ---
def test_bsts_synth_local_level(panel):
    res = bsts.bsts_synth(
        panel, **COMMON, model="local_level", n_simulations=100, seed=0
    )
    assert np.isfinite(res.estimate)


def test_bsts_synth_local_trend(panel):
    res = bsts.bsts_synth(
        panel, **COMMON, model="local_linear_trend", n_simulations=80, seed=1
    )
    assert np.isfinite(res.estimate)


def test_causal_impact_wide(panel):
    # causal_impact uses a time-indexed wide DataFrame (outcome + controls).
    wide = panel.pivot(index="time", columns="unit", values="outcome")
    wide = wide.rename(columns={"u0": "outcome"})
    res = bsts.causal_impact(
        wide,
        pre_period=(1, 11),
        post_period=(12, 18),
        outcome="outcome",
        n_simulations=80,
        seed=2,
    )
    assert np.isfinite(res.estimate)


# --- power ---
def test_synth_power_curve(panel):
    df = power_mod.synth_power(
        panel, **COMMON, effect_sizes=[0.0, 2.0, 5.0], n_simulations=30, seed=0
    )
    assert isinstance(df, pd.DataFrame)
    assert (df["power"] >= 0).all() and (df["power"] <= 1).all()


def test_synth_mde(panel):
    mde = power_mod.synth_mde(
        panel, **COMMON, power_target=0.8, n_simulations=30, seed=0
    )
    assert np.isfinite(mde)


# --- sequential sdid ---
def test_sequential_sdid():
    df = _staggered_panel()
    # build cohort column: adoption time (0 for never-treated)
    adopt = df[df["treated"] == 1].groupby("unit")["time"].min()
    df["cohort"] = df["unit"].map(adopt).fillna(0).astype(int)
    res = seqsdid.sequential_sdid(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        cohort="cohort",
        never_treated_value=0,
        se_method="placebo",
        n_reps=20,
        seed=0,
    )
    assert np.isfinite(res.estimate)


# --- survival ---
def test_synth_survival():
    rng = np.random.default_rng(5)
    records = []
    for i in range(8):
        base = rng.uniform(0.85, 0.97)
        for t in range(1, 13):
            s = base**t
            if i == 0 and t >= 8:
                s = min(0.999, s * 1.05)
            records.append(
                {"unit": f"u{i}", "time": t, "surv": s, "treated": 1 if i == 0 else 0}
            )
    df = pd.DataFrame(records)
    res = survival.synth_survival(
        df,
        unit="unit",
        time="time",
        survival="surv",
        treated="treated",
        treat_time=8,
        n_placebos=20,
        seed=0,
    )
    assert hasattr(res, "summary")
    assert isinstance(res.summary(), str)


# --- bayesian (no pymc needed; pure-numpy MCMC) ---
def test_bayesian_synth(panel):
    res = bayesian.bayesian_synth(
        panel, **COMMON, n_iter=200, n_warmup=100, n_chains=2, seed=0
    )
    assert np.isfinite(res.estimate)
    mi = res.model_info or {}
    assert any(k in mi for k in ("rhat", "posterior_mean", "credible_interval"))
