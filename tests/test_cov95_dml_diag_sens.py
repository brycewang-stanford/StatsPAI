"""Coverage tests for statspai.dml._diagnostics, _sensitivity, and the
PLR / IRM estimator branches.

Exercises summary string building, overlap tables (PLR-residual vs.
IRM-propensity paths), orthogonality / balance computation, the
robustness-value solver, benchmark covariates, the bias-bound scenario,
and error paths. Real synthetic data, real residuals from a fit DML
model — no fabricated numbers.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.dml import dml, dml_diagnostics, dml_sensitivity
from statspai.dml._sensitivity import _robustness_value


@pytest.fixture
def plr_result():
    rng = np.random.default_rng(7)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = np.cos(x1) + x2 + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + np.sin(x1) + x2**2 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    return dml(df, y="y", treat="d", covariates=["x1", "x2"], model="plr")


@pytest.fixture
def irm_result():
    rng = np.random.default_rng(11)
    n = 1200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    prob = 1.0 / (1.0 + np.exp(-(0.5 * x1 + x2)))
    d = rng.binomial(1, prob, n).astype(float)
    y = 3.0 * d + x1 + x2**2 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    return dml(df, y="y", treat="d", covariates=["x1", "x2"], model="irm")


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def test_diagnostics_plr_path(plr_result):
    diag = dml_diagnostics(plr_result)
    # PLR has no propensity → overlap uses |D - m(X)| residual.
    assert "residual" in diag._overlap_label.lower() or "D" in diag._overlap_label
    assert diag.method == "PLR"
    assert not diag.overlap_table.empty
    assert diag.score_sd > 0
    assert np.isfinite(diag.orth_stat)
    # Balance table built from the stashed design matrix.
    assert not diag.balance_table.empty
    assert set(["variable", "corr_d_resid", "corr_y_resid"]).issubset(
        diag.balance_table.columns
    )


def test_diagnostics_plr_summary(plr_result):
    diag = dml_diagnostics(plr_result)
    s = diag.summary()
    assert "DML Diagnostics" in s
    assert "Overlap" in s
    assert "Orthogonality" in s
    assert "Balance" in s


def test_diagnostics_irm_propensity_path(irm_result):
    diag = dml_diagnostics(irm_result, clip=0.05)
    assert "propensity" in diag._overlap_label.lower()
    # IRM overlap table uses the propensity quantile rows.
    assert "p1" in diag.overlap_table["quantile"].tolist()
    assert diag.n_clipped_low >= 0
    assert diag.n_clipped_high >= 0


def test_diagnostics_missing_residuals_raises():
    from statspai.core.results import CausalResult

    bogus = CausalResult(
        method="x",
        estimand="ATE",
        estimate=1.0,
        se=0.1,
        pvalue=0.5,
        ci=(0.8, 1.2),
        alpha=0.05,
        n_obs=10,
        model_info={},
    )
    with pytest.raises(ValueError, match="post-fit residuals"):
        dml_diagnostics(bogus)


def test_diagnostics_irm_overlap_warning(irm_result):
    # A large clip forces some units outside [clip, 1-clip] → a warning.
    diag = dml_diagnostics(irm_result, clip=0.3)
    # With clip=0.3 it is very likely some propensity falls outside;
    # the warning string is built when n_low+n_high>0.
    if diag.n_clipped_low + diag.n_clipped_high > 0:
        assert diag.overlap_warning is not None
        assert "%" in diag.overlap_warning


# --------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------
def test_robustness_value_solver():
    # Monotonic: bigger target needs more confounding.
    assert _robustness_value(0.0, 1.0) == 0.0
    rv_small = _robustness_value(0.5, 2.0)
    rv_big = _robustness_value(1.5, 2.0)
    assert 0.0 <= rv_small <= rv_big <= 1.0


def test_robustness_value_bad_s():
    assert np.isnan(_robustness_value(1.0, 0.0))
    assert np.isnan(_robustness_value(1.0, float("nan")))


def test_sensitivity_basic(plr_result):
    res = dml_sensitivity(plr_result)
    assert 0.0 <= res.rv_q <= 1.0
    assert 0.0 <= res.rv_qa <= res.rv_q + 1e-9
    assert res.s > 0
    assert np.isnan(res.bias_bound)  # no cf_y/cf_d given
    s = res.summary()
    assert "DML-OVB" in s
    assert "Robustness value" in s


def test_sensitivity_with_scenario(plr_result):
    res = dml_sensitivity(plr_result, cf_y=0.1, cf_d=0.1)
    assert np.isfinite(res.bias_bound)
    assert res.adjusted_estimate_low < res.estimate < res.adjusted_estimate_high
    s = res.summary()
    assert "Bias bound" in s
    assert "Adjusted estimate range" in s


def test_sensitivity_benchmarks(plr_result):
    res = dml_sensitivity(
        plr_result, benchmark_covariates=["x1", "x2"], k_y=1.0, k_d=1.0
    )
    assert not res.benchmarks.empty
    assert set(["variable", "cf_y_bench", "cf_d_bench", "bias_bound"]).issubset(
        res.benchmarks.columns
    )
    # benchmarks render in the summary
    assert "benchmarks" in res.summary()


def test_sensitivity_benchmark_unknown_covariate_skipped(plr_result):
    res = dml_sensitivity(plr_result, benchmark_covariates=["not_a_cov"])
    assert res.benchmarks.empty


def test_sensitivity_missing_residuals_raises():
    from statspai.core.results import CausalResult

    bogus = CausalResult(
        method="x",
        estimand="ATE",
        estimate=1.0,
        se=0.1,
        pvalue=0.5,
        ci=(0.8, 1.2),
        alpha=0.05,
        n_obs=10,
        model_info={},
    )
    with pytest.raises(ValueError, match="post-fit residuals"):
        dml_sensitivity(bogus)


def test_sensitivity_irm(irm_result):
    res = dml_sensitivity(irm_result, cf_y=0.05, cf_d=0.05)
    assert np.isfinite(res.rv_q)
    assert np.isfinite(res.bias_bound)


# --------------------------------------------------------------------------
# estimator branches: multi-rep PLR, IRM error paths
# --------------------------------------------------------------------------
def test_plr_multi_rep_aggregation():
    rng = np.random.default_rng(21)
    n = 500
    x = rng.normal(size=n)
    d = x + rng.normal(scale=0.5, size=n)
    y = 1.0 * d + x + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    res = dml(df, y="y", treat="d", covariates=["x"], n_rep=3)
    assert res.model_info["n_rep"] == 3
    assert "theta_all_reps" in res.model_info
    assert len(res.model_info["theta_all_reps"]) == 3


def test_irm_rejects_continuous_treatment():
    rng = np.random.default_rng(31)
    n = 300
    x = rng.normal(size=n)
    d = x + rng.normal(size=n)  # continuous
    y = d + x + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility):
        dml(df, y="y", treat="d", covariates=["x"], model="irm")


def test_irm_too_few_per_arm_raises():
    # n_folds larger than the minority-arm count.
    rng = np.random.default_rng(41)
    n = 40
    x = rng.normal(size=n)
    d = np.zeros(n)
    d[:3] = 1.0  # only 3 treated
    y = 3.0 * d + x + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    from statspai.exceptions import IdentificationFailure

    with pytest.raises(IdentificationFailure):
        dml(df, y="y", treat="d", covariates=["x"], model="irm", n_folds=5)


def test_irm_subgroup_mean_fallback():
    # ~14 treated rows, n_folds=2 → ~7 treated per training fold, below
    # _MIN_SUBGROUP_FIT (10) → g(1, X) uses the subgroup-mean fallback
    # (irm.py lines 108-121).
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=n)
    d = np.zeros(n)
    d[:14] = 1.0
    y = 3.0 * d + x + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    res = dml(df, y="y", treat="d", covariates=["x"], model="irm", n_folds=2)
    diags = res.model_info["diagnostics"]
    assert diags["n_subgroup_fallback_g1"] > 0


def test_irm_weighted_with_fallback():
    # Weighted IRM (lines 186-190) AND a small treated arm so the
    # weighted subgroup-mean fallback (lines 113-118) is exercised.
    rng = np.random.default_rng(4)
    n = 200
    x = rng.normal(size=n)
    d = np.zeros(n)
    d[:14] = 1.0
    y = 3.0 * d + x + rng.normal(scale=0.5, size=n)
    w = np.abs(rng.normal(size=n)) + 0.1
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    res = dml(
        df, y="y", treat="d", covariates=["x"], model="irm", n_folds=2, sample_weight=w
    )
    assert np.isfinite(res.estimate)
    assert res.model_info["diagnostics"]["weighted"] is True


def test_irm_control_arm_fallback():
    # ~14 control rows, n_folds=2 → small D=0 training subgroup →
    # g(0, X) uses the subgroup-mean fallback (irm.py lines 142-150).
    rng = np.random.default_rng(6)
    n = 200
    x = rng.normal(size=n)
    d = np.ones(n)
    d[:14] = 0.0
    y = 3.0 * d + x + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    res = dml(df, y="y", treat="d", covariates=["x"], model="irm", n_folds=2)
    assert res.model_info["diagnostics"]["n_subgroup_fallback_g0"] > 0


def test_irm_control_arm_weighted_fallback():
    rng = np.random.default_rng(7)
    n = 200
    x = rng.normal(size=n)
    d = np.ones(n)
    d[:14] = 0.0
    y = 3.0 * d + x + rng.normal(scale=0.5, size=n)
    w = np.abs(rng.normal(size=n)) + 0.1
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    res = dml(
        df, y="y", treat="d", covariates=["x"], model="irm", n_folds=2, sample_weight=w
    )
    assert np.isfinite(res.estimate)


def test_irm_regressor_ml_m_predict_path():
    # A regressor ml_m has no predict_proba → exercises the m_hat =
    # ml_m.predict(X_te) branch (irm.py line 170).
    rng = np.random.default_rng(5)
    n = 800
    x = rng.normal(size=n)
    prob = 1.0 / (1.0 + np.exp(-(0.5 * x)))
    d = rng.binomial(1, prob, n).astype(float)
    y = 2.0 * d + x + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    from sklearn.linear_model import LinearRegression

    res = dml(
        df,
        y="y",
        treat="d",
        covariates=["x"],
        model="irm",
        n_folds=3,
        ml_m=LinearRegression(),
    )
    assert np.isfinite(res.estimate)


def test_irm_diagnostics_populated(irm_result):
    diags = irm_result.model_info.get("diagnostics", {})
    assert "pscore_min" in diags
    assert "pscore_max" in diags
    assert "n_clipped_below" in diags


# --------------------------------------------------------------------------
# plot methods (matplotlib available) + summary warning branches +
# degenerate-input branches
# --------------------------------------------------------------------------
def test_diagnostics_plot(plr_result):
    import matplotlib

    matplotlib.use("Agg")
    diag = dml_diagnostics(plr_result)
    fig, axes = diag.plot()
    assert fig is not None
    assert axes.shape == (2, 2)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_diagnostics_plot_irm(irm_result):
    import matplotlib

    matplotlib.use("Agg")
    diag = dml_diagnostics(irm_result, clip=0.3)
    fig, axes = diag.plot(bins=15)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_sensitivity_plot(plr_result):
    import matplotlib

    matplotlib.use("Agg")
    res = dml_sensitivity(plr_result, benchmark_covariates=["x1"])
    fig, ax = res.plot()
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def _make_result(
    y_resid, d_resid, *, pscore=None, X=None, cov_names=None, estimate=1.0, se=0.1
):
    from statspai.core.results import CausalResult

    info = {"dml_model": "PLR", "_y_resid": y_resid, "_d_resid": d_resid}
    if pscore is not None:
        info["_pscore"] = pscore
    if X is not None:
        info["_X_design"] = X
        info["_covariate_names"] = cov_names
    return CausalResult(
        method="Double ML (PLR)",
        estimand="ATE",
        estimate=estimate,
        se=se,
        pvalue=0.5,
        ci=(0.0, 2.0),
        alpha=0.05,
        n_obs=len(y_resid),
        model_info=info,
    )


def test_diagnostics_summary_warnings():
    # A propensity outside the clip → overlap warning (line 90, rendered
    # in summary). Constant covariate column → balance zero-fill branch
    # (lines 291-293).
    rng = np.random.default_rng(55)
    n = 500
    y_resid = rng.normal(size=n)
    d_resid = rng.normal(size=n)
    pscore = np.clip(rng.uniform(0, 1, n), 0, 1)
    pscore[:50] = 0.001  # force overlap violations
    X = np.column_stack([np.ones(n), rng.normal(size=n)])  # col 0 constant
    res = _make_result(
        y_resid, d_resid, pscore=pscore, X=X, cov_names=["const_col", "x1"]
    )
    diag = dml_diagnostics(res, clip=0.02)
    assert diag.overlap_warning is not None
    s = diag.summary()
    assert "⚠" in s  # overlap warning rendered (line 90)
    # constant covariate row → zeros (lines 291-293)
    const_row = diag.balance_table[diag.balance_table["variable"] == "const_col"]
    assert float(const_row["corr_d_resid"].iloc[0]) == 0.0


def test_diagnostics_constant_score_path():
    # y_resid constant → score_sd == 0 → orth_stat 0.0 branch (line 272),
    # skew/kurtosis 0.0 branches.
    n = 100
    y_resid = np.full(n, 3.0)
    d_resid = np.linspace(-1, 1, n)
    res = _make_result(y_resid, d_resid)
    diag = dml_diagnostics(res)
    assert diag.score_sd == 0.0
    assert diag.orth_stat == 0.0
    assert diag.score_skew == 0.0


def test_sensitivity_constant_d_resid_raises():
    n = 100
    y_resid = np.linspace(-1, 1, n)
    d_resid = np.full(n, 2.0)  # zero variance
    res = _make_result(y_resid, d_resid)
    with pytest.raises(ValueError, match="D residual variance is 0"):
        dml_sensitivity(res)
