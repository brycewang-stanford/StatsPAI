"""Coverage tests for statspai.dml.model_averaging, panel_dml, and the
PLIV / IIVM weighted branches.

Targets: every weight_rule path (short_stacking / single_best /
inverse_risk / equal), weighted CLS + weighted per-candidate variance,
input validation, the panel within-transform (one-way and two-way, plus
weighted), cluster-robust SE, no-covariate panel path, and the
deprecated binary_treatment flag. Real synthetic data only.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.dml import dml, dml_model_averaging, dml_panel
from statspai.dml.model_averaging import _solve_cls_weights
from statspai.exceptions import DataInsufficient, MethodIncompatibility


@pytest.fixture
def avg_df():
    rng = np.random.default_rng(3)
    n = 400
    X = rng.normal(size=(n, 4))
    d = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=n)
    y = 1.0 * d + X[:, 0] + X[:, 2] + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(4)])
    df["y"] = y
    df["d"] = d
    return df


def _small_candidates():
    from sklearn.linear_model import LinearRegression, RidgeCV

    return [
        (LinearRegression(), LinearRegression(), "ols"),
        (RidgeCV(), RidgeCV(), "ridge"),
    ]


@pytest.mark.parametrize(
    "rule", ["short_stacking", "single_best", "inverse_risk", "equal"]
)
def test_averaging_all_weight_rules(avg_df, rule):
    res = dml_model_averaging(
        avg_df,
        y="y",
        treat="d",
        covariates=[f"x{j}" for j in range(4)],
        candidates=_small_candidates(),
        weight_rule=rule,
        n_folds=3,
    )
    assert np.isfinite(res.estimate)
    assert res.se >= 0
    assert res.model_info["weight_rule"] == rule
    assert set(res.model_info["weights"].keys()) == {"ols", "ridge"}


def test_averaging_short_stacking_reports_g_m_weights(avg_df):
    res = dml_model_averaging(
        avg_df,
        y="y",
        treat="d",
        covariates=[f"x{j}" for j in range(4)],
        candidates=_small_candidates(),
        weight_rule="short_stacking",
        n_folds=3,
    )
    assert "weights_g" in res.model_info
    assert "weights_m" in res.model_info


def test_averaging_default_candidates(avg_df):
    res = dml_model_averaging(
        avg_df,
        y="y",
        treat="d",
        covariates=[f"x{j}" for j in range(4)],
        n_folds=3,
        weight_rule="single_best",
    )
    # Default roster has lasso/ridge/rf/gbm.
    assert len(res.model_info["candidates"]) >= 1


def test_averaging_accepts_scalar_covariate(avg_df):
    res = dml_model_averaging(
        avg_df,
        y="y",
        treat="d",
        covariates="x0",
        candidates=_small_candidates(),
        n_folds=3,
        weight_rule="equal",
    )
    assert np.isfinite(res.estimate)


def test_averaging_missing_column(avg_df):
    with pytest.raises(ValueError, match="not found"):
        dml_model_averaging(avg_df, y="nope", treat="d", covariates=["x0"])


def test_averaging_bad_weight_rule(avg_df):
    with pytest.raises(ValueError, match="weight_rule must be"):
        dml_model_averaging(
            avg_df, y="y", treat="d", covariates=["x0"], weight_rule="bogus"
        )


def test_averaging_no_covariates(avg_df):
    with pytest.raises(ValueError, match="At least one covariate"):
        dml_model_averaging(avg_df, y="y", treat="d", covariates=[])


def test_averaging_empty_candidates(avg_df):
    with pytest.raises(ValueError, match="No candidate"):
        dml_model_averaging(avg_df, y="y", treat="d", covariates=["x0"], candidates=[])


def test_averaging_rejects_invalid_controls(avg_df):
    with pytest.raises(MethodIncompatibility, match="n_folds"):
        dml_model_averaging(
            avg_df,
            y="y",
            treat="d",
            covariates=["x0"],
            candidates=_small_candidates(),
            n_folds=1,
        )
    with pytest.raises(MethodIncompatibility, match="alpha"):
        dml_model_averaging(
            avg_df,
            y="y",
            treat="d",
            covariates=["x0"],
            candidates=_small_candidates(),
            alpha=1.0,
        )
    with pytest.raises(DataInsufficient, match="n_folds"):
        dml_model_averaging(
            avg_df.head(2),
            y="y",
            treat="d",
            covariates=["x0"],
            candidates=_small_candidates(),
            n_folds=3,
        )


def test_averaging_rejects_bad_candidates(avg_df):
    with pytest.raises(MethodIncompatibility, match="triple"):
        dml_model_averaging(
            avg_df,
            y="y",
            treat="d",
            covariates=["x0"],
            candidates=[("bad",)],
            n_folds=3,
        )
    cand = _small_candidates()
    cand[1] = (cand[1][0], cand[1][1], "ols")
    with pytest.raises(MethodIncompatibility, match="unique"):
        dml_model_averaging(
            avg_df,
            y="y",
            treat="d",
            covariates=["x0"],
            candidates=cand,
            n_folds=3,
        )


def test_averaging_all_nan_rows_raises():
    df = pd.DataFrame(
        {"y": [np.nan, np.nan], "d": [np.nan, np.nan], "x0": [np.nan, np.nan]}
    )
    with pytest.raises(ValueError, match="No rows remain"):
        dml_model_averaging(
            df,
            y="y",
            treat="d",
            covariates=["x0"],
            candidates=_small_candidates(),
            n_folds=2,
        )


def test_averaging_nonfinite_design_raises(avg_df):
    df = avg_df.copy()
    df.loc[0, "x0"] = np.inf
    with pytest.raises(MethodIncompatibility, match="finite"):
        dml_model_averaging(
            df,
            y="y",
            treat="d",
            covariates=["x0"],
            candidates=_small_candidates(),
            n_folds=3,
        )


def test_averaging_drops_missing(avg_df):
    df = avg_df.copy()
    df.loc[0, "y"] = np.nan
    res = dml_model_averaging(
        df,
        y="y",
        treat="d",
        covariates=[f"x{j}" for j in range(4)],
        candidates=_small_candidates(),
        n_folds=3,
        weight_rule="equal",
    )
    assert res.model_info["n_dropped_missing"] == 1


# --- weighted averaging paths ---
def test_averaging_weighted_short_stacking(avg_df):
    w = np.abs(np.random.default_rng(5).normal(size=len(avg_df))) + 0.1
    res = dml_model_averaging(
        avg_df,
        y="y",
        treat="d",
        covariates=[f"x{j}" for j in range(4)],
        candidates=_small_candidates(),
        weight_rule="short_stacking",
        n_folds=3,
        sample_weight=w,
    )
    assert np.isfinite(res.estimate)


def test_averaging_weighted_equal(avg_df):
    w = np.abs(np.random.default_rng(6).normal(size=len(avg_df))) + 0.1
    res = dml_model_averaging(
        avg_df,
        y="y",
        treat="d",
        covariates=[f"x{j}" for j in range(4)],
        candidates=_small_candidates(),
        weight_rule="equal",
        n_folds=3,
        sample_weight=w,
    )
    assert np.isfinite(res.estimate)


def test_averaging_weighted_column_name(avg_df):
    df = avg_df.copy()
    df["w"] = np.abs(np.random.default_rng(7).normal(size=len(df))) + 0.1
    res = dml_model_averaging(
        df,
        y="y",
        treat="d",
        covariates=[f"x{j}" for j in range(4)],
        candidates=_small_candidates(),
        n_folds=3,
        sample_weight="w",
        weight_rule="inverse_risk",
    )
    assert np.isfinite(res.estimate)


def test_averaging_weight_validations(avg_df):
    with pytest.raises(ValueError, match="non-negative"):
        dml_model_averaging(
            avg_df,
            y="y",
            treat="d",
            covariates=["x0"],
            candidates=_small_candidates(),
            n_folds=3,
            sample_weight=-np.ones(len(avg_df)),
        )
    with pytest.raises(ValueError, match="1-D of length"):
        dml_model_averaging(
            avg_df,
            y="y",
            treat="d",
            covariates=["x0"],
            candidates=_small_candidates(),
            n_folds=3,
            sample_weight=np.ones(3),
        )
    with pytest.raises(ValueError, match="sample_weight column"):
        dml_model_averaging(
            avg_df,
            y="y",
            treat="d",
            covariates=["x0"],
            candidates=_small_candidates(),
            n_folds=3,
            sample_weight="nope",
        )


from sklearn.base import BaseEstimator, RegressorMixin


class _NoWeightReg(BaseEstimator, RegressorMixin):
    """Minimal sklearn regressor whose .fit ignores sample_weight."""

    def fit(self, X, y):  # no sample_weight kwarg → TypeError on weighted call
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_averaging_weighted_fit_fallback_warns(avg_df):
    # A learner that does not accept sample_weight triggers the
    # unweighted-fallback warning inside _fit_candidate_plr (lines 149-157).
    w = np.abs(np.random.default_rng(8).normal(size=len(avg_df))) + 0.1
    cand = [(_NoWeightReg(), _NoWeightReg(), "noweight")]
    with pytest.warns(RuntimeWarning, match="does not accept"):
        res = dml_model_averaging(
            avg_df,
            y="y",
            treat="d",
            covariates=[f"x{j}" for j in range(4)],
            candidates=cand,
            weight_rule="equal",
            n_folds=3,
            sample_weight=w,
        )
    assert np.isfinite(res.estimate)


def test_solve_cls_weights_basic():
    rng = np.random.default_rng(0)
    n = 100
    p1 = rng.normal(size=n)
    p2 = rng.normal(size=n)
    target = 0.7 * p1 + 0.3 * p2
    w = _solve_cls_weights(target, np.column_stack([p1, p2]))
    assert w.shape == (2,)
    assert w.min() >= 0
    assert abs(w.sum() - 1.0) < 1e-6


def test_solve_cls_weights_validates_inputs():
    with pytest.raises(MethodIncompatibility, match="predictions"):
        _solve_cls_weights(np.arange(3.0), np.arange(3.0))
    with pytest.raises(MethodIncompatibility, match="sample_weight"):
        _solve_cls_weights(
            np.arange(3.0),
            np.ones((3, 2)),
            sample_weight=np.zeros(3),
        )


# --------------------------------------------------------------------------
# panel_dml
# --------------------------------------------------------------------------
@pytest.fixture
def panel_df():
    rng = np.random.default_rng(9)
    n_units, n_time = 60, 6
    rows = []
    for i in range(n_units):
        alpha = rng.normal()
        for t in range(n_time):
            x1 = rng.normal()
            x2 = rng.normal()
            d = 0.5 * x1 + alpha + rng.normal(scale=0.5)
            y = alpha + 1.0 * d + x1 + 0.5 * x2 + rng.normal(scale=0.5)
            rows.append({"pid": i, "year": t, "y": y, "d": d, "x1": x1, "x2": x2})
    return pd.DataFrame(rows)


def test_panel_basic(panel_df):
    res = dml_panel(panel_df, y="y", treat="d", covariates="x1", unit="pid", n_folds=4)
    assert np.isfinite(res.estimate)
    assert res.n_units == 60
    s = res.summary()
    assert "Long-panel" in s
    assert "β" in s or "causal" in s


def test_panel_two_way_fe(panel_df):
    res = dml_panel(
        panel_df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        unit="pid",
        time="year",
        include_time_fe=True,
        n_folds=4,
    )
    assert res.include_time_fe is True
    assert np.isfinite(res.estimate)


def test_panel_no_covariates(panel_df):
    res = dml_panel(panel_df, y="y", treat="d", covariates=[], unit="pid", n_folds=4)
    assert np.isfinite(res.estimate)


def test_panel_diagnostics(panel_df):
    res = dml_panel(
        panel_df, y="y", treat="d", covariates=["x1", "x2"], unit="pid", n_folds=4
    )
    d = res.diagnostics
    assert "y_resid_std" in d
    assert "within_r2_outcome" in d
    assert "omega_cluster" in d
    assert d["weighted"] is False


def test_panel_weighted(panel_df):
    w = np.abs(np.random.default_rng(13).normal(size=len(panel_df))) + 0.1
    res = dml_panel(
        panel_df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        unit="pid",
        n_folds=4,
        sample_weight=w,
    )
    assert res.diagnostics["weighted"] is True
    assert np.isfinite(res.estimate)


def test_panel_weighted_two_way(panel_df):
    w = np.abs(np.random.default_rng(14).normal(size=len(panel_df))) + 0.1
    res = dml_panel(
        panel_df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        unit="pid",
        time="year",
        include_time_fe=True,
        n_folds=4,
        sample_weight=w,
    )
    assert np.isfinite(res.estimate)


def test_panel_weighted_column_name(panel_df):
    df = panel_df.copy()
    df["w"] = np.abs(np.random.default_rng(15).normal(size=len(df))) + 0.1
    res = dml_panel(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        unit="pid",
        n_folds=4,
        sample_weight="w",
    )
    assert np.isfinite(res.estimate)


def test_panel_weighted_fit_fallback_warns(panel_df):
    # ml_g/ml_m without sample_weight support → RuntimeWarning fallback
    # inside _maybe_weighted_fit (panel_dml.py lines 472-481).
    w = np.abs(np.random.default_rng(16).normal(size=len(panel_df))) + 0.1
    with pytest.warns(RuntimeWarning, match="does not accept"):
        res = dml_panel(
            panel_df,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            unit="pid",
            n_folds=4,
            sample_weight=w,
            ml_g=_NoWeightReg(),
            ml_m=_NoWeightReg(),
        )
    assert np.isfinite(res.estimate)


def test_panel_n_folds_below_two(panel_df):
    with pytest.raises(MethodIncompatibility, match="n_folds must be"):
        dml_panel(panel_df, y="y", treat="d", covariates=["x1"], unit="pid", n_folds=1)


def test_panel_time_fe_without_time(panel_df):
    with pytest.raises(MethodIncompatibility, match="time must be provided"):
        dml_panel(
            panel_df,
            y="y",
            treat="d",
            covariates=["x1"],
            unit="pid",
            include_time_fe=True,
        )


def test_panel_missing_column(panel_df):
    with pytest.raises(MethodIncompatibility, match="missing columns"):
        dml_panel(panel_df, y="nope", treat="d", covariates=["x1"], unit="pid")


def test_panel_folds_exceed_units(panel_df):
    small = panel_df[panel_df["pid"] < 3]
    with pytest.raises(DataInsufficient, match="cannot exceed n_units"):
        dml_panel(small, y="y", treat="d", covariates=["x1"], unit="pid", n_folds=5)


def test_panel_weight_validations(panel_df):
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        dml_panel(
            panel_df,
            y="y",
            treat="d",
            covariates=["x1"],
            unit="pid",
            n_folds=4,
            sample_weight=-np.ones(len(panel_df)),
        )
    with pytest.raises(MethodIncompatibility, match="1-D of length"):
        dml_panel(
            panel_df,
            y="y",
            treat="d",
            covariates=["x1"],
            unit="pid",
            n_folds=4,
            sample_weight=np.ones(3),
        )
    with pytest.raises(MethodIncompatibility, match="sample_weight column"):
        dml_panel(
            panel_df,
            y="y",
            treat="d",
            covariates=["x1"],
            unit="pid",
            n_folds=4,
            sample_weight="nope",
        )


def test_panel_binary_treatment_deprecation():
    rng = np.random.default_rng(17)
    n_units, n_time = 40, 5
    rows = []
    for i in range(n_units):
        alpha = rng.normal()
        for t in range(n_time):
            x = rng.normal()
            d = float(rng.binomial(1, 0.5))
            y = alpha + d + x + rng.normal(scale=0.5)
            rows.append({"pid": i, "year": t, "y": y, "d": d, "x": x})
    df = pd.DataFrame(rows)
    with pytest.warns(DeprecationWarning, match="binary_treatment"):
        res = dml_panel(
            df,
            y="y",
            treat="d",
            covariates=["x"],
            unit="pid",
            n_folds=4,
            binary_treatment=True,
        )
    assert np.isfinite(res.estimate)


def test_panel_binary_treatment_nonbinary_raises(panel_df):
    with pytest.raises(MethodIncompatibility, match=r"requires D"):
        dml_panel(
            panel_df,
            y="y",
            treat="d",
            covariates=["x1"],
            unit="pid",
            n_folds=4,
            binary_treatment=True,
        )


# --------------------------------------------------------------------------
# PLIV / IIVM weighted scale-invariance & error branches
# --------------------------------------------------------------------------
@pytest.fixture
def pliv_df():
    rng = np.random.default_rng(19)
    n = 800
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    d = 0.8 * z + x + rng.normal(scale=0.5, size=n)
    y = 1.0 * d + x + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "d": d, "x": x, "z": z})


def test_pliv_diagnostics(pliv_df):
    res = dml(pliv_df, y="y", treat="d", covariates=["x"], model="pliv", instrument="z")
    diags = res.model_info["diagnostics"]
    assert "first_stage_partial_corr" in diags
    assert "first_stage_F_approx" in diags
    assert res.model_info["instrument"] == "z"
    assert res.model_info["ml_r"] is not None


def test_pliv_degenerate_instrument_collinear_with_x_raises():
    # Z is a deterministic function of X → ml_r absorbs all of its
    # variance → Var(z_resid)/Var(Z) collapses (case (i) guard).
    rng = np.random.default_rng(23)
    n = 600
    x = rng.normal(size=n)
    z = 2.0 * x + 1.0  # exactly collinear with the covariate
    d = x + rng.normal(scale=0.5, size=n)
    y = d + x + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x, "z": z})
    # Use linear ml_r so the residualisation absorbs the (linear) Z-on-X
    # relationship exactly, collapsing Var(z_resid)/Var(Z) → 0.
    with pytest.raises(RuntimeError, match="[Ww]eak|degenerate|collinear|orthogonal"):
        dml(
            df,
            y="y",
            treat="d",
            covariates=["x"],
            model="pliv",
            instrument="z",
            ml_g="linear",
            ml_m="linear",
            ml_r="linear",
        )


def test_pliv_weighted(pliv_df):
    # Weighted PLIV exercises the w_full normalisation (lines 50-55) and
    # the weighted variance / partial-corr branches (85-86, 106-107,
    # 142-143).
    w = np.abs(np.random.default_rng(29).normal(size=len(pliv_df))) + 0.1
    res = dml(
        pliv_df,
        y="y",
        treat="d",
        covariates=["x"],
        model="pliv",
        instrument="z",
        sample_weight=w,
    )
    assert np.isfinite(res.estimate)
    assert res.model_info["diagnostics"]["weighted"] is True


def test_pliv_weight_scale_invariant(pliv_df):
    w = np.abs(np.random.default_rng(31).normal(size=len(pliv_df))) + 0.1
    r1 = dml(
        pliv_df,
        y="y",
        treat="d",
        covariates=["x"],
        model="pliv",
        instrument="z",
        sample_weight=w,
    )
    r2 = dml(
        pliv_df,
        y="y",
        treat="d",
        covariates=["x"],
        model="pliv",
        instrument="z",
        sample_weight=5.0 * w,
    )
    assert r1.estimate == pytest.approx(r2.estimate, rel=1e-6)


def test_iivm_continuous_z_rejected():
    rng = np.random.default_rng(27)
    n = 400
    x = rng.normal(size=n)
    z = rng.normal(size=n)  # continuous
    d = rng.binomial(1, 0.5, n).astype(float)
    y = d + x + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x, "z": z})
    with pytest.raises(ValueError, match="binary"):
        dml(df, y="y", treat="d", covariates=["x"], model="iivm", instrument="z")
