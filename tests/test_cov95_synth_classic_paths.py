"""Coverage tests for classic SCM paths: covariates, special predictors,
V-weight methods, conformal inference override, and error branches.

Targets statspai.synth.scm and statspai.synth._core uncovered branches.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

scm = importlib.import_module("statspai.synth.scm")


def _panel(
    n_units=8, n_periods=16, treatment_time=11, effect=4.0, seed=51, with_cov=True
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
                row["x1"] = alphas[i] + 0.05 * t + rng.normal(0, 0.1)
                row["x2"] = betas[i] * t + rng.normal(0, 0.1)
            records.append(row)
    return pd.DataFrame(records)


COMMON = dict(
    outcome="outcome", unit="unit", time="time", treated_unit="u0", treatment_time=11
)

# Planted truth in _panel(): treated unit u0 gets +4.0 for t >= 11. With 8 units
# / 16 periods / treatment_time=11 the design fixes the panel geometry. SCM
# recovers the post-period ATT; across the predictor/V-weight variants exercised
# here the point estimate lands in ~[2.2, 4.05], so a generous band catches gross
# regressions (sign flips, off-by-a-factor) without being seed-fragile.
TRUE_EFFECT = 4.0


def _assert_simplex_weights(res):
    """Donor weights form a simplex: non-negative and summing to 1.

    The returned table only lists donors with retained (non-pruned) weight, so
    the row count is a subset of the 7-donor pool (some solvers drop exact-zero
    donors), hence 1..7 rather than ==7.
    """
    mi = res.model_info or {}
    w = mi.get("weights")
    assert w is not None
    vals = np.asarray(w["weight"].to_numpy(), dtype=float)
    assert 1 <= vals.shape[0] <= 7  # subset of the 7-donor pool
    assert np.all(vals >= -1e-8)  # non-negative (allow solver fuzz)
    assert vals.sum() == pytest.approx(1.0, abs=1e-6)  # convex combination


def _assert_recovers_effect(res, lo=0.5, hi=8.0):
    """Point estimate is finite, correctly signed (+), and in a band around 4.0."""
    est = res.estimate
    assert np.isfinite(est)
    assert est > 0  # planted effect is positive
    assert lo < est < hi  # generous recovery band around TRUE_EFFECT=4.0
    mi = res.model_info or {}
    # Geometry is fixed by the DGP regardless of estimator variant.
    assert mi.get("n_pre_periods") == 10
    assert mi.get("n_post_periods") == 6
    assert mi.get("n_donors") == 7
    # A good SCM pre-fit: small, finite, non-negative pre-treatment RMSE.
    rmse = mi.get("pre_treatment_rmse")
    assert rmse is not None and np.isfinite(rmse)
    assert 0.0 <= rmse < 5.0


@pytest.fixture
def panel():
    return _panel()


def test_classic_with_covariates_nested_v(panel):
    """Covariates trigger nested V optimization (_core.solve_synth_weights_adh)."""
    res = sp.synth(
        panel, **COMMON, method="classic", covariates=["x1", "x2"], placebo=False
    )
    _assert_recovers_effect(res)
    _assert_simplex_weights(res)
    mi = res.model_info or {}
    # V weights should be populated when predictors come from covariates: one
    # non-negative weight per predictor used (x1, x2 -> 2 rows).
    vw = mi.get("v_weights")
    assert vw is not None
    vvals = np.asarray(vw["v_weight"].to_numpy(), dtype=float)
    assert vvals.shape[0] == 2
    assert np.all(vvals >= -1e-8)
    assert vvals.sum() > 0


def test_classic_v_method_nested(panel):
    res = sp.synth(
        panel,
        **COMMON,
        method="classic",
        covariates=["x1"],
        v_method="nested",
        n_random_starts=2,
        placebo=False,
    )
    _assert_recovers_effect(res)
    _assert_simplex_weights(res)
    assert res.model_info.get("v_method") == "nested"


def test_classic_v_method_equal(panel):
    res = sp.synth(
        panel,
        **COMMON,
        method="classic",
        covariates=["x1", "x2"],
        v_method="equal",
        placebo=False,
    )
    _assert_recovers_effect(res)
    _assert_simplex_weights(res)
    mi = res.model_info or {}
    assert mi.get("v_method") == "equal"
    # 'equal' weighting => all predictor V-weights identical.
    vvals = np.asarray(mi["v_weights"]["v_weight"].to_numpy(), dtype=float)
    assert np.allclose(vvals, vvals[0])


def test_classic_no_standardize(panel):
    res = sp.synth(
        panel,
        **COMMON,
        method="classic",
        covariates=["x1"],
        standardize_predictors=False,
        placebo=False,
    )
    # Without standardization the single-covariate fit is noisier; keep the band
    # generous but still pin sign/geometry/simplex.
    _assert_recovers_effect(res)
    _assert_simplex_weights(res)


def test_classic_special_predictors_mean(panel):
    sp_spec = [("x1", [1, 2, 3], "mean")]
    res = sp.synth(
        panel, **COMMON, method="classic", special_predictors=sp_spec, placebo=False
    )
    _assert_recovers_effect(res)
    _assert_simplex_weights(res)


def test_classic_special_predictors_sum_and_slice(panel):
    sp_spec = [("x2", slice(1, 5), "sum")]
    res = sp.synth(
        panel, **COMMON, method="classic", special_predictors=sp_spec, placebo=False
    )
    _assert_recovers_effect(res)
    _assert_simplex_weights(res)


def test_classic_special_predictor_bad_op(panel):
    sp_spec = [("x1", [1, 2], "median")]
    with pytest.raises(ValueError):
        sp.synth(
            panel, **COMMON, method="classic", special_predictors=sp_spec, placebo=False
        )


def test_classic_special_predictor_missing_col(panel):
    sp_spec = [("nope", [1, 2], "mean")]
    with pytest.raises(ValueError):
        sp.synth(
            panel, **COMMON, method="classic", special_predictors=sp_spec, placebo=False
        )


def test_ridge_default_l2(panel):
    res = sp.synth(panel, **COMMON, method="ridge", placebo=False)
    _assert_recovers_effect(res)
    mi = res.model_info or {}
    # Ridge relaxes the simplex (allows L2-penalized weights) but still records a
    # positive penalization level.
    pen = mi.get("penalization")
    assert pen is not None and np.isfinite(pen) and pen > 0
    # SE finite and positive but not absurd.
    assert np.isfinite(res.se) and 0 < res.se < 100


def test_inference_conformal_override(panel):
    res = sp.synth(
        panel, **COMMON, method="classic", inference="conformal", grid_size=15
    )
    _assert_recovers_effect(res)
    # Conformal inference must produce a proper interval bracketing the estimate.
    lo, hi = res.ci
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
    assert lo <= res.estimate <= hi
    # A conformal p-value is a valid probability.
    assert 0.0 <= res.pvalue <= 1.0


def test_unknown_backend_raises(panel):
    with pytest.raises(ValueError):
        sp.synth(panel, **COMMON, method="classic", backend="matlab")


def test_r_backend_penalized_not_implemented(panel):
    with pytest.raises(NotImplementedError):
        sp.synth(panel, **COMMON, method="ridge", backend="r")


def test_r_backend_covariates_not_implemented(panel):
    with pytest.raises(NotImplementedError):
        sp.synth(panel, **COMMON, method="classic", covariates=["x1"], backend="synth")


def test_missing_column_raises():
    df = _panel(with_cov=False)
    with pytest.raises(ValueError):
        sp.synth(
            df,
            outcome="no_such",
            unit="unit",
            time="time",
            treated_unit="u0",
            treatment_time=11,
            method="classic",
        )


def test_treated_unit_not_found_raises(panel):
    with pytest.raises(ValueError):
        sp.synth(panel, **{**COMMON, "treated_unit": "ghost"}, method="classic")


def test_too_few_pre_periods_raises():
    df = _panel(with_cov=False)
    with pytest.raises(ValueError):
        # treatment_time=2 leaves only 1 pre-period
        sp.synth(
            df,
            outcome="outcome",
            unit="unit",
            time="time",
            treated_unit="u0",
            treatment_time=2,
            method="classic",
        )


def test_no_post_periods_raises():
    df = _panel(with_cov=False, n_periods=16)
    with pytest.raises(ValueError):
        sp.synth(
            df,
            outcome="outcome",
            unit="unit",
            time="time",
            treated_unit="u0",
            treatment_time=100,
            method="classic",
        )


def test_classic_placebo_full(panel):
    res = sp.synth(panel, **COMMON, method="classic", covariates=["x1"], placebo=True)
    _assert_recovers_effect(res)
    _assert_simplex_weights(res)
    mi = res.model_info or {}
    # placebo should populate gap distributions / rmspe ratios
    assert any(
        k in mi
        for k in (
            "placebo_gaps",
            "placebo_effects",
            "rmspe_ratios",
            "placebo_units",
            "pvalue",
        )
    )
    # The placebo run is over the donor pool: one placebo per donor unit.
    pu = mi.get("placebo_units")
    if pu is not None:
        assert len(pu) == 7  # 7 donor units each get a placebo fit
        assert "u0" not in list(pu)  # treated unit is not a placebo
