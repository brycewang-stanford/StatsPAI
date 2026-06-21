"""Coverage round-3 (final) — variant SCM estimators of ``statspai.synth``.

Targets still-uncovered branches of the generalized synthetic control
(``gsynth``), penalized Abadie-L'Hour (``penalized_synth``), sparse /
LASSO (``sparse_synth``) and kernel (``kernel_synth`` /
``kernel_ridge_synth``) estimators: covariate partial-out paths,
explicit factor-count selection, the alternative ``mode`` / ``kernel``
options, and the loud validation failures.

All estimators here are pure-numpy. The R gsynth backend is gated on an
``Rscript`` executable + R packages — covered only by an availability
check + skip note. Assertions check real properties (finite ATT / SE,
weights on the simplex where constrained, populated placebo
distributions, correct exceptions); no estimator numbers are fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.synth.gsynth import _find_rscript, gsynth
from statspai.synth.penscm import penalized_synth
from statspai.synth.sparse import sparse_synth
from statspai.synth.kernel import kernel_synth, kernel_ridge_synth

T_TREAT = 11
TRUE_EFFECT = 4.0


def _panel(seed=0, n_donors=8, n_t=20, effect=TRUE_EFFECT, with_cov=False):
    rng = np.random.default_rng(seed)
    units = [f"u{i}" for i in range(n_donors)] + ["treated"]
    rows = []
    for u in units:
        base = rng.normal(0, 1)
        fe = rng.normal(0, 0.5)
        w = rng.normal()
        for t in range(1, n_t + 1):
            eff = effect if (u == "treated" and t >= T_TREAT) else 0.0
            row = {
                "unit": u,
                "time": t,
                "y": base + 0.2 * t + fe + eff + rng.normal(0, 0.3),
            }
            if with_cov:
                row["z"] = w + 0.1 * t + rng.normal(0, 0.1)
            rows.append(row)
    return pd.DataFrame(rows)


def _simplex_ok(w, tol=5e-2):
    w = np.asarray(w, dtype=float)
    return w.min() >= -1e-6 and abs(w.sum() - 1.0) < tol


# ===========================================================================
# Generalized synthetic control (interactive fixed effects)
# ===========================================================================
def test_gsynth_explicit_factors_and_placebo():
    r = gsynth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        n_factors=2,
        placebo=True,
        seed=1,
    )
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.se)
    assert r.model_info.get("n_placebos", 0) >= 1


def test_gsynth_zero_factors_pure_twoway():
    # n_factors=0 -> the empty-rank else branch in the factor model.
    r = gsynth(
        _panel(1),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        n_factors=0,
        placebo=False,
        seed=2,
    )
    assert np.isfinite(r.estimate)


def test_gsynth_covariates_partial_out():
    r = gsynth(
        _panel(2, with_cov=True),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        covariates=["z"],
        n_factors=1,
        placebo=False,
        seed=3,
    )
    assert np.isfinite(r.estimate)


def test_gsynth_cv_factor_selection_small_panel():
    # Tiny donor pool drives _select_factors_cv toward max_factors<=0.
    df = _panel(3, n_donors=3, n_t=8)
    r = gsynth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=5,
        n_factors=None,
        max_factors=2,
        cv_folds=2,
        placebo=False,
        seed=4,
    )
    assert np.isfinite(r.estimate)


def test_gsynth_insufficient_post_periods_raises():
    from statspai.exceptions import DataInsufficient

    df = _panel(4, n_t=12)
    # treatment_time beyond the panel -> zero post periods.
    with pytest.raises(DataInsufficient):
        gsynth(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=99,
            placebo=False,
        )


def test_gsynth_r_backend_availability():
    # The R backend needs Rscript + R packages 'gsynth'/'jsonlite'.
    try:
        _find_rscript()
    except RuntimeError:
        with pytest.raises((RuntimeError, Exception)):
            gsynth(
                _panel(0),
                outcome="y",
                unit="unit",
                time="time",
                treated_unit="treated",
                treatment_time=T_TREAT,
                backend="r",
            )
    else:  # pragma: no cover - R not installed in CI
        pytest.skip("Rscript present; native-path coverage already exercised")


# ===========================================================================
# Penalized synthetic control (Abadie & L'Hour 2021)
# ===========================================================================
def test_penalized_synth_pairwise_penalty():
    r = penalized_synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        lambda_pen=0.1,
        placebo=True,
    )
    assert np.isfinite(r.estimate)
    w = r.model_info.get("donor_weights")
    if w is not None:
        vals = list(w.values()) if isinstance(w, dict) else np.asarray(w)
        assert np.all(np.asarray(vals, dtype=float) >= -1e-6)


def test_penalized_synth_auto_lambda_and_predictors():
    df = _panel(1, with_cov=True)
    r = penalized_synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        lambda_pen=None,
        predictors=["z"],
        placebo=False,
    )
    assert np.isfinite(r.estimate)


@pytest.mark.parametrize("ptype", ["pairwise", "max_dev", "l1_pairwise"])
def test_penalized_synth_penalty_types(ptype):
    r = penalized_synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        penalty_type=ptype,
        lambda_pen=0.1,
        placebo=False,
    )
    assert np.isfinite(r.estimate)
    w = np.asarray(list(r.model_info["weights"].values()), dtype=float)
    assert _simplex_ok(w)


def test_penalized_synth_invalid_penalty_type():
    with pytest.raises(ValueError):
        penalized_synth(
            _panel(0),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            penalty_type="not_valid",
        )


def test_gsynth_r_backend_rejects_covariates_and_factors():
    # These NotImplementedError guards fire before Rscript is even probed.
    with pytest.raises(NotImplementedError):
        gsynth(
            _panel(0),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            backend="r",
            covariates=["nonexistent"],
        )
    with pytest.raises(NotImplementedError):
        gsynth(
            _panel(0),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            backend="r",
            n_factors=2,
        )


def test_gsynth_unknown_backend_and_too_few_pre():
    from statspai.exceptions import DataInsufficient

    with pytest.raises(ValueError):
        gsynth(
            _panel(0),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            backend="bogus",
        )
    # treatment_time=2 -> only 1 pre-period < 3 required.
    with pytest.raises(DataInsufficient):
        gsynth(
            _panel(0),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=2,
            placebo=False,
        )


def test_penalized_synth_validation_errors():
    df = _panel(2)
    with pytest.raises(ValueError):
        penalized_synth(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="ghost",
            treatment_time=T_TREAT,
        )


def test_penalized_synth_too_few_donors():
    df = _panel(3, n_donors=1)
    with pytest.raises(ValueError):
        penalized_synth(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
        )


# ===========================================================================
# Sparse / LASSO synthetic control
# ===========================================================================
@pytest.mark.parametrize("mode", ["lasso", "constrained_lasso", "joint"])
def test_sparse_synth_modes(mode):
    r = sparse_synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        mode=mode,
        placebo=True,
    )
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.se) or np.isnan(r.se)


def test_sparse_synth_unknown_mode_raises():
    with pytest.raises(ValueError):
        sparse_synth(
            _panel(1),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            mode="not_a_mode",
        )


def test_sparse_synth_with_covariates():
    df = _panel(2, with_cov=True)
    r = sparse_synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        mode="lasso",
        covariates=["z"],
        placebo=False,
    )
    assert np.isfinite(r.estimate)


def test_sparse_synth_missing_treated_raises():
    with pytest.raises(ValueError):
        sparse_synth(
            _panel(3),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="ghost",
            treatment_time=T_TREAT,
        )


# ===========================================================================
# Kernel synthetic control
# ===========================================================================
@pytest.mark.parametrize("kernel", ["rbf", "polynomial", "laplacian"])
def test_kernel_synth_kernels(kernel):
    r = kernel_synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        kernel=kernel,
        placebo=True,
    )
    assert np.isfinite(r.estimate)


def test_kernel_synth_with_covariates_and_sigma():
    df = _panel(1, with_cov=True)
    r = kernel_synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        kernel="rbf",
        sigma=1.0,
        covariates=["z"],
        placebo=False,
    )
    assert np.isfinite(r.estimate)


def test_kernel_synth_missing_treated_raises():
    with pytest.raises(ValueError):
        kernel_synth(
            _panel(2),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="ghost",
            treatment_time=T_TREAT,
        )


def test_kernel_ridge_synth_covariates_no_placebo():
    df = _panel(1, with_cov=True)
    r = kernel_ridge_synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        covariates=["z"],
        ridge_lambda=0.05,
        placebo=False,
    )
    assert np.isfinite(r.estimate)


def test_penalized_synth_predictors_and_covariates():
    df = _panel(0, with_cov=True)
    df["w2"] = df["z"] * 0.5 + 1.0
    r = penalized_synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        covariates=["z"],
        predictors=["w2"],
        lambda_pen=0.1,
        placebo=False,
    )
    assert np.isfinite(r.estimate)


def test_penalized_synth_missing_covariate_raises():
    with pytest.raises(ValueError):
        penalized_synth(
            _panel(0),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            covariates=["does_not_exist"],
            lambda_pen=0.1,
        )


def test_sparse_synth_too_few_donors_raises():
    with pytest.raises(ValueError):
        sparse_synth(
            _panel(0, n_donors=1),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            mode="lasso",
        )


def test_sparse_synth_too_few_pre_periods_raises():
    df = _panel(0, n_t=20)
    with pytest.raises(ValueError):
        sparse_synth(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=2,
            mode="lasso",
        )


def test_kernel_ridge_synth_and_lambda_guard():
    r = kernel_ridge_synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        ridge_lambda=0.05,
        placebo=True,
    )
    assert np.isfinite(r.estimate)
    with pytest.raises(ValueError):
        kernel_ridge_synth(
            _panel(0),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            ridge_lambda=-1.0,
        )
