"""External parity: ``sp.dml_sensitivity`` vs ``DoubleML.sensitivity_analysis``.

Pins StatsPAI's DML omitted-variable-bias analysis (Chernozhukov,
Cinelli, Newey, Sharma & Syrgkanis 2022, "Long Story Short") against the
reference implementation in *doubleml-for-py*.

Why the reference is Python, not R
----------------------------------
The rest of the DML parity work (Track A modules 08 and 71) is pinned
against ``DoubleML`` for R. Sensitivity analysis is **not available
there**: ``DoubleML`` 1.0.2's R6 classes expose only ``initialize``,
``print``, ``fit``, ``bootstrap``, ``split_samples``,
``set_sample_splitting``, ``tune``, ``summary``, ``confint``,
``learner_names``, ``params_names``, ``set_ml_nuisance_params``,
``p_adjust``, ``get_params`` and ``clone`` -- no sensitivity method on
either ``DoubleMLPLR`` or the base class. The feature exists only in the
Python package, so that is the reference used here.

Shared-fold design
------------------
Both engines are given the same explicit fold partition and the same
closed-form nuisance learner, so the two PLR fits are numerically
identical (verified below before any sensitivity quantity is compared).
Anything that then differs is the sensitivity computation itself.

What is pinned, and what is not
-------------------------------
* ``bias_bound`` and the adjusted ``theta`` bounds: exact (1e-12).
* ``RV`` (robustness value at ``q=1``): exact to 1e-6.
* ``RVa``: **not** exact, and deliberately not claimed to be. StatsPAI
  solves for the confounding strength at which ``|theta| - z*se`` is
  exhausted using the *unadjusted* standard error; ``DoubleML`` lets the
  standard error itself move with the confounding scenario. The observed
  gap is ~1.4e-3 on this fixture and is asserted with that band rather
  than hidden inside a loose global tolerance.

Skipped automatically when ``doubleml`` is not installed -- it is an
optional pin, not a runtime dependency of StatsPAI.

References
----------
[@chernozhukov2022long], [@bach2022doubleml]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

doubleml = pytest.importorskip("doubleml")
from sklearn.linear_model import LinearRegression  # noqa: E402

N = 1500
K = 5
N_FOLDS = 5
COVARIATES = [f"x{i + 1}" for i in range(K)]
CF_Y = 0.05
CF_D = 0.05


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """Confounded PLR DGP with a closed-form-recoverable nuisance."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(N, K))
    d = 0.6 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(size=N)
    y = 0.8 * d + X[:, 0] + 0.5 * X[:, 2] + rng.normal(size=N)
    df = pd.DataFrame(X, columns=COVARIATES)
    df["d"] = d
    df["y"] = y
    return df


@pytest.fixture(scope="module")
def folds() -> np.ndarray:
    return np.arange(N) % N_FOLDS


@pytest.fixture(scope="module")
def sp_fit(data, folds):
    return sp.dml(
        data=data,
        y="y",
        d="d",
        X=COVARIATES,
        model="plr",
        model_y=LinearRegression(),
        model_d=LinearRegression(),
        n_folds=N_FOLDS,
        fold_indices=folds,
    )


@pytest.fixture(scope="module")
def dml_fit(data, folds):
    dml_data = doubleml.DoubleMLData(data, y_col="y", d_cols="d", x_cols=COVARIATES)
    obj = doubleml.DoubleMLPLR(
        dml_data,
        ml_l=LinearRegression(),
        ml_m=LinearRegression(),
        n_folds=N_FOLDS,
        n_rep=1,
        draw_sample_splitting=False,
    )
    obj.set_sample_splitting(
        [
            [
                (np.flatnonzero(folds != f), np.flatnonzero(folds == f))
                for f in range(N_FOLDS)
            ]
        ]
    )
    obj.fit()
    obj.sensitivity_analysis(cf_y=CF_Y, cf_d=CF_D, rho=1.0, level=0.95)
    return obj


def test_underlying_plr_fits_are_identical(sp_fit, dml_fit):
    """Precondition: any sensitivity gap must be the sensitivity code.

    If the two PLR fits themselves disagreed, every assertion below
    would be comparing sensitivity analyses of different estimates.
    """
    assert float(sp_fit.estimate) == pytest.approx(float(dml_fit.coef[0]), rel=1e-12)
    assert float(sp_fit.se) == pytest.approx(float(dml_fit.se[0]), rel=1e-12)


def test_bias_bound_matches_doubleml(sp_fit, dml_fit):
    """The maximum bias under (cf_y, cf_d) is the headline quantity."""
    sens = sp.dml_sensitivity(sp_fit, q=1.0, cf_y=CF_Y, cf_d=CF_D)
    reference = float(sp_fit.estimate) - float(
        dml_fit.sensitivity_params["theta"]["lower"][0]
    )
    assert float(sens.bias_bound) == pytest.approx(reference, rel=1e-12)


def test_adjusted_theta_bounds_match_doubleml(sp_fit, dml_fit):
    sens = sp.dml_sensitivity(sp_fit, q=1.0, cf_y=CF_Y, cf_d=CF_D)
    params = dml_fit.sensitivity_params["theta"]
    assert float(sens.adjusted_estimate_low) == pytest.approx(
        float(params["lower"][0]), rel=1e-12
    )
    assert float(sens.adjusted_estimate_high) == pytest.approx(
        float(params["upper"][0]), rel=1e-12
    )


def test_robustness_value_matches_doubleml(sp_fit, dml_fit):
    """RV_1: confounding strength that would zero out the estimate."""
    sens = sp.dml_sensitivity(sp_fit, q=1.0, cf_y=CF_Y, cf_d=CF_D)
    assert float(sens.rv_q) == pytest.approx(
        float(dml_fit.sensitivity_params["rv"][0]), rel=1e-6
    )


def test_rva_convention_gap_is_bounded_and_documented(sp_fit, dml_fit):
    """RVa differs by construction; the gap is pinned, not papered over.

    StatsPAI exhausts ``|theta| - z*se`` using the standard error of the
    *unadjusted* fit. DoubleML lets the standard error move with the
    confounding scenario, so its RVa is slightly smaller. This test
    fixes the size of that difference so a future change to either
    convention shows up as a failure rather than as drift.
    """
    sens = sp.dml_sensitivity(sp_fit, q=1.0, cf_y=CF_Y, cf_d=CF_D)
    reference = float(dml_fit.sensitivity_params["rva"][0])
    rel_gap = abs(float(sens.rv_qa) - reference) / reference
    assert rel_gap < 5e-3, (
        f"RVa gap {rel_gap:.3g} exceeds the documented convention band; "
        f"sp={sens.rv_qa:.10f} vs doubleml={reference:.10f}"
    )
    # No directional claim: the sign of the gap depends on how the
    # confounding-adjusted standard error moves relative to the original,
    # which is not monotone in (cf_y, cf_d). Only the magnitude is pinned.
    assert float(sens.rv_qa) <= float(sens.rv_q), (
        "RVa is the strength needed to lose significance, which cannot "
        "exceed the strength needed to zero out the estimate"
    )


def test_scaling_factor_uses_the_structural_residual(sp_fit):
    """S must subtract theta*(D - m), not stop at Y - l(X).

    This is the defect the parity pin caught: leaving the treatment's
    contribution in the numerator inflates S, which overstates the bias
    bound and understates the robustness value.
    """
    sens = sp.dml_sensitivity(sp_fit, q=1.0, cf_y=CF_Y, cf_d=CF_D)
    info = sp_fit.model_info
    y_resid = np.asarray(info["_y_resid"], dtype=float)
    d_resid = np.asarray(info["_d_resid"], dtype=float)
    theta = float(sp_fit.estimate)

    structural = y_resid - theta * d_resid
    expected = float(np.sqrt(np.mean(structural**2)) / np.sqrt(np.mean(d_resid**2)))
    assert float(sens.s) == pytest.approx(expected, rel=1e-12)

    # The pre-v1.21 quantity, kept here as an explicit non-equality so a
    # regression to it cannot pass silently.
    reduced_form = float(np.std(y_resid, ddof=1) / np.std(d_resid, ddof=1))
    assert float(sens.s) < reduced_form
