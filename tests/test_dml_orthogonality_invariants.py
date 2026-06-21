"""Methodological self-consistency invariants for ``sp.dml``.

These tests assert the *defining algebraic properties* of the
Double/Debiased ML estimators directly from a fitted result — they need
no external reference implementation (no ``doubleml`` install). Where the
machine-precision parity suite
(``tests/external_parity/test_dml_python_parity.py``) checks agreement
with the upstream DoubleML package, this module checks that the
estimator solves the moment it claims to solve and reports the variance
it claims to report. Concretely, for the partially-linear model with
Neyman-orthogonal score ``psi = (Y - g - theta*(D - m)) * (D - m)``:

1. **DML2 pooled identity** — ``theta_hat`` equals the closed-form
   pooled-moment solution ``sum(d_tilde*y_tilde)/sum(d_tilde**2)`` over
   *all* cross-fitted residuals (i.e. it is the DML2 estimator, not a
   per-fold DML1 average).
2. **Neyman moment solved** — the empirical score mean is ~0 at the
   solution.
3. **Sandwich variance identity** — the reported SE equals
   ``sqrt(mean(psi**2) / (J**2 * n))`` with ``J = -mean(d_tilde**2)``.
4. **Repeated cross-fitting aggregation** — with ``n_rep > 1`` the point
   estimate is the median of the per-rep estimates and the SE follows
   Chernozhukov et al. (2018, eq. 3.7):
   ``se = sqrt(median(se_r**2 + (theta_r - theta_med)**2))``.
5. **IRM AIPW moment** — for the interactive (binary-treatment) model
   ``theta_hat = mean(psi_AIPW)``, so the centered score has mean ~0.

References
----------
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
  Newey, W. & Robins, J. (2018). Double/debiased machine learning for
  treatment and structural parameters. *The Econometrics Journal*,
  21(1), C1-C68. [@chernozhukov2018double]
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

pytest.importorskip("sklearn")
from sklearn.linear_model import LassoCV, LogisticRegressionCV

_FIXTURE = (
    pathlib.Path(__file__).parent / "reference_parity" / "_fixtures" / "dml_data.csv"
)


@pytest.fixture(scope="module")
def dml_data() -> pd.DataFrame:
    return pd.read_csv(_FIXTURE)


@pytest.fixture(scope="module")
def x_cols() -> list:
    return [f"x{i}" for i in range(1, 11)]


@pytest.fixture(scope="module")
def plr_fit(dml_data, x_cols):
    np.random.seed(42)
    return sp.dml(
        data=dml_data,
        y="y",
        treat="d",
        covariates=x_cols,
        model="plr",
        ml_g=LassoCV(cv=5, random_state=42),
        ml_m=LassoCV(cv=5, random_state=42),
        n_folds=5,
        n_rep=1,
        random_state=42,
    )


def test_plr_is_dml2_pooled_estimator(plr_fit):
    """theta_hat == pooled-moment solution over ALL cross-fit residuals."""
    mi = plr_fit.model_info
    y_t = np.asarray(mi["_y_resid"], dtype=float)
    d_t = np.asarray(mi["_d_resid"], dtype=float)
    theta_pooled = float(np.sum(d_t * y_t) / np.sum(d_t * d_t))
    assert abs(float(plr_fit.estimate) - theta_pooled) < 1e-9, (
        f"PLR is not the DML2 pooled estimator: "
        f"theta={plr_fit.estimate:.10f}, pooled={theta_pooled:.10f}"
    )


def test_plr_neyman_moment_is_solved(plr_fit):
    """Empirical orthogonal-score mean ~ 0 at the solution."""
    mi = plr_fit.model_info
    y_t = np.asarray(mi["_y_resid"], dtype=float)
    d_t = np.asarray(mi["_d_resid"], dtype=float)
    theta = float(plr_fit.estimate)
    psi = (y_t - theta * d_t) * d_t
    # Scale-relative tolerance: |mean(psi)| should be vanishing vs the
    # score's own dispersion (the estimator solves sum(psi)=0 exactly).
    assert abs(np.mean(psi)) < 1e-8 * (np.std(psi) + 1.0)


def test_plr_sandwich_variance_identity(plr_fit):
    """Reported SE == sqrt(mean(psi^2) / (J^2 n)), J = -mean(d_tilde^2)."""
    mi = plr_fit.model_info
    y_t = np.asarray(mi["_y_resid"], dtype=float)
    d_t = np.asarray(mi["_d_resid"], dtype=float)
    theta = float(plr_fit.estimate)
    n = int(plr_fit.n_obs)
    psi = (y_t - theta * d_t) * d_t
    J = -np.mean(d_t * d_t)
    se_recomputed = float(np.sqrt(np.mean(psi**2) / (J**2 * n)))
    assert abs(float(plr_fit.se) - se_recomputed) < 1e-9, (
        f"PLR SE is not the DML sandwich variance: "
        f"se={plr_fit.se:.10f}, recomputed={se_recomputed:.10f}"
    )


def test_plr_nrep_median_and_variance_correction(dml_data, x_cols):
    """n_rep>1: theta = median(theta_r); se per Chernozhukov 2018 eq 3.7."""
    np.random.seed(42)
    res = sp.dml(
        data=dml_data,
        y="y",
        treat="d",
        covariates=x_cols,
        model="plr",
        ml_g=LassoCV(cv=5, random_state=42),
        ml_m=LassoCV(cv=5, random_state=42),
        n_folds=5,
        n_rep=3,
        random_state=42,
    )
    mi = res.model_info
    theta_r = np.asarray(mi["theta_all_reps"], dtype=float)
    se_r = np.asarray(mi["se_all_reps"], dtype=float)
    theta_med = float(np.median(theta_r))
    se_formula = float(np.sqrt(np.median(se_r**2 + (theta_r - theta_med) ** 2)))
    assert abs(float(res.estimate) - theta_med) < 1e-12
    assert abs(float(res.se) - se_formula) < 1e-12


def test_irm_aipw_moment_is_solved(dml_data, x_cols):
    """IRM theta = mean(psi_AIPW): the centered score has mean ~ 0."""
    np.random.seed(42)
    res = sp.dml(
        data=dml_data,
        y="y",
        treat="d_bin",
        covariates=x_cols,
        model="irm",
        ml_g=LassoCV(cv=5, random_state=42),
        ml_m=LogisticRegressionCV(cv=5, random_state=42, max_iter=2000),
        n_folds=5,
        n_rep=1,
        random_state=42,
    )
    # model_info["_y_resid"] is the centered AIPW score psi - theta_hat.
    centered = np.asarray(res.model_info["_y_resid"], dtype=float)
    assert abs(np.mean(centered)) < 1e-9
