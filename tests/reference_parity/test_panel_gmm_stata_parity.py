"""``sp.gmm`` against Stata ``gmm`` (linear and nonlinear IV) and R ``gmm::gmm``.

Complements ``test_general_gmm_parity.py`` (linear IV vs R ``gmm``).

References
----------
* ``_fixtures/panel_glmm_Stata.json`` (``gmm_lin_*``, ``gmm_nl_*``) --
  ``_generate_panel_glmm_stata.do``, Stata 18 ``gmm``: ``twostep`` /
  ``igmm`` (igmmeps / igmmweps 1e-13) / ``onestep``, conv_ptol 1e-13.
* ``_fixtures/panel_glmm_R.json`` (``gmm_nl_*``) -- ``_generate_panel_glmm_R.R``,
  gmm 1.9.1, optim reltol 1e-15.
* Data: ``_fixtures/general_gmm_data.csv`` (linear; from
  ``_generate_general_gmm_R.R``) and ``_fixtures/gmm_nl_data.csv``
  (``_generate_panel_glmm_data.py``).

Moment conditions
-----------------
linear:     E[z (y − b1 x1 − b0)] = 0
nonlinear:  E[z (y exp(−b1 x1 − b0) − 1)] = 0      (Mullahy's IV-Poisson)
with z = (z1, z2, z3, 1): four moments, two parameters.

Conventions each number depends on
----------------------------------
* Stata's first-step weight is ``winitial(unadjusted)`` = (Z'Z/N)⁻¹ -- passed
  as ``W=``; R's ``twoStep`` starts from the identity (``W=None``).
* S is uncentred in Stata (``center=False``, the default here) and centred by
  default in R (``center=True``); R is compared both ways.
* Robust VCE of the two-step estimator: Stata's documented formula uses the
  weight the last round was minimised with and S at the final estimate --
  ``sandwich_weight='estimation'``.  The default ``'reestimated'`` (S⁻¹ at
  the final estimate, R's ``gmm``) differs from Stata by 2.6e-6 (linear) /
  4.4e-5 (nonlinear) relative, which the test pins as the size of that
  convention gap.  The two coincide for iterated and one-step GMM.
* Hansen's J after ``onestep``: Stata recomputes an *unadjusted*
  (homoskedastic) weight for it, a different statistic from N·Q(β̂) at the
  estimation weight reported here; not compared.
* Jacobian: analytic here, numerical (``deriv()``) in Stata, which limits
  nonlinear SE agreement to ~2.5e-7.

Tolerance: coefficients rtol 1e-10 vs Stata (observed 1.2e-12 after the
Gauss-Newton finish added in this version -- BFGS alone stopped 1e-8 away
and reported ``converged=False``); SEs 1e-9 linear / 1e-6 nonlinear
(numerical Jacobian on Stata's side); R 1e-6 (optim slack 1.8e-7).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
S = json.loads((_FIX / "panel_glmm_Stata.json").read_text(encoding="utf-8"))
R = json.loads((_FIX / "panel_glmm_R.json").read_text(encoding="utf-8"))
NAMES = ["x1", "_cons"]


def _problem(csv, nonlinear):
    d = pd.read_csv(_FIX / csv)
    Z = np.column_stack([d["z1"], d["z2"], d["z3"], np.ones(len(d))])
    X = np.column_stack([d["x1"], np.ones(len(d))])
    y = d["y"].to_numpy(float)
    n = len(d)
    if nonlinear:

        def mom(th, _):
            return (y * np.exp(-X @ th) - 1.0)[:, None] * Z

        def jac(th, _):
            return -(Z.T @ (X * (y * np.exp(-X @ th))[:, None])) / n

    else:

        def mom(th, _):
            return (y - X @ th)[:, None] * Z

        def jac(th, _):
            return -(Z.T @ X) / n

    return d, Z, mom, jac


PROBLEMS = {"lin": ("general_gmm_data.csv", False), "nl": ("gmm_nl_data.csv", True)}


def _fit(tag, method, stata_weight=True, **kw):
    csv, nl = PROBLEMS[tag]
    d, Z, mom, jac = _problem(csv, nl)
    W0 = np.linalg.inv(Z.T @ Z / len(d)) if stata_weight else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.gmm(
            mom,
            np.zeros(2),
            d,
            W=W0,
            method=method,
            param_names=NAMES,
            jacobian=jac,
            tol=1e-13,
            maxiter=1000,
            **kw,
        )


SE_RTOL = {"lin": 1e-9, "nl": 1e-6}


@pytest.mark.parametrize("tag", ["lin", "nl"])
@pytest.mark.parametrize(
    "method,spec",
    [("twostep", "twostep"), ("iterative", "igmm"), ("onestep", "onestep")],
)
def test_matches_stata_gmm(tag, method, spec):
    r = _fit(tag, method, sandwich_weight="estimation")
    ref = S[f"gmm_{tag}_{spec}"]
    for nm in NAMES:
        np.testing.assert_allclose(r.params[nm], ref["b"][f"xb:{nm}"], rtol=1e-10)
        np.testing.assert_allclose(
            r.std_errors[nm], ref["se"][f"xb:{nm}"], rtol=SE_RTOL[tag]
        )
    if method != "onestep":
        np.testing.assert_allclose(r.diagnostics["J_stat"], ref["J"], rtol=1e-10)
    assert r.diagnostics["converged"]


@pytest.mark.parametrize("tag,gap", [("lin", 2.6e-6), ("nl", 4.4e-5)])
def test_default_twostep_se_differs_from_stata_by_the_weight_convention(tag, gap):
    """Default = S⁻¹ re-estimated at the final β̂ (R); Stata keeps the
    estimation weight.  Same β̂ and J; the SE gap is the convention."""
    a = _fit(tag, "twostep")
    b = _fit(tag, "twostep", sandwich_weight="estimation")
    np.testing.assert_array_equal(a.params.values, b.params.values)
    rel = np.max(np.abs(a.std_errors.values / b.std_errors.values - 1.0))
    assert 0.5 * gap < rel < 2.0 * gap


@pytest.mark.parametrize(
    "method,rm", [("twostep", "twostep"), ("iterative", "iterative")]
)
@pytest.mark.parametrize("center", [True, False])
def test_nonlinear_matches_r_gmm(method, rm, center):
    r = _fit("nl", method, stata_weight=False, center=center)
    ref = R[f"gmm_nl_{rm}_{'centered' if center else 'uncentered'}"]
    for nm in NAMES:
        np.testing.assert_allclose(r.params[nm], ref["coef"][nm], rtol=1e-6)
        np.testing.assert_allclose(r.std_errors[nm], ref["se"][nm], rtol=1e-6)
    np.testing.assert_allclose(r.diagnostics["J_stat"], ref["J"], rtol=1e-6)


def test_nonlinear_first_order_condition():
    """D'W ḡ = 0 at the optimum (reference-free)."""
    csv, _ = PROBLEMS["nl"]
    d, Z, mom, jac = _problem(csv, True)
    W0 = np.linalg.inv(Z.T @ Z / len(d))
    r = _fit("nl", "onestep")
    th = r.params[NAMES].to_numpy()
    foc = jac(th, d).T @ W0 @ mom(th, d).mean(axis=0)
    assert np.max(np.abs(foc)) < 1e-14
