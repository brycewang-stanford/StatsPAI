"""``sp.interactive_fe`` (Bai 2009) against Stata ``regife`` and R ``phtt::Eup``.

References
----------
* ``_fixtures/panel_glmm_Stata.json`` (``ife_regife``,
  ``ife_regife_cluster``) -- ``_generate_panel_glmm_stata.do``:
  ``regife y xa xb, ife(unit period, 2) noconstant`` (SSC regife, Gomez;
  reghdfe / ftools / require in a private ado directory), tolerance 1e-12.
* ``_fixtures/panel_glmm_R.json`` (``ife_phtt``) -- ``_generate_panel_glmm_R.R``:
  ``phtt::Eup(additive.effects = "none", factor.dim = 2)``, phtt 3.1.2
  (archived from CRAN), convergence 1e-12.
* Data: ``_fixtures/panel_ife_data.csv`` -- balanced 40 x 15 panel, two
  factors whose loadings also drive the regressors; no intercept, no
  additive effects.

Estimand and conventions
------------------------
* Slopes: the least-squares fixed point min_{β,Λ,F} ||Y − Xβ − ΛF'||², all
  three implementations.  Agreement 1e-11 (both references iterate to
  1e-12).
* Standard errors (Bai 2009): regressors enter through Z_j = M_Λ X_j M_F.
  Before this version StatsPAI used M_F X_j only (the loading-space
  projection was missing) and so reported neither reference's number.
* ``regife`` obtains its SEs from ``reghdfe`` with the loadings and
  factors absorbed as slopes (``i.period#c.λ``, ``i.unit#c.f``); reghdfe
  counts r(N + T) absorbed parameters and cannot net out the r² rotation
  redundancy.  ``dof='regife'`` reproduces that (homoskedastic and CR1
  cluster-by-unit) exactly; the default ``dof='exact'`` counts
  r(N + T − r), so it differs from regife by the factor
  sqrt((NT − k − r(N+T)) / (NT − k − r(N+T−r))) -- asserted below.
* ``phtt``'s error.type = 1 SE is sig2.hat (Z'Z)^{-1} with the same Z; its
  sig2.hat demeans the residuals *unit by unit* before squaring although
  the model has no unit effect (``sum(diag(var(residuals)))``), which we do
  not copy.  The test reconstructs phtt's number from our Z and residuals.

Tolerance: rtol 1e-9 on slopes and SEs (observed <= 2e-11).
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
D = pd.read_csv(_FIX / "panel_ife_data.csv")
N, T, K, RR = 40, 15, 2, 2
RTOL = 1e-9


def _fit(**kw):
    return sp.interactive_fe(
        D, "y", ["xa", "xb"], id="unit", time="period", n_factors=2, **kw
    )


def test_slopes_match_regife_and_phtt():
    r = _fit()
    for nm in ("xa", "xb"):
        np.testing.assert_allclose(r.params[nm], S["ife_regife"]["b"][nm], rtol=RTOL)
        np.testing.assert_allclose(r.params[nm], R["ife_phtt"]["b"][nm], rtol=RTOL)


def test_homoskedastic_se_matches_regife_with_its_dof():
    r = _fit(robust=False, dof="regife")
    for nm in ("xa", "xb"):
        np.testing.assert_allclose(
            r.std_errors[nm], S["ife_regife"]["se"][nm], rtol=RTOL
        )


def test_cluster_se_matches_regife_with_its_dof():
    r = _fit(robust=True, dof="regife")
    for nm in ("xa", "xb"):
        np.testing.assert_allclose(
            r.std_errors[nm], S["ife_regife_cluster"]["se"][nm], rtol=RTOL
        )


def test_exact_dof_differs_from_regife_by_the_rotation_count():
    ex, rg = _fit(robust=False), _fit(robust=False, dof="regife")
    nt = N * T
    factor = np.sqrt((nt - K - RR * (N + T)) / (nt - K - RR * (N + T - RR)))
    np.testing.assert_allclose(ex.std_errors / rg.std_errors, factor, rtol=1e-12)
    assert ex.data_info["df_resid"] == nt - K - RR * (N + T - RR) == 492


def test_phtt_se_reconstructed_from_our_z_and_residuals():
    r = _fit(robust=False)
    rss = float(np.sum(np.asarray(R["ife_phtt"]["rss"])))
    V = r.data_info["var_cov"]
    zz_inv = V / r.diagnostics["sigma2"]
    # phtt's sig2.hat: residuals demeaned unit by unit, over df = 488.
    y = D.pivot(index="unit", columns="period", values="y").to_numpy()
    xa = D.pivot(index="unit", columns="period", values="xa").to_numpy()
    xb = D.pivot(index="unit", columns="period", values="xb").to_numpy()
    E = y - r.params["xa"] * xa - r.params["xb"] * xb
    E = E - r.diagnostics["loadings"] @ r.diagnostics["factors"].T
    np.testing.assert_allclose(float(np.sum(E**2)), rss, rtol=1e-10)
    sig2_phtt = (
        float(np.sum((E - E.mean(axis=1, keepdims=True)) ** 2))
        / R["ife_phtt"]["degrees_of_freedom"]
    )
    np.testing.assert_allclose(sig2_phtt, R["ife_phtt"]["sig2_hat"], rtol=1e-10)
    se_phtt = np.sqrt(np.diag(zz_inv) * sig2_phtt)
    np.testing.assert_allclose(se_phtt[0], R["ife_phtt"]["se_type1"]["xa"], rtol=RTOL)
    np.testing.assert_allclose(se_phtt[1], R["ife_phtt"]["se_type1"]["xb"], rtol=RTOL)


def test_first_order_condition_holds():
    """sum_i X_i' M_F (Y_i - X_i b) = 0 at the fixed point (Bai 2009)."""
    r = _fit()
    F = r.diagnostics["factors"]
    MF = np.eye(T) - F @ np.linalg.solve(F.T @ F, F.T)

    def P(v):
        return D.pivot(index="unit", columns="period", values=v).to_numpy()

    E = P("y") - r.params["xa"] * P("xa") - r.params["xb"] * P("xb")
    for v in ("xa", "xb"):
        assert abs(np.sum(P(v) * (E @ MF))) < 1e-8


def test_pca_method_is_deprecated_and_identical():
    a = _fit()
    with pytest.warns(DeprecationWarning):
        b = _fit(method="pca")
    np.testing.assert_array_equal(a.params.values, b.params.values)


def test_unbalanced_panel_drops_units_loudly():
    d = D[~((D["unit"] == 3) & (D["period"] == 7))]
    with pytest.warns(RuntimeWarning, match="dropping 1 of 40 units"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sp.interactive_fe(
                d, "y", ["xa", "xb"], id="unit", time="period", n_factors=2
            )
