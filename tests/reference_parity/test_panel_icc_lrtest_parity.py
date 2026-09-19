"""``sp.icc`` and ``sp.lrtest`` against Stata ``estat icc`` / ``lrtest`` and R.

References
----------
* ``_fixtures/panel_glmm_Stata.json`` -- ``_generate_panel_glmm_stata.do``
  (Stata 18 ``mixed`` / ``melogit`` / ``meologit`` + ``estat icc``;
  ``lrtest``; ``e(k)``), convergence criteria 1e-12.
* ``_fixtures/panel_glmm_R.json`` -- ``_generate_panel_glmm_R.R``:
  ``anova()`` on lme4 ML fits, ``AIC()``, ``performance::icc`` (0.16.0),
  ``psych::ICC`` (2.6.5).
* Data: ``_fixtures/panel_glmm_data.csv``.

ICC conventions (``estat icc``)
-------------------------------
* ``mixed``: ρ = σ²_u / (σ²_u + σ²_e); after ``melogit`` / ``meologit`` the
  latent-scale ρ = σ²_u / (σ²_u + π²/3).
* SE: delta method on the observed-information covariance of the variance
  parameters (inverse Hessian of the ML / REML criterion, or of the Laplace
  log-likelihood for the logit GLMMs).
* CI: Wald on the logit scale, mapped back.
* Before this version ``sp.icc`` used a heuristic SE (var(log σ²_u) ≈
  2 / n_groups, var(log σ²_e) ≈ 2 / (n − p), no covariance) and returned
  NaN silently for every GLMM (it looked for a ``var(Residual)`` key).

LR-test conventions
-------------------
* chi2 = 2 (ll_full − ll_restricted), df = difference in the number of
  estimated parameters (Stata ``e(k)``, R ``attr(logLik, "df")``).
* Stata ``lrtest`` and R ``anova()`` report the naive χ²(df) tail even when
  the null puts a variance on the boundary (Stata prints a note):
  ``sp.lrtest(..., boundary=False)``.  ``sp.lrtest``'s default applies the
  χ̄² mixture; for adding one random slope under an unstructured covariance
  (df 2) that is the exact 50:50 χ²₁/χ²₂ mixture.

Also pinned: ``MixedResult.n_params`` against Stata ``e(k)`` and R's AIC
(it double-counted the residual variance before this version).

Tolerance: rtol 1e-6, the default parity budget (ML optima of
``sp.mixed`` sit ~1e-7 from Stata's / lme4's).  The ``meologit`` ICC SE is
not compared: Stata's Laplace-ologit SEs are not the Hessian of its own
objective (see test_panel_glmm_parity.py).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = Path(__file__).parent / "_fixtures"
S = json.loads((_FIX / "panel_glmm_Stata.json").read_text(encoding="utf-8"))
R = json.loads((_FIX / "panel_glmm_R.json").read_text(encoding="utf-8"))
D = pd.read_csv(_FIX / "panel_glmm_data.csv")
RTOL = 1e-6


def _q(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.fixture(scope="module")
def fits():
    out = {
        "ml": _q(
            sp.mixed, D, y="y_gau", x_fixed=["x1", "x2"], group="gid", method="ml"
        ),
        "reml": _q(
            sp.mixed, D, y="y_gau", x_fixed=["x1", "x2"], group="gid", method="reml"
        ),
        "ml_x1": _q(sp.mixed, D, y="y_gau", x_fixed=["x1"], group="gid", method="ml"),
        "bin": _q(sp.melogit, D, "y_bin", ["x1", "x2"], "gid"),
        "bin_x1": _q(sp.melogit, D, "y_bin", ["x1"], "gid"),
        "olog": _q(sp.meologit, D, "y_ord", ["x1", "x2"], "gid"),
        "gauss_meglm": _q(sp.meglm, D, "y_gau", ["x1", "x2"], "gid"),
        "rs_int": _q(
            sp.mixed, D, y="y_rs", x_fixed=["x1", "x2"], group="gid", method="ml"
        ),
        "rs_un": _q(
            sp.mixed,
            D,
            y="y_rs",
            x_fixed=["x1", "x2"],
            group="gid",
            x_random=["x1"],
            cov_type="unstructured",
            method="ml",
        ),
        "rs_ind": _q(
            sp.mixed,
            D,
            y="y_rs",
            x_fixed=["x1", "x2"],
            group="gid",
            x_random=["x1"],
            cov_type="diagonal",
            method="ml",
        ),
    }
    return out


def _icc_vs(res, ref, se=True):
    np.testing.assert_allclose(res.estimate, ref["icc"], rtol=RTOL)
    if se:
        np.testing.assert_allclose(res.se, ref["se"], rtol=RTOL)
        # The logit-scale interval amplifies a relative SE difference by
        # (1 - bound) z se / (rho (1 - rho)) -- 1.8 for the melogit ICC
        # (rho = 0.029), where the SEs agree to 7e-7 and the upper bound
        # to 1.24e-6.  Hence 2 x RTOL on the bounds.
        np.testing.assert_allclose(res.ci_lower, ref["lb"], rtol=2 * RTOL)
        np.testing.assert_allclose(res.ci_upper, ref["ub"], rtol=2 * RTOL)


# ---------------------------------------------------------------------------
# ICC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,spec",
    [
        ("ml", "mixed_ml_icc"),
        ("reml", "mixed_reml_icc"),
        ("bin", "bin_laplace_icc"),
        ("gauss_meglm", "mixed_ml_icc"),
    ],
)
def test_icc_matches_stata_estat_icc(fits, key, spec):
    _icc_vs(_q(sp.icc, fits[key]), S[spec])


def test_ologit_icc_point_estimate_matches_stata(fits):
    r = sp.icc(fits["olog"])
    # 2.4e-6: Stata's Laplace-ologit optimum sits that far from ours in
    # var(_cons) (see test_panel_glmm_parity.py); rtol 5e-6.
    np.testing.assert_allclose(r.estimate, S["ologit_laplace_icc"]["icc"], rtol=5e-6)


@pytest.mark.parametrize(
    "key,rkey", [("ml", "mixed_ml"), ("reml", "mixed_reml"), ("bin", "melogit")]
)
def test_icc_matches_performance_icc(fits, key, rkey):
    np.testing.assert_allclose(
        _q(sp.icc, fits[key]).estimate, R["icc_performance"][rkey], rtol=RTOL
    )


def test_balanced_reml_icc_is_the_anova_icc1():
    """Balanced one-way layout: REML ICC = (MSB − MSW)/(MSB + (k−1) MSW).

    Before this version ``sp.mixed`` stopped at L-BFGS-B's
    relative-function-change criterion and missed this identity by 1.0e-6;
    with the Newton finish it holds to ~1e-10.
    """
    bal = D.groupby("gid", sort=True).head(4).sort_values(["gid"], kind="stable")
    r = _q(sp.icc, _q(sp.mixed, bal, y="y_gau", x_fixed=[], group="gid", method="reml"))
    w = bal["y_gau"].to_numpy().reshape(-1, 4)
    n, k = w.shape
    msb = k * np.sum((w.mean(axis=1) - w.mean()) ** 2) / (n - 1)
    msw = np.sum((w - w.mean(axis=1, keepdims=True)) ** 2) / (n * (k - 1))
    anova = (msb - msw) / (msb + (k - 1) * msw)
    np.testing.assert_allclose(anova, R["icc_balanced"]["psych_icc1"], rtol=1e-12)
    np.testing.assert_allclose(r.estimate, anova, rtol=1e-9)


def test_icc_refuses_count_glmm_and_random_slopes(fits):
    pois = _q(sp.mepoisson, D, "y_pois", ["x1", "x2"], "gid", offset="lexpo")
    with pytest.raises(MethodIncompatibility, match="not defined"):
        sp.icc(pois)
    with pytest.raises(MethodIncompatibility, match="random-intercept"):
        sp.icc(fits["rs_un"])


# ---------------------------------------------------------------------------
# lrtest
# ---------------------------------------------------------------------------


def _lr_vs(res, ref):
    np.testing.assert_allclose(res.chi2, ref["chi2"], rtol=RTOL)
    assert res.df == ref["df"]
    np.testing.assert_allclose(res.p_value, ref["p"], rtol=10 * RTOL)


def test_lrtest_fixed_effect_restriction(fits):
    r = sp.lrtest(fits["ml_x1"], fits["ml"])
    assert not r.boundary_corrected
    _lr_vs(r, S["lr_fixed"])
    _lr_vs(r, R["lr_fixed"])


def test_lrtest_fixed_effect_restriction_glmm(fits):
    _lr_vs(sp.lrtest(fits["bin_x1"], fits["bin"]), S["lr_melogit_fixed"])


@pytest.mark.parametrize(
    "key,spec", [("rs_un", "lr_slope_un"), ("rs_ind", "lr_slope_ind")]
)
def test_lrtest_random_slope_naive_matches_stata_and_r(fits, key, spec):
    r = sp.lrtest(fits["rs_int"], fits[key], boundary=False)
    _lr_vs(r, S[spec])
    _lr_vs(r, R[spec])


def test_lrtest_random_slope_default_is_the_exact_mixture(fits):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the exact case must not warn
        r = sp.lrtest(fits["rs_int"], fits["rs_un"])
    assert r.boundary_corrected and r.df == 2
    np.testing.assert_allclose(
        r.p_value,
        0.5 * stats.chi2.sf(r.chi2, 1) + 0.5 * stats.chi2.sf(r.chi2, 2),
        rtol=1e-14,
    )
    r1 = sp.lrtest(fits["rs_int"], fits["rs_ind"])
    np.testing.assert_allclose(r1.p_value, 0.5 * stats.chi2.sf(r1.chi2, 1), rtol=1e-14)


def test_parameter_counts_match_stata_e_k_and_r_aic(fits):
    assert fits["ml"].n_params == S["mixed_ml"]["k"] == 5
    assert fits["rs_un"].n_params == S["rs_slope_un"]["k"] == 7
    assert fits["rs_ind"].n_params == S["rs_slope_ind"]["k"] == 6
    np.testing.assert_allclose(fits["ml"].aic, R["aic_mixed_ml"], rtol=1e-9)
