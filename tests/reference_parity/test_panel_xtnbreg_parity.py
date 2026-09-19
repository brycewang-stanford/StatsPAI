"""``sp.xtnbreg`` against Stata ``xtnbreg, fe`` / ``xtnbreg, re`` and R ``pglm``.

References
----------
* ``_fixtures/panel_glmm_Stata.json`` (specs ``xtnbreg_fe``,
  ``xtnbreg_fe_year``, ``xtnbreg_re``) -- ``_generate_panel_glmm_stata.do``,
  Stata 18, convergence criteria tightened to 1e-12.
* ``_fixtures/panel_glmm_R.json`` (``xtnbreg_pglm_within`` /
  ``xtnbreg_pglm_random``) -- ``_generate_panel_glmm_R.R``, pglm 0.2.4,
  Newton-Raphson with gradtol 1e-12.
* Data: ``_fixtures/panel_count_data.csv`` (``_generate_panel_glmm_data.py``),
  40 x 6 balanced count panel, two panels with all-zero outcomes.

The model (both references, and the likelihoods printed in Stata's [XT]
``xtnbreg`` Methods and formulas)
----------------------------------------------------------------------
Hausman-Hall-Griliches: y_it | γ_it ~ Poisson, γ_it | δ_i ~ gamma(λ_it, δ_i),
λ_it = exp(x_it'β) -- a negative binomial with panel-constant
variance-to-mean ratio 1 + δ_i.

* ``fe``: the likelihood conditional on Σ_t y_it.  The intercept is
  identified; panels with an all-zero outcome (or one observation) carry no
  information and are dropped: Stata ``note: 2 groups (12 obs) dropped``,
  so N = 228 of 240.
* ``re``: 1/(1 + δ_i) ~ Beta(r, s), reported as /ln_r and /ln_s (pglm
  reports r and s themselves as ``a`` and ``b``).
* Standard errors: inverse observed information.

Before this version ``sp.xtnbreg(model="fe")`` fitted an *unconditional*
NB-2 with entity dummies and ``model="re"`` the normal random-intercept NB-2
GLMM (``sp.menbreg``) -- neither is what ``xtnbreg`` computes, while the
fe result was labelled ``stata_equivalent = "xtnbreg, fe"``.  Both remain
available as ``model="ufe"`` / ``model="normal_re"``.

Tolerance
---------
rtol 1e-7 on coefficients and SEs (1e-12 on the log-likelihood).  Our
Newton-Raphson ends with a score of 7e-14; Stata's ``xtnbreg, fe`` estimate
carries a score of 3e-7 (evaluated with our analytic score, which the
log-likelihood identity to 1e-15 shows is the same function), which puts
its intercept 2.6e-8 from ours; pglm's intercept sits 2.7e-8 away (its
gradtol).  The RE fit agrees with Stata to 1e-10.
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
D = pd.read_csv(_FIX / "panel_count_data.csv")

RTOL = 1e-7


def _fit(model, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.xtnbreg("y ~ z1 + z2", data=D, entity="pid", model=model, **kw)


@pytest.fixture(scope="module")
def fe():
    return _fit("fe")


@pytest.fixture(scope="module")
def re():
    return _fit("re")


def test_fe_matches_stata(fe):
    ref = S["xtnbreg_fe"]
    for nm in ("z1", "z2", "_cons"):
        np.testing.assert_allclose(fe.params[nm], ref["b"][f"y:{nm}"], rtol=RTOL)
        np.testing.assert_allclose(fe.std_errors[nm], ref["se"][f"y:{nm}"], rtol=RTOL)
    np.testing.assert_allclose(fe.model_info["ll"], ref["ll"], rtol=1e-12)
    assert fe.data_info["nobs"] == ref["N"] == 228
    assert fe.model_info["n_groups_dropped"] == 2


def test_fe_with_year_effects_matches_stata():
    r = _fit("fe", time="year", time_effects=True)
    ref = S["xtnbreg_fe_year"]
    for nm in ("z1", "z2", "_cons"):
        np.testing.assert_allclose(r.params[nm], ref["b"][f"y:{nm}"], rtol=RTOL)
        np.testing.assert_allclose(r.std_errors[nm], ref["se"][f"y:{nm}"], rtol=RTOL)
    for yr in range(2, 7):
        np.testing.assert_allclose(
            r.params[f"year={yr}"], ref["b"][f"y:{yr}.year"], rtol=1e-7
        )
    np.testing.assert_allclose(r.model_info["ll"], ref["ll"], rtol=1e-12)


def test_re_matches_stata(re):
    ref = S["xtnbreg_re"]
    for nm in ("z1", "z2", "_cons"):
        np.testing.assert_allclose(re.params[nm], ref["b"][f"y:{nm}"], rtol=RTOL)
        np.testing.assert_allclose(re.std_errors[nm], ref["se"][f"y:{nm}"], rtol=RTOL)
    for nm in ("/ln_r", "/ln_s"):
        np.testing.assert_allclose(re.params[nm], ref["b"][nm], rtol=RTOL)
        np.testing.assert_allclose(re.std_errors[nm], ref["se"][nm], rtol=RTOL)
    np.testing.assert_allclose(re.model_info["ll"], ref["ll"], rtol=1e-12)
    assert re.data_info["nobs"] == ref["N"] == 240


def test_fe_matches_pglm_within(fe):
    ref = R["xtnbreg_pglm_within"]
    for nm, rn in (("z1", "z1"), ("z2", "z2"), ("_cons", "(Intercept)")):
        np.testing.assert_allclose(fe.params[nm], ref["b"][rn], rtol=1e-7)
        np.testing.assert_allclose(fe.std_errors[nm], ref["se"][rn], rtol=1e-7)
    np.testing.assert_allclose(fe.model_info["ll"], ref["ll"], rtol=1e-12)


def test_re_matches_pglm_random(re):
    ref = R["xtnbreg_pglm_random"]
    for nm, rn in (("z1", "z1"), ("z2", "z2"), ("_cons", "(Intercept)")):
        np.testing.assert_allclose(re.params[nm], ref["b"][rn], rtol=1e-7)
        np.testing.assert_allclose(re.std_errors[nm], ref["se"][rn], rtol=1e-7)
    # pglm estimates r and s directly; delta method from ln r / ln s.
    r_hat, s_hat = np.exp(re.params["/ln_r"]), np.exp(re.params["/ln_s"])
    np.testing.assert_allclose(r_hat, ref["b"]["a"], rtol=1e-7)
    np.testing.assert_allclose(s_hat, ref["b"]["b"], rtol=1e-7)
    np.testing.assert_allclose(
        r_hat * re.std_errors["/ln_r"], ref["se"]["a"], rtol=1e-7
    )
    np.testing.assert_allclose(
        s_hat * re.std_errors["/ln_s"], ref["se"]["b"], rtol=1e-7
    )
    np.testing.assert_allclose(re.model_info["ll"], ref["ll"], rtol=1e-12)


# ---------------------------------------------------------------------------
# Reference-free identities
# ---------------------------------------------------------------------------


def test_fe_conditional_likelihood_is_invariant_to_all_zero_panels():
    """An all-zero panel contributes exactly 0 to the conditional log L."""
    from statspai.regression._xtnbreg_hhg import fe_loglik

    X = np.column_stack([D["z1"], D["z2"], np.ones(len(D))])
    gidx = np.unique(D["pid"], return_inverse=True)[1]
    zero = D.groupby("pid")["y"].transform("sum").to_numpy() == 0
    assert zero.sum() == 12
    th = np.array([0.3, -0.2, 0.5])
    full = fe_loglik(
        th, X, D["y"].to_numpy(float), np.zeros(len(D)), gidx, gidx.max() + 1
    )
    keep = ~zero
    g2 = np.unique(D["pid"][keep], return_inverse=True)[1]
    kept = fe_loglik(
        th,
        X[keep],
        D["y"].to_numpy(float)[keep],
        np.zeros(keep.sum()),
        g2,
        g2.max() + 1,
    )
    np.testing.assert_allclose(full, kept, rtol=1e-13)


def test_score_is_zero_at_the_optimum(fe, re):
    from statspai.regression._xtnbreg_hhg import fe_score, re_score

    X = np.column_stack([D["z1"], D["z2"], np.ones(len(D))])
    y = D["y"].to_numpy(float)
    g = np.unique(D["pid"], return_inverse=True)[1]
    th_fe = fe.params[["z1", "z2", "_cons"]].to_numpy()
    assert (
        np.max(np.abs(fe_score(th_fe, X, y, np.zeros(len(D)), g, g.max() + 1))) < 1e-8
    )
    th_re = re.params[["z1", "z2", "_cons", "/ln_r", "/ln_s"]].to_numpy()
    assert (
        np.max(np.abs(re_score(th_re, X, y, np.zeros(len(D)), g, g.max() + 1))) < 1e-8
    )


def test_previous_estimators_remain_reachable():
    ufe = _fit("ufe")
    assert ufe.model_info["panel_model"] == "unconditional_fixed_effects"
    nre = _fit("normal_re")
    assert type(nre).__name__ == "MEGLMResult"


def test_fe_refuses_robust_standard_errors():
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        _fit("fe", robust="hc1")
