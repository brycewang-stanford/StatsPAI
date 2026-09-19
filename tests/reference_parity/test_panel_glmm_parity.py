"""GLMM family (``sp.mepoisson`` / ``menbreg`` / ``megamma`` / ``meologit`` /
``meglm``) against Stata's ``me*`` commands and R's lme4 / glmmTMB / ordinal.

References
----------
* ``_fixtures/panel_glmm_Stata.json`` -- ``_generate_panel_glmm_stata.do``
  (Stata 18: ``mepoisson``, ``melogit``, ``menbreg``, ``meglm``,
  ``meologit``, ``mixed``), integration method explicit on every call,
  convergence criteria tightened to 1e-12.
* ``_fixtures/panel_glmm_R.json`` -- ``_generate_panel_glmm_R.R`` (lme4
  2.0.1 ``glmer`` / ``glmer.nb`` / ``lmer``, glmmTMB 1.1.14 with its fit
  finished by Newton steps on TMB's own AD gradient, ordinal 2025.12.29
  ``clmm``).
* Data: ``_fixtures/panel_glmm_data.csv`` (``_generate_panel_glmm_data.py``),
  60 groups of 4-14 observations, one outcome per family.

Conventions every number depends on
-----------------------------------
* ``nAGQ=1`` is the Laplace approximation, ``nAGQ=k`` mode-curvature
  adaptive Gauss-Hermite quadrature: Stata ``intmethod(laplace)`` /
  ``intmethod(mcaghermite) intpoints(k)``, lme4 / clmm ``nAGQ``.  Stata's
  default ``mvaghermite`` re-centres the nodes at the posterior mean and is a
  different quadrature rule; it is not compared.
* The curvature of the log integrand at the mode is the *observed*
  information (``curvature='observed'``, the default): Stata, glmmTMB and
  clmm.  lme4 uses Fisher (PIRLS) weights -- identical for the canonical
  Poisson-log / binomial-logit links, different for NB-2-log, which
  ``glmer.nb`` is compared against with ``curvature='expected'``.
* Parameter vectors: log sd of the random intercept; ``log alpha`` (NB-2;
  Stata ``/lnalpha``, glmmTMB ``betadisp = -log alpha``), ``log phi``
  (gamma; Stata ``/logs`` is log sqrt(phi), glmmTMB ``betadisp = -log phi``),
  ``log sigma^2`` (Gaussian); ordered thresholds kappa_1 and
  log(kappa_k - kappa_{k-1}).
* Standard errors are the inverse Hessian of the approximated marginal
  log-likelihood over the full parameter vector (``vce(oim)``).

What is asserted, and at what tolerance
---------------------------------------
1. **Objective identity.**  Our marginal log-likelihood evaluated at the
   *reference's* parameter vector equals the reference's reported logLik,
   rtol 1e-11 (observed <= 7e-12).  This is the strongest statement
   available: it certifies the likelihood, the Laplace / AGHQ rule and the
   curvature convention independently of any optimiser.
2. **Estimates** at our optimum vs the reference optimum, rtol 1e-6, where
   the reference is converged.  Three Stata AGHQ fits (``pois_mcagh7``,
   ``nb_mcagh7``, ``gamma_mcagh7``) stop 3e-5 / 7e-6 / 1e-5 short of the
   optimum of their own objective even at criteria 1e-12: item 1 shows the
   objective is the same function, and our optimum attains a strictly
   larger value of it, so those are asserted that way instead (and Poisson
   AGHQ is pinned against lme4 at 1e-6).  ``clmm`` stops with a gradient of
   2e-3; likewise.

Defects this file pins (before this version):

* ``meglm(family='gaussian')`` held the residual variance at 1 -- not the
  Gaussian mixed model (x1 SE 0.0428 vs 0.0335).
* The Laplace path stopped at L-BFGS-B's relative-function-change criterion,
  ~2.7e-4 (relative) from the optimum; estimates are now finished with
  Newton steps.
* NB-2, gamma and ordinal-logit used the Fisher (expected) curvature in the
  Laplace approximation and the AGHQ node scaling -- a different
  approximation from the one Stata, glmmTMB and clmm compute (gamma _cons
  0.6196 vs 0.6372).  The Fisher version is kept as
  ``curvature='expected'`` and pinned against lme4's ``glmer.nb``.
* ``meologit`` reported fixed-effect SEs from the variance-components-fixed
  conditional information, which omits the covariance-parameter
  uncertainty, and no threshold SEs.
3. **Standard errors**, rtol 1e-6, against references whose Hessian is
   exact: glmmTMB's AD Hessian (NB-2 and gamma Laplace, observed 1.5e-7 /
   7e-8), Stata for the canonical-link Laplace fits (observed <= 4.4e-7) and
   lme4 for Poisson AGHQ.  Stata's reported SEs for the non-canonical Laplace
   fits (NB-2 4.5e-4, gamma 5.5e-3, ordinal 5e-5 off) and two AGHQ fits are
   *not* the Hessian of its own objective at its own estimates -- evaluated
   at Stata's parameter vector our Hessian (verified step-size stable to
   1e-7, and equal to TMB's AD Hessian where TMB applies) still differs by
   the same amount.  Recorded in ``docs/dev/campaign_phase3/panel_glmm.md``;
   not asserted.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.multilevel import _ordinal
from statspai.multilevel import glmm as _glmm

_FIX = Path(__file__).parent / "_fixtures"
S = json.loads((_FIX / "panel_glmm_Stata.json").read_text(encoding="utf-8"))
R = json.loads((_FIX / "panel_glmm_R.json").read_text(encoding="utf-8"))
D = pd.read_csv(_FIX / "panel_glmm_data.csv")

RTOL = 1e-6
LL_RTOL = 1e-11


def _close(ours, ref, rtol=RTOL, atol=0.0, msg=""):
    np.testing.assert_allclose(
        float(ours), float(ref), rtol=rtol, atol=atol, err_msg=msg
    )


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


# ---------------------------------------------------------------------------
# Objective evaluation at an arbitrary parameter vector
# ---------------------------------------------------------------------------

_GIDX = np.unique(D["gid"].to_numpy(), return_inverse=True)[1]
_NG = int(_GIDX.max()) + 1
_X3 = np.column_stack([np.ones(len(D)), D["x1"], D["x2"]])


def _glmm_ll(y, family, theta, nagq=1, offset=None, trials=None, curvature="observed"):
    nodes, w = _glmm._gh_nodes(nagq) if nagq > 1 else (None, np.ones(1))
    return -_glmm._glmm_nll_ri(
        np.asarray(theta, float),
        _X3,
        D[y].to_numpy(float),
        D[trials].to_numpy(float) if trials else np.ones(len(D)),
        D[offset].to_numpy(float) if offset else np.zeros(len(D)),
        _GIDX,
        _NG,
        3,
        "unstructured",
        _glmm._resolve_family(family),
        np.zeros(_NG),
        nagq,
        nodes,
        np.log(w),
        curvature == "observed",
    )


def _ologit_ll(theta, nagq=1):
    nodes, w = _glmm._gh_nodes(nagq) if nagq > 1 else (None, np.ones(1))
    return -_ordinal._ordinal_nll_ri(
        np.asarray(theta, float),
        D[["x1", "x2"]].to_numpy(float),
        D["y_ord"].to_numpy(int),
        np.zeros(len(D)),
        _GIDX,
        _NG,
        2,
        4,
        "unstructured",
        np.zeros(_NG),
        nagq,
        nodes,
        np.log(w),
        True,
    )


def _stata_theta(spec, y, disp=None):
    b = S[spec]["b"]
    th = [
        b[f"{y}:_cons"],
        b[f"{y}:x1"],
        b[f"{y}:x2"],
        0.5 * np.log(b["/var(_cons[gid])"]),
    ]
    if disp == "nb":
        th.append(b["/lnalpha"])
    elif disp == "gamma":
        th.append(2.0 * b["/logs"])
    elif disp == "gauss":
        th.append(np.log(b[f"/var(e.{y})"]))
    return th


def _ologit_theta(x1, x2, cuts, var_cons):
    k = np.asarray(cuts, float)
    return [
        x1,
        x2,
        k[0],
        np.log(k[1] - k[0]),
        np.log(k[2] - k[1]),
        0.5 * np.log(var_cons),
    ]


# ---------------------------------------------------------------------------
# Fits (module-scoped: each is fitted once)
# ---------------------------------------------------------------------------

_FITS = {
    "pois_laplace": lambda: sp.mepoisson(
        D, "y_pois", ["x1", "x2"], "gid", offset="lexpo"
    ),
    "pois_mcagh7": lambda: sp.mepoisson(
        D, "y_pois", ["x1", "x2"], "gid", offset="lexpo", nAGQ=7
    ),
    "binom_laplace": lambda: sp.melogit(
        D, "y_succ", ["x1", "x2"], "gid", trials="n_trials"
    ),
    "bin_laplace": lambda: sp.melogit(D, "y_bin", ["x1", "x2"], "gid"),
    "nb_laplace": lambda: sp.menbreg(D, "y_nb", ["x1", "x2"], "gid"),
    "nb_mcagh7": lambda: sp.menbreg(D, "y_nb", ["x1", "x2"], "gid", nAGQ=7),
    "nb_expected": lambda: sp.menbreg(
        D, "y_nb", ["x1", "x2"], "gid", curvature="expected"
    ),
    "gamma_laplace": lambda: sp.megamma(D, "y_gam", ["x1", "x2"], "gid"),
    "gamma_mcagh7": lambda: sp.megamma(D, "y_gam", ["x1", "x2"], "gid", nAGQ=7),
    "gauss_meglm": lambda: sp.meglm(D, "y_gau", ["x1", "x2"], "gid"),
    "ologit_laplace": lambda: sp.meologit(D, "y_ord", ["x1", "x2"], "gid"),
    "ologit_mcagh7": lambda: sp.meologit(D, "y_ord", ["x1", "x2"], "gid", nAGQ=7),
}
_CACHE: dict = {}


def fit(spec):
    if spec not in _CACHE:
        _CACHE[spec] = _quiet(_FITS[spec])
    return _CACHE[spec]


_YVAR = {
    "pois_laplace": "y_pois",
    "pois_mcagh7": "y_pois",
    "binom_laplace": "y_succ",
    "bin_laplace": "y_bin",
    "nb_laplace": "y_nb",
    "nb_mcagh7": "y_nb",
    "gamma_laplace": "y_gam",
    "gamma_mcagh7": "y_gam",
    "gauss_meglm": "y_gau",
}


# ---------------------------------------------------------------------------
# 1. Objective identity at the reference's parameter vector
# ---------------------------------------------------------------------------

_STATA_OBJECTIVES = [
    ("pois_laplace", "y_pois", "poisson", 1, "lexpo", None, None),
    ("pois_mcagh7", "y_pois", "poisson", 7, "lexpo", None, None),
    ("binom_laplace", "y_succ", "binomial", 1, None, "n_trials", None),
    ("bin_laplace", "y_bin", "binomial", 1, None, None, None),
    ("nb_laplace", "y_nb", "nbinomial", 1, None, None, "nb"),
    ("nb_mcagh7", "y_nb", "nbinomial", 7, None, None, "nb"),
    ("gamma_laplace", "y_gam", "gamma", 1, None, None, "gamma"),
    ("gamma_mcagh7", "y_gam", "gamma", 7, None, None, "gamma"),
    ("gauss_meglm", "y_gau", "gaussian", 1, None, None, "gauss"),
]


@pytest.mark.parametrize(
    "spec,y,family,nagq,offset,trials,disp",
    _STATA_OBJECTIVES,
    ids=[s[0] for s in _STATA_OBJECTIVES],
)
def test_loglik_at_stata_estimates_equals_stata_ll(
    spec, y, family, nagq, offset, trials, disp
):
    ll = _glmm_ll(y, family, _stata_theta(spec, y, disp), nagq, offset, trials)
    _close(ll, S[spec]["ll"], rtol=LL_RTOL, msg=spec)


@pytest.mark.parametrize("spec,nagq", [("ologit_laplace", 1), ("ologit_mcagh7", 7)])
def test_ologit_loglik_at_stata_estimates_equals_stata_ll(spec, nagq):
    b = S[spec]["b"]
    th = _ologit_theta(
        b["y_ord:x1"],
        b["y_ord:x2"],
        [b["/cut1"], b["/cut2"], b["/cut3"]],
        b["/var(_cons[gid])"],
    )
    _close(_ologit_ll(th, nagq), S[spec]["ll"], rtol=LL_RTOL, msg=spec)


@pytest.mark.parametrize("k", [1, 7])
def test_ologit_loglik_at_clmm_estimates_equals_clmm_ll(k):
    r = R[f"ologit_clmm_agq{k}"]
    b = r["b"]
    th = _ologit_theta(b["x1"], b["x2"], [b["1|2"], b["2|3"], b["3|4"]], r["var_cons"])
    _close(_ologit_ll(th, k), r["ll"], rtol=LL_RTOL)


@pytest.mark.parametrize(
    "key,y,family",
    [("nb_laplace_tmb", "y_nb", "nbinomial"), ("gamma_laplace_tmb", "y_gam", "gamma")],
)
def test_loglik_at_glmmtmb_estimates_equals_glmmtmb_ll(key, y, family):
    r = R[key]
    # betadisp is log(1/alpha) for nbinom2 and log(1/phi) for Gamma.
    th = [r["b"]["(Intercept)"], r["b"]["x1"], r["b"]["x2"], r["theta"], -r["betadisp"]]
    _close(_glmm_ll(y, family, th), r["ll"], rtol=LL_RTOL)


def test_expected_curvature_loglik_at_glmer_nb_estimates_equals_lme4_ll():
    """lme4's Fisher-weight Laplace is ``curvature='expected'``: same function.

    Agreement is 1.6e-10 relative (1.6e-7 absolute), not the 1e-12 of the
    other identities: glmer.nb's reported logLik sits that far above the
    Fisher-weight Laplace at its own estimates.  The source of that offset
    was not located (open item in the campaign notes); with the observed
    curvature the same parameters give a value 0.1 lower, so the
    identification of the convention is unambiguous.
    """
    r = R["nb_laplace_expected_lme4"]
    th = [
        r["b"]["(Intercept)"],
        r["b"]["x1"],
        r["b"]["x2"],
        0.5 * np.log(r["var_cons"]),
        -np.log(r["theta"]),
    ]
    _close(_glmm_ll("y_nb", "nbinomial", th, curvature="expected"), r["ll"], rtol=1e-9)
    assert abs(_glmm_ll("y_nb", "nbinomial", th) - r["ll"]) > 1e-2


# ---------------------------------------------------------------------------
# 2. Estimates at the optimum
# ---------------------------------------------------------------------------

_STATA_CONVERGED = [
    "pois_laplace",
    "binom_laplace",
    "bin_laplace",
    "nb_laplace",
    "gamma_laplace",
    "gauss_meglm",
]


@pytest.mark.parametrize("spec", _STATA_CONVERGED)
def test_estimates_match_stata(spec):
    r, y, b = fit(spec), _YVAR[spec], S[spec]["b"]
    for nm in ("_cons", "x1", "x2"):
        _close(r.params[nm], b[f"{y}:{nm}"], msg=f"{spec} {nm}")
    _close(
        r.variance_components["var(_cons)"], b["/var(_cons[gid])"], msg=f"{spec} var"
    )
    _close(r.log_likelihood, S[spec]["ll"], rtol=LL_RTOL, msg=f"{spec} ll")
    if spec.startswith("nb"):
        _close(np.log(r.dispersion), b["/lnalpha"], msg="lnalpha")
    if spec.startswith("gamma"):
        _close(0.5 * np.log(r.dispersion), b["/logs"], msg="logs")
    if spec == "gauss_meglm":
        _close(r.variance_components["var(Residual)"], b["/var(e.y_gau)"])


@pytest.mark.parametrize("spec", ["pois_mcagh7", "nb_mcagh7", "gamma_mcagh7"])
def test_optimum_dominates_stata_short_stop(spec):
    """Stata's AGHQ estimate is not the maximiser of its own objective.

    The objective is the same function (test above, rtol 1e-11); our
    optimum attains a strictly larger value of it.  Stata's estimates sit
    2.9e-5 (Poisson _cons), 6.5e-6 (NB var(_cons)) and 9.5e-6 (gamma) from
    ours.  Poisson AGHQ is pinned against lme4 at 1e-6 below.
    """
    r = fit(spec)
    assert r.log_likelihood > S[spec]["ll"]
    assert r.log_likelihood - S[spec]["ll"] < 1e-6


def test_ologit_mcagh7_matches_stata():
    r, b = fit("ologit_mcagh7"), S["ologit_mcagh7"]["b"]
    _close(r.params["x1"], b["y_ord:x1"])
    _close(r.params["x2"], b["y_ord:x2"])
    for i, c in enumerate(("/cut1", "/cut2", "/cut3")):
        _close(r.thresholds.iloc[i], b[c], msg=c)
    _close(r.log_likelihood, S["ologit_mcagh7"]["ll"], rtol=1e-12)


def test_ologit_laplace_matches_stata():
    r, b = fit("ologit_laplace"), S["ologit_laplace"]["b"]
    # Stata's ologit Laplace stops at 1.2e-6 (x1: 3e-8, cut3: 4.7e-7); our
    # log-likelihood at the optimum is >= Stata's.
    _close(r.params["x1"], b["y_ord:x1"])
    _close(r.params["x2"], b["y_ord:x2"])
    for i, c in enumerate(("/cut1", "/cut2", "/cut3")):
        _close(r.thresholds.iloc[i], b[c], rtol=2e-6, msg=c)
    assert r.log_likelihood >= S["ologit_laplace"]["ll"] - 1e-9


@pytest.mark.parametrize("k", [1, 7])
def test_ologit_optimum_dominates_clmm(k):
    r = fit("ologit_laplace" if k == 1 else "ologit_mcagh7")
    assert r.log_likelihood >= R[f"ologit_clmm_agq{k}"]["ll"]


@pytest.mark.parametrize(
    "spec,key",
    [
        ("pois_laplace", "pois_laplace"),
        ("pois_mcagh7", "pois_agq7"),
        ("binom_laplace", "binom_laplace"),
        ("bin_laplace", "bin_laplace"),
        ("gauss_meglm", "gauss_lmer_ml"),
        ("nb_laplace", "nb_laplace_tmb"),
        ("gamma_laplace", "gamma_laplace_tmb"),
    ],
)
def test_estimates_match_r(spec, key):
    r, ref = fit(spec), R[key]
    for nm, rn in (("_cons", "(Intercept)"), ("x1", "x1"), ("x2", "x2")):
        # atol 1e-7 only matters for the binomial intercept (0.014): lme4
        # stops 4e-8 (5e-7 of its SE) away, with a log-likelihood 7e-11
        # below ours; Stata agrees with us there to 7e-9 relative.
        _close(r.params[nm], ref["b"][rn], atol=1e-7, msg=f"{spec} {nm}")
    _close(r.variance_components["var(_cons)"], ref["var_cons"], msg=f"{spec} var")


def test_expected_curvature_matches_glmer_nb():
    r, ref = fit("nb_expected"), R["nb_laplace_expected_lme4"]
    # glmer.nb alternates theta and glmer fits; the estimates agree to 7e-7
    # and our objective at our optimum is >= our objective at lme4's.
    for nm, rn in (("_cons", "(Intercept)"), ("x1", "x1"), ("x2", "x2")):
        _close(r.params[nm], ref["b"][rn], msg=nm)
    th = [
        ref["b"]["(Intercept)"],
        ref["b"]["x1"],
        ref["b"]["x2"],
        0.5 * np.log(ref["var_cons"]),
        -np.log(ref["theta"]),
    ]
    assert r.log_likelihood >= _glmm_ll("y_nb", "nbinomial", th, curvature="expected")


# ---------------------------------------------------------------------------
# 3. Standard errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec", ["pois_laplace", "binom_laplace", "bin_laplace", "gauss_meglm"]
)
def test_se_match_stata_canonical_links(spec):
    r, y, se = fit(spec), _YVAR[spec], S[spec]["se"]
    for nm in ("_cons", "x1", "x2"):
        _close(r.bse[nm], se[f"{y}:{nm}"], msg=f"{spec} {nm}")


@pytest.mark.parametrize(
    "spec,key",
    [
        ("nb_laplace", "nb_laplace_tmb"),
        ("gamma_laplace", "gamma_laplace_tmb"),
        ("pois_mcagh7", "pois_agq7"),
    ],
)
def test_se_match_exact_hessian_references(spec, key):
    r, ref = fit(spec), R[key]["se"]
    for nm, rn in (("_cons", "(Intercept)"), ("x1", "x1"), ("x2", "x2")):
        _close(r.bse[nm], ref[rn], msg=f"{spec} {nm}")


def test_ologit_mcagh7_se_match_stata():
    r, se = fit("ologit_mcagh7"), S["ologit_mcagh7"]["se"]
    # 1.5e-6 at the two optima (which differ by 3.8e-7); 5.7e-7 when our
    # Hessian is evaluated at Stata's own estimates.
    _close(r.bse["x1"], se["y_ord:x1"], rtol=2e-6)
    _close(r.bse["x2"], se["y_ord:x2"], rtol=2e-6)
    for i, c in enumerate(("/cut1", "/cut2", "/cut3")):
        _close(r.thresholds_se.iloc[i], se[c], rtol=2e-6, msg=c)


# ---------------------------------------------------------------------------
# 4. Reference-free identities
# ---------------------------------------------------------------------------


def test_gaussian_meglm_is_the_ml_linear_mixed_model():
    """Identity link => the Laplace approximation is exact => ML LMM."""
    g = fit("gauss_meglm")
    m = _quiet(sp.mixed, D, y="y_gau", x_fixed=["x1", "x2"], group="gid", method="ml")
    for nm in ("_cons", "x1", "x2"):
        _close(g.params[nm], m.fixed_effects[nm], rtol=1e-8)
    _close(
        g.variance_components["var(_cons)"],
        m.variance_components["var(_cons)"],
        rtol=1e-6,
    )
    _close(
        g.variance_components["var(Residual)"],
        m.variance_components["var(Residual)"],
        rtol=1e-6,
    )
    _close(g.log_likelihood, m.log_likelihood, rtol=1e-12)


def test_curvature_choice_is_immaterial_for_canonical_links():
    obs = fit("pois_laplace")
    exp = _quiet(
        sp.mepoisson,
        D,
        "y_pois",
        ["x1", "x2"],
        "gid",
        offset="lexpo",
        curvature="expected",
    )
    np.testing.assert_allclose(obs.params.values, exp.params.values, rtol=1e-10)
    _close(obs.log_likelihood, exp.log_likelihood, rtol=1e-14)


def test_aghq_converges_to_the_integral_for_non_canonical_links():
    """Observed and Fisher curvature are two node scalings of one integral:
    at 15 nodes their marginal log-likelihoods agree to 1e-9 relative even
    though the Laplace values differ by 6e-5."""
    th = _stata_theta("nb_mcagh7", "y_nb", "nb")
    a = _glmm_ll("y_nb", "nbinomial", th, 15)
    b = _glmm_ll("y_nb", "nbinomial", th, 15, curvature="expected")
    _close(a, b, rtol=1e-9)
    la = _glmm_ll("y_nb", "nbinomial", th, 1)
    lb = _glmm_ll("y_nb", "nbinomial", th, 1, curvature="expected")
    assert abs(la - lb) > 1e-3
