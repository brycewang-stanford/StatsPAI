"""Round-2 ``misc_sens`` family vs the R packages that define each method.

Reference: ``_fixtures/misc_sens_R.json`` written by
``_generate_misc_sens_R.R`` (versions recorded in the JSON) on the CSVs
written once by ``_fixtures/_generate_misc_sens_data.py``; the R generator
also writes ``_fixtures/misc_sens_mi_imputed.csv`` (five ``mice``
imputations at 17 significant digits), which every side pools.

Conventions every number below depends on (the generator's header carries
the full list):

* ``mi_estimate`` / Rubin's rules: ``mice::pool`` -- ``b`` with the m - 1
  divisor, Barnard-Rubin df with ``dfcom`` = the complete-data residual df
  (``lm``'s ``df.residual``; StatsPAI reads ``data_info['df_resid']``),
  ``fmi = (riv + 2/(df + 3))/(riv + 1)``. Only the pooling is compared: the
  imputation step is stochastic (T3 at best) and both sides pool R's own
  completed datasets.
* ``mediate_sensitivity``: ``mediation::medsens`` lm/lm branch. With
  ``eps = sqrt(machine eps)`` (medsens' default stopping rule) the FGLS
  iterates are the same on both sides; StatsPAI's default iterates to the
  fixed point, compared with ``medsens(eps = 1e-26)``. ``err.cr.d`` is a grid
  argmin; StatsPAI's ``rho_at_zero`` is the exact crossing
  ``corr(resid(M ~ T + X), resid(Y ~ T + X))``.

Tolerance: 1e-10 relative wherever both sides evaluate closed forms or the
same deterministic iteration; the few looser entries say why where they
occur.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.imputation.mice import MICEResult

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "misc_sens_R.json").read_text(encoding="utf-8"))

TIGHT = 1e-10


def _close(ours, ref, rtol=TIGHT, atol=0.0):
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float),
        np.asarray(ref, dtype=float),
        rtol=rtol,
        atol=atol,
    )


# --------------------------------------------------------------------------
# Multiple imputation: pooling (mice::pool)
# --------------------------------------------------------------------------


def _mi_result():
    long = pd.read_csv(_FIX / "misc_sens_mi_imputed.csv")
    ds = [
        long.loc[long["imp"] == i, ["y", "x1", "x2", "x3"]].reset_index(drop=True)
        for i in range(1, 6)
    ]
    return MICEResult(ds, 5, len(ds[0]), {}, [], {}, True)


@pytest.fixture(scope="module")
def pooled():
    return sp.mi_estimate(_mi_result(), sp.regress, formula="y ~ x1 + x2 + x3")


@pytest.mark.parametrize(
    "ours, ref",
    [
        ("params", "estimate"),
        ("ubar", "ubar"),
        ("b", "b"),
        ("t", "t"),
        ("se", "se"),
        ("tvalues", "statistic"),
        ("df", "df"),
        ("riv", "riv"),
        ("lambda", "lambda"),
        ("fmi", "fmi"),
        ("pvalues", "p"),
        ("ci_lower", "ci_lo"),
        ("ci_upper", "ci_hi"),
    ],
)
def test_mi_pooling_matches_mice_pool(pooled, ours, ref):
    assert pooled["var_names"] == ["Intercept", "x1", "x2", "x3"]
    # p-values and CI endpoints go through t quantiles at a fractional df;
    # scipy and R agree on those to ~1e-12.
    _close(pooled[ours], R["mi"][ref], rtol=1e-9)
    assert pooled["dfcom"] == R["mi"]["dfcom"][0]


def test_mi_large_sample_df_is_rubin_1987(pooled):
    from statspai.imputation.mice import _rubins_rules

    ests = [
        {"params": r.params.to_numpy(), "var_cov": r.data_info["var_cov"]}
        for r in (
            sp.regress("y ~ x1 + x2 + x3", data=_mi_result().complete(i))
            for i in range(5)
        )
    ]
    inf = _rubins_rules(ests)  # no df_resid -> dfcom = inf
    _close(inf["df"], R["mi_inf"]["df"])
    _close(inf["fmi"], R["mi_inf"]["fmi"])
    # Identity: nu_old = (m - 1) / lambda^2, and the small-sample df is the
    # harmonic combination with nu_obs.
    _close(inf["df"], 4 / inf["lambda"] ** 2, rtol=1e-12)
    dfcom = pooled["dfcom"]
    nu_obs = dfcom * (dfcom + 1) * (1 - pooled["lambda"]) / (dfcom + 3)
    _close(pooled["df"], 1 / (1 / inf["df"] + 1 / nu_obs), rtol=1e-12)


def test_mi_pooled_covariance_uses_full_within_covariance(pooled):
    # Off-diagonal of T = U-bar + (1 + 1/m) B uses each fit's full vcov.
    fits = [
        sp.regress("y ~ x1 + x2 + x3", data=_mi_result().complete(i)) for i in range(5)
    ]
    U = np.mean([f.data_info["var_cov"] for f in fits], axis=0)
    Q = np.array([f.params.to_numpy() for f in fits])
    B = np.cov(Q.T, ddof=1)
    _close(pooled["var_cov"], U + 1.2 * B, rtol=1e-12)


# --------------------------------------------------------------------------
# Mediation sensitivity (mediation::medsens)
# --------------------------------------------------------------------------

MED = pd.read_csv(_FIX / "misc_sens_med.csv")


def _medsens(**kw):
    return sp.mediate_sensitivity(
        MED, y="y", treat="t", mediator="m", covariates=["x1", "x2"], n_grid=19, **kw
    )


def test_medsens_default_stopping_rule_matches_medsens():
    r = _medsens(eps=float(np.sqrt(np.finfo(float).eps)))
    ref = R["medsens"]
    _close(r.rho_grid, ref["rho"], rtol=0, atol=1e-15)
    # Same iterates, same stopping step: agreement is machine-level (the
    # largest-|rho| points take ~37 FGLS steps; 1e-12 is conservative).
    _close(r.acme_at_rho, ref["d0"], rtol=1e-11)
    _close(r.ci_lower, ref["lower"], rtol=1e-10)
    _close(r.ci_upper, ref["upper"], rtol=1e-10)
    assert r.rho_grid_zero == pytest.approx(ref["err_cr"], abs=1e-15)
    _close(r.r2_mediator, ref["r2_m"], rtol=1e-12)
    _close(r.r2_outcome, ref["r2_y"], rtol=1e-12)
    # medsens' thresholds are err.cr.d^2 on the grid; StatsPAI reports the
    # exact crossing's square. The grid version is reconstructible.
    _close(r.rho_grid_zero**2, ref["R2star_thresh"], rtol=1e-12)
    _close(
        r.rho_grid_zero**2 * (1 - r.r2_mediator) * (1 - r.r2_outcome),
        ref["R2tilde_thresh"],
        rtol=1e-12,
    )


def test_medsens_fixed_point_matches_tightly_iterated_medsens():
    r = _medsens()
    ref = R["medsens_fixed_point"]
    _close(r.acme_at_rho, ref["d0"], rtol=1e-10)
    _close(r.ci_lower, ref["lower"], rtol=1e-10)
    _close(r.ci_upper, ref["upper"], rtol=1e-10)
    # medsens' default stopping rule is ~1e-4 short of that fixed point at
    # |rho| = 0.9; StatsPAI's default is the fixed point.
    gap = np.max(np.abs(np.array(R["medsens"]["d0"]) / np.array(ref["d0"]) - 1))
    assert 1e-5 < gap < 1e-3


def test_medsens_crossing_point_is_the_residual_correlation():
    r = _medsens()
    _close(r.rho_at_zero, R["medsens"]["rho_tilde"], rtol=1e-12)
    at = sp.mediate_sensitivity(
        MED,
        y="y",
        treat="t",
        mediator="m",
        covariates=["x1", "x2"],
        rho_range=(r.rho_at_zero, r.rho_at_zero),
        n_grid=1,
    )
    assert abs(at.acme_at_rho[0]) < 1e-12
    # At rho = 0 the SUR is equation-by-equation OLS: ACME = a_T * b_M.
    zero = int(np.argmin(np.abs(r.rho_grid)))
    _close(r.acme_at_rho[zero], r.acme_at_zero, rtol=1e-12)


# --------------------------------------------------------------------------
# OVB benchmark bounds (sensemakr) and the sensitivity dashboard
# --------------------------------------------------------------------------

OVB = pd.read_csv(_FIX / "misc_sens_ovb.csv")


def test_calibrate_confounding_strength_matches_sensemakr_ovb_bounds():
    ref = R["ovb"]
    r = sp.calibrate_confounding_strength(
        ref["estimate"],
        ref["se"],
        observed_r2_outcome=ref["r2yxj"],
        observed_r2_treatment=ref["r2dxj"],
        dof=ref["dof"],
        multipliers=ref["k"],
    )
    c = r.curve
    assert c["feasible"].all()
    _close(c["r2_treatment"], ref["r2dz"], rtol=1e-12)
    _close(c["r2_outcome"], ref["r2yz"], rtol=1e-12)
    _close(c["adjusted_estimate"], ref["adjusted_estimate"], rtol=1e-12)
    _close(c["adjusted_se"], ref["adjusted_se"], rtol=1e-12)
    # t quantile at dof = 395 (scipy vs qt): ~1e-12.
    _close(c["adjusted_ci_low"], ref["adjusted_lo"], rtol=1e-10)
    _close(c["adjusted_ci_high"], ref["adjusted_hi"], rtol=1e-10)


def test_calibrate_inputs_are_the_benchmark_partial_r2():
    # The benchmark partial R2 fed to calibrate are the regressions' own:
    # R2_{D~x2|x1,x3} and R2_{Y~x2|d,x1,x3}, from t^2 / (t^2 + dof).
    def partial_r2(formula, term):
        f = sp.regress(formula, data=OVB)
        t = float(f.params[term] / f.std_errors[term])
        return t**2 / (t**2 + f.data_info["df_resid"])

    _close(partial_r2("d ~ x1 + x2 + x3", "x2"), R["ovb"]["r2dxj"], rtol=1e-12)
    _close(partial_r2("y ~ d + x1 + x2 + x3", "x2"), R["ovb"]["r2yxj"], rtol=1e-12)


def test_unified_sensitivity_components_match_sensemakr_and_evalue():
    fit = sp.regress("y ~ d + x1 + x2 + x3", data=OVB)
    dash = sp.unified_sensitivity(
        fit, treat="d", data=OVB, y="y", controls=["x1", "x2", "x3"]
    )
    ev = R["evalue_ols"]
    _close(dash.e_value_point, ev["e_point"], rtol=1e-12)
    _close(dash.e_value_ci, ev["e_ci"], rtol=1e-12)
    _close(dash.rr_observed, ev["rr"], rtol=1e-12)
    _close(dash.ci_observed, [ev["rr_lo"], ev["rr_hi"]], rtol=1e-12)
    _close(dash.sensemakr["rv_q1"], R["ovb"]["rv_q"], rtol=1e-10)
    _close(dash.sensemakr["rv_qa"], R["ovb"]["rv_qa"], rtol=1e-10)
    # Oster: the dashboard reports what sp.oster_delta reports (graded vs
    # psacalc in round 1), with the same default R_max.
    od = sp.oster_delta(
        OVB, y="y", x_base=["d"], x_controls=["x1", "x2", "x3"], r_max=0, n_boot=0
    )
    _close(dash.oster["delta"], od.model_info["delta_star"], rtol=1e-12)


def test_survival_sensitivity_breakpoint_is_the_evalue_bias_factor():
    ref = R["evalue_hr"]
    grid = np.linspace(1.0, 1.5, 50001)
    for sign in (1, -1):
        r = sp.survival_sensitivity(sign * ref["log_hr"], ref["se"], gamma_grid=grid)
        # smallest grid Gamma whose worst-case CI reaches the null
        _close(r.breakpoint, ref["hr_lo"], rtol=0, atol=1e-5)
    b = ref["hr_lo"]
    _close(b + np.sqrt(b * (b - 1)), ref["e_ci"], rtol=1e-12)


# --------------------------------------------------------------------------
# Mendelian randomisation leftovers (MendelianRandomization's LDL-C data,
# the round-1 file mr_ldl.csv)
# --------------------------------------------------------------------------

MR = pd.read_csv(_FIX / "mr_ldl.csv")
BX, BY, SX, SY = (MR[c].to_numpy() for c in ("bx", "by", "bxse", "byse"))


@pytest.mark.parametrize("prior", ["0.5", "0.1"])
def test_mr_bma_matches_zuber_summary_mvmr_bf(prior):
    df = pd.DataFrame(
        {"ldl": BX, "hdl": MR["hdl"], "tg": MR["tg"], "beta_y": BY, "se_y": SY}
    )
    r = sp.mr_bma(df, exposures=["ldl", "hdl", "tg"], prior_inclusion=float(prior))
    ref = R[f"mrbma_{prior}"]
    # Zuber's enumeration order ("1", "2", "3", "1,2", ...) is the
    # combinations() order used here.
    assert ref["tupel"] == ["1", "2", "3", "1,2", "1,3", "2,3", "1,2,3"]
    _close(r.model_priors, ref["pp"], rtol=1e-12)
    _close(r.marginal_inclusion.to_numpy(), ref["pp_marginal"], rtol=1e-12)
    _close(r.model_averaged_estimate.to_numpy(), ref["bma"], rtol=1e-12)


def test_mr_bma_bic_path_is_the_old_quantity():
    df = pd.DataFrame(
        {"ldl": BX, "hdl": MR["hdl"], "tg": MR["tg"], "beta_y": BY, "se_y": SY}
    )
    bf = sp.mr_bma(df, exposures=["ldl", "hdl", "tg"])
    bic = sp.mr_bma(df, exposures=["ldl", "hdl", "tg"], method="bic")
    assert "bic" in bic.best_models.columns and "log10_bf" in bf.best_models.columns
    assert not np.allclose(bf.model_priors, bic.model_priors)


def test_mr_mediation_is_mr_ivw_total_and_mr_mvivw_direct():
    x = pd.DataFrame(
        {
            "beta_x": BX,
            "se_x": SX,
            "beta_m": MR["hdl"],
            "se_m": MR["hdlse"],
            "beta_y": BY,
            "se_y": SY,
        }
    )
    r = sp.mr_mediation(x)
    ref = R["mr_mediation"]
    _close(r.total_effect, ref["total"], rtol=1e-12)
    _close(r.total_effect_se, ref["total_se"], rtol=1e-12)
    _close(r.direct_effect, ref["direct"], rtol=1e-12)
    _close(r.direct_effect_se, ref["direct_se"], rtol=1e-12)
    # identity: indirect is the difference of the two
    assert r.indirect_effect == r.total_effect - r.direct_effect


def test_mr_multivariable_floors_the_residual_standard_error_at_one():
    ref = R["mvivw_underdispersed"]
    assert ref["rse"] < 1
    df = pd.DataFrame(
        {"ldl": BX, "hdl": MR["hdl"], "tg": MR["tg"], "beta_y": BY, "se_y": 3 * SY}
    )
    r = sp.mr_multivariable(df, exposures=["ldl", "hdl", "tg"])
    _close(r.direct_effect["estimate"], ref["est"], rtol=1e-12)
    _close(r.direct_effect["se"], ref["se"], rtol=1e-12)


def test_mendelian_randomization_matches_mr_allmethods():
    ref = R["mr_allmethods"]
    m = dict(zip(ref["method"], range(len(ref["method"]))))
    res = sp.mendelian_randomization(
        MR,
        "bx",
        "by",
        "bxse",
        "byse",
        methods=["ivw", "egger", "weighted_median"],
        seed=1,
    )
    t = res.estimates.set_index("method")
    for ours, theirs in (("IVW", "IVW"), ("MR-Egger", "MR-Egger")):
        _close(t.loc[ours, "estimate"], ref["est"][m[theirs]], rtol=1e-12)
        _close(t.loc[ours, "se"], ref["se"][m[theirs]], rtol=1e-12)
    # Weighted median: deterministic estimate; bootstrap SE is Monte Carlo
    # on both sides (not compared).
    _close(
        t.loc["Weighted Median", "estimate"],
        ref["est"][m["Weighted median"]],
        rtol=1e-12,
    )
    _close(res.pleiotropy["intercept"], ref["est"][m["(intercept)"]], rtol=1e-12)


@pytest.mark.parametrize("loss", ["tukey", "huber", "l2"])
def test_grapple_sandwich_at_grapples_own_estimates(loss):
    """The variance formula, evaluated at GRAPPLE's (beta, tau2), gives
    GRAPPLE's SEs: exact for l2, ~1e-7 for tukey / huber, whose moments R
    obtains by integrate() (StatsPAI by quad)."""
    from statspai.mendelian.frontier.grapple import _grapple_vcov, _l2_rho
    from statspai.mendelian.frontier.raps import _gauss_moment, _rho

    rho = (
        _l2_rho if loss == "l2" else _rho(loss, {"tukey": 4.685, "huber": 1.345}[loss])
    )
    d = _gauss_moment(lambda x: rho(x))
    c1 = _gauss_moment(lambda x: rho(x, deriv=1) ** 2)
    c2 = _gauss_moment(lambda x: rho(x) ** 2) - d**2
    c4 = _gauss_moment(lambda x: rho(x, deriv=1) * x)
    ref = R["grapple"][loss]
    V = _grapple_vcov(
        ref["beta"], ref["tau2"], BX, BY, SX**2, SY**2, rho, c1, c2, c4, c4
    )
    tol = 1e-12 if loss == "l2" else 1e-6
    _close(np.sqrt(V[0, 0]), ref["se"], rtol=tol)
    _close(np.sqrt(V[1, 1]), ref["tau2_se"], rtol=max(tol, 1e-10))


@pytest.mark.parametrize("loss", ["tukey", "huber", "l2"])
def test_grapple_fit_is_aligned_and_solves_the_equations(loss):
    """Fitted values agree with grappleRobustEst to 1e-3 (beta) / 1e-4
    (tau2, SE): GRAPPLE stops optim at its default tolerance and its tau2
    root at bound * eps^0.25. StatsPAI solves both estimating equations to
    machine precision; at GRAPPLE's point the beta score is still O(1e-3).

    Tukey SEs are a documented divergence (docs/dev/cleanroom/grapple.md):
    GRAPPLE evaluates Var[rho(Z)] with the biweight on its max-1 scale but
    rho' on the paper's c^2/6 scale, which shrinks the tau2 SE by exactly
    6/c^2 and moves the beta SE by ~1e-3 here. StatsPAI keeps one scale; the
    same sandwich with GRAPPLE's mixed scale reproduces its numbers."""
    from statspai.mendelian.frontier.grapple import (
        _grapple_vcov,
        _l2_rho,
        _loss_moments,
        _robust_rho,
    )
    from statspai.mendelian.frontier.raps import _rho

    g = sp.grapple(BX, BY, SX, SY, loss=loss)
    ref = R["grapple"][loss]
    assert g.converged
    _close(g.estimate, ref["beta"], rtol=1e-3)
    _close(g.tau2, ref["tau2"], rtol=1e-4)
    if loss == "tukey":
        k = 4.685
        _close(g.tau2_se * 6 / k**2, ref["tau2_se"], rtol=1e-4)
        rho_t = _robust_rho("tukey", k)
        mom = _loss_moments("tukey", rho_t, k)
        V = _grapple_vcov(
            g.estimate,
            g.tau2,
            BX,
            BY,
            SX**2,
            SY**2,
            rho_t,
            mom["score_sq"],
            mom["rho_var"] * (6 / k**2) ** 2,
            mom["slope"],
            mom["slope"],
        )
        _close(np.sqrt(V[0, 0]), ref["se"], rtol=1e-4)
    else:
        _close(g.se, ref["se"], rtol=1e-4)
        _close(g.tau2_se, ref["tau2_se"], rtol=1e-4)
    rho = (
        _l2_rho if loss == "l2" else _rho(loss, {"tukey": 4.685, "huber": 1.345}[loss])
    )

    def score(b, t2):
        res = BY - BX * b
        v = SX**2 * b**2 + SY**2 + t2
        return np.sum(
            rho(res / np.sqrt(v), deriv=1) * (v * BX + res * SX**2 * b) / v**1.5
        )

    assert abs(score(g.estimate, g.tau2)) < 1e-12
    assert abs(score(ref["beta"], ref["tau2"])) > 1e-4


# --------------------------------------------------------------------------
# Transport / evidence synthesis
# --------------------------------------------------------------------------

SRC = pd.read_csv(_FIX / "misc_sens_src.csv")
TGT = pd.read_csv(_FIX / "misc_sens_tgt.csv")


@pytest.mark.parametrize("fn", ["transport_weights_fn", "transport_generalize"])
def test_transport_iosw_matches_glm_weights_and_hc0(fn):
    """Inverse-odds weights from a logit of source membership (glm), clipped
    at their 1% / 99% type-7 quantiles; effect = WLS coefficient, SE = HC0
    sandwich of that coefficient (weights held fixed)."""
    f = getattr(sp, fn)
    kw = dict(features=["x1", "x2"], treatment="a", outcome="y")
    r = f(SRC, TGT, **kw)
    ref = R["transport"]
    _close(r.effect_transported, ref["effect"], rtol=1e-12)
    _close(r.se_transported, ref["se_hc0"], rtol=1e-12)
    _close(r.ess, ref["ess"], rtol=1e-12)
    _close(r.max_weight, ref["max_w"], rtol=1e-12)
    _close(r.effect_source, ref["source_effect"], rtol=1e-12)


def test_transport_se_is_invariant_to_rescaling_the_weights():
    # Known-truth property the pre-1.30 SE (w instead of w^2) violated.
    r = sp.transport_weights_fn(
        SRC, TGT, features=["x1", "x2"], treatment="a", outcome="y"
    )
    w, a, y = r.weights, SRC["a"].to_numpy(), SRC["y"].to_numpy()

    def se(ww):
        v = 0.0
        for m in (a, 1 - a):
            wm = ww * m
            mu = (wm * y).sum() / wm.sum()
            v += (wm**2 * (y - mu) ** 2).sum() / wm.sum() ** 2
        return np.sqrt(v)

    _close(se(w), r.se_transported, rtol=1e-12)
    _close(se(7.3 * w), r.se_transported, rtol=1e-12)


def test_heterogeneity_of_effect_matches_metafor_dl():
    m = R["meta"]
    h = sp.heterogeneity_of_effect(m["yi"], m["si"])
    assert m["tau2"] > 0
    _close(
        [h.tau2, h.q_stat, h.q_pvalue, h.i2],
        [m["tau2"], m["Q"], m["Qp"], m["I2"]],
        rtol=1e-12,
    )


def test_synthesise_evidence_inverse_variance_is_metafor_fixed_effect():
    m = R["meta"]
    e = sp.synthesise_evidence(
        rct_estimate=0.50,
        rct_se=0.20,
        rwd_estimate=0.42,
        rwd_se=0.10,
        transport_shift=-0.05,
        transport_shift_se=0.02,
    )
    _close(e.pooled_estimate, m["fe_est"], rtol=1e-12)
    _close(e.pooled_se, m["fe_se"], rtol=1e-12)
    _close(e.pooled_ci, [m["fe_lo"], m["fe_hi"]], rtol=1e-12)


def test_identify_transport_against_causaleffect():
    """sp.identify_transport searches s-admissible pre-treatment sets for a
    covariates-only target; causaleffect::transport is the complete
    algorithm with target observational data (P*). They agree on the
    adjustment set where both apply (a, d) and differ, as documented, where
    only target outcome data identify the effect (b, c)."""
    ce = R["transport_id"]
    a = sp.identify_transport(
        sp.dag("X -> Y; W -> Y; W -> X; S -> W"),
        treatment="X",
        outcome="Y",
        selection_nodes="S",
    )
    assert a.transportable and sorted(a.admissible_set) == ["W"]
    assert ce["a"] == "\\sum_{W}P^*(Y|W,X)P^*(W)"
    d = sp.identify_transport(
        sp.dag("X -> Y; S -> X"), treatment="X", outcome="Y", selection_nodes="S"
    )
    assert d.transportable and not d.admissible_set
    for g, key in (("X -> Y; S -> Y", "b"), ("X -> Z; Z -> Y; S -> Z", "c")):
        r = sp.identify_transport(
            sp.dag(g), treatment="X", outcome="Y", selection_nodes="S"
        )
        assert not r.transportable
        assert "P^*(Y|" in ce[key]  # needs target outcome data
