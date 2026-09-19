"""Treatment-effects family vs the R / Stata implementations that define it.

References (identical bytes: ``_fixtures/teffects_{cs,wide,long,surv}.csv``,
written once by ``_fixtures/_generate_teffects_data.py``):

* ``_fixtures/teffects_R.json`` from ``_generate_teffects_R.R`` -- AIPW
  0.6.9.3, tmle 2.1.1, ipw 1.3.0 (+ geepack, sandwich), ltmle 1.3-0,
  DTRreg 2.4, CMAverse 0.1.0, geex 1.1.1, survival, AER; versions are
  recorded in the JSON.
* ``_fixtures/teffects_Stata.json`` from ``_fixtures/_generate_teffects_stata.do``
  -- Stata 18 ``teffects aipw``, ``leebounds`` (SSC, v1.5), ``tebounds``
  (SJ15-2 st0386), ``med4way`` (GitHub), ``doseresponse`` (SSC), and
  ``regress`` / ``logit`` with probability weights for the MSM stage.

Conventions every number below depends on (the generators' headers carry
the full list):

* AIPW / teffects: full-sample nuisance fits (``cross_fit=False``). R
  ``AIPW`` reports ``sd(EIF)/sqrt(n)`` (``se_method='influence'``); Stata's
  robust SE is the stacked M-estimation sandwich (``'sandwich'``, divisor
  ``n``). Both are pinned; they differ by the nuisance-estimation terms
  (0.26% here). R ``AIPW``'s ATT divides its control term by P(A=0) where
  the DR ATT divides by P(A=1); the test reconstructs R's number from that
  formula instead of copying it.
* tmle: ``fluctuation='per_arm'`` (``tmle::tmle``'s two clever covariates),
  ``q_bound=5e-4`` (``tmle``'s ``alpha = 0.9995``), glm learners with tight
  convergence, ``tmle(cvQinit = FALSE)`` on the R side.
* MSM: ``ipwtm(type = "all", trunc = 0.01)``; Gaussian densities use the ML
  residual SD (``density_sd='ml'``, geeglm's dispersion). Outcome stage
  SEs are CR1 (``G/(G-1) (N-1)/(N-k)``), which is ``vcovCL(type="HC1")``
  and Stata ``regress``; Stata ``logit`` omits ``(N-1)/(N-k)``.
* ltmle: glm learners, ``gbounds = c(0.01, 1)``, ``variance.method = "ic"``.
* Lee bounds: ``trimming='quantile'`` is Lee's sample-quantile rule;
  ``leebounds`` as shipped reproduces it on the side where its rounded
  threshold falls below the quantile observation, and drops that
  observation on the other side (block 4 of the do-file shows the
  mechanism; ``trimming='exact'`` reproduces the patched run).

Tolerances: 1e-10 relative wherever both sides run closed forms or
Newton-converged MLEs; 1e-8 where a sklearn ``lbfgs`` logistic fit sits in
the StatsPAI path (``tmle`` with learner libraries) or Stata's default
logit tolerance sits in the reference path (MSM weights).
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
R = json.loads((_FIX / "teffects_R.json").read_text(encoding="utf-8"))
ST = json.loads((_FIX / "teffects_Stata.json").read_text(encoding="utf-8"))
CS = pd.read_csv(_FIX / "teffects_cs.csv")
WIDE = pd.read_csv(_FIX / "teffects_wide.csv")
LONG = pd.read_csv(_FIX / "teffects_long.csv")
SURV = pd.read_csv(_FIX / "teffects_surv.csv")

TIGHT = 1e-10


def _close(ours, ref, rtol=TIGHT, atol=0.0):
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float),
        np.asarray(ref, dtype=float),
        rtol=rtol,
        atol=atol,
    )


def _nuisance(df, y="y", d="d", covs=("x1", "x2", "x3")):
    """Full-sample logit propensity and per-arm OLS, reference-free."""
    import statsmodels.api as sm

    X = sm.add_constant(df[list(covs)].to_numpy(float))
    D = df[d].to_numpy(float)
    Y = df[y].to_numpy(float)
    e = sm.Logit(D, X).fit(disp=0, tol=1e-12).predict(X)
    mu1 = X @ np.linalg.lstsq(X[D == 1], Y[D == 1], rcond=None)[0]
    mu0 = X @ np.linalg.lstsq(X[D == 0], Y[D == 0], rcond=None)[0]
    return Y, D, e, mu1, mu0


# --------------------------------------------------------------------------
# AIPW: Stata teffects aipw and R AIPW
# --------------------------------------------------------------------------

AIPW_KW = dict(y="y", treat="d", covariates=["x1", "x2", "x3"], cross_fit=False)


def test_aipw_matches_teffects_aipw_point_and_sandwich_se():
    r = sp.aipw(CS, se_method="sandwich", **AIPW_KW)
    ref = ST["aipw"]
    _close(r.estimate, ref["ate"])
    _close(r.se, ref["ate_se"])
    po, po_se = (
        r.model_info["potential_outcome_means"],
        r.model_info["potential_outcome_means_se"],
    )
    _close([po[0], po[1]], [ref["po0"], ref["po1"]])
    _close([po_se[0], po_se[1]], [ref["po0_se"], ref["po1_se"]])


def test_aipw_matches_r_aipw_influence_se():
    r = sp.aipw(CS, **AIPW_KW)
    ref = R["aipw"]
    _close(r.estimate, ref["ate"])
    _close(r.se, ref["ate_se"])
    po, po_se = (
        r.model_info["potential_outcome_means"],
        r.model_info["potential_outcome_means_se"],
    )
    _close([po[1], po[0]], [ref["po1"], ref["po0"]])
    _close([po_se[1], po_se[0]], [ref["po1_se"], ref["po0_se"]])


def test_aipw_two_se_conventions_differ_by_nuisance_terms_only():
    """Same point estimate; the sandwich adds the nuisance-estimation terms."""
    a = sp.aipw(CS, **AIPW_KW)
    b = sp.aipw(CS, se_method="sandwich", **AIPW_KW)
    assert a.estimate == b.estimate
    assert 0.0 < abs(a.se / b.se - 1) < 0.01


def test_aipw_att_identity_and_r_aipw_normalisation():
    """sp.aipw ATT is the DR ATT; R AIPW divides its control term by P(A=0).

    The DR ATT is ``mean(D (Y - mu0) - (1-D) e/(1-e) (Y - mu0)) / P(D=1)``
    and, being a ratio of means, has influence function
    ``(N_i - tau D_i) / p``. R ``AIPW`` (0.6.9.3, ``get_ATT_RD``) uses
    ``(1-D)/P(D=0)`` for the control term, so its estimate is a different
    number; rebuilding it from the same nuisances reproduces R to 1e-10,
    which locates the whole gap in that one normalisation.
    """
    Y, D, e, mu1, mu0 = _nuisance(CS)
    p = D.mean()
    psi = D * (Y - mu0) / p - (1 - D) * e / (1 - e) * (Y - mu0) / p
    tau = psi.mean()
    r = sp.aipw(CS, estimand="ATT", **AIPW_KW)
    _close(r.estimate, tau)
    _close(r.se, np.std(psi - tau * D / p, ddof=1) / np.sqrt(len(Y)))
    # R AIPW's formula, from the same nuisances.
    r_eif = D / p * Y - (D / p * mu0 + (1 - D) / (1 - p) * (Y - mu0) * e / (1 - e))
    _close(r_eif.mean(), R["aipw"]["att"])
    _close(
        np.std(r_eif - D / p * r_eif.mean(), ddof=1) / np.sqrt(len(Y)),
        R["aipw"]["att_se"],
    )
    assert abs(r.estimate - R["aipw"]["att"]) > 1e-4


def test_aipw_att_matches_doubleml_external_predictions():
    """Second reference for the ATT: DoubleML's ATTE score on the same
    full-sample nuisances (DoubleML's variance divides by n, ours by n-1)."""
    dml = pytest.importorskip("doubleml")
    from sklearn.linear_model import LinearRegression, LogisticRegression

    Y, D, e, mu1, mu0 = _nuisance(CS)
    data = dml.DoubleMLData(
        CS[["y", "d", "x1", "x2", "x3"]], "y", "d", ["x1", "x2", "x3"]
    )
    m = dml.DoubleMLIRM(
        data, LinearRegression(), LogisticRegression(), score="ATTE", n_folds=2
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(
            external_predictions={
                "d": {"ml_g0": mu0[:, None], "ml_g1": mu1[:, None], "ml_m": e[:, None]}
            }
        )
    r = sp.aipw(CS, estimand="ATT", **AIPW_KW)
    n = len(Y)
    _close(r.estimate, m.coef[0])
    _close(r.se * np.sqrt((n - 1) / n), m.se[0])


def test_aipw_rejects_sandwich_with_cross_fitting():
    with pytest.raises(ValueError, match="cross_fit=False"):
        sp.aipw(CS, y="y", treat="d", covariates=["x1"], se_method="sandwich")


# --------------------------------------------------------------------------
# Multi-valued AIPW: Stata teffects aipw with an mlogit propensity
# --------------------------------------------------------------------------


def test_multi_treatment_matches_teffects_aipw_multivalued():
    r = sp.multi_treatment(
        CS,
        y="ymt",
        treat="t3",
        covariates=["x1", "x2"],
        outcome_model="linear",
        se_method="sandwich",
    )
    ref = ST["aipw_multi"]
    det = r.detail.set_index("treatment")
    _close([det.loc[1, "estimate"], det.loc[2, "estimate"]], [ref["ate1"], ref["ate2"]])
    _close([det.loc[1, "se"], det.loc[2, "se"]], [ref["ate1_se"], ref["ate2_se"]])
    po = r.model_info["potential_outcomes"]
    _close([po[0], po[1], po[2]], ref["po"])
    _close([r.model_info["potential_outcomes_se"][k] for k in (0, 1, 2)], ref["po_se"])


def test_multi_treatment_two_arms_equals_binary_aipw():
    """Identity: with two arms the multinomial logit is the logit and the
    Cattaneo estimator is sp.aipw's full-sample AIPW (point and sandwich)."""
    r = sp.multi_treatment(
        CS,
        y="y",
        treat="d",
        covariates=["x1", "x2", "x3"],
        outcome_model="linear",
        se_method="sandwich",
    )
    a = sp.aipw(CS, se_method="sandwich", **AIPW_KW)
    _close(r.estimate, a.estimate)
    _close(r.se, a.se)


# --------------------------------------------------------------------------
# MSM and stabilized weights: R ipw::ipwtm, Stata pweighted regressions
# --------------------------------------------------------------------------

MSM_KW = dict(treat="a", id="id", time="t", time_varying=["l"], baseline=["v"])


def test_stabilized_weights_match_ipwtm_binomial():
    sw = sp.stabilized_weights(LONG, **MSM_KW)
    _close(sw, R["msm"]["sw_binary"])


def test_stabilized_weights_match_ipwtm_gaussian_with_ml_sd():
    kw = dict(MSM_KW, treat="ac")
    _close(sp.stabilized_weights(LONG, density_sd="ml", **kw), R["msm"]["sw_gaussian"])
    # The default divides the residual sum of squares by N - k instead.
    assert not np.allclose(
        sp.stabilized_weights(LONG, **kw), R["msm"]["sw_gaussian"], rtol=1e-6
    )


@pytest.mark.parametrize(
    "exposure,family,key",
    [
        ("cumulative", "gaussian", "cumulative"),
        ("ever", "gaussian", "ever"),
        ("current", "gaussian", "current"),
        ("cumulative", "binomial", "cumulative_binomial"),
    ],
)
def test_msm_matches_ipwtm_plus_vcovcl(exposure, family, key):
    y = "y" if family == "gaussian" else "yb"
    r = sp.msm(LONG, y=y, exposure=exposure, family=family, **MSM_KW)
    ref = R["msm"][key]
    _close(r.detail["estimate"], ref["coef"], rtol=1e-9)
    _close(r.detail["se"], ref["se"], rtol=1e-9)


def test_msm_matches_stata_pweighted_regress_and_logit():
    """Untruncated weights from Stata's own logits (default tolerance).

    ``regress, vce(cluster)`` uses the same CR1 factor as sp.msm; Stata's
    ``logit`` uses ``G/(G-1)`` alone, so sp.msm's binomial SE is Stata's
    times ``sqrt((N-1)/(N-k))`` -- a documented small-sample convention.
    """
    ref = ST["msm"]
    g = sp.msm(LONG, y="y", trim=0, **MSM_KW)
    _close(g.detail["estimate"], ref["coef"], rtol=1e-7)
    _close(g.detail["se"], ref["se"], rtol=1e-7)
    b = sp.msm(LONG, y="yb", trim=0, family="binomial", **MSM_KW)
    _close(b.detail["estimate"], ref["logit_coef"], rtol=1e-7)
    N, k = ref["N"], 3
    _close(
        b.detail["se"],
        np.asarray(ref["logit_se"]) * np.sqrt((N - 1) / (N - k)),
        rtol=1e-7,
    )


# --------------------------------------------------------------------------
# TMLE end to end with glm learners: tmle::tmle
# --------------------------------------------------------------------------


def _tmle(y, binary):
    from sklearn.linear_model import LinearRegression, LogisticRegression

    def lr():
        return LogisticRegression(penalty=None, tol=1e-12, max_iter=100000)

    return sp.tmle(
        CS,
        y=y,
        treat="d",
        covariates=["x1", "x2", "x3"],
        outcome_library=[lr() if binary else LinearRegression()],
        propensity_library=[lr()],
        fluctuation="per_arm",
        q_bound=5e-4,
    )


@pytest.mark.parametrize(
    "y,binary,key", [("yb", True, "tmle_binary"), ("y", False, "tmle_gaussian")]
)
def test_tmle_glm_learners_match_tmle_package(y, binary, key):
    r = _tmle(y, binary)
    _close(r.estimate, R[key]["psi"], rtol=1e-8)
    _close(r.se, R[key]["se"], rtol=1e-8)


def test_tmle_q_bound_is_the_only_gap_on_the_continuous_path():
    """With the default q_bound=1e-5 the continuous estimate moves by ~3e-6:
    tmle::tmle truncates initial Q at 5e-4 and some linear predictions fall
    outside the observed outcome range."""
    from sklearn.linear_model import LinearRegression, LogisticRegression

    lr = LogisticRegression(penalty=None, tol=1e-12, max_iter=100000)
    r = sp.tmle(
        CS,
        y="y",
        treat="d",
        covariates=["x1", "x2", "x3"],
        outcome_library=[LinearRegression()],
        propensity_library=[lr],
        fluctuation="per_arm",
    )
    gap = abs(r.estimate / R["tmle_gaussian"]["psi"] - 1)
    assert 1e-7 < gap < 1e-4


# --------------------------------------------------------------------------
# Longitudinal TMLE: R ltmle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("y,key", [("Yb", "ltmle_binary"), ("Y", "ltmle_gaussian")])
def test_ltmle_matches_r_ltmle(y, key):
    r = sp.ltmle(
        WIDE,
        y=y,
        treatments=["A0", "A1"],
        covariates_time=[["L0"], ["L1"]],
        baseline=["v"],
    )
    ref = R[key]
    _close(
        [r.psi_treated, r.psi_control, r.ate],
        [ref["psi1"], ref["psi0"], ref["ate"]],
        rtol=1e-9,
    )
    _close(r.se, ref["ate_se"], rtol=1e-9)


@pytest.mark.parametrize(
    "y,key", [("Yb", "ltmle_cens_binary"), ("Y", "ltmle_cens_gaussian")]
)
def test_ltmle_with_censoring_matches_r_ltmle(y, key):
    """Censoring after each treatment (C nodes, 1 = uncensored; later nodes
    missing). R's Q / g glms stop at glm's default deviance tolerance, so
    this row is pinned at 1e-8 (observed ~5e-10)."""
    wc = pd.read_csv(_FIX / "teffects_wide_cens.csv")
    r = sp.ltmle(
        wc,
        y=y,
        treatments=["A0", "A1"],
        covariates_time=[["L0"], ["L1"]],
        baseline=["v"],
        censoring=["C0", "C1"],
    )
    ref = R[key]
    _close(
        [r.psi_treated, r.psi_control, r.ate],
        [ref["psi1"], ref["psi0"], ref["ate"]],
        rtol=1e-8,
    )
    _close(r.se, ref["ate_se"], rtol=1e-8)


# --------------------------------------------------------------------------
# ICE g-formula: hand lm() + geex sandwich; MC g-formula identity
# --------------------------------------------------------------------------

ICE_KW = dict(
    id_col="id",
    time_col="id",
    treatment_cols=["A0", "A1"],
    confounder_cols=[["v", "L0"], ["L1"]],
    outcome_col="Y",
)


def test_ice_point_and_sandwich_se_match_r():
    r = sp.gformula_ice_fn(WIDE, treatment_strategy=[1, 1], **ICE_KW)
    _close(r.value, R["ice"]["psi"])
    _close(R["ice"]["psi_geex"], R["ice"]["psi"])
    _close(r.se, R["ice"]["se"], rtol=1e-9)


def test_ice_se_is_not_the_outcome_mean_se():
    """Regression: bootstrap=0 used to report sd(Y)/sqrt(n)."""
    r = sp.gformula_ice_fn(WIDE, treatment_strategy=[1, 1], **ICE_KW)
    naive = WIDE["Y"].std(ddof=1) / np.sqrt(len(WIDE))
    assert abs(r.se / naive - 1) > 0.05


def test_mc_gformula_converges_to_ice_with_linear_models():
    """T3 identity: with linear models on nested histories and L0 drawn from
    its empirical law, the MC g-formula mean is the ICE value up to Monte
    Carlo error (sd of the simulated predictions / sqrt(n_sim))."""
    kw = dict(
        treatment_cols=["A0", "A1"], confounder_cols=[["L0"], ["L1"]], outcome_col="Y"
    )
    ice = sp.gformula_ice_fn(
        WIDE, id_col="id", time_col="id", treatment_strategy=[1, 1], **kw
    )
    mc = sp.gformula_mc(
        WIDE,
        strategy=(1, 1),
        n_simulations=400000,
        bootstrap=0,
        return_trajectories=True,
        seed=11,
        **kw,
    )
    mc_err = float(np.std(mc.trajectories["Y_pred"])) / np.sqrt(400000)
    assert abs(mc.value - ice.value) < 4 * mc_err


# --------------------------------------------------------------------------
# G-estimation: R DTRreg
# --------------------------------------------------------------------------

GEST_KW = dict(
    y="Y",
    treatments=["A0", "A1"],
    covariates_by_stage=[["L0", "v"], ["L0", "v", "A0", "L1"]],
    n_bootstrap=2,
)


def test_g_estimation_matches_dtrreg_gest():
    r = sp.g_estimation(WIDE, **GEST_KW)
    _close(r.model_info["psi_estimates"], R["g_estimation"]["psi_logit"])


def test_g_estimation_uses_propensity_covariates():
    """Regression: propensity_covariates used to be silently ignored."""
    r = sp.g_estimation(WIDE, propensity_covariates=[["L0"], ["L1", "A0"]], **GEST_KW)
    _close(r.model_info["psi_estimates"], R["g_estimation"]["psi_logit_pcov"])


def test_g_estimation_linear_propensity_is_the_ols_coefficient():
    """Identity: with a linear-probability propensity on the stage
    covariates the g-estimate is the OLS coefficient on A_k (the pre-1.29
    behaviour)."""
    r = sp.g_estimation(WIDE, propensity_model="linear", **GEST_KW)
    X = np.column_stack([np.ones(len(WIDE)), WIDE[["L0", "v", "A0", "L1", "A1"]]])
    b = np.linalg.lstsq(X, WIDE["Y"], rcond=None)[0]
    _close(r.model_info["psi_estimates"][1], b[-1])


# --------------------------------------------------------------------------
# Lee bounds / SACE: Stata leebounds
# --------------------------------------------------------------------------


def _stata_rounded_lower():
    """Stata leebounds' shipped lower bound, rebuilt from the data.

    ``local lth = r(r1)`` keeps the 15-significant-digit decimal of the
    (1-p)-quantile; the kept set is ``y <= that decimal``. Here the decimal
    lies just below the quantile observation, which is therefore dropped.
    """
    y1 = np.sort(CS.loc[(CS.d == 1) & (CS.s == 1), "ys"].to_numpy())
    y0 = CS.loc[(CS.d == 0) & (CS.s == 1), "ys"].to_numpy()
    p1, p0 = CS.s[CS.d == 1].mean(), CS.s[CS.d == 0].mean()
    q = (p1 - p0) / p1
    P = (1 - q) * len(y1)
    lth = float(f"{y1[int(np.floor(P))]:.15g}")
    kept = y1[y1 <= lth]
    return kept.mean() - y0.mean(), len(kept)


def test_lee_bounds_upper_matches_leebounds_shipped():
    r = sp.lee_bounds(CS, y="ys", treat="d", selection="s", se_method="analytic")
    ref = ST["leebounds_d"]
    _close(r.model_info["upper_bound"], ref["upper"])
    _close(r.model_info["se_upper"] ** 2, ref["V_upper"])
    _close(r.model_info["trimming_fraction"], ref["trim"], rtol=1e-12)


def test_lee_bounds_lower_gap_is_leebounds_threshold_rounding():
    """Stata's shipped lower bound keeps one observation fewer than Lee's
    quantile rule; the 15-digit threshold reconstruction reproduces it."""
    lower, kept = _stata_rounded_lower()
    _close(lower, ST["leebounds_d"]["lower"])
    r = sp.lee_bounds(CS, y="ys", treat="d", selection="s", se_method="analytic")
    assert r.model_info["lower_bound"] > ST["leebounds_d"]["lower"] + 1e-3
    assert kept == 297


def test_lee_bounds_exact_trimming_matches_leebounds_with_exact_thresholds():
    """leebounds' tie branch (fractional trimming), reached once the
    thresholds are held exactly, is trimming='exact' -- bounds and the
    analytic variance of both."""
    r = sp.lee_bounds(
        CS, y="ys", treat="d", selection="s", se_method="analytic", trimming="exact"
    )
    ref = ST["leebounds_exact"]
    _close(
        [r.model_info["lower_bound"], r.model_info["upper_bound"]],
        [ref["lower"], ref["upper"]],
    )
    _close(
        [r.model_info["se_lower"] ** 2, r.model_info["se_upper"] ** 2],
        [ref["V_lower"], ref["V_upper"]],
    )


def test_lee_bounds_control_trimmed_mirror():
    """Flipping the treatment makes leebounds trim the control arm."""
    df = CS.assign(dflip=1 - CS.d)
    r = sp.lee_bounds(df, y="ys", treat="dflip", selection="s", se_method="analytic")
    ref = ST["leebounds_dflip"]
    assert ref["trimmed_control"] == 1
    _close(r.model_info["lower_bound"], ref["lower"])
    _close(r.model_info["se_lower"] ** 2, ref["V_lower"])


def test_lee_bounds_imbens_manski_ci_matches_on_leebounds_grid():
    """leebounds searches the Imbens-Manski critical value on a 0.001 grid;
    ours solves it exactly. With Stata's SEs and bounds the two intervals
    agree to the grid resolution times the SE."""
    ref = ST["leebounds_d"]
    se_l, se_u = np.sqrt(ref["V_lower"]), np.sqrt(ref["V_upper"])
    from statspai.bounds.lee_manski import _imbens_manski_cn

    c = _imbens_manski_cn(ref["upper"] - ref["lower"], max(se_l, se_u), 0.05)
    assert abs(ref["lower"] - c * se_l - ref["ci_lower"]) < 1e-3 * se_l
    assert abs(ref["upper"] + c * se_u - ref["ci_upper"]) < 1e-3 * se_u


def test_sace_equals_lee_bounds_and_uses_missing_outcomes():
    """Zhang-Rubin SACE bounds are Lee bounds under monotonicity. The
    outcome is missing for non-survivors (truncation by death); before 1.29
    those rows were dropped and the bounds collapsed to a point."""
    r = sp.survivor_average_causal_effect(
        CS, y="ys", treat="d", survival="s", n_boot=5, seed=0
    )
    lee = sp.lee_bounds(CS, y="ys", treat="d", selection="s", n_bootstrap=5)
    _close(
        [r.model_info["sace_lower"], r.model_info["sace_upper"]],
        [lee.model_info["lower_bound"], lee.model_info["upper_bound"]],
    )
    _close(r.model_info["sace_upper"], ST["leebounds_d"]["upper"])
    assert r.n_obs == len(CS)


def test_principal_strat_complier_effect_is_the_wald_late():
    r = sp.principal_strat(CS, y="y", treat="d", strata="s", n_boot=5, seed=0)
    _close(r.effects.loc[0, "estimate"], R["principal_strat"]["late"])


# --------------------------------------------------------------------------
# Manski / Horowitz-Manski: Stata tebounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "assumption,key", [("none", "worst"), ("mts", "mts"), ("mts_mtr", "mts_mtr")]
)
def test_manski_bounds_match_tebounds(assumption, key):
    r = sp.manski_bounds(
        CS,
        y="yb",
        treat="d",
        y_lower=0,
        y_upper=1,
        assumption=assumption,
        n_bootstrap=5,
    )
    _close(
        [r.model_info["lower_bound"], r.model_info["upper_bound"]],
        ST["tebounds"][key],
        rtol=1e-12,
        atol=1e-15,
    )


def test_manski_mtr_bounds_closed_form():
    r = sp.manski_bounds(
        CS, y="yb", treat="d", y_lower=0, y_upper=1, assumption="mtr", n_bootstrap=5
    )
    _close(
        [r.model_info["lower_bound"], r.model_info["upper_bound"]],
        [0.0, ST["tebounds"]["worst"][1]],
        rtol=1e-12,
        atol=1e-15,
    )


def test_manski_mts_mtr_refuted_raises():
    df = CS.assign(yneg=1 - CS.yb)
    with pytest.raises(ValueError, match="jointly refuted"):
        sp.manski_bounds(
            df,
            y="yneg",
            treat="d",
            y_lower=0,
            y_upper=1,
            assumption="mts_mtr",
            n_bootstrap=5,
        )


def test_horowitz_manski_equals_unconditional_worst_case():
    """Averaging stratum worst-case bounds reproduces the unconditional
    bounds exactly (they are linear in the cell means), including strata
    with one arm missing, which used to be dropped."""
    for covs in (["x3"], ["x1", "x3"]):
        r = sp.horowitz_manski(
            CS, y="yb", treatment="d", covariates=covs, y_lower=0, y_upper=1, n_boot=5
        )
        _close([r.lower, r.upper], ST["tebounds"]["worst"], rtol=1e-12, atol=1e-15)
    df = CS.assign(g=np.where(CS.x1 > 1.8, 9, 0), d2=np.where(CS.x1 > 1.8, 1, CS.d))
    r = sp.horowitz_manski(
        df, y="yb", treatment="d2", covariates=["g"], y_lower=0, y_upper=1, n_boot=5
    )
    m = sp.manski_bounds(df, y="yb", treat="d2", y_lower=0, y_upper=1, n_bootstrap=5)
    _close(
        [r.lower, r.upper],
        [m.model_info["lower_bound"], m.model_info["upper_bound"]],
        rtol=1e-12,
    )


# --------------------------------------------------------------------------
# Mediation: med4way / CMAverse
# --------------------------------------------------------------------------


def test_four_way_matches_med4way_and_cmaverse():
    r = sp.four_way_decomposition(
        CS, y="ym", treat="d", mediator="m", covariates=["x1"]
    )
    ours = [r.total_effect, r.cde, r.int_ref, r.int_med, r.pie]
    _close(ours, ST["med4way"]["b"], rtol=1e-12)
    f = R["four_way"]
    _close(ours, [f["te"], f["cde"], f["intref"], f["intmed"], f["pie"]], rtol=1e-12)


FW_KEYS = ["total_effect", "cde", "int_ref", "int_med", "pie"]


def test_four_way_delta_se_matches_cmaverse_ols_vcov():
    r = sp.four_way_decomposition(
        CS, y="ym", treat="d", mediator="m", covariates=["x1"]
    )
    f = R["four_way"]
    _close(
        [r.se[k] for k in FW_KEYS],
        [f["se_te"], f["se_cde"], f["se_intref"], f["se_intmed"], f["se_pie"]],
        rtol=1e-12,
    )


def test_four_way_delta_se_matches_med4way_ml_vcov():
    """med4way fits both linear models by ML (``ml maximize``), so its
    residual variances stop at ml's convergence tolerance: the variances
    agree to ~7e-10, the point estimates (closed-form OLS) to 1e-12."""
    r = sp.four_way_decomposition(
        CS, y="ym", treat="d", mediator="m", covariates=["x1"], vce="ml"
    )
    _close([r.se[k] ** 2 for k in FW_KEYS], ST["med4way"]["V_diag"], rtol=5e-9)


@pytest.mark.parametrize("tv", [True, False])
def test_interventional_effects_match_cmaverse(tv):
    kw = dict(y="yl", treat="d", mediator="m2", covariates=["x1"], n_mc=200, n_boot=3)
    if tv:
        kw["tv_confounders"] = ["l"]
    r = sp.mediate_interventional(CS, **kw).detail["estimate"].to_numpy()
    f = R["interventional"]
    pre = "tv_" if tv else "notv_"
    _close(r, [f[pre + "iie"], f[pre + "ide"], f[pre + "te"]], rtol=1e-9)


# --------------------------------------------------------------------------
# IPCW: survival::coxph (Breslow)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stabilize,key", [(False, "w_unstab"), (True, "w_stab")])
def test_ipcw_cox_matches_survival_coxph(stabilize, key):
    r = sp.ipcw(
        SURV,
        time="time",
        event="event",
        censor_covariates=["z1", "z2"],
        method="cox_ph",
        stabilize=stabilize,
        truncate=None,
    )
    _close(r.weights, R["ipcw"][key], rtol=1e-9, atol=1e-15)


def test_ipcw_censored_rows_have_zero_weight():
    r = sp.ipcw(SURV, time="time", event="event", censor_covariates=["z1", "z2"])
    w = np.asarray(r.weights)
    assert np.all(w[SURV.event.to_numpy() == 0] == 0)
    assert np.all(w[SURV.event.to_numpy() == 1] > 0)


# --------------------------------------------------------------------------
# Dose-response (Hirano-Imbens GPS): Stata doseresponse
# --------------------------------------------------------------------------


def test_dose_response_hirano_imbens_matches_doseresponse():
    """Normal GPS by ML and an outcome quadratic in (T, GPS) with interaction
    -- sp.dose_response with those two models is Stata doseresponse."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    r = sp.dose_response(
        CS,
        y="ydose",
        treat="tc",
        covariates=["x1", "x2"],
        n_dose_points=5,
        dose_range=(0.0, 2.0),
        treatment_model=LinearRegression(),
        outcome_model=make_pipeline(PolynomialFeatures(2), LinearRegression()),
        n_bootstrap=2,
    )
    _close(r.detail["dose"], ST["doseresponse"]["doses"], rtol=0, atol=1e-15)
    _close(r.detail["response"], ST["doseresponse"]["drf"], rtol=1e-9)


# --------------------------------------------------------------------------
# Policy value: closed form
# --------------------------------------------------------------------------


def test_policy_value_is_mean_reward_of_chosen_action():
    """policytree's value of a policy on a reward matrix cbind(0, gamma) is
    mean(Gamma[i, pi_i]); for a {0, 1} policy that is mean(gamma * pi)."""
    rng = np.random.default_rng(3)
    gamma = rng.normal(0.2, 1.0, 500)
    pol = (rng.uniform(size=500) < 0.4).astype(int)
    Gamma = np.column_stack([np.zeros(500), gamma])
    _close(sp.policy_value(gamma, pol), Gamma[np.arange(500), pol].mean(), rtol=1e-15)
