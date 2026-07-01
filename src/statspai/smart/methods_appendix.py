"""Auto-generated "Methods and Formulas" appendix for fitted results.

Stata's manuals ship a *Methods and Formulas* section for every estimator;
R packages anchor theirs to a JSS paper. For a tool that aspires to *write*
the paper, the killer feature is that **every reported number traces back to
(i) the estimand / estimator definition, (ii) the inference actually used, and
(iii) a verified citation** — so a referee can audit the methodology without
trusting the implementation blind.

This module is the engineering arm of that goal. Given a fitted
:class:`~statspai.core.results.CausalResult` it emits a compact methods
appendix that a careful econometrics referee would accept:

* **Estimand / estimator** — hand-curated, textbook-standard LaTeX definitions
  (e.g. the 2x2 DiD double difference, the sharp-RD limit contrast, the
  Imbens–Angrist Wald/LATE ratio). These are stored, never generated, mirroring
  the zero-hallucination policy of :data:`CausalResult._CITATIONS` (CLAUDE.md
  §10). An unregistered method degrades to an explicit ``(methods text not yet
  registered)`` placeholder — never an invented formula.

* **Inference** — read off the *actual* fitted object
  (``model_info['se_method']``, cluster variables, bootstrap replications,
  bandwidth, first-stage F, ...) plus the universal ``se`` / ``ci`` / ``pvalue``.
  Reporting the SE *method that the code ran* — rather than transcribing a
  sandwich formula that might not match — is both more honest and the whole
  point of "traceability".

* **Citation** — pulled from ``result.cite(...)``, which itself reads the
  single-source ``_CITATIONS`` table / ``paper.bib``. Methods text and citation
  resolve through the *same* key logic, so they stay in lockstep.

Notes
-----
The estimand / estimator formulas below are standard definitions from the
cited primary sources; standard-error *math* is deliberately **not** transcribed
(it is reported via the method recorded on the result instead) to avoid drift
between the appendix and the implementation.

Examples
--------
>>> import statspai as sp
>>> import numpy as np
>>> import pandas as pd
>>> rng = np.random.default_rng(0)
>>> ids = np.repeat(np.arange(40), 2)
>>> time = np.tile([0, 1], 40)
>>> treat = (ids >= 20).astype(int)
>>> y = 1.0 + 2.0 * treat * time + rng.normal(size=len(ids))
>>> df = pd.DataFrame({"id": ids, "time": time, "treat": treat, "y": y})
>>> res = sp.did(df, y="y", treat="treat", time="time", id="id")
>>> txt = sp.methods_appendix(res, format="markdown")
>>> "Estimand" in txt
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = ["methods_appendix", "MethodSpec"]


@dataclass(frozen=True)
class MethodSpec:
    """A verified methods-appendix entry for one estimator family.

    Attributes
    ----------
    key : str
        Canonical lookup key (matches the citation key where possible).
    name : str
        Human-readable estimator name for the section heading.
    estimand_latex : str
        LaTeX (no delimiters) for the target parameter.
    estimator_latex : str
        LaTeX (no delimiters) for the sample estimator.
    prose : str
        One- to three-sentence plain-language description of the estimator.
    assumptions : list of str
        Identifying assumptions, one short clause each.
    aliases : list of str
        Extra keys / substrings that should resolve to this spec.
    """

    key: str
    name: str
    estimand_latex: str
    estimator_latex: str
    prose: str
    assumptions: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
#  Curated methods table — core causal estimators (DiD family, RD, IV, SCM,
#  matching/weighting, DML/TMLE, event-study, DDD, gsynth, imputation DiD,
#  CiC, LP-DiD, QTE, mediation, front-door, g-computation, Manski bounds,
#  continuous-treatment DiD, Bartik/shift-share, proximal, RKD, bunching,
#  augmented SC, matrix completion, IV quantile regression, GMM, causal
#  forest, meta-learners, R-learner, Mendelian randomization, dose-response,
#  Cox PH, Kaplan-Meier, Lee bounds, policy learning, honest DiD, Oster
#  bounds, wild cluster bootstrap, deep IV, interference, marginal structural
#  models, network exposure, stacked DiD, surrogate index, Oaxaca / RIF / DFL /
#  Gelbach decompositions, Kitagawa, multiway clustering) plus the
#  common regression families (OLS, Poisson, logit, probit, panel fixed
#  effects).
#
#  Every formula is a standard definition from the primary source cited via
#  result.cite(); see module docstring. SE math is intentionally omitted and
#  reported from the fitted object instead.
# ---------------------------------------------------------------------------

_SPECS: List[MethodSpec] = [
    MethodSpec(
        key="did_2x2",
        name="Difference-in-Differences (2x2)",
        estimand_latex=(
            r"\tau_{\mathrm{ATT}} = "
            r"\mathbb{E}\!\left[Y_i(1) - Y_i(0) \mid D_i = 1\right]"
        ),
        estimator_latex=(
            r"\hat\tau = \left(\bar Y^{\,T}_{\mathrm{post}} "
            r"- \bar Y^{\,T}_{\mathrm{pre}}\right)"
            r" - \left(\bar Y^{\,C}_{\mathrm{post}} "
            r"- \bar Y^{\,C}_{\mathrm{pre}}\right)"
        ),
        prose=(
            "The canonical two-group, two-period difference-in-differences "
            "estimator: the change in the treated group's mean outcome minus the "
            "change in the comparison group's mean outcome."
        ),
        assumptions=[
            "Parallel trends: absent treatment, treated and control means would "
            "have moved in parallel.",
            "No anticipation of treatment in the pre-period.",
            "Stable group composition (SUTVA / no spillovers).",
        ],
        aliases=["did", "twfe_did", "2x2", "did_did"],
    ),
    MethodSpec(
        key="twfe_did",
        name="Two-Way Fixed Effects DiD",
        estimand_latex=r"\tau_{\mathrm{ATT}} \;\; (\text{under homogeneous effects})",
        estimator_latex=(
            r"Y_{it} = \alpha_i + \lambda_t + \tau\, D_{it} + \varepsilon_{it};"
            r"\quad \hat\tau = \text{OLS coefficient on } D_{it}"
        ),
        prose=(
            "Two-way (unit and time) fixed-effects regression of the outcome on a "
            "treatment indicator. With staggered timing and heterogeneous effects "
            "the OLS coefficient is a possibly negatively weighted average of "
            "group-time effects (Goodman-Bacon, 2021); prefer a "
            "heterogeneity-robust estimator in that case."
        ),
        assumptions=[
            "Parallel trends and no anticipation.",
            "Homogeneous treatment effects (else negative-weighting bias under "
            "staggered adoption).",
        ],
        aliases=["twfe", "fe_did"],
    ),
    MethodSpec(
        key="callaway_santanna",
        name="Callaway & Sant'Anna Group-Time ATT",
        estimand_latex=(
            r"\mathrm{ATT}(g,t) = \mathbb{E}\!\left[Y_t(g) - Y_t(0) \mid G_g = 1\right]"
        ),
        estimator_latex=(
            r"\widehat{\mathrm{ATT}}(g,t) = "
            r"\mathbb{E}\!\left[Y_t - Y_{g-1} \mid G=g\right]"
            r" - \mathbb{E}\!\left[Y_t - Y_{g-1} \mid C\right],\quad"
            r"\theta = \sum_{g,t} w(g,t)\,\widehat{\mathrm{ATT}}(g,t)"
        ),
        prose=(
            "Group-time average treatment effects on the treated, identified by "
            "comparing each treatment cohort's outcome change against a "
            "never-treated (or not-yet-treated) comparison group, then aggregated "
            "with user-chosen weights. The doubly robust variant additionally "
            "models the propensity and outcome evolution."
        ),
        assumptions=[
            "Conditional parallel trends relative to the comparison group.",
            "No anticipation prior to treatment.",
            "Overlap / common support of covariates.",
            "Irreversibility of treatment (staggered adoption).",
        ],
        aliases=["cs", "callaway", "santanna", "aggte", "csdid", "group_time"],
    ),
    MethodSpec(
        key="sun_abraham",
        name="Sun & Abraham Interaction-Weighted Event Study",
        estimand_latex=(
            r"\mathrm{CATT}_{e,\ell} = "
            r"\mathbb{E}\!\left[Y_{i,e+\ell}(e) - Y_{i,e+\ell}(0) "
            r"\mid E_i = e\right]"
        ),
        estimator_latex=(
            r"\hat\tau_\ell = \sum_{e} \widehat{\mathrm{CATT}}_{e,\ell}\;"
            r"\Pr\!\left(E_i = e \mid 0 \le e+\ell \le T\right)"
        ),
        prose=(
            "Interaction-weighted event-study estimator: saturate the two-way "
            "fixed-effects specification with cohort-by-relative-period "
            "interactions to recover clean cohort-specific effects, then average "
            "them using sample cohort shares. This removes the contamination that "
            "afflicts naive TWFE event-study leads/lags under heterogeneous "
            "effects."
        ),
        assumptions=[
            "Parallel trends across cohorts.",
            "No anticipation.",
            "Cohort shares identified within each relative period.",
        ],
        aliases=["sunab", "interaction_weighted", "iw_event_study"],
    ),
    MethodSpec(
        key="drdid",
        name="Doubly Robust DiD (Sant'Anna & Zhao)",
        estimand_latex=(
            r"\tau_{\mathrm{ATT}} = "
            r"\mathbb{E}\!\left[Y_i(1) - Y_i(0) \mid D_i = 1\right]"
        ),
        estimator_latex=(
            r"\widehat{\mathrm{ATT}}_{\mathrm{dr}} = "
            r"\mathbb{E}\!\left[\left(w_1(D) - w_0(D,X)\right)"
            r"\left(\Delta Y - \mu_{0,\Delta}(X)\right)\right]"
        ),
        prose=(
            "Doubly robust difference-in-differences combining an outcome-"
            "regression model for the trend, mu_{0,Delta}(X), with an inverse-"
            "propensity weighting model, via weights w_1 (treated) and w_0 "
            "(reweighted controls). Consistent if *either* the outcome model or "
            "the propensity model is correctly specified."
        ),
        assumptions=[
            "Conditional parallel trends given covariates X.",
            "Overlap: 0 < propensity < 1.",
            "Consistency if either the outcome or propensity model is correct.",
        ],
        aliases=["dr_did", "drdid_panel", "doubly_robust_did"],
    ),
    MethodSpec(
        key="did_multiplegt",
        name="de Chaisemartin & D'Haultfoeuille DID_M",
        estimand_latex=(
            r"\delta = \mathbb{E}\!\left[\sum_{(g,t)\in S} \frac{N_{g,t}}{N_S}\,"
            r"\mathrm{ATT}(g,t)\right]"
        ),
        estimator_latex=(
            r"\widehat{\mathrm{DID}}_M = \sum_{t} \frac{N_t}{N}\Big("
            r"\Delta\bar Y^{\,\text{switchers}}_t "
            r"- \Delta\bar Y^{\,\text{stayers}}_t\Big)"
        ),
        prose=(
            "Heterogeneity-robust DiD that compares, period by period, units whose "
            "treatment switches against units whose treatment stays constant, "
            "averaging over all such cells. Robust to heterogeneous and "
            "dynamic-in-the-static-version treatment effects."
        ),
        assumptions=[
            "Parallel trends.",
            "Strong exogeneity of treatment paths.",
            "No carryover effects (for the static estimator).",
        ],
        aliases=["dechaisemartin", "did_m", "didmultiplegt", "did_multiplegt_dyn"],
    ),
    MethodSpec(
        key="rdrobust",
        name="Regression Discontinuity (Local Polynomial)",
        estimand_latex=(
            r"\tau_{\mathrm{SRD}} = \lim_{x\downarrow c}\mathbb{E}[Y\mid X=x]"
            r" - \lim_{x\uparrow c}\mathbb{E}[Y\mid X=x]"
        ),
        estimator_latex=(
            r"\hat\tau = \hat\mu_+(c) - \hat\mu_-(c),\quad "
            r"\hat\mu_\pm = \text{local polynomial fit within bandwidth } h, "
            r"\text{ bias-corrected (CCT)}"
        ),
        prose=(
            "Sharp regression-discontinuity contrast of the outcome's right- and "
            "left-limits at the cutoff, estimated by weighted local polynomial "
            "regression within a data-driven bandwidth, with bias correction and "
            "robust confidence intervals (Calonico, Cattaneo & Titiunik). The "
            "fuzzy design divides this jump by the first-stage jump in treatment "
            "probability."
        ),
        assumptions=[
            "Continuity of potential-outcome means at the cutoff.",
            "No precise manipulation of the running variable (density continuity).",
            "Fuzzy design: monotonicity and a first-stage jump at the cutoff.",
        ],
        aliases=["rd", "rdd", "rd_robust", "rdrobust_sharp", "local_polynomial"],
    ),
    MethodSpec(
        key="iv",
        name="Instrumental Variables (2SLS / LATE)",
        estimand_latex=(
            r"\tau_{\mathrm{LATE}} = "
            r"\mathbb{E}\!\left[Y_i(1) - Y_i(0) \mid \text{compliers}\right]"
        ),
        estimator_latex=(
            r"\hat\tau = "
            r"\frac{\widehat{\mathrm{Cov}}(Y,Z)}{\widehat{\mathrm{Cov}}(D,Z)};\quad"
            r"\hat\beta_{2SLS} = \left(X' P_Z X\right)^{-1} X' P_Z Y,\;"
            r"P_Z = Z(Z'Z)^{-1}Z'"
        ),
        prose=(
            "Two-stage least squares / instrumental-variables estimator. With a "
            "single binary instrument the just-identified Wald ratio recovers the "
            "local average treatment effect for compliers (Imbens & Angrist); the "
            "general 2SLS form projects the endogenous regressors onto the "
            "instrument space."
        ),
        assumptions=[
            "Instrument relevance (non-zero first stage).",
            "Exclusion: the instrument affects Y only through D.",
            "Independence / as-good-as-random assignment of the instrument.",
            "Monotonicity (no defiers) for the LATE interpretation.",
        ],
        aliases=["2sls", "tsls", "ivreg", "liml", "late", "ivreg2", "iv_reg"],
    ),
    MethodSpec(
        key="synth",
        name="Synthetic Control Method",
        estimand_latex=r"\tau_{1t} = Y_{1t}(1) - Y_{1t}(0),\quad t > T_0",
        estimator_latex=(
            r"\hat Y_{1t}(0) = \sum_{j\ge 2} w_j^{*} Y_{jt},\quad "
            r"W^{*} = \arg\min_{W}\|X_1 - X_0 W\|_V "
            r"\;\text{s.t.}\; w_j\ge 0,\; \textstyle\sum_j w_j = 1"
        ),
        prose=(
            "Constructs a synthetic counterfactual for the treated unit as a "
            "convex combination of control units whose weights best reproduce the "
            "treated unit's pre-treatment outcomes and predictors. The treatment "
            "effect is the post-treatment gap between observed and synthetic "
            "outcomes (Abadie, Diamond & Hainmueller)."
        ),
        assumptions=[
            "Treated unit lies in the convex hull of controls (no extrapolation).",
            "Good pre-treatment fit over a long pre-period.",
            "No anticipation and no interference from the treatment.",
        ],
        aliases=["sc", "synthetic_control", "scm", "abadie"],
    ),
    MethodSpec(
        key="sdid",
        name="Synthetic Difference-in-Differences",
        estimand_latex=r"\tau_{\mathrm{ATT}}",
        estimator_latex=(
            r"(\hat\tau,\hat\mu,\hat\alpha,\hat\beta) = \arg\min \sum_{i,t}"
            r"\hat\omega_i\,\hat\lambda_t"
            r"\left(Y_{it} - \mu - \alpha_i - \beta_t - \tau D_{it}\right)^2"
        ),
        prose=(
            "Synthetic difference-in-differences combines the unit-reweighting of "
            "synthetic control with the time-reweighting and two-way structure of "
            "DiD: it fits a weighted two-way fixed-effects regression using "
            "data-driven unit weights omega and time weights lambda (Arkhangelsky "
            "et al.)."
        ),
        assumptions=[
            "Approximate parallel trends after reweighting.",
            "Adequate pre-treatment fit of the synthetic unit/time weights.",
            "No anticipation or interference.",
        ],
        aliases=["synthdid", "synth_did", "sequential_sdid"],
    ),
    MethodSpec(
        key="psm",
        name="Propensity Score / Nearest-Neighbor Matching",
        estimand_latex=(
            r"\tau_{\mathrm{ATT}} = "
            r"\mathbb{E}\!\left[Y_i(1) - Y_i(0) \mid D_i = 1\right]"
        ),
        estimator_latex=(
            r"\widehat{\mathrm{ATT}} = \frac{1}{N_1}\sum_{i:D_i=1}"
            r"\Big(Y_i - \sum_{j} w_{ij} Y_j\Big),\quad "
            r"w_{ij}\text{ matched on } \hat e(X)\text{ or } X"
        ),
        prose=(
            "Matches each treated unit to comparable control units on the "
            "estimated propensity score or covariates and contrasts their "
            "outcomes, averaging over treated units to estimate the ATT (or over "
            "all units for the ATE)."
        ),
        assumptions=[
            "Unconfoundedness / conditional independence given X.",
            "Overlap (common support) of the propensity score.",
        ],
        aliases=[
            "matching",
            "nnmatch",
            "match",
            "propensity_match",
            "genmatch",
            "optimal_match",
        ],
    ),
    MethodSpec(
        key="ipw",
        name="Inverse Propensity Weighting",
        estimand_latex=(
            r"\tau_{\mathrm{ATE}} = " r"\mathbb{E}\!\left[Y_i(1) - Y_i(0)\right]"
        ),
        estimator_latex=(
            r"\widehat{\mathrm{ATE}} = \frac{1}{N}\sum_{i}\left("
            r"\frac{D_i Y_i}{\hat e(X_i)} - \frac{(1-D_i) Y_i}{1-\hat e(X_i)}\right)"
        ),
        prose=(
            "Reweights observed outcomes by the inverse of the estimated "
            "treatment probability so that the weighted treated and control "
            "samples are covariate-balanced (Horvitz-Thompson). The stabilized "
            "variant normalizes the weights to reduce variance."
        ),
        assumptions=[
            "Unconfoundedness given X.",
            "Overlap: 0 < e(X) < 1.",
            "Correctly specified propensity model.",
        ],
        aliases=[
            "inverse_propensity",
            "iptw",
            "stabilized_weights",
            "horvitz_thompson",
        ],
    ),
    MethodSpec(
        key="aipw",
        name="Augmented IPW (Doubly Robust ATE)",
        estimand_latex=(
            r"\tau_{\mathrm{ATE}} = " r"\mathbb{E}\!\left[Y_i(1) - Y_i(0)\right]"
        ),
        estimator_latex=(
            r"\widehat{\mathrm{ATE}} = \frac{1}{N}\sum_i\Big[\mu_1(X_i) - \mu_0(X_i)"
            r" + \frac{D_i\,(Y_i - \mu_1(X_i))}{\hat e(X_i)}"
            r" - \frac{(1-D_i)\,(Y_i - \mu_0(X_i))}{1-\hat e(X_i)}\Big]"
        ),
        prose=(
            "Augmented inverse-propensity-weighting estimator: the IPW score plus "
            "an outcome-regression augmentation term. Doubly robust — consistent "
            "if either the outcome models mu_d or the propensity e is correctly "
            "specified — and semiparametrically efficient when both are."
        ),
        assumptions=[
            "Unconfoundedness given X.",
            "Overlap: 0 < e(X) < 1.",
            "Consistency if either the outcome or propensity model is correct.",
        ],
        aliases=["augmented_ipw", "doubly_robust", "dr_ate", "aipw_ate"],
    ),
    MethodSpec(
        key="dml",
        name="Double / Debiased Machine Learning",
        estimand_latex=(
            r"\theta_0 \;\; "
            r"(\text{e.g. } \mathrm{ATE}\text{ or partially-linear effect})"
        ),
        estimator_latex=(
            r"\frac{1}{N}\sum_{i} \psi\!\left(W_i; \hat\theta, \hat\eta_{-k(i)}\right)"
            r" = 0;\quad"
            r"\hat\theta = \frac{\sum_i \check V_i \check Y_i}{\sum_i \check V_i^2}"
            r"\;\;(\check V,\check Y\text{: cross-fitted residuals})"
        ),
        prose=(
            "Estimates a low-dimensional causal parameter using a Neyman-"
            "orthogonal score with cross-fitting: machine-learning nuisance "
            "functions (propensity and outcome) are fit on auxiliary folds, then "
            "the target parameter solves the orthogonal moment on the held-out "
            "folds, immunizing it to first-order nuisance error (Chernozhukov et "
            "al.)."
        ),
        assumptions=[
            "Unconfoundedness and overlap.",
            "Neyman orthogonality of the score.",
            "Nuisance estimators converge faster than N^{-1/4}.",
        ],
        aliases=["double_ml", "debiased_ml", "dml_panel", "ddml", "partially_linear"],
    ),
    MethodSpec(
        key="tmle",
        name="Targeted Maximum Likelihood Estimation",
        estimand_latex=r"\psi = \mathbb{E}\!\left[\mu_1(X) - \mu_0(X)\right]",
        estimator_latex=(
            r"\mu^{*} = \text{fluctuate}(\mu_0;\, H),\quad "
            r"H(D,X) = \frac{D}{\hat e(X)} - \frac{1-D}{1-\hat e(X)};\quad "
            r"\hat\psi = \frac{1}{N}\sum_i\left(\mu_1^{*}(X_i) - \mu_0^{*}(X_i)\right)"
        ),
        prose=(
            "Targeted maximum likelihood: start from an initial outcome model, "
            "then perform a targeting update along a parametric fluctuation that "
            "uses the clever covariate H, and plug the updated model into the "
            "substitution estimator. Doubly robust and asymptotically efficient "
            "(van der Laan & Rubin)."
        ),
        assumptions=[
            "Unconfoundedness given X.",
            "Overlap: 0 < e(X) < 1.",
            "Consistency if either the outcome or propensity model is correct.",
        ],
        aliases=["targeted_mle", "tmle_ate", "ltmle", "hal_tmle"],
    ),
    MethodSpec(
        key="event_study",
        name="Event-Study (Dynamic TWFE)",
        estimand_latex=(
            r"\tau_\ell = \mathbb{E}\!\left[Y_{i,e+\ell}(e) "
            r"- Y_{i,e+\ell}(0)\right]"
        ),
        estimator_latex=(
            r"Y_{it} = \alpha_i + \lambda_t "
            r"+ \sum_{\ell \neq -1} \beta_\ell \mathbf{1}\{t - E_i = \ell\} "
            r"+ \varepsilon_{it}"
        ),
        prose=(
            "Dynamic event-study specification: regress the outcome on a full "
            "set of leads and lags in event time relative to a normalized "
            "baseline period (here ell = -1). Leads probe pre-trends; lags trace "
            "the dynamic path of the effect. Under heterogeneous timing prefer an "
            "interaction-weighted estimator (Sun & Abraham)."
        ),
        assumptions=[
            "Parallel trends across event-time periods.",
            "No anticipation before the baseline period.",
            "Baseline-period normalization of the coefficient path.",
        ],
        aliases=["eventstudy", "dynamic_did", "leads_lags", "es_did"],
    ),
    MethodSpec(
        key="ddd",
        name="Triple Differences (DDD)",
        estimand_latex=r"\tau_{\mathrm{ATT}}",
        estimator_latex=(
            r"\hat\tau = \mathrm{DiD}_{\text{eligible}} "
            r"- \mathrm{DiD}_{\text{ineligible}}"
        ),
        prose=(
            "Triple-difference estimator: the difference between a "
            "difference-in-differences computed for an eligible (treated-"
            "exposed) group and one computed for an ineligible comparison "
            "group, netting out group-specific shocks that a single DiD would "
            "absorb into the trend."
        ),
        assumptions=[
            "Parallel trends in the third (eligibility) difference.",
            "No anticipation.",
            "Stable composition across the three differencing dimensions.",
        ],
        aliases=["triple_diff", "triple_difference", "ddd_heterogeneous"],
    ),
    MethodSpec(
        key="gsynth",
        name="Generalized Synthetic Control (Interactive FE)",
        estimand_latex=r"\tau_{it} = Y_{it}(1) - Y_{it}(0),\quad t > T_0",
        estimator_latex=(
            r"Y_{it}(0) = x_{it}'\beta + \lambda_i' f_t + \varepsilon_{it};"
            r"\quad \hat\tau_{it} = Y_{it} - \hat Y_{it}(0)"
        ),
        prose=(
            "Fits an interactive fixed-effects (latent factor) model on the "
            "control units, then imputes each treated unit's counterfactual from "
            "the estimated loadings and factors (Xu, 2017). Generalizes "
            "synthetic control to multiple treated units, covariates, and "
            "staggered adoption."
        ),
        assumptions=[
            "Low-rank factor structure correctly captures untreated outcomes.",
            "No anticipation and no interference.",
            "Treatment ignorable given latent factors and covariates.",
        ],
        aliases=[
            "generalized_synthetic_control",
            "interactive_fe",
            "interactive_fixed_effects",
            "ife",
            "gsc",
        ],
    ),
    MethodSpec(
        key="did_imputation",
        name="Imputation DiD (Borusyak-Jaravel-Spiess)",
        estimand_latex=r"\tau = \textstyle\sum_{it:\,D_{it}=1} w_{it}\,\tau_{it}",
        estimator_latex=(
            r"\hat Y_{it}(0) = \hat\alpha_i + \hat\lambda_t "
            r"\;\;(\text{fit on } D=0);\quad "
            r"\hat\tau_{it} = Y_{it} - \hat Y_{it}(0)"
        ),
        prose=(
            "Estimates the two-way (plus covariate) model on untreated "
            "observations only, imputes each treated observation's untreated "
            "potential outcome, and averages the gaps with efficient weights. "
            "Robust to heterogeneous timing and efficient under homoskedasticity "
            "(Borusyak, Jaravel & Spiess)."
        ),
        assumptions=[
            "Parallel trends and no anticipation.",
            "Correctly specified two-way (plus covariate) model for Y(0).",
        ],
        aliases=["bjs", "imputation_did", "borusyak_jaravel_spiess"],
    ),
    MethodSpec(
        key="cic",
        name="Changes-in-Changes (Athey-Imbens)",
        estimand_latex=(
            r"\mathrm{QTT}(\tau) = " r"F^{-1}_{Y_1(1)}(\tau) - F^{-1}_{Y_1(0)}(\tau)"
        ),
        estimator_latex=(
            r"\hat F_{Y_1(0)}(y) = "
            r"\hat F_{Y_{01}}\!\big(\hat F_{Y_{00}}^{-1}(\hat F_{Y_{10}}(y))\big)"
        ),
        prose=(
            "Nonparametric distributional generalization of DiD: maps the "
            "treated group's pre-period outcome distribution through the control "
            "group's observed change to construct the treated counterfactual "
            "distribution, yielding quantile treatment effects (Athey & Imbens)."
        ),
        assumptions=[
            "Scalar monotone production function in an unobservable.",
            "Time-invariant rank distribution within group.",
            "Common support of the outcome across groups and periods.",
        ],
        aliases=["changes_in_changes", "athey_imbens"],
    ),
    MethodSpec(
        key="lp_did",
        name="Local Projections DiD",
        estimand_latex=r"\tau_h \;\; (\text{dynamic ATT at horizon } h)",
        estimator_latex=(
            r"Y_{i,t+h} - Y_{i,t-1} = \alpha + \tau_h\,\Delta D_{it} "
            r"+ \text{controls} + \varepsilon_{it}"
        ),
        prose=(
            "Estimates dynamic treatment effects horizon by horizon by "
            "regressing the cumulative outcome change at each horizon on the "
            "treatment change, using not-yet-treated units as clean controls "
            "(Dube, Girardi, Jorda & Taylor)."
        ),
        assumptions=[
            "Parallel trends and no anticipation.",
            "A valid clean (not-yet-treated) control group at each horizon.",
        ],
        aliases=["local_projection_did", "lpdid", "dube_girardi"],
    ),
    MethodSpec(
        key="ols",
        name="Ordinary Least Squares",
        estimand_latex=r"\beta = \arg\min_b\,\mathbb{E}\!\left[(Y - X'b)^2\right]",
        estimator_latex=r"\hat\beta = (X'X)^{-1} X'Y",
        prose=(
            "Linear regression by minimizing the sum of squared residuals. The "
            "coefficient is the conditional-expectation slope under correct "
            "specification, and the best linear approximation otherwise."
        ),
        assumptions=[
            "Linearity in parameters and correct specification.",
            "Exogeneity: E[epsilon | X] = 0.",
            "No perfect collinearity; homoskedasticity for classical SE "
            "(else use robust / clustered SE).",
        ],
        aliases=["regress", "lm", "linear_regression", "least_squares"],
    ),
    MethodSpec(
        key="poisson",
        name="Poisson Regression",
        estimand_latex=r"\mathbb{E}[Y \mid X] = \exp(X'\beta)",
        estimator_latex=(
            r"\hat\beta:\;\; "
            r"\textstyle\sum_i \left(Y_i - \exp(X_i'\beta)\right) X_i = 0"
        ),
        prose=(
            "Exponential-mean regression for counts, fit by maximum likelihood. "
            "As the Poisson pseudo-MLE it is consistent for the conditional mean "
            "even when the data are not Poisson (e.g. PPML for multiplicative "
            "models)."
        ),
        assumptions=[
            "Correct exponential conditional-mean specification.",
            "Equidispersion for classical SE (else use robust SE).",
        ],
        aliases=["fepois", "ppml", "ppmlhdfe", "poisson_regression"],
    ),
    MethodSpec(
        key="logit",
        name="Logistic Regression",
        estimand_latex=(
            r"\Pr(Y=1 \mid X) = \Lambda(X'\beta) " r"= \frac{1}{1 + e^{-X'\beta}}"
        ),
        estimator_latex=r"\hat\beta = \arg\max_\beta\,\ell(\beta)\;\;(\text{MLE})",
        prose=(
            "Binary-outcome regression with the logistic link, fit by maximum "
            "likelihood. Coefficients are log-odds; marginal effects require "
            "post-estimation transformation."
        ),
        assumptions=[
            "Correct link and linear-index specification.",
            "Independent observations.",
            "No perfect separation.",
        ],
        aliases=["logistic", "logistic_regression"],
    ),
    MethodSpec(
        key="probit",
        name="Probit Regression",
        estimand_latex=r"\Pr(Y=1 \mid X) = \Phi(X'\beta)",
        estimator_latex=r"\hat\beta = \arg\max_\beta\,\ell(\beta)\;\;(\text{MLE})",
        prose=(
            "Binary-outcome regression with the probit (standard-normal CDF) "
            "link, fit by maximum likelihood. Phi denotes the standard normal "
            "CDF; coefficients index a latent-variable threshold-crossing model."
        ),
        assumptions=[
            "Correct link and linear-index specification.",
            "Independent observations with homoskedastic latent errors.",
        ],
        aliases=["probit_regression"],
    ),
    MethodSpec(
        key="fe",
        name="Panel Fixed Effects (Within Estimator)",
        estimand_latex=r"\beta \;\; (\text{within-unit slope})",
        estimator_latex=(
            r"\ddot Z_{it} = Z_{it} - \bar Z_i;\quad "
            r"\hat\beta = (\ddot X'\ddot X)^{-1}\ddot X'\ddot Y"
        ),
        prose=(
            "Controls for time-invariant unit heterogeneity by demeaning each "
            "variable within units (the within / LSDV estimator). Identifies the "
            "slope purely from within-unit variation."
        ),
        assumptions=[
            "Strict exogeneity of regressors given the unit effects.",
            "Within-unit variation in the regressors of interest.",
            "No time-varying confounders correlated with the regressors.",
        ],
        aliases=[
            "fixed_effects",
            "within",
            "feols",
            "panel_fe",
            "lsdv",
            "twfe_ols",
        ],
    ),
    MethodSpec(
        key="qte",
        name="Quantile Treatment Effects",
        estimand_latex=r"\tau(u) = F^{-1}_{Y(1)}(u) - F^{-1}_{Y(0)}(u)",
        estimator_latex=(
            r"\hat F_{Y(d)}(y) = \sum_i \omega_i^{d}\,\mathbf{1}\{Y_i \le y\};"
            r"\quad \hat\tau(u) = \hat F^{-1}_{Y(1)}(u) - \hat F^{-1}_{Y(0)}(u)"
        ),
        prose=(
            "Estimates the gap between the marginal outcome quantiles under "
            "treatment and control. Under unconfoundedness, Firpo's "
            "propensity-weighted estimator reweights each arm to recover the "
            "counterfactual quantile functions. Note this is a quantile of the "
            "two marginals, not of the individual effect (unless rank invariance "
            "holds)."
        ),
        assumptions=[
            "Unconfoundedness and overlap (for the weighting estimator).",
            "Rank invariance if a per-unit effect interpretation is desired.",
        ],
        aliases=[
            "quantile_treatment_effect",
            "qtet",
            "qte_hd_panel",
            "firpo_qte",
        ],
    ),
    MethodSpec(
        key="mediation",
        name="Causal Mediation (Natural Direct/Indirect Effects)",
        estimand_latex=(
            r"\mathrm{NDE} = \mathbb{E}[Y(1,M(0)) - Y(0,M(0))],\quad "
            r"\mathrm{NIE} = \mathbb{E}[Y(1,M(1)) - Y(1,M(0))]"
        ),
        estimator_latex=(
            r"\mathbb{E}[Y(t,M(t'))] = "
            r"\int \mathbb{E}[Y\mid t,m,X]\,dP(m\mid t',X)\,dP(X)"
        ),
        prose=(
            "Decomposes a total effect into the part transmitted through a "
            "mediator (natural indirect effect) and the part that is not "
            "(natural direct effect), via the mediation formula that integrates "
            "the outcome model over the mediator's conditional distribution."
        ),
        assumptions=[
            "No unmeasured treatment-outcome confounding.",
            "No unmeasured treatment-mediator confounding.",
            "No unmeasured mediator-outcome confounding.",
            "No mediator-outcome confounder is itself affected by treatment.",
        ],
        aliases=[
            "mediate",
            "natural_effects",
            "causal_mediation",
            "mediation_decompose",
            "mediate_interventional",
            "nde_nie",
        ],
    ),
    MethodSpec(
        key="front_door",
        name="Front-Door Adjustment",
        estimand_latex=r"\Pr(Y \mid \mathrm{do}(T=t))",
        estimator_latex=(
            r"\Pr(Y \mid \mathrm{do}(t)) = "
            r"\sum_m \Pr(m\mid t)\sum_{t'} \Pr(Y\mid m,t')\,\Pr(t')"
        ),
        prose=(
            "Identifies a causal effect through a fully-mediating variable M "
            "when the treatment-outcome back-door is confounded but the "
            "treatment-mediator and mediator-outcome links are not (Pearl's "
            "front-door criterion)."
        ),
        assumptions=[
            "M intercepts all directed paths from T to Y.",
            "No unconfounded treatment-mediator relationship.",
            "No unobserved mediator-outcome confounding given T.",
        ],
        aliases=["frontdoor", "front_door_adjustment", "pearl_frontdoor"],
    ),
    MethodSpec(
        key="g_computation",
        name="G-Computation (Standardization)",
        estimand_latex=r"\tau_{\mathrm{ATE}} = \mathbb{E}[Y(1) - Y(0)]",
        estimator_latex=(
            r"\hat\tau = \frac{1}{N}\sum_i" r"\big(\hat\mu(1,X_i) - \hat\mu(0,X_i)\big)"
        ),
        prose=(
            "Fits an outcome model mu(t,X) = E[Y | T=t, X] and averages its "
            "predicted treated-minus-control contrast over the empirical "
            "covariate distribution (the parametric g-formula / "
            "standardization)."
        ),
        assumptions=[
            "Unconfoundedness given X.",
            "Overlap: 0 < e(X) < 1.",
            "Correctly specified outcome model.",
        ],
        aliases=[
            "gcomputation",
            "g_formula",
            "standardization",
            "g_comp",
            "parametric_g_formula",
        ],
    ),
    MethodSpec(
        key="manski_bounds",
        name="Worst-Case (Manski) Bounds",
        estimand_latex=r"\tau_{\mathrm{ATE}} \in [\underline{\tau},\ \overline{\tau}]",
        estimator_latex=(
            r"\mathbb{E}[Y(1)] \in \big[\,"
            r"\mathbb{E}[Y\mid D{=}1]\Pr(D{=}1) + y_{\min}\Pr(D{=}0),\; "
            r"\mathbb{E}[Y\mid D{=}1]\Pr(D{=}1) + y_{\max}\Pr(D{=}0)\,\big]"
        ),
        prose=(
            "The no-assumption worst-case bounds on a potential-outcome mean: "
            "the observed values for the observed arm and the full outcome "
            "support for the counterfactual arm. The ATE bounds combine the "
            "bounds on E[Y(1)] and E[Y(0)] (Manski)."
        ),
        assumptions=[
            "Bounded outcome support [y_min, y_max].",
            "No further identifying assumptions (the point of partial "
            "identification).",
        ],
        aliases=["worst_case_bounds", "no_assumption_bounds", "manski_worst_case"],
    ),
    MethodSpec(
        key="continuous_did",
        name="Continuous-Treatment DiD (Dose-Response)",
        estimand_latex=r"\mathrm{ATT}(d\mid d) = \mathbb{E}[Y(d) - Y(0)\mid D=d]",
        estimator_latex=(
            r"\widehat{\mathrm{ATT}}(d\mid d) = "
            r"\mathbb{E}[\Delta Y\mid D=d] - \mathbb{E}[\Delta Y\mid D=0]"
        ),
        prose=(
            "Extends staggered DiD to a continuous (dose) treatment, identifying "
            "dose-specific level effects ATT(d|d) and the average causal "
            "response by comparing the outcome change of units receiving dose d "
            "against untreated units (Callaway, Goodman-Bacon & Sant'Anna). "
            "Requires a stronger parallel-trends condition holding across all "
            "doses."
        ),
        assumptions=[
            "Strong parallel trends across the full dose distribution.",
            "No anticipation.",
            "Correct handling of TWFE dose-heterogeneity bias.",
        ],
        aliases=["continuous_treatment_did", "dose_did", "cont_did", "cgs_did"],
    ),
    MethodSpec(
        key="bartik",
        name="Bartik / Shift-Share Instrument",
        estimand_latex=r"\beta\;\;(\text{2SLS coefficient instrumented by } B_i)",
        estimator_latex=(
            r"B_i = \sum_k z_{ik}\,g_k\;\;(\text{shares}\times\text{shocks});"
            r"\quad \hat\beta = \text{2SLS of } Y \text{ on } X \text{ using } B_i"
        ),
        prose=(
            "A shift-share instrument formed by interacting local exposure "
            "shares z_ik (e.g. industry composition) with common shocks g_k "
            "(e.g. national growth), then used to instrument the endogenous "
            "regressor. Identification rests either on exogenous shares "
            "(Goldsmith-Pinkham, Sorkin & Swift) or exogenous shocks (Borusyak, "
            "Hull & Jaravel)."
        ),
        assumptions=[
            "Instrument relevance (non-zero first stage).",
            "Exogeneity of the shares OR of the shocks (the two routes).",
        ],
        aliases=[
            "shift_share",
            "shift_share_iv",
            "bartik_instrument",
            "goldsmith_pinkham",
        ],
    ),
    MethodSpec(
        key="proximal",
        name="Proximal Causal Inference",
        estimand_latex=r"\tau_{\mathrm{ATE}} = \mathbb{E}[Y(1) - Y(0)]",
        estimator_latex=(
            r"\mathbb{E}[Y - h(W,A,X)\mid Z,A,X] = 0;\quad "
            r"\hat\tau = \mathbb{E}[h(W,1,X) - h(W,0,X)]"
        ),
        prose=(
            "Identifies causal effects under unmeasured confounding using a "
            "pair of negative-control proxies — a treatment-inducing proxy Z "
            "and an outcome-inducing proxy W — by solving for an outcome "
            "confounding bridge function h (Miao, Geng & Tchetgen Tchetgen; "
            "Cui et al.)."
        ),
        assumptions=[
            "Valid negative-control proxies (proxy independence conditions).",
            "Completeness / relevance of the proxies for the confounder.",
            "Overlap.",
        ],
        aliases=["proximal_causal", "negative_controls", "pci", "bridge_function"],
    ),
    MethodSpec(
        key="rkd",
        name="Regression Kink Design",
        estimand_latex=(
            r"\tau_{\mathrm{RKD}} = "
            r"\frac{m'_{+}(0) - m'_{-}(0)}{b'_{+}(0) - b'_{-}(0)}"
        ),
        estimator_latex=(
            r"\hat m'_{\pm}(0)\text{: local-polynomial one-sided slopes "
            r"within bandwidth } h;\quad "
            r"\hat\tau = \frac{\hat m'_{+} - \hat m'_{-}}{b'_{+} - b'_{-}}"
        ),
        prose=(
            "Identifies a causal effect from a kink (slope change) in a "
            "deterministic policy rule: the ratio of the change in the slope of "
            "the outcome's conditional mean at the threshold to the change in "
            "the slope of the policy. Estimated by one-sided local-polynomial "
            "derivatives (Card, Lee, Pei & Weber)."
        ),
        assumptions=[
            "Smooth (continuously differentiable) density of the running "
            "variable at the kink.",
            "Conditional-mean derivative continuous absent the kink.",
            "Deterministic, kinked policy (sharp design) or a first-stage kink "
            "(fuzzy design).",
        ],
        aliases=["regression_kink", "kink_design", "rkd_sharp", "rkd_fuzzy"],
    ),
    MethodSpec(
        key="bunching",
        name="Bunching Estimator",
        estimand_latex=(
            r"e \;\;(\text{behavioral elasticity from " r"excess mass at a kink/notch})"
        ),
        estimator_latex=(
            r"\hat B = \sum_{z\in[z^{*}-\delta,\,z^{*}]} "
            r"\big(c_z - \hat c_z^{0}\big);\quad "
            r"\hat e \approx \frac{\hat b^{*}}{z^{*}\,\Delta\log(1-\tau)}"
        ),
        prose=(
            "Recovers a behavioral elasticity from the excess mass (bunching) "
            "that piles up just below a kink or notch in a budget set, relative "
            "to a smooth counterfactual density. Normalized excess bunching maps "
            "to the elasticity via the kink's tax-rate change (Saez)."
        ),
        assumptions=[
            "Smooth counterfactual density absent the kink/notch.",
            "Structural model linking bunching mass to the elasticity.",
            "Optimization frictions are absent or bounded.",
        ],
        aliases=["bunching_estimator", "notch", "kink_bunching", "general_bunching"],
    ),
    MethodSpec(
        key="augsynth",
        name="Augmented Synthetic Control",
        estimand_latex=r"\tau_{1t} = Y_{1t}(1) - Y_{1t}(0),\quad t > T_0",
        estimator_latex=(
            r"\hat\tau_t = \Big(Y_{1t} - \sum_j \hat\omega_j Y_{jt}\Big) "
            r"- \Big(\hat m_t(X_1) - \sum_j \hat\omega_j \hat m_t(X_j)\Big)"
        ),
        prose=(
            "Augments the synthetic-control gap with an outcome-model (e.g. "
            "ridge) bias-correction term that absorbs the residual pre-treatment "
            "imbalance the SC weights leave behind — de-biasing SC when a perfect "
            "convex fit is unavailable (Ben-Michael, Feller & Rothstein)."
        ),
        assumptions=[
            "Synthetic-control assumptions (no anticipation, no interference).",
            "The outcome model captures the residual imbalance / bias term.",
        ],
        aliases=["augmented_synthetic_control", "augmented_sc", "ridge_asc", "asc"],
    ),
    MethodSpec(
        key="mc_panel",
        name="Matrix Completion (Causal Panel)",
        estimand_latex=r"\tau = \mathbb{E}[Y_{it}(1) - Y_{it}(0) \mid D_{it}=1]",
        estimator_latex=(
            r"\hat L = \arg\min_{L} \sum_{(i,t)\in\mathcal{O}} "
            r"(Y_{it}-L_{it})^2 + \lambda\|L\|_{*};\quad "
            r"\hat\tau = \mathrm{mean}_{(i,t):D=1}\big(Y_{it} - \hat L_{it}\big)"
        ),
        prose=(
            "Imputes the missing untreated potential outcomes by completing a "
            "low-rank matrix of Y(0) via nuclear-norm-regularized regression on "
            "the observed (untreated) cells, then averages the gaps over treated "
            "cells (Athey, Bayati, Doudchenko, Imbens & Khosravi)."
        ),
        assumptions=[
            "Low-rank (factor) structure of the untreated potential outcomes.",
            "Treatment / missingness ignorable given the factor structure.",
        ],
        aliases=["matrix_completion", "mc_synth", "mc_nnm", "nuclear_norm"],
    ),
    MethodSpec(
        key="ivqr",
        name="IV Quantile Regression",
        estimand_latex=(
            r"\beta(\tau)\;\;(\text{structural } \tau\text{-quantile effect})"
        ),
        estimator_latex=(
            r"\Pr\!\big[Y \le D'\beta(\tau) + X'\gamma(\tau) \,\big|\, Z, X\big] "
            r"= \tau,\quad Z \text{ excluded from the structural quantile}"
        ),
        prose=(
            "Estimates quantile treatment effects under endogeneity: for each "
            "quantile, the structural coefficient is chosen so that the "
            "instrument carries no residual explanatory power in the conditional "
            "quantile of the structural equation (Chernozhukov & Hansen)."
        ),
        assumptions=[
            "Rank similarity / invariance across the treatment.",
            "Instrument relevance and exclusion.",
            "Monotonicity of the structural quantile function.",
        ],
        aliases=["iv_quantile", "ivqreg", "instrumental_quantile", "ch_iv_quantile"],
    ),
    MethodSpec(
        key="gmm",
        name="Generalized Method of Moments",
        estimand_latex=r"\theta_0:\;\; \mathbb{E}[g(W,\theta_0)] = 0",
        estimator_latex=(
            r"\hat\theta = \arg\min_{\theta}\; "
            r"\bar g_n(\theta)' W_n \bar g_n(\theta),\quad "
            r"\bar g_n(\theta) = \tfrac{1}{n}\sum_i g(W_i,\theta)"
        ),
        prose=(
            "Estimates a parameter by making the sample analogue of a set of "
            "population moment conditions as close to zero as possible in a "
            "weighted quadratic norm; the efficient weight is the inverse "
            "moment-covariance (Hansen)."
        ),
        assumptions=[
            "Correct moment conditions: E[g(W, theta_0)] = 0.",
            "Global identification (the moments uniquely pin down theta_0).",
            "Regularity (finite moment covariance, smoothness).",
        ],
        aliases=["generalized_method_of_moments", "gmm_estimator", "two_step_gmm"],
    ),
    MethodSpec(
        key="causal_forest",
        name="Causal Forest (Generalized Random Forest)",
        estimand_latex=r"\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]",
        estimator_latex=(
            r"\hat\tau(x): \sum_i \alpha_i(x)\Big[(Y_i - \hat m(X_i)) "
            r"- (W_i - \hat e(X_i))\,\tau(x)\Big] = 0"
        ),
        prose=(
            "Estimates heterogeneous (conditional) treatment effects with an "
            "honest random forest whose adaptive neighborhood weights "
            "alpha_i(x) localize a residual-on-residual moment condition; this "
            "yields pointwise CATE estimates with valid confidence intervals "
            "(Wager & Athey; Athey, Tibshirani & Wager)."
        ),
        assumptions=[
            "Unconfoundedness given X.",
            "Overlap: 0 < e(X) < 1.",
            "Honesty (separate samples for splitting and estimation).",
        ],
        aliases=[
            "causal_forests",
            "grf",
            "generalized_random_forest",
            "cforest",
        ],
    ),
    MethodSpec(
        key="metalearner",
        name="Meta-Learners (S/T/X-Learner)",
        estimand_latex=r"\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]",
        estimator_latex=(
            r"\hat\tau^{T}(x) = \hat\mu_1(x) - \hat\mu_0(x);\quad "
            r"\hat\tau^{X}(x) = g(x)\hat\tau_0(x) + (1-g(x))\hat\tau_1(x)"
        ),
        prose=(
            "Reduces CATE estimation to off-the-shelf supervised learning: the "
            "T-learner fits separate response surfaces and differences them; the "
            "X-learner imputes individual effects and combines them with a "
            "propensity weight g(x); the S-learner fits a single model with "
            "treatment as a feature (Kunzel, Sekhon, Bickel & Yu)."
        ),
        assumptions=[
            "Unconfoundedness given X.",
            "Overlap: 0 < e(X) < 1.",
        ],
        aliases=[
            "metalearners",
            "meta_learner",
            "s_learner",
            "t_learner",
            "x_learner",
            "xlearner",
        ],
    ),
    MethodSpec(
        key="r_learner",
        name="R-Learner (Quasi-Oracle CATE)",
        estimand_latex=r"\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X=x]",
        estimator_latex=(
            r"\hat\tau = \arg\min_{\tau}\;\frac{1}{n}\sum_i\Big["
            r"(Y_i - \hat m(X_i)) - (W_i - \hat e(X_i))\,\tau(X_i)\Big]^2"
        ),
        prose=(
            "Estimates the CATE by minimizing the Robinson-style R-loss on "
            "cross-fitted outcome and propensity residuals, so first-stage "
            "nuisance error enters only at second order (the quasi-oracle "
            "property of Nie & Wager)."
        ),
        assumptions=[
            "Unconfoundedness given X.",
            "Overlap: 0 < e(X) < 1.",
            "Cross-fitting / Neyman-orthogonality of the R-loss.",
        ],
        aliases=["rlearner", "r_loss", "robinson_learner"],
    ),
    MethodSpec(
        key="mr",
        name="Mendelian Randomization (IVW)",
        estimand_latex=(r"\beta \;\;(\text{causal effect of exposure on outcome})"),
        estimator_latex=(
            r"\hat\beta_{\mathrm{IVW}} = "
            r"\frac{\sum_j \gamma_{Xj}\,\gamma_{Yj}\,\sigma_{Yj}^{-2}}"
            r"{\sum_j \gamma_{Xj}^2\,\sigma_{Yj}^{-2}}"
        ),
        prose=(
            "Uses genetic variants as instruments for a modifiable exposure: the "
            "inverse-variance-weighted estimator combines the per-variant "
            "Wald ratios (outcome association gamma_Y over exposure association "
            "gamma_X) weighted by precision (Burgess, Butterworth & Thompson)."
        ),
        assumptions=[
            "Relevance: the variants associate with the exposure.",
            "Independence: the variants are unconfounded with the outcome.",
            "Exclusion: the variants affect the outcome only via the exposure.",
        ],
        aliases=["mendelian_randomization", "ivw", "two_sample_mr", "mr_egger"],
    ),
    MethodSpec(
        key="dose_response",
        name="Continuous-Treatment Dose-Response (GPS)",
        estimand_latex=r"\mu(t) = \mathbb{E}[Y(t)],\quad t \in \mathcal{T}",
        estimator_latex=(
            r"R = r(t, X)\;\;(\text{GPS});\quad "
            r"\hat\mu(t) = \mathbb{E}_X\big[\,\mathbb{E}(Y \mid T=t, r(t,X))\,\big]"
        ),
        prose=(
            "Estimates the dose-response curve for a continuous treatment using "
            "the generalized propensity score R = r(t, X): model the outcome "
            "given the dose and the GPS, then average over the covariate "
            "distribution at each dose level (Hirano & Imbens)."
        ),
        assumptions=[
            "Weak unconfoundedness given the generalized propensity score.",
            "Overlap across the dose range.",
            "Correctly specified GPS model.",
        ],
        aliases=[
            "gps_dose_response",
            "continuous_treatment_gps",
            "drf",
            "generalized_propensity_score",
        ],
    ),
    MethodSpec(
        key="cox",
        name="Cox Proportional Hazards",
        estimand_latex=(
            r"\mathrm{HR} = e^{\beta},\quad "
            r"\lambda(t\mid X) = \lambda_0(t)\,e^{X'\beta}"
        ),
        estimator_latex=(
            r"\hat\beta = \arg\max_{\beta}\;\prod_{i:\,\delta_i=1} "
            r"\frac{e^{X_i'\beta}}{\sum_{j\in R(t_i)} e^{X_j'\beta}}"
        ),
        prose=(
            "Models the hazard as a baseline hazard times an exponential index "
            "of covariates; the coefficients (log hazard ratios) are estimated "
            "by maximizing the partial likelihood over the risk sets, leaving "
            "the baseline hazard unspecified (Cox)."
        ),
        assumptions=[
            "Proportional hazards (time-invariant covariate effects).",
            "Non-informative (independent) censoring.",
            "Correct functional form of the linear index.",
        ],
        aliases=["cox_ph", "coxph", "proportional_hazards", "cox_regression"],
    ),
    MethodSpec(
        key="kaplan_meier",
        name="Kaplan-Meier (Product-Limit) Estimator",
        estimand_latex=r"S(t) = \Pr(T > t)",
        estimator_latex=(
            r"\hat S(t) = \prod_{t_i \le t}\left(1 - \frac{d_i}{n_i}\right)"
        ),
        prose=(
            "Nonparametric product-limit estimator of the survival function "
            "from right-censored data: at each observed event time t_i it "
            "multiplies in the conditional survival (1 - deaths d_i over at-risk "
            "n_i), giving a step-function estimate of S(t) (Kaplan & Meier)."
        ),
        assumptions=[
            "Non-informative (independent) censoring.",
            "Survival times are identically distributed.",
        ],
        aliases=["km", "product_limit", "survival_function", "km_estimator"],
    ),
    MethodSpec(
        key="lee_bounds",
        name="Lee Bounds (Sample-Selection Trimming)",
        estimand_latex=(
            r"\mathrm{ATE}_{\text{selected}} \in "
            r"[\underline{\tau},\ \overline{\tau}]"
        ),
        estimator_latex=(
            r"p_0 = \frac{s_1 - s_0}{s_1};\quad "
            r"\overline{\tau},\underline{\tau} = "
            r"\mathbb{E}[Y\mid D{=}1,\text{trim } p_0\text{ tail}] - "
            r"\mathbb{E}[Y\mid D{=}0]"
        ),
        prose=(
            "Sharp bounds on the treatment effect for the always-selected "
            "subpopulation under sample selection: trim the over-selected "
            "treatment arm by the difference in selection rates p_0 from each "
            "tail, then contrast the trimmed means against the control (Lee)."
        ),
        assumptions=[
            "Monotonicity: treatment moves selection in one direction.",
            "Random assignment of treatment.",
            "SUTVA (no interference).",
        ],
        aliases=["lee_trimming", "trimming_bounds", "lee_2009"],
    ),
    MethodSpec(
        key="policy_tree",
        name="Policy Learning (Optimal Assignment Rule)",
        estimand_latex=(r"\pi^{*} = \arg\max_{\pi \in \Pi}\;\mathbb{E}[Y(\pi(X))]"),
        estimator_latex=(
            r"\hat\pi = \arg\max_{\pi \in \Pi}\;\frac{1}{n}\sum_i "
            r"\big(2\pi(X_i)-1\big)\,\hat\Gamma_i,\quad "
            r"\hat\Gamma_i = \text{AIPW score}"
        ),
        prose=(
            "Learns a treatment-assignment rule from a constrained policy class "
            "(e.g. a shallow decision tree) by maximizing a doubly-robust "
            "estimate of policy value, with regret guarantees relative to the "
            "best in-class rule (Athey & Wager)."
        ),
        assumptions=[
            "Unconfoundedness given X.",
            "Overlap: 0 < e(X) < 1.",
            "A restricted (finite-complexity) policy class.",
        ],
        aliases=["policy_learning", "policytree", "optimal_policy", "policy_value"],
    ),
    MethodSpec(
        key="honest_did",
        name="Honest DiD (Robust to Parallel-Trends Violations)",
        estimand_latex=(
            r"\theta \in \mathcal{S}(\bar M)\;\;" r"(\text{robust identified set})"
        ),
        estimator_latex=(
            r"\delta_{\text{post}} \in \Delta(\bar M);\quad "
            r"\mathcal{C}_{1-\alpha} = "
            r"\{\theta : \text{feasible under } \Delta(\bar M)\}"
        ),
        prose=(
            "Relaxes exact parallel trends to a bounded class of violations — "
            "smoothness restrictions or magnitudes relative to the observed "
            "pre-trends — and reports the robust confidence set and the "
            "breakdown value M-bar at which significance is lost (Rambachan & "
            "Roth)."
        ),
        assumptions=[
            "Post-treatment trend violation lies in the chosen restriction set "
            "Delta(M-bar).",
            "Valid event-study estimates and their covariance.",
        ],
        aliases=["rambachan_roth", "robust_did", "honest_parallel_trends"],
    ),
    MethodSpec(
        key="oster",
        name="Oster Bounds (Coefficient Stability)",
        estimand_latex=(
            r"\beta^{*}(\delta, R_{\max})\;\;" r"(\text{bias-adjusted effect})"
        ),
        estimator_latex=(
            r"\beta^{*} \approx \tilde\beta - \delta\,"
            r"(\mathring\beta - \tilde\beta)\,"
            r"\frac{R_{\max} - \tilde R}{\tilde R - \mathring R};\quad "
            r"\delta^{*}: \beta^{*}=0"
        ),
        prose=(
            "Bounds omitted-variable bias by assuming selection on unobservables "
            "is proportional (coefficient delta) to selection on observables, "
            "scaled to a maximum R-squared; reports the bias-adjusted effect and "
            "the delta that would explain the result away (Oster)."
        ),
        assumptions=[
            "Proportional selection (delta) on observables vs unobservables.",
            "A posited maximum R-squared (R_max).",
        ],
        aliases=["oster_bounds", "oster_delta", "coefficient_stability"],
    ),
    MethodSpec(
        key="wild_cluster_bootstrap",
        name="Wild Cluster Bootstrap",
        estimand_latex=(r"\text{cluster-robust CI / } p\text{-value for } \beta"),
        estimator_latex=(
            r"y_g^{*} = X_g\hat\beta_{H_0} + \hat\varepsilon_g\,w_g,\;"
            r"w_g \in \{-1,+1\};\quad "
            r"\{t^{*(b)}\}\text{ build the null distribution}"
        ),
        prose=(
            "Improves inference with few clusters by resampling cluster "
            "residuals with random signs (Rademacher weights) under the imposed "
            "null, recomputing the cluster-robust t-statistic to form an "
            "accurate bootstrap distribution where cluster-robust asymptotics "
            "fail (Cameron, Gelbach & Miller)."
        ),
        assumptions=[
            "Independence across clusters.",
            "The wild (sign-flip) bootstrap DGP approximates the error "
            "distribution.",
        ],
        aliases=[
            "wcb",
            "wild_bootstrap",
            "cgm_bootstrap",
            "wild_cluster_boot",
        ],
    ),
    MethodSpec(
        key="deepiv",
        name="Deep IV (Neural Instrumental Variables)",
        estimand_latex=(r"h(p,x) = \mathbb{E}[Y \mid \mathrm{do}(P=p), X=x]"),
        estimator_latex=(
            r"\text{(1) } \hat F(p\mid x,z);\quad "
            r"\text{(2) } \hat h = \arg\min_h \sum_i "
            r"\Big(Y_i - \int h(p,X_i)\,d\hat F(p\mid X_i,Z_i)\Big)^2"
        ),
        prose=(
            "Two-stage neural IV: first estimate the conditional treatment "
            "density given instruments and covariates with a mixture-density "
            "network, then fit the structural response function h by minimizing "
            "the integrated prediction loss against it (Hartford, Lewis, "
            "Leyton-Brown & Taddy)."
        ),
        assumptions=[
            "Instrument validity (relevance, exclusion, independence).",
            "The structural function lies in the neural hypothesis class.",
        ],
        aliases=["deep_iv", "neural_iv"],
    ),
    MethodSpec(
        key="interference",
        name="Causal Inference under Interference (Partial)",
        estimand_latex=(
            r"\mathrm{DE}(\alpha),\ \mathrm{IE}(\alpha_1,\alpha_0)\;\;"
            r"(\text{direct \& indirect effects})"
        ),
        estimator_latex=(
            r"\bar Y(\alpha) = \frac{1}{N}\sum_g \bar Y_g(\alpha);\quad "
            r"\widehat{\mathrm{DE}} = \bar Y(1;\alpha) - \bar Y(0;\alpha)"
        ),
        prose=(
            "Defines and estimates direct, indirect, total, and overall effects "
            "under partial interference (units interfere within groups but not "
            "across) via group-level average potential outcomes at coverage "
            "level alpha (Hudgens & Halloran)."
        ),
        assumptions=[
            "Partial interference: no interference across groups.",
            "A specified (counterfactual) treatment-allocation policy alpha.",
            "Stratified/two-stage randomization for identification.",
        ],
        aliases=["partial_interference", "spillover", "hudgens_halloran"],
    ),
    MethodSpec(
        key="msm",
        name="Marginal Structural Model (IPTW)",
        estimand_latex=(r"\mathbb{E}[Y(\bar a)] = g(\bar a;\beta)"),
        estimator_latex=(
            r"SW_i = \prod_t \frac{\Pr(A_t\mid \bar A_{t-1})}"
            r"{\Pr(A_t\mid \bar A_{t-1}, \bar L_t)};\quad "
            r"\hat\beta:\ \text{weighted regression of } Y \text{ on } \bar a"
        ),
        prose=(
            "Models the marginal mean of the outcome under a treatment history "
            "and fits it by inverse-probability-of-treatment weighting with "
            "stabilized weights — adjusting for time-varying confounders that "
            "are also mediators, which standard regression cannot (Robins, "
            "Hernán & Brumback)."
        ),
        assumptions=[
            "Sequential exchangeability (no unmeasured time-varying " "confounding).",
            "Positivity at each time point.",
            "Correctly specified treatment (weight) models.",
        ],
        aliases=["marginal_structural_model", "iptw", "msm_ipw"],
    ),
    MethodSpec(
        key="network_exposure",
        name="Causal Effects under Network Interference",
        estimand_latex=(
            r"\tau(a,a') = \mathbb{E}\big[Y_i(f_i{=}a) - Y_i(f_i{=}a')\big]"
        ),
        estimator_latex=(
            r"\hat\mu(a) = \frac{1}{N}\sum_i "
            r"\frac{\mathbf{1}\{f_i = a\}\,Y_i}{\pi_i(a)}\;\;"
            r"(\text{Horvitz-Thompson over exposure } f_i)"
        ),
        prose=(
            "Generalizes causal estimands to arbitrary network interference via "
            "an exposure mapping f_i that summarizes a unit's treatment "
            "neighborhood, then estimates exposure-specific means by "
            "inverse-probability (Horvitz-Thompson) weighting on the exposure "
            "probabilities (Aronow & Samii)."
        ),
        assumptions=[
            "A correctly specified exposure mapping.",
            "Known / estimable exposure probabilities from the design.",
            "Positivity of each exposure condition.",
        ],
        aliases=["aronow_samii", "network_interference", "exposure_mapping"],
    ),
    MethodSpec(
        key="stacked_did",
        name="Stacked (Event-Specific) DiD",
        estimand_latex=r"\tau\;\;(\text{clean-comparison ATT})",
        estimator_latex=(
            r"\text{stack per-event sub-experiments (treated + "
            r"clean controls),}\ \text{then a single TWFE on the stack:}\ "
            r"Y = \alpha_{i,e} + \lambda_{t,e} + \tau D + \varepsilon"
        ),
        prose=(
            "Builds a separate sub-experiment around each treatment event — the "
            "cohort plus not-yet/never-treated clean controls in an event-time "
            "window — stacks them, and runs one TWFE with event-specific fixed "
            "effects, avoiding the forbidden comparisons of pooled TWFE (Cengiz, "
            "Dube, Lindner & Zipperer)."
        ),
        assumptions=[
            "Parallel trends within each event-specific sub-experiment.",
            "No anticipation; clean controls per event window.",
        ],
        aliases=["stacked_regression_did", "stacked_event_study", "cengiz"],
    ),
    MethodSpec(
        key="surrogate",
        name="Surrogate Index",
        estimand_latex=(
            r"\tau_{\text{long}} = \mathbb{E}[Y_{\text{long}}(1) - "
            r"Y_{\text{long}}(0)]"
        ),
        estimator_latex=(
            r"S_i = \mathbb{E}[Y_{\text{long}}\mid \text{surrogates } "
            r"X_i]\ (\text{obs. sample});\quad "
            r"\hat\tau = \mathbb{E}[S\mid D{=}1] - \mathbb{E}[S\mid D{=}0]"
        ),
        prose=(
            "Estimates a long-term treatment effect from short-term surrogates "
            "by building a surrogate index — the predicted long-term outcome "
            "given the surrogates, learned in an observational sample — and "
            "contrasting it across arms in the experimental sample (Athey, "
            "Chetty, Imbens & Kang)."
        ),
        assumptions=[
            "Surrogacy: surrogates fully mediate treatment's effect on the "
            "long-term outcome.",
            "Comparability of the surrogate-outcome link across samples.",
        ],
        aliases=["surrogate_index", "surrogate_outcome", "long_term_effect"],
    ),
    MethodSpec(
        key="oaxaca",
        name="Blinder-Oaxaca Decomposition",
        estimand_latex=(
            r"\Delta = \underbrace{(\bar X_A - \bar X_B)'\beta_B}"
            r"_{\text{explained}} + \underbrace{\bar X_A'(\beta_A - \beta_B)}"
            r"_{\text{unexplained}}"
        ),
        estimator_latex=(
            r"\widehat\Delta = (\bar X_A - \bar X_B)'\hat\beta_B "
            r"+ \bar X_A'(\hat\beta_A - \hat\beta_B)"
        ),
        prose=(
            "Decomposes the mean outcome gap between two groups into an "
            "explained part (differences in covariates, valued at reference "
            "returns) and an unexplained part (differences in coefficients), "
            "using group-specific OLS (Oaxaca; Blinder)."
        ),
        assumptions=[
            "Correct linear specification within each group.",
            "A chosen reference (non-discriminatory) coefficient vector.",
            "Ignorability of the linear index.",
        ],
        aliases=["blinder_oaxaca", "oaxaca_blinder", "wage_decomposition"],
    ),
    MethodSpec(
        key="rif_decomposition",
        name="RIF / Unconditional Quantile Regression",
        estimand_latex=(
            r"\mathrm{RIF}(y;q_\tau) = q_\tau + "
            r"\frac{\tau - \mathbf{1}\{y \le q_\tau\}}{f_Y(q_\tau)}"
        ),
        estimator_latex=(
            r"\hat\gamma = \arg\min_\gamma \sum_i "
            r"\big(\widehat{\mathrm{RIF}}(Y_i;\nu) - X_i'\gamma\big)^2\;\;"
            r"(\text{then Oaxaca on the RIF})"
        ),
        prose=(
            "Replaces the outcome with the recentered influence function of a "
            "distributional statistic (e.g. a quantile or the Gini), so an "
            "OLS-style regression recovers the effect of covariates on the "
            "unconditional statistic and enables detailed decompositions (Firpo, "
            "Fortin & Lemieux)."
        ),
        assumptions=[
            "The distributional statistic admits an influence function.",
            "Standard OLS conditions on the RIF-transformed outcome.",
        ],
        aliases=["rif", "rifreg", "unconditional_quantile", "ffl_decompose"],
    ),
    MethodSpec(
        key="dfl_decompose",
        name="DiNardo-Fortin-Lemieux Reweighting",
        estimand_latex=(
            r"\psi(x) = \frac{\Pr(A\mid x)}{\Pr(B\mid x)}\cdot" r"\frac{\Pr(B)}{\Pr(A)}"
        ),
        estimator_latex=(
            r"\hat f^{C}_{Y}(y) = \int f_{Y\mid X}(y\mid x,B)\,"
            r"\hat\psi(x)\,dF_X(x\mid B)"
        ),
        prose=(
            "Semiparametric reweighting: constructs a counterfactual outcome "
            "density by reweighting one group to match another's covariate "
            "distribution, decomposing distributional differences into "
            "composition and structure effects (DiNardo, Fortin & Lemieux)."
        ),
        assumptions=[
            "Overlap / common support of covariates across groups.",
            "Ignorability: conditional outcome distribution is group-invariant.",
        ],
        aliases=["dfl", "dinardo_fortin_lemieux", "reweighting_decomposition"],
    ),
    MethodSpec(
        key="gelbach",
        name="Gelbach Conditional Decomposition",
        estimand_latex=(
            r"\hat\beta_{\text{base}} - \hat\beta_{\text{full}} " r"= \sum_k \delta_k"
        ),
        estimator_latex=(
            r"\delta_k = \hat\Gamma_k'\,\hat\beta_{\text{full},k},\quad "
            r"\hat\Gamma_k: \text{ regress covariate group } k "
            r"\text{ on the base regressors}"
        ),
        prose=(
            "Attributes the change in a coefficient when controls are added to "
            "each group of added covariates, using the omitted-variable-bias "
            "formula — giving an order-invariant, unambiguous decomposition "
            "(Gelbach)."
        ),
        assumptions=[
            "Linear base and full models.",
            "The full model is the reference specification.",
        ],
        aliases=[
            "gelbach_decomposition",
            "conditional_decomposition",
            "ovb_decomposition",
        ],
    ),
    MethodSpec(
        key="twoway_cluster",
        name="Multiway Cluster-Robust Variance",
        estimand_latex=(
            r"\mathrm{Var}(\hat\beta)\;\;(\text{robust to clustering "
            r"on } G \text{ and } H)"
        ),
        estimator_latex=(r"\hat V = \hat V_G + \hat V_H - \hat V_{G\cap H}"),
        prose=(
            "Variance estimator robust to correlation along two (or more) "
            "non-nested clustering dimensions at once: sum the one-way "
            "cluster-robust variances and subtract the intersection (Cameron, "
            "Gelbach & Miller)."
        ),
        assumptions=[
            "Independence across the intersection cells.",
            "The clustering dimensions capture the dependence structure.",
        ],
        aliases=["multiway_cluster", "two_way_cluster", "cgm_multiway"],
    ),
    MethodSpec(
        key="kitagawa",
        name="Kitagawa Rate Decomposition",
        estimand_latex=(r"\Delta = \Delta_{\text{rate}} + \Delta_{\text{composition}}"),
        estimator_latex=(
            r"\Delta = \sum_i (r^A_i - r^B_i)\tfrac{w^A_i + w^B_i}{2} "
            r"+ \sum_i (w^A_i - w^B_i)\tfrac{r^A_i + r^B_i}{2}"
        ),
        prose=(
            "Decomposes the difference between two aggregate rates into a "
            "component from differing category-specific rates and a component "
            "from differing composition — the demographic-standardization "
            "precursor to Oaxaca (Kitagawa)."
        ),
        assumptions=[
            "Well-defined, mutually exclusive categories.",
            "Category-specific rates and weights are observed for both groups.",
        ],
        aliases=["kitagawa_decompose", "rate_decomposition"],
    ),
]


# Build the lookup index: canonical key + every alias -> spec.
_INDEX: Dict[str, MethodSpec] = {}
for _spec in _SPECS:
    _INDEX[_spec.key] = _spec
    for _alias in _spec.aliases:
        _INDEX.setdefault(_alias, _spec)


def _resolve_spec(result: Any) -> Optional[MethodSpec]:
    """Resolve a result to its MethodSpec using the same logic as ``cite()``.

    Tries the explicit ``_citation_key``, then the normalized ``method`` /
    ``model_info`` hints, then a substring fallback. Returns ``None`` when no
    spec matches (caller emits a registered-placeholder rather than guessing).
    """
    candidates: List[str] = []
    ck = getattr(result, "_citation_key", None)
    if ck:
        candidates.append(str(ck))
    model_info = getattr(result, "model_info", None) or {}
    for hint_key in ("citation_key", "estimator", "model_type", "method"):
        hint = model_info.get(hint_key) if isinstance(model_info, dict) else None
        if hint:
            candidates.append(str(hint))
    method = getattr(result, "method", None)
    if method:
        candidates.append(str(method))

    norm = [c.lower().replace(" ", "_").replace("-", "_") for c in candidates]
    # 1. Exact match.
    for c in norm:
        if c in _INDEX:
            return _INDEX[c]
    # 2. Whole-token match (handles "synthetic_control_method" -> alias token).
    for c in norm:
        for tok in c.split("_"):
            if tok and tok in _INDEX:
                return _INDEX[tok]
    # 3. Guarded substring fallback. Only keys of length >= 4 may match as a
    #    substring, so short aliases ("sc", "rd", "iv", "did") can never fire
    #    on incidental matches like the "sc" inside "ob-sc-ure".
    for c in norm:
        for key, spec in _INDEX.items():
            if len(key) >= 4 and (key in c or c in key):
                return spec
    return None


def _inference_lines(result: Any) -> List[str]:
    """Human-readable inference facts read off the *fitted* object."""
    lines: List[str] = []
    model_info = getattr(result, "model_info", None)
    mi: Dict[str, Any] = model_info if isinstance(model_info, dict) else {}

    se_method = mi.get("se_method") or mi.get("inference_method") or mi.get("vcov")
    if se_method:
        lines.append(f"Standard errors: {se_method}.")

    cluster = mi.get("cluster_var") or mi.get("cluster")
    if cluster:
        if isinstance(cluster, (list, tuple)):
            cluster = ", ".join(str(c) for c in cluster)
        lines.append(f"Clustered by: {cluster}.")

    n_boot = (
        mi.get("n_boot")
        or mi.get("n_bootstrap")
        or mi.get("n_boot_valid")
        or mi.get("n_boot_effective")
    )
    if n_boot:
        lines.append(f"Bootstrap replications: {n_boot}.")

    bandwidth = mi.get("bandwidth_h") or mi.get("bandwidth")
    if bandwidth is not None:
        try:
            lines.append(f"Bandwidth (h): {float(bandwidth):.4g}.")
        except (TypeError, ValueError):
            lines.append(f"Bandwidth (h): {bandwidth}.")

    fsf = mi.get("first_stage_f") or mi.get("weak_iv_f")
    if fsf is not None:
        try:
            lines.append(f"First-stage F: {float(fsf):.2f}.")
        except (TypeError, ValueError):
            lines.append(f"First-stage F: {fsf}.")

    kernel = mi.get("kernel") or mi.get("kernel_type")
    if kernel:
        lines.append(f"Kernel: {kernel}.")

    # Universal facts straight off the result object.
    se = getattr(result, "se", None)
    if se is not None:
        try:
            lines.append(f"Point SE: {float(se):.4g}.")
        except (TypeError, ValueError):
            pass
    ci = getattr(result, "ci", None)
    alpha = getattr(result, "alpha", None)
    if ci is not None and isinstance(ci, (list, tuple)) and len(ci) == 2:
        try:
            level = int(round((1 - float(alpha)) * 100)) if alpha is not None else 95
            lines.append(f"{level}% CI: [{float(ci[0]):.4g}, {float(ci[1]):.4g}].")
        except (TypeError, ValueError):
            pass
    return lines


def _method_identity(result: Any) -> str:
    """Best available estimator identity recorded on the result."""
    method = getattr(result, "method", None)
    if not method:
        mi = getattr(result, "model_info", None)
        if isinstance(mi, dict):
            method = mi.get("method") or mi.get("model_type")
    return str(method) if method else "unknown"


def _provenance_line(result: Any, spec: Optional[MethodSpec]) -> str:
    """One-line, runtime-honest trace: package version + estimator + spec.

    Closes the "exact code path" leg of the traceability triple (formula +
    verified citation + code path). Only facts available at runtime are
    reported — the installed StatsPAI version and the estimator identity the
    result carries — never a fabricated git SHA.
    """
    _ver: Optional[str]
    try:
        from .. import __version__

        _ver = __version__
    except (AttributeError, ImportError):  # version is best-effort metadata
        _ver = None
    ver = f"StatsPAI v{_ver}" if _ver else "StatsPAI"
    ident = _method_identity(result)
    if spec is not None:
        tail = f"methods spec '{spec.key}'"
    else:
        tail = "no methods spec registered"
    return f"Produced by {ver}; estimator '{ident}' → {tail}."


def _math(latex: str, fmt: str) -> str:
    """Wrap a stored (delimiter-free) LaTeX formula for the target format."""
    if fmt == "latex":
        return f"\\[\n{latex}\n\\]"
    if fmt == "markdown":
        return f"$$\n{latex}\n$$"
    # text
    return f"    {latex}"


def _one_section(
    result: Any,
    fmt: str,
    *,
    include_assumptions: bool,
    include_diagnostics: bool,
    include_citation: bool,
    include_provenance: bool,
) -> str:
    spec = _resolve_spec(result)
    method_name = getattr(result, "method", None) or "Estimator"
    estimand = getattr(result, "estimand", None)

    parts: List[str] = []

    # --- heading -------------------------------------------------------
    heading = spec.name if spec is not None else str(method_name)
    if fmt == "latex":
        parts.append(f"\\subsection*{{{heading}}}")
    elif fmt == "markdown":
        parts.append(f"### {heading}")
    else:
        parts.append(heading)
        parts.append("=" * len(heading))

    if spec is None:
        note = (
            f"(Methods text not yet registered for method '{method_name}'. "
            "Estimand / inference reported from the fitted result below.)"
        )
        parts.append(note)
    else:
        parts.append(spec.prose)
        # Estimand
        lbl = "Estimand" if estimand is None else f"Estimand ({estimand})"
        if fmt == "markdown":
            parts.append(f"**{lbl}:**")
        elif fmt == "latex":
            parts.append(f"\\paragraph{{{lbl}.}}")
        else:
            parts.append(f"{lbl}:")
        parts.append(_math(spec.estimand_latex, fmt))
        # Estimator
        if fmt == "markdown":
            parts.append("**Estimator:**")
        elif fmt == "latex":
            parts.append("\\paragraph{Estimator.}")
        else:
            parts.append("Estimator:")
        parts.append(_math(spec.estimator_latex, fmt))

    # --- assumptions ---------------------------------------------------
    if include_assumptions and spec is not None and spec.assumptions:
        if fmt == "markdown":
            parts.append("**Identifying assumptions:**")
            parts.append("\n".join(f"- {a}" for a in spec.assumptions))
        elif fmt == "latex":
            parts.append("\\paragraph{Identifying assumptions.}")
            parts.append("\\begin{itemize}")
            parts.extend(f"  \\item {a}" for a in spec.assumptions)
            parts.append("\\end{itemize}")
        else:
            parts.append("Identifying assumptions:")
            parts.extend(f"  - {a}" for a in spec.assumptions)

    # --- inference -----------------------------------------------------
    if include_diagnostics:
        inf = _inference_lines(result)
        if inf:
            if fmt == "markdown":
                parts.append("**Inference (as fitted):**")
                parts.append("\n".join(f"- {x}" for x in inf))
            elif fmt == "latex":
                parts.append("\\paragraph{Inference (as fitted).}")
                parts.append("\\begin{itemize}")
                parts.extend(f"  \\item {x}" for x in inf)
                parts.append("\\end{itemize}")
            else:
                parts.append("Inference (as fitted):")
                parts.extend(f"  - {x}" for x in inf)

    # --- citation ------------------------------------------------------
    if include_citation:
        cite_fn = getattr(result, "cite", None)
        if callable(cite_fn):
            try:
                apa = cite_fn(format="apa")
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                apa = None
            if apa and not str(apa).lstrip().startswith("%"):
                if fmt == "markdown":
                    parts.append(f"**Reference:** {apa}")
                elif fmt == "latex":
                    parts.append(f"\\paragraph{{Reference.}} {apa}")
                else:
                    parts.append(f"Reference: {apa}")

    # --- provenance ----------------------------------------------------
    if include_provenance:
        prov = _provenance_line(result, spec)
        if fmt == "markdown":
            parts.append(f"*{prov}*")
        elif fmt == "latex":
            parts.append(f"\\paragraph{{Provenance.}} {prov}")
        else:
            parts.append(prov)

    sep = "\n\n" if fmt != "text" else "\n"
    return sep.join(parts)


def methods_appendix(
    results: Union[Any, Sequence[Any]],
    *,
    format: str = "latex",
    include_assumptions: bool = True,
    include_diagnostics: bool = True,
    include_citation: bool = True,
    include_provenance: bool = True,
) -> str:
    """Generate a referee-grade *Methods and Formulas* appendix for results.

    For each fitted result, emits the estimand and estimator definitions
    (verified LaTeX from the cited source), the identifying assumptions, the
    inference actually used (read off the fitted object), and the canonical
    citation. Multiple results sharing the same estimator family are emitted
    once. Unregistered methods degrade to an explicit placeholder — never an
    invented formula (CLAUDE.md §10).

    Parameters
    ----------
    results : CausalResult or sequence of CausalResult
        One or more fitted result objects exposing ``method`` / ``model_info``
        / ``cite``.
    format : {"latex", "markdown", "text"}, default ``"latex"``
        Output format. ``"latex"`` emits ``\\subsection*`` blocks with display
        math; ``"markdown"`` emits ``###`` headings with ``$$`` math;
        ``"text"`` emits a plain-text rendering.
    include_assumptions : bool, default True
        Include the identifying-assumptions list.
    include_diagnostics : bool, default True
        Include the inference block (SE method, clustering, bandwidth, F, CI).
    include_citation : bool, default True
        Append the APA-style reference from ``result.cite()``.
    include_provenance : bool, default True
        Append a one-line provenance trace (StatsPAI version + estimator
        identity + methods-spec key) — the "exact code path" leg of the
        formula / citation / code-path traceability triple.

    Returns
    -------
    str
        The assembled appendix.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> ids = np.repeat(np.arange(40), 2)
    >>> time = np.tile([0, 1], 40)
    >>> treat = (ids >= 20).astype(int)
    >>> y = 1.0 + 2.0 * treat * time + rng.normal(size=len(ids))
    >>> df = pd.DataFrame({"id": ids, "time": time, "treat": treat, "y": y})
    >>> res = sp.did(df, y="y", treat="treat", time="time", id="id")
    >>> txt = sp.methods_appendix(res, format="text")
    >>> "Estimand" in txt
    True
    """
    if format not in ("latex", "markdown", "text"):
        raise ValueError(
            f"format must be 'latex', 'markdown' or 'text'; got {format!r}"
        )

    if isinstance(results, (list, tuple)):
        items = list(results)
    else:
        items = [results]
    if not items:
        raise ValueError("methods_appendix requires at least one result.")

    sections: List[str] = []
    seen: set = set()
    for res in items:
        spec = _resolve_spec(res)
        dedup_key = spec.key if spec is not None else id(res)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        sections.append(
            _one_section(
                res,
                format,
                include_assumptions=include_assumptions,
                include_diagnostics=include_diagnostics,
                include_citation=include_citation,
                include_provenance=include_provenance,
            )
        )

    if format == "latex":
        body = "\n\n".join(sections)
        return "\\section*{Methods and Formulas}\n\n" + body
    if format == "markdown":
        body = "\n\n".join(sections)
        return "## Methods and Formulas\n\n" + body
    body = "\n\n".join(sections)
    return "Methods and Formulas\n" + "-" * 20 + "\n\n" + body
