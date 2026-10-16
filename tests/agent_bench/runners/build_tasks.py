"""CausalAgentBench v2 task set: 50 prompts with named, gradable answers.

Version 1 asked open questions ("Report the ATT") and graded against
hand-written ranges computed on no data. Version 2 keeps the 50 research
questions and the L1 / L2 / L3 split, and adds to every prompt:

* the data file(s) the trial receives (``data/MANIFEST.json``);
* ``report``: the named values the agent must return through
  ``submit_answer``, each with a one-line definition, so grading needs no
  reading of free text;
* ``headline``: the value task success is scored on;
* the ambiguities of v1 removed (v1's L1-12 asked for a "negative-weight
  share", which the Goodman-Bacon decomposition does not have; L1-03 asked
  for a "doubly-robust outcome regression" estimator).

Gold values are not here; ``build_golds.py`` computes them with reference
implementations on the frozen files.

Usage::

    python tests/agent_bench/runners/build_tasks.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
TASKS = HERE.parent / "prompts" / "prompts.json"

CARD_X = "exper, expersq, black, south, smsa"
MP = (
    "mpdta.csv is a county-year panel: lemp is log employment, first_treat the "
    "year a county was first treated (0 = never treated), year the period and "
    "countyreal the county."
)

#: (id, level, files, question, {report name: definition}, headline, estimator hint)
RAW: List[tuple] = [
    # ---------------------------------------------------------------- L1
    (
        "L1-01",
        "L1",
        ["card.csv"],
        f"Regress lwage on educ, {CARD_X} by OLS with HC1 robust standard errors.",
        {"beta_educ": "coefficient on educ", "se_educ": "its HC1 standard error"},
        "beta_educ",
        "ols",
    ),
    (
        "L1-02",
        "L1",
        ["card.csv"],
        "Estimate the effect of educ on lwage by 2SLS, instrumenting educ with "
        f"nearc4; exogenous controls {CARD_X}. Use HC1 robust standard errors.",
        {"beta_educ": "2SLS coefficient on educ", "se_educ": "its HC1 standard error"},
        "beta_educ",
        "iv_2sls",
    ),
    (
        "L1-03",
        "L1",
        ["mpdta.csv"],
        f"{MP} Estimate the Callaway-Sant'Anna group-time ATTs with the doubly "
        "robust estimator and never-treated units as controls, then aggregate "
        "them to the simple overall ATT.",
        {"att": "simple aggregated ATT", "se": "its analytic standard error"},
        "att",
        "callaway_santanna",
    ),
    (
        "L1-04",
        "L1",
        ["mpdta.csv"],
        f"{MP} Estimate the Sun-Abraham interaction-weighted event study (units "
        "clustered by county) and report its aggregate post-treatment ATT.",
        {"att": "aggregate ATT over post-treatment event times"},
        "att",
        "sun_abraham",
    ),
    (
        "L1-05",
        "L1",
        ["lee.csv"],
        "lee.csv has margin (the running variable, cutoff 0) and voteshare_next. "
        "Estimate the sharp RD effect with a local-linear fit, triangular kernel "
        "and an MSE-optimal bandwidth, with robust bias-corrected inference.",
        {
            "tau_conventional": "conventional local-linear estimate",
            "tau_bias_corrected": "bias-corrected estimate",
            "se_robust": "robust standard error",
        },
        "tau_conventional",
        "rdrobust",
    ),
    (
        "L1-06",
        "L1",
        ["lee.csv"],
        "Test for manipulation of margin at the cutoff 0 with the "
        "Cattaneo-Jansson-Ma local-polynomial density test.",
        {"p_value": "p-value of the robust density-discontinuity test"},
        "p_value",
        "rddensity",
    ),
    (
        "L1-07",
        "L1",
        ["basque.csv"],
        "basque.csv has gdppc by region and year; the Basque Country is the "
        "region with treated == 1 from 1970. Estimate a classical synthetic "
        "control for it from the other regions, matching on pre-1970 gdppc.",
        {"avg_post_gap": "mean of (treated - synthetic) gdppc over 1970 and later"},
        "avg_post_gap",
        "synth",
    ),
    (
        "L1-08",
        "L1",
        ["nsw_dw.csv"],
        "Estimate the ATT of treat on re78 by 1:1 nearest-neighbour matching on "
        "a logit propensity score, with replacement; covariates age, education, "
        "black, hispanic, married, re74, re75.",
        {"att": "matching estimate of the ATT"},
        "att",
        "psm",
    ),
    (
        "L1-09",
        "L1",
        ["card.csv"],
        "Estimate the effect of educ on lwage with double/debiased ML, partially "
        "linear model, linear-regression nuisance learners and 5-fold "
        f"cross-fitting; controls {CARD_X}.",
        {"theta": "DML estimate of the educ coefficient", "se": "its standard error"},
        "theta",
        "dml_plr",
    ),
    (
        "L1-10",
        "L1",
        ["synth_survival.csv"],
        "Fit a Cox proportional hazards model of (time, event) on x1, x2 and "
        "treat, Efron ties.",
        {
            "beta_x1": "log hazard ratio of x1",
            "beta_x2": "log hazard ratio of x2",
            "beta_treat": "log hazard ratio of treat",
        },
        "beta_treat",
        "coxph",
    ),
    (
        "L1-11",
        "L1",
        ["mpdta.csv"],
        f"{MP} Estimate the Callaway-Sant'Anna event study (doubly robust, "
        "never-treated controls, universal base period), then compute the "
        "Rambachan-Roth honest 95% confidence interval for the effect at event "
        "time 0 under the smoothness restriction with M = 0.05.",
        {"ci_lower": "lower end of the honest CI", "ci_upper": "upper end"},
        "ci_lower",
        "honest_did",
    ),
    (
        "L1-12",
        "L1",
        ["mpdta.csv"],
        f"{MP} Define the treatment indicator D = 1 when first_treat > 0 and "
        "year >= first_treat. Run the Goodman-Bacon decomposition of the "
        "two-way fixed-effects regression of lemp on D.",
        {
            "beta_twfe": "TWFE coefficient on D",
            "weight_later_vs_earlier": "total weight on comparisons that use "
            "already-treated units as controls",
        },
        "beta_twfe",
        "bacon",
    ),
    (
        "L1-13",
        "L1",
        ["mpdta.csv"],
        f"{MP} Estimate Wooldridge's extended two-way fixed effects (ETWFE) model "
        "and report the simple overall ATT.",
        {"att": "simple overall ATT"},
        "att",
        "etwfe",
    ),
    (
        "L1-14",
        "L1",
        ["mpdta.csv"],
        f"{MP} Estimate the Borusyak-Jaravel-Spiess imputation estimator and "
        "report the overall ATT.",
        {"att": "overall ATT"},
        "att",
        "did_imputation",
    ),
    (
        "L1-15",
        "L1",
        ["california.csv"],
        "california.csv is a state-year panel of cigsale; California "
        "(treated == 1) is treated from 1989. Estimate synthetic "
        "difference-in-differences.",
        {"att": "SDID estimate of the ATT"},
        "att",
        "sdid",
    ),
    (
        "L1-16",
        "L1",
        [],
        "Compute the E-value for an observed risk ratio of 2.5 with 95% CI "
        "(1.8, 3.2).",
        {
            "evalue_estimate": "E-value of the point estimate",
            "evalue_ci": "E-value of the CI limit closer to 1",
        },
        "evalue_estimate",
        "evalue",
    ),
    (
        "L1-17",
        "L1",
        ["panel.csv"],
        "Fit a linear mixed model of y on 1 + x1 with a random intercept for gid "
        "(REML).",
        {
            "beta_intercept": "fixed intercept",
            "beta_x1": "fixed slope on x1",
            "icc": "intraclass correlation of the random intercept",
        },
        "beta_x1",
        "lmm",
    ),
    (
        "L1-18",
        "L1",
        ["logit_panel.csv"],
        "Fit a logistic GLMM of y on 1 + x1 with a random intercept for gid, "
        "Laplace approximation.",
        {"beta_intercept": "fixed intercept", "beta_x1": "fixed slope on x1"},
        "beta_x1",
        "glmm_logit",
    ),
    (
        "L1-19",
        "L1",
        ["sfa.csv"],
        "Fit a stochastic production frontier lny ~ lnk + lnl with normal noise "
        "and half-normal inefficiency.",
        {
            "beta_lnk": "frontier coefficient on lnk",
            "beta_lnl": "frontier coefficient on lnl",
            "mean_efficiency": "sample mean of Battese-Coelli E[exp(-u) | e]",
        },
        "beta_lnk",
        "frontier",
    ),
    (
        "L1-20",
        "L1",
        ["wage_gap.csv"],
        "Run a threefold Blinder-Oaxaca decomposition of log_wage on educ and "
        "exper between men (female == 0, group A) and women (female == 1, "
        "group B).",
        {
            "gap": "mean log_wage of men minus women",
            "endowments": "endowments component",
            "coefficients": "coefficients component",
            "interaction": "interaction component",
        },
        "endowments",
        "oaxaca",
    ),
    # ---------------------------------------------------------------- L2
    (
        "L2-01",
        "L2",
        ["card.csv"],
        "Education is endogenous because of unobserved ability. Estimate the "
        "causal effect of educ on lwage with an appropriate design using the "
        "data in card.csv (it includes a college-proximity indicator nearc4).",
        {
            "effect": "estimated effect of one year of educ on lwage",
            "se": "its standard error",
        },
        "effect",
        None,
    ),
    (
        "L2-02",
        "L2",
        ["mpdta.csv"],
        f"{MP} Counties were treated at different times; two-way fixed effects "
        "is biased under heterogeneous timing. Estimate the overall ATT with a "
        "heterogeneity-robust estimator.",
        {"att": "overall ATT", "se": "its standard error"},
        "att",
        None,
    ),
    (
        "L2-03",
        "L2",
        ["lee.csv"],
        "voteshare_next is the next-election Democratic vote share and margin "
        "the previous-election margin (0 = tie). Estimate the incumbency "
        "advantage with a quasi-experimental design.",
        {"effect": "estimated incumbency advantage in vote-share points"},
        "effect",
        None,
    ),
    (
        "L2-04",
        "L2",
        ["nsw_dw.csv"],
        "treat is participation in a job training programme and re78 earnings "
        "in 1978; the other columns are pre-treatment covariates. Estimate the "
        "ATT.",
        {"att": "estimated ATT on re78"},
        "att",
        None,
    ),
    (
        "L2-05",
        "L2",
        ["basque.csv"],
        "The Basque Country (treated == 1) experienced terrorism from 1970. "
        "Estimate the gdppc gap relative to a counterfactual built from the "
        "other regions.",
        {"avg_post_gap": "average annual gap over 1970 and later"},
        "avg_post_gap",
        None,
    ),
    (
        "L2-06",
        "L2",
        ["california.csv"],
        "California (treated == 1) raised tobacco taxes from 1989. Estimate the "
        "effect on per-capita cigarette sales (cigsale).",
        {"att": "average effect on cigsale over the treated years"},
        "att",
        None,
    ),
    (
        "L2-07",
        "L2",
        ["mpdta.csv"],
        f"{MP} Treatment varies at the county level, observations are "
        "county-years. Estimate the TWFE effect of being treated (treat == 1 "
        "and year >= first_treat) on lemp with appropriate inference.",
        {"effect": "TWFE coefficient", "se": "its standard error"},
        "effect",
        None,
    ),
    (
        "L2-08",
        "L2",
        ["trade.csv"],
        "Estimate the effect of x1 and x2 on y absorbing firm and year fixed "
        "effects (200,000 firm-year rows).",
        {"beta_x1": "coefficient on x1", "beta_x2": "coefficient on x2"},
        "beta_x1",
        None,
    ),
    (
        "L2-09",
        "L2",
        ["mpdta.csv"],
        f"{MP} With D = 1 when first_treat > 0 and year >= first_treat, "
        "decompose the TWFE coefficient of lemp on D into its 2x2 comparisons.",
        {"beta_twfe": "TWFE coefficient being decomposed"},
        "beta_twfe",
        None,
    ),
    (
        "L2-10",
        "L2",
        ["mpdta.csv"],
        f"{MP} Quantify how robust the effect at event time 0 is to violations "
        "of parallel trends: report the honest 95% CI under a smoothness bound "
        "M = 0.05 (Callaway-Sant'Anna event study, never-treated controls).",
        {"ci_lower": "lower end", "ci_upper": "upper end"},
        "ci_lower",
        None,
    ),
    (
        "L2-11",
        "L2",
        ["nsw_dw.csv"],
        "Regress re78 on treat and all other covariates by OLS, then quantify "
        "how strong an omitted confounder would have to be to explain away the "
        "treat coefficient.",
        {"rv_q": "robustness value (q = 1) of the treat coefficient"},
        "rv_q",
        None,
    ),
    (
        "L2-12",
        "L2",
        ["mediation.csv"],
        "treat was randomised; m is a mediator and x a pre-treatment covariate. "
        "Decompose the effect of treat on y into the indirect effect through m "
        "and the direct effect.",
        {"acme": "average causal mediation effect", "ade": "average direct effect"},
        "acme",
        None,
    ),
    (
        "L2-13",
        "L2",
        ["synth_survival.csv"],
        "time is a duration and event its indicator; x1, x2 and treat are "
        "covariates. Estimate the hazard-ratio coefficients.",
        {"beta_treat": "log hazard ratio of treat"},
        "beta_treat",
        None,
    ),
    (
        "L2-14",
        "L2",
        ["panel.csv"],
        "Observations are clustered within gid. Estimate the effect of x1 on y "
        "accounting for the within-group correlation.",
        {"beta_x1": "coefficient on x1"},
        "beta_x1",
        None,
    ),
    (
        "L2-15",
        "L2",
        ["logit_panel.csv"],
        "y is binary and observations are clustered within gid. Fit a model with "
        "x1 as fixed effect and a random intercept by gid.",
        {"beta_x1": "coefficient on x1 (log-odds scale)"},
        "beta_x1",
        None,
    ),
    (
        "L2-16",
        "L2",
        ["sfa.csv"],
        "Estimate the production function lny ~ lnk + lnl allowing for "
        "firm-level inefficiency.",
        {"beta_lnk": "elasticity on lnk", "beta_lnl": "elasticity on lnl"},
        "beta_lnk",
        None,
    ),
    (
        "L2-17",
        "L2",
        ["wage_gap.csv"],
        "Decompose the male-female gap in mean log_wage into a part explained by "
        "educ and exper and an unexplained part.",
        {"gap": "mean gap (men minus women)", "explained": "explained part"},
        "explained",
        None,
    ),
    (
        "L2-18",
        "L2",
        ["wage_gap.csv"],
        "With a reweighting (DiNardo-Fortin-Lemieux) approach, estimate the "
        "mean log_wage women would have with men's distribution of educ and "
        "exper.",
        {"counterfactual_mean": "reweighted mean log_wage of women"},
        "counterfactual_mean",
        None,
    ),
    (
        "L2-19",
        "L2",
        [],
        "An observational study reports a risk ratio of 2.5 with 95% CI "
        "(1.8, 3.2). How strong would unmeasured confounding have to be to "
        "explain away the estimate?",
        {"evalue_estimate": "E-value of the point estimate"},
        "evalue_estimate",
        None,
    ),
    (
        "L2-20",
        "L2",
        ["macro.csv"],
        "Fit a vector autoregression in y1 and y2 with two lags and a constant.",
        {
            "y1_on_L1_y1": "coefficient on the first lag of y1 in the y1 equation",
            "y2_on_L1_y1": "coefficient on the first lag of y1 in the y2 equation",
        },
        "y1_on_L1_y1",
        None,
    ),
    # ---------------------------------------------------------------- L3
    (
        "L3-01",
        "L3",
        ["card.csv"],
        "What is the causal effect of an additional year of schooling on log "
        "wages in card.csv? Run a complete analysis with diagnostics and at "
        "least one robustness check.",
        {
            "effect": "your preferred estimate",
            "first_stage_f": "first-stage F statistic (null if not IV)",
        },
        "effect",
        None,
    ),
    (
        "L3-02",
        "L3",
        ["mpdta.csv"],
        f"{MP} Estimate the effect of the policy on lemp. Justify the estimator, "
        "test pre-trends and assess sensitivity.",
        {
            "att": "overall ATT",
            "pretrend_p_value": "p-value of your pre-trend test",
        },
        "att",
        None,
    ),
    (
        "L3-03",
        "L3",
        ["lee.csv"],
        "Estimate the U.S. House incumbency advantage from voteshare_next and "
        "margin. Run a complete RD analysis including a manipulation test.",
        {
            "effect": "preferred RD estimate",
            "density_p_value": "p-value of the manipulation test",
        },
        "effect",
        None,
    ),
    (
        "L3-04",
        "L3",
        ["nsw_dw.csv"],
        "Estimate the ATT of training on re78 with three estimators: a "
        "regression, a matching estimator and an ML-based one.",
        {
            "att_regression": "regression estimate",
            "att_matching": "matching estimate",
            "att_ml": "ML-based estimate",
        },
        "att_matching",
        None,
    ),
    (
        "L3-05",
        "L3",
        ["basque.csv"],
        "Quantify the gdppc cost of terrorism in the Basque Country (treated == "
        "1, from 1970) with synthetic control and at least one robustness test.",
        {
            "avg_post_gap": "average annual gap over 1970 and later",
            "placebo_p_value": "p-value of your placebo / permutation test",
        },
        "avg_post_gap",
        None,
    ),
    (
        "L3-06",
        "L3",
        ["trade.csv"],
        "Estimate the effects of x1 and x2 on y with firm and year fixed effects "
        "and standard errors clustered by firm.",
        {"beta_x1": "coefficient on x1", "se_x1": "its firm-clustered standard error"},
        "beta_x1",
        None,
    ),
    (
        "L3-07",
        "L3",
        ["panel.csv"],
        "Fit a random-intercept model of y on x1 by gid and a fixed-effects "
        "model, and choose between them with a Hausman test.",
        {
            "hausman_p_value": "Hausman test p-value",
            "beta_x1": "x1 coefficient of the chosen model",
        },
        "beta_x1",
        None,
    ),
    (
        "L3-08",
        "L3",
        ["wage_gap.csv"],
        "Decompose the male-female log_wage gap with Blinder-Oaxaca (twofold) "
        "and with DFL reweighting.",
        {
            "explained_oaxaca": "explained part, Blinder-Oaxaca",
            "explained_dfl": "explained part, reweighting",
        },
        "explained_oaxaca",
        None,
    ),
    (
        "L3-09",
        "L3",
        ["synth_survival.csv"],
        "Estimate the effect of treat on time-to-event with a Cox model "
        "(controls x1, x2) and test the proportional-hazards assumption.",
        {
            "beta_treat": "log hazard ratio of treat",
            "ph_global_p_value": "global PH test p-value",
        },
        "beta_treat",
        None,
    ),
    (
        "L3-10",
        "L3",
        ["macro.csv"],
        "Estimate the response of y2 to a one-unit shock to y1 at horizon 1, "
        "from a VAR(2) with y1 ordered first and from local projections with "
        "two lags of both variables as controls.",
        {
            "irf_var_h1": "VAR (Cholesky, y1 first) response of y2 at h = 1 to a unit y1 shock",
            "irf_lp_h1": "local-projection response of y2 at h = 1",
        },
        "irf_lp_h1",
        None,
    ),
]


def build() -> List[Dict[str, Any]]:
    tasks = []
    for tid, level, files, question, report, headline, hint in RAW:
        lines = "; ".join(f"{k} ({v})" for k, v in report.items())
        tasks.append(
            {
                "id": tid,
                "level": level,
                "files": files,
                "question": question,
                "report": report,
                "headline": headline,
                "method_named": hint,
                "instruction": f"Report through submit_answer: {lines}.",
            }
        )
    return tasks


def main() -> None:
    tasks = build()
    assert len(tasks) == 50 and len({t["id"] for t in tasks}) == 50
    TASKS.write_text(
        json.dumps({"version": 2, "tasks": tasks}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(tasks)} tasks to {TASKS}")


if __name__ == "__main__":
    main()
