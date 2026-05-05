"""Generate the 50-prompt CausalAgentBench dataset and gold rubrics.

Prompts are stored in prompts/prompts.json; gold answers (point
estimate, expected estimator, diagnostic checklist) in
golds/golds.json. Both files are deterministic; re-running this
script regenerates the same 50 entries.

The split is L1 × 20 (method named) + L2 × 20 (method inferred) +
L3 × 10 (full workflow). All 50 use a calibrated replica that ships
with sp.datasets so an agent in a sandbox can `read_csv` without
needing internet access.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROMPTS = HERE.parent / "prompts" / "prompts.json"
GOLDS = HERE.parent / "golds" / "golds.json"
SANDBOX = HERE.parent / "sandbox"
PROMPTS.parent.mkdir(parents=True, exist_ok=True)
GOLDS.parent.mkdir(parents=True, exist_ok=True)
SANDBOX.mkdir(parents=True, exist_ok=True)


def make_prompts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts: list[dict[str, Any]] = []
    golds: list[dict[str, Any]] = []
    pid = 0

    # ----- L1 (direct: method named) -- 20 prompts -----------------------
    L1_specs = [
        # Card returns to schooling
        ("L1-01", "card.csv", "OLS", "ols",
         "Run OLS of lwage on educ, exper, expersq, black, south, smsa "
         "with HC1 robust SEs.",
         {"beta_educ": 0.110, "se_educ": 0.0042}, 0.075),
        ("L1-02", "card.csv", "2SLS-IV", "iv_2sls",
         "Run 2SLS for the effect of educ on lwage using nearc4 as the "
         "instrument; controls are exper, expersq, black, south, smsa. "
         "Report HC1 robust SEs.",
         {"beta_educ": 0.142, "se_educ": 0.019}, 0.132),
        # mpdta CS-DiD
        ("L1-03", "mpdta.csv", "Callaway-Santanna staggered DiD", "callaway_santanna",
         "Run Callaway-Sant'Anna staggered DiD on lemp, with first_treat as "
         "the cohort variable, year as time, countyreal as unit. Use the "
         "doubly-robust outcome regression estimator and the never-treated "
         "control group. Report the simple ATT.",
         {"simple_ATT": -0.033, "se": 0.003}, -0.0454),
        ("L1-04", "mpdta.csv", "Sun-Abraham event study", "sun_abraham",
         "Run a Sun-Abraham event study on lemp, with first_treat as the "
         "cohort variable, year as time, countyreal as unit. Report the "
         "weighted-average post-treatment ATT.",
         {"weighted_avg_ATT": -0.034, "se": 0.009}, None),
        # Lee 2008 RD
        ("L1-05", "lee.csv", "RD bias-corrected", "rdrobust",
         "Run a sharp RD regression of voteshare_next on margin at cutoff 0 "
         "with bias-corrected robust inference. Report the conventional "
         "estimate and the robust-SE estimate.",
         {"conventional": 0.073, "robust": 0.062}, 0.080),
        ("L1-06", "lee.csv", "RD density manipulation test", "rddensity",
         "Run the Cattaneo-Jansson-Ma density-discontinuity test on margin "
         "at cutoff 0; report the p-value.",
         {"pvalue": 0.91}, None),
        # Basque SCM
        ("L1-07", "basque.csv", "Classical synthetic control", "synth_classic",
         "Run classical synthetic control on gdppc with the Basque Country as "
         "treated unit and 1970 as treatment time; donor pool is the other "
         "16 Spanish regions. Report the average post-1970 gap.",
         {"avg_post_gap": -0.66}, -0.855),
        # NSW-DW PSM
        ("L1-08", "nsw_dw.csv", "PSM 1:1 NN", "psm",
         "Run propensity-score matching with logistic propensity, 1:1 nearest-"
         "neighbour, with replacement; covariates are age, education, black, "
         "hispanic, married, re74, re75; outcome re78, treatment treat. "
         "Report the ATT.",
         {"att_psm": 2278}, 1794),
        # DML on Card
        ("L1-09", "card.csv", "DML PLR", "dml_plr",
         "Run DML PLR with linear-regression nuisance learners and 5-fold "
         "cross-fitting for the effect of educ on lwage; controls are exper, "
         "expersq, black, south, smsa. Report theta.",
         {"theta": 0.110}, None),
        # Cox PH on a built-in survival dataset
        ("L1-10", "synth_survival.csv", "Cox proportional hazards", "coxph",
         "Run a Cox proportional hazards model with duration time, event "
         "indicator event, and covariates x1, x2 (Efron ties). Report the "
         "log-hazard-ratio coefficients.",
         {"beta_x1": 0.46, "beta_x2": -0.59}, None),
        # Honest-DiD smoothness
        ("L1-11", "mpdta.csv", "Honest-DiD smoothness", "honest_did_smooth",
         "After running Callaway-Sant'Anna on mpdta, run honest-DiD under "
         "the smoothness restriction with M = 0.05 at relative time e = 0; "
         "report the robust 95% CI.",
         {"ci_lo": 0.254, "ci_hi": 0.746}, None),
        # Bacon decomposition
        ("L1-12", "mpdta.csv", "Goodman-Bacon decomposition", "bacon",
         "Run the Goodman-Bacon decomposition on lemp ~ treat with countyreal "
         "as unit and year as time. Report beta_twfe and the negative-weight "
         "share.",
         {"beta_twfe": -0.0375, "neg_weight_share": 0.33}, None),
        # ETWFE
        ("L1-13", "mpdta.csv", "Wooldridge ETWFE", "etwfe",
         "Run Wooldridge ETWFE on lemp with first_treat as cohort, year as "
         "time, countyreal as unit. Report the simple ATT.",
         {"att_etwfe": -0.038, "se": 0.006}, None),
        # BJS imputation
        ("L1-14", "mpdta.csv", "BJS imputation", "did_imputation",
         "Run BJS imputation DiD on lemp with first_treat as cohort, year as "
         "time, countyreal as unit. Report the simple ATT.",
         {"att_bjs": -0.022, "se": 0.006}, None),
        # Synthetic DID on California Prop99
        ("L1-15", "california.csv", "Synthetic DID", "sdid",
         "Run synthetic difference-in-differences on cigsale with California "
         "as treated unit and 1989 as treatment time. Report the ATT.",
         {"att_sdid": -17.25}, None),
        # E-value
        ("L1-16", None, "E-value (closed form)", "evalue",
         "Compute the E-value for an estimated risk ratio of 2.5 with 95% CI "
         "(1.8, 3.2) on the RR scale.",
         {"evalue_estimate": 4.44, "evalue_ci": 3.0}, None),
        # LMM
        ("L1-17", "panel.csv", "Linear mixed model", "lmm",
         "Fit a linear mixed model with random intercept by gid; outcome y, "
         "fixed effects 1 + x1. Report the fixed-effects intercept, beta_x1, "
         "and the ICC.",
         {"beta_intercept": 1.90, "beta_x1": 1.48, "icc": 0.72}, None),
        # GLMM logit
        ("L1-18", "logit_panel.csv", "GLMM logit (Laplace)", "glmm_logit",
         "Fit a logistic GLMM with random intercept by gid; outcome y, "
         "fixed effects 1 + x1; Laplace approximation. Report fixed effects.",
         {"beta_intercept": -0.54, "beta_x1": 0.61}, None),
        # SFA cross-section
        ("L1-19", "sfa.csv", "Stochastic frontier (cross-sec.)", "frontier",
         "Fit a half-normal stochastic production frontier of lny on lnk, "
         "lnl. Report the production-frontier coefficients and the mean "
         "JLMS efficiency.",
         {"beta_lnk": 0.59, "beta_lnl": 0.39, "mean_efficiency": 0.65}, None),
        # Blinder-Oaxaca
        ("L1-20", "wage_gap.csv", "Blinder-Oaxaca decomposition", "oaxaca",
         "Run a Blinder-Oaxaca threefold decomposition of log_wage on educ, "
         "exper across the female group. Report the gap, explained, and "
         "unexplained components.",
         {"gap": 0.221, "explained": 0.039, "unexplained": 0.183}, None),
    ]

    for spec in L1_specs:
        pid += 1
        prompts.append({
            "id": spec[0], "level": "L1", "csv": spec[1],
            "method_named": spec[2], "estimator_key": spec[3],
            "question": spec[4],
        })
        golds.append({
            "id": spec[0], "expected_estimator": spec[3],
            "expected_values": spec[5],
            "published_anchor": spec[6],
        })

    # ----- L2 (indirect: method inferred from problem structure) -- 20 ---
    L2_specs = [
        # Card endogeneity
        ("L2-01", "card.csv",
         "Education is endogenous because of unobserved ability. Suggest and "
         "apply an appropriate causal estimator using the data at card.csv "
         "(it includes a college-proximity indicator nearc4).",
         "iv_2sls",
         {"beta_educ": 0.142}),
        # mpdta heterogeneous timing
        ("L2-02", "mpdta.csv",
         "Counties were treated at different times in mpdta.csv. Two-way "
         "fixed-effects gives biased estimates with heterogeneous timing. "
         "Use a robust modern estimator.",
         "callaway_santanna",
         {"simple_ATT": -0.033}),
        # Lee
        ("L2-03", "lee.csv",
         "voteshare_next is the next-election Democratic vote share; margin "
         "is the previous-election margin. Estimate the incumbent advantage "
         "using a quasi-experimental approach.",
         "rdrobust",
         {"robust": 0.062}),
        # NSW-DW
        ("L2-04", "nsw_dw.csv",
         "treat is participation in a job training program; re78 is post-"
         "program earnings. Background covariates are observable. Estimate "
         "the ATT.",
         "psm",
         {"att_psm": 2278}),
        # Basque
        ("L2-05", "basque.csv",
         "The Basque Country experienced terrorism starting in 1970. Estimate "
         "the cumulative GDP-per-capita gap relative to a counterfactual.",
         "synth_classic",
         {"avg_post_gap": -0.66}),
        # synth panel
        ("L2-06", "california.csv",
         "California passed Prop 99 in 1988-1989 raising tobacco taxes. "
         "Estimate the effect on per-capita cigarette sales.",
         "sdid",
         {"att_sdid": -17.25}),
        # cluster SE
        ("L2-07", "mpdta.csv",
         "Treatment is at the county level, but observations are at the "
         "county-year level. Estimate the treatment effect on lemp with "
         "appropriate inference.",
         "ols_cluster",
         {"beta_treat": -0.056}),
        # HDFE
        ("L2-08", "trade.csv",
         "Estimate the effect of x1 and x2 on y absorbing firm and year "
         "fixed effects on a panel with 200,000 firm-year observations.",
         "fast_feols",
         {"beta_x1": 2.0, "beta_x2": -1.5}),
        # Bacon
        ("L2-09", "mpdta.csv",
         "Decompose the two-way fixed-effects coefficient into the various "
         "2x2 comparisons that contribute to it.",
         "bacon",
         {"beta_twfe": -0.0375}),
        # Honest DiD
        ("L2-10", "mpdta.csv",
         "Quantify how sensitive your DiD estimate would be if the parallel-"
         "trends assumption were violated by a small smoothness restriction.",
         "honest_did_smooth",
         {}),
        # Sensemakr
        ("L2-11", "nsw_dw.csv",
         "After estimating the ATT with covariate-adjusted OLS, quantify the "
         "minimum confounder strength that would invalidate the result.",
         "sensemakr",
         {"rv_q": 0.076}),
        # Mediation
        ("L2-12", "mediation.csv",
         "Decompose the total effect of treat on y through the mediator m "
         "into ACME (indirect) and ADE (direct) components.",
         "mediation",
         {"acme": 0.108}),
        # Cox
        ("L2-13", "synth_survival.csv",
         "time is duration, event is the event indicator, x1 and x2 are "
         "covariates. Estimate the hazard-ratio coefficients.",
         "coxph",
         {"beta_x1": 0.46}),
        # LMM
        ("L2-14", "panel.csv",
         "Observations are clustered within group gid. Fit a model that "
         "accounts for the within-group correlation in y, with x1 as the "
         "fixed effect.",
         "lmm",
         {"beta_x1": 1.48}),
        # GLMM
        ("L2-15", "logit_panel.csv",
         "Outcome y is binary, observations clustered within group gid. Fit "
         "a model with x1 as fixed effect and random intercept by gid.",
         "glmm_logit",
         {"beta_x1": 0.61}),
        # Frontier
        ("L2-16", "sfa.csv",
         "Estimate a production function lny ~ lnk + lnl that allows for "
         "unobserved firm-level inefficiency.",
         "frontier",
         {"beta_lnk": 0.59}),
        # Oaxaca
        ("L2-17", "wage_gap.csv",
         "Decompose the female-male wage gap in log_wage into a portion "
         "explained by educ + exper and a portion unexplained.",
         "oaxaca",
         {"gap": 0.221, "explained": 0.039}),
        # DFL
        ("L2-18", "wage_gap.csv",
         "Use a reweighting-based decomposition to estimate the counterfactual "
         "log_wage distribution if female workers had male covariate "
         "distributions.",
         "dfl",
         {"gap": 0.209}),
        # E-value
        ("L2-19", None,
         "An observational study reports a risk ratio of 2.5 with 95% CI "
         "(1.8, 3.2). Quantify the strength of unmeasured confounding "
         "needed to explain away the result.",
         "evalue",
         {"evalue_estimate": 4.44}),
        # VAR
        ("L2-20", "macro.csv",
         "Fit a vector-autoregressive model with two endogenous variables y1 "
         "and y2, two lags, constant trend.",
         "var",
         {"L1.y1->y1": 0.43}),
    ]
    for spec in L2_specs:
        pid += 1
        prompts.append({
            "id": spec[0], "level": "L2", "csv": spec[1],
            "method_named": None, "estimator_key": spec[3],
            "question": spec[2],
        })
        golds.append({
            "id": spec[0], "expected_estimator": spec[3],
            "expected_values": spec[4],
            "published_anchor": None,
        })

    # ----- L3 (full workflow) -- 10 prompts -----------------------------
    L3_specs = [
        ("L3-01", "card.csv",
         "What is the causal effect of an additional year of schooling on "
         "log wages, given the data at card.csv? Run a complete analysis "
         "with diagnostics and at least one robustness check.",
         "iv_2sls",
         {"beta_educ": [0.10, 0.16]}),
        ("L3-02", "mpdta.csv",
         "Estimate the effect of the policy change on county-level employment "
         "(lemp). Justify your choice of estimator, run pre-trend and "
         "sensitivity diagnostics, and present a publication-ready table.",
         "callaway_santanna",
         {"simple_ATT": [-0.06, -0.02]}),
        ("L3-03", "lee.csv",
         "Estimate the U.S. House incumbency advantage from voteshare_next "
         "and margin. Run a complete RD analysis including a placebo or "
         "manipulation diagnostic.",
         "rdrobust",
         {"robust": [0.04, 0.10]}),
        ("L3-04", "nsw_dw.csv",
         "Estimate the average treatment effect on the treated of training "
         "programme participation on 1978 earnings. Compare results across "
         "at least three estimators (a regression, a matching estimator, "
         "and an ML-based one).",
         "psm",
         {"att": [1500, 3000]}),
        ("L3-05", "basque.csv",
         "Quantify the GDP-per-capita cost of terrorism in the Basque "
         "Country. Run synthetic-control with at least one robustness "
         "test (placebo, leave-one-out, or backdating).",
         "synth_classic",
         {"avg_post_gap": [-1.0, -0.5]}),
        ("L3-06", "trade.csv",
         "Estimate the elasticity of y with respect to x1 and x2 in a panel "
         "with high-dimensional fixed effects. Choose the appropriate "
         "small-sample correction and present clustered standard errors.",
         "fast_feols",
         {"beta_x1": [1.95, 2.05]}),
        ("L3-07", "panel.csv",
         "Fit a linear mixed-effects model and compare it to a fixed-effects "
         "specification using a Hausman test. Report the chosen "
         "specification's fixed effects.",
         "lmm",
         {"beta_x1": [1.40, 1.55]}),
        ("L3-08", "wage_gap.csv",
         "Decompose the female-male wage gap. Run at least two distinct "
         "decompositions (Blinder-Oaxaca and a reweighting-based method), "
         "report both, and explain when they diverge.",
         "oaxaca",
         {"gap": [0.20, 0.23]}),
        ("L3-09", "synth_survival.csv",
         "Run a Cox proportional-hazards model and a Kaplan-Meier survival "
         "curve for the treatment effect on time-to-event. Test the "
         "proportional-hazards assumption.",
         "coxph",
         {"beta_x1": [0.40, 0.55]}),
        ("L3-10", "macro.csv",
         "Estimate the dynamic causal effect of a unit shock to y1 on y2 "
         "and y1 itself over a 6-period horizon, using both VAR-implied "
         "and local-projection IRFs. Compare them.",
         "local_projections",
         {"irf_h1": [0.40, 0.55]}),
    ]
    for spec in L3_specs:
        pid += 1
        prompts.append({
            "id": spec[0], "level": "L3", "csv": spec[1],
            "method_named": None, "estimator_key": spec[3],
            "question": spec[2],
        })
        golds.append({
            "id": spec[0], "expected_estimator": spec[3],
            "expected_value_range": spec[4],
            "published_anchor": None,
        })

    return prompts, golds


def main() -> None:
    prompts, golds = make_prompts()
    PROMPTS.write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    GOLDS.write_text(json.dumps(golds, indent=2), encoding="utf-8")
    n_l1 = sum(1 for p in prompts if p["level"] == "L1")
    n_l2 = sum(1 for p in prompts if p["level"] == "L2")
    n_l3 = sum(1 for p in prompts if p["level"] == "L3")
    print(f"OK -- wrote {len(prompts)} prompts ({n_l1} L1 + {n_l2} L2 + {n_l3} L3)")
    print(f"  prompts: {PROMPTS}")
    print(f"  golds:   {GOLDS}")


if __name__ == "__main__":
    main()
