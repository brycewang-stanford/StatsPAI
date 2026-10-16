"""Assemble golds/golds.json from the R reference values.

For each task: the headline value it is scored on, and one or more
*variants* -- an acceptable estimator with its reference values and a
tolerance. Task success means the submitted headline is within tolerance of
some variant; the method is correct when the submitted estimator is one of
the variants' estimators.

Tolerances (pre-registered with this file):

* ``EXACT`` (rel 1%): deterministic estimators whose R and Python / Stata
  implementations agree to many digits; 1% allows printed rounding.
* ``CONVENTION`` (rel 5%): estimators where implementations legitimately
  differ by an optimiser, a tie rule or a cross-fitting split (synthetic
  control weights, matching ties, DML folds, SFA / GLMM likelihoods).
* ``abs`` floors keep near-zero headlines from demanding impossible
  relative precision.

Diagnostic fields an L3 task asks for (a pre-trend p-value, a first-stage
F) are scored for presence (``required``), not value: their definition
depends on which test the agent chose.

Usage::

    Rscript golds/reference.R            # from tests/agent_bench/
    python runners/build_golds.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REF = ROOT / "golds" / "reference_values.json"
GOLDS = ROOT / "golds" / "golds.json"
TASKS = ROOT / "prompts" / "prompts.json"

EXACT = {"rel": 0.01, "abs": 1e-4}
CONVENTION = {"rel": 0.05, "abs": 1e-3}


def _v(ref: Dict[str, float], estimator: str, values: Dict[str, str], tol, source: str):
    return {
        "estimator": estimator,
        "values": {name: ref[key] for name, key in values.items()},
        "tolerance": tol,
        "source": source,
    }


def build(ref: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    R = ref
    g: Dict[str, Dict[str, Any]] = {}

    def task(tid: str, variants: List[dict], required: List[str] = ()) -> None:
        g[tid] = {"variants": variants, "required": list(required)}

    cs_all = [
        _v(
            R,
            "callaway_santanna",
            {"att": f"cs_{e}_{c}.att"},
            EXACT,
            f"R did::att_gt(est_method='{e}', control_group='{c}') + aggte(simple)",
        )
        for e in ("dr", "ipw", "reg")
        for c in ("nevertreated", "notyettreated")
    ] + [
        _v(R, "sun_abraham", {"att": "sa.att"}, EXACT, "R fixest sunab, agg='att'"),
        _v(
            R,
            "sun_abraham",
            {"att": "sa_eventtime.att"},
            EXACT,
            "sunab, equal-weight event-time mean",
        ),
        _v(R, "etwfe", {"att": "etwfe.att"}, EXACT, "R etwfe + emfx(simple)"),
        _v(R, "did_imputation", {"att": "bjs.att"}, EXACT, "R didimputation"),
    ]

    # ---------------------------------------------------------------- L1
    task(
        "L1-01",
        [
            _v(
                R,
                "ols",
                {"beta_educ": "ols.beta_educ", "se_educ": "ols.se_educ"},
                EXACT,
                "R lm + sandwich HC1",
            )
        ],
    )
    task(
        "L1-02",
        [
            _v(
                R,
                "iv_2sls",
                {"beta_educ": "iv.beta_educ", "se_educ": "iv.se_educ"},
                EXACT,
                "R AER::ivreg + sandwich HC1",
            )
        ],
    )
    task(
        "L1-03",
        [
            _v(
                R,
                "callaway_santanna",
                {"att": "cs_dr_nevertreated.att", "se": "cs_dr_nevertreated.se"},
                EXACT,
                "R did 2.x att_gt(dr, nevertreated), aggte simple, analytic SE",
            )
        ],
    )
    task(
        "L1-04",
        [
            _v(
                R,
                "sun_abraham",
                {"att": "sa.att"},
                EXACT,
                "R fixest sunab, summary(agg='att'): cells weighted by cohort size",
            ),
            _v(
                R,
                "sun_abraham",
                {"att": "sa_eventtime.att"},
                EXACT,
                "equal-weight mean of the post-treatment event-time IW coefficients",
            ),
        ],
    )
    task(
        "L1-05",
        [
            _v(
                R,
                "rdrobust",
                {
                    "tau_conventional": "rd.tau_conventional",
                    "tau_bias_corrected": "rd.tau_bias_corrected",
                    "se_robust": "rd.se_robust",
                },
                EXACT,
                "R rdrobust defaults",
            )
        ],
    )
    task(
        "L1-06",
        [
            _v(
                R,
                "rddensity",
                {"p_value": "rddensity.p_value"},
                {"rel": 0.02, "abs": 0.005},
                "R rddensity defaults, p_jk",
            )
        ],
    )
    task(
        "L1-07",
        [
            _v(
                R,
                "synth",
                {"avg_post_gap": "synth.avg_post_gap"},
                {"rel": 0.15, "abs": 0.02},
                "R Synth, pre-1970 gdppc as predictors",
            ),
            _v(
                R,
                "synth",
                {"avg_post_gap": "sc_basque.avg_post_gap"},
                {"rel": 0.15, "abs": 0.02},
                "R synthdid::sc_estimate",
            ),
        ],
    )
    task(
        "L1-08",
        [
            _v(
                R,
                "psm",
                {"att": "psm.att"},
                CONVENTION,
                "R Matching::Match(M=1, replace, ties=TRUE) on logit pscore",
            ),
            _v(R, "psm", {"att": "psm_noties.att"}, CONVENTION, "same, ties=FALSE"),
        ],
    )
    task(
        "L1-09",
        [
            _v(
                R,
                "dml_plr",
                {"theta": "dml.theta", "se": "dml.se"},
                CONVENTION,
                "R DoubleML PLR, regr.lm, 5 folds, mean of 20 splits",
            )
        ],
    )
    task(
        "L1-10",
        [
            _v(
                R,
                "coxph",
                {
                    "beta_x1": "cox.beta_x1",
                    "beta_x2": "cox.beta_x2",
                    "beta_treat": "cox.beta_treat",
                },
                EXACT,
                "R survival::coxph, Efron",
            )
        ],
    )
    task(
        "L1-11",
        [
            _v(
                R,
                "honest_did",
                {
                    "ci_lower": "honest_universal.ci_lower",
                    "ci_upper": "honest_universal.ci_upper",
                },
                CONVENTION,
                "R did (universal base) + HonestDiD smoothness, M=0.05, FLCI",
            )
        ],
    )
    task(
        "L1-12",
        [
            _v(
                R,
                "bacon",
                {
                    "beta_twfe": "bacon.beta_twfe",
                    "weight_later_vs_earlier": "bacon.weight_later_vs_earlier",
                },
                EXACT,
                "R bacondecomp",
            )
        ],
    )
    task(
        "L1-13",
        [_v(R, "etwfe", {"att": "etwfe.att"}, EXACT, "R etwfe + emfx(type='simple')")],
    )
    task(
        "L1-14", [_v(R, "did_imputation", {"att": "bjs.att"}, EXACT, "R didimputation")]
    )
    task(
        "L1-15",
        [_v(R, "sdid", {"att": "sdid.att"}, EXACT, "R synthdid::synthdid_estimate")],
    )
    task(
        "L1-16",
        [
            _v(
                R,
                "evalue",
                {
                    "evalue_estimate": "evalue.evalue_estimate",
                    "evalue_ci": "evalue.evalue_ci",
                },
                EXACT,
                "R EValue::evalues.RR",
            )
        ],
    )
    task(
        "L1-17",
        [
            _v(
                R,
                "lmm",
                {
                    "beta_intercept": "lmm.beta_intercept",
                    "beta_x1": "lmm.beta_x1",
                    "icc": "lmm.icc",
                },
                EXACT,
                "R lme4::lmer, REML",
            )
        ],
    )
    task(
        "L1-18",
        [
            _v(
                R,
                "glmm_logit",
                {"beta_intercept": "glmm.beta_intercept", "beta_x1": "glmm.beta_x1"},
                CONVENTION,
                "R lme4::glmer, Laplace",
            )
        ],
    )
    task(
        "L1-19",
        [
            _v(
                R,
                "frontier",
                {
                    "beta_lnk": "sfa.beta_lnk",
                    "beta_lnl": "sfa.beta_lnl",
                    "mean_efficiency": "sfa.mean_efficiency",
                },
                CONVENTION,
                "R frontier::sfa, half-normal",
            )
        ],
    )
    task(
        "L1-20",
        [
            _v(
                R,
                "oaxaca",
                {
                    "gap": "oaxaca.gap",
                    "endowments": "oaxaca.endowments",
                    "coefficients": "oaxaca.coefficients",
                    "interaction": "oaxaca.interaction",
                },
                EXACT,
                "threefold, group A = men, endowments at women's coefficients",
            )
        ],
    )

    # ---------------------------------------------------------------- L2
    task(
        "L2-01",
        [
            _v(
                R,
                "iv_2sls",
                {"effect": "iv.beta_educ"},
                EXACT,
                "2SLS with nearc4 and controls",
            ),
            _v(
                R,
                "iv_2sls",
                {"effect": "iv_nocontrols.beta_educ"},
                EXACT,
                "2SLS with nearc4, no controls",
            ),
        ],
    )
    task("L2-02", cs_all)
    task(
        "L2-03",
        [
            _v(
                R,
                "rdrobust",
                {"effect": "rd.tau_conventional"},
                CONVENTION,
                "rdrobust conventional",
            ),
            _v(
                R,
                "rdrobust",
                {"effect": "rd.tau_bias_corrected"},
                CONVENTION,
                "rdrobust bias-corrected",
            ),
        ],
    )
    task(
        "L2-04",
        [
            _v(R, "psm", {"att": "psm.att"}, CONVENTION, "PSM, Matching ties=TRUE"),
            _v(R, "psm", {"att": "psm_noties.att"}, CONVENTION, "PSM, ties=FALSE"),
            _v(R, "ipw", {"att": "ipw.att"}, CONVENTION, "normalised IPW ATT, logit"),
            _v(R, "aipw", {"att": "aipw.att"}, CONVENTION, "doubly robust ATT"),
            _v(
                R,
                "ols",
                {"att": "ols_adjust.att"},
                CONVENTION,
                "covariate-adjusted OLS",
            ),
        ],
    )
    task(
        "L2-05",
        [
            _v(
                R,
                "synth",
                {"avg_post_gap": "synth.avg_post_gap"},
                {"rel": 0.15, "abs": 0.02},
                "R Synth",
            ),
            _v(
                R,
                "synth",
                {"avg_post_gap": "sc_basque.avg_post_gap"},
                {"rel": 0.15, "abs": 0.02},
                "synthdid::sc_estimate",
            ),
            _v(
                R,
                "sdid",
                {"avg_post_gap": "sdid_basque.avg_post_gap"},
                {"rel": 0.15, "abs": 0.02},
                "synthdid::synthdid_estimate",
            ),
        ],
    )
    task(
        "L2-06",
        [
            _v(R, "sdid", {"att": "sdid.att"}, CONVENTION, "synthdid"),
            _v(
                R,
                "synth",
                {"att": "sc.att"},
                {"rel": 0.15, "abs": 0.5},
                "synthetic control",
            ),
            _v(R, "twfe", {"att": "did.att"}, CONVENTION, "difference in differences"),
        ],
    )
    task(
        "L2-07",
        [
            _v(
                R,
                "twfe",
                {"effect": "twfe.beta", "se": "twfe.se"},
                EXACT,
                "fixest TWFE, cluster countyreal",
            )
        ],
    )
    task(
        "L2-08",
        [
            _v(
                R,
                "hdfe_ols",
                {"beta_x1": "hdfe.beta_x1", "beta_x2": "hdfe.beta_x2"},
                EXACT,
                "fixest firm + year FE",
            )
        ],
    )
    task(
        "L2-09",
        [_v(R, "bacon", {"beta_twfe": "bacon.beta_twfe"}, EXACT, "bacondecomp")],
    )
    task(
        "L2-10",
        [
            _v(
                R,
                "honest_did",
                {
                    "ci_lower": "honest_universal.ci_lower",
                    "ci_upper": "honest_universal.ci_upper",
                },
                CONVENTION,
                "universal base period",
            ),
            _v(
                R,
                "honest_did",
                {
                    "ci_lower": "honest_varying.ci_lower",
                    "ci_upper": "honest_varying.ci_upper",
                },
                CONVENTION,
                "varying base period",
            ),
        ],
    )
    task(
        "L2-11", [_v(R, "sensemakr", {"rv_q": "sensemakr.rv_q"}, EXACT, "R sensemakr")]
    )
    task(
        "L2-12",
        [
            _v(
                R,
                "mediation",
                {"acme": "mediation.acme", "ade": "mediation.ade"},
                CONVENTION,
                "product of coefficients (linear models)",
            )
        ],
    )
    task(
        "L2-13",
        [
            _v(R, "coxph", {"beta_treat": "cox.beta_treat"}, EXACT, "coxph Efron"),
            _v(
                R,
                "coxph",
                {"beta_treat": "cox_breslow.beta_treat"},
                EXACT,
                "coxph Breslow",
            ),
        ],
    )
    task(
        "L2-14",
        [
            _v(R, "lmm", {"beta_x1": "lmm.beta_x1"}, EXACT, "random intercept (REML)"),
            _v(R, "lmm", {"beta_x1": "lmm_ml.beta_x1"}, EXACT, "random intercept (ML)"),
            _v(
                R,
                "ols_cluster",
                {"beta_x1": "pooled_ols.beta_x1"},
                EXACT,
                "pooled OLS, cluster-robust SEs",
            ),
            _v(
                R,
                "hdfe_ols",
                {"beta_x1": "fe.beta_x1"},
                EXACT,
                "within (group fixed effects)",
            ),
        ],
    )
    task(
        "L2-15",
        [
            _v(
                R,
                "glmm_logit",
                {"beta_x1": "glmm.beta_x1"},
                CONVENTION,
                "glmer Laplace",
            ),
            _v(
                R,
                "glmm_logit",
                {"beta_x1": "glmm_agq.beta_x1"},
                CONVENTION,
                "glmer AGQ",
            ),
        ],
    )
    task(
        "L2-16",
        [
            _v(
                R,
                "frontier",
                {"beta_lnk": "sfa.beta_lnk", "beta_lnl": "sfa.beta_lnl"},
                CONVENTION,
                "frontier::sfa half-normal",
            )
        ],
    )
    task(
        "L2-17",
        [
            _v(
                R,
                "oaxaca",
                {"gap": "oaxaca.gap", "explained": f"oaxaca_{ref}.explained"},
                EXACT,
                f"twofold, reference {ref}",
            )
            for ref in ("refA", "refB", "pooled", "pooled_dummy")
        ],
    )
    task(
        "L2-18",
        [
            _v(
                R,
                "dfl",
                {"counterfactual_mean": "dfl.counterfactual_mean"},
                EXACT,
                "DFL, logit on educ + exper",
            ),
            _v(
                R,
                "dfl",
                {"counterfactual_mean": "dfl_sq.counterfactual_mean"},
                EXACT,
                "DFL, with squares",
            ),
        ],
    )
    task(
        "L2-19",
        [
            _v(
                R,
                "evalue",
                {"evalue_estimate": "evalue.evalue_estimate"},
                EXACT,
                "EValue",
            )
        ],
    )
    task(
        "L2-20",
        [
            _v(
                R,
                "var",
                {"y1_on_L1_y1": "var.y1_on_L1_y1", "y2_on_L1_y1": "var.y2_on_L1_y1"},
                EXACT,
                "vars::VAR(p=2, const)",
            )
        ],
    )

    # ---------------------------------------------------------------- L3
    task(
        "L3-01",
        [
            _v(R, "iv_2sls", {"effect": "iv.beta_educ"}, EXACT, "2SLS with controls"),
            _v(
                R,
                "iv_2sls",
                {"effect": "iv_nocontrols.beta_educ"},
                EXACT,
                "2SLS without controls",
            ),
        ],
        required=["first_stage_f"],
    )
    task("L3-02", cs_all, required=["pretrend_p_value"])
    task(
        "L3-03",
        [
            _v(
                R,
                "rdrobust",
                {"effect": "rd.tau_conventional"},
                CONVENTION,
                "conventional",
            ),
            _v(
                R,
                "rdrobust",
                {"effect": "rd.tau_bias_corrected"},
                CONVENTION,
                "bias-corrected",
            ),
        ],
        required=["density_p_value"],
    )
    task(
        "L3-04",
        [
            _v(R, "psm", {"att_matching": "psm.att"}, CONVENTION, "PSM ties=TRUE"),
            _v(
                R,
                "psm",
                {"att_matching": "psm_noties.att"},
                CONVENTION,
                "PSM ties=FALSE",
            ),
        ],
        required=["att_regression", "att_ml"],
    )
    task(
        "L3-05",
        [
            _v(
                R,
                "synth",
                {"avg_post_gap": "synth.avg_post_gap"},
                {"rel": 0.15, "abs": 0.02},
                "Synth",
            ),
            _v(
                R,
                "synth",
                {"avg_post_gap": "sc_basque.avg_post_gap"},
                {"rel": 0.15, "abs": 0.02},
                "sc_estimate",
            ),
        ],
        required=["placebo_p_value"],
    )
    task(
        "L3-06",
        [
            _v(
                R,
                "hdfe_ols",
                {"beta_x1": "hdfe.beta_x1", "se_x1": "hdfe.se_x1"},
                EXACT,
                "fixest, cluster firm",
            )
        ],
    )
    task(
        "L3-07",
        [
            _v(
                R,
                "lmm",
                {"beta_x1": "re.beta_x1"},
                EXACT,
                "random effects (Hausman does not reject)",
            ),
            _v(R, "lmm", {"beta_x1": "lmm.beta_x1"}, EXACT, "lmer"),
        ],
        required=["hausman_p_value"],
    )
    task(
        "L3-08",
        [
            _v(
                R,
                "oaxaca",
                {"explained_oaxaca": f"oaxaca_{ref}.explained"},
                EXACT,
                f"twofold, reference {ref}",
            )
            for ref in ("refA", "refB", "pooled", "pooled_dummy")
        ],
        required=["explained_dfl"],
    )
    task(
        "L3-09",
        [_v(R, "coxph", {"beta_treat": "cox.beta_treat"}, EXACT, "coxph Efron")],
        required=["ph_global_p_value"],
    )
    task(
        "L3-10",
        [
            _v(
                R,
                "local_projections",
                {"irf_lp_h1": "lp.irf_lp_h1", "irf_var_h1": "var.irf_var_h1"},
                CONVENTION,
                "LP with two lags; VAR(2) Cholesky",
            )
        ],
    )
    return g


def main() -> None:
    ref = json.loads(REF.read_text(encoding="utf-8"))
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))["tasks"]
    golds = build(ref["values"])
    missing = [t["id"] for t in tasks if t["id"] not in golds]
    assert not missing, missing
    for t in tasks:
        entry = golds[t["id"]]
        entry["id"] = t["id"]
        entry["headline"] = t["headline"]
        for v in entry["variants"]:
            assert t["headline"] in v["values"], (t["id"], v)
            unknown = set(v["values"]) - set(t["report"])
            assert not unknown, (t["id"], unknown)
        assert set(entry["required"]) <= set(t["report"]), t["id"]
    out = {
        "version": 2,
        "reference": ref["meta"],
        "golds": [golds[t["id"]] for t in tasks],
    }
    GOLDS.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(out['golds'])} golds to {GOLDS}")


if __name__ == "__main__":
    main()
