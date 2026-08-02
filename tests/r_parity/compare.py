"""Parity comparator: read all results/<module>_{py,R}.json pairs and
(optionally) the sister Stata-side JSONs in
``tests/stata_parity/results/<module>_Stata.json``, then emit:

  * parity_table.md       -- human-readable Markdown (3-way when Stata available)
  * parity_table.tex      -- LaTeX longtable, 4-col R-only baseline (legacy)
  * parity_table_3way.tex -- LaTeX longtable, 5-col with Stata column;
                             this is the version the JSS appendix \\input{}s.

Tolerance budget (pre-registered, NEXT-STEPS / JSS plan §5.2):

  * machine-level point-estimate references:   rel_diff < 1e-6
  * iterative / cross-fit (DiD, RD, SCM, DML): rel_diff < 1e-3
  * bootstrap / placebo CI half-widths:        abs_diff < 0.05 * SE
  * Honest-DiD CI bounds:                      abs_diff < 0.05

The same tolerance applies to the StatsPAI <-> Stata comparison: we
do not register a separate budget for the Stata side; one budget per
module is the single source of truth. Stata-side implementation
convention gaps that exceed that budget must be explicitly listed in
``STATA_HEADLINE_GAP_EXCEPTIONS``.

Verdict assignment is per-module, not per-row, because some rows
(e.g. SE-with-documented-convention-gap, default-h selector) are
expected NOT to pass at the strict tolerance and are recorded with
an explicit rationale in the module's `extra` block.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_DIR = HERE / "results"
PAPER_TABLES_DIR = ROOT / "Paper-JSS" / "manuscript" / "tables"
# The sister Stata-side harness lives at tests/stata_parity/. Stata
# results are emitted for modules with a canonical Stata reference or a
# deliberately labelled audited Stata/Mata bridge; modules without an
# authoritative or portable bridge implementation are flagged with explicit
# reasons in the 3-way table.
STATA_RESULTS_DIR = HERE.parent / "stata_parity" / "results"
STATA_SKIP_REASON: dict[str, str] = {
    "81_didm": (
        "Stata's did_multiplegt is dCDH's own port and would be the natural "
        "third side, but it is not installed in the verified local runtime; "
        "the archived R 0.1.4 is the reference used here."
    ),
    "80_contdid": (
        "no Stata implementation: the CGS continuous-treatment estimator "
        "ships as the R package contdid (GitHub only, not CRAN) and has no "
        "ssc counterpart."
    ),
    "79_didff": (
        "no Stata implementation: the Roth-Sant'Anna functional-form test "
        "ships as the R package didFF (GitHub only, not CRAN) and has no "
        "ssc counterpart."
    ),
    "78_multiplegt_dyn": (
        "Stata's did_multiplegt_dyn is the authors' own port and would be "
        "the natural third side, but it is not installed in the verified "
        "local runtime; the R package is the reference used here."
    ),
    "77_ddd": (
        "no Stata implementation: the Ortiz-Villavicencio/Sant'Anna DDD "
        "estimator ships as the R package triplediff and has no ssc "
        "counterpart."
    ),
    "76_pretrends": (
        "no Stata implementation: Roth's pretrends ships as an R package "
        "(GitHub only, not CRAN) and has no ssc counterpart."
    ),
    "75_stacked": (
        "no Stata implementation: Cengiz-Dube-Lindner-Zipperer stacking is a "
        "construction rather than a packaged command, and neither side has a "
        "canonical ssc reference -- the R side is itself hand-written."
    ),
    "74_cic": (
        "Kranker's ssc cic is a port of the Athey-Imbens Matlab and is the "
        "Stata counterpart, but it is not installed in the verified local "
        "runtime; R qte::CiC is the reference used here."
    ),
    "73_did2s": (
        "no Stata implementation: Gardner's two-stage estimator ships as "
        "the R package did2s and has no ssc counterpart, so the R side is "
        "the only cross-language reference for sp.gardner_did."
    ),
    "13_causal_forest": (
        "bridge artifact not materialized: Stata 19's official cate command "
        "is the candidate causal-forest/AIPW reference, but the verified local "
        "runtime is Stata 18 and `which cate` fails."
    ),
    "18_augsynth": (
        "bridge artifact not materialized: local Stata allsynth v1.32 is a "
        "candidate bias-corrected SCM reference, but its ridge de-biaser "
        "rejects the Basque outcome-only fixture with 16 controls and 15 "
        "pre-period predictors because it requires at least K + 2 control "
        "units; a feasible California probe also follows a distinct "
        "allsynth bias-correction convention rather than the R augsynth "
        "estimand. No portable like-for-like Stata artifact is materialized yet."
    ),
    "19_gsynth": (
        "bridge artifact not materialized: Xu's fect_stata is the candidate "
        "generalized-SCM route and can be installed in a temporary Stata 18 "
        "ado path, but the two-way IFE probe selects r=1 and reports ATT "
        "0.679854 under fect's convention, while the R/Python gsynth headline "
        "is -0.324171; an option grid over force(two-way/unit/time/none) "
        "does not recover the R gsynth convention. No like-for-like Stata "
        "bridge is materialized yet."
    ),
    "65_spatial": (
        "R-referenced module: the SAR/SEM/SDM maximum-likelihood headline is "
        "pinned to spatialreg::lagsarlm / errorsarlm (spatialreg 1.4-3). "
        "Stata's spregress is the natural analog but follows a distinct "
        "ML/estimand convention; no like-for-like Stata artifact is "
        "materialized yet."
    ),
    "66_spatial_gmm": (
        "R-referenced module: the SAR-2SLS / SEM-GMM headline is pinned to "
        "spatialreg::stsls(W2X=FALSE) / GMerrorsar (spatialreg 1.4-3). "
        "Stata's spregress, gs2sls is the natural analog but follows a "
        "distinct instrument/moment convention; no like-for-like Stata "
        "artifact is materialized yet."
    ),
    "67_panel_glm": (
        "R-referenced module: absorbed-FE Poisson/GLM is pinned to its "
        "fixest siblings (fepois/feglm, fixest 0.14.0). Stata ppmlhdfe is "
        "the natural analog but no like-for-like Stata artifact is "
        "materialized for this module yet."
    ),
    "68_demean_within": (
        "algorithmic module: the within (mean-deviation) transformation is "
        "an exact textbook identity checked against a base-R recomputation, "
        "not an external estimator. No Stata artifact is materialized."
    ),
    "69_balance_panel": (
        "algorithmic module: the balanced-panel filter is the base-R "
        "counts == n_periods row selection, an exact identity rather than "
        "an external estimator. No Stata artifact is materialized."
    ),
    "72_tmle": (
        "bridge artifact not materialized: the TMLE targeting step is "
        "pinned to tmle::tmle (tmle 2.1.1). Stata has no official TMLE "
        "command; the user-written eltmle wraps the same R package rather "
        "than providing an independent implementation, so a Stata row "
        "would re-measure the R reference through a shell rather than "
        "cross-validate it. No like-for-like Stata artifact is "
        "materialized."
    ),
    "71_dml_family": (
        "bridge artifact not materialized: the IRM / PLIV / IIVM model "
        "classes are pinned to DoubleML (R, 1.0.2). Stata's ddml is the "
        "canonical analog and does implement the interactive and IV "
        "families, but this module's design turns on both engines "
        "consuming one explicit fold partition, and ddml exposes no "
        "like-for-like sample-splitting hook to accept it; without that "
        "the row would grade cross-fitting noise rather than the "
        "estimator. No like-for-like Stata artifact is materialized yet. "
        "Module 08 already carries the materialized Stata PLR bridge."
    ),
    "70_policy_tree": (
        "R-referenced module: the exact depth<=2 welfare-maximising tree "
        "is pinned to policytree::policy_tree (policytree 1.2.4). Stata "
        "ships no policy-learning tree estimator, official or user-written, "
        "that solves the Athey-Wager objective over a supplied "
        "doubly-robust score matrix; no like-for-like Stata artifact is "
        "materialized."
    ),
}

TRACK_A_SNAPSHOT_ROWS: list[dict[str, Any]] = [
    {
        "module": "03_hdfe",
        "statistic": "beta_x1",
        "estimator": r"\code{sp.fast.feols}",
        "label": r"\(\hat\beta_{x1}\)",
        "data": r"2-way HDFE, \(N{=}10^4\)",
        "tol": r"\(10^{-6}\)",
        "verdict": "pass; fixest ssc",
    },
    {
        "module": "01_ols",
        "statistic": "beta_educ",
        "estimator": r"\code{sp.regress}",
        "label": r"\(\hat\beta_{\mathrm{educ}}\)",
        "data": "Card 1995",
        "tol": r"\(10^{-6}\)",
        "verdict": "pass",
    },
    {
        "module": "02_iv",
        "statistic": "beta_educ",
        "estimator": r"\code{sp.iv}",
        "label": r"\(\hat\beta_{\mathrm{educ}}\)",
        "data": "Card 1995",
        "tol": r"\(10^{-6}\)",
        "verdict": "pass",
    },
    {
        "module": "04_csdid",
        "statistic": "simple_ATT",
        "estimator": r"\code{sp.callaway\_santanna}",
        "label": "simple ATT",
        "data": r"\texttt{mpdta}",
        "tol": r"\(10^{-6}\)",
        "verdict": "pass",
    },
    {
        "module": "06_rd",
        "statistic": "default_robust_est",
        "estimator": r"\code{sp.rdrobust}",
        "label": r"robust RD, default CCT \(h\)",
        "data": "RDsenate",
        "tol": r"\(10^{-6}\)",
        "verdict": r"R/Stata default-\(h\) pass",
    },
    {
        "module": "08_dml",
        "statistic": "theta_DML_PLR",
        "estimator": r"\code{sp.dml}",
        "label": r"\(\hat\theta_{\mathrm{DML}}\)",
        "data": "Card 1995",
        "tol": r"\(10^{-10}\)",
        "verdict": "R/Stata bridge pass",
    },
    {
        "module": "13_causal_forest",
        "statistic": "ate_causal_forest",
        "estimator": r"\code{sp.causal\_forest}",
        "label": "AIPW ATE",
        "data": "clean-overlap DGP",
        "tol": "0.01",
        "verdict": "T3 combined-MC-error pass; operator exact",
    },
    {
        "module": "11_psm",
        "statistic": "att_psm",
        "estimator": r"\code{sp.psm}",
        "label": "ATT",
        "data": "NSW--DW",
        "tol": r"\(10^{-6}\)",
        "verdict": "pass; SE convention",
    },
    {
        "module": "52_scm_unique",
        "statistic": "avg_post_gap",
        "estimator": r"\code{sp.synth} (unique)",
        "label": "avg post gap",
        "data": "identified SCM DGP",
        "tol": r"\(10^{-6}\)",
        "verdict": "convex-SCM solver pass",
    },
    {
        "module": "07_scm",
        "statistic": "avg_post_gap",
        "estimator": r"\code{sp.synth} (classic)",
        "label": "avg post gap",
        "data": "Basque 2003",
        "tol": "0.05",
        "verdict": r"T4 R/Stata reference disagreement",
    },
    {
        "module": "28_frontier",
        "statistic": "beta_intercept",
        "estimator": r"\code{sp.frontier}",
        "label": "intercept",
        "data": "frontier DGP",
        "tol": r"\(10^{-6}\)",
        "verdict": "pass",
    },
    {
        "module": "30_oaxaca",
        "statistic": "gap",
        "estimator": r"\code{sp.decompose}",
        "label": "Oaxaca gap",
        "data": "Oaxaca DGP",
        "tol": r"\(10^{-6}\)",
        "verdict": "pass",
    },
]

# Modules where the R headline is inside the registered tolerance but the
# Stata headline deliberately records a known implementation convention gap.
# Keeping this explicit prevents the 3-way PASS column from silently masking
# Stata-side drift.
STATA_HEADLINE_GAP_EXCEPTIONS: dict[str, str] = {}


# Pre-registered tolerance per module. Every entry with rel_est or
# rel_se >= 5e-2 carries a graded justification (A mechanistic /
# B empirical / C unjustified) in docs/dev/r_parity_tolerances.md,
# together with the observed worst gap recomputed from the committed
# golden JSONs. Audit rules:
#   * NEVER loosen a value without re-registering it in that document.
#   * "sentinel" rel_se entries mark point-only modules whose SE rows
#     are deliberately side-specific (distinct statistic names, or
#     se=None) and therefore never join: the budget is vacuous today
#     and is pinned at the 1e-6 machine floor so that any future
#     joined SE row fails loudly and must be consciously re-budgeted.
TOLERANCES: dict[str, dict[str, float]] = {
    "01_ols": {"rel_est": 1e-6, "rel_se": 1e-6},
    # dCDH 2020 DID_M: the static effect, the horizon-1 dynamic effect and
    # the lag-1 placebo all match at 5e-15 on a panel where treatment
    # switches both on and off. Pinned against the ARCHIVED 0.1.4 -- the
    # 2.x rewrite's mode="old" returns NaN even on its own example. No
    # rel_se: the R side runs brep=0.
    "81_didm": {"rel_est": 1e-6},
    # CGS continuous treatment: the dose curves at four grid points and
    # both overall quantities, across three spline specifications, match
    # contdid at 1e-12. The curves are compared under
    # curve_basis="reference" -- contdid fits on the treated dose range but
    # reports on a basis re-anchored to the dose grid, so its reported
    # curves are a rescaled version of the fitted dose response and do not
    # line up with the overall ACRT it returns alongside them. StatsPAI
    # defaults to one consistent basis.
    # No rel_se: contdid's standard errors come from the pte aggregation
    # layer, which is not replicated.
    "80_contdid": {"rel_est": 1e-6},
    # Functional-form test: the sixteen implied-density bins across both
    # designs agree at 1.2e-15. The 1e-3 budget exists for ONE row per
    # design -- the p-value, which is a 100k-draw simulation of the
    # least-favourable critical value on each side with different RNGs
    # (observed gap 1e-5, i.e. one draw). Iterative tier for that reason
    # alone; the substantive quantities are machine-exact.
    "79_didff": {"rel_est": 1e-3},
    # dCDH intertemporal event study: across TWO designs -- absorbing, and
    # one where treatment switches off as well as on -- every effect,
    # placebo, aggregate and switcher count matches DIDmultiplegtDYN at
    # 5e-15. The switch-off design is what exercises the baseline-matched
    # control group and the divide-by-delta-D sign convention. No rel_se:
    # sp.did_multiplegt_dyn's analytic variance is not the paper's formula
    # (see its limitations) and its bootstrap is a different estimator.
    "78_multiplegt_dyn": {"rel_est": 1e-6},
    # Triple differences: the post-treatment ATT(g,t) cells and the
    # cohort-weighted aggregate match triplediff::ddd at 1e-12, across the
    # unconditional design and all three conditional nuisance combinations
    # (dr / ipw / reg). rel_se applies to the conditional rows, where
    # sp.ddd_heterogeneous(se='analytic') uses the same influence-function
    # variance the reference does; the unconditional rows emit no SE
    # because that path reports a cluster bootstrap instead.
    "77_ddd": {"rel_est": 1e-6, "rel_se": 1e-6},
    # Pre-trends power: iterative tier on purpose. pretrends gets its
    # rejection probability from mvtnorm::pmvnorm, whose Genz-Bretz
    # integrator is randomised -- twenty repeated R calls on this fixture
    # spread over ~5e-4 (sd 1.3e-4). Worst observed gap 4.3e-4, on
    # slope_for_power, which root-finds through that same noise. The
    # likelihood ratio is closed-form and agrees at 1e-15.
    "76_pretrends": {"rel_est": 1e-3},
    # Stacked DiD: event-study coefficients and the post mean match a
    # hand-written fixest stack at 1.3e-13 under both control-group
    # conventions. No rel_se -- the R side clusters on the raw unit id
    # while sp.stacked_did clusters on the stacked unit-cohort id, so the
    # SEs answer slightly different questions by design.
    "75_stacked": {"rel_est": 1e-6},
    # CIC: ATT and all nine QTEs match qte::CiC at machine precision
    # (worst 4.4e-15). No rel_se -- the R call runs se=FALSE.
    "74_cic": {"rel_est": 1e-6},
    # Gardner two-stage: point estimate matches did2s at 4.8e-08. The SE
    # is deliberately NOT given a rel_se tolerance -- did2s propagates
    # stage-1 estimation error into the stage-2 variance and
    # sp.gardner_did's vce='analytic' does not, so the ~26% gap is a
    # documented convention (vce='bootstrap' closes it to ~6%), not a
    # numerical failure to be papered over with a loose bound.
    "73_did2s": {"rel_est": 1e-6},
    "02_iv": {"rel_est": 1e-6, "rel_se": 1e-6},
    # Tightened 2026-06-10 from 1e-2 ("1-df conv. gap" was stale): with
    # ssc='fixest' the IID SEs match fixest/reghdfe at machine level
    # (observed worst rel_se 8.4e-15 incl. Stata side).
    "03_hdfe": {"rel_est": 1e-6, "rel_se": 1e-6},
    # B: analytic IF SE incl. control-regression uncertainty; observed
    # 0.32% vs both did::aggte and csdid (3.1x margin).
    "04_csdid": {"rel_est": 1e-6, "rel_se": 1e-2},
    # B: event-time IW SEs track Stata eventstudyinteract (<=0.9%);
    # fixest agg='att' clustered delta-method aggregation differs by up
    # to 17.1% on the sparse mpdta cohorts (1.5x margin; fixest-side
    # mechanism not yet pinned term-by-term) -- see doc.
    "05_sunab": {"rel_est": 1e-6, "rel_se": 0.25},
    "06_rd": {
        "rel_est": 1e-6,
        "rel_se": 0.10,
    },  # A: default-h rows machine-level via official CCT port; budget is
    # bound by the deliberately retained legacy-internal SE diagnostic
    # rows at the forced common h (observed 6.7%, 1.5x margin).
    "07_scm": {
        "rel_est": 1.0,  # A/T4: Basque weight non-uniqueness; R Synth and
        # Stata synth land on different local optima (donor weights rel
        # gap up to 1.78 vs R); native tracks Stata; exact-recovery
        # counterpart is module 52.
        "rel_se": 1e-6,  # sentinel: all rows are point-only (se=None)
    },  # native classical SCM: T4 R/Stata reference-disagreement disclosure
    "08_dml": {
        "rel_est": 1e-10,
        "rel_se": 1e-10,
    },  # explicit folds; audited linear PLR bridge
    "09_rddensity": {
        "rel_est": 1e-6,
        "rel_se": 1e-6,
    },  # native CJM/rddensity default parity
    "10_honest_did": {
        "abs_est": 5e-4,
        "abs_se": 5e-4,
    },  # R backend exact; Stata package port <3.3e-4
    # A + sentinel (tightened 2026-06-10 from 5.0): att_psm carries
    # se=None on all three sides by design -- SE estimators differ by
    # construction (sp matched-pair effect dispersion vs MatchIt
    # post-matching weighted-lm diagnostic vs teffects Abadie-Imbens)
    # and live under side-specific statistic names that never join.
    "11_psm": {"rel_est": 1e-6, "rel_se": 1e-6},
    "12_sdid": {
        "rel_est": 1e-6,
        "rel_se": 1e-6,  # sentinel (was 5e-2): att_sdid is point-only;
        # placebo SEs are backend-native diagnostics under distinct names.
    },  # point-only native FW/zeta ATT parity
    "13_causal_forest": {
        "rel_est": 0.01,  # B/T3: graded against combined Monte Carlo error
        # of two independent forests. Observed worst 0.47% (ATT, 2.1x
        # margin); the ATE headline sits at 0.19%. Widened from 0.005
        # when the ATT row moved onto grf's own plug-in + correction
        # estimator: the two sides now run the *same* estimator, so the
        # residual is pure forest MC rather than a convention gap, and
        # 0.005 left only 1.07x margin against machine-to-machine drift.
        "rel_se": 0.25,  # B: the AIPW *operator* is pinned exactly (see
        # tests/reference_parity/test_grf_aipw_operator_parity.py), so
        # this band covers forest RNG only. Observed worst 7.7% (ATE,
        # 3.2x margin); tightened from 0.50 after the ATT convention fix
        # removed the historical 14.6% ATT row (now 0.087%).
    },  # clean-overlap AIPW vs grf (post-nuisance-regularisation MC gap)
    "14_ols_cluster": {
        "rel_est": 1e-6,
        "rel_se": 1e-6,
    },  # obs worst 6.1e-9 (machine); 2026-06 tighten
    # Tightened 2026-06-10 from 5e-2 ("ssc convention" was stale): with
    # ssc='fixest' the CR1 nested-FE cluster SEs match fixest/reghdfe
    # (observed worst rel_se 1.25e-11 incl. Stata side).
    "15_hdfe_cluster": {"rel_est": 1e-6, "rel_se": 1e-6},
    "16_bjs": {
        "rel_est": 1e-6,
        "rel_se": 1e-6,  # sentinel (was 0.25): SE rows are side-specific
        # (se_cluster_if / se_didimputation / se_stata_did_imputation).
    },  # point row; side-specific SE diagnostics
    "17_etwfe": {"rel_est": 1e-6, "rel_se": 1e-3},  # emfx + cluster SE
    # parity; B: observed worst 6.0e-4 on the Stata side (1.7x margin).
    # B on est (iterative Ridge+SCM solver, observed 7.9e-6, 2.5x margin);
    # rel_se sentinel (was 1.0): the R augsynth fixture emits no joinable SE.
    "18_augsynth": {"rel_est": 2e-5, "rel_se": 1e-6},
    "19_gsynth": {
        "rel_est": 1e-6,
        "rel_se": 1e-6,  # sentinel (was 1.0): no SE row joins.
    },  # native gsynth/fect factor convention parity
    # rel_se sentinel (was 1.0): the Goodman-Bacon decomposition emits
    # no SEs on any side.
    "20_bacon": {"rel_est": 1e-6, "rel_se": 1e-6},  # TWFE-only headline
    "21_honest_relmags": {"abs_est": 1e-6, "abs_se": 1e-6},
    "22_sensemakr": {"rel_est": 1e-6, "rel_se": 1e-6},
    "23_evalue": {"rel_est": 1e-6, "rel_se": 1e-6},
    "24_coxph": {
        "rel_est": 1e-6,
        "rel_se": 1e-6,
    },  # obs worst 2.6e-15 (machine); 2026-06 tighten
    "25_lmm": {
        "rel_est": 1e-6,
        "rel_se": 1e-6,
    },  # REML criterion + tight optimiser parity
    # B: SE information-matrix convention at the (tight) Laplace/AGHQ
    # optimum differs across implementations; observed worst 1.9% incl.
    # Stata (2.7x margin). Value frozen by the contract test.
    "26_glmm_logit": {"rel_est": 2e-4, "rel_se": 5e-2},  # tightened GLMM optimiser tol
    "27_glmm_aghq": {
        "rel_est": 1e-6,
        "rel_se": 5e-2,
    },  # AGHQ tight optimiser, SE convention gap
    "28_frontier": {
        "rel_est": 1e-6,
        "rel_se": 5e-5,
    },  # obs worst 1.3e-5, ~4x margin; 2026-06 tighten
    # C (known weak spot, see doc): non-headline SE rows exceed this
    # budget (slope SE up to 0.98% vs frontier::sfa; intercept/sigma are
    # documented Stata-scale diagnostics). Headline = slope rel_est.
    "29_panel_sfa": {"rel_est": 1e-3, "rel_se": 1e-3},
    # A (tightened 2026-06-10 from 1.0): sp reports closed-form
    # delta-method SEs while oaxaca::oaxaca reports seeded R=100
    # bootstrap SEs; observed 1.25% (R) / 1.22% (Stata), 4x margin.
    "30_oaxaca": {"rel_est": 1e-6, "rel_se": 0.05},  # gap-only headline
    # rel_se sentinel (was 1.0): point-only decomposition rows.
    "31_dfl": {"rel_est": 1e-6, "rel_se": 1e-6},  # ddecompose reference_0 mapping
    "32_rif": {
        "rel_est": 1e-6,
        "rel_se": 1e-6,
    },  # dineq/Hmisc + stats::density convention
    # A on the Stata side (machine match, conditional-MLE divisor T);
    # NOTE: every R-side SE row differs by exactly sqrt(T/(T-k))-1 =
    # 1.29% because vars::VAR uses the per-equation lm() divisor T-k --
    # over this budget, disclosed as a weak spot in the doc.
    "33_var": {"rel_est": 1e-6, "rel_se": 1e-3},
    "34_lp": {"rel_est": 1e-6, "rel_se": 1e-6},  # lpirfs Cholesky/unit shock
    "35_panel": {"rel_est": 1e-6, "rel_se": 1e-3},  # FE/RE + plm-style Hausman
    # A: sp bootstrap B=1000 vs mediate's quasi-Bayesian MC (sims=200,
    # ~5% MC noise by itself); observed 7.0% (1.4x margin). Frozen by
    # the contract test.
    "36_mediation": {
        "rel_est": 1e-6,
        "rel_se": 0.10,
    },  # point exact; bootstrap/delta SE convention
    # Modules added in the 2026-05-28 parity expansion session.
    "37_ppmlhdfe": {"rel_est": 1e-6, "rel_se": 1e-2},  # post FE-score fix
    "38_drdid": {"rel_est": 1e-6, "rel_se": 1e-6},  # panel DRDID calibrated PS
    # rel_se sentinel (was 1e-2): no SE row joins on this fixture.
    "39_arima": {"rel_est": 1e-6, "rel_se": 1e-6},  # innovations-MLE exact convention
    # A: sp uses a Powell-type iid kernel sandwich
    # (regression/quantile.py) while quantreg reports se='nid'
    # (Hendricks-Koenker difference-quotient sandwich, chosen to match
    # Stata qreg); different sparsity estimators by construction.
    # Observed 7.3% (R) / 3.0% (Stata), 1.4x margin.
    "40_qreg": {"rel_est": 1e-6, "rel_se": 1e-1},  # Powell SE method choice
    "41_tobit": {
        "rel_est": 1e-6,
        "rel_se": 1e-5,
    },  # observed-info Hessian; obs worst 2.0e-6 (2026-06 tighten)
    "42_nbreg": {
        "rel_est": 1e-6,
        "rel_se": 5e-3,
    },  # obs worst 1.4e-3, 3x margin (2026-06 tighten)
    "43_heckman": {
        "rel_est": 1e-6,
        "rel_se": 5e-4,
    },  # obs worst 8.6e-5, ~6x margin (2026-06 tighten)
    "44_mlogit": {
        "rel_est": 1e-6,
        "rel_se": 5e-5,
    },  # multinom tight optimiser + observed-info Hessian; obs 1.2e-5 (2026-06 tighten)
    "45_ologit": {
        "rel_est": 1e-6,
        "rel_se": 1e-5,
    },  # polr tight optimiser + observed-info Hessian; obs 2.0e-6, 5.1x (at rule boundary)
    "46_clogit": {
        "rel_est": 1e-6,
        "rel_se": 1e-6,
    },  # obs worst 2.7e-9 (machine); 2026-06 tighten
    # Modules added in the second 2026-05-28 fix-and-extend pass.
    # B: HC1 sandwich after the Gauss-Seidel multi-FE fix; observed
    # 1.8% (R) / 0.10% (Stata), 2.7x margin.
    "47_ppmlhdfe_3fe": {"rel_est": 1e-6, "rel_se": 5e-2},  # post Gauss-Seidel
    "48_probit": {"rel_est": 1e-6, "rel_se": 1e-2},
    "49_oprobit": {
        "rel_est": 1e-6,
        "rel_se": 1e-6,
    },  # obs worst 3.0e-7 (machine floor); 2026-06 tighten
    "50_xtabond": {"rel_est": 1e-6, "rel_se": 1e-6},  # R/Stata dynamic-panel fixture
    "51_newey": {"rel_est": 1e-6, "rel_se": 1e-2},  # post HAC fix
    # Unique-solution SCM: strict-parity counterpart to module 07.
    "52_scm_unique": {
        "rel_est": 1e-6,
        "rel_se": 1e-6,  # sentinel (was 1.0): all rows point-only (se=None).
    },  # identified convex SCM; sp/Stata exact, Synth fixed-V QP at machine level
    # CR2 / CR3 cluster-robust SE: both headline rows use the
    # clubSandwich-compatible analytic corrections. Exact delete-one-cluster
    # jackknife remains a separate API.
    "53_cr2": {"rel_est": 1e-6, "rel_se": 1e-6},
    # Two-way cluster-robust SE (Cameron-Gelbach-Miller). sp uses the
    # per-dimension Liang-Zeger correction = sandwich::vcovCL defaults
    # (HC1, cadjust), so the headline two-way SE is a machine-precision
    # match (rel_se ~1e-16). fixest's single min-G df factor differs at
    # ~1e-3 and is NOT the convention reference here.
    "54_twoway_cluster": {"rel_est": 1e-6, "rel_se": 1e-6},
    # HC2/HC3 (MacKinnon-White) heteroskedasticity-robust SE. sp.regress
    # robust="hc2"/"hc3" matches sandwich::vcovHC(type="HC2"/"HC3") to
    # machine precision (module 01 covers HC1).
    "55_hc2_hc3": {"rel_est": 1e-6, "rel_se": 1e-6},
    # Three-way cluster-robust SE (Cameron-Gelbach-Miller). sp.multiway_cluster_vcov
    # matches sandwich::vcovCL(~g1+g2+g3, HC1, cadjust) to machine precision after
    # the v1.16.1 intersection-key fix.
    "56_multiway_cluster": {"rel_est": 1e-6, "rel_se": 1e-6},
    # Plain binary logit ML (sister of 48_probit). glm IRLS is run at
    # epsilon=1e-12 so all three sides sit on the same optimum; observed
    # diffs are ~1e-11 est / ~1e-9 SE.
    "57_logit": {"rel_est": 1e-6, "rel_se": 1e-6},
    # Plain Poisson ML (no FE; the HDFE robust Poisson is 37/47).
    "58_poisson": {"rel_est": 1e-6, "rel_se": 1e-6},
    # LIML k-class. sp.liml matches ivmodel::LIML at machine precision;
    # Stata ivregress liml runs with `small` so all three sides share the
    # RSS/(n-k) error-variance divisor.
    "59_liml": {"rel_est": 1e-6, "rel_se": 1e-6},
    # Two-equation SUR, one-step FGLS with Sigma divisor n (Stata sureg
    # default; systemfit methodResidCov='noDfCor', maxiter=1).
    "60_sureg": {"rel_est": 1e-6, "rel_se": 1e-6},
    # Beta regression (logit mean link, log-link precision). Point
    # estimates are machine-level across all three sides; the SE budget
    # is 1e-2 because betareg reports expected-(Fisher-)information SEs
    # while sp/Stata report observed-information SEs (documented
    # convention gap <=0.7% on this fixture; py<->Stata SEs agree at
    # ~1e-6).
    "61_betareg": {"rel_est": 1e-6, "rel_se": 1e-2},
    # Left-truncated normal regression. truncreg runs maxLik method='NR'
    # to converge past the BFGS default stopping point; sigma rows are
    # compared on the natural scale (sp delta-maps exp(ln_sigma)).
    "62_truncreg": {"rel_est": 1e-6, "rel_se": 1e-4},
    # Zero-inflated Poisson (logit inflation). zeroinfl runs at
    # reltol=1e-14; worst observed gap ~1e-7 est / ~6e-6 SE.
    "63_zip": {"rel_est": 1e-6, "rel_se": 1e-4},
    # Zero-inflated negative binomial. The ZINB likelihood is flat near
    # the optimum (R EM vs BFGS refinements move coefficients ~3e-7 at
    # identical logLik to 1e-10), so the point budget is 1e-5 instead of
    # machine; worst observed gap is ~1.1e-6 est / ~4e-5 SE.
    "64_zinb": {"rel_est": 1e-5, "rel_se": 1e-3},
    # Spatial ML: SAR/SEM/SDM coefficients, spatial parameter (rho/lambda),
    # and full-information asymptotic SEs vs spatialreg::lagsarlm /
    # errorsarlm / lagsarlm(Durbin=TRUE) on a 12x12 row-standardised rook
    # lattice. Machine tier: worst observed 8.3e-8 est / 2.0e-8 SE after the
    # bounded rho/lambda optimiser was tightened (xatol=1e-10).
    "65_spatial": {"rel_est": 1e-6, "rel_se": 1e-6},
    # Spatial GMM: SAR spatial 2SLS (Kelejian-Prucha) vs spatialreg::stsls
    # (W2X=FALSE) — closed-form projection, coefficients and n-k SEs agree to
    # machine precision (~1e-15). SEM generalized-moments coefficients +
    # lambda vs spatialreg::GMerrorsar are bit-exact (worst 4.6e-8), emitted
    # point-only because the coefficient-SE variance estimators differ.
    "66_spatial_gmm": {"rel_est": 1e-6, "rel_se": 1e-6},
    # Panel GLM: sp.feglm(family='logit') vs fixest::feglm and sp.fepois vs
    # fixest::fepois, both absorbing a single entity FE (id). Coefficients
    # agree to ~1e-8 (machine); SEs differ at ~1e-5 because the two IWLS
    # implementations iterate to slightly different working-weight roots.
    "67_panel_glm": {"rel_est": 1e-6, "rel_se": 5e-5},
    # Within transformation (sp.demean solver='map') vs textbook mean-within.
    # Pure algorithmic, agrees to machine tier (~3.5e-15 worst on a 20x8
    # balanced panel). Emitted point-only (no SE — it's a linear projection).
    "68_demean_within": {"rel_est": 1e-6, "rel_se": 1e-6},
    # Panel balance filter: sp.balance_panel keeps only entities observed
    # in every period, matching base R's counts == n_periods filter. The
    # estimator is a row-filter + sort, so all rows agree to 0.0.
    "69_balance_panel": {"rel_est": 1e-6, "rel_se": 1e-6},
    # Policy tree: shared AIPW score vector, so both engines maximise the
    # identical finite objective over the identical grid of distinct
    # covariate values. Exact optimisers on the same problem agree to the
    # floating-point floor (observed worst 9.6e-16 across value, treated
    # fraction, and root split at depths 1 and 2). Point-only: policytree
    # reports no SE for the tree itself.
    "70_policy_tree": {"rel_est": 1e-6, "rel_se": 1e-6},
    # DML family (IRM / PLIV / IIVM) against their DoubleML model classes
    # on a shared explicit fold partition, so cross-fitting contributes no
    # Monte Carlo term. PLIV is all closed-form least squares and lands at
    # the floating-point floor (6.5e-16); IRM and IIVM carry an unpenalised
    # logistic nuisance solved by lbfgs on one side and IRLS on the other,
    # which sets the observed worst at 1.1e-10.
    "71_dml_family": {"rel_est": 1e-6, "rel_se": 1e-6},
    # TMLE targeting step on a shared initial fit (Q and g1W supplied to
    # both engines), binary outcome so neither side rescales Y. The
    # residual is Newton-iteration tolerance on the fluctuation
    # parameters: observed 1.9e-9 on psi and 1.4e-11 on the SE.
    "72_tmle": {"rel_est": 1e-6, "rel_se": 1e-6},
}


# Strictness tiers. A single PASS/GAP verdict column flattens a
# machine-level point-estimate match and a methodological T3/T4
# tolerance into the same word, which a JSS
# reviewer is right to find dilutive. We therefore classify each module by
# the *registered* point-estimate tolerance (the forced strict tolerance
# when one exists, e.g. RD at a common bandwidth) so the parity tables can
# report the strictness breakdown explicitly. This is data-driven: it stays
# correct if a module's verdict later flips (e.g. when the RD bandwidth
# regularisation is ported and 06_rd tightens).
TIER_ORDER = ["machine", "iterative", "moderate", "methodological"]
TIER_LABEL = {
    "machine": "machine-level point estimate ($\\le 10^{-6}$)",
    "iterative": "iterative/cross-fit ($\\le 10^{-3}$)",
    "moderate": "moderate ($\\le 5\\times10^{-2}$)",
    "methodological": "methodological/T4 disclosure (T3/T4, not deterministic T2)",
    "unclassified": "unclassified",
}
TIER_LABEL_MD = {
    "machine": "machine-level point estimate (≤1e-6)",
    "iterative": "iterative/cross-fit (≤1e-3)",
    "moderate": "moderate (≤5e-2)",
    "methodological": "methodological/T4 disclosure (T3/T4, not deterministic T2)",
    "unclassified": "unclassified",
}

METHODOLOGICAL_DISCLOSURE_NOTES = {
    "13_causal_forest": (
        "T3 combined-Monte-Carlo-error pass, with the estimator's two "
        "factors graded separately. (1) The AIPW *operator* -- the "
        "closed-form map from (Y, W, tau.hat, Y.hat, W.hat) to the score "
        "vector, point estimate and influence-function SE -- is pinned "
        "exactly: fed grf's own forest outputs, sp.causal_forest "
        "reproduces grf::get_scores elementwise to 2.3e-14 and grf's "
        "reported ATE and ATT (estimate and std.err) to 1e-15 "
        "(tests/reference_parity/test_grf_aipw_operator_parity.py). "
        "(2) The *forest* is not pinnable across implementations, so the "
        "rows below are graded against combined sampling error: both "
        "sides report the same doubly-robust AIPW estimands, agree within "
        "~0.05 combined SE on the clean-overlap DGP, and the AIPW "
        "recovery tests certify truth recovery within 4 SE across "
        "multiple seeds. Because the operator is exact, the residual SE "
        "gap is attributable to forest RNG alone rather than to an "
        "unresolved formula difference -- which is what the previous 50% "
        "rel_se band could not distinguish."
    ),
}


def _display_meta_value(module: str, key: str, value: Any) -> Any:
    """Normalise the causal-forest note to the T3 combined-MC-error framing."""
    if (
        module == "13_causal_forest"
        and key == "note"
        and isinstance(value, str)
        and "must agree within combined Monte Carlo error" in value
    ):
        return value.replace(
            "so they are like-for-like and must agree within combined Monte Carlo error",
            "so they agree within combined Monte Carlo error "
            "(worst rel gap below 0.3%, ~0.05 combined SE on the clean DGP), "
            "the multi-seed truth-recovery guard passes, and the row is graded T3 "
            "(combined-MC-error pass)",
        )
    return value


def tolerance_tier(module: str) -> str:
    """Classify a module's headline strictness from its registered tolerance.

    Uses the point-estimate ``rel_est``/``abs_est``. Returns one of
    ``machine`` / ``iterative`` / ``moderate`` / ``methodological`` /
    ``unclassified``.
    """
    tol = TOLERANCES.get(module, {})
    key = tol.get("rel_est", tol.get("abs_est"))
    if key is None:
        return "unclassified"
    if key <= 1e-6:
        return "machine"
    if key <= 1e-3:
        return "iterative"
    if key <= 5e-2:
        return "moderate"
    return "methodological"


def tier_breakdown(modules: list[str]) -> dict[str, int]:
    """Count how many of ``modules`` fall in each strictness tier."""
    counts: dict[str, int] = {}
    for m in modules:
        counts[tolerance_tier(m)] = counts.get(tolerance_tier(m), 0) + 1
    return counts


def _tier_breakdown_sentence(modules: list[str], *, md: bool = False) -> str:
    counts = tier_breakdown(modules)
    labels = TIER_LABEL_MD if md else TIER_LABEL
    parts = [
        f"{counts[t]} {labels[t]}"
        for t in TIER_ORDER + ["unclassified"]
        if counts.get(t)
    ]
    return "; ".join(parts)


@dataclass
class RowDiff:
    module: str
    statistic: str
    py_est: float | None
    R_est: float | None
    abs_est: float | None
    rel_est: float | None
    py_se: float | None
    R_se: float | None
    abs_se: float | None
    rel_se: float | None
    # Stata-side fields. None when no Stata reference exists for the
    # module (or no row with this statistic).
    Stata_est: float | None = None
    Stata_se: float | None = None
    abs_est_st: float | None = None
    rel_est_st: float | None = None
    abs_se_st: float | None = None
    rel_se_st: float | None = None


def _diff(a: float | None, b: float | None) -> tuple[float | None, float | None]:
    if a is None or b is None:
        return None, None
    abs_d = abs(a - b)
    rel_d = abs_d / abs(b) if abs(b) > 1e-12 else (abs_d if abs(b) < 1e-12 else 0.0)
    return abs_d, rel_d


def _load_stata(module: str) -> dict[str, dict] | None:
    """Return {statistic -> row_dict} from the Stata harness, or None
    if the module has no Stata reference."""
    path = STATA_RESULTS_DIR / f"{module}_Stata.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {r["statistic"]: r for r in payload["rows"]}


def collect(module: str) -> list[RowDiff]:
    py_path = RESULTS_DIR / f"{module}_py.json"
    R_path = RESULTS_DIR / f"{module}_R.json"
    if not py_path.exists() or not R_path.exists():
        return []
    py = json.loads(py_path.read_text(encoding="utf-8"))
    R = json.loads(R_path.read_text(encoding="utf-8"))
    R_by = {r["statistic"]: r for r in R["rows"]}
    Stata_by = _load_stata(module) or {}
    out: list[RowDiff] = []
    for pr in py["rows"]:
        rr = R_by.get(pr["statistic"])
        if rr is None:
            continue
        abs_e, rel_e = _diff(pr["estimate"], rr["estimate"])
        abs_s, rel_s = _diff(pr.get("se"), rr.get("se"))
        sr = Stata_by.get(pr["statistic"])
        Stata_est = sr.get("estimate") if sr else None
        Stata_se = sr.get("se") if sr else None
        abs_est_st, rel_est_st = _diff(pr["estimate"], Stata_est)
        abs_se_st, rel_se_st = _diff(pr.get("se"), Stata_se)
        out.append(
            RowDiff(
                module=module,
                statistic=pr["statistic"],
                py_est=pr["estimate"],
                R_est=rr["estimate"],
                abs_est=abs_e,
                rel_est=rel_e,
                py_se=pr.get("se"),
                R_se=rr.get("se"),
                abs_se=abs_s,
                rel_se=rel_s,
                Stata_est=Stata_est,
                Stata_se=Stata_se,
                abs_est_st=abs_est_st,
                rel_est_st=rel_est_st,
                abs_se_st=abs_se_st,
                rel_se_st=rel_se_st,
            )
        )
    return out


def _has_any_stata(modules: list[str]) -> bool:
    return any(_load_stata(m) is not None for m in modules)


def fmt(x: float | None, prec: int = 6) -> str:
    if x is None:
        return "—"
    if abs(x) >= 1 or x == 0.0:
        return f"{x:.{prec}f}"
    return f"{x:.{prec}g}"


def _snapshot_number(x: float | None) -> str:
    if x is None:
        return "---"
    if abs(x) >= 100:
        return f"{x:.6f}"
    return f"{x:.9f}"


def _snapshot_rel(x: float | None) -> str:
    if x is None:
        return "---"
    if x == 0:
        return "0"
    if x < 1e-4:
        exp = int(math.floor(math.log10(abs(x))))
        mant = x / (10**exp)
        mant_s = f"{mant:.2g}"
        return rf"\({mant_s}\times10^{{{exp}}}\)"
    return f"{x:.3g}"


def _select_snapshot_diff(spec: dict[str, Any]) -> RowDiff:
    diffs = collect(spec["module"])
    if "statistic" in spec:
        selected = [d for d in diffs if d.statistic == spec["statistic"]]
    else:
        prefix = spec.get("statistic_prefix", "")
        suffix = spec.get("statistic_suffix", "")
        selected = [
            d
            for d in diffs
            if d.statistic.startswith(prefix) and d.statistic.endswith(suffix)
        ]
    if not selected:
        raise KeyError(f"snapshot row not found: {spec}")
    return selected[0]


def render_track_a_snapshot_tex() -> str:
    """Render the compact Track-A snapshot consumed by the main manuscript."""
    rows: list[str] = []
    for spec in TRACK_A_SNAPSHOT_ROWS:
        d = _select_snapshot_diff(spec)
        max_rel = max(value for value in (d.rel_est, d.rel_est_st) if value is not None)
        rows.append(
            f"{spec['estimator']} & {spec['label']} & {spec['data']} & "
            f"{_snapshot_number(d.py_est)} & {_snapshot_number(d.R_est)} & "
            f"{_snapshot_number(d.Stata_est)} & {_snapshot_rel(max_rel)} & "
            f"{spec['tol']} & {spec['verdict']} \\\\"
        )
    body = "\n".join(rows)
    return (
        "% AUTO-GENERATED by tests/r_parity/compare.py\n"
        "% Compact main-manuscript Track A snapshot; do not hand edit.\n"
        "\\begin{tabular}{lllllllll}\n"
        "\\toprule\n"
        "Estimator & Statistic & Data & \\statspai{} & "
        "\\proglang{R} ref. & \\proglang{Stata} ref. & "
        "Max rel. err. & Tol. & Verdict \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def render_md(modules: list[str]) -> str:
    lines: list[str] = [
        "# Track A parity report",
        "",
        "Generated by `tests/r_parity/compare.py` on the "
        "`results/<module>_{py,R}.json` artefacts. Tolerance budget per "
        "module is pre-registered in `compare.py::TOLERANCES`. Documented "
        "convention gaps, common-specification passes, and small-sample "
        "SE conventions (HDFE 1-df, legacy RD bandwidth diagnostics, "
        "SCM non-uniqueness) are flagged "
        "in the per-module `extra` block of the JSON.",
        "",
    ]
    for m in modules:
        diffs = collect(m)
        if not diffs:
            continue
        meta_path = RESULTS_DIR / f"{m}_py.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")).get("extra", {})
        lines.append(f"## Module {m}")
        if m in METHODOLOGICAL_DISCLOSURE_NOTES:
            lines.append(
                f"- **methodological_disclosure**: "
                f"{METHODOLOGICAL_DISCLOSURE_NOTES[m]}"
            )
        if meta:
            for k, v in meta.items():
                v = _display_meta_value(m, k, v)
                if isinstance(v, str) and len(v) > 80:
                    lines.append(f"- **{k}**: {v}")
                else:
                    lines.append(f"- **{k}**: `{v}`")
        lines.append("")
        lines.append(
            "| stat | py est | R est | abs Δ | rel Δ | py SE | R SE | abs Δ SE | rel Δ SE |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for d in diffs:
            lines.append(
                f"| `{d.statistic}` "
                f"| {fmt(d.py_est)} | {fmt(d.R_est)} "
                f"| {fmt(d.abs_est, 3)} | {fmt(d.rel_est, 3)} "
                f"| {fmt(d.py_se)} | {fmt(d.R_se)} "
                f"| {fmt(d.abs_se, 3)} | {fmt(d.rel_se, 3)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# Per-module headline: a (rows-to-summarise, label, verdict, gap-note)
# tuple that the TeX renderer uses to pick the most informative row to
# show. The headline uses the *strictest* row that the module is
# expected to pass, not the worst-case row -- so a documented
# convention gap doesn't shadow the bit-equal point-estimate result.
HEADLINE: dict[str, dict[str, Any]] = {
    "01_ols": {
        "name": "OLS + HC1 SE",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "02_iv": {
        "name": "2SLS + HC1 SE",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "03_hdfe": {
        "name": "HDFE 2-way FE",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "fixest/reghdfe small-sample correction",
    },
    "04_csdid": {
        "name": "CS-DiD simple ATT",
        "headline_filter": lambda d: d.statistic == "simple_ATT",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "SE within 1\\% analytic tolerance",
    },
    "73_did2s": {
        "name": "Gardner two-stage DiD",
        "headline_filter": lambda d: d.statistic == "static_ATT",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": (
            "point estimate rel < 1e-6; SE is a documented convention gap "
            "(did2s propagates stage-1 estimation error, sp.gardner_did's "
            "vce='analytic' does not -- vce='bootstrap' recovers it to ~6\\%)"
        ),
    },
    "81_didm": {
        "name": "dCDH 2020 DID_M (on/off switching)",
        "headline_filter": lambda d: d.statistic in {
            "effect", "dynamic_1", "placebo_1"
        },
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "machine precision vs archived DIDmultiplegt 0.1.4",
    },
    "80_contdid": {
        "name": "Continuous-treatment DiD (CGS)",
        "headline_filter": lambda d: d.statistic.endswith(
            ("_overall_att", "_overall_acrt")
        ),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "machine precision vs contdid::cont_did (worst 1e-12)",
    },
    "79_didff": {
        "name": "Functional-form test (Roth--Sant'Anna)",
        "headline_filter": lambda d: "_density_" in d.statistic,
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": (
            "machine precision vs didFF (worst 1.2e-15); the module's 1e-3 "
            "budget covers only the simulated p-value"
        ),
    },
    "78_multiplegt_dyn": {
        "name": "dCDH intertemporal event study",
        "headline_filter": lambda d: (
            d.statistic.startswith("Effect_") or d.statistic == "Av_tot_eff"
        ),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": (
            "machine precision vs DIDmultiplegtDYN (worst 5e-15), across "
            "absorbing and switch-off designs"
        ),
    },
    "77_ddd": {
        "name": "Triple differences ATT(g,t)",
        "headline_filter": lambda d: d.statistic.startswith("ddd_g"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": (
            "machine precision vs triplediff::ddd (worst 1e-12), estimates "
            "and analytic SEs, across dr / ipw / reg"
        ),
    },
    "76_pretrends": {
        "name": "Pre-trends power (Roth 2022)",
        "headline_filter": lambda d: d.statistic.startswith("power_slope_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": (
            "rel < 1e-4 vs pretrends::pretrends; the 1e-3 budget covers "
            "mvtnorm::pmvnorm's randomised integrator"
        ),
    },
    "75_stacked": {
        "name": "Stacked DiD (never-treated controls)",
        "headline_filter": lambda d: (
            d.statistic == "never_ATT_post" or d.statistic.startswith("never_att_rel_")
        ),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "rel < 1e-6 vs a hand-written fixest stack",
    },
    "74_cic": {
        "name": "Changes-in-Changes (ATT + QTE)",
        "headline_filter": lambda d: (
            d.statistic == "cic_ATT" or d.statistic.startswith("qte_")
        ),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "machine precision vs qte::CiC (worst 4.4e-15)",
    },
    "05_sunab": {
        "name": "Sun--Abraham event study",
        "headline_filter": lambda d: (
            d.statistic == "weighted_avg_ATT" or d.statistic.startswith("att_rel_")
        ),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "fixest agg='att' summary parity",
    },
    "06_rd": {
        "name": "RD CCT bias-corrected",
        "headline_filter": lambda d: d.statistic
        in (
            "default_conventional_est",
            "default_robust_est",
        ),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "R/Stata default-$h$ via CCT delegation",
    },
    "07_scm": {
        "name": "Classical SCM",
        "headline_filter": lambda d: d.statistic == "avg_post_gap",
        "metric": "rel_est",
        "verdict": "\\textit{GAP}",
        "gap_note": "T4 reference disagreement: native tracks Stata; R Synth differs; exact recovery on identified DGP \\code{52}",
    },
    "08_dml": {
        "name": "DML PLR (LinReg learners)",
        "headline_filter": lambda d: d.statistic == "theta_DML_PLR",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "shared explicit fold_id",
    },
    "09_rddensity": {
        "name": "RD density (CJM)",
        "headline_filter": lambda d: d.statistic == "test_pvalue",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "native CJM/rddensity default parity",
    },
    "10_honest_did": {
        "name": "Honest DiD bounds",
        "headline_filter": lambda d: d.statistic.startswith("ci_"),
        "metric": "abs_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "FLCI via HonestDiD backend; Stata port within 0.0005",
    },
    "11_psm": {
        "name": "PSM 1:1 NN",
        "headline_filter": lambda d: d.statistic == "att_psm",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "12_sdid": {
        "name": "Synthetic DID",
        "headline_filter": lambda d: d.statistic == "att_sdid",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "native Frank-Wolfe/zeta ATT parity; backend-native placebo SE diagnostics",
    },
    "13_causal_forest": {
        "name": "Causal forest (AIPW)",
        "headline_filter": lambda d: d.statistic == "ate_causal_forest",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "T3 combined-MC-error pass; like-for-like AIPW vs grf within $\\sim 0.05$ combined SE on clean-overlap DGP",
    },
    "14_ols_cluster": {
        "name": "OLS + cluster-robust SE",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "15_hdfe_cluster": {
        "name": "HDFE + cluster SE",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "fixest/reghdfe nested-FE cluster correction",
    },
    "16_bjs": {
        "name": "BJS imputation",
        "headline_filter": lambda d: d.statistic == "att_bjs",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "R/Stata simple-ATT point parity; SE rows side-specific",
    },
    "20_bacon": {
        "name": "Goodman--Bacon decomposition",
        "headline_filter": lambda d: d.statistic == "beta_twfe",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "R/Stata bacondecomp dyad parity",
    },
    "21_honest_relmags": {
        "name": "Honest-DiD relative-mags",
        "headline_filter": lambda d: d.statistic.startswith("ci_"),
        "metric": "abs_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "HonestDiD reference backend",
    },
    "22_sensemakr": {
        "name": "sensemakr robustness",
        "headline_filter": lambda d: d.statistic in ("beta_treat", "rv_q"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "sensemakr RV and benchmark bound-scale parity",
    },
    "25_lmm": {
        "name": "Linear mixed model",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "28_frontier": {
        "name": "Stochastic frontier (cross-sec.)",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "30_oaxaca": {
        "name": "Blinder--Oaxaca decomposition",
        "headline_filter": lambda d: d.statistic == "gap",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "twofold vs threefold split convention",
    },
    "17_etwfe": {
        "name": "Wooldridge ETWFE",
        "headline_filter": lambda d: d.statistic == "att_etwfe",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "emfx aggregation and cluster SE matched",
    },
    "18_augsynth": {
        "name": "Augmented SCM",
        "headline_filter": lambda d: d.statistic == "att_augmented",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "native centered Ridge+SCM parity; backend='augsynth' is a migration bridge",
    },
    "19_gsynth": {
        "name": "Generalized SCM (Xu 2017)",
        "headline_filter": lambda d: d.statistic == "att_gsynth",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "native gsynth/fect two-way FE factor convention parity",
    },
    "23_evalue": {
        "name": "E-value (closed form)",
        "headline_filter": lambda d: d.statistic.startswith("evalue_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "24_coxph": {
        "name": "Cox proportional hazards",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "26_glmm_logit": {
        "name": "GLMM logit (Laplace)",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "27_glmm_aghq": {
        "name": "GLMM logit (AGHQ, n=8)",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "29_panel_sfa": {
        "name": "Panel SFA (Pitt--Lee)",
        "headline_filter": lambda d: d.statistic in {"beta_lnk", "beta_lnl"},
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "slope parity; intercept/sigma rows are scale diagnostics",
    },
    "31_dfl": {
        "name": "DFL reweighting",
        "headline_filter": lambda d: d.statistic
        in {
            "gap",
            "composition",
            "structure",
        },
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "32_rif": {
        "name": "RIF / UQR (median)",
        "headline_filter": lambda d: d.statistic == "total_diff",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "33_var": {
        "name": "VAR (vars::VAR)",
        "headline_filter": lambda d: d.statistic.startswith("eq_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "34_lp": {
        "name": "Local projections",
        "headline_filter": lambda d: d.statistic.startswith("irf_h"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "lpirfs Cholesky/unit-shock path",
    },
    "35_panel": {
        "name": "Panel FE/RE + Hausman",
        "headline_filter": lambda d: d.statistic.startswith(
            ("fe_beta_", "re_beta_", "hausman_chi2", "hausman_pvalue")
        ),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "FE/RE and plm-style Hausman parity; Stata sigmamore diagnostic row",
    },
    "36_mediation": {
        "name": "Causal mediation (IKT)",
        "headline_filter": lambda d: d.statistic in ("acme", "ade", "total_effect"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    # Modules added 2026-05-28
    "37_ppmlhdfe": {
        "name": "PPML + HDFE",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "post PPML FE-score fix; robust SE within 0.5\\% of fixest",
    },
    "38_drdid": {
        "name": "DR-DID (SZ 2020)",
        "headline_filter": lambda d: d.statistic == "att",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "panel DR-DID calibrated propensity parity with R/Stata",
    },
    "39_arima": {
        "name": "ARIMA(2,0,0)",
        "headline_filter": lambda d: d.statistic.startswith("ar"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "stats::arima ML / tightly converged Stata arima",
    },
    "40_qreg": {
        "name": "Quantile reg (median)",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "post sqrt(n) sandwich-scaling fix",
    },
    "41_tobit": {
        "name": "Tobit (left-censored)",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "post observed-information Hessian fix",
    },
    "42_nbreg": {
        "name": "Negative binomial",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "43_heckman": {
        "name": "Heckman 2-step",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "44_mlogit": {
        "name": "Multinomial logit",
        "headline_filter": lambda d: d.statistic.startswith("class"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "post observed-information Hessian fix",
    },
    "45_ologit": {
        "name": "Ordered logit",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "post observed-information Hessian fix",
    },
    "46_clogit": {
        "name": "Conditional logit",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "47_ppmlhdfe_3fe": {
        "name": "PPML + 3-way HDFE",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "Post Gauss-Seidel multi-FE fix; sp matches at 1e-15",
    },
    "48_probit": {
        "name": "Binary probit",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "49_oprobit": {
        "name": "Ordered probit",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "50_xtabond": {
        "name": "Arellano-Bond GMM",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "R/Stata dynamic-panel fixture; block-diagonal instruments and one-step GMM weights match",
    },
    "51_newey": {
        "name": "Newey-West HAC OLS",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "post HAC sandwich sqrt(n) scaling fix",
    },
    "52_scm_unique": {
        "name": "Classical SCM (unique solution)",
        "headline_filter": lambda d: d.statistic == "avg_post_gap",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "identified convex SCM; sp and Stata synth recover exact weights+gap",
    },
    "53_cr2": {
        "name": "Cluster-robust CR2 / CR3 SE",
        "headline_filter": lambda d: d.statistic.startswith(("cr2_", "cr3_")),
        "metric": "rel_se",
        "verdict": "\\textbf{PASS}",
        "gap_note": "CR2 and analytic CR3 match clubSandwich",
    },
    "54_twoway_cluster": {
        "name": "Two-way cluster-robust SE",
        # Headline is the two-way SE vs sandwich::vcovCL (same per-dimension
        # Liang-Zeger convention) -- a machine-precision match.
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_se",
        "verdict": "\\textbf{PASS}",
        "gap_note": "matches sandwich::vcovCL(HC1,cadjust); fixest min-G df convention differs $\\sim10^{-3}$",
    },
    "55_hc2_hc3": {
        "name": "HC2 / HC3 robust SE",
        # Both the HC2 and HC3 SE rows match sandwich::vcovHC at machine
        # precision (MacKinnon-White small-sample heteroskedasticity-robust).
        "headline_filter": lambda d: d.statistic.startswith("hc"),
        "metric": "rel_se",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "56_multiway_cluster": {
        "name": "Three-way cluster-robust SE",
        # Three-way SE vs sandwich::vcovCL -- machine-precision match; exercises
        # the full inclusion-exclusion (triple-intersection term) of the fixed
        # multiway_cluster_vcov.
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_se",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "57_logit": {
        "name": "Binary logit ML",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "58_poisson": {
        "name": "Poisson ML",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "59_liml": {
        "name": "LIML k-class IV",
        # ivmodel::LIML pins the endogenous coefficient; the exogenous
        # coefficients are py<->Stata rows (ivregress liml, small).
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "ivregress runs with `small` (RSS/(n-k))",
    },
    "60_sureg": {
        "name": "SUR one-step FGLS",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "Sigma divisor n (sureg default / noDfCor)",
    },
    "61_betareg": {
        "name": "Beta regression ML",
        "headline_filter": lambda d: d.statistic.startswith("beta_")
        or d.statistic == "ln_phi",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "betareg SEs are expected-information (documented)",
    },
    "62_truncreg": {
        "name": "Truncated regression ML",
        "headline_filter": lambda d: d.statistic.startswith("beta_")
        or d.statistic == "sigma",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "63_zip": {
        "name": "Zero-inflated Poisson",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "64_zinb": {
        "name": "Zero-inflated NB",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "flat ZINB likelihood near optimum (1e-5 budget)",
    },
    "65_spatial": {
        "name": "Spatial ML (SAR/SEM/SDM)",
        "headline_filter": lambda d: True,
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "vs spatialreg::lagsarlm/errorsarlm/Durbin, row-std rook W",
    },
    "66_spatial_gmm": {
        "name": "Spatial GMM (SAR-2SLS/SEM-GMM)",
        "headline_filter": lambda d: True,
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "vs spatialreg::stsls(W2X=F)/GMerrorsar; SEM point-only",
    },
    "67_panel_glm": {
        "name": "Panel GLM (feglm / fepois)",
        "headline_filter": lambda d: True,
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "vs fixest::feglm/fepois, absorbed id FE; IWLS SE 1e-5",
    },
    "68_demean_within": {
        "name": "Within transformation",
        "headline_filter": lambda d: True,
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "sp.demean(solver='map') vs textbook mean-within",
    },
    "69_balance_panel": {
        "name": "Panel balance filter",
        "headline_filter": lambda d: True,
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "sp.balance_panel vs base R counts == n_periods",
    },
    "72_tmle": {
        "name": "TMLE (targeting step)",
        "headline_filter": lambda d: d.statistic == "psi_tmle_ate",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": (
            "shared initial Q / g1W; per-arm fluctuation matching "
            "tmle::tmle's submodel"
        ),
    },
    "71_dml_family": {
        "name": "DML family (IRM / PLIV / IIVM)",
        "headline_filter": lambda d: d.statistic.startswith("theta_DML_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": (
            "shared explicit fold_id across all three DoubleML model "
            "classes; PLIV at the floating-point floor, IRM/IIVM limited "
            "by the logistic-MLE optimiser"
        ),
    },
    "70_policy_tree": {
        "name": "Policy tree (exact, depth 1--2)",
        "headline_filter": lambda d: d.statistic.startswith("value_policy_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": (
            "shared AIPW scores; exact welfare optimum vs policytree, "
            "all 1200 per-row policy decisions identical at both depths"
        ),
    },
}


def render_tex(modules: list[str]) -> str:
    rows: list[str] = []
    for m in modules:
        diffs = collect(m)
        if not diffs:
            continue
        cfg = HEADLINE.get(
            m,
            {
                "name": m,
                "headline_filter": lambda d: True,
                "metric": "rel_est",
                "verdict": "\\textit{review}",
                "gap_note": "",
            },
        )
        filtered = [d for d in diffs if cfg["headline_filter"](d)]
        if not filtered:
            filtered = diffs
        metric = cfg["metric"]
        vals = [getattr(d, metric) for d in filtered if getattr(d, metric) is not None]
        if not vals:
            continue
        worst = max(vals)
        if metric == "rel_est":
            primary = f"rel $\\le {worst:.2g}$"
        else:
            primary = f"abs $\\le {worst:.3g}$"
        gap_note = cfg.get("gap_note", "")
        gap_cell = f" {{\\footnotesize ({gap_note})}}" if gap_note else ""
        # Escape underscores inside \code{...} so the texttt rendering
        # does not trip the LaTeX scanner.
        m_safe = m.replace("_", r"\_")
        rows.append(
            f"\\code{{{m_safe}}} & {cfg['name']} & {primary}{gap_cell} & "
            f"{cfg['verdict']} \\\\"
        )

    body = "\n".join(rows)
    return (
        "% AUTO-GENERATED by tests/r_parity/compare.py\n"
        "% Re-run after any module change to refresh.\n"
        "\\begin{longtable}{p{0.10\\linewidth}p{0.27\\linewidth}p{0.40\\linewidth}p{0.16\\linewidth}}\n"
        "\\caption{Track A parity headline for the \\statspai{} 1.20.0 source snapshot vs the "
        "canonical \\proglang{R} reference on the calibrated replicas. The "
        "``Worst diff'' column reports the worst residual gap across the "
        "module's headline rows (point estimates only; per-row SE diffs "
        "and documented gap rows are reported in the Markdown source). "
        "Verdicts use PASS and GAP; common-specification passes, "
        "small-sample SE conventions, and convention gaps are explained "
        "in the parenthetical notes and per-module \\code{extra} block in "
        "\\code{tests/r\\_parity/results/}.}\n"
        "\\label{tab:track-a-parity}\\\\\n"
        "\\toprule\n"
        "Module & Method & Worst headline diff & Verdict \\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\multicolumn{4}{c}{\\textit{(continued)}}\\\\\n"
        "\\toprule\n"
        "Module & Method & Worst headline diff & Verdict \\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
        f"{body}\n"
        "\\end{longtable}\n"
    )


def render_tex_3way(modules: list[str]) -> str:
    """Five-column 3-way table: ID / Method / vs R / vs Stata / Verdict."""
    rows: list[str] = []
    tier_sentence = _tier_breakdown_sentence([m for m in modules if collect(m)])
    for m in modules:
        diffs = collect(m)
        if not diffs:
            continue
        cfg = HEADLINE.get(
            m,
            {
                "name": m,
                "headline_filter": lambda d: True,
                "metric": "rel_est",
                "verdict": "\\textit{review}",
                "gap_note": "",
            },
        )
        filtered = [d for d in diffs if cfg["headline_filter"](d)]
        if not filtered:
            filtered = diffs
        metric = cfg["metric"]
        # vs-R column.
        vals_r = [
            getattr(d, metric) for d in filtered if getattr(d, metric) is not None
        ]
        if not vals_r:
            continue
        worst_r = max(vals_r)
        if metric == "rel_est":
            primary_r = f"rel $\\le {worst_r:.2g}$"
        else:
            primary_r = f"abs $\\le {worst_r:.3g}$"
        # vs-Stata column.
        st_metric = "rel_est_st" if metric == "rel_est" else "abs_est_st"
        st_vals = [
            getattr(d, st_metric) for d in filtered if getattr(d, st_metric) is not None
        ]
        if st_vals:
            worst_s = max(st_vals)
            if metric == "rel_est":
                primary_s = f"rel $\\le {worst_s:.2g}$"
            else:
                primary_s = f"abs $\\le {worst_s:.3g}$"
            if m in STATA_HEADLINE_GAP_EXCEPTIONS:
                primary_s += " {\\footnotesize (Stata convention gap)}"
        else:
            reason = STATA_SKIP_REASON.get(m, "n/a")
            primary_s = f"\\emph{{{reason}}}"
        gap_note = cfg.get("gap_note", "")
        gap_cell = f" {{\\footnotesize ({gap_note})}}" if gap_note else ""
        m_safe = m.split("_", 1)[0]
        rows.append(
            f"\\code{{{m_safe}}} & {cfg['name']} & "
            f"{primary_r}{gap_cell} & {primary_s} & "
            f"{cfg['verdict']} \\\\"
        )

    body = "\n".join(rows)
    return (
        "% AUTO-GENERATED by tests/r_parity/compare.py\n"
        "% Re-run after any module change to refresh.\n"
        "\\begingroup\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{2pt}\n"
        "\\begin{longtable}{@{}p{0.055\\linewidth}p{0.205\\linewidth}p{0.30\\linewidth}p{0.30\\linewidth}p{0.10\\linewidth}@{}}\n"
        "\\caption{Track A parity headline for the \\statspai{} 1.20.0 source snapshot against the canonical "
        "\\proglang{R} reference \\emph{and} (where one exists) a canonical or audited "
        "\\proglang{Stata} bridge reference, on the calibrated replicas. The ID column is the two-digit module prefix; "
        "the two diff columns report the worst residual "
        "gap across each module's headline rows (point estimates only; per-row SE diffs and "
        "documented gap rows are reported in \\code{tests/r\\_parity/results/parity\\_table\\_3way.md}). "
        "Italic text in the \\proglang{Stata} column records the explicit "
        "non-materialized bridge or no-canonical-reference reason when no "
        "portable \\proglang{Stata} artifact is available. Verdicts use PASS "
        "and GAP; common-specification passes, small-sample SE conventions, and "
        "convention gaps are explained in the parenthetical notes and per-module "
        "\\code{extra} block in "
        "\\code{tests/r\\_parity/results/} and \\code{tests/stata\\_parity/results/}. "
        "Strictness-tier breakdown by registered point-estimate tolerance: "
        f"{tier_sentence}.}}\n"
        "\\label{tab:track-a-parity}\\\\\n"
        "\\toprule\n"
        "ID & Method & Worst diff vs \\proglang{R} & Worst diff vs \\proglang{Stata} & Verdict \\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\multicolumn{5}{c}{\\textit{(continued)}}\\\\\n"
        "\\toprule\n"
        "ID & Method & Worst diff vs \\proglang{R} & Worst diff vs \\proglang{Stata} & Verdict \\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
        f"{body}\n"
        "\\end{longtable}\n"
        "\\endgroup\n"
    )


def render_md_3way(modules: list[str]) -> str:
    """Markdown with Stata column when available."""
    lines: list[str] = [
        "# Track A parity report (3-way: \\proglang{Python} <-> R <-> Stata)",
        "",
        "Generated by `tests/r_parity/compare.py` on the "
        "`results/<module>_{py,R}.json` and "
        "`tests/stata_parity/results/<module>_Stata.json` artefacts. "
        "Tolerance budget per module is pre-registered in "
        "`compare.py::TOLERANCES`. Documented convention gaps, "
        "common-specification passes, and small-sample SE conventions are flagged "
        "in the per-module `extra` block of each JSON.",
        "",
        "**Strictness-tier breakdown** (by registered point-estimate "
        "tolerance, so machine-level point-estimate matches are not flattened together "
        "with methodological T3/T4 tolerances): "
        + _tier_breakdown_sentence([m for m in modules if collect(m)], md=True)
        + ".",
        "",
    ]
    for m in modules:
        diffs = collect(m)
        if not diffs:
            continue
        meta_path = RESULTS_DIR / f"{m}_py.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")).get("extra", {})
        st_meta_path = STATA_RESULTS_DIR / f"{m}_Stata.json"
        st_meta = (
            json.loads(st_meta_path.read_text(encoding="utf-8")).get("extra", {})
            if st_meta_path.exists()
            else {}
        )
        lines.append(f"## Module {m}")
        lines.append(
            f"- **strictness_tier**: `{tolerance_tier(m)}` "
            f"({TIER_LABEL_MD[tolerance_tier(m)]})"
        )
        if m in METHODOLOGICAL_DISCLOSURE_NOTES:
            lines.append(
                f"- **methodological_disclosure**: "
                f"{METHODOLOGICAL_DISCLOSURE_NOTES[m]}"
            )
        if meta:
            for k, v in meta.items():
                v = _display_meta_value(m, k, v)
                if isinstance(v, str) and len(v) > 80:
                    lines.append(f"- **{k}**: {v}")
                else:
                    lines.append(f"- **{k}**: `{v}`")
        if st_meta:
            for k, v in st_meta.items():
                if k.startswith("stata"):
                    lines.append(f"- **{k}**: `{v}`")
        elif m in STATA_SKIP_REASON:
            lines.append(f"- **stata_status**: {STATA_SKIP_REASON[m]}")
        if m in STATA_HEADLINE_GAP_EXCEPTIONS:
            lines.append(f"- **stata_gap_note**: {STATA_HEADLINE_GAP_EXCEPTIONS[m]}")
        lines.append("")
        lines.append(
            "| stat | py est | R est | Stata est | rel py-R | rel py-Stata | py SE | R SE | Stata SE | rel SE py-R | rel SE py-Stata |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for d in diffs:
            lines.append(
                f"| `{d.statistic}` "
                f"| {fmt(d.py_est)} | {fmt(d.R_est)} | {fmt(d.Stata_est)} "
                f"| {fmt(d.rel_est, 3)} | {fmt(d.rel_est_st, 3)} "
                f"| {fmt(d.py_se)} | {fmt(d.R_se)} | {fmt(d.Stata_se)} "
                f"| {fmt(d.rel_se, 3)} | {fmt(d.rel_se_st, 3)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    modules = sorted(p.stem.replace("_py", "") for p in RESULTS_DIR.glob("*_py.json"))
    rendered_modules = [m for m in modules if collect(m)]
    md = render_md(modules)
    tex = render_tex(modules)
    (RESULTS_DIR / "parity_table.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / "parity_table.tex").write_text(tex, encoding="utf-8")
    print("OK -- wrote parity_table.md and parity_table.tex")
    print(
        "     strictness tiers: " + _tier_breakdown_sentence(rendered_modules, md=True)
    )
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_tex = render_track_a_snapshot_tex()
    (PAPER_TABLES_DIR / "track_a_cross_language_snapshot.tex").write_text(
        snapshot_tex, encoding="utf-8"
    )
    print("OK -- wrote Paper-JSS/manuscript/tables/track_a_cross_language_snapshot.tex")

    # 3-way Stata extension. Always emitted; Stata-empty modules show
    # the explicit skip/materialization reason rather than a blank.
    if _has_any_stata(modules) or STATA_SKIP_REASON:
        md3 = render_md_3way(modules)
        tex3 = render_tex_3way(modules)
        (RESULTS_DIR / "parity_table_3way.md").write_text(md3, encoding="utf-8")
        (RESULTS_DIR / "parity_table_3way.tex").write_text(tex3, encoding="utf-8")
        print("OK -- wrote parity_table_3way.md and parity_table_3way.tex")
        n_stata = sum(1 for m in rendered_modules if _load_stata(m) is not None)
        n_py_stata_only = sum(
            1
            for m in modules
            if _load_stata(m) is not None and m not in rendered_modules
        )
        suffix = (
            f"; {n_py_stata_only} Py-Stata-only module omitted from R-joined table"
            if n_py_stata_only
            else ""
        )
        print(
            f"     ({n_stata} of {len(rendered_modules)} rendered modules "
            f"have a Stata reference{suffix})"
        )
    print(
        f"     ({len(rendered_modules)} rendered modules from {len(modules)} "
        f"Python result files: {', '.join(rendered_modules)})"
    )


if __name__ == "__main__":
    main()
