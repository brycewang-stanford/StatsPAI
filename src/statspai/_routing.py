"""Machine-readable estimator routing (roadmap W4).

The seven ``docs/guides/choosing_*_estimator.md`` guides hold the decision
logic — target parameter, design, instrument strength, running-variable
shape, estimand — that turns a research question into an estimator call.
Until now that logic lived only in Markdown that is not shipped in the
wheel, so an agent had to guess or read source. This module states the
same decisions as data:

* :func:`decision_guide` returns a family's questions (with the allowed
  answers) and routes (which answers lead to which call, why, what the
  route additionally assumes, and where in the guide to read more);
* :func:`route` takes answers and returns the matching routes, the
  questions still unanswered, and the one question that would narrow
  the choice most.

The tables are written from the guides, not the other way round; the
test-suite checks that every ``call`` is a registered function and every
``read_more`` anchor is a heading in the guide, so the two cannot drift
apart silently. The guides themselves are packaged
(``statspai/agent/_guides/``) and served over MCP as ``statspai://guide/{family}``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

AnswerSpec = Union[str, Sequence[str]]


@dataclass(frozen=True)
class Question:
    key: str
    text: str
    options: Dict[str, str]  # answer -> meaning


@dataclass(frozen=True)
class Route:
    when: Dict[str, AnswerSpec]  # question key -> accepted answer(s)
    call: str  # registered function name
    example: str
    why: str
    assumptions_added: List[str] = field(default_factory=list)
    read_more: str = ""  # heading text in the family's guide
    also: List[str] = field(default_factory=list)  # follow-up functions

    def matches(self, answers: Dict[str, str]) -> bool:
        for key, accepted in self.when.items():
            given = answers.get(key)
            if given is None:
                return False
            allowed = [accepted] if isinstance(accepted, str) else list(accepted)
            if given not in allowed:
                return False
        return True


@dataclass(frozen=True)
class Family:
    name: str
    guide: str  # file name under docs/guides
    title: str
    questions: List[Question]
    routes: List[Route]


# ---------------------------------------------------------------------------
# DiD
# ---------------------------------------------------------------------------

_DID = Family(
    name="did",
    guide="choosing_did_estimator.md",
    title="Choosing a DID estimator",
    questions=[
        Question(
            "design",
            "What is the treatment design?",
            {
                "2x2": "two periods, one treated and one control group",
                "repeated_cross_section": "2x2 but units are not followed over time",
                "ddd": "triple differences (a within-group eligibility split)",
                "staggered": "units adopt at different calendar times",
            },
        ),
        Question(
            "timing_random",
            "Was adoption timing actually randomised (a lottery, a phased launch)?",
            {"yes": "timing assigned at random", "no": "timing chosen / observational"},
        ),
        Question(
            "covariates",
            "Do covariates need to enter the identifying assumption?",
            {
                "none": "unconditional parallel trends",
                "yes": "conditional parallel trends",
            },
        ),
        Question(
            "target",
            "What is the target parameter?",
            {
                "overall_att": "one number for all treated units",
                "event_study": "effects by time since treatment",
                "cohort": "effects by adoption cohort",
                "calendar": "effects by calendar period",
            },
        ),
        Question(
            "pretrends_concern",
            "Are pre-trends a live concern (visible drift or low test power)?",
            {"yes": "yes", "no": "no"},
        ),
        Question(
            "few_treated",
            "Is the number of treated units / clusters very small (one or a handful)?",
            {"yes": "yes", "no": "no"},
        ),
    ],
    routes=[
        Route(
            when={"design": "2x2", "covariates": "none"},
            call="did",
            example="sp.did(df, y='y', treat='treated', time='post')",
            why="Two periods, two groups: the 2x2 estimator is the estimand itself.",
            assumptions_added=["Parallel trends between the two groups"],
            read_more='1. Two-period, two-group ("2x2 DID")',
            also=["honest_did", "pretrends_test"],
        ),
        Route(
            when={"design": "2x2", "covariates": "yes"},
            call="drdid",
            example="sp.drdid(df, y='y', group='d', time='post', covariates=[...])",
            why=(
                "Under conditional parallel trends the TWFE coefficient with additive "
                "covariates is not the ATT; the doubly-robust estimator is consistent "
                "if either nuisance model is right."
            ),
            assumptions_added=["Conditional parallel trends given the covariates"],
            read_more="Q3 — Do you need covariates, and how do they enter?",
        ),
        Route(
            when={"design": "repeated_cross_section"},
            call="did",
            example="sp.did(df, y='y', treat='treated', time='post', panel=False)",
            why="Repeated cross-sections need the RCS estimator, not the panel one.",
            read_more='1. Two-period, two-group ("2x2 DID")',
        ),
        Route(
            when={"design": "ddd", "covariates": "none"},
            call="ddd",
            example="sp.ddd(df, y='y', treat='d', time='t', subgroup='eligible')",
            why=(
                "Triple differences remove a group-specific shock common to eligible "
                "and ineligible units."
            ),
            read_more='1. Two-period, two-group ("2x2 DID")',
        ),
        Route(
            when={"design": "ddd", "covariates": "yes"},
            call="ddd",
            example=(
                "sp.ddd(df, y='y', treat='d', time='t', subgroup='eligible', "
                "covariates=[...], id='unit', method='dr')"
            ),
            why=(
                "Covariates must not go in the 3WFE regression; use the conditional "
                "(DR) triple-difference estimator."
            ),
            read_more='1. Two-period, two-group ("2x2 DID")',
        ),
        Route(
            when={"design": "staggered", "timing_random": "yes"},
            call="staggered_rollout",
            example=(
                "sp.staggered_rollout(df, y='y', g='first_treat', t='year', i='id', "
                "estimand='simple')"
            ),
            why=(
                "With randomised timing parallel trends is neither assumed nor needed; "
                "the design-based estimator uses the randomisation and reports "
                "Neyman / adjusted standard errors."
            ),
            assumptions_added=["Adoption timing is as good as random"],
            read_more="Q4 — What is random?",
            also=["staggered_cs", "staggered_sa"],
        ),
        Route(
            when={"design": "staggered", "timing_random": "no", "covariates": "none"},
            call="callaway_santanna",
            example=(
                "sp.callaway_santanna(df, y='y', g='first_treat', t='year', i='id')"
            ),
            why=(
                "Group-time ATT(g,t) with never/not-yet-treated controls is robust to "
                "heterogeneous effects where static TWFE is not; aggregate with "
                "sp.aggte."
            ),
            assumptions_added=[
                "Parallel trends for the chosen control group (PT-GT-NEV / PT-GT-NYT)"
            ],
            read_more="2b. Staggered + heterogeneous effects",
            also=["aggte", "sun_abraham", "did_imputation", "bacon_decomposition"],
        ),
        Route(
            when={"design": "staggered", "timing_random": "no", "covariates": "yes"},
            call="callaway_santanna",
            example=(
                "sp.callaway_santanna(df, y='y', g='first_treat', t='year', i='id', "
                "x=[...], estimator='dr')"
            ),
            why="Doubly-robust group-time ATT under conditional parallel trends.",
            assumptions_added=["Conditional parallel trends given the covariates"],
            read_more="Q3 — Do you need covariates, and how do they enter?",
        ),
        Route(
            when={"design": "staggered", "target": "event_study"},
            call="aggte",
            example="sp.aggte(cs, type='dynamic')",
            why=(
                "Event-study aggregation of the group-time ATTs; sp.sun_abraham is the "
                "interaction-weighted alternative."
            ),
            read_more="Q1 — What is the target parameter?",
            also=["sun_abraham", "uniform_bands"],
        ),
        Route(
            when={"design": "staggered", "target": "cohort"},
            call="aggte",
            example="sp.aggte(cs, type='group')",
            why="Cohort-specific effects θ(g).",
            read_more="Q1 — What is the target parameter?",
        ),
        Route(
            when={"design": "staggered", "target": "calendar"},
            call="aggte",
            example="sp.aggte(cs, type='calendar')",
            why="Calendar-period effects θ(t).",
            read_more="Q1 — What is the target parameter?",
        ),
        Route(
            when={"pretrends_concern": "yes"},
            call="honest_did",
            example="sp.honest_did(result, m_grid=[0.0, 0.1, 0.2])",
            why=(
                "Rambachan-Roth honest confidence sets bound the post-treatment effect "
                "under restricted deviations from parallel trends; a passed pre-test "
                "is not evidence for it."
            ),
            read_more="3. Sensitivity and robustness",
            also=["pretrends_power", "pretrends_test"],
        ),
        Route(
            when={"few_treated": "yes"},
            call="did_few_treated",
            example=(
                "sp.did_few_treated(df, y='y', treat='treated', time='post', id='id')"
            ),
            why=(
                "Cluster-robust SEs over-reject badly with one or a few treated "
                "clusters; the placebo-based inversion is the appropriate inference."
            ),
            read_more="3. Sensitivity and robustness",
        ),
    ],
)

# ---------------------------------------------------------------------------
# IV
# ---------------------------------------------------------------------------

_IV = Family(
    name="iv",
    guide="choosing_iv_estimator.md",
    title="Choosing an IV estimator",
    questions=[
        Question(
            "strength",
            "How strong is the instrument (first-stage / effective F)?",
            {
                "strong": "F > 10 (ideally > 30)",
                "weak": "F < 10",
                "very_weak": "F near or below 5",
            },
        ),
        Question(
            "n_instruments",
            "How many instruments?",
            {"few": "1-3", "many": "many relative to n"},
        ),
        Question(
            "exogeneity",
            "How credible is the exclusion restriction?",
            {
                "tight": "exclusion holds",
                "plausible": "small direct effect possible",
                "untestable": "cannot be defended",
            },
        ),
        Question(
            "instrument_type",
            "What kind of instrument?",
            {
                "standard": "excluded variable(s)",
                "shift_share": "shares x shocks (Bartik)",
            },
        ),
        Question(
            "estimand",
            "Which estimand?",
            {
                "late": "complier average effect",
                "mte": "marginal treatment effects over the unobserved resistance",
                "distribution": "distributional / quantile effects",
            },
        ),
    ],
    routes=[
        Route(
            when={"strength": "strong", "n_instruments": "few", "exogeneity": "tight"},
            call="ivreg",
            example="sp.ivreg('y ~ x1 + (d ~ z1 + z2)', data=df, robust='hc1')",
            why=(
                "2SLS with robust SEs; report first-stage F, endogeneity and "
                "over-identification tests."
            ),
            assumptions_added=["Instrument relevance and exclusion"],
            read_more="1. The default: 2SLS with robust SE",
            also=["iv_diag", "effective_f_test"],
        ),
        Route(
            when={"strength": "weak"},
            call="liml",
            example="sp.liml(df, y='y', x_endog=['d'], z=['z1', 'z2'], fuller=1)",
            why=(
                "LIML / Fuller are less biased than 2SLS under weak instruments; "
                "report Anderson-Rubin sets alongside."
            ),
            read_more="2. Weak instruments",
            also=["anderson_rubin_ci", "tF_critical_value"],
        ),
        Route(
            when={"strength": "very_weak"},
            call="anderson_rubin_ci",
            example="sp.anderson_rubin_ci(result)",
            why=(
                "Anderson-Rubin confidence sets are valid at any first-stage strength; "
                "do not report 2SLS t-ratios."
            ),
            read_more="2. Weak instruments",
            also=["conditional_lr_ci", "effective_f_test"],
        ),
        Route(
            when={"n_instruments": "many"},
            call="iv",
            example="sp.iv('y ~ (d ~ z1 + ... + z50)', data=df, method='ujive')",
            why=(
                "Jackknife IV (UJIVE) removes the many-instrument bias of 2SLS; "
                "post-lasso IV selects instruments with valid inference."
            ),
            read_more="3. Many instruments",
            also=["jive", "rlasso_iv"],
        ),
        Route(
            when={"exogeneity": "plausible"},
            call="iv",
            example="sp.iv('y ~ (d ~ z)', data=df, method='ltz', gamma_grid=[...])",
            why=(
                "Conley-Hansen-Rossi plausibly-exogenous analysis shows how the "
                "conclusion moves with a direct Z→Y channel."
            ),
            read_more="4. Plausibly exogenous instruments",
        ),
        Route(
            when={"exogeneity": "untestable"},
            call="partial_identification",
            example="sp.partial_identification(df, y='y', treat='d', instrument='z')",
            why=(
                "Without a defensible exclusion restriction only bounds are identified."
            ),
            read_more="4. Plausibly exogenous instruments",
        ),
        Route(
            when={"instrument_type": "shift_share"},
            call="bartik",
            example="sp.bartik(df, y='y', shares=[...], shocks=[...], ...)",
            why=(
                "Shift-share designs need shock-level (Adão-Kolesár-Morales / "
                "Borusyak-Hull-Jaravel) inference, not plain 2SLS SEs."
            ),
            read_more="7. Shift-share / Bartik IV",
            also=["shift_share_se", "ssaggregate"],
        ),
        Route(
            when={"estimand": "distribution"},
            call="dist_iv",
            example="sp.dist_iv(df, y='y', treat='d', instrument='z')",
            why="Distributional / quantile effects for compliers rather than the LATE.",
            read_more="6. Fuzzy / discrete treatments",
            also=["beyond_average_late"],
        ),
    ],
)

# ---------------------------------------------------------------------------
# RD
# ---------------------------------------------------------------------------

_RD = Family(
    name="rd",
    guide="choosing_rd_estimator.md",
    title="Choosing an RD estimator",
    questions=[
        Question(
            "assignment",
            "Is treatment deterministic at the cutoff?",
            {
                "sharp": "P(D=1 | X >= c) = 1",
                "fuzzy": "the cutoff shifts treatment probability",
                "none": "no change in treatment probability",
            },
        ),
        Question(
            "running",
            "What does the running variable look like?",
            {
                "continuous": "continuous score",
                "discrete": "discrete / time-based (few mass points)",
                "kink": "the slope, not the level, changes at the cutoff",
                "two_dimensional": "two running variables (geographic boundary)",
                "multiple_cutoffs": "several cutoffs (districts, cohorts)",
            },
        ),
        Question(
            "inference",
            "Which inference framework?",
            {
                "local_polynomial": "continuity-based (CCT robust bias-corrected)",
                "local_randomization": "as-if random in a window",
                "honest": (
                    "Armstrong-Kolesár honest CIs under a bound on the second "
                    "derivative"
                ),
            },
        ),
        Question(
            "heterogeneity",
            "Do you need heterogeneous effects at the cutoff?",
            {"yes": "yes", "no": "no"},
        ),
        Question(
            "manipulation",
            "Is sorting / manipulation at the cutoff a concern?",
            {"yes": "yes", "no": "no"},
        ),
    ],
    routes=[
        Route(
            when={
                "assignment": "sharp",
                "running": "continuous",
                "inference": "local_polynomial",
            },
            call="rdrobust",
            example=(
                "sp.rdrobust(df, y='y', x='running', c=0.0, kernel='triangular', "
                "bwselect='mserd')"
            ),
            why=(
                "Calonico-Cattaneo-Titiunik bias-corrected local polynomial with "
                "robust CIs is the default sharp RD."
            ),
            assumptions_added=[
                "Continuity of the potential-outcome regression functions at the cutoff"
            ],
            read_more="1. The default: sharp RD with CCT-robust CI",
            also=["rddensity", "rdplot", "rdbwselect"],
        ),
        Route(
            when={"assignment": "fuzzy", "running": "continuous"},
            call="rdrobust",
            example="sp.rdrobust(df, y='y', x='running', c=0.0, fuzzy='treatment')",
            why=(
                "Fuzzy RD is a local Wald ratio; report the first-stage jump and a "
                "Kitagawa-type validity test."
            ),
            assumptions_added=["Monotonicity of compliance at the cutoff"],
            read_more="2. Fuzzy RD",
            also=["kitagawa_test"],
        ),
        Route(
            when={"assignment": "none"},
            call="bunching",
            example="sp.bunching(df, x='running', cutoff=0.0)",
            why=(
                "No change in treatment probability means RD is not identified; a "
                "bunching design or DiD is the fallback."
            ),
            read_more="8. When NOT to use RD",
        ),
        Route(
            when={"running": "discrete"},
            call="rdit",
            example="sp.rdit(df, y='y', x='date', c=cutoff_date)",
            why=(
                "Regression discontinuity in time / discrete running variables need "
                "the RDiT machinery and clustered-by-mass-point inference."
            ),
            read_more="3. Decision tree for method variants",
        ),
        Route(
            when={"running": "kink"},
            call="rkd",
            example="sp.rkd(df, y='y', x='running', c=0.0)",
            why="A slope change identifies a regression kink effect, not a level jump.",
            read_more="3. Decision tree for method variants",
        ),
        Route(
            when={"running": "two_dimensional"},
            call="rd2d",
            example="sp.rd2d(df, y='y', x1='lat', x2='lon', boundary=...)",
            why="Two running variables define a boundary; effects vary along it.",
            read_more="3. Decision tree for method variants",
        ),
        Route(
            when={"running": "multiple_cutoffs"},
            call="rdmc",
            example="sp.rdmc(df, y='y', x='running', c='cutoff_col')",
            why="Multiple cutoffs are pooled with cutoff-specific normalisation.",
            read_more="3. Decision tree for method variants",
        ),
        Route(
            when={"inference": "local_randomization"},
            call="rdrandinf",
            example="sp.rdrandinf(df, y='y', x='running', c=0.0, window=(-w, w))",
            why=(
                "Local randomization treats units in a window as an experiment; choose "
                "the window with sp.rdwinselect."
            ),
            assumptions_added=["As-if random assignment inside the window"],
            read_more="3. Decision tree for method variants",
            also=["rdwinselect"],
        ),
        Route(
            when={"inference": "honest"},
            call="rd_honest",
            example="sp.rd_honest(df, y='y', x='running', c=0.0, M=0.05)",
            why=(
                "Honest CIs are valid uniformly over functions with bounded second "
                "derivative M."
            ),
            read_more="3. Decision tree for method variants",
        ),
        Route(
            when={"heterogeneity": "yes"},
            call="rdhte",
            example="sp.rdhte(df, y='y', x='running', c=0.0, covs=[...])",
            why=(
                "Heterogeneous RD effects by covariate; sp.rd_forest for a "
                "nonparametric version."
            ),
            read_more="3. Decision tree for method variants",
            also=["rd_forest"],
        ),
        Route(
            when={"manipulation": "yes"},
            call="rddensity",
            example="sp.rddensity(df, x='running', c=0.0)",
            why=(
                "A density discontinuity at the cutoff is evidence of sorting; pair "
                "with a donut-hole specification."
            ),
            read_more="4. Mandatory diagnostics",
            also=["bunching"],
        ),
    ],
)

# ---------------------------------------------------------------------------
# Matching / weighting
# ---------------------------------------------------------------------------

_MATCHING = Family(
    name="matching",
    guide="choosing_matching_estimator.md",
    title="Choosing a matching / weighting estimator",
    questions=[
        Question(
            "estimand",
            "Which estimand?",
            {
                "att": "effect on the treated",
                "ate": "effect on the population",
                "ato": "effect on the overlap population",
                "atc": "effect on the controls",
                "cate": "conditional on covariates",
            },
        ),
        Question(
            "covariates",
            "How many covariates relative to n?",
            {"few": "a handful", "many": "high-dimensional"},
        ),
        Question(
            "overlap",
            "Is overlap (common support) good?",
            {
                "good": "propensity scores away from 0 and 1",
                "poor": "thin or no overlap in parts of the covariate space",
            },
        ),
    ],
    routes=[
        Route(
            when={"estimand": "att", "covariates": "few"},
            call="ebalance",
            example="sp.ebalance(df, y='y', treat='d', covariates=[...])",
            why=(
                "Entropy balancing hits exact moment balance for the treated with no "
                "model search."
            ),
            assumptions_added=["Unconfoundedness given the balanced moments"],
            read_more='1. Entropy balancing (ebal) — the "just works" default for ATT',
            also=["match", "love_plot"],
        ),
        Route(
            when={"estimand": "ate", "covariates": "few"},
            call="cbps",
            example="sp.cbps(df, y='y', treat='d', covariates=[...], estimand='ATE')",
            why=(
                "Covariate-balancing propensity scores target ATE balance directly; "
                "sp.aipw is the doubly-robust alternative."
            ),
            read_more="3. Covariate Balancing Propensity Score (CBPS)",
            also=["aipw"],
        ),
        Route(
            when={"estimand": "ato"},
            call="overlap_weights",
            example="sp.overlap_weights(df, y='y', treat='d', covariates=[...])",
            why=(
                "Overlap weights emphasise units with propensity near 0.5 and are "
                "bounded by construction."
            ),
            read_more="4. Overlap weights (ATO)",
        ),
        Route(
            when={"estimand": "atc"},
            call="match",
            example="sp.match(df, y='y', treat='d', covariates=[...], estimand='ATC')",
            why=(
                "Nearest-neighbour matching with the control units as the target "
                "population."
            ),
            read_more="2. Nearest-neighbor matching",
        ),
        Route(
            when={"estimand": "cate"},
            call="metalearner",
            example=(
                "sp.metalearner(df, y='y', treat='d', covariates=[...], learner='x')"
            ),
            why=(
                "Meta-learners and causal forests estimate conditional effects; report "
                "calibration, not only the CATE map."
            ),
            read_more="6. Meta-learners (for heterogeneous effects)",
            also=["causal_forest", "calibrate_cate"],
        ),
        Route(
            when={"covariates": "many"},
            call="dml",
            example="sp.dml(df, y='y', treat='d', covariates=[...], model='irm')",
            why=(
                "Double / debiased ML handles high-dimensional nuisances with "
                "cross-fitting and valid inference."
            ),
            read_more="5. Doubly-robust estimators",
        ),
        Route(
            when={"overlap": "poor"},
            call="trimming",
            example="sp.trimming(df, treatment='d', covariates=[...], method='crump')",
            why=(
                "Trim to the overlap region (Crump et al.) or switch to overlap "
                "weights; report the trimmed share."
            ),
            read_more="8. Mandatory diagnostics",
            also=["overlap_weights", "overlap_plot"],
        ),
    ],
)

# ---------------------------------------------------------------------------
# ML causal
# ---------------------------------------------------------------------------

_ML = Family(
    name="ml_causal",
    guide="choosing_ml_causal_estimator.md",
    title="Choosing an ML-based causal estimator",
    questions=[
        Question(
            "goal",
            "What do you need?",
            {
                "ate": "a population average effect",
                "late": "an IV effect with ML nuisances",
                "cate": "the conditional effect function",
            },
        ),
        Question(
            "treatment",
            "Treatment type?",
            {"binary": "binary", "continuous": "continuous"},
        ),
        Question(
            "outcome",
            "Outcome type?",
            {"continuous": "continuous", "binary": "binary"},
        ),
        Question(
            "cate_style",
            "For a CATE, which style?",
            {
                "tree": "nonparametric, honest forest",
                "dr_rloss": "doubly-robust / R-loss meta-learner",
            },
        ),
    ],
    routes=[
        Route(
            when={"goal": "ate", "treatment": "binary", "outcome": "continuous"},
            call="dml",
            example="sp.dml(df, y='y', treat='d', covariates=[...], model='irm')",
            why=(
                "Interactive regression model DML gives the ATE with Neyman-orthogonal "
                "scores."
            ),
            read_more="`dml` — Double / Debiased ML [chernozhukov2018double]",
        ),
        Route(
            when={"goal": "ate", "treatment": "continuous"},
            call="dml",
            example="sp.dml(df, y='y', treat='d', covariates=[...], model='plr')",
            why="Partially linear DML for a continuous treatment.",
            read_more="`dml` — Double / Debiased ML [chernozhukov2018double]",
        ),
        Route(
            when={"goal": "ate", "outcome": "binary"},
            call="tmle",
            example="sp.tmle(df, y='y', treat='d', covariates=[...])",
            why=(
                "TMLE with a Super Learner respects the outcome's support and targets "
                "the ATE with the efficient influence function."
            ),
            read_more="`tmle` — Targeted Maximum Likelihood [vanderlaan2006targeted]",
        ),
        Route(
            when={"goal": "late"},
            call="dml",
            example=(
                "sp.dml(df, y='y', treat='d', covariates=[...], instruments=['z'], "
                "model='iivm')"
            ),
            why=(
                "Interactive IV model (binary instrument) or partially linear IV for "
                "the LATE with ML nuisances."
            ),
            read_more="`dml` — Double / Debiased ML [chernozhukov2018double]",
        ),
        Route(
            when={"goal": "cate", "cate_style": "tree"},
            call="causal_forest",
            example="sp.causal_forest(df, y='y', treat='d', covariates=[...])",
            why=(
                "Honest generalized random forest with out-of-bag CATEs and "
                "doubly-robust ATE / BLP / calibration."
            ),
            read_more=(
                "`causal_forest` — honest random forest [athey2019generalized; "
                "wager2018estimation]"
            ),
            also=["best_linear_projection", "calibrate_cate", "rate"],
        ),
        Route(
            when={"goal": "cate", "cate_style": "dr_rloss"},
            call="metalearner",
            example=(
                "sp.metalearner(df, y='y', treat='d', covariates=[...], learner='dr')"
            ),
            why=(
                "DR- / R-learners are doubly robust for the CATE and accept any base "
                "learner."
            ),
            read_more=(
                "`metalearner` — S/T/X/R/DR-Learner [kunzel2019metalearners; "
                "nie2021quasi]"
            ),
        ),
    ],
)

# ---------------------------------------------------------------------------
# QTE
# ---------------------------------------------------------------------------

_QTE = Family(
    name="qte",
    guide="choosing_qte_estimator.md",
    title="Choosing a QTE estimator",
    questions=[
        Question(
            "estimand",
            "Which quantile estimand?",
            {
                "unconditional": "unconditional QTE for everyone",
                "treated": "QTT among the treated",
                "conditional": "coefficient in a conditional quantile regression",
                "distribution": "the whole counterfactual distribution",
            },
        ),
        Question(
            "design",
            "What is the design?",
            {
                "cross_section": "unconfoundedness in a cross-section",
                "endogenous": "endogenous treatment with an instrument",
                "two_period": "two periods / repeated cross-sections",
                "three_period_panel": "three balanced panel periods",
                "panel_many_controls": "panel with many controls",
            },
        ),
        Question(
            "outcome",
            "Outcome type?",
            {"continuous": "continuous", "discrete": "discrete / mass points"},
        ),
    ],
    routes=[
        Route(
            when={"estimand": "unconditional", "design": "cross_section"},
            call="qte",
            example=(
                "sp.qte(df, y='wage', treatment='program', covariates=[...], "
                "method='firpo_qte')"
            ),
            why="Firpo's IPW unconditional QTE under unconfoundedness and overlap.",
            read_more="`sp.qte` — cross-section",
        ),
        Route(
            when={"estimand": "treated", "design": "cross_section"},
            call="qte",
            example=(
                "sp.qte(df, y='wage', treatment='program', covariates=[...], "
                "method='firpo_qtt')"
            ),
            why="The same contrast among the treated.",
            read_more="`sp.qte` — cross-section",
        ),
        Route(
            when={"estimand": "conditional"},
            call="qte",
            example=(
                "sp.qte(df, y='wage', treatment='program', covariates=[...], "
                "method='conditional_qr')"
            ),
            why=(
                "A conditional quantile-regression coefficient; no causal reading "
                "without rank invariance."
            ),
            read_more="`sp.qte` — cross-section",
        ),
        Route(
            when={"design": "endogenous"},
            call="dist_iv",
            example="sp.dist_iv(df, y='y', treat='d', instrument='z')",
            why=(
                "Distributional IV for compliers (random assignment, exclusion, "
                "monotonicity)."
            ),
            read_more="`sp.dist_iv` / `sp.beyond_average_late` — endogenous treatment",
            also=["beyond_average_late"],
        ),
        Route(
            when={"design": "two_period", "outcome": "continuous"},
            call="qdid",
            example="sp.qdid(df, y='y', treat='d', time='post', method='cic')",
            why=(
                "Changes-in-changes (Athey-Imbens) is preferred; the quantile-DiD "
                "variant needs a constant rank-shift assumption."
            ),
            read_more="`sp.qdid` — repeated cross-section / two-period panel",
        ),
        Route(
            when={"design": "two_period", "outcome": "discrete"},
            call="qdid",
            example="sp.qdid(df, y='y', treat='d', time='post', method='cic')",
            why=(
                "With discrete outcomes changes-in-changes identifies only bounds; "
                "report them as bounds."
            ),
            read_more="`sp.qdid` — repeated cross-section / two-period panel",
        ),
        Route(
            when={"design": "three_period_panel"},
            call="panel_qtet",
            example="sp.panel_qtet(df, y='y', treat='d', unit='id', time='year')",
            why=(
                "Callaway-Li distributional DiD with copula stability needs three "
                "balanced periods."
            ),
            read_more="`sp.panel_qtet` — three-period panel, Callaway & Li (2019)",
        ),
        Route(
            when={"design": "panel_many_controls"},
            call="qte_hd_panel",
            example=(
                "sp.qte_hd_panel(df, y='y', treat='d', unit='id', time='year', "
                "method='canay')"
            ),
            why=(
                "Canay's two-step treats the unit effect as a location shift; large T "
                "needed."
            ),
            read_more="`sp.qte_hd_panel` — panel with many controls",
        ),
        Route(
            when={"estimand": "distribution"},
            call="distributional_te",
            example=(
                "sp.distributional_te(df, y='y', treat='d', covariates=[...], "
                "method='dr')"
            ),
            why="The whole counterfactual distribution rather than selected quantiles.",
            read_more="`sp.distributional_te` — the whole counterfactual distribution",
        ),
    ],
)

# ---------------------------------------------------------------------------
# Dynamic panel
# ---------------------------------------------------------------------------

_DYNPANEL = Family(
    name="dynamic_panel",
    guide="choosing_dynamic_panel_estimator.md",
    title="Choosing a dynamic panel estimator",
    questions=[
        Question(
            "persistence",
            "How persistent is the series?",
            {
                "moderate": "rho well below 1",
                "near_unit_root": "rho near 1, or difference GMM returns rho >= 1",
            },
        ),
        Question(
            "gaps", "Does the panel have interior holes?", {"yes": "yes", "no": "no"}
        ),
        Question(
            "instrument_count",
            "Do instruments outnumber units?",
            {"yes": "T large enough that they do", "no": "no"},
        ),
        Question(
            "heteroskedasticity",
            "Heteroskedasticity suspected?",
            {"yes": "yes (almost always)", "no": "no"},
        ),
    ],
    routes=[
        Route(
            when={"persistence": "moderate"},
            call="xtabond",
            example="sp.xtabond(df, y='n', x=['w', 'k'], id='id', time='year', lags=2)",
            why="Arellano-Bond difference GMM for short T and moderate persistence.",
            assumptions_added=["Sequential exogeneity; no AR(2) in differenced errors"],
            read_more="2. Difference GMM — `sp.xtabond`",
        ),
        Route(
            when={"persistence": "near_unit_root"},
            call="xtdpdsys",
            example=(
                "sp.xtdpdsys(df, y='n', x=['w', 'k'], id='id', time='year', "
                "twostep=True)"
            ),
            why=(
                "System GMM adds the level equation; lagged differences instrument "
                "levels when the series is persistent."
            ),
            assumptions_added=["Stationarity of initial conditions"],
            read_more="3. System GMM — `sp.xtdpdsys` / `method='system'`",
        ),
        Route(
            when={"gaps": "yes"},
            call="xtabond",
            example="sp.xtabond(..., orthogonal=True)",
            why="Forward orthogonal deviations keep observations with interior holes.",
            read_more="`orthogonal=True` — gaps",
        ),
        Route(
            when={"instrument_count": "yes"},
            call="xtabond",
            example="sp.xtabond(..., collapse=True)",
            why=(
                "Collapsing the instrument matrix stops proliferation from weakening "
                "the Hansen test."
            ),
            read_more="`collapse=True` — instrument proliferation",
        ),
        Route(
            when={"heteroskedasticity": "yes"},
            call="xtabond",
            example="sp.xtabond(..., twostep=True, robust=True)",
            why="Two-step with Windmeijer-corrected robust SEs.",
            read_more="`twostep=True, robust=True` — inference",
        ),
    ],
)

FAMILIES: Dict[str, Family] = {
    f.name: f for f in (_DID, _IV, _RD, _MATCHING, _ML, _QTE, _DYNPANEL)
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _family(name: str) -> Family:
    key = (name or "").strip().lower()
    if key not in FAMILIES:
        from .exceptions import MethodIncompatibility

        raise MethodIncompatibility(
            f"Unknown routing family {name!r}.",
            recovery_hint=f"Choose one of {sorted(FAMILIES)}.",
            diagnostics={"family": name, "available": sorted(FAMILIES)},
        )
    return FAMILIES[key]


def decision_guide(family: Optional[str] = None) -> Dict[str, Any]:
    """Return the machine-readable decision table for an estimator family.

    Parameters
    ----------
    family : str, optional
        One of ``'did'``, ``'iv'``, ``'rd'``, ``'matching'``,
        ``'ml_causal'``, ``'qte'``, ``'dynamic_panel'``. ``None`` lists the
        families with their questions.

    Returns
    -------
    dict
        ``{'family', 'title', 'guide', 'questions': [...], 'routes': [...]}``
        — every route names a registered function, an example call, why it
        is the right one, the assumptions it adds and the guide heading to
        read.

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.decision_guide('did')
    >>> [q['key'] for q in g['questions']][:3]
    ['design', 'timing_random', 'covariates']
    >>> sorted(sp.decision_guide())
    ['did', 'dynamic_panel', 'iv', 'matching', 'ml_causal', 'qte', 'rd']
    """
    if family is None:
        return {
            name: {
                "title": f.title,
                "guide": f.guide,
                "questions": [q.key for q in f.questions],
            }
            for name, f in FAMILIES.items()
        }
    f = _family(family)
    return {
        "family": f.name,
        "title": f.title,
        "guide": f.guide,
        "questions": [asdict(q) for q in f.questions],
        "routes": [asdict(r) for r in f.routes],
    }


def route(family: str, **answers: str) -> Dict[str, Any]:
    """Route a research question to estimator calls.

    Answer the family's questions (see :func:`decision_guide`) as keyword
    arguments; every route whose conditions the answers satisfy is
    returned, most specific first, together with the questions still
    unanswered and the single question that would narrow the choice most.

    Parameters
    ----------
    family : str
        ``'did'``, ``'iv'``, ``'rd'``, ``'matching'``, ``'ml_causal'``,
        ``'qte'`` or ``'dynamic_panel'``.
    **answers
        ``question_key=answer``; unknown keys and unknown answers raise.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.route('did', design='staggered', timing_random='no', covariates='none')
    >>> r['routes'][0]['call']
    'callaway_santanna'
    >>> sp.route('rd', assignment='sharp', running='continuous',
    ...          inference='local_polynomial')['routes'][0]['call']
    'rdrobust'
    """
    f = _family(family)
    valid = {q.key: q for q in f.questions}
    from .exceptions import MethodIncompatibility

    for key, val in answers.items():
        if key not in valid:
            raise MethodIncompatibility(
                f"Unknown question {key!r} for family {f.name!r}.",
                recovery_hint=f"Questions: {sorted(valid)}.",
                diagnostics={"family": f.name, "unknown": key},
            )
        if val not in valid[key].options:
            raise MethodIncompatibility(
                f"Unknown answer {val!r} for question {key!r}.",
                recovery_hint=f"Answers: {sorted(valid[key].options)}.",
                diagnostics={"family": f.name, "question": key, "answer": val},
            )
    matched = [r for r in f.routes if r.matches(answers)]
    # Most specific (most conditions satisfied) first, stable otherwise.
    matched.sort(key=lambda r: -len(r.when))
    # Routes that could still match once more questions are answered.
    pending = [
        r
        for r in f.routes
        if r not in matched
        and all(
            answers.get(k) is None
            or answers[k] in ([v] if isinstance(v, str) else list(v))
            for k, v in r.when.items()
        )
    ]
    unanswered = [q.key for q in f.questions if q.key not in answers]
    # The question that appears in the most still-possible routes.
    counts: Dict[str, int] = {}
    for r in pending:
        for k in r.when:
            if k not in answers:
                counts[k] = counts.get(k, 0) + 1
    next_q = (
        max(counts, key=lambda k: (counts[k], -unanswered.index(k))) if counts else None
    )
    return {
        "family": f.name,
        "answers": dict(answers),
        "routes": [asdict(r) for r in matched],
        "pending_routes": [r.call for r in pending],
        "unanswered": unanswered,
        "next_question": (asdict(valid[next_q]) if next_q else None),
        "guide": f.guide,
    }


__all__ = ["FAMILIES", "Family", "Question", "Route", "decision_guide", "route"]
