"""
Function registry for AI agent consumption.

Provides machine-readable metadata (JSON-schema-compatible) for every
public StatsPAI function, enabling LLM agents to discover, understand,
and call the right estimator without reading source code.

Usage
-----
>>> import statspai as sp
>>> sp.list_functions()                 # human-friendly list
>>> sp.describe_function('did')         # detailed schema for one function
>>> sp.search_functions('treatment')    # keyword search
>>> sp.function_schema('regress')       # OpenAI function-calling schema
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ParamSpec:
    """Specification for a single function parameter."""
    name: str
    type: str
    required: bool = True
    default: Any = None
    description: str = ""
    enum: Optional[List[str]] = None


@dataclass
class FailureMode:
    """One failure mode for agent-native recovery.

    Parameters
    ----------
    symptom : str
        What the agent observes (exception class, warning text, pattern).
    exception : str
        Fully-qualified exception name (``"statspai.AssumptionViolation"``
        or ``"ValueError"``). Agents should ``except`` on this.
    remedy : str
        One-sentence, actionable recovery hint.
    alternative : str, optional
        ``sp.xxx`` to try next when this failure mode triggers.
    """
    symptom: str
    exception: str
    remedy: str
    alternative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionSpec:
    """Machine-readable specification for a StatsPAI function.

    Agent-native fields (``assumptions`` / ``failure_modes`` /
    ``alternatives`` / ``typical_n_min`` / ``pre_conditions``) are
    optional — any entry without them still renders correctly; only
    the agent-card layer surfaces the extras.
    """
    name: str
    category: str
    description: str
    params: List[ParamSpec] = field(default_factory=list)
    returns: str = ""
    example: str = ""
    tags: List[str] = field(default_factory=list)
    reference: str = ""  # paper / method reference
    # ------------------------------------------------------------------ #
    #  Agent-native metadata (all optional; populate per-estimator)
    # ------------------------------------------------------------------ #
    assumptions: List[str] = field(default_factory=list)
    """Identifying / statistical assumptions, human-readable one-liners."""
    pre_conditions: List[str] = field(default_factory=list)
    """Data-shape preconditions the agent should verify before calling."""
    failure_modes: List[FailureMode] = field(default_factory=list)
    """Common failures + recovery paths (see :class:`FailureMode`)."""
    alternatives: List[str] = field(default_factory=list)
    """Ranked ``sp.xxx`` fallbacks when this estimator is a poor fit."""
    typical_n_min: Optional[int] = None
    """Rule-of-thumb minimum sample size; ``None`` if not applicable."""

    def to_openai_schema(self) -> Dict[str, Any]:
        """Export as OpenAI function-calling compatible JSON schema."""
        properties = {}
        required = []
        for p in self.params:
            prop: Dict[str, Any] = {"description": p.description}
            # Map Python types to JSON schema types
            type_map = {
                "str": "string", "int": "integer", "float": "number",
                "bool": "boolean", "DataFrame": "string",
                "ndarray": "string", "list": "array",
                "EconometricResults": "string",
            }
            prop["type"] = type_map.get(p.type, "string")
            # JSON schema requires "items" for array types
            if prop["type"] == "array":
                prop["items"] = {"type": "string"}
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def agent_card(self) -> Dict[str, Any]:
        """Return the agent-native view of this function.

        This is the structured payload rendered into guide
        ``## For Agents`` sections, surfaced by :func:`sp.describe_function`
        and consumed by :func:`sp.recommend` / agent code. It is a
        superset of :meth:`to_openai_schema` plus the agent-native
        fields (``assumptions`` / ``failure_modes`` / ``alternatives`` /
        ``typical_n_min`` / ``pre_conditions``).
        """
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "signature": self.to_openai_schema(),
            "pre_conditions": list(self.pre_conditions),
            "assumptions": list(self.assumptions),
            "failure_modes": [fm.to_dict() for fm in self.failure_modes],
            "alternatives": list(self.alternatives),
            "typical_n_min": self.typical_n_min,
            "reference": self.reference,
            "example": self.example,
        }


# ====================================================================== #
#  Registry
# ====================================================================== #

_REGISTRY: Dict[str, FunctionSpec] = {}
_BASE_REGISTRY_BUILT = False


def register(spec: FunctionSpec) -> FunctionSpec:
    """Register a function specification."""
    _REGISTRY[spec.name] = spec
    return spec


def _build_registry():
    """Populate the registry with all public StatsPAI functions.

    Idempotent via the ``_BASE_REGISTRY_BUILT`` sentinel.  The older
    ``if _REGISTRY: return`` gate was unsafe — any user or test that
    called :func:`register` before the first ``_ensure_full_registry``
    would cause the hand-written block below to be skipped, stripping
    agent-native metadata from flagship families like ``regress``.
    """
    global _BASE_REGISTRY_BUILT
    if _BASE_REGISTRY_BUILT:
        return  # already built

    # -- Regression ---------------------------------------------------- #
    register(FunctionSpec(
        name="regress",
        category="regression",
        description="OLS regression with robust/clustered standard errors. The workhorse of econometric analysis.",
        params=[
            ParamSpec("formula", "str", True, description="R-style formula, e.g. 'y ~ x1 + x2'"),
            ParamSpec("data", "DataFrame", True, description="pandas DataFrame with variables"),
            ParamSpec("robust", "str", False, "nonrobust", "Standard error type", ["nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"]),
            ParamSpec("cluster", "str", False, description="Column name for cluster-robust SEs"),
        ],
        returns="EconometricResults",
        example='sp.regress("wage ~ education + experience", data=df, robust="hc1")',
        tags=["regression", "ols", "linear", "robust"],
        pre_conditions=[
            "data is a pandas DataFrame with every variable in formula as a column",
            "outcome is numeric; non-numeric regressors should be categorical (handled via patsy)",
            "no perfect collinearity among regressors",
        ],
        assumptions=[
            "Conditional mean independence: E[u|X] = 0",
            "No perfect collinearity",
            "For valid inference: homoskedastic errors (relax with robust='hc1'/'hc3')",
            "For cluster-robust SEs: enough clusters (≥ 30–50) and no cross-cluster dependence",
        ],
        failure_modes=[
            FailureMode(
                symptom="Singular design / LinAlgError",
                exception="NumericalInstability",
                remedy="Drop collinear regressors or check dummy-variable coding.",
                alternative="sp.vif",
            ),
            FailureMode(
                symptom="Heteroskedasticity test rejects (sp.het_test)",
                exception="AssumptionWarning",
                remedy="Re-estimate with robust='hc1' (or 'hc3' for n < 250).",
                alternative="",
            ),
            FailureMode(
                symptom="Few clusters (< 30) with cluster-robust SEs",
                exception="AssumptionWarning",
                remedy="Use wild cluster bootstrap (sp.wild_cluster_bootstrap) or CR3 adjustment.",
                alternative="sp.wild_cluster_bootstrap",
            ),
        ],
        alternatives=["iv", "heckman", "qreg", "tobit"],
        typical_n_min=30,
    ))

    register(FunctionSpec(
        name="iv",
        category="regression",
        description="Unified IV estimation: 2SLS, LIML, Fuller, GMM, JIVE. Includes first-stage F, Sargan/Hansen J, and Hausman diagnostics.",
        params=[
            ParamSpec("formula", "str", True, description="IV formula: 'y ~ (endog ~ instruments) + exog'"),
            ParamSpec("data", "DataFrame", True, description="pandas DataFrame"),
            ParamSpec("method", "str", False, "2sls", "Estimation method", ["2sls", "liml", "fuller", "gmm", "jive"]),
            ParamSpec("robust", "str", False, "nonrobust", "Standard error type", ["nonrobust", "hc0", "hc1", "hc2", "hc3"]),
            ParamSpec("cluster", "str", False, description="Column name for cluster-robust SEs"),
            ParamSpec("fuller_alpha", "float", False, 1.0, "Fuller constant (method='fuller' only)"),
        ],
        returns="EconometricResults",
        example='sp.iv("wage ~ (education ~ parent_edu + distance) + experience", data=df, method="liml")',
        tags=["iv", "2sls", "liml", "fuller", "gmm", "jive", "instrumental", "variable", "endogeneity", "weak-instruments"],
        reference="Wooldridge (2010); Stock & Yogo (2005); Fuller (1977); Hansen (1982)",
        pre_conditions=[
            "formula includes the (endog ~ instruments) parenthesised block",
            "at least as many instruments as endogenous regressors (order condition)",
            "instruments are not themselves endogenous in the outcome equation",
        ],
        assumptions=[
            "Relevance: instruments predict the endogenous regressor (first-stage F ≥ 10 rule of thumb)",
            "Exclusion: instruments affect outcome only through the endogenous regressor",
            "Monotonicity (for LATE interpretation under heterogeneous effects)",
        ],
        failure_modes=[
            FailureMode(
                symptom="First-stage F < 10 (Stock-Yogo 5% bias)",
                exception="AssumptionWarning",
                remedy="Use weak-IV-robust inference (Anderson-Rubin) or LIML.",
                alternative="sp.anderson_rubin_ci",
            ),
            FailureMode(
                symptom="Over-identification test rejects (sp.estat 'overid')",
                exception="AssumptionViolation",
                remedy="At least one instrument is invalid; drop instruments or switch to just-identified LIML.",
                alternative="sp.iv",
            ),
            FailureMode(
                symptom="Hausman endogeneity test fails to reject",
                exception="AssumptionWarning",
                remedy="OLS may be consistent and more efficient; report both.",
                alternative="sp.regress",
            ),
            FailureMode(
                symptom="Many instruments (≥ 10) cause many-IV bias",
                exception="NumericalInstability",
                remedy="Use LIML or JIVE which are robust to many weak instruments.",
                alternative="sp.iv",
            ),
        ],
        alternatives=["deepiv", "bartik", "proximal", "regress"],
        typical_n_min=100,
    ))

    register(FunctionSpec(
        name="ivreg",
        category="regression",
        description="Two-stage least squares (2SLS) IV regression. Alias for sp.iv(method='2sls').",
        params=[
            ParamSpec("formula", "str", True, description="IV formula: 'y ~ (endog ~ instruments) + exog'"),
            ParamSpec("data", "DataFrame", True, description="pandas DataFrame"),
            ParamSpec("robust", "str", False, "nonrobust", "Standard error type"),
        ],
        returns="EconometricResults",
        example='sp.ivreg("wage ~ (education ~ parent_edu + distance) + experience", data=df)',
        tags=["iv", "2sls", "instrumental", "variable", "endogeneity"],
    ))

    register(FunctionSpec(
        name="qreg",
        category="regression",
        description="Quantile regression at specified quantile(s).",
        params=[
            ParamSpec("formula", "str", True, description="'y ~ x1 + x2'"),
            ParamSpec("data", "DataFrame", True),
            ParamSpec("q", "float", False, 0.5, "Quantile (0-1)"),
        ],
        returns="EconometricResults",
        example='sp.qreg("wage ~ education", data=df, q=0.9)',
        tags=["quantile", "robust", "distribution"],
    ))

    register(FunctionSpec(
        name="heckman",
        category="regression",
        description="Heckman two-step selection model correcting for sample selection bias.",
        params=[
            ParamSpec("formula", "str", True, description="Outcome equation formula"),
            ParamSpec("select_formula", "str", True, description="Selection equation formula"),
            ParamSpec("data", "DataFrame", True),
        ],
        returns="EconometricResults",
        example='sp.heckman("wage ~ education + experience", select_formula="employed ~ age + kids", data=df)',
        tags=["selection", "heckman", "bias"],
        reference="Heckman (1979)",
    ))

    register(FunctionSpec(
        name="tobit",
        category="regression",
        description="Tobit model for censored dependent variables.",
        params=[
            ParamSpec("formula", "str", True),
            ParamSpec("data", "DataFrame", True),
            ParamSpec("lower", "float", False, 0.0, "Lower censoring point"),
            ParamSpec("upper", "float", False, None, "Upper censoring point"),
        ],
        returns="EconometricResults",
        example='sp.tobit("hours ~ wage + kids", data=df, lower=0)',
        tags=["censored", "tobit", "limited"],
        reference="Tobin (1958)",
    ))

    # -- Causal Inference ---------------------------------------------- #
    register(FunctionSpec(
        name="did",
        category="causal",
        description="Difference-in-Differences. Supports 2x2, DDD, staggered (Callaway-Sant'Anna, Sun-Abraham), and Synthetic DID.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome variable"),
            ParamSpec("treat", "str", True, description="Treatment indicator or first-treatment-period column"),
            ParamSpec("time", "str", True, description="Time period column"),
            ParamSpec("id", "str", False, description="Unit identifier (for staggered DID / SDID)"),
            ParamSpec("method", "str", False, "auto", "Estimator: 'auto', '2x2', 'ddd', 'cs', 'sa', 'sdid'",
                      ["auto", "2x2", "ddd", "callaway_santanna", "cs", "sun_abraham", "sa", "sdid"]),
            ParamSpec("subgroup", "str", False, None, "Affected-subgroup column for DDD"),
        ],
        returns="CausalResult",
        example='sp.did(df, y="wage", treat="treated", time="post")',
        tags=["did", "causal", "treatment", "panel", "staggered", "ddd", "sdid"],
        reference="Roth et al. (2023); Callaway & Sant'Anna (2021); Goodman-Bacon (2021)",
        pre_conditions=[
            "data is panel or repeated cross-section with a time column",
            "treat column is binary (0/1) for 2x2, or first-treatment-period (int) for staggered",
            "at least one pre-treatment period (≥ 2 periods for 2x2; ≥ 3 recommended for event study)",
            "for staggered designs: id column identifying units across time",
        ],
        assumptions=[
            "Parallel trends: treated and control groups would have followed the same trajectory absent treatment",
            "No anticipation: outcomes in pre-treatment periods are unaffected by future treatment",
            "SUTVA: no spillovers between units",
            "For staggered / heterogeneous effects: use CS or SA — TWFE can produce negative weights (Goodman-Bacon)",
        ],
        failure_modes=[
            FailureMode(
                symptom="Pre-trend joint test p < 0.05 (or underpowered at 0.10)",
                exception="AssumptionViolation",
                remedy="Use sp.sensitivity_rr (Rambachan & Roth honest CI) or switch to sp.callaway_santanna.",
                alternative="sp.sensitivity_rr",
            ),
            FailureMode(
                symptom="Staggered treatment timing with TWFE method",
                exception="AssumptionWarning",
                remedy="TWFE can give negative weights; use Callaway-Sant'Anna, Sun-Abraham, or BJS imputation.",
                alternative="sp.callaway_santanna",
            ),
            FailureMode(
                symptom="Pre-trend test underpowered (Roth 2022)",
                exception="AssumptionWarning",
                remedy="Check sp.pretrends_power — if low, report honest CI via sp.sensitivity_rr.",
                alternative="sp.sensitivity_rr",
            ),
            FailureMode(
                symptom="Few clusters at unit level",
                exception="AssumptionWarning",
                remedy="Use wild cluster bootstrap (sp.wild_cluster_bootstrap).",
                alternative="sp.wild_cluster_bootstrap",
            ),
        ],
        alternatives=[
            "callaway_santanna",
            "sun_abraham",
            "did_imputation",
            "sdid",
            "synth",
        ],
        typical_n_min=50,
    ))

    register(FunctionSpec(
        name="ddd",
        category="causal",
        description="Triple Differences (DDD). Extends 2x2 DID with a within-unit subgroup comparison to eliminate additional confounders.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome variable"),
            ParamSpec("treat", "str", True, description="Binary treatment group indicator"),
            ParamSpec("time", "str", True, description="Binary time period indicator"),
            ParamSpec("subgroup", "str", True, description="Binary affected-subgroup indicator (1=affected, 0=unaffected)"),
            ParamSpec("cluster", "str", False, None, "Cluster variable for standard errors"),
        ],
        returns="CausalResult",
        example='sp.ddd(df, y="employment", treat="nj", time="post", subgroup="low_wage")',
        tags=["ddd", "triple", "did", "causal", "subgroup"],
        reference="Gruber (1994); Olden & Møen (2022)",
    ))

    register(FunctionSpec(
        name="did_analysis",
        category="causal",
        description="One-call comprehensive DID workflow: design detection, Bacon decomposition, estimation, event study, and sensitivity analysis.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome variable"),
            ParamSpec("treat", "str", True, description="Treatment indicator or first-treatment-period column"),
            ParamSpec("time", "str", True, description="Time period column"),
            ParamSpec("id", "str", False, description="Unit identifier (for staggered DID)"),
            ParamSpec("method", "str", False, "auto", "Estimator: 'auto', '2x2', 'cs', 'sa', 'sdid'"),
            ParamSpec("run_bacon", "bool", False, True, "Run Bacon decomposition for staggered designs"),
            ParamSpec("run_event_study", "bool", False, True, "Run event study for dynamic effects"),
            ParamSpec("run_sensitivity", "bool", False, True, "Run honest_did sensitivity analysis"),
        ],
        returns="DIDAnalysis",
        example='report = sp.did_analysis(df, y="earnings", treat="first_treat", time="year", id="worker")\nprint(report.summary())',
        tags=["did", "workflow", "analysis", "bacon", "event_study", "sensitivity", "diagnostic"],
        reference="Cunningham (2021, The Mixtape Ch.9)",
    ))

    register(FunctionSpec(
        name="callaway_santanna",
        category="causal",
        description="Callaway-Sant'Anna (2021) staggered DID with group-time ATTs. Robust to heterogeneous treatment effects and staggered adoption.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome variable"),
            ParamSpec("g", "str", True, description="First-treatment-period column (0 = never-treated)"),
            ParamSpec("t", "str", True, description="Time period column"),
            ParamSpec("i", "str", True, description="Unit identifier"),
            ParamSpec("control_group", "str", False, "nevertreated",
                      "Control group", ["nevertreated", "notyettreated"]),
            ParamSpec("anticipation", "int", False, 0, "Number of anticipation periods"),
        ],
        returns="CausalResult",
        example='sp.callaway_santanna(df, y="earnings", g="first_treat", t="year", i="worker")',
        tags=["did", "staggered", "causal", "cs", "group_time"],
        reference="Callaway & Sant'Anna (2021) J. Econometrics",
        pre_conditions=[
            "panel data with unit × time × outcome",
            "g column is integer: first-treated period or 0 for never-treated",
            "at least one never-treated or late-treated control group",
            "≥ 2 pre-treatment periods per cohort",
        ],
        assumptions=[
            "Parallel trends conditional on X (if covariates supplied)",
            "No anticipation (or adjust via anticipation= parameter)",
            "Overlap: positive propensity for each cohort",
            "SUTVA",
        ],
        failure_modes=[
            FailureMode(
                symptom="Pre-trend test on aggregated ATT(g,t) rejects",
                exception="AssumptionViolation",
                remedy="Use sp.sensitivity_rr for honest CI, or add covariates for conditional parallel trends.",
                alternative="sp.sensitivity_rr",
            ),
            FailureMode(
                symptom="Cohort with only one unit — insufficient variation",
                exception="DataInsufficient",
                remedy="Aggregate small cohorts or drop; check sp.diagnose_result.",
                alternative="",
            ),
            FailureMode(
                symptom="All units treated at the same time (no staggering)",
                exception="MethodIncompatibility",
                remedy="Fall back to 2x2 DID via sp.did(method='2x2').",
                alternative="sp.did",
            ),
        ],
        alternatives=[
            "sun_abraham",
            "did_imputation",
            "sdid",
            "did",
        ],
        typical_n_min=50,
    ))

    register(FunctionSpec(
        name="rdrobust",
        category="causal",
        description="RD estimation: sharp, fuzzy, kink, and donut-hole designs with robust inference.",
        params=[
            ParamSpec("y", "str", True, description="Outcome variable"),
            ParamSpec("x", "str", True, description="Running variable"),
            ParamSpec("data", "DataFrame", True),
            ParamSpec("c", "float", False, 0.0, "Cutoff value"),
            ParamSpec("fuzzy", "str", False, None, "Treatment variable for fuzzy RD"),
            ParamSpec("deriv", "int", False, 0, "Derivative order (0=RD, 1=RKD)"),
            ParamSpec("donut", "float", False, 0.0, "Donut-hole radius"),
            ParamSpec("kernel", "str", False, "triangular", "Kernel type", ["triangular", "epanechnikov", "uniform"]),
        ],
        returns="CausalResult",
        example='sp.rdrobust(df, y="score", x="income", c=10000)',
        tags=["rd", "discontinuity", "causal", "bandwidth", "kink", "donut", "fuzzy"],
        reference="Calonico, Cattaneo, Titiunik (2014)",
        pre_conditions=[
            "running variable x is continuous with support on both sides of c",
            "treatment assignment is determined by the cutoff c (sharp) or probabilistically at c (fuzzy)",
            "sufficient mass of observations within the optimal bandwidth",
        ],
        assumptions=[
            "Continuity of potential outcomes in x at c (Hahn, Todd, van der Klaauw 2001)",
            "No manipulation of x at c (McCrary density test)",
            "Local randomization only in a neighborhood of c — extrapolation away from c is not identified",
            "Covariate balance at c (optional but recommended)",
        ],
        failure_modes=[
            FailureMode(
                symptom="McCrary density test p < 0.05",
                exception="AssumptionViolation",
                remedy="Use donut-hole RD (donut=<δ>) or partial-identification bounds.",
                alternative="sp.rdrobust",
            ),
            FailureMode(
                symptom="Covariate imbalance at cutoff (sp.rdbalance rejects)",
                exception="AssumptionViolation",
                remedy="Include covariates as controls, narrow bandwidth, or report as caveat.",
                alternative="",
            ),
            FailureMode(
                symptom="Effect unstable across bandwidth halvings",
                exception="AssumptionWarning",
                remedy="Report sp.rdbwsensitivity and sp.rd_honest (Armstrong-Kolesár honest CI).",
                alternative="sp.rd_honest",
            ),
            FailureMode(
                symptom="Placebo cutoffs show significant 'effects'",
                exception="AssumptionViolation",
                remedy="The RD signal is noise; seek an alternative identification strategy.",
                alternative="sp.bounds",
            ),
        ],
        alternatives=["rd_honest", "rdrbounds", "bounds"],
        typical_n_min=500,
    ))

    register(FunctionSpec(
        name="synth",
        category="causal",
        description=(
            "Unified synthetic control estimator. method= selects variant: "
            "'classic', 'demeaned', 'detrended', 'unconstrained', 'elastic_net', "
            "'augmented', 'sdid', 'gsynth', 'staggered'. "
            "inference= selects: 'placebo', 'conformal', 'bootstrap', 'jackknife'."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("unit", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("treated_unit", "str", False, description="Treated unit (not needed for staggered)"),
            ParamSpec("treatment_time", "int", False, description="First treatment period"),
            ParamSpec("method", "str", False, "classic",
                      "SCM variant: classic/demeaned/detrended/unconstrained/elastic_net/augmented/sdid/gsynth/staggered"),
            ParamSpec("inference", "str", False, None,
                      "Inference method: placebo/conformal/bootstrap/jackknife"),
            ParamSpec("treatment", "str", False, None, "Binary treatment column (staggered only)"),
        ],
        returns="CausalResult",
        example='sp.synth(data=df, outcome="gdp", unit="state", time="year", treated_unit="CA", treatment_time=1989, method="demeaned")',
        tags=["synth", "synthetic", "causal", "comparative", "scm", "factor", "staggered", "conformal"],
        reference="Abadie et al. (2010); Ferman & Pinto (2021); Doudchenko & Imbens (2016); Xu (2017); Ben-Michael et al. (2022); Chernozhukov et al. (2021)",
        pre_conditions=[
            "panel data in long form (unit × time × outcome)",
            "single treated unit (classic) or a treatment-timing column (staggered)",
            "≥ 10 donor (untreated) units with similar pre-treatment trajectories",
            "≥ 10 pre-treatment periods (fewer → large weight on any one year)",
        ],
        assumptions=[
            "Treatment effect on the treated is identified by the counterfactual implicit in the donor weights",
            "No spillover from treated unit to donors (SUTVA)",
            "Donor pool contains units whose outcomes plausibly track the treated counterfactual",
            "Pre-treatment fit (RMSPE) is small relative to post-treatment effect for placebo inference",
        ],
        failure_modes=[
            FailureMode(
                symptom="Pre-treatment RMSPE > post-treatment effect",
                exception="AssumptionWarning",
                remedy="Poor pre-fit — switch to method='demeaned'/'augmented' or enlarge donor pool.",
                alternative="sp.synth",
            ),
            FailureMode(
                symptom="Placebo p-value ≥ 0.1 despite visible gap",
                exception="AssumptionWarning",
                remedy="Use inference='conformal' (valid under weak assumptions) or report ranked placebo statistic.",
                alternative="sp.synth",
            ),
            FailureMode(
                symptom="All weight concentrated on one donor",
                exception="AssumptionWarning",
                remedy="Interpolation bias risk — check method='elastic_net' or augmented SCM.",
                alternative="sp.synth",
            ),
            FailureMode(
                symptom="Treated unit outside donor convex hull",
                exception="IdentificationFailure",
                remedy="Extrapolation needed — use method='unconstrained' or 'augmented'.",
                alternative="sp.synth",
            ),
        ],
        alternatives=["sdid", "did", "matrix_completion", "causal_impact"],
        typical_n_min=10,  # donors (units); time periods enforced separately
    ))

    register(FunctionSpec(
        name="dml",
        category="causal",
        description=(
            "Double/Debiased Machine Learning for treatment effect estimation. "
            "Supports partially linear (PLR), interactive regression (IRM, binary D), "
            "partially linear IV (PLIV), and interactive IV (IIVM, binary D/binary Z → LATE)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome"),
            ParamSpec("treat", "str", True, description="Treatment variable"),
            ParamSpec("covariates", "list", True, description="List of control variable names"),
            ParamSpec("model", "str", False, "plr", "DML model family",
                      ["plr", "irm", "pliv", "iivm"]),
            ParamSpec("instrument", "str", False, description="Instrument (required for pliv/iivm)"),
            ParamSpec("n_folds", "int", False, 5, "Cross-fitting folds"),
            ParamSpec("n_rep", "int", False, 1, "Repeated cross-fitting splits (median aggregation)"),
        ],
        returns="CausalResult",
        example='sp.dml(df, y="wage", treat="training", covariates=["age","edu"], model="plr")',
        tags=["dml", "ml", "causal", "semiparametric", "iivm", "plr", "irm", "pliv"],
        reference="Chernozhukov et al. (2018) Econometrics Journal",
        pre_conditions=[
            "data is tabular (DataFrame); covariates include all confounders conditional on which unconfoundedness holds",
            "cross-fitting folds ≥ 2 (default 5) — more folds → lower variance, higher compute",
            "for irm / iivm: treatment (and for iivm: instrument) is binary 0/1",
            "for pliv / iivm: instrument column supplied",
        ],
        assumptions=[
            "Unconfoundedness: Y(d) ⊥ D | X (conditional ignorability)",
            "Overlap: 0 < P(D=1 | X) < 1 for the estimand support (strong for IRM)",
            "Nuisance-function estimators converge at op(n^{-1/4}) — fast enough that orthogonal moments give √n CATE",
            "For IV variants (PLIV/IIVM): relevance + exclusion + monotonicity",
        ],
        failure_modes=[
            FailureMode(
                symptom="Extreme propensity scores (≈ 0 or 1)",
                exception="statspai.AssumptionViolation",
                remedy="Trim sample to 0.05 < e(x) < 0.95 or use overlap weights (sp.overlap_weights).",
                alternative="sp.overlap_weights",
            ),
            FailureMode(
                symptom="Nuisance models cross-val R² near zero",
                exception="statspai.AssumptionWarning",
                remedy="Nuisances not learnable — DML bias guarantees don't apply; re-featurize or pick a different model family.",
                alternative="",
            ),
            FailureMode(
                symptom="Large Monte-Carlo variance across folds (n_rep > 1)",
                exception="statspai.NumericalInstability",
                remedy="Increase n_rep to 10+ and aggregate by median; check for leakage.",
                alternative="",
            ),
            FailureMode(
                symptom="IIVM first-stage compliance rate near zero",
                exception="statspai.AssumptionWarning",
                remedy="Instrument too weak for LATE; fall back to Anderson-Rubin inference.",
                alternative="sp.anderson_rubin_ci",
            ),
        ],
        alternatives=["metalearner", "causal_forest", "tmle", "aipw"],
        typical_n_min=500,
    ))

    register(FunctionSpec(
        name="causal_forest",
        category="causal",
        description="Causal Forest for heterogeneous treatment effect estimation (CATE).",
        params=[
            ParamSpec("formula", "str", True, description="'y ~ treatment | x1 + x2' (pipe separates covariates)"),
            ParamSpec("data", "DataFrame", True),
            ParamSpec("n_trees", "int", False, 100),
        ],
        returns="CausalResult",
        example='sp.causal_forest("y ~ treat | x1 + x2 + x3", data=df)',
        tags=["forest", "cate", "heterogeneous", "ml"],
        reference="Athey, Tibshirani & Wager (2019) Annals of Statistics",
        pre_conditions=[
            "formula uses pipe separator: 'y ~ treatment | x_1 + x_2 + ...'",
            "treatment is binary 0/1 (use sp.multi_arm_forest for multi-valued)",
            "covariates are numeric; encode categoricals beforehand",
            "n ≥ ~1000 for stable CATE — forests are data-hungry",
        ],
        assumptions=[
            "Unconfoundedness: Y(d) ⊥ D | X",
            "Overlap: 0 < P(D=1 | X) < 1 for the estimand support",
            "Honest splitting: splits and estimates use disjoint samples (enforced by default)",
            "Smoothness: CATE is Lipschitz in X (forests approximate smooth functions)",
        ],
        failure_modes=[
            FailureMode(
                symptom="Calibration test (sp.calibration_test) rejects",
                exception="statspai.AssumptionViolation",
                remedy="CATE predictions are miscalibrated — increase n_trees, add variables, or switch to a DR-Learner.",
                alternative="sp.metalearner",
            ),
            FailureMode(
                symptom="Variance of CATE estimates too large to be useful",
                exception="statspai.DataInsufficient",
                remedy="Need more observations or narrower conditioning set; consider GATE on discrete subgroups.",
                alternative="sp.gate_test",
            ),
            FailureMode(
                symptom="Extreme propensity scores in part of the covariate space",
                exception="statspai.AssumptionViolation",
                remedy="Trim to overlap region via sp.trimming or restrict estimand to overlap support.",
                alternative="sp.trimming",
            ),
        ],
        alternatives=["metalearner", "dml", "multi_arm_forest", "iv_forest"],
        typical_n_min=1000,
    ))

    register(FunctionSpec(
        name="metalearner",
        category="causal",
        description="Meta-learner framework for CATE: S-, T-, X-, R-, DR-Learner.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("method", "str", False, "t", "Learner type", ["s", "t", "x", "r", "dr"]),
        ],
        returns="Meta-learner result with CATE predictions",
        example='sp.metalearner(df, y="outcome", treatment="treat", covariates=["x1","x2"], method="x")',
        tags=["metalearner", "cate", "heterogeneous", "s-learner", "t-learner", "x-learner"],
        reference="Künzel, Sekhon, Bickel & Yu (2019) PNAS; Nie & Wager (2021) Biometrika",
        pre_conditions=[
            "binary treatment (0/1)",
            "covariates numeric; categoricals encoded",
            "enough treated AND control to train separate outcome models (T/X/DR-Learner)",
            "n ≥ 500 for S/T; n ≥ 1000 for X/R/DR (they do 2+ learning steps)",
        ],
        assumptions=[
            "Unconfoundedness: Y(d) ⊥ D | X",
            "Overlap: 0 < P(D=1 | X) < 1",
            "For R-Learner / DR-Learner: orthogonality between treatment residual and outcome residual",
            "Base learner expressivity adequate for the true CATE function",
        ],
        failure_modes=[
            FailureMode(
                symptom="Large divergence across learner types",
                exception="statspai.AssumptionWarning",
                remedy="Use sp.compare_metalearners to identify which learner is biased; DR-Learner is safest under model misspecification.",
                alternative="sp.compare_metalearners",
            ),
            FailureMode(
                symptom="S-Learner estimates near zero regardless of true effect",
                exception="statspai.AssumptionWarning",
                remedy="S-Learner regularization smooths treatment coefficient toward zero; use T/X/DR instead.",
                alternative="sp.metalearner",
            ),
            FailureMode(
                symptom="X-Learner fails when treated group is very small",
                exception="statspai.DataInsufficient",
                remedy="X-Learner needs well-identified control-outcome model; fall back to T-Learner or weighted T-Learner.",
                alternative="sp.metalearner",
            ),
        ],
        alternatives=["causal_forest", "dml", "tmle", "bcf"],
        typical_n_min=500,
    ))

    register(FunctionSpec(
        name="match",
        category="causal",
        description="Propensity score and covariate matching for treatment effect estimation.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("method", "str", False, "nearest", "Matching method", ["nearest", "caliper", "mahalanobis"]),
        ],
        returns="MatchEstimator result",
        example='sp.match(df, treatment="treat", outcome="y", covariates=["x1","x2"])',
        tags=["matching", "propensity", "psm", "treatment"],
        reference="Rosenbaum & Rubin (1983); Ho et al. (2007) Political Analysis; Stuart (2010) Statistical Science",
        pre_conditions=[
            "binary treatment 0/1",
            "covariates are pre-treatment (temporally prior to D)",
            "enough control units for each treated unit under the chosen method (k:1 matching)",
            "covariates numeric; categoricals one-hot or handled by caliper/mahalanobis",
        ],
        assumptions=[
            "Unconfoundedness / CIA: Y(d) ⊥ D | X",
            "Overlap / common support: treated X-values are in the control X-support",
            "SUTVA: no interference between matched units",
            "Covariates are selected before looking at outcomes (no post-treatment conditioning)",
        ],
        failure_modes=[
            FailureMode(
                symptom="Covariate imbalance after matching (max |SMD| > 0.1)",
                exception="statspai.AssumptionViolation",
                remedy="Re-match with stricter caliper, add interactions, or switch to sp.ebalance (entropy balancing).",
                alternative="sp.ebalance",
            ),
            FailureMode(
                symptom="Poor propensity score overlap (density plots, treated mass where controls are sparse)",
                exception="statspai.AssumptionViolation",
                remedy="Apply sp.trimming (Crump 2009) or redefine the estimand to the overlap region.",
                alternative="sp.trimming",
            ),
            FailureMode(
                symptom="Too few matched controls per treated unit",
                exception="statspai.DataInsufficient",
                remedy="Relax caliper, allow with-replacement, or use entropy balancing / overlap weights.",
                alternative="sp.ebalance",
            ),
            FailureMode(
                symptom="Results highly sensitive to match specification",
                exception="statspai.AssumptionWarning",
                remedy="Report sp.rosenbaum_bounds (sensitivity to unobserved confounding) and compare multiple matching methods.",
                alternative="sp.rosenbaum_bounds",
            ),
        ],
        alternatives=["ebalance", "cbps", "optimal_match", "sbw", "ipw"],
        typical_n_min=200,
    ))

    register(FunctionSpec(
        name="tmle",
        category="causal",
        description="Targeted Maximum Likelihood Estimation for ATE/ATT with double-robustness.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("covariates", "list", True),
        ],
        returns="TMLE result",
        example='sp.tmle(df, y="outcome", treatment="treat", covariates=["x1","x2","x3"])',
        tags=["tmle", "doubly-robust", "semiparametric"],
        reference="van der Laan & Rose (2011) Targeted Learning",
        pre_conditions=[
            "binary treatment 0/1",
            "covariates comprise the confounding set",
            "n ≥ 500 for asymptotic efficiency",
        ],
        assumptions=[
            "Unconfoundedness: Y(d) ⊥ D | X",
            "Overlap: 0 < P(D=1 | X) < 1 on the estimand support",
            "Consistent estimation of at least one of Q(a, x) = E[Y|A, X] or g(x) = P(A=1|X) (double robustness)",
            "Super-learner candidates include reasonable approximations",
        ],
        failure_modes=[
            FailureMode(
                symptom="Extreme propensity scores (ATE IF denominator ≈ 0)",
                exception="statspai.NumericalInstability",
                remedy="Bound propensity scores away from 0/1 (e.g. 0.025 / 0.975) or trim.",
                alternative="sp.trimming",
            ),
            FailureMode(
                symptom="Super-learner cross-validated risk not improving over baseline",
                exception="statspai.AssumptionWarning",
                remedy="Nuisances not learnable; widen the candidate library or use stronger base learners.",
                alternative="",
            ),
        ],
        alternatives=["dml", "aipw", "metalearner", "ltmle"],
        typical_n_min=500,
    ))

    # -- Panel / Time Series ------------------------------------------- #
    register(FunctionSpec(
        name="panel",
        category="panel",
        description=(
            "Unified panel regression: FE, RE, between, FD, pooled OLS, "
            "two-way FE, Mundlak/Chamberlain CRE, Arellano-Bond, "
            "Blundell-Bond system GMM. Results include built-in "
            "diagnostics: .hausman_test(), .bp_lm_test(), "
            ".f_test_effects(), .pesaran_cd_test(), .compare(method)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("formula", "str", True, description="Regression formula: 'y ~ x1 + x2'"),
            ParamSpec("entity", "str", True, description="Unit identifier column"),
            ParamSpec("time", "str", True, description="Time column"),
            ParamSpec("method", "str", False, "fe", "Estimation method",
                      ["fe", "re", "be", "fd", "pooled", "twoway",
                       "mundlak", "cre", "chamberlain", "ab", "system"]),
            ParamSpec("robust", "str", False, "nonrobust",
                      "Standard errors: nonrobust, robust, kernel, driscoll-kraay"),
            ParamSpec("cluster", "str", False,
                      description="Cluster variable: entity, time, or twoway"),
            ParamSpec("lags", "int", False, 1, "AR lags for dynamic panel (ab/system)"),
            ParamSpec("gmm_lags", "str", False, "(2, 5)", "GMM instrument lag range"),
            ParamSpec("twostep", "bool", False, False, "Two-step GMM"),
        ],
        returns="PanelResults",
        example='sp.panel(df, "wage ~ edu + exp", entity="worker", time="year", method="fe")',
        tags=["panel", "fe", "re", "fixed-effects", "twoway", "mundlak",
              "cre", "chamberlain", "arellano-bond", "system-gmm", "dynamic"],
        reference="Wooldridge (2010); Mundlak (1978); Arellano & Bond (1991)",
    ))

    register(FunctionSpec(
        name="panel_compare",
        category="panel",
        description=(
            "Estimate the same model with multiple panel methods and "
            "return a side-by-side comparison table."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("formula", "str", True),
            ParamSpec("entity", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("methods", "list", False,
                      description="List of methods to compare, default: pooled/fe/re/twoway/mundlak"),
        ],
        returns="DataFrame",
        example='sp.panel_compare(df, "wage ~ edu + exp", entity="id", time="year")',
        tags=["panel", "comparison", "diagnostics"],
    ))

    register(FunctionSpec(
        name="xtabond",
        category="panel",
        description="Arellano-Bond / Blundell-Bond GMM for dynamic panels (standalone).",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Dependent variable"),
            ParamSpec("x", "list", False, description="Exogenous regressors"),
            ParamSpec("id", "str", False, "id", "Unit identifier"),
            ParamSpec("time", "str", False, "time", "Time column"),
            ParamSpec("lags", "int", False, 1),
            ParamSpec("method", "str", False, "difference", "difference or system",
                      ["difference", "system"]),
            ParamSpec("twostep", "bool", False, False),
        ],
        returns="CausalResult",
        example='sp.xtabond(df, y="output", x=["capital", "labor"], id="firm", time="year")',
        tags=["gmm", "dynamic", "panel", "arellano-bond"],
        reference="Arellano & Bond (1991); Blundell & Bond (1998)",
    ))

    register(FunctionSpec(
        name="causal_impact",
        category="panel",
        description="Bayesian structural time series for causal impact analysis.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("intervention_time", "str", True, description="Date/index of intervention"),
        ],
        returns="CausalImpactEstimator result",
        example='sp.causal_impact(df, outcome="sales", intervention_time="2020-03-15")',
        tags=["timeseries", "bayesian", "impact", "intervention"],
        reference="Brodersen et al. (2015)",
    ))

    # -- Survey -------------------------------------------------------- #
    register(FunctionSpec(
        name="svydesign",
        category="survey",
        description="Declare a complex survey design for design-corrected estimation.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("weights", "str", True, description="Sampling weights column"),
            ParamSpec("strata", "str", False, description="Stratification variable"),
            ParamSpec("cluster", "str", False, description="PSU cluster variable"),
            ParamSpec("fpc", "str", False, description="Finite population correction column"),
        ],
        returns="SurveyDesign",
        example='design = sp.svydesign(df, weights="pw", strata="region", cluster="psu")',
        tags=["survey", "weights", "design", "sampling"],
    ))

    # -- Diagnostics & Output ------------------------------------------ #
    register(FunctionSpec(
        name="outreg2",
        category="output",
        description="Export regression results to publication-quality tables (Excel, LaTeX, Word).",
        params=[
            ParamSpec("results", "list", True, description="One or more EconometricResults objects"),
            ParamSpec("filename", "str", True, description="Output file path (.xlsx, .tex, .docx)"),
        ],
        returns="None (writes file)",
        example='sp.outreg2(result1, result2, filename="table1.xlsx")',
        tags=["output", "table", "publication", "export"],
    ))

    register(FunctionSpec(
        name="modelsummary",
        category="output",
        description="Summary table comparing multiple models side by side.",
        params=[
            ParamSpec("results", "list", True, description="List of EconometricResults"),
        ],
        returns="DataFrame",
        example='sp.modelsummary([r1, r2, r3])',
        tags=["output", "summary", "comparison"],
    ))

    register(FunctionSpec(
        name="sensemakr",
        category="diagnostics",
        description="Sensitivity analysis for omitted variable bias (Cinelli & Hazlett 2020).",
        params=[
            ParamSpec("result", "EconometricResults", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("benchmark_covariates", "list", False, description="Covariates for benchmarking"),
        ],
        returns="Sensitivity analysis result",
        example='sp.sensemakr(result, treatment="education", benchmark_covariates=["experience"])',
        tags=["sensitivity", "omitted-variable", "robustness"],
        reference="Cinelli & Hazlett (2020)",
    ))

    register(FunctionSpec(
        name="spec_curve",
        category="robustness",
        description="Specification curve analysis — run many model specifications and visualise robustness.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("controls", "list", True, description="All potential control variables"),
        ],
        returns="SpecCurveResult",
        example='sp.spec_curve(df, y="outcome", treatment="treat", controls=["x1","x2","x3","x4"])',
        tags=["robustness", "specification", "multiverse"],
        reference="Simonsohn, Simmons & Nelson (2020)",
    ))

    # -- IPW -------------------------------------------------------------- #
    register(FunctionSpec(
        name="ipw",
        category="causal",
        description="Inverse Probability Weighting for ATE/ATT/ATC with propensity score trimming.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("estimand", "str", False, "ATE", "Target estimand", ["ATE", "ATT", "ATC"]),
            ParamSpec("trim", "float", False, 0.0, "Propensity score trimming threshold"),
        ],
        returns="CausalResult",
        example='sp.ipw(df, y="wage", treat="training", covariates=["age","edu"], estimand="ATT")',
        tags=["ipw", "weighting", "propensity", "treatment"],
        reference="Hirano, Imbens & Ridder (2003)",
    ))

    # -- DAG -------------------------------------------------------------- #
    register(FunctionSpec(
        name="dag",
        category="causal",
        description=(
            "Declare a causal DAG and perform identification analysis: "
            "backdoor/frontdoor adjustment sets, d-separation, path enumeration, "
            "bad controls detection, variable role classification, do-operator."
        ),
        params=[
            ParamSpec("spec", "str", True, description='Edge spec: "Z -> X; Z -> Y; X -> Y"'),
        ],
        returns=(
            "DAG object with .adjustment_sets(), .frontdoor_sets(), .backdoor_paths(), "
            ".bad_controls(), .do(), .summary(), .d_separated(), .plot()"
        ),
        example='g = sp.dag("Z -> X; Z -> Y; X -> Y"); print(g.summary("X", "Y"))',
        tags=["dag", "causal", "graph", "adjustment", "backdoor", "frontdoor", "collider", "bad control"],
        reference="Pearl (2009); Cunningham (2021)",
    ))
    register(FunctionSpec(
        name="dag_example",
        category="causal",
        description=(
            "Load a classic textbook DAG: confounding, collider, mediation, "
            "discrimination, movie_star, police, frontdoor, bad_control_earnings, m_bias."
        ),
        params=[
            ParamSpec("name", "str", True, description="Example name, e.g. 'discrimination'"),
        ],
        returns="DAG object with pre-built structure",
        example='g = sp.dag_example("discrimination"); print(g.summary("D", "Y"))',
        tags=["dag", "causal", "example", "textbook", "mixtape"],
        reference="Cunningham (2021) ch.3",
    ))

    # -- Event Study ------------------------------------------------------ #
    register(FunctionSpec(
        name="event_study",
        category="causal",
        description="Traditional OLS event study with lead/lag dummies, TWFE, and pre-trend test.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat_time", "str", True, description="Column with unit's treatment time"),
            ParamSpec("time", "str", True, description="Calendar time column"),
            ParamSpec("unit", "str", True, description="Unit identifier column"),
            ParamSpec("window", "list", False, [-4, 4], "Relative time window [min, max]"),
        ],
        returns="CausalResult with event_study DataFrame and pre-trend test",
        example='sp.event_study(df, y="wage", treat_time="first_treat", time="year", unit="worker")',
        tags=["event-study", "did", "lead-lag", "twfe", "parallel-trends"],
        reference="Freyaldenhoven, Hansen & Shapiro (2019)",
    ))

    # -- Augmented Synthetic Control -------------------------------------- #
    register(FunctionSpec(
        name="augsynth",
        category="causal",
        description="Augmented Synthetic Control with ridge bias correction (Ben-Michael et al. 2021).",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("unit", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("treated_unit", "str", True),
            ParamSpec("treatment_time", "int", True),
        ],
        returns="CausalResult with period-level effects and placebo inference",
        example='sp.augsynth(df, outcome="gdp", unit="state", time="year", treated_unit="CA", treatment_time=1989)',
        tags=["synth", "augmented", "scm", "bias-correction"],
        reference="Ben-Michael, Feller & Rothstein (2021)",
    ))

    # -- Spatial ---------------------------------------------------------- #
    register(FunctionSpec(
        name="sar",
        category="spatial",
        description="Spatial Autoregressive (Lag) Model: Y = ρWY + Xβ + ε via ML.",
        params=[
            ParamSpec("W", "ndarray", True, description="(n,n) spatial weights matrix"),
            ParamSpec("data", "DataFrame", True),
            ParamSpec("formula", "str", True, description="'y ~ x1 + x2'"),
        ],
        returns="EconometricResults with ρ (rho) parameter",
        example='sp.sar(W, data=df, formula="crime ~ income + education")',
        tags=["spatial", "sar", "lag", "ml", "weights"],
        reference="Anselin (1988)",
    ))

    register(FunctionSpec(
        name="sem",
        category="spatial",
        description="Spatial Error Model: Y = Xβ + u, u = λWu + ε via ML.",
        params=[
            ParamSpec("W", "ndarray", True, description="(n,n) spatial weights matrix"),
            ParamSpec("data", "DataFrame", True),
            ParamSpec("formula", "str", True),
        ],
        returns="EconometricResults with λ (lambda) parameter",
        example='sp.sem(W, data=df, formula="crime ~ income + education")',
        tags=["spatial", "sem", "error", "ml"],
        reference="Anselin (1988)",
    ))

    register(FunctionSpec(
        name="sdm",
        category="spatial",
        description="Spatial Durbin Model: Y = ρWY + Xβ + WXθ + ε with direct/indirect effects.",
        params=[
            ParamSpec("W", "ndarray", True, description="(n,n) spatial weights matrix"),
            ParamSpec("data", "DataFrame", True),
            ParamSpec("formula", "str", True),
        ],
        returns="EconometricResults with ρ, β, θ, and effect decomposition",
        example='sp.sdm(W, data=df, formula="crime ~ income + education")',
        tags=["spatial", "sdm", "durbin", "spillover"],
        reference="LeSage & Pace (2009)",
    ))

    # -- Bootstrap -------------------------------------------------------- #
    register(FunctionSpec(
        name="bootstrap",
        category="inference",
        description="General bootstrap inference: nonparametric, cluster, block. Percentile/BCa/normal CIs.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("statistic", "str", True, description="Function f(df) -> float"),
            ParamSpec("n_boot", "int", False, 1000),
            ParamSpec("cluster", "str", False, description="Cluster variable for cluster bootstrap"),
            ParamSpec("ci_method", "str", False, "percentile", "CI method", ["percentile", "bca", "normal"]),
        ],
        returns="BootstrapResult with estimate, se, ci, pvalue",
        example='sp.bootstrap(df, lambda d: d["y"].mean(), n_boot=2000)',
        tags=["bootstrap", "inference", "ci", "resampling"],
        reference="Efron & Tibshirani (1993)",
    ))

    # -- Diagnostics (new) ------------------------------------------------ #
    register(FunctionSpec(
        name="diagnose_result",
        category="diagnostics",
        description="Method-aware diagnostic battery: auto-selects tests by model type (OLS/DID/RDD/IV/SCM).",
        params=[
            ParamSpec("result", "EconometricResults", True, description="Fitted result from any StatsPAI estimator"),
        ],
        returns="Dict with method_type and checks list",
        example='sp.diagnose_result(result)',
        tags=["diagnostics", "robustness", "battery", "auto"],
    ))

    # -- G-methods family ------------------------------------------------- #
    register(FunctionSpec(
        name="g_computation",
        category="causal",
        description=(
            "Parametric g-formula (standardization) estimator. "
            "ATE/ATT for binary D, or dose-response curve for continuous D. "
            "Consistent under correctly-specified outcome model; not doubly robust."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome"),
            ParamSpec("treat", "str", True, description="Treatment variable"),
            ParamSpec("covariates", "list", True, description="Baseline covariates"),
            ParamSpec("estimand", "str", False, "ATE", "Target estimand",
                      ["ATE", "ATT", "dose_response"]),
            ParamSpec("treat_values", "list", False, description="Dose grid (required for dose_response)"),
            ParamSpec("n_boot", "int", False, 500, "Bootstrap replications for SE"),
        ],
        returns="CausalResult",
        example='sp.g_computation(df, y="wage", treat="trained", covariates=["age","edu"])',
        tags=["g-computation", "g-formula", "standardization", "causal", "robins"],
        reference="Robins (1986); Hernán & Robins (2020) ch. 13",
    ))

    register(FunctionSpec(
        name="front_door",
        category="causal",
        description=(
            "Pearl's front-door adjustment: identifies ATE with unmeasured "
            "confounding when a mediator fully transmits the effect of D on Y. "
            "Supports binary or continuous mediator; integrate_by controls "
            "Pearl (marginal) vs Fulcher et al. (conditional) aggregation."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome"),
            ParamSpec("treat", "str", True, description="Binary treatment (0/1)"),
            ParamSpec("mediator", "str", True, description="Fully-transmitting mediator"),
            ParamSpec("covariates", "list", False, description="Pre-treatment covariates"),
            ParamSpec("mediator_type", "str", False, "auto", "Mediator model",
                      ["auto", "binary", "continuous"]),
            ParamSpec("integrate_by", "str", False, "marginal",
                      "MC integration formulation (continuous M only)",
                      ["marginal", "conditional"]),
        ],
        returns="CausalResult",
        example='sp.front_door(df, y="y", treat="d", mediator="m", covariates=["x"])',
        tags=["front-door", "pearl", "causal", "mediator", "unobserved-confounding"],
        reference="Pearl (1995); Fulcher et al. (2020)",
    ))

    register(FunctionSpec(
        name="msm",
        category="causal",
        description=(
            "Marginal Structural Models for time-varying treatments with "
            "time-varying confounders. Uses stabilized IPTW and cluster-robust "
            "inference. Handles binary or continuous treatment; exposure summary "
            "can be current, cumulative, or ever."
        ),
        params=[
            ParamSpec("data", "DataFrame", True, description="Long-format panel (unit × time)"),
            ParamSpec("y", "str", True, description="Outcome"),
            ParamSpec("treat", "str", True, description="Time-varying treatment"),
            ParamSpec("id", "str", True, description="Unit identifier"),
            ParamSpec("time", "str", True, description="Period identifier"),
            ParamSpec("time_varying", "list", True,
                      description="Time-varying confounders (pre-treatment)"),
            ParamSpec("baseline", "list", False, description="Baseline covariates"),
            ParamSpec("exposure", "str", False, "cumulative",
                      "Exposure summary", ["cumulative", "current", "ever"]),
            ParamSpec("family", "str", False, "gaussian",
                      "Outcome family", ["gaussian", "binomial"]),
            ParamSpec("trim", "float", False, 0.01, "Weight truncation quantile"),
        ],
        returns="CausalResult",
        example=('sp.msm(panel, y="Y", treat="A", id="id", time="t", '
                 'time_varying=["L_lag"], baseline=["V"])'),
        tags=["msm", "iptw", "time-varying", "robins", "g-methods", "causal"],
        reference="Robins, Hernán & Brumback (2000); Cole & Hernán (2008)",
    ))

    register(FunctionSpec(
        name="mediate_interventional",
        category="causal",
        description=(
            "Interventional (in)direct effects (VanderWeele, Vansteelandt, "
            "Robins 2014). Identifies mediation effects in the presence of "
            "treatment-induced mediator-outcome confounders where natural "
            "(in)direct effects are not identified."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome"),
            ParamSpec("treat", "str", True, description="Binary treatment"),
            ParamSpec("mediator", "str", True, description="Mediator variable"),
            ParamSpec("covariates", "list", False, description="Baseline covariates"),
            ParamSpec("tv_confounders", "list", False,
                      description="Treatment-induced M-Y confounders"),
        ],
        returns="CausalResult (IIE; IDE and Total in .detail)",
        example=('sp.mediate_interventional(df, y="y", treat="d", mediator="m", '
                 'tv_confounders=["L"])'),
        tags=["mediation", "interventional", "indirect-effect", "causal"],
        reference="VanderWeele, Vansteelandt & Robins (2014)",
    ))

    register(FunctionSpec(
        name="proximal",
        category="causal",
        description=(
            "Proximal Causal Inference via linear 2SLS on the outcome bridge. "
            "Identifies ATE with unmeasured confounding using two proxy "
            "variables: a treatment-side Z (instrument for W) and an "
            "outcome-side W (endogenous bridge regressor)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome"),
            ParamSpec("treat", "str", True, description="Treatment"),
            ParamSpec("proxy_z", "list", True, description="Treatment-side proxies (instruments for W)"),
            ParamSpec("proxy_w", "list", True, description="Outcome-side proxies (endogenous)"),
            ParamSpec("covariates", "list", False, description="Baseline covariates"),
            ParamSpec("bridge", "str", False, "linear", "Bridge function family",
                      ["linear"]),
            ParamSpec("n_boot", "int", False, 0, "Bootstrap SE replications"),
        ],
        returns="CausalResult",
        example='sp.proximal(df, y="y", treat="d", proxy_z=["z"], proxy_w=["w"])',
        tags=["proximal", "unobserved-confounding", "bridge", "causal", "2sls"],
        reference="Tchetgen Tchetgen et al. (2020); Miao, Geng & Tchetgen Tchetgen (2018)",
        pre_conditions=[
            "at least one treatment-side proxy Z (independent of outcome given U, X)",
            "at least one outcome-side proxy W (independent of treatment given U, X)",
            "proxy_z and proxy_w measure the same unmeasured confounder U from different angles",
            "n ≥ 1000 — 2SLS on proxies is noisy",
        ],
        assumptions=[
            "Existence of an outcome bridge function h(w, a, x) that recovers E[Y(a) | U, X]",
            "Z and W are conditionally independent given U and (A, X)",
            "Z ⊥ Y | U, A, X (exclusion on Z)",
            "W ⊥ A | U, X (exclusion on W)",
            "Z is relevant for W given A, X (bridge first stage)",
        ],
        failure_modes=[
            FailureMode(
                symptom="First-stage (Z → W) too weak",
                exception="statspai.AssumptionWarning",
                remedy="Try richer Z or more proxies; without first-stage strength the bridge is underidentified.",
                alternative="sp.iv",
            ),
            FailureMode(
                symptom="Proxies collapse to nearly-constant",
                exception="statspai.DataInsufficient",
                remedy="Proxy variation insufficient — redesign measurement or fall back to sensitivity (sp.sensemakr).",
                alternative="sp.sensemakr",
            ),
            FailureMode(
                symptom="Estimate highly sensitive to bridge specification",
                exception="statspai.AssumptionWarning",
                remedy="Report multiple bridge families; compare with sp.negative_control_outcome / _exposure.",
                alternative="sp.negative_control_outcome",
            ),
        ],
        alternatives=[
            "negative_control_outcome",
            "negative_control_exposure",
            "double_negative_control",
            "iv",
            "sensemakr",
        ],
        typical_n_min=1000,
    ))

    register(FunctionSpec(
        name="principal_strat",
        category="causal",
        description=(
            "Principal Stratification (Frangakis & Rubin 2002). "
            "'monotonicity' method identifies the complier PCE (= LATE) and "
            "reports Zhang-Rubin sharp bounds on the always-survivor SACE. "
            "'principal_score' uses Ding-Lu covariate weighting to "
            "point-identify stratum-specific effects under principal ignorability."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome"),
            ParamSpec("treat", "str", True, description="Binary treatment"),
            ParamSpec("strata", "str", True, description="Binary post-treatment variable"),
            ParamSpec("covariates", "list", False, description="Baseline covariates (required for principal_score)"),
            ParamSpec("method", "str", False, "monotonicity", "Identification strategy",
                      ["monotonicity", "principal_score"]),
            ParamSpec("n_boot", "int", False, 500, "Bootstrap replications"),
        ],
        returns="PrincipalStratResult",
        example='sp.principal_strat(df, y="y", treat="d", strata="s")',
        tags=["principal-stratification", "sace", "late", "compliance", "causal"],
        reference="Frangakis & Rubin (2002); Zhang & Rubin (2003); Ding & Lu (2017)",
        pre_conditions=[
            "binary treatment",
            "binary post-treatment stratum variable (compliance, survival, employment, …)",
            "covariates required when method='principal_score' (for Ding-Lu weighting)",
            "n ≥ 300 per (treat × stratum) cell for stable bounds",
        ],
        assumptions=[
            "Monotonicity (no defiers) for method='monotonicity'",
            "Principal ignorability for method='principal_score' (strata ⊥ Y(d) | X)",
            "SUTVA and exclusion restriction for the never-takers / always-takers interpretation",
            "Overlap in the principal score when method='principal_score'",
        ],
        failure_modes=[
            FailureMode(
                symptom="Zhang-Rubin bounds include 0 and both signs",
                exception="statspai.AssumptionWarning",
                remedy="Strata partition too weak for point identification — add covariates and use method='principal_score'.",
                alternative="sp.principal_strat",
            ),
            FailureMode(
                symptom="Complier share near zero",
                exception="statspai.DataInsufficient",
                remedy="Low compliance — report only bounds; LATE SE explodes.",
                alternative="sp.bounds",
            ),
            FailureMode(
                symptom="Principal score fails overlap",
                exception="statspai.AssumptionViolation",
                remedy="Principal-score inversion is unstable — restrict to overlap region or fall back to method='monotonicity'.",
                alternative="sp.trimming",
            ),
        ],
        alternatives=["survivor_average_causal_effect", "iv", "bounds"],
        typical_n_min=500,
    ))

    register(FunctionSpec(
        name="mediate",
        category="causal",
        description=(
            "Mediation analysis (Imai-Keele-Tingley 2010). Decomposes the "
            "total effect into natural direct effect (NDE) and natural "
            "indirect effect (NIE) via an interventional or sequential-"
            "ignorability identification strategy."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome"),
            ParamSpec("treat", "str", True, description="Binary treatment"),
            ParamSpec("mediator", "str", True, description="Mediator variable"),
            ParamSpec("covariates", "list", False, description="Pre-treatment confounders"),
            ParamSpec("n_sim", "int", False, 1000, "Monte Carlo sims for NDE/NIE"),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="MediationAnalysis with .NDE, .NIE, .total, .proportion_mediated",
        example='sp.mediate(df, y="y", treat="d", mediator="m")',
        tags=["mediation", "NDE", "NIE", "imai-keele-tingley", "causal"],
        reference="Imai, Keele & Tingley (2010) Psych Methods; VanderWeele (2015) Explanation in Causal Inference",
        pre_conditions=[
            "binary treatment 0/1",
            "mediator is a post-treatment variable causally between treat and y",
            "pre-treatment covariates capture confounding for T–Y, M–Y, T–M",
            "n ≥ 500 for stable NDE/NIE bootstrap CIs",
        ],
        assumptions=[
            "Sequential ignorability: (Y(t,m), M(t)) ⊥ T | X; Y(t,m) ⊥ M | T, X",
            "No post-treatment confounder of the mediator-outcome relationship (classical Imai-Keele-Tingley)",
            "SUTVA on both mediator and outcome",
        ],
        failure_modes=[
            FailureMode(
                symptom="NDE + NIE do not sum to total effect (difference vs product decomposition)",
                exception="statspai.AssumptionWarning",
                remedy="Nonlinear / interactive mediator model — use sp.mediate_interventional or four-way decomposition.",
                alternative="sp.mediate_interventional",
            ),
            FailureMode(
                symptom="Sensitivity to unobserved T-M / M-Y confounder unknown",
                exception="statspai.AssumptionWarning",
                remedy="Always report sp.mediate_sensitivity (Imai-Keele-Yamamoto ρ bound).",
                alternative="sp.mediate_sensitivity",
            ),
            FailureMode(
                symptom="Post-treatment confounder L suspected",
                exception="statspai.AssumptionViolation",
                remedy="Use sp.four_way_decomposition (VanderWeele 2014) which handles L.",
                alternative="sp.four_way_decomposition",
            ),
        ],
        alternatives=[
            "mediate_sensitivity",
            "mediate_interventional",
            "four_way_decomposition",
            "proximal",
        ],
        typical_n_min=500,
    ))

    register(FunctionSpec(
        name="bartik",
        category="causal",
        description=(
            "Bartik / shift-share IV estimator (Adão-Kolesár-Morales 2019; "
            "Borusyak-Hull-Jaravel 2022). Uses pre-period industry / group "
            "shares × exogenous shocks as an instrument for local outcome "
            "exposure."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome (e.g. local wage growth)"),
            ParamSpec("shares", "str", True, description="Pre-period share column (e.g. industry share)"),
            ParamSpec("shocks", "str", True, description="Shock column (e.g. industry-level change)"),
            ParamSpec("unit", "str", True, description="Region / unit identifier"),
            ParamSpec("time", "str", False, description="Time column (panel)"),
            ParamSpec("covariates", "list", False),
        ],
        returns="BartikIV result",
        example=(
            'sp.bartik(df, y="wage_growth", shares="industry_share_t0", '
            'shocks="industry_shock", unit="region", time="year")'
        ),
        tags=["bartik", "shift-share", "iv", "causal", "labor", "trade"],
        reference="Adão, Kolesár & Morales (2019) QJE; Borusyak, Hull & Jaravel (2022) ReStud",
        pre_conditions=[
            "pre-period shares are pre-determined (measured strictly before the outcome window)",
            "shocks are as-good-as-random conditional on unit-level controls",
            "≥ 50 regions for AKM shift-share SE to be well-sized",
            "enough industries / groups (n_shares × avg_share_concentration not too concentrated)",
        ],
        assumptions=[
            "Exogeneity of shocks conditional on pre-period exposure structure (Borusyak-Hull-Jaravel)",
            "Shock-level IV: shocks are independent of region-level unobserved trends",
            "Asymptotic framework: many shocks (L → ∞) — check via sp.ssaggregate Herfindahl",
            "First-stage relevance: Bartik predicts local exposure",
        ],
        failure_modes=[
            FailureMode(
                symptom="Herfindahl of shares too concentrated (one industry dominates)",
                exception="statspai.AssumptionWarning",
                remedy="Shift-share SE unreliable — use Adão-Kolesár-Morales shock-level SE via sp.shift_share_se.",
                alternative="sp.shift_share_se",
            ),
            FailureMode(
                symptom="First-stage F < 10",
                exception="statspai.AssumptionWarning",
                remedy="Shares don't predict exposure enough — report weak-IV-robust CI (sp.anderson_rubin_ci).",
                alternative="sp.anderson_rubin_ci",
            ),
            FailureMode(
                symptom="Shocks correlate with pre-trends",
                exception="statspai.AssumptionViolation",
                remedy="Shock exogeneity fails — drop the violating shock dimension or add trend controls.",
                alternative="",
            ),
        ],
        alternatives=["iv", "shift_share_se", "shift_share_political",
                      "shift_share_political_panel"],
        typical_n_min=100,
    ))

    register(FunctionSpec(
        name="bayes_rd",
        category="bayes",
        description=(
            "Bayesian sharp Regression Discontinuity — full posterior over "
            "the RD jump via local polynomial with prior regularisation on "
            "bandwidth and bias-correction slopes. Reports HDI, rhat, ESS, "
            "divergences."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("x", "str", True, description="Running variable"),
            ParamSpec("c", "float", False, 0.0, "Cutoff value"),
            ParamSpec("p", "int", False, 1, "Polynomial order"),
            ParamSpec("kernel", "str", False, "triangular",
                      enum=["triangular", "epanechnikov", "uniform"]),
            ParamSpec("draws", "int", False, 2000),
            ParamSpec("tune", "int", False, 1000),
            ParamSpec("chains", "int", False, 4),
        ],
        returns="CausalResult with .posterior, .rhat, .ess_bulk, .divergences",
        example='sp.bayes_rd(df, y="y", x="running_var", c=0.0)',
        tags=["bayes", "rd", "sharp", "posterior", "bandwidth"],
        reference="Chib & Jacobi (2016); Branson et al. (2019)",
        pre_conditions=[
            "pymc installed",
            "running variable x is continuous with mass on both sides of c",
            "enough observations within the optimal bandwidth (≥ 50 on each side)",
            "draws × chains ≥ 8000 for reliable tail HDI",
        ],
        assumptions=[
            "Continuity at the cutoff (same as frequentist RD)",
            "No manipulation / bunching at c (McCrary / rddensity clean)",
            "Local polynomial + prior-regularised bandwidth captures the CEF",
            "HMC convergence within thresholds",
        ],
        failure_modes=[
            FailureMode(
                symptom="R-hat > 1.01 or divergences > 0",
                exception="statspai.ConvergenceFailure",
                remedy="Increase tune / target_accept; non-centered polynomial coefficients.",
                alternative="sp.rdrobust",
            ),
            FailureMode(
                symptom="Posterior mass outside the plausible effect range",
                exception="statspai.AssumptionWarning",
                remedy="Prior too wide — report sensitivity to prior_sd over {1, 5, 20} × OLS jump SE.",
                alternative="",
            ),
        ],
        alternatives=["rdrobust", "rd_honest", "bayes_fuzzy_rd"],
        typical_n_min=500,
    ))

    register(FunctionSpec(
        name="bayes_fuzzy_rd",
        category="bayes",
        description=(
            "Bayesian fuzzy RD: joint model of first-stage jump in "
            "treatment probability and outcome jump, yielding posterior "
            "over the LATE at the cutoff (Wald ratio)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treatment", "str", True, description="Take-up / treatment received"),
            ParamSpec("x", "str", True, description="Running variable"),
            ParamSpec("c", "float", False, 0.0, "Cutoff"),
            ParamSpec("p", "int", False, 1, "Polynomial order"),
            ParamSpec("draws", "int", False, 2000),
            ParamSpec("tune", "int", False, 1000),
            ParamSpec("chains", "int", False, 4),
        ],
        returns="CausalResult with .posterior, .rhat, .ess_bulk, .divergences",
        example='sp.bayes_fuzzy_rd(df, y="y", treatment="d", x="score", c=0.5)',
        tags=["bayes", "rd", "fuzzy", "late", "wald"],
        reference="Geneletti, O'Keeffe & Baio (2015); Chib & Jacobi (2016)",
        pre_conditions=[
            "pymc installed",
            "running variable continuous on both sides of c",
            "first-stage take-up probability must jump at c (verify with sp.rdrobust on the treatment)",
            "enough draws to resolve Wald-ratio tail mass",
        ],
        assumptions=[
            "Continuity of potential outcomes at c",
            "First-stage relevance (posterior on take-up jump concentrated away from 0)",
            "Exclusion / monotonicity: running variable affects outcome only via treatment at c",
            "HMC convergence",
        ],
        failure_modes=[
            FailureMode(
                symptom="Posterior on first-stage take-up jump straddles zero",
                exception="statspai.AssumptionWarning",
                remedy="Weak fuzzy first stage — report posterior CI width; Wald-ratio divergence symptom.",
                alternative="sp.anderson_rubin_ci",
            ),
            FailureMode(
                symptom="Divergences > 0 near the cutoff",
                exception="statspai.ConvergenceFailure",
                remedy="Reparameterize ratio as log-ratio or raise target_accept to 0.98.",
                alternative="sp.rdrobust",
            ),
        ],
        alternatives=["rdrobust", "bayes_rd", "anderson_rubin_ci"],
        typical_n_min=800,
    ))

    register(FunctionSpec(
        name="bayes_mte",
        category="bayes",
        description=(
            "Bayesian Marginal Treatment Effect (Heckman-Vytlacil 2005). "
            "Full posterior over the MTE curve under essential heterogeneity, "
            "with bivariate-normal latent errors. Derives ATE / ATT / LATE / "
            "PRTE as posterior linear functionals."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treatment", "str", True, description="Binary treatment"),
            ParamSpec("instrument", "str", True, description="Instrument(s)"),
            ParamSpec("covariates", "list", False),
            ParamSpec("draws", "int", False, 2000),
            ParamSpec("tune", "int", False, 1000),
            ParamSpec("chains", "int", False, 4),
            ParamSpec("n_grid", "int", False, 20, "Grid points for MTE curve"),
        ],
        returns="CausalResult with .mte_grid, .posterior, .rhat, .divergences",
        example='sp.bayes_mte(df, y="y", treatment="d", instrument="z")',
        tags=["bayes", "mte", "heckman-vytlacil", "hte", "late"],
        reference="Heckman & Vytlacil (2005, 2007); Brinch, Mogstad & Wiswall (2017)",
        pre_conditions=[
            "pymc installed",
            "binary treatment + at least one continuous instrument",
            "enough variation in the propensity score (≥ 3 instrument values or continuous)",
            "n ≥ 500 for stable MTE posterior across grid points",
        ],
        assumptions=[
            "Binary treatment, latent index model Y = T Y₁ + (1-T) Y₀",
            "Instrument relevance: propensity score varies",
            "Monotonicity / LATE assumption (no defiers)",
            "Joint normality of structural errors (bivariate normal for tractable MTE)",
            "Support of propensity score determines which estimands (ATE/ATT/PRTE) are identified",
        ],
        failure_modes=[
            FailureMode(
                symptom="Propensity-score support thin — ATE endpoints {0,1} not covered",
                exception="statspai.IdentificationFailure",
                remedy="Only report estimands on the supported P-range; ATE not identified.",
                alternative="sp.iv",
            ),
            FailureMode(
                symptom="R-hat > 1.01 or divergences > 0",
                exception="statspai.ConvergenceFailure",
                remedy="Increase tune and target_accept; Cholesky-parameterise the bivariate error covariance.",
                alternative="sp.bayes_iv",
            ),
            FailureMode(
                symptom="Posterior MTE curve wildly oscillates",
                exception="statspai.NumericalInstability",
                remedy="Grid too fine for data support — reduce n_grid or use GP smoothing.",
                alternative="",
            ),
        ],
        alternatives=["iv", "bayes_iv", "deepiv", "metalearner"],
        typical_n_min=500,
    ))

    # -- v0.9.16 breadth-expansion: Target Trial Emulation ----------- #
    register(FunctionSpec(
        name="target_trial_protocol",
        category="target_trial",
        description=(
            "Create a 7-component target trial protocol (Hernan-Robins / "
            "JAMA 2022 framework). Formalizes eligibility, treatment "
            "strategies, time zero, follow-up, outcome, causal contrast, "
            "and analysis plan before any estimation."
        ),
        params=[
            ParamSpec("eligibility", "str | list | callable", True),
            ParamSpec("treatment_strategies", "list", True),
            ParamSpec("assignment", "str", True,
                      description="'randomization' or 'observational emulation'"),
            ParamSpec("time_zero", "str", True),
            ParamSpec("followup_end", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("causal_contrast", "str", False, "ITT",
                      enum=["ITT", "per-protocol", "as-treated", "observational-analogue"]),
            ParamSpec("analysis_plan", "str", False),
            ParamSpec("baseline_covariates", "list", False),
            ParamSpec("time_varying_covariates", "list", False),
        ],
        returns="TargetTrialProtocol",
        example='proto = sp.target_trial_protocol(eligibility="age >= 50", ...)',
        tags=["target_trial", "epidemiology", "observational", "JAMA"],
        reference="Hernan & Robins (2016); JAMA (2022)",
    ))
    register(FunctionSpec(
        name="clone_censor_weight",
        category="target_trial",
        description=(
            "Clone-Censor-Weight (CCW) for sustained-treatment target "
            "trials. Clones each subject per strategy, artificially "
            "censors on deviation, and re-weights via IPCW."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("id_col", "str", True),
            ParamSpec("time_col", "str", True),
            ParamSpec("treatment_col", "str", True),
            ParamSpec("strategies", "dict[str, callable]", True),
            ParamSpec("censor_covariates", "list", False),
            ParamSpec("stabilize", "bool", False, True),
        ],
        returns="CloneCensorWeightResult",
        tags=["target_trial", "ccw", "longitudinal", "dynamic_strategy"],
        reference="Cain et al. 2010; Hernan et al. 2016",
    ))
    register(FunctionSpec(
        name="ipcw",
        category="censoring",
        description=(
            "Inverse Probability of Censoring Weights -- corrects for "
            "informative censoring under conditional independent "
            "censoring given covariates."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("time", "str", True),
            ParamSpec("event", "str", True),
            ParamSpec("censor_covariates", "list", True),
            ParamSpec("treatment_covariates", "list", False),
            ParamSpec("stabilize", "bool", False, True),
            ParamSpec("method", "str", False, "pooled_logistic",
                      enum=["pooled_logistic", "cox_ph"]),
            ParamSpec("truncate", "tuple", False, (0.01, 0.99)),
        ],
        returns="IPCWResult",
        tags=["censoring", "weighting", "survival", "What If"],
        reference="Robins & Finkelstein (2000); Cole & Hernan (2008)",
    ))

    # -- v0.9.16 breadth-expansion: DAG / SCM -------------------------- #
    register(FunctionSpec(
        name="identify",
        category="dag",
        description=(
            "Shpitser-Pearl ID algorithm: decide if P(Y | do(X)) is "
            "non-parametrically identifiable on a semi-Markovian DAG, "
            "return the do-free estimand or a witness hedge."
        ),
        params=[
            ParamSpec("dag", "DAG", True),
            ParamSpec("treatment", "str | set", True),
            ParamSpec("outcome", "str | set", True),
        ],
        returns="IdentificationResult",
        example='sp.identify(sp.dag("Z->X;Z->Y;X->Y"), treatment="X", outcome="Y")',
        tags=["dag", "identification", "scm", "pearl"],
        reference="Shpitser & Pearl (2006); Tian & Pearl (2002)",
    ))
    register(FunctionSpec(
        name="swig",
        category="dag",
        description=(
            "Build a Single-World Intervention Graph (SWIG) by "
            "node-splitting intervened variables. Bridges Pearl's SCM "
            "and Hernan-Robins potential-outcome languages."
        ),
        params=[
            ParamSpec("dag", "DAG", True),
            ParamSpec("intervention", "dict | list", True),
        ],
        returns="SWIGGraph",
        tags=["dag", "swig", "counterfactual"],
        reference="Richardson & Robins (2013)",
    ))

    # -- v0.9.16 breadth-expansion: Causal Discovery (ICP) ----------- #
    register(FunctionSpec(
        name="icp",
        category="causal_discovery",
        description=(
            "Invariant Causal Prediction: infer direct parents of Y by "
            "testing invariance of P(Y | X_S) across environments."
        ),
        params=[
            ParamSpec("X", "DataFrame", True),
            ParamSpec("y", "ndarray", True),
            ParamSpec("environment", "ndarray", True),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("method", "str", False, "linear",
                      enum=["linear", "nonlinear"]),
            ParamSpec("max_subset_size", "int", False),
        ],
        returns="ICPResult",
        tags=["causal_discovery", "invariance", "icp"],
        reference="Peters, Bühlmann & Meinshausen (2016)",
    ))

    # -- v0.9.16 breadth-expansion: Transportability ------------------ #
    register(FunctionSpec(
        name="transport_weights_fn",
        category="transport",
        description=(
            "Density-ratio (inverse odds of sampling) weighting to "
            "transport an effect estimated in the source population to "
            "a named target population."
        ),
        params=[
            ParamSpec("source", "DataFrame", True),
            ParamSpec("target", "DataFrame", True),
            ParamSpec("features", "list", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("truncate", "tuple", False, (0.01, 0.99)),
        ],
        returns="TransportWeightResult",
        tags=["transport", "external_validity", "weighting"],
        reference="Stuart et al. (2011); Dahabreh et al. (2020)",
    ))
    register(FunctionSpec(
        name="identify_transport",
        category="transport",
        description=(
            "Pearl-Bareinboim transportability: enumerate s-admissible "
            "adjustment sets on a selection diagram; returns the "
            "transport formula or NOT identifiable."
        ),
        params=[
            ParamSpec("dag", "DAG", True),
            ParamSpec("treatment", "str | set", True),
            ParamSpec("outcome", "str | set", True),
            ParamSpec("selection_nodes", "set", True),
        ],
        returns="TransportIdentificationResult",
        tags=["transport", "selection_diagram", "bareinboim"],
        reference="Bareinboim & Pearl (2013)",
    ))

    # -- v0.9.16 breadth-expansion: Off-Policy Evaluation ------------- #
    register(FunctionSpec(
        name="OPEResult",
        category="ope",
        description=(
            "Container returned by sp.ope.* estimators (IPS, SNIPS, DR, "
            "Switch-DR, DM). Reports value, SE, CI, importance-ratio "
            "diagnostics."
        ),
        params=[],
        returns="OPEResult",
        tags=["ope", "contextual_bandits", "rl"],
        reference="Dudik, Langford & Li (2011); Swaminathan & Joachims (2015)",
    ))

    # -- v0.9.16 breadth-expansion: CEVAE ---------------------------- #
    register(FunctionSpec(
        name="cevae",
        category="neural_causal",
        description=(
            "Causal Effect Variational Auto-Encoder: infer a latent "
            "confounder Z from noisy proxies X, then estimate ITE via "
            "counterfactual decoding. Uses PyTorch when available, "
            "else a numpy linear-variational fallback."
        ),
        params=[
            ParamSpec("X", "ndarray", True),
            ParamSpec("treatment", "ndarray", True),
            ParamSpec("outcome", "ndarray", True),
            ParamSpec("z_dim", "int", False, 4),
            ParamSpec("hidden", "int", False, 32),
            ParamSpec("lr", "float", False, 1e-2),
            ParamSpec("n_epochs", "int", False, 200),
            ParamSpec("seed", "int", False, 0),
        ],
        returns="CEVAEResult",
        tags=["neural_causal", "vae", "latent_confounder"],
        reference="Louizos et al. (2017)",
    ))

    # -- v0.9.16 breadth-expansion: Parametric g-formula ------------- #
    register(FunctionSpec(
        name="gformula_ice_fn",
        category="g-formula",
        description=(
            "Parametric g-formula via Iterative Conditional Expectation "
            "(ICE) -- sequential regression of the outcome on treatment "
            "and time-varying confounders, with recursive plug-in of "
            "the target strategy. Consistent under correctly-specified "
            "nuisance models; handles time-varying confounding that "
            "vanilla adjustment cannot."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("id_col", "str", True),
            ParamSpec("time_col", "str", True),
            ParamSpec("treatment_cols", "list", True),
            ParamSpec("confounder_cols", "list | list[list]", True),
            ParamSpec("outcome_col", "str", True),
            ParamSpec("treatment_strategy", "list | callable", True),
            ParamSpec("bootstrap", "int", False, 0),
        ],
        returns="ICEResult",
        tags=["g-formula", "longitudinal", "time_varying_confounding",
              "What If", "bang_robins"],
        reference="Robins (1986); Bang & Robins (2005)",
    ))

    # -- v0.9.17 three-school completion: Epidemiology primitives ---- #
    register(FunctionSpec(
        name="odds_ratio",
        category="epi",
        description=(
            "Odds ratio from a 2x2 table with Woolf (asymptotic) or "
            "Fisher-exact CI. Haldane-Anscombe correction for zero cells."
        ),
        params=[
            ParamSpec("a", "float | 2x2 array", True,
                      description="a (exposed, outcome+) count or 2x2 array"),
            ParamSpec("b", "float", False,
                      description="b (exposed, outcome-) count"),
            ParamSpec("c", "float", False,
                      description="c (unexposed, outcome+) count"),
            ParamSpec("d", "float", False,
                      description="d (unexposed, outcome-) count"),
            ParamSpec("method", "str", False, "woolf",
                      description="CI method",
                      enum=["woolf", "exact"]),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="OR2x2Result",
        example="sp.epi.odds_ratio(50, 20, 30, 40)",
        tags=["epidemiology", "odds_ratio", "2x2", "contingency"],
        reference="Woolf (1955); Rothman, Greenland & Lash (2008)",
    ))
    register(FunctionSpec(
        name="relative_risk",
        category="epi",
        description=(
            "Relative risk (risk ratio) from a 2x2 table with Katz "
            "log-RR CI. Haldane correction for zero cells."
        ),
        params=[
            ParamSpec("a", "float | 2x2 array", True),
            ParamSpec("b", "float", False),
            ParamSpec("c", "float", False),
            ParamSpec("d", "float", False),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="RR2x2Result",
        example="sp.epi.relative_risk(50, 950, 10, 990)",
        tags=["epidemiology", "relative_risk", "risk_ratio"],
        reference="Katz (1978); Rothman, Greenland & Lash (2008)",
    ))
    register(FunctionSpec(
        name="risk_difference",
        category="epi",
        description=(
            "Risk difference (absolute risk reduction) with Wald or "
            "Newcombe hybrid-score CI."
        ),
        params=[
            ParamSpec("a", "float | 2x2 array", True),
            ParamSpec("b", "float", False),
            ParamSpec("c", "float", False),
            ParamSpec("d", "float", False),
            ParamSpec("method", "str", False, "wald",
                      enum=["wald", "newcombe"]),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="RD2x2Result",
        tags=["epidemiology", "risk_difference", "absolute_risk"],
        reference="Newcombe (1998)",
    ))
    register(FunctionSpec(
        name="attributable_risk",
        category="epi",
        description=(
            "Attributable fractions in the exposed (AF) and in the "
            "population (Levin PAF) with delta-method CI."
        ),
        params=[
            ParamSpec("a", "float | 2x2 array", True),
            ParamSpec("b", "float", False),
            ParamSpec("c", "float", False),
            ParamSpec("d", "float", False),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="ARResult",
        tags=["epidemiology", "PAF", "attributable_fraction", "Levin"],
        reference="Levin (1953); Greenland (2001)",
    ))
    register(FunctionSpec(
        name="incidence_rate_ratio",
        category="epi",
        description=(
            "Person-time incidence rate ratio with exact Poisson CI "
            "(Clopper-Pearson on conditional binomial)."
        ),
        params=[
            ParamSpec("events_exposed", "float", True),
            ParamSpec("pt_exposed", "float", True,
                      description="Person-time at risk (exposed)"),
            ParamSpec("events_unexposed", "float", True),
            ParamSpec("pt_unexposed", "float", True),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("method", "str", False, "exact",
                      enum=["exact", "wald"]),
        ],
        returns="IRRResult",
        tags=["epidemiology", "incidence_rate", "person_time", "poisson"],
        reference="Breslow & Day (1987)",
    ))
    register(FunctionSpec(
        name="mantel_haenszel",
        category="epi",
        description=(
            "Mantel-Haenszel pooled OR or RR across K strata, with "
            "Robins-Breslow-Greenland variance and Cochran's Q "
            "homogeneity check."
        ),
        params=[
            ParamSpec("tables", "array (K, 2, 2)", True,
                      description="Stack of K per-stratum 2x2 tables"),
            ParamSpec("measure", "str", False, "OR", enum=["OR", "RR"]),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="MantelHaenszelResult",
        tags=["epidemiology", "stratification", "mantel_haenszel",
              "confounding"],
        reference="Mantel & Haenszel (1959); Robins, Breslow & Greenland (1986)",
    ))
    register(FunctionSpec(
        name="breslow_day_test",
        category="epi",
        description=(
            "Breslow-Day test for homogeneity of the odds ratio across "
            "strata, with Tarone correction."
        ),
        params=[
            ParamSpec("tables", "array (K, 2, 2)", True),
            ParamSpec("tarone_correction", "bool", False, True),
        ],
        returns="tuple (chi2, p_value)",
        tags=["epidemiology", "homogeneity", "stratification"],
        reference="Breslow & Day (1980); Tarone (1985)",
    ))
    register(FunctionSpec(
        name="direct_standardize",
        category="epi",
        description=(
            "Direct age/covariate standardization of a rate using "
            "external standard-population weights."
        ),
        params=[
            ParamSpec("events", "list | ndarray", True),
            ParamSpec("population", "list | ndarray", True),
            ParamSpec("standard_weights", "list | ndarray", True),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="StandardizedRateResult",
        tags=["epidemiology", "standardization", "age_adjustment"],
        reference="Rothman, Greenland & Lash (2008) ch. 3",
    ))
    register(FunctionSpec(
        name="indirect_standardize",
        category="epi",
        description=(
            "Indirect standardization -> SMR (standardized morbidity / "
            "mortality ratio) with Garwood exact Poisson CI."
        ),
        params=[
            ParamSpec("observed", "float", True),
            ParamSpec("events_reference", "list | ndarray", True),
            ParamSpec("population_reference", "list | ndarray", True),
            ParamSpec("population_study", "list | ndarray", True),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="SMRResult",
        tags=["epidemiology", "SMR", "standardization"],
        reference="Breslow & Day (1987) Vol. II",
    ))
    register(FunctionSpec(
        name="bradford_hill",
        category="epi",
        description=(
            "Structured 9-viewpoint Bradford-Hill causal-assessment "
            "rubric with prerequisite check (temporality required) and "
            "narrative verdict."
        ),
        params=[
            ParamSpec("evidence", "dict", False,
                      description="Optional dict mapping viewpoint -> [0,1] score"),
            ParamSpec("strength", "float", False),
            ParamSpec("consistency", "float", False),
            ParamSpec("specificity", "float", False),
            ParamSpec("temporality", "float", False),
            ParamSpec("biological_gradient", "float", False),
            ParamSpec("plausibility", "float", False),
            ParamSpec("coherence", "float", False),
            ParamSpec("experiment", "float", False),
            ParamSpec("analogy", "float", False),
            ParamSpec("notes", "dict", False),
        ],
        returns="BradfordHillResult",
        tags=["epidemiology", "causal_assessment", "bradford_hill"],
        reference="Hill (1965)",
    ))

    # -- v0.9.17: Mendelian randomization diagnostics ---------------- #
    register(FunctionSpec(
        name="mr_heterogeneity",
        category="mendelian",
        description=(
            "Cochran's Q (IVW) or Ruecker's Q' (Egger) heterogeneity "
            "statistic with I^2, used to detect horizontal pleiotropy."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("beta_outcome", "ndarray", True),
            ParamSpec("se_outcome", "ndarray", True),
            ParamSpec("method", "str", False, "ivw", enum=["ivw", "egger"]),
        ],
        returns="HeterogeneityResult",
        tags=["mendelian_randomization", "heterogeneity", "pleiotropy"],
        reference="Bowden et al. (2017)",
    ))
    register(FunctionSpec(
        name="mr_pleiotropy_egger",
        category="mendelian",
        description=(
            "Formal MR-Egger intercept test for directional "
            "(unbalanced) horizontal pleiotropy."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("beta_outcome", "ndarray", True),
            ParamSpec("se_outcome", "ndarray", True),
        ],
        returns="PleiotropyResult",
        tags=["mendelian_randomization", "egger", "pleiotropy"],
        reference="Bowden et al. (2015)",
    ))
    register(FunctionSpec(
        name="mr_leave_one_out",
        category="mendelian",
        description=(
            "Drop-one IVW sensitivity — per-SNP table of estimates when "
            "each SNP is removed in turn."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("beta_outcome", "ndarray", True),
            ParamSpec("se_outcome", "ndarray", True),
            ParamSpec("snp_ids", "list", False),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="LeaveOneOutResult",
        tags=["mendelian_randomization", "sensitivity", "leave_one_out"],
    ))
    register(FunctionSpec(
        name="mr_steiger",
        category="mendelian",
        description=(
            "Steiger directionality test — verifies that the SNPs "
            "explain more variance in the exposure than the outcome, "
            "supporting the assumed causal direction."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("se_exposure", "ndarray", True),
            ParamSpec("n_exposure", "int | ndarray", True),
            ParamSpec("beta_outcome", "ndarray", True),
            ParamSpec("se_outcome", "ndarray", True),
            ParamSpec("n_outcome", "int | ndarray", True),
            ParamSpec("eaf", "ndarray", False,
                      description="Effect-allele frequencies"),
        ],
        returns="SteigerResult",
        tags=["mendelian_randomization", "directionality", "steiger"],
        reference="Hemani et al. (2017)",
    ))
    register(FunctionSpec(
        name="mr_presso",
        category="mendelian",
        description=(
            "MR-PRESSO global test + per-SNP outlier detection + "
            "outlier-corrected IVW estimate + distortion test."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("beta_outcome", "ndarray", True),
            ParamSpec("se_exposure", "ndarray", True),
            ParamSpec("se_outcome", "ndarray", True),
            ParamSpec("n_boot", "int", False, 1000),
            ParamSpec("sig_threshold", "float", False, 0.05),
            ParamSpec("seed", "int", False),
        ],
        returns="MRPressoResult",
        tags=["mendelian_randomization", "outlier_detection", "presso"],
        reference="Verbanck et al. (2018)",
    ))
    register(FunctionSpec(
        name="mr_radial",
        category="mendelian",
        description=(
            "Radial IVW MR (Bowden 2018) with per-SNP Bonferroni-"
            "thresholded outlier flagging."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("beta_outcome", "ndarray", True),
            ParamSpec("se_outcome", "ndarray", True),
            ParamSpec("snp_ids", "list", False),
        ],
        returns="RadialResult",
        tags=["mendelian_randomization", "radial", "outlier_detection"],
        reference="Bowden et al. (2018)",
    ))

    # -- v0.9.17: Longitudinal dispatcher ---------------------------- #
    register(FunctionSpec(
        name="longitudinal_analyze",
        category="longitudinal",
        description=(
            "Unified longitudinal causal-effect estimator. Auto-routes "
            "to IPW (no time-varying confounders) / MSM (dynamic regime "
            "with time-varying confounders) / parametric g-formula ICE "
            "(static regime). Accepts a string DSL or callable for the "
            "treatment regime."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("id", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("time_varying", "list", False),
            ParamSpec("baseline", "list", False),
            ParamSpec("regime", "str | Regime | list | callable", False,
                      "always_treat"),
            ParamSpec("method", "str", False, "auto",
                      enum=["auto", "msm", "g-formula", "ipw"]),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("trim", "float", False, 0.01),
        ],
        returns="LongitudinalResult",
        example=(
            "sp.longitudinal_analyze(df, id='pid', time='visit', "
            "treatment='drug', outcome='cd4', "
            "time_varying=['cd4_lag'], "
            "regime='if cd4_lag < 200 then 1 else 0')"
        ),
        tags=["longitudinal", "what_if", "g_methods", "msm", "ipw",
              "dynamic_regime"],
        reference="Hernan & Robins (2020) Causal Inference: What If",
    ))
    register(FunctionSpec(
        name="longitudinal_contrast",
        category="longitudinal",
        description=(
            "Plug-in estimator of E[Y(regime_a)] - E[Y(regime_b)] with "
            "delta-method SE."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("id", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("regime_a", "str | Regime", True),
            ParamSpec("regime_b", "str | Regime", True),
        ],
        returns="dict",
        tags=["longitudinal", "regime_contrast", "g_methods"],
    ))
    register(FunctionSpec(
        name="regime",
        category="longitudinal",
        description=(
            "Build a dynamic or static treatment regime from a string "
            "DSL, list, callable, or scalar. Supports "
            "'if <cond> then <a> else <b>', 'always_treat', "
            "'never_treat', and arbitrary safe expressions. Parsed via "
            "a whitelisted AST walker — no dynamic code execution."
        ),
        params=[
            ParamSpec("rule", "str | list | callable | scalar", True),
            ParamSpec("name", "str", False),
            ParamSpec("K", "int", False, 1),
        ],
        returns="Regime",
        example=(
            'sp.regime("if cd4 < 200 then 1 else 0")'
        ),
        tags=["longitudinal", "regime", "DSL", "what_if"],
    ))

    # -- v0.9.17: Target-trial publication report ------------------- #
    register(FunctionSpec(
        name="target_trial_report",
        category="target_trial",
        description=(
            "Render a target-trial emulation result as a publication-"
            "ready Methods + Results block (Markdown / LaTeX / plain "
            "text), tracking the JAMA 2022 7-component spec."
        ),
        params=[
            ParamSpec("result", "TargetTrialResult", True),
            ParamSpec("fmt", "str", False, "markdown",
                      enum=["markdown", "latex", "text"]),
            ParamSpec("title", "str", False),
        ],
        returns="str",
        tags=["target_trial", "reporting", "publication"],
        reference="Hernan, Wang & Leaf (JAMA 2022)",
    ))

    # -- v0.9.17: DAG -> estimator recommender ----------------------- #
    register(FunctionSpec(
        name="dag_recommend_estimator",
        category="dag",
        description=(
            "Inspect a declared DAG and recommend a StatsPAI estimator "
            "for (exposure, outcome) with a plain-English identification "
            "story. Priority: backdoor adjustment -> IV -> frontdoor -> "
            "not-identifiable. Also available as DAG.recommend_estimator()."
        ),
        params=[
            ParamSpec("dag", "DAG", True),
            ParamSpec("exposure", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("candidate_instruments", "list[str]", False),
        ],
        returns="EstimatorRecommendation",
        example="sp.dag('X -> Y; Z -> X; Z -> Y').recommend_estimator('X', 'Y')",
        tags=["dag", "identification", "estimator_recommendation"],
        reference="Pearl (2009); Greenland, Pearl & Robins (1999)",
    ))

    # -- v0.9.17: Estimand-first DSL -------------------------------- #
    register(FunctionSpec(
        name="causal_question",
        category="workflow",
        description=(
            "Declare a causal question up front (estimand-first). "
            ".identify() picks an estimator and lists identifying "
            "assumptions; .estimate() runs the analysis; .report() "
            "produces a Markdown Methods + Results paragraph. Auto-"
            "routes to IV / RD / DiD / longitudinal / selection-on-"
            "observables based on supplied fields."
        ),
        params=[
            ParamSpec("treatment", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("data", "DataFrame", False),
            ParamSpec("population", "str", False),
            ParamSpec("estimand", "str", False, "ATE",
                      enum=["ATE", "ATT", "ATU", "LATE", "CATE", "ITT"]),
            ParamSpec("design", "str", False, "auto",
                      enum=["auto", "rct", "selection_on_observables",
                            "iv", "natural_experiment", "policy_shock",
                            "regression_discontinuity",
                            "synthetic_control", "did", "event_study",
                            "longitudinal_observational"]),
            ParamSpec("time_structure", "str", False, "cross_section",
                      enum=["cross_section", "panel",
                            "repeated_cross_section", "longitudinal",
                            "time_series", "pre_post"]),
            ParamSpec("time", "str", False),
            ParamSpec("id", "str", False),
            ParamSpec("covariates", "list[str]", False),
            ParamSpec("instruments", "list[str]", False),
            ParamSpec("running_variable", "str", False),
            ParamSpec("cutoff", "float", False),
        ],
        returns="CausalQuestion",
        example=(
            "q = sp.causal_question(treatment='D', outcome='Y', "
            "design='did', time='year', id='unit', data=df); "
            "q.identify(); q.estimate(); q.report()"
        ),
        tags=["workflow", "estimand", "DSL", "target_trial",
              "identification"],
        reference="Hernan (2016); Angrist & Pischke (2008)",
    ))

    # -- v0.9.17: MR deepening (mode + F-stat) ---------------------- #
    register(FunctionSpec(
        name="mr_mode",
        category="mendelian",
        description=(
            "Weighted or simple mode-based MR estimator (Hartwig 2017). "
            "Consistent under the ZEMPA (zero-mode pleiotropy) "
            "assumption — more permissive than the median's 50% rule."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("beta_outcome", "ndarray", True),
            ParamSpec("se_exposure", "ndarray", True),
            ParamSpec("se_outcome", "ndarray", True),
            ParamSpec("method", "str", False, "weighted",
                      enum=["weighted", "simple"]),
            ParamSpec("n_boot", "int", False, 1000),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("seed", "int", False),
        ],
        returns="ModeBasedResult",
        tags=["mendelian_randomization", "mode", "hartwig",
              "zempa", "robust"],
        reference="Hartwig, Davey Smith & Bowden (2017)",
    ))
    register(FunctionSpec(
        name="mr_f_statistic",
        category="mendelian",
        description=(
            "Per-SNP F-statistic summary for instrument strength. "
            "Flags weak-instrument risk when any F < 10 (Staiger-Stock)."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("se_exposure", "ndarray", True),
            ParamSpec("n_samples", "int", False),
        ],
        returns="FStatisticResult",
        tags=["mendelian_randomization", "instrument_strength",
              "f_statistic", "weak_iv"],
        reference="Staiger & Stock (1997)",
    ))

    # -- v0.9.17: Clinical diagnostics ------------------------------ #
    register(FunctionSpec(
        name="sensitivity_specificity",
        category="epi",
        description=(
            "Sensitivity, specificity, PPV, NPV, LR+ / LR- with Wilson "
            "score CIs.  Accepts either raw binary labels or "
            "pre-computed confusion counts."
        ),
        params=[
            ParamSpec("y_true", "array", False),
            ParamSpec("y_pred", "array", False),
            ParamSpec("tp", "int", False),
            ParamSpec("fn", "int", False),
            ParamSpec("fp", "int", False),
            ParamSpec("tn", "int", False),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="DiagnosticTestResult",
        tags=["epidemiology", "clinical", "diagnostic_test",
              "sensitivity", "specificity"],
        reference="Altman & Bland (1994)",
    ))
    register(FunctionSpec(
        name="roc_curve",
        category="epi",
        description=(
            "ROC curve with AUC (trapezoidal) and Hanley-McNeil (1982) "
            "standard error."
        ),
        params=[
            ParamSpec("y_true", "array", True),
            ParamSpec("scores", "array", True),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="ROCResult",
        tags=["epidemiology", "ROC", "AUC", "binary_classification"],
        reference="Hanley & McNeil (1982)",
    ))
    register(FunctionSpec(
        name="cohen_kappa",
        category="epi",
        description=(
            "Cohen's kappa for inter-rater agreement on nominal or "
            "ordinal scales. Supports linear / quadratic weighting."
        ),
        params=[
            ParamSpec("rater_a", "array", True),
            ParamSpec("rater_b", "array", True),
            ParamSpec("weights", "str", False, "unweighted",
                      enum=["unweighted", "linear", "quadratic"]),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="KappaResult",
        tags=["epidemiology", "agreement", "kappa",
              "inter_rater_reliability"],
        reference="Cohen (1960); Landis & Koch (1977)",
    ))

    # -- v0.9.17: Pre-registration ---------------------------------- #
    register(FunctionSpec(
        name="preregister",
        category="workflow",
        description=(
            "Write a pre-analysis plan (CausalQuestion) to YAML / JSON "
            "for OSF, AEA RCT Registry, or a repo-local PAP.  Includes "
            "a metadata block with timestamp and statspai version."
        ),
        params=[
            ParamSpec("question", "CausalQuestion | dict", True),
            ParamSpec("filename", "str | Path", True),
            ParamSpec("fmt", "str", False, "auto",
                      enum=["auto", "yaml", "json"]),
            ParamSpec("registry_url", "str", False),
            ParamSpec("note", "str", False),
        ],
        returns="Path",
        tags=["workflow", "preregistration", "reproducibility",
              "analysis_plan"],
        reference="Nosek et al. (2018) PNAS",
    ))
    register(FunctionSpec(
        name="load_preregister",
        category="workflow",
        description=(
            "Load a pre-registration file back into a CausalQuestion."
        ),
        params=[
            ParamSpec("filename", "str | Path", True),
        ],
        returns="CausalQuestion",
        tags=["workflow", "preregistration", "reproducibility"],
    ))

    # -- v0.9.17: Unified sensitivity dashboard --------------------- #
    register(FunctionSpec(
        name="unified_sensitivity",
        category="robustness",
        description=(
            "Run every applicable sensitivity analysis in one shot: "
            "E-value, Oster delta (when R^2 inputs given), Rosenbaum "
            "Gamma (when matched structure exposed), Sensemakr "
            "(regression models), and a breakdown-frontier bias "
            "estimate. Also available as result.sensitivity()."
        ),
        params=[
            ParamSpec("result", "CausalResult | EconometricResults", True),
            ParamSpec("r2_treated", "float", False),
            ParamSpec("r2_controlled", "float", False),
            ParamSpec("rho_max", "float", False, 1.0),
            ParamSpec("include_oster", "bool", False, True),
            ParamSpec("include_rosenbaum", "bool", False, True),
            ParamSpec("include_sensemakr", "bool", False, True),
        ],
        returns="SensitivityDashboard",
        example="sp.did(df, ...).sensitivity()",
        tags=["sensitivity", "robustness", "evalue", "oster",
              "rosenbaum"],
        reference=(
            "VanderWeele & Ding (2017); Oster (2019); "
            "Rosenbaum (2002); Cinelli & Hazlett (2020)"
        ),
    ))

    # -- Long-term effects via surrogate indices ---------------------- #
    register(FunctionSpec(
        name="surrogate_index",
        category="surrogate",
        description=(
            "Athey-Chetty-Imbens-Kang surrogate-index estimator for the "
            "long-term ATE: combines an experimental sample (treatment + "
            "short-term surrogate) with an observational sample "
            "(surrogate + long-term outcome) to extrapolate the effect on "
            "the long-term outcome."
        ),
        params=[
            ParamSpec("experimental", "DataFrame", True),
            ParamSpec("observational", "DataFrame", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("surrogates", "list", True),
            ParamSpec("long_term_outcome", "str", True),
            ParamSpec("covariates", "list", False),
            ParamSpec("model", "str", False, "ols"),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("n_boot", "int", False, 0,
                      "Bootstrap replicates (0 = analytic delta-method SE)"),
        ],
        returns="CausalResult",
        example=(
            "sp.surrogate_index(exp, obs, treatment='T', "
            "surrogates=['s1','s2'], long_term_outcome='Y')"
        ),
        tags=["surrogate", "long_term", "causal", "ate"],
        reference=(
            "Athey, Chetty, Imbens & Kang (2019). NBER WP 26463."
        ),
    ))

    register(FunctionSpec(
        name="long_term_from_short",
        category="surrogate",
        description=(
            "Long-term ATE under multi-wave short-term surrogates; extends "
            "the classical surrogate index to sustained treatments via "
            "iterated conditional expectations (Ghassami et al. 2024)."
        ),
        params=[
            ParamSpec("experimental", "DataFrame", True),
            ParamSpec("observational", "DataFrame", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("surrogates_waves", "list", True,
                      description="List of wave column lists"),
            ParamSpec("long_term_outcome", "str", True),
            ParamSpec("covariates", "list", False),
            ParamSpec("n_boot", "int", False, 200),
        ],
        returns="CausalResult",
        example=(
            "sp.long_term_from_short(exp, obs, treatment='T', "
            "surrogates_waves=[['s1'],['s2','s3']], long_term_outcome='Y')"
        ),
        tags=["surrogate", "long_term", "multi_wave"],
        reference="Tran, Bibaut & Kallus (arXiv:2311.08527, 2023).",
    ))

    # -- Next-gen evidence synthesis (RCT + RWD + AI/ML) ------------ #
    register(FunctionSpec(
        name="synthesise_evidence",
        category="transport",
        description=(
            "Inverse-variance pooling of an RCT and RWD estimate with "
            "optional transport shift (Dahabreh et al. 2020; arXiv:2511.19735 2025)."
        ),
        params=[
            ParamSpec("rct_estimate", "float", True),
            ParamSpec("rct_se", "float", True),
            ParamSpec("rwd_estimate", "float", True),
            ParamSpec("rwd_se", "float", True),
            ParamSpec("transport_shift", "float", False, 0.0),
            ParamSpec("transport_shift_se", "float", False, 0.0),
            ParamSpec("weight_mode", "str", False, "inverse_variance",
                      enum=["inverse_variance", "rct_heavy"]),
        ],
        returns="EvidenceSynthesisResult",
        tags=["transport", "rwe", "synthesis"],
        reference="arXiv:2511.19735 (2025); Dahabreh et al. 2020.",
    ))
    register(FunctionSpec(
        name="heterogeneity_of_effect",
        category="transport",
        description=(
            "DerSimonian-Laird tau² / Q / I² heterogeneity statistics for "
            "multi-study evidence synthesis."
        ),
        params=[
            ParamSpec("estimates", "list", True),
            ParamSpec("ses", "list", True),
        ],
        returns="HeterogeneityResult",
        tags=["transport", "rwe", "heterogeneity"],
    ))
    register(FunctionSpec(
        name="rwd_rct_concordance",
        category="transport",
        description=(
            "Report-card: does the RWD estimate fall inside the RCT's 95% CI?"
        ),
        params=[
            ParamSpec("rct_estimate", "float", True),
            ParamSpec("rct_se", "float", True),
            ParamSpec("rwd_estimate", "float", True),
        ],
        returns="ConcordanceResult",
        tags=["transport", "rwe", "concordance"],
    ))

    # -- LLM causal-reasoning evaluator ----------------------------- #
    register(FunctionSpec(
        name="llm_causal_assess",
        category="dag",
        description=(
            "Level-1 (knowledge) and Level-2 (deductive reasoning) "
            "evaluation of an LLM's causal-reasoning ability."
        ),
        params=[
            ParamSpec("level1_items", "DataFrame", False),
            ParamSpec("level2_items", "DataFrame", False),
            ParamSpec("llm_client", "callable", True),
            ParamSpec("llm_identifier", "str", False, "llm"),
        ],
        returns="LLMCausalAssessResult",
        tags=["llm", "causal", "benchmark"],
        reference=(
            "arXiv:2403.09606; 2409.09822; 2503.09326; 2509.00987."
        ),
    ))
    register(FunctionSpec(
        name="pairwise_causal_benchmark",
        category="dag",
        description=(
            "Pairwise causal-direction discovery benchmark for an LLM."
        ),
        params=[
            ParamSpec("ground_truth", "DataFrame", True),
            ParamSpec("llm_client", "callable", True),
        ],
        returns="PairwiseBenchmarkResult",
        tags=["llm", "causal_discovery", "benchmark", "pairwise"],
        reference="Kıcıman et al. 2023; arXiv:2509.00987.",
    ))

    # -- Causal RL primitives ---------------------------------------- #
    register(FunctionSpec(
        name="causal_bandit",
        category="causal_rl",
        description=(
            "Bareinboim-Forney-Pearl contextual causal bandit: pick the optimal "
            "arm by Monte-Carlo estimation of E[Y(a) | context]."
        ),
        params=[
            ParamSpec("arms", "list", True),
            ParamSpec("reward_fn", "callable", True),
            ParamSpec("context", "dict", False),
            ParamSpec("n_samples", "int", False, 500),
        ],
        returns="CausalBanditResult",
        tags=["causal_rl", "bandit", "pearl"],
        reference="Bareinboim, Forney & Pearl (NeurIPS 2015). 'Bandits with Unobserved Confounders: A Causal Approach.'",
    ))
    register(FunctionSpec(
        name="counterfactual_policy_optimization",
        category="causal_rl",
        description=(
            "Counterfactual policy evaluation under a linear-Gaussian SCM "
            "via noise inversion (Oberst-Sontag 2019, Buesing et al. 2019)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("state", "str", True),
            ParamSpec("action", "str", True),
            ParamSpec("reward", "str", True),
            ParamSpec("target_policy", "callable", True),
        ],
        returns="CFPolicyResult",
        tags=["causal_rl", "counterfactual", "scm"],
        reference="Oberst & Sontag (ICML 2019); Buesing et al. 2019.",
    ))
    register(FunctionSpec(
        name="structural_mdp",
        category="causal_rl",
        description=(
            "Fit a linear SVAR for a Markov decision process and roll out "
            "counterfactual trajectories under alternative policies."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("state_cols", "list", True),
            ParamSpec("action_cols", "list", True),
            ParamSpec("reward", "str", True),
            ParamSpec("next_state_cols", "list", False),
            ParamSpec("time", "str", False),
            ParamSpec("trajectory", "str", False),
        ],
        returns="StructuralMDPResult",
        tags=["causal_rl", "mdp", "svar", "counterfactual"],
        reference="arXiv:2512.18135 (2025).",
    ))

    # -- Overlap-weighted DID + DL propensity ------------------------ #
    register(FunctionSpec(
        name="overlap_weighted_did",
        category="causal",
        description=(
            "2x2 DID with overlap weights w=e(X)(1-e(X)), focusing the "
            "ATT on the subpopulation where treatment assignment is most "
            "ambiguous (Econ Letters 2025)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("covariates", "list", False),
            ParamSpec("ps_model", "str", False, "logit",
                      enum=["logit", "gbm", "dl"]),
        ],
        returns="CausalResult",
        tags=["did", "overlap", "propensity", "causal"],
        reference="Li, Morgan, Zaslavsky (JASA 2018); Econ Letters 2025.",
    ))
    register(FunctionSpec(
        name="dl_propensity_score",
        category="matching",
        description=(
            "Neural-net propensity score estimator (arXiv:2404.04794, 2024)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("hidden_sizes", "list", False),
        ],
        returns="ndarray",
        tags=["propensity", "neural_net", "matching"],
        reference="arXiv:2404.04794 (2024).",
    ))

    # -- Continuous + interference conformal ------------------------ #
    register(FunctionSpec(
        name="conformal_continuous",
        category="conformal_causal",
        description=(
            "Split-conformal prediction bands for continuous-treatment "
            "dose-response curves (Schröder, Frauen, Schweisthal, Heß, Melnychuk, Feuerriegel 2024, arXiv:2407.03094)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("test_data", "DataFrame", True),
            ParamSpec("alpha", "float", False, 0.1),
        ],
        returns="ContinuousConformalResult",
        tags=["conformal", "continuous_treatment", "dose_response"],
        reference="arXiv:2407.03094 (2024).",
    ))
    register(FunctionSpec(
        name="conformal_interference",
        category="conformal_causal",
        description=(
            "Cluster-exchangeable split-conformal prediction under "
            "network interference (2509.21660 systematic review)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("cluster", "str", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("test_clusters", "list", True),
            ParamSpec("alpha", "float", False, 0.1),
        ],
        returns="InterferenceConformalResult",
        tags=["conformal", "interference", "cluster"],
        reference="arXiv:2509.21660 (2025).",
    ))

    # -- Sharp OPE + Causal-Policy Forest ---------------------------- #
    register(FunctionSpec(
        name="sharp_ope_unobserved",
        category="ope",
        description=(
            "Sharp bounds on off-policy value under unobserved confounding "
            "via the marginal-sensitivity Gamma-model (Kallus, Mao, Uehara 2025)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("actions", "str", True),
            ParamSpec("rewards", "str", True),
            ParamSpec("logging_prob", "str", True),
            ParamSpec("target_prob", "str", True),
            ParamSpec("gamma", "float", False, 1.5),
        ],
        returns="SharpOPEResult",
        tags=["ope", "sensitivity", "sharp", "bandit"],
        reference="Hess, Frauen, Melnychuk & Feuerriegel (arXiv:2502.13022, 2025).",
    ))
    register(FunctionSpec(
        name="causal_policy_forest",
        category="ope",
        description=(
            "Forest of doubly-robust policy trees: ensembles depth-limited "
            "trees over AIPW-scored actions to reduce variance and give "
            "honest policy-value SE (2025)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("actions", "str", True),
            ParamSpec("rewards", "str", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("n_trees", "int", False, 20),
            ParamSpec("depth", "int", False, 3),
        ],
        returns="CausalPolicyForestResult",
        tags=["ope", "policy_learning", "forest", "aipw"],
        reference="arXiv:2512.22846 (2025).",
    ))

    # -- Orthogonal network HTE + inward/outward spillover ----------- #
    register(FunctionSpec(
        name="network_hte",
        category="interference",
        description=(
            "Orthogonal learning of direct + spillover effects under "
            "network interference via cross-fitted double-residualisation "
            "(Wu & Yuan 2025, arXiv:2509.18484)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("neighbor_exposure", "str", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("n_folds", "int", False, 5),
        ],
        returns="NetworkHTEResult",
        tags=["interference", "network", "hte", "orthogonal"],
        reference="Wu & Yuan (arXiv:2509.18484, 2025).",
    ))
    register(FunctionSpec(
        name="inward_outward_spillover",
        category="interference",
        description=(
            "Decompose network spillover into inward (incoming edges to "
            "unit i) and outward (from i to neighbours) components."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("inward_exposure", "str", True),
            ParamSpec("outward_exposure", "str", True),
        ],
        returns="InwardOutwardResult",
        tags=["interference", "spillover", "directional"],
        reference="Fang, Airoldi & Forastiere (arXiv:2506.06615, 2025).",
    ))

    # -- Bayesian Double Machine Learning ---------------------------- #
    register(FunctionSpec(
        name="bayes_dml",
        category="bayes",
        description=(
            "Bayesian Double Machine Learning (DiTraglia & Liu 2025): "
            "Normal-Normal conjugate update on a DML point estimate, with "
            "optional full PyMC MCMC over the orthogonal moment equation."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("model", "str", False, "plr",
                      enum=["plr", "irm", "pliv"]),
            ParamSpec("prior_mean", "float", False, 0.0),
            ParamSpec("prior_sd", "float", False, 10.0),
            ParamSpec("mode", "str", False, "conjugate",
                      enum=["conjugate", "full"]),
        ],
        returns="BayesianDMLResult",
        example=(
            "sp.bayes_dml(df, y='y', treatment='d', "
            "covariates=['x1','x2'])"
        ),
        tags=["bayes", "dml", "double_ml", "posterior"],
        reference="DiTraglia & Liu (arXiv:2508.12688, 2025). DML framework: Chernozhukov et al. (2018).",
        pre_conditions=[
            "prior_sd is weakly informative relative to the expected effect scale",
            "for mode='full': pymc installed (sp.bayes extra)",
            "treatment is numeric (binary for irm, continuous for plr)",
        ],
        assumptions=[
            "Standard DML unconfoundedness + overlap (see sp.dml)",
            "Normal-Normal prior/likelihood update valid on the DML asymptotic linearization (mode='conjugate')",
            "Weak prior dominance: posterior concentrates around DML point when prior_sd is large",
        ],
        failure_modes=[
            FailureMode(
                symptom="Strong prior shifts posterior noticeably from DML point",
                exception="statspai.AssumptionWarning",
                remedy="Report sensitivity to prior_sd over [1, 10, 100] × DML SE; document prior choice.",
                alternative="sp.dml",
            ),
            FailureMode(
                symptom="Full-mode MCMC R-hat > 1.01 or ESS < 400",
                exception="statspai.ConvergenceFailure",
                remedy="Increase tune / draws; reparameterise to non-centered; check divergences.",
                alternative="sp.bayes_dml",
            ),
        ],
        alternatives=["dml", "bayes_did", "bayes_mte"],
        typical_n_min=500,
    ))

    register(FunctionSpec(
        name="bayes_did",
        category="bayes",
        description=(
            "Bayesian Difference-in-Differences with staggered adoption — "
            "hierarchical ATT(g,t) posterior with optional cohort / unit "
            "random effects. Full MCMC with PyMC; reports R-hat, ESS, "
            "divergences, and 94% HDI."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome"),
            ParamSpec("g", "str", True, description="First-treatment-period column (0 = never-treated)"),
            ParamSpec("t", "str", True, description="Time period column"),
            ParamSpec("i", "str", True, description="Unit identifier"),
            ParamSpec("draws", "int", False, 2000, "Post-warmup draws per chain"),
            ParamSpec("tune", "int", False, 1000, "Warmup draws per chain"),
            ParamSpec("chains", "int", False, 4, "Number of parallel chains"),
            ParamSpec("target_accept", "float", False, 0.9, "HMC target acceptance rate"),
        ],
        returns="CausalResult with .posterior, .rhat, .ess_bulk, .ess_tail, .divergences",
        example=(
            "sp.bayes_did(df, y='wage', g='first_treat', t='year', i='worker_id')"
        ),
        tags=["bayes", "did", "staggered", "hierarchical", "posterior"],
        reference="Callaway & Sant'Anna (2021); Gelman & Hill (2006) hierarchical models",
        pre_conditions=[
            "pymc installed (pip install 'statspai[bayes]')",
            "staggered-panel shape: unit × time × outcome with g-column",
            "≥ 2 pre-treatment periods per cohort",
            "enough draws for posterior summaries (≥ 2000 post-warmup)",
        ],
        assumptions=[
            "Parallel trends (conditional on covariates if supplied)",
            "No anticipation (or modelled via explicit anticipation parameter)",
            "Hierarchical prior regularises small cohorts toward the grand mean",
            "HMC / NUTS reaches stationary distribution (R-hat ≤ 1.01, ESS ≥ 400)",
        ],
        failure_modes=[
            FailureMode(
                symptom="Max R-hat > 1.01",
                exception="statspai.ConvergenceFailure",
                remedy="Raise tune ≥ 4000 and target_accept ≥ 0.95; check priors for weak identification.",
                alternative="sp.callaway_santanna",
            ),
            FailureMode(
                symptom="Min bulk ESS < 400",
                exception="statspai.ConvergenceWarning",
                remedy="Increase draws or chains; consider reparameterization.",
                alternative="",
            ),
            FailureMode(
                symptom="Post-warmup divergences > 0",
                exception="statspai.ConvergenceFailure",
                remedy="Raise target_accept to 0.95–0.99; switch to non-centered random effects.",
                alternative="",
            ),
            FailureMode(
                symptom="Posterior concentrates at a single cohort",
                exception="statspai.DataInsufficient",
                remedy="Cohort sizes too uneven — aggregate small cohorts or use partial pooling strength.",
                alternative="sp.callaway_santanna",
            ),
        ],
        alternatives=["callaway_santanna", "did", "bayes_dml", "bayes_mte"],
        typical_n_min=200,
    ))

    register(FunctionSpec(
        name="bayes_iv",
        category="bayes",
        description=(
            "Bayesian instrumental variables with full posterior over "
            "structural parameters. Handles weak instruments via shrinkage "
            "priors and reports posterior mass near zero on the first-stage "
            "coefficient."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome"),
            ParamSpec("treatment", "str", True, description="Endogenous treatment"),
            ParamSpec("instrument", "str", True, description="Instrument(s) — str or list"),
            ParamSpec("covariates", "list", False, description="Exogenous controls"),
            ParamSpec("draws", "int", False, 2000),
            ParamSpec("tune", "int", False, 1000),
            ParamSpec("chains", "int", False, 4),
        ],
        returns="CausalResult with .posterior, .rhat, .ess_bulk, .divergences",
        example=(
            "sp.bayes_iv(df, y='wage', treatment='education', "
            "instrument='quarter_of_birth')"
        ),
        tags=["bayes", "iv", "2sls", "weak-iv", "posterior"],
        reference="Kleibergen & Zivot (2003); Chen et al. (2018) weak-IV Bayesian",
        pre_conditions=[
            "pymc installed",
            "instrument column(s) exist; exclusion restriction is defensible a priori",
            "draws × chains ≥ 8000 for reliable tail quantiles",
        ],
        assumptions=[
            "Relevance (posterior on first-stage coef concentrated away from 0)",
            "Exclusion: instrument → outcome only through treatment",
            "Monotonicity (for LATE interpretation)",
            "HMC convergence diagnostics within thresholds",
        ],
        failure_modes=[
            FailureMode(
                symptom="Posterior on first-stage coef straddles zero",
                exception="statspai.AssumptionWarning",
                remedy="Weak instrument — report posterior credible interval width and caveat LATE interpretation.",
                alternative="sp.anderson_rubin_ci",
            ),
            FailureMode(
                symptom="Divergences > 0",
                exception="statspai.ConvergenceFailure",
                remedy="Raise target_accept; use Cholesky-parameterized bivariate error.",
                alternative="",
            ),
            FailureMode(
                symptom="R-hat > 1.01",
                exception="statspai.ConvergenceFailure",
                remedy="Longer tune; non-centered structural error parameterization.",
                alternative="",
            ),
        ],
        alternatives=["iv", "anderson_rubin_ci", "deepiv", "bayes_mte"],
        typical_n_min=300,
    ))

    # -- Multivariable / mediation / BMA MR -------------------------- #
    register(FunctionSpec(
        name="mr_multivariable",
        category="mendelian",
        description=(
            "Multivariable Mendelian randomization (Sanderson-Windmeijer "
            "2019): direct causal effects of multiple correlated exposures "
            "via weighted least-squares on SNP-summary data, with "
            "conditional F-statistics for instrument strength."
        ),
        params=[
            ParamSpec("snp_associations", "DataFrame", True),
            ParamSpec("outcome", "str", False, "beta_y"),
            ParamSpec("outcome_se", "str", False, "se_y"),
            ParamSpec("exposures", "list", False),
        ],
        returns="MVMRResult",
        example=(
            "sp.mr_multivariable(df, outcome='beta_y', se_outcome='se_y', "
            "exposures=['beta_ldl','beta_hdl'])"
        ),
        tags=["mr", "mvmr", "multivariable", "mendelian"],
        reference="Sanderson et al. (IJE 2019); Yao et al. (arXiv:2509.11519).",
    ))

    register(FunctionSpec(
        name="mr_mediation",
        category="mendelian",
        description=(
            "Two-step (network) MR: decompose the total causal effect of "
            "an exposure on an outcome into direct + indirect (mediated) "
            "components."
        ),
        params=[
            ParamSpec("snp_associations", "DataFrame", True),
            ParamSpec("beta_exposure", "str", False, "beta_x"),
            ParamSpec("beta_mediator", "str", False, "beta_m"),
            ParamSpec("beta_outcome", "str", False, "beta_y"),
        ],
        returns="MediationMRResult",
        tags=["mr", "mediation", "two_step"],
        reference="Burgess, Daniel, Butterworth, Thompson (IJE 2015).",
    ))

    register(FunctionSpec(
        name="mr_bma",
        category="mendelian",
        description=(
            "MR Bayesian model averaging over exposure subsets (Zuber et "
            "al. 2020). Outputs marginal inclusion probabilities and top "
            "posterior models."
        ),
        params=[
            ParamSpec("snp_associations", "DataFrame", True),
            ParamSpec("outcome", "str", False, "beta_y"),
            ParamSpec("outcome_se", "str", False, "se_y"),
            ParamSpec("exposures", "list", False),
            ParamSpec("max_model_size", "int", False, None),
        ],
        returns="MRBMAResult",
        tags=["mr", "bma", "bayesian", "model_averaging"],
        reference="Zuber, Colijn, Staley, Burgess (Nat Comm 2020).",
    ))

    # -- v1.6 MR Frontier: MR-Lap / MR-Clust / GRAPPLE / MR-cML ------ #
    register(FunctionSpec(
        name="mr_lap",
        category="mendelian",
        description=(
            "Sample-overlap-corrected IVW MR (Burgess-Davies-Thompson "
            "2016 closed-form correction). Removes first-order bias "
            "when exposure and outcome GWAS share participants; "
            "requires overlap_fraction and overlap_rho (e.g. from "
            "LD-score regression)."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("beta_outcome", "ndarray", True),
            ParamSpec("se_exposure", "ndarray", True),
            ParamSpec("se_outcome", "ndarray", True),
            ParamSpec("overlap_fraction", "float", False, 1.0),
            ParamSpec("overlap_rho", "float", False, 0.0),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="MRLapResult",
        tags=["mendelian_randomization", "mr_lap", "sample_overlap",
              "bias_correction"],
        reference=(
            "Burgess, Davies & Thompson (2016) Genet Epidemiol 40(7); "
            "Mounier & Kutalik (2023) Genet Epidemiol 47(4)."
        ),
    ))
    register(FunctionSpec(
        name="mr_clust",
        category="mendelian",
        description=(
            "Clustered Mendelian randomization via finite Gaussian "
            "mixture on Wald ratios (Foley et al. 2021). EM with "
            "SNP-specific measurement SE; optional 'null' cluster at "
            "theta=0; K selected by BIC. Returns per-cluster estimate, "
            "SNP-to-cluster responsibilities, and the K-path."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("beta_outcome", "ndarray", True),
            ParamSpec("se_exposure", "ndarray", True),
            ParamSpec("se_outcome", "ndarray", True),
            ParamSpec("K_range", "tuple", False, (1, 5)),
            ParamSpec("include_null", "bool", False, True),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("seed", "int", False, 0),
        ],
        returns="MRClustResult",
        tags=["mendelian_randomization", "mr_clust", "clustered_pleiotropy",
              "mixture_model"],
        reference=(
            "Foley, Mason, Kirk & Burgess (2021) Bioinformatics 37(4)."
        ),
    ))
    register(FunctionSpec(
        name="grapple",
        category="mendelian",
        description=(
            "GRAPPLE: profile-likelihood MR with joint weak-instrument "
            "and balanced-pleiotropy robustness (Wang et al. 2021). "
            "Model: beta_y = beta*beta_x + u, Var(u) = se_y^2 + "
            "beta^2*se_x^2 + tau^2; jointly MLE over (beta, tau^2) via "
            "L-BFGS-B; SE from observed Fisher info."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("beta_outcome", "ndarray", True),
            ParamSpec("se_exposure", "ndarray", True),
            ParamSpec("se_outcome", "ndarray", True),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("beta_init", "float", False),
            ParamSpec("tau2_init", "float", False, 1e-4),
        ],
        returns="GrappleResult",
        tags=["mendelian_randomization", "grapple", "profile_likelihood",
              "weak_instruments", "pleiotropy"],
        reference="Wang, Zhao, Bowden, Hemani et al. (2021) PLoS Genet 17(6).",
    ))
    register(FunctionSpec(
        name="mr_cml",
        category="mendelian",
        description=(
            "MR-cML-BIC: constrained maximum-likelihood MR with "
            "L0-sparse pleiotropy (Xue, Shen & Pan 2021). Block-"
            "coordinate descent jointly updates causal beta, true "
            "exposure effects, and a K-sparse pleiotropy vector; K "
            "selected by BIC. Robust to correlated + uncorrelated "
            "pleiotropy simultaneously."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("beta_outcome", "ndarray", True),
            ParamSpec("se_exposure", "ndarray", True),
            ParamSpec("se_outcome", "ndarray", True),
            ParamSpec("K_max", "int", False),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="MRcMLResult",
        tags=["mendelian_randomization", "mr_cml", "constrained_ml",
              "sparse_pleiotropy", "bic"],
        reference="Xue, Shen & Pan (2021) AJHG 108(7).",
    ))
    register(FunctionSpec(
        name="mr_raps",
        category="mendelian",
        description=(
            "MR-RAPS: Robust Adjusted Profile Score for two-sample "
            "summary-data MR (Zhao et al. 2020, Annals of Statistics). "
            "Profile-likelihood MR with Tukey biweight loss + weak-"
            "instrument correction; resistant to a small fraction of "
            "gross pleiotropy outliers. Complements GRAPPLE (Gaussian) "
            "with a robust-loss variant of the same structural model."
        ),
        params=[
            ParamSpec("beta_exposure", "ndarray", True),
            ParamSpec("beta_outcome", "ndarray", True),
            ParamSpec("se_exposure", "ndarray", True),
            ParamSpec("se_outcome", "ndarray", True),
            ParamSpec("tuning_c", "float", False, 4.685),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("beta_init", "float", False),
            ParamSpec("tau2_init", "float", False, 1e-4),
        ],
        returns="MRRapsResult",
        tags=["mendelian_randomization", "mr_raps",
              "robust_profile_score", "pleiotropy", "outlier_resistant"],
        reference=(
            "Zhao, Wang, Hemani, Bowden & Small (2020) "
            "Annals of Statistics 48(3)."
        ),
    ))

    # -- TARGET 21-item checklist ------------------------------------ #
    register(FunctionSpec(
        name="target_trial_checklist",
        category="target_trial",
        description=(
            "Render the JAMA/BMJ 2025 TARGET Statement 21-item reporting "
            "checklist as a completed Markdown table, auto-filled from a "
            "TargetTrialResult and flagged for any remaining TODO items."
        ),
        params=[
            ParamSpec("result", "TargetTrialResult", True),
            ParamSpec("fmt", "str", False, "markdown",
                      enum=["markdown", "text"]),
        ],
        returns="str",
        example="sp.target_trial_checklist(res, fmt='markdown')",
        tags=["target_trial", "reporting", "tte", "checklist"],
        reference=(
            "Hernán et al. (2025). TARGET Statement. "
            "JAMA/BMJ Sept 2025."
        ),
    ))

    # -- Longitudinal Bayesian Causal Forest ------------------------ #
    register(FunctionSpec(
        name="bcf_longitudinal",
        category="causal",
        description=(
            "Hierarchical Bayesian Causal Forest for longitudinal data "
            "(BCFLong) — allows mu_t(X), tau_t(X) to evolve across time "
            "with unit-level random intercepts."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("unit", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("n_trees_mu", "int", False, 200),
            ParamSpec("n_trees_tau", "int", False, 50),
            ParamSpec("n_bootstrap", "int", False, 100),
        ],
        returns="BCFLongResult",
        example=(
            "sp.bcf_longitudinal(df, outcome='y', treatment='d', "
            "unit='id', time='t', covariates=['x1','x2'])"
        ),
        tags=["bcf", "longitudinal", "panel", "hte"],
        reference="Prevot, Häring, Nichols, Holmes & Ganjgahi (arXiv:2508.08418, 2025).",
    ))

    # -- Time-series causal discovery extensions --------------------- #
    register(FunctionSpec(
        name="lpcmci",
        category="causal_discovery",
        description=(
            "Latent-PCMCI: time-series causal discovery allowing hidden "
            "common causes. Outputs a lag-specific adjacency tensor with "
            "typed edges (directed, bidirected, uncertain)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("variables", "list", False),
            ParamSpec("tau_max", "int", False, 3),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="LPCMCIResult",
        example="sp.lpcmci(df, variables=['gdp','inflation'], tau_max=4)",
        tags=["causal_discovery", "time_series", "latent", "lpcmci"],
        reference="Gerhardus & Runge (NeurIPS 2020).",
    ))
    register(FunctionSpec(
        name="dynotears",
        category="causal_discovery",
        description=(
            "DYNOTEARS: continuous-optimisation structure learning for "
            "structural VARs. Returns contemporaneous (W) and lagged (A) "
            "adjacency matrices with the contemporaneous part enforced "
            "to be acyclic via the NOTEARS h(W) penalty."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("variables", "list", False),
            ParamSpec("lag", "int", False, 1),
            ParamSpec("lambda_w", "float", False, 0.05),
            ParamSpec("lambda_a", "float", False, 0.05),
            ParamSpec("threshold", "float", False, 0.1),
        ],
        returns="DYNOTEARSResult",
        example="sp.dynotears(df, lag=2)",
        tags=["causal_discovery", "time_series", "notears", "svar"],
        reference="Pamfil et al. (AISTATS 2020).",
    ))

    # -- Sequential SDID (Arkhangelsky-Samkov 2024) ------------------ #
    register(FunctionSpec(
        name="sequential_sdid",
        category="causal",
        description=(
            "Sequential Synthetic DID for staggered-adoption panels "
            "(Arkhangelsky & Samkov 2024): processes cohorts in adoption "
            "order using not-yet-treated donors, avoiding TWFE negative "
            "weights and SDID overlap failures."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("unit", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("cohort", "str", True,
                      description="First-treated period column; never-treated = 0"),
            ParamSpec("never_treated_value", "Any", False, 0),
            ParamSpec("se_method", "str", False, "placebo",
                      enum=["placebo", "bootstrap", "jackknife"]),
            ParamSpec("n_reps", "int", False, 200),
            ParamSpec("cohort_weights", "str", False, "size",
                      enum=["size", "equal"]),
        ],
        returns="CausalResult",
        example=(
            "sp.sequential_sdid(df, outcome='y', unit='id', time='t', "
            "cohort='first_treat')"
        ),
        tags=["sdid", "synth", "staggered", "sequential"],
        reference="Arkhangelsky & Samkov (arXiv:2404.00164, 2024).",
    ))

    # -- Algorithmic fairness diagnostics ----------------------------- #
    register(FunctionSpec(
        name="counterfactual_fairness",
        category="fairness",
        description=(
            "Kusner-Loftus-Russell-Silva (2018) counterfactual-fairness "
            "test: compares factual vs. SCM-intervened predictions to "
            "measure path-specific dependence of a classifier on the "
            "protected attribute."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("predictor", "callable", True,
                      description="Callable(DataFrame) -> predictions"),
            ParamSpec("protected", "str", True),
            ParamSpec("scm_intervention", "callable", True),
            ParamSpec("threshold", "float", False, 0.05),
        ],
        returns="FairnessResult",
        example=(
            "sp.counterfactual_fairness(df, predictor=model.predict_proba, "
            "protected='gender', scm_intervention=scm_fn)"
        ),
        tags=["fairness", "counterfactual", "causal"],
        reference="Kusner, Loftus, Russell, Silva (2018), NeurIPS.",
    ))

    register(FunctionSpec(
        name="orthogonal_to_bias",
        category="fairness",
        description=(
            "Residualize features against the protected attribute as a "
            "pre-processing step toward counterfactual fairness."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("features", "list", True),
            ParamSpec("protected", "str", True),
        ],
        returns="DataFrame",
        example=(
            "sp.orthogonal_to_bias(df, features=['income','edu'], "
            "protected='gender')"
        ),
        tags=["fairness", "preprocessing", "residualize"],
        reference="Chen & Zhu (arXiv:2403.17852v3, 2024).",
    ))

    register(FunctionSpec(
        name="demographic_parity",
        category="fairness",
        description=(
            "Demographic-parity gap between groups defined by the "
            "protected attribute."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("predictions", "str", True),
            ParamSpec("protected", "str", True),
            ParamSpec("threshold", "float", False, 0.1),
        ],
        returns="FairnessResult",
        tags=["fairness", "parity", "audit"],
        reference="EEOC 80%-rule; Dwork et al. (2012).",
    ))

    register(FunctionSpec(
        name="equalized_odds",
        category="fairness",
        description=(
            "Hardt-Price-Srebro equalized-odds gap — max of TPR and FPR "
            "group differences."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("predictions", "str", True),
            ParamSpec("labels", "str", True),
            ParamSpec("protected", "str", True),
            ParamSpec("threshold", "float", False, 0.1),
        ],
        returns="FairnessResult",
        tags=["fairness", "equalized_odds", "audit"],
        reference="Hardt, Price, Srebro (2016), NeurIPS.",
    ))

    register(FunctionSpec(
        name="fairness_audit",
        category="fairness",
        description=(
            "One-shot dashboard combining demographic parity, equalized "
            "odds, and (optionally) counterfactual fairness."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("predictions", "str", True),
            ParamSpec("protected", "str", True),
            ParamSpec("labels", "str", False),
            ParamSpec("predictor", "callable", False),
            ParamSpec("scm_intervention", "callable", False),
        ],
        returns="FairnessAudit",
        tags=["fairness", "audit", "dashboard"],
    ))

    register(FunctionSpec(
        name="proximal_surrogate_index",
        category="surrogate",
        description=(
            "Proximal surrogate-index estimator: long-term ATE when an "
            "unobserved U confounds S→Y, using a proxy W and 2SLS-style "
            "bridge-function identification (Imbens-Kallus-Mao-Wang 2025, JRSS-B)."
        ),
        params=[
            ParamSpec("experimental", "DataFrame", True),
            ParamSpec("observational", "DataFrame", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("surrogates", "list", True),
            ParamSpec("proxies", "list", True),
            ParamSpec("long_term_outcome", "str", True),
            ParamSpec("covariates", "list", False),
            ParamSpec("n_boot", "int", False, 200),
        ],
        returns="CausalResult",
        example=(
            "sp.proximal_surrogate_index(exp, obs, treatment='T', "
            "surrogates=['s'], proxies=['w'], long_term_outcome='Y')"
        ),
        tags=["surrogate", "long_term", "proximal", "unobserved_confounding"],
        reference="Imbens, Kallus, Mao & Wang (2025). JRSS-B 87(2), 362-388. arXiv:2202.07234.",
    ))

    # ------------------------------------------------------------------
    # v1.1 additions (doc-alignment sprint — Gardner, Ahrens MA-DML,
    # Kernel IV, Continuous LATE, HAL-TMLE, Synth Survival, RD aliases)
    # ------------------------------------------------------------------

    register(FunctionSpec(
        name="gardner_did",
        category="causal",
        description=(
            "Gardner (2021) two-stage DID. Stage-1 fits two-way FEs on "
            "untreated observations; Stage-2 regresses the residualised "
            "outcome on treatment dummies (ATT or event study). Numerically "
            "close to Borusyak-Jaravel-Spiess imputation with unit-clustered SEs."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome column"),
            ParamSpec("group", "str", True, description="Unit/panel-id column"),
            ParamSpec("time", "str", True, description="Time column"),
            ParamSpec("first_treat", "str", True,
                      description="First-treatment-period column; 0/NaN/inf = never treated"),
            ParamSpec("controls", "list", False, None, "Additional covariates"),
            ParamSpec("event_study", "bool", False, False,
                      "If True, report coefficients by relative time k = t - first_treat"),
            ParamSpec("horizon", "list", False, None,
                      "Relative-time leads/lags to report (default range(-5, 6))"),
            ParamSpec("cluster", "str", False, None,
                      "Cluster variable for Stage-2 SEs (defaults to group)"),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult",
        example='sp.gardner_did(df, y="wage", group="county", time="year", first_treat="first_treat", event_study=True)',
        tags=["did", "causal", "staggered", "two-stage", "did2s"],
        reference="Gardner (2021), arXiv:2207.05943. Butts & Gardner (2022), R Journal 14(3).",
    ))

    register(FunctionSpec(
        name="dml_model_averaging",
        category="causal",
        description=(
            "Model-averaging DML (PLR) per Ahrens et al. (2025, JAE). Fits "
            "DML-PLR under multiple candidate nuisance learners and reports "
            "a risk-weighted (or equal/single-best) average of their θ "
            "estimates with a covariance-adjusted SE."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome column"),
            ParamSpec("treat", "str", True, description="Treatment column"),
            ParamSpec("covariates", "list", True, description="Covariate columns X"),
            ParamSpec("candidates", "list", False, None,
                      "List of (ml_g, ml_m, label) sklearn triples; defaults to Lasso/Ridge/RF/GBM"),
            ParamSpec("n_folds", "int", False, 5),
            ParamSpec("seed", "int", False, 0),
            ParamSpec("weight_rule", "str", False, "inverse_risk",
                      "Weighting of candidate estimators",
                      ["inverse_risk", "equal", "single_best"]),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="DMLAveragingResult",
        example=('sp.dml_model_averaging(df, y="y", treat="d", '
                 'covariates=[f"x{j}" for j in range(10)])'),
        tags=["dml", "causal", "model_averaging", "ensemble", "plr"],
        reference="Ahrens, Hansen, Schaffer & Wiemann (2025). JAE 40(3):249-269. DOI 10.1002/jae.3103.",
    ))

    # -- v1.7 long-panel DML (Semenova-Chernozhukov 2023) -------------- #
    register(FunctionSpec(
        name="dml_panel",
        category="causal",
        description=(
            "Long-panel Double/Debiased ML (Semenova-Chernozhukov 2023 "
            "simplified). Absorbs unit (and optional time) fixed "
            "effects via within-transform, cross-fits ML nuisance "
            "learners with folds that split units, and reports "
            "cluster-robust SE at the unit level. PLR moment "
            "(continuous or binary treatment)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome column"),
            ParamSpec("treat", "str", True, description="Treatment column"),
            ParamSpec("covariates", "list", True,
                      description="Covariate columns X_it"),
            ParamSpec("unit", "str", True,
                      description="Unit ID column (FE + clustering)"),
            ParamSpec("time", "str", False, None,
                      description="Time column (required if include_time_fe)"),
            ParamSpec("ml_g", "sklearn estimator", False,
                      description="Outcome nuisance learner"),
            ParamSpec("ml_m", "sklearn estimator", False,
                      description="Treatment nuisance learner"),
            ParamSpec("n_folds", "int", False, 5),
            ParamSpec("include_time_fe", "bool", False, False),
            ParamSpec("binary_treatment", "bool", False, False),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("seed", "int", False, 0),
        ],
        returns="DMLPanelResult",
        example=(
            'sp.dml_panel(df, y="log_wage", treat="union", '
            'covariates=["exper","educ"], unit="pid", time="year", '
            'include_time_fe=True)'
        ),
        tags=["dml", "causal", "panel", "fixed_effects",
              "cluster_robust_se", "long_panel"],
        reference=(
            "Semenova & Chernozhukov (2023) Econometrics Journal 26(2); "
            "Chernozhukov et al. (2018); Cameron & Miller (2015)."
        ),
        pre_conditions=[
            "long panel: at least unit and outcome columns; include_time_fe=True needs time column",
            "enough units (clusters) for cluster-robust SE — ≥ 30 ideally",
            "enough periods per unit for within-transform to leave variation in the treatment",
            "covariates are time-varying (pure time-invariant ones get absorbed by unit FE)",
        ],
        assumptions=[
            "Conditional unconfoundedness within unit: E[ε_it | X_it, α_i, λ_t] = 0",
            "Strict exogeneity conditional on covariates (weaker than standard FE)",
            "Nuisance learners converge fast enough (op(n^{-1/4})) after within-transform",
            "Cluster-robust inference valid: ≥ 30 units; no cross-unit dependence at t given X",
        ],
        failure_modes=[
            FailureMode(
                symptom="Few units (< 30) — cluster-robust SE under-coverage",
                exception="statspai.DataInsufficient",
                remedy="Use wild cluster bootstrap (sp.wild_cluster_bootstrap) or CR3 jackknife.",
                alternative="sp.wild_cluster_bootstrap",
            ),
            FailureMode(
                symptom="Within-unit variation in treatment is near zero",
                exception="statspai.DataInsufficient",
                remedy="Unit FE absorbs almost all treatment variation — switch to between estimator or cross-section.",
                alternative="sp.dml",
            ),
            FailureMode(
                symptom="Nuisance cross-val R² near zero on demeaned outcomes",
                exception="statspai.AssumptionWarning",
                remedy="ML nuisances not learnable on within-transformed data; use sp.panel FE or richer features.",
                alternative="sp.panel",
            ),
            FailureMode(
                symptom="Large residual serial correlation within unit",
                exception="statspai.AssumptionWarning",
                remedy="Cluster-robust SE handles within-unit correlation, but report Driscoll-Kraay (sp.panel robust='driscoll-kraay') if cross-sectional dependence likely.",
                alternative="sp.panel",
            ),
        ],
        alternatives=["dml", "panel", "msm", "bayes_dml"],
        typical_n_min=500,
    ))

    register(FunctionSpec(
        name="kernel_iv",
        category="causal",
        description=(
            "Kernel IV regression with uniform confidence bands (Lob et al. 2025). "
            "Estimates the structural function h*(d) = E[Y | do(D=d)] via kernel-weighted "
            "local averaging under a continuous instrument Z, with wild-bootstrap uniform SEs."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True, description="Continuous treatment D"),
            ParamSpec("instrument", "str", True, description="Continuous instrument Z"),
            ParamSpec("grid", "ndarray", False, None, "Grid of d-values (default 30 quantile-evenly spaced)"),
            ParamSpec("bandwidth", "float", False, None, "Silverman default"),
            ParamSpec("ridge", "float", False, 0.001, "Tikhonov regularisation"),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("n_boot", "int", False, 100),
            ParamSpec("seed", "int", False, 0),
        ],
        returns="KernelIVResult",
        example='sp.kernel_iv(df, y="wage", treat="schooling", instrument="compulsory")',
        tags=["iv", "kernel", "non-parametric", "uniform-ci", "continuous"],
        reference="Lob et al. (2025). arXiv:2511.21603.",
    ))

    register(FunctionSpec(
        name="continuous_iv_late",
        category="causal",
        description=(
            "LATE with a continuous instrument (Xie et al. 2025). Estimates the "
            "LATE on the maximal complier class via quantile-bin Wald ratios, "
            "weighted by the bin-pair with the largest first-stage response."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("instrument", "str", True, description="Continuous instrument"),
            ParamSpec("n_quantiles", "int", False, 4, "Number of instrument quantile bins"),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("n_boot", "int", False, 200),
            ParamSpec("seed", "int", False, 0),
        ],
        returns="ContinuousLATEResult",
        example='sp.continuous_iv_late(df, y="y", treat="d", instrument="z", n_quantiles=5)',
        tags=["iv", "late", "continuous-instrument", "complier"],
        reference="Zeng et al. (2025). arXiv:2504.03063.",
    ))

    register(FunctionSpec(
        name="hal_tmle",
        category="causal",
        description=(
            "TMLE with Highly Adaptive Lasso (HAL) nuisance learners "
            "(Qian & van der Laan 2025). Two variants: 'delta' plugs HAL into "
            "standard TMLE; 'projection' shrinks the targeting step using "
            "a tangent-space projection for reduced variance."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True, description="Binary treatment"),
            ParamSpec("covariates", "list", True),
            ParamSpec("variant", "str", False, "delta",
                      "HAL-TMLE variant", ["delta", "projection"]),
            ParamSpec("lambda_outcome", "float", False, None,
                      "Outcome L1 penalty; None → 5-fold CV"),
            ParamSpec("C_propensity", "float", False, 1.0,
                      "Inverse L1 penalty for HAL propensity classifier"),
            ParamSpec("max_anchors_per_col", "int", False, 40),
            ParamSpec("n_folds", "int", False, 5),
            ParamSpec("estimand", "str", False, "ATE", "Estimand", ["ATE", "ATT"]),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("random_state", "int", False, 42),
        ],
        returns="CausalResult",
        example='sp.hal_tmle(df, y="y", treat="d", covariates=["x1","x2","x3"])',
        tags=["tmle", "hal", "semiparametric", "causal", "double-robust"],
        reference="Li, Qiu, Wang & van der Laan (2025). arXiv:2506.17214.",
    ))

    register(FunctionSpec(
        name="synth_survival",
        category="causal",
        description=(
            "Synthetic Survival Control (Han & Shah 2025, arXiv:2511.14133). Fits a convex "
            "combination of donor Kaplan-Meier curves on the complementary "
            "log-log scale to match the treated arm's pre-treatment survival, "
            "then reports the post-treatment survival gap with placebo UCBs."
        ),
        params=[
            ParamSpec("data", "DataFrame", True,
                      description="Long panel with one row per (unit, time) and a precomputed KM survival"),
            ParamSpec("unit", "str", True, description="Unit/panel-id column"),
            ParamSpec("time", "str", True),
            ParamSpec("survival", "str", True,
                      description="Column with survival probability S_i(t)"),
            ParamSpec("treated", "str", True,
                      description="Boolean column or name of the single treated unit"),
            ParamSpec("treat_time", "float", True),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("n_placebos", "int", False, 100),
            ParamSpec("seed", "int", False, 0),
        ],
        returns="SyntheticSurvivalResult",
        example=('sp.synth_survival(df, unit="arm", time="month", '
                 'survival="km", treated="tr", treat_time=6)'),
        tags=["synth", "scm", "survival", "causal", "kaplan-meier"],
        reference="Han & Shah (2025). arXiv:2511.14133.",
    ))

    register(FunctionSpec(
        name="bridge",
        category="causal",
        description=(
            "Unified dispatcher for six causal-inference bridging theorems "
            "(2025-2026): DiD≡SC (Shi-Athey), EWM≡CATE (Ferman), "
            "IPW≡DR≡CB (Zhao-Percival), Bunching≡RDD (Lu-Wang-Xie), "
            "DR-via-Calibration (Zhang), Long-term-surrogate≡PCI (Imbens-Kallus-Mao-Wang). "
            "Reports both path estimates + doubly-robust recommendation."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("kind", "str", True,
                      "Which bridge to invoke",
                      ["did_sc", "ewm_cate", "cb_ipw", "kink_rdd",
                       "dr_calib", "surrogate_pci"]),
        ],
        returns="BridgeResult",
        example='sp.bridge(df, kind="did_sc", y="wage", group="state", time="year", first_treat="g")',
        tags=["bridge", "causal", "identification", "doubly-robust"],
        reference=(
            "Sun-Xie-Zhang (2503.11375); Ferman et al. (2510.26723); "
            "Zhao-Percival (2310.18563); Lu-Wang-Xie (2404.09117); "
            "Zhang et al. (2411.02771); Imbens-Kallus-Mao-Wang (2202.07234, JRSS-B 2025)."
        ),
    ))

    register(FunctionSpec(
        name="causal_dqn",
        category="causal",
        description=(
            "Causal deep Q-network (Li, Zhang, Bareinboim 2025, arXiv:2510.21110) for offline policy "
            "learning under unobserved confounding. Learns a "
            "confounding-robust Q-function via bootstrap data augmentation."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("states", "list", True),
            ParamSpec("action", "str", True),
            ParamSpec("reward", "str", True),
            ParamSpec("next_states", "list", False, None),
            ParamSpec("terminal", "str", False, None),
            ParamSpec("gamma", "float", False, 0.95),
            ParamSpec("n_episodes", "int", False, 50),
        ],
        returns="CausalDQNResult",
        example='sp.causal_dqn(df, states=["s1","s2"], action="a", reward="r")',
        tags=["rl", "causal", "policy", "offline"],
        reference="Li, Zhang & Bareinboim (2025). arXiv:2510.21110. Cunha et al. (2512.18135).",
    ))

    register(FunctionSpec(
        name="fortified_pci",
        category="causal",
        description=(
            "Fortified proximal causal inference (Yu, Shi & Tchetgen Tchetgen 2025). "
            "Adds a bridge-function stability constraint that gives robust "
            "ATT under mild misspecification of the outcome/treatment bridge."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("proxy_z", "list", True, description="Treatment-side proxies"),
            ParamSpec("proxy_w", "list", True, description="Outcome-side proxies"),
            ParamSpec("covariates", "list", False, None),
        ],
        returns="CausalResult",
        example='sp.fortified_pci(df, y="y", treat="d", proxy_z=["z"], proxy_w=["w"])',
        tags=["proximal", "pci", "unobserved-confounding", "fortified"],
        reference="Yu, Shi & Tchetgen Tchetgen (2025). arXiv:2506.13152.",
    ))

    register(FunctionSpec(
        name="bidirectional_pci",
        category="causal",
        description=(
            "Bidirectional proximal causal inference (Min, Zhang & Luo 2025). "
            "Solves for both outcome and treatment bridges simultaneously "
            "in a single two-way regression system."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("proxy_z", "list", True),
            ParamSpec("proxy_w", "list", True),
            ParamSpec("covariates", "list", False, None),
        ],
        returns="CausalResult",
        example='sp.bidirectional_pci(df, y="y", treat="d", proxy_z=["z"], proxy_w=["w"])',
        tags=["proximal", "pci", "bidirectional"],
        reference="Min, Zhang & Luo (2025). arXiv:2507.13965.",
    ))

    register(FunctionSpec(
        name="pci_mtp",
        category="causal",
        description=(
            "Proximal causal inference for modified treatment policies "
            "(Park & Ying 2025). Estimates the effect of a policy that "
            "shifts the treatment distribution (e.g., raises the dose by 10%) "
            "under unobserved confounding identified by PCI."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("proxy_z", "list", True),
            ParamSpec("proxy_w", "list", True),
            ParamSpec("policy", "Callable", True, description="Function D → D_shifted"),
        ],
        returns="CausalResult",
        example='sp.pci_mtp(df, y="y", treat="d", proxy_z=["z"], proxy_w=["w"], policy=lambda d: d+0.1)',
        tags=["proximal", "mtp", "modified-treatment-policy", "pci"],
        reference="Olivas-Martinez, Gilbert & Rotnitzky (2025). arXiv:2512.12038.",
    ))

    register(FunctionSpec(
        name="cluster_cross_interference",
        category="causal",
        description=(
            "Cluster-randomised trial under cross-cluster interference "
            "(Ding et al. 2025). Estimates direct + spillover effects when "
            "treatment of one cluster affects outcomes in adjacent clusters."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("cluster", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("exposure", "str", True,
                      description="Column with neighbours' treatment share"),
        ],
        returns="CrossClusterRCTResult",
        example='sp.cluster_cross_interference(df, y="y", cluster="city", treatment="d", exposure="neighbour_d")',
        tags=["interference", "spillover", "cluster-rct", "sutva"],
        reference="Leung (2023). arXiv:2310.18836.",
    ))

    register(FunctionSpec(
        name="beyond_average_late",
        category="causal",
        description=(
            "Beyond-average LATE (Xie-Wu 2025). Identifies the entire "
            "treatment-effect distribution among compliers under incomplete "
            "compliance, not just its mean."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("instrument", "str", True),
            ParamSpec("quantiles", "list", False, None,
                      "Quantiles τ at which to evaluate QTE (default 0.1..0.9 step 0.1)"),
        ],
        returns="BeyondAverageResult",
        example='sp.beyond_average_late(df, y="y", treat="d", instrument="z")',
        tags=["iv", "qte", "late", "complier", "distribution"],
        reference="Byambadalai, Hirata, Oka & Yasui (2025). arXiv:2509.15594.",
    ))

    register(FunctionSpec(
        name="conformal_fair_ite",
        category="causal",
        description=(
            "Counterfactual-fair conformal prediction for ITE (2025). "
            "Wraps standard conformal ITE intervals with a demographic-parity "
            "adjustment, giving distribution-free coverage under protected-attribute shifts."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("protected", "str", True, description="Protected-attribute column"),
            ParamSpec("alpha", "float", False, 0.1),
        ],
        returns="FairConformalResult",
        example='sp.conformal_fair_ite(df, y="y", treat="d", covariates=["x1","x2"], protected="race")',
        tags=["conformal", "fairness", "ite", "counterfactual"],
        reference="arXiv:2510.08724 / 2510.12822 (2025).",
    ))

    # ------------------------------------------------------------------
    # v1.1 frontier sprint (v3-doc Sprint 1): Abadie-Zhao, rbc bootstrap,
    # evidence-without-injustice, JAMA TARGET, harvest DID, BCF ordinal
    # + factor exposure, causal MAS, shift-share political, assimilation.
    # ------------------------------------------------------------------

    register(FunctionSpec(
        name="synth_experimental_design",
        category="synth",
        description=(
            "Abadie-Zhao (2025/2026) experimental-design synthetic controls: "
            "picks the best k candidate units to treat by minimising the "
            "sum of per-unit pre-period synthetic-control MSPEs."
        ),
        params=[
            ParamSpec("data", "DataFrame", True, description="Long-format panel"),
            ParamSpec("unit", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("k", "int", True, description="Number of units to treat"),
            ParamSpec("candidates", "list", False),
            ParamSpec("donors", "list", False),
            ParamSpec("risk", "str", False, "mspe", enum=["mspe", "rmse"]),
            ParamSpec("concentration_weight", "float", False, 0.0),
            ParamSpec("penalization", "float", False, 0.0),
            ParamSpec("n_random", "int", False, 500),
        ],
        returns="SynthExperimentalDesignResult",
        example="sp.synth_experimental_design(df, unit='u', time='t', outcome='y', k=5)",
        tags=["synth", "experimental_design", "selection", "abadie"],
        reference="Abadie & Zhao (2025/2026), MIT / Cambridge UP.",
    ))

    register(FunctionSpec(
        name="evidence_without_injustice",
        category="fairness",
        description=(
            "Kwak-Pleasants (2025) evidence-without-injustice counterfactual "
            "fairness test.  Freezes admissible-evidence features at their "
            "factual values and tests whether predictions still change under "
            "do(A=a')."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("predictor", "callable", True),
            ParamSpec("protected", "str", True),
            ParamSpec("admissible_features", "list", True),
            ParamSpec("scm_intervention", "callable", True),
            ParamSpec("alternative_values", "list", False),
            ParamSpec("threshold", "float", False, 0.05),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("n_boot", "int", False, 500),
        ],
        returns="EvidenceWithoutInjusticeResult",
        example=(
            "sp.fairness.evidence_without_injustice("
            "df, predictor, protected='race', admissible_features=['credit'], "
            "scm_intervention=fn)"
        ),
        tags=["fairness", "counterfactual", "algorithmic_bias", "kwak_pleasants"],
        reference="Loi, Di Bello & Cangiotti (arXiv:2510.12822, 2025).",
    ))

    register(FunctionSpec(
        name="harvest_did",
        category="did",
        description=(
            "Harvesting DID / Event Study (Borusyak et al. MIT/NBER 34550, "
            "2025).  Extracts every valid 2x2 DID comparison from a staggered "
            "panel, combines them with inverse-variance weights, and reports "
            "event-study + pretrend Wald tests."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("unit", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("treat", "str", False),
            ParamSpec("cohort", "str", False),
            ParamSpec("horizons", "list", False),
            ParamSpec("reference", "int", False, -1),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("weighting", "str", False, "precision",
                      enum=["precision", "equal", "n_treated"]),
        ],
        returns="CausalResult",
        example="sp.harvest_did(df, unit='u', time='t', outcome='y', treat='D')",
        tags=["did", "event_study", "harvest", "staggered"],
        reference="MIT / NBER WP 34550, 2025.",
    ))

    register(FunctionSpec(
        name="bcf_ordinal",
        category="causal",
        description=(
            "Bayesian Causal Forest for ordered / dose-level treatment "
            "(Zorzetto et al. 2026).  Estimates cumulative dose-response "
            "curves via chained BCF between consecutive levels."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("baseline", "str", False),
            ParamSpec("n_trees_mu", "int", False, 200),
            ParamSpec("n_trees_tau", "int", False, 50),
            ParamSpec("n_bootstrap", "int", False, 100),
            ParamSpec("n_folds", "int", False, 5),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("random_state", "int", False, 42),
        ],
        returns="BCFOrdinalResult",
        example='sp.bcf_ordinal(df, y="Y", treat="dose", covariates=["x1","x2"])',
        tags=["bcf", "ordinal", "dose_response", "bayesian"],
        reference="Zorzetto et al. (2026) working paper.",
    ))

    register(FunctionSpec(
        name="bcf_factor_exposure",
        category="causal",
        description=(
            "BCF on PCA-factor scores of a high-dimensional exposure vector "
            "(arXiv:2601.16595, 2026).  Compresses exposures via SVD or "
            "user-supplied loadings, then fits one BCF per factor."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("exposures", "list", True),
            ParamSpec("covariates", "list", True),
            ParamSpec("n_factors", "int", False, 3),
            ParamSpec("binarize", "str", False, "median",
                      enum=["median", "zero", "none"]),
            ParamSpec("loadings", "DataFrame", False),
            ParamSpec("n_bootstrap", "int", False, 100),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="BCFFactorExposureResult",
        example=(
            'sp.bcf_factor_exposure(df, y="Y", exposures=["z1","z2","z3"], '
            'covariates=["x1","x2"], n_factors=2)'
        ),
        tags=["bcf", "factor_analysis", "exposure_mixture", "bayesian"],
        reference="arXiv:2601.16595 (2026).",
    ))

    register(FunctionSpec(
        name="causal_mas",
        category="causal_llm",
        description=(
            "Multi-agent LLM causal discovery (arXiv:2509.00987, 2025). "
            "Runs proposer / critic / domain-expert / synthesiser agents "
            "over several rounds, returns per-edge confidence + audit log."
        ),
        params=[
            ParamSpec("variables", "list", True),
            ParamSpec("domain", "str", False, ""),
            ParamSpec("treatment", "str", False),
            ParamSpec("outcome", "str", False),
            ParamSpec("instruments", "list", False),
            ParamSpec("confounders", "list", False),
            ParamSpec("rounds", "int", False, 3),
            ParamSpec("final_threshold", "float", False, 0.5),
            ParamSpec("client", "object", False, description="LLM chat client"),
        ],
        returns="CausalMASResult",
        example=(
            "sp.causal_llm.causal_mas(variables=['age','sex','treatment','outcome'])"
        ),
        tags=["llm", "causal_discovery", "multi_agent", "dag"],
        reference="arXiv:2509.00987 (2025).",
    ))

    # ------------------------------------------------------------------
    # P1-C: data → publication-draft pipeline (v1.6)
    # ------------------------------------------------------------------
    register(FunctionSpec(
        name="paper",
        category="workflow",
        description=(
            "End-to-end 'data + question -> publication draft' "
            "pipeline. Parses a natural-language question, runs "
            "sp.causal() (diagnose + recommend + estimate + robustness), "
            "and assembles a Markdown / LaTeX / Word draft with EDA, "
            "identification verdict, estimator rationale, results, and "
            "robustness sections."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("question", "str", True,
                      description="Natural-language causal question"),
            ParamSpec("y", "str", False,
                      description="Outcome column (overrides parser)"),
            ParamSpec("treatment", "str", False),
            ParamSpec("covariates", "list", False),
            ParamSpec("id", "str", False),
            ParamSpec("time", "str", False),
            ParamSpec("running_var", "str", False),
            ParamSpec("instrument", "str", False),
            ParamSpec("cutoff", "float", False),
            ParamSpec("cohort", "str", False),
            ParamSpec("cluster", "str", False),
            ParamSpec("design", "str", False,
                      enum=["did", "rd", "iv", "rct", "observational",
                            "synth"]),
            ParamSpec("dag", "DAG", False),
            ParamSpec("fmt", "str", False, "markdown",
                      enum=["markdown", "tex", "docx"]),
            ParamSpec("output_path", "str", False),
            ParamSpec("include_eda", "bool", False, True),
            ParamSpec("include_robustness", "bool", False, True),
            ParamSpec("cite", "bool", False, True),
            ParamSpec("strict", "bool", False, False),
        ],
        returns="PaperDraft",
        example=(
            "sp.paper(df, 'effect of training on wages', design='did', "
            "treatment='trained', y='wage', time='year', id='worker_id')"
        ),
        tags=["workflow", "agent-native", "report", "publication",
              "end_to_end"],
        reference=(
            "Workflow design 2026-04-21 P1 spec; builds on "
            "sp.causal() (CausalWorkflow)."
        ),
        assumptions=[
            "Question parser is heuristic — explicit kwargs always win",
            "Underlying sp.causal() determines design when not specified",
        ],
        pre_conditions=[
            "data must contain the outcome column (`y` or parsed)",
            "If treatment given, it must be a column",
        ],
        failure_modes=[
            FailureMode(
                symptom="ValueError 'Could not determine the outcome y'",
                exception="ValueError",
                remedy=(
                    "Pass `y=...` explicitly or include 'effect of X "
                    "on Y' in the question text"
                ),
            ),
            FailureMode(
                symptom="Pipeline notes section appears in draft",
                exception="(none — informational)",
                remedy=(
                    "One pipeline stage failed; inspect "
                    "`draft.workflow.diagnostics` and pipeline_errors"
                ),
            ),
        ],
        alternatives=[
            "causal",       # workflow without paper rendering
            "recommend",    # estimator selection only
        ],
    ))

    # ------------------------------------------------------------------
    # P1-B: causal_text MVP (v1.6 experimental)
    # ------------------------------------------------------------------
    register(FunctionSpec(
        name="text_treatment_effect",
        category="causal_text",
        description=(
            "[experimental] Veitch-Wang-Blei (2020) text-as-treatment "
            "ATE estimation. Embeds a text column into n_components "
            "features (default hash embedder, deterministic) and uses "
            "them as confounder adjustment in OLS with HC1 SEs."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("text_col", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("covariates", "list", False),
            ParamSpec("embedder", "str", False, "hash",
                      enum=["hash", "sbert"]),
            ParamSpec("n_components", "int", False, 20),
            ParamSpec("seed", "int", False, 0),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="TextTreatmentResult",
        example=(
            "sp.text_treatment_effect(df, text_col='review', "
            "outcome='revenue', treatment='positive_label', "
            "n_components=20)"
        ),
        tags=["causal_text", "text_as_treatment", "embedding",
              "experimental", "agent-native"],
        reference="Veitch, Sridhar & Blei (UAI 2019); arXiv:1905.12741.",
        assumptions=[
            "All text-derived confounding is captured by the embedding",
            "Treatment is conditionally exogenous given embedding+covariates",
            "Linear outcome in treatment (HC1 OLS)",
        ],
        pre_conditions=[
            "data has the text/outcome/treatment columns",
            "n_obs >= max(20, n_components+4)",
        ],
        failure_modes=[
            FailureMode(
                symptom="DataInsufficient: 'Need at least N rows'",
                exception="statspai.DataInsufficient",
                remedy=(
                    "Lower n_components or supply more data"
                ),
            ),
            FailureMode(
                symptom=(
                    "ImportError on embedder='sbert'"
                ),
                exception="ImportError",
                remedy=(
                    "Install sentence-transformers: "
                    "`pip install sentence-transformers` or use "
                    "embedder='hash'"
                ),
                alternative="embedder='hash'",
            ),
        ],
        alternatives=[
            "sp.regress: plain OLS without text adjustment",
            "sp.dml: double machine learning with manual text features",
        ],
        typical_n_min=200,
    ))
    register(FunctionSpec(
        name="llm_annotator_correct",
        category="causal_text",
        description=(
            "[experimental] Egami et al. (2024) measurement-error "
            "correction for downstream OLS coefficients when the "
            "treatment indicator was produced by an LLM (or any "
            "imperfect classifier). Uses a small human-validated "
            "subset to estimate p_01 and p_10, then corrects the "
            "naive coefficient by 1/(1-p_01-p_10)."
        ),
        params=[
            ParamSpec("annotations_llm", "Series", True),
            ParamSpec("outcome", "Series", True),
            ParamSpec("annotations_human", "Series", True,
                      description="NaN where unavailable; >=30 valid rows"),
            ParamSpec("covariates", "DataFrame", False),
            ParamSpec("method", "str", False, "hausman",
                      enum=["hausman"]),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="LLMAnnotatorResult",
        example=(
            "sp.llm_annotator_correct(annotations_llm=df.llm_label, "
            "annotations_human=df.human_label, outcome=df.y)"
        ),
        tags=["causal_text", "measurement_error", "llm_annotator",
              "hausman", "experimental", "agent-native"],
        reference=(
            "Egami, Hinck, Stewart & Wei (NeurIPS 2024); "
            "arXiv:2306.04746. Hausman et al. (1998)."
        ),
        assumptions=[
            "Binary treatment indicator",
            "Misclassification is independent of outcome conditional on T",
            "Validation subset is representative of the full sample",
        ],
        pre_conditions=[
            "annotations_llm is binary (0/1)",
            ">=30 rows with both LLM and human labels",
            "Both T_human classes present in validation set",
        ],
        failure_modes=[
            FailureMode(
                symptom=(
                    "DataInsufficient: 'At least 30 validation rows'"
                ),
                exception="statspai.DataInsufficient",
                remedy=(
                    "Hand-label more rows so that annotations_human has "
                    ">=30 non-NaN entries spanning both classes"
                ),
            ),
            FailureMode(
                symptom=(
                    "IdentificationFailure: '1-p_01-p_10 <= 0'"
                ),
                exception="statspai.IdentificationFailure",
                remedy=(
                    "Misclassification too severe — re-prompt the LLM "
                    "or hand-label"
                ),
            ),
        ],
        alternatives=[
            "sp.regress with raw LLM label (biased — for comparison only)",
        ],
        typical_n_min=300,
    ))

    # ------------------------------------------------------------------
    # P1-A: closed-loop LLM-assisted causal discovery (v1.6)
    # ------------------------------------------------------------------
    register(FunctionSpec(
        name="llm_dag_constrained",
        category="dag",
        description=(
            "Closed-loop LLM-assisted DAG discovery: iterate "
            "LLM-propose -> constrained PC -> CI-test validate -> demote, "
            "until edge set converges or max_iter is hit. Returns a final "
            "DAG with per-edge LLM confidence and CI-test p-value."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("variables", "list", False,
                      description="Subset of columns to include"),
            ParamSpec("descriptions", "dict", False,
                      description="Variable -> human description"),
            ParamSpec("oracle", "callable", False,
                      description="LLM oracle f(vars, desc)->[(a,b[,conf])]"),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("ci_test", "str", False, "fisherz",
                      enum=["fisherz"]),
            ParamSpec("max_iter", "int", False, 3),
            ParamSpec("high_conf_threshold", "float", False, 0.7),
            ParamSpec("low_conf_threshold", "float", False, 0.3),
            ParamSpec("forbid_low_conf", "bool", False, False),
        ],
        returns="LLMConstrainedDAGResult",
        example=(
            "sp.llm_dag_constrained(df, variables=['X','Y','Z'], "
            "oracle=lambda v, d: [('X','Y',0.9)], max_iter=3)"
        ),
        tags=["llm", "causal_discovery", "dag", "background_knowledge",
              "agent-native"],
        reference=(
            "Kıcıman et al. arXiv:2305.00050; Long et al. arXiv:2307.02390; "
            "Jiralerspong et al. arXiv:2402.01207."
        ),
        assumptions=[
            "Faithfulness (PC's CI tests reflect d-separation)",
            "Causal sufficiency (no unmeasured confounder among `variables`)",
            "Linear/Gaussian relationships (Fisher-Z partial correlation)",
        ],
        pre_conditions=[
            "data has at least 2 numeric columns intersecting `variables`",
            "n_obs >> number of variables (PC unstable when p ~ n)",
        ],
        failure_modes=[
            FailureMode(
                symptom="ValueError 'Variable X not in data.columns'",
                exception="ValueError",
                remedy="Pass only column names that exist in data",
            ),
            FailureMode(
                symptom="Loop never converges (max_iter reached)",
                exception="(none — returns converged=False)",
                remedy=(
                    "Inspect iteration_log for oscillating edges; "
                    "raise alpha or lower high_conf_threshold"
                ),
                alternative="sp.llm_dag_propose (single-shot)",
            ),
        ],
        alternatives=[
            "sp.llm_dag_propose: single-shot LLM proposal without CI loop",
            "sp.pc_algorithm: data-only PC (no LLM)",
            "sp.causal_mas: multi-agent LLM consensus",
        ],
        typical_n_min=200,
    ))
    register(FunctionSpec(
        name="llm_dag_validate",
        category="dag",
        description=(
            "Per-edge CI-test validation of a declared DAG. For each "
            "directed edge a->b, run partial-correlation independence "
            "test conditioning on parents(b)\\{a}. Edges with p>alpha "
            "are flagged unsupported."
        ),
        params=[
            ParamSpec("dag", "DAG", True),
            ParamSpec("data", "DataFrame", True),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("ci_test", "str", False, "fisherz",
                      enum=["fisherz"]),
        ],
        returns="DAGValidationResult",
        example=(
            "sp.llm_dag_validate(my_dag, df, alpha=0.05)"
        ),
        tags=["dag", "validation", "ci_test", "background_knowledge",
              "agent-native"],
        reference="Spirtes-Glymour-Scheines (2000); standard CI-test logic.",
        assumptions=[
            "Faithfulness", "Linear/Gaussian (Fisher-Z)",
        ],
        failure_modes=[
            FailureMode(
                symptom="Many supported=False edges",
                exception="(none — informational)",
                remedy=(
                    "DAG may be misspecified; rerun discovery or check "
                    "for nonlinearity / unmeasured confounders"
                ),
                alternative="sp.llm_dag_constrained",
            ),
        ],
        typical_n_min=200,
    ))

    register(FunctionSpec(
        name="shift_share_political",
        category="bartik",
        description=(
            "Park-Xu (2026) political-science shift-share IV: long-difference "
            "Bartik IV with AKM shock-cluster SE, Rotemberg top-K, and "
            "share-balance diagnostics."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("unit", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("endog", "str", True),
            ParamSpec("shares", "DataFrame", True),
            ParamSpec("shocks", "Series", True),
            ParamSpec("covariates", "list", False),
            ParamSpec("leave_one_out", "bool", False, True),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="ShiftSharePoliticalResult",
        example=(
            "sp.shift_share_political(df, unit='state', time='year', "
            "outcome='vote', endog='expo', shares=S, shocks=g)"
        ),
        tags=["bartik", "shift_share", "iv", "political_science"],
        reference="Park & Xu (arXiv:2603.00135, 2026).",
    ))

    register(FunctionSpec(
        name="causal_kalman",
        category="assimilation",
        description=(
            "Closed-form Kalman filter over a stream of causal-effect "
            "estimates + SEs.  Produces a running posterior over the "
            "time-varying (or static) causal effect."
        ),
        params=[
            ParamSpec("estimates", "list", True),
            ParamSpec("standard_errors", "list", True),
            ParamSpec("prior_mean", "float", False, 0.0),
            ParamSpec("prior_var", "float", False, 1.0),
            ParamSpec("process_var", "float", False, 0.0),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="AssimilationResult",
        example="sp.causal_kalman(ests, ses, prior_mean=0.0, prior_var=1.0)",
        tags=["assimilation", "kalman", "streaming", "bayesian"],
        reference="Nature Communications 2026.",
    ))

    register(FunctionSpec(
        name="assimilative_causal",
        category="assimilation",
        description=(
            "End-to-end Assimilative Causal Inference pipeline (Nature "
            "Communications 2026): for each data batch, apply `estimator` "
            "to get (θ̂, SE), then fuse via Kalman filtering or particle filter."
        ),
        params=[
            ParamSpec("batches", "list", True),
            ParamSpec("estimator", "callable", True,
                      description="Maps a batch to (theta_hat, se)"),
            ParamSpec("prior_mean", "float", False, 0.0),
            ParamSpec("prior_var", "float", False, 1.0),
            ParamSpec("process_var", "float", False, 0.0),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("backend", "str", False, "kalman",
                      enum=["kalman", "particle"]),
        ],
        returns="AssimilationResult",
        example=(
            "sp.assimilative_causal(batches, "
            "lambda df: (sp.regress('y~d', data=df).params['d'], "
            "sp.regress('y~d', data=df).std_errors['d']))"
        ),
        tags=["assimilation", "streaming", "bayesian", "rwe"],
        reference="Nature Communications 2026.",
    ))

    # ------------------------------------------------------------------
    # v1.4 Sprint 2 additions:
    # shift_share_political_panel, particle_filter, LLM SDK adapters
    # ------------------------------------------------------------------

    register(FunctionSpec(
        name="shift_share_political_panel",
        category="bartik",
        description=(
            "Multi-period panel shift-share IV (Park-Xu 2026 §4.2): "
            "pooled 2SLS with unit/time/two-way FEs over a time-varying "
            "Bartik instrument.  Reports per-period event-study, "
            "aggregate Rotemberg top-K, and share-balance F-tests."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("unit", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("endog", "str", True),
            ParamSpec("shares", "DataFrame", True),
            ParamSpec("shocks", "DataFrame", True),
            ParamSpec("covariates", "list", False),
            ParamSpec("cluster", "str", False, "unit",
                      enum=["unit", "time", "twoway"]),
            ParamSpec("fe", "str", False, "two-way",
                      enum=["two-way", "unit", "time", "none"]),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="ShiftSharePoliticalPanelResult",
        example=(
            "sp.shift_share_political_panel(df, unit='state', time='year', "
            "outcome='vote', endog='exp', shares=S, shocks=G)"
        ),
        tags=["bartik", "shift_share", "iv", "panel", "political_science"],
        reference="Park & Xu (arXiv:2603.00135, 2026) §4.2.",
    ))

    register(FunctionSpec(
        name="particle_filter",
        category="assimilation",
        description=(
            "Bootstrap SIR particle filter for non-Gaussian assimilative "
            "causal inference.  Supports arbitrary prior sampler, "
            "transition sampler, and observation log-pdf, with systematic "
            "resampling triggered by an ESS threshold."
        ),
        params=[
            ParamSpec("estimates", "list", True),
            ParamSpec("standard_errors", "list", True),
            ParamSpec("prior_mean", "float", False, 0.0),
            ParamSpec("prior_var", "float", False, 1.0),
            ParamSpec("process_sd", "float", False, 0.0),
            ParamSpec("n_particles", "int", False, 2000),
            ParamSpec("ess_resample_threshold", "float", False, 0.5),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="AssimilationResult",
        example=(
            "sp.assimilation.particle_filter(ests, ses, n_particles=3000, "
            "random_state=0)"
        ),
        tags=["assimilation", "particle_filter", "streaming", "bayesian"],
        reference="Gordon-Salmond-Smith 1993; Douc-Cappé 2005.",
    ))

    register(FunctionSpec(
        name="openai_client",
        category="causal_llm",
        description=(
            "Construct an OpenAI-compatible LLM client for use with "
            "sp.causal_llm.causal_mas.  Requires the optional openai>=1.0 "
            "extra.  Supports custom base_url for Azure / vLLM / Ollama."
        ),
        params=[
            ParamSpec("model", "str", False, "gpt-4o-mini"),
            ParamSpec("api_key", "str", False),
            ParamSpec("base_url", "str", False),
            ParamSpec("organization", "str", False),
            ParamSpec("temperature", "float", False, 0.0),
            ParamSpec("max_tokens", "int", False, 1024),
            ParamSpec("max_retries", "int", False, 3),
        ],
        returns="LLMClient",
        example="sp.causal_llm.openai_client(model='gpt-4o-mini')",
        tags=["llm", "openai", "adapter"],
        reference="OpenAI Python SDK v1.x.",
    ))

    register(FunctionSpec(
        name="anthropic_client",
        category="causal_llm",
        description=(
            "Construct an Anthropic-compatible LLM client for use with "
            "sp.causal_llm.causal_mas.  Requires the optional "
            "anthropic>=0.30 extra.  Defaults to Claude Opus 4.7."
        ),
        params=[
            ParamSpec("model", "str", False, "claude-opus-4-7"),
            ParamSpec("api_key", "str", False),
            ParamSpec("base_url", "str", False),
            ParamSpec("temperature", "float", False, 0.0),
            ParamSpec("max_tokens", "int", False, 1024),
            ParamSpec("max_retries", "int", False, 3),
        ],
        returns="LLMClient",
        example="sp.causal_llm.anthropic_client(model='claude-opus-4-7')",
        tags=["llm", "anthropic", "claude", "adapter"],
        reference="Anthropic Python SDK v0.30+.",
    ))

    register(FunctionSpec(
        name="echo_client",
        category="causal_llm",
        description=(
            "Deterministic scripted-response LLM client for testing "
            "sp.causal_llm.causal_mas without network access."
        ),
        params=[
            ParamSpec("response_fn", "callable", True,
                      description="Maps (role, prompt) -> str"),
        ],
        returns="LLMClient",
        example=(
            "sp.causal_llm.echo_client(lambda r, p: 'age -> treatment')"
        ),
        tags=["llm", "testing", "adapter"],
        reference="StatsPAI test utility.",
    ))

    # =================================================================== #
    #  v1.5 unified family dispatchers (mirror sp.synth / sp.decompose /  #
    #  sp.dml): one entry per family with a method/kind/design switch.    #
    # =================================================================== #

    register(FunctionSpec(
        name="mr",
        category="causal",
        description=(
            "Unified Mendelian Randomization dispatcher. "
            "method= selects the estimator: "
            "'ivw' / 'egger' / 'median' / 'penalized_median' / 'mode' / "
            "'all' (runs IVW+Egger+Median together) / "
            "'mvmr' / 'mediation' / 'bma' (multi-exposure) / "
            "'presso' / 'radial' / 'leave_one_out' / 'steiger' / "
            "'heterogeneity' / 'pleiotropy_egger' / 'f_statistic' "
            "(diagnostics).  Kwargs are passed through to the target "
            "function unchanged; see sp.mendelian_family guide."
        ),
        params=[
            ParamSpec("method", "str", False, "ivw",
                      "MR estimator / diagnostic — call "
                      "sp.mr_available_methods() for the full list."),
        ],
        returns="dict | MRResult | MVMRResult | MediationMRResult | MRBMAResult | MRPressoResult | RadialResult | LeaveOneOutResult | SteigerResult | HeterogeneityResult | PleiotropyResult | FStatisticResult | ModeBasedResult",
        example=(
            'sp.mr("ivw", beta_exposure=bx, beta_outcome=by, '
            'se_exposure=sx, se_outcome=sy)'
        ),
        tags=["mr", "mendelian", "iv", "causal", "dispatcher",
              "genetic", "two-sample"],
        reference=(
            "Burgess et al. 2013; Bowden et al. 2015/2016/2017/2018; "
            "Verbanck et al. 2018; Hartwig et al. 2017; Sanderson et al. "
            "2019; Zuber et al. 2020."
        ),
        pre_conditions=[
            "SNP-summary statistics for exposure and outcome aligned by SNP",
            "beta_exposure / beta_outcome / se_exposure / se_outcome arrays of equal length",
            "≥ 10 genetic instruments for reliable IVW/median/mode; ≥ 20 for robust Egger intercept",
            "mvmr needs SNP × exposure associations matrix",
        ],
        assumptions=[
            "Relevance: SNPs predict exposure (F-statistic ≥ 10 per SNP or set-F)",
            "Independence: SNPs ⊥ confounders of exposure-outcome",
            "Exclusion restriction: SNPs affect outcome only through exposure (InSIDE for Egger; ≥ 50% valid for median; modal for mode-based)",
            "Monotonicity when interpreting LATE on genetically-shifted subpopulation",
        ],
        failure_modes=[
            FailureMode(
                symptom="Egger intercept p < 0.05 — directional pleiotropy",
                exception="statspai.AssumptionViolation",
                remedy="Use weighted-median or mode-based estimator; report Egger intercept + I² as pleiotropy diagnostic.",
                alternative="sp.mr_median",
            ),
            FailureMode(
                symptom="Q-statistic rejects homogeneity (Cochran's Q p < 0.05)",
                exception="statspai.AssumptionWarning",
                remedy="Heterogeneity across SNPs — run sp.mr_presso to detect/remove outliers.",
                alternative="sp.mr_presso",
            ),
            FailureMode(
                symptom="Set-F < 10 (weak instruments in aggregate)",
                exception="statspai.AssumptionWarning",
                remedy="Weak-IV bias in IVW — use debiased IVW or LAP-type estimator (sp.mr_lap).",
                alternative="sp.mr_lap",
            ),
            FailureMode(
                symptom="Steiger test flags reverse causation",
                exception="statspai.IdentificationFailure",
                remedy="SNPs explain more outcome variance than exposure — direction of effect questionable.",
                alternative="",
            ),
        ],
        alternatives=[
            "mr_ivw", "mr_egger", "mr_median", "mr_presso",
            "mr_multivariable", "iv",
        ],
        typical_n_min=10,
    ))

    register(FunctionSpec(
        name="conformal",
        category="causal",
        description=(
            "Unified conformal causal inference dispatcher. "
            "kind= selects the estimator: "
            "'cate' / 'counterfactual' / 'ite' (Lei-Candès 2021 base) / "
            "'weighted' (TBCR 2019 primitive) / "
            "'density' / 'multidp' / 'debiased' / 'fair' "
            "(2025-2026 frontier) / "
            "'continuous' (dose-response) / "
            "'interference' (cluster-exchangeable).  Kwargs pass through "
            "to the target function; see sp.conformal_family guide."
        ),
        params=[
            ParamSpec("kind", "str", False, "cate",
                      "Conformal estimator — call "
                      "sp.conformal_available_kinds() for the full list."),
        ],
        returns=(
            "CausalResult | ConformalCounterfactualResult | "
            "ConformalITEResult | ConformalDensityResult | "
            "MultiDPConformalResult | DebiasedConformalResult | "
            "FairConformalResult | ContinuousConformalResult | "
            "InterferenceConformalResult | tuple"
        ),
        example=(
            'sp.conformal("cate", data=df, y="y", treat="d", '
            'covariates=["x1", "x2"], alpha=0.1)'
        ),
        tags=["conformal", "causal", "prediction_interval", "cate",
              "ite", "dispatcher", "distribution-free", "coverage"],
        reference=(
            "Lei & Candès 2021 JRSS-B; Tibshirani et al. 2019 NeurIPS; "
            "Kim-Jeong-Barber-Lee 2024; Romano et al. 2019."
        ),
        pre_conditions=[
            "calibration sample disjoint from training sample (auto-split or user-supplied)",
            "exchangeability between calibration and test distributions (weighted variants for covariate shift)",
            "for CATE / ITE variants: unconfoundedness + overlap on covariates",
            "≥ 500 calibration observations for reliable finite-sample coverage at alpha ≤ 0.1",
        ],
        assumptions=[
            "Exchangeability of calibration and test points (base case)",
            "For kind='weighted': known or estimable density ratio between calibration and test",
            "For kind='cate' / 'ite': selection-on-observables with correct propensity / outcome model",
            "For kind='interference': cluster-exchangeable exchangeability",
        ],
        failure_modes=[
            FailureMode(
                symptom="Calibration and test distributions differ (covariate shift)",
                exception="statspai.AssumptionViolation",
                remedy="Use kind='weighted' with estimated density ratios.",
                alternative="",
            ),
            FailureMode(
                symptom="Calibration set too small — intervals wide",
                exception="statspai.DataInsufficient",
                remedy="Increase calibration sample or raise alpha; coverage gets loose below ~100.",
                alternative="",
            ),
            FailureMode(
                symptom="Miscalibrated nuisance (propensity / outcome) for CATE/ITE",
                exception="statspai.AssumptionWarning",
                remedy="Use kind='debiased' which orthogonalises via DML-style nuisance handling.",
                alternative="",
            ),
        ],
        alternatives=[
            "conformal_cate",
            "weighted_conformal_prediction",
            "conformal_counterfactual",
        ],
        typical_n_min=500,
    ))

    register(FunctionSpec(
        name="interference",
        category="causal",
        description=(
            "Unified interference / spillover dispatcher. "
            "design= selects the estimator: "
            "'partial' (Hudgens-Halloran cluster) / "
            "'network_exposure' (Aronow-Samii HT) / "
            "'peer_effects' (Manski / Bramoullé linear-in-means) / "
            "'network_hte' (Wu & Yuan 2025 orthogonal, arXiv:2509.18484) / "
            "'inward_outward' (directed network; Fang, Airoldi & Forastiere 2025, arXiv:2506.06615) / "
            "'cluster_matched_pair' (Bai 2022) / "
            "'cluster_cross' (Ding et al. 2025) / "
            "'cluster_staggered' (Zhou et al. 2025) / "
            "'dnc_gnn' (Zhao et al. 2026).  Kwargs pass through "
            "to the target function; see sp.interference_family guide."
        ),
        params=[
            ParamSpec("design", "str", False, "partial",
                      "Interference design — call "
                      "sp.interference_available_designs() for the "
                      "full list."),
        ],
        returns=(
            "CausalResult | NetworkExposureResult | PeerEffectsResult | "
            "NetworkHTEResult | InwardOutwardResult | MatchedPairResult "
            "| CrossClusterRCTResult | StaggeredClusterRCTResult | "
            "DNCGNNDiDResult"
        ),
        example=(
            'sp.interference("partial", data=df, y="y", '
            'treat="d", cluster="household")'
        ),
        tags=["interference", "spillover", "sutva", "network", "peer",
              "cluster_rct", "dispatcher", "causal"],
        reference=(
            "Hudgens & Halloran 2008 JASA; Aronow & Samii 2017 AoAS; "
            "Manski 1993; Bramoullé-Djebbari-Fortin 2009; "
            "Wu & Yuan 2025 (arXiv:2509.18484); Bai 2022; Ding et al. 2025; "
            "Zhou et al. 2025; Zhao et al. 2026."
        ),
        pre_conditions=[
            "clustered data OR network / adjacency matrix",
            "treatment varies within cluster (or exposure is well-defined on the network)",
            "enough clusters (≥ 30) for cluster-robust inference",
        ],
        assumptions=[
            "Partial interference (within-cluster spillover only) OR an explicit exposure mapping",
            "SUTVA modulo the declared spillover structure",
            "Correctly specified exposure function (e.g. fraction-treated, neighbour-share)",
            "Overlap: positive probability of every (treatment × exposure) cell",
        ],
        failure_modes=[
            FailureMode(
                symptom="Few clusters (< 30) with cluster-level inference",
                exception="statspai.DataInsufficient",
                remedy="Use wild cluster bootstrap or permutation; CR3 jackknife for < 50.",
                alternative="sp.wild_cluster_bootstrap",
            ),
            FailureMode(
                symptom="Very few treated per cluster",
                exception="statspai.DataInsufficient",
                remedy="Saturation DID (Baird et al.) or cluster-level estimand instead of individual.",
                alternative="sp.cluster_matched_pair",
            ),
            FailureMode(
                symptom="Exposure mapping misspecified",
                exception="statspai.AssumptionWarning",
                remedy="Report sensitivity to multiple exposure functions (fraction / any / k-NN).",
                alternative="sp.network_exposure",
            ),
        ],
        alternatives=[
            "network_exposure", "peer_effects", "cluster_matched_pair",
            "cluster_cross_interference", "cluster_staggered_rollout",
        ],
        typical_n_min=500,
    ))

    # -- Distributional / continuous-treatment / multi-valued / network families --
    register(FunctionSpec(
        name="qdid",
        category="causal",
        description=(
            "Quantile Difference-in-Differences (Athey & Imbens 2006 CIC). "
            "Estimates QTE at multiple quantiles via changes-in-changes on "
            "a 2×2 design with bootstrap SE."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome"),
            ParamSpec("group", "str", True, description="Binary treated / control group"),
            ParamSpec("time", "str", True, description="Binary pre / post indicator"),
            ParamSpec("quantiles", "list", False,
                      description="Quantiles to estimate, defaults to [0.1, ..., 0.9]"),
            ParamSpec("n_boot", "int", False, 500),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="QTEResult",
        example='sp.qdid(df, y="wage", group="treat", time="post")',
        tags=["qte", "qdid", "cic", "distributional", "did", "causal"],
        reference="Athey & Imbens (2006) Econometrica — Changes-in-Changes",
        pre_conditions=[
            "panel or repeated cross-section",
            "group is binary 0/1",
            "time is binary 0/1 (pre / post)",
            "outcome is continuous",
        ],
        assumptions=[
            "CIC rank invariance: the quantile rank in the untreated distribution is stable across groups",
            "Continuous outcome support covering both groups in both periods",
            "SUTVA (no cross-group spillovers)",
        ],
        failure_modes=[
            FailureMode(
                symptom="Outcome heavily discrete / zero-inflated",
                exception="statspai.AssumptionViolation",
                remedy="CIC rank-matching is unstable on discrete supports — use QTE regression (sp.qte) or Firpo-RIF.",
                alternative="sp.qte",
            ),
            FailureMode(
                symptom="Bootstrap CI across quantiles varies wildly",
                exception="statspai.DataInsufficient",
                remedy="Thin tails at extreme quantiles — restrict to [0.2, 0.8] or raise n_boot to 2000.",
                alternative="",
            ),
        ],
        alternatives=["qte", "did", "rifreg"],
        typical_n_min=500,
    ))

    register(FunctionSpec(
        name="qte",
        category="causal",
        description=(
            "Quantile Treatment Effect via quantile regression or IPW "
            "weighting. Returns QTE at supplied quantiles with bootstrap "
            "SE."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treatment", "str", True),
            ParamSpec("quantiles", "list", False),
            ParamSpec("method", "str", False, "quantile_regression",
                      "Estimation method",
                      ["quantile_regression", "ipw"]),
            ParamSpec("controls", "list", False),
            ParamSpec("n_boot", "int", False, 500),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="QTEResult",
        example='sp.qte(df, y="earnings", treatment="training", quantiles=[0.25, 0.5, 0.75])',
        tags=["qte", "quantile", "distributional", "causal"],
        reference="Koenker & Bassett (1978); Firpo (2007); Chernozhukov & Hansen (2005)",
        pre_conditions=[
            "binary or continuous treatment (method='quantile_regression' supports both; 'ipw' needs binary)",
            "continuous outcome",
            "controls cover the confounding set (for 'quantile_regression')",
            "overlap when method='ipw'",
        ],
        assumptions=[
            "For 'quantile_regression': unconfoundedness conditional on controls",
            "For 'ipw': unconfoundedness + overlap 0 < e(x) < 1",
            "Correct parametric quantile model (sensitivity tested via multiple quantiles)",
        ],
        failure_modes=[
            FailureMode(
                symptom="Large IPW weights (method='ipw')",
                exception="statspai.AssumptionViolation",
                remedy="Extreme propensities — trim (sp.trimming) or switch to doubly-robust DR-QTE.",
                alternative="sp.trimming",
            ),
            FailureMode(
                symptom="Quantile crossing",
                exception="statspai.AssumptionWarning",
                remedy="Use rearrangement (Chernozhukov-Fernandez-Val-Galichon) or monotone constraints.",
                alternative="",
            ),
        ],
        alternatives=["qdid", "rifreg", "cic", "metalearner"],
        typical_n_min=500,
    ))

    register(FunctionSpec(
        name="dose_response",
        category="causal",
        description=(
            "Dose-response function for a continuous treatment under "
            "unconfoundedness. Uses generalised propensity-score weighting "
            "or double ML for the conditional expectation E[Y(d)]."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True, description="Continuous treatment / dose"),
            ParamSpec("covariates", "list", True),
            ParamSpec("n_dose_points", "int", False, 20),
            ParamSpec("dose_range", "tuple", False,
                      description="(lo, hi) over which to evaluate dose-response"),
            ParamSpec("n_boot", "int", False, 500),
        ],
        returns="DoseResponseResult",
        example='sp.dose_response(df, y="y", treat="dose", covariates=["x1","x2"])',
        tags=["continuous_treatment", "dose_response", "gps", "causal"],
        reference="Hirano & Imbens (2004); Kennedy et al. (2017) JRSSB",
        pre_conditions=[
            "treat is continuous (numeric, not binary)",
            "covariates comprise the confounding set",
            "n ≥ 1000 for stable dose-response curves",
            "weak overlap: positive density of treatment across the confounder range",
        ],
        assumptions=[
            "Weak unconfoundedness: Y(d) ⊥ D | X for each d",
            "Generalised overlap: positive conditional density of D at each evaluated dose",
            "Smoothness of dose-response function (for local-polynomial / kernel smoothing)",
        ],
        failure_modes=[
            FailureMode(
                symptom="Sparse data at extreme doses",
                exception="statspai.DataInsufficient",
                remedy="Narrow dose_range; CIs at tails will be wide and uninformative.",
                alternative="",
            ),
            FailureMode(
                symptom="Heavy-tailed generalised propensity weights",
                exception="statspai.AssumptionViolation",
                remedy="Use stabilised weights or restrict to common-support dose window.",
                alternative="",
            ),
        ],
        alternatives=["dml", "metalearner", "causal_forest"],
        typical_n_min=1000,
    ))

    register(FunctionSpec(
        name="spillover",
        category="causal",
        description=(
            "Direct + spillover treatment effect estimation under partial "
            "interference (within-cluster). Uses the Hudgens-Halloran "
            "decomposition with chosen exposure function."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("cluster", "str", True,
                      description="Cluster column (interference boundary)"),
            ParamSpec("covariates", "list", False),
            ParamSpec("exposure_fn", "str", False, "fraction",
                      "Exposure function",
                      ["fraction", "any", "count"]),
            ParamSpec("n_bootstrap", "int", False, 500),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult (direct + spillover effects)",
        example='sp.spillover(df, y="y", treat="d", cluster="village")',
        tags=["spillover", "interference", "partial", "cluster"],
        reference="Hudgens & Halloran (2008); Basse & Feller (2018)",
        pre_conditions=[
            "data has a cluster column defining the interference boundary",
            "treatment varies within clusters",
            "≥ 30 clusters for cluster-robust inference",
        ],
        assumptions=[
            "Partial interference: spillover only within cluster, not across",
            "Correct exposure function (fraction / any / count — sensitivity tested)",
            "Overlap: every (treatment × exposure) cell has positive probability",
        ],
        failure_modes=[
            FailureMode(
                symptom="No within-cluster variation in treatment",
                exception="statspai.DataInsufficient",
                remedy="Assignments are cluster-level — use sp.cluster_matched_pair or cluster-level ATE.",
                alternative="sp.cluster_matched_pair",
            ),
            FailureMode(
                symptom="Exposure function misspecified",
                exception="statspai.AssumptionWarning",
                remedy="Compare estimates under exposure_fn in {fraction, any, count}.",
                alternative="",
            ),
        ],
        alternatives=["network_exposure", "cluster_matched_pair", "peer_effects"],
        typical_n_min=500,
    ))

    register(FunctionSpec(
        name="multi_treatment",
        category="causal",
        description=(
            "Effects of multi-valued (3+ level) treatments via AIPW. "
            "Returns pairwise contrasts versus a reference level."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True, description="Multi-valued treatment (int)"),
            ParamSpec("covariates", "list", True),
            ParamSpec("reference", "int", False,
                      description="Reference treatment level (defaults to 0 / smallest)"),
            ParamSpec("n_bootstrap", "int", False, 500),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult with pairwise contrasts",
        example='sp.multi_treatment(df, y="wage", treat="program", covariates=["age","edu"])',
        tags=["multi_treatment", "multi_arm", "aipw", "causal"],
        reference="Robins et al. (1994); Imbens (2000); Yang et al. (2016)",
        pre_conditions=[
            "treat is integer-valued with ≥ 2 distinct levels",
            "covariates comprise the confounding set",
            "enough units per treatment arm (≥ 50 per arm)",
            "overlap: every treatment arm has positive probability at each x",
        ],
        assumptions=[
            "Generalised unconfoundedness: Y(a) ⊥ T | X for all a",
            "Generalised overlap: 0 < P(T=a | X) < 1 for each arm a",
            "SUTVA across arms",
            "Correctly specified (or ML-approximated) nuisance models",
        ],
        failure_modes=[
            FailureMode(
                symptom="Some arm has near-zero propensity in the data",
                exception="statspai.AssumptionViolation",
                remedy="Violates overlap — drop that arm or use bounds.",
                alternative="sp.bounds",
            ),
            FailureMode(
                symptom="Tiny treatment cells (< 30)",
                exception="statspai.DataInsufficient",
                remedy="Collapse sparse arms or use regularised multinomial propensity.",
                alternative="",
            ),
        ],
        alternatives=["multi_arm_forest", "dml", "metalearner"],
        typical_n_min=300,
    ))

    register(FunctionSpec(
        name="network_exposure",
        category="causal",
        description=(
            "Aronow-Samii Horvitz-Thompson estimator for arbitrary "
            "interference via a user-supplied exposure mapping. Handles "
            "Bernoulli / complete randomisation designs with simulated "
            "conservative variance."
        ),
        params=[
            ParamSpec("Y", "array", True, description="Outcome vector"),
            ParamSpec("Z", "array", True, description="Treatment vector (0/1)"),
            ParamSpec("adjacency", "array", True,
                      description="Adjacency matrix (n x n) or sparse"),
            ParamSpec("mapping", "str", False, "as4",
                      "Exposure mapping",
                      ["as4", "as3", "as2", "custom"]),
            ParamSpec("p_treat", "float", False,
                      description="Marginal treatment probability"),
            ParamSpec("design", "str", False, "bernoulli",
                      "Randomisation design", ["bernoulli", "complete"]),
            ParamSpec("n_sim", "int", False, 2000),
        ],
        returns="NetworkExposureResult with per-exposure HT estimates",
        example='sp.network_exposure(Y=y, Z=z, adjacency=A, mapping="as4")',
        tags=["interference", "network", "aronow_samii",
              "horvitz_thompson"],
        reference="Aronow & Samii (2017) AoAS",
        pre_conditions=[
            "adjacency is a binary n × n matrix encoding network ties",
            "Y, Z have same length n",
            "randomisation design is known (bernoulli with p_treat, or complete)",
            "n_sim ≥ 2000 for stable Monte Carlo variance",
        ],
        assumptions=[
            "Exposure mapping is correctly specified (as4 / as3 / as2 — Aronow-Samii hierarchy)",
            "Positivity: every exposure level has positive probability under the design",
            "Network adjacency is fixed / known (measurement error in ties introduces bias)",
        ],
        failure_modes=[
            FailureMode(
                symptom="Some exposure level has < 5 observed units",
                exception="statspai.DataInsufficient",
                remedy="Switch to a coarser mapping (as4 → as3) or increase sample size.",
                alternative="",
            ),
            FailureMode(
                symptom="Variance estimate extremely conservative (wide CI)",
                exception="statspai.AssumptionWarning",
                remedy="HT-style variance is conservative by design — use sp.spillover for cluster case.",
                alternative="sp.spillover",
            ),
        ],
        alternatives=["spillover", "peer_effects", "cluster_matched_pair"],
        typical_n_min=200,
    ))

    # ------------------------------------------------------------------
    # DiD frontier: continuous treatment + on/off switching
    # ------------------------------------------------------------------
    register(FunctionSpec(
        name="continuous_did",
        category="causal",
        description=(
            "DiD with continuous treatment intensity. Four modes: (i) "
            "'twfe' TWFE with dose×post interaction; (ii) 'att_gt' dose-"
            "quantile group-time ATT versus the untreated (dose=0) arm "
            "with bootstrap SE (heuristic); (iii) 'dose_response' local-"
            "linear regression of ΔY=Y_post−Y_pre on baseline dose; (iv) "
            "'cgs' Callaway-Goodman-Bacon-Sant'Anna (2024) ATT(d|g,t) MVP "
            "— 2-period design, OR only, bootstrap SE, [待核验] markers "
            "on paper formulas. Full CGS parity (cohort aggregation, DR/"
            "IPW, analytical IF variance) is on the roadmap — see "
            "docs/rfc/continuous_did_cgs.md."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome variable"),
            ParamSpec("dose", "str", True,
                      description="Continuous treatment / dose variable"),
            ParamSpec("time", "str", True, description="Time period column"),
            ParamSpec("id", "str", True, description="Unit identifier"),
            ParamSpec("post", "str", False, None,
                      "Binary post-treatment indicator "
                      "(inferred from t_pre / t_post if omitted)"),
            ParamSpec("t_pre", "int", False, None,
                      "Last pre-treatment period"),
            ParamSpec("t_post", "int", False, None,
                      "First post-treatment period"),
            ParamSpec("method", "str", False, "att_gt",
                      "Estimation mode",
                      ["att_gt", "twfe", "dose_response", "cgs"]),
            ParamSpec("n_quantiles", "int", False, 5,
                      "Number of dose quantiles for discretisation"),
            ParamSpec("controls", "list", False, None, "Control variables"),
            ParamSpec("cluster", "str", False, None,
                      "Cluster variable for SE (TWFE mode)"),
            ParamSpec("n_boot", "int", False, 500,
                      "Bootstrap replications for SE"),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("seed", "int", False, None),
        ],
        returns="CausalResult",
        example=(
            'sp.continuous_did(df, y="wage", dose="training_hours", '
            'time="year", id="worker_id", t_pre=2019, t_post=2020)'
        ),
        tags=["did", "continuous_treatment", "dose_response", "causal",
              "acrt", "frontier"],
        reference=(
            "Callaway, Goodman-Bacon & Sant'Anna (2024) "
            "[@callaway2024difference]; de Chaisemartin & D'Haultfœuille "
            "(2018) [@dechaisemartin2018fuzzy]."
        ),
        pre_conditions=[
            "panel data with unit × time × outcome × continuous dose",
            "at least one unit with dose == 0 acts as untreated control "
            "(or the lowest dose quantile is used as control)",
            "both a pre and a post period per unit",
        ],
        assumptions=[
            "Parallel trends in potential outcomes across dose levels",
            "No anticipation of treatment",
            "Strong parallel trends (CGS 2024) required for ATT(d|g,t) "
            "interpretation in att_gt mode",
            "Overlap: positive density of dose in the treated support",
        ],
        failure_modes=[
            FailureMode(
                symptom="No units with dose == 0",
                exception="DataInsufficient",
                remedy=("Lowest-dose quantile is auto-used as the control arm; "
                        "pass an explicit never-treated indicator via post= if "
                        "this is not intended."),
                alternative="callaway_santanna",
            ),
            FailureMode(
                symptom="Dose collapses to one quantile",
                exception="DataInsufficient",
                remedy=("Not enough variation in dose. Lower n_quantiles, or "
                        "check that dose is truly continuous in the baseline "
                        "period."),
                alternative="",
            ),
            FailureMode(
                symptom=("SE appears too small — you want the CGS 2024 "
                        "analytical influence-function variance"),
                exception="",
                remedy=("Current modes use bootstrap / OLS SE. The CGS 2024 "
                        "analytical IF is tracked in "
                        "docs/rfc/continuous_did_cgs.md; until landed, "
                        "inflate n_boot or cluster bootstrap manually."),
                alternative="",
            ),
        ],
        alternatives=[
            "callaway_santanna",
            "did_multiplegt",
            "dose_response",
        ],
        typical_n_min=100,
    ))

    register(FunctionSpec(
        name="did_multiplegt",
        category="causal",
        description=(
            "de Chaisemartin & D'Haultfœuille (2020) DID_M estimator. "
            "Weighted average of consecutive-period DID cells where "
            "treatment 'switchers' are compared to 'stayers'. Handles "
            "treatments that switch on AND off (unlike Callaway-Sant'Anna "
            "which assumes staggered adoption). Supports placebo lags, "
            "dynamic horizons, cluster bootstrap SE, joint placebo test "
            "and average-cumulative-effect summary from dCDH (2024). The "
            "heteroskedastic-weights variant and full dCDH (2024) "
            "intertemporal event-study (did_multiplegt_dyn Stata) are on "
            "the roadmap — see docs/rfc/multiplegt_dyn.md."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome variable"),
            ParamSpec("group", "str", True, description="Unit identifier"),
            ParamSpec("time", "str", True, description="Time period column"),
            ParamSpec("treatment", "str", True,
                      description="Binary current-treatment indicator "
                                  "(may switch on and off)"),
            ParamSpec("controls", "list", False, None,
                      "Controls residualised via first differences"),
            ParamSpec("placebo", "int", False, 0,
                      "Number of pre-treatment placebo lags"),
            ParamSpec("dynamic", "int", False, 0,
                      "Number of post-treatment dynamic horizons"),
            ParamSpec("cluster", "str", False, None,
                      "Cluster variable for bootstrap (defaults to group)"),
            ParamSpec("n_boot", "int", False, 100,
                      "Cluster-bootstrap replications"),
            ParamSpec("seed", "int", False, None),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns=(
            "CausalResult with placebo / dynamic event-study in "
            "model_info['event_study'], joint placebo Wald test, and "
            "avg_cumulative_effect summary."
        ),
        example=(
            'sp.did_multiplegt(df, y="wage", group="county", time="year", '
            'treatment="treated", placebo=2, dynamic=3, cluster="state", '
            'n_boot=200, seed=42)'
        ),
        tags=["did", "dcdh", "switchers", "on_off", "event_study",
              "placebo", "dynamic", "causal"],
        reference=(
            "de Chaisemartin & D'Haultfœuille (2020) "
            "[@dechaisemartin2020two]; 2022 survey "
            "[@dechaisemartin2022fixed]; 2024 joint placebo + avg "
            "cumulative [@dechaisemartin2024difference]."
        ),
        pre_conditions=[
            "long-format panel with one row per unit × period",
            "treatment is binary (0/1) and may vary over time within a unit",
            "at least two periods observed per unit so a first difference "
            "can be computed",
        ],
        assumptions=[
            "Parallel trends between switchers and stayers",
            "Stable treatment effects across consecutive periods (for the "
            "DID_M weighted average interpretation)",
            "No anticipation",
            "Cluster-bootstrap validity requires G large and clusters "
            "independent",
        ],
        failure_modes=[
            FailureMode(
                symptom="No switching cells (nobody changes treatment)",
                exception="DataInsufficient",
                remedy=("did_multiplegt identifies effects only from treatment "
                        "switches. Fall back to callaway_santanna if the "
                        "design is staggered adoption."),
                alternative="callaway_santanna",
            ),
            FailureMode(
                symptom=("Joint placebo test rejects — parallel trends "
                         "unlikely"),
                exception="AssumptionViolation",
                remedy=("Inspect model_info['event_study'] by placebo lag; "
                        "consider honest_did sensitivity bounds or add "
                        "controls."),
                alternative="honest_did",
            ),
            FailureMode(
                symptom=("Bootstrap SE unstable with small G"),
                exception="",
                remedy=("Raise n_boot, or switch to a wild cluster bootstrap. "
                        "The analytical influence-function SE from dCDH "
                        "(2020) is not yet implemented — see RFC."),
                alternative="",
            ),
        ],
        alternatives=[
            "callaway_santanna",
            "sun_abraham",
            "did_imputation",
            "gardner_did",
            "wooldridge_did",
        ],
        typical_n_min=50,
    ))

    # ==================================================================
    # DiD family: rich specs for previously auto-registered estimators
    # (added 2026-04-24 as part of docs/rfc/did_roadmap_gap_audit.md §5)
    # ==================================================================

    register(FunctionSpec(
        name="did_2x2",
        category="causal",
        description=(
            "Canonical 2×2 DID: two groups (treated / control) × two periods "
            "(pre / post). Point estimate via either group-means differencing "
            "or OLS on the treat × post interaction; optional covariates, "
            "robust / cluster SE, and sample weights."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome variable"),
            ParamSpec("treat", "str", True,
                      description="Binary treatment-group indicator (0/1)"),
            ParamSpec("time", "str", True, description="Time / period indicator"),
            ParamSpec("covariates", "list", False, None,
                      "Covariates included additively; for DR use sp.drdid"),
            ParamSpec("cluster", "str", False, None,
                      "Column for cluster-robust SE (defaults to treat)"),
            ParamSpec("robust", "bool", False, True,
                      "Heteroskedasticity-robust SE when no cluster provided"),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("weights", "str", False, None,
                      "Optional column name for sampling weights"),
        ],
        returns="CausalResult",
        example=(
            'sp.did_2x2(df, y="earnings", treat="treated", time="year")'
        ),
        tags=["did", "2x2", "canonical", "causal"],
        reference="Card & Krueger (1994); Angrist & Pischke (2009) MHE Ch.5",
        pre_conditions=[
            "data has exactly two time periods (pre, post)",
            "treat is 0/1 constant within unit (unit-level, not time-varying)",
            "at least a handful of treated and control units",
        ],
        assumptions=[
            "Parallel trends",
            "No anticipation",
            "SUTVA (no spillovers)",
        ],
        failure_modes=[
            FailureMode(
                symptom="Staggered timing (> 2 periods with varying treat start)",
                exception="MethodIncompatibility",
                remedy="Use sp.callaway_santanna / sp.sun_abraham / sp.did_imputation.",
                alternative="callaway_santanna",
            ),
            FailureMode(
                symptom="Very few clusters at the group level",
                exception="AssumptionWarning",
                remedy="Use wild cluster bootstrap via sp.wild_cluster_bootstrap.",
                alternative="wild_cluster_bootstrap",
            ),
        ],
        alternatives=["drdid", "did_analysis", "callaway_santanna"],
        typical_n_min=30,
    ))

    register(FunctionSpec(
        name="drdid",
        category="causal",
        description=(
            "Doubly-robust DiD (Sant'Anna & Zhao 2020). Combines outcome "
            "regression with IPW; consistent if either model is correct. "
            "Primary estimator for 2×2 DiD with covariates."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome variable"),
            ParamSpec("group", "str", True,
                      description="Unit-level treatment indicator (0/1)"),
            ParamSpec("time", "str", True, description="Time period column"),
            ParamSpec("covariates", "list", False, None, "Covariates X"),
            ParamSpec("method", "str", False, "dr",
                      "Estimation method",
                      ["dr", "or", "ipw", "reg", "stdipw"]),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("n_boot", "int", False, None,
                      "Bootstrap replications (None → analytical SE)"),
            ParamSpec("seed", "int", False, None),
        ],
        returns="CausalResult",
        example=(
            'sp.drdid(df, y="y", group="d", time="t", '
            'covariates=["age","edu"])'
        ),
        tags=["did", "dr", "doubly_robust", "causal", "2x2", "ipw"],
        reference="Sant'Anna & Zhao (2020) J. Econometrics 219(1) "
                  "[@santanna2020doubly]",
        pre_conditions=[
            "panel or repeated cross-section with 2 periods",
            "group is a binary unit-level treatment indicator",
            "covariates have non-zero variance and overlap",
        ],
        assumptions=[
            "Conditional parallel trends given X",
            "Overlap / positivity: 0 < P(D=1|X) < 1",
            "Correct specification of at least one nuisance model",
            "No anticipation",
        ],
        failure_modes=[
            FailureMode(
                symptom="Propensity score near 0/1 (overlap violation)",
                exception="AssumptionViolation",
                remedy="Trim extreme propensity scores or use sp.ipw_trim.",
                alternative="",
            ),
        ],
        alternatives=["did_2x2", "callaway_santanna", "wooldridge_did"],
        typical_n_min=100,
    ))

    register(FunctionSpec(
        name="sun_abraham",
        category="causal",
        description=(
            "Sun-Abraham (2021) interaction-weighted event-study. Fixes the "
            "contamination in dynamic event-study TWFE coefficients from "
            "other relative-time bins by using cohort-specific interaction "
            "weights. Canonical companion to Callaway-Sant'Anna for event "
            "studies."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("g", "str", True,
                      description="First-treatment period (0 = never-treated)"),
            ParamSpec("t", "str", True, description="Time period column"),
            ParamSpec("i", "str", True, description="Unit identifier"),
            ParamSpec("event_window", "tuple", False, None,
                      "(lead, lag) window for event-study coefficients"),
            ParamSpec("control_group", "str", False, "nevertreated",
                      "Control arm", ["nevertreated", "notyettreated"]),
            ParamSpec("covariates", "list", False, None),
            ParamSpec("cluster", "str", False, None,
                      "Cluster variable (defaults to i)"),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult with event-study in model_info['event_study']",
        example=(
            'sp.sun_abraham(df, y="y", g="first_treat", t="year", i="unit")'
        ),
        tags=["did", "event_study", "sun_abraham", "iw", "staggered", "causal"],
        reference="Sun & Abraham (2021) J. Econometrics 225(2) "
                  "[@sun2021estimating]",
        pre_conditions=[
            "panel with unit × time × outcome",
            "g is the first-treatment period (int), 0 / NaN for never-treated",
            "≥ 2 pre-periods per cohort for event-study leads",
        ],
        assumptions=[
            "Parallel trends across cohorts",
            "No anticipation within event_window lead horizon",
            "SUTVA",
        ],
        failure_modes=[
            FailureMode(
                symptom="No never-treated cohort when control_group='nevertreated'",
                exception="DataInsufficient",
                remedy="Pass control_group='notyettreated' or add never-treated units.",
                alternative="callaway_santanna",
            ),
        ],
        alternatives=["callaway_santanna", "did_imputation", "gardner_did",
                      "wooldridge_did"],
        typical_n_min=50,
    ))

    register(FunctionSpec(
        name="did_imputation",
        category="causal",
        description=(
            "Borusyak-Jaravel-Spiess (2024) imputation DiD. Fits a TWFE "
            "model on untreated observations only, imputes counterfactual "
            "Y(0) for treated obs, and averages the imputation residuals. "
            "Efficient under no-anticipation + parallel trends; analytical "
            "SE via bjs_inference."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("group", "str", True, description="Unit identifier"),
            ParamSpec("time", "str", True, description="Time period column"),
            ParamSpec("first_treat", "str", True,
                      description="First-treatment period; 0 = never-treated"),
            ParamSpec("controls", "list", False, None),
            ParamSpec("horizon", "list", False, None,
                      "Relative-time leads / lags (default: all available)"),
            ParamSpec("cluster", "str", False, None),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult",
        example=(
            'sp.did_imputation(df, y="y", group="i", time="t", '
            'first_treat="g_first")'
        ),
        tags=["did", "imputation", "bjs", "efficient", "event_study", "causal"],
        reference="Borusyak, Jaravel & Spiess (2024) RES "
                  "[@borusyak2024revisiting]; Borusyak & Jaravel (2022) "
                  "[@borusyak2022quasi]",
        pre_conditions=[
            "panel with unit × time × outcome",
            "first_treat encodes cohort (first treated period or 0)",
            "≥ 1 never-treated unit OR ≥ 1 late-treated cohort",
        ],
        assumptions=[
            "Parallel trends in absolute levels",
            "No anticipation (no pre-treatment reaction)",
            "SUTVA",
        ],
        failure_modes=[
            FailureMode(
                symptom="All units treated (no untreated observations to fit)",
                exception="DataInsufficient",
                remedy="Impossible to impute Y(0); use sp.did_multiplegt if on/off switching.",
                alternative="did_multiplegt",
            ),
        ],
        alternatives=["callaway_santanna", "sun_abraham", "gardner_did",
                      "wooldridge_did"],
        typical_n_min=50,
    ))

    register(FunctionSpec(
        name="wooldridge_did",
        category="causal",
        description=(
            "Wooldridge (2021) extended TWFE (ETWFE). Saturated TWFE "
            "regression with cohort × post interactions; recovers "
            "cohort-specific ATTs. Numerically equivalent to CS / SA / BJS "
            "under the saturated specification."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("group", "str", True, description="Unit identifier"),
            ParamSpec("time", "str", True),
            ParamSpec("first_treat", "str", True,
                      description="First-treatment period; 0 = never-treated"),
            ParamSpec("controls", "list", False, None),
            ParamSpec("cluster", "str", False, None),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult",
        example=(
            'sp.wooldridge_did(df, y="y", group="i", time="t", '
            'first_treat="g")'
        ),
        tags=["did", "twfe", "wooldridge", "etwfe", "staggered", "causal"],
        reference="Wooldridge (2021) working paper "
                  "[@wooldridge2021two]; McDermott (2023) R etwfe package",
        pre_conditions=[
            "panel with unit × time × outcome",
            "first_treat cohort column (first period treated, 0 = never)",
        ],
        assumptions=[
            "Parallel trends per cohort",
            "No anticipation",
            "SUTVA",
        ],
        failure_modes=[
            FailureMode(
                symptom="Singleton cohorts with one unit",
                exception="DataInsufficient",
                remedy="Aggregate small cohorts or drop them.",
                alternative="callaway_santanna",
            ),
        ],
        alternatives=["callaway_santanna", "sun_abraham", "did_imputation",
                      "etwfe"],
        typical_n_min=50,
    ))

    register(FunctionSpec(
        name="etwfe",
        category="causal",
        description=(
            "Extended Two-Way Fixed Effects (Wooldridge 2021). Explicit API "
            "mirroring the R etwfe package. Alias for sp.wooldridge_did; "
            "same numerical output."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("group", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("first_treat", "str", True),
            ParamSpec("controls", "list", False, None),
            ParamSpec("cluster", "str", False, None),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("xvar", "list", False, None,
                      "R-style alias for controls"),
            ParamSpec("panel", "bool", False, True,
                      "If False, treat data as repeated cross-section"),
        ],
        returns="CausalResult",
        example='sp.etwfe(df, y="y", group="i", time="t", first_treat="g")',
        tags=["did", "etwfe", "twfe", "wooldridge", "staggered", "causal",
              "r_parity"],
        reference="Wooldridge (2021) [@wooldridge2021two]",
        alternatives=["wooldridge_did", "callaway_santanna", "did_imputation"],
        typical_n_min=50,
    ))

    register(FunctionSpec(
        name="bacon_decomposition",
        category="causal",
        description=(
            "Goodman-Bacon (2021) decomposition of the TWFE DiD coefficient "
            "into a weighted sum of underlying 2×2 DID comparisons: "
            "treated-vs-never, treated-vs-notyet, earlier-vs-later. "
            "Diagnoses when TWFE is contaminated by already-treated units "
            "acting as controls."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True,
                      description="Time-varying binary treatment indicator"),
            ParamSpec("time", "str", True),
            ParamSpec("id", "str", True),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="dict with 'weights', 'estimates', 'summary'",
        example=(
            'sp.bacon_decomposition(df, y="y", treat="d", time="t", id="i")'
        ),
        tags=["did", "diagnostic", "bacon", "twfe", "decomposition"],
        reference="Goodman-Bacon (2021) J. Econometrics "
                  "[@goodmanbacon2021difference]",
        pre_conditions=[
            "panel with time-varying binary treatment",
            "≥ 2 treatment cohorts OR 1 cohort + never-treated",
        ],
        assumptions=[
            "Treatment is absorbing (staggered adoption, no reversal)",
            "Standard DiD assumptions hold within each 2x2 comparison",
        ],
        failure_modes=[
            FailureMode(
                symptom="Treatment switches on and off (dCDH setting)",
                exception="MethodIncompatibility",
                remedy="Bacon decomp assumes absorbing treatment. Use "
                       "sp.did_multiplegt for on/off switching.",
                alternative="did_multiplegt",
            ),
        ],
        alternatives=["did_multiplegt", "callaway_santanna"],
        typical_n_min=30,
    ))

    register(FunctionSpec(
        name="ddd",
        category="causal",
        description=(
            "Triple Differences (DDD) estimator. Adds a within-treatment-group "
            "subgroup that is unaffected by treatment as an additional "
            "control dimension, relaxing parallel trends from 'same trend "
            "across groups' to 'same differential trend across subgroups "
            "within groups'."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True,
                      description="Primary treatment indicator"),
            ParamSpec("time", "str", True),
            ParamSpec("subgroup", "str", True,
                      description="Within-group subgroup (1=affected, 0=not)"),
            ParamSpec("covariates", "list", False, None),
            ParamSpec("cluster", "str", False, None),
            ParamSpec("robust", "bool", False, True),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("weights", "str", False, None),
        ],
        returns="CausalResult",
        example=(
            'sp.ddd(df, y="y", treat="policy_state", time="year", '
            'subgroup="eligible")'
        ),
        tags=["did", "ddd", "triple", "causal"],
        reference=(
            "Gruber (1994) JPE — classical DDD; Olden & Møen (2022) "
            "[@olden2022triple] — heterogeneity-robust variant (separate "
            "function on the roadmap)."
        ),
        pre_conditions=[
            "treat × time × subgroup variation exists",
            "subgroup is binary and meaningful within treatment group",
        ],
        assumptions=[
            "Parallel trends in the DDD differential (weaker than DID PT)",
            "No anticipation",
            "SUTVA",
        ],
        failure_modes=[
            FailureMode(
                symptom="Staggered adoption with heterogeneous effects",
                exception="AssumptionWarning",
                remedy="Textbook DDD can have negative weights with staggered "
                       "timing. The Olden-Møen (2022) / Strezhnev (2023) "
                       "heterogeneity-robust DDD is on the roadmap "
                       "(see docs/rfc/did_roadmap_gap_audit.md §4).",
                alternative="",
            ),
        ],
        alternatives=["did_2x2", "callaway_santanna"],
        typical_n_min=100,
    ))

    register(FunctionSpec(
        name="cic",
        category="causal",
        description=(
            "Changes-in-Changes (Athey & Imbens 2006). Nonparametric "
            "quantile DiD that identifies the full counterfactual outcome "
            "distribution for treated units, not just the mean. Reports "
            "quantile treatment effects (QTE) via empirical-CDF "
            "transformation; bootstrap SE."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("group", "str", True,
                      description="Treatment-group indicator (0/1)"),
            ParamSpec("time", "str", True,
                      description="Period indicator (0=pre, 1=post)"),
            ParamSpec("quantiles", "list", False, None,
                      "Quantile grid (default: deciles)"),
            ParamSpec("n_boot", "int", False, 500),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("seed", "int", False, None),
            ParamSpec("n_grid", "int", False, 100,
                      "Grid size for inverse-CDF mapping"),
        ],
        returns="CausalResult with quantile-specific effects in detail",
        example='sp.cic(df, y="y", group="d", time="t")',
        tags=["did", "cic", "quantile", "qte", "nonparametric", "causal"],
        reference="Athey & Imbens (2006) Econometrica 74(2) "
                  "[@athey2006identification]",
        pre_conditions=[
            "continuous-ish outcome with sufficient support overlap "
            "between treated and control",
            "2 periods, 2 groups",
        ],
        assumptions=[
            "Rank-invariance of untreated potential outcomes across periods",
            "Time-invariant group-level production technology "
            "(distributional DiD)",
            "SUTVA",
        ],
        failure_modes=[
            FailureMode(
                symptom="Discrete outcome with few support points",
                exception="AssumptionWarning",
                remedy="CIC quantile transformation degenerates; use sp.qte "
                       "or sp.drdid for mean effects.",
                alternative="qte",
            ),
        ],
        alternatives=["qte", "drdid", "did_2x2"],
        typical_n_min=200,
    ))

    register(FunctionSpec(
        name="stacked_did",
        category="causal",
        description=(
            "Stacked DiD (Cengiz, Dube, Lindner, Zipperer 2019). For each "
            "treatment cohort, constructs a sub-experiment with only that "
            "cohort + clean (never-treated or not-yet-treated) controls, "
            "then TWFE on the stacked panel. Robust to staggered-adoption "
            "contamination at the cost of dropping late-treated units in "
            "early sub-experiments."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("group", "str", True, description="Unit identifier"),
            ParamSpec("time", "str", True),
            ParamSpec("first_treat", "str", True),
            ParamSpec("window", "tuple", False, (-5, 5),
                      "Event-time (lead, lag) window per sub-experiment"),
            ParamSpec("controls", "list", False, None),
            ParamSpec("cluster", "str", False, None),
            ParamSpec("never_treated_only", "bool", False, False,
                      "Use only never-treated as controls (drops late-treated)"),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult with event-study coefficients",
        example=(
            'sp.stacked_did(df, y="y", group="i", time="t", '
            'first_treat="g", window=(-4, 4))'
        ),
        tags=["did", "stacked", "event_study", "cengiz", "staggered", "causal"],
        reference="Cengiz, Dube, Lindner & Zipperer (2019) QJE "
                  "[@cengiz2019effect]",
        pre_conditions=[
            "staggered adoption with ≥ 2 cohorts",
            "window horizon available per cohort (else dropped)",
        ],
        assumptions=[
            "Parallel trends within each sub-experiment",
            "No anticipation within window",
            "SUTVA",
        ],
        failure_modes=[
            FailureMode(
                symptom="No clean controls for the latest cohort",
                exception="DataInsufficient",
                remedy="Late cohort's sub-experiment is dropped; check "
                       "coverage in model_info. Consider sp.callaway_santanna.",
                alternative="callaway_santanna",
            ),
        ],
        alternatives=["callaway_santanna", "sun_abraham", "did_imputation"],
        typical_n_min=100,
    ))

    register(FunctionSpec(
        name="event_study",
        category="causal",
        description=(
            "Traditional OLS event-study with entity and time FEs. Generates "
            "relative-time dummies around the treatment date, omits a "
            "reference period, and estimates via TWFE + optional clustered "
            "SE. Exposed for users who want the classical specification "
            "alongside CS / SA / BJS; not robust to staggered-effect "
            "heterogeneity — use sp.sun_abraham for that."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat_time", "str", True,
                      description="First-treatment period column"),
            ParamSpec("time", "str", True),
            ParamSpec("unit", "str", True, description="Unit identifier"),
            ParamSpec("window", "tuple", False, (-4, 4),
                      "(lead, lag) horizons"),
            ParamSpec("ref_period", "int", False, -1,
                      "Reference relative-time period to omit"),
            ParamSpec("covariates", "list", False, None),
            ParamSpec("cluster", "str", False, None),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult with event_study DataFrame",
        example=(
            'sp.event_study(df, y="y", treat_time="g", time="t", unit="i", '
            'window=(-3, 3))'
        ),
        tags=["did", "event_study", "twfe", "ols", "lead_lag"],
        reference="Standard event-study; Roth (2022) on pre-test bias; "
                  "Sun & Abraham (2021) on dynamic contamination.",
        pre_conditions=[
            "panel with unit × time × outcome",
            "treat_time column gives first-treatment period (or 0/NaN)",
        ],
        assumptions=[
            "Parallel trends across event time",
            "No anticipation beyond window lead",
            "SUTVA",
        ],
        failure_modes=[
            FailureMode(
                symptom="Staggered heterogeneity — TWFE event-study biased",
                exception="AssumptionWarning",
                remedy="Use sp.sun_abraham for contamination-robust "
                       "event-study coefficients.",
                alternative="sun_abraham",
            ),
        ],
        alternatives=["sun_abraham", "callaway_santanna", "did_imputation"],
        typical_n_min=50,
    ))

    register(FunctionSpec(
        name="did_analysis",
        category="causal",
        description=(
            "Workflow wrapper that runs a full DiD pipeline: auto-detects "
            "2×2 vs. staggered, runs the right estimator (CS by default), "
            "optionally runs Bacon decomposition, event study, and "
            "Rambachan-Roth sensitivity, and aggregates into a "
            "DIDAnalysis report object."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True,
                      description="Binary treatment or first-treat column"),
            ParamSpec("time", "str", True),
            ParamSpec("id", "str", True),
            ParamSpec("method", "str", False, "auto",
                      "Estimator selection",
                      ["auto", "2x2", "cs", "sa", "sdid"]),
            ParamSpec("run_bacon", "bool", False, True),
            ParamSpec("run_event_study", "bool", False, True),
            ParamSpec("run_sensitivity", "bool", False, True),
        ],
        returns="DIDAnalysis",
        example=(
            'sp.did_analysis(df, y="earnings", treat="first_treat", '
            'time="year", id="worker")'
        ),
        tags=["did", "workflow", "analysis", "bacon", "event_study",
              "sensitivity"],
        reference=("DiD workflow synthesis; Roth, Sant'Anna, Bilinski & Poe "
                   "(2023) survey cited throughout [待核验 — bib key for "
                   "the 2023 DiD-what's-trending survey not yet added to "
                   "paper.bib]."),
        alternatives=["callaway_santanna", "harvest_did"],
        typical_n_min=50,
    ))

    register(FunctionSpec(
        name="harvest_did",
        category="causal",
        description=(
            "Harvest every valid 2×2 DID comparison from a staggered panel "
            "and aggregate them via precision-weighted / simple / "
            "cohort-weighted averages. Agnostic to cohort structure; "
            "useful for robustness comparisons against CS / SA / BJS."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("unit", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("outcome", "str", True),
            ParamSpec("treat", "str", False, None,
                      "Time-varying treat indicator (for dynamic harvesting)"),
            ParamSpec("cohort", "str", False, None,
                      "First-treat cohort column (for static harvesting)"),
            ParamSpec("never_value", "Any", False, 0),
            ParamSpec("horizons", "list", False, None),
            ParamSpec("reference", "str", False, "pre",
                      "Reference period convention",
                      ["pre", "first_treat"]),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult with all 2x2 comparisons in detail",
        example=(
            'sp.harvest_did(df, unit="i", time="t", outcome="y", '
            'cohort="g_first")'
        ),
        tags=["did", "harvest", "2x2", "aggregation", "staggered",
              "event_study"],
        reference="Synthesis of Goodman-Bacon (2021) + CS (2021) + "
                  "precision-weighted DiD; see docs/guides/harvest_did.md.",
        alternatives=["callaway_santanna", "bacon_decomposition",
                      "did_analysis"],
        typical_n_min=100,
    ))

    register(FunctionSpec(
        name="overlap_weighted_did",
        category="causal",
        description=(
            "Overlap-weighted 2×2 DiD. Weights observations by "
            "e(X)(1-e(X)), where e(X) is the estimated propensity score, "
            "placing highest weight on units with the most overlap between "
            "treated and control covariate distributions. Useful when "
            "overlap is poor at the tails."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True,
                      description="Binary treatment indicator"),
            ParamSpec("time", "str", True),
            ParamSpec("covariates", "list", True, description="Covariates X"),
            ParamSpec("ps_model", "str", False, "logit",
                      "Propensity score model",
                      ["logit", "rf", "gbm", "dl"]),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult",
        example=(
            'sp.overlap_weighted_did(df, y="y", treat="d", time="t", '
            'covariates=["age","edu"])'
        ),
        tags=["did", "overlap", "propensity", "weighted", "causal"],
        reference="Li, Morgan & Zaslavsky (2018) JASA on overlap weights; "
                  "applied to DiD by several authors [待核验 — specific "
                  "citation to be confirmed].",
        pre_conditions=[
            "2 periods, binary treat",
            "covariates with variation",
        ],
        assumptions=[
            "Overlap weights target the sub-population with positive overlap",
            "Correct PS model OR outcome model for DR variant",
        ],
        alternatives=["drdid", "did_2x2"],
        typical_n_min=200,
    ))

    register(FunctionSpec(
        name="cohort_anchored_event_study",
        category="causal",
        description=(
            "Cohort-anchored event study. Instead of averaging across "
            "cohorts at each relative-time bin (which can contaminate "
            "leads / lags with other cohorts' dynamics), estimates "
            "separate event-study paths per cohort and then aggregates "
            "with cohort weights. Successor to the Rambachan-Roth "
            "sensitivity-friendly specification."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("id", "str", True),
            ParamSpec("leads", "int", False, 3),
            ParamSpec("lags", "int", False, 5),
            ParamSpec("cluster", "str", False, None),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult with cohort-specific event-study paths",
        example=(
            'sp.cohort_anchored_event_study(df, y="y", treat="d", '
            'time="t", id="i", leads=3, lags=5)'
        ),
        tags=["did", "event_study", "cohort_anchored", "staggered", "causal"],
        reference="Rambachan & Roth (2023) ReStud "
                  "[@rambachan2023more]; design-robust extensions [待核验 "
                  "— specific citation to be confirmed].",
        alternatives=["sun_abraham", "callaway_santanna", "event_study"],
        typical_n_min=80,
    ))

    register(FunctionSpec(
        name="design_robust_event_study",
        category="causal",
        description=(
            "Design-robust event study with explicit negative-weight "
            "diagnostics per cohort × relative-time cell. Reports which "
            "event-study coefficients receive negative weights in TWFE "
            "and flags the affected horizons."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("id", "str", True),
            ParamSpec("leads", "int", False, 3),
            ParamSpec("lags", "int", False, 5),
            ParamSpec("cluster", "str", False, None),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult with weight diagnostics in model_info",
        example=(
            'sp.design_robust_event_study(df, y="y", treat="d", '
            'time="t", id="i")'
        ),
        tags=["did", "event_study", "design_robust", "negative_weights",
              "diagnostic"],
        reference="de Chaisemartin & D'Haultfœuille (2020) negative-weight "
                  "diagnostic [@dechaisemartin2020two]; design-robust "
                  "specifications [待核验 — specific citation to be "
                  "confirmed].",
        alternatives=["sun_abraham", "bacon_decomposition",
                      "cohort_anchored_event_study"],
        typical_n_min=80,
    ))

    register(FunctionSpec(
        name="did_misclassified",
        category="causal",
        description=(
            "Staggered DiD robust to treatment-timing misclassification "
            "and anticipation. Adjusts the CS-style aggregation for a "
            "user-supplied misclassification probability pi_misclass and "
            "a known anticipation horizon. Use when first-treat dates "
            "are noisy (e.g., survey-reported)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True,
                      description="First-treatment period (possibly noisy)"),
            ParamSpec("time", "str", True),
            ParamSpec("id", "str", True),
            ParamSpec("pi_misclass", "float", False, 0.0,
                      "P(observed treat ≠ true treat) — between 0 and 1"),
            ParamSpec("anticipation_periods", "int", False, 0),
            ParamSpec("cluster", "str", False, None),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult adjusted for misclassification / anticipation",
        example=(
            'sp.did_misclassified(df, y="y", treat="d", time="t", id="i", '
            'pi_misclass=0.05, anticipation_periods=1)'
        ),
        tags=["did", "measurement_error", "anticipation", "robustness",
              "staggered"],
        reference=("Measurement-error DiD literature — adjustment follows "
                   "a standard attenuation-correction pattern; specific "
                   "citation [待核验]."),
        pre_conditions=[
            "pi_misclass is between 0 and 0.5 (else identification flips)",
            "Known anticipation horizon",
        ],
        alternatives=["callaway_santanna"],
        typical_n_min=200,
    ))

    register(FunctionSpec(
        name="did_multiplegt_dyn",
        category="causal",
        description=(
            "dCDH (2024) intertemporal event-study DiD (MVP — see "
            "docs/rfc/multiplegt_dyn.md). At each horizon l ∈ {-placebo, "
            "..., dynamic}, compares Y_{F+l} − Y_{F-1} between units "
            "first switching at F and a not-yet-treated or never-treated "
            "control set held stable across the horizon. **MVP caveats**: "
            "analytical influence-function variance [待核验] is not yet "
            "implemented (SE via cluster bootstrap); switch-off events "
            "are ignored; heteroskedastic-weights variant pending."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("group", "str", True, description="Unit identifier"),
            ParamSpec("time", "str", True),
            ParamSpec("treatment", "str", True,
                      description="Binary treatment (0/1), switch-on only in MVP"),
            ParamSpec("placebo", "int", False, 0,
                      "Number of pre-treatment placebo horizons"),
            ParamSpec("dynamic", "int", False, 3,
                      "Number of post-treatment dynamic horizons"),
            ParamSpec("control", "str", False, "not_yet_treated",
                      "Control group", ["not_yet_treated", "never_treated"]),
            ParamSpec("cluster", "str", False, None,
                      "Cluster column (defaults to group)"),
            ParamSpec("n_boot", "int", False, 500),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("seed", "int", False, None),
        ],
        returns=(
            "CausalResult with event_study in model_info + joint placebo "
            "and overall Wald tests + [待核验] MVP warning"
        ),
        example=(
            'sp.did_multiplegt_dyn(df, y="y", group="i", time="t", '
            'treatment="d", placebo=2, dynamic=4)'
        ),
        tags=["did", "dcdh", "dynamic", "event_study", "intertemporal",
              "mvp", "causal"],
        reference=("de Chaisemartin & D'Haultfœuille (2024) "
                   "[@dechaisemartin2024difference]; DOI "
                   "10.1162/rest_a_01414; paper-parity pending "
                   "(see docs/rfc/multiplegt_dyn.md)."),
        pre_conditions=[
            "long-format panel with binary time-varying treatment",
            "at least some units switching on from d=0 to d=1",
            "enough horizons pre/post to compute long differences",
        ],
        assumptions=[
            "Parallel trends between switchers and controls per horizon",
            "No anticipation prior to F",
            "Stable control treatment across horizon window",
            "SUTVA",
            "[待核验] MVP omits the paper's analytical IF variance",
        ],
        failure_modes=[
            FailureMode(
                symptom="No units switch from 0 to 1",
                exception="",
                remedy="did_multiplegt_dyn identifies from switch-on events. "
                       "Use sp.callaway_santanna if design is standard "
                       "staggered adoption.",
                alternative="callaway_santanna",
            ),
            FailureMode(
                symptom="Joint placebo test rejects",
                exception="AssumptionViolation",
                remedy="Parallel trends unlikely; inspect event_study, "
                       "consider sp.honest_did sensitivity.",
                alternative="honest_did",
            ),
            FailureMode(
                symptom="Switch-off events present",
                exception="",
                remedy="MVP silently ignores switch-off events. For full "
                       "treatment-reversal handling, wait for paper-parity "
                       "implementation or use sp.did_multiplegt (2020 DID_M).",
                alternative="did_multiplegt",
            ),
        ],
        alternatives=["did_multiplegt", "callaway_santanna", "sun_abraham",
                      "did_imputation", "lp_did"],
        typical_n_min=100,
    ))

    register(FunctionSpec(
        name="did_timevarying_covariates",
        category="causal",
        description=(
            "DiD with time-varying covariates frozen at baseline (Caetano, "
            "Callaway, Payne & Rodrigues 2022 [待核验]). Avoids the "
            "bad-controls bias that arises when treatment affects the "
            "covariates: freezes X at period g + baseline_offset (default "
            "g-1) per cohort and uses the frozen values as controls in a "
            "per-(g, t) outcome-regression DiD. Aggregates via cohort-size "
            "weights."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("unit", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("cohort", "str", True,
                      description="First-treatment period (never_value = never-treated)"),
            ParamSpec("covariates", "list", True,
                      description="Time-varying covariates to freeze at baseline"),
            ParamSpec("never_value", "Any", False, 0),
            ParamSpec("baseline_offset", "int", False, -1,
                      "Offset relative to first-treatment period for freezing"),
            ParamSpec("n_boot", "int", False, 500),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("seed", "int", False, None),
        ],
        returns="CausalResult with per-(g, t) decomposition in detail",
        example=(
            'sp.did_timevarying_covariates(df, y="earnings", unit="i", '
            'time="year", cohort="g", covariates=["age","prior_wage"])'
        ),
        tags=["did", "timevarying", "covariates", "bad_controls",
              "staggered", "causal"],
        reference=("Caetano, Callaway, Payne & Rodrigues (2022) — exact "
                   "paper version + DOI [待核验 — not yet in paper.bib]."),
        pre_conditions=[
            "staggered adoption with ≥ 1 never-treated unit",
            "covariates column(s) exist for the baseline period per cohort",
            "integer-valued time column",
        ],
        assumptions=[
            "Conditional parallel trends given frozen baseline X",
            "No anticipation",
            "SUTVA",
        ],
        failure_modes=[
            FailureMode(
                symptom="No observation at baseline period for some units",
                exception="",
                remedy="Fallback uses the first observed period; review "
                       "detail coverage.",
                alternative="",
            ),
            FailureMode(
                symptom="Covariate measured with error or missing",
                exception="",
                remedy="Impute (sp.mice_impute) or restrict to a complete "
                       "sub-sample before calling.",
                alternative="",
            ),
        ],
        alternatives=["callaway_santanna", "drdid", "wooldridge_did"],
        typical_n_min=150,
    ))

    register(FunctionSpec(
        name="ddd_heterogeneous",
        category="causal",
        description=(
            "Heterogeneity-robust triple differences (DDD) for staggered "
            "adoption. Decomposes DDD into per-(cohort, time) cells via a "
            "Callaway-Sant'Anna-style aggregation, with the unaffected "
            "subgroup's DID as a placebo. Avoids the negative-weight issue "
            "that textbook TWFE DDD inherits from TWFE DID (Goodman-Bacon "
            "2021 analogue)."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("unit", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("cohort", "str", True,
                      description="First-treatment period (never_value = never-treated)"),
            ParamSpec("subgroup", "str", True,
                      description="Binary within-group subgroup indicator (1=affected, 0=placebo)"),
            ParamSpec("never_value", "Any", False, 0,
                      description="Value in cohort for never-treated units"),
            ParamSpec("n_boot", "int", False, 500),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("seed", "int", False, None),
        ],
        returns=("CausalResult with per-(g, t) decomposition in detail + "
                 "placebo joint test in model_info"),
        example=(
            'sp.ddd_heterogeneous(df, y="y", unit="i", time="t", '
            'cohort="g_first", subgroup="eligible")'
        ),
        tags=["did", "ddd", "triple", "heterogeneity", "staggered", "causal"],
        reference=("Olden & Møen (2022) *The Econometrics Journal* "
                   "[@olden2022triple]; Strezhnev (2023) working paper "
                   "[待核验 — bib key not yet added to paper.bib]; "
                   "Callaway & Sant'Anna (2021) [@callaway2021difference] "
                   "for the group-time aggregation template."),
        pre_conditions=[
            "staggered adoption panel with ≥ 1 never-treated unit",
            "binary within-group subgroup (affected vs unaffected)",
            "≥ 1 pre-treatment period per cohort",
        ],
        assumptions=[
            "Parallel trends relaxed to: same differential trend across "
            "treated vs never-treated, within both affected and unaffected "
            "subgroups",
            "No anticipation",
            "SUTVA",
        ],
        failure_modes=[
            FailureMode(
                symptom="No never-treated units",
                exception="ValueError",
                remedy=("First-cut implementation requires never-treated "
                        "controls; not-yet-treated variant is on the roadmap."),
                alternative="ddd",
            ),
            FailureMode(
                symptom=("placebo_joint_test rejects — DDD parallel-trends "
                         "assumption violated"),
                exception="AssumptionViolation",
                remedy=("Inspect per-(g, t) did_placebo values in the "
                        "detail DataFrame; add controls or apply "
                        "sp.honest_did sensitivity to the affected arm."),
                alternative="honest_did",
            ),
        ],
        alternatives=["ddd", "callaway_santanna", "wooldridge_did"],
        typical_n_min=200,
    ))

    register(FunctionSpec(
        name="lp_did",
        category="causal",
        description=(
            "Local-Projections DiD (Dube-Girardi-Jordà-Taylor 2023). At each "
            "event-time horizon h ∈ {-P, ..., H}, runs a separate OLS of "
            "Y_{t+h} − Y_{t-1} on the treatment change Δd_{t} with time FE "
            "and cluster-robust SE, using 'not-yet-treated' or 'never-treated' "
            "units as controls. Event-study β_h paths are returned in "
            "``model_info['event_study']``."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("unit", "str", True, description="Unit identifier"),
            ParamSpec("time", "str", True,
                      description="Integer period (consecutive)"),
            ParamSpec("treatment", "str", True,
                      description="Binary time-varying treatment (0/1)"),
            ParamSpec("horizons", "tuple", False, (-3, 5),
                      "(min, max) event-time horizons to estimate"),
            ParamSpec("controls", "list", False, None),
            ParamSpec("clean_controls", "str", False, "not_yet_treated",
                      "Control selection",
                      ["not_yet_treated", "never_treated"]),
            ParamSpec("time_fe", "bool", False, True),
            ParamSpec("cluster", "str", False, None,
                      "Cluster variable (defaults to unit)"),
            ParamSpec("alpha", "float", False, 0.05),
        ],
        returns="CausalResult with event-study in model_info['event_study']",
        example=(
            'sp.lp_did(df, y="y", unit="i", time="t", treatment="d", '
            'horizons=(-3, 5))'
        ),
        tags=["did", "lp_did", "local_projections", "event_study",
              "staggered", "causal"],
        reference=("Dube, Girardi, Jordà & Taylor (2023) NBER Working Paper "
                   "[待核验 — NBER WP ID + arXiv ID not yet added to "
                   "paper.bib]. Jordà (2005) AER on local projections more "
                   "broadly."),
        pre_conditions=[
            "long-format panel with consecutive integer time",
            "treatment is binary 0/1 and time-varying",
            "horizons feasible: enough periods for Y_{t-1} and Y_{t+H}",
        ],
        assumptions=[
            "Parallel trends across event time (standard DiD)",
            "No anticipation within the pre-treatment horizon",
            "SUTVA",
            "Stable treatment across the clean-control window",
        ],
        failure_modes=[
            FailureMode(
                symptom=("Horizon-0 n_obs is tiny because few units switch "
                         "on in the clean-control window"),
                exception="DataInsufficient",
                remedy=("Widen clean_controls='never_treated' → "
                        "'not_yet_treated' or shorten horizons."),
                alternative="callaway_santanna",
            ),
            FailureMode(
                symptom=("Placebo CIs don't cover zero — parallel trends "
                         "suspect"),
                exception="AssumptionViolation",
                remedy=("Apply sp.honest_did to the event-study paths for "
                        "Rambachan-Roth sensitivity bounds."),
                alternative="honest_did",
            ),
        ],
        alternatives=["callaway_santanna", "sun_abraham", "did_imputation",
                      "gardner_did"],
        typical_n_min=100,
    ))

    register(FunctionSpec(
        name="did_bcf",
        category="causal",
        description=(
            "Bayesian Causal Forests DiD. Fits a BART-style ensemble with "
            "treatment and prognostic terms on the DiD residuals, "
            "providing heterogeneous treatment-effect posterior draws per "
            "unit. Useful for machine-learning DiD with covariates."
        ),
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True),
            ParamSpec("treat", "str", True),
            ParamSpec("time", "str", True),
            ParamSpec("id", "str", True),
            ParamSpec("covariates", "list", False, None),
            ParamSpec("n_trees", "int", False, 200),
            ParamSpec("alpha", "float", False, 0.05),
            ParamSpec("seed", "int", False, None),
        ],
        returns=("CausalResult with posterior ATT + unit-level CATE "
                 "posterior draws in model_info"),
        example='sp.did_bcf(df, y="y", treat="d", time="t", id="i")',
        tags=["did", "bcf", "bart", "bayesian", "heterogeneous", "causal"],
        reference=("Hahn, Murray & Carvalho (2020) BCF prior; applied to "
                   "DiD by multiple authors — specific citation [待核验]."),
        alternatives=["did_imputation", "drdid"],
        typical_n_min=200,
    ))

    _BASE_REGISTRY_BUILT = True


# ====================================================================== #
#  Auto-registration from statspai.__all__
# ====================================================================== #
#
# Hand-written specs above cover ~41 canonical estimators.  The package
# exposes several hundred more symbols via ``statspai.__all__``.  The
# auto-registration pass below ensures sp.help() / sp.list_functions()
# / sp.search_functions() can still surface those names, using
# inspect.signature + docstring as a lightweight fallback spec.
#
# Design rules
# ------------
# * Never overwrite a hand-written entry.
# * Extract params from ``inspect.signature``; default="required" when no
#   default is set.  Type hints are stringified best-effort.
# * First non-empty docstring line becomes the description; fall back to
#   "(no description)".
# * Category comes from the object's ``__module__`` via the help module's
#   prefix table.
# * Idempotent: a sentinel flag prevents re-scanning on repeat calls.

_FULL_REGISTRY_BUILT = False


def _stringify_annotation(ann: Any) -> str:
    if ann is inspect._empty:
        return "Any"
    if isinstance(ann, str):
        return ann
    if hasattr(ann, "__name__"):
        return ann.__name__
    return str(ann).replace("typing.", "")


def _first_doc_line(doc: Optional[str]) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _auto_spec_from_callable(name: str, obj: Any) -> Optional[FunctionSpec]:
    """Build a minimal FunctionSpec by introspecting a callable.

    Returns None if introspection fails (e.g. C-extension without sig).
    """
    from .help import _infer_category  # lazy to avoid cycle

    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        sig = None

    params: List[ParamSpec] = []
    if sig is not None:
        for p in sig.parameters.values():
            if p.name == "self" or p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            required = p.default is inspect._empty
            default = None if required else p.default
            params.append(ParamSpec(
                name=p.name,
                type=_stringify_annotation(p.annotation),
                required=required,
                default=default,
                description="",
            ))

    doc = inspect.getdoc(obj) or ""
    desc = _first_doc_line(doc) or f"({name} — no description)"
    category = _infer_category(obj)
    return FunctionSpec(
        name=name,
        category=category,
        description=desc,
        params=params,
        returns="",
        example="",
        tags=[],
    )


def _ensure_full_registry() -> None:
    """Populate the registry with hand-written specs + auto-registered tail.

    Idempotent.  Call this from any entry point that needs *complete*
    coverage (sp.help(), sp.list_functions() without filter, etc.).
    """
    global _FULL_REGISTRY_BUILT
    _build_registry()
    if _FULL_REGISTRY_BUILT:
        return

    import statspai as _sp  # safe: called post-import from user code

    exported = getattr(_sp, "__all__", None) or dir(_sp)
    for name in exported:
        if name in _REGISTRY:
            continue
        obj = getattr(_sp, name, None)
        if obj is None:
            continue
        # Skip submodules — the help system treats those separately.
        if inspect.ismodule(obj):
            continue
        # Skip non-callables that aren't classes (e.g. constants).
        if not (inspect.isfunction(obj) or inspect.isclass(obj)
                or inspect.isbuiltin(obj) or inspect.ismethod(obj)
                or callable(obj)):
            continue
        spec = _auto_spec_from_callable(name, obj)
        if spec is not None:
            _REGISTRY[name] = spec

    _FULL_REGISTRY_BUILT = True


# ====================================================================== #
#  Public query API
# ====================================================================== #

def list_functions(category: Optional[str] = None) -> List[str]:
    """
    List all registered StatsPAI functions, optionally filtered by category.

    Auto-registers every function in ``statspai.__all__`` on first call
    (hand-written specs take precedence), so coverage is the full public
    surface — not just the 41 canonical estimators.
    """
    _ensure_full_registry()
    if category:
        return [k for k, v in _REGISTRY.items() if v.category == category]
    return list(_REGISTRY.keys())


def describe_function(name: str) -> Dict[str, Any]:
    """
    Return the full specification for a function as a dictionary.

    >>> sp.describe_function('did')
    {'name': 'did', 'category': 'causal', ...}
    """
    _ensure_full_registry()
    if name not in _REGISTRY:
        # Keep error message compact — full registry may contain 200+ names.
        hand_written = sorted(
            k for k, v in _REGISTRY.items() if not getattr(v, "_auto", False)
        )
        hint = ", ".join(hand_written[:15]) + ", ..."
        raise KeyError(f"Unknown function '{name}'. Examples: {hint}")
    return _REGISTRY[name].to_dict()


def function_schema(name: str) -> Dict[str, Any]:
    """
    Return an OpenAI function-calling compatible JSON schema.

    Useful for LLM tool-use / agent integrations.

    >>> schema = sp.function_schema('regress')
    >>> # Feed to OpenAI's function_call or Anthropic's tool_use
    """
    _ensure_full_registry()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown function '{name}'")
    return _REGISTRY[name].to_openai_schema()


def search_functions(query: str) -> List[Dict[str, str]]:
    """
    Keyword search across function names, descriptions, and tags.

    All query words must appear (AND logic), but not necessarily as a
    contiguous substring. This matches "panel data" against a function
    whose description contains "panel" and "data" separately.

    Returns a list of ``{'name': ..., 'description': ..., 'category': ...}``,
    sorted by relevance (number of word hits).

    >>> sp.search_functions('treatment effect')
    [{'name': 'did', ...}, {'name': 'dml', ...}, ...]
    """
    _ensure_full_registry()
    words = query.lower().split()
    if not words:
        return []

    scored = []
    for spec in _REGISTRY.values():
        text = f"{spec.name} {spec.description} {' '.join(spec.tags)}".lower()
        # All words must appear
        if all(w in text for w in words):
            # Score: count total word occurrences for ranking
            score = sum(text.count(w) for w in words)
            scored.append((score, {
                "name": spec.name,
                "description": spec.description,
                "category": spec.category,
            }))

    # Sort by score descending (most relevant first)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]


def all_schemas() -> List[Dict[str, Any]]:
    """
    Export all function schemas at once (for bulk agent tool registration).

    >>> schemas = sp.all_schemas()
    >>> # Register all as tools in your LLM framework
    """
    _ensure_full_registry()
    return [spec.to_openai_schema() for spec in _REGISTRY.values()]


def agent_card(name: str) -> Dict[str, Any]:
    """Return the agent-native metadata card for a function.

    Unlike :func:`function_schema` (OpenAI tool-call signature only),
    this includes identifying assumptions, pre-conditions, failure
    modes with recovery hints, ranked alternative functions, and
    the typical minimum sample size. It's the payload an agent should
    inspect *before* calling the function, and the payload rendered
    into each guide's ``## For Agents`` block.

    >>> card = sp.agent_card('did')
    >>> [a for a in card['assumptions']]
    ['Parallel trends', 'No anticipation', 'SUTVA', ...]
    """
    _ensure_full_registry()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown function '{name}'")
    return _REGISTRY[name].agent_card()


def agent_cards(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Bulk export of agent cards, optionally filtered by category.

    Only entries with at least one agent-native field populated are
    returned — auto-registered specs without assumptions / failure
    modes are skipped to keep the output signal-dense.

    >>> cards = sp.agent_cards(category='causal')
    >>> # Feed to an agent's tool catalog or doc generator
    """
    _ensure_full_registry()
    out: List[Dict[str, Any]] = []
    for spec in _REGISTRY.values():
        if category and spec.category != category:
            continue
        if not (spec.assumptions or spec.failure_modes
                or spec.alternatives or spec.pre_conditions
                or spec.typical_n_min):
            continue
        out.append(spec.agent_card())
    return out
