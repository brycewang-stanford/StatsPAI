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
class FunctionSpec:
    """Machine-readable specification for a StatsPAI function."""
    name: str
    category: str
    description: str
    params: List[ParamSpec] = field(default_factory=list)
    returns: str = ""
    example: str = ""
    tags: List[str] = field(default_factory=list)
    reference: str = ""  # paper / method reference

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


# ====================================================================== #
#  Registry
# ====================================================================== #

_REGISTRY: Dict[str, FunctionSpec] = {}


def register(spec: FunctionSpec) -> FunctionSpec:
    """Register a function specification."""
    _REGISTRY[spec.name] = spec
    return spec


def _build_registry():
    """Populate the registry with all public StatsPAI functions."""
    if _REGISTRY:
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
        reference="Chernozhukov et al. (2018)",
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
        reference="Athey, Tibshirani, Wager (2019)",
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
        reference="Kunzel et al. (2019)",
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
        reference="van der Laan & Rose (2011)",
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
            "Athey-Chetty-Imbens surrogate-index estimator for the "
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
            "Athey, Chetty, Imbens, Pollmann, Taubinsky (2019). NBER WP 26463."
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
        reference="Ghassami, Yang, Shpitser, Tchetgen Tchetgen (arXiv:2311.08527, 2024).",
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
            "Bareinboim-Pearl contextual causal bandit: pick the optimal "
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
        reference="Bareinboim & Pearl (NIPS 2015).",
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
        reference="Kallus, Mao, Uehara (arXiv:2502.13022, 2025).",
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
            "(Parmigiani et al. 2025)."
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
        reference="Parmigiani et al. (arXiv:2509.18484, 2025).",
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
        reference="Li, Ratkovic et al. (arXiv:2506.06615, 2025).",
    ))

    # -- Bayesian Double Machine Learning ---------------------------- #
    register(FunctionSpec(
        name="bayes_dml",
        category="bayes",
        description=(
            "Bayesian Double Machine Learning (Chernozhukov et al. 2025): "
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
        reference="Chernozhukov et al. (arXiv:2508.12688, 2025).",
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
        reference="Alessi, Zorzetto et al. (arXiv:2508.08418, 2025).",
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
        reference="Marchesin et al. (arXiv:2403.17852v3, 2025).",
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
            "bridge-function identification (Imbens-Kallus-Mao 2026)."
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
        reference="Imbens, Kallus, Mao (arXiv:2601.17712, 2026).",
    ))


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
