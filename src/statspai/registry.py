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

import json
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
        description="Double/Debiased Machine Learning for treatment effect estimation.",
        params=[
            ParamSpec("data", "DataFrame", True),
            ParamSpec("y", "str", True, description="Outcome"),
            ParamSpec("treatment", "str", True, description="Treatment variable"),
            ParamSpec("controls", "list", True, description="List of control variable names"),
            ParamSpec("n_folds", "int", False, 5, "Cross-fitting folds"),
        ],
        returns="EconometricResults",
        example='sp.dml(data=df, y="outcome", treatment="treat", controls=["x1","x2"])',
        tags=["dml", "ml", "causal", "semiparametric"],
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


# ====================================================================== #
#  Public query API
# ====================================================================== #

def list_functions(category: Optional[str] = None) -> List[str]:
    """
    List all registered StatsPAI functions, optionally filtered by category.

    Categories: regression, causal, panel, survey, output, diagnostics, robustness.
    """
    _build_registry()
    if category:
        return [k for k, v in _REGISTRY.items() if v.category == category]
    return list(_REGISTRY.keys())


def describe_function(name: str) -> Dict[str, Any]:
    """
    Return the full specification for a function as a dictionary.

    >>> sp.describe_function('did')
    {'name': 'did', 'category': 'causal', ...}
    """
    _build_registry()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown function '{name}'. Available: {available}")
    return _REGISTRY[name].to_dict()


def function_schema(name: str) -> Dict[str, Any]:
    """
    Return an OpenAI function-calling compatible JSON schema.

    Useful for LLM tool-use / agent integrations.

    >>> schema = sp.function_schema('regress')
    >>> # Feed to OpenAI's function_call or Anthropic's tool_use
    """
    _build_registry()
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
    _build_registry()
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
    _build_registry()
    return [spec.to_openai_schema() for spec in _REGISTRY.values()]
