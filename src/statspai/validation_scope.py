"""Configuration- and output-level validation scope for the validated core.

A registry tier (``certified`` / ``validated``) is attached to a *function*.
The evidence behind it is attached to *configurations* and *outputs*: a
Track A module runs one estimator variant, with one inference option, on
one kind of data, through one entry point, and compares some outputs (a
point estimate, a standard error, an interval, a diagnostic) but not
others. ``sp.dml`` being certified does not mean that the configuration a
user just ran -- say, the partially linear model with gradient-boosting
learners -- is the one the parity row exercised; and a row that pins the
2SLS coefficient and its classical, HC1 and CR1 standard errors says
nothing about the HC3 standard error of the same coefficient.

This module records, for the estimators listed in ``SCOPE_FUNCTIONS`` (the
paper's validation suite and the event-study estimators added since), which
configurations each artifact exercised and which outputs it compared. Every dimension has an enumerated domain; every row lists the
values it actually ran, never a wildcard. A dimension can be ignored only
for an output that provably does not depend on it (the 2SLS point
estimate does not depend on the covariance estimator), and the reason is
stored next to the exemption.

``sp.validation_scope(result)`` reads the configuration from a fitted
result and reports, per output, the strongest evidence that matches it:

* ``"reference"`` -- a deterministic known-truth (T1) or same-byte
  cross-language (T2) row ran this configuration and compared this output;
* ``"seed_equivalence"`` -- a seed-replicated equivalence test against a
  stated margin (T3);
* ``"stochastic_screen"`` -- a stochastic comparison within a Monte Carlo
  tolerance that is not an equivalence test (S);
* ``"coverage_simulation"`` -- a Monte Carlo coverage row on a known DGP (B);
* ``"disclosure"`` -- a documented disagreement or non-uniqueness (T4),
  which is a statement about the references, not evidence of correctness;
* ``"not_covered"``.

The overall ``status`` summarises the primary outputs (the point estimate
and its standard error, or the diagnostic for a test):

* ``"covered"`` -- every primary output has T1/T2 evidence;
* ``"estimate_only"`` -- the point estimate has T1/T2 evidence and some
  other primary output (typically the standard error) does not;
* ``"stochastic_only"`` -- no primary output has T1/T2 evidence, but some
  has T3, S or B evidence;
* ``"disclosure_only"`` -- the only matching rows are T4 disclosures;
* ``"not_covered"`` -- nothing matches.

Dimensions the fit did not record are reported under ``unchecked`` and
never match. Explicit queries with a value outside a dimension's domain
raise. Every artifact path is checked to exist, and every row is checked
against the configurations its artifact runs, by
``tests/test_validation_scope.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

__all__ = ["validation_scope", "ValidationScope", "SCOPE_FUNCTIONS"]

#: Evidence kinds, strongest first, and the per-output status each earns.
KINDS: Tuple[str, ...] = ("T1", "T2", "T3", "S", "B", "T4")
_STATUS_OF_KIND = {
    "T1": "reference",
    "T2": "reference",
    "T3": "seed_equivalence",
    "S": "stochastic_screen",
    "B": "coverage_simulation",
    "T4": "disclosure",
}
_OUTPUT_STATUS_ORDER = (
    "reference",
    "seed_equivalence",
    "stochastic_screen",
    "coverage_simulation",
    "disclosure",
    "not_covered",
)
#: ``vcov`` -- the joint covariance of a reported vector (every entry, not
#: only its diagonal): what joint tests, simultaneous bands and HonestDiD
#: consume. ``joint_test`` -- a Wald / F statistic and its reference
#: distribution. Neither is implied by an ``se`` row. ``interval`` -- the
#: p-values and confidence limits a fit reports from its SE, which pin the
#: critical value's distribution and degrees of freedom (an ``se`` row does
#: not; see the ``crit_dist`` dimension of the event-study maps).
OUTPUTS: Tuple[str, ...] = (
    "estimate",
    "se",
    "coverage",
    "diagnostic",
    "vcov",
    "joint_test",
    "interval",
)


def _vals(*values: str) -> FrozenSet[str]:
    return frozenset(values)


@dataclass(frozen=True)
class _Row:
    """One artifact, the configurations it ran, and the outputs it compared."""

    kind: str
    artifact: str  # repo-relative path
    config: Mapping[str, FrozenSet[str]]  # dimension -> values actually run
    outputs: Tuple[str, ...]  # outputs compared
    compares: str  # human-readable description
    entry_point: str  # the call the artifact makes


@dataclass(frozen=True)
class _Scope:
    function: str
    domains: Mapping[str, Tuple[str, ...]]  # dimension -> legal values
    extract: Callable[[Any], Dict[str, Optional[str]]]
    rows: Tuple[_Row, ...]
    primary: Tuple[str, ...] = ("estimate", "se")
    #: dimension -> (outputs that do not depend on it, reason)
    invariant: Mapping[str, Tuple[Tuple[str, ...], str]] = field(default_factory=dict)
    note: str = ""
    #: dimension -> value assumed when a by-name query omits it. Only for a
    #: dimension added after the map was first published, whose omitted value
    #: is what every earlier call ran; a fitted result always reports its own.
    #: A callable receives the rest of the query and returns the value (for a
    #: dimension whose default depends on the others).
    defaults: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dimensions(self) -> Tuple[str, ...]:
        return tuple(self.domains)

    def ignored(self, dim: str, output: str) -> bool:
        spec = self.invariant.get(dim)
        return spec is not None and output in spec[0]


def _mi(result: Any) -> Dict[str, Any]:
    return dict(getattr(result, "model_info", None) or {})


#: Boolean ``model_info`` flags to their domain labels; a missing flag maps
#: to None (typed ``Any`` keys because ``model_info`` values are untyped).
_TRUE_FALSE: Dict[Any, str] = {True: "true", False: "false"}
_SHARE_VARIANCE: Dict[Any, str] = {True: "estimated", False: "fixed"}
_HORIZON: Dict[Any, str] = {True: "horizons", False: "overall"}


def _lower(v: Any) -> Optional[str]:
    return None if v is None else str(v).lower()


def _set(v: Any) -> str:
    return "none" if not v else "set"


# --------------------------------------------------------------------------- #
#  Configuration extractors (read only what the fit recorded)
# --------------------------------------------------------------------------- #


def _vce_linear(mi: Mapping[str, Any]) -> Optional[str]:
    """The covariance estimator a regress / iv fit actually computed."""
    vtype = str(mi.get("vcov_type") or "").lower()
    if "wild" in vtype:
        return "wild"
    if "cr2" in vtype:
        return "cr2"
    if "cr3" in vtype:
        return "cr3"
    if "two-way" in vtype or "multiway" in vtype:
        return "cluster_multiway"
    if "conley" in vtype:
        return "conley"
    if "jackknife" in vtype:
        return "jackknife"
    cluster = mi.get("cluster")
    if cluster is not None:
        if isinstance(cluster, (list, tuple)) and len(cluster) > 1:
            return "cluster_multiway"
        return "cr1"
    robust = mi.get("robust")
    if isinstance(robust, dict):  # e.g. {"CRV1": "g"}
        return "cr1"
    robust = _lower(robust)
    if robust in {"nonrobust", None}:
        return "classical"
    return robust


def _x_regress(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    return {
        "vce": _vce_linear(mi),
        "weights": _set(mi.get("weights") or mi.get("weighted")),
    }


def _x_panel(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    vce = _lower(mi.get("vce_type"))
    if vce is None:
        if mi.get("cluster"):
            vce = "cluster"
        elif _lower(mi.get("robust")) == "robust":
            vce = "robust"
        elif _lower(mi.get("robust")) in (None, "nonrobust", "unadjusted"):
            vce = "unadjusted"
    return {
        "method": _lower(mi.get("method")),
        "vce": vce,
        "ssc": _lower(mi.get("ssc")) or "linearmodels",
        "weights": _set(mi.get("weights")),
    }


def _x_iv(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    model_type = (_lower(mi.get("model_type")) or "").replace("iv-", "")
    # the GMM label carries its step count ("IV-GMM (2-step)")
    model_type = model_type.split(" ")[0]
    estimator = model_type if model_type in {"2sls", "liml", "fuller", "gmm"} else None
    from .smart.audit import _overid_degree  # shared identification rule

    view = dict(mi)
    view.update(getattr(r, "diagnostics", None) or {})
    degree = _overid_degree(view)
    ident = None if degree is None else ("just" if degree == 0 else "over")
    absorbed = mi.get("absorb") or mi.get("absorbed_fe") or mi.get("absorb_vars")
    return {
        "estimator": estimator,
        "vce": _vce_linear(mi),
        "identification": ident,
        "absorb": _set(absorbed),
        # sp.iv reports ivregress, small statistics unless small=False
        "small": "false" if mi.get("small") is False else "true",
    }


def _x_feols(r: Any) -> Dict[str, Optional[str]]:
    weights = getattr(r, "weights", None)
    if weights is None:
        weights = getattr(r, "_weights", None)
    return {
        "vcov": _lower(getattr(r, "vcov_type", None)),
        "ssc": _lower(getattr(r, "ssc", None)),
        "weights": _set(weights is not None and weights is not False),
    }


def _cs_estimator(label: Any) -> Optional[str]:
    """The ATT(g, t) estimator behind a recorded label.

    The panel path records ``estimator.upper()``; the repeated cross-section
    path appends the engine (``"DR (RCS, Sant'Anna-Zhao)"``, ``"REG (RCS,
    cell-mean)"``, `` -- unbalanced panel``). ``'stdipw'`` is ``'ipw'`` under
    Stata's name (the same code path).
    """
    name = _lower(label)
    if name is None:
        return None
    name = name.split(" ", 1)[0]
    return "ipw" if name == "stdipw" else name


def _cs_sample(r: Any, mi: Mapping[str, Any]) -> Optional[str]:
    if mi.get("allow_unbalanced_panel"):
        return "unbalanced_panel"
    if mi.get("panel") is False:
        return "rcs"
    units, periods = mi.get("n_units"), mi.get("n_periods")
    n_obs = getattr(r, "n_obs", None)
    if mi.get("balance") == "pair" and units and periods and n_obs is not None:
        # balance='pair' on a panel with missing unit-periods: each cell uses
        # the units seen in both of its periods (csdid 1.8x's rule).
        if int(n_obs) < int(units) * int(periods):
            return "unbalanced_pair"
    return "panel"


def _x_cs(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    anticipation = mi.get("anticipation")
    return {
        "sample": _cs_sample(r, mi),
        "estimator": _cs_estimator(mi.get("estimator")),
        "control_group": _lower(mi.get("control_group")),
        # Consulted only under the not-yet-treated comparison set; under the
        # never-treated one every value runs the same estimator.
        "notyet_cutoff": (
            "period"
            if _lower(mi.get("control_group")) == "nevertreated"
            else _lower(mi.get("notyet_cutoff", "period"))
        ),
        "weights": "weighted" if mi.get("weighted") else "none",
        "covariates": None if "covariates" not in mi else _set(mi.get("covariates")),
        "inference": "bootstrap" if mi.get("bstrap") else "analytic",
        "base_period": _lower(mi.get("base_period")),
        "anticipation": (
            None if anticipation is None else ("0" if int(anticipation) == 0 else ">0")
        ),
        "clustering": _set(mi.get("clustervars")),
        "parity": (mi.get("conventions") or {}).get("parity") or "none",
        "variance_convention": _cs_variance_convention(mi),
    }


def _cs_query_variance_convention(cfg: Mapping[str, Optional[str]]) -> str:
    """The convention a default fit of a by-name query's configuration has."""
    relevant = (
        cfg.get("sample") in ("rcs", "unbalanced_panel")
        and cfg.get("estimator") == "dr"
        and cfg.get("covariates") == "set"
    )
    if not relevant:
        return "na"
    # parity='r' / 'stata' set the reference sign; the default is 'delta'
    return "reference" if cfg.get("parity") in ("r", "stata") else "delta"


def _cs_variance_convention(mi: Mapping[str, Any]) -> str:
    """``'delta'`` / ``'reference'`` where the choice changes the SE, else ``'na'``.

    The two influence functions differ only for the DR estimator with
    covariates on the repeated-cross-section path (``panel=False`` or an
    unbalanced panel with ``allow_unbalanced_panel=True``); elsewhere both
    give the same numbers, so evidence from either applies.
    """
    rcs_path = mi.get("panel") is False or bool(mi.get("allow_unbalanced_panel"))
    dr = _cs_estimator(mi.get("estimator")) == "dr"
    if not (rcs_path and dr and mi.get("covariates")):
        return "na"
    conv = mi.get("variance_convention") or (mi.get("conventions") or {}).get(
        "variance_convention"
    )
    return _lower(conv) or "delta"


def _crit_cell(mi: Mapping[str, Any]) -> Optional[str]:
    """The distribution behind a regression event study's p-values and CIs.

    ``model_info['conventions']`` records ``crit_dist`` (``'normal'`` or
    ``'t'``) and ``crit_df``; a t with ``G - 1`` degrees of freedom (``G``
    the recorded cluster count) is ``'t_cluster'``, any other t is
    ``'t_resid'`` (``N - K``).
    """
    conv = mi.get("conventions") or {}
    dist = conv.get("crit_dist")
    if dist in (None, "normal", "bootstrap"):
        return dist
    df, g = conv.get("crit_df"), mi.get("n_clusters")
    if df is not None and g is not None and int(df) == int(g) - 1:
        return "t_cluster"
    return "t_resid"


def _x_sa(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    # ``control_group`` is recorded as passed, and is ignored when
    # ``control_cohort=`` names the control; the label says which ran.
    label = str(mi.get("control_cohort") or "")
    control: Optional[str]
    if label == "nevertreated":
        control = "nevertreated"
    elif label.startswith("lastcohort"):
        control = "lastcohort"
    elif label:
        control = "control_cohort"
    else:
        control = _lower(mi.get("control_group"))
    return {
        "control_group": control,
        "aggregation": _lower(mi.get("summary_aggregation")),
        "share_variance": _SHARE_VARIANCE.get(mi.get("share_variance")),
        "window_rule": _lower(mi.get("window_rule")),
        "crit_dist": _crit_cell(mi),
    }


def _x_bjs(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    return {
        "fe": _lower(mi.get("y0_fe")),
        "covariates": _lower(mi.get("y0_covariates")),
        "vce": _lower(mi.get("vce")),
        "weights": _set(mi.get("weights")),
        "horizon": _HORIZON.get(mi.get("horizon_requested")),
    }


def _x_did2s(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    return {
        "vce": _lower(mi.get("vce")),
        "weights": _set(mi.get("weights")),
        "covariates": _lower(mi.get("covariates")),
        "event_study": "on" if mi.get("event_study") is not None else "off",
        # treat= / fe= / second_stage= build both stages differently
        "form": "general" if "first_stage_fe" in mi else "first_treat",
    }


def _x_twfe_es(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    dates = mi.get("n_adoption_dates")
    window = mi.get("window")
    ref = mi.get("ref_periods")
    return {
        "adoption": (
            None if dates is None else ("single_date" if dates == 1 else "staggered")
        ),
        "window": (
            None
            if window is None
            else ("-4_4" if tuple(window) == (-4, 4) else "other")
        ),
        "ref_period": None if ref is None else ("-1" if list(ref) == [-1] else "other"),
        "covariates": None if "covariates" not in mi else _set(mi.get("covariates")),
        "weights": _set(mi.get("weights")),
        "cluster": _lower(mi.get("cluster_level")),
        "bins": None if "bin_width" not in mi else _set(mi.get("bin_width")),
        "intensity": None if "intensity" not in mi else _set(mi.get("intensity")),
        "absorb": None if "absorb" not in mi else _set(mi.get("absorb")),
        "crit_dist": _crit_cell(mi),
    }


def _x_etwfe(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    family = mi.get("family")
    return {
        "family": (
            None if family is None else ("linear" if family == "linear" else "glm")
        ),
        "panel": _TRUE_FALSE.get(mi.get("panel")),
        "cgroup": _lower(mi.get("cgroup")),
        "controls": None if "controls" not in mi else _set(mi.get("controls")),
        "xvar": None if "xvar" not in mi else _set(mi.get("xvar")),
        "weights": _set(mi.get("weights")),
        "agg_weights": _lower(mi.get("agg_weights")),
    }


def _x_etwfe_glm(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    family = _lower(mi.get("family"))
    cgroup = _lower(mi.get("cgroup"))
    return {
        "family": "logit" if family == "binomial" else family,
        # the fe='unit' / hettype path records Stata's spelling
        "cgroup": "nevertreated" if cgroup == "never" else cgroup,
        "fe": _lower(mi.get("fe")),
        "hettype": _lower(mi.get("hettype")),
        "scale": _lower(mi.get("scale")),
        "response_se": _lower(mi.get("response_se")),
        "controls": None if "controls" not in mi else _set(mi.get("controls")),
        "xvar": None if "xvar" not in mi else _set(mi.get("xvar")),
        "weights": _set(mi.get("weights")),
    }


def _x_rd(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    vce: Optional[str]
    if mi.get("cluster") is not None:
        vce = "cluster"
    else:
        vce = _lower(mi.get("vce"))
    return {
        "design": _lower(mi.get("rd_type")),
        "bwselect": _lower(mi.get("bwselect")),
        "kernel": _lower(mi.get("kernel")),
        "p": None if mi.get("polynomial_p") is None else str(mi.get("polynomial_p")),
        "vce": vce,
        "covariates": None if "covariates" not in mi else _set(mi.get("covariates")),
        "code_path": "port" if mi.get("cct_delegation") else "native",
        "weights": None if "weighted" not in mi else _set(mi.get("weighted")),
    }


def _x_rddensity(r: Any) -> Dict[str, Optional[str]]:
    return {"code_path": _lower(_mi(r).get("backend"))}


def _x_synth(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    v = _lower(mi.get("v_method"))
    nonunique = mi.get("weight_solution_nonunique")
    return {
        "v_method": v,
        # The solver records whether near-optimal starts landed on different
        # donor-weight classes; certification holds only for identified fits.
        "weights": (
            None if nonunique is None else ("nonunique" if nonunique else "unique")
        ),
        "code_path": _lower(mi.get("backend")),
    }


def _x_sdid(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    # Only the treat= path records "design"; treated_unit= is always a block.
    via_treat = "design" in mi
    return {
        "method": _lower(mi.get("estimator")),
        "se_method": _lower(mi.get("se_method")),
        "code_path": _lower(mi.get("backend")),
        "interface": "treat" if via_treat else "treated_unit",
        "design": _lower(mi.get("design")) if via_treat else "block",
        "covariates": _lower(mi.get("covariate_method")) or "none",
    }


def _x_dcdh_dyn(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    horizons = list(mi.get("horizons") or [])
    n_placebo = sum(1 for h in horizons if h < 0)
    group, cluster = mi.get("group"), mi.get("cluster_var")
    balanced = mi.get("balanced_panel")
    flag: Dict[Any, str] = {True: "true", False: "false"}
    return {
        "se_method": _lower(mi.get("se_method")),
        "aggregation": _lower(mi.get("aggregation")),
        "control": _lower(mi.get("control")),
        "switchers": (
            None if "switchers" not in mi else (_lower(mi["switchers"]) or "both")
        ),
        "same_switchers": flag.get(mi.get("same_switchers")),
        "controls": None if "controls" not in mi else _set(mi.get("controls")),
        "trends_nonparam": (
            None if "trends_nonparam" not in mi else _set(mi.get("trends_nonparam"))
        ),
        "normalized": flag.get(mi.get("normalized")),
        "weights": _set(mi.get("weights")),
        "cluster": (
            None
            if group is None or cluster is None
            else ("group" if cluster == group else "coarser")
        ),
        "treatment": _lower(mi.get("treatment_kind")),
        "panel": (
            None if balanced is None else ("balanced" if balanced else "unbalanced")
        ),
        "placebo": None if not horizons else ("0" if n_placebo == 0 else ">0"),
    }


_LINEAR_LEARNERS = {"LinearRegression", "LogisticRegression"}


def _x_dml(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    learners_used = {str(mi.get("ml_g")), str(mi.get("ml_m"))}
    if mi.get("default_learners"):
        learners = "default"
    elif learners_used <= _LINEAR_LEARNERS:
        learners = "linear"
    else:
        learners = "other"
    n_rep = mi.get("n_rep")
    if "trimming_threshold" not in mi:
        ipw = "n/a"  # plr / pliv have no propensity weights
    else:
        trim = float(mi["trimming_threshold"])
        norm = bool(mi.get("normalize_ipw"))
        known = {(0.01, False): "trim0.01", (1e-12, False): "trim1e-12"}
        ipw = known.get((trim, norm), "other")
    return {
        "model": _lower(mi.get("dml_model")),
        "score": _lower(mi.get("score")),
        "learners": learners,
        "n_folds": None if mi.get("n_folds") is None else str(mi.get("n_folds")),
        "n_rep": None if n_rep is None else ("1" if int(n_rep) == 1 else ">1"),
        "ipw": ipw,
    }


#: grf's tuning defaults as they appear on a fitted ``CausalForest``.
_FOREST_DEFAULTS = {
    "min_samples_leaf": 5,
    "max_depth": None,
    "max_samples": 0.5,
    "honest": True,
    "honesty_fraction": 0.5,
    "honesty_prune_leaves": True,
    "mtry": None,
    "alpha": 0.05,
    "imbalance_penalty": 0.0,
    "stabilize_splits": True,
    "split_rule": "grf",
    "_user_model_y": False,
    "_user_model_t": False,
}


def _x_forest(r: Any) -> Dict[str, Optional[str]]:
    discrete = getattr(r, "discrete_treatment", None)
    n_trees = getattr(r, "n_estimators", None)
    missing = object()
    tuning: Optional[str] = "grf_defaults"
    for attr, default in _FOREST_DEFAULTS.items():
        got = getattr(r, attr, missing)
        if got is missing:
            tuning = None
            break
        if got != default:
            tuning = "custom"
            break
    if getattr(r, "fe", None) is not None:
        design = "fixed_effects"
    elif getattr(r, "_clusters", None) is not None:
        design = "clustered"
    else:
        design = "iid"
    if discrete is None:
        treatment = None
    else:
        values = getattr(r, "_treatment_values", None)
        binary = bool(discrete) and (values is None or len(values) == 2)
        treatment = "binary" if binary else "continuous"
    return {
        "treatment": treatment,
        "trees": None if n_trees is None else str(int(n_trees)),
        "tuning": tuning,
        "design": design,
    }


def _x_psm(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    return {
        "method": _lower(mi.get("method")),
        "distance": _lower(mi.get("distance")),
        "n_matches": None if mi.get("n_matches") is None else str(mi.get("n_matches")),
        "replace": _lower(mi.get("replace")),
        "caliper": "none" if mi.get("caliper") is None else "set",
        "bias_correction": _lower(mi.get("bias_correction")),
        "se_method": _lower(mi.get("se_method")),
    }


def _x_nnmatch(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    vce = _lower(mi.get("vce"))
    vce_nn = mi.get("vce_nn")
    return {
        "estimand": _lower(getattr(r, "estimand", None)),
        "metric": _lower(mi.get("metric")),
        "n_matches": None if mi.get("n_matches") is None else str(mi.get("n_matches")),
        "exact": None if "exact" not in mi else _set(mi.get("exact")),
        "bias_adjust": None if "bias_adjust" not in mi else _set(mi.get("bias_adjust")),
        # the iid variance has no neighbour count; 'na' keeps it one cell
        "vce": vce,
        "vce_nn": "na" if vce == "iid" else (None if vce_nn is None else str(vce_nn)),
        "caliper": "none" if mi.get("caliper") is None else "set",
    }


_HONEST_SOURCES = (
    "callaway_santanna",
    "sun_abraham",
    "event_study",
    "etwfe",
    "gardner_did",
    "did_imputation",
    "moments",
    "supplied",
    "other",
)


def _x_honest(r: Any) -> Dict[str, Optional[str]]:
    """``sp.honest_did`` returns a table; its configuration is in ``attrs``."""
    at = dict(getattr(r, "attrs", None) or {})
    alpha = at.get("alpha")
    source = at.get("source")
    return {
        "restriction": at.get("restriction"),
        "shape": at.get("shape", "none"),
        "interval": at.get("interval"),
        "code_path": at.get("code_path"),
        "target": at.get("target"),
        "window": at.get("window"),
        "alpha": None if alpha is None else ("0.05" if alpha == 0.05 else "other"),
        "m_grid": at.get("m_grid"),
        "rm_grid": at.get("rm_grid"),
        "source": (
            None
            if source is None
            else (source if source in _HONEST_SOURCES else "other")
        ),
    }


# --------------------------------------------------------------------------- #
#  The evidence map
# --------------------------------------------------------------------------- #

_R = "tests/r_parity/"
_RP = "tests/reference_parity/"
_B = "tests/coverage_monte_carlo/results_b1000/coverage_b1000.json"
_DML_PY = "tests/external_parity/test_dml_classifier_nuisance_and_irm_sensitivity.py"
_DML_IVT = "tests/test_dml_ivtype_reference.py"
_DML_OPT = "tests/reference_parity/test_dml_options_doubleml_parity.py"
_VCE = _RP + "test_vce_grammar_stata_parity.py"
_ATTACH = _RP + "test_validation_entry_points.py"
_JOINT = _RP + "test_joint_wald_stata_parity.py"
_REGW = _RP + "test_regress_vce_weights_stata_parity.py"
_MENU = _RP + "test_vce_menu_parity.py"
#: parity= / crit_dist= of the regression event studies and did_2x2
_ES_CRIT = _RP + "test_es_crit_parity.py"
_EST_SE = ("estimate", "se")
_EST = ("estimate",)
_COV = ("coverage",)

_REG_VCE = (
    "classical",
    "hc0",
    "hc1",
    "hc2",
    "hc3",
    "hc4",
    "cr1",
    "cr2",
    "cr3",
    "cluster_multiway",
    "hac",
    "wild",
    "conley",
    "jackknife",
)
_VCE_INVARIANT = (
    _EST,
    "the least-squares / k-class point estimate is computed before, and "
    "independently of, the covariance estimator",
)

SCOPES: Dict[str, _Scope] = {}


def _dml_cfg(
    model: str,
    scores: Tuple[str, ...],
    learners: str,
    n_folds: Tuple[str, ...],
    n_rep: str,
    ipw: str,
) -> Dict[str, FrozenSet[str]]:
    """One ``sp.dml`` row's configuration, every dimension spelled out."""
    return {
        "model": _vals(model),
        "score": _vals(*scores),
        "learners": _vals(learners),
        "n_folds": _vals(*n_folds),
        "n_rep": _vals(n_rep),
        "ipw": _vals(ipw),
    }


def _add(scope: _Scope) -> None:
    SCOPES[scope.function] = scope


_add(
    _Scope(
        "regress",
        {"vce": _REG_VCE, "weights": ("none", "set")},
        _x_regress,
        (
            _Row(
                "T2",
                _R + "01_ols.py",
                {"vce": _vals("hc1"), "weights": _vals("none")},
                _EST_SE,
                "coefficients and HC1 SEs vs lm+sandwich",
                "sp.regress(robust='hc1')",
            ),
            _Row(
                "T2",
                _VCE,
                {
                    "vce": _vals("classical", "hc1", "hc3", "cr1"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "coefficients and SEs vs Stata regress (ols, robust, hc3, cluster)",
                "sp.regress(vce=...)",
            ),
            _Row(
                "T2",
                _RP + "test_r2_postest_parity.py",
                {"vce": _vals("classical", "hc1", "cr1"), "weights": _vals("none")},
                ("joint_test",),
                "joint Wald F of the factor dummies (and its t(G - 1) / F(q, G - 1) "
                "reference law under clustering) vs Stata contrast's joint row",
                "sp.test(result, 'C(g)[T.2] = ... = 0')",
            ),
            _Row(
                "T2",
                _JOINT,
                {
                    "vce": _vals("classical", "hc1", "hc2", "hc3", "cr1"),
                    "weights": _vals("none"),
                },
                ("joint_test",),
                "Wald F, both degrees of freedom and p-value of three joint "
                "hypotheses vs Stata test / testparm after regress",
                "sp.regress(formula, data, robust=..., cluster=...)",
            ),
            _Row(
                "T2",
                _R + "55_hc2_hc3.py",
                {"vce": _vals("hc2", "hc3"), "weights": _vals("none")},
                _EST_SE,
                "HC2 / HC3 SEs vs sandwich::vcovHC",
                "sp.regress(robust='hc2'|'hc3')",
            ),
            _Row(
                "T2",
                _RP + "test_linear_model_extensions_parity.py",
                {"vce": _vals("hc4"), "weights": _vals("none", "set")},
                _EST_SE,
                "HC4 SEs vs sandwich::vcovHC(type = 'HC4'), unweighted and "
                "weighted, to 1e-9",
                "sp.regress(F_ALL, df, robust=...)",
            ),
            _Row(
                "T2",
                _R + "14_ols_cluster.py",
                {"vce": _vals("cr1"), "weights": _vals("none")},
                _EST_SE,
                "CR1 SEs vs sandwich::vcovCL",
                "sp.regress(cluster=...)",
            ),
            _Row(
                "T2",
                _R + "51_newey.py",
                {"vce": _vals("hac"), "weights": _vals("none")},
                _EST_SE,
                "Newey-West SEs vs sandwich::NeweyWest",
                "sp.regress(robust='hac')",
            ),
            _Row(
                "T2",
                _REGW,
                {
                    "vce": _vals("classical", "hc1", "hc2", "hc3", "cr1", "cr2"),
                    "weights": _vals("set"),
                },
                _EST_SE,
                "weighted coefficients and SEs vs Stata regress [aw=w] under "
                "ols, robust, hc2, hc3, cluster and vce(hc2 g), to 1e-10",
                "sp.regress(FORMULA, data, weights=..., vce=...)",
            ),
            _Row(
                "T2",
                _REGW,
                {"vce": _vals("cr2"), "weights": _vals("none")},
                _EST_SE,
                "CR2 SEs vs Stata regress, vce(hc2 g), to 1e-10",
                "sp.regress(FORMULA, data, vce='cr2', cluster=...)",
            ),
            _Row(
                "T1",
                _REGW,
                {"vce": _vals("cr3"), "weights": _vals("none", "set")},
                ("se",),
                "CR3 equals the brute-force delete-one-cluster jackknife "
                "sum_g (b_(g) - b)(b_(g) - b)' to 1e-10",
                "sp.regress(FORMULA, data, vce='cr3', cluster=...)",
            ),
            _Row(
                "T4",
                _REGW,
                {"vce": _vals("cr3"), "weights": _vals("none")},
                ("se",),
                "Stata's vce(jackknife, cluster(g) mse) is (G - 1)/G times "
                "this variance; equal to 1e-6 after that factor (Stata keeps "
                "its replicates in single precision)",
                "sp.regress(FORMULA, data, vce='cr3', cluster=...)",
            ),
            _Row(
                "T4",
                _REGW,
                {"vce": _vals("cluster_multiway"), "weights": _vals("none", "set")},
                ("se",),
                "two-way SEs use one G_min/(G_min - 1) factor; Stata 18 scales "
                "each component by its own. Both are rebuilt from the same "
                "scores to 1e-10, and sp.twoway_cluster gives Stata's number",
                "sp.regress(FORMULA, data, cluster=[a, b])",
            ),
            _Row(
                "B",
                _B,
                {"vce": _vals("hc1"), "weights": _vals("none")},
                _COV,
                "95% CI coverage on a known-truth RCT DGP",
                "sp.regress(robust='hc1')",
            ),
            _Row(
                "T2",
                _MENU,
                {
                    "vce": _vals(
                        "classical",
                        "hc0",
                        "hc1",
                        "hc2",
                        "hc3",
                        "hc4",
                        "cr1",
                        "cr3",
                        "hac",
                        "cluster_multiway",
                    ),
                    "weights": _vals("none", "set"),
                },
                _EST_SE,
                "coefficients and SEs vs R on lm, unweighted and weighted: "
                "sandwich vcovHC HC0-HC4, vcovCL, NeweyWest (4 lags, with and "
                "without adjust), clubSandwich CR3, and fixest two-way "
                "clustering at its default ssc() (one G_min factor, the "
                "convention sp.regress uses), to 1e-9 (observed 1e-13). "
                "Two-way only: sp.regress refuses three-way clustering",
                "sp.regress(F, data, weights=..., **opts)",
            ),
            _Row(
                "T2",
                _MENU,
                {"vce": _vals("cr2"), "weights": _vals("none")},
                _EST_SE,
                "CR2 SEs vs clubSandwich::vcovCR on lm, to 1e-9",
                "sp.regress(F, data, vce='cr2', cluster='g')",
            ),
            _Row(
                "T2",
                _MENU,
                {"vce": _vals("jackknife"), "weights": _vals("none", "set")},
                _EST_SE,
                "vce='jackknife' is the CR3 variance (a fit reports vce 'cr3'); "
                "equal to clubSandwich CR3, unweighted and weighted, to 1e-9",
                "sp.regress(F, data, vce='jackknife', cluster='g')",
            ),
            _Row(
                "T4",
                _MENU,
                {"vce": _vals("cr2"), "weights": _vals("set")},
                ("se",),
                "weighted CR2 has two unbiased adjustments: StatsPAI and Stata "
                "apply CR2 to the sqrt(w)-transformed design, clubSandwich "
                "(inverse_var=TRUE) uses the Pustejovsky-Tipton symmetric form; "
                "up to 0.53% apart here, both rebuilt from the fit to 1e-10 and "
                "both satisfying A_g M_g A_g' = W_g^-1",
                "sp.regress(F, data, vce='cr2', cluster='g', weights='w')",
            ),
            _Row(
                "T2",
                _MENU,
                {"vce": _vals("wild"), "weights": _vals("none")},
                _EST_SE,
                "the SE reported with the wild bootstrap is the CR1 one: vs "
                "sandwich::vcovCL and Stata regress, vce(cluster g), to 1e-9",
                "sp.regress(F, data, vce='wild', cluster='g')",
            ),
            _Row(
                "T2",
                _MENU,
                {"vce": _vals("wild"), "weights": _vals("none")},
                ("diagnostic",),
                "WCR bootstrap p-values, 12 clusters and all 2^12 Rademacher "
                "sign vectors enumerated: equal to fwildclusterboot::boottest "
                "and Stata boottest exactly",
                "sp.regress(F, data, vce='wild', cluster='g')",
            ),
            _Row(
                "T2",
                _MENU,
                {"vce": _vals("conley"), "weights": _vals("none")},
                _EST_SE,
                "Conley SEs (uniform kernel, 100 km) vs Stata acreg, to 1e-9",
                "sp.regress(F, data, vce='conley', ...)",
            ),
            _Row(
                "T2",
                _MENU,
                {"vce": _vals("hac"), "weights": _vals("none", "set")},
                _EST_SE,
                "hac_small=True vs Stata newey, lag(4), unweighted and [aw=w], "
                "to 1e-9",
                "sp.regress(F, data, robust='hac', hac_small=True, ...)",
            ),
        ),
        invariant={"vce": _VCE_INVARIANT},
    )
)

_add(
    _Scope(
        "iv",
        {
            "estimator": ("2sls", "liml", "fuller", "gmm"),
            "vce": _REG_VCE,
            "identification": ("just", "over"),
            "absorb": ("none", "set"),
            "small": ("true", "false"),
        },
        _x_iv,
        (
            _Row(
                "T2",
                _JOINT,
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("classical", "hc1", "cr1"),
                    "identification": _vals("over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                ("joint_test",),
                "Wald F, both degrees of freedom and p-value vs Stata ivregress "
                "2sls, small + test (the df-adjusted convention sp.ivreg "
                "reports; Stata's default ivregress is an unadjusted chi2)",
                "sp.ivreg(formula, data, robust=..., cluster=...)",
            ),
            _Row(
                "T2",
                _RP + "test_iv_card_aer_parity.py",
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("classical", "hc1", "cr1"),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "coefficient and classical / HC1 / CR1 SEs vs AER::ivreg + sandwich on "
                "the original Card extract, just- and over-identified",
                "sp.iv(formula, data, robust=..., cluster=...)",
            ),
            _Row(
                "T2",
                _RP + "test_iv_card_aer_parity.py",
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("classical"),
                    "identification": _vals("over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                ("diagnostic",),
                "Sargan over-identification statistic vs AER",
                "sp.iv(formula, data)",
            ),
            _Row(
                "T2",
                _R + "02_iv.py",
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("hc1"),
                    "identification": _vals("just"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "coefficients and HC1 SEs vs AER::ivreg (via sp.ivreg, "
                "bit-identical to sp.iv: " + _ATTACH + ")",
                "sp.ivreg(robust='hc1')",
            ),
            _Row(
                "T2",
                _ATTACH,
                {
                    "estimator": _vals("liml"),
                    "vce": _vals("classical"),
                    "identification": _vals("over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "LIML coefficient, SE and kappa vs ivmodel on Track A "
                "module 59's bytes, through sp.iv(method='liml')",
                "sp.iv(formula, data, method='liml')",
            ),
            _Row(
                "B",
                _B,
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("hc1"),
                    "identification": _vals("just"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _COV,
                "95% CI coverage, strong single instrument (via sp.ivreg)",
                "sp.ivreg(robust='hc1')",
            ),
            _Row(
                "T2",
                _RP + "test_iv_small_Stata_parity.py",
                {
                    "estimator": _vals("2sls", "liml"),
                    "vce": _vals("classical", "hc1", "cr1"),
                    "identification": _vals("just"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "coefficient, SE and p-value vs Stata ivregress 2sls / liml, small "
                "under default, robust and cluster variances (SE to 1e-12)",
                "sp.iv(formula, data, method=..., small=True)",
            ),
            _Row(
                "T2",
                _RP + "test_iv_small_Stata_parity.py",
                {
                    "estimator": _vals("2sls", "liml"),
                    "vce": _vals("classical", "hc0", "cr1"),
                    "identification": _vals("just"),
                    "absorb": _vals("none"),
                    "small": _vals("false"),
                },
                _EST_SE,
                "the same without small: N divisor, HC0 under vce(robust), no "
                "finite-sample cluster factor, z p-values, vs Stata ivregress",
                "sp.iv(formula, data, method=..., small=False)",
            ),
            _Row(
                "T2",
                _RP + "test_iv_gmm_stata_parity.py",
                {
                    "estimator": _vals("gmm"),
                    "vce": _vals("hc1", "cr1"),
                    "identification": _vals("over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "two-step GMM coefficients, SEs and Hansen J vs Stata ivregress gmm, "
                "small under wmatrix(robust) and wmatrix(cluster), to 5e-9",
                "sp.iv(formula, data, method='gmm')",
            ),
            _Row(
                "T2",
                _RP + "test_iv_gmm_stata_parity.py",
                {
                    "estimator": _vals("gmm"),
                    "vce": _vals("hc0", "cr1"),
                    "identification": _vals("over"),
                    "absorb": _vals("none"),
                    "small": _vals("false"),
                },
                _EST_SE,
                "the same without small (ivregress gmm's default): no "
                "finite-sample factor, z p-values",
                "sp.iv(formula, data, method='gmm', small=False)",
            ),
            _Row(
                "T2",
                _RP + "test_iv_gmm_stata_parity.py",
                {
                    "estimator": _vals("gmm"),
                    "vce": _vals("classical"),
                    "identification": _vals("over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "under the unadjusted weight matrix, both variances sp.iv offers: "
                "the default sandwich is ivregress gmm, wmatrix(unadjusted) "
                "vce(robust), and gmm_vcov='efficient' is wmatrix(unadjusted); "
                "neither carries a finite-sample factor",
                "sp.iv(formula, data, method='gmm')",
            ),
            _Row(
                "T2",
                _RP + "test_iv_hdfe_stata_parity.py",
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("classical", "hc1", "cr1", "cluster_multiway"),
                    "identification": _vals("just"),
                    "absorb": _vals("set"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "absorbed two-way fixed effects: coefficient and SE vs Stata ivreghdfe "
                "under iid, robust, one-way and two-way clustering (SE to 1e-10)",
                "sp.iv(formula, data, absorb=[...])",
            ),
            _Row(
                "T2",
                _RP + "test_iv_hdfe_stata_parity.py",
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("hc1", "cr1"),
                    "identification": _vals("over"),
                    "absorb": _vals("set"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "absorbed, over-identified 2SLS vs Stata ivreghdfe",
                "sp.iv(formula, data, absorb=[...])",
            ),
            _Row(
                "T2",
                _RP + "test_iv_hdfe_stata_parity.py",
                {
                    "estimator": _vals("liml"),
                    "vce": _vals("hc1", "cr1"),
                    "identification": _vals("over"),
                    "absorb": _vals("set"),
                    "small": _vals("true"),
                },
                _EST,
                "absorbed LIML coefficient and kappa vs Stata ivreghdfe, liml (1e-10)",
                "sp.iv(formula, data, absorb=[...], method='liml')",
            ),
            _Row(
                "T4",
                _RP + "test_iv_hdfe_stata_parity.py",
                {
                    "estimator": _vals("liml"),
                    "vce": _vals("hc1", "cr1"),
                    "identification": _vals("over"),
                    "absorb": _vals("set"),
                    "small": _vals("true"),
                },
                ("se",),
                "absorbed LIML SE within 5e-5 of ivreghdfe: ivreg2 builds the k-class "
                "meat from X_hat, StatsPAI from AX",
                "sp.iv(formula, data, absorb=[...], method='liml')",
            ),
            _Row(
                "T2",
                _RP + "test_iv_hdfe_stata_parity.py",
                {
                    "estimator": _vals("fuller"),
                    "vce": _vals("hc1"),
                    "identification": _vals("over"),
                    "absorb": _vals("set"),
                    "small": _vals("true"),
                },
                _EST,
                "absorbed Fuller(1) coefficient and kappa vs Stata ivreghdfe, "
                "fuller(1)",
                "sp.iv(formula, data, absorb=[...], method='fuller')",
            ),
            _Row(
                "T4",
                _RP + "test_iv_hdfe_stata_parity.py",
                {
                    "estimator": _vals("fuller"),
                    "vce": _vals("hc1"),
                    "identification": _vals("over"),
                    "absorb": _vals("set"),
                    "small": _vals("true"),
                },
                ("se",),
                "absorbed Fuller(1) SE within 5e-5 of ivreghdfe (same k-class meat "
                "convention)",
                "sp.iv(formula, data, absorb=[...], method='fuller')",
            ),
            _Row(
                "T2",
                _MENU,
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals(
                        "classical",
                        "hc0",
                        "hc1",
                        "hc2",
                        "hc3",
                        "cr1",
                        "cluster_multiway",
                    ),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "coefficients and SEs vs ivreg::ivreg + sandwich (vcov, vcovHC "
                "HC0-HC3, vcovCL) and vs fixest IV with two- and three-way "
                "clustering at its default ssc(), to 1e-9 (observed 1e-13); "
                "sp.iv zeroes negative eigenvalues of the multiway matrix, as "
                "fixest does",
                "sp.iv(IV[ident], data, **opts)",
            ),
            _Row(
                "T2",
                _MENU,
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("cr2", "cr3", "jackknife"),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "CR2 / CR3 SEs vs clubSandwich::vcovCR on ivreg::ivreg, to 1e-9; "
                "vce='jackknife' is CR3 (a fit reports vce 'cr3')",
                "sp.ivreg(IV[ident], data, vce=..., cluster='g')",
            ),
            _Row(
                "T2",
                _MENU,
                {
                    "estimator": _vals("2sls", "liml"),
                    "vce": _vals("classical", "hc1", "cr1"),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "coefficients and SEs vs Stata ivregress 2sls / liml, small "
                "(unadjusted, robust, cluster), to 1e-9",
                "sp.iv(IV[ident], data, method=..., small=True)",
            ),
            _Row(
                "T2",
                _MENU,
                {
                    "estimator": _vals("2sls", "liml"),
                    "vce": _vals("classical", "hc0", "cr1"),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
                    "small": _vals("false"),
                },
                _EST_SE,
                "the same without small (ivregress's default), to 1e-9",
                "sp.iv(IV[ident], data, method=..., small=False)",
            ),
            _Row(
                "T2",
                _MENU,
                {
                    "estimator": _vals("liml"),
                    "vce": _vals("hc0"),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "robust='hc0' carries no finite-sample factor: its LIML SE is "
                "ivregress liml, vce(robust)'s, to 1e-9",
                "sp.iv(IV[ident], data, method='liml', robust='hc0')",
            ),
            _Row(
                "T2",
                _MENU,
                {
                    "estimator": _vals("fuller"),
                    "vce": _vals("classical", "hc0", "hc1", "cr1"),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "Fuller(1) coefficients and SEs vs Stata ivreg2, fuller(1): "
                "small classical / robust / cluster, and robust without small "
                "for HC0, to 1e-9",
                "sp.iv(IV[ident], data, method='fuller', ...)",
            ),
            _Row(
                "T2",
                _MENU,
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("cluster_multiway"),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "sp.ivreg(cluster=[a, b]) reports the raw two-way matrix and "
                "equals Stata ivreg2, cluster(g yr) small, to 1e-9; where that "
                "matrix is not PSD it differs from sp.iv's projected one (next "
                "row)",
                "sp.ivreg(IV[ident], data, cluster=['g', 'yr'])",
            ),
            _Row(
                "T4",
                _MENU,
                {
                    "estimator": _vals("2sls", "liml", "fuller"),
                    "vce": _vals("cluster_multiway"),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                ("se",),
                "multiway: ivreg2 reports the raw inclusion-exclusion matrix, "
                "sp.iv zeroes its negative eigenvalues (fixest's fix). ivreg2's "
                "matrix is rebuilt from three one-way sp.iv fits to 1e-9, and "
                "its projection is sp.iv's to 1e-9",
                "sp.iv(IV[ident], data, method=..., cluster=['g', 'yr'])",
            ),
            _Row(
                "T2",
                _MENU,
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("conley"),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "Conley SEs (uniform kernel, 100 km) vs Stata acreg, to 1e-9; "
                "a negative variance is nan, as acreg reports it missing",
                "sp.ivreg(IV[ident], data, vce='conley', ...)",
            ),
            _Row(
                "T2",
                _MENU,
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("wild"),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                _EST_SE,
                "the endogenous coefficient's SE reported with the WRE bootstrap "
                "is ivreg2's one-way cluster SE (small), to 1e-9",
                "sp.ivreg(IV[ident], data, vce='wild', cluster='g')",
            ),
            _Row(
                "T2",
                _MENU,
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("wild"),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
                    "small": _vals("true"),
                },
                ("diagnostic",),
                "WRE p-value of the endogenous coefficient, all 2^12 Rademacher "
                "sign vectors enumerated: equal to Stata boottest after ivreg2 "
                "exactly",
                "sp.ivreg(IV[ident], data, vce='wild', cluster='g')",
            ),
        ),
        invariant={"vce": _VCE_INVARIANT},
        note="sp.iv and sp.ivreg share this map; they are asserted bit-identical "
        "on these configurations in " + _ATTACH + ". Module 59 itself runs "
        "sp.liml, a separate code path, so its row is attached through that test.",
        # added 2026-10-03; every earlier row and by-name query ran the default
        defaults={"small": "true"},
    )
)
SCOPES["ivreg"] = SCOPES["iv"]

_add(
    _Scope(
        "fast.feols",
        {
            "vcov": ("iid", "hc1", "cr1"),
            "ssc": ("fixest", "statspai"),
            "weights": ("none", "set"),
        },
        _x_feols,
        (
            _Row(
                "T2",
                _R + "03_hdfe.py",
                {
                    "vcov": _vals("iid"),
                    "ssc": _vals("fixest"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "coefficients and iid SEs vs fixest::feols",
                "sp.fast.feols(vcov='iid')",
            ),
            _Row(
                "T2",
                _R + "15_hdfe_cluster.py",
                {
                    "vcov": _vals("cr1"),
                    "ssc": _vals("fixest"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "CR1 SEs vs fixest::feols(cluster=)",
                "sp.fast.feols(vcov='cr1')",
            ),
            _Row(
                "B",
                _B,
                {
                    "vcov": _vals("cr1"),
                    "ssc": _vals("fixest"),
                    "weights": _vals("none"),
                },
                _COV,
                "95% CI coverage of the within estimator on a two-way FE panel DGP",
                "sp.fast.feols(vcov='cr1')",
            ),
            _Row(
                "T2",
                _RP + "test_fast_feols_weights_fixest_parity.py",
                {
                    "vcov": _vals("iid", "hc1", "cr1"),
                    "ssc": _vals("fixest"),
                    "weights": _vals("none", "set"),
                },
                _EST_SE,
                "one and two absorbed dimensions, with and without weights, and "
                "nine clustered layouts (nested / not nested) vs fixest::feols",
                "sp.fast.feols(formula, data, weights=..., vcov=...)",
            ),
        ),
        invariant={
            "vcov": _VCE_INVARIANT,
            "ssc": (_EST, "the small-sample factor scales the covariance only"),
        },
        note="The Track B row 'sp.panel two-way FE' is family-level evidence for "
        "sp.panel; this entry point has its own coverage row. ssc='statspai' (the "
        "pre-1.31 default) has no reference row; its CR1 SE over-covers on the "
        "Track B panel (mechanisms/feols_ssc.py).",
    )
)

_PANEL_SSC = _RP + "test_panel_ssc_stata_parity.py"
_PANEL_METHODS = (
    "fe",
    "twoway",
    "pooled",
    "fd",
    "re",
    "be",
    "mundlak",
    "chamberlain",
)


def _panel_row(
    kind: str,
    artifact: str,
    methods: Tuple[str, ...],
    vce: Tuple[str, ...],
    ssc: str,
    weights: str,
    outputs: Tuple[str, ...],
    compares: str,
    entry: str,
) -> _Row:
    return _Row(
        kind,
        artifact,
        {
            "method": _vals(*methods),
            "vce": _vals(*vce),
            "ssc": _vals(ssc),
            "weights": _vals(weights),
        },
        outputs,
        compares,
        entry,
    )


_add(
    _Scope(
        "panel",
        {
            "method": _PANEL_METHODS,
            "vce": ("unadjusted", "robust", "cluster"),
            "ssc": ("linearmodels", "stata", "fixest"),
            "weights": ("none", "set"),
        },
        _x_panel,
        (
            _panel_row(
                "T2",
                _R + "35_panel.py",
                ("fe", "re"),
                ("unadjusted",),
                "linearmodels",
                "none",
                _EST,
                "FE / RE coefficients vs plm::plm (SE budget 1e-3)",
                "sp.panel(data=df, ...)",
            ),
            *(
                _panel_row(
                    "T2",
                    _JOINT,
                    ("fe",),
                    ("unadjusted",),
                    ssc,
                    "none",
                    ("joint_test",),
                    "joint F of the time effects vs Stata xtreg, fe + testparm",
                    "sp.panel(data, formula, method='fe')",
                )
                for ssc in ("linearmodels", "stata", "fixest")
            ),
            *(
                _panel_row(
                    "T2",
                    _JOINT,
                    ("fe",),
                    ("cluster",),
                    ssc,
                    "none",
                    ("joint_test",),
                    "joint F of the time effects on (q, G - 1) vs Stata xtreg, fe "
                    "vce(cluster) + testparm",
                    "sp.panel(data, formula, method='fe', cluster=..., ssc=...)",
                )
                for ssc in ("stata", "fixest")
            ),
            _panel_row(
                "T4",
                _JOINT,
                ("fe",),
                ("cluster",),
                "linearmodels",
                "none",
                ("joint_test",),
                "default small-sample convention under clustering: the statistic "
                "is 1.7% above xtreg's and is referred to F(q, N - K) instead of "
                "F(q, G - 1), so the p-value is less than half of Stata's on 60 "
                "clusters; pass ssc='stata' for the xtreg number",
                "sp.panel(data, formula, method='fe', cluster=...)",
            ),
            _panel_row(
                "T2",
                _PANEL_SSC,
                ("fe", "twoway", "pooled", "fd", "re"),
                ("unadjusted", "robust", "cluster"),
                "stata",
                "none",
                _EST_SE,
                "coefficients, SEs and t / z df vs Stata xtreg / areg / regress",
                "sp.panel(data, ..., ssc=ssc)",
            ),
            _panel_row(
                "T2",
                _PANEL_SSC,
                ("be",),
                ("unadjusted",),
                "stata",
                "none",
                _EST_SE,
                "coefficients and SEs vs Stata xtreg, be",
                "sp.panel(data, ..., ssc=ssc)",
            ),
            _panel_row(
                "T2",
                _PANEL_SSC,
                ("mundlak",),
                ("unadjusted", "cluster"),
                "stata",
                "none",
                _EST_SE,
                "coefficients and SEs vs Stata xtreg, re with unit means",
                "sp.panel(data, ..., ssc=ssc)",
            ),
            _panel_row(
                "T2",
                _PANEL_SSC,
                ("chamberlain",),
                ("cluster",),
                "stata",
                "none",
                _EST_SE,
                "coefficients and SEs vs Stata xtreg, re with Chamberlain terms",
                "sp.panel(data, ..., ssc=ssc)",
            ),
            _panel_row(
                "T2",
                _PANEL_SSC,
                ("fe", "twoway", "pooled"),
                ("unadjusted", "cluster"),
                "stata",
                "set",
                _EST_SE,
                "weighted fits vs Stata xtreg / regress [aw=]",
                "sp.panel(data, ..., ssc=ssc)",
            ),
            _panel_row(
                "T2",
                _PANEL_SSC,
                ("fe", "twoway", "pooled", "fd"),
                ("unadjusted", "robust", "cluster"),
                "fixest",
                "none",
                _EST_SE,
                "coefficients, SEs and t df vs R fixest default ssc()",
                "sp.panel(data, ..., ssc=ssc)",
            ),
            _panel_row(
                "T2",
                _PANEL_SSC,
                ("fe", "twoway", "pooled"),
                ("unadjusted", "robust", "cluster"),
                "fixest",
                "set",
                _EST_SE,
                "weighted fits vs R fixest weights=",
                "sp.panel(data, ..., ssc=ssc)",
            ),
        ),
        invariant={
            "vce": (_EST, "the covariance is computed after the coefficients"),
            "ssc": (_EST, "the small-sample convention scales the covariance only"),
        },
        note="The default ssc (linearmodels' own scaling) has no SE reference row "
        "beyond the unadjusted FE / RE budget of module 35; robust and clustered "
        "SEs are reference-checked only under ssc='stata' or ssc='fixest'.",
    )
)

_CS_DOMAINS = {
    "sample": ("panel", "unbalanced_panel", "rcs", "unbalanced_pair"),
    "estimator": ("dr", "drimp", "reg", "ipw", "ipw_abadie"),
    "control_group": ("nevertreated", "notyettreated"),
    "notyet_cutoff": ("period", "asinr", "cohort"),
    "weights": ("none", "weighted"),
    "covariates": ("none", "set"),
    "inference": ("analytic", "bootstrap"),
    "base_period": ("universal", "varying"),
    "anticipation": ("0", ">0"),
    "clustering": ("none", "set"),
    "parity": ("none", "r", "stata"),
    # 'na' where the DR variance sign cannot matter (see
    # _cs_variance_convention); R did, DRDID and csdid all use 'reference'.
    "variance_convention": ("na", "delta", "reference"),
}
#: The values most CS rows run; ``_cs(...)`` overrides them per row.
_CS_BASE = {
    "sample": ("panel",),
    "estimator": ("dr",),
    "control_group": ("nevertreated",),
    "notyet_cutoff": ("period",),
    "weights": ("none",),
    "covariates": ("none",),
    "inference": ("analytic",),
    "base_period": ("universal",),
    "anticipation": ("0",),
    "clustering": ("none",),
    "parity": ("none",),
    "variance_convention": ("na",),
}


def _cs(**over: Tuple[str, ...]) -> Dict[str, FrozenSet[str]]:
    return {k: _vals(*over.get(k, v)) for k, v in _CS_BASE.items()}


_CS_OPT = _RP + "test_cs_options_R_parity.py"
_CS_96 = _RP + "test_csdid_options_stata_parity.py"
_CS_GAPS = _RP + "test_cs_gaps_parity.py"
_CS_NONEVER = _RP + "test_cs_no_never_unbalanced_parity.py"
_CS_PAR = _RP + "test_cs_parity_switch.py"
_CS_PAR_RU = _RP + "test_cs_parity_rc_unbal.py"
#: parity= rows compare the fit's simple ATT and its SE, the pre-test (Wald
#: statistic and chi-squared p-value) and, through sp.aggte, the four
#: aggregations and every cell.
_CS_PAR_OUT = ("estimate", "se", "joint_test")
_add(
    _Scope(
        "callaway_santanna",
        _CS_DOMAINS,
        _x_cs,
        (
            _Row(
                "T2",
                _R + "04_csdid.py",
                _cs(estimator=("reg",)),
                _EST_SE,
                "simple / dynamic / group ATT and analytic SEs vs did and csdid",
                "sp.callaway_santanna(estimator='reg') + sp.aggte(bstrap=False)",
            ),
            _Row(
                "T2",
                _RP + "test_cs_weighted_parity.py",
                _cs(
                    estimator=("dr", "reg", "ipw"),
                    control_group=("nevertreated", "notyettreated"),
                    weights=("none", "weighted"),
                    covariates=("none", "set"),
                ),
                _EST_SE,
                "simple / dynamic / group ATT and analytic SE vs did::att_gt over "
                "dr/reg/ipw x never/not-yet x weights x covariates (72 cells)",
                "sp.callaway_santanna(...)",
            ),
            _Row(
                "T2",
                _RP + "test_aggte_mpdta_parity.py",
                _cs(base_period=("varying",)),
                _EST_SE,
                "did::mpdta: the simple, dynamic, group and calendar aggregates "
                "and their analytic SEs vs did::aggte at its default varying "
                "base period, to 1e-9 absolute",
                "sp.aggte(cs_result, type=agg_type, bstrap=False)",
            ),
            _Row(
                "T2",
                _RP + "test_aggte_vcov_r_parity.py",
                _cs(),
                ("vcov",),
                "every entry of the dynamic event-study covariance (off-diagonal "
                "blocks included) vs did::aggte influence functions; the matrix "
                "sp.honest_did and sp.uniform_bands consume",
                "sp.aggte(type='dynamic')",
            ),
            _Row(
                "T2",
                _CS_GAPS,
                _cs(base_period=("universal", "varying"), anticipation=("0", ">0")),
                _EST_SE,
                "every ATT(g, t) and its analytic SE at anticipation 0, 1 and 2 "
                "under both base periods vs did::att_gt, to 1e-7 (the (g, t) "
                "grid itself is compared)",
                "sp.callaway_santanna(panel, ...)",
            ),
            _Row(
                "T2",
                _CS_GAPS,
                _cs(
                    sample=("unbalanced_panel",),
                    estimator=("dr", "ipw", "reg"),
                    covariates=("none", "set"),
                    base_period=("varying",),
                    variance_convention=("na", "reference"),
                ),
                _EST_SE,
                "allow_unbalanced_panel=True: every ATT(g, t) and its analytic SE "
                "vs did::att_gt(allow_unbalanced_panel = TRUE) for dr / ipw / reg "
                "with and without a covariate, to 1e-7",
                "sp.callaway_santanna(unbalanced, ...)",
            ),
            _Row(
                "T2",
                _CS_NONEVER,
                _cs(
                    sample=("panel", "rcs"),
                    estimator=("reg", "dr"),
                    control_group=("notyettreated",),
                    base_period=("universal", "varying"),
                    anticipation=("0", ">0"),
                ),
                _EST_SE,
                "a design without never-treated units: every ATT(g, t), the "
                "simple aggregate and their analytic SEs vs did 2.5.1 on panel "
                "and repeated cross-section data, both base periods, "
                "anticipation 0 and 1, to 1e-8",
                "sp.callaway_santanna(df, **kw)",
            ),
            _Row(
                "T2",
                _CS_NONEVER,
                _cs(
                    sample=("unbalanced_panel",),
                    estimator=("reg", "dr"),
                    control_group=("nevertreated", "notyettreated"),
                    base_period=("varying",),
                ),
                _EST_SE,
                "allow_unbalanced_panel=True under both control groups: every "
                "ATT(g, t), the simple aggregate and their SEs vs did 2.5.1, "
                "to 1e-8",
                "sp.callaway_santanna(df, **kw)",
            ),
            _Row(
                "T2",
                _RP + "test_cs_notyet_cutoff_parity.py",
                _cs(
                    sample=("panel", "rcs"),
                    control_group=("notyettreated",),
                    covariates=("none", "set"),
                    base_period=("universal", "varying"),
                    anticipation=("0", ">0"),
                    variance_convention=("na", "reference"),
                ),
                _EST_SE,
                "not-yet-treated comparison set at the default notyet_cutoff: "
                "every ATT(g, t) and its SE vs did 2.5.1 on panel and repeated "
                "cross-section data, both base periods, anticipation 0 and 1, "
                "with and without a covariate, to 1e-8",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _RP + "test_cs_rcs_parity.py",
                _cs(
                    sample=("rcs",),
                    estimator=("dr", "ipw", "reg"),
                    control_group=("nevertreated", "notyettreated"),
                    covariates=("set",),
                    base_period=("varying",),
                    variance_convention=("na", "reference"),
                ),
                _EST_SE,
                "did::mpdta as repeated cross-sections with lpop: the simple "
                "aggregate and its SE vs did 2.5.1 (panel = FALSE) for dr / ipw "
                "/ reg under both control groups",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _RP + "test_cs_rcs_parity.py",
                _cs(
                    sample=("rcs",),
                    control_group=("nevertreated", "notyettreated"),
                    covariates=("set",),
                    base_period=("varying",),
                    variance_convention=("delta",),
                ),
                ("estimate",),
                "default variance_convention='delta': every ATT(g, t) equals the "
                "'reference' fit's (1e-12), which matches did 2.5.1; the SE is "
                "the paper's influence function (drdid_rc), not a cross-software "
                "number",
                "sp.callaway_santanna(panel=False, estimator='dr', x=...)",
            ),
            _Row(
                "T2",
                _RP + "test_rcs_aggregation_conventions.py",
                _cs(sample=("rcs",), estimator=("reg",)),
                _EST,
                "repeated cross-sections: the four overall aggregates and the "
                "cohort cells vs did::aggte, and agg_weights='csdid' vs Stata "
                "csdid, to 1e-12 (point estimates only)",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _RP + "test_csdid_conventions_stata_parity.py",
                _cs(
                    estimator=("reg",),
                    control_group=("notyettreated",),
                    notyet_cutoff=("period", "asinr", "cohort"),
                ),
                _EST,
                "every ATT(g, t) vs Stata csdid method(reg) notyet long2 under "
                "notyet_cutoff 'asinr', 'cohort' and the default, to 1e-9 "
                "(fixture 82; point estimates only)",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _RP + "test_csdid_conventions_stata_parity.py",
                _cs(estimator=("ipw", "ipw_abadie"), covariates=("set",)),
                _EST,
                "every ATT(g, t) vs Stata csdid method(stdipw) and method(ipw) "
                "(Abadie) with lpop, to 1e-9 (fixture 82; point estimates only)",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_96,
                _cs(
                    estimator=("reg",),
                    control_group=("notyettreated",),
                    notyet_cutoff=("asinr", "cohort"),
                ),
                _EST_SE,
                "notyet_cutoff 'asinr' and 'cohort': every ATT(g, t) and its "
                "SE vs Stata csdid notyet long2 [asinr] method(reg), to 1e-8 "
                "(fixture 96; R did has only the 'period' rule)",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_96,
                _cs(estimator=("ipw_abadie",), covariates=("set",)),
                _EST_SE,
                "every ATT(g, t) and its SE vs Stata csdid method(ipw) with "
                "lpop, to 1e-7 (fixture 96)",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_96,
                _cs(sample=("unbalanced_pair",), covariates=("set",)),
                _EST_SE,
                "the default balance='pair' on an unbalanced panel (each cell "
                "uses the units seen in both of its periods): every ATT(g, t) "
                "and its SE vs Stata csdid 1.81 method(dripw) with lpop, to "
                "1e-7 (fixture 96)",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_96,
                _cs(
                    sample=("unbalanced_pair",),
                    estimator=("reg",),
                    control_group=("notyettreated",),
                    notyet_cutoff=("cohort",),
                ),
                _EST_SE,
                "the default balance='pair' on an unbalanced panel: every "
                "ATT(g, t) and its SE vs Stata csdid 1.81 notyet long2 "
                "method(reg), to 1e-8 (fixture 96)",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_OPT,
                _cs(
                    sample=("panel", "rcs"),
                    estimator=("drimp", "ipw_abadie"),
                    covariates=("set",),
                ),
                _EST_SE,
                "the improved doubly robust and the Abadie IPW estimators "
                "(DRDID::drdid_imp_* / ipw_did_* as att_gt's est_method) on "
                "panel and repeated cross-section data: every ATT(g, t), the four "
                "aggregates, their per-cell rows and SEs vs did 2.5.1, to 1e-7",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_OPT,
                _cs(
                    estimator=("drimp",),
                    control_group=("notyettreated",),
                    covariates=("set",),
                    base_period=("varying",),
                ),
                _EST_SE,
                "drimp under the not-yet-treated comparison and the varying "
                "base: cells, four aggregates and SEs vs did 2.5.1, to 1e-7",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_OPT,
                _cs(estimator=("ipw",), covariates=("set",), clustering=("set",)),
                _EST_SE,
                "'ipw' and 'stdipw' with lpop and a 29-state cluster: cells, four "
                "aggregates and analytic clustered SEs vs did 2.5.1, to 1e-7",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_OPT,
                _cs(sample=("rcs",), estimator=("ipw",), covariates=("set",)),
                _EST_SE,
                "'ipw' and 'stdipw' on repeated cross-sections: cells, four "
                "aggregates and SEs vs did 2.5.1 (std_ipw_did_rc), to 1e-7",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_OPT,
                _cs(weights=("weighted",), covariates=("set",), anticipation=(">0",)),
                _EST_SE,
                "anticipation = 1 with unit weights and lpop: cells, four "
                "aggregates, per-cell rows and SEs vs did 2.5.1, to 1e-7",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_OPT,
                _cs(
                    estimator=("reg",),
                    control_group=("notyettreated",),
                    base_period=("varying",),
                    anticipation=(">0",),
                    clustering=("set",),
                ),
                _EST_SE,
                "anticipation = 1, not-yet-treated, varying base, 29-state "
                "cluster: cells, four aggregates and clustered SEs vs did 2.5.1, "
                "to 1e-7",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _RP + "test_cs_rc_cluster_csdid_parity.py",
                _cs(sample=("panel", "rcs"), clustering=("set",)),
                _EST_SE,
                "clustered analytic SEs vs Stata csdid (v1) cluster() to 1e-9: "
                "every ATT(g,t), group and simple aggregates, on a panel with "
                "a time-invariant cluster (mpdta) and on repeated "
                "cross-sections clustered on a row-level city x year",
                "sp.callaway_santanna(clustervars=...)",
            ),
            _Row(
                "T2",
                _RP + "test_cs_rc_weighted_aggte_did_parity.py",
                _cs(
                    sample=("rcs",),
                    weights=("weighted",),
                    clustering=("none", "set"),
                ),
                _EST_SE,
                "weighted repeated cross-sections (panel=False): simple, "
                "dynamic, group and calendar aggregates and their analytic "
                "SEs vs did::aggte, with and without a row-level cluster",
                "sp.callaway_santanna(panel=False, weights=...) + sp.aggte",
            ),
            _Row(
                "T2",
                _RP + "test_cs_balance_full_did_parity.py",
                _cs(control_group=("nevertreated", "notyettreated")),
                _EST_SE,
                "unbalanced panel under balance='full' (drop units not "
                "observed in every period): every ATT(g,t) and the four "
                "aggregates vs did::att_gt at its default "
                "allow_unbalanced_panel = FALSE",
                "sp.callaway_santanna(balance='full')",
            ),
            _Row(
                "T2",
                _RP + "test_cs_cluster_analytic_parity.py",
                _cs(
                    sample=("panel", "unbalanced_panel"),
                    control_group=("nevertreated", "notyettreated"),
                    base_period=("universal", "varying"),
                    clustering=("set",),
                ),
                _EST_SE,
                "clustered analytic SEs vs R did (>= 2.5.0) clustervars= to "
                "1e-8 on nine clusters of unequal size: every ATT(g,t) and "
                "the simple / dynamic / group / calendar aggregates, on a "
                "balanced and an unbalanced panel",
                "sp.callaway_santanna(clustervars=...)",
            ),
            _Row(
                "T2",
                _RP + "test_stata_did_commands_parity.py",
                _cs(
                    control_group=("nevertreated", "notyettreated"),
                    covariates=("set",),
                    base_period=("universal", "varying"),
                ),
                _EST_SE,
                "a covariate that varies over time, read in the earlier period "
                "of each cell: every ATT(g, t) and its SE vs R did::att_gt, "
                "to 1e-7",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T4",
                _RP + "test_stata_did_commands_parity.py",
                _cs(
                    estimator=("dr", "drimp"),
                    control_group=("nevertreated", "notyettreated"),
                    covariates=("set",),
                    base_period=("universal", "varying"),
                ),
                _EST_SE,
                "csdid lines run through sp.stata vs Stata csdid method(dripw) / "
                "method(drimp): cells and estat aggregates agree to 2e-6, held "
                "to 1e-5 because Stata's logit stops at its default tolerance "
                "(a documented optimiser difference, not a 1e-6 parity row)",
                "sp.stata(",
            ),
            _Row(
                "T2",
                _CS_PAR,
                _cs(
                    parity=("r",),
                    control_group=("nevertreated", "notyettreated"),
                    covariates=("none", "set"),
                    base_period=("varying",),
                    clustering=("none", "set"),
                ),
                _CS_PAR_OUT,
                "parity='r' on mpdta and on mpdta with 35 always-treated "
                "counties: every ATT(g, t) and SE, the pre-test, the fit's "
                "simple ATT and the four sp.aggte aggregations (overall and "
                "every cell) vs R did 2.5.1 (bstrap = FALSE), to 1e-10 "
                "without lpop and 1e-8 with it",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_PAR,
                _cs(parity=("r",), base_period=("varying",), anticipation=(">0",)),
                _CS_PAR_OUT,
                "parity='r' with anticipation = 1 on the always-treated panel: "
                "the same comparison vs R did 2.5.1, to 1e-10",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_PAR,
                _cs(
                    parity=("stata",),
                    control_group=("nevertreated", "notyettreated"),
                    notyet_cutoff=("period", "cohort"),
                    covariates=("none", "set"),
                    base_period=("varying",),
                    clustering=("none", "set"),
                ),
                _CS_PAR_OUT,
                "parity='stata' on mpdta and on mpdta with 35 always-treated "
                "counties: every ATT(g, t) and SE, estat pretrend, the fit's "
                "simple ATT and estat simple / group / calendar / event "
                "(estimates, SEs, confidence limits) vs Stata csdid 1.81, to "
                "1e-10 without lpop and 1e-7 with it",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_PAR,
                _cs(
                    parity=("stata",),
                    base_period=("varying",),
                    inference=("bootstrap",),
                ),
                _EST_SE,
                "parity='stata' with bstrap=True: the fit's simple ATT and its "
                "SE are csdid's analytic estat simple after wboot, to 1e-10 "
                "(the bootstrap SEs of the cells are not compared)",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_PAR_RU,
                _cs(
                    parity=("r",),
                    sample=("rcs", "panel", "unbalanced_panel"),
                    control_group=("nevertreated", "notyettreated"),
                    weights=("none", "weighted"),
                    covariates=("none", "set"),
                    base_period=("varying",),
                    clustering=("none", "set"),
                    variance_convention=("na", "reference"),
                ),
                _CS_PAR_OUT,
                "parity='r' on a repeated cross-section and an unbalanced panel "
                "built from mpdta (R's default drop of incomplete units, and "
                "allow_unbalanced_panel = TRUE), unweighted and with sampling "
                "weights (and weighted on the balanced mpdta): every ATT(g, t) "
                "and SE, the pre-test, the fit's simple ATT and the four "
                "aggregations vs R did 2.5.1, to 1e-10 without lpop and 1e-7 "
                "with it",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_PAR_RU,
                _cs(
                    parity=("stata",),
                    sample=("rcs", "unbalanced_pair"),
                    control_group=("nevertreated", "notyettreated"),
                    notyet_cutoff=("period", "cohort"),
                    weights=("none", "weighted"),
                    covariates=("none", "set"),
                    base_period=("varying",),
                    clustering=("none", "set"),
                    variance_convention=("na", "reference"),
                ),
                _CS_PAR_OUT,
                "parity='stata' on the same repeated cross-section (csdid "
                "without ivar()) and unbalanced panel (ivar()), unweighted and "
                "with [iw=w]: every ATT(g, t) and SE, estat pretrend, the fit's "
                "simple ATT (estat simple) and estat simple / group / calendar "
                "/ event with csdid's per-cell treated counts (weight mass "
                "when weighted) vs Stata csdid 1.81, to 1e-10 without lpop and "
                "1e-6 with it",
                "sp.callaway_santanna(",
            ),
            _Row(
                "T2",
                _CS_PAR_RU,
                _cs(
                    parity=("stata",),
                    control_group=("nevertreated", "notyettreated"),
                    notyet_cutoff=("period", "cohort"),
                    weights=("weighted",),
                    covariates=("none", "set"),
                    base_period=("varying",),
                    clustering=("none", "set"),
                ),
                _CS_PAR_OUT,
                "parity='stata' on the balanced mpdta with sampling weights: "
                "the same comparison vs csdid [iw=w], to 1e-10 without lpop "
                "and 1e-6 with it",
                "sp.callaway_santanna(",
            ),
            _Row(
                "S",
                _RP + "test_cs_inference_parity.py",
                _cs(inference=("bootstrap",), clustering=("none", "set")),
                ("se",),
                "multiplier-bootstrap SEs within 8-10% and uniform critical values "
                "within 5% of did at 9,999 draws (one seed per side; a tolerance "
                "screen, not an equivalence test)",
                "sp.callaway_santanna(bstrap=True)",
            ),
            _Row(
                "B",
                _B,
                _cs(estimator=("reg",)),
                _COV,
                "95% CI coverage of the simple ATT on a staggered DGP",
                "sp.callaway_santanna(estimator='reg')",
            ),
        ),
        invariant={
            "inference": (
                _EST,
                "the multiplier bootstrap resamples influence functions for SEs "
                "and bands; the ATT(g,t) and their aggregates are unchanged",
            ),
            "base_period": (
                _EST_SE,
                "the headline is the simple aggregate of the post-treatment "
                "cells, whose base period is g - 1 - anticipation under both "
                "schemes; only the pre-treatment placebos (and so the event-"
                "study covariance) differ (asserted equal in "
                "tests/test_validation_scope.py)",
            ),
        },
        defaults={
            "sample": "panel",
            "notyet_cutoff": "period",
            "parity": "none",
            # what a default fit records for that configuration
            "variance_convention": _cs_query_variance_convention,
        },
        note="'stdipw' is read as 'ipw' (the same estimator). sample="
        "'unbalanced_pair' is an unbalanced panel fitted under the default "
        "balance='pair' (csdid 1.8x's pairwise rule); R did drops incomplete "
        "units instead (balance='full'), so its references are Stata csdid's. "
        "The headline is the simple ATT of the fit; sp.aggte results do not "
        "carry the fit's configuration and read as unchecked. parity='r' / "
        "'stata' fits are their own cells: under 'stata' off a balanced panel "
        "the headline is csdid's count-weighted estat simple, and both set "
        "the varying base period and the reference DR variance sign, so the "
        "rows without parity do not vouch for them.",
    )
)

_add(
    _Scope(
        "sun_abraham",
        {
            "control_group": ("nevertreated", "lastcohort", "control_cohort"),
            "aggregation": ("fixest_att", "event_time"),
            "share_variance": ("estimated", "fixed"),
            "window_rule": ("report", "bin", "reference"),
            "crit_dist": ("normal", "t_resid", "t_cluster"),
        },
        _x_sa,
        (
            _Row(
                "T2",
                _R + "05_sunab.py",
                {
                    "control_group": _vals("nevertreated"),
                    "aggregation": _vals("fixest_att"),
                    "share_variance": _vals("estimated", "fixed"),
                    "window_rule": _vals("report"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "cohort-size-weighted ATT (agg='att') and SE vs fixest::sunab and "
                "eventstudyinteract",
                "sp.sun_abraham(aggregation='fixest_att')",
            ),
            _Row(
                "T2",
                _RP + "test_sunab_event_time_aggregate_parity.py",
                {
                    "control_group": _vals("nevertreated"),
                    "aggregation": _vals("event_time"),
                    "share_variance": _vals("estimated", "fixed"),
                    "window_rule": _vals("report"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "equal-weighted post-period event-time average and its SE vs the "
                "same linear combination of fixest::sunab coefficients, on both "
                "share_variance settings",
                "sp.sun_abraham()",
            ),
            _Row(
                "B",
                _B,
                {
                    "control_group": _vals("nevertreated"),
                    "aggregation": _vals("event_time"),
                    "share_variance": _vals("estimated"),
                    "window_rule": _vals("report"),
                    "crit_dist": _vals("normal"),
                },
                _COV,
                "95% CI coverage of the default event-time aggregate",
                "sp.sun_abraham()",
            ),
            _Row(
                "T2",
                _RP + "test_event_study_vcov_R_parity.py",
                {
                    "control_group": _vals("nevertreated"),
                    "aggregation": _vals("event_time"),
                    "share_variance": _vals("fixed"),
                    "window_rule": _vals("report"),
                    "crit_dist": _vals("normal"),
                },
                ("vcov",),
                "every entry of the event-study covariance (off-diagonal blocks "
                "included) vs fixest::sunab's A V A'; the matrix sp.uniform_bands "
                "and sp.honest_did consume",
                "sp.sun_abraham(share_variance=False)",
            ),
            _Row(
                "T2",
                _RP + "test_dcdh_textbook_stata_parity.py",
                {
                    "control_group": _vals("nevertreated"),
                    "aggregation": _vals("event_time"),
                    "share_variance": _vals("estimated"),
                    "window_rule": _vals("bin"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "event-time coefficients and SEs with both ends of the window "
                "pooled, on an unbalanced panel, unweighted and with weights "
                "that vary over time, vs Stata eventstudyinteract",
                "sp.sun_abraham(window_rule='bin')",
            ),
            _Row(
                "T2",
                _RP + "test_sunab_options_parity.py",
                {
                    "control_group": _vals("lastcohort", "control_cohort"),
                    "aggregation": _vals("event_time", "fixest_att"),
                    "share_variance": _vals("estimated", "fixed"),
                    "window_rule": _vals("report"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "the 2007 cohort as control on mpdta (the last cohort, and the "
                "same cohort named by control_cohort=): event-time effects "
                "and SEs vs eventstudyinteract control_cohort(c2007) with the "
                "share term and vs fixest::sunab(ref.c=2007) without it; both "
                "summary aggregates vs fixest",
                "sp.sun_abraham(control_group='lastcohort')",
            ),
            _Row(
                "T2",
                _RP + "test_sunab_control_cohort_parity.py",
                {
                    "control_group": _vals("nevertreated", "control_cohort"),
                    "aggregation": _vals("event_time"),
                    "share_variance": _vals("estimated"),
                    "window_rule": _vals("report"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "event-time effects and SEs with the never-treated, and with "
                "control_cohort= naming the 2007 cohort (as a value and as a "
                "0/1 column), vs eventstudyinteract control_cohort()",
                "sp.sun_abraham(control_cohort=2007)",
            ),
            _Row(
                "T2",
                _RP + "test_sunab_options_parity.py",
                {
                    "control_group": _vals("nevertreated", "lastcohort"),
                    "aggregation": _vals("event_time", "fixest_att"),
                    "share_variance": _vals("estimated", "fixed"),
                    "window_rule": _vals("reference"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "windows (-3, 2) and (-2, 1) with the relative times outside "
                "them in the omitted category: event-time effects and SEs vs "
                "eventstudyinteract run on the in-window dummies only and vs "
                "fixest::sunab(ref.p=c(-1, <outside>)); both summary "
                "aggregates vs fixest",
                "sp.sun_abraham(window_rule='reference')",
            ),
            _Row(
                "T2",
                _ES_CRIT,
                {
                    "control_group": _vals("nevertreated"),
                    "aggregation": _vals("event_time"),
                    "share_variance": _vals("estimated", "fixed"),
                    "window_rule": _vals("report"),
                    "crit_dist": _vals("t_resid", "t_cluster"),
                },
                ("estimate", "se", "interval"),
                "parity='stata' / 'r' on the castle-doctrine panel, weighted, "
                "clustered by state and (without a unit id) robust: every "
                "event-time SE, p-value and confidence limit vs Stata "
                "eventstudyinteract (share term, t(G-1) / t(N-K)) and R "
                "fixest::sunab (fixed shares, same t), to 1e-10",
                "sp.sun_abraham(",
            ),
        ),
        invariant={
            "aggregation": (
                ("vcov",),
                "the event-time covariance is formed before the summary "
                "aggregate is chosen",
            ),
            "crit_dist": (
                ("estimate", "se", "vcov"),
                "the critical value turns an SE into a p-value and an "
                "interval; coefficients and SEs are computed before it is "
                "chosen (asserted equal across parity= in "
                "tests/reference_parity/test_es_crit_parity.py)",
            ),
        },
        defaults={"share_variance": "estimated", "crit_dist": "normal"},
        note="window_rule only matters with event_window=; 'report' fits the "
        "saturated regression and is what every row without a window runs. "
        "'reference' (the behaviour before 1.39.0) is kept to reproduce "
        "earlier output; it is what eventstudyinteract computes when only "
        "the in-window dummies are passed, and is pinned against it. "
        "control_group='control_cohort' is any explicit control_cohort=; "
        "the evidence ran the 2007 cohort of mpdta. "
        "share_variance changes the per-period SEs only (module 05 pins both "
        "conventions: fixest for fixed shares, eventstudyinteract for the Prop. 3 "
        "term); the overall aggregate's SE treats cohort shares as fixed on both "
        "settings. The joint event-study covariance has a reference only with "
        "share_variance=False: fixest holds the shares fixed, and no reference "
        "pins the off-diagonal blocks of the Prop. 3 matrix.",
    )
)

_BJS_BASE = {
    "vce": _vals("analytic"),
    "weights": _vals("none"),
    "horizon": _vals("overall"),
}
_BJS_OPT = _RP + "test_bjs_fe_covariates_parity.py"
_BJS_SE = _RP + "test_option_fixture_bindings.py"
_BJS_94 = _RP + "test_bjs_options_parity.py"

_add(
    _Scope(
        "did_imputation",
        {
            "fe": ("unit_time", "time", "none", "custom"),
            "covariates": (
                "none",
                "controls",
                "time_covariates",
                "unit_covariates",
                "mixed",
            ),
            "vce": ("analytic", "bootstrap", "none"),
            "weights": ("none", "set"),
            "horizon": ("overall", "horizons"),
        },
        _x_bjs,
        (
            _Row(
                "T2",
                _R + "16_bjs.py",
                {**_BJS_BASE, "fe": _vals("unit_time"), "covariates": _vals("none")},
                _EST,
                "overall ATT vs didimputation and Stata did_imputation",
                "sp.did_imputation()",
            ),
            _Row(
                "T2",
                _BJS_OPT,
                {
                    **_BJS_BASE,
                    "fe": _vals("unit_time", "time", "none"),
                    "covariates": _vals("none"),
                },
                _EST,
                "overall ATT under fe(i t) / fe(t) / fe(.) vs Stata did_imputation "
                "run in double precision",
                "sp.did_imputation(fe=...)",
            ),
            _Row(
                "T2",
                _BJS_OPT,
                {
                    **_BJS_BASE,
                    "fe": _vals("unit_time"),
                    "covariates": _vals("controls", "time_covariates"),
                },
                _EST,
                "overall ATT with controls() / timecontrols() vs Stata did_imputation",
                "sp.did_imputation(controls=...|time_covariates=...)",
            ),
            _Row(
                "T4",
                _BJS_OPT,
                {
                    **_BJS_BASE,
                    "fe": _vals("unit_time"),
                    "covariates": _vals("unit_covariates"),
                },
                _EST,
                "unitcontrols(): Stata's iteration does not settle below 1e-6 "
                "relative (its ATT moves 6e-7 across tol() settings); StatsPAI "
                "equals the dense least-squares fit to 1e-11, Stata sits 1.4e-6 "
                "from it",
                "sp.did_imputation(unit_covariates=...)",
            ),
            _Row(
                "T2",
                _BJS_SE,
                {
                    **_BJS_BASE,
                    "fe": _vals("unit_time", "time", "none"),
                    "covariates": _vals("none"),
                },
                ("se",),
                "SE under fe(i t) / fe(t) / fe(.) vs Stata did_imputation",
                "sp.did_imputation(fe=...)",
            ),
            _Row(
                "T2",
                _BJS_SE,
                {
                    **_BJS_BASE,
                    "fe": _vals("unit_time"),
                    "covariates": _vals(
                        "controls", "time_covariates", "unit_covariates"
                    ),
                },
                ("se",),
                "SE with controls() / timecontrols() / unitcontrols() vs Stata "
                "did_imputation",
                "sp.did_imputation(controls=...|time_covariates=...|"
                "unit_covariates=...)",
            ),
            _Row(
                "T2",
                _R + "84_bjs_pretrends.py",
                {
                    "fe": _vals("unit_time"),
                    "covariates": _vals("none"),
                    "vce": _vals("analytic"),
                    "weights": _vals("none"),
                    "horizon": _vals("horizons"),
                },
                _EST_SE,
                "per-horizon ATTs, the pre-treatment lead vector and their SEs vs "
                "didimputation and Stata did_imputation",
                "sp.did_imputation(horizon=[...], pretrend_method='bjs')",
            ),
            _Row(
                "T2",
                _RP + "test_event_study_vcov_Stata_parity.py",
                {
                    "fe": _vals("unit_time"),
                    "covariates": _vals("none"),
                    "vce": _vals("analytic"),
                    "weights": _vals("none"),
                    "horizon": _vals("horizons"),
                },
                ("vcov",),
                "every entry of the per-horizon covariance vs Stata "
                "did_imputation's e(V)",
                "sp.did_imputation(horizon=[...])",
            ),
            _Row(
                "T2",
                _BJS_94,
                {
                    "fe": _vals("custom"),
                    "covariates": _vals("none"),
                    "vce": _vals("analytic"),
                    "weights": _vals("none"),
                    "horizon": _vals("overall", "horizons"),
                },
                _EST_SE,
                "fe(countyreal region#year) (overall and four horizons) and "
                "fe(region year) vs Stata did_imputation; the estimates are "
                "also the dense least-squares imputation to 1e-10",
                "sp.did_imputation(fe=[...])",
            ),
            _Row(
                "T2",
                _BJS_94,
                {
                    "fe": _vals("unit_time"),
                    "covariates": _vals("controls", "mixed"),
                    "vce": _vals("analytic"),
                    "weights": _vals("none"),
                    "horizon": _vals("overall"),
                },
                _EST_SE,
                "a time-varying control, and a control next to timecontrols(), "
                "vs Stata did_imputation; next to unitcontrols() the file "
                "compares the SE at 1e-8 and holds the estimate to 5e-6 "
                "(Stata's iteration moved it 2.8e-6 between two runs), so "
                "that combination's estimate rests on the exact "
                "least-squares solution",
                "sp.did_imputation(controls=..., time_covariates=...)",
            ),
            _Row(
                "T2",
                _BJS_94,
                {
                    "fe": _vals("unit_time"),
                    "covariates": _vals("none"),
                    "vce": _vals("analytic"),
                    "weights": _vals("set"),
                    "horizon": _vals("overall", "horizons"),
                },
                _EST_SE,
                "[aw=w] overall and over four horizons vs Stata did_imputation",
                "sp.did_imputation(weights=...)",
            ),
            _Row(
                "T2",
                _BJS_94,
                {
                    "fe": _vals("custom"),
                    "covariates": _vals("controls"),
                    "vce": _vals("analytic"),
                    "weights": _vals("set"),
                    "horizon": _vals("overall"),
                },
                _EST_SE,
                "[aw=w] with fe(countyreal region#year) and controls(x) vs "
                "Stata did_imputation",
                "sp.did_imputation(weights=..., fe=[...], controls=...)",
            ),
        ),
        invariant={
            "vce": (_EST, "the covariance is computed after the imputation"),
        },
        note="The Stata rows for the Y(0)-model options were regenerated in "
        "double precision on 2026-10-02; in single precision Stata sat 2e-6 "
        "from the exact answer. Stata's estimates come from reghdfe's "
        "fixed-effect solver (tolerance 1e-8, not exposed) and sit up to "
        "9e-7 from the exact imputation, which StatsPAI equals. "
        "unitcontrols() stays a disclosure for the estimate (Stata's "
        "iteration does not settle below 1e-6 there; fixture 94's run with "
        "controls() next to it moved 2.8e-6 between two runs). The "
        "bootstrap SE has no reference row.",
    )
)

_DID2S_BASE = {
    "vce": _vals("analytic"),
    "weights": _vals("none"),
    "covariates": _vals("none"),
    "form": _vals("first_treat"),
}

_add(
    _Scope(
        "gardner_did",
        {
            "vce": ("analytic", "stage2", "bootstrap", "none"),
            "weights": ("none", "set"),
            "covariates": ("none", "set"),
            "event_study": ("off", "on"),
            "form": ("first_treat", "general"),
        },
        _x_did2s,
        (
            _Row(
                "T2",
                _RP + "test_stata_did_commands_parity.py",
                {
                    "vce": _vals("analytic"),
                    "weights": _vals("none", "set"),
                    "covariates": _vals("none", "set"),
                    "event_study": _vals("off"),
                    "form": _vals("general"),
                },
                _EST_SE,
                "the general form sp.gardner_did(treat=, fe=, second_stage=) vs "
                "Stata did2s in eight specifications: unit() demeaning, a "
                "region-by-year cell effect, a covariate, [aw=], a factor and a "
                "list of dummies in the second stage, a coarser cluster",
                "sp.gardner_did(treat=, fe=)",
            ),
            _Row(
                "T2",
                _R + "73_did2s.py",
                {**_DID2S_BASE, "event_study": _vals("off")},
                _EST_SE,
                "static two-stage ATT and its corrected two-stage clustered SE vs "
                "did2s::did2s and Stata did2s",
                "sp.gardner_did()",
            ),
            _Row(
                "T2",
                _RP + "test_event_study_vcov_R_parity.py",
                {**_DID2S_BASE, "event_study": _vals("on")},
                ("estimate", "se", "vcov"),
                "event-study coefficients and every entry of their covariance "
                "(diagonal and off-diagonal) vs did2s::did2s(i(rel_year))",
                "sp.gardner_did(event_study=True)",
            ),
            _Row(
                "T2",
                _RP + "test_gardner_stage2_R_parity.py",
                {**_DID2S_BASE, "vce": _vals("stage2"), "event_study": _vals("off")},
                _EST_SE,
                "the legacy second-stage-only SE vs the hand-run second stage: "
                "lm(y_tilde ~ D) with sandwich::vcovCL(type='HC1') on the "
                "county (Stata regress, vce(cluster)), first stage by dense lm",
                "sp.gardner_did(vce='stage2')",
            ),
        ),
        invariant={
            "vce": (_EST, "the covariance is computed after the two stages"),
        },
        note="vce='stage2' is the pre-correction convention and understates "
        "(it treats the first stage as known). Its reference row pins that "
        "the number is the hand-run second-stage SE it documents, not that "
        "it is a valid SE; the static form only, since the event-study "
        "headline under stage2 treats the horizons as independent and has "
        "no counterpart (its per-horizon SEs are compared in the same file). "
        "The bootstrap SE has no reference row; covariates and weights have "
        "one in the general form (treat= / fe=) only, one at a time.",
    )
)

_ES_OPT = _RP + "test_event_study_options_R_parity.py"

_TWFE_ES_RAN = {
    "adoption": _vals("single_date"),
    "window": _vals("-4_4"),
    "ref_period": _vals("-1"),
    "covariates": _vals("none"),
    "weights": _vals("none"),
    "cluster": _vals("unit"),
    "bins": _vals("none"),
    "intensity": _vals("none"),
    "absorb": _vals("none"),
    "crit_dist": _vals("normal"),
}

_add(
    _Scope(
        "event_study",
        {
            "adoption": ("single_date", "staggered"),
            "window": ("-4_4", "other"),
            "ref_period": ("-1", "other"),
            "covariates": ("none", "set"),
            "weights": ("none", "set"),
            "cluster": ("unit", "other"),
            "bins": ("none", "set"),
            "intensity": ("none", "set"),
            "absorb": ("none", "set"),
            "crit_dist": ("normal", "t_resid", "t_cluster"),
        },
        _x_twfe_es,
        (
            _Row(
                "T2",
                _R + "85_twfe_event_study.py",
                dict(_TWFE_ES_RAN),
                _EST_SE,
                "every lead and lag coefficient and its unit-clustered SE vs "
                "fixest::feols(i(rel, ref=-1)) and Stata reghdfe",
                "sp.event_study(window=(-4, 4), cluster='unit')",
            ),
            _Row(
                "T2",
                _RP + "test_event_study_vcov_R_parity.py",
                dict(_TWFE_ES_RAN),
                ("vcov",),
                "every entry of the event-time covariance (off-diagonal blocks "
                "included) vs fixest::feols",
                "sp.event_study(window=(-4, 4), cluster='unit')",
            ),
            _Row(
                "T2",
                _RP + "test_event_study_grid_R_parity.py",
                {
                    "adoption": _vals("single_date", "staggered"),
                    "window": _vals("-4_4", "other"),
                    "ref_period": _vals("-1", "other"),
                    "covariates": _vals("none", "set"),
                    "weights": _vals("none", "set"),
                    "cluster": _vals("unit"),
                    "bins": _vals("none"),
                    "intensity": _vals("none"),
                    "absorb": _vals("none"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "the full 32-cell grid (adoption x window (-4, 4) / (-3, 5) x "
                "reference -1 / -2 x a time-varying covariate x unit-level "
                "weights) vs fixest::feols: every coefficient and SE to 1e-10",
                "sp.event_study(covariates=..., weights=...)",
            ),
            _Row(
                "T2",
                _ES_OPT,
                {
                    "adoption": _vals("staggered"),
                    "window": _vals("-4_4", "other"),
                    "ref_period": _vals("-1", "other"),
                    "covariates": _vals("none"),
                    "weights": _vals("none"),
                    "cluster": _vals("unit"),
                    "bins": _vals("none"),
                    "intensity": _vals("none"),
                    "absorb": _vals("none"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "staggered adoption with eight other windows (narrower, wider "
                "than the data, asymmetric, no leads) and five other "
                "references (-2, the clipped endpoint -4, a span, both "
                "interval forms), vs fixest::feols",
                "sp.event_study(...)",
            ),
            _Row(
                "T2",
                _ES_OPT,
                {
                    "adoption": _vals("staggered"),
                    "window": _vals("-4_4"),
                    "ref_period": _vals("-1"),
                    "covariates": _vals("none"),
                    "weights": _vals("none", "set"),
                    "cluster": _vals("unit"),
                    "bins": _vals("none"),
                    "intensity": _vals("none"),
                    "absorb": _vals("none"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "staggered adoption, unweighted and with weights constant "
                "within a unit or varying over time, on the balanced and an "
                "unbalanced panel, vs fixest::feols",
                "sp.event_study(...)",
            ),
            _Row(
                "T2",
                _ES_OPT,
                {
                    "adoption": _vals("staggered"),
                    "window": _vals("-4_4"),
                    "ref_period": _vals("-1"),
                    "covariates": _vals("set"),
                    "weights": _vals("set"),
                    "cluster": _vals("unit"),
                    "bins": _vals("none"),
                    "intensity": _vals("none"),
                    "absorb": _vals("none"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "staggered adoption with a covariate and weights, vs fixest::feols",
                "sp.event_study(...)",
            ),
            _Row(
                "T2",
                _ES_OPT,
                {
                    "adoption": _vals("staggered"),
                    "window": _vals("-4_4"),
                    "ref_period": _vals("-1"),
                    "covariates": _vals("none", "set"),
                    "weights": _vals("none"),
                    "cluster": _vals("other"),
                    "bins": _vals("none"),
                    "intensity": _vals("none"),
                    "absorb": _vals("none"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "clustered by state (coarser than the unit) and by period, "
                "and by state with a covariate on an unbalanced panel, vs "
                "fixest::feols",
                "sp.event_study(...)",
            ),
            _Row(
                "T2",
                _ES_OPT,
                {
                    "adoption": _vals("single_date"),
                    "window": _vals("-4_4"),
                    "ref_period": _vals("-1"),
                    "covariates": _vals("none"),
                    "weights": _vals("set"),
                    "cluster": _vals("other"),
                    "bins": _vals("none"),
                    "intensity": _vals("none"),
                    "absorb": _vals("none"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "a single adoption date, weighted, clustered by state, vs "
                "fixest::feols",
                "sp.event_study(...)",
            ),
            _Row(
                "T2",
                _ES_OPT,
                {
                    "adoption": _vals("staggered"),
                    "window": _vals("other"),
                    "ref_period": _vals("other"),
                    "covariates": _vals("none"),
                    "weights": _vals("none"),
                    "cluster": _vals("unit"),
                    "bins": _vals("set"),
                    "intensity": _vals("none"),
                    "absorb": _vals("none", "set"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "bins of width 2 and 3 with a bin-aligned span and an "
                "interval reference, and with a region-by-time effect "
                "absorbed, vs fixest::feols",
                "sp.event_study(...)",
            ),
            _Row(
                "T2",
                _ES_OPT,
                {
                    "adoption": _vals("single_date"),
                    "window": _vals("other"),
                    "ref_period": _vals("other"),
                    "covariates": _vals("set"),
                    "weights": _vals("none"),
                    "cluster": _vals("unit"),
                    "bins": _vals("set"),
                    "intensity": _vals("none"),
                    "absorb": _vals("none"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "bins of width 2 on a single adoption date with a covariate, "
                "vs fixest::feols",
                "sp.event_study(...)",
            ),
            _Row(
                "T2",
                _ES_OPT,
                {
                    "adoption": _vals("single_date"),
                    "window": _vals("-4_4"),
                    "ref_period": _vals("-1"),
                    "covariates": _vals("none"),
                    "weights": _vals("none", "set"),
                    "cluster": _vals("unit"),
                    "bins": _vals("none"),
                    "intensity": _vals("set"),
                    "absorb": _vals("none"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "intensity x period regressors from a common adoption date, "
                "unweighted and weighted, vs fixest::feols",
                "sp.event_study(...)",
            ),
            _Row(
                "T2",
                _ES_OPT,
                {
                    "adoption": _vals("single_date"),
                    "window": _vals("other"),
                    "ref_period": _vals("-1"),
                    "covariates": _vals("none"),
                    "weights": _vals("none"),
                    "cluster": _vals("other"),
                    "bins": _vals("none"),
                    "intensity": _vals("set"),
                    "absorb": _vals("none"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "intensity from a common date, window (-6, 5), clustered by "
                "state, vs fixest::feols",
                "sp.event_study(...)",
            ),
            _Row(
                "T2",
                _ES_OPT,
                {
                    "adoption": _vals("staggered"),
                    "window": _vals("-4_4"),
                    "ref_period": _vals("-1"),
                    "covariates": _vals("none"),
                    "weights": _vals("none"),
                    "cluster": _vals("unit"),
                    "bins": _vals("none"),
                    "intensity": _vals("set"),
                    "absorb": _vals("none"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "intensity on a staggered adoption column, vs fixest::feols",
                "sp.event_study(...)",
            ),
            _Row(
                "T2",
                _ES_OPT,
                {
                    "adoption": _vals("staggered"),
                    "window": _vals("-4_4"),
                    "ref_period": _vals("-1"),
                    "covariates": _vals("none", "set"),
                    "weights": _vals("none"),
                    "cluster": _vals("unit"),
                    "bins": _vals("none"),
                    "intensity": _vals("none"),
                    "absorb": _vals("set"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "region-by-time and state-by-time effects absorbed, the "
                "latter with a covariate, vs fixest::feols",
                "sp.event_study(...)",
            ),
            _Row(
                "T2",
                _ES_OPT,
                {
                    "adoption": _vals("single_date"),
                    "window": _vals("-4_4"),
                    "ref_period": _vals("-1"),
                    "covariates": _vals("none"),
                    "weights": _vals("set"),
                    "cluster": _vals("unit"),
                    "bins": _vals("none"),
                    "intensity": _vals("none"),
                    "absorb": _vals("set"),
                    "crit_dist": _vals("normal"),
                },
                _EST_SE,
                "a region-by-time effect absorbed on a single adoption date, "
                "weighted, vs fixest::feols",
                "sp.event_study(...)",
            ),
            _Row(
                "T2",
                _ES_CRIT,
                {
                    "adoption": _vals("staggered"),
                    "window": _vals("other"),
                    "ref_period": _vals("-1"),
                    "covariates": _vals("none"),
                    "weights": _vals("set"),
                    "cluster": _vals("unit", "other"),
                    "bins": _vals("none"),
                    "intensity": _vals("none"),
                    "absorb": _vals("none"),
                    "crit_dist": _vals("t_cluster"),
                },
                ("estimate", "se", "interval"),
                "parity='stata' / 'r' on the castle-doctrine panel, window "
                "(-5, 4), weighted, clustered by state and by region, alpha "
                "0.05 and 0.10: every lead and lag SE, p-value and confidence "
                "limit vs Stata reghdfe and R fixest (t(G-1)), and the "
                "headline average vs lincom, to 1e-10",
                "sp.event_study(",
            ),
            _Row(
                "T2",
                _ES_CRIT,
                {
                    "adoption": _vals("staggered"),
                    "window": _vals("other"),
                    "ref_period": _vals("-1"),
                    "covariates": _vals("none"),
                    "weights": _vals("none"),
                    "cluster": _vals("unit"),
                    "bins": _vals("none"),
                    "intensity": _vals("none"),
                    "absorb": _vals("none"),
                    "crit_dist": _vals("t_cluster"),
                },
                ("estimate", "se", "interval"),
                "parity='stata' / 'r' on the no-fault divorce panel, window "
                "(-10, 15), clustered by state: every lead and lag SE, "
                "p-value and confidence limit vs Stata reghdfe and R fixest "
                "(t(40)), to 1e-10",
                "sp.event_study(",
            ),
        ),
        note="On a staggered panel this saturated dynamic TWFE regression is "
        "the contaminated estimator the heterogeneity-robust methods "
        "replace: the staggered rows pin that sp.event_study computes the "
        "regression it documents (every coefficient and SE as fixest::feols "
        "computes it), not that its coefficients are clean dynamic effects. "
        "'other' windows and references are credited on the evidence of "
        "thirteen other windows and six other reference specifications "
        "(-2, -3, the clipped endpoint -4, a span, both interval forms), "
        "not one. With absorb= the SE follows reghdfe's count of redundant "
        "fixed effects; fixest's nested K does not detect that a^time spans "
        "the time effects, so its reference fit drops the redundant term "
        "(same coefficients, exact K). Before 2026-10-15 unbalanced panels "
        "and weights varying within a unit were demeaned by one pass of "
        "unit and time means, which is not the two-way projection; the "
        "option fixture pins the corrected fit. crit_dist is the "
        "distribution of the p-values and intervals: 'normal' is the default "
        "before 1.41, 't_cluster' / 't_resid' are t(G-1) / t(N-K) "
        "(parity='stata' / 'r' or crit_dist='t'); rows that compare only "
        "coefficients and SEs ran the default.",
        invariant={
            "crit_dist": (
                ("estimate", "se", "vcov", "joint_test"),
                "the critical value turns an SE into a p-value and an "
                "interval; coefficients, SEs and the pre-trend test are "
                "computed before it is chosen (asserted equal across parity= "
                "in tests/reference_parity/test_es_crit_parity.py)",
            ),
        },
        defaults={"crit_dist": "normal"},
    )
)

_ETWFE_BASE = {
    "family": _vals("linear"),
    "controls": _vals("none"),
    "xvar": _vals("none"),
    "weights": _vals("none"),
    "agg_weights": _vals("estimation"),
}
_ETWFE_NOTYET_RCS = {**_ETWFE_BASE, "panel": _vals("false"), "cgroup": _vals("notyet")}
_ETWFE_NEVER = {**_ETWFE_BASE, "panel": _vals("true"), "cgroup": _vals("nevertreated")}
_ETWFE_NOTYET_PANEL = {**_ETWFE_BASE, "panel": _vals("true"), "cgroup": _vals("notyet")}
_ETWFE_OPT = _RP + "test_etwfe_options_parity.py"
_ETWFE_OPT_PANEL = {
    **_ETWFE_BASE,
    "panel": _vals("true"),
    "cgroup": _vals("notyet", "nevertreated"),
}
_ETWFE_EMFX_R = "tests/test_etwfe_emfx_reference.py"

_add(
    _Scope(
        "etwfe",
        {
            "family": ("linear", "glm"),
            "panel": ("true", "false"),
            "cgroup": ("notyet", "nevertreated"),
            "controls": ("none", "set"),
            "xvar": ("none", "set"),
            "weights": ("none", "set"),
            "agg_weights": ("estimation", "unit"),
        },
        _x_etwfe,
        (
            _Row(
                "T2",
                _R + "17_etwfe.py",
                dict(_ETWFE_NOTYET_RCS),
                _EST,
                "simple ATT (treated-observation-weighted) vs etwfe::emfx("
                "type='simple') and Stata jwdid",
                "sp.etwfe(panel=False)",
            ),
            _Row(
                "T2",
                _R + "17_etwfe.py",
                dict(_ETWFE_NEVER),
                _EST,
                "simple ATT with never-treated controls vs etwfe(cgroup='never') "
                "and Stata jwdid, never",
                "sp.etwfe(cgroup='nevertreated')",
            ),
            _Row(
                "T4",
                _R + "17_etwfe.py",
                dict(_ETWFE_NOTYET_RCS),
                ("se",),
                "delta-method SE of the simple ATT: 2.4e-6 from etwfe::emfx, whose "
                "Jacobian is a forward difference (StatsPAI's is analytic), and "
                "6e-4 from Stata jwdid, which leaves absorbed unit effects out of "
                "K; inside the registered 1e-3 budget, outside same-byte parity",
                "sp.etwfe(panel=False)",
            ),
            _Row(
                "T4",
                _R + "17_etwfe.py",
                dict(_ETWFE_NEVER),
                ("se",),
                "delta-method SE of the simple ATT: 2.0e-6 from etwfe::emfx "
                "(forward-difference Jacobian in the reference) and 6e-4 from "
                "Stata jwdid (K convention)",
                "sp.etwfe(cgroup='nevertreated')",
            ),
            _Row(
                "T2",
                _RP + "test_event_study_vcov_R_parity.py",
                dict(_ETWFE_NOTYET_RCS),
                ("vcov",),
                "event-time effects and every entry of their covariance vs "
                "etwfe::emfx(type='event') with a Richardson-extrapolated Jacobian",
                "sp.etwfe(panel=False)",
            ),
            _Row(
                "T2",
                _RP + "test_event_study_vcov_R_parity.py",
                dict(_ETWFE_NEVER),
                ("vcov",),
                "event-time effects (leads included) and every entry of their "
                "covariance vs etwfe::emfx(type='event'), never-treated controls",
                "sp.etwfe(cgroup='nevertreated')",
            ),
            _Row(
                "T2",
                _ATTACH,
                dict(_ETWFE_NOTYET_PANEL),
                ("estimate", "vcov"),
                "the default call is bit-identical (estimate, SE, event-study "
                "covariance) to the panel=False fit that module 17 and the "
                "covariance test pin, on module 17's bytes",
                "sp.etwfe()",
            ),
            _Row(
                "T4",
                _ATTACH,
                dict(_ETWFE_NOTYET_PANEL),
                ("se",),
                "same SE as the panel=False fit, so the same convention "
                "difference from etwfe::emfx and jwdid applies",
                "sp.etwfe()",
            ),
            _Row(
                "T2",
                "tests/test_etwfe_weights_reference.py",
                {
                    "family": _vals("linear"),
                    "panel": _vals("true"),
                    "cgroup": _vals("notyet", "nevertreated"),
                    "controls": _vals("none"),
                    "xvar": _vals("none"),
                    "weights": _vals("set"),
                    "agg_weights": _vals("estimation"),
                },
                _EST,
                "weighted simple / event / group / calendar ATTs vs Stata jwdid [pw=] "
                "+ estat on two real panels (1e-6)",
                "sp.etwfe(panel, ..., weights=...)",
            ),
            _Row(
                "T4",
                "tests/test_etwfe_weights_reference.py",
                {
                    "family": _vals("linear"),
                    "panel": _vals("true"),
                    "cgroup": _vals("notyet", "nevertreated"),
                    "controls": _vals("none"),
                    "xvar": _vals("none"),
                    "weights": _vals("set"),
                    "agg_weights": _vals("estimation"),
                },
                ("se",),
                "Stata's SEs are rebuilt to 1e-9 after sqrt((n - K)/(n - K_Stata)): "
                "reghdfe leaves the cohort levels nested in the unit cluster out of K, "
                "fixest counts them",
                "sp.etwfe(panel, ..., weights=...)",
            ),
            _Row(
                "T2",
                "tests/test_etwfe_weights_reference.py",
                {
                    "family": _vals("linear"),
                    "panel": _vals("true"),
                    "cgroup": _vals("notyet", "nevertreated"),
                    "controls": _vals("none"),
                    "xvar": _vals("none"),
                    "weights": _vals("set"),
                    "agg_weights": _vals("unit"),
                },
                _EST,
                "weighted ATTs under R emfx's aggregation rule vs etwfe::emfx on the "
                "same two panels (1e-6)",
                "sp.etwfe(panel, ..., weights=...)",
            ),
            _Row(
                "T4",
                "tests/test_etwfe_weights_reference.py",
                {
                    "family": _vals("linear"),
                    "panel": _vals("true"),
                    "cgroup": _vals("notyet", "nevertreated"),
                    "controls": _vals("none"),
                    "xvar": _vals("none"),
                    "weights": _vals("set"),
                    "agg_weights": _vals("unit"),
                },
                ("se",),
                "SEs within 1e-5 of etwfe::emfx; R's carry marginaleffects' "
                "finite-difference Jacobian noise (up to 3.3e-6)",
                "sp.etwfe(panel, ..., weights=...)",
            ),
            _Row(
                "T2",
                _ETWFE_EMFX_R,
                dict(_ETWFE_OPT_PANEL),
                _EST_SE,
                "simple / event / group / calendar ATTs vs etwfe::emfx on the "
                "castle-doctrine panel: estimates to 1e-9, SEs to 1e-6 (R's "
                "forward-difference Jacobian; R's own sqrt(w' V w) agrees to "
                "1e-13)",
                "sp.etwfe(castle, ...)",
            ),
            _Row(
                "T2",
                _ETWFE_OPT,
                {**_ETWFE_OPT_PANEL, "agg_weights": _vals("estimation", "unit")},
                _EST_SE,
                "every emfx aggregation (simple / group / calendar / event, "
                "leads included) vs etwfe::emfx with a Richardson Jacobian: "
                "estimates to 1e-9, SEs to 1e-7 (observed 3.1e-9); unweighted, "
                "where the two agg_weights rules coincide",
                "sp.etwfe(panel, **args)",
            ),
            _Row(
                "T2",
                _ETWFE_OPT,
                {
                    **_ETWFE_BASE,
                    "panel": _vals("false"),
                    "cgroup": _vals("notyet"),
                },
                _EST_SE,
                "panel=False without cluster= (HC1) vs etwfe(vcov='hetero') + "
                "emfx, every aggregation: estimates to 1e-9, SEs to 1e-7",
                "sp.etwfe(panel, **args)",
            ),
            _Row(
                "T2",
                _ETWFE_OPT,
                {
                    **_ETWFE_OPT_PANEL,
                    "weights": _vals("set"),
                    "agg_weights": _vals("estimation", "unit"),
                },
                _EST_SE,
                "weighted fit vs etwfe(weights = ~w): agg_weights='unit' vs "
                "emfx(), agg_weights='estimation' vs emfx(wts = 'w'); every "
                "aggregation, estimates to 1e-9, SEs to 1e-7",
                "sp.etwfe(panel, **args)",
            ),
            _Row(
                "T2",
                _ETWFE_OPT,
                {**_ETWFE_OPT_PANEL, "xvar": _vals("set")},
                _EST_SE,
                "continuous moderator vs etwfe(xvar = xv) + emfx(by_xvar = "
                "FALSE), every aggregation, plus the per-cohort moderator "
                "slopes vs fixest's coefficients and vcov; a categorical "
                "moderator vs the same per-level design fitted by fixest "
                "(etwfe 0.6.2 sums the level slopes of the period term)",
                "sp.etwfe(panel, **args)",
            ),
            _Row(
                "T2",
                _ETWFE_OPT,
                {**_ETWFE_OPT_PANEL, "controls": _vals("set")},
                _EST_SE,
                "additive time-varying control (jwdid's exovar(); R etwfe "
                "interacts controls with the cells, another model) vs fixest "
                "on etwfe's cell basis + marginaleffects::slopes as emfx calls "
                "it, and vs Stata jwdid, exovar() without ivar(): estimates "
                "and SEs to 1e-9",
                "sp.etwfe(panel, **args)",
            ),
            _Row(
                "T2",
                _ETWFE_OPT,
                {**_ETWFE_OPT_PANEL, "family": _vals("glm")},
                _EST_SE,
                "family='poisson' / 'logit' vs etwfe(family=) + emfx on both "
                "scales (glm.tol = 1e-12, Richardson Jacobian): estimates to "
                "1e-9, SEs to 1e-7",
                "sp.etwfe(panel, **args)",
            ),
            _Row(
                "T2",
                _ETWFE_OPT,
                {
                    **_ETWFE_BASE,
                    "family": _vals("glm"),
                    "panel": _vals("true"),
                    "cgroup": _vals("notyet"),
                    "xvar": _vals("set"),
                },
                _EST_SE,
                "Poisson with a continuous moderator vs etwfe(family = "
                "'poisson', xvar = xv) + emfx, both scales",
                "sp.etwfe(panel, **args)",
            ),
        ),
        note="Every option value has a same-byte (T2) row for the estimate "
        "and the SE in "
        + _ETWFE_OPT
        + ", against R etwfe with marginaleffects' Richardson Jacobian (and, "
        "for the additive control, Stata jwdid). Module 17's headline SE stays "
        "a disclosure row: there R's forward-difference Jacobian leaves 2e-6 "
        "and Stata counts K differently (docs/dev/r_parity_tolerances.md). The "
        "default call (panel=True, not-yet-treated controls) also inherits the "
        "panel=False rows through a bit-identity test in "
        + _ATTACH
        + ". Not run: panel=False with cgroup='nevertreated' (raises), weights "
        "with xvar (raises), and controls or weights combined with xvar or a "
        "GLM family.",
    )
)

_ETWFE_GLM_RAN = {
    "family": _vals("poisson", "logit"),
    "cgroup": _vals("notyet"),
    "fe": _vals("cohort"),
    "hettype": _vals("timecohort"),
    "scale": _vals("response"),
    "response_se": _vals("profile"),
    "controls": _vals("none"),
    "xvar": _vals("none"),
    "weights": _vals("none"),
}
_ETWFE_GLM = _RP + "test_etwfe_glm_parity.py"

#: The nonlinear branch of sp.etwfe has its own option axes, so it has its
#: own map; sp.validation_scope routes a fit with family='poisson'/'logit'
#: here (``function="etwfe_glm"`` in a by-name query).
_add(
    _Scope(
        "etwfe_glm",
        {
            "family": ("poisson", "logit", "gaussian"),
            "cgroup": ("notyet", "nevertreated"),
            "fe": ("cohort", "unit"),
            "hettype": ("timecohort", "time", "cohort", "event", "twfe"),
            "scale": ("response", "link"),
            "response_se": ("profile", "margins", "unconditional"),
            "controls": ("none", "set"),
            "xvar": ("none", "set"),
            "weights": ("none", "set"),
        },
        _x_etwfe_glm,
        (
            _Row(
                "T2",
                _ETWFE_GLM,
                dict(_ETWFE_GLM_RAN),
                _EST,
                "average marginal effect on the response scale (simple and by "
                "event time) vs etwfe::emfx on the same panel, family poisson "
                "and binomial",
                "sp.etwfe(family='poisson'|'logit')",
            ),
            _Row(
                "T4",
                _ETWFE_GLM,
                dict(_ETWFE_GLM_RAN),
                ("se",),
                "clustered delta-method SE: about 1e-5 from etwfe::emfx run "
                "with fixest's defaults (the test budget is 1e-4). Not a "
                "convention: the reference's forward-difference Jacobian, "
                "glm.tol = 1e-8 and a bread taken at the penultimate IRLS "
                "iterate; with those removed the SEs agree to 1e-9 ("
                + _RP
                + "test_etwfe_options_parity.py)",
                "sp.etwfe(family='poisson'|'logit')",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_options_parity.py",
                {
                    **_ETWFE_GLM_RAN,
                    "cgroup": _vals("notyet", "nevertreated"),
                    "scale": _vals("response", "link"),
                },
                _EST_SE,
                "every emfx aggregation on both scales vs etwfe(family=) + emfx "
                "with a Richardson Jacobian, glm.tol = 1e-12 and a restart from "
                "the solution: estimates to 1e-9, SEs to 1e-7 (observed 9.4e-10)",
                "sp.etwfe(panel, **args)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_options_parity.py",
                {
                    **_ETWFE_GLM_RAN,
                    "family": _vals("poisson"),
                    "scale": _vals("response", "link"),
                    "xvar": _vals("set"),
                },
                _EST_SE,
                "Poisson with a continuous moderator (R etwfe's design) vs "
                "etwfe(family = 'poisson', xvar = xv) + emfx, both scales",
                "sp.etwfe(panel, **args)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_options_parity.py",
                {
                    **_ETWFE_GLM_RAN,
                    "cgroup": _vals("notyet", "nevertreated"),
                    "response_se": _vals("unconditional"),
                },
                _EST,
                "response-scale aggregates under response_se='unconditional' vs "
                "Stata jwdid, method(poisson|logit) + estat (no ivar): 1e-9",
                "sp.etwfe(panel, **args)",
            ),
            _Row(
                "T4",
                _RP + "test_etwfe_options_parity.py",
                {
                    **_ETWFE_GLM_RAN,
                    "cgroup": _vals("notyet", "nevertreated"),
                    "response_se": _vals("unconditional"),
                },
                ("se",),
                "unconditional SEs vs Stata estat ..., vce(unconditional), every "
                "aggregation: equal to 1e-9 after sqrt((n - K) / (n - 1)) -- "
                "Stata's ML commands scale the clustered sandwich by G/(G-1) "
                "only, fixest and StatsPAI also by (n - 1)/(n - K); R has no "
                "unconditional variance",
                "sp.etwfe(panel, **args)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_linear_jwdid_parity.py",
                {
                    "family": _vals("gaussian"),
                    "cgroup": _vals("notyet"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort", "time", "cohort", "event", "twfe"),
                    "scale": _vals("link"),
                    "response_se": _vals("profile"),
                    "controls": _vals("none"),
                    "xvar": _vals("none"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "linear ETWFE with unit effects vs Stata jwdid (reghdfe): simple and "
                "event-time ATTs, estimates to 1e-10 and SEs to 1e-9; every hettype",
                "sp.etwfe(df, y, group, time, first_treat, fe='unit' | hettype=...)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_linear_jwdid_parity.py",
                {
                    "family": _vals("gaussian"),
                    "cgroup": _vals("nevertreated"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort", "cohort", "event"),
                    "scale": _vals("link"),
                    "response_se": _vals("profile"),
                    "controls": _vals("none"),
                    "xvar": _vals("none"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "linear ETWFE with unit effects vs Stata jwdid (reghdfe): simple and "
                "event-time ATTs, estimates to 1e-10 and SEs to 1e-9; never-treated "
                "controls",
                "sp.etwfe(df, y, group, time, first_treat, fe='unit' | hettype=...)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_linear_jwdid_parity.py",
                {
                    "family": _vals("gaussian"),
                    "cgroup": _vals("notyet"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort", "event"),
                    "scale": _vals("link"),
                    "response_se": _vals("profile"),
                    "controls": _vals("none"),
                    "xvar": _vals("set"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "linear ETWFE with unit effects vs Stata jwdid (reghdfe): simple and "
                "event-time ATTs, estimates to 1e-10 and SEs to 1e-9; with a "
                "categorical or continuous moderator, and per-level ATTs",
                "sp.etwfe(df, y, group, time, first_treat, fe='unit' | hettype=...)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_linear_jwdid_parity.py",
                {
                    "family": _vals("gaussian"),
                    "cgroup": _vals("nevertreated"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort"),
                    "scale": _vals("link"),
                    "response_se": _vals("profile"),
                    "controls": _vals("none"),
                    "xvar": _vals("set"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "linear ETWFE with unit effects vs Stata jwdid (reghdfe): simple and "
                "event-time ATTs, estimates to 1e-10 and SEs to 1e-9; moderator with "
                "never-treated controls",
                "sp.etwfe(df, y, group, time, first_treat, fe='unit' | hettype=...)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_poisson_jwdid_parity.py",
                {
                    "family": _vals("poisson"),
                    "cgroup": _vals("notyet"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort", "time", "cohort", "event", "twfe"),
                    "scale": _vals("response"),
                    "response_se": _vals("profile"),
                    "controls": _vals("none"),
                    "xvar": _vals("none"),
                    "weights": _vals("none"),
                },
                _EST,
                "Poisson ETWFE with unit effects vs Stata jwdid, method(ppmlhdfe): "
                "link-scale estimates to 1e-9 and response-scale to 1e-6; every "
                "hettype",
                "sp.etwfe(panel, family='poisson', fe='unit', ...)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_poisson_jwdid_parity.py",
                {
                    "family": _vals("poisson"),
                    "cgroup": _vals("nevertreated"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort", "cohort", "event"),
                    "scale": _vals("response"),
                    "response_se": _vals("profile"),
                    "controls": _vals("none"),
                    "xvar": _vals("none"),
                    "weights": _vals("none"),
                },
                _EST,
                "Poisson ETWFE with unit effects vs Stata jwdid, method(ppmlhdfe): "
                "link-scale estimates to 1e-9 and response-scale to 1e-6; "
                "never-treated controls",
                "sp.etwfe(panel, family='poisson', fe='unit', ...)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_poisson_jwdid_parity.py",
                {
                    "family": _vals("poisson"),
                    "cgroup": _vals("notyet"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort", "event"),
                    "scale": _vals("response"),
                    "response_se": _vals("profile"),
                    "controls": _vals("none"),
                    "xvar": _vals("set"),
                    "weights": _vals("none"),
                },
                _EST,
                "Poisson ETWFE with unit effects vs Stata jwdid, method(ppmlhdfe): "
                "link-scale estimates to 1e-9 and response-scale to 1e-6; with a "
                "categorical or continuous moderator, and per-level ATTs",
                "sp.etwfe(panel, family='poisson', fe='unit', ...)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_poisson_jwdid_parity.py",
                {
                    "family": _vals("poisson"),
                    "cgroup": _vals("nevertreated"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort"),
                    "scale": _vals("response"),
                    "response_se": _vals("profile"),
                    "controls": _vals("none"),
                    "xvar": _vals("set"),
                    "weights": _vals("none"),
                },
                _EST,
                "Poisson ETWFE with unit effects vs Stata jwdid, method(ppmlhdfe): "
                "link-scale estimates to 1e-9 and response-scale to 1e-6; moderator "
                "with never-treated controls",
                "sp.etwfe(panel, family='poisson', fe='unit', ...)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_poisson_jwdid_parity.py",
                {
                    "family": _vals("poisson"),
                    "cgroup": _vals("notyet"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort"),
                    "scale": _vals("response"),
                    "response_se": _vals("margins"),
                    "controls": _vals("none"),
                    "xvar": _vals("none", "set"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "response-scale ATTs and SEs under response_se='margins' vs Stata "
                "jwdid estat (margins), SEs to 1e-6",
                "sp.etwfe(panel, family='poisson', fe='unit', response_se='margins')",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_poisson_jwdid_parity.py",
                {
                    "family": _vals("poisson"),
                    "cgroup": _vals("notyet"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort", "time", "cohort", "event", "twfe"),
                    "scale": _vals("link"),
                    "response_se": _vals("profile"),
                    "controls": _vals("none"),
                    "xvar": _vals("none"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "Poisson ETWFE on the link scale vs Stata jwdid, "
                "method(ppmlhdfe): estimates to 1e-9, SEs to 1e-7; every hettype",
                "sp.etwfe(df, family='poisson', fe='unit', scale='link', ...)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_poisson_jwdid_parity.py",
                {
                    "family": _vals("poisson"),
                    "cgroup": _vals("nevertreated"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort", "cohort", "event"),
                    "scale": _vals("link"),
                    "response_se": _vals("profile"),
                    "controls": _vals("none"),
                    "xvar": _vals("none"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "Poisson ETWFE on the link scale vs Stata jwdid, "
                "method(ppmlhdfe): estimates to 1e-9, SEs to 1e-7; never-treated "
                "controls",
                "sp.etwfe(df, family='poisson', fe='unit', scale='link', ...)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_poisson_jwdid_parity.py",
                {
                    "family": _vals("poisson"),
                    "cgroup": _vals("notyet"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort", "event"),
                    "scale": _vals("link"),
                    "response_se": _vals("profile"),
                    "controls": _vals("none"),
                    "xvar": _vals("set"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "Poisson ETWFE on the link scale vs Stata jwdid, "
                "method(ppmlhdfe): estimates to 1e-9, SEs to 1e-7; with a moderator",
                "sp.etwfe(df, family='poisson', fe='unit', scale='link', ...)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_poisson_jwdid_parity.py",
                {
                    "family": _vals("poisson"),
                    "cgroup": _vals("nevertreated"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort"),
                    "scale": _vals("link"),
                    "response_se": _vals("profile"),
                    "controls": _vals("none"),
                    "xvar": _vals("set"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "Poisson ETWFE on the link scale vs Stata jwdid, "
                "method(ppmlhdfe): estimates to 1e-9, SEs to 1e-7; moderator, "
                "never-treated controls",
                "sp.etwfe(df, family='poisson', fe='unit', scale='link', ...)",
            ),
            _Row(
                "T2",
                _RP + "test_etwfe_linear_jwdid_parity.py",
                {
                    "family": _vals("gaussian"),
                    "cgroup": _vals("notyet", "nevertreated"),
                    "fe": _vals("unit"),
                    "hettype": _vals("timecohort"),
                    "scale": _vals("link"),
                    "response_se": _vals("profile"),
                    "controls": _vals("set"),
                    "xvar": _vals("none"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "additive time-varying control with unit effects vs Stata "
                "jwdid, exovar(): simple ATT to 1e-10, SE to 1e-9",
                "sp.etwfe(df, y, group, time, first_treat, fe='unit' | hettype=...)",
            ),
        ),
        note="Cohort effects (R etwfe's design): estimates and SEs are "
        "same-byte against R etwfe for both families, both control groups and "
        "both scales (test_etwfe_options_parity.py); the older 1e-5 SE gap of "
        "test_etwfe_glm_parity.py is reference noise, not a convention. "
        "response_se='unconditional' has only Stata as a reference, equal "
        "after one documented small-sample factor. Unit effects are compared "
        "with Stata jwdid. weights= raises on the nonlinear families (no row); "
        "controls on the nonlinear families and the logit with xvar are not run.",
    )
)

_add(
    _Scope(
        "rdrobust",
        {
            "design": ("sharp", "fuzzy", "kink"),
            "bwselect": (
                "mserd",
                "msetwo",
                "msesum",
                "msecomb1",
                "msecomb2",
                "cerrd",
                "certwo",
                "cersum",
                "cercomb1",
                "cercomb2",
                "cct",
                "manual",
            ),
            "kernel": ("triangular", "epanechnikov", "uniform"),
            "p": ("0", "1", "2", "3", "4"),
            "vce": ("nn", "hc0", "hc1", "hc2", "hc3", "cluster"),
            "covariates": ("none", "set"),
            "code_path": ("native", "port"),
            "weights": ("none", "set"),
        },
        _x_rd,
        (
            _Row(
                "T2",
                _R + "06_rd.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "conventional / robust estimates, SEs and h, b vs rdrobust (R and "
                "Stata)",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _R + "06_rd.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("manual"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "estimates and SEs at a forced common bandwidth h = 15",
                "sp.rdrobust(h=15)",
            ),
            _Row(
                "B",
                _B,
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _COV,
                "robust bias-corrected CI coverage on a known-jump DGP",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_weights_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("set"),
                },
                _EST_SE,
                "weighted estimates, SEs and h vs R rdrobust(weights=)",
                "sp.rdrobust(weights=)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_weights_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("msetwo"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("hc1"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("set"),
                },
                _EST_SE,
                "weighted msetwo / hc1 vs R rdrobust",
                "sp.rdrobust(weights=)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_weights_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("cerrd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("2"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("set"),
                },
                _EST_SE,
                "weighted cerrd, p = 2 vs R rdrobust",
                "sp.rdrobust(weights=)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_weights_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("nn"),
                    "covariates": _vals("set"),
                    "code_path": _vals("native"),
                    "weights": _vals("set"),
                },
                _EST_SE,
                "weighted, covariate-adjusted vs R rdrobust",
                "sp.rdrobust(weights=)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_weights_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("cluster"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("set"),
                },
                _EST_SE,
                "weighted, clustered vs R rdrobust",
                "sp.rdrobust(weights=)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_weights_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("uniform"),
                    "p": _vals("1"),
                    "vce": _vals("hc3"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("set"),
                },
                _EST_SE,
                "weighted hc3 / uniform kernel vs R rdrobust",
                "sp.rdrobust(weights=)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_weights_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("manual"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("set"),
                },
                _EST_SE,
                "weighted at fixed h, b vs R rdrobust",
                "sp.rdrobust(weights=)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_weights_parity.py",
                {
                    "design": _vals("fuzzy"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("set"),
                },
                _EST_SE,
                "weighted fuzzy RD vs R rdrobust",
                "sp.rdrobust(weights=)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_weights_parity.py",
                {
                    "design": _vals("fuzzy"),
                    "bwselect": _vals("msecomb2"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("set"),
                },
                _EST_SE,
                "weighted fuzzy RD, msecomb2 vs R rdrobust",
                "sp.rdrobust(weights=)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_weights_parity.py",
                {
                    "design": _vals("fuzzy"),
                    "bwselect": _vals("msecomb2"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "fuzzy RD, msecomb2 (comb recursion now keeps the first stage)",
                "sp.rdrobust(weights=)",
            ),
            _Row(
                "T2",
                _RP + "test_rdrobust_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals(
                        "mserd", "msetwo", "msesum", "cerrd", "certwo", "cersum"
                    ),
                    "kernel": _vals("triangular", "epanechnikov", "uniform"),
                    "p": _vals("1", "2"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "36-cell grid vs R rdrobust: both estimates, both SEs and h, b on each "
                "side, all within 1e-9",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rdrobust_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals(
                        "mserd", "msetwo", "msesum", "cerrd", "certwo", "cersum"
                    ),
                    "kernel": _vals("triangular", "epanechnikov", "uniform"),
                    "p": _vals("0", "3", "4"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "54-cell grid at polynomial orders 0, 3 and 4 vs R rdrobust: the same "
                "eight outputs, within 1e-11 (p = 0), 1e-9 (p = 3) and 1e-7 (p = 4)",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_vce_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1", "2"),
                    "vce": _vals("nn", "hc0", "hc1", "hc2", "hc3"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "every vce kind at p = 1, 2 vs R rdrobust: estimates and SEs to 1e-9",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_vce_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("nn", "hc0", "hc2", "hc3"),
                    "covariates": _vals("set"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "vce kinds with a covariate vs R rdrobust",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_vce_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1", "2"),
                    "vce": _vals("cluster"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "clustered estimates, SEs and bandwidth at p = 1, 2 vs R rdrobust",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_vce_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("cluster"),
                    "covariates": _vals("set"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "clustered with a covariate vs R rdrobust",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_cluster_cer_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("cerrd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("cluster"),
                    "covariates": _vals("set"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "clustered, covariate-adjusted fit on a CER bandwidth: bias-corrected "
                "estimate and robust SE vs R rdrobust",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rdrobust_fuzzy_parity.py",
                {
                    "design": _vals("fuzzy"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1", "2"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none", "set"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "two-sided fuzzy RD at p = 1, 2, with and without covariates, vs R "
                "rdrobust",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rdrobust_fuzzy_parity.py",
                {
                    "design": _vals("fuzzy"),
                    "bwselect": _vals("cerrd", "msetwo"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "fuzzy RD under the other bandwidth selectors vs R rdrobust",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rdrobust_fuzzy_parity.py",
                {
                    "design": _vals("fuzzy"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("uniform"),
                    "p": _vals("1"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "fuzzy RD, uniform kernel vs R rdrobust",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rdrobust_fuzzy_parity.py",
                {
                    "design": _vals("fuzzy"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("hc0", "hc1", "hc2", "hc3"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "fuzzy RD under hc0-hc3: conventional estimate, conventional and "
                "robust SE vs R rdrobust",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rdrobust_params_parity.py",
                {
                    "design": _vals("kink"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("2"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "kink (deriv = 1, p = 2): bandwidth, conventional estimate and its SE "
                "vs R rdrobust",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rdrobust_comb_parity.py",
                {
                    "design": _vals("sharp"),
                    "bwselect": _vals("msecomb1", "msecomb2", "cercomb1", "cercomb2"),
                    "kernel": _vals("triangular", "epanechnikov", "uniform"),
                    "p": _vals("1", "2"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "24-cell comb-selector grid vs R rdrobust 4.0.0 (2.2e-12) and Stata "
                "rdrobust 10.0.0 (2.5e-9, the distance between the two references): "
                "both estimates, both SEs and h, b on each side",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rdrobust_comb_parity.py",
                {
                    "design": _vals("fuzzy"),
                    "bwselect": _vals("msecomb1", "msecomb2", "cercomb1", "cercomb2"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "fuzzy RD with two-sided noncompliance under the four comb selectors "
                "vs R and Stata rdrobust",
                "sp.rdrobust(df, y, x, c)",
            ),
            # The extractor reports deriv = 1 as design 'kink' for sharp and
            # fuzzy fits alike, so only cells the artifact ran for both are
            # listed (its uniform-kernel and clustered cells are sharp only).
            _Row(
                "T2",
                _RP + "test_rd_iv_rd_R_parity.py",
                {
                    "design": _vals("kink"),
                    "bwselect": _vals("mserd"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1", "2"),
                    "vce": _vals("nn"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "deriv = 1, sharp and fuzzy: both estimates, both SEs, h and b vs R "
                "rdrobust(deriv = 1); p = 2 also vs Stata rdrobust",
                "sp.rdrobust(df, y, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rd_iv_rd_R_parity.py",
                {
                    "design": _vals("kink"),
                    "bwselect": _vals("mserd", "manual"),
                    "kernel": _vals("triangular"),
                    "p": _vals("1"),
                    "vce": _vals("hc1"),
                    "covariates": _vals("none"),
                    "code_path": _vals("native"),
                    "weights": _vals("none"),
                },
                _EST_SE,
                "deriv = 1, sharp and fuzzy, hc1 at the selected and at a fixed h: "
                "estimates and SEs vs R and Stata rdrobust",
                "sp.rdrobust(df, y, x, c)",
            ),
        ),
        note="The R reference rows run on the Senate data and on two simulated "
        "designs built so that a wrong variance or a dropped covariate moves "
        "the answer. Kink designs with the uniform kernel or a cluster (sharp "
        "only), kink designs at p >= 3, the Epanechnikov kernel outside the "
        "sharp grids, and the comb selectors with covariates have no reference "
        "row. bwselect='cct' (code_path 'port') runs the authors' own rdrobust "
        "Python port; comparing it with R compares the reference with itself, "
        "so it has no row and its native twin is bwselect='mserd'. Other "
        "bandwidth selectors are also pinned for sp.rdbwselect (Track A module "
        "88).",
    )
)

_add(
    _Scope(
        "rddensity",
        {"code_path": ("native", "rddensity")},
        _x_rddensity,
        (
            _Row(
                "T2",
                _R + "09_rddensity.py",
                {"code_path": _vals("native")},
                ("diagnostic",),
                "density-difference test statistic and p-value vs rddensity",
                "sp.rddensity(backend='native')",
            ),
            _Row(
                "T2",
                _RP + "test_rddensity_parity.py",
                {"code_path": _vals("native")},
                ("diagnostic", "estimate", "se"),
                "three designs x p in {2, 3} vs R rddensity 2.6: the statistic "
                "(result.estimate) and p-value, the jackknife SE of the density "
                "difference (result.se) and the one-sided SEs, bandwidths, "
                "densities and the order-p fit, all within 1e-8",
                "sp.rddensity(df, x, c)",
            ),
            _Row(
                "T2",
                _RP + "test_rddensity_all_stata.py",
                {"code_path": _vals("native")},
                ("diagnostic", "estimate"),
                "Lee senate data vs Stata rddensity, all: robust and conventional "
                "statistics, p-values, one-sided densities and SEs, and the "
                "binomial-test table",
                "sp.rddensity(df, x)",
            ),
        ),
        primary=("diagnostic",),
        note="backend='r' (code_path 'rddensity') runs R rddensity itself through "
        "Rscript, so a comparison with R would compare the reference with itself; "
        "it has no row. result.estimate is the test statistic and result.se the "
        "jackknife SE of the density difference.",
    )
)

_add(
    _Scope(
        "synth",
        {
            "v_method": ("equal", "nested", "regression"),
            "weights": ("unique", "nonunique"),
            "code_path": ("native", "synth"),
        },
        _x_synth,
        (
            _Row(
                "T2",
                _R + "52_scm_unique.py",
                {
                    "v_method": _vals("equal"),
                    "weights": _vals("unique"),
                    "code_path": _vals("native"),
                },
                _EST,
                "donor weights and average gap vs Synth and synth on a uniquely "
                "identified DGP",
                "sp.synth(method='classic')",
            ),
            _Row(
                "T4",
                _R + "07_scm.py",
                {
                    "v_method": _vals("nested"),
                    "weights": _vals("unique", "nonunique"),
                    "code_path": _vals("native"),
                },
                _EST,
                "Basque gap: reference implementations disagree (non-unique V, W)",
                "sp.synth(method='classic', special predictors)",
            ),
            _Row(
                "T1",
                _RP + "test_synth_unique_options_parity.py",
                {
                    "v_method": _vals("nested", "regression"),
                    "weights": _vals("unique"),
                    "code_path": _vals("native"),
                },
                _EST,
                "module 52's design with special predictors: true donor weights and "
                "a gap of exactly 2 under the nested V search and regression V; R "
                "Synth's nested search recovers the same to 4e-14",
                "sp.synth(method='classic', special_predictors=...)",
            ),
            _Row(
                "T2",
                _RP + "test_synth_unique_options_parity.py",
                {
                    "v_method": _vals("equal"),
                    "weights": _vals("unique"),
                    "code_path": _vals("native"),
                },
                ("se",),
                "placebo SD (result.se): every placebo's weights certified as the "
                "unique simplex least-squares optimum (KKT); placebo effects vs R "
                "Synth with equal unscaled V to 1e-12 (one vertex optimum to ipop's "
                "1e-5), the SD to 9e-7",
                "sp.synth(method='classic')",
            ),
            _Row(
                "T4",
                _RP + "test_synth_regression_v_stata.py",
                {
                    "v_method": _vals("regression"),
                    "weights": _vals("unique"),
                    "code_path": _vals("native"),
                },
                _EST,
                "Proposition 99 vs Stata synth: V and pre-period RMSPE to 1e-9; "
                "Stata builds its gaps from donor weights rounded to three decimals "
                "(ATT -18.70 against -18.81 at the exact weights), and rounding ours "
                "reproduces its path to 1e-9",
                "sp.synth(v_method='regression')",
            ),
            _Row(
                "T4",
                _RP + "test_synth2_stata_parity.py",
                {
                    "v_method": _vals("regression"),
                    "weights": _vals("unique"),
                    "code_path": _vals("native"),
                },
                _EST,
                "synth2 placebo / leave-one-out reports: pre-period MSPEs to 1e-8, "
                "paths and effects reproduced from weights rounded as synth stores "
                "them",
                "sp.synth(method='classic', v_method='regression')",
            ),
        ),
        primary=("estimate",),
        note="Classical SCM only; other sp.synth methods have their own modules. "
        "Placebo p-values are not part of any row. backend='synth' (code_path "
        "'synth') runs R Synth itself, so it has no row. Equal V with non-unique "
        "weights returns the minimum-norm optimum, a selection no reference "
        "shares; it has no row. Regression V (Stata synth's default) is "
        "certified only on the identified design, because Stata reports gaps "
        "from rounded weights. The identified design has a zero-loss solution, "
        "where V does not move W: its nested row checks the nested code path end "
        "to end, not that the outer search finds the global V on a design "
        "without one.",
    )
)

_DYN_DOMAINS = {
    "se_method": ("analytic", "bootstrap"),
    "aggregation": ("simple", "switchers"),
    "control": ("not_yet_treated", "never_treated"),
    "switchers": ("both", "in", "out"),
    "same_switchers": ("false", "true"),
    "controls": ("none", "set"),
    "trends_nonparam": ("none", "set"),
    "normalized": ("false", "true"),
    "weights": ("none", "set"),
    "cluster": ("group", "coarser"),
    "treatment": ("binary", "discrete", "continuous"),
    "panel": ("balanced", "unbalanced"),
    "placebo": ("0", ">0"),
}
_DYN_BASE = {
    "se_method": "analytic",
    "aggregation": "switchers",
    "control": "not_yet_treated",
    "switchers": "both",
    "same_switchers": "false",
    "controls": "none",
    "trends_nonparam": "none",
    "normalized": "false",
    "weights": "none",
    "cluster": "group",
    "treatment": "binary",
    "panel": "balanced",
    "placebo": ">0",
}


def _dyn_cfg(**over: Any) -> Dict[str, FrozenSet[str]]:
    """One ``did_multiplegt_dyn`` configuration: the base, with overrides."""
    cfg = {k: _vals(v) for k, v in _DYN_BASE.items()}
    for k, v in over.items():
        cfg[k] = _vals(*v) if isinstance(v, tuple) else _vals(v)
    return cfg


_DYN97 = _RP + "test_dcdh_option_values_stata_parity.py"
_DYN85 = _RP + "test_multiplegt_dyn_options_parity.py"
_DYN_R = _RP + "test_multiplegt_dyn_parity.py"
_DYN_ROPT = _RP + "test_dcdh_options_parity.py"
_DYN_TB = _RP + "test_dcdh_textbook_stata_parity.py"
_DYN_CASTLE = "tests/test_did_multiplegt_dyn_castle_reference.py"
_DYN_EP = "sp.did_multiplegt_dyn()"
_DYN_ALL = ("estimate", "se", "vcov", "joint_test")
_DYN_EST_SE_JOINT = ("estimate", "se", "joint_test")
_BOTH_AGG = ("simple", "switchers")
_DYN97_TEXT = (
    "every effect and placebo, its analytic SE, the switcher count, "
    "Av_tot_eff and its SE, every entry of e(V) and the joint Wald p-values "
    "vs Stata did_multiplegt_dyn (option fixture 97, double precision, "
    "1e-6; observed 1e-13); the 'simple' headline and SE against the mean "
    "of Stata's effects and the quadratic form in its e(V)"
)


def _dyn97(note: str, outputs: Tuple[str, ...] = _DYN_ALL, **over: Any) -> _Row:
    """A row of option fixture 97: one option value at a time vs Stata."""
    return _Row(
        "T2",
        _DYN97,
        _dyn_cfg(aggregation=_BOTH_AGG, **over),
        outputs,
        _DYN97_TEXT + "; " + note,
        _DYN_EP,
    )


def _dyn_tb(note: str, **over: Any) -> _Row:
    """A row of the textbook-shaped panel vs Stata (count treatment)."""
    over.setdefault("treatment", "discrete")
    return _Row(
        "T2",
        _DYN_TB,
        _dyn_cfg(**over),
        _DYN_EST_SE_JOINT,
        "effects, placebos, analytic SEs, Av_tot_eff and the joint Wald "
        "p-values vs Stata did_multiplegt_dyn; " + note,
        _DYN_EP,
    )


def _dyn_ropt(note: str, **over: Any) -> _Row:
    """A row of the DIDmultiplegtDYN 2.3.4 option panel (estimates, 1e-12)."""
    return _Row(
        "T2",
        _DYN_ROPT,
        _dyn_cfg(aggregation="simple", **over),
        _EST,
        "vs DIDmultiplegtDYN 2.3.4 at 1e-12: " + note,
        _DYN_EP,
    )


def _dyn85(note: str, **over: Any) -> _Row:
    """A row of option fixture 85 (sample options vs Stata, estimates)."""
    return _Row(
        "T2",
        _DYN85,
        _dyn_cfg(**over),
        _EST,
        "effects, placebos, switcher counts and Av_tot_eff vs Stata "
        "did_multiplegt_dyn on a switch-in / switch-out panel; " + note,
        _DYN_EP,
    )


_add(
    _Scope(
        "did_multiplegt_dyn",
        _DYN_DOMAINS,
        _x_dcdh_dyn,
        (
            _dyn97("base fit, with and without effects_equal(all)"),
            _dyn97("switchers(in)", switchers="in"),
            _dyn97("switchers(out)", switchers="out"),
            _dyn97("same_switchers", same_switchers="true"),
            _dyn97("only_never_switchers", control="never_treated"),
            _dyn97("trends_nonparam(reg)", trends_nonparam="set"),
            _dyn97("normalized", normalized="true"),
            _dyn97("weight(w), time-varying", weights="set"),
            _dyn97("cluster(cl), six groups per cluster", cluster="coarser"),
            _dyn97("controls(x)", controls="set"),
            _dyn97(
                "effects(1) placebo(1), no joint test to form",
                ("estimate", "se", "vcov"),
            ),
            _dyn97(
                "trends_nonparam, normalized, weight, cluster and switchers(in) "
                "together",
                switchers="in",
                trends_nonparam="set",
                normalized="true",
                weights="set",
                cluster="coarser",
            ),
            _dyn97(
                "continuous(1) and continuous(2), placebo(1), with "
                "never-switchers and switches both ways",
                treatment="continuous",
            ),
            _dyn97("unbalanced panel, 4% of rows dropped", panel="unbalanced"),
            _Row(
                "S",
                _DYN97,
                _dyn_cfg(se_method="bootstrap"),
                ("se",),
                "bootstrap(200, s) SEs of every horizon and of Av_tot_eff: mean "
                "over five seeds vs Stata's mean over five seeds within four "
                "pooled Monte Carlo SEs (Stata's random stream cannot be "
                "reproduced)",
                _DYN_EP,
            ),
            _dyn85("pooled"),
            _dyn85("switchers(in) / switchers(out)", switchers=("in", "out")),
            _dyn85("same_switchers", same_switchers="true"),
            _Row(
                "T2",
                _R + "78_multiplegt_dyn.py",
                _dyn_cfg(se_method="bootstrap", aggregation=_BOTH_AGG),
                _EST,
                "every effect and placebo and Av_tot_eff vs DIDmultiplegtDYN on "
                "an absorbing staggered panel and a switch-on / switch-off "
                "panel (SEs not compared)",
                _DYN_EP,
            ),
            _Row(
                "T2",
                _DYN_R,
                _dyn_cfg(),
                ("estimate", "se"),
                "every effect and placebo, its analytic SE, Av_tot_eff and its "
                "SE vs DIDmultiplegtDYN on Track A module 78's panel",
                _DYN_EP,
            ),
            _Row(
                "T2",
                _DYN_R,
                _dyn_cfg(aggregation="simple"),
                _EST,
                "every effect and placebo vs DIDmultiplegtDYN (the 'simple' "
                "headline is their mean; its SE needs the covariance, which is "
                "not compared here)",
                _DYN_EP,
            ),
            _dyn_ropt("every effect and placebo, no option"),
            _dyn_ropt("every effect and placebo, controls", controls="set"),
            _dyn_ropt(
                "every effect and placebo, trends_nonparam", trends_nonparam="set"
            ),
            _dyn_ropt("every effect and placebo, normalized", normalized="true"),
            _dyn_ropt(
                "the effects (not the placebo), continuous=1 and 2, on a panel "
                "where every group switches",
                treatment="continuous",
            ),
            _dyn_tb("count treatment, periods four years apart, effects_equal(all)"),
            _dyn_tb(
                "count treatment, normalized, effects_equal(all)", normalized="true"
            ),
            _dyn_tb(
                "count treatment, same_switchers", same_switchers="true", placebo="0"
            ),
            _dyn_tb(
                "count treatment, weight() cluster(state)",
                weights="set",
                cluster="coarser",
            ),
            _dyn_tb(
                "count treatment, controls(x) with the slope-estimation term",
                controls="set",
            ),
            _dyn_tb(
                "count treatment, controls(x) weight() cluster(state)",
                controls="set",
                weights="set",
                cluster="coarser",
            ),
            _dyn_tb(
                "count treatment, controls(x) normalized",
                controls="set",
                normalized="true",
                placebo="0",
            ),
            _dyn_tb(
                "count treatment, controls(x) trends_nonparam(cohort)",
                controls="set",
                trends_nonparam="set",
                placebo="0",
            ),
            _dyn_tb(
                "count treatment, controls(x) switchers(in)",
                controls="set",
                switchers="in",
                placebo="0",
            ),
            _dyn_tb(
                "binary treatment, missing outcomes, controls(x)",
                treatment="binary",
                controls="set",
                panel="unbalanced",
            ),
            _Row(
                "T2",
                _DYN_CASTLE,
                _dyn_cfg(cluster=("group", "coarser"), weights=("none", "set")),
                ("estimate", "se"),
                "castle-doctrine panel: every effect and placebo, analytic SEs "
                "and Av_tot_eff vs DIDmultiplegtDYN and Stata, unweighted and "
                "weight(popwt); cluster(region) vs Stata",
                _DYN_EP,
            ),
        ),
        invariant={
            "se_method": (
                _EST,
                "the effects are computed before, and independently of, the "
                "variance; the bootstrap resamples afterwards",
            ),
            "cluster": (_EST, "the cluster enters the variance only"),
        },
        note="The 'simple' headline (the default) is the equal-weight mean of "
        "the effects; no reference reports it, so a row covers it only when it "
        "compared every effect, and its SE only with the joint covariance "
        "(fixture 97). On an unbalanced panel where a period in the middle has "
        "no not-yet-switched control, the reference drops a group StatsPAI "
        "keeps as a control (the departure documented in the module "
        "docstring); no row runs that design. design= / by_path= / "
        "normalized_weights= describe the treatment paths and are not "
        "dimensions; by_path is pinned in test_dcdh_textbook_stata_parity.py. "
        "Bootstrap SEs are only screened (S). The number of effects is not a "
        "dimension (every row runs several, fixture 97 also one); "
        "effects_equal= only adds a Wald test, compared where a row lists "
        "joint_test and ran it (fixture 97's base fit, the textbook rows).",
    )
)

_add(
    _Scope(
        "sdid",
        {
            "method": ("sdid", "sc", "did"),
            "se_method": ("placebo", "bootstrap", "jackknife", "noinference"),
            "code_path": ("native", "synthdid", "r"),
            "interface": ("treated_unit", "treat"),
            "design": ("block", "staggered"),
            "covariates": ("none", "projected"),
        },
        _x_sdid,
        (
            _Row(
                "T2",
                _R + "12_sdid.py",
                {
                    "method": _vals("sdid"),
                    "se_method": _vals("placebo"),
                    "code_path": _vals("native"),
                    "interface": _vals("treated_unit"),
                    "design": _vals("block"),
                    "covariates": _vals("none"),
                },
                _EST,
                "point ATT vs synthdid (the SE is not part of this row)",
                "sp.sdid(backend='native')",
            ),
            _Row(
                "T2",
                _RP + "test_sdid_staggered_parity.py",
                {
                    "method": _vals("sdid", "did", "sc"),
                    "se_method": _vals("jackknife", "noinference"),
                    "code_path": _vals("native"),
                    "interface": _vals("treat"),
                    "design": _vals("block", "staggered"),
                    "covariates": _vals("none", "projected"),
                },
                _EST_SE,
                "ATT, per-cohort tau and jackknife SE vs Stata sdid 2.0.2 "
                "(Prop 99 block, quota staggered, projected covariates)",
                "sp.sdid(treat=...)",
            ),
            _Row(
                "B",
                _B,
                {
                    "method": _vals("sdid"),
                    "se_method": _vals("placebo"),
                    "code_path": _vals("native"),
                    "interface": _vals("treated_unit"),
                    "design": _vals("block"),
                    "covariates": _vals("none"),
                },
                _COV,
                "placebo-SE CI coverage, one treated unit",
                "sp.sdid()",
            ),
            _Row(
                "T2",
                _RP + "test_did_synth_R_parity.py",
                {
                    "method": _vals("sdid", "sc", "did"),
                    "se_method": _vals("jackknife"),
                    "code_path": _vals("native"),
                    "interface": _vals("treated_unit"),
                    "design": _vals("block"),
                    "covariates": _vals("none"),
                },
                _EST_SE,
                "ATT, unit and time weights at 1e-9 and the fixed-weight "
                "jackknife SE at 1e-9 vs synthdid 0.0.9 (Prop. 99 replica, one "
                "treated unit, where the jackknife is undefined on both sides; "
                "a panel with five treated units); also replays R's own placebo "
                "and bootstrap index vectors draw for draw at 1e-9",
                "sp.sdid(treated_unit=...)",
            ),
            _Row(
                "T2",
                _RP + "test_sdid_parity.py",
                {
                    "method": _vals("sdid"),
                    "se_method": _vals("placebo"),
                    "code_path": _vals("native"),
                    "interface": _vals("treated_unit"),
                    "design": _vals("block"),
                    "covariates": _vals("none"),
                },
                _EST,
                "point ATT vs synthdid on the shipped Prop. 99 panel",
                "sp.sdid()",
            ),
            _Row(
                "S",
                _RP + "test_sdid_resampling_screen.py",
                {
                    "method": _vals("sdid", "sc", "did"),
                    "se_method": _vals("placebo", "bootstrap"),
                    "code_path": _vals("native"),
                    "interface": _vals("treated_unit"),
                    "design": _vals("block"),
                    "covariates": _vals("none"),
                },
                ("se",),
                "placebo and bootstrap SEs, mean over five seeds x 200 draws vs "
                "synthdid_se's mean over five seeds, within four pooled Monte "
                "Carlo SEs (one and five treated units; the bootstrap only with "
                "five, it is undefined with one)",
                "sp.sdid(treated_unit=...)",
            ),
            _Row(
                "S",
                _RP + "test_sdid_resampling_screen.py",
                {
                    "method": _vals("sdid", "sc", "did"),
                    "se_method": _vals("placebo"),
                    "code_path": _vals("native"),
                    "interface": _vals("treat"),
                    "design": _vals("block"),
                    "covariates": _vals("none"),
                },
                ("se",),
                "vce(placebo) SE on the Prop. 99 block design, mean over five "
                "seeds x 200 draws vs Stata sdid 2.0.2 over five seeds",
                "sp.sdid(treat=...)",
            ),
            _Row(
                "S",
                _RP + "test_sdid_resampling_screen.py",
                {
                    "method": _vals("sdid"),
                    "se_method": _vals("placebo", "bootstrap"),
                    "code_path": _vals("native"),
                    "interface": _vals("treat"),
                    "design": _vals("staggered"),
                    "covariates": _vals("none"),
                },
                ("se",),
                "vce(placebo) and vce(bootstrap) SEs on the staggered quota "
                "panel, mean over five seeds x 200 draws vs Stata sdid 2.0.2 "
                "over five seeds",
                "sp.sdid(treat=...)",
            ),
        ),
        invariant={
            "se_method": (
                _EST,
                "the SE method resamples after the unit and time weights are "
                "solved; the ATT is unchanged",
            )
        },
        note="Placebo and bootstrap SEs are Monte Carlo draws whose random "
        "stream no reference shares: the replication map is pinned draw for "
        "draw against R (test_did_synth_R_parity.py), the end-to-end SE only "
        "screened (S). backend='synthdid' / 'r' (code_path 'synthdid' / 'r') "
        "runs R synthdid itself and has no row by design.",
        # added with sp.sdid(treat=...); an older by-name query means the
        # treated_unit= block design without covariates
        defaults={"interface": "treated_unit", "design": "block", "covariates": "none"},
    )
)

_add(
    _Scope(
        "dml",
        {
            "model": ("plr", "irm", "pliv", "iivm"),
            "score": ("partialling out", "iv-type", "ate", "atte", "late"),
            "learners": ("linear", "default", "other"),
            "n_folds": tuple(str(k) for k in range(2, 21)),
            "n_rep": ("1", ">1"),
            # propensity trimming / normalisation of the IRM and IIVM scores
            "ipw": ("n/a", "trim0.01", "trim1e-12", "other"),
        },
        _x_dml,
        (
            _Row(
                "T2",
                _R + "08_dml.py",
                {
                    "model": _vals("plr"),
                    "score": _vals("partialling out"),
                    "learners": _vals("linear"),
                    "n_folds": _vals("5"),
                    "n_rep": _vals("1"),
                    "ipw": _vals("n/a"),
                },
                _EST_SE,
                "theta and SE vs DoubleML on shared folds",
                "sp.dml(model='plr')",
            ),
            _Row(
                "T2",
                _R + "71_dml_family.py",
                {
                    "model": _vals("irm"),
                    "score": _vals("ate"),
                    "learners": _vals("linear"),
                    "n_folds": _vals("5"),
                    "n_rep": _vals("1"),
                    "ipw": _vals("trim1e-12"),
                },
                _EST_SE,
                "theta and SE vs DoubleML on shared folds",
                "sp.dml(model='irm')",
            ),
            _Row(
                "T2",
                _R + "71_dml_family.py",
                {
                    "model": _vals("pliv"),
                    "score": _vals("partialling out"),
                    "learners": _vals("linear"),
                    "n_folds": _vals("5"),
                    "n_rep": _vals("1"),
                    "ipw": _vals("n/a"),
                },
                _EST_SE,
                "theta and SE vs DoubleML on shared folds",
                "sp.dml(model='pliv')",
            ),
            _Row(
                "T2",
                _R + "71_dml_family.py",
                {
                    "model": _vals("iivm"),
                    "score": _vals("late"),
                    "learners": _vals("linear"),
                    "n_folds": _vals("5"),
                    "n_rep": _vals("1"),
                    "ipw": _vals("trim1e-12"),
                },
                _EST_SE,
                "theta and SE vs DoubleML on shared folds",
                "sp.dml(model='iivm')",
            ),
            _Row(
                "T2",
                _DML_PY,
                {
                    "model": _vals("plr"),
                    "score": _vals("partialling out"),
                    "learners": _vals("other"),
                    "n_folds": _vals("4"),
                    "n_rep": _vals("1"),
                    "ipw": _vals("n/a"),
                },
                _EST_SE,
                "random-forest regressor for Y and classifier for a 0/1 D "
                "(predict_proba), vs doubleml-for-py on shared folds",
                "sp.dml(model='plr')",
            ),
            _Row(
                "T2",
                _DML_PY,
                {
                    "model": _vals("irm"),
                    "score": _vals("ate", "atte"),
                    "learners": _vals("linear"),
                    "n_folds": _vals("4"),
                    "n_rep": _vals("1"),
                    "ipw": _vals("trim0.01"),
                },
                _EST_SE,
                "theta, and the omitted-variable-bias bounds of "
                "sp.dml_sensitivity, vs doubleml-for-py on shared folds",
                "sp.dml(model='irm')",
            ),
            _Row(
                "B",
                _B,
                {
                    "model": _vals("plr"),
                    "score": _vals("partialling out"),
                    "learners": _vals("default"),
                    "n_folds": _vals("5"),
                    "n_rep": _vals("1"),
                    "ipw": _vals("n/a"),
                },
                _COV,
                "95% CI coverage with the default learners: 0.883 (bias 0.5 MC SD, "
                "mean SE 0.88 of the MC SD); see mechanisms/dml_plr_learners.py",
                "sp.dml(model='plr', n_folds=5)",
            ),
            _Row(
                "B",
                _B,
                {
                    "model": _vals("irm"),
                    "score": _vals("ate"),
                    "learners": _vals("default"),
                    "n_folds": _vals("5"),
                    "n_rep": _vals("1"),
                    "ipw": _vals("trim0.01"),
                },
                _COV,
                "95% CI coverage with the default learners",
                "sp.causal_question(design='dml')",
            ),
            _Row(
                "T2",
                _DML_IVT,
                _dml_cfg("plr", ("iv-type",), "linear", ("5",), "1", "n/a"),
                _EST_SE,
                "IV-type score with OLS nuisances vs doubleml-for-py on a shared "
                "partition (1e-10)",
                "sp.dml(model='plr', score='IV-type')",
            ),
            _Row(
                "T2",
                _DML_IVT,
                _dml_cfg("plr", ("iv-type",), "other", ("5",), "1", "n/a"),
                _EST_SE,
                "IV-type score with cross-validated lasso nuisances vs doubleml-for-"
                "py on a shared partition (1e-8)",
                "sp.dml(model='plr', score='IV-type')",
            ),
            _Row(
                "T2",
                _DML_IVT,
                _dml_cfg("irm", ("ate", "atte"), "linear", ("5",), "1", "trim0.01"),
                _EST_SE,
                "ATE and ATTE scores vs doubleml-for-py on a shared partition",
                "sp.dml(model='irm')",
            ),
            _Row(
                "T2",
                _DML_OPT,
                _dml_cfg(
                    "plr",
                    ("partialling out", "iv-type"),
                    "linear",
                    ("2", "3", "10"),
                    "1",
                    "n/a",
                ),
                _EST_SE,
                "both scores at 2, 3 and 10 folds vs doubleml-for-py 0.11.3 on "
                "shared folds (7e-16)",
                "sp.dml(df, model='plr')",
            ),
            _Row(
                "T2",
                _DML_OPT,
                _dml_cfg("irm", ("ate",), "linear", ("2", "3", "10"), "1", "trim0.01"),
                _EST_SE,
                "ATE at 2, 3 and 10 folds vs doubleml-for-py on shared folds",
                "sp.dml(df, model='irm')",
            ),
            _Row(
                "T2",
                _DML_OPT,
                _dml_cfg("irm", ("atte",), "linear", ("3",), "1", "trim0.01"),
                _EST_SE,
                "ATTE at 3 folds vs doubleml-for-py on shared folds",
                "sp.dml(df, model='irm')",
            ),
            _Row(
                "T2",
                _DML_OPT,
                _dml_cfg("irm", ("ate", "atte"), "linear", ("3",), "1", "other"),
                _EST_SE,
                "trimming at 0.1 (binding; ATE) and normalize_ipw=True (ATE, ATTE) "
                "vs doubleml-for-py on shared folds",
                "sp.dml(df, model='irm')",
            ),
            _Row(
                "T2",
                _DML_OPT,
                _dml_cfg(
                    "pliv", ("partialling out",), "linear", ("2", "3", "10"), "1", "n/a"
                ),
                _EST_SE,
                "PLIV at 2, 3 and 10 folds vs doubleml-for-py on shared folds",
                "sp.dml(df, model='pliv')",
            ),
            _Row(
                "T2",
                _DML_OPT,
                _dml_cfg(
                    "iivm", ("late",), "linear", ("2", "3", "10"), "1", "trim0.01"
                ),
                _EST_SE,
                "IIVM at 2, 3 and 10 folds vs doubleml-for-py on shared folds",
                "sp.dml(df, model='iivm')",
            ),
            _Row(
                "T2",
                _DML_OPT,
                _dml_cfg("iivm", ("late",), "linear", ("3",), "1", "other"),
                _EST_SE,
                "IIVM with trimming at 0.1 (binding) and normalize_ipw=True vs "
                "doubleml-for-py on shared folds",
                "sp.dml(df, model='iivm')",
            ),
            _Row(
                "T2",
                _DML_OPT,
                _dml_cfg(
                    "plr", ("partialling out", "iv-type"), "linear", ("5",), ">1", "n/a"
                ),
                _EST,
                "n_rep = 3 on sp.dml's own folds: every repetition's estimate and SE "
                "and the median estimate vs doubleml-for-py and R DoubleML",
                "sp.dml(df, model='plr', n_rep=3)",
            ),
            _Row(
                "T2",
                _DML_OPT,
                _dml_cfg("irm", ("ate",), "linear", ("5",), ">1", "trim0.01"),
                _EST,
                "n_rep = 3 (stratified folds): every repetition's estimate and SE "
                "and the median estimate vs doubleml-for-py",
                "sp.dml(df, model='irm', n_rep=3)",
            ),
            _Row(
                "T4",
                _DML_OPT,
                _dml_cfg(
                    "plr", ("partialling out", "iv-type"), "linear", ("5",), ">1", "n/a"
                ),
                ("se",),
                "aggregated SE under n_rep > 1: R and Python DoubleML use different "
                "rules (1.8e-3 apart); sp.dml uses median(se_r^2 + dev_r^2); each "
                "reference's SE is rebuilt from the shared repetitions to 1e-12",
                "sp.dml(df, model='plr', n_rep=3)",
            ),
            _Row(
                "T4",
                _DML_OPT,
                _dml_cfg("irm", ("ate",), "linear", ("5",), ">1", "trim0.01"),
                ("se",),
                "aggregated SE under n_rep > 1: Python DoubleML's median-upper-bound "
                "rule, rebuilt from the shared repetitions to 1e-12",
                "sp.dml(df, model='irm', n_rep=3)",
            ),
        ),
        note="With default (flexible) learners the estimate depends on the learner, so "
        "deterministic parity exists only for fixed learners and folds; the default-"
        "learner PLR interval under-covers (0.883) on the Track B design. ipw 'other' "
        "rows ran trimming at 0.1 and normalize_ipw=True at the 0.01 threshold. "
        "n_folds 6-9 and 11-20 have no row. Under n_rep > 1 the per-repetition "
        "estimates and SEs and the median estimate match DoubleML, but the "
        "aggregated SE is a convention the two DoubleML ports disagree on.",
    )
)

_add(
    _Scope(
        "causal_forest",
        {
            "treatment": ("binary", "continuous"),
            "trees": tuple(
                str(k) for k in (100, 200, 300, 500, 1000, 2000, 4000, 8000)
            ),
            "tuning": ("grf_defaults", "custom"),
            "design": ("iid", "clustered", "fixed_effects"),
        },
        _x_forest,
        (
            _Row(
                "T3",
                _RP + "test_grf_seed_mc_equivalence.py",
                {
                    "treatment": _vals("binary"),
                    "trees": _vals("500", "2000", "8000"),
                    "tuning": _vals("grf_defaults"),
                    "design": _vals("iid"),
                },
                _EST,
                "seed-replicated AIPW ATE/ATT vs grf at 500, 2,000 and 8,000 trees: "
                "TOST equivalence within 0.1 sampling SE on two datasets",
                "sp.causal_forest(...).average_treatment_effect()",
            ),
            _Row(
                "S",
                _R + "13_causal_forest.py",
                {
                    "treatment": _vals("binary"),
                    "trees": _vals("2000"),
                    "tuning": _vals("grf_defaults"),
                    "design": _vals("iid"),
                },
                _EST_SE,
                "single-draw AIPW ATE/ATT and SE vs grf (one fit per engine)",
                "sp.causal_forest(n_estimators=2000)",
            ),
            _Row(
                "B",
                _B,
                {
                    "treatment": _vals("binary"),
                    "trees": _vals("2000"),
                    "tuning": _vals("grf_defaults"),
                    "design": _vals("iid"),
                },
                _COV,
                "coverage of the forest's own AIPW interval (2,000 trees)",
                "sp.causal_question(design='causal_forest')",
            ),
        ),
        note="Only tree counts that were run are listed; 'custom' tuning (honesty, "
        "sample fraction, leaf size, mtry, nuisance models, ...) has no row. A "
        "continuous treatment is compared with grf only in the Card worked example.",
    )
)

_PSMOPT = _RP + "test_psm_options_parity.py"


def _psm_cfg(
    method: str = "nearest",
    distance: str = "propensity",
    n_matches: Tuple[str, ...] = ("1",),
    replace: str = "true",
    caliper: Tuple[str, ...] = ("none",),
    bias_correction: str = "false",
    se_method: Tuple[str, ...] = ("abadie_imbens",),
) -> Dict[str, FrozenSet[str]]:
    return {
        "method": _vals(method),
        "distance": _vals(distance),
        "n_matches": _vals(*n_matches),
        "replace": _vals(replace),
        "caliper": _vals(*caliper),
        "bias_correction": _vals(bias_correction),
        "se_method": _vals(*se_method),
    }


_add(
    _Scope(
        "psm",
        {
            # a caliper is its own axis (and the radius width); there is no
            # method="caliper", and the score is matched on, not its logit
            "method": ("nearest", "kernel", "radius", "stratify"),
            "distance": ("propensity", "mahalanobis"),
            "n_matches": tuple(str(k) for k in range(1, 11)),
            "replace": ("true", "false"),
            "caliper": ("none", "set"),
            "bias_correction": ("true", "false"),
            # psmatch2: the kernel / radius default; analytic: stratification
            "se_method": (
                "abadie_imbens",
                "abadie_imbens_2016",
                "psmatch2",
                "analytic",
                "bootstrap",
            ),
        },
        _x_psm,
        (
            _Row(
                "T2",
                _R + "11_psm.py",
                {
                    "method": _vals("nearest"),
                    "distance": _vals("propensity"),
                    "n_matches": _vals("1"),
                    "replace": _vals("true"),
                    "caliper": _vals("none"),
                    "bias_correction": _vals("false"),
                    "se_method": _vals("abadie_imbens", "abadie_imbens_2016"),
                },
                _EST,
                "ATT vs MatchIt",
                "sp.psm(method='nn')",
            ),
            _Row(
                "T2",
                _R + "11_psm.py",
                {
                    "method": _vals("nearest"),
                    "distance": _vals("propensity"),
                    "n_matches": _vals("1"),
                    "replace": _vals("true"),
                    "caliper": _vals("none"),
                    "bias_correction": _vals("false"),
                    "se_method": _vals("abadie_imbens_2016"),
                },
                ("se",),
                "Abadie-Imbens (2016) estimated-score SE vs Stata teffects " "psmatch",
                "sp.psm(se_method='abadie_imbens_2016')",
            ),
            _Row(
                "T2",
                _RP + "test_matching_r_parity.py",
                {
                    "method": _vals("nearest"),
                    "distance": _vals("propensity"),
                    "n_matches": _vals("1", "2"),
                    "replace": _vals("false"),
                    "caliper": _vals("none"),
                    "bias_correction": _vals("false"),
                    "se_method": _vals("abadie_imbens", "abadie_imbens_2016"),
                },
                _EST,
                "k:1 propensity-score matching without replacement: ATT vs "
                "MatchIt to 1e-9 (reached through sp.match, the same fit)",
                "sp.match(lalonde, ..., method='nearest', replace=False)",
            ),
            _Row(
                "T2",
                _RP + "test_matching_r_parity.py",
                {
                    "method": _vals("nearest"),
                    "distance": _vals("mahalanobis"),
                    "n_matches": _vals("1"),
                    "replace": _vals("false"),
                    "caliper": _vals("none"),
                    "bias_correction": _vals("false"),
                    "se_method": _vals("abadie_imbens", "abadie_imbens_2016"),
                },
                _EST,
                "Mahalanobis matching without replacement, pooled covariance, "
                "data and closest orderings: ATT vs MatchIt to 1e-9",
                "sp.match(lalonde, ..., distance='mahalanobis', replace=False)",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _psm_cfg(n_matches=tuple(str(k) for k in range(1, 11))),
                _EST_SE,
                "ATT and Abadie-Imbens (2006) SE vs Stata psmatch2 neighbor(k) "
                "ai(1), k = 1..10 (and ai(2) at k = 1), on tie-free data, to "
                "1e-9; with ties see test_psm_ties_se_parity.py (psmatch2 "
                "ai(1) ties, every row order)",
                "sp.match(df, y='y', treat='d', covariates=X, n_matches=k)",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _psm_cfg(
                    n_matches=tuple(str(k) for k in range(1, 11)),
                    se_method=("abadie_imbens_2016",),
                ),
                _EST_SE,
                "ATT and Abadie-Imbens (2016) SE vs Stata teffects psmatch, "
                "atet nneighbor(k), k = 1..10, to 1e-9",
                "sp.match(df, ..., se_method='abadie_imbens_2016')",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _psm_cfg(caliper=("set",), se_method=("abadie_imbens_2016",)),
                _EST_SE,
                "ATT and ATE with caliper(0.2) vs Stata teffects psmatch, to "
                "1e-9; teffects reads the caliper as a condition (caliper "
                "0.1 is refused on both sides), so the matches are unchanged",
                "sp.match(df, ..., caliper=0.2, se_method='abadie_imbens_2016')",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _psm_cfg(caliper=("none", "set"), se_method=("psmatch2",)),
                _EST_SE,
                "ATT and psmatch2's own SE vs Stata psmatch2 neighbor(1), with "
                "no caliper and with caliper(0.01) / caliper(0.005) that leave "
                "5 / 15 treated unmatched, to 1e-9",
                "sp.match(df, ..., caliper=0.01, se_method='psmatch2')",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _psm_cfg(caliper=("set",)),
                _EST,
                "ATT with caliper 0.01 / 0.005 vs Stata psmatch2 caliper() and "
                "R Matching::Match(caliper=), to 1e-9",
                "sp.match(df, ..., caliper=0.01)",
            ),
            _Row(
                "T4",
                _PSMOPT,
                _psm_cfg(caliper=("set",)),
                ("se",),
                "Abadie-Imbens SE under a caliper: psmatch2 ai(1) applies the "
                "caliper to the same-arm search and sets sigma^2 = y^2/2 when "
                "no same-arm unit is inside it; rebuilt from our matches to "
                "1e-9. Matching::Match's caliper SE is unexplained",
                "sp.match(df, ..., caliper=0.01)",
            ),
            _Row(
                "T4",
                _PSMOPT,
                _psm_cfg(replace="false"),
                ("se",),
                "Abadie-Imbens SE without replacement: psmatch2 noreplacement "
                "ai(1) also draws the same-arm neighbours without replacement; "
                "rebuilt from our matches to 1e-9 (Matching reports no SE)",
                "sp.match(df, ..., replace=False, m_order='smallest')",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _psm_cfg(distance="mahalanobis", n_matches=("1", "3")),
                _EST_SE,
                "ATT and Abadie-Imbens SE (same-arm neighbours on the "
                "Mahalanobis distance, pooled covariance) vs Stata psmatch2 "
                "mahalanobis() ai(1), k = 1 and 3, to 1e-9",
                "sp.match(df, ..., distance='mahalanobis')",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _psm_cfg(method="kernel", se_method=("psmatch2",)),
                _EST_SE,
                "kernel matching, all five kernels at bwidth 0.06 and the "
                "Epanechnikov at 0.02: ATT and SE vs Stata psmatch2 kernel, "
                "to 1e-9",
                "sp.match(df, ..., method='kernel', kernel=k)",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _psm_cfg(method="radius", caliper=("set",), se_method=("psmatch2",)),
                _EST_SE,
                "radius matching, caliper 0.02 and 0.05: ATT and SE vs Stata "
                "psmatch2 radius, to 1e-9",
                "sp.match(df, ..., method='radius', caliper=0.02)",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _psm_cfg(method="stratify", se_method=("analytic",)),
                _EST_SE,
                "five quantile classes of the score (the default strata): ATT "
                "and SE vs Stata atts with those blocks, to 1e-9",
                "sp.match(df, ..., method='stratify')",
            ),
            _Row(
                "T2",
                _RP + "test_qiu_jiaping_methods_stata.py",
                _psm_cfg(method="stratify", se_method=("analytic",)),
                _EST_SE,
                "the blocks of sp.pscore (strata=): ATT and SE vs Stata atts "
                "on Lalonde, to 1e-12",
                "sp.match(data, ..., method='stratify', strata='b1')",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _psm_cfg(bias_correction="true"),
                _EST_SE,
                "bias-adjusted ATT (regression on the covariates over the "
                "matched controls, weighted by use) and its SE vs R "
                "Matching::Match(BiasAdjust = TRUE), to 1e-9",
                "sp.match(df, ..., bias_correction=True)",
            ),
        ),
        invariant={
            "se_method": (_EST, "the variance estimator does not change the matches")
        },
        note="Option values were compared one at a time on tie-free data, "
        "not crossed. With ties in the score the default Abadie-Imbens SE "
        "keeps every tied same-arm neighbour, as psmatch2 , ai(1) ties does "
        "(test_psm_ties_se_parity.py, Lalonde and NSW to the last digit). No "
        "reference reproduces the SE of matching without replacement "
        "(psmatch2's differs by a rule documented in its T4 row; Matching "
        "reports none); bias correction is compared on the score distance "
        "only; the bootstrap has no row.",
    )
)

_NNMATCH = _RP + "test_nnmatch_teffects_parity.py"


def _nn_cfg(
    estimand: str,
    n_matches: Tuple[str, ...] = ("1",),
    vce_nn: Tuple[str, ...] = ("2",),
    caliper: Tuple[str, ...] = ("none",),
) -> Dict[str, FrozenSet[str]]:
    """An nnmatch cell on the Mahalanobis metric with the robust variance."""
    return {
        "estimand": _vals(estimand),
        "metric": _vals("mahalanobis"),
        "n_matches": _vals(*n_matches),
        "exact": _vals("none"),
        "bias_adjust": _vals("none"),
        "vce": _vals("robust"),
        "vce_nn": _vals(*vce_nn),
        "caliper": _vals(*caliper),
    }


#: sp.match(method='nnmatch') is Abadie-Imbens covariate matching with its
#: own option axes; sp.validation_scope routes such a fit here
#: (``function="nnmatch"`` in a by-name query).
_add(
    _Scope(
        "nnmatch",
        {
            "estimand": ("att", "ate"),
            "metric": ("mahalanobis", "ivariance", "euclidean"),
            "n_matches": ("1", "2", "3", "4", "5"),
            "exact": ("none", "set"),
            "bias_adjust": ("none", "set"),
            "vce": ("robust", "iid"),
            "vce_nn": ("2", "3", "4", "na"),
            "caliper": ("none", "set"),
        },
        _x_nnmatch,
        (
            _Row(
                "T2",
                _NNMATCH,
                {
                    "estimand": _vals("att"),
                    "metric": _vals("mahalanobis"),
                    "n_matches": _vals("1"),
                    "exact": _vals("none"),
                    "bias_adjust": _vals("none"),
                    "vce": _vals("robust"),
                    "vce_nn": _vals("2"),
                    "caliper": _vals("none"),
                },
                _EST_SE,
                "ATT, defaults vs Stata teffects nnmatch, estimate and SE to 1e-10",
                "sp.match(df, y, treat, covariates, method='nnmatch', ...)",
            ),
            _Row(
                "T2",
                _NNMATCH,
                {
                    "estimand": _vals("ate"),
                    "metric": _vals("mahalanobis"),
                    "n_matches": _vals("1"),
                    "exact": _vals("none"),
                    "bias_adjust": _vals("none"),
                    "vce": _vals("robust"),
                    "vce_nn": _vals("2"),
                    "caliper": _vals("none"),
                },
                _EST_SE,
                "ATE vs Stata teffects nnmatch, estimate and SE to 1e-10",
                "sp.match(df, y, treat, covariates, method='nnmatch', ...)",
            ),
            _Row(
                "T2",
                _NNMATCH,
                {
                    "estimand": _vals("att"),
                    "metric": _vals("mahalanobis"),
                    "n_matches": _vals("1"),
                    "exact": _vals("none"),
                    "bias_adjust": _vals("set"),
                    "vce": _vals("robust"),
                    "vce_nn": _vals("2"),
                    "caliper": _vals("none"),
                },
                _EST_SE,
                "ATT with bias adjustment on all covariates vs Stata teffects "
                "nnmatch, estimate and SE to 1e-10",
                "sp.match(df, y, treat, covariates, method='nnmatch', ...)",
            ),
            _Row(
                "T2",
                _NNMATCH,
                {
                    "estimand": _vals("att"),
                    "metric": _vals("ivariance"),
                    "n_matches": _vals("3"),
                    "exact": _vals("none"),
                    "bias_adjust": _vals("none"),
                    "vce": _vals("robust"),
                    "vce_nn": _vals("2"),
                    "caliper": _vals("none"),
                },
                _EST_SE,
                "ATT, inverse-variance metric, three matches vs Stata teffects "
                "nnmatch, estimate and SE to 1e-10",
                "sp.match(df, y, treat, covariates, method='nnmatch', ...)",
            ),
            _Row(
                "T2",
                _NNMATCH,
                {
                    "estimand": _vals("att"),
                    "metric": _vals("euclidean"),
                    "n_matches": _vals("1"),
                    "exact": _vals("none"),
                    "bias_adjust": _vals("none"),
                    "vce": _vals("robust"),
                    "vce_nn": _vals("2"),
                    "caliper": _vals("none"),
                },
                _EST_SE,
                "ATT, Euclidean metric vs Stata teffects nnmatch, estimate and SE to "
                "1e-10",
                "sp.match(df, y, treat, covariates, method='nnmatch', ...)",
            ),
            _Row(
                "T2",
                _NNMATCH,
                {
                    "estimand": _vals("att"),
                    "metric": _vals("mahalanobis"),
                    "n_matches": _vals("1"),
                    "exact": _vals("none"),
                    "bias_adjust": _vals("none"),
                    "vce": _vals("iid"),
                    "vce_nn": _vals("na"),
                    "caliper": _vals("none"),
                },
                _EST_SE,
                "ATT, vce(iid) vs Stata teffects nnmatch, estimate and SE to 1e-10",
                "sp.match(df, y, treat, covariates, method='nnmatch', ...)",
            ),
            _Row(
                "T2",
                _NNMATCH,
                {
                    "estimand": _vals("att"),
                    "metric": _vals("mahalanobis"),
                    "n_matches": _vals("1"),
                    "exact": _vals("none"),
                    "bias_adjust": _vals("none"),
                    "vce": _vals("robust"),
                    "vce_nn": _vals("4"),
                    "caliper": _vals("none"),
                },
                _EST_SE,
                "ATT, vce(robust, nn(4)) vs Stata teffects nnmatch, estimate and SE "
                "to 1e-10",
                "sp.match(df, y, treat, covariates, method='nnmatch', ...)",
            ),
            _Row(
                "T2",
                _NNMATCH,
                {
                    "estimand": _vals("att"),
                    "metric": _vals("mahalanobis"),
                    "n_matches": _vals("1"),
                    "exact": _vals("set"),
                    "bias_adjust": _vals("set"),
                    "vce": _vals("robust"),
                    "vce_nn": _vals("2"),
                    "caliper": _vals("none"),
                },
                _EST_SE,
                "ATT with exact matching and a partial bias adjustment vs Stata "
                "teffects nnmatch, estimate and SE to 1e-10",
                "sp.match(df, y, treat, covariates, method='nnmatch', ...)",
            ),
            _Row(
                "T2",
                _NNMATCH,
                {
                    "estimand": _vals("ate"),
                    "metric": _vals("mahalanobis"),
                    "n_matches": _vals("2"),
                    "exact": _vals("none"),
                    "bias_adjust": _vals("set"),
                    "vce": _vals("robust"),
                    "vce_nn": _vals("2"),
                    "caliper": _vals("none"),
                },
                _EST_SE,
                "ATE, two matches, bias adjustment vs Stata teffects nnmatch, "
                "estimate and SE to 1e-10",
                "sp.match(df, y, treat, covariates, method='nnmatch', ...)",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _nn_cfg("att", n_matches=("1", "4", "5")),
                _EST_SE,
                "ATT, nneighbor(1), (4), (5) vs Stata teffects nnmatch on the "
                "tie-free option data, estimate and SE to 1e-9",
                "sp.match(df, ..., method='nnmatch', n_matches=4)",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _nn_cfg("ate", n_matches=("4", "5")),
                _EST_SE,
                "ATE, nneighbor(4), (5) vs Stata teffects nnmatch, estimate "
                "and SE to 1e-9",
                "sp.match(df, ..., method='nnmatch', n_matches=4)",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _nn_cfg("att", vce_nn=("3",)),
                _EST_SE,
                "ATT, vce(robust, nn(3)) vs Stata teffects nnmatch, estimate "
                "and SE to 1e-9",
                "sp.match(df, ..., method='nnmatch', vce_nn=3)",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _nn_cfg("ate", n_matches=("4",), vce_nn=("3",)),
                _EST_SE,
                "ATE, nneighbor(4) vce(robust, nn(3)) vs Stata teffects "
                "nnmatch, estimate and SE to 1e-9",
                "sp.match(df, ..., method='nnmatch', vce_nn=3)",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _nn_cfg("att", n_matches=("4",), caliper=("set",)),
                _EST_SE,
                "ATT, nneighbor(4) caliper(3) vs Stata teffects nnmatch, "
                "estimate and SE to 1e-9. teffects reads the caliper as a "
                "condition: met, nothing changes; at caliper 2 it stops, "
                "sp.match keeps fewer matches and warns",
                "sp.match(df, ..., method='nnmatch', caliper=3.0)",
            ),
            _Row(
                "T2",
                _PSMOPT,
                _nn_cfg("ate", caliper=("set",)),
                _EST_SE,
                "ATE, caliper(3) vs Stata teffects nnmatch, estimate and SE to " "1e-9",
                "sp.match(df, ..., method='nnmatch', caliper=3.0)",
            ),
        ),
        note="Nine Stata commands on the NSW-PSID sample and nine on the "
        "tie-free option data, one cell each; the options are not crossed "
        "beyond those. A caliper is compared only where every unit meets "
        "it, which is the only case teffects estimates.",
    )
)

_HD_BASE = {
    "restriction": ("smoothness",),
    "shape": ("none",),
    "interval": ("flci",),
    "code_path": ("native",),
    "target": ("single",),
    "window": ("full",),
    "alpha": ("0.05",),
    "m_grid": ("zero_only", "zero", "positive"),
    "rm_grid": ("n/a",),
    "source": ("supplied",),
}


def _hd(**over: Tuple[str, ...]) -> Dict[str, FrozenSet[str]]:
    cfg = dict(_HD_BASE, **over)
    if "relative_magnitude" in cfg["restriction"] and "rm_grid" not in over:
        cfg["rm_grid"] = ("honestdid",)
    return {k: _vals(*v) for k, v in cfg.items()}


_HD_RM = {"restriction": ("relative_magnitude",), "interval": ("arp_conditional",)}
_HD_OPT = _RP + "test_honest_did_options_R_parity.py"
_HD_MOM = _RP + "test_honest_did_moments_R_parity.py"
_HD_STATA = _RP + "test_honest_did_lvec_window_Stata_parity.py"
_HD_BACKEND = _RP + "test_honest_did_backend_parity.py"
_HD_RESTR = _RP + "test_honest_restrictions_R_parity.py"
_add(
    _Scope(
        "honest_did",
        {
            "restriction": (
                "smoothness",
                "relative_magnitude",
                "smoothness_relative_magnitude",
            ),
            "shape": ("none", "sign", "monotone"),
            "interval": (
                "flci",
                "arp_conditional",
                "arp_c_lf",
                "arp_c_f",
                "worst_case_bias",
            ),
            "code_path": ("native", "r"),
            "target": ("single", "average", "weights"),
            "window": ("full", "restricted"),
            "alpha": ("0.05", "other"),
            "m_grid": ("zero_only", "zero", "positive"),
            "rm_grid": ("honestdid", "extended", "user", "n/a"),
            "source": _HONEST_SOURCES,
        },
        _x_honest,
        (
            _Row(
                "T2",
                _R + "10_honest_did.py",
                _hd(),
                ("diagnostic",),
                "FLCI bounds at five M on a diagonal covariance vs HonestDiD with "
                "the exact folded-normal quantile (M > 0, 1.4e-8) and vs the "
                "closed form at M = 0 (2.2e-9); Stata honestdid joins the rows",
                "sp.honest_did(res, e=0, m_grid=M_GRID, method='smoothness')",
            ),
            _Row(
                "T2",
                _R + "21_honest_relmags.py",
                _hd(**_HD_RM),
                ("diagnostic",),
                "relative-magnitudes Conditional set at five Mbar vs "
                "HonestDiD::createSensitivityResults_relativeMagnitudes: the same "
                "accepted grid points",
                "sp.honest_did(res, method='relative_magnitude')",
            ),
            _Row(
                "T2",
                _RP + "test_honest_rm_R_parity.py",
                _hd(**_HD_RM),
                ("diagnostic",),
                "through the public entry point on a supplied event study and "
                "covariance: the Conditional set equals HonestDiD's to 1e-12 "
                "(the engine alone also on correlated covariances and an "
                "average target)",
                "sp.honest_did(res, method='relative_magnitude')",
            ),
            _Row(
                "S",
                _RP + "test_honest_rm_R_parity.py",
                _hd(
                    restriction=("relative_magnitude",),
                    interval=("arp_c_lf",),
                    target=("single", "average"),
                ),
                ("diagnostic",),
                "the conditional least-favourable hybrid (HonestDiD's default) "
                "within one grid step of HonestDiD: its first-stage critical "
                "value is simulated with each side's own RNG",
                "sp.honest_did(res, method='relative_magnitude')",
            ),
            _Row(
                "T2",
                _HD_MOM,
                _hd(**_HD_RM, target=("single", "average"), source=("moments",)),
                ("diagnostic",),
                "BCdata_EventStudy as raw arrays: the Conditional set for the "
                "first post period and the post-period average equals "
                "HonestDiD's to 1e-10",
                "sp.honest_did_from_moments(",
            ),
            _Row(
                "T4",
                _HD_MOM,
                _hd(
                    **_HD_RM,
                    target=("single", "average"),
                    rm_grid=("extended",),
                    source=("moments",),
                ),
                ("diagnostic",),
                "where HonestDiD's set reaches the end of its +/-20 SD grid it "
                "reports the grid end; the default native grid extends and "
                "closes further out (asserted wider, a documented departure)",
                "sp.honest_did_from_moments(",
            ),
            _Row(
                "T2",
                _HD_OPT,
                _hd(
                    **_HD_RM,
                    target=("single", "average", "weights"),
                    alpha=("0.05", "other"),
                    rm_grid=("honestdid", "user"),
                    source=("moments",),
                ),
                ("diagnostic",),
                "BCdata_EventStudy: the Conditional set for a later post period, "
                "the average, arbitrary weights, alpha = 0.10 and a user grid "
                "(grid_lb / grid_ub / grid_points) equals HonestDiD's to 2e-13",
                "sp.honest_did_from_moments(",
            ),
            _Row(
                "T2",
                _HD_OPT,
                _hd(
                    **_HD_RM,
                    target=("single", "average", "weights"),
                    window=("full", "restricted"),
                    alpha=("0.05", "other"),
                    source=("callaway_santanna",),
                ),
                ("diagnostic",),
                "did::mpdta through sp.callaway_santanna vs did::att_gt + aggte "
                "+ HonestDiD (the README pipeline): the event study and its "
                "covariance to 1e-14, then the Conditional set for e = 0, 1, the "
                "average and weights, alpha = 0.10 and window (-3, 2) to 2e-13",
                "sp.honest_did(cs, **kw)",
            ),
            _Row(
                "S",
                _HD_OPT,
                _hd(
                    restriction=("relative_magnitude",),
                    interval=("arp_c_lf",),
                    target=("single", "average"),
                    source=("callaway_santanna",),
                ),
                ("diagnostic",),
                "did::mpdta through sp.callaway_santanna: the default C-LF hybrid "
                "for e = 0 and the average within one grid step of HonestDiD "
                "(each side simulates the critical value with its own RNG)",
                "sp.honest_did(cs, **kw)",
            ),
            _Row(
                "T1",
                _HD_OPT,
                _hd(
                    target=("single", "average", "weights"),
                    window=("full", "restricted"),
                    alpha=("0.05", "other"),
                    m_grid=("zero_only",),
                    source=("moments", "callaway_santanna"),
                ),
                ("diagnostic",),
                "the FLCI at M = 0 on correlated covariances: the closed form "
                "(minimum-variance estimator unbiased under a linear trend, "
                "+/- z sd), computed in R with base linear algebra, to 1e-12",
                "sp.honest_did(cs, **kw)",
            ),
            _Row(
                "T4",
                _HD_OPT,
                _hd(
                    target=("single", "average", "weights"),
                    window=("full", "restricted"),
                    alpha=("0.05", "other"),
                    source=("moments", "callaway_santanna"),
                ),
                ("diagnostic",),
                "FLCI on correlated covariances vs HonestDiD's program at tight "
                "tolerance: the half-length (the program's optimum) agrees to "
                "3e-10, the centre only to 1e-5 of the half-length at M > 0 "
                "(the half-length is flat in h, so each solver's noise moves "
                "the centre); M = 0 is the closed form",
                "sp.honest_did(cs, **kw)",
            ),
            _Row(
                "T4",
                _HD_MOM,
                _hd(
                    target=("single", "average"),
                    m_grid=("positive",),
                    source=("moments",),
                ),
                ("diagnostic",),
                "FLCI vs HonestDiD as shipped: within 1e-3, its Monte Carlo "
                "folded-normal quantile",
                "sp.honest_did_from_moments(",
            ),
            _Row(
                "T2",
                _HD_STATA,
                _hd(
                    **_HD_RM,
                    target=("single", "average", "weights"),
                    window=("restricted",),
                    source=("etwfe",),
                ),
                ("diagnostic",),
                "a windowed Poisson ETWFE event study: the native Conditional set "
                "for e = 1, the average and weights equals R HonestDiD run on "
                "the same windowed inputs to 1e-10 (needs R; marked slow)",
                "sp.honest_did(fit, backend='r', ...) vs sp.honest_did(fit, ...)",
            ),
            _Row(
                "T4",
                _HD_STATA,
                _hd(
                    restriction=("smoothness", "relative_magnitude"),
                    interval=("flci", "arp_conditional"),
                    target=("single", "average"),
                    window=("restricted",),
                    rm_grid=("n/a", "honestdid"),
                    source=("etwfe",),
                ),
                ("diagnostic",),
                "vs Stata honestdid 1.3.0: the FLCI to 1e-5 (Stata simulates the "
                "folded-normal quantile) and the Conditional set to one grid "
                "step (the port lays its grid out from the estimates)",
                "sp.honest_did(fit, method=...)",
            ),
            _Row(
                "T2",
                _HD_BACKEND,
                _hd(
                    **_HD_RM,
                    source=("callaway_santanna",),
                ),
                ("diagnostic",),
                "did::mpdta Callaway-Sant'Anna fit: the native Conditional set "
                "equals R HonestDiD run through backend='r' on the same "
                "covariance to 1e-10 (needs R)",
                "sp.honest_did(cs_result, method='relative_magnitude')",
            ),
            _Row(
                "T2",
                _HD_RESTR,
                _hd(
                    restriction=("smoothness",),
                    shape=("none", "sign", "monotone"),
                    interval=("arp_conditional",),
                    target=("single", "average"),
                    rm_grid=("honestdid",),
                    source=("moments",),
                ),
                ("diagnostic",),
                "smoothness with the conditional test, alone and with each sign "
                "and shape restriction (DeltaSD / SDPB / SDNB / SDI / SDD), on a "
                "correlated 3 pre / 3 post and a one-post-period event study: "
                "the same grid (identified set at beta = 0 +/- 20 sd) and the "
                "same accepted points as HonestDiD::computeConditionalCS_* to "
                "1e-12",
                "sp.honest_did_from_moments(",
            ),
            _Row(
                "T2",
                _HD_RESTR,
                _hd(
                    restriction=("relative_magnitude", "smoothness_relative_magnitude"),
                    shape=("none", "sign", "monotone"),
                    interval=("arp_conditional",),
                    target=("single", "average"),
                    source=("moments",),
                ),
                ("diagnostic",),
                "relative magnitudes with sign / shape restrictions (DeltaRMPB "
                "... RMD) and of second differences (DeltaSDRM and its sign / "
                "shape variants): the Conditional set equals "
                "HonestDiD::createSensitivityResults_relativeMagnitudes to 1e-12",
                "sp.honest_did_from_moments(",
            ),
            _Row(
                "T4",
                _HD_RESTR,
                _hd(
                    restriction=("smoothness",),
                    shape=("none", "sign", "monotone"),
                    interval=("arp_c_f",),
                    target=("single",),
                    rm_grid=("honestdid",),
                    source=("moments",),
                ),
                ("diagnostic",),
                "the conditional-FLCI hybrid within one grid step of HonestDiD "
                "with the exact folded-normal quantile (two of the shipped "
                "package): its grid and first stage are the level alpha/10 "
                "FLCI, whose centre the two solvers pin only to ~1e-5",
                "sp.honest_did_from_moments(",
            ),
            _Row(
                "S",
                _HD_RESTR,
                _hd(
                    restriction=(
                        "smoothness",
                        "relative_magnitude",
                        "smoothness_relative_magnitude",
                    ),
                    shape=("none", "sign"),
                    interval=("arp_c_lf",),
                    target=("single",),
                    rm_grid=("honestdid",),
                    source=("moments",),
                ),
                ("diagnostic",),
                "the C-LF hybrid under DeltaSD / SDNB / RMPB / SDRM: HonestDiD's "
                "bound inside the spread of the native bound over ten seeds "
                "plus one grid step (each side simulates the first-stage "
                "critical value with its own RNG)",
                "sp.honest_did_from_moments(",
            ),
        ),
        primary=("diagnostic",),
        defaults={"shape": "none"},
        note="The diagnostic is the confidence set [ci_lower, ci_upper] at each M. "
        "backend='r' (code_path 'r') runs HonestDiD itself, so a comparison with "
        "R would compare the reference with itself; it has no row. "
        "interval='worst_case_bias' is the warned fallback when no joint "
        "covariance is available and has no reference. The FLCI's half-length "
        "is pinned to 3e-10 on correlated covariances, its centre only to 1e-5 "
        "of the half-length (T4); at M = 0 it is a closed form (T1). m_grid "
        "'zero_only' is a grid of M = 0 alone; 'zero' also has M > 0.",
    )
)


#: Functions with a configuration map.
def _register_satellite_maps() -> None:
    """Maps kept in their own modules (they read this module's primitives)."""
    import sys

    from . import (
        _validation_scope_did2x2,
        _validation_scope_dynpanel,
        _validation_scope_fe,
        _validation_scope_ldv,
        _validation_scope_te,
    )

    _validation_scope_ldv.register(_add, sys.modules[__name__])
    _validation_scope_fe.register(_add, sys.modules[__name__])
    _validation_scope_te.register(_add, sys.modules[__name__])
    _validation_scope_dynpanel.register(_add, sys.modules[__name__])
    _validation_scope_did2x2.register(_add, sys.modules[__name__])


_register_satellite_maps()

SCOPE_FUNCTIONS = tuple(sorted(SCOPES))


class ValidationScope(dict):
    """Evidence for one fitted configuration (a ``dict``; prints as a table).

    Returned by :func:`validation_scope`. Keys: ``function``,
    ``configuration``, ``status`` (``"covered"``, ``"estimate_only"``,
    ``"stochastic_only"``, ``"disclosure_only"`` or ``"not_covered"``),
    ``outputs`` (per-output status and evidence), ``matched``,
    ``near_misses``, ``unchecked``, ``invariant`` and ``note``.

    Examples
    --------
    >>> import statspai as sp
    >>> scope = sp.validation_scope(function="causal_forest",
    ...                             treatment="binary", trees="2000",
    ...                             tuning="grf_defaults", design="iid")
    >>> scope["status"]
    'stochastic_only'
    >>> scope["outputs"]["estimate"]["status"]
    'seed_equivalence'
    """

    __slots__ = ()

    def _render(self) -> str:
        lines = [
            f"Validation scope: sp.{self['function']}  [{self['status']}]",
            "configuration: "
            + ", ".join(f"{k}={v}" for k, v in self["configuration"].items()),
        ]
        for out, info in self["outputs"].items():
            arts = ", ".join(
                f"{e['kind']} {e['artifact'].rsplit('/', 1)[-1]}"
                for e in info["evidence"]
            )
            lines.append(f"  {out:<10} {info['status']:<20} {arts}")
        if self["unchecked"]:
            lines.append(
                "unchecked (not recorded by the fit): " + ", ".join(self["unchecked"])
            )
        if self["near_misses"]:
            lines.append("evidence differing in one dimension:")
            for row in self["near_misses"]:
                lines.append(
                    f"  {row['kind']:<3} {row['artifact']} (differs in "
                    f"{row['differs_in']})"
                )
        if self.get("note"):
            lines.append(f"note: {self['note']}")
        return "\n".join(lines)

    __str__ = _render
    __repr__ = _render


def _match(
    scope: _Scope, row: _Row, cfg: Mapping[str, Optional[str]], output: str
) -> Tuple[bool, List[str]]:
    """Whether ``row`` covers ``output`` at ``cfg``; the dimensions it misses."""
    misses: List[str] = []
    unknown = False
    for dim in scope.dimensions:
        if scope.ignored(dim, output):
            continue
        got = cfg.get(dim)
        if got is None:
            unknown = True
        elif got not in row.config[dim]:
            misses.append(dim)
    return (not misses and not unknown), misses


def validation_scope(
    result: Any = None, *, function: Optional[str] = None, **configuration: Any
) -> ValidationScope:
    """Which validation artifacts cover the configuration and outputs that were run?

    Parameters
    ----------
    result : fitted result, optional
        A result from one of the validation-suite estimators. Its
        configuration (estimator variant, inference option, treatment type,
        code path, ...) is read from the fitted object.
    function : str, optional
        Name of the estimator (``"dml"``, ``"callaway_santanna"``, ...),
        required when ``result`` is omitted and otherwise inferred.
    **configuration
        Dimension values to use instead of (or in addition to) those read
        from ``result``, e.g. ``validation_scope(function="dml",
        model="plr", learners="default", ...)``. Values must lie in the
        dimension's domain.

    Returns
    -------
    ValidationScope
        A ``dict`` with ``function``, ``configuration``, ``status``,
        ``outputs`` (for each of estimate / se / coverage / diagnostic: its
        status and the matching artifacts), ``matched`` (artifacts covering
        at least one output, with the outputs they cover), ``near_misses``
        (artifacts differing in one dimension), ``unchecked`` (dimensions
        the fit did not record), ``invariant`` (dimensions ignored for some
        outputs, with the reason) and ``note``.

    Raises
    ------
    MethodIncompatibility
        For a function without a map, an unknown dimension, or a value
        outside a dimension's domain.

    Examples
    --------
    >>> import statspai as sp
    >>> card = sp.datasets.card_1995()
    >>> fit = sp.iv("lwage ~ exper + expersq + black + south + smsa"
    ...             " + (educ ~ nearc4)", data=card, method="liml",
    ...             robust="hc3")
    >>> scope = sp.validation_scope(fit)
    >>> scope["status"]
    'estimate_only'
    >>> scope["outputs"]["se"]["status"]
    'not_covered'

    Notes
    -----
    The registry tier of a function summarises its evidence; this function
    answers the narrower question a user or agent has after a fit. An
    option absent from the map is reported as not covered rather than
    inferred from a neighbouring configuration, and a point-estimate row
    never vouches for a standard error it did not compare.
    """
    from .exceptions import MethodIncompatibility

    if type(result).__name__ == "EstimationResult" and hasattr(result, "underlying"):
        # sp.causal_question(...).estimate() wraps the estimator it ran.
        result = result.underlying
    name = function or _function_of(result)
    if (
        name == "etwfe"
        and result is not None
        and _mi(result).get("estimator") == "etwfe_glm"
    ):
        # Same entry point, different option axes: the nonlinear branch.
        name = "etwfe_glm"
    if (
        name == "psm"
        and result is not None
        and _mi(result).get("estimator") == "nnmatch"
    ):
        # sp.match(method='nnmatch'): covariate matching, its own map.
        name = "nnmatch"
    if name is None or name not in SCOPES:
        raise MethodIncompatibility(
            f"No configuration-level evidence map for {name!r}.",
            recovery_hint=(
                "validation_scope covers these estimators: "
                + ", ".join(SCOPE_FUNCTIONS)
                + ". For other functions read "
                "sp.describe_function(name)['validation_notes']."
            ),
            diagnostics={"function": name},
        )
    scope = SCOPES[name]
    cfg: Dict[str, Optional[str]] = {d: None for d in scope.dimensions}
    if result is not None:
        cfg.update(scope.extract(result))
    else:
        cfg.update({k: v for k, v in scope.defaults.items() if not callable(v)})
    for key, value in configuration.items():
        if key not in scope.domains:
            raise MethodIncompatibility(
                f"{key!r} is not a dimension of the {name} map.",
                recovery_hint=f"Dimensions: {', '.join(scope.dimensions)}.",
                diagnostics={"function": name, "dimension": key},
            )
        val = _lower(value)
        if val not in scope.domains[key]:
            raise MethodIncompatibility(
                f"{value!r} is not a value of the {name} map's {key!r} dimension.",
                recovery_hint=f"Values: {', '.join(scope.domains[key])}.",
                diagnostics={"function": name, "dimension": key, "value": repr(value)},
            )
        cfg[key] = val
    if result is None:
        for key, value in scope.defaults.items():
            if callable(value) and key not in configuration:
                cfg[key] = value(cfg)

    outputs: Dict[str, Dict[str, Any]] = {}
    matched: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    near: List[Dict[str, Any]] = []
    for out in OUTPUTS:
        evidence = []
        for row in scope.rows:
            if out not in row.outputs:
                continue
            ok, misses = _match(scope, row, cfg, out)
            rec = {
                "kind": row.kind,
                "artifact": row.artifact,
                "compares": row.compares,
                "entry_point": row.entry_point,
            }
            if ok:
                evidence.append(rec)
                row_key = (row.kind, row.artifact, row.compares)
                entry = matched.setdefault(
                    row_key,
                    {
                        **rec,
                        "configuration": {k: sorted(v) for k, v in row.config.items()},
                        "outputs": [],
                    },
                )
                entry["outputs"].append(out)
            elif len(misses) == 1 and not any(
                n["artifact"] == row.artifact and n["differs_in"] == misses[0]
                for n in near
            ):
                near.append({**rec, "differs_in": misses[0]})
        if (
            not evidence
            and out not in scope.primary
            and not any(out in row.outputs for row in scope.rows)
        ):
            continue
        kinds = [e["kind"] for e in evidence]
        status = "not_covered"
        for k in KINDS:
            if k in kinds:
                status = _STATUS_OF_KIND[k]
                break
        outputs[out] = {"status": status, "evidence": evidence}

    prim = [outputs.get(o, {"status": "not_covered"})["status"] for o in scope.primary]
    if all(s == "reference" for s in prim):
        status = "covered"
    elif (
        "estimate" in scope.primary
        and prim[scope.primary.index("estimate")] == "reference"
    ):
        status = "estimate_only"
    elif any(
        s
        in {"reference", "seed_equivalence", "stochastic_screen", "coverage_simulation"}
        for s in (o["status"] for o in outputs.values())
    ):
        status = "stochastic_only"
    elif any(o["status"] == "disclosure" for o in outputs.values()):
        status = "disclosure_only"
    else:
        status = "not_covered"

    unchecked = [
        d
        for d in scope.dimensions
        if cfg.get(d) is None and not all(scope.ignored(d, o) for o in scope.primary)
    ]
    return ValidationScope(
        {
            "function": name,
            "configuration": cfg,
            "status": status,
            "outputs": outputs,
            "matched": list(matched.values()),
            "near_misses": near,
            "unchecked": unchecked,
            "invariant": {
                d: {"outputs": list(o), "reason": r}
                for d, (o, r) in scope.invariant.items()
            },
            "note": scope.note,
        }
    )


_METHOD_TO_FUNCTION = (
    ("did_multiplegt_dyn", "did_multiplegt_dyn"),
    ("callaway and sant'anna", "callaway_santanna"),
    ("sun and abraham", "sun_abraham"),
    ("sun-abraham", "sun_abraham"),
    ("rd estimation", "rdrobust"),
    ("density test", "rddensity"),
    ("ols event study (twfe)", "event_study"),
    ("etwfe", "etwfe"),
    ("gardner", "gardner_did"),
    ("imputation estimator", "did_imputation"),
    ("bjs imputation", "did_imputation"),
    ("synthetic control", "synth"),
    ("synthetic difference", "sdid"),
    ("double ml", "dml"),
    ("g-computation", "g_computation"),
    # sp.g_computation(by_arm=True); "regression adjustment" alone would
    # also catch sp.lm_lin.
    ("outcome regression per arm", "g_computation"),
    ("regression adjustment (ipw-weighted", "g_computation"),
    ("matching", "psm"),
)


def _function_of(result: Any) -> Optional[str]:
    if result is None:
        return None
    # The exact public call, when the estimator records it
    # (``core._call_record``): no guessing from labels.
    from .core._call_record import get_call_record

    record = get_call_record(result)
    if record is not None and record.function in SCOPES:
        return record.function
    if type(result).__name__ == "DataFrame":
        # sp.honest_did returns its sensitivity table, labelled in attrs.
        at = getattr(result, "attrs", None) or {}
        return "honest_did" if at.get("function") == "honest_did" else None
    if type(result).__name__ == "CausalForest":
        return "causal_forest"
    if type(result).__name__ == "FeolsResult":
        return "fast.feols"
    if type(result).__name__ == "PanelResults":
        return "panel"
    mi = _mi(result)
    model_type = _lower(mi.get("model_type")) or ""
    if model_type.startswith("iv") or "2sls" in model_type:
        return "iv"
    if model_type == "ols":
        return "regress"
    method = (
        (_lower(getattr(result, "method", None)) or "")
        + " "
        + (_lower(mi.get("estimator_label")) or "")
        + " "
        + (_lower(mi.get("estimator")) or "")
    )
    for token, fn in _METHOD_TO_FUNCTION:
        if token in method:
            return fn
    return None
