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

This module records, for the twelve estimators of the paper's validation
suite, which configurations each artifact exercised and which outputs it
compared. Every dimension has an enumerated domain; every row lists the
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
#: distribution. Neither is implied by an ``se`` row.
OUTPUTS: Tuple[str, ...] = (
    "estimate",
    "se",
    "coverage",
    "diagnostic",
    "vcov",
    "joint_test",
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

    @property
    def dimensions(self) -> Tuple[str, ...]:
        return tuple(self.domains)

    def ignored(self, dim: str, output: str) -> bool:
        spec = self.invariant.get(dim)
        return spec is not None and output in spec[0]


def _mi(result: Any) -> Dict[str, Any]:
    return dict(getattr(result, "model_info", None) or {})


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


def _x_iv(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    model_type = (_lower(mi.get("model_type")) or "").replace("iv-", "")
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


def _x_cs(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    anticipation = mi.get("anticipation")
    return {
        "estimator": _lower(mi.get("estimator")),
        "control_group": _lower(mi.get("control_group")),
        "weights": "weighted" if mi.get("weighted") else "none",
        "covariates": None if "covariates" not in mi else _set(mi.get("covariates")),
        "inference": "bootstrap" if mi.get("bstrap") else "analytic",
        "base_period": _lower(mi.get("base_period")),
        "anticipation": (
            None if anticipation is None else ("0" if int(anticipation) == 0 else ">0")
        ),
        "clustering": _set(mi.get("clustervars")),
    }


def _x_sa(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
    return {
        "control_group": _lower(mi.get("control_group")),
        "aggregation": _lower(mi.get("summary_aggregation")),
    }


def _x_rd(r: Any) -> Dict[str, Optional[str]]:
    mi = _mi(r)
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
    return {
        "method": _lower(mi.get("estimator")),
        "se_method": _lower(mi.get("se_method")),
        "code_path": _lower(mi.get("backend")),
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


# --------------------------------------------------------------------------- #
#  The evidence map
# --------------------------------------------------------------------------- #

_R = "tests/r_parity/"
_RP = "tests/reference_parity/"
_B = "tests/coverage_monte_carlo/results_b1000/coverage_b1000.json"
_VCE = _RP + "test_vce_grammar_stata_parity.py"
_ATTACH = _RP + "test_validation_entry_points.py"
_EST_SE = ("estimate", "se")
_EST = ("estimate",)
_COV = ("coverage",)

_REG_VCE = (
    "classical",
    "hc0",
    "hc1",
    "hc2",
    "hc3",
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
                _R + "55_hc2_hc3.py",
                {"vce": _vals("hc2", "hc3"), "weights": _vals("none")},
                _EST_SE,
                "HC2 / HC3 SEs vs sandwich::vcovHC",
                "sp.regress(robust='hc2'|'hc3')",
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
                "B",
                _B,
                {"vce": _vals("hc1"), "weights": _vals("none")},
                _COV,
                "95% CI coverage on a known-truth RCT DGP",
                "sp.regress(robust='hc1')",
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
        },
        _x_iv,
        (
            _Row(
                "T2",
                _RP + "test_iv_card_aer_parity.py",
                {
                    "estimator": _vals("2sls"),
                    "vce": _vals("classical", "hc1", "cr1"),
                    "identification": _vals("just", "over"),
                    "absorb": _vals("none"),
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
                },
                _COV,
                "95% CI coverage, strong single instrument (via sp.ivreg)",
                "sp.ivreg(robust='hc1')",
            ),
        ),
        invariant={"vce": _VCE_INVARIANT},
        note="sp.iv and sp.ivreg share this map; they are asserted bit-identical "
        "on these configurations in " + _ATTACH + ". Module 59 itself runs "
        "sp.liml, a separate code path, so its row is attached through that test.",
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

_CS_DOMAINS = {
    "estimator": ("dr", "reg", "ipw"),
    "control_group": ("nevertreated", "notyettreated"),
    "weights": ("none", "weighted"),
    "covariates": ("none", "set"),
    "inference": ("analytic", "bootstrap"),
    "base_period": ("universal", "varying"),
    "anticipation": ("0", ">0"),
    "clustering": ("none", "set"),
}
_add(
    _Scope(
        "callaway_santanna",
        _CS_DOMAINS,
        _x_cs,
        (
            _Row(
                "T2",
                _R + "04_csdid.py",
                {
                    "estimator": _vals("reg"),
                    "control_group": _vals("nevertreated"),
                    "weights": _vals("none"),
                    "covariates": _vals("none"),
                    "inference": _vals("analytic"),
                    "base_period": _vals("universal"),
                    "anticipation": _vals("0"),
                    "clustering": _vals("none"),
                },
                _EST_SE,
                "simple / dynamic / group ATT and analytic SEs vs did and csdid",
                "sp.callaway_santanna(estimator='reg') + sp.aggte(bstrap=False)",
            ),
            _Row(
                "T2",
                _RP + "test_cs_weighted_parity.py",
                {
                    "estimator": _vals("dr", "reg", "ipw"),
                    "control_group": _vals("nevertreated", "notyettreated"),
                    "weights": _vals("none", "weighted"),
                    "covariates": _vals("none", "set"),
                    "inference": _vals("analytic"),
                    "base_period": _vals("universal"),
                    "anticipation": _vals("0"),
                    "clustering": _vals("none"),
                },
                _EST_SE,
                "simple / dynamic / group ATT and analytic SE vs did::att_gt over "
                "dr/reg/ipw x never/not-yet x weights x covariates (72 cells)",
                "sp.callaway_santanna(...)",
            ),
            _Row(
                "T2",
                _RP + "test_aggte_vcov_r_parity.py",
                {
                    "estimator": _vals("dr"),
                    "control_group": _vals("nevertreated"),
                    "weights": _vals("none"),
                    "covariates": _vals("none"),
                    "inference": _vals("analytic"),
                    "base_period": _vals("universal"),
                    "anticipation": _vals("0"),
                    "clustering": _vals("none"),
                },
                ("vcov",),
                "every entry of the dynamic event-study covariance (off-diagonal "
                "blocks included) vs did::aggte influence functions; the matrix "
                "sp.honest_did and sp.uniform_bands consume",
                "sp.aggte(type='dynamic')",
            ),
            _Row(
                "S",
                _RP + "test_cs_inference_parity.py",
                {
                    "estimator": _vals("dr"),
                    "control_group": _vals("nevertreated"),
                    "weights": _vals("none"),
                    "covariates": _vals("none"),
                    "inference": _vals("bootstrap"),
                    "base_period": _vals("universal"),
                    "anticipation": _vals("0"),
                    "clustering": _vals("none", "set"),
                },
                ("se",),
                "multiplier-bootstrap SEs within 8-10% and uniform critical values "
                "within 5% of did at 9,999 draws (one seed per side; a tolerance "
                "screen, not an equivalence test)",
                "sp.callaway_santanna(bstrap=True)",
            ),
            _Row(
                "B",
                _B,
                {
                    "estimator": _vals("reg"),
                    "control_group": _vals("nevertreated"),
                    "weights": _vals("none"),
                    "covariates": _vals("none"),
                    "inference": _vals("analytic"),
                    "base_period": _vals("universal"),
                    "anticipation": _vals("0"),
                    "clustering": _vals("none"),
                },
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
            )
        },
    )
)

_add(
    _Scope(
        "sun_abraham",
        {
            "control_group": ("nevertreated", "lastcohort"),
            "aggregation": ("fixest_att", "event_time"),
        },
        _x_sa,
        (
            _Row(
                "T2",
                _R + "05_sunab.py",
                {
                    "control_group": _vals("nevertreated"),
                    "aggregation": _vals("fixest_att"),
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
                },
                _COV,
                "95% CI coverage of the default event-time aggregate",
                "sp.sun_abraham()",
            ),
        ),
        note="share_variance changes the per-period SEs only (module 05 pins both "
        "conventions: fixest for fixed shares, eventstudyinteract for the Prop. 3 "
        "term); the overall aggregate's SE treats cohort shares as fixed on both "
        "settings.",
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
                "conventional / robust estimates, SEs and h, b vs rdrobust (R and Stata)",
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
        ),
        note="Other bandwidth selectors are pinned for sp.rdbwselect (Track A module "
        "88), not for this entry point.",
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
        ),
        primary=("diagnostic",),
    )
)

_add(
    _Scope(
        "synth",
        {
            "v_method": ("equal", "nested"),
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
        ),
        primary=("estimate",),
        note="Classical SCM only; other sp.synth methods have their own modules. "
        "Placebo p-values are not part of any row.",
    )
)

_add(
    _Scope(
        "sdid",
        {
            "method": ("sdid", "sc", "did"),
            "se_method": ("placebo", "bootstrap", "jackknife"),
            "code_path": ("native", "synthdid", "r"),
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
                },
                _EST,
                "point ATT vs synthdid (the SE is not part of this row)",
                "sp.sdid(backend='native')",
            ),
            _Row(
                "B",
                _B,
                {
                    "method": _vals("sdid"),
                    "se_method": _vals("placebo"),
                    "code_path": _vals("native"),
                },
                _COV,
                "placebo-SE CI coverage, one treated unit",
                "sp.sdid()",
            ),
        ),
        invariant={
            "se_method": (
                _EST,
                "the SE method resamples after the unit and time weights are "
                "solved; the ATT is unchanged",
            )
        },
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
        ),
        note="With default (flexible) learners the estimate depends on the learner, so "
        "deterministic parity exists only for fixed learners and folds; the default-"
        "learner PLR interval under-covers (0.883) on the Track B design.",
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

_add(
    _Scope(
        "psm",
        {
            "method": ("nearest", "caliper", "kernel", "radius", "stratify"),
            "distance": ("propensity", "mahalanobis", "logit"),
            "n_matches": tuple(str(k) for k in range(1, 11)),
            "replace": ("true", "false"),
            "caliper": ("none", "set"),
            "bias_correction": ("true", "false"),
            "se_method": ("abadie_imbens", "abadie_imbens_2016", "bootstrap"),
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
        ),
        invariant={
            "se_method": (_EST, "the variance estimator does not change the matches")
        },
        note="The default Abadie-Imbens (2006) SE has no reference row.",
    )
)

#: Functions with a configuration map.
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
                    f"  {row['kind']:<3} {row['artifact']} (differs in {row['differs_in']})"
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
    ...             " + (educ ~ nearc4)", data=card, robust="hc3")
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
    if name is None or name not in SCOPES:
        raise MethodIncompatibility(
            f"No configuration-level evidence map for {name!r}.",
            recovery_hint=(
                "validation_scope covers the twelve validation-suite estimators: "
                + ", ".join(SCOPE_FUNCTIONS)
                + ". For other functions read sp.describe_function(name)['validation_notes']."
            ),
            diagnostics={"function": name},
        )
    scope = SCOPES[name]
    cfg: Dict[str, Optional[str]] = {d: None for d in scope.dimensions}
    if result is not None:
        cfg.update(scope.extract(result))
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
                key = (row.kind, row.artifact, row.compares)
                entry = matched.setdefault(
                    key,
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
    ("callaway and sant'anna", "callaway_santanna"),
    ("sun and abraham", "sun_abraham"),
    ("sun-abraham", "sun_abraham"),
    ("rd estimation", "rdrobust"),
    ("density test", "rddensity"),
    ("synthetic control", "synth"),
    ("synthetic difference", "sdid"),
    ("double ml", "dml"),
    ("matching", "psm"),
)


def _function_of(result: Any) -> Optional[str]:
    if result is None:
        return None
    if type(result).__name__ == "CausalForest":
        return "causal_forest"
    if type(result).__name__ == "FeolsResult":
        return "fast.feols"
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
