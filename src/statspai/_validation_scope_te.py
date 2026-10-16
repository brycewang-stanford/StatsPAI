"""Validation-scope maps for the treatment-effect estimators.

``sp.aipw`` / ``ipw`` / ``tmle`` / ``ebalance`` / ``g_computation``. Same
contract as :mod:`statspai.validation_scope`: each row lists the
configurations its artifact runs and the outputs it compares.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def register(add: Callable[[Any], None], vs: Any) -> None:
    _Row, _Scope, _vals, _mi, _lower, _set = (
        vs._Row,
        vs._Scope,
        vs._vals,
        vs._mi,
        vs._lower,
        vs._set,
    )
    RP = "tests/reference_parity/"
    EST_SE = ("estimate", "se")
    TEFF = RP + "test_teffects_design_stata_parity.py"
    TEFF_R = RP + "test_teffects_R_parity.py"
    PROBIT = RP + "test_teffects_probit_stata.py"
    se_invariant = (
        ("estimate",),
        "the point estimate is computed before, and independently of, the "
        "variance estimator",
    )

    def cfg(**dims: Any) -> Dict[str, Any]:
        return {k: _vals(*v) for k, v in dims.items()}

    # ------------------------------------------------------------------ aipw
    def x_aipw(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        clipped = mi.get("n_propensity_clipped")
        return {
            "estimand": _lower(getattr(r, "estimand", None)),
            "cross_fit": {True: "true", False: "false"}.get(mi.get("cross_fit")),
            "se_method": _lower(mi.get("se_method")),
            "outcome_model": _lower(mi.get("outcome_model")),
            "propensity": _lower(mi.get("propensity")),
            "trim": None if clipped is None else ("binding" if clipped else "inactive"),
            "weights": _set(mi.get("weights")),
            "cluster": _set(mi.get("cluster")),
        }

    def aipw_cfg(**over: Any) -> Dict[str, Any]:
        base = dict(
            estimand=["ate"],
            cross_fit=["false"],
            se_method=["sandwich"],
            outcome_model=["linear"],
            propensity=["logit"],
            trim=["inactive"],
            weights=["none"],
            cluster=["none"],
        )
        base.update(over)
        return cfg(**base)

    add(
        _Scope(
            "aipw",
            {
                "estimand": ("ate", "att"),
                "cross_fit": ("true", "false"),
                "se_method": ("influence", "sandwich"),
                "outcome_model": ("linear", "logit", "probit", "poisson"),
                "propensity": ("logit", "known"),
                "trim": ("inactive", "binding"),
                "weights": ("none", "set"),
                "cluster": ("none", "set"),
            },
            x_aipw,
            (
                _Row(
                    "T2",
                    TEFF,
                    aipw_cfg(weights=["none", "set"], cluster=["none", "set"]),
                    EST_SE,
                    "ATE, potential-outcome means and their SEs vs Stata "
                    "teffects aipw: plain, pweights, cluster, both; 1e-6",
                    "sp.aipw(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_aipw_outcome_models_stata_parity.py",
                    aipw_cfg(
                        outcome_model=["logit", "probit", "poisson"],
                        weights=["none", "set"],
                    ),
                    EST_SE,
                    "logit / probit / poisson outcome models vs Stata teffects "
                    "aipw, unweighted and weighted, 1e-6",
                    "sp.aipw(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_aipw_outcome_models_stata_parity.py",
                    aipw_cfg(
                        outcome_model=["logit", "probit", "poisson"], cluster=["set"]
                    ),
                    EST_SE,
                    "nonlinear outcome models, clustered, vs Stata teffects aipw",
                    "sp.aipw(...)",
                ),
                _Row(
                    "T2",
                    TEFF_R,
                    aipw_cfg(se_method=["influence"]),
                    EST_SE,
                    "ATE and influence-function SE vs R AIPW, 1e-10",
                    "sp.aipw(...)",
                ),
                _Row(
                    "T1",
                    TEFF_R,
                    aipw_cfg(estimand=["att"], se_method=["influence"]),
                    EST_SE,
                    "ATT equals the hand-written doubly robust ATT and its "
                    "influence-function SE from the same nuisances",
                    "sp.aipw(...)",
                ),
                _Row(
                    "T1",
                    RP + "test_aipw_crossfit_identity.py",
                    aipw_cfg(
                        estimand=["ate", "att"],
                        cross_fit=["true"],
                        se_method=["influence"],
                    ),
                    EST_SE,
                    "the default cross-fitted fit equals an independent rebuild "
                    "(same fold draw, statsmodels logit and per-arm OLS, AIPW "
                    "scores, sd(psi)/sqrt(n)), 1e-9",
                    "sp.aipw(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_yuksel_aydede_causal_ml_parity.py",
                    aipw_cfg(se_method=["influence"]),
                    EST_SE,
                    "ATE and SE vs the textbook's R code, 1e-8",
                    "sp.aipw(...)",
                ),
            ),
            invariant={"se_method": se_invariant},
            note="trim reads n_propensity_clipped: the default 0.01 bound "
            "clips nothing on most data, and such fits equal the untrimmed "
            "estimator.",
        )
    )

    # ------------------------------------------------------------------ ipw
    def x_ipw(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        estimand = _lower(mi.get("estimand"))
        trim = mi.get("trim", mi.get("trimming_threshold"))
        return {
            "estimand": "atc" if estimand == "atu" else estimand,
            "se_method": _lower(mi.get("se_method")),
            "ps_model": _lower(mi.get("ps_model")) or "known",
            "normalized": {True: "true", False: "false"}.get(mi.get("normalized")),
            "trim": None if trim is None else ("zero" if not trim else "set"),
            "weights": _set(mi.get("weights")),
            "cluster": _set(mi.get("cluster")),
        }

    def ipw_cfg(**over: Any) -> Dict[str, Any]:
        base = dict(
            estimand=["ate", "att"],
            se_method=["sandwich"],
            ps_model=["logit"],
            normalized=["true"],
            trim=["zero"],
            weights=["none"],
            cluster=["none"],
        )
        base.update(over)
        return cfg(**base)

    add(
        _Scope(
            "ipw",
            {
                "estimand": ("ate", "att", "atc", "ato", "atm"),
                "se_method": ("sandwich", "bootstrap"),
                "ps_model": ("logit", "probit", "known"),
                "normalized": ("true", "false"),
                "trim": ("zero", "set"),
                "weights": ("none", "set"),
                "cluster": ("none", "set"),
            },
            x_ipw,
            (
                _Row(
                    "T2",
                    TEFF,
                    ipw_cfg(weights=["none", "set"], cluster=["none", "set"]),
                    EST_SE,
                    "ATE / ATT and sandwich SEs vs Stata teffects ipw: plain, "
                    "pweights, cluster, both; 1e-6",
                    "sp.ipw(...)",
                ),
                _Row(
                    "T2",
                    PROBIT,
                    ipw_cfg(ps_model=["logit", "probit"]),
                    EST_SE,
                    "logit and probit propensity models vs Stata teffects ipw "
                    "on Lalonde",
                    "sp.ipw(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_barrett_causal_inference_in_r_parity.py",
                    ipw_cfg(estimand=["ate", "att", "atm", "ato"]),
                    EST_SE,
                    "ATE / ATT / ATM / ATO and sandwich SEs vs R propensity::ipw "
                    "(its SE carries n/(n - 1): the documented divisor), 1e-9",
                    "sp.ipw(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_barrett_causal_inference_in_r_parity.py",
                    ipw_cfg(estimand=["atc"]),
                    ("estimate",),
                    "ATC vs R propensity::ipw",
                    "sp.ipw(...)",
                ),
                _Row(
                    "T1",
                    RP + "test_ipw_parity.py",
                    ipw_cfg(estimand=["ate", "att", "atc"], se_method=["bootstrap"]),
                    ("estimate",),
                    "saturated design: equals the closed-form cell-mean "
                    "estimators, 1e-9",
                    "sp.ipw(...)",
                ),
                _Row(
                    "T1",
                    RP + "test_ipw_parity.py",
                    ipw_cfg(
                        estimand=["ate"], se_method=["bootstrap"], normalized=["false"]
                    ),
                    ("estimate",),
                    "saturated design: the Horvitz-Thompson ATE equals the "
                    "closed form",
                    "sp.ipw(...)",
                ),
            ),
            invariant={"se_method": se_invariant},
            note="The bootstrap SE has no reference row (only a calibration "
            "screen); trimmed fits have none either.",
        )
    )

    # ------------------------------------------------------------------ tmle
    def x_tmle(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        src = mi.get("nuisance_source") or {}
        supplied = (
            isinstance(src, dict) and src and all(v == "supplied" for v in src.values())
        )
        se = _lower(mi.get("se_method")) or ""
        return {
            "estimand": _lower(mi.get("estimand")),
            "fluctuation": _lower(mi.get("fluctuation")),
            "nuisance": (
                "supplied"
                if supplied
                else (
                    "default_library"
                    if mi.get("library") == "default"
                    else "user_library"
                )
            ),
            "se_method": "bootstrap" if "bootstrap" in se else "eif",
            "outcome": _lower(mi.get("outcome_type")),
            "weights": _set(mi.get("weights")),
            "cluster": _set(mi.get("cluster")),
        }

    def tmle_cfg(**over: Any) -> Dict[str, Any]:
        base = dict(
            estimand=["ate"],
            fluctuation=["per_arm"],
            nuisance=["user_library"],
            se_method=["eif"],
            outcome=["binary", "continuous"],
            weights=["none"],
            cluster=["none"],
        )
        base.update(over)
        return cfg(**base)

    PARAMS = RP + "test_tmle_parameters_R_parity.py"
    add(
        _Scope(
            "tmle",
            {
                "estimand": ("ate", "att", "atc", "ey1", "ey0", "rr", "or"),
                "fluctuation": ("single", "per_arm"),
                "nuisance": ("supplied", "user_library", "default_library"),
                "se_method": ("eif", "bootstrap"),
                "outcome": ("binary", "continuous"),
                "weights": ("none", "set"),
                "cluster": ("none", "set"),
            },
            x_tmle,
            (
                _Row(
                    "T2",
                    RP + "test_tmle_design_R_parity.py",
                    tmle_cfg(weights=["none", "set"], cluster=["none", "set"]),
                    EST_SE,
                    "ATE and SE vs R tmle 2.1.1 (glm nuisances, obsWeights=, "
                    "id=), gaussian and binomial, four design cases, 1e-6",
                    "sp.tmle(...)",
                ),
                _Row(
                    "T2",
                    PARAMS,
                    tmle_cfg(
                        estimand=["ate", "ey1", "ey0"],
                        nuisance=["supplied"],
                        weights=["none", "set"],
                        cluster=["none", "set"],
                    ),
                    EST_SE,
                    "arm means and their difference vs R tmle from shared "
                    "initial fits, four design cases, 1e-8",
                    "sp.tmle(...)",
                ),
                _Row(
                    "T2",
                    PARAMS,
                    tmle_cfg(
                        estimand=["rr", "or"],
                        nuisance=["supplied"],
                        outcome=["binary"],
                        cluster=["none", "set"],
                    ),
                    EST_SE,
                    "risk ratio and odds ratio vs R tmle, 1e-8",
                    "sp.tmle(...)",
                ),
                _Row(
                    "T2",
                    PARAMS,
                    tmle_cfg(
                        estimand=["rr", "or"],
                        nuisance=["supplied"],
                        outcome=["binary"],
                        weights=["set"],
                        cluster=["none", "set"],
                    ),
                    ("estimate",),
                    "weighted risk ratio and odds ratio vs R tmle",
                    "sp.tmle(...)",
                ),
                _Row(
                    "T4",
                    PARAMS,
                    tmle_cfg(
                        estimand=["or"],
                        nuisance=["supplied"],
                        outcome=["binary"],
                        weights=["set"],
                        cluster=["none", "set"],
                    ),
                    ("se",),
                    "R tmle does not centre the weighted log-OR influence curve "
                    "(variance 0.3-2% larger); ours is the centred one, and R's "
                    "number is rebuilt from it to 1e-9",
                    "sp.tmle(...)",
                ),
                _Row(
                    "T2",
                    "tests/r_parity/72_tmle.py",
                    tmle_cfg(nuisance=["supplied"], outcome=["binary"]),
                    EST_SE,
                    "ATE and SE vs R tmle (Track A module 72)",
                    "sp.tmle(fluctuation=...)",
                ),
                _Row(
                    "T2",
                    RP + "test_tmle_parity.py",
                    tmle_cfg(fluctuation=["single"], outcome=["binary"]),
                    EST_SE,
                    "default single fluctuation vs a frozen base-R glm TMLE, "
                    "estimate and SE 1e-9",
                    "sp.tmle(...)",
                ),
            ),
            note="ATT / ATC differ from R tmle's (another valid TMLE), so they "
            "have no reference row; the bootstrap SE has none either. The "
            "reference rows fit the nuisances with a user-supplied glm library; "
            "a fit with the built-in Super Learner library is its own cell.",
        )
    )

    # ------------------------------------------------------------------ ebalance
    def x_ebal(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        m = mi.get("moments_balanced")
        moments: Optional[str]
        if isinstance(m, (list, tuple)):
            moments = str(m[0]) if len(set(m)) == 1 else "mixed"
        else:
            moments = None if m is None else str(m)
        return {
            "moments": moments,
            "dof_adjust": {True: "true", False: "false"}.get(mi.get("dof_adjust")),
            "vce": _lower(mi.get("vce")),
        }

    add(
        _Scope(
            "ebalance",
            {
                "moments": ("1", "2", "3", "mixed"),
                "dof_adjust": ("true", "false"),
                "vce": ("mestimation", "naive"),
            },
            x_ebal,
            (
                _Row(
                    "T2",
                    RP + "test_ebalance_weightit_parity.py",
                    cfg(
                        moments=["1", "2", "3"],
                        dof_adjust=["false"],
                        vce=["mestimation"],
                    ),
                    EST_SE,
                    "ATT (1e-9) and M-estimation SE (1e-7) vs R WeightIt "
                    "1.7.0, two simulated designs and Lalonde",
                    "sp.ebalance(...)",
                ),
                _Row(
                    "T4",
                    RP + "test_ebalance_weightit_parity.py",
                    cfg(moments=["1", "2", "3"], dof_adjust=["false"], vce=["naive"]),
                    ("se",),
                    "the naive SE ignores the estimated weights; WeightIt "
                    "reports the M-estimation one",
                    "sp.ebalance(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_the_effect_stata_parity.py",
                    cfg(
                        moments=["1", "mixed"], dof_adjust=["true"], vce=["mestimation"]
                    ),
                    ("estimate",),
                    "ATT vs Stata ebalance with targets(1) and mixed "
                    "per-covariate moments, 1e-8",
                    "sp.ebalance(...)",
                ),
            ),
            invariant={"vce": se_invariant},
        )
    )

    # ------------------------------------------------------------------ g-comp
    def x_gcomp(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        by_arm = bool(mi.get("by_arm"))
        se = _lower(mi.get("se_method")) or "bootstrap"
        ml = _lower(mi.get("ml_Q")) or "ols"
        estimand = _lower(mi.get("estimand"))
        return {
            "estimand": "atc" if estimand == "atu" else estimand,
            "by_arm": "true" if by_arm else "false",
            "ps": _set(mi.get("ps_covariates")),
            "se_method": se,
            "ml_q": "ols" if ml in ("ols", "linearregression") else "other",
        }

    SCAN = "tests/test_inference_scan3_weighting.py"
    add(
        _Scope(
            "g_computation",
            {
                "estimand": ("ate", "att", "atc"),
                "by_arm": ("true", "false"),
                "ps": ("none", "set"),
                "se_method": ("bootstrap", "analytic"),
                "ml_q": ("ols", "other"),
            },
            x_gcomp,
            (
                _Row(
                    "T1",
                    SCAN,
                    cfg(
                        estimand=["ate", "att", "atc"],
                        by_arm=["false"],
                        ps=["none"],
                        se_method=["bootstrap"],
                        ml_q=["ols"],
                    ),
                    ("estimate",),
                    "without a treatment interaction every estimand equals the "
                    "OLS coefficient, 1e-10",
                    "sp.g_computation(...)",
                ),
                _Row(
                    "T1",
                    SCAN,
                    cfg(
                        estimand=["ate"],
                        by_arm=["false"],
                        ps=["none"],
                        se_method=["bootstrap"],
                        ml_q=["ols"],
                    ),
                    ("se",),
                    "the bootstrap SE and percentile interval equal the same "
                    "seeded resamples refitted by hand, 1e-9",
                    "sp.g_computation(...)",
                ),
                _Row(
                    "T1",
                    SCAN,
                    cfg(
                        estimand=["ate", "att"],
                        by_arm=["true"],
                        ps=["none", "set"],
                        se_method=["analytic"],
                        ml_q=["ols"],
                    ),
                    EST_SE,
                    "regression adjustment (and IPW-weighted RA) equal an "
                    "independent stacked M-estimator, estimate and SE",
                    "sp.g_computation(...)",
                ),
                _Row(
                    "T1",
                    RP + "test_gformula_parity.py",
                    cfg(
                        estimand=["ate"],
                        by_arm=["false"],
                        ps=["none"],
                        se_method=["bootstrap"],
                        ml_q=["other"],
                    ),
                    ("estimate",),
                    "a saturated regression tree as Q gives the cell-mean "
                    "contrast, 1e-10",
                    "sp.g_computation(...)",
                ),
            ),
            invariant={"se_method": se_invariant},
        )
    )
