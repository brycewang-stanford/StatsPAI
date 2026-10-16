"""Validation-scope maps for the likelihood-based regression family.

``sp.logit`` / ``probit`` / ``poisson`` / ``nbreg`` / ``glm`` / ``tobit`` /
``qreg`` / ``heckman``. Same contract as the maps in
:mod:`statspai.validation_scope` (which imports and registers these): every
row lists the configurations its artifact actually runs, and the outputs it
compares. Kept in its own module so the main file stays readable.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

#: Small-sample conventions of the ML sandwich, spelled as the fits record
#: them (``model_info['robust']``): ``robust`` is Stata's N/(N - 1), ``hc1``
#: N/(N - K), ``hc0`` none.
_ML_VCE = ("classical", "robust", "hc0", "hc1", "cluster")


def register(add: Callable[[Any], None], vs: Any) -> None:
    """Add the maps through ``validation_scope._add`` (``vs`` is that module)."""
    _Row, _Scope, _vals, _mi, _lower, _set = (
        vs._Row,
        vs._Scope,
        vs._vals,
        vs._mi,
        vs._lower,
        vs._set,
    )
    RP = "tests/reference_parity/"
    R = "tests/r_parity/"
    EST_SE = ("estimate", "se")
    VCE_GRAMMAR = RP + "test_vce_grammar_stata_parity.py"
    COUNT_W = RP + "test_count_weights_stata_parity.py"
    JOINT = RP + "test_joint_wald_stata_parity.py"
    POSTEST = RP + "test_postestimation_stata_parity.py"
    LDV = RP + "test_ldv_design_stata_parity.py"
    ML_SANDWICH = RP + "test_ml_sandwich_r_parity.py"
    vce_invariant = (
        ("estimate",),
        "the maximum-likelihood point estimate is computed before, and "
        "independently of, the covariance estimator",
    )

    def ml_vce(mi: Dict[str, Any]) -> Optional[str]:
        if mi.get("cluster"):
            return "cluster"
        robust = _lower(mi.get("robust"))
        if robust in (None, "nonrobust", "oim"):
            return "classical"
        return str(robust)

    def weighted(mi: Dict[str, Any], r: Any) -> str:
        di = getattr(r, "data_info", None) or {}
        flag = (
            mi.get("weights")
            or mi.get("n_effective_weights")
            or mi.get("weighted")
            or di.get("weights") is not None
        )
        return str(_set(flag))

    # ------------------------------------------------------------------ logit
    def x_binary(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        return {
            "vce": ml_vce(mi),
            "weights": weighted(mi, r),
            "perfect_prediction": _lower(mi.get("perfect_prediction")) or "drop",
        }

    # perfect_prediction='keep' keeps completely separated terms (their
    # coefficients run off); no reference covers it.
    binary_domains = {
        "vce": _ML_VCE,
        "weights": ("none", "set"),
        "perfect_prediction": ("drop", "keep"),
    }

    add(
        _Scope(
            "logit",
            binary_domains,
            x_binary,
            (
                _Row(
                    "T2",
                    R + "57_logit.py",
                    {
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                        "perfect_prediction": _vals("drop"),
                    },
                    EST_SE,
                    "coefficients and model-based SEs vs R glm(binomial) and "
                    "Stata logit, 1e-6",
                    "sp.logit(formula=...)",
                ),
                _Row(
                    "T2",
                    VCE_GRAMMAR,
                    {
                        "vce": _vals("classical", "robust", "cluster"),
                        "weights": _vals("none"),
                        "perfect_prediction": _vals("drop"),
                    },
                    EST_SE,
                    "coefficients and SEs vs Stata logit, vce(oim | robust | "
                    "cluster g), 1e-6",
                    "sp.logit(...)",
                ),
                _Row(
                    "T2",
                    COUNT_W,
                    {
                        "vce": _vals("classical", "robust", "cluster"),
                        "weights": _vals("set"),
                        "perfect_prediction": _vals("drop"),
                    },
                    EST_SE,
                    "weighted coefficients and SEs vs Stata logit [iw] / [fw] "
                    "(model-based) and [pw] (robust, cluster), 1e-6",
                    "sp.logit(...)",
                ),
                _Row(
                    "T1",
                    COUNT_W,
                    {
                        "vce": _vals("classical"),
                        "weights": _vals("set"),
                        "perfect_prediction": _vals("drop"),
                    },
                    EST_SE,
                    "frequency weights equal the expanded data, 2e-6",
                    "sp.logit(...)",
                ),
                _Row(
                    "T2",
                    ML_SANDWICH,
                    {
                        "vce": _vals("hc0", "hc1"),
                        "weights": _vals("none", "set"),
                        "perfect_prediction": _vals("drop"),
                    },
                    EST_SE,
                    "HC0 / HC1 sandwich SEs vs R sandwich::vcovHC on glm, "
                    "unweighted and weighted, 1e-8",
                    "sp.logit(...)",
                ),
                _Row(
                    "T2",
                    JOINT,
                    {
                        "vce": _vals("classical", "robust"),
                        "weights": _vals("none"),
                        "perfect_prediction": _vals("drop"),
                    },
                    ("joint_test",),
                    "Wald chi2, df and p-value of joint hypotheses vs Stata "
                    "test after logit, 1e-8",
                    "sp.logit(...)",
                ),
                _Row(
                    "T2",
                    POSTEST,
                    {
                        "vce": _vals("classical", "cluster"),
                        "weights": _vals("none"),
                        "perfect_prediction": _vals("drop"),
                    },
                    ("joint_test",),
                    "Wald test and lincom vs Stata after logit, 1e-6",
                    "sp.logit(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_logit_perfect_prediction_Stata_parity.py",
                    {
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                        "perfect_prediction": _vals("drop"),
                    },
                    EST_SE,
                    "coefficients and SEs after Stata's perfect-prediction "
                    "drop, vs Stata logit",
                    "sp.logit(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_regression_stories_r_parity.py",
                    {
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                        "perfect_prediction": _vals("drop"),
                    },
                    EST_SE,
                    "coefficients and SEs vs R glm(binomial)",
                    "sp.logit(...)",
                ),
            ),
            invariant={"vce": vce_invariant},
            note="vce 'robust' is Stata's N/(N - 1) sandwich, 'hc1' N/(N - K), "
            "'hc0' none; weights with a model-based vce are Stata [fw] / [iw], "
            "with robust or cluster [pw].",
        )
    )

    # ------------------------------------------------------------------ probit
    add(
        _Scope(
            "probit",
            binary_domains,
            x_binary,
            (
                _Row(
                    "T2",
                    R + "48_probit.py",
                    {
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                        "perfect_prediction": _vals("drop"),
                    },
                    EST_SE,
                    "coefficients vs R glm(binomial(probit)) and coefficients "
                    "and observed-information SEs vs Stata probit, 1e-6",
                    "sp.probit(formula=...)",
                ),
                _Row(
                    "T4",
                    R + "48_probit.py",
                    {
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                        "perfect_prediction": _vals("drop"),
                    },
                    ("se",),
                    "R glm reports expected-information SEs, Stata and this "
                    "package observed-information; they differ on a "
                    "non-canonical link (budget 1e-2)",
                    "sp.probit(formula=...)",
                ),
                _Row(
                    "T2",
                    VCE_GRAMMAR,
                    {
                        "vce": _vals("classical", "robust", "cluster"),
                        "weights": _vals("none"),
                        "perfect_prediction": _vals("drop"),
                    },
                    EST_SE,
                    "coefficients and SEs vs Stata probit, vce(oim | robust | "
                    "cluster g), 1e-6",
                    "sp.probit(...)",
                ),
                _Row(
                    "T2",
                    COUNT_W,
                    {
                        "vce": _vals("classical", "robust", "cluster"),
                        "weights": _vals("set"),
                        "perfect_prediction": _vals("drop"),
                    },
                    EST_SE,
                    "weighted coefficients and SEs vs Stata probit [iw] / [fw] "
                    "/ [pw], 1e-6",
                    "sp.probit(...)",
                ),
                _Row(
                    "T4",
                    ML_SANDWICH,
                    {
                        "vce": _vals("hc0", "hc1"),
                        "weights": _vals("none", "set"),
                        "perfect_prediction": _vals("drop"),
                    },
                    ("se",),
                    "R sandwich::vcovHC builds its bread from glm's expected "
                    "information, this package (like Stata) from the observed "
                    "one: 2-3% apart on probit; sp.glm(information='expected') "
                    "reproduces R to 1e-8",
                    "sp.probit(...)",
                ),
                _Row(
                    "T2",
                    POSTEST,
                    {
                        "vce": _vals("classical", "cluster"),
                        "weights": _vals("none"),
                        "perfect_prediction": _vals("drop"),
                    },
                    ("joint_test",),
                    "Wald test and lincom vs Stata after probit, 1e-6",
                    "sp.probit(...)",
                ),
            ),
            invariant={"vce": vce_invariant},
            note="Same conventions as logit.",
        )
    )

    # ------------------------------------------------------------------ poisson
    def x_poisson(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        di = getattr(r, "data_info", None) or {}
        return {
            "vce": ml_vce(mi),
            "weights": weighted(mi, r),
            "exposure": _set(
                di.get("exposure") is not None or di.get("offset") is not None
            ),
        }

    add(
        _Scope(
            "poisson",
            {"vce": _ML_VCE, "weights": ("none", "set"), "exposure": ("none", "set")},
            x_poisson,
            (
                _Row(
                    "T2",
                    R + "58_poisson.py",
                    {
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                        "exposure": _vals("none"),
                    },
                    EST_SE,
                    "coefficients and SEs vs R glm(poisson) and Stata poisson, 1e-6",
                    "sp.poisson(formula=...)",
                ),
                _Row(
                    "T2",
                    VCE_GRAMMAR,
                    {
                        "vce": _vals("classical", "robust", "cluster"),
                        "weights": _vals("none"),
                        "exposure": _vals("none"),
                    },
                    EST_SE,
                    "coefficients and SEs vs Stata poisson, vce(oim | robust | "
                    "cluster g), 1e-6",
                    "sp.poisson(...)",
                ),
                _Row(
                    "T2",
                    COUNT_W,
                    {
                        "vce": _vals("classical", "robust", "cluster"),
                        "weights": _vals("set"),
                        "exposure": _vals("none"),
                    },
                    EST_SE,
                    "weighted coefficients, SEs and log-likelihood vs Stata "
                    "poisson [iw] / [fw] / [pw], 1e-6",
                    "sp.poisson(...)",
                ),
                _Row(
                    "T1",
                    COUNT_W,
                    {
                        "vce": _vals("classical"),
                        "weights": _vals("set"),
                        "exposure": _vals("none"),
                    },
                    EST_SE,
                    "frequency weights equal the expanded data",
                    "sp.poisson(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_accounting_research_parity.py",
                    {
                        "vce": _vals("classical", "hc1"),
                        "weights": _vals("none"),
                        "exposure": _vals("none"),
                    },
                    EST_SE,
                    "coefficients and SEs vs R glm and sandwich::vcovHC(type = "
                    "'HC1'), 1e-6",
                    "sp.poisson(...)",
                ),
                _Row(
                    "T2",
                    ML_SANDWICH,
                    {
                        "vce": _vals("hc0", "hc1"),
                        "weights": _vals("none", "set"),
                        "exposure": _vals("none"),
                    },
                    EST_SE,
                    "HC0 / HC1 sandwich SEs vs R sandwich::vcovHC on glm, "
                    "unweighted and weighted, 1e-8",
                    "sp.poisson(...)",
                ),
                _Row(
                    "T2",
                    ML_SANDWICH,
                    {
                        "vce": _vals("classical", "hc0"),
                        "weights": _vals("none"),
                        "exposure": _vals("set"),
                    },
                    EST_SE,
                    "coefficients and SEs with exposure= vs R glm(offset = "
                    "log(exposure)), model-based and HC0, 1e-8",
                    "sp.poisson(...)",
                ),
                _Row(
                    "T2",
                    JOINT,
                    {
                        "vce": _vals("classical", "robust"),
                        "weights": _vals("none"),
                        "exposure": _vals("none"),
                    },
                    ("joint_test",),
                    "Wald chi2 of joint hypotheses vs Stata test after poisson",
                    "sp.poisson(...)",
                ),
            ),
            invariant={"vce": vce_invariant},
        )
    )

    # ------------------------------------------------------------------ nbreg
    def x_nbreg(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        return {
            "vce": ml_vce(mi),
            "weights": weighted(mi, r),
            "dispersion": _lower(mi.get("dispersion_type")),
        }

    add(
        _Scope(
            "nbreg",
            {
                "vce": ("classical", "robust", "cluster"),
                "weights": ("none", "set"),
                "dispersion": ("nb2", "nb1"),
            },
            x_nbreg,
            (
                _Row(
                    "T2",
                    R + "42_nbreg.py",
                    {
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                        "dispersion": _vals("nb2"),
                    },
                    EST_SE,
                    "coefficients vs R MASS::glm.nb and coefficients and SEs vs "
                    "Stata nbreg",
                    "sp.nbreg(formula=...)",
                ),
                _Row(
                    "T4",
                    R + "42_nbreg.py",
                    {
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                        "dispersion": _vals("nb2"),
                    },
                    ("se",),
                    "glm.nb's SEs are conditional on the dispersion estimate; "
                    "this package's (and Stata's) are joint (1.4e-3 apart)",
                    "sp.nbreg(formula=...)",
                ),
                _Row(
                    "T2",
                    VCE_GRAMMAR,
                    {
                        "vce": _vals("classical", "robust", "cluster"),
                        "weights": _vals("none"),
                        "dispersion": _vals("nb2", "nb1"),
                    },
                    EST_SE,
                    "NB2 and NB1 coefficients and SEs vs Stata nbreg, "
                    "vce(oim | robust | cluster g), 1e-6",
                    "sp.nbreg(...)",
                ),
                _Row(
                    "T1",
                    COUNT_W,
                    {
                        "vce": _vals("classical"),
                        "weights": _vals("set"),
                        "dispersion": _vals("nb2"),
                    },
                    EST_SE,
                    "frequency weights equal the expanded data",
                    "sp.nbreg(...)",
                ),
            ),
            invariant={"vce": vce_invariant},
        )
    )

    # ------------------------------------------------------------------ glm
    canonical = {"gaussian": "identity", "binomial": "logit", "poisson": "log"}

    def x_glm(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        fam = (_lower(mi.get("family")) or "").replace(" ", "_")
        link = _lower(mi.get("link")) or ""
        pair = f"{fam}_{link}"
        known = {
            "gaussian_identity",
            "gaussian_log",
            "binomial_logit",
            "binomial_probit",
            "binomial_cloglog",
            "binomial_cauchit",
            "poisson_log",
            "gamma_log",
            "inverse_gaussian_log",
        }
        info = _lower(mi.get("information"))
        if info is not None and canonical.get(fam) == link:
            # The two informations coincide on a canonical link.
            info = "observed"
        return {
            "family_link": pair if pair in known else "other",
            "vce": ml_vce(mi),
            "weights": weighted(mi, r),
            "information": info,
        }

    add(
        _Scope(
            "glm",
            {
                "family_link": (
                    "gaussian_identity",
                    "gaussian_log",
                    "binomial_logit",
                    "binomial_probit",
                    "binomial_cloglog",
                    "binomial_cauchit",
                    "poisson_log",
                    "gamma_log",
                    "inverse_gaussian_log",
                    "other",
                ),
                "vce": (
                    "classical",
                    "robust",
                    "hc0",
                    "hc1",
                    "hc2",
                    "hc3",
                    "hac",
                    "cluster",
                ),
                "weights": ("none", "set"),
                "information": ("observed", "expected"),
            },
            x_glm,
            (
                _Row(
                    "T2",
                    VCE_GRAMMAR,
                    {
                        "family_link": _vals(
                            "gaussian_identity",
                            "gaussian_log",
                            "binomial_logit",
                            "binomial_probit",
                            "binomial_cloglog",
                            "poisson_log",
                        ),
                        "vce": _vals("classical", "robust", "cluster"),
                        "weights": _vals("none"),
                        "information": _vals("observed"),
                    },
                    EST_SE,
                    "coefficients and SEs vs Stata glm, vce(oim | robust | "
                    "cluster g), six family/link pairs, 1e-6",
                    "sp.glm(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_glm_parity.py",
                    {
                        "family_link": _vals("binomial_logit", "poisson_log"),
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                        "information": _vals("observed"),
                    },
                    EST_SE,
                    "coefficients (1e-9) and SEs (1e-3) vs R glm",
                    "sp.glm(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_glm_dispersion_stata_r_parity.py",
                    {
                        "family_link": _vals("gamma_log", "inverse_gaussian_log"),
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                        "information": _vals("observed", "expected"),
                    },
                    EST_SE,
                    "coefficients and SEs vs Stata glm (observed information) "
                    "and R summary.glm (expected information), 1e-6",
                    "sp.glm(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_linear_model_extensions_parity.py",
                    {
                        "family_link": _vals("binomial_cauchit"),
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                        "information": _vals("expected"),
                    },
                    EST_SE,
                    "cauchit coefficients and SEs vs R glm, 1e-6",
                    "sp.glm(...)",
                ),
                _Row(
                    "T2",
                    ML_SANDWICH,
                    {
                        "family_link": _vals("binomial_logit", "poisson_log"),
                        "vce": _vals("hc0", "hc1", "hc2", "hc3"),
                        "weights": _vals("none"),
                        "information": _vals("observed"),
                    },
                    EST_SE,
                    "HC0-HC3 sandwich SEs vs R sandwich::vcovHC on glm, 1e-8",
                    "sp.glm(...)",
                ),
                _Row(
                    "T2",
                    ML_SANDWICH,
                    {
                        "family_link": _vals("binomial_probit"),
                        "vce": _vals("hc0"),
                        "weights": _vals("none"),
                        "information": _vals("expected"),
                    },
                    ("se",),
                    "probit HC0 with expected-information bread vs R "
                    "sandwich::vcovHC, 1e-8",
                    "sp.glm(...)",
                ),
            ),
            invariant={"vce": vce_invariant, "information": vce_invariant},
            note="On a canonical link the observed and expected informations "
            "coincide, so such fits read as information 'observed'.",
        )
    )

    # ------------------------------------------------------------------ tobit
    def x_tobit(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        method = "scls" if "scls" in (_lower(mi.get("method")) or "") else "mle"
        lo, hi = mi.get("lower_limit"), mi.get("upper_limit")
        lo_set = lo is not None and lo != float("-inf")
        hi_set = hi is not None and hi != float("inf")
        censoring = (
            "two_sided"
            if lo_set and hi_set
            else "left" if lo_set else "right" if hi_set else None
        )
        vce = _lower(mi.get("vce"))
        if mi.get("cluster"):
            vce = "cluster"
        elif vce in (None, "nonrobust", "oim"):
            vce = "classical"
        return {
            "method": method,
            "vce": vce,
            "weights": _set(mi.get("weights")),
            "censoring": censoring,
        }

    tobit_cfg = lambda m, v, w, c: {  # noqa: E731
        "method": _vals(*m),
        "vce": _vals(*v),
        "weights": _vals(*w),
        "censoring": _vals(*c),
    }
    add(
        _Scope(
            "tobit",
            {
                "method": ("mle", "scls"),
                "vce": ("classical", "robust", "cluster"),
                "weights": ("none", "set"),
                "censoring": ("left", "right", "two_sided"),
            },
            x_tobit,
            (
                _Row(
                    "T2",
                    R + "41_tobit.py",
                    tobit_cfg(["mle"], ["classical"], ["none"], ["left"]),
                    EST_SE,
                    "coefficients and SEs vs R censReg and Stata tobit, ll(0), 1e-6",
                    "sp.tobit(data=...)",
                ),
                _Row(
                    "T2",
                    RP + "test_count_quantile_parity.py",
                    tobit_cfg(["mle"], ["classical"], ["none"], ["left"]),
                    EST_SE,
                    "coefficients (1e-5) and SEs (5e-5) vs AER::tobit",
                    "sp.tobit(...)",
                ),
                _Row(
                    "T2",
                    LDV,
                    tobit_cfg(
                        ["mle"], ["classical", "robust", "cluster"], ["none"], ["left"]
                    ),
                    EST_SE,
                    "coefficients, SEs and sigma vs Stata tobit, ll(0), "
                    "vce(oim | robust | cluster g), 1e-6",
                    "sp.tobit(...)",
                ),
                _Row(
                    "T2",
                    LDV,
                    tobit_cfg(["mle"], ["robust", "cluster"], ["set"], ["left"]),
                    EST_SE,
                    "weighted coefficients and SEs vs Stata tobit [pw], 1e-6",
                    "sp.tobit(...)",
                ),
                _Row(
                    "T2",
                    LDV,
                    tobit_cfg(["mle"], ["classical"], ["none"], ["two_sided"]),
                    EST_SE,
                    "two-sided [0, 1.5] coefficients and SEs vs Stata tobit, 1e-6",
                    "sp.tobit(...)",
                ),
                _Row(
                    "T2",
                    LDV,
                    tobit_cfg(["mle"], ["cluster"], ["set"], ["two_sided"]),
                    EST_SE,
                    "two-sided, weighted and clustered, vs Stata tobit, 1e-6",
                    "sp.tobit(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_ldv_gaps_stata_parity.py",
                    tobit_cfg(
                        ["mle"], ["robust", "cluster"], ["none", "set"], ["two_sided"]
                    ),
                    EST_SE,
                    "two-sided [0, 1.5] vs Stata tobit, vce(robust | cluster g) "
                    "and [pw=w], 1e-6",
                    "sp.tobit(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_ldv_gaps_stata_parity.py",
                    tobit_cfg(
                        ["mle"], ["classical", "robust", "cluster"], ["none"], ["right"]
                    ),
                    EST_SE,
                    "right-censored ul(1.5) vs Stata tobit, vce(oim | robust | "
                    "cluster g), 1e-6",
                    "sp.tobit(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_binary_r2_and_scls_parity.py",
                    tobit_cfg(["scls"], ["classical"], ["none"], ["left"]),
                    ("estimate",),
                    "SCLS coefficients vs micsr::tobit1(method = 'trimmed'), 1e-8",
                    "sp.tobit(...)",
                ),
            ),
            invariant={"vce": vce_invariant},
            note="Weights force the robust (pw) variance, as Stata does.",
        )
    )

    # ------------------------------------------------------------------ qreg
    def x_qreg(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        vce = _lower(mi.get("vce")) or "iid"
        if mi.get("cluster"):
            vce = "cluster"
        return {"vce": vce, "weights": _set(mi.get("weighted"))}

    add(
        _Scope(
            "qreg",
            {
                "vce": ("iid", "robust", "nid", "kernel", "ker", "cluster", "powell"),
                "weights": ("none", "set"),
            },
            x_qreg,
            (
                _Row(
                    "T2",
                    R + "40_qreg.py",
                    {"vce": _vals("nid"), "weights": _vals("none")},
                    EST_SE,
                    "median coefficients and SEs vs R quantreg::rq(se = 'nid') "
                    "and Stata qreg, vce(robust), 1e-6",
                    "sp.qreg(...)",
                ),
                _Row(
                    "T2",
                    LDV,
                    {"vce": _vals("iid", "robust"), "weights": _vals("none")},
                    EST_SE,
                    "coefficients, SEs and p-values at tau 0.25 / 0.5 / 0.75 vs "
                    "Stata qreg, 1e-6",
                    "sp.qreg(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_qreg_cluster_stata_parity.py",
                    {"vce": _vals("kernel", "cluster"), "weights": _vals("none")},
                    ("estimate", "se", "vcov"),
                    "full covariance vs Stata qreg2 (kernel and cluster), four "
                    "quantiles, 1e-10",
                    "sp.qreg(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_ldv_gaps_stata_parity.py",
                    {"vce": _vals("robust"), "weights": _vals("set")},
                    EST_SE,
                    "qreg [pw=w] at tau 0.25 / 0.5 / 0.75 vs Stata, 1e-10",
                    "sp.qreg(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_linear_model_extensions_parity.py",
                    {"vce": _vals("ker", "nid"), "weights": _vals("none", "set")},
                    EST_SE,
                    "coefficients and SEs vs quantreg (se = 'ker' / 'nid'), "
                    "unweighted and weighted, 1e-8",
                    "sp.qreg(...)",
                ),
            ),
            invariant={"vce": vce_invariant},
        )
    )

    # ------------------------------------------------------------------ heckman
    def x_heckman(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        two_step = "two" in (_lower(mi.get("method")) or "")
        vce = "classical" if two_step else _lower(mi.get("vce"))
        if not two_step and mi.get("cluster"):
            vce = "cluster"
        if vce in (None, "nonrobust", "oim"):
            vce = "classical"
        return {
            "method": "twostep" if two_step else "ml",
            "vce": vce,
            "weights": _set(mi.get("weights")),
        }

    add(
        _Scope(
            "heckman",
            {
                "method": ("twostep", "ml"),
                "vce": ("classical", "robust", "cluster"),
                "weights": ("none", "set"),
            },
            x_heckman,
            (
                _Row(
                    "T2",
                    R + "43_heckman.py",
                    {
                        "method": _vals("twostep"),
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                    },
                    EST_SE,
                    "two-step coefficients and SEs vs sampleSelection::heckit "
                    "and Stata heckman, twostep, 1e-6",
                    "sp.heckman(...)",
                ),
                _Row(
                    "T1",
                    RP + "test_heckman_se_parity.py",
                    {
                        "method": _vals("twostep"),
                        "vce": _vals("classical"),
                        "weights": _vals("none"),
                    },
                    ("se",),
                    "two-step SEs equal a hand-computed Heckman (1979) variance, 1e-8",
                    "sp.heckman(...)",
                ),
                _Row(
                    "T2",
                    LDV,
                    {
                        "method": _vals("ml"),
                        "vce": _vals("classical", "robust", "cluster"),
                        "weights": _vals("none"),
                    },
                    EST_SE,
                    "ML coefficients and SEs vs Stata heckman, vce(oim | robust "
                    "| cluster g), 1e-6",
                    "sp.heckman(...)",
                ),
                _Row(
                    "T2",
                    LDV,
                    {
                        "method": _vals("ml"),
                        "vce": _vals("robust", "cluster"),
                        "weights": _vals("set"),
                    },
                    EST_SE,
                    "weighted ML coefficients and SEs vs Stata heckman [pw], 1e-6",
                    "sp.heckman(...)",
                ),
            ),
            invariant={"vce": vce_invariant},
            note="The two-step variance is the conventional Heckman (1979) one "
            "and takes no vce.",
        )
    )
