"""Validation-scope maps for the high-dimensional fixed-effects estimators.

``sp.feols`` (the pyfixest-backed entry point, not ``sp.fast.feols``) and
``sp.ppmlhdfe``. Same contract as :mod:`statspai.validation_scope`.

``sp.feols``'s own ``model_info['vcov_type']`` is pyfixest's coarse label
(``"CRV"`` for one-way, two-way and CRV3 alike), so the extractor reads the
fitted pyfixest object, and whether ``ssc=`` was passed from the recorded
call.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def _record_kwargs(r: Any) -> Dict[str, Any]:
    from .core._call_record import get_call_record

    rec = get_call_record(r)
    return dict(rec.kwargs) if rec is not None else {}


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
    vce_invariant = (
        ("estimate",),
        "the point estimate is computed before, and independently of, the "
        "covariance estimator",
    )

    def cfg(**dims: Any) -> Dict[str, Any]:
        return {k: _vals(*v) for k, v in dims.items()}

    # ------------------------------------------------------------------ feols
    def x_feols(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        pf = getattr(r, "_pyfixest_fit", None)
        fe_text = str(mi.get("fixed_effects") or "None")
        fe = (
            "none"
            if fe_text in ("None", "", "0")
            else ("multi" if "+" in fe_text else "one")
        )
        formula = str(mi.get("formula") or "")
        iv = "true" if formula.count("~") > 1 else "false"
        vtype = _lower(mi.get("vcov_type")) or ""
        detail = str(getattr(pf, "_vcov_type_detail", "") or "")
        clusters = getattr(pf, "_clustervar", None) or []
        if any(k in vtype for k in ("wild", "conley", "cr2", "jackknife")):
            vcov = "other"
        elif detail == "iid":
            vcov = "iid"
        elif detail in ("hetero", "HC1"):
            vcov = "hetero"
        elif detail == "CRV1":
            vcov = "cluster_multiway" if len(clusters) > 1 else "cr1"
        elif detail == "CRV3":
            vcov = "cr3"
        else:
            vcov = "other" if detail else None
        weighted = getattr(pf, "_weights_name", None) or mi.get("n_effective_weights")
        return {
            "fe": fe,
            "iv": iv,
            "vcov": vcov,
            "weights": _set(weighted),
            "ssc": "custom" if _record_kwargs(r).get("ssc") is not None else "default",
        }

    add(
        _Scope(
            "feols",
            {
                "fe": ("none", "one", "multi"),
                "iv": ("false", "true"),
                "vcov": ("iid", "hetero", "cr1", "cr3", "cluster_multiway", "other"),
                "weights": ("none", "set"),
                "ssc": ("default", "custom"),
            },
            x_feols,
            (
                _Row(
                    "T2",
                    RP + "test_feols_grid_fixest_parity.py",
                    cfg(
                        fe=["none", "one", "multi"],
                        iv=["false"],
                        vcov=["iid", "hetero", "cr1", "cluster_multiway"],
                        weights=["none", "set"],
                        ssc=["default"],
                    ),
                    EST_SE,
                    "24-cell grid (absorbed effects x vcov x weights) vs R "
                    "fixest defaults, 1e-6 (observed <= 1.2e-8)",
                    "sp.feols(...)",
                ),
                _Row(
                    "T2",
                    "tests/external_parity/test_gow_ding_accounting.py",
                    cfg(
                        fe=["multi"],
                        iv=["false"],
                        vcov=["iid", "cluster_multiway"],
                        weights=["none"],
                        ssc=["default"],
                    ),
                    EST_SE,
                    "two-way absorbed DiD coefficient and SEs (iid and "
                    "two-way clustered) vs R fixest, 1e-8",
                    "sp.feols(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_sasp_within_parity.py",
                    cfg(
                        fe=["one"],
                        iv=["false"],
                        vcov=["cr1"],
                        weights=["none"],
                        ssc=["default"],
                    ),
                    EST_SE,
                    "within estimator and clustered SE vs Stata xtreg, fe "
                    "robust, abs 1e-6",
                    "sp.feols(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_castle_stata_parity.py",
                    cfg(
                        fe=["multi"],
                        iv=["false"],
                        vcov=["cr1"],
                        weights=["none", "set"],
                        ssc=["default"],
                    ),
                    EST_SE,
                    "TWFE coefficient and clustered SE vs Stata xtreg, fe "
                    "vce(cluster), unweighted and [aweight], abs 1e-6",
                    "sp.feols(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_feols_multiref_R_parity.py",
                    cfg(
                        fe=["multi"],
                        iv=["false"],
                        vcov=["cr1"],
                        weights=["none"],
                        ssc=["default"],
                    ),
                    EST_SE,
                    "event-study coefficients and clustered SEs vs R fixest, 1e-11",
                    "sp.feols(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_accounting_research_parity.py",
                    cfg(
                        fe=["multi"],
                        iv=["false"],
                        vcov=["cluster_multiway"],
                        weights=["none"],
                        ssc=["default"],
                    ),
                    EST_SE,
                    "two-way clustered SEs vs R fixest vcov = ~firm + year, 1e-9",
                    "sp.feols(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_ssc_presets_Stata_parity.py",
                    cfg(
                        fe=["none"],
                        iv=["false", "true"],
                        vcov=["cr1", "hetero"],
                        weights=["none"],
                        ssc=["custom"],
                    ),
                    ("se",),
                    "sp.ssc(preset) SEs vs Stata regress / ivregress (robust "
                    "and cluster), 1e-10",
                    "sp.feols(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_ssc_presets_Stata_parity.py",
                    cfg(
                        fe=["one"],
                        iv=["false"],
                        vcov=["cr1"],
                        weights=["none"],
                        ssc=["custom"],
                    ),
                    ("se",),
                    "sp.ssc(preset) clustered SEs vs Stata areg, reghdfe and "
                    "xtreg, fe, 1e-10",
                    "sp.feols(...)",
                ),
            ),
            invariant={"vcov": vce_invariant, "ssc": vce_invariant},
            note="Backed by pyfixest (third-party provenance, Track A table "
            "marks it with a dagger). Wild-bootstrap, CR2 and Conley fits read "
            "as vcov 'other' and have no reference row here.",
        )
    )

    # ------------------------------------------------------------------ ppmlhdfe
    def x_ppml(r: Any) -> Dict[str, Optional[str]]:
        mi = _mi(r)
        fes = mi.get("absorbed_fe") or []
        cluster = mi.get("cluster")
        if isinstance(cluster, (list, tuple)) and len(cluster) > 1:
            vce = "cluster_multiway"
        elif cluster:
            vce = "cr1"
        else:
            vce = _lower(mi.get("robust"))
            vce = "classical" if vce in (None, "nonrobust") else vce
        return {
            "fe": "one" if len(fes) == 1 else ("multi" if fes else None),
            "vce": vce,
            "weights": _set(mi.get("n_effective_weights")),
            "ssc": _lower(_record_kwargs(r).get("ssc")) or "stata",
        }

    add(
        _Scope(
            "ppmlhdfe",
            {
                "fe": ("one", "multi"),
                "vce": ("classical", "robust", "hc0", "hc1", "cr1", "cluster_multiway"),
                "weights": ("none", "set"),
                "ssc": ("stata", "fixest", "none"),
            },
            x_ppml,
            (
                _Row(
                    "T2",
                    "tests/r_parity/37_ppmlhdfe.py",
                    cfg(
                        fe=["one"],
                        vce=["robust"],
                        weights=["none"],
                        ssc=["stata", "fixest"],
                    ),
                    EST_SE,
                    "coefficients and robust SEs vs Stata ppmlhdfe, vce(robust) "
                    "and (ssc='fixest') R fixest::fepois(vcov = 'hetero')",
                    "sp.ppmlhdfe(formula=...)",
                ),
                _Row(
                    "T2",
                    "tests/r_parity/47_ppmlhdfe_3fe.py",
                    cfg(
                        fe=["multi"],
                        vce=["robust"],
                        weights=["none"],
                        ssc=["stata", "fixest"],
                    ),
                    EST_SE,
                    "three absorbed effects, same comparisons as module 37",
                    "sp.ppmlhdfe(formula=...)",
                ),
                _Row(
                    "T2",
                    RP + "test_count_weights_stata_parity.py",
                    cfg(fe=["one"], vce=["robust"], weights=["set"], ssc=["stata"]),
                    EST_SE,
                    "[pw=w] coefficients, SEs and log-likelihood vs Stata ppmlhdfe",
                    "sp.ppmlhdfe(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_count_weights_stata_parity.py",
                    cfg(
                        fe=["one"], vce=["cr1"], weights=["none", "set"], ssc=["stata"]
                    ),
                    EST_SE,
                    "clustered, with and without [pw=w], vs Stata ppmlhdfe",
                    "sp.ppmlhdfe(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_ppmlhdfe_singletons_Stata_parity.py",
                    cfg(fe=["multi"], vce=["cr1"], weights=["none"], ssc=["stata"]),
                    EST_SE,
                    "singleton and separation drops, interacted absorbed "
                    "effects, clustered SEs vs Stata ppmlhdfe, 1e-7",
                    "sp.ppmlhdfe(...)",
                ),
                _Row(
                    "T2",
                    RP + "test_ppmlhdfe_twoway_parity.py",
                    cfg(
                        fe=["multi"],
                        vce=["cr1", "cluster_multiway"],
                        weights=["none"],
                        ssc=["stata"],
                    ),
                    ("se",),
                    "one- and two-way clustered SEs vs Stata ppmlhdfe "
                    "cluster(ca cb), abs 5e-7",
                    "sp.ppmlhdfe(...)",
                ),
            ),
            invariant={"vce": vce_invariant, "ssc": vce_invariant},
        )
    )
