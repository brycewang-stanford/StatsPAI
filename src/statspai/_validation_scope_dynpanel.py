"""Validation-scope maps for dynamic-panel GMM: ``sp.xtabond`` / ``sp.xtdpdsys``.

Same contract as :mod:`statspai.validation_scope` (which imports and
registers these): every row lists the configurations its artifact actually
runs, and the outputs it compares.

``sp.xtdpdsys`` is a pass-through to ``sp.xtabond(method='system', ...)``
that only changes two defaults (``h=2``, ``iv_equation='diff'``), so the two
maps share one set of rows; the ``model`` dimension is read with each
function's own defaults. A row names the call its artifact makes, and a
system-GMM row run through one name is evidence for the same configuration
reached through the other.

The grid is enumerated cell by cell (``scripts/build_evidence_inventory.py``),
so options that only exist together are fused into one dimension instead of
being crossed: ``model`` carries the estimator, the one-step weight ``h``,
where the standard instruments enter (``iv_equation``) and the constant;
``regressors`` carries the regressor classes and the period dummies. Values
no reference ran are grouped (``h1``, ``other``) without loss: a group no row
lists is not covered whichever member was run.

Several options are not stored in ``model_info`` (the instrument classes and
their windows, ``h``, ``iv_equation``, the number of lags of y). They are
read from the recorded call (``core._call_record``); a result without one
reports those dimensions as unchecked.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

#: Instrument-class defaults of ``sp.xtabond`` (absolute lags, the xtabond2
#: convention): predetermined 1+, endogenous 2+.
_CLASS_DEFAULT_MIN = {"predetermined": 1, "endogenous": 2}

#: ``model``: estimator x one-step weight x standard-instrument placement x
#: constant. ``no_iv`` = no standard instruments (no x, no period dummies).
_SYSTEM_MODELS: Tuple[str, ...] = (
    "system_h3_iv_both",
    "system_h3_iv_diff",
    "system_h3_no_iv",
    "system_h2_iv_both",
    "system_h2_iv_diff",
    "system_h2_no_iv",
    "system_iv_level",
    "system_no_constant",
)
_MODELS: Tuple[str, ...] = ("difference", "ah") + _SYSTEM_MODELS + ("h1",)

#: ``regressors``: what enters besides the lags of y. ``*_window`` = the
#: class at a user-set lag window; ``other`` = any other combination (a class
#: at its default window, both classes, x together with period dummies, ...).
_REGRESSORS: Tuple[str, ...] = (
    "none",
    "x",
    "x_pre_window",
    "x_endo_window",
    "dummies",
    "x_endo_window_dummies",
    "other",
)


def _call_arguments(r: Any) -> Optional[Dict[str, Any]]:
    """The arguments of the recorded public call, positional ones by name."""
    from .core._call_record import get_call_record

    rec = get_call_record(r)
    if rec is None:
        return None
    out: Dict[str, Any] = {}
    try:
        import statspai

        fn = getattr(statspai, rec.function)
        params = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        out.update({p.name: v for p, v in zip(params, rec.args)})
    except (AttributeError, TypeError, ValueError):
        if rec.args:
            return None
    out.update(rec.kwargs)
    return out


def _class_window(kw: Dict[str, Any], cls: str) -> str:
    """``none`` / ``default_window`` / ``custom_window`` for one class."""
    if not kw.get(cls):
        return "none"
    window = kw.get(f"{cls}_lags")
    if window is None:
        return "default_window"
    lo, hi = (tuple(window) + (None, None))[:2]
    if (lo is None or lo == _CLASS_DEFAULT_MIN[cls]) and hi is None:
        return "default_window"
    return "custom_window"


def _regressors(kw: Dict[str, Any], dummies: bool) -> str:
    exog = bool(kw.get("x"))
    pre = _class_window(kw, "predetermined")
    endo = _class_window(kw, "endogenous")
    if pre == "none" and endo == "none":
        return {
            (False, False): "none",
            (True, False): "x",
            (False, True): "dummies",
        }.get((exog, dummies), "other")
    if exog and pre == "custom_window" and endo == "none" and not dummies:
        return "x_pre_window"
    if exog and endo == "custom_window" and pre == "none":
        return "x_endo_window_dummies" if dummies else "x_endo_window"
    return "other"


def register(add: Callable[[Any], None], vs: Any) -> None:
    """Add the maps through ``validation_scope._add`` (``vs`` is that module)."""
    _Row, _Scope, _vals, _mi, _lower = (
        vs._Row,
        vs._Scope,
        vs._vals,
        vs._mi,
        vs._lower,
    )
    RP = "tests/reference_parity/"
    ABD = RP + "test_dynpanel_abdata_parity.py"
    HANSEN = RP + "test_hansen_methods_stata_parity.py"
    DGP = RP + "test_gmm_dynamic_panel_parity.py"
    TRACK_A = "tests/r_parity/50_xtabond.py"
    EST_SE = ("estimate", "se")
    ALL3 = ("estimate", "se", "diagnostic")
    XTABOND = "sp.xtabond(...)"
    SYSTEM = "sp.xtabond(method='system')"
    XTDPDSYS = "sp.xtdpdsys(...)"

    domains: Dict[str, Tuple[str, ...]] = {
        "model": _MODELS,
        "steps": ("onestep", "twostep", "beyond_two"),
        "robust": ("true", "false"),
        "cluster": ("unit", "coarser"),
        "transform": ("fd", "fod"),
        "collapse": ("false", "true"),
        "gmm_window": ("default", "max_capped", "min_raised", "min_and_max"),
        "regressors": _REGRESSORS,
        "lags": ("1", "2", "3plus"),
        "panel": ("gap_free", "gapped"),
    }
    robust_invariant = {
        "robust": (
            ("estimate",),
            "the coefficients are solved before, and independently of, the "
            "choice between the classical and the robust / Windmeijer "
            "variance (robust= only switches the reported VCE)",
        )
    }

    def extractor(default_h: int, default_iv: str) -> Callable[[Any], Dict]:
        def extract(r: Any) -> Dict[str, Optional[str]]:
            mi = _mi(r)
            kw = _call_arguments(r)
            label = _lower(mi.get("method")) or ""
            method: Optional[str]
            if "system" in label:
                method = "system"
            elif "anderson" in label:
                method = "ah"
            elif "difference" in label:
                method = "difference"
            else:
                method = None

            req = mi.get("steps_requested")
            steps: Optional[str]
            if isinstance(req, str):
                # 'iterated' / 'cue'
                steps = "beyond_two" if req in ("iterated", "cue") else None
            elif isinstance(req, int) and not isinstance(req, bool):
                steps = {1: "onestep", 2: "twostep"}.get(req, "beyond_two")
            else:
                steps = None

            # A cluster variable constant within unit with as many groups as
            # units is a relabelling of the unit: the default grouping.
            n_cl, n_units = mi.get("n_clusters"), mi.get("n_units")
            cluster = (
                "unit"
                if mi.get("cluster") is None or (n_cl is not None and n_cl == n_units)
                else "coarser"
            )

            gmm_window: Optional[str] = None
            gl = mi.get("gmm_lags")
            if isinstance(gl, (tuple, list)) and len(gl) == 2:
                lo, hi = gl
                if method == "ah":
                    hi = None  # Anderson-Hsiao uses the minimum lag only
                if lo == 2:
                    gmm_window = "default" if hi is None else "max_capped"
                else:
                    gmm_window = "min_raised" if hi is None else "min_and_max"

            dummies = bool(mi.get("time_dummies"))
            out: Dict[str, Optional[str]] = {
                "model": None,
                "steps": steps,
                "robust": {True: "true", False: "false"}.get(mi.get("robust")),
                "cluster": cluster,
                "transform": {
                    "first differences": "fd",
                    "orthogonal deviations": "fod",
                }.get(str(mi.get("transform"))),
                "collapse": "true" if mi.get("collapse") else "false",
                "gmm_window": gmm_window,
                "regressors": None,
                "lags": None,
                "panel": "gapped" if mi.get("gap_warning") else "gap_free",
            }
            if kw is None or method is None:
                return out
            out["regressors"] = _regressors(kw, dummies)
            # model_info does not store the lags of y; the call does
            # (default 1 in both signatures).
            lags = kw.get("lags", 1)
            if isinstance(lags, int) and not isinstance(lags, bool) and lags >= 1:
                out["lags"] = str(lags) if lags < 3 else "3plus"
            h = kw.get("h", default_h)
            if h == 1:
                out["model"] = "h1"
            elif method != "system":
                # Without a level equation h(2) and h(3) are the same matrix:
                # the transform's own error covariance.
                out["model"] = method if h in (2, 3) else None
            elif not mi.get("constant"):
                out["model"] = "system_no_constant"
            else:
                iv = (
                    "none"
                    if not (kw.get("x") or dummies)
                    else (kw.get("iv_equation") or default_iv)
                )
                if iv == "level":
                    out["model"] = "system_iv_level"
                elif h in (2, 3) and iv in ("both", "diff", "none"):
                    tail = {"both": "iv_both", "diff": "iv_diff", "none": "no_iv"}
                    out["model"] = f"system_h{h}_{tail[iv]}"
            return out

        return extract

    def cfg(**over: Any) -> Dict[str, Any]:
        base: Dict[str, List[str]] = dict(
            model=["difference"],
            steps=["onestep"],
            robust=["true"],
            cluster=["unit"],
            transform=["fd"],
            collapse=["false"],
            gmm_window=["default"],
            regressors=["none"],
            lags=["1"],
            panel=["gap_free"],
        )
        base.update(over)
        return {k: _vals(*v) for k, v in base.items()}

    def sys_cfg(**over: Any) -> Dict[str, Any]:
        """System GMM with xtabond2's defaults and ``x=['w', 'k']``."""
        base: Dict[str, Any] = dict(model=["system_h3_iv_both"], regressors=["x"])
        base.update(over)
        return cfg(**base)

    rows = (
        # ------------------------------------------------- difference GMM
        _Row(
            "T2",
            ABD,
            cfg(steps=["onestep", "twostep"], robust=["true", "false"]),
            ALL3,
            "abdata, one lag of y: one-step robust / classical and two-step "
            "Windmeijer / conventional: every coefficient and SE, e(N) and "
            "the instrument count vs Stata 18 xtabond (noconstant), rtol "
            "1e-9; AR(1)/AR(2) z vs e(arm1)/e(arm2) 1e-9; one-step Sargan and "
            "post-two-step Hansen J vs e(sargan) 1e-9; one-step Hansen J vs "
            "xtabond2 1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(lags=["2"]),
            ALL3,
            "abdata, two lags of y, one-step robust, vs Stata 18 xtabond: "
            "coefficients, SEs, e(N), instrument count and AR z, 1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(gmm_window=["max_capped"]),
            EST_SE,
            "gmm_lags=(2, 4) vs Stata xtabond, maxldep(3), 1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(gmm_window=["min_raised"]),
            EST_SE,
            "gmm_lags=(3, None) vs xtabond2 gmm(L.n, lag(2 .)), 1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(collapse=["true"]),
            ALL3,
            "collapsed instruments vs xtabond2, collapse: coefficients and "
            "SEs 1e-9, 7 instruments, Hansen J 1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(steps=["onestep", "twostep"], regressors=["x"], lags=["2"]),
            ALL3,
            "Arellano-Bond (1991) table 4 (two lags of y, x = l(0/1).w "
            "l(0/2).k, also with year dummies passed in x), one-step robust "
            "and two-step Windmeijer, vs Stata xtabond: coefficients, SEs "
            "and AR z, 1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(regressors=["x"], lags=["2"]),
            ("estimate",),
            "Arellano-Bond (1991) table 4 (two lags of y), one-step, vs R "
            "plm::pgmm, 1e-8",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(),
            EST_SE,
            "AR(1) one-step robust vs R pdynmc 0.9.12: estimate 1e-6, SE 1e-5",
            XTABOND,
        ),
        _Row(
            "T2",
            TRACK_A,
            cfg(regressors=["x"]),
            EST_SE,
            "Track A module 50 (simulated balanced panel, one lag of y, one "
            "exogenous x): coefficients and robust SEs vs R plm::pgmm and "
            "Stata xtabond, 1e-6 (observed 1e-15)",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(regressors=["x_pre_window", "x_endo_window"], lags=["2"]),
            EST_SE,
            "two lags of y, x = l(0/2).k, w predetermined with "
            "predetermined_lags=(2, None) / endogenous with "
            "endogenous_lags=(3, None), vs Stata xtabond pre() / endogenous() "
            "with lagstruct(1, .), 1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(regressors=["x"]),
            ("diagnostic",),
            "x = w k, one-step robust: Hansen J vs xtabond2 1e-9 and AR z "
            "(cluster on the panel id) vs xtabond2 1e-8",
            XTABOND,
        ),
        _Row(
            "T4",
            ABD,
            cfg(regressors=["none", "x"]),
            ("diagnostic",),
            "Sargan statistic: xtabond's scale; xtabond2's is exactly "
            "N*/(N* - k) larger, asserted to 1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(regressors=["x"], transform=["fod"], steps=["onestep", "twostep"]),
            ALL3,
            "forward orthogonal deviations, one-step robust and two-step "
            "Windmeijer, vs xtabond2, orthogonal: coefficients, SEs, e(N), "
            "instrument count and AR z, 1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(regressors=["x"], cluster=["coarser"]),
            ALL3,
            "cluster on industry (9 clusters), one-step robust, vs xtabond2, "
            "cluster(): coefficients and SEs 1e-9, AR z 1e-8",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(model=["ah"], regressors=["none", "x"]),
            EST_SE,
            "Anderson-Hsiao with the level and with the differenced "
            "instrument, with and without x = w k, vs xtabond2 with one "
            "pooled instrument: coefficients, SEs, e(N), instrument count, "
            "1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(panel=["gapped"], steps=["onestep", "twostep"]),
            EST_SE,
            "abdata with 1980 dropped for every third firm, one-step robust "
            "and two-step Windmeijer, vs xtabond2 (instruments written on the "
            "level): coefficients, SEs, e(N), instrument count, 1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(panel=["gapped"], transform=["fod"]),
            EST_SE,
            "gapped panel, forward orthogonal deviations, vs xtabond2, 1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(panel=["gapped"], regressors=["x"]),
            EST_SE,
            "gapped panel, x = w k, vs xtabond2, 1e-9",
            XTABOND,
        ),
        _Row(
            "T2",
            ABD,
            cfg(
                panel=["gapped"],
                collapse=["true"],
                gmm_window=["max_capped", "min_and_max"],
            ),
            EST_SE,
            "gapped panel, one collapsed instrument at depths 2 to 6 "
            "(gmm_lags=(d, d)), vs xtabond2, 1e-9",
            XTABOND,
        ),
        _Row(
            "T1",
            DGP,
            cfg(regressors=["x"]),
            ("estimate",),
            "the Arellano-Bond moment system rebuilt by hand and solved by "
            "sp.gmm with the same one-step weight gives the same (rho, beta), "
            "1e-5 (sp.gmm's BFGS stopping rule)",
            XTABOND,
        ),
        _Row(
            "S",
            DGP,
            cfg(regressors=["x"]),
            ("estimate",),
            "known DGP (rho 0.5, beta 1): Monte Carlo mean of rho over 20 "
            "panels within 4 MC standard errors, beta within 4 SEs on one "
            "large draw",
            XTABOND,
        ),
        # ------------------------------------------------- system GMM
        _Row(
            "T2",
            ABD,
            sys_cfg(steps=["onestep", "twostep"]),
            ALL3,
            "system GMM, x = w k, one-step robust and two-step Windmeijer, vs "
            "xtabond2: coefficients and SEs 1e-9, 38 instruments, AR z 1e-9, "
            "Hansen J 1e-9; difference-in-Hansen groups vs xtabond2's printed "
            "output, abs 0.005",
            SYSTEM,
        ),
        _Row(
            "T2",
            ABD,
            sys_cfg(steps=["twostep"], collapse=["true"]),
            ALL3,
            "system GMM, two-step Windmeijer, collapsed, vs xtabond2: "
            "coefficients and SEs 1e-9, 11 instruments, Hansen J 1e-9",
            SYSTEM,
        ),
        _Row(
            "T2",
            ABD,
            sys_cfg(transform=["fod"], steps=["onestep", "twostep"]),
            ALL3,
            "system GMM with forward orthogonal deviations, one-step robust "
            "and two-step Windmeijer, vs xtabond2, orthogonal: coefficients, "
            "SEs and AR z, 1e-9",
            SYSTEM,
        ),
        _Row(
            "T2",
            ABD,
            sys_cfg(cluster=["coarser"]),
            ALL3,
            "system GMM clustered on industry, one-step robust, vs xtabond2: "
            "coefficients and SEs 1e-9, AR z 1e-8",
            SYSTEM,
        ),
        _Row(
            "T2",
            ABD,
            cfg(model=["system_h3_no_iv"], panel=["gapped"]),
            EST_SE,
            "gapped panel, system GMM (AR(1), constant), vs xtabond2, 1e-9",
            SYSTEM,
        ),
        _Row(
            "T4",
            ABD,
            sys_cfg(),
            ("diagnostic",),
            "Sargan statistic: xtabond's scale; xtabond2's is exactly "
            "N*/(N* - k) larger, asserted to 1e-9",
            SYSTEM,
        ),
        # ------------------------------- system GMM through sp.xtdpdsys (h=2)
        _Row(
            "T2",
            ABD,
            cfg(model=["system_h2_no_iv"], steps=["onestep", "twostep"]),
            EST_SE,
            "AR(1), one-step robust and two-step Windmeijer, vs Stata 18 "
            "xtdpdsys: coefficients and SEs 1e-9, instrument count",
            XTDPDSYS,
        ),
        _Row(
            "T2",
            ABD,
            cfg(
                model=["system_h2_iv_diff"],
                regressors=["x"],
                steps=["onestep", "twostep"],
            ),
            EST_SE,
            "x = w k in the differenced equation only, one-step robust and "
            "two-step Windmeijer, vs Stata 18 xtdpdsys, 1e-9",
            XTDPDSYS,
        ),
        _Row(
            "T2",
            ABD,
            cfg(model=["system_h2_iv_diff"], regressors=["x"], robust=["false"]),
            EST_SE,
            "x = w k, one-step classical, vs Stata 18 xtdpdsys, 1e-9",
            XTDPDSYS,
        ),
        _Row(
            "T2",
            HANSEN,
            cfg(
                model=["system_h2_iv_both"],
                regressors=["dummies"],
                lags=["2"],
                robust=["false"],
            ),
            ALL3,
            "two lags of y and period dummies, one-step classical, vs Stata "
            "18 xtdpd (h=2, iv() in both equations): coefficients, SEs and "
            "Sargan, 1e-9",
            XTDPDSYS,
        ),
        _Row(
            "T2",
            HANSEN,
            cfg(model=["system_h2_iv_both"], regressors=["dummies"], lags=["2"]),
            ("se",),
            "same, one-step vce(robust): SEs vs xtdpd, 1e-9",
            XTDPDSYS,
        ),
        _Row(
            "T2",
            HANSEN,
            cfg(
                model=["system_h2_iv_both"],
                regressors=["dummies"],
                lags=["2"],
                steps=["twostep"],
            ),
            EST_SE,
            "same, two-step Windmeijer, vs xtdpd, 1e-8",
            XTDPDSYS,
        ),
        _Row(
            "T2",
            HANSEN,
            cfg(
                model=["system_h2_no_iv"],
                lags=["2"],
                steps=["twostep"],
                robust=["false"],
            ),
            ALL3,
            "two lags of y, two-step with a singular weight matrix (early "
            "years kept for six firms), conventional SEs, vs xtdpd: "
            "coefficients and SEs (generalized inverse by sweep, as Mata's "
            "invsym) and Hansen J, 1e-9",
            XTDPDSYS,
        ),
        _Row(
            "T2",
            HANSEN,
            cfg(model=["system_h2_no_iv"], lags=["2"], steps=["twostep"]),
            ("se",),
            "same singular-weight fit, Windmeijer SEs (the zeroed moments "
            "left out of the one-step covariance) vs xtdpd, 1e-9",
            XTDPDSYS,
        ),
        _Row(
            "T2",
            HANSEN,
            cfg(
                model=["system_h2_iv_both"],
                regressors=["x_endo_window_dummies"],
                lags=["2"],
                gmm_window=["max_capped"],
                steps=["twostep"],
            ),
            EST_SE,
            "xtdpd with two lags of y, k exogenous, w endogenous "
            "(dgmmiv(n w, lag(2 4))) and period dummies, two-step Windmeijer, "
            "translated by sp.stata, vs Stata 18 xtdpd: coefficients and SEs, "
            "1e-8",
            "sp.stata(...)",
        ),
    )

    identity = _Row(
        "T1",
        ABD,
        sys_cfg(),
        ("estimate",),
        "with iv_equation='both', h=3 the coefficients equal "
        "sp.xtabond(method='system') (itself matched to xtabond2), 1e-14",
        XTDPDSYS,
    )
    note = (
        "sp.xtdpdsys forwards every argument to sp.xtabond(method='system'); "
        "system rows run through either name cover both. model = estimator x "
        "one-step weight h x where x and the period dummies are instrumented "
        "x constant; regressors = x / predetermined / endogenous (at a "
        "user-set lag window) / period dummies; lags = lags of y (read from "
        "the call; references ran one and two). Untested: three or more lags "
        "of y, h=1, "
        "iv_equation='level', h=3 with iv_equation='diff', constant=False in "
        "system GMM, the default predetermined / endogenous windows, period "
        "dummies in difference GMM, steps above two / 'iterated' / 'cue', "
        "two-step with a coarser cluster."
    )

    add(
        _Scope(
            "xtabond",
            domains,
            extractor(default_h=3, default_iv="both"),
            rows,
            invariant=robust_invariant,
            note=note,
        )
    )
    sys_models = _SYSTEM_MODELS + ("h1",)
    add(
        _Scope(
            "xtdpdsys",
            dict(domains, model=sys_models),
            extractor(default_h=2, default_iv="diff"),
            tuple(r for r in rows if r.config["model"] <= set(sys_models))
            + (identity,),
            invariant=robust_invariant,
            note=note,
        )
    )
