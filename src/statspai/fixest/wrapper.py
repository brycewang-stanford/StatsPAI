"""
``sp.feols`` / ``sp.fepois`` / ``sp.feglm``: fixest-style estimation.

Plain ``sp.feols`` / ``sp.fepois`` calls run on StatsPAI's own engine
(:mod:`statspai.fixest._native`); the rest delegates to pyfixest. Either
way the result is StatsPAI's ``EconometricResults``, compatible with
``sp.regtable`` and the rest of the StatsPAI ecosystem.
"""

import contextlib
import functools
import importlib.util
import inspect
import re
import threading
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..core._call_record import records_call
from ..core._design_guardrails import check_design as _check_design
from ..core._vcov_spec import markout_clusters, stata_vce_to_fixest
from ..core.results import EconometricResults
from ..core.utils import (
    _widen_narrow_integers,
    r_formula_idioms,
    raise_if_formula_names_missing,
)
from ..exceptions import ColumnNotFound, MethodIncompatibility, NumericalInstability
from ..output._lineage import records_provenance
from .adapter import _multi_fit_to_results, _pyfixest_to_econometric_results

#: vcov sentinels (case-insensitive) that select the wild cluster bootstrap.
_WILD_VCOV = frozenset({"wild", "wildbootstrap", "wild_cluster", "wcr", "boottest"})

#: fixest-style varying-slope FE, e.g. ``firm[year]`` / ``firm[[year]]``.
_VARYING_SLOPE_RE = re.compile(r"([A-Za-z_]\w*)\s*\[\[?\s*([^\]]+?)\s*\]\]?")

#: Stata-style varying-slope FE, e.g. ``i.pref#c.year``.
_STATA_SLOPE_RE = re.compile(r"i\.\s*([A-Za-z_]\w*)\s*#\s*c\.\s*([A-Za-z_]\w*)")


def _require_frame_columns(
    data: pd.DataFrame,
    terms: List[str],
    named: Dict[str, Any],
    context: str,
) -> None:
    """Raise ColumnNotFound for formula terms / named columns absent from data."""
    missing: Dict[str, Any] = {t: t for t in terms if t not in data.columns}
    missing.update({k: v for k, v in named.items() if v not in data.columns})
    if missing:
        raise ColumnNotFound.from_missing(missing, data.columns, context=context)


def _name_lookup_errors() -> Tuple[Type[BaseException], ...]:
    """What pyfixest raises for a name that is not a column.

    A formula term it cannot find surfaces from formulaic as a
    ``FormulaicError``; a misspelled cluster column as a ``KeyError``.
    """
    try:
        from formulaic.errors import FormulaicError
    except ImportError:  # pragma: no cover - formulaic ships with pyfixest
        return (LookupError, ValueError)
    return (LookupError, ValueError, FormulaicError)


def _raise_if_names_missing(
    pf_kwargs: Dict[str, Any], exc: BaseException, context: str
) -> None:
    """Re-raise a pyfixest name-lookup failure as ColumnNotFound.

    Covers a misspelled formula variable and a misspelled cluster column
    (``vcov={"CRV1": "firm"}`` / ``cluster="firm"``), which pyfixest reports
    as a bare ``KeyError``.  Returns when neither explains ``exc``.

    Also turns pyfixest's bare "Demeaning failed after N iterations" into a
    typed error that says how to get an answer.
    """
    if (
        isinstance(exc, ValueError)
        and "Demeaning failed" in str(exc)
        and context != "sp.fepois"  # sp.fepois points to sp.ppmlhdfe itself
    ):
        raise NumericalInstability(
            f"{context}: pyfixest's fixed-effect demeaning stopped before "
            f"converging ({exc}). This happens on poorly connected "
            "fixed-effect graphs (chains of groups linked by few "
            "observations, many singleton groups).",
            recovery_hint=(
                "Fit a formula the native engine takes (y ~ x1 + x2 | fe1 + "
                "fe2 with backend='auto'; it finishes unconverged projections "
                "with a sparse direct solve), or use sp.hdfe_ols, or pass "
                "fixef_maxiter= larger than the default."
            ),
            diagnostics={"pyfixest_error": str(exc)},
        ) from exc
    data = pf_kwargs.get("data")
    fml = pf_kwargs.get("fml")
    if not isinstance(data, pd.DataFrame):
        return
    if isinstance(fml, str):
        raise_if_formula_names_missing(fml, data, exc, context=context)
    vcov = pf_kwargs.get("vcov")
    if isinstance(exc, KeyError) and isinstance(vcov, dict):
        parts = [
            p.strip()
            for v in vcov.values()
            if isinstance(v, str)
            for p in re.split(r"[+^]", v)
            if p.strip()
        ]
        labels = (
            ["cluster"]
            if len(parts) == 1
            else [f"cluster[{i}]" for i in range(len(parts))]
        )
        missing = {lab: p for lab, p in zip(labels, parts) if p not in data.columns}
        if missing:
            raise ColumnNotFound.from_missing(
                missing, data.columns, context=context
            ) from exc


def _fe_part(fml: str) -> str:
    """Return the fixed-effects segment of a fixest formula (empty if none).

    ``y ~ x | fe1 + fe2`` -> ``fe1 + fe2``. For IV formulas
    (``y ~ x | fe | d ~ z``) only the FE segment is returned.
    """
    parts = fml.split("|")
    return parts[1] if len(parts) > 1 else ""


def _has_varying_slopes(fml: str) -> bool:
    fe = _fe_part(fml)
    return bool(fe) and (
        _VARYING_SLOPE_RE.search(fe) is not None
        or _STATA_SLOPE_RE.search(fe) is not None
    )


def _translate_varying_slopes(fml: str) -> Tuple[str, bool]:
    """fixest FE syntax -> ``sp.hdfe_ols`` absorb syntax.

    ``f[x]`` -> ``f + i.f#c.x`` (fixed effect and slope); ``f[[x]]`` ->
    ``i.f#c.x`` (slope only); ``f[x1, x2]`` gives one slope per variable.
    Returns ``(formula, needs_constant)``: R fixest keeps a global intercept
    when every absorbed term is a pure slope (``y ~ x | f[[z]]``), whereas
    Stata ``reghdfe absorb(i.f#c.z)`` does not, so a constant FE is added.
    """
    parts = fml.split("|")
    fe_terms = [t.strip() for t in parts[1].split("+") if t.strip()]
    out: List[str] = []
    has_intercept_fe = False
    for term in fe_terms:
        m = re.fullmatch(r"([A-Za-z_]\w*)\s*(\[\[?)\s*([^\]]+?)\s*\]\]?", term)
        if m is None:
            out.append(term)
            if not _STATA_SLOPE_RE.fullmatch(term):
                has_intercept_fe = True
            continue
        factor, brackets, vars_ = m.group(1), m.group(2), m.group(3)
        if brackets == "[":
            out.append(factor)
            has_intercept_fe = True
        for v in (x.strip() for x in vars_.split(",") if x.strip()):
            out.append(f"i.{factor}#c.{v}")
    rest = "|".join(parts[2:])
    new = parts[0].rstrip() + " | " + " + ".join(out)
    if rest:
        new += " |" + rest
    return new, not has_intercept_fe


def _native_eligible(
    fml: str, vcov: Any, ssc: Any, extra: Dict[str, Any], weights: Any = None
) -> bool:
    """Whether ``backend='native'`` can fit this ``feols`` call."""
    if ssc is not None or extra or fml.count("|") > 1:
        return False
    if weights is not None and isinstance(vcov, str) and vcov.lower() != "iid":
        return False  # the native robust menu is unweighted
    if any(tok in fml for tok in ("sw(", "csw(", "sw0(", "csw0(", "C(", "i(")):
        return False
    if vcov is None or isinstance(vcov, dict):
        return vcov is None or (
            len(vcov) == 1 and str(next(iter(vcov))).upper() in ("CRV1", "CLUSTER")
        )
    return str(vcov).lower() in ("iid", "hetero", "hc1", "robust")


def _native_vcov(p: Any) -> Any:
    """The ``vcov`` a native plan answers, in the form the notes read."""
    if p.vcov == "cr1":
        if isinstance(p.cluster, list):
            return {"CRV1": "+".join(p.cluster)}
        return {"CRV1": p.cluster}
    return "hetero" if p.vcov == "hc1" else "iid"


def _native_feols(
    fml: str,
    data: pd.DataFrame,
    *,
    vcov: Any,
    cluster: Optional[str],
    weights: Optional[str],
    ssc: Any,
    fixef_rm: str,
    lean: bool,
    extra: Dict[str, Any],
    strict: bool,
) -> Optional[Union[EconometricResults, List[EconometricResults]]]:
    """``sp.feols`` on StatsPAI's engine, or ``None`` to delegate.

    ``None`` when the call needs something only pyfixest implements, or
    when the native fit refuses absorbed / collinear regressors that
    fixest drops and pyfixest is installed to drop them.
    """
    from . import _native

    models = _native.expand_multiple(fml)
    if models is not None:
        return _native_feols_multi(
            models,
            data,
            vcov=vcov,
            cluster=cluster,
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            lean=lean,
            extra=extra,
        )
    p = _native.plan(
        fml,
        data,
        vcov=vcov,
        cluster=cluster,
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        extra=extra,
        multiway_ok=True,
        iv_ok=True,
    )
    if not isinstance(p, _native.Plan):
        return None
    try:
        out = _native.fit_feols(p, fml, lean=lean)
    except Exception as exc:
        if strict or not _native.refused(exc) or not _pyfixest_installed():
            raise
        return None
    v = _native_vcov(p)
    _note_cluster_sizes(out, p.data, v)
    _note_weights(out, p.data, v, weights)
    _note_multiway_adjustment(
        out, v, int(out.model_info.get("multiway_negative_eigenvalues", 0) or 0)
    )
    _check_design(
        out,
        data,
        formula=fml,
        entry="sp.feols",
        fe=p.fe,
        cluster=p.cluster,
        vcov=p.vcov,
        weights=weights,
    )
    return out


def _native_feols_multi(
    models: List[str],
    data: pd.DataFrame,
    *,
    vcov: Any,
    cluster: Optional[str],
    weights: Optional[str],
    ssc: Any,
    fixef_rm: str,
    lean: bool,
    extra: Dict[str, Any],
) -> Optional[List[EconometricResults]]:
    """A multiple-estimation ``sp.feols`` call on the native engine.

    All or nothing: ``None`` (delegate the whole call) unless every model
    is one the native engine fits on complete rows, so the list never mixes
    engines or samples. Like the delegate's list, the per-call notes
    (cluster sizes, weights) are not attached.
    """
    from . import _native

    plans = []
    for f in models:
        p = _native.plan(
            f,
            data,
            vcov=vcov,
            cluster=cluster,
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            extra=extra,
            multiway_ok=True,
            iv_ok=True,
        )
        if not isinstance(p, _native.Plan) or p.n_missing:
            return None
        plans.append((p, f))
    out: List[EconometricResults] = []
    for p, f in plans:
        try:
            out.append(_native.fit_feols(p, f, lean=lean))
        except Exception as exc:
            if not _native.refused(exc) or not _pyfixest_installed():
                raise
            return None
    return out


def _native_fepois(
    fml: str,
    data: pd.DataFrame,
    *,
    vcov: Any,
    cluster: Optional[str],
    weights: Optional[str],
    ssc: Any,
    fixef_rm: str,
    iwls_tol: float,
    iwls_maxiter: int,
    extra: Dict[str, Any],
    strict: bool,
) -> Optional[EconometricResults]:
    """``sp.fepois`` on StatsPAI's engine, or ``None`` to delegate."""
    from . import _native

    p = _native.plan(
        fml,
        data,
        vcov=vcov,
        cluster=cluster,
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        extra=extra,
        poisson=True,
        multiway_ok=True,
    )
    if not isinstance(p, _native.Plan):
        if strict:
            raise MethodIncompatibility(
                f"fepois(backend='native') cannot fit this call: {p}.",
                recovery_hint="Use backend='auto' to let pyfixest fit it.",
            )
        return None
    try:
        out = _native.fit_fepois(p, fml, tol=iwls_tol, maxiter=iwls_maxiter)
    except Exception as exc:
        if strict or not _native.refused(exc) or not _pyfixest_installed():
            raise
        return None
    _note_multiway_adjustment(
        out,
        _native_vcov(p),
        int(out.model_info.get("multiway_negative_eigenvalues", 0) or 0),
    )
    _check_design(
        out,
        data,
        formula=fml,
        entry="sp.fepois",
        fe=p.fe,
        cluster=p.cluster,
        vcov=p.vcov,
        weights=weights,
        poisson=True,
    )
    return out


def _native_feglm(
    fml: str,
    data: pd.DataFrame,
    *,
    family: str,
    vcov: Any,
    cluster: Optional[str],
    extra: Dict[str, Any],
    strict: bool,
    bias_correction: bool = False,
) -> Optional[EconometricResults]:
    """``sp.feglm`` (logit / probit) on StatsPAI's engine, or ``None``."""
    from . import _native

    fam = str(family).lower()
    negbin = fam == "negbin"
    rest = dict(extra)
    weights = rest.pop("weights", None)
    p: Any
    if fam not in ("logit", "probit", "negbin"):
        p = f"family={family!r} (native: 'logit', 'probit', 'negbin')"
    else:
        p = _native.plan(
            fml,
            data,
            vcov=vcov,
            cluster=cluster,
            weights=weights,
            ssc=rest.pop("ssc", None),
            fixef_rm="none",
            extra=rest,
            binomial=not negbin,
            poisson=negbin,
            multiway_ok=True,
        )
    if bias_correction and fam not in ("logit", "probit"):
        raise MethodIncompatibility(
            f"feglm: bias_correction=True is defined for family='logit' / "
            f"'probit', not {family!r}.",
            recovery_hint="Use family='logit' or 'probit', or drop bias_correction=.",
        )
    if not isinstance(p, _native.Plan):
        if strict or negbin or bias_correction:
            # pyfixest has no negative binomial: nothing to delegate to
            raise MethodIncompatibility(
                f"feglm(family={family!r}) cannot fit this call natively: {p}.",
                recovery_hint=(
                    "Write the call as 'y ~ x1 + x2 | fe1 + fe2' with numeric "
                    "regressors and vcov in 'iid' / 'hetero' / {'CRV1': 'a'} "
                    "/ {'CRV1': 'a + b'}"
                    + ("." if negbin else ", or use backend='auto'.")
                ),
            )
        return None
    try:
        if negbin:
            from ._native_negbin import fit_fenegbin

            out = fit_fenegbin(p, fml)
        else:
            out = _native.fit_feglm(p, fml, family=fam, bias_correction=bias_correction)
    except Exception as exc:
        if (
            strict
            or negbin
            or bias_correction
            or not _native.refused(exc)
            or not _pyfixest_installed()
        ):
            raise
        return None
    _note_multiway_adjustment(
        out,
        _native_vcov(p),
        int(out.model_info.get("multiway_negative_eigenvalues", 0) or 0),
    )
    return out


def _pyfixest_installed() -> bool:
    try:
        return importlib.util.find_spec("pyfixest") is not None
    except (ImportError, ValueError):  # a module entry of None / no spec
        return False


def _feols_varying_slopes(
    fml: str,
    data: pd.DataFrame,
    *,
    vcov: Any,
    cluster: Optional[str],
    weights: Optional[str],
    ssc: Any,
    fixef_rm: str,
    extra: Dict[str, Any],
) -> EconometricResults:
    """Varying-slope FE through the native kernel, reported like feols."""
    from ..panel.feols import hdfe_ols as _hdfe_ols

    if extra:
        raise MethodIncompatibility(
            f"feols with varying slopes does not support {sorted(extra)}.",
            recovery_hint="Drop them, or absorb the slopes with sp.hdfe_ols.",
        )
    if ssc is not None:
        raise MethodIncompatibility(
            "feols with varying slopes uses the native HDFE kernel, which "
            "does not take pyfixest ssc= objects.",
            recovery_hint="Drop ssc=, or call sp.hdfe_ols directly.",
        )
    if "|" in fml.split("|", 2)[-1] and fml.count("|") > 1:
        raise MethodIncompatibility(
            "feols with varying slopes does not support an IV part.",
        )
    kw: Dict[str, Any] = {}
    label = "iid"
    if isinstance(vcov, dict):
        ((kind, cl),) = vcov.items()
        if str(kind).upper() not in ("CRV1", "CLUSTER"):
            raise MethodIncompatibility(
                f"feols with varying slopes supports vcov='iid', 'hetero' or "
                f"{{'CRV1': col}}; got {vcov!r}."
            )
        cluster = cl
    elif isinstance(vcov, str):
        v = vcov.lower()
        if v in ("hetero", "hc1", "robust"):
            if cluster is not None:
                raise MethodIncompatibility(
                    f"feols: vcov={vcov!r} and cluster={cluster!r} were both "
                    "given; the heteroskedasticity-robust variance would "
                    "drop the clustering.",
                    recovery_hint="Pass cluster= alone (or "
                    f"vcov={{'CRV1': {cluster!r}}}) for cluster-robust "
                    f"standard errors, or vcov={vcov!r} alone.",
                    diagnostics={"vcov": vcov, "cluster": cluster},
                )
            kw["vce"] = "robust"
            label = "hetero"
        elif v not in ("iid",):
            raise MethodIncompatibility(
                f"feols with varying slopes supports vcov='iid', 'hetero' or "
                f"{{'CRV1': col}}; got {vcov!r}."
            )
    if cluster is not None:
        kw["cluster"] = cluster
        label = f"CRV1:{cluster}"
    translated, needs_constant = _translate_varying_slopes(fml)
    work = data
    if needs_constant:
        # fixest keeps (and reports) the intercept when only slopes are
        # absorbed; carry it as an explicit regressor.
        work = data.assign(__const__=1.0)
        lhs, rhs = translated.split("|", 1)
        dep, regs = lhs.split("~", 1)
        translated = f"{dep.strip()} ~ __const__ + {regs.strip()} |{rhs}"
    res = _hdfe_ols(
        translated,
        data=work,
        weights=weights,
        drop_singletons=(fixef_rm == "singleton"),
        tol=1e-12,
        **kw,
    )
    params = res.params.rename({"__const__": "Intercept"})
    ses = res.std_errors.rename({"__const__": "Intercept"})
    names = list(params.index)
    k = len(names)
    V = np.asarray(res.vcov, dtype=float)
    data_info: Dict[str, Any] = {
        "nobs": int(res.n_obs),
        "df_model": k,
        "df_resid": int(res.df_resid),
        "var_cov": V if V.shape == (k, k) else None,
        "var_names": names,
        "residuals": np.asarray(res.residuals),
    }
    n_cl = (res.cluster_info or {}).get("n_clusters")
    if cluster is not None and n_cl:
        data_info["df_inference"] = int(min(n_cl)) - 1
    model_info: Dict[str, Any] = {
        "model_type": "OLS (StatsPAI native HDFE)",
        "method": "High-Dimensional Fixed Effects"
        + (" (varying slopes)" if _has_varying_slopes(fml) else ""),
        "formula": fml,
        "absorbed_formula": translated,
        "vcov_type": label,
        "fixed_effects": _fe_part(fml),
        "backend": "statspai-native",
        "implementation": "native",
        "cluster": cluster,
    }
    if cluster is not None and n_cl:
        model_info["n_clusters"] = int(min(n_cl))
        _eff = (res.cluster_info or {}).get("n_clusters_effective")
        if _eff is not None:
            model_info["n_clusters_effective"] = float(_eff)
    diagnostics = {
        "R-squared": float(res.r2),
        "Adj. R-squared": float(res.r2_a),
        "R-squared (within)": float(res.r2_within),
        "Adj. R-squared (within)": float(res.r2_a_within),
    }
    return EconometricResults(
        params=params.rename(None),
        std_errors=ses.rename(None),
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def _reject_silent_varying_slopes(fml: str) -> None:
    """Fail loudly on varying-slope FE syntax that pyfixest silently ignores.

    pyfixest (as of 0.50.1) *parses* ``y ~ d | county + pref[year]`` without
    error but never absorbs the slope: the fit is bit-identical to
    ``y ~ d | county``. Stata's ``reghdfe absorb(county i.pref#c.year)``
    absorbs it, so the two disagree — silently, and by a wide margin.

    Returning a quietly-wrong coefficient is the worst possible outcome, so we
    refuse to run and point at the two paths that are actually correct.
    """
    fe = _fe_part(fml)
    if not fe:
        return

    slope = _VARYING_SLOPE_RE.search(fe)
    stata = _STATA_SLOPE_RE.search(fe)
    if slope is None and stata is None:
        return

    factor, var = (slope or stata).groups()  # type: ignore[union-attr]
    shown = (slope or stata).group(0)  # type: ignore[union-attr]
    raise MethodIncompatibility(
        f"feols() cannot absorb the varying slope {shown!r}: the pyfixest "
        f"backend accepts this syntax but silently drops the slope, returning "
        f"the same estimate as if you had omitted it entirely.\n"
        f"Two paths give the correct (Stata reghdfe-equivalent) answer:\n"
        f"  1. Estimate the slope instead of absorbing it — move it to the "
        f"right-hand side:\n"
        f"       sp.feols('... + i({factor}, {var}) | <other FE>', data=...)\n"
        f"  2. Absorb it with StatsPAI's own HDFE kernel:\n"
        f"       sp.hdfe_ols('... | <other FE> + i.{factor}#c.{var}', data=...)"
    )


_MULTI_REF_RE = re.compile(
    r"i\(\s*(?P<f>[^\W\d]\w*)\s*(?:,\s*(?P<v>[^\W\d]\w*)\s*)?,"
    r"\s*ref\s*=\s*(?:\[(?P<lst>[^\]]*)\]|c\((?P<rc>[^)]*)\))\s*\)"
)


def _parse_ref_levels(text: str, column: pd.Series) -> List[Any]:
    """Reference levels written in a formula, typed like the column."""
    out: List[Any] = []
    for tok in (t.strip().strip("'\"") for t in text.split(",") if t.strip()):
        match = [v for v in pd.unique(column.dropna()) if str(v) == tok]
        if not match:
            try:
                num = float(tok)
                match = [v for v in pd.unique(column.dropna()) if v == num]
            except ValueError:
                match = []
        if not match:
            raise MethodIncompatibility(
                f"i(..., ref=[...]): level {tok!r} is not a value of "
                f"'{column.name}'.",
            )
        out.append(match[0])
    return out


def _expand_multi_ref(
    fml: str, data: pd.DataFrame
) -> Tuple[str, pd.DataFrame, Dict[str, str]]:
    """``i(f, ref=[a, b])`` (fixest ``ref = c(a, b)``) for pyfixest.

    pyfixest takes one reference level. Omitting the dummies of levels
    ``a`` and ``b`` is the same model as merging ``b`` into ``a`` and using
    that merged level as the base, so the term is rewritten on a copy of the
    factor with ``b -> a`` and the coefficient names are mapped back.
    Returns the new formula, the (possibly augmented) data and the
    name mapping ``{pyfixest name: user-facing name}``.
    """
    renames: Dict[str, str] = {}
    if "ref" not in fml:
        return fml, data, renames
    work = data
    counter = 0

    def repl(m: "re.Match[str]") -> str:
        nonlocal work, counter
        f, v = m.group("f"), m.group("v")
        levels = _parse_ref_levels(m.group("lst") or m.group("rc") or "", data[f])
        if len(levels) == 1:  # ref=[a] is ref=a
            ref1 = repr(levels[0]) if isinstance(levels[0], str) else str(levels[0])
            return f"i({f}, {v}, ref={ref1})" if v else f"i({f}, ref={ref1})"
        counter += 1
        # No leading underscores: pyfixest drops ``__x`` terms silently.
        tmp = f"{f}_multiref{counter}"
        while tmp in data.columns:
            counter += 1
            tmp = f"{f}_multiref{counter}"
        if work is data:
            work = data.copy()
        base = levels[0]
        work[tmp] = data[f].where(~data[f].isin(levels[1:]), base)
        for lev in pd.unique(data[f].dropna()):
            if lev in levels:
                continue
            suffix = f":{v}" if v else ""
            renames[f"{tmp}::{lev}{suffix}"] = f"{f}::{lev}{suffix}"
        ref = repr(base) if isinstance(base, str) else str(base)
        return f"i({tmp}, {v}, ref={ref})" if v else f"i({tmp}, ref={ref})"

    new = _MULTI_REF_RE.sub(repl, fml)
    return new, work, renames


def _rename_terms(res: Any, renames: Dict[str, str]) -> Any:
    """Map coefficient names on a fitted result (see ``_expand_multi_ref``)."""
    if not renames:
        return res
    for attr in (
        "params",
        "std_errors",
        "tvalues",
        "pvalues",
        "conf_int_lower",
        "conf_int_upper",
    ):
        val = getattr(res, attr, None)
        if isinstance(val, pd.Series):
            setattr(res, attr, val.rename(index=lambda n: renames.get(str(n), n)))
    di = getattr(res, "data_info", None)
    if isinstance(di, dict) and isinstance(di.get("var_names"), list):
        di["var_names"] = [renames.get(str(n), n) for n in di["var_names"]]
    return res


def _note_cluster_sizes(result: Any, data: pd.DataFrame, vcov: Any) -> None:
    """Record the effective number of clusters of a one-way clustered fit.

    Warns, as ``sp.regress`` and ``sp.panel`` do, when many clusters are
    few in effect. The key is read from ``data`` as passed, so rows the
    fit dropped still count; for a diagnostic of cluster sizes that is
    close enough.
    """
    if not isinstance(vcov, dict) or len(vcov) != 1:
        return
    column = next(iter(vcov.values()))
    if not isinstance(column, str) or column not in getattr(data, "columns", ()):
        return
    from ..core._agent_summary import warn_if_clusters_unequal

    keys = data[column].dropna()
    if keys.empty:
        return
    info = dict(getattr(result, "model_info", None) or {})
    info["n_clusters_effective"] = warn_if_clusters_unequal(keys, column, stacklevel=4)
    # Few clusters ever treated: plain regressors only (a term such as
    # i(x) or a transformed variable is not a column of ``data``).
    names = [str(n) for n in getattr(result, "params", pd.Series(dtype=float)).index]
    plain = [n for n in names if n in data.columns]
    if plain:
        frame = data[plain + [column]].dropna()
        try:
            X = frame[plain].to_numpy(dtype=float)
        except (TypeError, ValueError):
            X = None  # a non-numeric regressor column: nothing to scan
        if X is not None and len(frame):
            from ..core._agent_summary import warn_if_few_ever_treated

            few = warn_if_few_ever_treated(
                X, plain, frame[column].to_numpy(), column, stacklevel=4
            )
            if few:
                info["few_treated_clusters"] = few
    result.model_info = info


def _note_weights(result: Any, data: pd.DataFrame, vcov: Any, weights: Any) -> None:
    """Record the Kish effective sizes of a weighted fit, and warn.

    Same diagnostic as ``sp.panel`` and ``sp.hdfe_ols``. Weights and the
    cluster key are read from ``data`` on the rows where both are present.
    """
    if not isinstance(weights, str) or weights not in getattr(data, "columns", ()):
        return
    info = dict(getattr(result, "model_info", None) or {})
    label = str(info.get("vcov_type", "")).lower()
    column = None
    if isinstance(vcov, dict) and len(vcov) == 1:
        column = next(iter(vcov.values()))
        if not isinstance(column, str) or column not in data.columns:
            column = None
    frame = data[[weights] + ([column] if column else [])].dropna()
    if frame.empty:
        return
    if label == "iid":
        kind = "classical"
    elif label in ("hetero", "hc1", "hc2", "hc3"):
        kind = "robust" if label == "hetero" else label
    elif column is not None:
        kind = "cluster"
    else:
        kind = "other"
    from ..core._agent_summary import note_weight_diagnostics

    note_weight_diagnostics(
        info,
        frame[weights].to_numpy(dtype=float),
        kind,
        entry="sp.feols",
        robust_option="vcov='hetero' or vcov={'CRV1': <cluster>}",
        classical_reading="analytic weights (Stata [aw=])",
        cluster_keys=frame[column].to_numpy() if column else None,
        cluster=column,
        stacklevel=5,
    )
    result.model_info = info


def _require_hac_ids(vcov: Any, kwargs: Dict[str, Any], data: Any, entry: str) -> None:
    """Driscoll-Kraay / Newey-West need the panel's time (and unit) ids.

    pyfixest reads them from ``vcov_kwargs``; without them it failed with
    ``KeyError: None`` (``sp.vcov(fit, 'DK')`` reached that).
    """
    if not isinstance(vcov, str) or vcov.upper() not in ("DK", "NW"):
        return
    kind = vcov.upper()
    vk = kwargs.get("vcov_kwargs") or {}
    need = ["time_id", "lag"] + (["panel_id"] if kind == "DK" else [])
    missing = [k for k in need if vk.get(k) is None]
    example = "vcov_kwargs={'time_id': 'year', 'panel_id': 'firm', 'lag': 2}"
    label = "Driscoll-Kraay" if kind == "DK" else "Newey-West"
    if missing:
        raise MethodIncompatibility(
            f"{entry}: vcov={vcov!r} ({label}) needs {missing} in vcov_kwargs: "
            "the time variable, the unit variable (panel HAC) and the number "
            "of lags.",
            recovery_hint=f"sp.{entry}(..., vcov={kind!r}, {example}); after a "
            "fit, refit with these arguments (sp.vcov cannot add them).",
            diagnostics={"vcov": vcov, "missing": missing},
        )
    if isinstance(data, pd.DataFrame):
        absent = {
            k: vk[k]
            for k in ("time_id", "panel_id")
            if vk.get(k) is not None and vk[k] not in data.columns
        }
        if absent:
            raise ColumnNotFound.from_missing(absent, list(data.columns))


def _check_pyfixest() -> Any:
    """Import pyfixest or raise a structured ``MissingDependencyError``.

    The error is an ``ImportError`` (back-compatible) and a
    ``StatsPAIError`` whose diagnostics carry the install command.
    """
    from .._optional_deps import require_optional

    return require_optional(
        "pyfixest",
        extra="fixest",
        purpose="high-dimensional fixed effects estimation",
    )


def _prefer_compiled_demeaner(pf_fn: Any, pf_kwargs: Dict[str, Any]) -> None:
    """Ask pyfixest for its compiled demeaner unless the caller chose one.

    pyfixest's default is the numba alternating-projection demeaner, which
    converges slowly with many sparse fixed-effect levels: on 400,000 rows
    with 100,000 firm effects it took 41 s where ``demeaner_backend='rust'``
    took 0.2 s, with the same coefficients and standard errors (both stop
    at ``fixef_tol``). pyfixest 0.60 made the compiled demeaner its default
    and turned ``demeaner_backend`` into a deprecated legacy argument that
    may not be combined with its typed ``demeaner=``, so nothing is added
    when the function takes ``demeaner``. Releases without the option or
    without the compiled core keep their default.
    """
    if "demeaner_backend" in pf_kwargs or "demeaner" in pf_kwargs:
        return
    params = inspect.signature(pf_fn).parameters
    if "demeaner" in params or "demeaner_backend" not in params:
        return
    if importlib.util.find_spec("pyfixest.core._core_impl") is None:
        return
    pf_kwargs["demeaner_backend"] = "rust"


_FIRST_STAGE_PATCH_LOCK = threading.RLock()


@contextlib.contextmanager
def _first_stage_demeaner(fml: str, pf_kwargs: Dict[str, Any]) -> Iterator[None]:
    """Fit an IV model's first stage with the model's own demeaner.

    pyfixest's ``Feiv.first_stage`` refits the first stage through
    ``pyfixest.estimation.feols`` without passing ``demeaner_backend`` on,
    so it ran the numba demeaner (and the numba count of fixed effects
    nested in the clusters) while the model ran the compiled one. With
    1,000,000 rows, two fixed effects and clustering on one of them that
    first stage took 17 s, 10 s of it compiling numba code, where the model
    took under a second; the first-stage coefficients and F statistic agree
    to 1e-14 between the two demeaners. The function is swapped for the
    duration of the fit only, under a lock so that concurrent fits do not
    restore each other's copy.
    """
    backend = pf_kwargs.get("demeaner_backend")
    if backend is None or "|" not in fml or "~" not in fml.split("|", 1)[1]:
        yield
        return
    try:
        module = importlib.import_module("pyfixest.estimation")
    except ImportError:
        yield
        return
    with _FIRST_STAGE_PATCH_LOCK:
        original = getattr(module, "feols", None)
        if original is None or "demeaner" in inspect.signature(original).parameters:
            # pyfixest 0.60+ hands the first stage the model's resolved
            # ``demeaner=``; adding the legacy argument would be refused.
            yield
            return
        setattr(module, "feols", functools.partial(original, demeaner_backend=backend))
        try:
            yield
        finally:
            setattr(module, "feols", original)


_FE_TERM = re.compile(r"__fixed_effect__\((.+?)\)$")


def _written_order_encoder(original: Any) -> Any:
    """Wrap pyfixest's ``_encode_fixed_effects`` to keep the written order.

    pyfixest builds the fixed-effect formula from ``set(fixed_effects.split
    ("+"))``, so the order in which the alternating projections sweep the
    effects follows Python's string hashing and changes from one process to
    the next (``PYTHONHASHSEED``). The projections stop at ``fixef_tol``, so
    the residuals, and with them every standard error, then differed between
    runs of the same call: by 4e-9 relative for a one-way clustered SE on
    20,000 rows with two effects, and up to 1.7e-6 for two-way clustering.
    The wrapper sorts the terms back into the order the formula writes
    them; output it does not recognise is passed through unchanged.
    """

    @functools.wraps(original)
    def encode(fixed_effects: str, data: pd.DataFrame) -> str:
        out = str(original(fixed_effects=fixed_effects, data=data))
        written = [
            t.replace("^", "_")
            for t in dict.fromkeys(re.split(r"\s*\+\s*", fixed_effects.strip()))
        ]
        if not out.endswith(" - 1"):
            return out
        terms = out[: -len(" - 1")].split(" + ")
        names = [m.group(1) if (m := _FE_TERM.match(t)) else None for t in terms]
        if None in names or sorted(names) != sorted(written):  # type: ignore[type-var]
            return out
        by_name = dict(zip(names, terms))
        return " + ".join(by_name[w] for w in written) + " - 1"

    return encode


@contextlib.contextmanager
def _written_fe_order() -> Iterator[None]:
    """Run a pyfixest fit with its fixed effects in the written order."""
    try:
        module = importlib.import_module("pyfixest.estimation.formula.model_matrix")
    except ImportError:
        yield
        return
    with _FIRST_STAGE_PATCH_LOCK:
        original = getattr(module, "_encode_fixed_effects", None)
        if original is None:
            yield
            return
        setattr(module, "_encode_fixed_effects", _written_order_encoder(original))
        try:
            yield
        finally:
            setattr(module, "_encode_fixed_effects", original)


# --------------------------------------------------------------------------- #
#  feols — OLS / IV with high-dimensional fixed effects
# --------------------------------------------------------------------------- #


def _feols_wild(
    fml: str,
    data: pd.DataFrame,
    cluster: str,
    *,
    reps: int,
    weight_type: str,
    seed: Optional[int],
    weights: Optional[str],
    ssc: Optional[Any],
    fixef_rm: str,
    collin_tol: float,
    lean: bool,
    extra: Dict[str, Any],
) -> EconometricResults:
    """Native ``feols(..., vce="wild")`` — Stata ``boottest``-style inference.

    Fits the model once with cluster-robust (CRV1) SEs — which also stores the
    within-transformed design on the result (see ``fixest/adapter.py``) — then
    runs the WCR wild cluster bootstrap (Cameron-Gelbach-Miller 2008) on that
    partialled-out design for every non-absorbed coefficient.  Point estimates
    are the feols estimates; p-values and confidence intervals come from the
    bootstrap.  This is the verified within wild bootstrap (byte-identical to
    ``sp.regress`` on FE-demeaned data), promoted from the standalone
    ``sp.wild_cluster_boot`` to a first-class ``vce=`` option.
    """
    base = feols(
        fml,
        data,
        vcov={"CRV1": cluster},
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        collin_tol=collin_tol,
        lean=lean,
        **extra,
    )
    if isinstance(base, list):
        raise MethodIncompatibility(
            "feols(vce='wild') does not support multiple-estimation formulas "
            "(csw/sw/csw0). Fit each model separately."
        )

    from ..inference.jackknife import wild_cluster_boot

    se = base.std_errors.copy()
    pvals = pd.Series(np.nan, index=base.params.index, dtype=float)
    ci_lo = pd.Series(np.nan, index=base.params.index, dtype=float)
    ci_hi = pd.Series(np.nan, index=base.params.index, dtype=float)
    iv = (base.data_info or {}).get("iv")
    targets = list(base.params.index)
    if iv is not None:
        # 2SLS: the WRE bootstrap of the endogenous coefficients (as
        # sp.ivreg(vce='wild')); the exogenous ones keep their CRV1
        # inference. (Before, least squares on the structural design was
        # bootstrapped and reported as the IV fit's.)
        targets = list(iv["endog_names"])
        pvals = pd.Series(
            np.asarray(base.pvalues, dtype=float), index=base.params.index
        )
        from scipy import stats as _st

        dfi = base.data_info.get("df_inference") or base.data_info.get("df_resid")
        crit = float(_st.t.ppf(0.975, dfi) if dfi else _st.norm.ppf(0.975))
        ci_lo = base.params - crit * base.std_errors
        ci_hi = base.params + crit * base.std_errors
    for var in targets:
        out = wild_cluster_boot(
            base,
            data,
            cluster=cluster,
            variable=str(var),
            n_boot=reps,
            weight_type=weight_type,
            seed=seed,
        )
        pvals[var] = out["p_boot"]
        ci_lo[var], ci_hi[var] = out["ci_boot"]
        if iv is None:
            se[var] = out["se_cluster"]
        # IV: the fit's CRV1 standard error stands (the bootstrap's own
        # studentisation does not charge the absorbed effects)

    base.std_errors = se
    base.pvalues = pvals
    base.conf_int_lower = ci_lo
    base.conf_int_upper = ci_hi
    base.model_info = dict(base.model_info)
    base.model_info["vcov_type"] = (
        f"wild cluster bootstrap (Cameron-Gelbach-Miller 2008, {reps} reps, "
        f"{weight_type})"
        if iv is None
        else "WRE wild cluster bootstrap (Davidson-MacKinnon 2010, "
        f"{reps} reps, {weight_type}; endogenous coefficients)"
    )
    if iv is not None:
        base.model_info["wild_endogenous"] = targets
    base.model_info["n_boot"] = reps
    base.model_info["cluster"] = cluster
    return base


def _feols_bias_reduced(
    fml: str,
    data: pd.DataFrame,
    kind: str,
    *,
    cluster: Optional[str],
    conley_lat: Optional[str],
    conley_lon: Optional[str],
    conley_cutoff: Optional[float],
    weights: Optional[str],
    ssc: Optional[Any],
    fixef_rm: str,
    collin_tol: float,
    lean: bool,
    extra: Dict[str, Any],
) -> EconometricResults:
    """CR2/CR3/jackknife (Pustejovsky-Tipton) or Conley spatial HAC on feols.

    Fits the FE model once (populating the stored within design), then applies
    the requested bias-reduced / spatial estimator to the partialled-out design.
    CR2/CR3 match R clubSandwich (plm); Conley matches Stata acreg on the
    FE-demeaned data.
    """
    from scipy import stats as _stats

    from ..inference.jackknife import conley_vcov_ols, cr_vcov_ols

    if kind == "conley":
        if conley_lat is None or conley_lon is None or conley_cutoff is None:
            raise MethodIncompatibility(
                "feols(vce='conley') requires conley_lat=, conley_lon=, and "
                "conley_cutoff= (planar distance cutoff in km)."
            )
        _require_frame_columns(
            data,
            [],
            {"conley_lat": conley_lat, "conley_lon": conley_lon},
            "feols(vce='conley')",
        )
        base = feols(
            fml,
            data,
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            lean=lean,
            **extra,
        )
    else:
        if cluster is None:
            raise MethodIncompatibility(
                f"feols(vce={kind!r}) requires cluster=... (a cluster-robust "
                "small-sample correction)."
            )
        _require_frame_columns(data, [], {"cluster": cluster}, f"feols(vce={kind!r})")
        base = feols(
            fml,
            data,
            vcov={"CRV1": cluster},
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            lean=lean,
            **extra,
        )
    if isinstance(base, list):
        raise MethodIncompatibility(
            f"feols(vce={kind!r}) does not support multiple-estimation formulas."
        )

    if kind == "conley":
        se = conley_vcov_ols(base, data, conley_lat, conley_lon, conley_cutoff)
        label = f"Conley spatial HAC (acreg planar, {conley_cutoff} km)"
    else:
        power = 0.5 if kind == "cr2" else 1.0
        cl_codes = pd.Categorical(data[cluster]).codes
        # small_sample=False matches clubSandwich's CR2/CR3 for FE (plm) exactly.
        se = cr_vcov_ols(base, cl_codes, power=power, small_sample=False)
        label = {
            "cr2": "CR2 cluster-robust (clubSandwich, Pustejovsky-Tipton 2018)",
            "cr3": "CR3 cluster-robust (clubSandwich jackknife-type)",
            "jackknife": "CR3 cluster-robust (clubSandwich jackknife-type)",
        }[kind]

    base.std_errors = se
    if kind == "conley":
        z = base.params / se
        base.pvalues = pd.Series(2 * _stats.norm.sf(np.abs(z)), index=base.params.index)
        crit = _stats.norm.ppf(0.975)
        base.conf_int_lower = base.params - crit * se
        base.conf_int_upper = base.params + crit * se
    else:
        # t(G - 1), as for the one-way cluster variance and as Stata's
        # vce(hc2 clustvar); the normal was used through 1.36.0.
        from ..inference.jackknife import set_cluster_t_inference

        set_cluster_t_inference(base, se, len(np.unique(cl_codes)))
    base.model_info = dict(base.model_info)
    base.model_info["vcov_type"] = label
    if cluster is not None:
        base.model_info["cluster"] = cluster
    return base


#: vcov sentinels selecting the GLM bias-reduced cluster menu (CR2/CR3/jackknife).
_GLM_BR_VCOV = frozenset({"cr2", "cr3", "jackknife"})


def _sm_glm_family(family_str: Optional[str]) -> Any:
    """Map a pyfixest family string to a statsmodels GLM family instance."""
    from statsmodels.api import families as _fam

    f = (family_str or "poisson").lower()
    if f == "poisson":
        return _fam.Poisson()
    if f in ("logit", "binomial"):
        return _fam.Binomial()
    if f == "probit":
        return _fam.Binomial(link=_fam.links.Probit())
    if f == "gaussian":
        return _fam.Gaussian()
    raise MethodIncompatibility(
        f"vce='CR2'/'CR3' not supported for family={family_str!r}; "
        "supported: poisson, logit, probit, gaussian.",
        recovery_hint="Use cluster= (CRV1) or vce='wild' instead.",
    )


def _feglm_bias_reduced(
    fml: str,
    data: pd.DataFrame,
    kind: str,
    *,
    family_str: Optional[str],
    cluster: Optional[str],
    is_pois: bool,
    ssc: Optional[Any],
    fixef_rm: str,
    collin_tol: float,
    extra: Dict[str, Any],
) -> EconometricResults:
    """CR2/CR3 (Pustejovsky-Tipton) cluster-robust SEs for fepois / feglm.

    Reproduces R ``clubSandwich::vcovCR(glm, type="CR2"/"CR3")`` to machine
    precision by fitting the GLM with the fixed effects as **dummies** and
    applying the IRLS-weighted bias-reduced adjustment (see
    :func:`statspai.inference.jackknife.glm_cr_vcov`). Unlike OLS, the
    weighted-projection FWL equivalence does *not* carry the CR2 leverage
    through fixed-effect absorption, so the dummy design is required to match
    the reference — feasible for modest FE dimensionality only.
    """
    from scipy import stats as _stats

    from ..inference.jackknife import glm_cr_vcov

    who = "fepois" if is_pois else "feglm"
    if cluster is None:
        raise MethodIncompatibility(
            f"{who}(vce={kind!r}) requires cluster=... (a cluster-robust "
            "small-sample correction).",
            recovery_hint="Pass cluster='firm' (or another id column).",
        )
    _require_frame_columns(data, [], {"cluster": cluster}, f"{who}(vce={kind!r})")

    # Canonical estimate + full result object from the native estimator.
    if is_pois:
        base = fepois(
            fml,
            data,
            vcov={"CRV1": cluster},
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            **extra,
        )
    else:
        base = feglm(fml, data, family=family_str, vcov={"CRV1": cluster}, **extra)
    if isinstance(base, list):
        raise MethodIncompatibility(
            f"{who}(vce={kind!r}) does not support multiple-estimation formulas."
        )

    # Parse the formula into outcome / regressors / fixed effects.
    lhs, rhs = fml.split("~", 1)
    y_var = lhs.strip()
    fe_vars: List[str] = []
    if "|" in rhs:
        cov_part, fe_part = rhs.split("|", 1)
        fe_vars = [t.strip() for t in fe_part.replace("+", " ").split() if t.strip()]
    else:
        cov_part = rhs
    cov_vars = [
        t.strip() for t in cov_part.split("+") if t.strip() and t.strip() != "1"
    ]

    need = [y_var] + cov_vars + fe_vars + [cluster]
    _require_frame_columns(
        data, [y_var] + cov_vars + fe_vars, {"cluster": cluster}, f"{who}(vce={kind!r})"
    )
    work = data.loc[:, need].dropna().reset_index(drop=True)

    base_n = int(base.data_info.get("nobs", len(work)))
    if base_n != len(work):
        raise MethodIncompatibility(
            f"{who}(vce={kind!r}): the fitted sample ({base_n} obs) differs from "
            f"the complete-case design ({len(work)} obs) — singleton / separation "
            "dropping breaks the CR2/CR3 dummy-design alignment.",
            recovery_hint="Use cluster= (CRV1) or vce='wild' for this model.",
        )

    # Align result coefficients to design columns by name (simple regressors
    # only; the constant is named "Intercept" by pyfixest in the no-FE case).
    colmap = {nm: i for i, nm in enumerate(cov_vars)}
    colmap["Intercept"] = len(cov_vars)
    try:
        cov_pos = [colmap[str(nm)] for nm in base.params.index]
    except KeyError:
        raise MethodIncompatibility(
            f"{who}(vce={kind!r}) supports simple additive regressors only; "
            "factor() / interaction terms are not yet handled.",
            recovery_hint="Use cluster= (CRV1) or vce='wild' for this model.",
        )

    y = work[y_var].to_numpy(dtype=float)
    Xc = work[cov_vars].to_numpy(dtype=float)
    blocks: List[np.ndarray] = [Xc, np.ones((len(work), 1))]
    for fe in fe_vars:
        d = pd.get_dummies(work[fe], prefix=fe, drop_first=True).astype(float).values
        if d.shape[1]:
            blocks.append(d)
    X = np.column_stack(blocks)

    # Full-dummy CR2/CR3 costs O(p^3) in the bread (p = regressors + FE dummies)
    # plus O(n_g^3) per cluster; guard against high-dimensional FE.
    if X.shape[1] >= len(work) or X.shape[1] > 1000:
        raise MethodIncompatibility(
            f"{who}(vce={kind!r}): the fixed-effects dummy design has "
            f"{X.shape[1]} columns — too many for the full-dummy CR2/CR3 "
            "(which cannot absorb the FE without losing the leverage the "
            "reference needs).",
            recovery_hint="Use cluster= (CRV1) or vce='wild' for "
            "high-dimensional fixed effects.",
        )

    fam = _sm_glm_family("poisson" if is_pois else family_str)
    codes = pd.factorize(work[cluster])[0]
    power = 0.5 if kind == "cr2" else 1.0
    se_full = glm_cr_vcov(X, y, fam, codes, power=power)
    se = pd.Series(
        [float(se_full[cov_pos[i]]) for i in range(len(cov_pos))],
        index=list(base.params.index),
    )

    z = base.params / se
    base.std_errors = se
    base.pvalues = pd.Series(2 * _stats.norm.sf(np.abs(z)), index=base.params.index)
    crit = _stats.norm.ppf(0.975)
    base.conf_int_lower = base.params - crit * se
    base.conf_int_upper = base.params + crit * se
    base.model_info = dict(base.model_info)
    base.model_info["vcov_type"] = {
        "cr2": "CR2 cluster-robust (clubSandwich glm, Pustejovsky-Tipton 2018)",
        "cr3": "CR3 cluster-robust (clubSandwich glm jackknife-type)",
        "jackknife": "CR3 cluster-robust (clubSandwich glm jackknife-type)",
    }[kind]
    base.model_info["cluster"] = cluster
    return base


def _feglm_wild(
    fml: str,
    data: pd.DataFrame,
    *,
    family_str: Optional[str],
    cluster: Optional[str],
    is_pois: bool,
    n_boot: int,
    weight_type: str,
    seed: Optional[int],
    ssc: Optional[Any],
    fixef_rm: str,
    collin_tol: float,
    extra: Dict[str, Any],
) -> EconometricResults:
    """Score wild cluster bootstrap p-values for fepois / feglm.

    Runs the restricted (null-imposed) score wild cluster bootstrap of
    Kline-Santos (2012) — the method Stata ``boottest`` uses after ``poisson`` /
    ``logit`` — on the fixed-effects-as-dummies design with ``boottest``'s
    exact studentization (cluster-share-centered CRVE denominator + strict
    exceedance counting; see
    :func:`statspai.inference.jackknife.glm_score_wild_boot`). In the
    enumerated regime (``2^G <= wild_reps``) the p-values are **bit-exact**
    matches to Stata ``boottest`` (verified for Poisson and logit). Point
    estimates and SE/CI stay the cluster-robust (CRV1) values; only the
    p-values are replaced by the wild-bootstrap p-values.
    """
    from ..inference.jackknife import glm_score_wild_boot

    who = "fepois" if is_pois else "feglm"
    if cluster is None:
        raise MethodIncompatibility(
            f"{who}(vce='wild') requires cluster=... — the wild *cluster* "
            "bootstrap resamples cluster scores.",
            recovery_hint="Pass cluster='firm' (or another id column).",
        )
    _require_frame_columns(data, [], {"cluster": cluster}, f"{who}(vce='wild')")

    if is_pois:
        base = fepois(
            fml,
            data,
            vcov={"CRV1": cluster},
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            **extra,
        )
    else:
        base = feglm(fml, data, family=family_str, vcov={"CRV1": cluster}, **extra)
    if isinstance(base, list):
        raise MethodIncompatibility(
            f"{who}(vce='wild') does not support multiple-estimation formulas."
        )

    lhs, rhs = fml.split("~", 1)
    y_var = lhs.strip()
    fe_vars: List[str] = []
    if "|" in rhs:
        cov_part, fe_part = rhs.split("|", 1)
        fe_vars = [t.strip() for t in fe_part.replace("+", " ").split() if t.strip()]
    else:
        cov_part = rhs
    cov_vars = [
        t.strip() for t in cov_part.split("+") if t.strip() and t.strip() != "1"
    ]

    need = [y_var] + cov_vars + fe_vars + [cluster]
    _require_frame_columns(
        data, [y_var] + cov_vars + fe_vars, {"cluster": cluster}, f"{who}(vce='wild')"
    )
    work = data.loc[:, need].dropna().reset_index(drop=True)
    if int(base.data_info.get("nobs", len(work))) != len(work):
        raise MethodIncompatibility(
            f"{who}(vce='wild'): the fitted sample differs from the complete-case "
            "design (singleton / separation dropping) — cannot align the wild "
            "bootstrap design.",
            recovery_hint="Use cluster= (CRV1) for this model.",
        )
    try:
        [cov_vars.index(nm) for nm in base.params.index]
    except ValueError:
        raise MethodIncompatibility(
            f"{who}(vce='wild') supports simple additive regressors only; "
            "factor() / interaction terms are not yet handled.",
            recovery_hint="Use cluster= (CRV1) for this model.",
        )

    y = work[y_var].to_numpy(dtype=float)
    Xc = work[cov_vars].to_numpy(dtype=float)
    blocks: List[np.ndarray] = [Xc, np.ones((len(work), 1))]
    for fe in fe_vars:
        d = pd.get_dummies(work[fe], prefix=fe, drop_first=True).astype(float).values
        if d.shape[1]:
            blocks.append(d)
    X = np.column_stack(blocks)
    if X.shape[1] >= len(work) or X.shape[1] > 1000:
        raise MethodIncompatibility(
            f"{who}(vce='wild'): the fixed-effects dummy design has "
            f"{X.shape[1]} columns — too many for the dummy-design score "
            "bootstrap (which refits the GLM per coefficient).",
            recovery_hint="Use cluster= (CRV1) for high-dimensional fixed effects.",
        )

    fam = _sm_glm_family("poisson" if is_pois else family_str)
    codes = pd.factorize(work[cluster])[0]
    pvals = {}
    for nm in base.params.index:
        out = glm_score_wild_boot(
            X,
            y,
            fam,
            codes,
            test_idx=cov_vars.index(nm),
            n_boot=n_boot,
            weight_type=weight_type,
            seed=seed,
        )
        pvals[nm] = out["p_boot"]

    base.pvalues = pd.Series(
        [pvals[nm] for nm in base.params.index], index=base.params.index
    )
    base.model_info = dict(base.model_info)
    base.model_info["vcov_type"] = (
        f"score wild cluster bootstrap (Kline-Santos 2012, {weight_type}); "
        "point estimates + SE/CI are cluster-robust CRV1"
    )
    base.model_info["cluster"] = cluster
    base.model_info["n_boot"] = n_boot
    return base


def _feglm_conley(
    fml: str,
    data: pd.DataFrame,
    *,
    family_str: Optional[str],
    is_pois: bool,
    conley_lat: Optional[str],
    conley_lon: Optional[str],
    conley_cutoff: Optional[float],
    ssc: Optional[Any],
    fixef_rm: str,
    collin_tol: float,
    extra: Dict[str, Any],
) -> EconometricResults:
    """Conley spatial-HAC SEs for fepois / feglm (conleyreg convention).

    Score sandwich on the FE-as-dummies design with R ``conleyreg``'s
    spherical (haversine, R=6371.01 km) uniform kernel — the reference
    implementation for GLM Conley (Stata ``acreg`` supports OLS/2SLS only).
    Matches ``conleyreg::conleyreg(model="poisson"/"logit",
    kernel="uniform", dist_comp="spherical")`` to ~1e-7. Guarded against
    high-dimensional FE like the CR2/CR3 path.
    """
    from scipy import stats as _stats

    from ..inference.jackknife import glm_conley_vcov

    who = "fepois" if is_pois else "feglm"
    if conley_lat is None or conley_lon is None or conley_cutoff is None:
        raise MethodIncompatibility(
            f"{who}(vce='conley') requires conley_lat=, conley_lon= and "
            "conley_cutoff= (km).",
            recovery_hint="Pass the coordinate columns and distance cutoff.",
        )
    _require_frame_columns(
        data,
        [],
        {"conley_lat": conley_lat, "conley_lon": conley_lon},
        f"{who}(vce='conley')",
    )

    if is_pois:
        base = fepois(
            fml, data, ssc=ssc, fixef_rm=fixef_rm, collin_tol=collin_tol, **extra
        )
    else:
        base = feglm(fml, data, family=family_str, **extra)
    if isinstance(base, list):
        raise MethodIncompatibility(
            f"{who}(vce='conley') does not support multiple-estimation formulas."
        )

    lhs, rhs = fml.split("~", 1)
    y_var = lhs.strip()
    fe_vars: List[str] = []
    if "|" in rhs:
        cov_part, fe_part = rhs.split("|", 1)
        fe_vars = [t.strip() for t in fe_part.replace("+", " ").split() if t.strip()]
    else:
        cov_part = rhs
    cov_vars = [
        t.strip() for t in cov_part.split("+") if t.strip() and t.strip() != "1"
    ]

    need = [y_var] + cov_vars + fe_vars + [conley_lat, conley_lon]
    _require_frame_columns(
        data,
        [y_var] + cov_vars + fe_vars,
        {"conley_lat": conley_lat, "conley_lon": conley_lon},
        f"{who}(vce='conley')",
    )
    work = data.loc[:, need].dropna().reset_index(drop=True)
    if int(base.data_info.get("nobs", len(work))) != len(work):
        raise MethodIncompatibility(
            f"{who}(vce='conley'): the fitted sample differs from the "
            "complete-case design — cannot align coordinates.",
            recovery_hint="Use cluster= (CRV1) for this model.",
        )
    # column map: covariates first, then the constant (pyfixest names it
    # "Intercept" in the no-FE case)
    colmap = {nm: i for i, nm in enumerate(cov_vars)}
    colmap["Intercept"] = len(cov_vars)
    try:
        cov_pos = [colmap[str(nm)] for nm in base.params.index]
    except KeyError:
        raise MethodIncompatibility(
            f"{who}(vce='conley') supports simple additive regressors only.",
            recovery_hint="Use cluster= (CRV1) for this model.",
        )

    y = work[y_var].to_numpy(dtype=float)
    Xc = work[cov_vars].to_numpy(dtype=float)
    blocks: List[np.ndarray] = [Xc, np.ones((len(work), 1))]
    for fe in fe_vars:
        d = pd.get_dummies(work[fe], prefix=fe, drop_first=True).astype(float).values
        if d.shape[1]:
            blocks.append(d)
    X = np.column_stack(blocks)
    if X.shape[1] >= len(work) or X.shape[1] > 1000:
        raise MethodIncompatibility(
            f"{who}(vce='conley'): the fixed-effects dummy design has "
            f"{X.shape[1]} columns — too many for the reference-matching "
            "spatial HAC (conleyreg's construction is dummy-based).",
            recovery_hint="Use cluster= (CRV1) for high-dimensional " "fixed effects.",
        )

    fam = _sm_glm_family("poisson" if is_pois else family_str)
    se_full = glm_conley_vcov(
        X,
        y,
        fam,
        work[conley_lat].to_numpy(dtype=float),
        work[conley_lon].to_numpy(dtype=float),
        float(conley_cutoff),
    )
    se = pd.Series(
        [float(se_full[cov_pos[i]]) for i in range(len(cov_pos))],
        index=list(base.params.index),
    )

    z = base.params / se
    base.std_errors = se
    base.pvalues = pd.Series(2 * _stats.norm.sf(np.abs(z)), index=base.params.index)
    crit = _stats.norm.ppf(0.975)
    base.conf_int_lower = base.params - crit * se
    base.conf_int_upper = base.params + crit * se
    base.model_info = dict(base.model_info)
    base.model_info["vcov_type"] = (
        f"Conley spatial HAC (conleyreg spherical, {conley_cutoff} km)"
    )
    return base


def _count_kept_singletons(out: Any, fml: Any, data: Any) -> int:
    """Observations in a fixed-effect group of one that a fit kept."""
    first = out[0] if isinstance(out, list) and out else out
    info = getattr(first, "model_info", None) or {}
    if "fe_singleton_obs" in info:
        return int(info["fe_singleton_obs"] or 0)
    if not isinstance(fml, str) or not isinstance(data, pd.DataFrame):
        return 0
    terms = [t.strip() for t in _fe_part(fml).split("+") if t.strip()]
    if not terms or any(t not in data.columns for t in terms):
        return 0  # interacted or sloped effects: not counted
    single = np.zeros(len(data), dtype=bool)
    for t in terms:
        counts = data[t].value_counts()
        single |= data[t].isin(counts.index[counts == 1]).to_numpy()
    return int(single.sum())


def _user_stacklevel() -> int:
    """``stacklevel`` that attributes a warning issued by the caller to the
    first frame outside ``statspai``. Frames named like ``importlib``'s
    bootstrap (the frame-input shim) are skipped by ``warnings`` itself and
    so are not counted."""
    import os
    import sys

    package = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
    level = 1
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_code.co_filename
        internal = "importlib" in name and "_bootstrap" in name
        if not internal and not os.path.abspath(name).startswith(package):
            return level
        if not internal:
            level += 1
        frame = frame.f_back
    return 1


def _singleton_default_notice(entry: str) -> Any:
    """Announce the change of the ``fixef_rm`` default (StatsPAI 1.40).

    Without ``fixef_rm=`` the fit keeps singletons, as before; when it kept
    some, a ``FutureWarning`` says that R ``fixest`` and Stata ``reghdfe``
    drop them by default and that StatsPAI will from 1.40. Passing
    ``fixef_rm`` (either value) silences it.
    """

    def deco(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("fixef_rm") is not None:
                return fn(*args, **kwargs)
            kwargs["fixef_rm"] = "none"
            out = fn(*args, **kwargs)
            fml = args[0] if args else kwargs.get("fml")
            data = args[1] if len(args) > 1 else kwargs.get("data")
            n = _count_kept_singletons(out, fml, data)
            if n:
                warnings.warn(
                    f"{entry}: {n} observation{'s' if n > 1 else ''} sit in "
                    "fixed-effect groups of one (singletons) and were kept. "
                    "R fixest and Stata reghdfe drop them by default, and so "
                    "will StatsPAI from 1.40 (fixef_rm='singleton'). "
                    "Coefficients do not change; the number of observations "
                    "and the degrees of freedom behind the standard errors "
                    "do. Pass fixef_rm='singleton' to adopt the new default "
                    "now, or fixef_rm='none' to keep singletons; either "
                    "silences this.",
                    FutureWarning,
                    stacklevel=_user_stacklevel(),
                )
            return out

        return wrapper

    return deco


@records_call("feols")
@accepts_aliases(vce="vcov")
@records_provenance("sp.feols")
@markout_clusters
@_singleton_default_notice("sp.feols")
def feols(
    fml: str,
    data: pd.DataFrame,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    *,
    weights: Optional[str] = None,
    ssc: Optional[Any] = None,
    fixef_rm: Optional[str] = None,
    collin_tol: float = 1e-6,
    lean: bool = False,
    cluster: Optional[str] = None,
    wild_reps: int = 999,
    wild_weight_type: str = "rademacher",
    seed: Optional[int] = None,
    conley_lat: Optional[str] = None,
    conley_lon: Optional[str] = None,
    conley_cutoff: Optional[float] = None,
    backend: str = "auto",
    weights_as: Optional[str] = None,
    **kwargs: Any,
) -> Union[EconometricResults, List[EconometricResults]]:
    """
    Estimate OLS / IV with high-dimensional fixed effects.

    Uses the Frisch-Waugh-Lovell theorem for fast absorption of
    high-dimensional fixed effects.

    Parameters
    ----------
    fml : str
        A pyfixest-style formula. Examples:

        - ``"Y ~ X1 + X2"`` — plain OLS
        - ``"Y ~ X1 | firm + year"`` — two-way fixed effects
        - ``"Y ~ 1 | firm | X1 ~ Z1"`` — IV with fixed effects
        - ``"Y ~ X1 | csw0(firm, year)"`` — multiple estimations
        - ``"Y ~ X1 | firm + state[year]"`` — varying slopes: state fixed
          effects plus state-specific year trends (``state[[year]]`` for the
          slopes alone). pyfixest does not absorb these, so such formulas run
          on StatsPAI's own HDFE kernel (``model_info['backend'] ==
          'statspai-native'``), matching R ``fixest`` for ``vcov`` iid /
          hetero / ``{'CRV1': col}`` and ``weights=``.

    data : pd.DataFrame
        Input dataset.
    vcov : str or dict, optional
        Variance-covariance estimator (``vce=`` is the canonical alias).

        - ``"iid"`` — classical
        - ``"HC1"``, ``"HC2"``, ``"HC3"`` — heteroskedasticity-robust
        - ``{"CRV1": "firm"}`` — cluster-robust
        - ``{"CRV1": "firm + year"}`` — two-way clustering
        - ``vce="CR2"`` / ``"CR3"`` / ``"jackknife"`` (with ``cluster=``) —
          Pustejovsky-Tipton bias-reduced cluster-robust on the FE-absorbed
          within design; matches R ``clubSandwich::vcovCR(plm)``.
        - ``vce="wild"`` (with ``cluster=``) — WCR wild cluster bootstrap
          (Cameron-Gelbach-Miller 2008); validated against Stata ``boottest``.
        - ``vce="conley"`` (with ``conley_lat=/conley_lon=/conley_cutoff=``)
          — Conley spatial HAC (Stata ``acreg`` planar-distance convention).
        - ``"DK"`` / ``"NW"`` with ``vcov_kwargs={'time_id': 't',
          'panel_id': 'id', 'lag': L}`` — Driscoll-Kraay / Newey-West panel
          HAC through pyfixest (R ``fixest`` ``DK(L)`` / ``NW(L)`` with
          ``panel.id = ~id + t``); without the ids they raise.
    weights : str, optional
        Column name for regression weights. Alone they are analytic
        weights (Stata ``[aw=]``, R ``fixest`` ``weights=``).
    weights_as : {'analytic', 'sampling'}, optional
        What the weights are. ``'analytic'`` is what ``weights=`` alone
        already means. ``'sampling'`` (Stata ``[pw=]``) gives the same
        coefficients with ``vcov='hetero'``, unless ``vcov=``, ``vce=``
        or ``cluster=`` asks for another. Saying which also silences the
        warning about dispersed weights under the classical variance.
        ``'frequency'`` is refused.
    ssc : optional
        Small-sample correction via ``pyfixest.ssc()``.
    fixef_rm : {'none', 'singleton'}, optional
        Singleton fixed-effect groups: ``"none"`` keeps them,
        ``"singleton"`` drops them (R ``fixest`` and Stata ``reghdfe`` do by
        default). Unset, they are kept, with a ``FutureWarning`` when there
        are any: the default becomes ``"singleton"`` in StatsPAI 1.40.
    collin_tol : float, default 1e-6
        Collinearity tolerance.
    lean : bool, default False
        If True, drop large intermediate arrays to save memory.
    cluster : str, optional
        Cluster id column for the extended ``vce=`` menu; also a shorthand
        for one-way ``{"CRV1": cluster}``.
    wild_reps : int, default 999
        Bootstrap replications for ``vce="wild"``.
    wild_weight_type : str, default "rademacher"
        Wild weight distribution (``"rademacher"``, ``"webb"``, ``"mammen"``).
    seed : int, optional
        RNG seed for ``vce="wild"``.
    conley_lat, conley_lon : str, optional
        Coordinate columns (decimal degrees) for ``vce="conley"``.
    conley_cutoff : float, optional
        Conley distance cutoff in km for ``vce="conley"``.
    backend : {'auto', 'pyfixest', 'native'}, default 'auto'
        ``'auto'`` fits on StatsPAI's own engine (``sp.fast.feols``, on
        the Rust kernels when ``statspai-hdfe`` is installed) whenever the
        call is ``y ~ x1 + ... | fe1 + ...`` with numeric regressors and
        ``i(factor[, var], ref=value)`` terms, or the IV form
        ``y ~ x | fe | d1 + ... ~ z1 + ...`` with absorbed effects;
        ``vcov`` in ``iid`` / ``hetero`` / ``{'CRV1': 'a'}`` /
        ``{'CRV1': 'a + b'}``; no ``ssc=`` or extra pyfixest keyword.
        Multiple estimation (``y1 + y2 ~``, ``sw()``, ``csw()``, ...) is
        native when every model is. Other calls go through pyfixest (also
        when regressors are absorbed or collinear, which fixest drops).
        ``model_info['backend']`` says which ran. The native IV first-stage
        F is fixest's ``ivwald1``. The native fit follows fixest's
        conventions: same coefficients, SEs within the
        demeaning tolerance, rows with missing values dropped first.
        ``'pyfixest'`` always delegates. ``'native'`` uses StatsPAI's
        engine for everything it supports, including varying slopes
        (``sp.hdfe_ols``), and refuses the rest.
    **kwargs
        Additional arguments passed to ``pyfixest.feols()``.

    Returns
    -------
    EconometricResults or list of EconometricResults
        Single result for simple formulas, list for multiple estimations
        (e.g. ``csw0``/``sw``/``sw0`` syntax).

    Examples
    --------
    Two-way fixed effects with clustered SEs:

    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> firm = rng.integers(0, 8, n)
    >>> year = rng.integers(0, 5, n)
    >>> x1 = rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x1 + 0.1 * firm + 0.2 * year + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "firm": firm, "year": year})
    >>> res = sp.feols("y ~ x1 | firm + year", data=df, vcov={"CRV1": "firm"})  # doctest: +SKIP
    >>> "x1" in res.params.index  # doctest: +SKIP
    True

    Multiple estimation (``csw0`` returns a list, one fit per FE set):

    >>> results = sp.feols("y ~ x1 | csw0(firm, year)", data=df)  # doctest: +SKIP
    >>> summaries = [r.summary() for r in results]  # doctest: +SKIP

    Use with outreg2 to build a regression table:

    >>> r1 = sp.feols("y ~ x1", data=df)  # doctest: +SKIP
    >>> r2 = sp.feols("y ~ x1 | firm", data=df)  # doctest: +SKIP
    >>> sp.outreg2(r1, r2, filename="table.xlsx")  # doctest: +SKIP
    """
    vcov = stata_vce_to_fixest(vcov, function="feols")
    _require_hac_ids(vcov, kwargs, data, "feols")
    if weights_as is not None:
        # Resolve the declaration once, then run the ordinary call inside
        # it: the fit has several return paths and the weight diagnostic
        # sits at the end of each.
        from ..core._agent_summary import declaring_weights_as, resolve_weights_as

        declared = resolve_weights_as(
            weights_as,
            has_weights=weights is not None,
            native="analytic",
            entry="sp.feols",
        )
        if (
            declared == "sampling"
            and vcov is None
            and cluster is None
            and kwargs.get("vce") is None
        ):
            vcov = "hetero"
        with declaring_weights_as(declared):
            return feols(
                fml,
                data,
                vcov,
                weights=weights,
                ssc=ssc,
                fixef_rm=fixef_rm,
                collin_tol=collin_tol,
                lean=lean,
                cluster=cluster,
                wild_reps=wild_reps,
                wild_weight_type=wild_weight_type,
                seed=seed,
                conley_lat=conley_lat,
                conley_lon=conley_lon,
                conley_cutoff=conley_cutoff,
                backend=backend,
                **kwargs,
            )
    # The native varying-slope parser (sp.hdfe_ols) reads Stata's i.f /
    # c.x / ib2.f / i.f#c.x itself on both sides of '|', so that route gets
    # the formula as written rather than the patsy translation.
    fml_as_written = fml
    fml = r_formula_idioms(fml, fixest_fe=True)
    # numpy integer arithmetic wraps around silently: I(x**2) on an int16
    # column (what pd.read_stata returns for a Stata int) is garbage.
    data = _widen_narrow_integers(data)
    # ``i(rel, ref=[-1, -5])``: several reference levels (fixest's
    # ``ref = c(-1, -5)``); pyfixest takes one, so rewrite and map back.
    if isinstance(fml, str) and isinstance(data, pd.DataFrame):
        fml2, data2, renames = _expand_multi_ref(fml, data)
        if fml2 != fml:
            res = feols(
                fml2,
                data2,
                vcov=vcov,
                weights=weights,
                ssc=ssc,
                fixef_rm=fixef_rm,
                collin_tol=collin_tol,
                lean=lean,
                cluster=cluster,
                wild_reps=wild_reps,
                wild_weight_type=wild_weight_type,
                seed=seed,
                conley_lat=conley_lat,
                conley_lon=conley_lon,
                conley_cutoff=conley_cutoff,
                **kwargs,
            )
            fits = res if isinstance(res, list) else [res]
            for r in fits:
                if renames and not any(str(n) in renames for n in r.params.index):
                    raise MethodIncompatibility(
                        "feols: the i(..., ref=[...]) terms were not estimated; "
                        "check the factor has levels besides the references.",
                    )
                _rename_terms(r, renames)
            return res
    if backend not in ("auto", "pyfixest", "native"):
        raise MethodIncompatibility(
            f"feols: backend must be 'auto', 'pyfixest' or 'native', got {backend!r}."
        )
    if backend != "pyfixest" and not _has_varying_slopes(fml):
        native = _native_feols(
            fml,
            data,
            vcov=vcov,
            cluster=cluster,
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            lean=lean,
            extra=kwargs,
            strict=False,
        )
        if native is not None:
            return native
    if backend == "native" or _has_varying_slopes(fml):
        # Varying slopes always take this path: pyfixest parses fixest's
        # varying-slope syntax but silently drops the slope (see
        # _reject_silent_varying_slopes).
        return _feols_varying_slopes(
            fml_as_written,
            data,
            vcov=vcov,
            cluster=cluster,
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            extra=kwargs,
        )
    if (
        backend == "pyfixest"
        and len(data) >= 1_000_000
        and _native_eligible(fml, vcov, ssc, kwargs, weights)
    ):
        warnings.warn(
            f"feols: {len(data):,} rows through pyfixest; backend='native' fits "
            "this regression on StatsPAI's HDFE kernel with the same "
            "coefficients and SEs (within ~1e-9), typically 10-60x faster.",
            UserWarning,
            stacklevel=3,
        )

    # Wild cluster bootstrap path (Stata ``boottest`` / ``vce()``-style):
    #   sp.feols("y ~ x | firm", data=df, vce="wild", cluster="firm")
    if isinstance(vcov, str) and vcov.lower() in _WILD_VCOV:
        if cluster is None:
            raise MethodIncompatibility(
                "feols(vce='wild') requires cluster=... — the wild *cluster* "
                "bootstrap resamples residuals within clusters."
            )
        _require_frame_columns(data, [], {"cluster": cluster}, "feols(vce='wild')")
        return _feols_wild(
            fml,
            data,
            cluster,
            reps=wild_reps,
            weight_type=wild_weight_type,
            seed=seed,
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            lean=lean,
            extra=kwargs,
        )

    # Bias-reduced / spatial SEs on the FE-absorbed (within) design. feols stores
    # its within-transformed design on the result, so CR2/CR3 (Bell-McCaffrey /
    # jackknife-type) and Conley spatial HAC operate on the partialled-out model.
    # CR2/CR3 match R clubSandwich (plm) to machine precision; Conley matches
    # Stata acreg run on the FE-demeaned data.
    if isinstance(vcov, str) and vcov.lower() in (
        "cr2",
        "cr3",
        "jackknife",
        "conley",
    ):
        return _feols_bias_reduced(
            fml,
            data,
            vcov.lower(),
            cluster=cluster,
            conley_lat=conley_lat,
            conley_lon=conley_lon,
            conley_cutoff=conley_cutoff,
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            lean=lean,
            extra=kwargs,
        )

    # Grammar convenience: `cluster="firm"` is shorthand for one-way CRV1, so the
    # SE keyword reads the same as `sp.regress(..., cluster=...)`. An explicit
    # `vcov=` always wins if both are given.
    if cluster is not None and vcov is None:
        vcov = {"CRV1": cluster}

    pf = _check_pyfixest()

    pf_kwargs: Dict[str, Any] = {
        "fml": fml,
        "data": data,
        "fixef_rm": fixef_rm,
        "collin_tol": collin_tol,
        "lean": lean,
        **kwargs,
    }
    if vcov is not None:
        pf_kwargs["vcov"] = vcov
    if weights is not None:
        pf_kwargs["weights"] = weights
    if ssc is not None:
        pf_kwargs["ssc"] = ssc

    _prefer_compiled_demeaner(pf.feols, pf_kwargs)
    try:
        with _written_fe_order(), _first_stage_demeaner(pf_kwargs["fml"], pf_kwargs):
            fit = pf.feols(**pf_kwargs)
    except _name_lookup_errors() as exc:
        _raise_if_names_missing(pf_kwargs, exc, "sp.feols")
        raise

    # Multiple estimation returns FixestMulti
    if hasattr(fit, "all_fitted_models"):
        return _multi_fit_to_results(fit, vcov=None)

    n_negative = _adjust_multiway_vcov(fit, vcov)
    out = _pyfixest_to_econometric_results(fit)
    _note_cluster_sizes(out, data, vcov)
    _note_weights(out, data, vcov, weights)
    _note_multiway_adjustment(out, vcov, n_negative)
    if weights is None:
        _note_iv_diagnostics(fit, out)
    _check_design(
        out,
        data,
        formula=fml,
        entry="sp.feols",
        cluster=cluster,
        vcov=vcov,
        weights=weights,
    )
    return out


def _note_iv_diagnostics(fit: Any, out: Any) -> None:
    """First-stage F, Sargan and Wu-Hausman statistics of an IV fit.

    The statistics R's fixest reports as ``fitstat(m, ~ ivf1 + sargan +
    wh)``, all under homoskedastic errors, computed on the design pyfixest
    has already projected the fixed effects out of. They are stored under
    the names ``sp.ivreg`` uses, which is where ``sp.estat`` looks.
    """
    if not getattr(fit, "_is_iv", False):
        return
    try:
        names = [str(c) for c in fit._coefnames]
        exog = {str(c) for c in fit._coefnames_z}
        endo = [j for j, c in enumerate(names) if c not in exog]
        first = None
        if len(endo) == 1:
            # pyfixest fits the first stage while it estimates an IV model;
            # refitting it here recomputed the same numbers (bit for bit)
            # at the cost of a second fixed-effects regression.
            if getattr(fit, "_f_stat_1st_stage", None) is None:
                fit.first_stage()
            p1 = getattr(fit, "_p_value_1st_stage", None)
            first = (float(fit._f_stat_1st_stage), None if p1 is None else float(p1))
        _iv_diagnostics_core(
            out,
            Y=np.asarray(fit._Y, dtype=float).ravel(),
            X=np.asarray(fit._X, dtype=float),
            Z=np.asarray(fit._Z, dtype=float),
            u=np.asarray(fit._u_hat, dtype=float).ravel(),
            names=names,
            endo=endo,
            k_total=int(getattr(fit, "_df_k", len(names)) or len(names)),
            first=first,
        )
    except (
        np.linalg.LinAlgError,
        ValueError,
        TypeError,
        AttributeError,
        ZeroDivisionError,
    ) as exc:  # diagnostics must not cost the user the fit
        from ..workflow._degradation import record_degradation

        record_degradation(
            out, section="feols_iv_diagnostics", exc=exc, detail="IV diagnostics"
        )


def _iv_diagnostics_core(
    out: Any,
    *,
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    u: np.ndarray,
    names: List[str],
    endo: List[int],
    k_total: int,
    first: Optional[Tuple[float, Optional[float]]],
) -> None:
    """First-stage F, Sargan and Wu-Hausman on the residualised design.

    ``Y`` / ``X`` / ``Z`` / ``u`` are the outcome, second-stage regressors,
    instruments (exogenous regressors included) and residuals after the
    fixed effects are projected out (and scaled by the square root of the
    weights); ``first`` is the first-stage F and its p-value.
    """
    from scipy import stats

    n = len(Y)
    if not endo or X.shape[0] != n or Z.shape[0] != n:
        return
    diag = out.diagnostics
    info = dict(out.model_info)
    if first is not None:
        f1, p1 = first
        diag[f"First-stage F ({names[endo[0]]})"] = f1
        info["first_stage_f"] = f1
        if p1 is not None:
            diag[f"First-stage F p-value ({names[endo[0]]})"] = float(p1)
            info["first_stage_f_pvalue"] = float(p1)
    df_over = Z.shape[1] - X.shape[1]
    if df_over > 0:
        fitted = Z @ np.linalg.lstsq(Z, u, rcond=None)[0]
        sargan = n * float(u @ fitted) / float(u @ u)
        diag["Sargan statistic"] = sargan
        diag["Sargan p-value"] = float(stats.chi2.sf(sargan, df_over))
        diag["Sargan df"] = int(df_over)
    E = X[:, endo]
    V = E - Z @ np.linalg.lstsq(Z, E, rcond=None)[0]
    XA = np.column_stack([X, V])
    rss1 = float(np.sum((Y - XA @ np.linalg.lstsq(XA, Y, rcond=None)[0]) ** 2))
    rss0 = float(np.sum((Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]) ** 2))
    q = len(endo)
    df2 = n - max(k_total, X.shape[1]) - q
    if df2 > 0 and rss1 > 0:
        wh = ((rss0 - rss1) / q) / (rss1 / df2)
        diag["Hausman F-stat"] = float(wh)
        diag["Hausman p-value"] = float(stats.f.sf(wh, q, df2))
    out.model_info = info


def _adjust_multiway_vcov(fit: Any, vcov: Any) -> int:
    """Make a multiway clustered covariance positive semi-definite.

    The inclusion-exclusion estimator can have negative eigenvalues when
    one clustering dimension is small. R's fixest then sets them to zero
    (``vcov_fix = TRUE``, its default, after Cameron, Gelbach and Miller
    2011); pyfixest reports the matrix as computed, with a missing standard
    error where a variance is negative. The adjustment is applied to the
    fit, pyfixest recomputes its inference from the adjusted matrix, and
    the number of eigenvalues that were negative is returned.
    """
    if not isinstance(vcov, dict):
        return 0
    spec = str(next(iter(vcov.values()), ""))
    V = getattr(fit, "_vcov", None)
    if "+" not in spec or V is None or not hasattr(fit, "get_inference"):
        return 0
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or not np.all(np.isfinite(V)):
        return 0
    from ..inference.jackknife import psd_adjust

    fixed, n_negative = psd_adjust(V)
    if n_negative:
        fit._vcov = fixed
        fit.get_inference()
    return n_negative


def _note_multiway_adjustment(out: Any, vcov: Any, n_negative: int) -> None:
    if not n_negative:
        return
    spec = str(next(iter(vcov.values()), ""))
    if isinstance(getattr(out, "diagnostics", None), dict):
        out.diagnostics["Multiway VCOV negative eigenvalues"] = n_negative
    warnings.warn(
        f"feols: the covariance clustered on {spec!r} was not positive "
        f"semi-definite ({n_negative} negative eigenvalue"
        f"{'s' if n_negative > 1 else ''}); they were set to zero, the "
        "adjustment of Cameron, Gelbach and Miller (2011) that R's fixest "
        "applies. This happens when one clustering dimension has few "
        "clusters relative to the number of coefficients; treat joint "
        "tests with care.",
        RuntimeWarning,
        stacklevel=3,
    )


# --------------------------------------------------------------------------- #
#  fepois — Poisson regression with high-dimensional fixed effects
# --------------------------------------------------------------------------- #


@records_call("fepois")
@accepts_aliases(vce="vcov")
@records_provenance("sp.fepois")
@markout_clusters
@_singleton_default_notice("sp.fepois")
def fepois(
    fml: str,
    data: pd.DataFrame,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    *,
    weights: Optional[str] = None,
    ssc: Optional[Any] = None,
    fixef_rm: Optional[str] = None,
    collin_tol: float = 1e-6,
    iwls_tol: float = 1e-8,
    iwls_maxiter: int = 25,
    cluster: Optional[str] = None,
    wild_reps: int = 9999,
    wild_weight_type: str = "rademacher",
    seed: Optional[int] = None,
    conley_lat: Optional[str] = None,
    conley_lon: Optional[str] = None,
    conley_cutoff: Optional[float] = None,
    backend: str = "auto",
    **kwargs: Any,
) -> Union[EconometricResults, List[EconometricResults]]:
    """
    Estimate Poisson regression with high-dimensional fixed effects.

    Parameters
    ----------
    fml : str
        pyfixest formula. E.g. ``"Y ~ X1 | firm"``.
    data : pd.DataFrame
        Input dataset.
    vcov : str or dict, optional
        Variance-covariance estimator (``vce=`` is the canonical alias).
        Besides the pyfixest values (``"iid"``, ``"HC1"``,
        ``{"CRV1": "firm"}``, ...), accepts the extended menu:

        - ``vce="CR2"`` / ``"CR3"`` / ``"jackknife"`` (with ``cluster=``) —
          clubSandwich glm bias-reduced cluster-robust SEs on the
          FE-as-dummies design; matches R ``clubSandwich::vcovCR(glm)``.
        - ``vce="wild"`` (with ``cluster=``) — restricted score wild cluster
          bootstrap (Kline-Santos 2012) with Stata ``boottest``'s exact
          studentization; bit-exact vs ``boottest`` in the enumerated regime.
    weights : str, optional
        Column name for regression weights (not supported with the extended
        ``vce=`` menu).
    ssc : optional
        Small-sample correction.
    fixef_rm : {'none', 'singleton'}, optional
        Singleton fixed-effect groups: ``"none"`` keeps them,
        ``"singleton"`` drops them (R ``fixest`` does by default). Unset,
        they are kept, with a ``FutureWarning`` when there are any: the
        default becomes ``"singleton"`` in StatsPAI 1.40.
    collin_tol : float, default 1e-6
        Collinearity tolerance.
    iwls_tol : float, default 1e-8
        IWLS convergence tolerance.
    iwls_maxiter : int, default 25
        Max IWLS iterations.
    cluster : str, optional
        Cluster id column for the extended ``vce=`` menu; also a shorthand
        for one-way ``{"CRV1": cluster}``.
    wild_reps : int, default 9999
        Replications for ``vce="wild"``. When ``2**G <= wild_reps`` the full
        Rademacher grid is enumerated (deterministic).
    wild_weight_type : str, default "rademacher"
        Wild weight distribution (``"rademacher"`` or ``"webb"``).
    seed : int, optional
        RNG seed for sampled (non-enumerated) ``vce="wild"`` draws.
    backend : {'auto', 'pyfixest', 'native'}, default 'auto'
        ``'auto'`` fits on StatsPAI's own PPML-HDFE engine
        (``sp.fast.fepois``, on the Rust kernels when ``statspai-hdfe`` is
        installed) whenever the call is ``y ~ x1 + ... | fe1 + ...`` with
        numeric regressors, ``vcov`` in ``iid`` / ``hetero`` /
        ``{'CRV1': 'a'}`` / ``{'CRV1': 'a + b'}`` and no ``ssc=`` or extra
        pyfixest keyword, and through pyfixest otherwise. Both follow
        fixest: observations in a fixed-effect group whose outcomes are
        all zero are dropped, rows with missing values are dropped,
        singletons are kept unless ``fixef_rm='singleton'``.
        ``model_info['backend']`` says which
        ran. ``'pyfixest'`` always delegates; ``'native'`` refuses calls
        the native engine cannot fit.
    **kwargs
        Additional arguments passed to ``pyfixest.fepois()``.

    Returns
    -------
    EconometricResults or list of EconometricResults

    Examples
    --------
    Poisson regression (PPML-style) with firm fixed effects:

    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> firm = rng.integers(0, 8, n)
    >>> x1 = rng.normal(size=n)
    >>> mu = np.exp(0.3 + 0.5 * x1 + 0.1 * firm)
    >>> df = pd.DataFrame({"y": rng.poisson(mu), "x1": x1, "firm": firm})
    >>> res = sp.fepois("y ~ x1 | firm", data=df)  # doctest: +SKIP
    >>> "x1" in res.params.index  # doctest: +SKIP
    True

    Bias-reduced cluster-robust SEs (matches R ``clubSandwich::vcovCR``):

    >>> res = sp.fepois("y ~ x1 | firm", data=df, vce="CR2",  # doctest: +SKIP
    ...                 cluster="firm")
    """
    vcov = stata_vce_to_fixest(vcov, function="fepois")
    _require_hac_ids(vcov, kwargs, data, "fepois")
    # numpy integer arithmetic wraps around silently: I(x**2) on an int16
    # column (what pd.read_stata returns for a Stata int) is garbage.
    fml = r_formula_idioms(fml, fixest_fe=True)
    data = _widen_narrow_integers(data)
    _reject_silent_varying_slopes(fml)
    if backend not in ("auto", "pyfixest", "native"):
        raise MethodIncompatibility(
            f"fepois: backend must be 'auto', 'pyfixest' or 'native', got {backend!r}."
        )
    if backend != "pyfixest":
        native = _native_fepois(
            fml,
            data,
            vcov=vcov,
            cluster=cluster,
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            iwls_tol=iwls_tol,
            iwls_maxiter=iwls_maxiter,
            extra=kwargs,
            strict=backend == "native",
        )
        if native is not None:
            return native

    # Conley spatial HAC (conleyreg spherical convention).
    if isinstance(vcov, str) and vcov.lower() == "conley":
        if weights is not None:
            raise MethodIncompatibility(
                "fepois(vce='conley') does not support weights=.",
                recovery_hint="Drop weights= or use cluster= (CRV1).",
            )
        return _feglm_conley(
            fml,
            data,
            family_str="poisson",
            is_pois=True,
            conley_lat=conley_lat,
            conley_lon=conley_lon,
            conley_cutoff=conley_cutoff,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            extra=kwargs,
        )

    # Score wild cluster bootstrap p-values (Kline-Santos 2012, boottest-style).
    if isinstance(vcov, str) and vcov.lower() in (_WILD_VCOV | _GLM_BR_VCOV):
        if weights is not None:
            raise MethodIncompatibility(
                f"fepois(vce={vcov!r}) does not support weights= — the "
                "extended SE menu (wild / CR2 / CR3) is unweighted.",
                recovery_hint="Drop weights= or use cluster= (CRV1).",
            )
    if isinstance(vcov, str) and vcov.lower() in _WILD_VCOV:
        return _feglm_wild(
            fml,
            data,
            family_str="poisson",
            cluster=cluster,
            is_pois=True,
            n_boot=wild_reps,
            weight_type=wild_weight_type,
            seed=seed,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            extra=kwargs,
        )

    # Bias-reduced cluster SEs (CR2 / CR3 / jackknife) via the clubSandwich glm
    # adjustment on the FE-as-dummies design — matches R clubSandwich exactly.
    if isinstance(vcov, str) and vcov.lower() in _GLM_BR_VCOV:
        return _feglm_bias_reduced(
            fml,
            data,
            vcov.lower(),
            family_str="poisson",
            cluster=cluster,
            is_pois=True,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            extra=kwargs,
        )

    # Grammar convenience: cluster="firm" is one-way CRV1 (reads like sp.regress).
    if cluster is not None and vcov is None:
        vcov = {"CRV1": cluster}

    pf = _check_pyfixest()

    pf_kwargs: Dict[str, Any] = {
        "fml": fml,
        "data": data,
        "fixef_rm": fixef_rm,
        "collin_tol": collin_tol,
        "iwls_tol": iwls_tol,
        "iwls_maxiter": iwls_maxiter,
        **kwargs,
    }
    if vcov is not None:
        pf_kwargs["vcov"] = vcov
    if weights is not None:
        pf_kwargs["weights"] = weights
    if ssc is not None:
        pf_kwargs["ssc"] = ssc

    try:
        _prefer_compiled_demeaner(pf.fepois, pf_kwargs)
        with _written_fe_order():
            fit = pf.fepois(**pf_kwargs)
    except _name_lookup_errors() as exc:
        _raise_if_names_missing(pf_kwargs, exc, "sp.fepois")
        if not isinstance(exc, ValueError) or "Demeaning failed" not in str(exc):
            raise
        # pyfixest's alternating projections give up on some crossed
        # multi-way designs (five crossed effects on 10^5 rows in a
        # replication) that sp.ppmlhdfe solves; say so instead of leaving
        # the user with a backend message.
        raise NumericalInstability(
            f"sp.fepois: the pyfixest backend could not absorb the fixed "
            f"effects ({exc}).",
            recovery_hint=(
                "Use sp.ppmlhdfe(formula, data=..., absorb=..., cluster=...), "
                "the native PPML-HDFE solver with Stata ppmlhdfe's defaults "
                "(singleton and separation rules); its point estimates and "
                "clustered SEs match ppmlhdfe."
            ),
            diagnostics={"backend": "pyfixest", "formula": fml},
            alternative_functions=["sp.ppmlhdfe"],
        ) from exc

    if hasattr(fit, "all_fitted_models"):
        return _multi_fit_to_results(fit, vcov=None)

    return _pyfixest_to_econometric_results(fit)


# --------------------------------------------------------------------------- #
#  feglm — GLM with high-dimensional fixed effects
# --------------------------------------------------------------------------- #


@records_call("feglm")
@accepts_aliases(vce="vcov")
@records_provenance("sp.feglm")
@markout_clusters
def feglm(
    fml: str,
    data: pd.DataFrame,
    family: str = "gaussian",
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    *,
    cluster: Optional[str] = None,
    wild_reps: int = 9999,
    wild_weight_type: str = "rademacher",
    seed: Optional[int] = None,
    conley_lat: Optional[str] = None,
    conley_lon: Optional[str] = None,
    conley_cutoff: Optional[float] = None,
    backend: str = "auto",
    bias_correction: bool = False,
    **kwargs: Any,
) -> Union[EconometricResults, List[EconometricResults]]:
    """
    Estimate GLM (logit, probit, negative binomial, Gaussian) with
    high-dimensional fixed effects.

    Parameters
    ----------
    fml : str
        pyfixest formula.
    data : pd.DataFrame
        Input dataset.
    family : str, default "gaussian"
        GLM family: ``"gaussian"``, ``"logit"``, ``"probit"``, or
        ``"negbin"``: the negative binomial (NB2, variance
        ``mu + mu^2 / theta``) with the fixed effects estimated as
        parameters, as ``fixest::fenegbin`` fits it (not Stata's conditional
        ``xtnbreg, fe``). ``theta`` is estimated jointly; it and its
        standard error are in ``model_info['theta']`` /
        ``model_info['theta_se']`` (fixest lists it as the coefficient
        ``.theta``). Rows its effects fit perfectly (groups with only zero
        outcomes, singletons) are dropped, as fixest does; ``weights=`` are
        likelihood weights. Native engine only: the ``vce=`` options that
        go through pyfixest are refused for it.
    vcov : str or dict, optional
        Variance-covariance estimator (``vce=`` is the canonical alias). Also
        accepts ``vce="CR2"``/``"CR3"``/``"jackknife"`` (with ``cluster=``)
        for the clubSandwich bias-reduced cluster-robust SEs, and
        ``vce="wild"`` (with ``cluster=``) for the restricted score wild
        cluster bootstrap (Kline-Santos 2012; bit-exact vs Stata ``boottest``
        in the enumerated regime).
    cluster : str, optional
        Cluster id column for the extended ``vce=`` menu (also a shorthand
        for one-way ``{"CRV1": cluster}``).
    wild_reps : int, default 9999
        Replications for ``vce="wild"``. When ``2**G <= wild_reps`` the full
        Rademacher grid is enumerated (deterministic).
    wild_weight_type : str, default "rademacher"
        Wild weight distribution (``"rademacher"`` or ``"webb"``).
    seed : int, optional
        RNG seed for sampled (non-enumerated) ``vce="wild"`` draws.
    backend : {'auto', 'pyfixest', 'native'}, default 'auto'
        ``'auto'`` fits ``family='logit'`` / ``'probit'`` on StatsPAI's own
        IRLS engine whenever the call is ``y ~ x1 + ... | fe1 + ...`` with
        numeric regressors, ``vcov`` in ``iid`` / ``hetero`` /
        ``{'CRV1': 'a'}`` / ``{'CRV1': 'a + b'}`` and at most ``weights=``
        among the extra keywords, and through pyfixest otherwise (``family='gaussian'``
        always). The native fit follows ``fixest::feglm``: observations in
        a fixed-effect group whose outcomes are all 0 or all 1 are dropped,
        p-values are normal, and IRLS runs to a relative deviance change of
        1e-10. ``'pyfixest'`` always delegates; ``'native'`` refuses calls
        the native engine cannot fit.
    bias_correction : bool, default False
        With ``family='logit'`` / ``'probit'`` and unit (and period)
        effects, correct the slopes for the incidental-parameter bias
        (order ``1/T`` from unit effects, ``1/N`` from period effects) by
        the analytical correction for strictly exogenous regressors, as
        R ``alpaca::biasCorr(L = 0)`` does, and report the variance at
        the corrected slopes without small-sample factors (alpaca's
        convention): ``vcov`` ``'iid'``, ``'hetero'`` or one-way
        ``{'CRV1': c}``. The uncorrected slopes and the shift are in
        ``model_info['coef_uncorrected']`` / ``['bias_shift']``.
        Unweighted fits with one or two absorbed effects; native engine.
        The correction assumes regressors uncorrelated with past and
        future errors (no lagged outcomes). Bib key:
        ``fernandezval2016individual``.
    **kwargs
        Additional arguments passed to ``pyfixest.feglm()``.

    Returns
    -------
    EconometricResults or list of EconometricResults

    Examples
    --------
    Logit GLM with firm fixed effects:

    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> firm = rng.integers(0, 8, n)
    >>> x1 = rng.normal(size=n)
    >>> p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x1)))
    >>> y = (rng.random(n) < p).astype(int)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "firm": firm})
    >>> res = sp.feglm("y ~ x1 | firm", data=df, family="logit")  # doctest: +SKIP
    >>> "x1" in res.params.index  # doctest: +SKIP
    True
    """
    vcov = stata_vce_to_fixest(vcov, function="feglm")
    _require_hac_ids(vcov, kwargs, data, "feglm")
    # numpy integer arithmetic wraps around silently: I(x**2) on an int16
    # column (what pd.read_stata returns for a Stata int) is garbage.
    fml = r_formula_idioms(fml, fixest_fe=True)
    data = _widen_narrow_integers(data)
    _reject_silent_varying_slopes(fml)
    if backend not in ("auto", "pyfixest", "native"):
        raise MethodIncompatibility(
            f"feglm: backend must be 'auto', 'pyfixest' or 'native', got {backend!r}."
        )
    if bias_correction and backend == "pyfixest":
        raise MethodIncompatibility(
            "feglm(bias_correction=True) runs on StatsPAI's engine.",
            recovery_hint="Drop backend='pyfixest'.",
        )
    if backend == "pyfixest" and str(family).lower() == "negbin":
        raise MethodIncompatibility(
            "feglm(family='negbin') runs on StatsPAI's engine; pyfixest has "
            "no negative binomial.",
            recovery_hint="Drop backend='pyfixest'.",
        )
    if backend != "pyfixest":
        native = _native_feglm(
            fml,
            data,
            family=family,
            vcov=vcov,
            cluster=cluster,
            extra=kwargs,
            strict=backend == "native",
            bias_correction=bool(bias_correction),
        )
        if native is not None:
            return native

    # Conley spatial HAC (conleyreg spherical convention).
    if isinstance(vcov, str) and vcov.lower() == "conley":
        return _feglm_conley(
            fml,
            data,
            family_str=family,
            is_pois=False,
            conley_lat=conley_lat,
            conley_lon=conley_lon,
            conley_cutoff=conley_cutoff,
            ssc=None,
            fixef_rm="none",
            collin_tol=1e-6,
            extra=kwargs,
        )

    # Score wild cluster bootstrap p-values (Kline-Santos 2012, boottest-style).
    if isinstance(vcov, str) and vcov.lower() in _WILD_VCOV:
        return _feglm_wild(
            fml,
            data,
            family_str=family,
            cluster=cluster,
            is_pois=False,
            n_boot=wild_reps,
            weight_type=wild_weight_type,
            seed=seed,
            ssc=None,
            fixef_rm="none",
            collin_tol=1e-6,
            extra=kwargs,
        )

    # Bias-reduced cluster SEs (CR2 / CR3 / jackknife) via the clubSandwich glm
    # adjustment on the FE-as-dummies design — matches R clubSandwich exactly.
    if isinstance(vcov, str) and vcov.lower() in _GLM_BR_VCOV:
        return _feglm_bias_reduced(
            fml,
            data,
            vcov.lower(),
            family_str=family,
            cluster=cluster,
            is_pois=False,
            ssc=None,
            fixef_rm="none",
            collin_tol=1e-6,
            extra=kwargs,
        )

    if cluster is not None and vcov is None:
        vcov = {"CRV1": cluster}

    pf = _check_pyfixest()

    pf_kwargs: Dict[str, Any] = {"fml": fml, "data": data, "family": family, **kwargs}
    if vcov is not None:
        pf_kwargs["vcov"] = vcov

    _prefer_compiled_demeaner(pf.feglm, pf_kwargs)
    try:
        with _written_fe_order():
            fit = pf.feglm(**pf_kwargs)
    except _name_lookup_errors() as exc:
        _raise_if_names_missing(pf_kwargs, exc, "sp.feglm")
        raise

    if hasattr(fit, "all_fitted_models"):
        return _multi_fit_to_results(fit, vcov=None)

    return _pyfixest_to_econometric_results(fit)


# --------------------------------------------------------------------------- #
#  etable — convenience re-export
# --------------------------------------------------------------------------- #


def etable(
    *results: EconometricResults,
    **kwargs: Any,
) -> Any:
    """
    Display a pyfixest-style regression table for StatsPAI results.

    If every result carries a ``_pyfixest_fit`` reference, uses pyfixest's
    native ``etable``. Otherwise returns one row per term with
    ``"coef*** (se)"`` cells, formatted by :func:`sp.regtable` so both tables
    report identical numbers.

    Parameters
    ----------
    *results : EconometricResults
        One or more fitted results.
    **kwargs
        Passed to ``pyfixest.etable()``.

    Returns
    -------
    str or DataFrame

    Examples
    --------
    Side-by-side comparison of nested specifications:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 200
    >>> df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
    ...                    "firm": rng.integers(0, 10, n)})
    >>> df["wage"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(0, 0.5, n)
    >>> r1 = sp.regress("wage ~ x1", data=df)
    >>> r2 = sp.regress("wage ~ x1 + x2", data=df)
    >>> tab = sp.etable(r1, r2)  # DataFrame: one coefficient column per model
    >>> list(tab.columns)
    ['(1)', '(2)']

    With ``sp.feols`` fits (requires the ``fixest`` extra), pyfixest's
    native styled table is returned instead:

    >>> f1 = sp.feols("wage ~ x1 | firm", data=df)  # doctest: +SKIP
    >>> f2 = sp.feols("wage ~ x1 + x2 | firm", data=df)  # doctest: +SKIP
    >>> tab = sp.etable(f1, f2)  # doctest: +SKIP

    A list works too: ``sp.etable([r1, r2])``.
    """
    # ``sp.etable([r1, r2])`` used to be read as a single "result" with no
    # ``params`` and returned an empty DataFrame; so did any non-result
    # argument. Both are caller errors that must be visible.
    if len(results) == 1 and isinstance(results[0], (list, tuple)):
        results = tuple(results[0])
    if not results:
        raise TypeError("etable() needs at least one fitted result.")
    not_results = [
        (i + 1, type(r).__name__)
        for i, r in enumerate(results)
        if getattr(r, "params", None) is None and not hasattr(r, "_pyfixest_fit")
    ]
    if not_results:
        # TypeError is the released contract (1.29.0); the message names each
        # offending position and says how to recover.
        which = ", ".join(f"argument {i} ({name})" for i, name in not_results)
        raise TypeError(
            "etable() expects fitted results (e.g. from sp.regress, sp.feols, "
            f"sp.logit); {which} has no coefficients. Recovery: pass only "
            "fitted model results to sp.etable."
        )

    pf_fits = [
        getattr(r, "_pyfixest_fit") for r in results if hasattr(r, "_pyfixest_fit")
    ]

    # pyfixest's styled table only when *every* result is a pyfixest fit; a
    # mixed call used to drop the non-pyfixest columns without a word.
    if pf_fits and len(pf_fits) == len(results):
        pf = _check_pyfixest()
        return pf.etable(pf_fits, **kwargs)

    # Non-pyfixest results: sp.regtable's cells folded into pyfixest's compact
    # "coef (se)" layout, one row per term. One formatter for every table
    # entry point, so stars, rounding and term extraction cannot drift apart.
    from ..output.regression_table import regtable

    table = regtable(*results, **{k: kwargs[k] for k in ("fmt",) if k in kwargs})
    terms = table._resolve_vars(table._all_models_flat())
    frame = table.to_dataframe().iloc[: 2 * len(terms)]
    rows: Dict[str, Dict[str, str]] = {}
    for k, term in enumerate(terms):
        coef, se = frame.iloc[2 * k], frame.iloc[2 * k + 1]
        rows[str(term)] = {
            col: f"{coef[col]} {se[col]}".strip() if coef[col] else ""
            for col in frame.columns
        }
    return pd.DataFrame(rows).T
