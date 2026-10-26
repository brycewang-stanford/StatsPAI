"""Counterfactual design matrices for response-scale postestimation.

``sp.margins`` needs ``X(frame)`` -- the model's design evaluated on an
arbitrary (counterfactual) covariate frame -- and, for continuous variables,
its derivative.  Two backends produce it:

``_TermDesign``
    Parses the coefficient names themselves (plain columns, the intercept,
    treatment-coded ``C(g)[T.level]`` dummies and ``a:b`` interactions).
    Derivatives are analytic.  Used whenever every term is of that form, so
    results for such models are unchanged from the pre-1.32 implementation.

``_FormulaDesign``
    Rebuilds the design with patsy from the fitted formula, so ``I(x**2)``,
    ``np.log(x)``, ``center(x)`` and factor codings are honoured.  The design
    is differentiated numerically (central differences on the *raw* variable),
    which is the chain rule through every transform.  The formula is the
    picklable recipe stored at fit time (``model_info['formula']`` or the
    provenance record); patsy ``DesignInfo`` itself cannot be pickled.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import ColumnNotFound, MethodIncompatibility

_INTERCEPT_TOKENS = ("Intercept", "const", "_cons")

# Treatment-coded dummy, e.g. ``C(group)[T.2]`` or ``group[T.b]``.
CAT_TERM_RE = re.compile(r"^(?:C\(\s*)?([A-Za-z_]\w*)\s*(?:,[^)]*)?\)?\[T\.(.+)\]$")
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


def factor_value(level: str, obs_val: Any) -> float:
    """1.0 if ``obs_val`` is the dummy's ``level``."""
    try:
        return 1.0 if float(obs_val) == float(level) else 0.0
    except (TypeError, ValueError):
        return 1.0 if str(obs_val) == str(level) else 0.0


def model_formula(result: Any) -> Optional[str]:
    """The formula a result was fitted from, if it recorded one."""
    mi = getattr(result, "model_info", None) or {}
    f = mi.get("formula")
    if isinstance(f, str) and "~" in f:
        return f
    prov = getattr(result, "_provenance", None)
    params = getattr(prov, "params", None) or {}
    f = params.get("formula") if isinstance(params, dict) else None
    if isinstance(f, str) and "~" in f:
        return f
    return None


def model_setting(result: Any, key: str) -> Any:
    """``offset`` / ``exposure`` / ``weights`` as recorded at fit time.

    Estimators disagree on where they put these (``model_info`` for GLM /
    logit, ``data_info`` for ``sp.poisson``, the provenance record for
    ``sp.regress``); reading only ``model_info`` silently dropped a Poisson
    exposure from the prediction (AME off by the mean exposure).
    """
    for holder in ("model_info", "data_info"):
        d = getattr(result, holder, None) or {}
        if d.get(key) is not None:
            return d[key]
    prov = getattr(result, "_provenance", None)
    params = getattr(prov, "params", None) or {}
    if isinstance(params, dict) and params.get(key) is not None:
        return params[key]
    return None


def _simple_part(part: str) -> bool:
    return (
        part in _INTERCEPT_TOKENS
        or CAT_TERM_RE.match(part) is not None
        or _IDENT_RE.match(part) is not None
    )


_NEEDS_DATA_HINT = (
    "A formula with factor, transformed or interaction terms is rebuilt from "
    "the original columns: pass data=<the frame the model was fitted on>."
)


class _TermDesign:
    """Design from coefficient names (plain / C()-dummy / interaction terms)."""

    backend = "terms"

    def __init__(self, terms: Sequence[str], frame: pd.DataFrame) -> None:
        self.terms = [str(t) for t in terms]
        variables: List[str] = []
        levels: Dict[str, List[str]] = {}
        for term in self.terms:
            for part in term.split(":"):
                if part in _INTERCEPT_TOKENS:
                    continue
                m = CAT_TERM_RE.match(part)
                name = m.group(1) if m is not None else part
                if m is not None:
                    lv = levels.setdefault(name, [])
                    if m.group(2) not in lv:
                        lv.append(m.group(2))
                if name not in variables:
                    variables.append(name)
        self.variables = variables
        self._dummy_levels = levels
        self.factors: Dict[str, Tuple[List[Any], Any]] = {}
        for var, dummies in levels.items():
            if var not in frame.columns:
                continue  # raw column absent (data=None): not usable as factor
            observed = pd.unique(frame[var].dropna())
            base = [v for v in observed if not any(factor_value(d, v) for d in dummies)]
            coded = [
                next((v for v in observed if factor_value(d, v)), d) for d in dummies
            ]
            if len(base) == 1:
                self.factors[var] = (list(base) + coded, base[0])

    def _part(self, part: str, frame: pd.DataFrame) -> np.ndarray:
        n = len(frame)
        if part in _INTERCEPT_TOKENS:
            return np.ones(n)
        m = CAT_TERM_RE.match(part)
        if m is not None:
            base, level = m.group(1), m.group(2)
            if base not in frame.columns:
                raise ColumnNotFound.from_missing(
                    [base],
                    list(frame.columns),
                    message=f"margins: {base!r} is not in the data.",
                    extra_hint=_NEEDS_DATA_HINT,
                )
            return np.array([factor_value(level, v) for v in frame[base]], dtype=float)
        if part not in frame.columns:
            raise ColumnNotFound.from_missing(
                [part],
                list(frame.columns),
                message=f"margins: {part!r} is not in the data.",
                extra_hint=_NEEDS_DATA_HINT,
            )
        out: np.ndarray = frame[part].to_numpy(dtype=float)
        return out

    def build(self, frame: pd.DataFrame) -> np.ndarray:
        cols = []
        for term in self.terms:
            v = np.ones(len(frame))
            for part in term.split(":"):
                v = v * self._part(part, frame)
            cols.append(v)
        return np.column_stack(cols)

    def derivative(self, frame: pd.DataFrame, var: str, eps: float) -> np.ndarray:
        """``d X / d var`` (product rule over ``a:b``), analytic."""
        cols = []
        for term in self.terms:
            parts = term.split(":")
            deriv = np.zeros(len(frame))
            for i, part in enumerate(parts):
                if part != var:
                    continue
                others: np.ndarray = np.ones(len(frame))
                for j, other in enumerate(parts):
                    if j != i:
                        others = others * self._part(other, frame)
                deriv = deriv + others
            cols.append(deriv)
        return np.column_stack(cols)

    def is_factor(self, var: str) -> bool:
        return var in self._dummy_levels


class _FormulaDesign:
    """Design rebuilt by patsy from the fitted formula."""

    backend = "formula"

    def __init__(self, formula: str, frame: pd.DataFrame, names: Sequence[str]):
        import patsy

        from ..core.utils import (
            _coerce_string_extension_dtypes,
            formula_eval_env,
            r_formula_idioms,
            raise_if_formula_names_missing,
        )

        rhs = r_formula_idioms(formula.split("~", 1)[1].strip())
        frame = _coerce_string_extension_dtypes(frame, rhs)
        try:
            di = patsy.dmatrix(
                rhs, frame, eval_env=formula_eval_env(), NA_action="drop"
            ).design_info
        except Exception as exc:  # patsy raises many types
            raise_if_formula_names_missing(rhs, frame, exc, context="margins")
            raise MethodIncompatibility(
                f"margins: could not rebuild the design of {formula!r} on the "
                f"supplied data ({type(exc).__name__}: {exc}).",
                recovery_hint="Pass the estimation data (all formula variables).",
            ) from exc

        def _canon(c: str) -> str:
            return "Intercept" if c in _INTERCEPT_TOKENS else c

        if [_canon(c) for c in di.column_names] != [_canon(str(n)) for n in names]:
            raise MethodIncompatibility(
                "margins: rebuilding the formula on this data gives design "
                f"columns {list(di.column_names)}, but the model has "
                f"{list(names)} (typically a factor level absent from, or "
                "added to, the data passed to margins).",
                recovery_hint="Pass the estimation sample as data=.",
            )
        self.design_info = di
        self._columns = set(frame.columns)
        numeric_vars: List[str] = []
        factor_vars: Dict[str, Tuple[List[Any], Any]] = {}
        for factor, info in di.factor_infos.items():
            used = _names_in(factor.code, self._columns)
            if info.type == "categorical":
                cats = list(info.categories)
                base = _base_level(di, factor, cats)
                for v in used:
                    factor_vars[v] = (cats, base)
            else:
                for v in used:
                    if v not in numeric_vars:
                        numeric_vars.append(v)
        both = sorted(set(numeric_vars) & set(factor_vars))
        self._ambiguous = set(both)
        self.factors = {k: v for k, v in factor_vars.items() if k not in both}
        self.variables = [v for v in numeric_vars if v not in both] + [
            v for v in self.factors if v not in numeric_vars
        ]

    def build(self, frame: pd.DataFrame) -> np.ndarray:
        import patsy

        from ..core.utils import _coerce_string_extension_dtypes

        (X,) = patsy.build_design_matrices(
            [self.design_info],
            _coerce_string_extension_dtypes(frame, self.design_info),
            NA_action="raise",
        )
        return np.asarray(X, dtype=float)

    def derivative(self, frame: pd.DataFrame, var: str, eps: float) -> np.ndarray:
        """Central difference of the whole design in the raw variable."""
        if var in self._ambiguous:
            raise MethodIncompatibility(
                f"margins: {var!r} enters both as a factor and numerically; "
                "its marginal effect is not defined.",
            )
        x = frame[var].to_numpy(dtype=float)
        h = eps * np.maximum(np.abs(x), 1.0)
        up = frame.copy()
        dn = frame.copy()
        up[var] = x + h
        dn[var] = x - h
        grad: np.ndarray = (self.build(up) - self.build(dn)) / (2.0 * h)[:, None]
        return grad

    def is_factor(self, var: str) -> bool:
        return var in self.factors


def _names_in(code: str, columns: set) -> List[str]:
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        return [c for c in columns if re.search(rf"\b{re.escape(c)}\b", code)]
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in columns and node.id not in out:
            out.append(node.id)
    return out


def _base_level(di: Any, factor: Any, cats: List[Any]) -> Any:
    """Level whose contrast row is all zero (treatment reference)."""
    for term, subterms in di.term_codings.items():
        for sub in subterms:
            cm = sub.contrast_matrices.get(factor)
            if cm is None:
                continue
            rows = np.asarray(cm.matrix)
            zero = [i for i in range(rows.shape[0]) if not np.any(rows[i])]
            if len(zero) == 1:
                return cats[zero[0]]
    return cats[0]


def _structural_formula(formula: str) -> str:
    """An IV formula with its ``(endog ~ instruments)`` block replaced by
    the endogenous regressors: the equation the coefficients belong to.

    ``y ~ w + (np.log(x) ~ z1 + z2)`` becomes ``y ~ w + np.log(x)``; the
    linearmodels spelling ``[x ~ z]`` is read the same way. A formula
    without such a block is returned as it is.
    """
    if "~" not in formula:
        return formula
    head, rhs = formula.split("~", 1)
    if "~" not in rhs:
        return formula
    depth, start, opener = 0, -1, ""
    for i, ch in enumerate(rhs):
        if ch in "([":
            if depth == 0:
                start, opener = i, ch
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0 and start >= 0 and "~" in rhs[start:i]:
                closes = ")" if opener == "(" else "]"
                if ch != closes:
                    continue
                endog = rhs[start + 1 : i].split("~", 1)[0].strip()
                return f"{head}~{rhs[:start]}{endog}{rhs[i + 1 :]}"
    return formula


def fitted_factor_levels(design_info: Any, columns: Sequence[Any]) -> Dict[str, Any]:
    """Levels of each factor of a fitted formula, ``{column: [levels]}``.

    The reference level has no dummy, so the coefficient names alone do not
    say which values of a factor the model saw; this record (plain values,
    picklable) does, and :func:`reject_unseen_levels` reads it at prediction
    time.  Only factors that are one column of the data (``C(g)``, ``g``,
    ``C(g, Treatment(...))``) are recorded.
    """
    out: Dict[str, Any] = {}
    infos = getattr(design_info, "factor_infos", None) or {}
    names = {str(c) for c in columns}
    for factor, info in infos.items():
        if getattr(info, "type", None) != "categorical":
            continue
        code = str(getattr(factor, "code", ""))
        used = _names_in(code, names)
        bare = len(used) == 1 and (
            code == used[0] or re.match(rf"^C\(\s*{re.escape(used[0])}\s*[,)]", code)
        )
        if not bare:
            continue
        out[used[0]] = [c.item() if hasattr(c, "item") else c for c in info.categories]
    return out


def reject_unseen_levels(
    levels: Any,
    frame: pd.DataFrame,
    context: str = "predict()",
    only: Optional[Sequence[str]] = None,
) -> None:
    """Raise when ``frame`` holds a factor level the model was not fitted on.

    A dummy-coded design gives such a row zeros in every dummy of the
    factor, which is the row of the reference level: the prediction would
    be the reference level's, without a word. ``context`` names the caller
    in the message; ``only`` restricts the check to these columns.
    """
    if not isinstance(levels, dict):
        return
    for var, seen in levels.items():
        if var not in frame.columns or (only is not None and var not in only):
            continue
        known = set()
        for level in seen:
            known.add(str(level))
            try:
                known.add(float(level))
            except (TypeError, ValueError):
                continue
        unseen = []
        for value in pd.unique(frame[var].dropna()):
            try:
                number: Any = float(value)
            except (TypeError, ValueError):
                number = None
            if str(value) not in known and number not in known:
                unseen.append(value.item() if hasattr(value, "item") else value)
        if unseen:
            raise MethodIncompatibility(
                f"{context}: {var!r} takes the value(s) {unseen} in the new "
                f"data, which the model was not fitted on (levels at fit "
                f"time: {list(seen)}). The model has no coefficient for "
                f"them, so there is no prediction for these rows.",
                recovery_hint=(
                    "Drop these rows, recode them to a fitted level, or "
                    "refit the model on data that includes the level."
                ),
                diagnostics={"variable": var, "unseen_levels": unseen},
            )


def known_factor_levels(result: Any, design: Any = None) -> Dict[str, Any]:
    """Levels of each factor the fit saw, ``{column: [levels]}``.

    The record the fit wrote (``data_info['factor_levels']``), or, for a
    fit that wrote none, the levels ``design`` read off the estimation data
    (the base level and one level per dummy).
    """
    recorded = (getattr(result, "data_info", None) or {}).get("factor_levels")
    if isinstance(recorded, dict):
        return recorded
    factors = getattr(design, "factors", None) or {}
    return {var: list(levels) for var, (levels, _base) in factors.items()}


def design_for(result: Any, frame: pd.DataFrame) -> Any:
    """Pick the backend that can evaluate this model's design on ``frame``."""
    terms = [str(t) for t in result.params.index]
    simple = all(_simple_part(p) for t in terms for p in t.split(":"))
    if simple:
        return _TermDesign(terms, frame)
    formula = model_formula(result)
    if formula is None:
        odd = [p for t in terms for p in t.split(":") if not _simple_part(p)]
        raise MethodIncompatibility(
            f"margins: model terms {odd} are transformations and the fit did "
            "not record its formula, so the design cannot be re-evaluated.",
            recovery_hint="Refit with a formula-based estimator (sp.regress, "
            "sp.logit, sp.probit, sp.glm) and pass data=.",
        )
    structural = _structural_formula(formula)
    fitted = _design_from_fitted_frame(result, structural, terms, frame)
    if fitted is not None:
        return fitted
    return _FormulaDesign(structural, frame, terms)


def _fitted_frame(result: Any) -> Optional[pd.DataFrame]:
    """The DataFrame the model was fitted on, if the result still reaches it."""
    from ..core._call_record import get_call_record

    record = get_call_record(result)
    if record is None:
        return None
    for ref in record.frames().values():
        found = ref.get()
        if isinstance(found, pd.DataFrame):
            return found
    return None


def _design_from_fitted_frame(
    result: Any, formula: str, terms: Sequence[str], frame: pd.DataFrame
) -> Optional[_FormulaDesign]:
    """The formula's design as memorised on the frame the model was fitted
    on, to be evaluated on ``frame``.

    patsy memorises stateful transforms (``center``, ``standardize``,
    ``bs`` ...) and factor levels on the frame it first sees. Rebuilt on new
    rows, ``center(x)`` would centre on their mean and a factor missing from
    them would change the columns. The fitted frame gives the coding the
    coefficients belong to; it is used only when, evaluated on the fit's
    rows, it reproduces the stored design (the frame may have been edited
    since). None when ``frame`` is that frame or it is not available.
    """
    fitted = _fitted_frame(result)
    if fitted is None or fitted is frame:
        return None
    try:
        design = _FormulaDesign(formula, fitted, terms)
    except Exception:  # noqa: BLE001 - patsy and pandas raise many types
        return None
    if any(v not in frame.columns for v in design.variables):
        return None
    data_info = getattr(result, "data_info", None) or {}
    X = data_info.get("X")
    if X is None:
        return design
    X = np.asarray(X, dtype=float)
    cols = list(design.variables)
    dep = data_info.get("dependent_var")
    if isinstance(dep, str) and dep in fitted.columns:
        cols.append(dep)
    rows = fitted.loc[fitted[cols].notna().all(axis=1)] if cols else fitted
    try:
        rebuilt = design.build(rows)
    except Exception:  # noqa: BLE001
        return None
    if rebuilt.shape != X.shape:
        # the fit dropped further rows (weights, clusters, singletons):
        # its sample cannot be matched row by row; the coding stands
        return design
    scale = max(1.0, float(np.nanmax(np.abs(X)))) if X.size else 1.0
    if np.allclose(rebuilt, X, rtol=1e-8, atol=1e-10 * scale):
        return design
    return None


__all__ = [
    "CAT_TERM_RE",
    "design_for",
    "factor_value",
    "fitted_factor_levels",
    "known_factor_levels",
    "model_formula",
    "model_setting",
    "reject_unseen_levels",
]
