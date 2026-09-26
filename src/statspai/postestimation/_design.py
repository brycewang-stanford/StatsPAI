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

from ..exceptions import MethodIncompatibility

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
                raise MethodIncompatibility(f"margins: {base!r} is not in the data.")
            return np.array([factor_value(level, v) for v in frame[base]], dtype=float)
        if part not in frame.columns:
            raise MethodIncompatibility(f"margins: {part!r} is not in the data.")
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

        from ..core.utils import _coerce_string_extension_dtypes

        rhs = formula.split("~", 1)[1].strip()
        frame = _coerce_string_extension_dtypes(frame)
        try:
            di = patsy.dmatrix(rhs, frame, NA_action="drop").design_info
        except Exception as exc:  # patsy raises many types
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
            _coerce_string_extension_dtypes(frame),
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
    return _FormulaDesign(formula, frame, terms)


__all__ = [
    "CAT_TERM_RE",
    "design_for",
    "factor_value",
    "model_formula",
    "model_setting",
]
