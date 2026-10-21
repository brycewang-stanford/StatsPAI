"""Margins after limited-dependent-variable models.

``sp.margins`` / ``sp.margins_at`` after ``mlogit``, ``ologit`` / ``oprobit``,
``tobit``, ``heckman`` (``method='ml'``), ``heckprobit`` and ``ivprobit``.

These models are not a single index pushed through an inverse link, which is
what :mod:`.margins` handles: an ordered or multinomial model predicts one
probability per outcome, a tobit has several conditional means, a selection
model combines two equations and a correlation. Each adapter below writes the
prediction as a function of the equation indices ``x'b_e`` and the ancillary
parameters, and the shared engine does what Stata's ``margins`` does:

* the derivative of the prediction with respect to a continuous variable is
  the derivative along the direction in which the variable moves every index
  (a five-point stencil in that direction, step ``2e-3`` in index units times
  the model's scale, so the truncation error is ~1e-13);
* a factor variable is a discrete change from its base level;
* the delta-method standard error uses the Jacobian of the averaged quantity
  with respect to the full parameter vector on the estimation scale (a
  five-point stencil, step ``1e-4 * max(1, |theta_j|)``) and the full ML
  covariance, which every adapted estimator stores.

The prediction options and their defaults are Stata 18's (``predict(...)``
after each command); the evidence is ``tests/test_ldv_margins_stata.py``.
The adapters are in :mod:`._ldv_models`.
"""

from __future__ import annotations

import re
from itertools import product as itertools_product
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility, StatsPAIWarning
from ._ldv_models import _LDVModel, _same, ldv_model

__all__ = ["ldv_model", "ldv_margins", "ldv_margins_at"]

_STENCIL = (
    (-2.0, 1.0 / 12.0),
    (-1.0, -8.0 / 12.0),
    (1.0, 8.0 / 12.0),
    (2.0, -1.0 / 12.0),
)
_INDEX_STEP = 2e-3
_THETA_STEP = 1e-4


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


class _Context:
    """Estimation sample, designs and weights shared by the LDV margins."""

    def __init__(
        self,
        model: _LDVModel,
        data: Optional[pd.DataFrame],
        at: Optional[Dict[str, Any]],
        subset: Any,
        predict: Optional[str],
        outcome: Any,
    ) -> None:
        from .margins import _apply_subset, _averaging_weights

        self.model = model
        self.opt = model.parse(predict)
        frame = data
        if frame is None:
            from .marginaleffects import _fitted_frame

            frame = _fitted_frame(model.result)
            if frame is None:
                raise MethodIncompatibility(
                    f"margins after {model.kind} needs the covariate values: "
                    "pass data=<estimation sample>.",
                )
        designs = model.designs(frame)
        self.variables: List[str] = []
        for d in designs:
            for v in d.variables:
                if v not in self.variables:
                    self.variables.append(v)
        if model.formula:
            # Stata lists dydx(*) in the order the command names the
            # variables; patsy puts factor terms first.
            rhs = model.formula.split("~", 1)[1]

            def _pos(v: str) -> int:
                m = re.search(rf"(?<![\w.]){re.escape(v)}(?!\w)", rhs)
                return m.start() if m else len(rhs)

            self.variables.sort(key=_pos)
        extra = []
        for key in ("weights", "cluster"):
            spec = (getattr(model.result, "model_info", None) or {}).get(key)
            if isinstance(spec, str) and spec in frame.columns:
                extra.append(spec)
        mask = model.sample_mask(frame, self.variables + extra)
        kept = frame.loc[mask]
        self.n_dropped = int((~mask).sum())
        if self.n_dropped and model.nobs is not None and len(kept) != int(model.nobs):
            import warnings

            warnings.warn(
                StatsPAIWarning(
                    f"margins: averaging over {len(kept)} complete row(s) of "
                    f"data, but the {model.kind} fit used {int(model.nobs)}.",
                    recovery_hint="Pass the data the model was fitted on.",
                ),
                stacklevel=4,
            )
        if subset is not None:
            kept = _apply_subset(kept, data, subset)
        bind = getattr(model, "bind_base", None)
        if bind is not None:
            bind(kept)
        if at:
            unknown = [k for k in at if k not in self.variables]
            if unknown:
                raise MethodIncompatibility(
                    f"margins: at() variable(s) {unknown} are not in the model "
                    "(Stata: 'not found in list of covariates', r(322)).",
                    recovery_hint=f"Model variables: {self.variables}.",
                    diagnostics={"not_in_model": unknown},
                )
            kept = kept.copy()
            for name, value in at.items():
                kept[name] = value
        self.frame = kept
        self.factors: Dict[str, Tuple[List[Any], Any]] = {}
        self.factor_vars: set = set()
        self.designs = model.designs(kept)
        for d in self.designs:
            for v in d.variables:
                if d.is_factor(v):
                    self.factor_vars.add(v)
            self.factors.update(getattr(d, "factors", {}))
        w, self.weights_name = _averaging_weights(model.result, kept, data)
        if isinstance(w, str):
            raise MethodIncompatibility(
                f"margins: the fit used weights={w!r}, which Stata margins "
                "averages with, but that column is not in the data.",
                recovery_hint="Pass data= containing the weight column.",
            )
        self.weights = None if w is None else np.asarray(w, dtype=float)
        outs = model.outcomes(self.opt)
        if outcome is None:
            self.cols = list(range(len(outs)))
        else:
            wanted = outcome if isinstance(outcome, (list, tuple)) else [outcome]
            self.cols = []
            for o in wanted:
                hit = [j for j, c in enumerate(outs) if c is not None and _same(c, o)]
                if not hit:
                    raise MethodIncompatibility(
                        f"margins: category={o!r} is not an outcome of this "
                        f"{model.kind} prediction; outcomes are "
                        f"{[c for c in outs if c is not None]}.",
                    )
                self.cols.extend(hit)
        self.outcomes = [outs[j] for j in self.cols]
        self.multi = outs[0] is not None

    # -- pieces --------------------------------------------------------------
    def mean(self, values: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.asarray(values.mean(axis=0))
        w = self.weights / self.weights.sum()
        return np.asarray(np.tensordot(w, values, axes=(0, 0)))

    def build(self, frame: pd.DataFrame) -> List[np.ndarray]:
        return [d.build(frame) for d in self.designs]

    def derivative(self, var: str, frame: pd.DataFrame, eps: float) -> List[np.ndarray]:
        out = []
        for d, terms in zip(self.designs, self.model.eq_terms):
            if var in d.variables:
                out.append(d.derivative(frame, var, eps))
            else:
                out.append(np.zeros((len(frame), len(terms))))
        return out

    def at_means(self, Xs: List[np.ndarray]) -> List[np.ndarray]:
        return [self.mean(X)[None, :] for X in Xs]

    def indices(self, theta: np.ndarray, Xs: List[np.ndarray]) -> List[np.ndarray]:
        return [X @ theta[p] for X, p in zip(Xs, self.model.eq_pos)]

    def prediction(self, theta: np.ndarray, Xs: List[np.ndarray]) -> np.ndarray:
        P = self.model.predict(theta, self.indices(theta, Xs), self.opt)
        return np.asarray(P)[:, self.cols]

    def slope(
        self,
        theta: np.ndarray,
        Xs: List[np.ndarray],
        Ds: List[np.ndarray],
        t: np.ndarray,
    ) -> np.ndarray:
        """Unit derivatives of the prediction along d index / d var."""
        idx = self.indices(theta, Xs)
        dirs = [D @ theta[p] for D, p in zip(Ds, self.model.eq_pos)]
        acc = 0.0
        for s, c in _STENCIL:
            shifted = [i + (s * t) * d for i, d in zip(idx, dirs)]
            acc = acc + c * np.asarray(self.model.predict(theta, shifted, self.opt))
        _ret: np.ndarray = np.asarray(acc)[:, self.cols] / t[:, None]
        return _ret

    def step(self, Ds: List[np.ndarray]) -> np.ndarray:
        dirs = [np.abs(D @ self.model.theta[p]) for D, p in zip(Ds, self.model.eq_pos)]
        size = np.max(np.column_stack(dirs), axis=1)
        _ret: np.ndarray = (
            _INDEX_STEP * self.model.scale(self.model.theta) / np.maximum(size, 1.0)
        )
        return _ret

    def inference(
        self, f: Callable[[np.ndarray], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = self.model.theta
        est = np.asarray(f(theta), dtype=float).reshape(-1)
        J = np.zeros((est.size, theta.size))
        for j in range(theta.size):
            h = _THETA_STEP * max(1.0, abs(theta[j]))
            acc = np.zeros(est.size)
            for s, c in _STENCIL:
                th = theta.copy()
                th[j] += s * h
                acc += c * np.asarray(f(th), dtype=float).reshape(-1)
            J[:, j] = acc / h
        V = self.model.V
        se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", J, V, J), 0.0, None))
        return est, se, J

    def attrs(self) -> Dict[str, Any]:
        return {
            "n": int(len(self.frame)),
            "n_dropped": int(self.n_dropped),
            "weights": self.weights_name,
            "design_backend": "ldv",
            "model": self.model.kind,
            "predict": repr(self.opt),
            "prediction_scale": self.model.expression(self.opt),
        }


def _require_no_interactions(ctx: _Context) -> None:
    if any(":" in t for terms in ctx.model.eq_terms for t in terms) or any(
        getattr(d, "backend", "") == "formula" for d in ctx.designs
    ):
        raise MethodIncompatibility(
            f"margins(method='mem') after {ctx.model.kind} supports models "
            "whose terms are plain variables and factor indicators; with "
            "interactions or transformations Stata's atmeans holds each "
            "variable, not each term, at its mean.",
            recovery_hint="Use method='ame', or margins_at with explicit values.",
        )


def ldv_margins(
    model: _LDVModel,
    data: Optional[pd.DataFrame],
    variables: Optional[List[str]],
    at: Optional[Dict[str, Any]],
    method: str,
    eps: float,
    alpha: float,
    subset: Any,
    predict: Optional[str],
    outcome: Any,
) -> pd.DataFrame:
    """``sp.margins`` (dy/dx) after a limited-dependent-variable model."""
    from .margins import _inference_dict, _level_label, _same_level

    ctx = _Context(model, data, at, subset, predict, outcome)
    if variables is None:
        variables = [
            v
            for v in ctx.variables
            if v in ctx.frame.columns and (v not in ctx.factor_vars or v in ctx.factors)
        ]
    else:
        unknown = [v for v in variables if v not in ctx.variables]
        if unknown:
            raise MethodIncompatibility(
                f"margins: {unknown} do not enter the model.",
                recovery_hint=f"Variables in the model: {ctx.variables}.",
            )
    if method == "mem":
        _require_no_interactions(ctx)
    frame = ctx.frame
    rows: List[Dict[str, Any]] = []
    base_levels: Dict[str, Any] = {}

    def emit(label: str, est: np.ndarray, se: np.ndarray) -> None:
        for j, out in enumerate(ctx.outcomes):
            row = _inference_dict(float(est[j]), float(se[j]), np.inf, alpha, label)
            if ctx.multi:
                row["outcome"] = out
            rows.append(row)

    for var in variables:
        if var in ctx.factor_vars:
            if var not in ctx.factors:
                raise MethodIncompatibility(
                    f"margins: {var!r} enters as a factor but its levels cannot "
                    "be read from the data (pass data= with the raw column).",
                )
            levels, base = ctx.factors[var]
            base_levels[var] = base
            Xb = ctx.build(frame.assign(**{var: base}))
            if method == "mem":
                Xb = ctx.at_means(Xb)
            for lev in levels:
                if _same_level(lev, base):
                    continue
                Xl = ctx.build(frame.assign(**{var: lev}))
                if method == "mem":
                    Xl = ctx.at_means(Xl)

                def f_disc(th: np.ndarray, _Xl: Any = Xl, _Xb: Any = Xb) -> np.ndarray:
                    diff = ctx.prediction(th, _Xl) - ctx.prediction(th, _Xb)
                    return diff[0] if method == "mem" else ctx.mean(diff)

                est, se, _ = ctx.inference(f_disc)
                emit(f"{_level_label(lev)}.{var}", est, se)
            continue
        Xs = ctx.build(frame)
        Ds = ctx.derivative(var, frame, eps)
        if method == "mem":
            Xs, Ds = ctx.at_means(Xs), ctx.at_means(Ds)
        t = ctx.step(Ds)

        def f_cont(
            th: np.ndarray, _Xs: Any = Xs, _Ds: Any = Ds, _t: Any = t
        ) -> np.ndarray:
            unit = ctx.slope(th, _Xs, _Ds, _t)
            return unit[0] if method == "mem" else ctx.mean(unit)

        est, se, _ = ctx.inference(f_cont)
        emit(var, est, se)

    cols = ["variable", "dy/dx", "se", "z", "pvalue", "ci_lower", "ci_upper"]
    if ctx.multi:
        cols.insert(1, "outcome")
    out = pd.DataFrame(rows, columns=cols)
    out.attrs.update(ctx.attrs())
    if base_levels:
        out.attrs["base_levels"] = base_levels
    return out


def ldv_margins_at(
    model: _LDVModel,
    data: pd.DataFrame,
    at: Dict[str, Any],
    alpha: float,
    contrast: Optional[str],
    subset: Any,
    predict: Optional[str],
    outcome: Any,
) -> pd.DataFrame:
    """``sp.margins_at`` (predictive margins) after an LDV model."""
    from .margins import _inference_dict

    ctx = _Context(model, data, None, subset, predict, outcome)
    unknown = [v for v in at if v not in ctx.variables]
    if unknown:
        raise MethodIncompatibility(
            f"margins_at: {unknown} do not enter the model.",
            recovery_hint=f"Variables in the model: {ctx.variables}.",
        )
    at_vars = list(at.keys())
    grid = list(itertools_product(*[np.atleast_1d(at[v]).tolist() for v in at_vars]))
    designs = [ctx.build(ctx.frame.assign(**dict(zip(at_vars, p)))) for p in grid]
    m = len(ctx.cols)

    def f_all(th: np.ndarray) -> np.ndarray:
        return np.concatenate([ctx.mean(ctx.prediction(th, Xs)) for Xs in designs])

    est, se, J = ctx.inference(f_all)
    crit = stats.norm.ppf(1 - alpha / 2)
    rows: List[Dict[str, Any]] = []
    if contrast is None:
        for g, point in enumerate(grid):
            for j, out in enumerate(ctx.outcomes):
                k = g * m + j
                row: Dict[str, Any] = dict(zip(at_vars, point))
                if ctx.multi:
                    row["outcome"] = out
                row.update(
                    margin=float(est[k]),
                    se=float(se[k]),
                    ci_lower=float(est[k] - crit * se[k]),
                    ci_upper=float(est[k] + crit * se[k]),
                )
                rows.append(row)
    else:
        if len(grid) < 2:
            raise MethodIncompatibility(
                "margins_at: a contrast needs at least two grid points.",
            )
        for g in range(1, len(grid)):
            ref = 0 if contrast == "first" else g - 1
            for j, out in enumerate(ctx.outcomes):
                a, b = g * m + j, ref * m + j
                diff = float(est[a] - est[b])
                grad = J[a] - J[b]
                s = float(np.sqrt(max(float(grad @ model.V @ grad), 0.0)))
                inf = _inference_dict(diff, s, np.inf, alpha, "")
                row = dict(zip(at_vars, grid[g]))
                if ctx.multi:
                    row["outcome"] = out
                row.update(
                    versus=", ".join(f"{v}={x}" for v, x in zip(at_vars, grid[ref])),
                    contrast=diff,
                    se=s,
                    z=inf["z"],
                    pvalue=inf["pvalue"],
                    ci_lower=inf["ci_lower"],
                    ci_upper=inf["ci_upper"],
                )
                rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs.update(ctx.attrs())
    return out
