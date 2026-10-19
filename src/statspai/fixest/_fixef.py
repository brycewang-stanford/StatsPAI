"""Absorbed fixed effects after ``sp.feols`` / ``sp.fepois`` / ``sp.feglm``.

The fit absorbs the effects and reports only the slopes. Three things need
the effects themselves:

* ``result.fixef()`` -- the estimated effect of every level (R
  ``fixest::fixef``, Stata ``reghdfe, savefe``);
* ``result.predict(newdata)`` -- fixest's ``predict``: ``x'b`` plus the
  effects of the row's levels, through the inverse link;
* ``sp.margins`` / ``sp.avg_slopes`` / ``sp.avg_predictions`` after a
  non-linear fit -- marginaleffects on a fixest model evaluates the link
  at ``x'b`` plus the effects, holding the effects fixed (they enter as an
  offset; the standard errors are the delta method in ``b`` alone).

At fit time the result keeps, for the rows of the estimation sample, the
level of each effect and the sum of the effects on the link scale
(``eta - x'b``). The individual effects are recovered from that sum on
demand (alternating projections on an exactly consistent system) and
normalised as fixest reports them, a convention read off its output (the
package is GPL and used as a black box): with two effects, in each
connected set of levels the level of the second effect seen in the set's
first row is 0; with three or more, the level of every effect after the
first seen in the first row is 0. Sums of effects -- and so every
prediction and marginal effect -- do not depend on the normalisation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility, StatsPAIWarning

__all__ = [
    "FEState",
    "attach",
    "fe_offset",
    "fe_state",
    "fixef",
    "has_fixed_effects",
]


@dataclass
class FEState:
    """The absorbed effects of one fit, on its estimation sample."""

    names: List[str]
    codes: List[np.ndarray]  # per sample row, codes in first-seen order
    levels: List[np.ndarray]  # level labels, indexed by code
    fe_sum: np.ndarray  # per sample row, sum of the effects (link scale)
    sample_index: Optional[pd.Index]  # labels of the sample rows (native fits)
    alpha: Optional[List[np.ndarray]] = field(default=None, repr=False)
    max_error: float = float("nan")


def fe_state(result: Any) -> Optional[FEState]:
    """The :class:`FEState` of ``result``, or None for a fit without one.

    A fit pyfixest answered gets one built from its estimation data and
    fitted index on first use (so its effects are recovered and normalised
    exactly as a native fit's).
    """
    state = getattr(result, "_fe_state", None)
    if isinstance(state, FEState):
        return state
    fit = _delegate_fe_fit(result)
    if fit is None:
        return None
    _require_additive(fit)
    state = _delegate_state(result, fit)
    if state is not None:
        result._fe_state = state
    return state


def attach(
    result: Any,
    frame: pd.DataFrame,
    keep: np.ndarray,
    fe: List[str],
    eta: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
) -> None:
    """Record the absorbed effects of a native fit on ``result``.

    ``frame`` is the data the fit used (complete rows), ``keep`` its rows in
    the estimation sample, ``eta`` the fitted linear index on those rows
    and ``X`` the raw regressors in coefficient order.
    """
    if not fe:
        return
    keep = np.asarray(keep, dtype=bool)
    rows = frame.loc[keep] if not keep.all() else frame
    codes, levels = [], []
    for name in fe:
        c, u = pd.factorize(rows[name], sort=False)
        codes.append(c.astype(np.int64))
        levels.append(np.asarray(u, dtype=object))
    fe_sum = np.asarray(eta, dtype=float) - np.asarray(X, dtype=float) @ np.asarray(
        beta, dtype=float
    )
    result._fe_state = FEState(
        names=list(fe),
        codes=codes,
        levels=levels,
        fe_sum=fe_sum,
        sample_index=rows.index,
    )


# ---------------------------------------------------------------------------
# Recovering the effects
# ---------------------------------------------------------------------------


def _solve(state: FEState, tol: float = 1e-13, maxiter: int = 20_000) -> None:
    d = state.fe_sum
    counts = [
        np.bincount(c, minlength=len(lv)) for c, lv in zip(state.codes, state.levels)
    ]
    alpha = [np.zeros(len(lv)) for lv in state.levels]
    scale = max(1.0, float(np.max(np.abs(d))) if d.size else 1.0)
    K = len(alpha)
    total = np.zeros_like(d)
    err = np.inf
    for _ in range(maxiter if K > 1 else 1):
        for k in range(K):
            part = alpha[k][state.codes[k]]
            r = d - (total - part)
            new = np.bincount(state.codes[k], weights=r, minlength=len(alpha[k]))
            new = new / counts[k]
            total += new[state.codes[k]] - part
            alpha[k] = new
        err = float(np.max(np.abs(d - total))) / scale if d.size else 0.0
        if err < tol:
            break
    if K > 1 and err >= 1e-9:
        warnings.warn(
            StatsPAIWarning(
                "fixef: the fixed effects reproduce the fitted values only to "
                f"{err:.1e} (relative) after {maxiter} passes.",
                recovery_hint="The levels are weakly connected; sums of "
                "effects may be inaccurate at that order.",
                diagnostics={"max_relative_error": err},
            ),
            stacklevel=3,
        )
    _normalise(state, alpha)
    state.alpha = alpha
    state.max_error = err


def _components(c1: np.ndarray, c2: np.ndarray, g1: int, g2: int) -> np.ndarray:
    """Connected set of each row (levels of two effects as a bipartite graph)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = g1 + g2
    adj = coo_matrix((np.ones(len(c1)), (c1, c2 + g1)), shape=(n, n)).tocsr()
    _, lab = connected_components(adj, directed=False)
    return lab[c1]


def _normalise(state: FEState, alpha: List[np.ndarray]) -> None:
    K = len(alpha)
    if K < 2 or state.fe_sum.size == 0:
        return
    c = state.codes
    if K == 2:
        comp = _components(c[0], c[1], len(alpha[0]), len(alpha[1]))
        _, first_rows = np.unique(comp, return_index=True)
        for r in first_rows:
            shift = alpha[1][c[1][r]]
            in_comp = comp == comp[r]
            lv1 = np.unique(c[0][in_comp])
            lv2 = np.unique(c[1][in_comp])
            alpha[1][lv2] -= shift
            alpha[0][lv1] += shift
        return
    for k in range(1, K):
        shift = alpha[k][c[k][0]]
        alpha[k] = alpha[k] - shift
        alpha[0] = alpha[0] + shift


def fixef(result: Any) -> Dict[str, pd.Series]:
    """The estimated fixed effects of a ``sp.feols`` / ``fepois`` / ``feglm`` fit.

    Returns
    -------
    dict
        One Series per absorbed effect, indexed by its levels (sorted),
        on the scale of the linear index (log for ``fepois``).
    """
    state = fe_state(result)
    if state is None:
        fit = _delegate_fe_fit(result)
        if fit is not None:
            return _delegate_fixef(result, fit)
        raise MethodIncompatibility(
            "fixef(): this result does not carry absorbed fixed effects.",
            recovery_hint="Use it after sp.feols / sp.fepois / sp.feglm with a "
            "'| fe' part in the formula.",
        )
    if state.alpha is None:
        _solve(state)
    assert state.alpha is not None
    out: Dict[str, pd.Series] = {}
    for name, lv, a in zip(state.names, state.levels, state.alpha):
        s = pd.Series(a, index=pd.Index(lv, name=name), name=name)
        try:
            s = s.sort_index()
        except TypeError:
            pass
        out[name] = s
    return out


def fe_offset(result: Any, frame: pd.DataFrame, *, unseen: str = "nan") -> np.ndarray:
    """Sum of the effects of the levels in ``frame`` (link scale).

    ``unseen='nan'`` gives NaN (with a warning) for a row whose level of
    some effect the fit never saw, as fixest's ``predict`` does;
    ``unseen='raise'`` refuses such rows.
    """
    state = fe_state(result)
    if state is None:
        fit = _delegate_fe_fit(result)
        if fit is None:
            return np.zeros(len(frame))
        return _delegate_offset(result, fit, frame, unseen)
    missing = [n for n in state.names if n not in frame.columns]
    if missing:
        raise MethodIncompatibility(
            f"the model absorbed fixed effects {state.names}; the data lacks "
            f"{missing}.",
            recovery_hint="Include the fixed-effect columns in the data.",
            diagnostics={"missing_columns": missing},
        )
    effects = fixef(result)
    total = np.zeros(len(frame))
    bad = np.zeros(len(frame), dtype=bool)
    for name in state.names:
        s = effects[name]
        vals = frame[name].map(s)
        miss = vals.isna().to_numpy()
        if miss.any() and frame[name].dtype != s.index.dtype:
            # e.g. integer levels read back as floats
            try:
                alt = pd.Series(s.to_numpy(), index=s.index.astype(frame[name].dtype))
                vals = frame[name].map(alt)
                miss = vals.isna().to_numpy()
            except (TypeError, ValueError):
                pass
        bad |= miss
        total = total + np.where(miss, 0.0, vals.to_numpy(dtype=float, na_value=0.0))
    return _flag_unseen(total, bad, unseen)


def _flag_unseen(total: np.ndarray, bad: np.ndarray, unseen: str) -> np.ndarray:
    if bad.any():
        if unseen == "raise":
            raise MethodIncompatibility(
                f"{int(bad.sum())} row(s) of the data have a fixed-effect level "
                "the fit never saw; their effect is not estimated.",
                recovery_hint="Evaluate on rows whose levels are in the "
                "estimation sample.",
                diagnostics={"n_unseen": int(bad.sum())},
            )
        warnings.warn(
            StatsPAIWarning(
                f"predict(): {int(bad.sum())} row(s) have a fixed-effect level "
                "the fit never saw; their prediction is NaN (as in fixest).",
                recovery_hint="Drop those rows or use levels from the "
                "estimation sample.",
                diagnostics={"n_unseen": int(bad.sum())},
            ),
            stacklevel=3,
        )
        total[bad] = np.nan
    return total


def has_fixed_effects(result: Any) -> bool:
    """Whether the fit absorbed fixed effects (native or through pyfixest)."""
    return fe_state(result) is not None or _delegate_fe_fit(result) is not None


# ---------------------------------------------------------------------------
# Fits pyfixest answered (backend='pyfixest', or calls the native engine
# does not fit): its own fixef() / predict() carry the effects.
# ---------------------------------------------------------------------------


def _delegate_fe_fit(result: Any) -> Any:
    fit = getattr(result, "_pyfixest_fit", None)
    if fit is None or not getattr(fit, "_fixef", None):
        return None
    return fit


def _require_additive(fit: Any) -> None:
    if "[" in str(getattr(fit, "_fixef", "")) or "^" in str(getattr(fit, "_fixef", "")):
        raise MethodIncompatibility(
            "this fit has varying slopes or combined fixed effects "
            f"({fit._fixef!r}); the effects are not an additive term of the "
            "linear index here.",
            recovery_hint="Use result._pyfixest_fit.predict(newdata=...) "
            "directly, or enter the slopes as interactions.",
        )


def _delegate_state(result: Any, fit: Any) -> Optional[FEState]:
    """FEState from pyfixest's estimation data, or None if it cannot be
    aligned with its fitted values (then pyfixest's own effects are used)."""
    from ..postestimation._design import design_for

    data = getattr(fit, "_data", None)
    names = [t.strip() for t in str(fit._fixef).split("+")]
    if not isinstance(data, pd.DataFrame) or any(n not in data for n in names):
        return None
    try:
        eta = np.asarray(fit.predict(type="link"), dtype=float).ravel()
        X = np.asarray(design_for(result, data).build(data), dtype=float)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        # pyfixest internals vary by version: say that its own effects
        # (its normalisation) are used instead
        warnings.warn(
            StatsPAIWarning(
                "fixef / predict: the fixed effects could not be recovered "
                f"from pyfixest's estimation data ({type(exc).__name__}); "
                "pyfixest's own effects are used.",
                recovery_hint="Fit with backend='auto' for fixest's " "normalisation.",
            ),
            stacklevel=3,
        )
        return None
    beta = np.asarray(result.params, dtype=float)
    if eta.shape[0] != len(data) or X.shape != (len(data), beta.size):
        return None
    codes, levels = [], []
    for name in names:
        c, u = pd.factorize(data[name], sort=False)
        if (c < 0).any():
            return None
        codes.append(c.astype(np.int64))
        levels.append(np.asarray(u, dtype=object))
    return FEState(
        names=names,
        codes=codes,
        levels=levels,
        fe_sum=eta - X @ beta,
        sample_index=None,
    )


def _delegate_fixef(result: Any, fit: Any) -> Dict[str, pd.Series]:
    _require_additive(fit)
    raw = fit.fixef()
    out: Dict[str, pd.Series] = {}
    data = getattr(fit, "_data", None)
    for key, values in raw.items():
        name = str(key)
        if name.startswith("C(") and name.endswith(")"):
            name = name[2:-1]
        s = pd.Series(dict(values), dtype=float, name=name)
        if isinstance(data, pd.DataFrame) and name in data.columns:
            # pyfixest labels levels as strings and leaves its reference
            # level out (its effect is 0)
            col = data[name]
            levels = pd.unique(col.dropna())
            s = pd.Series(
                [float(dict(values).get(str(v), 0.0)) for v in levels],
                index=pd.Index(levels, name=name),
                name=name,
            )
        try:
            s = s.sort_index()
        except TypeError:
            pass
        out[name] = s
    return out


def _delegate_offset(
    result: Any, fit: Any, frame: pd.DataFrame, unseen: str
) -> np.ndarray:
    _require_additive(fit)
    from ..postestimation._design import design_for

    eta = np.asarray(fit.predict(newdata=frame, type="link"), dtype=float)
    X = np.asarray(design_for(result, frame).build(frame), dtype=float)
    total = eta - X @ np.asarray(result.params, dtype=float)
    bad = ~np.isfinite(total)
    total = np.where(bad, 0.0, total)
    return _flag_unseen(total, bad, unseen)
