"""Absorbed fixed effects after ``sp.hdfe_ols`` (Stata ``reghdfe, savefe``).

At fit time :func:`attach` keeps, for the rows of the estimation sample,
the level of each absorbed effect and the sum of the effects
``y - x'b - e``. :func:`fixef` recovers the individual effects from that
sum by an exact sparse solve (``fast/_fe_direct``: no alternating
projections, which crawl on the poorly connected graphs of worker-firm
data) and normalises them as ``reghdfe`` saves them (Stata 18, reghdfe
6.13, checked on ``tests/reference_parity/_fixtures/akm_twoway*``):

* for every effect after the first, its (weighted) mean over the rows of
  each connected set formed with the first effect is zero, the first
  effect absorbing the shift;
* the first effect then has mean zero over the sample, and ``reghdfe``
  reports what is left as ``_cons = ybar - xbar'b``.

Sums of effects, hence fitted values, do not depend on the normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility, NumericalInstability

__all__ = ["HdfeFEState", "attach", "fixef", "row_effects"]


@dataclass
class HdfeFEState:
    names: List[str]
    codes: List[np.ndarray]
    levels: List[pd.Index]
    fe_sum: np.ndarray
    weights: Optional[np.ndarray]
    sample_index: pd.Index


def attach(
    result: Any,
    frame: pd.DataFrame,
    keep: np.ndarray,
    fe_terms: List[Any],
    y: np.ndarray,
    X: np.ndarray,
    coef: np.ndarray,
    resid: np.ndarray,
    weights: Optional[np.ndarray],
) -> None:
    """Record what :func:`fixef` needs (estimation-sample rows only)."""
    keep = np.asarray(keep, dtype=bool)
    rows = frame.iloc[np.flatnonzero(keep)]
    b = np.where(np.isfinite(coef), coef, 0.0)
    Xk = np.asarray(X, dtype=float)[keep]
    yk = np.asarray(y, dtype=float)[keep]
    wk = None if weights is None else np.asarray(weights, dtype=float)[keep]
    xb = Xk @ b
    fe_sum = yk - xb - np.asarray(resid, dtype=float)
    names, codes, levels = [], [], []
    for t in fe_terms:
        cols = [a.name for a in t.atoms]
        if len(cols) == 1:
            c, u = pd.factorize(rows[cols[0]], sort=True)
            lev = pd.Index(u, name=t.label)
        else:
            key = pd.MultiIndex.from_frame(rows[cols])
            c, u = pd.factorize(key, sort=True)
            lev = pd.MultiIndex.from_tuples(list(u), names=cols)
        # an interaction is keyed "a^b" however it was written (a^b, a:b,
        # i.a#i.b), the spelling the Stata translator emits for a#b
        names.append(t.label if len(cols) == 1 else "^".join(cols))
        codes.append(c.astype(np.int64))
        levels.append(lev)
    result._hdfe_fe_state = HdfeFEState(
        names=names,
        codes=codes,
        levels=levels,
        fe_sum=fe_sum,
        weights=wk,
        sample_index=rows.index,
    )


def _solve(state: HdfeFEState) -> List[np.ndarray]:
    from scipy import sparse

    from ..fast._fe_direct import _Solver, fe_design

    K = len(state.codes)
    n = len(state.fe_sum)
    sizes = [int(c.max()) + 1 if c.size else 0 for c in state.codes]
    order = sorted(range(K), key=lambda j: -sizes[j])
    D = sparse.csc_matrix(fe_design([state.codes[j] for j in order], n))
    w = np.ones(n) if state.weights is None else state.weights
    DtW = D.multiply(w[:, None]).T.tocsr()
    M = DtW @ D
    diag = M.diagonal()
    level_w = np.where(diag > 0, diag, 1.0)
    solver = _Solver(M, sizes[order[0]] if K > 1 else 0)
    r = state.fe_sum.copy()
    a = np.zeros(D.shape[1])
    scale = max(1.0, float(np.max(np.abs(r))) if r.size else 1.0)
    worst = np.inf
    for _ in range(50):
        g = DtW @ r
        worst = float(np.max(np.abs(g / level_w))) if g.size else 0.0
        if worst <= 1e-13 * scale:
            break
        step = solver(g)
        a += step
        r -= D @ step
    if float(np.max(np.abs(r))) > 1e-8 * scale:
        raise NumericalInstability(
            "hdfe_ols.fixef: the fixed effects reproduce the fitted values "
            f"only to {float(np.max(np.abs(r))):.1e}.",
            recovery_hint="The fit's own convergence limits this; refit with a "
            "smaller tol= in sp.hdfe_ols.",
            diagnostics={"max_abs_error": float(np.max(np.abs(r)))},
        )
    offsets = np.cumsum([0] + [sizes[j] for j in order])
    out: List[np.ndarray] = [np.zeros(0)] * K
    for pos, j in enumerate(order):
        out[j] = a[offsets[pos] : offsets[pos + 1]].copy()
    return out


def _components(c0: np.ndarray, ck: np.ndarray, n0: int, nk: int) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = n0 + nk
    adj = coo_matrix((np.ones(len(c0)), (c0, ck + n0)), shape=(n, n)).tocsr()
    _, lab = connected_components(adj, directed=False)
    return lab[c0]


def _normalise(state: HdfeFEState, alpha: List[np.ndarray]) -> None:
    c = state.codes
    w = np.ones(len(state.fe_sum)) if state.weights is None else state.weights
    for k in range(1, len(alpha)):
        comp = _components(c[0], c[k], len(alpha[0]), len(alpha[k]))
        m = np.bincount(comp, weights=w * alpha[k][c[k]]) / np.bincount(comp, weights=w)
        # every level of effect k (and of effect 0) sits in one component
        comp_k = np.zeros(len(alpha[k]), dtype=np.int64)
        comp_k[c[k]] = comp
        comp_0 = np.zeros(len(alpha[0]), dtype=np.int64)
        comp_0[c[0]] = comp
        alpha[k] -= m[comp_k]
        alpha[0] += m[comp_0]


def _effects(result: Any, absorb_constant: bool):
    state = getattr(result, "_hdfe_fe_state", None)
    if not isinstance(state, HdfeFEState):
        raise MethodIncompatibility(
            "fixef(): this sp.hdfe_ols fit kept no fixed-effect state; it is "
            "recorded for OLS fits with plain absorbed factors (no "
            "instruments, no absorbed slopes).",
            recovery_hint="Refit without the IV part / slope terms, or use "
            "sp.feols(...).fixef().",
        )
    alpha = _solve(state)
    _normalise(state, alpha)
    w = np.ones(len(state.fe_sum)) if state.weights is None else state.weights
    mean0 = float(np.average(alpha[0][state.codes[0]], weights=w))
    # mean0 = ybar - xbar'b (the residuals average zero): reghdfe's _cons
    alpha[0] = alpha[0] - mean0 + (mean0 if absorb_constant else 0.0)
    return state, alpha, (0.0 if absorb_constant else mean0)


def fixef(result: Any, absorb_constant: bool = True) -> Dict[str, pd.Series]:
    """Effects of each absorbed term, indexed by level (sorted)."""
    state, alpha, constant = _effects(result, absorb_constant)
    out: Dict[str, pd.Series] = {}
    for name, a, lev in zip(state.names, alpha, state.levels):
        s = pd.Series(a, index=lev, name=name)
        s.attrs["constant"] = constant
        out[name] = s
    return out


def row_effects(
    result: Any, absorb_constant: bool = False
) -> Tuple[pd.Index, Dict[str, np.ndarray]]:
    """Each term's effect on every estimation-sample row (``reghdfe``'s
    saved variables with the default ``absorb_constant=False``), and the
    index labels of those rows in the data the fit was given."""
    state, alpha, _ = _effects(result, absorb_constant)
    return state.sample_index, {
        name: a[c] for name, a, c in zip(state.names, alpha, state.codes)
    }
