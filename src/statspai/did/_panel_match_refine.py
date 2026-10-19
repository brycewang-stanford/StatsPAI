"""Refinement weights for ``sp.panel_match`` matched sets.

Refinement reweights the controls of each matched set by how close they
are to the treated unit on time-varying covariates:

* ``'mahalanobis'`` -- for each period ``t-L, ..., t`` the Mahalanobis
  distance of each control from the treated unit, with the covariance of
  the set's controls in that period; distances are averaged over the
  periods and the ``size_match`` closest controls share equal weight.
* ``'ps_weight'`` / ``'cbps_weight'`` -- a logit (or CBPS) propensity score
  fitted on the pooled period-``t`` rows of all sets; controls are weighted
  by their odds ``e / (1 - e)``, normalised within the set.
* ``'ps_match'`` / ``'cbps_match'`` -- the ``size_match`` controls closest
  on that score share equal weight.

Covariates are columns of the balanced grid, optionally lagged within the
unit. Missing covariate values are set to 0 and an indicator column per
covariate flags them, so a missing value is itself matched on (R
``PanelMatch`` does the same).
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.linalg import get_lapack_funcs, lu_factor

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._panel_match_sets import Grid, MatchedSet

__all__ = ["parse_covariates", "covariate_tensor", "refine"]

_LAG_RE = re.compile(
    r"^(?:I\(\s*)?lag\(\s*([^,\s()]+)\s*,\s*(\d+)\s*(?::\s*(\d+))?\s*\)(?:\s*\))?$"
)
_ROUND = 5  # Mahalanobis distances are compared to 5 decimals (ties)


def parse_covariates(specs: Sequence[str]) -> List[Tuple[str, int]]:
    """``['x', 'lag(y, 1:4)', 'lag(z, 2)']`` -> ``[(x,0), (y,1)...(y,4), (z,2)]``."""
    out: List[Tuple[str, int]] = []
    for raw in specs:
        spec = str(raw).strip()
        m = _LAG_RE.match(spec)
        if m:
            name, a, b = m.group(1), int(m.group(2)), m.group(3)
            lags = range(a, int(b) + 1) if b is not None else [a]
            if b is not None and int(b) < a:
                raise MethodIncompatibility(
                    f"panel_match: lag range in {spec!r} runs backwards.",
                    recovery_hint="Write lag(x, 1:4), lowest lag first.",
                )
            out.extend((name, int(k)) for k in lags)
        elif re.fullmatch(r"[A-Za-z_.][\w.]*", spec):
            out.append((spec, 0))
        else:
            raise MethodIncompatibility(
                f"panel_match: cannot read covariate {spec!r}.",
                recovery_hint=(
                    "Give a column name or lag(column, k) / lag(column, a:b), "
                    "e.g. covariates=['x1', 'lag(y, 1:4)']."
                ),
            )
    return out


def covariate_tensor(grid: Grid, terms: List[Tuple[str, int]]) -> np.ndarray:
    """``(N, P, 2K)``: K covariate values (missing -> 0) then K indicators."""
    N, P = grid.treat.shape
    vals = np.full((N, P, len(terms)), np.nan)
    for k, (name, lag) in enumerate(terms):
        col = grid.columns[name]
        if lag == 0:
            vals[:, :, k] = col
        elif lag < P:
            vals[:, lag:, k] = col[:, : P - lag]
    miss = np.isnan(vals)
    return np.concatenate([np.where(miss, 0.0, vals), miss.astype(float)], axis=2)


def _drop_degenerate(M: np.ndarray) -> np.ndarray:
    """Columns of ``M`` that are not constant and not a copy of an earlier one."""
    keep = []
    for j in range(M.shape[1]):
        col = M[:, j]
        if np.all(col == col[0]):
            continue
        if any(np.array_equal(col, M[:, k]) for k in keep):
            continue
        keep.append(j)
    return np.asarray(keep, dtype=int)


_GECON = None


def _solve_fails(A: np.ndarray) -> bool:
    """Whether ``solve(A)`` stops in R: an exactly singular LU factor, or a
    1-norm reciprocal condition estimate below machine epsilon."""
    global _GECON
    if _GECON is None:
        (_GECON,) = get_lapack_funcs(("gecon",), (A,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lu, piv = lu_factor(A, check_finite=False)
    if np.any(np.diag(lu) == 0):
        return True
    anorm = float(np.max(np.sum(np.abs(A), axis=0)))
    rcond, _ = _GECON(lu, anorm, norm="1")
    return bool(rcond < np.finfo(float).eps)


def _period_distance(X: np.ndarray, x0: np.ndarray, diagonal: bool) -> np.ndarray:
    """Mahalanobis distance of each row of ``X`` (controls) from ``x0``."""

    def cov_of(M):
        if diagonal:
            return np.diag(M.var(axis=0, ddof=1))
        return np.atleast_2d(np.cov(M, rowvar=False, ddof=1))

    S = cov_of(X)
    if abs(np.linalg.det(S)) <= 1e-5:
        keep = _drop_degenerate(X)
        if 0 < keep.size < X.shape[1]:
            X, x0 = X[:, keep], x0[keep]
            S = cov_of(X)
    diff = X - x0[None, :]
    if _solve_fails(S):
        full = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
        Q = np.linalg.pinv(full, rcond=np.sqrt(np.finfo(float).eps))
    else:
        Q = np.linalg.solve(S, np.eye(S.shape[0]))
    d2 = np.sum((diff @ Q) * diff, axis=1)
    with np.errstate(invalid="ignore"):
        return np.sqrt(d2)


def _nearest_weights(dist: np.ndarray, size: int, rounding: int | None) -> np.ndarray:
    """Equal weight on the ``size`` closest controls; ties at the cutoff
    bring every tied control in."""
    n = dist.size
    if n == 1:
        return np.ones(1)
    if n < size:
        return np.full(n, 1.0 / n)
    beat = np.max(np.sort(dist)[: size + 1])
    d = dist
    if rounding is not None:
        d, beat = np.round(dist, rounding), np.round(beat, rounding)
    if np.sum(d < beat) < size:
        sel = d <= beat
    else:
        sel = d < beat
    return np.where(sel, 1.0 / sel.sum(), 0.0)


def _mahalanobis(sets, C, lag, size, diagonal):
    for s in sets:
        if s.controls.size == 1:
            s.weights = np.ones(1)
            continue
        per = [
            _period_distance(C[s.controls, p, :], C[s.unit, p, :], diagonal)
            for p in range(s.period - lag, s.period + 1)
        ]
        s.weights = _nearest_weights(np.mean(per, axis=0), size, _ROUND)


def _logit_glm(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logit MLE by iteratively reweighted least squares, started and
    stopped as R ``glm.fit`` (deviance change below 1e-8)."""
    mu = (y + 0.5) / 2.0
    eta = np.log(mu / (1 - mu))
    dev_old = np.inf
    beta = np.zeros(X.shape[1])
    for _ in range(25):
        var = mu * (1 - mu)
        z = eta + (y - mu) / var
        w = np.sqrt(var)
        beta, *_ = np.linalg.lstsq(X * w[:, None], z * w, rcond=None)
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        dev = -2.0 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
        if abs(dev - dev_old) / (abs(dev) + 0.1) < 1e-8:
            break
        dev_old = dev
    else:
        warnings.warn(
            "panel_match: the propensity-score logit did not converge in 25 "
            "iterations (separation?).",
            RuntimeWarning,
            stacklevel=3,
        )
    return beta


def _propensity(sets, C, D, method):
    """Fit the pooled period-t score; return per-set (control, treated) scores."""
    rows: Dict[Tuple[int, int], None] = {}
    for s in sets:
        for u in list(s.controls) + [s.unit]:
            rows.setdefault((int(u), s.period), None)
    keys = list(rows)
    units = np.array([k[0] for k in keys])
    times = np.array([k[1] for k in keys])
    Xp = C[units, times, :]
    yp = D[units, times]
    keep = _drop_degenerate(Xp)
    Xp = Xp[:, keep]
    design = np.column_stack([np.ones(len(yp)), Xp])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise DataInsufficient(
            "panel_match: the propensity-score covariates are linearly "
            "dependent on the pooled treatment-period rows.",
            recovery_hint="Drop redundant covariates or lags.",
            diagnostics={"rows": int(len(yp)), "columns": int(design.shape[1])},
        )
    if method.startswith("cbps"):
        from ..matching.cbps import _fit_cbps

        beta, *_ = _fit_cbps(design, yp.astype(float), "ATT", "over")
    else:
        beta = _logit_glm(design, yp.astype(float))
    out = []
    for s in sets:
        Xs = C[np.append(s.controls, s.unit), s.period, :][:, keep]
        eta = np.column_stack([np.ones(len(Xs)), Xs]) @ beta
        out.append(1.0 - 1.0 / (1.0 + np.exp(eta)))
    return out


def refine(
    sets: List[MatchedSet],
    grid: Grid,
    *,
    method: str,
    C: np.ndarray | None,
    lag: int,
    size_match: int,
    diagonal: bool,
    D: np.ndarray,
) -> None:
    """Attach ``weights`` to every set in place."""
    if method == "none":
        for s in sets:
            s.weights = np.full(s.controls.size, 1.0 / s.controls.size)
        return
    if method == "mahalanobis":
        _mahalanobis(sets, C, lag, size_match, diagonal)
        return
    scores = _propensity(sets, C, D, method)
    for s, ps in zip(sets, scores):
        ctrl, treated = ps[:-1], ps[-1]
        if ctrl.size == 1:
            s.weights = np.ones(1)
        elif method.endswith("_weight"):
            odds = ctrl / (1.0 - ctrl)
            tot = odds.sum()
            s.weights = odds / tot if tot > 0 else np.full(ctrl.size, 1.0 / ctrl.size)
        else:
            s.weights = _nearest_weights(np.abs(treated - ctrl), size_match, None)
