"""Refinement of matched sets for ``sp.panel_match``.

Weights within each matched set (Imai, Kim and Wang 2023, eqs. 12-17):

* ``none``: equal weights over the set.
* ``mahalanobis``: average over t-L .. t of the Mahalanobis distance
  between the treated observation and each control (covariance of the
  set's controls in that period); the ``size_match`` closest controls
  (ties kept) share equal weight.
* ``ps_match`` / ``cbps_match``: the same with the distance
  ``|e_i - e_c|`` between propensity scores (probability scale, as
  PanelMatch 3.1.5 does; the paper's eq. 14 writes it on the logit
  scale) fitted on the treated and control observations of all sets at
  their period t.
* ``ps_weight`` / ``cbps_weight``: weights proportional to the odds
  ``e / (1 - e)`` of the controls.

Conventions observed on PanelMatch 3.1.5 used as a black box (outputs on
our fixtures only) are listed in ``docs/dev/modules/did.md``.

Clean-room module: no PanelMatch source was consulted.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import lapack

from ..exceptions import ConvergenceFailure
from ._pmatch_sets import MatchedSet

_EPS = np.finfo(float).eps


def equal_weights(sets: List[MatchedSet]) -> None:
    for s in sets:
        s.weights = (
            np.full(s.controls.size, 1.0 / s.controls.size)
            if not s.empty
            else np.zeros(0)
        )


def _keep_closest(dist: np.ndarray, size: int) -> np.ndarray:
    """Equal weights over the ``size`` smallest distances, ties kept."""
    cut = np.sort(dist)[size - 1]
    keep = dist <= cut
    w: np.ndarray = keep / keep.sum()
    return w


def _inverse(S: np.ndarray) -> np.ndarray:
    """Inverse by LU; pseudo-inverse when LU is singular or the reciprocal
    1-norm condition number is below machine epsilon."""
    if S.shape[0] == 0:
        return S
    lu, piv, info = lapack.dgetrf(S)
    if info == 0:
        anorm = np.abs(S).sum(axis=0).max()
        rcond, cinfo = lapack.dgecon(lu, anorm, norm="1")
        if cinfo == 0 and rcond >= _EPS:
            return np.linalg.solve(S, np.eye(S.shape[0]))
    return np.linalg.pinv(S, rcond=np.sqrt(_EPS))


def _reduce_columns(X: np.ndarray) -> np.ndarray:
    """Column indices that are neither constant nor duplicates of an
    earlier column (used when the covariance is near singular)."""
    keep: List[int] = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.all(col == col[0]):
            continue
        if any(np.array_equal(col, X[:, k]) for k in keep):
            continue
        keep.append(j)
    return np.asarray(keep, dtype=int)


def _period_distance(X: np.ndarray, x0: np.ndarray, diagonal: bool) -> np.ndarray:
    S = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
    if diagonal:
        S = np.diag(np.diag(S))
    if abs(np.linalg.det(S)) <= 1e-5:
        cols = _reduce_columns(X)
        X, x0 = X[:, cols], x0[cols]
        S = (
            np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
            if cols.size
            else np.zeros((0, 0))
        )
        if diagonal:
            S = np.diag(np.diag(S))
    diff = X - x0
    Si = _inverse(S)
    q = np.einsum("ij,jk,ik->i", diff, Si, diff) if Si.size else np.zeros(X.shape[0])
    dist: np.ndarray = np.sqrt(np.maximum(q, 0.0))
    return dist


def mahalanobis_weights(
    sets: List[MatchedSet], Z: np.ndarray, lag: int, size: int, diagonal: bool
) -> None:
    for s in sets:
        n = s.controls.size
        if n == 0:
            s.weights = np.zeros(0)
            continue
        if n <= size:
            s.weights = np.full(n, 1.0 / n)
            continue
        periods = range(s.t - lag, s.t + 1)
        d = np.mean(
            [
                _period_distance(Z[s.controls, p, :], Z[s.unit, p, :], diagonal)
                for p in periods
            ],
            axis=0,
        )
        s.weights = _keep_closest(np.round(np.asarray(d), 5), size)


# ----------------------------------------------------------------------
# propensity scores
# ----------------------------------------------------------------------


def _independent_columns(X: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Columns kept in order, dropping each one that is (numerically) a
    linear combination of the columns kept before it."""
    keep: List[int] = []
    for j in range(X.shape[1]):
        trial = X[:, keep + [j]]
        sv = np.linalg.svd(trial, compute_uv=False)
        if sv[-1] > tol * max(sv[0], 1.0):
            keep.append(j)
    return np.asarray(keep, dtype=int)


def _logit_irls(
    X: np.ndarray, y: np.ndarray, tol: float = 1e-8, maxit: int = 25
) -> np.ndarray:
    """Logistic MLE by iteratively reweighted least squares, started at
    mu = (y + 0.5) / 2 and stopped when the relative deviance change is
    below ``tol`` (the textbook IRLS recipe with R ``glm``'s documented
    default stopping rule)."""
    mu = (y + 0.5) / 2.0
    eta = np.log(mu / (1 - mu))

    def deviance(m: np.ndarray) -> float:
        m = np.clip(m, 1e-300, 1 - 1e-16)
        return float(-2 * np.sum(y * np.log(m) + (1 - y) * np.log(1 - m)))

    dev_old = deviance(mu)
    beta = np.zeros(X.shape[1])
    for _ in range(maxit):
        w = mu * (1 - mu)
        z = eta + (y - mu) / w
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        dev = deviance(mu)
        if abs(dev - dev_old) / (abs(dev) + 0.1) < tol:
            out: np.ndarray = beta
            return out
        dev_old = dev
    raise ConvergenceFailure(
        "panel_match: the propensity-score logit did not converge",
        recovery_hint=(
            "Check for separation in the refinement covariates or use "
            "refinement='mahalanobis'."
        ),
    )


def _ps_rows(sets: List[MatchedSet]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    units, times, resp = [], [], []
    for s in sets:
        if s.empty:
            continue
        units.append(s.unit)
        times.append(s.t)
        resp.append(1.0)
        units.extend(s.controls.tolist())
        times.extend([s.t] * s.controls.size)
        resp.extend([0.0] * s.controls.size)
    frame = pd.DataFrame({"u": units, "t": times, "r": resp}).drop_duplicates(
        ["u", "t"]
    )
    return frame["u"].to_numpy(), frame["t"].to_numpy(), frame["r"].to_numpy()


def propensity_index(
    sets: List[MatchedSet], Z: np.ndarray, method: str
) -> Tuple[np.ndarray, dict]:
    """Linear index ``logit e`` for every (unit, period) cell used by a
    set (NaN elsewhere) and fit diagnostics."""
    u, t, r = _ps_rows(sets)
    X = np.column_stack([np.ones(u.size), Z[u, t, :]])
    cols = _independent_columns(X)
    if 0 not in cols:  # keep the intercept first
        cols = np.concatenate([[0], cols])
    Xk = X[:, cols]
    info: Dict[str, Any]
    if method == "ps":
        beta = _logit_irls(Xk, r)
        info = {"model": "logit"}
    else:
        from ..matching.cbps import cbps as _cbps

        names = [f"z{j}" for j in range(Xk.shape[1] - 1)]
        frame = pd.DataFrame(Xk[:, 1:], columns=names)
        frame["_r"] = r.astype(int)
        frame["_y"] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = _cbps(frame, "_y", "_r", names, estimand="ATT", n_bootstrap=0)
        beta = np.asarray(fit.model_info["beta"], dtype=float)
        info = {
            "model": "cbps",
            "converged": bool(fit.model_info.get("converged", True)),
        }
    index = np.full(Z.shape[:2], np.nan)
    index[u, t] = Xk @ beta
    info["cells"] = (u, t, 1.0 / (1.0 + np.exp(-index[u, t])))
    return index, info


def ps_weights(sets: List[MatchedSet], index: np.ndarray) -> None:
    for s in sets:
        if s.empty:
            s.weights = np.zeros(0)
            continue
        odds = np.exp(index[s.controls, s.t])
        s.weights = odds / odds.sum()


def ps_match_weights(sets: List[MatchedSet], index: np.ndarray, size: int) -> None:
    for s in sets:
        n = s.controls.size
        if n == 0:
            s.weights = np.zeros(0)
            continue
        if n <= size:
            s.weights = np.full(n, 1.0 / n)
            continue
        e = 1.0 / (1.0 + np.exp(-index[np.append(s.controls, s.unit), s.t]))
        s.weights = _keep_closest(np.abs(e[:-1] - e[-1]), size)


def refine(
    sets: List[MatchedSet],
    method: str,
    Z: Optional[np.ndarray],
    lag: int,
    size: int,
    diagonal: bool,
) -> dict:
    info: dict = {}
    if method == "none" or not any(not s.empty for s in sets):
        equal_weights(sets)
    elif method == "mahalanobis":
        assert Z is not None  # refinement methods are validated with covariates
        mahalanobis_weights(sets, Z, lag, size, diagonal)
    else:
        assert Z is not None  # refinement methods are validated with covariates
        family, how = method.split("_")
        index, info = propensity_index(sets, Z, family)
        if how == "weight":
            ps_weights(sets, index)
        else:
            ps_match_weights(sets, index, size)
    return info
