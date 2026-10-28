"""Non-diagonal ``Omega = E[ee' | w]`` for the efficient recentered instrument.

``sp.recentered_iv(resid_cov=...)`` accepts either a dense ``n x n``
covariance or an equicorrelated within-cluster model built on
``resid_var``. Both are turned into a solver ``M -> Omega^{-1} M`` on the
estimation rows; :func:`efficient_transform_cov` then adds the rank-one
``psi psi'`` term by Sherman-Morrison, as ``efficient_transform`` does for
a diagonal ``Omega``. Nothing here is public.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from .._input_validation import check_columns
from ..exceptions import MethodIncompatibility

Solver = Callable[[np.ndarray], np.ndarray]

_COV_KEYS = {"cluster", "rho"}


def check_resid_cov(resid_cov: Any, has_var: bool, n_rows: int) -> Optional[str]:
    """Validate the form of ``resid_cov``; return ``'dense'`` / ``'equicorrelated'``."""
    if resid_cov is None:
        return None
    if isinstance(resid_cov, Mapping):
        extra = set(resid_cov) - _COV_KEYS
        if extra or not _COV_KEYS <= set(resid_cov):
            raise MethodIncompatibility(
                "resid_cov as a dict must have exactly the keys 'cluster' and "
                f"'rho', got {sorted(resid_cov)}",
                recovery_hint=(
                    "Pass resid_cov={'cluster': 'cz', 'rho': 0.3} with "
                    "resid_var= as the diagonal, or a dense n x n array."
                ),
            )
        if not has_var:
            raise MethodIncompatibility(
                "The equicorrelated resid_cov needs resid_var= for the "
                "diagonal: Omega_ij = rho * sqrt(v_i v_j) within a cluster.",
                recovery_hint="Pass resid_var= (Var[eps_i | w]) as well.",
            )
        rho = resid_cov["rho"]
        if not isinstance(
            rho, (int, float, np.floating, np.integer)
        ) or not np.isfinite(rho):
            raise MethodIncompatibility(
                f"resid_cov['rho'] must be a finite number, got {rho!r}",
                recovery_hint="Use the within-cluster residual correlation, e.g. 0.3.",
            )
        if not -1.0 < float(rho) < 1.0:
            raise MethodIncompatibility(
                f"resid_cov['rho'] must lie in (-1, 1), got {rho!r}",
                recovery_hint="A correlation of +-1 makes Omega singular.",
                diagnostics={"rho": float(rho)},
            )
        if not isinstance(resid_cov["cluster"], str):
            raise MethodIncompatibility(
                "resid_cov['cluster'] must be a column name",
                recovery_hint="Name the column that holds the cluster ids.",
            )
        return "equicorrelated"
    arr = np.asarray(resid_cov)
    if arr.ndim != 2 or arr.shape != (n_rows, n_rows):
        raise MethodIncompatibility(
            f"resid_cov must be an n x n matrix with n = {n_rows} rows of data, "
            f"got shape {arr.shape}",
            recovery_hint=(
                "Pass the full covariance of the residuals over the rows of "
                "data, or resid_cov={'cluster': ..., 'rho': ...}."
            ),
            diagnostics={"shape": list(arr.shape), "n": n_rows},
        )
    if has_var:
        raise MethodIncompatibility(
            "A dense resid_cov already holds the variances on its diagonal; "
            "resid_var= would be a second, possibly conflicting, diagonal.",
            recovery_hint="Drop resid_var=, or use the equicorrelated dict form.",
        )
    return "dense"


def dense_solver(cov: Any, idx: np.ndarray) -> Tuple[Solver, Dict[str, Any]]:
    """Cholesky solver for the rows ``idx`` of a dense covariance."""
    full = np.asarray(cov, dtype=float)
    sub = full[np.ix_(idx, idx)]
    if not np.all(np.isfinite(sub)):
        raise MethodIncompatibility(
            "resid_cov has missing or infinite entries on the estimation rows",
            recovery_hint="Fill every entry of the covariance matrix.",
        )
    scale = float(np.max(np.abs(sub))) if sub.size else 0.0
    asym = float(np.max(np.abs(sub - sub.T))) if sub.size else 0.0
    if asym > 1e-10 * max(scale, 1e-300):
        raise MethodIncompatibility(
            "resid_cov is not symmetric",
            recovery_hint="Pass a covariance matrix, e.g. (C + C.T) / 2.",
            diagnostics={"max_asymmetry": asym, "max_abs": scale},
        )
    sub = 0.5 * (sub + sub.T)
    try:
        factor = cho_factor(sub, lower=True, check_finite=False)
    except LinAlgError as exc:
        raise MethodIncompatibility(
            "resid_cov is not positive definite on the estimation rows, so "
            "E[ee' | w]^{-1} does not exist",
            recovery_hint=(
                "Pass a positive definite covariance (no zero variances, no "
                "perfectly correlated rows); add a small ridge if needed."
            ),
            diagnostics={"error": str(exc)},
        ) from exc

    def solve(M: np.ndarray) -> np.ndarray:
        return np.asarray(cho_solve(factor, M, check_finite=False), dtype=float)

    return solve, {"resid_cov": "dense"}


def equicorrelated_solver(
    data: pd.DataFrame,
    spec: Mapping[str, Any],
    idx: np.ndarray,
    omega: np.ndarray,
) -> Tuple[Solver, Dict[str, Any]]:
    """Closed-form solver for an equicorrelated within-cluster ``Omega``.

    Within cluster ``g`` of size ``m``,
    ``Omega_g = S ((1 - rho) I + rho 1 1') S`` with ``S = diag(sqrt(v))``,
    so ``Omega_g^{-1} x = S^{-1} (u - c 1 1'u) / (1 - rho)`` with
    ``u = S^{-1} x`` and ``c = rho / (1 - rho + m rho)``. Positive
    definite iff ``-1/(m - 1) < rho < 1`` for every cluster.
    """
    col = spec["cluster"]
    check_columns(data, {"resid_cov cluster": col}, function="recentered_iv")
    raw = data[col].to_numpy()[idx]
    if pd.isna(raw).any():
        raise MethodIncompatibility(
            f"Cluster column {col!r} has missing values on the estimation rows",
            recovery_hint="Fill or drop the rows without a cluster id.",
        )
    codes, uniq = pd.factorize(raw, sort=True)
    sizes = np.bincount(codes)
    rho = float(spec["rho"])
    bad = (1.0 - rho + sizes * rho) <= 0
    if np.any(bad):
        m = int(sizes[bad].max())
        raise MethodIncompatibility(
            f"rho = {rho} makes Omega singular or indefinite: a cluster of "
            f"size {m} needs rho > {-1.0 / (m - 1):.6g}",
            recovery_hint="Use a larger rho, or check the cluster ids.",
            diagnostics={"rho": rho, "largest_bad_cluster": m},
        )
    sd = np.sqrt(omega)
    c = rho / (1.0 - rho + sizes * rho)

    def solve(M: np.ndarray) -> np.ndarray:
        X = np.asarray(M, dtype=float)
        vec = X.ndim == 1
        U = X.reshape(len(sd), -1) / sd[:, None]
        G = np.zeros((len(sizes), U.shape[1]))
        np.add.at(G, codes, U)
        out = (U - c[codes, None] * G[codes]) / (1.0 - rho) / sd[:, None]
        return np.asarray(out.ravel() if vec else out)

    info = {
        "resid_cov": "equicorrelated",
        "rho": rho,
        "n_clusters": int(len(uniq)),
    }
    return solve, info


def efficient_transform_cov(
    V: np.ndarray, solve: Solver, psi: Optional[np.ndarray]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """``(Omega + psi psi')^{-1} V`` for a general ``Omega`` given its solver.

    Sherman-Morrison: with ``a = Omega^{-1} psi`` and ``q = psi' a``,
    ``(Omega + psi psi')^{-1} v = Omega^{-1} v - a (a' v) / (1 + q)``.
    Linear in ``V`` and fixed given ``w``, so every shock draw goes
    through the same map and randomization inference stays exact.
    """
    V = np.asarray(V, dtype=float)
    vec = V.ndim == 1
    M = V.reshape(V.shape[0], -1)
    out = solve(M)
    info: Dict[str, float] = {}
    if psi is not None:
        a = solve(psi)
        q = float(psi @ a)
        out = out - np.outer(a, a @ M) / (1.0 + q)
        info["nu"] = q / (1.0 + q)
        info["psi_omega_psi"] = q
    return (out.ravel() if vec else out), info
