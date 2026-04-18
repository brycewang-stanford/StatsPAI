"""
Shared primitives for linear and generalized linear mixed models.

Covers:
    * Formula / column parsing and per-group matrix pre-computation.
    * Parameterisation of random-effect covariance matrices
      (identity / diagonal / unstructured) via a packed theta vector.
    * Marginal covariance V_j = Z_j G Z_j' + Phi_j and its log-determinant
      through a Cholesky factorisation.
    * Pseudo-data stacking utilities used by both the LMM GLS step and
      the Laplace-approximation inner loop of the GLMM.

Kept deliberately free of plotting / result-object code so it can be
reused from ``lmm.py`` and ``glmm.py`` without creating cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Grouping utilities
# ---------------------------------------------------------------------------

def _as_str_list(x) -> List[str]:
    """Normalise a column spec to a list of strings."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _build_group_keys(df: pd.DataFrame, groups: Sequence[str]) -> np.ndarray:
    """
    Return an (n,) array of hashable keys identifying the finest
    group crossing for *nested* multi-level designs.

    ``groups`` are ordered outermost → innermost, e.g.
    ``["school", "class"]`` means classes are nested within schools and
    the returned key identifies a single class uniquely.
    """
    if len(groups) == 0:
        raise ValueError("at least one grouping variable is required")
    if len(groups) == 1:
        return df[groups[0]].values
    # Tuple-based key preserves hierarchical identity across levels.
    return pd.MultiIndex.from_frame(df[list(groups)]).values


# ---------------------------------------------------------------------------
# Covariance parameterisation
# ---------------------------------------------------------------------------

def _n_cov_params(q: int, cov_type: str) -> int:
    """Number of free parameters for the chosen covariance structure."""
    if cov_type == "identity":
        return 1
    if cov_type == "diagonal":
        return q
    if cov_type == "unstructured":
        return q * (q + 1) // 2
    raise ValueError(f"unknown cov_type {cov_type!r}")


def _unpack_G(theta: np.ndarray, q: int, cov_type: str) -> np.ndarray:
    """
    Rebuild a q×q PSD covariance matrix from its packed parameters.

    The parameterisation keeps all structures **unconstrained**, so
    numerical optimisation does not need box constraints:

    * identity     — single log-variance σ²   applied to I_q.
    * diagonal     — q log-variances.
    * unstructured — lower-triangular Cholesky L with log-diagonal,
                     G = L L'.
    """
    if cov_type == "identity":
        s2 = np.exp(theta[0])
        return s2 * np.eye(q)

    if cov_type == "diagonal":
        return np.diag(np.exp(theta[:q]))

    if cov_type == "unstructured":
        L = np.zeros((q, q))
        idx = 0
        for i in range(q):
            for j in range(i + 1):
                if i == j:
                    L[i, j] = np.exp(theta[idx])
                else:
                    L[i, j] = theta[idx]
                idx += 1
        return L @ L.T

    raise ValueError(f"unknown cov_type {cov_type!r}")


def _initial_theta(q: int, cov_type: str, s2_init: float) -> np.ndarray:
    """Sensible starting values for the covariance parameters."""
    # Aim for a small random-effect variance (~10% of residual var) at
    # the start; this is conservative and easy to escape from.
    log_s2 = np.log(max(s2_init, 1e-6))

    if cov_type == "identity":
        return np.array([log_s2], dtype=float)

    if cov_type == "diagonal":
        return np.full(q, log_s2, dtype=float)

    if cov_type == "unstructured":
        theta = np.zeros(_n_cov_params(q, "unstructured"))
        idx = 0
        for i in range(q):
            for j in range(i + 1):
                if i == j:
                    theta[idx] = 0.5 * log_s2  # diag of L
                idx += 1
        return theta

    raise ValueError(f"unknown cov_type {cov_type!r}")


# ---------------------------------------------------------------------------
# Per-group pre-processed block
# ---------------------------------------------------------------------------

@dataclass
class _GroupBlock:
    """
    Pre-computed matrices for a single level-2 group.

    These are expensive to construct — touching pandas for each group on
    every likelihood evaluation would be prohibitive.  We do it once and
    hand the raw numpy arrays to the hot loop.
    """
    key: object
    y: np.ndarray               # (n_j,)
    X: np.ndarray               # (n_j, p)
    Z: np.ndarray               # (n_j, q)  (intercept-first ordering)
    n: int

    def V(self, G: np.ndarray, sigma2: float) -> np.ndarray:
        """Marginal covariance V_j = Z_j G Z_j' + sigma² I."""
        return self.Z @ G @ self.Z.T + sigma2 * np.eye(self.n)


def _group_blocks(
    df: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    x_random: Sequence[str],
    group_col_name: str,
) -> Tuple[List[_GroupBlock], List[str], List[str]]:
    """
    Split a dataframe into per-group numeric blocks.

    Returns the list of blocks plus the column-name vectors for fixed
    and random effects (with the intercept name prepended).  An explicit
    column named ``__intercept__`` is expected to exist in ``df``.
    """
    fixed_names = ["_cons"] + list(x_fixed)
    random_names = ["_cons"] + list(x_random)

    blocks: List[_GroupBlock] = []
    for key, sub in df.groupby(group_col_name, sort=False):
        y_j = sub[y].to_numpy(dtype=float)
        X_j = sub[["__intercept__"] + list(x_fixed)].to_numpy(dtype=float)
        if x_random:
            Z_j = sub[["__intercept__"] + list(x_random)].to_numpy(dtype=float)
        else:
            Z_j = sub[["__intercept__"]].to_numpy(dtype=float)
        blocks.append(_GroupBlock(key=key, y=y_j, X=X_j, Z=Z_j, n=len(sub)))
    return blocks, fixed_names, random_names


# ---------------------------------------------------------------------------
# Marginal-covariance Cholesky helper
# ---------------------------------------------------------------------------

def _solve_V(V: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve V x = B via Cholesky and return (x, log|V|).

    Raises ``np.linalg.LinAlgError`` if V is not positive-definite.
    """
    L = np.linalg.cholesky(V)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    # solve L z = B, then L' x = z
    z = np.linalg.solve(L, B)
    x = np.linalg.solve(L.T, z)
    return x, logdet


# ---------------------------------------------------------------------------
# Formula helpers (intercept handling)
# ---------------------------------------------------------------------------

def _prepare_frame(
    data: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    group_cols: Sequence[str],
    x_random: Optional[Sequence[str]] = None,
    weights: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validate columns and return a cleaned copy with ``__intercept__``
    already attached.
    """
    all_cols = [y] + list(x_fixed) + list(group_cols)
    if x_random:
        all_cols.extend(x_random)
    if weights:
        all_cols.append(weights)
    missing = [c for c in all_cols if c not in data.columns]
    if missing:
        raise KeyError(f"missing columns in data: {missing}")
    # Dedup while preserving order — random-slope variables frequently
    # also appear in ``x_fixed``, and pandas creates duplicated columns
    # when asked to index a column twice.
    seen: set = set()
    unique_cols: List[str] = []
    for c in all_cols:
        if c not in seen:
            unique_cols.append(c)
            seen.add(c)
    df = data[unique_cols].dropna().copy()
    df["__intercept__"] = 1.0
    return df


__all__ = [
    "_as_str_list",
    "_build_group_keys",
    "_n_cov_params",
    "_unpack_G",
    "_initial_theta",
    "_GroupBlock",
    "_group_blocks",
    "_solve_V",
    "_prepare_frame",
]
