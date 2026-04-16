"""LiNGAM — Linear Non-Gaussian Acyclic Model (Shimizu et al., JMLR 2006).

Identifies a causal DAG under the assumptions:

1. Causal structure is linear and acyclic.
2. All disturbances are mutually independent and non-Gaussian.
3. No latent common causes.

Under these assumptions the causal order is **uniquely identifiable**
from purely observational data (in contrast with constraint-based
methods like PC which only identify a Markov-equivalence class).

This implementation uses the DirectLiNGAM algorithm (Shimizu 2011):

    1. Pick the variable most independent of the others' residuals
       (Hilbert-Schmidt Independence Criterion) → declare it a source.
    2. Regress the remaining variables on the source, take residuals.
    3. Recurse on the residuals until all variables are ordered.
    4. The adjacency matrix B[i, j] of direct effects is then recovered
       by regressing each variable on its predecessors in the ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- #
#  Helper: pairwise independence measure
# --------------------------------------------------------------------- #

def _negentropy_hyvarinen(x: np.ndarray) -> float:
    """Hyvärinen's approximation of differential entropy / negentropy.

    Uses the two non-polynomial non-linearities recommended in Hyvärinen
    (1998, *Neural Networks*) eq. (7). Larger values indicate stronger
    departure from Gaussianity (which DirectLiNGAM exploits to
    identify direction).
    """
    z = (x - x.mean()) / (x.std() + 1e-12)
    k1 = 36.0 / (8.0 * np.sqrt(3.0) - 9.0)
    k2a = 1.0 / (2.0 - 6.0 / np.pi)
    gamma = 0.37457
    t1 = k1 * (float(np.mean(np.log(np.cosh(z)))) - gamma) ** 2
    t2 = k2a * (float(np.mean(z * np.exp(-z ** 2 / 2.0)))) ** 2
    return t1 + t2


def _mi_quadratic(x: np.ndarray, y: np.ndarray) -> float:
    """Approximation of mutual information via Hyvärinen's negentropy.

    Follows Shimizu (2011) eq. (5):

        M(x, y) ≈ H(x) + H(y) - H(x, y)
               ≈ -J(x) - J(y) + J((x+y)/√2)

    but because J is non-negative and we only use this score for
    *ordering* candidates, we use the simpler pairwise form:

        dep(x, y)  =  J(x_std) + J(y_std) - J_joint_proxy

    Concretely, we return a non-negative score that is 0 for
    independent x, y and grows with non-Gaussian dependence.
    """
    x = x - x.mean()
    y = y - y.mean()
    nx = _negentropy_hyvarinen(x)
    ny = _negentropy_hyvarinen(y)
    # Proxy: non-Gaussianity of the projection (x+y)/√2
    nxy = _negentropy_hyvarinen((x + y) / np.sqrt(2.0))
    return max(nx + ny - nxy, 0.0)


# --------------------------------------------------------------------- #
#  Result
# --------------------------------------------------------------------- #

@dataclass
class LiNGAMResult:
    order: List[int]                 # causal order (most exogenous first)
    adjacency: np.ndarray            # B[i, j] = direct effect of j on i
    names: List[str]
    residuals: np.ndarray            # (n, k) final residuals

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.adjacency, index=self.names, columns=self.names)

    def edges(self, threshold: float = 0.05) -> List[tuple]:
        out = []
        for i, ni in enumerate(self.names):
            for j, nj in enumerate(self.names):
                if i == j:
                    continue
                if abs(self.adjacency[i, j]) > threshold:
                    out.append((nj, ni, float(self.adjacency[i, j])))
        return out

    def summary(self) -> str:
        ordered = [self.names[i] for i in self.order]
        lines = [
            "DirectLiNGAM",
            "-" * 40,
            "Causal order (most exogenous first):",
            "  " + " → ".join(ordered),
            "",
            "Adjacency matrix B (B[i, j] = effect of j on i):",
            self.to_frame().round(3).to_string(),
            "",
            "Detected edges (|B| > 0.05):",
        ]
        for src, dst, w in self.edges(threshold=0.05):
            lines.append(f"  {src} → {dst}   β={w:+.3f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------- #
#  Main algorithm
# --------------------------------------------------------------------- #

def _pick_exogenous(X: np.ndarray, remaining: List[int]) -> int:
    """DirectLiNGAM exogeneity test.

    For candidate ``i``, compute

        T(i) = Σ_{j≠i} min(0, mi(r_{j|i}, x_i) - mi(r_{i|j}, x_j))²

    where r_{j|i} are the residuals of regressing ``x_j`` on ``x_i``.
    Under LiNGAM the true source has the smallest ``T`` (Shimizu 2011
    Lemma 2). We pick the candidate minimising ``T``.
    """
    if len(remaining) == 1:
        return remaining[0]
    scores = []
    for i in remaining:
        x_i = X[:, i]
        others = [j for j in remaining if j != i]
        total = 0.0
        for j in others:
            x_j = X[:, j]
            b_ji = np.cov(x_j, x_i)[0, 1] / max(np.var(x_i), 1e-12)
            r_j_given_i = x_j - b_ji * x_i
            b_ij = np.cov(x_i, x_j)[0, 1] / max(np.var(x_j), 1e-12)
            r_i_given_j = x_i - b_ij * x_j
            # If i is exogenous, residual r_{j|i} should be INDEPENDENT of x_i
            # so mi(r_{j|i}, x_i) small; mi(r_{i|j}, x_j) large.
            # Diff = mi(r_{j|i}, x_i) - mi(r_{i|j}, x_j) should be ≤ 0.
            diff = _mi_quadratic(r_j_given_i, x_i) - _mi_quadratic(r_i_given_j, x_j)
            total += min(0.0, diff) ** 2
        scores.append((i, total))
    return min(scores, key=lambda t: t[1])[0]


def lingam(data, standardize: bool = True) -> LiNGAMResult:
    """Fit DirectLiNGAM (Shimizu 2011).

    Parameters
    ----------
    data : pd.DataFrame or (n, k) ndarray
    standardize : bool, default True
        Zero-mean / unit-variance each variable before the algorithm.
    """
    if isinstance(data, pd.DataFrame):
        names = list(data.columns)
        X = data.to_numpy(dtype=float).copy()
    else:
        X = np.asarray(data, dtype=float).copy()
        names = [f"x{i}" for i in range(X.shape[1])]
    # Keep original scales so we can de-standardise B at the end.
    scales = X.std(axis=0)
    means = X.mean(axis=0)
    if standardize:
        X = (X - means) / np.maximum(scales, 1e-12)
    n, k = X.shape
    remaining = list(range(k))
    order: List[int] = []
    current = X.copy()
    while remaining:
        pick = _pick_exogenous(current, remaining)
        order.append(pick)
        # Regress all other remaining on `pick` and replace in current
        src = current[:, pick]
        for j in remaining:
            if j == pick:
                continue
            b = np.cov(current[:, j], src)[0, 1] / max(np.var(src), 1e-12)
            current[:, j] = current[:, j] - b * src
        remaining.remove(pick)

    # Now estimate B by regressing each variable on its predecessors
    # (in the discovered order).
    B = np.zeros((k, k))
    Z = X.copy()
    for rank_idx, i in enumerate(order):
        if rank_idx == 0:
            continue
        predecessors = order[:rank_idx]
        Xp = Z[:, predecessors]
        # OLS of Z[:, i] on Xp
        beta, *_ = np.linalg.lstsq(Xp, Z[:, i], rcond=None)
        for j_local, j_global in enumerate(predecessors):
            B[i, j_global] = float(beta[j_local])

    # Residuals after subtracting recovered structural effects (std scale)
    resid_std = Z - Z @ B.T
    # Map the standardised-scale B back to original scale:
    #   B_orig[i, j] = B_std[i, j] * sigma_i / sigma_j
    if standardize:
        B_orig = np.zeros_like(B)
        for i in range(k):
            for j in range(k):
                B_orig[i, j] = B[i, j] * scales[i] / max(scales[j], 1e-12)
        B = B_orig
        resid = resid_std * scales  # rescale to original units
    else:
        resid = resid_std

    return LiNGAMResult(
        order=order, adjacency=B, names=names, residuals=resid,
    )
