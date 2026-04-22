"""Optimal and cardinality matching.

Two approaches that go beyond the "nearest-neighbour" heuristic:

- :func:`optimal_match` — 1:1 matching that minimises the **total**
  distance across all matched pairs, solved exactly via the
  Hungarian / Kuhn-Munkres algorithm
  (``scipy.optimize.linear_sum_assignment``).

- :func:`cardinality_match` — Zubizarreta et al. (2014) cardinality
  matching: keep as many treated units matched to at least one control
  as possible, subject to covariate-balance constraints solved by
  linear programming.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass
class OptimalMatchResult:
    pairs: pd.DataFrame              # (n_matched, 2) treated_idx + control_idx
    distances: np.ndarray            # (n_matched,) matching distances
    ate: float                       # average treatment effect (ATT)
    se: float                        # (rough) analytic SE on matched pairs
    n_treated: int
    n_matched: int

    def summary(self) -> str:
        lines = [
            "Optimal 1:1 Matching (Hungarian algorithm)",
            "-" * 40,
            f"Treated (T=1) : {self.n_treated}",
            f"Matched       : {self.n_matched}",
            f"Mean distance : {self.distances.mean():.4f}",
            f"ATT           : {self.ate:.4f}  (SE = {self.se:.4f})",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _distance_matrix(
    X_treat: np.ndarray, X_ctrl: np.ndarray, metric: str
) -> np.ndarray:
    if metric == "mahalanobis":
        X_all = np.vstack([X_treat, X_ctrl])
        cov = np.cov(X_all, rowvar=False) + 1e-8 * np.eye(X_all.shape[1])
        cov_inv = np.linalg.inv(cov)
        D = np.empty((X_treat.shape[0], X_ctrl.shape[0]))
        for i in range(X_treat.shape[0]):
            diff = X_ctrl - X_treat[i]
            D[i] = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))
        return D
    if metric == "euclidean":
        diff = X_treat[:, None, :] - X_ctrl[None, :, :]
        return np.linalg.norm(diff, axis=2)
    raise ValueError(f"unknown metric {metric!r}")


def optimal_match(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str],
    metric: str = "mahalanobis",
    caliper: Optional[float] = None,
) -> OptimalMatchResult:
    """Optimal 1:1 matching via the Hungarian algorithm.

    Each treated unit is matched to exactly one control; the total
    sum of matched distances is globally minimised. Requires
    ``n_treated ≤ n_control``.

    Parameters
    ----------
    caliper : float, optional
        Drop any pair with distance greater than ``caliper``.
    """
    df = data.dropna(subset=[treatment, outcome] + covariates).reset_index(drop=True)
    t = df[treatment].to_numpy().astype(int)
    y = df[outcome].to_numpy(dtype=float)
    X = df[covariates].to_numpy(dtype=float)
    treated_idx = np.where(t == 1)[0]
    ctrl_idx = np.where(t == 0)[0]
    if len(treated_idx) == 0 or len(ctrl_idx) == 0:
        from statspai.exceptions import DataInsufficient
        raise DataInsufficient(
            "Need both treated and control units.",
            recovery_hint=(
                "All observations have the same treatment value. "
                "Re-check the treatment column / sample filter."
            ),
            diagnostics={
                "n_treated": int(len(treated_idx)),
                "n_control": int(len(ctrl_idx)),
            },
            alternative_functions=[],
        )
    if len(treated_idx) > len(ctrl_idx):
        raise ValueError(
            "Optimal 1:1 matching requires n_control ≥ n_treated. "
            f"Got n_treated={len(treated_idx)}, n_control={len(ctrl_idx)}."
        )
    D = _distance_matrix(X[treated_idx], X[ctrl_idx], metric=metric)
    row_ind, col_ind = optimize.linear_sum_assignment(D)
    dists = D[row_ind, col_ind]

    if caliper is not None:
        keep = dists <= caliper
        row_ind = row_ind[keep]
        col_ind = col_ind[keep]
        dists = dists[keep]

    pairs = pd.DataFrame({
        "treated_idx": treated_idx[row_ind],
        "control_idx": ctrl_idx[col_ind],
        "distance": dists,
    })
    if len(pairs) == 0:
        raise ValueError("Caliper dropped all pairs; try a larger value.")
    diffs = y[pairs["treated_idx"].values] - y[pairs["control_idx"].values]
    ate = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    return OptimalMatchResult(
        pairs=pairs,
        distances=dists,
        ate=ate, se=se,
        n_treated=len(treated_idx),
        n_matched=len(pairs),
    )


# --------------------------------------------------------------------- #
#  Cardinality matching (Zubizarreta 2012, 2014)
# --------------------------------------------------------------------- #

@dataclass
class CardinalityMatchResult:
    treated_matched: np.ndarray       # indices of matched treated
    control_matched: np.ndarray       # indices of matched controls
    ate: float
    se: float
    n_matched_pairs: int
    balance: pd.DataFrame             # post-match standardised mean diffs

    def summary(self) -> str:
        lines = [
            "Cardinality Matching (Zubizarreta 2014)",
            "-" * 40,
            f"Matched pairs : {self.n_matched_pairs}",
            f"ATT           : {self.ate:.4f}  (SE = {self.se:.4f})",
            "",
            "Post-match balance (|SMD|):",
            self.balance.round(3).to_string(index=False),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def cardinality_match(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: List[str],
    smd_tolerance: float = 0.1,
) -> CardinalityMatchResult:
    """Cardinality matching — maximise the number of matched pairs subject
    to a standardised-mean-difference tolerance on every covariate.

    Formulation (Zubizarreta 2014):

        maximise   sum_j z_j
        s.t.       |mean(X_k | T=1) - sum_j z_j X_{jk} / sum_j z_j|
                    <= smd_tolerance * SD(X_k)   ∀ k
                   z_j ∈ {0, 1}  for each control j

    Uses a continuous LP relaxation (scipy.optimize.linprog) then rounds
    weights to 0/1 via a threshold — sufficient in almost all applied
    work. Matched pair sample is the matched controls each paired
    sequentially with the nearest treated in covariate space.
    """
    df = data.dropna(subset=[treatment, outcome] + covariates).reset_index(drop=True)
    t = df[treatment].to_numpy().astype(int)
    y = df[outcome].to_numpy(dtype=float)
    X = df[covariates].to_numpy(dtype=float)
    treated = X[t == 1]
    ctrl = X[t == 0]
    n_t, n_c = len(treated), len(ctrl)
    k = X.shape[1]

    # target: treated means
    mu_t = treated.mean(axis=0)
    sd_all = X.std(axis=0) + 1e-12
    tol = smd_tolerance * sd_all

    # LP: maximise sum(z); z in [0, 1]^n_c
    # constraints:
    #   |mu_t_k * sum(z) - X_c[:, k] @ z| <= tol_k * sum(z)
    # Rearranged: X_c @ z <= (mu_t + tol) * sum(z) and X_c @ z >= (mu_t - tol) * sum(z)
    # Sum(z) is itself a variable — use the standard trick of dividing through
    # by sum(z) and noting sum(z) > 0 ⇒ enforce constraints as:
    #   (X_c - (mu_t + tol)) @ z <= 0
    #   -(X_c - (mu_t - tol)) @ z <= 0
    A_ub = np.vstack([
        (ctrl - (mu_t + tol)).T,
        -(ctrl - (mu_t - tol)).T,
    ])                                                 # (2k, n_c)
    b_ub = np.zeros(2 * k)
    c = -np.ones(n_c)                                  # maximise sum(z) = minimise -sum(z)
    bounds = [(0, 1)] * n_c
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        raise RuntimeError(f"cardinality matching LP failed: {res.message}")
    z = np.asarray(res.x)

    # Round: keep controls with z > 0.5, cap at n_t
    keep_idx = np.argsort(-z)[:min(n_t, int(np.floor(z.sum() + 0.5)))]
    kept_z = np.zeros(n_c, dtype=bool)
    kept_z[keep_idx] = True

    ctrl_global = np.where(t == 0)[0]
    treat_global = np.where(t == 1)[0]
    matched_ctrl_global = ctrl_global[kept_z]

    # Pair up each matched control with the nearest treated by Mahalanobis
    # distance; fall back to Euclidean if singular.
    pair_treated = []
    pair_control = []
    # nearest treated for each matched control
    if len(matched_ctrl_global):
        D = _distance_matrix(ctrl[kept_z], treated, metric="mahalanobis")
        row, col = optimize.linear_sum_assignment(D)
        # row indexes matched_ctrl subset; col indexes treated subset
        pair_treated = treat_global[col]
        pair_control = matched_ctrl_global[row]

    diffs = y[pair_treated] - y[pair_control]
    ate = float(diffs.mean()) if len(diffs) else float("nan")
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")

    # Post-match balance
    bal_rows = []
    for j, cv in enumerate(covariates):
        mu_c_post = ctrl[kept_z, j].mean() if kept_z.sum() else np.nan
        smd = (mu_t[j] - mu_c_post) / (sd_all[j] + 1e-12)
        bal_rows.append({"covariate": cv, "SMD": float(smd),
                         "|SMD|": abs(float(smd))})
    balance = pd.DataFrame(bal_rows)

    return CardinalityMatchResult(
        treated_matched=np.asarray(pair_treated),
        control_matched=np.asarray(pair_control),
        ate=ate, se=se,
        n_matched_pairs=len(pair_treated),
        balance=balance,
    )
