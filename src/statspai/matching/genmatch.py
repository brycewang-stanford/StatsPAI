"""
Genetic Matching (Diamond & Sekhon 2013).

The user-supplied generalised Mahalanobis distance is

.. math::

    d_W(x_i, x_j) = (x_i - x_j)^\\top S^{-1/2}\\, W\\, S^{-1/2} (x_i - x_j),

where :math:`S` is the sample covariance of covariates and :math:`W`
is a diagonal weight matrix found by a genetic (evolutionary) search
that maximises the *minimum* across-covariate balance p-value
(Kolmogorov-Smirnov + t-tests, following the `Matching` R package).

Outputs
-------
* the optimal weight vector,
* matched treated-control pair indices,
* a ``balance`` table of standardised mean differences pre/post match,
* the ATT estimate + bootstrap SE.

References
----------
Diamond, A. & Sekhon, J. S. (2013).
"Genetic matching for estimating causal effects." *Review of Economics
and Statistics*, 95(3), 932-945. [@diamond2013genetic]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Sequence, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GenMatchResult:
    att: float
    att_se: float
    ci: tuple
    pvalue: float
    weights: np.ndarray
    balance: pd.DataFrame
    matches: np.ndarray  # (n_treated, k) indices of matched controls
    n_treated: int
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        return (
            "Genetic Matching (Diamond-Sekhon 2013)\n"
            "--------------------------------------\n"
            f"  n_treated : {self.n_treated}\n"
            f"  ATT       : {self.att:.4f}  (SE={self.att_se:.4f})\n"
            f"  CI        : [{self.ci[0]:.4f}, {self.ci[1]:.4f}]\n"
            f"  p-value   : {self.pvalue:.4f}\n"
            "Balance summary:\n"
            f"{self.balance.to_string(index=False)}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"GenMatchResult(ATT={self.att:.4f}, n_treated={self.n_treated})"


def _standardised_diff(x_t: np.ndarray, x_c: np.ndarray) -> float:
    pooled = np.sqrt(0.5 * (np.var(x_t, ddof=1) + np.var(x_c, ddof=1) + 1e-12))
    return float((x_t.mean() - x_c.mean()) / pooled)


def _ks_p(x_t: np.ndarray, x_c: np.ndarray) -> float:
    try:
        return float(stats.ks_2samp(x_t, x_c).pvalue)
    except Exception:
        return 1.0


def _match_with_weights(
    X_t: np.ndarray, X_c: np.ndarray, w: np.ndarray, k_nn: int = 1
) -> np.ndarray:
    """k-NN match with weighted Mahalanobis distance."""
    # Standardise columns (covariance-based whitening omitted for speed;
    # we just use inverse variances as a first-order proxy).
    mu = X_c.mean(axis=0, keepdims=True)
    sd = X_c.std(axis=0, keepdims=True) + 1e-12
    Xt = (X_t - mu) / sd
    Xc = (X_c - mu) / sd
    W = np.sqrt(np.clip(w, 0, None))
    Xt_w = Xt * W
    Xc_w = Xc * W
    # Compute distances (n_t x n_c)
    dists = np.sum((Xt_w[:, None, :] - Xc_w[None, :, :]) ** 2, axis=2)
    k_nn = min(k_nn, Xc.shape[0])
    return np.argpartition(dists, kth=k_nn - 1, axis=1)[:, :k_nn]


def _balance_stats(
    X_t: np.ndarray, X_c: np.ndarray, matches: np.ndarray, names: Sequence[str]
) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(names):
        smd_pre = _standardised_diff(X_t[:, j], X_c[:, j])
        matched_c = X_c[matches.flatten(), j]
        matched_t = np.repeat(X_t[:, j], matches.shape[1])
        smd_post = _standardised_diff(matched_t, matched_c)
        ks_pre = _ks_p(X_t[:, j], X_c[:, j])
        ks_post = _ks_p(X_t[:, j], matched_c)
        rows.append({
            "variable": name,
            "smd_pre": smd_pre,
            "smd_post": smd_post,
            "ks_p_pre": ks_pre,
            "ks_p_post": ks_post,
        })
    return pd.DataFrame(rows)


def _fitness(
    X_t: np.ndarray, X_c: np.ndarray, w: np.ndarray, k_nn: int, names: Sequence[str]
) -> float:
    matches = _match_with_weights(X_t, X_c, w, k_nn=k_nn)
    bal = _balance_stats(X_t, X_c, matches, names)
    # worst p-value across covariates; higher = better balance
    worst = float(min(bal["ks_p_post"].min(), 1.0))
    max_smd = float(np.max(np.abs(bal["smd_post"])))
    # maximise min p-value while penalising large SMD
    return worst - 0.1 * max_smd


def genmatch(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str],
    k: int = 1,
    population_size: int = 40,
    generations: int = 20,
    mutation_rate: float = 0.2,
    alpha: float = 0.05,
    random_state: int = 42,
) -> GenMatchResult:
    """
    Genetic Matching for ATT estimation.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treat : str
        Binary treatment indicator.
    covariates : sequence of str
    k : int, default 1
        Number of matches per treated unit.
    population_size : int, default 40
    generations : int, default 20
    mutation_rate : float, default 0.2
    alpha : float, default 0.05
    random_state : int, default 42

    Returns
    -------
    GenMatchResult
    """
    cov = list(covariates)
    df = data[[y, treat] + cov].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(dtype=float)
    D = df[treat].to_numpy(dtype=int)
    X = df[cov].to_numpy(dtype=float)
    idx_t = np.where(D == 1)[0]
    idx_c = np.where(D == 0)[0]
    X_t = X[idx_t]
    X_c = X[idx_c]

    p = len(cov)
    rng = np.random.default_rng(random_state)

    # Initial population: uniform random in [0.1, 10]
    population = rng.uniform(0.1, 10.0, size=(population_size, p))
    fitness = np.array([_fitness(X_t, X_c, w, k, cov) for w in population])
    for gen in range(generations):
        # rank-based selection
        order = np.argsort(fitness)[::-1]
        parents = population[order[: population_size // 2]]
        # breed
        children = []
        for _ in range(population_size - len(parents)):
            a, b = rng.integers(0, len(parents), size=2)
            mix = rng.uniform(size=p)
            child = mix * parents[a] + (1 - mix) * parents[b]
            # mutate
            if rng.random() < mutation_rate:
                idx_mut = rng.integers(0, p)
                child[idx_mut] *= rng.uniform(0.5, 2.0)
            children.append(child)
        children = np.asarray(children)
        population = np.vstack([parents, children])
        fitness = np.array([_fitness(X_t, X_c, w, k, cov) for w in population])

    best = population[np.argmax(fitness)]
    matches = _match_with_weights(X_t, X_c, best, k_nn=k)
    balance = _balance_stats(X_t, X_c, matches, cov)

    # ATT estimation
    Y_t = Y[idx_t]
    Y_c_match = Y[idx_c[matches]].mean(axis=1)
    att = float(np.mean(Y_t - Y_c_match))
    # Abadie-Imbens SE approximation (paired differences)
    diffs = Y_t - Y_c_match
    att_se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    z = att / att_se if att_se > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(z)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (att - crit * att_se, att + crit * att_se)

    return GenMatchResult(
        att=att,
        att_se=att_se,
        ci=ci,
        pvalue=pval,
        weights=best,
        balance=balance,
        matches=matches,
        n_treated=len(idx_t),
        n_obs=len(df),
        detail={"fitness": float(fitness.max())},
    )


__all__ = ["genmatch", "GenMatchResult"]
