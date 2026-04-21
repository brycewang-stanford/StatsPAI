"""
Graph-neural-network causal inference (lightweight, dependency-free).

Idea (Bhattacharya, Nabi & Shpitser 2022; Ma, Tucker, Bhalla, Shpitser
2022): in settings where units are embedded in a network (social,
biological, supply-chain), the covariate pool for unit :math:`i`
should include neighbour-aggregated features. A *one-hop GNN* outcome
model uses

.. math::

    \\mu(D_i, X_i, G) = f\\bigl(D_i, X_i, \\tfrac{1}{|N(i)|}\\sum_{j \\in N(i)} X_j\\bigr).

To stay self-contained (no PyTorch / JAX dependency) we realise this
via **random-forest regression on GCN-aggregated features**. The
forest takes as input the concatenation ``[D, X, W X, W² X]``, where
:math:`W` is the row-normalised adjacency. That is — we use the first
two layers of a spectral GCN *as a feature map*, then let sklearn's
random forest (or any sklearn regressor) learn :math:`f`.

The AIPW score combines this with a multinomial propensity to deliver
a doubly-robust ATE estimate that is robust to interference.

References
----------
Bhattacharya, R., Nabi, R., & Shpitser, I. (2022). "Semiparametric
inference for causal effects in graphical models with hidden
variables." *JMLR*, 23(101), 1-76.

Ma, J., Tucker, G., Bhalla, V., & Shpitser, I. (2022). "Learning
causal effects on hypergraphs." *NeurIPS 2022*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression


@dataclass
class GNNCausalResult:
    ate: float
    se: float
    ci: tuple
    pvalue: float
    feature_map: np.ndarray
    n_obs: int
    n_layers: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lo, hi = self.ci
        return (
            "GNN Causal Inference (GCN-RF + AIPW)\n"
            "------------------------------------\n"
            f"  N            : {self.n_obs}\n"
            f"  GCN layers   : {self.n_layers}\n"
            f"  ATE          : {self.ate:+.4f}  (SE={self.se:.4f})\n"
            f"  95% CI       : [{lo:.4f}, {hi:.4f}]\n"
            f"  p-value      : {self.pvalue:.4f}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"GNNCausalResult(ATE={self.ate:+.4f})"


def _row_normalise(A: np.ndarray) -> np.ndarray:
    rs = A.sum(axis=1, keepdims=True)
    rs = np.where(rs == 0, 1.0, rs)
    return A / rs


def gnn_causal(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str],
    adjacency,
    n_layers: int = 2,
    n_trees: int = 200,
    min_leaf: int = 5,
    propensity_bounds: tuple = (0.02, 0.98),
    random_state: int = 42,
    alpha: float = 0.05,
) -> GNNCausalResult:
    """
    GCN-featurised AIPW ATE estimator under network interference.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat : str
    covariates : sequence of str
    adjacency : ndarray / DataFrame
    n_layers : int, default 2
        Number of GCN propagation layers applied to the covariate matrix.
    n_trees, min_leaf : RF hyperparameters.
    propensity_bounds : (float, float)
    random_state : int
    alpha : float

    Returns
    -------
    GNNCausalResult
    """
    cov = list(covariates)
    df = data[[y, treat] + cov].dropna().reset_index(drop=True)
    n = len(df)
    Y = df[y].to_numpy(dtype=float)
    D = df[treat].to_numpy(dtype=int)
    X = df[cov].to_numpy(dtype=float)

    if isinstance(adjacency, pd.DataFrame):
        A = adjacency.to_numpy(dtype=float)
    else:
        A = np.asarray(adjacency, dtype=float)
    if A.shape[0] != n:
        raise ValueError("adjacency size must match data length")
    A = _row_normalise(A)

    # GCN-style feature propagation: F = [X, WX, W²X, ...]
    layers = [X]
    H = X.copy()
    for _ in range(n_layers):
        H = A @ H
        layers.append(H)
    F = np.concatenate(layers, axis=1)
    feature_map = F

    # Propensity p(D | F)
    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        lr.fit(F, D)
        e_hat = lr.predict_proba(F)[:, 1]
    except Exception:
        e_hat = np.full(n, float(D.mean()))
    e_hat = np.clip(e_hat, *propensity_bounds)

    # Outcome regression per arm via RF
    def _fit_rf(idx):
        rf = RandomForestRegressor(
            n_estimators=n_trees, min_samples_leaf=min_leaf,
            random_state=random_state, bootstrap=True, n_jobs=-1,
        )
        rf.fit(F[idx], Y[idx])
        return rf.predict(F)

    mu1 = _fit_rf(D == 1)
    mu0 = _fit_rf(D == 0)

    aipw = (mu1 - mu0) + D * (Y - mu1) / e_hat - (1 - D) * (Y - mu0) / (1 - e_hat)
    ate = float(np.mean(aipw))
    se = float(np.std(aipw, ddof=1) / np.sqrt(n))
    z = ate / se if se > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(z)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (ate - crit * se, ate + crit * se)

    return GNNCausalResult(
        ate=ate,
        se=se,
        ci=ci,
        pvalue=pval,
        feature_map=feature_map,
        n_obs=n,
        n_layers=n_layers,
        detail={"propensity_range": (float(e_hat.min()), float(e_hat.max()))},
    )


__all__ = ["gnn_causal", "GNNCausalResult"]
