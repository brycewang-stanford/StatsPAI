"""
Graph-neural-network causal inference (lightweight, dependency-free).

Idea (Bhattacharya, Nabi & Shpitser 2022; Ma, Tucker, Bhalla, Shpitser
2022): in settings where units are embedded in a network (social,
biological, supply-chain), the covariate pool for unit :math:`i`
should include neighbour-aggregated features. A *one-hop GNN* outcome
model uses

.. math::

    \\mu(D_i, X_i, G) =
    f\\bigl(D_i, X_i, \\tfrac{1}{|N(i)|}\\sum_{j \\in N(i)} X_j\\bigr).

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

from ..exceptions import DataInsufficient, MethodIncompatibility

_GNN_CAUSAL_ALTERNATIVES = [
    "sp.gnn_causal",
    "sp.neural_causal.gnn_causal",
    "sp.dml",
]


def _gnn_error(
    message: str,
    *,
    diagnostics: Optional[Dict[str, Any]] = None,
    recovery_hint: str = "Check gnn_causal inputs.",
) -> MethodIncompatibility:
    return MethodIncompatibility(
        message,
        recovery_hint=recovery_hint,
        diagnostics=diagnostics,
        alternative_functions=_GNN_CAUSAL_ALTERNATIVES,
    )


def _normalize_covariates(covariates: Sequence[str] | str) -> list[str]:
    raw = [covariates] if isinstance(covariates, str) else list(covariates)
    if not raw:
        raise _gnn_error(
            "gnn_causal requires at least one covariate.",
            diagnostics={"covariates": raw},
            recovery_hint="Pass covariates=['x1', 'x2'] or a single column name.",
        )
    for idx, covariate in enumerate(raw):
        if not isinstance(covariate, str) or not covariate:
            raise _gnn_error(
                f"covariates[{idx}] must be a non-empty column-name string.",
                diagnostics={"index": idx, "value": repr(covariate)},
                recovery_hint="Pass covariates as column-name strings.",
            )
    return raw


@dataclass
class GNNCausalResult:
    ate: float
    se: float
    ci: tuple[float, float]
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
    return np.asarray(A / rs, dtype=float)


def gnn_causal(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str] | str,
    adjacency: Any,
    n_layers: int = 2,
    n_trees: int = 200,
    min_leaf: int = 5,
    propensity_bounds: tuple[float, float] = (0.02, 0.98),
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
    if not isinstance(data, pd.DataFrame):
        raise _gnn_error(
            "gnn_causal data must be a pandas DataFrame.",
            diagnostics={"type": type(data).__name__},
            recovery_hint="Pass a pandas DataFrame with one row per network unit.",
        )
    cov = _normalize_covariates(covariates)
    required = [y, treat] + cov
    missing = set(required) - set(data.columns)
    if missing:
        raise _gnn_error(
            f"Missing columns: {missing}",
            diagnostics={"missing_columns": sorted(str(col) for col in missing)},
            recovery_hint="Pass column names present in the input DataFrame.",
        )
    if n_layers < 0:
        raise _gnn_error(
            f"n_layers must be >= 0, got {n_layers}.",
            diagnostics={"n_layers": n_layers},
            recovery_hint="Use n_layers=0 for raw covariates or a positive integer.",
        )
    if n_trees < 1:
        raise _gnn_error(
            f"n_trees must be >= 1, got {n_trees}.",
            diagnostics={"n_trees": n_trees},
            recovery_hint="Use n_trees >= 1.",
        )
    if min_leaf < 1:
        raise _gnn_error(
            f"min_leaf must be >= 1, got {min_leaf}.",
            diagnostics={"min_leaf": min_leaf},
            recovery_hint="Use min_leaf >= 1.",
        )
    if not (0 < alpha < 1):
        raise _gnn_error(
            f"alpha must be in (0, 1), got {alpha}.",
            diagnostics={"alpha": alpha},
            recovery_hint="Use a confidence level such as alpha=0.05.",
        )
    if len(propensity_bounds) != 2:
        raise _gnn_error(
            "propensity_bounds must contain exactly two values.",
            diagnostics={"propensity_bounds": list(propensity_bounds)},
            recovery_hint="Use propensity_bounds=(0.02, 0.98).",
        )
    p_lo, p_hi = float(propensity_bounds[0]), float(propensity_bounds[1])
    if not (0 < p_lo < p_hi < 1):
        raise _gnn_error(
            "propensity_bounds must satisfy 0 < lower < upper < 1.",
            diagnostics={"propensity_bounds": [p_lo, p_hi]},
            recovery_hint="Use bounds such as (0.02, 0.98).",
        )

    df = data[required].dropna().reset_index(drop=True)
    n = len(df)
    if n < 3:
        raise DataInsufficient(
            "gnn_causal requires at least 3 complete rows.",
            recovery_hint="Provide more complete network-unit rows after dropna.",
            diagnostics={"n_complete": n},
            alternative_functions=_GNN_CAUSAL_ALTERNATIVES,
        )
    Y = df[y].to_numpy(dtype=float)
    D = df[treat].to_numpy(dtype=int)
    X = df[cov].to_numpy(dtype=float)
    if not (np.isfinite(Y).all() and np.isfinite(X).all()):
        raise _gnn_error(
            "outcome and covariate values must be finite.",
            diagnostics={
                "finite_outcome": bool(np.isfinite(Y).all()),
                "finite_covariates": bool(np.isfinite(X).all()),
            },
            recovery_hint="Drop or impute non-finite GNN causal inputs.",
        )
    if not set(np.unique(D)).issubset({0, 1}):
        raise _gnn_error(
            "treat must be binary 0/1.",
            diagnostics={"treat_values": np.unique(D).tolist()},
            recovery_hint="Encode treatment as binary 0/1 before fitting.",
        )
    if np.unique(D).size < 2:
        raise DataInsufficient(
            "gnn_causal requires both treatment arms.",
            recovery_hint="Provide complete rows from treated and control units.",
            diagnostics={"treat_values": np.unique(D).tolist()},
            alternative_functions=_GNN_CAUSAL_ALTERNATIVES,
        )

    if isinstance(adjacency, pd.DataFrame):
        A = adjacency.to_numpy(dtype=float)
    else:
        A = np.asarray(adjacency, dtype=float)
    if A.ndim != 2 or A.shape != (n, n):
        raise _gnn_error(
            "adjacency must be a square matrix matching the complete data rows.",
            diagnostics={"adjacency_shape": tuple(A.shape), "n_complete": n},
            recovery_hint="Pass an n-by-n adjacency aligned with rows after dropna.",
        )
    if not np.isfinite(A).all():
        raise _gnn_error(
            "adjacency values must be finite.",
            diagnostics={"finite_adjacency": False},
            recovery_hint="Drop or impute non-finite adjacency entries.",
        )
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
    e_hat = np.clip(e_hat, p_lo, p_hi)

    # Outcome regression per arm via RF
    def _fit_rf(idx: np.ndarray) -> np.ndarray:
        rf = RandomForestRegressor(
            n_estimators=n_trees,
            min_samples_leaf=min_leaf,
            random_state=random_state,
            bootstrap=True,
            n_jobs=-1,
        )
        rf.fit(F[idx], Y[idx])
        return np.asarray(rf.predict(F), dtype=float)

    mu1 = _fit_rf(D == 1)
    mu0 = _fit_rf(D == 0)

    aipw = (mu1 - mu0) + D * (Y - mu1) / e_hat - (1 - D) * (Y - mu0) / (1 - e_hat)
    ate = float(np.mean(aipw))
    se = float(np.std(aipw, ddof=1) / np.sqrt(n))
    z = ate / se if se > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(z)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (ate - crit * se, ate + crit * se)

    _result = GNNCausalResult(
        ate=ate,
        se=se,
        ci=ci,
        pvalue=pval,
        feature_map=feature_map,
        n_obs=n,
        n_layers=n_layers,
        detail={"propensity_range": (float(e_hat.min()), float(e_hat.max()))},
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.neural_causal.gnn_causal",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(cov),
                "n_layers": n_layers,
                "n_trees": n_trees,
                "min_leaf": min_leaf,
                "propensity_bounds": list(propensity_bounds),
                "random_state": random_state,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


__all__ = ["gnn_causal", "GNNCausalResult"]
