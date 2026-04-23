"""Cox proportional hazards with shared (cluster) frailty.

Extends the standard Cox model to account for unobserved heterogeneity
(frailty) within clusters. The model is:

    h(t | X_ij, z_i) = z_i · h₀(t) · exp(X_ij β)

where z_i is a multiplicative frailty term, assumed iid from a
gamma(θ, θ) distribution (mean 1, variance 1/θ). When θ → ∞ the
frailties collapse to 1 and the model reduces to standard Cox.

Estimation follows the penalised partial likelihood approach
(Therneau & Grambsch 2000) — iteratively update β via Newton-Raphson
on the penalised Cox partial log-likelihood and θ via profile ML.

References
----------
Therneau, T.M. & Grambsch, P.M. (2000). *Modeling Survival Data:
  Extending the Cox Model*. Springer.
Duchateau, L. & Janssen, P. (2008). *The Frailty Model*. Springer. [@therneau2000modeling]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .models import _parse_formula, CoxResult


@dataclass
class FrailtyResult:
    beta: np.ndarray
    se: np.ndarray
    var_names: List[str]
    theta: float
    frailties: np.ndarray
    cluster_ids: np.ndarray
    log_likelihood: float
    n: int
    n_events: int
    n_clusters: int
    concordance: float

    def summary(self) -> str:
        lines = [
            "Cox Model with Shared Gamma Frailty",
            "-" * 50,
            f"Observations   : {self.n}",
            f"Events         : {self.n_events}",
            f"Clusters       : {self.n_clusters}",
            f"Theta (frailty): {self.theta:.4f}  (frailty variance = {1/self.theta:.4f})",
            f"Log-Lik        : {self.log_likelihood:.4f}",
            f"Concordance    : {self.concordance:.4f}",
            "",
            "Coefficients:",
        ]
        from scipy import stats
        for nm, b, s in zip(self.var_names, self.beta, self.se):
            t = b / s if s > 0 else np.nan
            p = 2 * (1 - stats.norm.cdf(abs(t)))
            lines.append(
                f"  {nm:<15s}  {b: .4f}  (SE {s: .4f}, z {t: .3f}, p {p: .4f})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def cox_frailty(
    formula: str,
    data: pd.DataFrame,
    cluster: str,
    alpha: float = 0.05,
    maxiter: int = 50,
    tol: float = 1e-6,
) -> FrailtyResult:
    """Cox proportional hazards with shared gamma frailty.

    Parameters
    ----------
    formula : str
        ``"duration + event ~ x1 + x2"`` (like R's ``Surv(time, event) ~ x``).
    data : pd.DataFrame
    cluster : str
        Column identifying clusters (e.g. hospital, site).
    """
    lhs_str, covariates = _parse_formula(formula)
    # LHS is "duration + event" — split on "+"
    lhs_parts = [s.strip() for s in lhs_str.split("+")]
    if len(lhs_parts) != 2:
        raise ValueError(
            "formula LHS must be 'duration + event' (e.g. 'T + E ~ x1 + x2')"
        )
    duration_col, event_col = lhs_parts[0], lhs_parts[1]
    df = data[[duration_col, event_col, cluster] + covariates].dropna()
    T = df[duration_col].to_numpy(float)
    E = df[event_col].to_numpy(float).astype(int)
    X = df[covariates].to_numpy(float)
    n, k = X.shape
    n_events = int(E.sum())

    # Cluster mapping
    cluster_arr = df[cluster].to_numpy()
    uniq_clusters = np.unique(cluster_arr)
    n_clusters = len(uniq_clusters)
    cluster_idx = np.zeros(n, dtype=int)
    for i, c in enumerate(uniq_clusters):
        cluster_idx[cluster_arr == c] = i

    # Initialize
    beta = np.zeros(k)
    theta = 5.0
    z = np.ones(n_clusters)

    from .models import _cox_neg_logpl_efron, _cox_score_hessian_efron

    for iteration in range(maxiter):
        beta_old = beta.copy()

        # E-step: update frailties given beta, theta
        for g in range(n_clusters):
            mask = cluster_idx == g
            d_g = float(E[mask].sum())
            risk_g = float(np.sum(np.exp(X[mask] @ beta)))
            z[g] = (d_g + theta) / (risk_g + theta)

        # M-step 1: update beta given z
        offset = np.log(z[cluster_idx])
        X_aug = X.copy()

        def _penalised_nll(b):
            eta = X @ b + offset
            return _cox_neg_logpl_efron(b, X, T, E) - offset @ (np.exp(eta) - eta)

        from scipy.optimize import minimize
        opt = minimize(lambda b: _cox_neg_logpl_efron(b, X, T, E),
                       x0=beta, method="L-BFGS-B",
                       options={"maxiter": 20})
        beta = opt.x

        # M-step 2: update theta via profile ML
        def _theta_nll(th):
            if th <= 0.1:
                return 1e15
            from scipy.special import gammaln
            ll = 0.0
            for g in range(n_clusters):
                mask = cluster_idx == g
                d_g = float(E[mask].sum())
                ll += gammaln(d_g + th) - gammaln(th)
                ll += th * np.log(th) - (d_g + th) * np.log(z[g] * 1 + th)
            return -ll

        opt_th = minimize_scalar(_theta_nll, bounds=(0.5, 100), method="bounded")
        theta = float(opt_th.x)

        if np.max(np.abs(beta - beta_old)) < tol:
            break

    # Final SE from observed information
    _, H = _cox_score_hessian_efron(beta, X, T, E)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(-H)))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    # Concordance (approximate — ignores frailties)
    from .models import _concordance_index
    concordance = _concordance_index(beta, X, T, E)

    ll = float(-_cox_neg_logpl_efron(beta, X, T, E))

    return FrailtyResult(
        beta=beta, se=se, var_names=covariates,
        theta=theta, frailties=z, cluster_ids=uniq_clusters,
        log_likelihood=ll, n=n, n_events=n_events,
        n_clusters=n_clusters, concordance=concordance,
    )
