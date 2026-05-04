"""
Off-Policy Evaluation (OPE).

Given logged data ``{(X_i, A_i, R_i, π_b(A_i|X_i))}`` collected under a
behaviour policy :math:`\\pi_b` and a proposed *target* policy
:math:`\\pi_e(a|x)`, estimate the target-policy value

.. math::

    V(\\pi_e) = E_{X, A \\sim \\pi_e}[R(X, A)].

Four classical estimators are provided:

* :func:`direct_method` — pure outcome regression (Q-model).
* :func:`ips` — inverse propensity score.
* :func:`snips` — self-normalised IPS (bias-reduction for large weights).
* :func:`doubly_robust` — DR combining Q-model and IPS residual.

All estimators accept either pre-computed propensity scores (from
logging), pre-computed Q-values from a model, or will fit simple
logistic / random-forest nuisance models on the fly.

References
----------
Dudik, M., Langford, J., & Li, L. (2011). "Doubly robust policy
evaluation and learning." *ICML*.

Swaminathan, A. & Joachims, T. (2015). "The self-normalized estimator
for counterfactual learning." *NeurIPS*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any, Callable, Union

import numpy as np
import pandas as pd
from scipy import stats

# sklearn is imported lazily inside the functions that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches ope.


@dataclass
class OPEResult:
    estimator: str
    value: float
    se: float
    ci: tuple
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lo, hi = self.ci
        return (
            f"Off-Policy Evaluation ({self.estimator})\n"
            f"  V(pi_e)  = {self.value:.4f}\n"
            f"  SE       = {self.se:.4f}\n"
            f"  95% CI   = [{lo:.4f}, {hi:.4f}]\n"
            f"  N        = {self.n_obs}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"OPEResult({self.estimator}, V={self.value:.4f})"


# --------------------------------------------------------------------
# Policy helpers
# --------------------------------------------------------------------

def _target_prob(pi_target, X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    pi_target can be:
      * a 2-D array (n, K) with probabilities per action, indexed by A,
      * a 1-D array of length n (already P(a=A_i | X_i)),
      * a callable taking (X) and returning (n, K).
    """
    if callable(pi_target):
        probs = pi_target(X)
        return np.asarray([probs[i, A[i]] for i in range(len(A))])
    arr = np.asarray(pi_target, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[np.arange(len(A)), A]
    raise ValueError("pi_target must be (n,), (n,K), or callable")


def _fit_propensity(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    try:
        lr = LogisticRegression(solver="lbfgs", max_iter=500)
        lr.fit(X, A)
        probs = lr.predict_proba(X)
        return probs[np.arange(len(A)), A]
    except Exception:
        return np.full(len(A), 1.0 / len(np.unique(A)))


def _fit_q(X: np.ndarray, A: np.ndarray, R: np.ndarray, n_actions: int) -> np.ndarray:
    """Return (n, K) Q-hat matrix via a single RF on (X, A one-hot)."""
    from sklearn.ensemble import RandomForestRegressor
    oh = np.eye(n_actions)[A]
    features = np.column_stack([X, oh])
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0)
    rf.fit(features, R)
    Q = np.zeros((len(A), n_actions))
    for a in range(n_actions):
        oh_a = np.zeros((len(A), n_actions))
        oh_a[:, a] = 1
        Q[:, a] = rf.predict(np.column_stack([X, oh_a]))
    return Q


# --------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------


def direct_method(
    X: np.ndarray, A: np.ndarray, R: np.ndarray,
    pi_target, n_actions: Optional[int] = None, alpha: float = 0.05,
) -> OPEResult:
    """Direct outcome regression (plug-in Q-model) OPE."""
    X = np.asarray(X); A = np.asarray(A); R = np.asarray(R)
    if n_actions is None:
        n_actions = int(A.max()) + 1
    Q = _fit_q(X, A, R, n_actions)
    # Compute pi_target probability matrix for all actions
    if callable(pi_target):
        pi_mat = pi_target(X)
    else:
        pi_mat = np.asarray(pi_target)
        if pi_mat.ndim == 1:
            # Deterministic policy: action vector
            tmp = np.zeros((len(A), n_actions))
            tmp[np.arange(len(A)), pi_mat.astype(int)] = 1.0
            pi_mat = tmp
    V_per = (Q * pi_mat).sum(axis=1)
    V = float(V_per.mean())
    se = float(V_per.std(ddof=1) / np.sqrt(len(A)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return OPEResult("direct", V, se, (V - crit * se, V + crit * se), n_obs=len(A))


def ips(
    X: np.ndarray, A: np.ndarray, R: np.ndarray,
    pi_target, pi_behavior: Optional[np.ndarray] = None,
    clip: float = 50.0, alpha: float = 0.05,
) -> OPEResult:
    """Inverse propensity score OPE."""
    X = np.asarray(X); A = np.asarray(A); R = np.asarray(R)
    pi_t = _target_prob(pi_target, X, A)
    pi_b = pi_behavior if pi_behavior is not None else _fit_propensity(X, A.astype(int))
    pi_b = np.clip(pi_b, 1 / clip, 1.0)
    ratio = np.clip(pi_t / pi_b, 0, clip)
    V_per = ratio * R
    V = float(V_per.mean())
    se = float(V_per.std(ddof=1) / np.sqrt(len(A)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return OPEResult("IPS", V, se, (V - crit * se, V + crit * se), n_obs=len(A))


def snips(
    X: np.ndarray, A: np.ndarray, R: np.ndarray,
    pi_target, pi_behavior: Optional[np.ndarray] = None,
    clip: float = 50.0, alpha: float = 0.05,
) -> OPEResult:
    """Self-normalised IPS (bias-reduction for large IS weights)."""
    X = np.asarray(X); A = np.asarray(A); R = np.asarray(R)
    pi_t = _target_prob(pi_target, X, A)
    pi_b = pi_behavior if pi_behavior is not None else _fit_propensity(X, A.astype(int))
    pi_b = np.clip(pi_b, 1 / clip, 1.0)
    ratio = np.clip(pi_t / pi_b, 0, clip)
    denom = max(float(ratio.sum()), 1e-6)
    V = float((ratio * R).sum() / denom)
    # Approximate SE via delta method (ratio estimator).
    numer_var = np.var(ratio * R, ddof=1)
    denom_var = np.var(ratio, ddof=1)
    cov = np.cov(ratio * R, ratio, ddof=1)[0, 1]
    n = len(A)
    se = float(np.sqrt(max(numer_var / denom - V * cov * 2 / denom + V ** 2 * denom_var / denom, 0)
                        / n))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return OPEResult("SNIPS", V, se, (V - crit * se, V + crit * se), n_obs=n)


def doubly_robust(
    X: np.ndarray, A: np.ndarray, R: np.ndarray,
    pi_target, pi_behavior: Optional[np.ndarray] = None,
    n_actions: Optional[int] = None, clip: float = 50.0,
    alpha: float = 0.05,
) -> OPEResult:
    """Doubly-robust OPE (Dudik et al. 2011)."""
    X = np.asarray(X); A = np.asarray(A); R = np.asarray(R)
    if n_actions is None:
        n_actions = int(A.max()) + 1
    pi_t_a = _target_prob(pi_target, X, A)
    pi_b = pi_behavior if pi_behavior is not None else _fit_propensity(X, A.astype(int))
    pi_b = np.clip(pi_b, 1 / clip, 1.0)

    Q = _fit_q(X, A, R, n_actions)
    if callable(pi_target):
        pi_mat = pi_target(X)
    else:
        pi_mat_raw = np.asarray(pi_target)
        if pi_mat_raw.ndim == 1:
            tmp = np.zeros((len(A), n_actions))
            tmp[np.arange(len(A)), pi_mat_raw.astype(int)] = 1.0
            pi_mat = tmp
        else:
            pi_mat = pi_mat_raw
    Q_pi = (Q * pi_mat).sum(axis=1)
    resid = R - Q[np.arange(len(A)), A]
    ratio = np.clip(pi_t_a / pi_b, 0, clip)
    V_per = Q_pi + ratio * resid
    V = float(V_per.mean())
    se = float(V_per.std(ddof=1) / np.sqrt(len(A)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return OPEResult("Doubly Robust", V, se, (V - crit * se, V + crit * se), n_obs=len(A))


__all__ = ["direct_method", "ips", "snips", "doubly_robust", "OPEResult"]
