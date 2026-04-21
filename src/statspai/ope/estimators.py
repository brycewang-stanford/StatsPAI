"""
Off-Policy Evaluation (OPE) estimators for contextual bandits.

Given logged data :math:`(X_i, A_i, R_i, \\pi_b(A_i | X_i))` collected
under a behaviour policy :math:`\\pi_b`, estimate the value
:math:`V(\\pi_e) = \\mathbb{E}[R | A \\sim \\pi_e]` of an evaluation
policy :math:`\\pi_e`.

Implemented estimators
----------------------
* Direct Method (DM)
* Inverse Propensity Scoring (IPS)
* Self-Normalized IPS (SNIPS) — Swaminathan & Joachims 2015
* Doubly Robust (DR) — Dudík, Langford, Li 2011
* Switch-DR / CAB — Wang, Agarwal, Dudík 2017

References
----------
Dudik, Langford, Li (2011). Doubly robust policy evaluation and
learning. ICML.
Swaminathan & Joachims (2015). The self-normalized estimator for
counterfactual learning. NeurIPS.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass
class OPEResult:
    method: str
    value: float
    se: float
    ci: tuple[float, float]
    diagnostics: dict

    def summary(self) -> str:
        return (
            f"OPE({self.method}): V(pi_e) = {self.value:.4f} "
            f"(SE {self.se:.4f}, 95% CI [{self.ci[0]:.4f}, {self.ci[1]:.4f}])"
        )


def direct_method(
    reward_model,
    X: np.ndarray,
    pi_e: np.ndarray,
) -> OPEResult:
    """Plug-in estimator using a fitted reward model ``Q(x, a)``.

    Parameters
    ----------
    reward_model : callable
        ``reward_model(X, a)`` → vector of length n predicting E[R | X, a].
    X : (n, d) array
    pi_e : (n, K) array
        Evaluation policy probabilities over K actions at each X_i.
    """
    n, K = pi_e.shape
    V = np.zeros(n)
    for a in range(K):
        V += pi_e[:, a] * reward_model(X, a)
    val = float(V.mean())
    se = float(V.std(ddof=1) / np.sqrt(max(n, 1)))
    return OPEResult(
        method="DM",
        value=val,
        se=se,
        ci=(val - 1.96 * se, val + 1.96 * se),
        diagnostics={"n": int(n)},
    )


def ips(
    actions: np.ndarray,
    rewards: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
    clip: float | None = 1000.0,
) -> OPEResult:
    """Inverse Propensity Scoring (aka IPS / Horvitz-Thompson)."""
    a = np.asarray(actions, dtype=int)
    r = np.asarray(rewards, dtype=float)
    pe_at_a = pi_e[np.arange(len(a)), a]
    pb_at_a = pi_b[np.arange(len(a)), a]
    rho = pe_at_a / np.clip(pb_at_a, 1e-6, None)
    if clip is not None:
        rho = np.clip(rho, 0, clip)
    val_per = rho * r
    val = float(val_per.mean())
    se = float(val_per.std(ddof=1) / np.sqrt(max(len(a), 1)))
    return OPEResult(
        method="IPS",
        value=val,
        se=se,
        ci=(val - 1.96 * se, val + 1.96 * se),
        diagnostics={
            "ess_rho": float(rho.sum() ** 2 / (rho ** 2).sum()),
            "max_rho": float(rho.max()),
        },
    )


def snips(
    actions: np.ndarray,
    rewards: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
    clip: float | None = 1000.0,
) -> OPEResult:
    """Self-Normalized IPS -- reduces variance at small bias cost."""
    a = np.asarray(actions, dtype=int)
    r = np.asarray(rewards, dtype=float)
    rho = pi_e[np.arange(len(a)), a] / np.clip(
        pi_b[np.arange(len(a)), a], 1e-6, None
    )
    if clip is not None:
        rho = np.clip(rho, 0, clip)
    val = float((rho * r).sum() / max(rho.sum(), 1e-12))
    # Delta-method SE for self-normalized estimator
    m1 = (rho * r).mean()
    m2 = rho.mean()
    n = len(a)
    var = (
        (1.0 / m2 ** 2) * (rho * r - val * rho).var(ddof=1) / n
    )
    se = float(np.sqrt(var))
    return OPEResult(
        method="SNIPS",
        value=val,
        se=se,
        ci=(val - 1.96 * se, val + 1.96 * se),
        diagnostics={"ess_rho": float(rho.sum() ** 2 / (rho ** 2).sum())},
    )


def doubly_robust(
    X: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
    reward_model,
    clip: float | None = 1000.0,
) -> OPEResult:
    """Doubly Robust estimator (Dudik, Langford, Li 2011)."""
    a = np.asarray(actions, dtype=int)
    r = np.asarray(rewards, dtype=float)
    n, K = pi_e.shape

    # Direct-method baseline
    V_dm = np.zeros(n)
    for k in range(K):
        V_dm += pi_e[:, k] * reward_model(X, k)

    # IPS correction on residuals
    q_at_a = reward_model(X, a)
    rho = pi_e[np.arange(n), a] / np.clip(pi_b[np.arange(n), a], 1e-6, None)
    if clip is not None:
        rho = np.clip(rho, 0, clip)
    correction = rho * (r - q_at_a)
    per_sample = V_dm + correction
    val = float(per_sample.mean())
    se = float(per_sample.std(ddof=1) / np.sqrt(max(n, 1)))
    return OPEResult(
        method="DR",
        value=val,
        se=se,
        ci=(val - 1.96 * se, val + 1.96 * se),
        diagnostics={
            "max_rho": float(rho.max()),
            "ess_rho": float(rho.sum() ** 2 / (rho ** 2).sum()),
        },
    )


def switch_dr(
    X: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
    reward_model,
    tau: float = 10.0,
) -> OPEResult:
    """Switch-DR (Wang, Agarwal, Dudík 2017): fall back to the DM
    whenever the importance ratio exceeds ``tau``."""
    a = np.asarray(actions, dtype=int)
    r = np.asarray(rewards, dtype=float)
    n, K = pi_e.shape
    V_dm = np.zeros(n)
    for k in range(K):
        V_dm += pi_e[:, k] * reward_model(X, k)

    q_at_a = reward_model(X, a)
    rho = pi_e[np.arange(n), a] / np.clip(pi_b[np.arange(n), a], 1e-6, None)
    mask = rho <= tau
    correction = np.where(mask, rho * (r - q_at_a), 0.0)
    per_sample = V_dm + correction
    val = float(per_sample.mean())
    se = float(per_sample.std(ddof=1) / np.sqrt(max(n, 1)))
    return OPEResult(
        method="Switch-DR",
        value=val,
        se=se,
        ci=(val - 1.96 * se, val + 1.96 * se),
        diagnostics={"tau": float(tau), "switched_frac": float((~mask).mean())},
    )


def evaluate(
    method: str,
    *,
    X: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    pi_b: np.ndarray | None = None,
    pi_e: np.ndarray | None = None,
    reward_model=None,
    **kw,
) -> OPEResult:
    """Dispatch-by-name for OPE methods.

    method : {"DM", "IPS", "SNIPS", "DR", "Switch-DR"}
    """
    method_upper = method.upper().replace("_", "-")
    if method_upper == "DM":
        return direct_method(reward_model, X, pi_e)
    if method_upper == "IPS":
        return ips(actions, rewards, pi_b, pi_e, **kw)
    if method_upper == "SNIPS":
        return snips(actions, rewards, pi_b, pi_e, **kw)
    if method_upper == "DR":
        return doubly_robust(X, actions, rewards, pi_b, pi_e, reward_model, **kw)
    if method_upper == "SWITCH-DR":
        return switch_dr(X, actions, rewards, pi_b, pi_e, reward_model, **kw)
    raise ValueError(f"Unknown OPE method: {method!r}")
