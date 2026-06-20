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
from typing import Any, Callable, TypeVar
import numpy as np

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility

RewardModel = Callable[[np.ndarray, Any], Any]
_T = TypeVar("_T")
_OPE_METHODS = ("DM", "IPS", "SNIPS", "DR", "SWITCH-DR")


def _ope_error(
    message: str,
    *,
    method: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> MethodIncompatibility:
    payload = dict(diagnostics or {})
    if method is not None:
        payload.setdefault("method", method)
    return MethodIncompatibility(
        message,
        recovery_hint=(
            "Use one of DM, IPS, SNIPS, DR, or Switch-DR and provide the "
            "required logged-policy arrays."
        ),
        diagnostics=payload,
        alternative_functions=[
            "sp.ope.evaluate",
            "sp.ope.ips",
            "sp.ope.snips",
            "sp.ope.doubly_robust",
        ],
    )


def _require(value: _T | None, name: str, method: str) -> _T:
    if value is None:
        raise _ope_error(
            f"OPE method {method} requires {name}.",
            method=method,
            diagnostics={"missing_argument": name},
        )
    return value


@dataclass
class OPEResult(ResultProtocolMixin):
    """Canonical Off-Policy Evaluation result.

    All `sp.ope.*` and `sp.direct_method/ips/snips/doubly_robust` estimators
    return this class (the policy_learning OPEResult subclass is a thin alias
    that adds an ``estimator`` attribute for back-compat). ``isinstance(res,
    sp.OPEResult)`` therefore holds for results from either entry point.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n, K = 500, 3
    >>> pi_b = np.full((n, K), 1.0 / K)          # uniform behaviour policy
    >>> pi_e = np.tile([0.6, 0.3, 0.1], (n, 1))  # evaluation policy to score
    >>> actions = np.array([rng.choice(K, p=pi_b[i]) for i in range(n)])
    >>> rewards = (actions == 0).astype(float) + rng.normal(0, 0.1, n)
    >>> res = sp.ope.ips(actions, rewards, pi_b, pi_e)
    >>> type(res).__name__
    'OPEResult'
    >>> res.method
    'IPS'
    >>> bool(isinstance(res, sp.OPEResult))
    True
    """

    method: str
    value: float
    se: float
    ci: tuple[float, float]
    diagnostics: dict[str, Any]

    @property
    def estimator(self) -> str:
        """Backwards-compatible alias for :attr:`method`."""
        return self.method

    @property
    def n_obs(self) -> int:
        """Return ``diagnostics['n']`` or ``diagnostics['n_obs']`` if present."""
        return int(self.diagnostics.get("n_obs", self.diagnostics.get("n", 0)))

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"OPE({self.method}): V(pi_e) = {self.value:.4f} "
            f"(SE {self.se:.4f}, 95% CI [{lo:.4f}, {hi:.4f}])"
        )


def direct_method(
    reward_model: RewardModel,
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
    _result = OPEResult(
        method="DM",
        value=val,
        se=se,
        ci=(val - 1.96 * se, val + 1.96 * se),
        diagnostics={"n": int(n)},
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.ope.direct_method",
            params={"n_actions": int(K), "n_obs": int(n)},
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


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
            "ess_rho": float(rho.sum() ** 2 / (rho**2).sum()),
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
    rho = pi_e[np.arange(len(a)), a] / np.clip(pi_b[np.arange(len(a)), a], 1e-6, None)
    if clip is not None:
        rho = np.clip(rho, 0, clip)
    val = float((rho * r).sum() / max(rho.sum(), 1e-12))
    # Delta-method SE for self-normalized estimator
    m2 = rho.mean()
    n = len(a)
    var = (1.0 / m2**2) * (rho * r - val * rho).var(ddof=1) / n
    se = float(np.sqrt(var))
    return OPEResult(
        method="SNIPS",
        value=val,
        se=se,
        ci=(val - 1.96 * se, val + 1.96 * se),
        diagnostics={"ess_rho": float(rho.sum() ** 2 / (rho**2).sum())},
    )


def doubly_robust(
    X: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
    reward_model: RewardModel,
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
            "ess_rho": float(rho.sum() ** 2 / (rho**2).sum()),
        },
    )


def switch_dr(
    X: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    pi_b: np.ndarray,
    pi_e: np.ndarray,
    reward_model: RewardModel,
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
    reward_model: RewardModel | None = None,
    **kw: Any,
) -> OPEResult:
    """Dispatch-by-name for OPE methods.

    method : {"DM", "IPS", "SNIPS", "DR", "Switch-DR"}
    """
    if not isinstance(method, str) or not method:
        raise _ope_error(
            "OPE method must be a non-empty string.",
            diagnostics={"method": repr(method)},
        )
    method_upper = method.upper().replace("_", "-")
    if method_upper == "DM":
        return direct_method(
            _require(reward_model, "reward_model", method_upper),
            _require(X, "X", method_upper),
            _require(pi_e, "pi_e", method_upper),
        )
    if method_upper == "IPS":
        return ips(
            _require(actions, "actions", method_upper),
            _require(rewards, "rewards", method_upper),
            _require(pi_b, "pi_b", method_upper),
            _require(pi_e, "pi_e", method_upper),
            **kw,
        )
    if method_upper == "SNIPS":
        return snips(
            _require(actions, "actions", method_upper),
            _require(rewards, "rewards", method_upper),
            _require(pi_b, "pi_b", method_upper),
            _require(pi_e, "pi_e", method_upper),
            **kw,
        )
    if method_upper == "DR":
        return doubly_robust(
            _require(X, "X", method_upper),
            _require(actions, "actions", method_upper),
            _require(rewards, "rewards", method_upper),
            _require(pi_b, "pi_b", method_upper),
            _require(pi_e, "pi_e", method_upper),
            _require(reward_model, "reward_model", method_upper),
            **kw,
        )
    if method_upper == "SWITCH-DR":
        return switch_dr(
            _require(X, "X", method_upper),
            _require(actions, "actions", method_upper),
            _require(rewards, "rewards", method_upper),
            _require(pi_b, "pi_b", method_upper),
            _require(pi_e, "pi_e", method_upper),
            _require(reward_model, "reward_model", method_upper),
            **kw,
        )
    raise _ope_error(
        f"Unknown OPE method: {method!r}",
        method=method_upper,
        diagnostics={"valid_methods": list(_OPE_METHODS)},
    )
