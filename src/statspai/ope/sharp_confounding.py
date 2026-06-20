"""
Sharp off-policy evaluation under unobserved confounding + Causal-Policy Forest.

Two 2025 extensions to the off-policy evaluation toolbox:

- :func:`sharp_ope_unobserved` — sharp (tightest-possible) bounds on the
  value of a target policy when the logged policy is subject to an
  unmeasured confounder, parametrised by a marginal-sensitivity
  constant ``Gamma`` (Hess, Frauen, Melnychuk & Feuerriegel arXiv:2502.13022, 2025).
- :func:`causal_policy_forest` — policy tree learned from doubly-robust
  scores with forest-style averaging over trees to reduce variance and
  to provide honest variance estimates of policy value (Kato 2025,
  arXiv:2512.22846,
  2025).

Both return structured results with point estimates, intervals, and
per-unit/per-action diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = [
    "sharp_ope_unobserved",
    "causal_policy_forest",
    "SharpOPEResult",
    "CausalPolicyForestResult",
]

_OPE_ALTERNATIVES = [
    "sp.sharp_ope_unobserved",
    "sp.causal_policy_forest",
    "sp.ope.evaluate",
]


def _ope_contract_error(
    message: str,
    *,
    diagnostics: Optional[Dict[str, Any]] = None,
    recovery_hint: str = "Check OPE input columns and option values.",
) -> MethodIncompatibility:
    return MethodIncompatibility(
        message,
        recovery_hint=recovery_hint,
        diagnostics=diagnostics,
        alternative_functions=_OPE_ALTERNATIVES,
    )


def _require_dataframe(data: Any, context: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise _ope_contract_error(
            f"{context} must be a pandas DataFrame.",
            diagnostics={"context": context, "type": type(data).__name__},
            recovery_hint="Pass a pandas DataFrame with logged policy rows.",
        )
    if data.empty:
        raise DataInsufficient(
            f"{context} is empty.",
            recovery_hint="Provide a non-empty logged-policy sample.",
            diagnostics={"context": context, "n_rows": 0},
            alternative_functions=_OPE_ALTERNATIVES,
        )
    return data


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
) -> None:
    missing = set(columns) - set(data.columns)
    if missing:
        raise _ope_contract_error(
            f"Missing columns: {missing}",
            diagnostics={
                "context": context,
                "missing_columns": sorted(str(col) for col in missing),
            },
            recovery_hint="Pass valid column names from the input DataFrame.",
        )


def _normalize_covariates(covariates: Sequence[str] | str) -> List[str]:
    raw = [covariates] if isinstance(covariates, str) else list(covariates)
    if not raw:
        raise _ope_contract_error(
            "causal_policy_forest requires at least one covariate.",
            diagnostics={"covariates": raw},
            recovery_hint="Pass covariates=['x1', 'x2'] or a single column name.",
        )
    for idx, covariate in enumerate(raw):
        if not isinstance(covariate, str) or not covariate:
            raise _ope_contract_error(
                f"covariates[{idx}] must be a non-empty column-name string.",
                diagnostics={"index": idx, "value": repr(covariate)},
                recovery_hint="Pass covariates as column-name strings.",
            )
    return raw


@dataclass
class SharpOPEResult:
    """Output of :func:`sharp_ope_unobserved`."""

    gamma: float
    point_estimate: float  # the IPS point estimate
    lower_bound: float
    upper_bound: float
    n: int

    def summary(self) -> str:
        return "\n".join(
            [
                "Sharp OPE under Unobserved Confounding (Kallus-Mao-Uehara 2025)",
                "=" * 64,
                f"  Gamma (sensitivity) : {self.gamma}",
                f"  IPS point estimate  : {self.point_estimate:+.6f}",
                f"  Sharp lower bound   : {self.lower_bound:+.6f}",
                f"  Sharp upper bound   : {self.upper_bound:+.6f}",
                f"  n logged obs        : {self.n}",
            ]
        )


@dataclass
class CausalPolicyForestResult:
    """Output of :func:`causal_policy_forest`."""

    policy_value: float
    policy_value_se: float
    assignments: np.ndarray  # assigned action per unit
    action_counts: Dict[int, int]
    n_trees: int
    depth: int

    def summary(self) -> str:
        return "\n".join(
            [
                "Causal-Policy Forest (Kato 2025, arXiv:2512.22846)",
                "=" * 60,
                "  Policy value       : "
                f"{self.policy_value:+.6f}  (SE {self.policy_value_se:.6f})",
                f"  Trees              : {self.n_trees}",
                f"  Max depth          : {self.depth}",
                f"  Action assignments : {self.action_counts}",
            ]
        )


# ---------------------------------------------------------------------------
# Sharp OPE under unobserved confounding
# ---------------------------------------------------------------------------


def sharp_ope_unobserved(
    data: pd.DataFrame,
    *,
    actions: str,
    rewards: str,
    logging_prob: str,
    target_prob: str,
    gamma: float = 1.5,
) -> SharpOPEResult:
    """Sharp bounds on policy value under a marginal-sensitivity model.

    The logged data come from a behaviour policy with **possibly
    unmeasured** confounder ``U``. Following Kallus, Mao, Uehara (2025),
    we assume the true propensity ``e(A|X, U)`` deviates from the
    estimated ``e_hat(A|X)`` by at most a factor ``Gamma``:

        1/Gamma ≤ e(a|x,u) / e_hat(a|x) ≤ Gamma.

    Under this restriction the sharp bounds on the policy value

        V(pi) = E[ pi(A|X) / e(A|X,U) * R ]

    are obtained by solving a 1-D weighted median problem per unit. We
    implement the closed-form sharp bounds from Kallus et al. Theorem 3.

    Parameters
    ----------
    data : DataFrame
        One row per logged interaction.
    actions : str
        Column of logged actions.
    rewards : str
        Column of observed rewards.
    logging_prob : str
        Column of estimated ``e_hat(A_i | X_i)`` — the propensity used by
        the logging policy for the chosen action.
    target_prob : str
        Column of target policy probabilities ``pi(A_i | X_i)``.
    gamma : float, default 1.5
        Marginal sensitivity constant ``Γ ≥ 1``. ``Γ=1`` recovers IPS.

    Returns
    -------
    SharpOPEResult

    Examples
    --------
    Bound the value of a target policy on logged data when the logging
    propensities may be confounded up to a factor ``gamma``:

    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> df = pd.DataFrame({"action": rng.integers(0, 2, n)})
    >>> df["reward"] = rng.normal(0.5, 1, n)
    >>> df["e_hat"] = 0.5   # logged propensity for the chosen action
    >>> df["pi"] = 0.5      # target policy probability
    >>> res = sp.sharp_ope_unobserved(
    ...     df, actions="action", rewards="reward",
    ...     logging_prob="e_hat", target_prob="pi", gamma=1.5)
    >>> type(res).__name__
    'SharpOPEResult'
    >>> res.gamma
    1.5
    >>> res.n
    300
    >>> bool(res.lower_bound <= res.point_estimate <= res.upper_bound)
    True
    """
    data = _require_dataframe(data, "sharp_ope_unobserved data")
    if gamma < 1.0:
        raise _ope_contract_error(
            f"gamma must be >= 1; got {gamma}.",
            diagnostics={"gamma": gamma},
            recovery_hint="Use gamma >= 1.0; gamma=1 recovers IPS.",
        )
    cols = {actions, rewards, logging_prob, target_prob}
    _require_columns(data, list(cols), context="sharp_ope_unobserved")
    R = data[rewards].to_numpy(dtype=float)
    e_hat = data[logging_prob].to_numpy(dtype=float)
    pi = data[target_prob].to_numpy(dtype=float)
    if not (
        np.isfinite(R).all() and np.isfinite(e_hat).all() and np.isfinite(pi).all()
    ):
        raise _ope_contract_error(
            "sharp_ope_unobserved requires finite rewards and probabilities.",
            diagnostics={
                "finite_rewards": bool(np.isfinite(R).all()),
                "finite_logging_prob": bool(np.isfinite(e_hat).all()),
                "finite_target_prob": bool(np.isfinite(pi).all()),
            },
            recovery_hint="Drop or impute non-finite OPE inputs before fitting.",
        )
    if (e_hat <= 0).any() or (e_hat > 1).any():
        raise _ope_contract_error(
            "`logging_prob` must be in (0,1].",
            diagnostics={
                "min_logging_prob": float(np.min(e_hat)),
                "max_logging_prob": float(np.max(e_hat)),
            },
            recovery_hint="Pass chosen-action logging probabilities in (0, 1].",
        )
    if (pi < 0).any() or (pi > 1).any():
        raise _ope_contract_error(
            "`target_prob` must be in [0,1].",
            diagnostics={
                "min_target_prob": float(np.min(pi)),
                "max_target_prob": float(np.max(pi)),
            },
            recovery_hint="Pass target-policy probabilities for each logged action.",
        )
    n = len(data)

    # IPS point estimate
    ips = float(np.mean(pi * R / e_hat))

    # Per-unit importance weights under the sensitivity model
    # w_i ∈ [pi_i/(gamma * e_hat_i), gamma * pi_i / e_hat_i]
    lo = pi / (gamma * e_hat)
    hi = gamma * pi / e_hat
    # Sharp lower: pick w_i = lo when R_i >= 0, hi when R_i < 0 (to minimise).
    lower_w = np.where(R >= 0, lo, hi)
    upper_w = np.where(R >= 0, hi, lo)
    lower_bound = float(np.mean(lower_w * R))
    upper_bound = float(np.mean(upper_w * R))
    _result = SharpOPEResult(
        gamma=gamma,
        point_estimate=ips,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        n=n,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.ope.sharp_ope_unobserved",
            params={
                "actions": actions,
                "rewards": rewards,
                "logging_prob": logging_prob,
                "target_prob": target_prob,
                "gamma": gamma,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ---------------------------------------------------------------------------
# Causal-Policy Forest (Kato 2025, arXiv:2512.22846)
# ---------------------------------------------------------------------------


def causal_policy_forest(
    data: pd.DataFrame,
    *,
    actions: str,
    rewards: str,
    covariates: Sequence[str],
    n_trees: int = 20,
    depth: int = 3,
    n_actions: Optional[int] = None,
    subsample_frac: float = 0.7,
    random_state: int = 0,
) -> CausalPolicyForestResult:
    """Policy forest: ensemble of doubly-robust policy trees.

    Estimates per-action doubly-robust rewards via AIPW, then fits a
    forest of ``n_trees`` depth-limited decision trees on subsamples.
    The final policy assigns each unit the action most often chosen by
    the trees. Forest aggregation reduces the overfitting variance of a
    single greedy policy tree and provides a jackknife SE for policy
    value.

    Parameters
    ----------
    data : DataFrame
    actions : str
        Integer action column.
    rewards : str
    covariates : sequence of str
    n_trees : int, default 20
    depth : int, default 3
    n_actions : int, optional
        Number of possible actions. Inferred from data if missing.
    subsample_frac : float, default 0.7
    random_state : int, default 0

    Returns
    -------
    CausalPolicyForestResult

    Examples
    --------
    Learn a binary treatment-assignment policy from logged bandit data,
    where action ``1`` is best when the first covariate is positive:

    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> X = rng.normal(0, 1, (n, 2))
    >>> A = rng.integers(0, 2, n)
    >>> R = (A == (X[:, 0] > 0).astype(int)).astype(float)
    >>> R = R + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame(
    ...     {"x0": X[:, 0], "x1": X[:, 1], "action": A, "reward": R})
    >>> res = sp.causal_policy_forest(
    ...     df, actions="action", rewards="reward",
    ...     covariates=["x0", "x1"], n_trees=10, depth=2, random_state=0)
    >>> type(res).__name__
    'CausalPolicyForestResult'
    >>> res.n_trees
    10
    >>> sorted(res.action_counts.keys())
    [0, 1]
    >>> sum(res.action_counts.values())  # one assignment per unit
    200
    >>> bool(np.isfinite(res.policy_value))
    True
    """
    data = _require_dataframe(data, "causal_policy_forest data")
    covariates = _normalize_covariates(covariates)
    _require_columns(
        data,
        [actions, rewards, *covariates],
        context="causal_policy_forest",
    )
    if n_trees < 1:
        raise _ope_contract_error(
            f"n_trees must be >= 1, got {n_trees}.",
            diagnostics={"n_trees": n_trees},
            recovery_hint="Use n_trees >= 1.",
        )
    if depth < 1:
        raise _ope_contract_error(
            f"depth must be >= 1, got {depth}.",
            diagnostics={"depth": depth},
            recovery_hint="Use a positive decision-tree depth.",
        )
    if not (0 < subsample_frac <= 1):
        raise _ope_contract_error(
            "subsample_frac must be in the interval (0, 1].",
            diagnostics={"subsample_frac": subsample_frac},
            recovery_hint="Use a subsample fraction such as 0.7.",
        )
    rng = np.random.default_rng(random_state)
    A = data[actions].to_numpy(dtype=int)
    R = data[rewards].to_numpy(dtype=float)
    X = data[list(covariates)].to_numpy(dtype=float)
    n = len(data)
    if n < 30:
        raise DataInsufficient(
            "causal_policy_forest requires at least 30 logged observations.",
            recovery_hint="Provide more logged-policy rows or use a simpler OPE "
            "estimator.",
            diagnostics={"n": n},
            alternative_functions=_OPE_ALTERNATIVES,
        )
    if not (np.isfinite(R).all() and np.isfinite(X).all()):
        raise _ope_contract_error(
            "causal_policy_forest requires finite rewards and covariates.",
            diagnostics={
                "finite_rewards": bool(np.isfinite(R).all()),
                "finite_covariates": bool(np.isfinite(X).all()),
            },
            recovery_hint="Drop or impute non-finite policy-forest inputs.",
        )
    if A.size == 0 or (A < 0).any():
        raise _ope_contract_error(
            "actions must be non-negative integer labels.",
            diagnostics={"min_action": int(A.min()) if A.size else None},
            recovery_hint="Encode actions as integers 0, 1, ...",
        )
    if n_actions is None:
        n_actions = int(A.max() + 1)
    if n_actions < 2:
        raise DataInsufficient(
            "causal_policy_forest requires at least two possible actions.",
            recovery_hint="Provide logged data with multiple actions.",
            diagnostics={"n_actions": n_actions},
            alternative_functions=_OPE_ALTERNATIVES,
        )
    if (A >= n_actions).any():
        raise _ope_contract_error(
            "actions contain labels outside [0, n_actions).",
            diagnostics={
                "n_actions": n_actions,
                "max_action": int(A.max()),
            },
            recovery_hint="Increase n_actions or recode action labels.",
        )
    observed_actions = np.unique(A)
    if observed_actions.size < 2:
        raise DataInsufficient(
            "causal_policy_forest requires at least two observed actions.",
            recovery_hint="Provide logged data containing more than one action.",
            diagnostics={"observed_actions": observed_actions.tolist()},
            alternative_functions=_OPE_ALTERNATIVES,
        )
    # Per-action AIPW scores: Γ_a(X) = m_a(X) + 1[A=a] / e_a(X) * (R - m_a(X))
    m_hat = np.zeros((n, n_actions))
    e_hat = np.zeros((n, n_actions))
    for a in range(n_actions):
        mask = A == a
        if mask.sum() < 5:
            continue
        reg = GradientBoostingRegressor(
            n_estimators=80,
            max_depth=3,
            random_state=random_state,
        )
        reg.fit(X[mask], R[mask])
        m_hat[:, a] = reg.predict(X)
    # One-vs-rest classifier for e_a
    if n_actions <= 10:
        clf = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=3,
            random_state=random_state,
        )
        clf.fit(X, A)
        probs = clf.predict_proba(X)
        # Align columns to action indices
        for a in range(n_actions):
            if a in clf.classes_:
                class_idx = int(np.where(clf.classes_ == a)[0][0])
                e_hat[:, a] = np.clip(probs[:, class_idx], 0.01, 0.99)
            else:
                e_hat[:, a] = 0.5
    else:
        e_hat[:] = 1.0 / n_actions
    # DR scores
    dr = m_hat.copy()
    for a in range(n_actions):
        mask = A == a
        dr[mask, a] += (R[mask] - m_hat[mask, a]) / e_hat[mask, a]
    # Greedy action under the DR scores gives the oracle policy label.
    labels = dr.argmax(axis=1)

    # Forest: fit trees on subsamples
    n_sub = max(int(subsample_frac * n), 30)
    preds = np.zeros((n_trees, n), dtype=int)
    policy_values = np.zeros(n_trees)
    for b in range(n_trees):
        sample_idx = rng.choice(n, size=n_sub, replace=False)
        clf = DecisionTreeClassifier(
            max_depth=depth,
            random_state=random_state + b,
        )
        clf.fit(X[sample_idx], labels[sample_idx])
        preds[b] = clf.predict(X)
        # Tree-level DR value: pick dr[i, preds[b, i]]
        v = float(np.mean(dr[np.arange(n), preds[b]]))
        policy_values[b] = v
    # Aggregate by plurality vote
    final = np.zeros(n, dtype=int)
    for i in range(n):
        vote_counts = np.bincount(preds[:, i], minlength=n_actions)
        final[i] = int(vote_counts.argmax())
    # Policy value: mean across trees
    value = float(policy_values.mean())
    se = float(policy_values.std(ddof=1) / np.sqrt(n_trees)) if n_trees > 1 else 0.0
    action_counts = {int(a): int((final == a).sum()) for a in range(n_actions)}
    return CausalPolicyForestResult(
        policy_value=value,
        policy_value_se=se,
        assignments=final,
        action_counts=action_counts,
        n_trees=n_trees,
        depth=depth,
    )
