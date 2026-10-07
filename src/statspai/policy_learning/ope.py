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

All four OPE estimators accept either pre-computed propensity scores (from
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

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats

from ..exceptions import (
    AssumptionViolation,
    ConvergenceWarning,
    IdentificationFailure,
    MethodIncompatibility,
)

# Single canonical OPEResult lives in ``ope.estimators``; re-exporting it
# here so ``isinstance(sp.direct_method(...), sp.OPEResult)`` is True
# regardless of which entry point the user picked. The historical
# policy_learning OPEResult dataclass (with extra ``estimator/n_obs/detail``
# fields) is replicated as properties / diagnostics on the canonical class.
from ..ope.estimators import OPEResult as _CanonicalOPEResult

# sklearn is imported lazily inside the functions that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches ope.


OPEResult = _CanonicalOPEResult


def _wrap(
    estimator: str, value: float, se: float, ci: tuple, n_obs: int, **detail: Any
) -> OPEResult:
    """Pack policy_learning-style returns into the canonical OPEResult."""
    diag: Dict[str, Any] = {"n_obs": int(n_obs), **detail}
    return OPEResult(method=estimator, value=value, se=se, ci=ci, diagnostics=diag)


# --------------------------------------------------------------------
# Policy helpers
# --------------------------------------------------------------------


def _target_prob(pi_target: Any, X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    pi_target can be:
      * a 2-D array (n, K) with probabilities per action, indexed by A,
      * a 1-D array of length n (already P(a=A_i | X_i)),
      * a callable taking (X) and returning (n, K).
    """
    if callable(pi_target):
        probs = pi_target(X)
        return np.asarray([probs[i, A[i]] for i in range(len(A))], dtype=float)
    arr = np.asarray(pi_target, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return np.asarray(arr[np.arange(len(A)), A], dtype=float)
    raise ValueError("pi_target must be (n,), (n,K), or callable")


def _fit_propensity(X: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Fit behaviour-policy propensities; returns (probs, fallback_flag)."""
    from sklearn.linear_model import LogisticRegression

    try:
        lr = LogisticRegression(solver="lbfgs", max_iter=500)
        lr.fit(X, A)
        probs = lr.predict_proba(X)
        return probs[np.arange(len(A)), A], False
    except Exception as exc:
        warnings.warn(
            "off-policy evaluation: behavior-policy logistic regression "
            f"failed ({type(exc).__name__}: {exc}); falling back to "
            "uniform propensities 1/K. Importance weights are degraded — "
            "supply pi_behavior explicitly if known.",
            ConvergenceWarning,
            stacklevel=3,
        )
        return np.full(len(A), 1.0 / len(np.unique(A))), True


def _behaviour_ratio(pi_t: np.ndarray, pi_b: Any, n: int, clip: float) -> np.ndarray:
    """Importance weights ``pi_e(A|X) / pi_b(A|X)`` capped at ``clip``.

    ``clip`` caps the *weight* (``obp``'s ``lambda_``); it does not floor the
    behaviour propensity. Before 1.30.0 ``pi_b`` was also floored at
    ``1/clip``, which shrank the weight of every unit with
    ``pi_b < 1/clip`` even when its true weight was below the cap.
    """
    pb = np.asarray(pi_b, dtype=float).ravel()
    if pb.shape[0] != n:
        raise MethodIncompatibility(
            f"pi_behavior must have length {n}; got {pb.shape[0]}.",
            recovery_hint="Pass one logging propensity per logged action.",
        )
    if np.any(pb <= 0) or not np.isfinite(pb).all():
        raise AssumptionViolation(
            "pi_behavior must be strictly positive and finite: an action "
            "logged with probability 0 has an undefined importance weight.",
            recovery_hint=(
                "Importance weighting needs positivity: every logged action must have "
                "a positive, finite logging propensity."
            ),
        )
    return np.minimum(pi_t / pb, clip)


def _policy_matrix(pi_target: Any, X: np.ndarray, n: int, n_actions: int) -> np.ndarray:
    """(n, K) evaluation-policy probabilities; a 1-D input is an action vector."""
    if callable(pi_target):
        return np.asarray(pi_target(X), dtype=float)
    arr = np.asarray(pi_target)
    if arr.ndim == 1:
        acts = arr.astype(int)
        if not np.array_equal(acts, arr) or acts.min() < 0 or acts.max() >= n_actions:
            raise MethodIncompatibility(
                "A 1-D pi_target for direct_method / doubly_robust is a "
                "deterministic action vector with integer entries in [0, "
                f"{n_actions}).",
                recovery_hint=(
                    "Pass integer action labels in [0, n_actions) or an (n, n_actions)"
                    " probability matrix."
                ),
            )
        out = np.zeros((n, n_actions))
        out[np.arange(n), acts] = 1.0
        return out
    return np.asarray(arr, dtype=float)


def _resolve_q(
    q_hat: Optional[np.ndarray],
    X: np.ndarray,
    A: np.ndarray,
    R: np.ndarray,
    n_actions: int,
    n_folds: int = 1,
) -> np.ndarray:
    if q_hat is None:
        if n_folds > 1:
            return _fit_q_cross(X, A, R, n_actions, n_folds)
        return _fit_q(X, A, R, n_actions)
    Q = np.asarray(q_hat, dtype=float)
    if Q.shape != (len(A), n_actions) or not np.isfinite(Q).all():
        raise MethodIncompatibility(
            f"q_hat must be a finite (n, n_actions) = ({len(A)}, {n_actions}) matrix.",
            recovery_hint="Pass q_hat=None to fit the outcome model internally.",
        )
    return Q


def _fit_q(
    X: np.ndarray,
    A: np.ndarray,
    R: np.ndarray,
    n_actions: int,
    X_eval: Optional[np.ndarray] = None,
) -> np.ndarray:
    """(n_eval, K) Q-hat matrix from one random forest on (X, A one-hot)."""
    from sklearn.ensemble import RandomForestRegressor

    oh = np.eye(n_actions)[A]
    features = np.column_stack([X, oh])
    rf = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0
    )
    rf.fit(features, R)
    X_out = X if X_eval is None else X_eval
    Q = np.zeros((len(X_out), n_actions))
    for a in range(n_actions):
        oh_a = np.zeros((len(X_out), n_actions))
        oh_a[:, a] = 1
        Q[:, a] = rf.predict(np.column_stack([X_out, oh_a]))
    return Q


def _fit_q_cross(
    X: np.ndarray, A: np.ndarray, R: np.ndarray, n_actions: int, n_folds: int
) -> np.ndarray:
    """Cross-fitted Q-hat: each row is predicted by a forest that did not see it.

    A forest evaluated on its own training rows fits part of the noise,
    which shrinks the residual the doubly robust correction relies on.
    """
    from sklearn.model_selection import KFold

    n = len(A)
    Q = np.zeros((n, n_actions))
    folds = KFold(n_splits=min(n_folds, n), shuffle=True, random_state=0)
    for train, test in folds.split(X):
        Q[test] = _fit_q(X[train], A[train], R[train], n_actions, X_eval=X[test])
    return Q


# --------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------


def direct_method(
    X: np.ndarray,
    A: np.ndarray,
    R: np.ndarray,
    pi_target: Any,
    n_actions: Optional[int] = None,
    alpha: float = 0.05,
    q_hat: Optional[np.ndarray] = None,
) -> OPEResult:
    """Direct outcome regression (plug-in Q-model) OPE.

    ``V = mean_i sum_a pi_e(a|X_i) Q(X_i, a)``. ``Q`` is ``q_hat`` when
    given (an ``(n, n_actions)`` matrix of reward-model predictions),
    otherwise a random forest on ``(X, one-hot A)``. A 1-D ``pi_target`` is
    a deterministic action vector. With the same ``Q`` this is ``obp``'s
    ``DirectMethod``.

    Examples
    --------
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> X = rng.normal(size=(n, 3))
    >>> A = rng.integers(0, 3, size=n)
    >>> R = rng.normal(size=n)
    >>> pi_target = rng.dirichlet(np.ones(3), size=n)
    >>> res = sp.direct_method(X, A, R, pi_target)
    >>> float(res.value)  # doctest: +SKIP
    """
    X = np.asarray(X)
    A = np.asarray(A)
    R = np.asarray(R)
    if n_actions is None:
        n_actions = int(A.max()) + 1
    Q = _resolve_q(q_hat, X, A, R, n_actions)
    pi_mat = _policy_matrix(pi_target, X, len(A), n_actions)
    V_per = (Q * pi_mat).sum(axis=1)
    V = float(V_per.mean())
    se = float(V_per.std(ddof=1) / np.sqrt(len(A)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return _wrap(
        "direct",
        V,
        se,
        (V - crit * se, V + crit * se),
        n_obs=len(A),
        n_actions=int(n_actions),
    )


def ips(
    X: np.ndarray,
    A: np.ndarray,
    R: np.ndarray,
    pi_target: Any,
    pi_behavior: Optional[np.ndarray] = None,
    clip: float = 50.0,
    alpha: float = 0.05,
) -> OPEResult:
    """Inverse propensity score OPE.

    ``V = mean_i w_i R_i`` with ``w_i = min(pi_e(A_i|X_i) / pi_b(A_i|X_i),
    clip)``; ``pi_target`` 1-D is ``pi_e(A_i|X_i)``, 2-D is ``(n, K)``
    probabilities. ``clip`` is ``obp``'s ``lambda_`` (``np.inf`` for no
    cap). SE ``sd(w R)/sqrt(n)``.

    .. versionchanged:: 1.30.0
       ``clip`` caps the weight only; ``pi_b`` is no longer floored at
       ``1/clip``.

    Notes
    -----
    If ``pi_behavior`` is None and the internal behavior-policy logistic
    regression fails, propensities fall back to uniform ``1/K``; a
    ``ConvergenceWarning`` is emitted and
    ``diagnostics['propensity_fallback']`` is set to True.

    Examples
    --------
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> X = rng.normal(size=(n, 3))
    >>> A = rng.integers(0, 3, size=n)
    >>> R = rng.normal(size=n)
    >>> pi_target = rng.dirichlet(np.ones(3), size=n)
    >>> res = sp.ips(X, A, R, pi_target)
    >>> v = float(res.value)
    """
    X = np.asarray(X)
    A = np.asarray(A)
    R = np.asarray(R)
    pi_t = _target_prob(pi_target, X, A)
    if pi_behavior is not None:
        pi_b, pi_b_fallback = pi_behavior, False
    else:
        pi_b, pi_b_fallback = _fit_propensity(X, A.astype(int))
    ratio = _behaviour_ratio(pi_t, pi_b, len(A), clip)
    V_per = ratio * R
    V = float(V_per.mean())
    se = float(V_per.std(ddof=1) / np.sqrt(len(A)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return _wrap(
        "IPS",
        V,
        se,
        (V - crit * se, V + crit * se),
        n_obs=len(A),
        ess=float(ratio.sum() ** 2 / max(np.sum(ratio**2), 1e-12)),
        weight_max=float(np.max(ratio)),
        weight_mean=float(np.mean(ratio)),
        propensity_fallback=pi_b_fallback,
    )


def snips(
    X: np.ndarray,
    A: np.ndarray,
    R: np.ndarray,
    pi_target: Any,
    pi_behavior: Optional[np.ndarray] = None,
    clip: float = 50.0,
    alpha: float = 0.05,
) -> OPEResult:
    """Self-normalised IPS (bias-reduction for large IS weights).

    ``V = sum_i w_i R_i / sum_i w_i`` (``obp``'s
    ``SelfNormalizedInverseProbabilityWeighting``), weights as in
    :func:`ips`. SE is the delta-method ``sd(w (R - V)) / (mean(w) sqrt(n))``.

    .. versionchanged:: 1.30.0
       The SE divided by ``sum(w)`` where ``mean(w)`` belongs, so it was
       too small by a factor of about ``sqrt(n)`` (0.00039 against a
       bootstrap 0.0174 at n = 2000). ``clip`` no longer floors ``pi_b``.

    Notes
    -----
    If ``pi_behavior`` is None and the internal behavior-policy logistic
    regression fails, propensities fall back to uniform ``1/K``; a
    ``ConvergenceWarning`` is emitted and
    ``diagnostics['propensity_fallback']`` is set to True.

    Examples
    --------
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> X = rng.normal(size=(n, 3))
    >>> A = rng.integers(0, 3, size=n)
    >>> R = rng.normal(size=n)
    >>> pi_target = rng.dirichlet(np.ones(3), size=n)
    >>> res = sp.snips(X, A, R, pi_target)
    >>> v = float(res.value)
    """
    X = np.asarray(X)
    A = np.asarray(A)
    R = np.asarray(R)
    pi_t = _target_prob(pi_target, X, A)
    if pi_behavior is not None:
        pi_b, pi_b_fallback = pi_behavior, False
    else:
        pi_b, pi_b_fallback = _fit_propensity(X, A.astype(int))
    ratio = _behaviour_ratio(pi_t, pi_b, len(A), clip)
    n = len(A)
    w_bar = float(ratio.mean())
    if w_bar <= 0:
        raise IdentificationFailure(
            "SNIPS: all importance weights are zero.",
            recovery_hint=(
                "The evaluation policy puts no mass on any logged action; check "
                "overlap between pi_target and the logged actions."
            ),
        )
    V = float(np.mean(ratio * R) / w_bar)
    # Delta method for the ratio of means: the influence function is
    # (w_i R_i - V w_i) / mean(w).
    se = float(np.std(ratio * (R - V), ddof=1) / (w_bar * np.sqrt(n)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return _wrap(
        "SNIPS",
        V,
        se,
        (V - crit * se, V + crit * se),
        n_obs=n,
        ess=float(ratio.sum() ** 2 / max(np.sum(ratio**2), 1e-12)),
        weight_max=float(np.max(ratio)),
        weight_mean=float(np.mean(ratio)),
        propensity_fallback=pi_b_fallback,
    )


def doubly_robust(
    X: np.ndarray,
    A: np.ndarray,
    R: np.ndarray,
    pi_target: Any,
    pi_behavior: Optional[np.ndarray] = None,
    n_actions: Optional[int] = None,
    clip: float = 50.0,
    alpha: float = 0.05,
    q_hat: Optional[np.ndarray] = None,
    n_folds: int = 1,
) -> OPEResult:
    """Doubly-robust OPE (Dudik et al. 2011).

    ``V = mean_i [ sum_a pi_e(a|X_i) Q(X_i, a) + w_i (R_i - Q(X_i, A_i)) ]``
    with ``w_i = min(pi_e(A_i|X_i) / pi_b(A_i|X_i), clip)``. ``Q`` is
    ``q_hat`` when given, else a random forest. With ``n_folds > 1`` the
    forest is cross-fitted, so that each observation's ``Q`` comes from a
    forest that did not see it; the default ``n_folds=1`` fits and
    predicts on the same rows. A 1-D ``pi_target`` is a
    deterministic action vector. With the same ``Q`` and
    ``clip = lambda_`` this is ``obp``'s ``DoublyRobust``.

    Notes
    -----
    If ``pi_behavior`` is None and the internal behavior-policy logistic
    regression fails, propensities fall back to uniform ``1/K``; a
    ``ConvergenceWarning`` is emitted and
    ``diagnostics['propensity_fallback']`` is set to True.

    Examples
    --------
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> X = rng.normal(size=(n, 3))
    >>> A = rng.integers(0, 3, size=n)
    >>> R = rng.normal(size=n)
    >>> pi_target = rng.dirichlet(np.ones(3), size=n)
    >>> res = sp.doubly_robust(X, A, R, pi_target)
    >>> v = float(res.value)  # doctest: +SKIP
    """
    X = np.asarray(X)
    A = np.asarray(A)
    R = np.asarray(R)
    if n_actions is None:
        n_actions = int(A.max()) + 1
    A = A.astype(int)
    pi_mat = _policy_matrix(pi_target, X, len(A), n_actions)
    # The target probability of the *logged* action comes from the same
    # matrix: before 1.30.0 a 1-D action vector was read here as a vector
    # of probabilities, so the correction term was weighted by action
    # indices.
    pi_t_a = pi_mat[np.arange(len(A)), A]
    if pi_behavior is not None:
        pi_b, pi_b_fallback = pi_behavior, False
    else:
        pi_b, pi_b_fallback = _fit_propensity(X, A)

    if n_folds < 1:
        raise MethodIncompatibility(
            "n_folds must be at least 1.",
            recovery_hint="Use n_folds=1 for no cross-fitting.",
        )
    Q = _resolve_q(q_hat, X, A, R, n_actions, n_folds=int(n_folds))
    Q_pi = (Q * pi_mat).sum(axis=1)
    resid = R - Q[np.arange(len(A)), A]
    ratio = _behaviour_ratio(pi_t_a, pi_b, len(A), clip)
    V_per = Q_pi + ratio * resid
    V = float(V_per.mean())
    se = float(V_per.std(ddof=1) / np.sqrt(len(A)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return _wrap(
        "Doubly Robust",
        V,
        se,
        (V - crit * se, V + crit * se),
        n_obs=len(A),
        ess=float(ratio.sum() ** 2 / max(np.sum(ratio**2), 1e-12)),
        weight_max=float(np.max(ratio)),
        n_actions=int(n_actions),
        propensity_fallback=pi_b_fallback,
    )


__all__ = ["direct_method", "ips", "snips", "doubly_robust", "OPEResult"]
