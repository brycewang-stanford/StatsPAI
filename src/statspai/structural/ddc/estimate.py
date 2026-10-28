"""``sp.ddc``: estimation of single-agent dynamic discrete choice models.

Infinite-horizon, stationary, discrete-state models with additively
separable i.i.d. type-I extreme value shocks and flow utility linear in
the parameters. Three estimators of the same structural parameters:

* ``'nfxp'``: Rust's nested fixed point. The Bellman fixed point is
  re-solved at every trial parameter (successive approximation, then
  Newton-Kantorovich steps, so ``beta = 0.9999`` is cheap), and the partial
  likelihood of the observed choices is maximised with an analytic
  gradient (implicit function theorem) followed by Newton polishing.
* ``'ccp'``: Hotz-Miller conditional choice probabilities. Frequency CCPs
  give the value function by one matrix inversion, so the choice values
  are linear in the parameters and the pseudo-likelihood is a conditional
  logit with an offset (the pseudo-likelihood form of the Hotz-Miller
  estimator, which is Aguirregabiria and Mira's one-stage estimator).
* ``'npl'``: Aguirregabiria and Mira's nested pseudo-likelihood. Repeats
  the CCP step with the probabilities implied by the last estimate; its
  fixed point is the partial maximum likelihood estimator.

References
----------
[@rust1987optimal]
[@hotz1993conditional]
[@aguirregabiria2002swapping]
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize

from ...core._call_record import records_call
from ...exceptions import ConvergenceWarning, MethodIncompatibility
from ._core import (
    centred_scores,
    fit_linear_logit,
    hm_valuation,
    loglik_from_P,
    softmax_rows,
    solve_value,
    value_derivative,
)
from ._data import (
    build_features,
    check_increments,
    check_transitions,
    encode_panel,
    estimate_transitions,
    increments_matrix,
)
from ._result import DDCResult

__all__ = ["ddc"]


def _model_scores(
    phi: np.ndarray, F: np.ndarray, beta: float, theta: np.ndarray, counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Model CCPs, value, per-(s, a) scores and the log-likelihood at theta."""
    u = phi @ theta
    V, v, P, _ = solve_value(u, F, beta)
    g = centred_scores(value_derivative(phi, F, beta, P), P)
    return V, v, P, g, loglik_from_P(counts, P)


def _numerical_hessian(
    grad_fn: Callable[[np.ndarray], np.ndarray], theta: np.ndarray
) -> np.ndarray:
    K = theta.size
    H = np.empty((K, K))
    for k in range(K):
        h = 1e-5 * max(1.0, abs(theta[k]))
        e = np.zeros(K)
        e[k] = h
        H[:, k] = (grad_fn(theta + e) - grad_fn(theta - e)) / (2 * h)
    return 0.5 * (H + H.T)


def _nfxp(
    phi: np.ndarray,
    F: np.ndarray,
    beta: float,
    counts: np.ndarray,
    theta0: np.ndarray,
    tol: float,
) -> Tuple[np.ndarray, bool, int]:
    """Maximise the partial likelihood; returns (theta, converged, iterations)."""
    N = counts.sum()
    cache: Dict[str, Any] = {"V": None}

    def solve(theta: np.ndarray) -> np.ndarray:
        V, v, P, _ = solve_value(phi @ theta, F, beta, V0=cache["V"])
        cache["V"] = V
        return P

    def negll(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        P = solve(theta)
        g = centred_scores(value_derivative(phi, F, beta, P), P)
        grad = np.einsum("sj,sjk->k", counts, g)
        return -loglik_from_P(counts, P) / N, -grad / N

    res = optimize.minimize(
        negll, theta0, jac=True, method="BFGS", options={"gtol": 1e-9, "maxiter": 2000}
    )
    theta = res.x
    iterations = int(res.nit)

    def grad_fn(th: np.ndarray) -> np.ndarray:
        return np.asarray(-negll(th)[1] * N)

    converged = False
    for _ in range(20):
        g = grad_fn(theta)
        H = _numerical_hessian(grad_fn, theta)
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        iterations += 1
        if np.max(np.abs(step)) <= tol * max(1.0, np.max(np.abs(theta))):
            theta = theta + step
            converged = True
            break
        if float(g @ step) <= 0:  # Hessian not negative definite here
            break
        f0 = -negll(theta)[0]
        t = 1.0
        while t > 1e-8 and -negll(theta + t * step)[0] < f0 - 1e-15:
            t *= 0.5
        theta = theta + t * step
        if np.max(np.abs(t * step)) <= tol * max(1.0, np.max(np.abs(theta))):
            converged = True
            break
    return theta, converged, iterations


def _reset_indices(
    reset_actions: Optional[Sequence[Any]], utility: Any, labels: Sequence[Any]
) -> List[int]:
    """Model indices of the actions that restart the state at 0."""
    if reset_actions is None:
        return [1] if isinstance(utility, str) else []
    unknown = [r for r in reset_actions if r not in labels]
    if unknown:
        raise MethodIncompatibility(
            f"reset_actions {unknown} are not action labels",
            recovery_hint=f"use labels from {list(labels)}",
        )
    return [list(labels).index(r) for r in reset_actions]


def _ccp_first_stage(counts: np.ndarray, smoothing: float) -> np.ndarray:
    J = counts.shape[1]
    n_s = counts.sum(axis=1, keepdims=True)
    return np.asarray((counts + smoothing) / (n_s + J * smoothing))


def _vcov(
    vce: str,
    phi: np.ndarray,
    F: np.ndarray,
    beta: float,
    theta: np.ndarray,
    counts: np.ndarray,
    enc: Dict[str, Any],
) -> np.ndarray:
    """Variance of theta from the partial likelihood at theta."""
    K = theta.size

    def grad_fn(th: np.ndarray) -> np.ndarray:
        _, _, P, g, _ = _model_scores(phi, F, beta, th, counts)
        return np.asarray(np.einsum("sj,sjk->k", counts, g))

    _, _, P, g, _ = _model_scores(phi, F, beta, theta, counts)
    H = _numerical_hessian(grad_fn, theta)
    A = -H
    if vce == "hessian":
        return np.linalg.inv(A)
    obs = g[enc["s"], enc["a"]]  # (N, K) per-observation scores
    if vce == "opg":
        return np.linalg.inv(obs.T @ obs)
    # sandwich, clustered by unit when the panel id is known
    if enc["ids"] is not None:
        codes = pd.factorize(enc["ids"])[0]
        sc = np.zeros((codes.max() + 1, K))
        np.add.at(sc, codes, obs)
    else:
        sc = obs
    Ainv = np.linalg.inv(A)
    return np.asarray(Ainv @ (sc.T @ sc) @ Ainv)


@records_call("ddc")
def ddc(
    data: pd.DataFrame,
    choice: str,
    state: str,
    id: Optional[str] = None,
    time: Optional[str] = None,
    method: Literal["nfxp", "ccp", "npl"] = "nfxp",
    utility: Union[str, np.ndarray, Callable] = "rust",
    beta: float = 0.95,
    transition: Union[str, np.ndarray, Dict[Any, np.ndarray]] = "auto",
    actions: Optional[Sequence[Any]] = None,
    n_states: Optional[int] = None,
    cost: Literal["linear", "quadratic"] = "linear",
    cost_scale: float = 0.001,
    reset_actions: Optional[Sequence[Any]] = None,
    param_names: Optional[Sequence[str]] = None,
    theta0: Optional[Sequence[float]] = None,
    ccp_smoothing: float = 0.5,
    npl_max_iter: int = 100,
    vce: Literal["hessian", "opg", "sandwich"] = "hessian",
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> DDCResult:
    """Estimate a single-agent dynamic discrete choice model.

    The agent chooses action ``a`` in state ``s`` to maximise
    ``u(s, a; theta) + eps_a + beta * E[V(s') | s, a]`` with ``eps`` i.i.d.
    type-I extreme value and ``u = phi(s, a)' theta``. The discount factor
    is fixed; transitions are estimated in a first stage (or supplied) and
    the utility parameters maximise the partial likelihood of the choices.

    Parameters
    ----------
    data : pd.DataFrame
        One row per unit and period.
    choice : str
        Action column.
    state : str
        Discretised state: integer codes ``0, ..., S-1``.
    id, time : str, optional
        Panel identifiers. Needed to estimate transitions (consecutive
        periods of the same unit); ``id`` also clusters ``vce='sandwich'``.
    method : {'nfxp', 'ccp', 'npl'}, default 'nfxp'
        Nested fixed point, two-step Hotz-Miller CCP, or nested
        pseudo-likelihood.
    utility : 'rust', array or callable, default 'rust'
        ``'rust'`` is Rust's engine-replacement model: action 0 keeps
        (``-c(s)``), action 1 replaces (``-RC - c(0)``), with
        ``c(s) = theta_1 * cost_scale * s`` (plus ``theta_2 (cost_scale s)^2``
        when ``cost='quadratic'``). Otherwise an array ``phi`` of shape
        ``(S, J, K)`` or a callable ``f(s, j) -> (K,)`` giving the features
        of state ``s`` and action index ``j``.
    beta : float, default 0.95
        Discount factor in ``[0, 1)``. ``beta = 0`` is a static logit.
    transition : 'auto', 'estimate', 'increments', array or dict
        ``'estimate'``: action-specific frequency matrices from consecutive
        rows. ``'increments'``: Rust's first stage, one distribution of
        state increments pooled over states, measured from 0 after a
        ``reset_actions`` choice. ``'auto'`` is ``'increments'`` for
        ``utility='rust'`` and ``'estimate'`` otherwise. An array of shape
        ``(J, S, S)`` (or ``{action: (S, S)}``) is used as given. A 1-D
        array ``p`` is an increment distribution, ``p[d] = Pr(s' = s + d)``
        (from 0 after a ``reset_actions`` choice, capped at the last state),
        such as Rust's published ``theta_3`` or one computed from the
        ``increment`` column of ``sp.datasets.rust_bus()``.
    actions : sequence, optional
        Action labels in model order (default: sorted unique values).
    n_states : int, optional
        Number of states (default: largest code + 1).
    cost : {'linear', 'quadratic'}, default 'linear'
        Maintenance cost for ``utility='rust'``.
    cost_scale : float, default 0.001
        Scale of the state in the Rust cost function (Rust's 0.001).
    reset_actions : sequence, optional
        Action labels that restart the state at 0 (``'increments'`` or a
        1-D increment distribution only; default the second action for
        ``utility='rust'``).
    param_names : sequence of str, optional
        Names of the utility parameters for array / callable utilities.
    theta0 : sequence of float, optional
        Starting values (NFXP default: the CCP estimate).
    ccp_smoothing : float, default 0.5
        Pseudo-count added to every (state, action) cell of the frequency
        CCPs, so unvisited states start at ``1 / J``.
    npl_max_iter : int, default 100
        Maximum NPL iterations.
    vce : {'hessian', 'opg', 'sandwich'}, default 'hessian'
        Inverse observed information of the partial likelihood, outer
        product of scores, or the sandwich (clustered by ``id`` when given).
        All are conditional on the first-stage transition estimates.
    tol : float, default 1e-8
        Convergence tolerance on the parameters.
    alpha : float, default 0.05
        Level of the confidence intervals.

    Returns
    -------
    DDCResult
        ``params``, ``se``, ``loglik``, model ``ccp``, ``value``,
        ``transition`` and ``.counterfactual()``.

    Notes
    -----
    The standard errors of all three methods use the partial-likelihood
    information at the estimate. For NFXP and converged NPL this is the
    partial MLE's variance; for the two-step CCP estimator it is its
    asymptotic variance too, because the pseudo-likelihood estimators are
    asymptotically equivalent to the partial MLE in single-agent models
    [@aguirregabiria2002swapping]. None of them account for the sampling
    error of the first-stage transitions (Rust's third, full-likelihood
    step is not implemented). The discount factor is not estimated.

    Rust's own bus data are ``sp.datasets.rust_bus()``. Fitting its
    ``rust_sample`` rows with ``transition=`` set to the frequencies of its
    ``increment`` column and ``vce='opg'`` (Rust's BHHH variance) gives
    the estimates, standard errors and log-likelihoods of Rust (1987),
    Tables VIII-X (``tests/external_parity/test_rust1987_bus.py``). The
    default ``transition='increments'`` measures the step after a
    replacement from state 0, one bin below Rust's convention; on groups
    1-4 that moves ``RC`` and ``theta_1`` by about 0.01 to 0.02.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_ddc(n_units=200, n_periods=100, seed=0)  # RC=10, theta_1=2.5
    >>> fit = sp.ddc(df, "replace", "mileage", id="bus", time="t",
    ...              beta=0.9999, n_states=90)
    >>> fit.params.round(1).tolist()  # doctest: +SKIP
    [9.3, 2.2]
    >>> cf = fit.counterfactual(params={"RC": 5.0})
    >>> cf.attrs["long_run"]["counterfactual"][1] > cf.attrs["long_run"]["base"][1]
    True

    References
    ----------
    [@rust1987optimal]
    [@hotz1993conditional]
    [@aguirregabiria2002swapping]
    """
    if method not in ("nfxp", "ccp", "npl"):
        raise MethodIncompatibility(
            f"method={method!r} is not supported",
            recovery_hint="use 'nfxp', 'ccp' or 'npl'",
        )
    if vce not in ("hessian", "opg", "sandwich"):
        raise MethodIncompatibility(
            f"vce={vce!r} is not supported",
            recovery_hint="use 'hessian', 'opg' or 'sandwich'",
        )
    beta = float(beta)
    if not 0.0 <= beta < 1.0:
        raise MethodIncompatibility(
            f"beta={beta} is outside [0, 1); the infinite-horizon value is undefined",
            recovery_hint="pass a discount factor below one, e.g. 0.95 or 0.9999",
        )
    if ccp_smoothing <= 0:
        raise MethodIncompatibility(
            "ccp_smoothing must be positive",
            recovery_hint="a small pseudo-count such as 0.5 keeps log CCPs finite",
        )

    enc = encode_panel(data, choice, state, id, time, actions, n_states)
    S, labels, counts = enc["S"], enc["labels"], enc["counts"]
    J = len(labels)
    phi, names = build_features(utility, S, J, cost, cost_scale, param_names)
    K = phi.shape[2]
    diag: Dict[str, Any] = {}

    # transitions --------------------------------------------------------
    if isinstance(transition, str):
        mode = transition
        if mode == "auto":
            mode = "increments" if isinstance(utility, str) else "estimate"
        if mode not in ("estimate", "increments"):
            raise MethodIncompatibility(
                f"transition={transition!r} is not supported",
                recovery_hint="use 'auto', 'estimate', 'increments' or an array",
            )
        reset_idx = _reset_indices(reset_actions, utility, labels)
        F, tparams = estimate_transitions(enc, mode, reset_idx, diag)
        tparams["mode"] = mode
    elif not isinstance(transition, dict) and np.ndim(transition) == 1:
        p_inc = check_increments(transition)
        reset_idx = _reset_indices(reset_actions, utility, labels)
        F = increments_matrix(p_inc, S, J, reset_idx)
        tparams = {
            "mode": "increments_supplied",
            "increments": list(range(p_inc.size)),
            "increment_probs": p_inc.tolist(),
        }
    else:
        F = check_transitions(transition, S, J, labels)
        tparams = {"mode": "supplied"}

    unvisited = np.flatnonzero(counts.sum(axis=1) == 0)
    if unvisited.size:
        diag["unvisited_states"] = unvisited.tolist()

    # estimation -----------------------------------------------------------
    P_hat = _ccp_first_stage(counts, ccp_smoothing)
    start = np.zeros(K) if theta0 is None else np.asarray(theta0, dtype=float)
    if start.shape != (K,):
        raise MethodIncompatibility(
            f"theta0 needs {K} values", recovery_hint=f"one per parameter {names}"
        )
    converged = True
    if method in ("ccp", "npl") or theta0 is None:
        Z, e = hm_valuation(phi, F, beta, P_hat)
        theta_ccp, conv_ccp, it_ccp = fit_linear_logit(Z, e, counts, start)
    if method == "ccp":
        theta, converged, iterations = theta_ccp, conv_ccp, it_ccp
    elif method == "npl":
        theta, P = theta_ccp, P_hat
        converged = False
        for iterations in range(1, npl_max_iter + 1):
            P = softmax_rows(Z @ theta + e)
            Z, e = hm_valuation(phi, F, beta, P)
            theta_new, _, _ = fit_linear_logit(Z, e, counts, theta)
            step = np.max(np.abs(theta_new - theta))
            theta = theta_new
            if step <= tol * max(1.0, np.max(np.abs(theta))):
                converged = True
                break
        diag["npl_iterations"] = iterations
    else:
        theta, converged, iterations = _nfxp(
            phi, F, beta, counts, theta_ccp if theta0 is None else start, tol
        )
    if not converged:
        warnings.warn(
            ConvergenceWarning(
                f"sp.ddc(method={method!r}) did not converge",
                recovery_hint=(
                    "try another theta0, a larger npl_max_iter, or method='nfxp'"
                ),
            ),
            stacklevel=2,
        )

    V, v, P, _, loglik = _model_scores(phi, F, beta, theta, counts)
    vc = _vcov(vce, phi, F, beta, theta, counts, enc)
    if not np.all(np.isfinite(vc)) or np.any(np.diag(vc) <= 0):
        warnings.warn(
            ConvergenceWarning(
                "the information matrix is not positive definite at the estimate",
                recovery_hint="check identification of the utility parameters",
            ),
            stacklevel=2,
        )
    idx = pd.Index(names, name="param")
    params = pd.Series(theta, index=idx, name="coef")
    se = pd.Series(np.sqrt(np.clip(np.diag(vc), 0, None)), index=idx, name="se")
    states = pd.Index(range(S), name="state")
    acts = pd.Index(labels, name="action")
    return DDCResult(
        method=method,
        params=params,
        se=se,
        vcov=pd.DataFrame(vc, index=idx, columns=idx),
        loglik=loglik,
        ccp=pd.DataFrame(P, index=states, columns=acts),
        ccp_first_stage=(
            None
            if method == "nfxp"
            else pd.DataFrame(P_hat, index=states, columns=acts)
        ),
        value=pd.Series(V, index=states, name="value"),
        choice_values=pd.DataFrame(v, index=states, columns=acts),
        transition=F,
        transition_params=tparams,
        beta=beta,
        n_obs=int(counts.sum()),
        n_units=None if enc["ids"] is None else int(pd.Series(enc["ids"]).nunique()),
        n_states=S,
        converged=bool(converged),
        iterations=int(iterations),
        vce=vce,
        alpha=alpha,
        estimand="structural utility parameters",
        diagnostics=diag,
        _phi=phi,
    )
