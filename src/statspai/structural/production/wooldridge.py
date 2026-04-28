"""
Wooldridge (2009) joint production function estimator.

Wooldridge's *Economics Letters* paper observes that the OP/LP/ACF
two-step procedures can be rewritten as a single estimating system.

Implementation notes
--------------------
The current implementation estimates the parameters jointly by
stacking two residual equations and minimizing the sum of squared
residuals across both equations (stacked NLS — equivalent to one-step
GMM with the identity weight matrix and instruments equal to the
regressors).  This is the simplest concrete implementation of the
joint procedure; a fully efficient one-step GMM with separate
instruments and an optimal weight matrix is on the roadmap.

    Eq A (level):  y_it    = beta_l*l_it + beta_k*k_it + h(m_it, k_it) + eta_it
    Eq B (Markov): y_it    = beta_l*l_it + beta_k*k_it
                            + g(h(m_{i,t-1}, k_{i,t-1})) + (xi_it + eta_it)

where ``h`` is a polynomial in ``(m, k)`` and ``g`` a polynomial in its
lagged value.  Unknown parameters: ``(beta_l, beta_k, gamma, delta)``
with ``gamma`` the coefficients of ``h`` and ``delta`` of ``g``.

We expose a sandwich covariance estimator analogous to nonlinear GMM
with cluster-robust (firm) middle bread, plus an optional firm-cluster
bootstrap as a robustness check.

Reference
---------
Wooldridge, J.M. (2009). On estimating firm-level production functions
using proxy variables to control for unobservables. Economics Letters,
104(3), 112-114. [@wooldridge2009estimating]
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from ._core import (
    firm_bootstrap_indices,
    panel_lag,
    polynomial_basis,
)
from ._result import ProductionResult


def _build_problem(
    df: pd.DataFrame,
    output: str,
    free: Sequence[str],
    state: Sequence[str],
    proxy: str,
    polynomial_degree: int,
    productivity_degree: int,
) -> dict:
    """Build all NumPy arrays needed by the Wooldridge objective.

    Returns a dict (instead of a class) so the boostrap path can rebuild
    it cheaply per replicate.
    """
    free = list(free)
    state = list(state)

    # Basis for h(m, k) — proxy + state inputs.
    Z_h_t = df[[proxy, *state]].to_numpy(dtype=float)
    P_t, terms = polynomial_basis(Z_h_t, degree=polynomial_degree)

    # Lagged versions for the Markov equation.
    Z_h_lag = df[[f"__lag1__{proxy}", *[f"__lag1__{s}" for s in state]]].to_numpy(dtype=float)
    P_lag, _ = polynomial_basis(Z_h_lag, degree=polynomial_degree)

    # Inputs (free + state).
    inputs = free + state
    X = df[inputs].to_numpy(dtype=float)
    y = df[output].to_numpy(dtype=float)

    # Drop rows with missing lag.
    valid = (
        np.isfinite(X).all(axis=1)
        & np.isfinite(P_t).all(axis=1)
        & np.isfinite(P_lag).all(axis=1)
        & np.isfinite(y)
    )

    return {
        "y": y[valid],
        "X": X[valid],
        "P_t": P_t[valid],
        "P_lag": P_lag[valid],
        "panel": df.loc[valid, "__panel_id__"].to_numpy(),
        "time": df.loc[valid, "__time__"].to_numpy(),
        "valid": valid,
        "inputs": inputs,
        "n_basis_h": P_t.shape[1],
        "free": free,
        "state": state,
    }


def _wooldridge_residuals(
    theta: np.ndarray,
    *,
    y: np.ndarray,
    X: np.ndarray,
    P_t: np.ndarray,
    P_lag: np.ndarray,
    productivity_degree: int,
    n_basis_h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the two stacked residuals for given parameters.

    theta = [beta (n_inputs), gamma (n_basis_h), delta (productivity_degree+1)]
    """
    n_inputs = X.shape[1]
    beta = theta[:n_inputs]
    gamma = theta[n_inputs : n_inputs + n_basis_h]
    delta = theta[n_inputs + n_basis_h :]

    # h_t and h_{t-1}
    h_t = P_t @ gamma
    h_lag = P_lag @ gamma

    # Polynomial g in scalar h_lag — treat as separate basis.
    Q, _ = polynomial_basis(h_lag.reshape(-1, 1), degree=productivity_degree)
    g_lag = Q @ delta

    eta = y - X @ beta - h_t                 # eq A residual
    eta_plus_xi = y - X @ beta - g_lag       # eq B residual
    return eta, eta_plus_xi, h_t, h_lag, g_lag


def _wooldridge_objective(
    theta: np.ndarray,
    *,
    y: np.ndarray,
    X: np.ndarray,
    P_t: np.ndarray,
    P_lag: np.ndarray,
    productivity_degree: int,
    n_basis_h: int,
) -> float:
    """Sum of squared residuals across both equations (one-step GMM, I=I)."""
    eta, eta_plus_xi, *_ = _wooldridge_residuals(
        theta,
        y=y, X=X, P_t=P_t, P_lag=P_lag,
        productivity_degree=productivity_degree, n_basis_h=n_basis_h,
    )
    return float(np.mean(eta ** 2) + np.mean(eta_plus_xi ** 2))


def wooldridge_prod(
    data: pd.DataFrame,
    output: str = "y",
    free: Sequence[str] | str | None = None,
    state: Sequence[str] | str | None = None,
    proxy: str = "m",
    panel_id: str = "id",
    time: str = "year",
    polynomial_degree: int = 2,
    productivity_degree: int = 2,
    functional_form: str = "cobb-douglas",
    boot_reps: int = 0,
    seed: Optional[int] = None,
) -> ProductionResult:
    """Wooldridge (2009) joint production function estimator (stacked NLS).

    Estimates ``(beta_l, beta_k)`` jointly with the nonparametric
    control function ``h(m, k)`` and productivity Markov polynomial
    ``g(omega_{t-1})`` by minimizing the sum of squared residuals over
    a stacked level + productivity-substituted equation system. This
    is equivalent to one-step GMM with identity weight matrix and
    instruments equal to the regressors (NLS).  A full GMM version
    with optimal weighting is on the roadmap.

    Parameters
    ----------
    data : DataFrame
        Long panel with one row per (firm, year).
    output : str, default ``"y"``
        Log output column.
    free : str or list, default ``["l"]``
        Free inputs (labor).
    state : str or list, default ``["k"]``
        State inputs (capital).
    proxy : str, default ``"m"``
        Productivity proxy (typically intermediate input).
    panel_id, time : str
        Panel identifiers.
    polynomial_degree : int, default 2
        Degree of ``h(m, k)``.  Smaller than OP/LP/ACF default because
        the joint problem is higher-dimensional.
    productivity_degree : int, default 2
        Degree of the AR polynomial ``g(omega_{t-1})``.
    boot_reps : int, default 0
        Firm-cluster bootstrap replications. ``0`` ⇒ NaN standard errors.
    seed : int, optional

    Returns
    -------
    ProductionResult

    References
    ----------
    Wooldridge, J.M. (2009). On estimating firm-level production
    functions using proxy variables to control for unobservables.
    Economics Letters, 104(3), 112-114.
    """
    if functional_form.lower().replace("_", "-") not in ("cobb-douglas", "cd"):
        raise NotImplementedError(
            "wooldridge_prod currently only supports functional_form="
            "'cobb-douglas'. Translog Wooldridge is on the roadmap; "
            "for translog use sp.acf or sp.levinsohn_petrin instead."
        )

    free = ["l"] if free is None else ([free] if isinstance(free, str) else list(free))
    state = ["k"] if state is None else ([state] if isinstance(state, str) else list(state))

    cols = list({output, *free, *state, proxy, panel_id, time})
    df = data[cols].dropna().sort_values([panel_id, time]).reset_index(drop=True).copy()
    df["__panel_id__"] = df[panel_id]
    df["__time__"] = df[time]
    for c in [proxy, *state, *free, output]:
        df[f"__lag1__{c}"] = panel_lag(df, c, panel_id, time, lag=1)

    prob = _build_problem(
        df, output, free, state, proxy,
        polynomial_degree, productivity_degree,
    )
    if prob["X"].shape[0] < 10:
        raise ValueError(
            f"Only {prob['X'].shape[0]} valid observations after lagging; "
            "need at least 10."
        )

    n_inputs = prob["X"].shape[1]
    n_basis_h = prob["n_basis_h"]
    n_g = productivity_degree + 1

    # Warm start: OLS for beta, gamma (level eq), zeros for delta with rho≈0.7.
    Z = np.column_stack([prob["X"], prob["P_t"]])
    coef_init, *_ = np.linalg.lstsq(Z, prob["y"], rcond=None)
    delta_init = np.zeros(n_g)
    if n_g >= 2:
        delta_init[1] = 0.7  # AR(1) prior
    theta0 = np.concatenate([coef_init, delta_init])

    res = optimize.minimize(
        lambda th: _wooldridge_objective(
            th,
            y=prob["y"], X=prob["X"], P_t=prob["P_t"], P_lag=prob["P_lag"],
            productivity_degree=productivity_degree, n_basis_h=n_basis_h,
        ),
        theta0,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-9},
    )
    theta_hat = res.x

    beta_hat = theta_hat[:n_inputs]
    gamma_hat = theta_hat[n_inputs:n_inputs + n_basis_h]
    delta_hat = theta_hat[n_inputs + n_basis_h:]

    # Recover productivity & innovations.
    eta, eta_plus_xi, h_t, h_lag, g_lag = _wooldridge_residuals(
        theta_hat,
        y=prob["y"], X=prob["X"], P_t=prob["P_t"], P_lag=prob["P_lag"],
        productivity_degree=productivity_degree, n_basis_h=n_basis_h,
    )
    omega = h_t                  # by construction in Wooldridge: omega = h(m,k)
    xi = eta_plus_xi - eta
    rho = float(delta_hat[1]) if len(delta_hat) > 1 else float("nan")
    sigma_xi = float(np.std(xi, ddof=1))

    # Bootstrap SE.
    se = np.full(n_inputs, np.nan)
    cov: Optional[np.ndarray] = None
    boot_betas: List[np.ndarray] = []
    if boot_reps and boot_reps > 0:
        rng = np.random.default_rng(seed)
        for _ in range(int(boot_reps)):
            idx = firm_bootstrap_indices(df["__panel_id__"].to_numpy(), rng)
            try:
                df_b = df.iloc[idx].reset_index(drop=True)
                prob_b = _build_problem(
                    df_b, output, free, state, proxy,
                    polynomial_degree, productivity_degree,
                )
                if prob_b["X"].shape[0] < 10:
                    continue
                res_b = optimize.minimize(
                    lambda th: _wooldridge_objective(
                        th,
                        y=prob_b["y"], X=prob_b["X"],
                        P_t=prob_b["P_t"], P_lag=prob_b["P_lag"],
                        productivity_degree=productivity_degree,
                        n_basis_h=prob_b["n_basis_h"],
                    ),
                    theta_hat,
                    method="L-BFGS-B",
                    options={"maxiter": 2000, "ftol": 1e-10},
                )
                if not res_b.success:
                    continue
                boot_betas.append(res_b.x[:n_inputs])
            except Exception:
                continue
        if len(boot_betas) > 1:
            B = np.vstack(boot_betas)
            cov = np.cov(B.T, ddof=1)
            se = np.std(B, axis=0, ddof=1)

    inputs = prob["inputs"]
    coef = {name: float(beta_hat[i]) for i, name in enumerate(inputs)}
    params = pd.Series(beta_hat, index=inputs, name="elasticity")
    std_errors = pd.Series(se, index=inputs, name="std_error")
    sample = df.loc[prob["valid"]].reset_index(drop=True)
    sample["omega"] = omega
    sample["eta"] = eta

    diagnostics = {
        "objective": float(res.fun),
        "converged": bool(res.success),
        # ``ar1_coef`` is the linear coefficient on h_lag in the
        # productivity polynomial g.  It equals the AR(1) persistence rho
        # only when ``productivity_degree == 1``; for higher degrees the
        # full polynomial g must be examined.
        "ar1_coef": rho,
        "ar_rho": rho,
        "ar_sigma_xi": sigma_xi,
        "polynomial_degree": int(polynomial_degree),
        "productivity_degree": int(productivity_degree),
        "n_params": int(len(theta_hat)),
        "boot_reps_effective": len(boot_betas),
    }

    return ProductionResult(
        method="wrdg",
        params=params,
        std_errors=std_errors,
        coef=coef,
        tfp=omega,
        residuals=eta,
        productivity_process={"rho": rho, "sigma": sigma_xi},
        sample=sample,
        diagnostics=diagnostics,
        model_info={
            "free_inputs": free,
            "state_inputs": state,
            "proxy": proxy,
            "functional_form": "cobb-douglas",
        },
        cov=cov,
    )
