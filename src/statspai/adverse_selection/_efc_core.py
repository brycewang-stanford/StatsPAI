"""Shared pieces of the Einav-Finkelstein-Cullen welfare estimator.

The closed-form linear-curve quantities, the stacked OLS influence
functions, the delta-method helpers and the bootstrap used by
:func:`statspai.efc_welfare` (``tests.py``) and by its non-linear demand
specifications (``_efc_nonlinear.py``).
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from ..exceptions import AssumptionWarning, MethodIncompatibility, NumericalInstability

_EFC_ROWS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "price_eq",
    "coverage_eq",
    "price_eff",
    "coverage_eff",
    "dwl",
    "surplus_eff",
    "dwl_share",
)


def _efc_quantities(
    alpha: float, beta: float, gamma: float, delta: float
) -> np.ndarray:
    """Equilibrium, efficient allocation and deadweight loss from four numbers.

    ``D(p) = alpha + beta p`` and ``AC(p) = gamma + delta p``. In coverage
    space the inverse demand is ``P(D) = (D - alpha) / beta`` and the gap
    between demand and marginal cost is linear, ``g(D) = c0 + c1 D``.
    """

    def inv(q: float) -> float:
        return (q - alpha) / beta

    c0 = -(1.0 - delta) * alpha / beta - gamma
    c1 = (1.0 - 2.0 * delta) / beta

    def surplus(q: float) -> float:
        return c0 * q + 0.5 * c1 * q * q

    # Competitive equilibrium: price equals average cost.
    q_eq = float(np.clip(alpha + beta * gamma / (1.0 - delta), 0.0, 1.0))
    p_eq = gamma + delta * inv(q_eq)
    # Efficient coverage maximises the surplus on [0, 1].
    cands = [0.0, 1.0]
    if c1 != 0.0:
        root = -c0 / c1
        if 0.0 < root < 1.0:
            cands.append(root)
    q_eff = max(cands, key=surplus)
    p_eff = inv(q_eff)
    dwl = surplus(q_eff) - surplus(q_eq)
    s_eff = surplus(q_eff)
    share = dwl / s_eff if s_eff > 0 else np.nan
    return np.array(
        [alpha, beta, gamma, delta, p_eq, q_eq, p_eff, q_eff, dwl, s_eff, share]
    )


def _ols_influence(
    X: np.ndarray, y: np.ndarray, n_total: int, label: str
) -> Tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and their influence rows scaled to ``n_total``."""
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise NumericalInstability(
            f"the {label} regression is rank deficient",
            recovery_hint="Drop collinear covariates; price must vary within "
            "the sample used for this regression.",
        )
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ (X.T @ y)
    e = y - X @ b
    return b, n_total * (X * e[:, None]) @ xtx_inv


def _efc_fit(
    df: pd.DataFrame, price: str, coverage: str, cost: str, xs: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Stacked parameters (demand, cost, covariate means) and influence rows."""
    n = len(df)
    X = np.column_stack(
        [np.ones(n), df[price].to_numpy(dtype=float)]
        + [df[c].to_numpy(dtype=float) for c in xs]
    )
    d = df[coverage].to_numpy(dtype=float)
    covered = d == 1
    b_d, if_d = _ols_influence(X, d, n, "demand")
    b_c, if_c_sub = _ols_influence(
        X[covered], df.loc[covered, cost].to_numpy(dtype=float), n, "average cost"
    )
    if_c = np.zeros((n, X.shape[1]))
    if_c[covered] = if_c_sub
    xbar = X[:, 2:].mean(axis=0)
    if_x = X[:, 2:] - xbar
    theta = np.concatenate([b_d, b_c, xbar])
    return theta, np.hstack([if_d, if_c, if_x])


def _efc_from_theta(theta: np.ndarray, m: int) -> np.ndarray:
    k = 2 + m
    b_d, b_c, xbar = theta[:k], theta[k : 2 * k], theta[2 * k :]
    alpha = b_d[0] + float(xbar @ b_d[2:])
    gamma = b_c[0] + float(xbar @ b_c[2:])
    return _efc_quantities(alpha, b_d[1], gamma, b_c[1])


def _stacked_vcov(infl: np.ndarray, clusters: Optional[np.ndarray]) -> np.ndarray:
    n = infl.shape[0]
    if clusters is not None:
        infl = pd.DataFrame(infl).groupby(clusters).sum().to_numpy()
    out: np.ndarray = np.asarray(infl.T @ infl) / n**2
    return out


def _numeric_jacobian(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray
) -> np.ndarray:
    f0 = f(x)
    jac = np.empty((f0.size, x.size))
    for j in range(x.size):
        h = 1e-6 * max(1.0, abs(x[j]))
        up, dn = x.copy(), x.copy()
        up[j] += h
        dn[j] -= h
        jac[:, j] = (f(up) - f(dn)) / (2 * h)
    return jac


def _efc_bootstrap(
    df: pd.DataFrame,
    price: str,
    coverage: str,
    cost: str,
    xs: List[str],
    clusters: Optional[np.ndarray],
    n_boot: int,
    seed: Optional[int],
    estimate: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    n_out: int = len(_EFC_ROWS),
    failures: Tuple[Type[BaseException], ...] = (
        NumericalInstability,
        np.linalg.LinAlgError,
    ),
) -> Tuple[np.ndarray, int]:
    """Resample rows (or clusters) and refit; ``estimate`` maps a resample to
    the reported quantities (default: the linear curves)."""
    if n_boot < 2:
        raise MethodIncompatibility(
            "n_boot must be at least 2", recovery_hint="Pass e.g. n_boot=500."
        )
    rng = np.random.default_rng(seed)
    if clusters is None:
        groups = [np.array([i]) for i in range(len(df))]
    else:
        codes, uniq = pd.factorize(clusters)
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
        groups = [order[bounds[g] : bounds[g + 1]] for g in range(len(uniq))]
    out = np.full((n_boot, n_out), np.nan)
    failed = 0
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[g] for g in pick])
        sub = df.iloc[rows].reset_index(drop=True)
        try:
            if estimate is None:
                theta, _ = _efc_fit(sub, price, coverage, cost, xs)
                out[b] = _efc_from_theta(theta, len(xs))
            else:
                out[b] = estimate(sub)
        except failures:
            failed += 1
    if failed > n_boot // 10:
        warnings.warn(
            AssumptionWarning(
                f"{failed} of {n_boot} bootstrap draws could not be fitted",
                recovery_hint="The curves are weakly identified in resamples; "
                "treat the bootstrap interval with caution.",
            ),
            stacklevel=3,
        )
    return out, failed
