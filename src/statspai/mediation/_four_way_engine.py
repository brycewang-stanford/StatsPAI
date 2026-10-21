"""Regression-based four-way decomposition for any outcome / mediator model.

The four components of VanderWeele (2014) are differences of four
counterfactual means

    g(a, a') = E[ mu(a, M_{a'}) | c ],     g_m(a) = mu(a, m*),

where ``mu(a, m) = h(theta0 + theta1 a + theta2 m + theta3 a m + theta4'c)``
is the outcome model (``h`` the identity for a linear outcome, ``exp`` for a
log-linear one -- Poisson, or logistic read as a risk under the rare-outcome
approximation) and ``M_{a'}`` follows the mediator model at exposure ``a'``:

    CDE     = g_m(a1) - g_m(a0)
    INT_ref = g(a1, a0) - g(a0, a0) - CDE
    INT_med = g(a1, a1) - g(a0, a1) - g(a1, a0) + g(a0, a0)
    PIE     = g(a0, a1) - g(a0, a0)

The expectation over the mediator is closed form: ``E[M]`` for a linear
outcome, and for ``h = exp`` the moment generating function of the
mediator -- ``exp(s mu_M + s^2 sigma^2 / 2)`` for a normal mediator,
``1 - p + p e^s`` for a binary one, with ``s = theta2 + theta3 a``. For a
log-linear outcome each component is divided by ``g(a0, a0)``, giving the
excess-relative-risk scale on which Stata ``med4way`` reports them
(``ereri_*``); they then sum to ``RR - 1``.

Standard errors use the delta method with the exact Jacobian, obtained by
complex-step differentiation of the same function (no truncation error),
and a block-diagonal covariance of the outcome coefficients, the mediator
coefficients and, for a normal mediator, its variance.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np

from ..exceptions import ConvergenceFailure, NumericalInstability

#: Output order of :func:`components`.
NAMES = (
    "total_effect",
    "cde",
    "int_ref",
    "int_med",
    "pie",
    "prop_cde",
    "prop_int_ref",
    "prop_int_med",
    "prop_pie",
    "prop_mediated",
    "prop_interaction",
    "prop_eliminated",
)


def fit_glm(X: np.ndarray, y: np.ndarray, family: str) -> Tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood logit / Poisson fit; returns (coef, inverse information).

    Newton-Raphson on the canonical link (observed = expected information),
    with step halving, iterated until the coefficients stop moving.
    """
    n, k = X.shape
    if np.linalg.matrix_rank(X) < k:
        raise NumericalInstability(
            f"four_way_decomposition: the {family} design is rank deficient.",
            recovery_hint="Drop collinear covariates.",
        )
    if family == "logistic":
        p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        beta = np.zeros(k)
        beta[0] = np.log(p0 / (1 - p0))

        def mean(eta: np.ndarray) -> np.ndarray:
            mu: np.ndarray = 1.0 / (1.0 + np.exp(-eta))
            return mu

        def loglik(eta: np.ndarray) -> float:
            return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

        def weight(mu: np.ndarray) -> np.ndarray:
            return mu * (1.0 - mu)

    else:
        beta = np.zeros(k)
        beta[0] = np.log(max(y.mean(), 1e-8))

        def mean(eta: np.ndarray) -> np.ndarray:
            mu: np.ndarray = np.exp(eta)
            return mu

        def loglik(eta: np.ndarray) -> float:
            return float(np.sum(y * eta - np.exp(eta)))

        def weight(mu: np.ndarray) -> np.ndarray:
            return mu

    eta = X @ beta
    ll = loglik(eta)
    for _ in range(200):
        mu = mean(eta)
        H = X.T @ (X * weight(mu)[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        t = 1.0
        while True:
            cand = beta + t * step
            eta_c = X @ cand
            ll_c = loglik(eta_c)
            if np.isfinite(ll_c) and ll_c >= ll - 1e-12 * abs(ll):
                break
            t /= 2.0
            if t < 1e-10:
                break
        beta, eta, ll = cand, eta_c, ll_c
        if np.max(np.abs(t * step)) <= 1e-12 * (1.0 + np.max(np.abs(beta))):
            break
    else:  # pragma: no cover - 200 Newton steps without convergence
        raise ConvergenceFailure(
            f"four_way_decomposition: the {family} model did not converge.",
            recovery_hint="Check for separation or rescale the covariates.",
        )
    if np.max(np.abs(beta)) > 30 and family == "logistic":
        raise ConvergenceFailure(
            "four_way_decomposition: a logistic coefficient diverges "
            "(complete or quasi-complete separation).",
            recovery_hint="Drop or coarsen the separating covariate.",
        )
    mu = mean(eta)
    V = np.linalg.inv(X.T @ (X * weight(mu)[:, None]))
    return beta, V


def components(
    yreg: str,
    mreg: str,
    k_theta: int,
    k_beta: int,
    cvals: np.ndarray,
    a0: float,
    a1: float,
    m_star: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Map the stacked parameters ``(theta, beta[, sigma2])`` to :data:`NAMES`.

    Written with operations that accept complex input, so that the
    complex-step derivative of the returned function is exact.
    """
    log_linear = yreg != "linear"

    def f(p: np.ndarray) -> np.ndarray:
        th = p[:k_theta]
        be = p[k_theta : k_theta + k_beta]
        s2 = p[k_theta + k_beta] if mreg == "linear" else None

        def base(a: float) -> Any:
            return th[0] + th[1] * a + th[4:] @ cvals

        def slope(a: float) -> Any:
            return th[2] + th[3] * a

        def med_index(a: float) -> Any:
            return be[0] + be[1] * a + be[2:] @ cvals

        def g(a: float, ap: float) -> Any:
            s = slope(a)
            if mreg == "linear":
                if not log_linear:
                    return base(a) + s * med_index(ap)
                return np.exp(base(a) + s * med_index(ap) + 0.5 * s * s * s2)
            pr = 1.0 / (1.0 + np.exp(-med_index(ap)))
            if not log_linear:
                return base(a) + s * pr
            return np.exp(base(a)) * (1.0 - pr + pr * np.exp(s))

        def g_m(a: float) -> Any:
            lin = base(a) + slope(a) * m_star
            return np.exp(lin) if log_linear else lin

        cde = g_m(a1) - g_m(a0)
        int_ref = g(a1, a0) - g(a0, a0) - cde
        int_med = g(a1, a1) - g(a0, a1) - g(a1, a0) + g(a0, a0)
        pie = g(a0, a1) - g(a0, a0)
        comp = [cde, int_ref, int_med, pie]
        if log_linear:
            ref = g(a0, a0)
            comp = [c / ref for c in comp]
        cde, int_ref, int_med, pie = comp
        te = cde + int_ref + int_med + pie
        return np.array(
            [
                te,
                cde,
                int_ref,
                int_med,
                pie,
                cde / te,
                int_ref / te,
                int_med / te,
                pie / te,
                (pie + int_med) / te,
                (int_ref + int_med) / te,
                (int_ref + int_med + pie) / te,
            ]
        )

    return f


def delta_method(
    f: Callable[[np.ndarray], np.ndarray], params: np.ndarray, V: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimates and delta-method SEs with a complex-step Jacobian."""
    est = f(params.astype(complex)).real
    h = 1e-30
    J = np.empty((est.size, params.size))
    for j in range(params.size):
        z = params.astype(complex)
        z[j] += 1j * h
        J[:, j] = f(z).imag / h
    var = np.einsum("ij,jk,ik->i", J, V, J)
    return est, np.sqrt(np.maximum(var, 0.0))


def as_dict(values: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(NAMES, values)}
