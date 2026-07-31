"""Fixed-length confidence intervals for Rambachan & Roth (2023) Delta^SD.

The native ``sp.honest_did`` path used to return a *worst-case-bias* interval,
``theta_hat ± bias_bound ± z·SE`` — the worst-case bias bolted onto an ordinary
Wald interval.  That is not the Rambachan-Roth confidence set: it ignores the
pre-period covariance structure, and on real data it comes out **narrower**
than the reference at every M, which overstates robustness.

This module implements the actual FLCI.  Following Rambachan-Roth §3 (and the
``HonestDiD`` reference implementation), the interval is built from an affine
estimator

.. math::

    \\hat\\theta(l) = l_{post}' \\hat\\beta_{post} - l_{pre}' \\hat\\beta_{pre}

whose half-length is ``q_{1-alpha}(|N(bias/h, 1)|) * h``, where ``h`` bounds the
estimator's standard deviation and ``bias`` is the worst-case bias over the
smoothness set ``Delta^SD(M)``.  For a fixed ``h`` the worst-case bias is a
convex program; the reported interval minimises the resulting half-length over
``h``.

Formulation
-----------
With ``K`` pre-periods and ``S`` post-periods, the decision variable is
``x = [u; w]`` of length ``2K``:

* objective  ``min  c + sum(u)``  with
  ``c = sum_s |<1..s, l_post[S-s:]>| - <1..S, l_post>``
* ``u >= |cumsum(w)|`` componentwise (the absolute-value constraints)
* ``sum(w) == <1..S, l_post>``
* ``x' A_q x + A_l' x + A_c <= h^2`` (the estimator's variance)

which is a convex QCQP — linear objective, linear constraints, one convex
quadratic — so it is solved with SLSQP rather than by taking on a
convex-optimisation dependency.

Accuracy note
-------------
``HonestDiD`` computes the folded-normal quantile by simulation
(``.qfoldednormal``: 10^6 draws at a fixed seed), which carries roughly 2e-3 of
Monte Carlo error — at ``mu = 0`` it returns 1.96224 where the exact value is
``z_{0.975} = 1.95996``.  :func:`_folded_normal_quantile` here inverts the
folded-normal CDF exactly, so this implementation is *more* accurate than the
reference and agrees with it to about that same 2e-3.

References
----------
Rambachan, A. and Roth, J. (2023). "A More Credible Approach to Parallel
Trends." *Review of Economic Studies*, 90(5), 2555-2591. [@rambachan2023more]
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
from scipy import optimize, stats

from ..exceptions import ConvergenceFailure, MethodIncompatibility

__all__ = ["FLCIResult", "flci_delta_sd", "folded_normal_quantile"]


class FLCIResult(NamedTuple):
    """Optimal fixed-length CI and the affine estimator that attains it."""

    estimate: float
    half_length: float
    ci_lower: float
    ci_upper: float
    #: Weights on the pre-period coefficients (the extrapolation rule).
    pre_period_weights: np.ndarray
    #: The ``h`` (standard-deviation bound) that minimised the half-length.
    h: float
    worst_case_bias: float


def folded_normal_quantile(p: float, mu: float, sd: float = 1.0) -> float:
    """Exact ``p``-quantile of ``|N(mu, sd^2)|``.

    ``HonestDiD`` simulates this; inverting the CDF directly avoids ~2e-3 of
    Monte Carlo error. Sanity check: ``mu = 0`` returns ``z_{(1+p)/2}``.
    """
    if not 0.0 < p < 1.0:
        raise MethodIncompatibility(
            f"p must be in (0, 1); got {p}.",
            recovery_hint="Pass a probability such as 0.95.",
            diagnostics={"p": p},
        )
    mu = float(abs(mu))

    def _cdf_gap(q: float) -> float:
        return stats.norm.cdf((q - mu) / sd) - stats.norm.cdf((-q - mu) / sd) - p

    hi = mu + sd * 20.0
    return float(optimize.brentq(_cdf_gap, 0.0, hi, xtol=1e-12, rtol=1e-14))


def _variance_matrices(sigma: np.ndarray, n_pre: int, l_post: np.ndarray) -> tuple:
    """(A_quadratic, A_linear, A_constant) giving Var(theta_hat) as a
    quadratic form in ``x = [u; w]``."""
    w_to_l = np.eye(n_pre)
    for col in range(n_pre - 1):
        w_to_l[col + 1, col] = -1.0
    # u does not enter the variance; only w does.
    stack = np.hstack([np.zeros((n_pre, n_pre)), w_to_l])

    sigma_pre = sigma[:n_pre, :n_pre]
    sigma_pre_post = sigma[:n_pre, n_pre:]
    sigma_post = float(l_post @ sigma[n_pre:, n_pre:] @ l_post)

    a_quad = stack.T @ sigma_pre @ stack
    a_lin = 2.0 * stack.T @ sigma_pre_post @ l_post
    return a_quad, a_lin, sigma_post


def flci_delta_sd(
    betahat: np.ndarray,
    sigma: np.ndarray,
    n_pre: int,
    n_post: int,
    m_bar: float,
    l_post: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_grid: int = 100,
) -> FLCIResult:
    """Optimal FLCI under the smoothness restriction ``Delta^SD(M)``.

    Parameters
    ----------
    betahat, sigma
        Event-study coefficients (pre-periods first, then post) and their
        covariance matrix.
    n_pre, n_post
        Counts of pre- and post-treatment coefficients in ``betahat``.
    m_bar
        The smoothness bound ``M``: the second difference of the underlying
        trend violation is bounded by ``M`` per period.
    l_post
        Weights picking out the post-treatment target. Defaults to the first
        post-treatment period.
    alpha
        One minus the nominal coverage.
    """
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    if betahat.size != n_pre + n_post:
        raise MethodIncompatibility(
            f"betahat has {betahat.size} entries but n_pre + n_post = "
            f"{n_pre + n_post}.",
            recovery_hint="Check the event-study window.",
            diagnostics={"n_beta": int(betahat.size)},
        )
    if sigma.shape != (betahat.size, betahat.size):
        raise MethodIncompatibility(
            f"sigma must be {betahat.size}x{betahat.size}; got {sigma.shape}.",
            recovery_hint="Pass the full event-study covariance matrix.",
            diagnostics={"shape": list(sigma.shape)},
        )
    if n_pre < 1:
        raise MethodIncompatibility(
            "the FLCI needs at least one pre-treatment period.",
            recovery_hint="Use a design with pre-periods, or "
            "method='relative_magnitude'.",
            diagnostics={"n_pre": n_pre},
        )

    l_post = np.eye(n_post)[0] if l_post is None else np.asarray(l_post, dtype=float)

    a_quad, a_lin, a_const = _variance_matrices(sigma, n_pre, l_post)
    steps = np.arange(1, n_post + 1, dtype=float)
    sum_target = float(steps @ l_post)
    const = (
        sum(
            abs(np.arange(1, s + 1) @ l_post[n_post - s :])
            for s in range(1, n_post + 1)
        )
        - sum_target
    )
    tril = np.tril(np.ones((n_pre, n_pre)))

    def variance(x: np.ndarray) -> float:
        return float(x @ a_quad @ x + a_lin @ x + a_const)

    base_cons = [
        {"type": "ineq", "fun": lambda x: x[:n_pre] - tril @ x[n_pre:]},
        {"type": "ineq", "fun": lambda x: x[:n_pre] + tril @ x[n_pre:]},
        {"type": "eq", "fun": lambda x: x[n_pre:].sum() - sum_target},
    ]
    x0 = np.concatenate([np.ones(n_pre), np.full(n_pre, sum_target / n_pre)])

    # h ranges from the minimum attainable SD up to the SD of the
    # minimum-bias estimator.
    var_fit = optimize.minimize(
        variance,
        x0,
        constraints=base_cons,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not var_fit.success:
        raise ConvergenceFailure(
            "FLCI: could not find the minimum-variance affine estimator.",
            recovery_hint="Check sigma for near-singularity, or use " "backend='r'.",
            diagnostics={"message": var_fit.message},
        )
    h_min = float(np.sqrt(max(var_fit.fun, 0.0)))
    w_min_bias = np.concatenate([np.zeros(n_pre - 1), [sum_target]])
    h_max = float(
        np.sqrt(max(variance(np.concatenate([np.zeros(n_pre), w_min_bias])), 0.0))
    )
    if not np.isfinite(h_max) or h_max <= h_min:
        h_max = h_min * 1.5 + 1e-8

    def worst_case_bias(h: float):
        cons = base_cons + [
            {"type": "ineq", "fun": lambda x, h=h: h**2 - variance(x)}
        ]
        fit = optimize.minimize(
            lambda x: const + x[:n_pre].sum(),
            x0,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not fit.success:
            return np.inf, None
        return float(fit.fun), fit.x

    best = None
    for h in np.linspace(h_min, h_max, n_grid):
        bias, x = worst_case_bias(h)
        if not np.isfinite(bias) or x is None:
            continue
        half = folded_normal_quantile(1 - alpha, m_bar * bias / h) * h
        if best is None or half < best[0]:
            best = (half, h, bias, x)

    if best is None:  # pragma: no cover - only on a pathological sigma
        raise ConvergenceFailure(
            "FLCI: the worst-case-bias program did not solve at any h.",
            recovery_hint="Use backend='r', or widen the event-study window.",
            diagnostics={"h_min": h_min, "h_max": h_max},
        )

    half_length, h_star, bias_star, x_star = best
    w = x_star[n_pre:]
    w_to_l = np.eye(n_pre)
    for col in range(n_pre - 1):
        w_to_l[col + 1, col] = -1.0
    l_pre = w_to_l @ w

    estimate = float(l_post @ betahat[n_pre:] - l_pre @ betahat[:n_pre])
    return FLCIResult(
        estimate=estimate,
        half_length=float(half_length),
        ci_lower=estimate - float(half_length),
        ci_upper=estimate + float(half_length),
        pre_period_weights=l_pre,
        h=float(h_star),
        worst_case_bias=float(m_bar * bias_star),
    )
