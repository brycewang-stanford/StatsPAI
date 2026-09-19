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

    \\hat\\theta(l) = l_{pre}' \\hat\\beta_{pre} + l_{post}' \\hat\\beta_{post}

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
from scipy import optimize, special, stats

from ..exceptions import ConvergenceFailure, MethodIncompatibility

__all__ = [
    "FLCIResult",
    "event_study_moments",
    "flci_delta_sd",
    "folded_normal_quantile",
]


def event_study_moments(result) -> Optional[tuple]:
    """Extract ``(betahat, sigma, event_times)`` for the event study.

    The FLCI needs the *joint* covariance of the event-study coefficients —
    that is exactly what the old worst-case-bias path threw away. A
    Callaway-Sant'Anna fit carries per-unit influence functions for each
    ATT(g, t) cell, so the event-study covariance is
    ``W (Psi' Psi / n^2) W'`` where ``W`` is the event-study aggregation
    weight matrix.

    Returns ``None`` when the covariance cannot be recovered (no influence
    functions attached), so callers can fall back rather than fabricate one.
    """
    inf_matrix = getattr(result, "_influence_funcs", None)
    detail = getattr(result, "detail", None)
    if inf_matrix is None or detail is None:
        return None
    required = {"group", "relative_time", "att"}
    if not required.issubset(set(detail.columns)):
        return None

    model_info = getattr(result, "model_info", None) or {}
    n_units = model_info.get("n_units", getattr(result, "n_obs", None))
    if not n_units:
        return None

    try:
        from .aggte import _weights_dynamic
    except ImportError:  # pragma: no cover - internal layout guard
        return None

    finite = np.isfinite(np.asarray(detail["att"], dtype=float))
    if not finite.all():
        detail = detail.loc[finite].reset_index(drop=True)
        inf_matrix = inf_matrix[:, np.asarray(finite)]

    labels, weights = _weights_dynamic(
        detail, model_info.get("cohort_sizes"), -np.inf, np.inf
    )
    if weights.shape[0] < 2:
        return None

    betahat = weights @ np.asarray(detail["att"], dtype=float)
    psi = np.asarray(inf_matrix, dtype=float) @ weights.T
    sigma = (psi.T @ psi) / float(n_units) ** 2
    return betahat, sigma, np.asarray(labels, dtype=int)


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

    # special.ndtr is the standard-normal CDF stats.norm.cdf evaluates, without
    # the distribution-object overhead (this runs inside every FLCI solve).
    def _cdf_gap(q: float) -> float:
        return special.ndtr((q - mu) / sd) - special.ndtr((-q - mu) / sd) - p

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
    return _FLCIProgram(betahat, sigma, n_pre, n_post, l_post, n_grid).ci(m_bar, alpha)


class _FLCIProgram:
    """The ``Delta^SD`` FLCI program for one ``(betahat, sigma, l_post)``.

    The worst-case bias ``b(h)`` of the best affine estimator whose standard
    deviation is at most ``h`` does not depend on ``M``; the half-length for a
    given ``M`` is ``min_h q_{1-alpha}(|N(M b(h) / h, 1)|) h``. Solving ``b(h)``
    is the expensive step (one SLSQP program), so it is memoised: a sweep over
    many ``M`` -- which is what :func:`breakdown_m_sd` does -- reuses every
    solve.

    The minimisation over ``h`` is a grid scan (``n_grid`` points between the
    minimum attainable SD and the SD of the minimum-bias estimator) followed
    by a bounded Brent refinement inside the bracket around the best grid
    point. Through 1.28.0 the best grid point itself was reported, which left
    a discretisation error of order ``(h_max - h_min) / n_grid`` in ``h``
    (7e-4 relative on a CI bound of the event study in
    ``tests/reference_parity/test_did_synth_R_parity.py`` at ``M = 0.01``);
    the refinement removes it.
    """

    def __init__(
        self,
        betahat: np.ndarray,
        sigma: np.ndarray,
        n_pre: int,
        n_post: int,
        l_post: np.ndarray,
        n_grid: int = 100,
    ) -> None:
        self.betahat = betahat
        self.n_pre = n_pre
        self.l_post = l_post
        a_quad, a_lin, a_const = _variance_matrices(sigma, n_pre, l_post)
        steps = np.arange(1, n_post + 1, dtype=float)
        sum_target = float(steps @ l_post)
        self.const = (
            sum(
                abs(np.arange(1, s + 1) @ l_post[n_post - s :])
                for s in range(1, n_post + 1)
            )
            - sum_target
        )
        tril = np.tril(np.ones((n_pre, n_pre)))

        def variance(x: np.ndarray) -> float:
            return float(x @ a_quad @ x + a_lin @ x + a_const)

        a_sym = a_quad + a_quad.T
        eye = np.eye(n_pre)
        jac_minus = np.hstack([eye, -tril])
        jac_plus = np.hstack([eye, tril])
        jac_sum = np.concatenate([np.zeros(n_pre), np.ones(n_pre)])
        self._variance = variance
        self._variance_jac = lambda x: a_sym @ x + a_lin
        # Linear objective and constraints: exact Jacobians, so SLSQP does not
        # finite-difference them (and is not limited by that step's error).
        self._obj_jac = np.concatenate([np.ones(n_pre), np.zeros(n_pre)])
        self._base_cons = [
            {
                "type": "ineq",
                "fun": lambda x: x[:n_pre] - tril @ x[n_pre:],
                "jac": lambda x: jac_minus,
            },
            {
                "type": "ineq",
                "fun": lambda x: x[:n_pre] + tril @ x[n_pre:],
                "jac": lambda x: jac_plus,
            },
            {
                "type": "eq",
                "fun": lambda x: x[n_pre:].sum() - sum_target,
                "jac": lambda x: jac_sum,
            },
        ]
        self._x0 = np.concatenate([np.ones(n_pre), np.full(n_pre, sum_target / n_pre)])

        # h ranges from the minimum attainable SD up to the SD of the
        # minimum-bias estimator.
        var_fit = optimize.minimize(
            variance,
            self._x0,
            jac=self._variance_jac,
            constraints=self._base_cons,
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not var_fit.success:
            raise ConvergenceFailure(
                "FLCI: could not find the minimum-variance affine estimator.",
                recovery_hint="Check sigma for near-singularity, or use "
                "backend='r'.",
                diagnostics={"message": var_fit.message},
            )
        self.h_min = float(np.sqrt(max(var_fit.fun, 0.0)))
        w_min_bias = np.concatenate([np.zeros(n_pre - 1), [sum_target]])
        self._x0_min_bias = np.concatenate([np.abs(tril @ w_min_bias), w_min_bias])
        h_max = float(
            np.sqrt(max(variance(np.concatenate([np.zeros(n_pre), w_min_bias])), 0.0))
        )
        if not np.isfinite(h_max) or h_max <= self.h_min:
            h_max = self.h_min * 1.5 + 1e-8
        self.h_max = h_max
        self.grid = np.linspace(self.h_min, self.h_max, int(n_grid))
        self._cache: dict = {}

    def bias(self, h: float) -> tuple:
        """``(b(h), x(h))``: worst-case bias per unit of ``M`` and its solution."""
        key = float(h)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        cons = self._base_cons + [
            {
                "type": "ineq",
                "fun": lambda x, h=key: h**2 - self._variance(x),
                "jac": lambda x: -self._variance_jac(x),
            }
        ]
        # Two starting points. SLSQP stops after one iteration at the
        # default start (u = 1, w uniform) whenever the variance constraint
        # is slack there, reporting "success" at a point that is not the
        # optimum of this convex program: through 1.28.0 that froze b(h) at
        # the start's objective for every h above some threshold (b = 4.0
        # where HonestDiD's ECOS solve gives 3.60 ... 3.00 on the e = 1 event
        # study of tests/reference_parity/test_did_synth_R_parity.py), which
        # cut the h search short and widened the FLCI. The second start is the
        # minimum-bias estimator, feasible for every h >= h_max; the smaller
        # objective of the two converged solves is kept.
        best = None
        for x0 in (self._x0, self._x0_min_bias):
            fit = optimize.minimize(
                lambda x: self.const + x[: self.n_pre].sum(),
                x0,
                jac=lambda x: self._obj_jac,
                constraints=cons,
                method="SLSQP",
                options={"maxiter": 500, "ftol": 1e-12},
            )
            if fit.success and (best is None or fit.fun < best.fun):
                best = fit
        out = (float(best.fun), best.x) if best is not None else (np.inf, None)
        self._cache[key] = out
        return out

    def half_length(self, h: float, m_bar: float, alpha: float) -> float:
        b, x = self.bias(h)
        if not np.isfinite(b) or x is None:
            return np.inf
        return folded_normal_quantile(1 - alpha, m_bar * b / h) * h

    def ci(self, m_bar: float, alpha: float, refine: bool = True) -> FLCIResult:
        halves = np.array([self.half_length(h, m_bar, alpha) for h in self.grid])
        if not np.isfinite(halves).any():  # pragma: no cover - pathological sigma
            raise ConvergenceFailure(
                "FLCI: the worst-case-bias program did not solve at any h.",
                recovery_hint="Use backend='r', or widen the event-study window.",
                diagnostics={"h_min": self.h_min, "h_max": self.h_max},
            )
        i = int(np.argmin(halves))
        h_star = float(self.grid[i])
        lo = float(self.grid[max(i - 1, 0)])
        hi = float(self.grid[min(i + 1, len(self.grid) - 1)])
        if refine and hi > lo:
            ref = optimize.minimize_scalar(
                lambda h: self.half_length(h, m_bar, alpha),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1e-12 * hi},
            )
            if np.isfinite(ref.fun) and ref.fun <= halves[i]:
                h_star = float(ref.x)
        bias_star, x_star = self.bias(h_star)
        half_length = self.half_length(h_star, m_bar, alpha)

        n_pre = self.n_pre
        w = x_star[n_pre:]
        w_to_l = np.eye(n_pre)
        for col in range(n_pre - 1):
            w_to_l[col + 1, col] = -1.0
        l_pre = w_to_l @ w

        # The affine estimator is the full weight vector dotted with betahat.
        # `_w_to_l` already carries the sign that turns w into the pre-period
        # extrapolation weights, so this is an addition, not a subtraction --
        # matching HonestDiD's `optimalVec = c(optimal.l, l_vec)`.
        estimate = float(
            l_pre @ self.betahat[:n_pre] + self.l_post @ self.betahat[n_pre:]
        )
        return FLCIResult(
            estimate=estimate,
            half_length=float(half_length),
            ci_lower=estimate - float(half_length),
            ci_upper=estimate + float(half_length),
            pre_period_weights=l_pre,
            h=float(h_star),
            worst_case_bias=float(m_bar * bias_star),
        )


def breakdown_m_sd(
    betahat: np.ndarray,
    sigma: np.ndarray,
    n_pre: int,
    n_post: int,
    l_post: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_grid: int = 100,
    xtol: float = 1e-10,
) -> float:
    """Breakdown value ``M* = sup{M : 0 not in FLCI(M)}`` under ``Delta^SD(M)``.

    The Rambachan-Roth breakdown value is defined through the confidence set
    itself, so it is found by root-finding on the FLCI bound that faces zero:
    ``f(M) = max(lower(M), -upper(M))`` is positive exactly when the interval
    excludes zero. Returns ``0.0`` when ``FLCI(0)`` already covers zero.
    """
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    l_post = np.eye(n_post)[0] if l_post is None else np.asarray(l_post, dtype=float)
    prog = _FLCIProgram(betahat, sigma, n_pre, n_post, l_post, n_grid)

    def f(m: float, refine: bool = True) -> float:
        r = prog.ci(m, alpha, refine=refine)
        return max(r.ci_lower, -r.ci_upper)

    if f(0.0) <= 0.0:
        return 0.0
    hi = max(prog.h_min, 1e-8)
    for _ in range(200):
        if f(hi, refine=False) <= 0.0:
            break
        hi *= 2.0
    else:  # pragma: no cover - would need an unbounded worst-case bias
        raise ConvergenceFailure(
            "breakdown M: the FLCI never covers zero as M grows.",
            recovery_hint="Check the event-study inputs.",
            diagnostics={"last_M": hi},
        )
    # Stage 1: locate the crossing on the grid-scanned FLCI, whose bias
    # solves are all cached after the first evaluation (cheap).
    m0 = float(optimize.brentq(lambda m: f(m, False), 0.0, hi, xtol=1e-9 * hi))
    # Stage 2: polish on the refined FLCI inside a bracket around m0 that is
    # widened until it straddles the refined crossing.
    width = max(1e-3 * m0, 1e-10)
    lo_b, hi_b = max(m0 - width, 0.0), m0 + width
    for _ in range(60):
        f_lo, f_hi = f(lo_b), f(hi_b)
        if f_lo > 0.0 >= f_hi:
            break
        if f_lo <= 0.0:
            lo_b = max(lo_b - 2.0 * width, 0.0)
        if f_hi > 0.0:
            hi_b = hi_b + 2.0 * width
        width *= 2.0
    else:  # pragma: no cover - the refined and scanned crossings coincide
        raise ConvergenceFailure(
            "breakdown M: could not bracket the refined FLCI crossing.",
            recovery_hint="Check the event-study inputs.",
            diagnostics={"grid_root": m0},
        )
    return float(optimize.brentq(f, lo_b, hi_b, xtol=xtol * max(m0, 1e-12), rtol=1e-14))
