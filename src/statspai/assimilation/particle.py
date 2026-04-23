"""
Bootstrap particle filter for non-Gaussian assimilative causal inference.

Complements the exact Kalman closed form in
:mod:`statspai.assimilation.kalman` for cases where either the dynamics
or the observation model is non-linear / non-Gaussian — e.g.

* skewed causal-effect priors (log-normal, half-normal)
* heavy-tailed per-batch observation noise (Student-t)
* informative observation noise whose variance depends on θ

Uses the canonical sequential-importance-resampling (SIR) bootstrap
particle filter with **systematic resampling** and an effective-sample-
size trigger.  The shipped default dynamics = Gaussian random walk, so
``causal_kalman`` and ``assimilative_causal(backend='particle')`` agree
to within Monte-Carlo noise under the Kalman-compatible DGP — exactly
what you want for a sanity test.

References
----------
Gordon, Salmond, Smith (1993).  "Novel approach to nonlinear/non-
Gaussian Bayesian state estimation."  *IEE Proc. F.* [@gordon1993novel]

Douc, Cappé (2005).  "Comparison of resampling schemes for particle
filtering."  *ISPA*. [@douc2005comparison]
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from .kalman import AssimilationResult


__all__ = [
    "particle_filter",
    "assimilative_causal_particle",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling — returns indices into the particle array.

    Low-variance resampling that draws ``N`` uniformly-spaced pointers
    from a single uniform random variate.  Strictly dominates
    multinomial resampling in practice (Douc-Cappé 2005).
    """
    N = len(weights)
    if N == 0:
        return np.empty(0, dtype=int)
    positions = (rng.uniform() + np.arange(N)) / N
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # avoid floating-point tail gap
    idx = np.searchsorted(cumulative, positions)
    idx = np.clip(idx, 0, N - 1)
    return idx.astype(int)


def _normal_log_pdf(x: np.ndarray, mu: np.ndarray, sd: float) -> np.ndarray:
    """Log pdf of N(mu, sd^2) evaluated at x — vectorised."""
    sd = max(float(sd), 1e-12)
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)


# ---------------------------------------------------------------------------
# Core particle filter
# ---------------------------------------------------------------------------


def particle_filter(
    estimates: Sequence[float],
    standard_errors: Sequence[float],
    *,
    prior_sampler: Optional[Callable[[np.random.Generator, int], np.ndarray]] = None,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    transition_sampler: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    process_sd: float = 0.0,
    observation_log_pdf: Optional[Callable[[float, np.ndarray, float], np.ndarray]] = None,
    n_particles: int = 2000,
    ess_resample_threshold: float = 0.5,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> AssimilationResult:
    """SIR bootstrap particle filter over a stream of causal estimates.

    Parameters
    ----------
    estimates, standard_errors : sequences
        Batch-level ``(θ̂_t, σ_t)``.
    prior_sampler : callable, optional
        ``fn(rng, n) -> ndarray`` draws ``n`` particles from ``p(θ_0)``.
        Defaults to ``N(prior_mean, prior_var)``.
    prior_mean, prior_var : float
        Used only when ``prior_sampler=None``.
    transition_sampler : callable, optional
        ``fn(particles, rng) -> ndarray`` propagates the particles one
        step (the state-transition density).  Defaults to a Gaussian
        random walk with SD ``process_sd``.
    process_sd : float
        Used only when ``transition_sampler=None``.  ``0`` = static
        effect.
    observation_log_pdf : callable, optional
        ``fn(theta_hat_t, particles, sigma_t) -> ndarray`` returns
        log ``p(θ̂_t | θ_t)`` for every particle.  Defaults to the
        Gaussian obs model ``N(θ_t, σ_t^2)``.
    n_particles : int, default 2000
    ess_resample_threshold : float, default 0.5
        Resample whenever ``ESS / N`` falls below this value.
    alpha : float, default 0.05
    random_state : int, optional

    Returns
    -------
    AssimilationResult

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> T = 15
    >>> tau = 0.5
    >>> ests = [tau + rng.normal(0, 0.1) for _ in range(T)]
    >>> ses = [0.1] * T
    >>> # Default Gaussian obs + random-walk dynamics — should match the
    >>> # Kalman filter to within Monte-Carlo noise.
    >>> res_p = sp.assimilation.particle_filter(
    ...     ests, ses, n_particles=3000, random_state=0,
    ... )
    >>> abs(res_p.final_mean - 0.5) < 0.15
    True
    """
    theta = np.asarray(estimates, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    if theta.shape != se.shape:
        raise ValueError(
            "estimates/standard_errors shape mismatch: "
            f"{theta.shape} vs {se.shape}"
        )
    if np.any(se <= 0):
        raise ValueError("all standard_errors must be > 0")
    if n_particles < 100:
        raise ValueError("n_particles must be >= 100 for a stable filter")
    if not (0 < ess_resample_threshold <= 1):
        raise ValueError("ess_resample_threshold must be in (0, 1]")

    rng = np.random.default_rng(random_state)

    # --- Initial particles & weights --------------------------------------
    if prior_sampler is None:
        sd0 = np.sqrt(max(prior_var, 1e-12))
        particles = rng.normal(prior_mean, sd0, size=n_particles)
    else:
        particles = np.asarray(prior_sampler(rng, n_particles), dtype=float)
        if particles.shape != (n_particles,):
            raise ValueError(
                f"prior_sampler must return shape ({n_particles},); "
                f"got {particles.shape}"
            )
    log_w = np.full(n_particles, -np.log(n_particles))

    # --- Transition & observation defaults -------------------------------
    if transition_sampler is None:
        sd_w = float(process_sd)
        def _default_trans(p: np.ndarray, r: np.random.Generator) -> np.ndarray:
            if sd_w <= 0:
                return p
            return p + r.normal(0.0, sd_w, size=p.shape)
        trans_fn = _default_trans
    else:
        trans_fn = transition_sampler

    if observation_log_pdf is None:
        def _default_obs(y: float, p: np.ndarray, s: float) -> np.ndarray:
            return _normal_log_pdf(p, np.full_like(p, y), s)
        obs_fn = _default_obs
    else:
        obs_fn = observation_log_pdf

    T = len(theta)
    m = np.empty(T)
    sd_out = np.empty(T)
    ci = np.empty((T, 2))
    innov = np.empty(T)
    ess_arr = np.empty(T)

    base_var = float(np.mean(se ** 2))
    base_var = max(base_var, 1e-12)
    q_lo = alpha / 2
    q_hi = 1 - alpha / 2

    # --- Sequential loop --------------------------------------------------
    for t in range(T):
        # Forecast
        particles = trans_fn(particles, rng)
        # Analysis: incorporate θ̂_t
        log_like = obs_fn(float(theta[t]), particles, float(se[t]))
        log_w = log_w + log_like
        # Normalise in log-space for numerical stability
        log_w_max = np.max(log_w)
        w = np.exp(log_w - log_w_max)
        w_sum = w.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            raise RuntimeError(
                f"All particle weights collapsed to zero at t={t}. "
                "Increase n_particles or widen the prior / process noise."
            )
        w = w / w_sum
        log_w = np.log(np.maximum(w, 1e-300))

        # Posterior summaries
        mean_t = float(np.sum(w * particles))
        var_t = float(np.sum(w * (particles - mean_t) ** 2))
        m[t] = mean_t
        sd_out[t] = float(np.sqrt(max(var_t, 0.0)))
        innov[t] = float(theta[t] - mean_t)
        # Weighted quantiles via cumulative sum of sorted weights
        order = np.argsort(particles)
        sp_ = particles[order]
        cw = np.cumsum(w[order])
        cw[-1] = 1.0
        ci_lo = float(sp_[np.searchsorted(cw, q_lo)])
        ci_hi = float(sp_[np.searchsorted(cw, q_hi)])
        ci[t] = (ci_lo, ci_hi)
        ess_arr[t] = float(1.0 / np.sum(w ** 2))

        # Resample if ESS low
        if ess_arr[t] / n_particles < ess_resample_threshold:
            idx = _systematic_resample(w, rng)
            particles = particles[idx]
            log_w = np.full(n_particles, -np.log(n_particles))

    return AssimilationResult(
        posterior_mean=m,
        posterior_sd=sd_out,
        posterior_ci=ci,
        innovations=innov,
        ess=ess_arr,
        final_mean=float(m[-1]),
        final_sd=float(sd_out[-1]),
        final_ci=(float(ci[-1, 0]), float(ci[-1, 1])),
        method="causal_particle",
        diagnostics={
            "T": int(T),
            "n_particles": int(n_particles),
            "ess_resample_threshold": float(ess_resample_threshold),
            "average_obs_var": float(np.mean(se ** 2)),
            "alpha": float(alpha),
        },
    )


def assimilative_causal_particle(
    batches: Sequence[Any],
    estimator: Callable[[Any], Tuple[float, float]],
    **kwargs: Any,
) -> AssimilationResult:
    """Same as :func:`assimilative_causal` but forces the particle filter.

    Forwards every kwarg to :func:`particle_filter`.
    """
    estimates = []
    ses = []
    for b in batches:
        theta, se = estimator(b)
        estimates.append(float(theta))
        ses.append(float(se))
    return particle_filter(estimates, ses, **kwargs)
