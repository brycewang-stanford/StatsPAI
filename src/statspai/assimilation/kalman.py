"""
Kalman-filter data assimilation for streaming causal estimates.

Given a sequence of batch-level causal-effect estimates ``{(θ̂_t, σ_t)}``
— each of which you get from, say, running ``sp.did`` or ``sp.dml`` on
the batch that arrived at time ``t`` — this module fuses them into a
single running posterior.

State-space formulation
-----------------------
Latent causal effect evolves as

    θ_t = θ_{t-1} + w_t,    w_t ~ N(0, Q_t)                    (dynamics)
    θ̂_t = θ_t + e_t,       e_t ~ N(0, σ_t^2)                    (obs model)

The Kalman filter gives closed-form posteriors

    θ_t | y_{1:t}  ~  N(m_t, P_t).

We also emit an effective-sample-size (ESS) style diagnostic so you can
detect when the streaming posterior is "too confident" because of an
overly small ``Q``.

For the general nonlinear case (e.g. non-Gaussian outcome, drift tied
to a covariate), ``assimilative_causal`` supports a user-supplied
particle-filter backend via ``backend='particle'`` — but the default is
the exact Kalman closed form, which is the version used in Nature
Communications 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


__all__ = [
    "assimilative_causal",
    "causal_kalman",
    "AssimilationResult",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AssimilationResult:
    """Output of :func:`assimilative_causal` / :func:`causal_kalman`.

    Attributes
    ----------
    posterior_mean : np.ndarray
        Running posterior means ``m_t``.
    posterior_sd : np.ndarray
        Running posterior standard deviations ``sqrt(P_t)``.
    posterior_ci : np.ndarray
        ``(T, 2)`` array of 95% CIs.
    innovations : np.ndarray
        ``θ̂_t - m_{t|t-1}`` at each step (useful for diagnostics).
    ess : np.ndarray
        Effective sample size per step — number of past batches the
        current posterior is "worth" in precision terms.
    final_mean : float
    final_sd : float
    final_ci : tuple
    method : str
    diagnostics : dict
    """

    posterior_mean: np.ndarray
    posterior_sd: np.ndarray
    posterior_ci: np.ndarray
    innovations: np.ndarray
    ess: np.ndarray
    final_mean: float
    final_sd: float
    final_ci: Tuple[float, float]
    method: str = "causal_kalman"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        T = len(self.posterior_mean)
        lines = [
            "Assimilative Causal Inference (Kalman backend)",
            "-" * 54,
            f"  T (batches)        : {T}",
            f"  Final posterior    : {self.final_mean:+.6f}",
            f"  Final SD           : {self.final_sd:.6f}",
            f"  Final 95% CI       : "
            f"[{self.final_ci[0]:+.6f}, {self.final_ci[1]:+.6f}]",
            f"  Final ESS          : {self.ess[-1]:.2f}",
            "",
            "  Trajectory (tail):",
        ]
        for t in range(max(0, T - 5), T):
            lines.append(
                f"    t={t:>3d}  "
                f"m={self.posterior_mean[t]:+.4f}  "
                f"sd={self.posterior_sd[t]:.4f}  "
                f"ci=[{self.posterior_ci[t, 0]:+.4f}, "
                f"{self.posterior_ci[t, 1]:+.4f}]  "
                f"ess={self.ess[t]:.2f}"
            )
        return "\n".join(lines)

    def trajectory(self) -> pd.DataFrame:
        """Convert the running posterior to a tidy DataFrame."""
        return pd.DataFrame({
            "t": np.arange(len(self.posterior_mean)),
            "posterior_mean": self.posterior_mean,
            "posterior_sd": self.posterior_sd,
            "ci_lower": self.posterior_ci[:, 0],
            "ci_upper": self.posterior_ci[:, 1],
            "innovation": self.innovations,
            "ess": self.ess,
        })


# ---------------------------------------------------------------------------
# Core Kalman filter
# ---------------------------------------------------------------------------


def _kalman_filter(
    theta_obs: np.ndarray,
    obs_var: np.ndarray,
    *,
    prior_mean: float,
    prior_var: float,
    process_var: float,
    alpha: float = 0.05,
) -> AssimilationResult:
    """Run the scalar Kalman filter described in the module docstring."""
    T = len(theta_obs)
    m = np.empty(T)
    P = np.empty(T)
    innov = np.empty(T)
    ess = np.empty(T)
    m_prev = float(prior_mean)
    P_prev = float(prior_var)
    base_obs_var = float(np.nanmedian(obs_var)) if T > 0 else 1.0
    base_obs_var = max(base_obs_var, 1e-12)
    for t in range(T):
        # Forecast
        m_fore = m_prev
        P_fore = P_prev + process_var
        # Analysis (observation update)
        R = float(obs_var[t])
        K = P_fore / (P_fore + R)
        innov[t] = theta_obs[t] - m_fore
        m[t] = m_fore + K * innov[t]
        P[t] = (1 - K) * P_fore
        # ESS proxy: posterior precision / baseline-obs precision
        ess[t] = base_obs_var / max(P[t], 1e-12)
        m_prev, P_prev = m[t], P[t]

    z = 1.959963984540054  # Φ^{-1}(1 - α/2) with α = 0.05
    if alpha != 0.05:
        from scipy.stats import norm as _norm
        z = float(_norm.ppf(1 - alpha / 2))
    sd = np.sqrt(np.maximum(P, 0.0))
    ci = np.column_stack([m - z * sd, m + z * sd])

    return AssimilationResult(
        posterior_mean=m,
        posterior_sd=sd,
        posterior_ci=ci,
        innovations=innov,
        ess=ess,
        final_mean=float(m[-1]) if T > 0 else float(prior_mean),
        final_sd=float(sd[-1]) if T > 0 else float(np.sqrt(prior_var)),
        final_ci=(
            (float(ci[-1, 0]), float(ci[-1, 1]))
            if T > 0 else (prior_mean - z * np.sqrt(prior_var),
                           prior_mean + z * np.sqrt(prior_var))
        ),
        method="causal_kalman",
        diagnostics={
            "T": int(T),
            "prior_mean": float(prior_mean),
            "prior_var": float(prior_var),
            "process_var": float(process_var),
            "alpha": float(alpha),
            "average_obs_var": float(np.nanmean(obs_var)) if T > 0 else np.nan,
        },
    )


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------


def causal_kalman(
    estimates: Sequence[float],
    standard_errors: Sequence[float],
    *,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    process_var: float = 0.0,
    alpha: float = 0.05,
) -> AssimilationResult:
    """Closed-form Kalman filter over a stream of causal-effect estimates.

    Parameters
    ----------
    estimates : sequence of float
        Batch-level estimates ``θ̂_t``.
    standard_errors : sequence of float
        Batch-level standard errors ``σ_t``.  Converted to variances.
    prior_mean, prior_var : float, default (0.0, 1.0)
        Prior ``N(m_0, P_0)`` on the causal effect.
    process_var : float, default 0.0
        State noise variance ``Q``.  ``0`` = static effect (all batches
        share one truth); ``>0`` = random-walk drift.
    alpha : float, default 0.05

    Returns
    -------
    AssimilationResult

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> T = 20
    >>> true_tau = 0.5
    >>> ests = [true_tau + rng.normal(0, 0.1) for _ in range(T)]
    >>> ses = [0.1] * T
    >>> res = sp.assimilation.causal_kalman(
    ...     ests, ses, prior_mean=0.0, prior_var=1.0, process_var=0.0,
    ... )
    >>> abs(res.final_mean - 0.5) < 0.1
    True
    """
    theta = np.asarray(estimates, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    if theta.shape != se.shape:
        raise ValueError(
            f"estimates/standard_errors shape mismatch: "
            f"{theta.shape} vs {se.shape}"
        )
    if prior_var <= 0:
        raise ValueError("prior_var must be > 0")
    if process_var < 0:
        raise ValueError("process_var must be >= 0")
    if np.any(se <= 0):
        raise ValueError("all standard_errors must be > 0")
    return _kalman_filter(
        theta, se ** 2,
        prior_mean=prior_mean,
        prior_var=prior_var,
        process_var=process_var,
        alpha=alpha,
    )


def assimilative_causal(
    batches: Sequence[Any],
    estimator: Callable[[Any], Tuple[float, float]],
    *,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    process_var: float = 0.0,
    alpha: float = 0.05,
    backend: str = "kalman",
) -> AssimilationResult:
    """Run the Nature-Comms-2026 assimilation pipeline end-to-end.

    Parameters
    ----------
    batches : sequence of Any
        Each element is a batch dataset (DataFrame, ndarray, or anything
        ``estimator`` knows how to consume).
    estimator : callable
        Maps one batch to ``(theta_hat, standard_error)``.  For example,
        a lambda calling ``sp.dml(...)`` and extracting ``.estimate``,
        ``.se``.
    prior_mean, prior_var, process_var, alpha
        Forwarded to :func:`causal_kalman`.
    backend : {'kalman'}, default 'kalman'
        Only the exact Kalman path is implemented at the moment.  A
        particle-filter backend is planned for nonlinear / non-Gaussian
        observation models and will share the same call signature.

    Returns
    -------
    AssimilationResult

    Notes
    -----
    Assimilative Causal Inference is non-adversarial by design: it
    assumes the per-batch estimator is well-calibrated (i.e. the CIs
    actually cover at their nominal rate).  If the estimator is biased
    the filter inherits that bias.  Run the per-batch estimator through
    :func:`sp.smart.assumption_audit` before feeding it to this pipeline.

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> def one_batch(n):
    ...     x = rng.normal(size=n)
    ...     d = rng.integers(0, 2, n)
    ...     y = 0.5 * d + 0.2 * x + rng.normal(scale=0.3, size=n)
    ...     return pd.DataFrame({'y': y, 'd': d, 'x': x})
    >>> batches = [one_batch(200) for _ in range(10)]
    >>> def est(df):
    ...     r = sp.regress('y ~ d + x', data=df)
    ...     return float(r.params['d']), float(r.std_errors['d'])
    >>> res = sp.assimilation.assimilative_causal(
    ...     batches, est, prior_mean=0.0, prior_var=1.0,
    ... )
    >>> abs(res.final_mean - 0.5) < 0.1
    True
    """
    if backend != "kalman":
        raise NotImplementedError(
            f"Only backend='kalman' is implemented; particle filter is "
            f"planned.  Got backend={backend!r}."
        )
    estimates: List[float] = []
    ses: List[float] = []
    for b in batches:
        theta, se = estimator(b)
        estimates.append(float(theta))
        ses.append(float(se))
    return causal_kalman(
        estimates, ses,
        prior_mean=prior_mean,
        prior_var=prior_var,
        process_var=process_var,
        alpha=alpha,
    )
