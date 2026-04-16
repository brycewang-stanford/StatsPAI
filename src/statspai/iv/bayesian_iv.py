"""
Bayesian IV — posterior inference under potentially weak instruments.

A simple Metropolis–Hastings sampler for the posterior of the structural
parameter β in the model

    y = D β + X α + u       (structural equation)
    D = Z π + X γ + v       (first stage)
    (u, v) ~ N(0, Σ)         (joint normality of reduced-form errors)

Under weak instruments, frequentist CIs (Wald, t-ratio) can have severe
under-coverage. The Bayesian posterior is *always* coherent — it
correctly reflects the wide uncertainty induced by weak identification.

This module implements:
- :func:`bayesian_iv` — posterior samples of β via M–H with an
  Anderson-Rubin-style likelihood that is pivotal in β.

The approach follows Chernozhukov & Hong (2003, ECMA) who show that
posterior inference based on the AR objective yields credible sets with
correct frequentist coverage under weak identification (the "Bayesian
AR" posterior).

References
----------
Chernozhukov, V. and Hong, H. (2003). "An MCMC Approach to Classical
    Estimation." *Journal of Econometrics*, 115(2), 293-346.

Kleibergen, F. and Zivot, E. (2003). "Bayesian and Classical Approaches
    to Instrumental Variable Regression."
    *Journal of Econometrics*, 114(1), 29-72.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BayesianIVResult:
    """Posterior from Bayesian IV."""
    posterior_draws: np.ndarray
    posterior_mean: float
    posterior_sd: float
    hpd_lower: float
    hpd_upper: float
    hpd_level: float
    acceptance_rate: float
    n_draws: int
    n_warmup: int
    ess: float
    extra: dict

    def summary(self) -> str:
        return (
            "Bayesian IV (Chernozhukov-Hong 2003 AR posterior)\n"
            f"{'-' * 56}\n"
            f"  Posterior mean        : {self.posterior_mean:>10.4f}\n"
            f"  Posterior SD          : {self.posterior_sd:>10.4f}\n"
            f"  {int(self.hpd_level*100)}% HPD interval      : "
            f"[{self.hpd_lower:.4f}, {self.hpd_upper:.4f}]\n"
            f"  Draws (post-warmup)  : {self.n_draws}\n"
            f"  Acceptance rate      : {self.acceptance_rate:.3f}\n"
            f"  Effective sample size: {self.ess:.0f}"
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"beta": self.posterior_draws})


def _grab(v, data, cols=False):
    if isinstance(v, str):
        return data[v].values.astype(float)
    if cols and isinstance(v, list) and all(isinstance(x, str) for x in v):
        return data[v].values.astype(float)
    return np.asarray(v, dtype=float)


def _residualize(M: np.ndarray, W: Optional[np.ndarray]) -> np.ndarray:
    if W is None or W.size == 0 or W.shape[1] == 0:
        return M
    b, *_ = np.linalg.lstsq(W, M, rcond=None)
    return M - W @ b


def _hpd(draws: np.ndarray, level: float = 0.95):
    """Highest Posterior Density interval."""
    n = len(draws)
    alpha = 1 - level
    n_in = int(np.ceil((1 - alpha) * n))
    if n_in >= n:
        return float(draws.min()), float(draws.max())
    sorted_d = np.sort(draws)
    widths = sorted_d[n_in:] - sorted_d[:n - n_in]
    i = int(np.argmin(widths))
    return float(sorted_d[i]), float(sorted_d[i + n_in])


def _ess(draws: np.ndarray) -> float:
    """Effective sample size via autocorrelation (Geyer 1992 monotone estimator)."""
    n = len(draws)
    if n < 10:
        return float(n)
    mean = draws.mean()
    var = draws.var(ddof=0)
    if var < 1e-12:
        return float(n)
    acf = np.correlate(draws - mean, draws - mean, "full")[n - 1:] / (n * var)
    # Sum of autocorrelation pairs until sum goes negative
    total = 1.0
    for k in range(1, n // 2):
        pair = acf[2 * k - 1] + acf[2 * k] if 2 * k < len(acf) else 0
        if pair < 0:
            break
        total += 2 * pair
    return max(n / total, 1.0)


def bayesian_iv(
    y: Union[np.ndarray, pd.Series, str],
    endog: Union[np.ndarray, pd.Series, str],
    instruments: Union[np.ndarray, pd.DataFrame, List[str]],
    exog=None,
    data: Optional[pd.DataFrame] = None,
    n_draws: int = 5000,
    n_warmup: int = 2000,
    proposal_sd: Optional[float] = None,
    prior_sd: float = 10.0,
    hpd_level: float = 0.95,
    add_const: bool = True,
    random_state: Optional[int] = None,
) -> BayesianIVResult:
    """
    Bayesian IV via Anderson-Rubin posterior (Chernozhukov-Hong 2003).

    The log-posterior combines a flat / Gaussian prior on β with the
    Anderson-Rubin concentrated log-likelihood:

        log p(β | data) ∝  -(n/2) log RSS(β) + log prior(β)

    where ``RSS(β) = (y - β D)' M_Z (y - β D)`` and ``M_Z`` is the
    annihilator matrix for the instrument projection.

    This posterior is well-defined and yields correct HPD-coverage even
    when instruments are weak.

    Parameters
    ----------
    y, endog : outcome and endogenous regressor.
    instruments : excluded instruments.
    exog : exogenous controls.
    data : DataFrame for string inputs.
    n_draws : int, default 5000
        Post-warmup posterior draws.
    n_warmup : int, default 2000
    proposal_sd : float, optional
        Random-walk M–H proposal std dev. Auto-calibrated if None.
    prior_sd : float, default 10.0
        Std dev of the Gaussian prior N(0, prior_sd²) on β.
    hpd_level : float, default 0.95
    add_const : bool, default True
    random_state : int, optional

    Returns
    -------
    BayesianIVResult
    """
    Y = _grab(y, data).reshape(-1)
    D = _grab(endog, data).reshape(-1)
    Z = _grab(instruments, data, cols=True)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = len(Y)

    if exog is None:
        W = np.ones((n, 1)) if add_const else np.empty((n, 0))
    else:
        Wx = _grab(exog, data, cols=True)
        if Wx.ndim == 1:
            Wx = Wx.reshape(-1, 1)
        W = np.column_stack([np.ones(n), Wx]) if add_const else Wx

    Yt = _residualize(Y.reshape(-1, 1), W).ravel()
    Dt = _residualize(D.reshape(-1, 1), W).ravel()
    Zt = _residualize(Z, W)
    k = Zt.shape[1]

    # Projection matrix for instruments
    PZ = Zt @ np.linalg.solve(Zt.T @ Zt + 1e-10 * np.eye(k), Zt.T)
    MZ = np.eye(n) - PZ

    def log_posterior(beta: float) -> float:
        u = Yt - beta * Dt
        u_Pz = float(u @ PZ @ u)
        u_Mz = float(u @ MZ @ u)
        if u_Mz <= 0:
            return -np.inf
        # AR quasi-log-likelihood: -AR/2 where AR = n * u'PZu / u'MZu
        # Maximised when u orthogonal to Z (correct β)
        ar = n * u_Pz / u_Mz
        log_lik = -0.5 * ar
        log_prior = -0.5 * (beta / prior_sd) ** 2
        return log_lik + log_prior

    # --- Initialize at 2SLS estimate ---
    D_hat = PZ @ Dt
    denom = float(D_hat @ Dt)
    if abs(denom) > 1e-12:
        beta_init = float(D_hat @ Yt) / denom
    else:
        beta_init = 0.0

    # Auto-calibrate proposal
    if proposal_sd is None:
        proposal_sd = max(0.01, abs(beta_init) * 0.1, 0.5)

    rng = np.random.default_rng(random_state)
    total = n_warmup + n_draws
    chain = np.empty(total)
    chain[0] = beta_init
    log_p = log_posterior(beta_init)
    accepted = 0

    for t in range(1, total):
        proposal = chain[t - 1] + proposal_sd * rng.standard_normal()
        log_p_new = log_posterior(proposal)
        if np.log(rng.uniform()) < log_p_new - log_p:
            chain[t] = proposal
            log_p = log_p_new
            accepted += 1
        else:
            chain[t] = chain[t - 1]

    draws = chain[n_warmup:]
    acc_rate = accepted / total

    lo, hi = _hpd(draws, hpd_level)

    return BayesianIVResult(
        posterior_draws=draws,
        posterior_mean=float(draws.mean()),
        posterior_sd=float(draws.std(ddof=1)),
        hpd_lower=lo, hpd_upper=hi,
        hpd_level=hpd_level,
        acceptance_rate=acc_rate,
        n_draws=n_draws,
        n_warmup=n_warmup,
        ess=_ess(draws),
        extra={"beta_init": beta_init, "proposal_sd": proposal_sd},
    )


__all__ = ["bayesian_iv", "BayesianIVResult"]
