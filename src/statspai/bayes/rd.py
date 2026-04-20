"""Bayesian sharp regression discontinuity via PyMC.

Fits a local polynomial model around a known cutoff within a
user-supplied bandwidth, with independent slope coefficients on each
side of the cutoff and a normal prior on the discontinuity ``tau``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ._base import (
    BayesianCausalResult,
    _require_pymc,
    _summarise_posterior,
)


def _local_polynomial_design(
    x: np.ndarray,
    cutoff: float,
    treated: np.ndarray,
    poly: int,
) -> np.ndarray:
    """Build a design matrix with independent polynomial fits per side.

    Columns: ``treated, (x-c), (x-c)^2, ..., (x-c)^p,
              treated*(x-c), ..., treated*(x-c)^p``.
    The intercept is handled separately in the PyMC model.
    """
    x_c = x - cutoff
    cols = [treated.astype(float)]
    for k in range(1, poly + 1):
        cols.append(x_c ** k)
    for k in range(1, poly + 1):
        cols.append(treated * (x_c ** k))
    return np.column_stack(cols)


def bayes_rd(
    data: pd.DataFrame,
    y: str,
    running: str,
    cutoff: float = 0.0,
    bandwidth: Optional[float] = None,
    poly: int = 1,
    *,
    prior_tau: Tuple[float, float] = (0.0, 10.0),
    prior_slope_sigma: float = 10.0,
    prior_noise: float = 5.0,
    rope: Optional[Tuple[float, float]] = None,
    hdi_prob: float = 0.95,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_state: int = 42,
    progressbar: bool = False,
) -> BayesianCausalResult:
    """Bayesian sharp regression discontinuity.

    Within the bandwidth ``[cutoff - bw, cutoff + bw]`` the outcome
    is modelled as a polynomial (default order 1) in ``running - cutoff``
    with independent slopes on each side. The causal parameter is the
    jump ``tau`` at the cutoff, with a weakly informative Normal prior.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    running : str
        Running / forcing variable column.
    cutoff : float, default 0.0
        Threshold above which treatment applies (``running >= cutoff``).
    bandwidth : float, optional
        Half-width of the local window around ``cutoff``. If ``None``,
        uses the rule-of-thumb ``0.5 * std(running)``.
    poly : int, default 1
        Polynomial order (1 = local linear; 2 = local quadratic).
        Stata's ``rdrobust`` defaults to 1.
    prior_tau : (float, float), default ``(0.0, 10.0)``
        Mean / SD of the Normal prior on the discontinuity.
    prior_slope_sigma, prior_noise : float
        Prior scales for the polynomial slopes and residual SD.
    rope : (float, float), optional
        Region of practical equivalence.
    hdi_prob : float, default 0.95
    draws, tune, chains, target_accept, random_state, progressbar : see
        :func:`bayes_did`.

    Returns
    -------
    BayesianCausalResult
        Posterior summary of the local average treatment effect
        (LATE) at the cutoff.
    """
    pm, _ = _require_pymc()

    for c in (y, running):
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found in data")
    if poly < 1:
        raise ValueError(f"poly must be >= 1, got {poly}")

    clean = data[[y, running]].dropna().reset_index(drop=True)
    if len(clean) < 10:
        raise ValueError(
            f"Need at least 10 observations after dropping NA, got {len(clean)}."
        )
    x_full = clean[running].to_numpy(dtype=float)

    # Bandwidth
    bw = float(bandwidth) if bandwidth is not None else 0.5 * float(x_full.std())
    if not np.isfinite(bw) or bw <= 0:
        raise ValueError(f"Invalid bandwidth {bw}")

    mask = (x_full >= cutoff - bw) & (x_full <= cutoff + bw)
    if mask.sum() < 10:
        raise ValueError(
            f"Only {int(mask.sum())} observations inside "
            f"[{cutoff - bw}, {cutoff + bw}]; widen the bandwidth."
        )

    sub = clean.loc[mask].reset_index(drop=True)
    Y = sub[y].to_numpy(dtype=float)
    x = sub[running].to_numpy(dtype=float)
    treated = (x >= cutoff).astype(float)
    if treated.sum() == 0 or (1 - treated).sum() == 0:
        raise ValueError(
            "Local window contains observations on only one side of the cutoff."
        )

    Z = _local_polynomial_design(x, cutoff, treated, poly)
    mu_tau, sigma_tau = prior_tau

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0.0, sigma=prior_slope_sigma)
        # Independent beta for each design column. Column 0 is the
        # treated indicator (the jump) — re-labelled for clarity as tau
        # via a deterministic, but we also draw a dedicated tau prior
        # so users can inspect it directly.
        tau = pm.Normal('tau', mu=mu_tau, sigma=sigma_tau)
        other_cols = Z.shape[1] - 1
        beta_poly = pm.Normal(
            'beta_poly',
            mu=0.0,
            sigma=prior_slope_sigma,
            shape=other_cols,
        )
        linpred = alpha + tau * Z[:, 0] + pm.math.dot(Z[:, 1:], beta_poly)

        sigma = pm.HalfNormal('sigma', sigma=prior_noise)
        pm.Normal('y_obs', mu=linpred, sigma=sigma, observed=Y)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_state,
            progressbar=progressbar,
            return_inferencedata=True,
        )

    summary = _summarise_posterior(trace, 'tau', hdi_prob=hdi_prob, rope=rope)

    model_info = {
        'draws': draws,
        'tune': tune,
        'chains': chains,
        'target_accept': target_accept,
        'cutoff': cutoff,
        'bandwidth': bw,
        'poly': poly,
        'n_inside': int(mask.sum()),
        'n_treated_local': int(treated.sum()),
        'n_control_local': int((1 - treated).sum()),
        'prior_tau': prior_tau,
        'prior_slope_sigma': prior_slope_sigma,
        'prior_noise': prior_noise,
    }

    return BayesianCausalResult(
        method=f'Bayesian sharp RD (poly={poly})',
        estimand='LATE',
        posterior_mean=summary['posterior_mean'],
        posterior_median=summary['posterior_median'],
        posterior_sd=summary['posterior_sd'],
        hdi_lower=summary['hdi_lower'],
        hdi_upper=summary['hdi_upper'],
        prob_positive=summary['prob_positive'],
        prob_rope=summary.get('prob_rope'),
        rhat=summary['rhat'],
        ess=summary['ess'],
        n_obs=int(mask.sum()),
        hdi_prob=hdi_prob,
        trace=trace,
        model_info=model_info,
    )
