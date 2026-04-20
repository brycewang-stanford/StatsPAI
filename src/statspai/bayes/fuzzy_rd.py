"""Bayesian fuzzy regression discontinuity via PyMC.

Models the intent-to-treat effect on Y and the first-stage effect on
D at the cutoff jointly, then forms the LATE as the ratio
``itt_Y / itt_D`` (Wald estimator). Using a deterministic ratio at
the posterior level means the resulting posterior on ``late``
automatically inherits both noise channels and becomes heavy-tailed
under weak compliance (``itt_D ≈ 0``).
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


def _local_poly_design(
    x: np.ndarray, cutoff: float, treated: np.ndarray, poly: int,
) -> np.ndarray:
    """Design matrix for one-side-dependent polynomial in (x - cutoff)."""
    x_c = x - cutoff
    cols = []
    for k in range(1, poly + 1):
        cols.append(x_c ** k)
    for k in range(1, poly + 1):
        cols.append(treated * (x_c ** k))
    if not cols:
        return np.zeros((len(x), 0))
    return np.column_stack(cols)


def bayes_fuzzy_rd(
    data: pd.DataFrame,
    y: str,
    treat: str,
    running: str,
    cutoff: float = 0.0,
    bandwidth: Optional[float] = None,
    poly: int = 1,
    *,
    prior_late: Tuple[float, float] = (0.0, 10.0),
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
    """Bayesian fuzzy RD — LATE at the cutoff via joint ITT-Y / ITT-D posterior.

    Models within-bandwidth observations:

    .. code-block:: text

        Y_i = a_Y + itt_Y * I(x_i >= c) + poly_Y(x_i - c) + eps_Y
        D_i = a_D + itt_D * I(x_i >= c) + poly_D(x_i - c) + eps_D
        LATE := itt_Y / itt_D   (deterministic posterior)

    Each polynomial gets independent slopes on each side (standard
    local-linear RD practice). Priors on the ITT jumps are
    Normal(mean, ``prior_slope_sigma``) (weakly informative); LATE
    inherits its prior from ``prior_late`` but the ratio form means
    the posterior is data-driven once compliance is non-trivial.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    treat : str
        Realised treatment uptake (0/1). May disagree with
        ``running >= cutoff`` when compliance is partial.
    running : str
        Running variable.
    cutoff : float, default 0.0
    bandwidth : float, optional
        Default: ``0.5 * std(running)``.
    poly : int, default 1
    prior_late : (float, float)
        Normal prior on the LATE (used as a regularising prior on the
        deterministic ratio — it is implemented as a prior on the
        ``itt_Y`` magnitude so that the ratio inherits natural scale).
    prior_slope_sigma, prior_noise : float
    rope, hdi_prob, draws, tune, chains, target_accept, random_state,
    progressbar : see :func:`bayes_did`.

    Returns
    -------
    BayesianCausalResult
        Posterior over the LATE.
    """
    pm, _ = _require_pymc()

    for c in (y, treat, running):
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found in data")
    if poly < 1:
        raise ValueError(f"poly must be >= 1, got {poly}")

    clean = data[[y, treat, running]].dropna().reset_index(drop=True)
    if len(clean) < 20:
        raise ValueError(
            f"Need at least 20 observations after dropping NA, "
            f"got {len(clean)}."
        )

    x_full = clean[running].to_numpy(dtype=float)
    bw = float(bandwidth) if bandwidth is not None else 0.5 * float(x_full.std())
    if not np.isfinite(bw) or bw <= 0:
        raise ValueError(f"Invalid bandwidth {bw}")

    mask = (x_full >= cutoff - bw) & (x_full <= cutoff + bw)
    if mask.sum() < 20:
        raise ValueError(
            f"Only {int(mask.sum())} observations inside "
            f"[{cutoff - bw}, {cutoff + bw}]; widen the bandwidth."
        )

    sub = clean.loc[mask].reset_index(drop=True)
    Y = sub[y].to_numpy(dtype=float)
    D = sub[treat].to_numpy(dtype=float)
    uniq_D = np.unique(D)
    if not set(uniq_D).issubset({0.0, 1.0}):
        raise ValueError(
            f"Treatment uptake '{treat}' must be binary 0/1; "
            f"got unique values {uniq_D}."
        )
    x = sub[running].to_numpy(dtype=float)
    treated_side = (x >= cutoff).astype(float)

    poly_design = _local_poly_design(x, cutoff, treated_side, poly)

    mu_late, sigma_late = prior_late

    with pm.Model() as model:
        # Outcome equation
        a_Y = pm.Normal('a_Y', mu=0.0, sigma=prior_slope_sigma)
        itt_Y = pm.Normal('itt_Y', mu=0.0, sigma=prior_slope_sigma)
        # First-stage equation
        a_D = pm.Normal('a_D', mu=0.0, sigma=prior_slope_sigma)
        itt_D = pm.Normal('itt_D', mu=0.0, sigma=prior_slope_sigma)

        if poly_design.shape[1] > 0:
            beta_poly_Y = pm.Normal(
                'beta_poly_Y', mu=0.0,
                sigma=prior_slope_sigma, shape=poly_design.shape[1],
            )
            beta_poly_D = pm.Normal(
                'beta_poly_D', mu=0.0,
                sigma=prior_slope_sigma, shape=poly_design.shape[1],
            )
            poly_Y = pm.math.dot(poly_design, beta_poly_Y)
            poly_D = pm.math.dot(poly_design, beta_poly_D)
        else:
            poly_Y = 0.0
            poly_D = 0.0

        mu_Y = a_Y + itt_Y * treated_side + poly_Y
        mu_D = a_D + itt_D * treated_side + poly_D

        sigma_Y = pm.HalfNormal('sigma_Y', sigma=prior_noise)
        sigma_D = pm.HalfNormal('sigma_D', sigma=prior_noise)

        pm.Normal('y_obs', mu=mu_Y, sigma=sigma_Y, observed=Y)
        pm.Normal('d_obs', mu=mu_D, sigma=sigma_D, observed=D)

        # Deterministic LATE = itt_Y / itt_D. pm.Deterministic writes
        # it into the trace so the downstream summary machinery picks
        # it up automatically.
        #
        # Guard: if a posterior sample places itt_D extremely close to
        # zero, the ratio can explode. We clamp the denominator's
        # absolute value to a tiny floor by switching to the nearest
        # sign-preserving floor when too small.
        eps_floor = 1e-6
        safe_denom = pm.math.where(
            pm.math.abs(itt_D) < eps_floor,
            pm.math.where(itt_D >= 0, eps_floor, -eps_floor),
            itt_D,
        )
        late = pm.Deterministic('late', itt_Y / safe_denom)

        # A weakly informative prior on `late` via a soft potential —
        # keeps the ratio away from the pathological regime when
        # itt_D ~ 0 in degenerate DGPs, without impairing recovery on
        # well-identified cases.
        pm.Potential(
            'late_prior',
            pm.logp(pm.Normal.dist(mu=mu_late, sigma=sigma_late), late),
        )

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_state,
            progressbar=progressbar,
            return_inferencedata=True,
        )

    summary = _summarise_posterior(
        trace, 'late', hdi_prob=hdi_prob, rope=rope,
    )

    # Also summarise the first-stage strength for diagnostics
    itt_D_post = trace.posterior['itt_D'].values.ravel()
    first_stage_mean = float(np.mean(itt_D_post))
    first_stage_sd = float(np.std(itt_D_post, ddof=1))
    first_stage_prob_nonzero = float(
        np.mean(np.abs(itt_D_post) > 0.05)  # heuristic threshold
    )

    model_info = {
        'draws': draws,
        'tune': tune,
        'chains': chains,
        'target_accept': target_accept,
        'cutoff': cutoff,
        'bandwidth': bw,
        'poly': poly,
        'n_inside': int(mask.sum()),
        'n_treated_local': int(D.sum()),
        'first_stage_mean': first_stage_mean,
        'first_stage_sd': first_stage_sd,
        'first_stage_prob_nonzero': first_stage_prob_nonzero,
        'prior_late': prior_late,
        'prior_slope_sigma': prior_slope_sigma,
        'prior_noise': prior_noise,
    }

    return BayesianCausalResult(
        method=f'Bayesian fuzzy RD (poly={poly})',
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
