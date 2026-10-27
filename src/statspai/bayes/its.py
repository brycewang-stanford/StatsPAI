"""Bayesian interrupted time series via PyMC.

A Bayesian counterpart to :func:`statspai.timeseries.its`: the same segmented
design (level + slope change at a known intervention) with weakly informative
priors, returning a full posterior on the immediate level change plus a
posterior-predictive counterfactual with credible bands. The counterfactual is
stored under the shared keys read by :func:`statspai.counterfactual_data`, so
``sp.counterfactual_plot`` works on the Bayesian fit too.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ._base import (
    BayesianCausalResult,
    _require_pymc,
    _sample_model,
    _summarise_posterior,
)


@accepts_aliases(seed="random_state")
def bayes_its(
    data: pd.DataFrame,
    y: str,
    time: Optional[str] = None,
    intervention: Optional[int] = None,
    *,
    prior_level: Tuple[float, float] = (0.0, 10.0),
    prior_slope_sigma: float = 10.0,
    prior_trend_sigma: float = 10.0,
    prior_noise: float = 5.0,
    rope: Optional[Tuple[float, float]] = None,
    hdi_prob: float = 0.95,
    inference: str = "nuts",
    advi_iterations: int = 20000,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_state: int = 42,
    progressbar: bool = False,
) -> BayesianCausalResult:
    """Bayesian interrupted time series (segmented regression).

    Models ``y`` as a pre-intervention linear trend plus an immediate level
    change and a slope change at ``intervention`` (a row position, as in
    :func:`sp.its`). The reported estimand is the posterior of the immediate
    **level change**; the slope-change posterior and a posterior-predictive
    counterfactual (with credible bands) are stored in ``model_info`` /
    ``detail``.

    Parameters
    ----------
    data : pd.DataFrame
        Observed series, one row per period. Rows are put in time order
        (stable sort on ``time``) before fitting, as in :func:`sp.its`; a
        repeated time point raises ``MethodIncompatibility``.
    y : str
        Outcome column.
    time : str, optional
        Time column (used for the trend term and the x-axis). Defaults to the
        row index, with the rows taken to be in time order.
    intervention : int
        Integer row position in the time-ordered series where the
        intervention begins (``0 < intervention < n``).
    prior_level : (float, float), default (0.0, 10.0)
        Mean / SD of the Normal prior on the level change (the estimand).
    prior_slope_sigma, prior_trend_sigma, prior_noise : float
        Prior scales for the slope change, the pre-trend, and the residual SD.
    rope : (float, float), optional
        Region of practical equivalence for the level change.
    hdi_prob : float, default 0.95
    inference, advi_iterations, draws, tune, chains, target_accept,
    random_state, progressbar :
        Sampler controls; see :func:`sp.bayes_did`.

    Returns
    -------
    BayesianCausalResult
        ``estimand='level change'``. ``model_info['slope_change']`` holds the
        slope-change posterior summary; ``detail`` carries the observed /
        counterfactual series with credible bands.

    Examples
    --------
    >>> import statspai as sp  # doctest: +SKIP
    >>> res = sp.bayes_its(df, y='y', time='t', intervention=30,
    ...                    draws=500, tune=500, chains=2)  # doctest: +SKIP
    >>> res.posterior_mean  # posterior mean level change  # doctest: +SKIP
    """
    check_columns(data, {"y": y, "time": time}, function="sp.bayes_its")
    from ..timeseries.its import _time_ordered

    # Time order first (a repeated period is refused before PyMC loads).
    work = _time_ordered(data, time, "sp.bayes_its")
    pm, _ = _require_pymc()

    if intervention is None:
        raise ValueError("intervention (integer row position) is required.")

    Y = work[y].to_numpy(dtype=float)
    n = len(Y)
    intervention = int(intervention)
    if not 0 < intervention < n:
        raise ValueError(
            f"intervention index {intervention} must satisfy 0 < i < n={n}."
        )
    t = (
        work[time].to_numpy(dtype=float)
        if time is not None
        else np.arange(n, dtype=float)
    )
    if not (np.isfinite(Y).all() and np.isfinite(t).all()):
        raise ValueError("outcome/time contains NaN or inf; clean the series first.")

    idx = np.arange(n)
    D = (idx >= intervention).astype(float)
    t_post = np.where(idx >= intervention, idx - intervention, 0).astype(float)
    t_c = t - float(t.mean())  # centre the trend for conditioning

    mu_level, sigma_level = prior_level

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=prior_trend_sigma)
        beta_trend = pm.Normal("beta_trend", mu=0.0, sigma=prior_trend_sigma)
        tau_level = pm.Normal("tau_level", mu=mu_level, sigma=sigma_level)
        tau_slope = pm.Normal("tau_slope", mu=0.0, sigma=prior_slope_sigma)
        sigma = pm.HalfNormal("sigma", sigma=prior_noise)
        mu = alpha + beta_trend * t_c + tau_level * D + tau_slope * t_post
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=Y)

    trace = _sample_model(
        model,
        inference=inference,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_state=random_state,
        progressbar=progressbar,
        advi_iterations=advi_iterations,
    )

    summary = _summarise_posterior(trace, "tau_level", hdi_prob=hdi_prob, rope=rope)
    slope_summary = _summarise_posterior(trace, "tau_slope", hdi_prob=hdi_prob)

    # Posterior-predictive counterfactual (no level/slope change): draws of
    # alpha + beta_trend * t_c, evaluated over the whole series.
    post = trace.posterior
    a = post["alpha"].values.ravel()
    bt = post["beta_trend"].values.ravel()
    cf_draws = a[:, None] + bt[:, None] * t_c[None, :]  # (n_draws, n)
    counterfactual = cf_draws.mean(axis=0)
    lo = (1.0 - hdi_prob) / 2.0
    cf_lower = np.quantile(cf_draws, lo, axis=0)
    cf_upper = np.quantile(cf_draws, 1.0 - lo, axis=0)

    detail = {
        "time": t,
        "observed": Y,
        "counterfactual": counterfactual,
        "cf_lower": cf_lower,
        "cf_upper": cf_upper,
        "post": D.astype(bool),
    }

    model_info = {
        "inference": inference,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "target_accept": target_accept,
        "intervention_time": intervention,
        "slope_change": {
            "posterior_mean": slope_summary["posterior_mean"],
            "hdi_lower": slope_summary["hdi_lower"],
            "hdi_upper": slope_summary["hdi_upper"],
            "prob_positive": slope_summary["prob_positive"],
        },
        "prior_level": prior_level,
        "prior_slope_sigma": prior_slope_sigma,
        "prior_trend_sigma": prior_trend_sigma,
        "prior_noise": prior_noise,
        # counterfactual contract (mirrors sp.its detail)
        "observed": Y,
        "counterfactual": counterfactual,
        "times": t,
        "detail": detail,
    }

    result = BayesianCausalResult(
        method="Bayesian ITS (segmented)",
        estimand="level change",
        posterior_mean=summary["posterior_mean"],
        posterior_median=summary["posterior_median"],
        posterior_sd=summary["posterior_sd"],
        hdi_lower=summary["hdi_lower"],
        hdi_upper=summary["hdi_upper"],
        prob_positive=summary["prob_positive"],
        prob_rope=summary.get("prob_rope"),
        rhat=summary["rhat"],
        ess=summary["ess"],
        n_obs=n,
        hdi_prob=hdi_prob,
        trace=trace,
        model_info=model_info,
    )
    return result
