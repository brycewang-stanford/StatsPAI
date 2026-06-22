"""Bayesian synthetic control via PyMC.

A Bayesian counterpart to :func:`statspai.synth.synth`. Donor weights live on the
simplex (a Dirichlet prior, matching the classic Abadie-Diamond-Hainmueller
convex-hull constraint) and are fit to the pre-treatment outcome path; the
post-treatment gap between the treated unit and its synthetic counterpart is the
estimand. Because weights are sampled, the counterfactual trajectory comes with
genuine credible bands — the small-N honesty that motivates a Bayesian SC when
there are few pre-periods or donors.

The counterfactual is stored under the shared keys read by
:func:`statspai.counterfactual_data`, so ``sp.counterfactual_plot`` works here too.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._base import BayesianCausalResult, _require_pymc, _sample_model


def _wide_panel(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    donors: Optional[Sequence[Any]],
) -> "Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]":
    """Pivot long panel to (times, treated_vec, donor_matrix, donor_names)."""
    for c in (outcome, unit, time):
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found in data")
    wide = data.pivot_table(index=time, columns=unit, values=outcome)
    if treated_unit not in wide.columns:
        raise ValueError(
            f"treated_unit={treated_unit!r} not found among units "
            f"{list(wide.columns)[:8]}..."
        )
    if donors is None:
        donor_names = [u for u in wide.columns if u != treated_unit]
    else:
        donor_names = [u for u in donors if u != treated_unit]
        missing = [u for u in donor_names if u not in wide.columns]
        if missing:
            raise ValueError(f"donor units not found: {missing}")
    if len(donor_names) < 2:
        raise ValueError(f"Need at least 2 donor units; got {len(donor_names)}.")
    sub = wide[[treated_unit] + donor_names].dropna(axis=0, how="any")
    if sub.empty:
        raise ValueError("No time periods with complete treated + donor observations.")
    times = np.asarray(sub.index.tolist())
    treated = sub[treated_unit].to_numpy(dtype=float)
    donor_mat = sub[donor_names].to_numpy(dtype=float)
    return times, treated, donor_mat, donor_names


def bayes_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    *,
    donors: Optional[Sequence[Any]] = None,
    prior_weights_alpha: float = 1.0,
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
    """Bayesian synthetic control.

    Fits convex donor weights ``w`` (Dirichlet prior on the simplex) to the
    treated unit's pre-treatment outcome path, then reports the posterior of the
    average post-treatment gap (ATT) between the treated unit and its synthetic
    counterpart. The posterior over ``w`` yields a counterfactual trajectory
    with credible bands.

    Parameters
    ----------
    data : pd.DataFrame
        Long panel (one row per unit-time).
    outcome, unit, time : str
        Outcome, unit-id and time column names.
    treated_unit : any
        The treated unit's id.
    treatment_time : any
        First treated period (inclusive); periods ``< treatment_time`` are the
        pre-treatment fitting window.
    donors : sequence, optional
        Donor pool. Defaults to every unit except the treated one.
    prior_weights_alpha : float, default 1.0
        Dirichlet concentration. ``1.0`` is uniform over the simplex; ``<1``
        favours sparse weights.
    prior_noise : float, default 5.0
        Half-Normal scale for the pre-period fit residual SD.
    rope : (float, float), optional
        Region of practical equivalence for the ATT.
    hdi_prob : float, default 0.95
    inference, advi_iterations, draws, tune, chains, target_accept,
    random_state, progressbar :
        Sampler controls; see :func:`sp.bayes_did`.

    Returns
    -------
    BayesianCausalResult
        ``estimand='ATT'``. ``model_info`` carries posterior-mean donor
        weights and the counterfactual trajectory with credible bands.

    Examples
    --------
    >>> import statspai as sp  # doctest: +SKIP
    >>> df = sp.california_prop99()  # doctest: +SKIP
    >>> res = sp.bayes_synth(  # doctest: +SKIP
    ...     df, outcome='packspercapita', unit='state',
    ...     time='year', treated_unit='California',
    ...     treatment_time=1989, draws=500, chains=2,
    ... )
    >>> res.posterior_mean  # posterior mean ATT  # doctest: +SKIP
    """
    pm, _ = _require_pymc()

    times, treated, donor_mat, donor_names = _wide_panel(
        data, outcome, unit, time, treated_unit, donors
    )
    pre_mask = np.asarray(times) < treatment_time
    post_mask = ~pre_mask
    if pre_mask.sum() < 2:
        raise ValueError(f"Need >= 2 pre-treatment periods; got {int(pre_mask.sum())}.")
    if post_mask.sum() < 1:
        raise ValueError("No post-treatment periods (check treatment_time).")

    Y_pre = donor_mat[pre_mask]
    Y_post_donor = donor_mat[post_mask]
    y_pre = treated[pre_mask]
    y_post = treated[post_mask]
    n_donors = donor_mat.shape[1]

    ybar_post_treated = float(y_post.mean())
    ybar_post_donor = Y_post_donor.mean(axis=0)  # (J,)

    with pm.Model() as model:
        w = pm.Dirichlet("w", a=np.full(n_donors, float(prior_weights_alpha)))
        sigma = pm.HalfNormal("sigma", sigma=prior_noise)
        mu_pre = pm.math.dot(Y_pre, w)
        pm.Normal("y_pre_obs", mu=mu_pre, sigma=sigma, observed=y_pre)
        pm.Deterministic("att", ybar_post_treated - pm.math.dot(ybar_post_donor, w))

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

    from ._base import _summarise_posterior

    summary = _summarise_posterior(trace, "att", hdi_prob=hdi_prob, rope=rope)

    # Counterfactual trajectory + credible bands from the weight posterior.
    w_draws = trace.posterior["w"].values.reshape(-1, n_donors)  # (S, J)
    synth_draws = donor_mat @ w_draws.T  # (T, S)
    counterfactual = synth_draws.mean(axis=1)
    lo = (1.0 - hdi_prob) / 2.0
    cf_lower = np.quantile(synth_draws, lo, axis=1)
    cf_upper = np.quantile(synth_draws, 1.0 - lo, axis=1)
    w_mean = w_draws.mean(axis=0)

    detail = {
        "time": np.asarray(times),
        "observed": treated,
        "counterfactual": counterfactual,
        "cf_lower": cf_lower,
        "cf_upper": cf_upper,
        "post": post_mask,
    }
    model_info = {
        "inference": inference,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "target_accept": target_accept,
        "treated_unit": treated_unit,
        "treatment_time": treatment_time,
        "n_donors": n_donors,
        "n_pre": int(pre_mask.sum()),
        "n_post": int(post_mask.sum()),
        "weights": {str(n): float(v) for n, v in zip(donor_names, w_mean)},
        "prior_weights_alpha": prior_weights_alpha,
        "prior_noise": prior_noise,
        # counterfactual contract
        "observed": treated,
        "counterfactual": counterfactual,
        "times": np.asarray(times),
        "detail": detail,
    }

    return BayesianCausalResult(
        method="Bayesian synthetic control",
        estimand="ATT",
        posterior_mean=summary["posterior_mean"],
        posterior_median=summary["posterior_median"],
        posterior_sd=summary["posterior_sd"],
        hdi_lower=summary["hdi_lower"],
        hdi_upper=summary["hdi_upper"],
        prob_positive=summary["prob_positive"],
        prob_rope=summary.get("prob_rope"),
        rhat=summary["rhat"],
        ess=summary["ess"],
        n_obs=int(donor_mat.shape[0]),
        hdi_prob=hdi_prob,
        trace=trace,
        model_info=model_info,
    )
