"""Bayesian Difference-in-Differences via PyMC.

Supports both 2×2 (no unit / time structure) and panel DID with
hierarchical Gaussian random effects on unit and/or time. The causal
parameter is the ATT coefficient ``tau`` on ``treat * post``.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ._base import (
    BayesianCausalResult,
    BayesianDIDResult,
    _az_hdi_compat,
    _require_pymc,
    _sample_model,
    _summarise_posterior,
)


def _prepare_did_frame(
    data: pd.DataFrame,
    y: str,
    treat: str,
    post: str,
    unit: Optional[str],
    time: Optional[str],
    covariates: Optional[List[str]],
) -> dict:
    """Validate, drop NA, and extract arrays for DID."""
    required = [y, treat, post]
    for c in required:
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found in data")

    cols = list(required)
    if unit is not None:
        if unit not in data.columns:
            raise ValueError(f"Column '{unit}' (unit) not found in data")
        cols.append(unit)
    if time is not None:
        if time not in data.columns:
            raise ValueError(f"Column '{time}' (time) not found in data")
        cols.append(time)
    if covariates:
        for c in covariates:
            if c not in data.columns:
                raise ValueError(f"Covariate '{c}' not found in data")
        cols.extend(covariates)

    clean = data[cols].dropna().reset_index(drop=True)
    n = len(clean)
    if n < 4:
        raise ValueError(
            f"DID needs at least 4 observations after dropping NA, got {n}."
        )

    Y = clean[y].to_numpy(dtype=float)
    T = clean[treat].to_numpy(dtype=float)
    P = clean[post].to_numpy(dtype=float)

    # Validate 0/1 coding
    for name, arr in [('treat', T), ('post', P)]:
        uniq = np.unique(arr)
        if not set(uniq).issubset({0.0, 1.0}):
            raise ValueError(
                f"'{name}' must be binary 0/1; got unique values {uniq}."
            )

    unit_idx: Optional[np.ndarray] = None
    n_units: int = 0
    if unit is not None:
        codes, uniques = pd.factorize(clean[unit], sort=True)
        unit_idx = np.asarray(codes, dtype=np.int64)
        n_units = int(len(uniques))
        if n_units < 2:
            raise ValueError(
                f"Unit column '{unit}' has < 2 distinct values; cannot fit "
                "panel random effect."
            )

    time_idx: Optional[np.ndarray] = None
    n_times: int = 0
    if time is not None:
        codes, uniques = pd.factorize(clean[time], sort=True)
        time_idx = np.asarray(codes, dtype=np.int64)
        n_times = int(len(uniques))
        if n_times < 2:
            raise ValueError(
                f"Time column '{time}' has < 2 distinct values; cannot fit "
                "time random effect."
            )

    X: Optional[np.ndarray] = None
    if covariates:
        X = clean[covariates].to_numpy(dtype=float)

    return {
        'n': n,
        'Y': Y,
        'T': T,
        'P': P,
        'DID': T * P,
        'unit_idx': unit_idx,
        'n_units': n_units,
        'time_idx': time_idx,
        'n_times': n_times,
        'X': X,
        'covariates': list(covariates) if covariates else [],
    }


def bayes_did(
    data: pd.DataFrame,
    y: str,
    treat: str,
    post: str,
    unit: Optional[str] = None,
    time: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    *,
    cohort: Optional[str] = None,
    prior_ate: Tuple[float, float] = (0.0, 10.0),
    prior_unit_sigma: float = 5.0,
    prior_time_sigma: float = 5.0,
    prior_noise: float = 5.0,
    prior_covariate_sigma: float = 10.0,
    rope: Optional[Tuple[float, float]] = None,
    hdi_prob: float = 0.95,
    inference: str = 'nuts',
    advi_iterations: int = 20000,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_state: int = 42,
    progressbar: bool = False,
) -> BayesianDIDResult:
    """Bayesian difference-in-differences via PyMC.

    Two model shapes:

    - **2×2** (no ``unit``/``time``): ``y = a + b1*treat + b2*post + tau*treat*post + X*beta + eps``.
      Both group effects are Normal(0, ``prior_covariate_sigma``).

    - **Panel** (``unit`` or ``time`` supplied): hierarchical Gaussian
      random effects replace the dummies. ATT is the coefficient on
      ``treat * post``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    treat : str
        Binary 0/1 indicator for "ever-treated".
    post : str
        Binary 0/1 indicator for the post-treatment period.
    unit, time : str, optional
        Panel indices. If supplied, replaces the corresponding main
        effect with a hierarchical Gaussian random effect. Omit for
        the 2×2 model.
    covariates : list of str, optional
        Additional time-varying regressors. Standardised inside PyMC
        via Normal(0, ``prior_covariate_sigma``) priors.
    cohort : str, optional
        Column name identifying each treated unit's cohort (typically
        the first-treatment period in a staggered design). When
        supplied, the coefficient on ``treat*post`` is replaced by a
        per-cohort vector ``tau_c`` with a shared ``Normal(prior_ate)``
        prior; the result populates
        :attr:`BayesianDIDResult.cohort_summaries` so
        ``tidy(terms='per_cohort')`` returns one row per cohort.
        Untreated / never-treated units should carry a sentinel
        cohort value (e.g. ``-1``, ``'never'``) — they are grouped
        into a single cohort whose τ is still estimated but typically
        shrinks toward zero because their ``treat*post`` is
        identically zero. The top-level posterior ATT is the
        size-weighted mean of the cohort τ's over **treated units**.
    prior_ate : (float, float), default ``(0.0, 10.0)``
        Mean and SD of the Normal prior on ``tau``.
    prior_unit_sigma, prior_time_sigma : float
        Half-Normal scale on the random-effect SDs.
    prior_noise : float
        Half-Normal scale on the residual SD.
    prior_covariate_sigma : float
        Normal SD on fixed effects and covariate slopes.
    rope : (float, float), optional
        Region of practical equivalence for the ATT. If supplied the
        result includes ``prob_rope = P(lo < tau < hi | data)``.
    hdi_prob : float, default 0.95
    draws, tune, chains : int
    target_accept : float, default 0.9
        Higher value if you see divergences.
    random_state : int, default 42
    progressbar : bool, default False
        PyMC progress bar — off by default because this is often
        called inside notebooks / scripts and noise hurts readability.

    Returns
    -------
    BayesianCausalResult
        Posterior summary of the ATT.
    """
    pm, _ = _require_pymc()

    prep = _prepare_did_frame(data, y, treat, post, unit, time, covariates)
    n = prep['n']
    Y = prep['Y']
    T = prep['T']
    P = prep['P']
    DID = prep['DID']
    X = prep['X']

    use_unit_re = prep['unit_idx'] is not None
    use_time_re = prep['time_idx'] is not None

    # Cohort wiring: factorise once so the model gets integer codes and
    # the result carries back the user's original labels for tidy(). We
    # factorise on the dropna'd frame to keep index alignment with DID.
    cohort_codes: Optional[np.ndarray] = None
    cohort_labels: list = []
    if cohort is not None:
        if cohort not in data.columns:
            raise ValueError(f"Column '{cohort}' (cohort) not found in data")
        cols_for_factor = [y, treat, post]
        if unit is not None:
            cols_for_factor.append(unit)
        if time is not None:
            cols_for_factor.append(time)
        if covariates:
            cols_for_factor.extend(covariates)
        cols_for_factor.append(cohort)
        clean_for_cohort = (
            data[cols_for_factor].dropna().reset_index(drop=True)
        )
        codes, uniques = pd.factorize(clean_for_cohort[cohort], sort=True)
        cohort_codes = np.asarray(codes, dtype=np.int64)
        cohort_labels = list(uniques)
        n_cohorts = len(cohort_labels)
        if n_cohorts < 2:
            raise ValueError(
                f"cohort column '{cohort}' has < 2 distinct values; "
                "per-cohort ATT requires at least 2 cohorts."
            )

    mu_ate, sigma_ate = prior_ate

    with pm.Model() as model:
        intercept = pm.Normal('intercept', mu=0.0, sigma=prior_covariate_sigma)

        # Main effects: use dummy coefficients when no random effect is used
        if use_unit_re:
            sigma_unit = pm.HalfNormal('sigma_unit', sigma=prior_unit_sigma)
            alpha_unit = pm.Normal(
                'alpha_unit',
                mu=0.0,
                sigma=sigma_unit,
                shape=prep['n_units'],
            )
            unit_effect = alpha_unit[prep['unit_idx']]
            treat_main = 0.0  # absorbed by unit FE for static treat
            beta_treat = pm.Normal(
                'beta_treat', mu=0.0, sigma=prior_covariate_sigma,
            )
            # Keep beta_treat for models where treat varies within unit (rare);
            # if treat is fully collinear with unit, the posterior concentrates
            # on the prior — documented behaviour.
            treat_main = beta_treat * T
        else:
            beta_treat = pm.Normal(
                'beta_treat', mu=0.0, sigma=prior_covariate_sigma,
            )
            treat_main = beta_treat * T
            unit_effect = 0.0

        if use_time_re:
            sigma_time = pm.HalfNormal('sigma_time', sigma=prior_time_sigma)
            gamma_time = pm.Normal(
                'gamma_time',
                mu=0.0,
                sigma=sigma_time,
                shape=prep['n_times'],
            )
            time_effect = gamma_time[prep['time_idx']]
            post_main = 0.0  # absorbed by time FE
        else:
            beta_post = pm.Normal(
                'beta_post', mu=0.0, sigma=prior_covariate_sigma,
            )
            post_main = beta_post * P
            time_effect = 0.0

        # The causal parameter. Two parameterisations:
        #   - scalar `tau` (single-τ model, v0.9.15 behaviour)
        #   - vector `tau_cohort` of length n_cohorts when `cohort` is
        #     supplied; per-unit contribution is tau_cohort[c_i] * DID_i
        if cohort_codes is None:
            tau = pm.Normal('tau', mu=mu_ate, sigma=sigma_ate)
            did_contribution = tau * DID
        else:
            tau_cohort = pm.Normal(
                'tau_cohort',
                mu=mu_ate,
                sigma=sigma_ate,
                shape=len(cohort_labels),
            )
            # Row i's DID contribution uses its cohort's τ. Casting to
            # PyMC via fancy-indexing on a numpy int64 array is the
            # idiomatic pattern in this module (cf. `alpha_unit[unit_idx]`).
            did_contribution = tau_cohort[cohort_codes] * DID

        linpred = (
            intercept
            + treat_main
            + post_main
            + unit_effect
            + time_effect
            + did_contribution
        )

        if X is not None:
            beta_cov = pm.Normal(
                'beta_cov',
                mu=0.0,
                sigma=prior_covariate_sigma,
                shape=X.shape[1],
            )
            linpred = linpred + pm.math.dot(X, beta_cov)

        sigma = pm.HalfNormal('sigma', sigma=prior_noise)
        pm.Normal('y_obs', mu=linpred, sigma=sigma, observed=Y)

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

    shape_label = (
        'panel' if (use_unit_re or use_time_re) else '2x2'
    )
    model_info = {
        'inference': inference,
        'draws': draws,
        'tune': tune,
        'chains': chains,
        'target_accept': target_accept,
        'prior_ate': prior_ate,
        'prior_unit_sigma': prior_unit_sigma,
        'prior_time_sigma': prior_time_sigma,
        'prior_noise': prior_noise,
        'prior_covariate_sigma': prior_covariate_sigma,
        'use_unit_re': use_unit_re,
        'use_time_re': use_time_re,
        'n_units': prep['n_units'],
        'n_times': prep['n_times'],
        'covariates': prep['covariates'],
    }

    # Single-τ model: identical to v0.9.15 output.
    if cohort_codes is None:
        summary = _summarise_posterior(trace, 'tau', hdi_prob=hdi_prob, rope=rope)
        method = f"Bayesian DID ({shape_label})"
        return BayesianDIDResult(
            method=method,
            estimand='ATT',
            posterior_mean=summary['posterior_mean'],
            posterior_median=summary['posterior_median'],
            posterior_sd=summary['posterior_sd'],
            hdi_lower=summary['hdi_lower'],
            hdi_upper=summary['hdi_upper'],
            prob_positive=summary['prob_positive'],
            prob_rope=summary.get('prob_rope'),
            rhat=summary['rhat'],
            ess=summary['ess'],
            n_obs=n,
            hdi_prob=hdi_prob,
            trace=trace,
            model_info=model_info,
        )

    # Per-cohort model: walk the posterior of `tau_cohort` (chains,
    # draws, n_cohorts) to build per-cohort summaries AND a
    # treated-size-weighted average as the top-level ATT.
    _, az = _require_pymc()
    tau_post = trace.posterior['tau_cohort'].values  # (chains, draws, n_c)
    tau_flat = tau_post.reshape(-1, tau_post.shape[-1])   # (S, n_c)

    # Per-cohort summaries keyed by str(label) so tidy() can match a
    # 'cohort:2019' style label without caring about int vs. str.
    cohort_summaries: dict = {}
    treated_mask = prep['T'] * prep['P'] > 0  # DID==1 indicates treated-post
    treated_counts_per_cohort = np.zeros(len(cohort_labels), dtype=float)
    for idx, label in enumerate(cohort_labels):
        col = tau_flat[:, idx]
        hdi = _az_hdi_compat(col, hdi_prob=hdi_prob)
        cohort_summaries[str(label)] = {
            'posterior_mean': float(col.mean()),
            'posterior_median': float(np.median(col)),
            'posterior_sd': float(col.std(ddof=1)),
            'hdi_lower': float(hdi[0]),
            'hdi_upper': float(hdi[1]),
            'prob_positive': float(np.mean(col > 0)),
        }
        # Count units in cohort idx that are treated-post. Gives the
        # treated-unit weighting used for the pooled ATT below.
        in_cohort = (cohort_codes == idx)
        treated_counts_per_cohort[idx] = float(
            (in_cohort & treated_mask).sum()
        )

    # Size-weighted pooled ATT posterior. When no treated-post rows
    # exist (e.g. all cohorts are pure controls), fall back to equal
    # weights so the pooled row is defined.
    total_treated = float(treated_counts_per_cohort.sum())
    if total_treated > 0:
        w = treated_counts_per_cohort / total_treated
    else:
        w = np.full(len(cohort_labels), 1.0 / len(cohort_labels))
    pooled_samples = tau_flat @ w        # (S,)
    pooled_hdi = _az_hdi_compat(pooled_samples, hdi_prob=hdi_prob)
    pooled_mean = float(pooled_samples.mean())
    pooled_median = float(np.median(pooled_samples))
    pooled_sd = float(pooled_samples.std(ddof=1))
    prob_positive = float(np.mean(pooled_samples > 0))
    prob_rope = None
    if rope is not None:
        lo, hi = rope
        prob_rope = float(
            np.mean((pooled_samples > lo) & (pooled_samples < hi))
        )

    # rhat/ess on the vector of per-cohort τ's; report the worst case.
    try:
        rhat_series = az.rhat(trace, var_names=['tau_cohort'])[
            'tau_cohort'
        ].values
        rhat = float(np.nanmax(rhat_series))
    except Exception:
        rhat = float('nan')
    try:
        ess_series = az.ess(trace, var_names=['tau_cohort'])[
            'tau_cohort'
        ].values
        ess = float(np.nanmin(ess_series))
    except Exception:
        ess = float('nan')

    model_info['cohort_column'] = cohort
    model_info['n_cohorts'] = len(cohort_labels)
    model_info['cohort_weights'] = w.tolist()
    method = f"Bayesian DID ({shape_label}, {len(cohort_labels)} cohorts)"
    return BayesianDIDResult(
        method=method,
        estimand='ATT',
        posterior_mean=pooled_mean,
        posterior_median=pooled_median,
        posterior_sd=pooled_sd,
        hdi_lower=float(pooled_hdi[0]),
        hdi_upper=float(pooled_hdi[1]),
        prob_positive=prob_positive,
        prob_rope=prob_rope,
        rhat=rhat,
        ess=ess,
        n_obs=n,
        hdi_prob=hdi_prob,
        trace=trace,
        model_info=model_info,
        cohort_summaries=cohort_summaries,
        cohort_labels=list(cohort_labels),
    )
