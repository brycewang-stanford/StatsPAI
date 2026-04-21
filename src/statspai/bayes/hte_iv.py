"""Heterogeneous-effect Bayesian IV.

Extends :func:`bayes_iv` with a linear CATE-by-covariate model:

.. code-block:: text

    D_i = pi_0 + pi_Z' Z_i + pi_X' X_i + v_i                        (first stage)
    tau(M_i) = tau_0 + tau_hte' (M_i - M_bar)                       (CATE)
    Y_i = alpha + tau(M_i) * D_i + beta_X' X_i + rho * v_hat_i + eps_i   (structural)

The posterior gives the average LATE (``tau_0``) and a table of
slopes on each effect modifier. ``prob_positive`` on any individual
slope answers "is the CATE systematically higher when this variable
is higher?".
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._base import (
    BayesianHTEIVResult,
    _az_hdi_compat,
    _require_pymc,
    _sample_model,
    _summarise_posterior,
)


def _prepare_hte_iv_frame(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: Union[str, Sequence[str]],
    effect_modifiers: Sequence[str],
    covariates: Optional[List[str]],
) -> dict:
    if isinstance(instrument, str):
        iv_cols = [instrument]
    else:
        iv_cols = list(instrument)
    mod_cols = list(effect_modifiers)
    cov_cols = list(covariates) if covariates else []

    for c in [y, treat] + iv_cols + mod_cols + cov_cols:
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found in data")
    if not mod_cols:
        raise ValueError("effect_modifiers must list at least one covariate.")

    # Reject overlap between effect_modifiers and covariates. If the
    # same column appears in both, the structural equation at fit
    # time includes it as (M - M_bar) * D *and* as plain beta_X * X,
    # while the first-stage plug-in v_hat only includes it through X.
    # That asymmetry is confusing and near-certain to be a user bug.
    # Force the caller to pick a lane.
    overlap = set(mod_cols) & set(cov_cols)
    if overlap:
        raise ValueError(
            f"effect_modifiers and covariates must not overlap. "
            f"Columns in both: {sorted(overlap)}. Pass the column as "
            "one role only — either as an effect modifier (the LATE "
            "slopes on it) or as a control (it enters both stages "
            "but does not modify the LATE)."
        )

    all_cols = [y, treat] + iv_cols + mod_cols + cov_cols
    # Deduplicate while preserving order (modifiers may overlap with
    # covariates; that's fine, we just don't want NA-drop redundancy)
    seen = set()
    uniq_cols = [c for c in all_cols if not (c in seen or seen.add(c))]
    clean = data[uniq_cols].dropna().reset_index(drop=True)
    n = len(clean)
    if n < 40:
        raise ValueError(
            f"bayes_hte_iv needs at least 40 observations; got {n}."
        )

    Y = clean[y].to_numpy(dtype=float)
    D = clean[treat].to_numpy(dtype=float)
    Z = clean[iv_cols].to_numpy(dtype=float)
    M = clean[mod_cols].to_numpy(dtype=float)
    X = clean[cov_cols].to_numpy(dtype=float) if cov_cols else None
    return {
        'n': n, 'Y': Y, 'D': D, 'Z': Z, 'M': M, 'X': X,
        'iv_cols': iv_cols, 'mod_cols': mod_cols, 'cov_cols': cov_cols,
    }


def bayes_hte_iv(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: Union[str, Sequence[str]],
    effect_modifiers: Sequence[str],
    covariates: Optional[List[str]] = None,
    *,
    prior_late: Tuple[float, float] = (0.0, 10.0),
    prior_hte_sigma: float = 5.0,
    prior_coef_sigma: float = 10.0,
    prior_noise: float = 5.0,
    inference: str = 'nuts',
    advi_iterations: int = 20000,
    rope: Optional[Tuple[float, float]] = None,
    hdi_prob: float = 0.95,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_state: int = 42,
    progressbar: bool = False,
) -> BayesianHTEIVResult:
    """Bayesian IV with linear CATE-by-covariate heterogeneity.

    Parameters
    ----------
    data, y, treat, instrument, covariates :
        Same semantics as :func:`bayes_iv`.
    effect_modifiers : sequence of str
        Covariates whose linear interaction with ``treat`` is the
        heterogeneity signal. The model centres these at their sample
        mean so ``tau_0`` is interpretable as the average LATE at
        ``modifiers = mean``.
    prior_hte_sigma : float
        Prior SD on each element of ``tau_hte`` (the slope of LATE on
        the corresponding modifier).
    inference : {'nuts', 'advi'}, default 'nuts'
        Sampler. ADVI is a fast mean-field approximation; see the
        module-level caveat.
    advi_iterations : int, default 20000
        ADVI iterations (ignored for NUTS).
    prior_late, prior_coef_sigma, prior_noise, rope, hdi_prob, draws,
    tune, chains, target_accept, random_state, progressbar : see
        :func:`bayes_did`.

    Returns
    -------
    BayesianHTEIVResult
        `BayesianCausalResult` with a `cate_slopes` DataFrame and
        `predict_cate(values)` method.
    """
    pm, _ = _require_pymc()
    prep = _prepare_hte_iv_frame(
        data, y, treat, instrument, effect_modifiers, covariates,
    )
    n = prep['n']
    Y = prep['Y']
    D = prep['D']
    Z = prep['Z']
    M = prep['M']
    X = prep['X']
    mod_cols = prep['mod_cols']
    n_instr = Z.shape[1]
    n_mod = M.shape[1]

    # Pre-compute first-stage residuals (control function plug-in,
    # same shortcut as bayes_iv).
    W_fs = [np.ones((n, 1)), Z]
    if X is not None:
        W_fs.append(X)
    W_fs_mat = np.hstack(W_fs)
    pi_ols, *_ = np.linalg.lstsq(W_fs_mat, D, rcond=None)
    v_hat = D - W_fs_mat @ pi_ols

    modifier_means = M.mean(axis=0)
    M_centered = M - modifier_means

    mu_late, sigma_late = prior_late

    with pm.Model() as model:
        # First stage (model for pi_Z so we can report identification strength)
        pi_intercept = pm.Normal('pi_intercept', mu=0.0, sigma=prior_coef_sigma)
        pi_Z = pm.Normal(
            'pi_Z', mu=0.0, sigma=prior_coef_sigma, shape=n_instr,
        )
        first_stage = pi_intercept + pm.math.dot(Z, pi_Z)
        if X is not None:
            pi_X = pm.Normal(
                'pi_X', mu=0.0, sigma=prior_coef_sigma, shape=X.shape[1],
            )
            first_stage = first_stage + pm.math.dot(X, pi_X)
        sigma_v = pm.HalfNormal('sigma_v', sigma=prior_noise)
        pm.Normal('d_obs', mu=first_stage, sigma=sigma_v, observed=D)

        # Heterogeneous structural equation
        alpha = pm.Normal('alpha', mu=0.0, sigma=prior_coef_sigma)
        tau_0 = pm.Normal('tau_0', mu=mu_late, sigma=sigma_late)
        tau_hte = pm.Normal(
            'tau_hte', mu=0.0, sigma=prior_hte_sigma, shape=n_mod,
        )
        # tau(M_i) = tau_0 + tau_hte' (M_i - M_bar)
        tau_i = tau_0 + pm.math.dot(M_centered, tau_hte)
        rho = pm.Normal('rho_cf', mu=0.0, sigma=prior_coef_sigma)
        structural = alpha + tau_i * D + rho * v_hat
        if X is not None:
            beta_X = pm.Normal(
                'beta_X', mu=0.0, sigma=prior_coef_sigma, shape=X.shape[1],
            )
            structural = structural + pm.math.dot(X, beta_X)
        sigma_eps = pm.HalfNormal('sigma_eps', sigma=prior_noise)
        pm.Normal('y_obs', mu=structural, sigma=sigma_eps, observed=Y)

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

    # Primary estimand: tau_0 (average LATE at modifier means)
    summary = _summarise_posterior(
        trace, 'tau_0', hdi_prob=hdi_prob, rope=rope,
    )

    # Build the CATE-slopes DataFrame from tau_hte
    tau_hte_post = trace.posterior['tau_hte'].values  # (chains, draws, n_mod)
    flat = tau_hte_post.reshape(-1, n_mod)
    slope_rows = []
    _, az = _require_pymc()
    for k, name in enumerate(mod_cols):
        col = flat[:, k]
        hdi = _az_hdi_compat(col, hdi_prob=hdi_prob)
        slope_rows.append({
            'term': name,
            'estimate': float(np.mean(col)),
            'std_error': float(np.std(col, ddof=1)),
            'hdi_low': float(hdi[0]),
            'hdi_high': float(hdi[1]),
            'prob_positive': float(np.mean(col > 0)),
        })
    cate_slopes = pd.DataFrame(slope_rows)

    model_info = {
        'inference': inference,
        'draws': draws,
        'tune': tune,
        'chains': chains,
        'target_accept': target_accept,
        'prior_late': prior_late,
        'prior_hte_sigma': prior_hte_sigma,
        'prior_coef_sigma': prior_coef_sigma,
        'prior_noise': prior_noise,
        'instruments': prep['iv_cols'],
        'effect_modifiers': mod_cols,
        'covariates': prep['cov_cols'],
        'modifier_means': modifier_means.tolist(),
    }

    result = BayesianHTEIVResult(
        method=(
            f"Bayesian HTE-IV "
            f"({n_instr} instrument{'s' if n_instr > 1 else ''}, "
            f"{n_mod} modifier{'s' if n_mod > 1 else ''})"
        ),
        estimand='LATE (avg)',
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
        cate_slopes=cate_slopes,
        effect_modifiers=list(mod_cols),
        _modifier_means=modifier_means,
    )
    return result
