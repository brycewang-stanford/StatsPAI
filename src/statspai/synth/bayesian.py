"""
Bayesian Synthetic Control Method.

Places a Dirichlet prior on donor weights and uses Metropolis-Hastings
MCMC to sample from the joint posterior of (weights, sigma).  The
posterior distribution of treatment effects yields credible intervals
that incorporate both estimation uncertainty and model uncertainty —
avoiding the need for asymptotic approximations or permutation-based
inference.

Model
-----
Prior:      w ~ Dirichlet(alpha)
            sigma ~ InverseGamma(a0, b0)  [weakly informative]
Likelihood: Y1_pre | w, sigma ~ N(Y0_pre.T @ w, sigma^2 I)
Posterior:  p(w, sigma | Y1_pre) via MCMC
Effect:     tau_t = Y1_post_t - Y0_post_t.T @ w   for each posterior draw

References
----------
Vives, J. and Martinez, A. (2024). "Bayesian Synthetic Control Methods."
*Journal of Computational and Graphical Statistics*.

Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N. and Scott, S. L.
(2015). "Inferring causal impact using Bayesian structural time-series
models." *The Annals of Applied Statistics*, 9(1), 247-274. [@brodersen2015inferring]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.special import gammaln

from ..core.results import CausalResult


# ====================================================================== #
#  Public API
# ====================================================================== #


def bayesian_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit,
    treatment_time,
    covariates: Optional[List[str]] = None,
    n_iter: int = 2000,
    n_warmup: int = 1000,
    n_chains: int = 2,
    dirichlet_alpha: float = 1.0,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Bayesian Synthetic Control Method.

    Estimates the ATT by placing a Dirichlet prior on donor weights and
    sampling from the posterior via Metropolis-Hastings MCMC.  Returns
    full posterior credible intervals for the treatment effect.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format with columns for unit, time, and outcome.
    outcome : str
        Name of the outcome variable column.
    unit : str
        Name of the unit identifier column.
    time : str
        Name of the time period column.
    treated_unit : scalar
        Value in *unit* that identifies the treated unit.
    treatment_time : scalar
        First period of treatment (inclusive).
    covariates : list of str, optional
        Additional pre-treatment predictors to include in the matching
        objective.  Covariates are appended to the pre-treatment outcome
        series for each unit before fitting.
    n_iter : int, default 2000
        Total MCMC iterations per chain (including warmup).
    n_warmup : int, default 1000
        Number of warmup (burn-in) iterations for adaptation.  Must be
        strictly less than *n_iter*.
    n_chains : int, default 2
        Number of independent MCMC chains.  Multiple chains enable the
        R-hat convergence diagnostic.
    dirichlet_alpha : float, default 1.0
        Concentration parameter for the symmetric Dirichlet prior on
        donor weights.  ``alpha = 1`` gives a uniform prior on the
        simplex; values < 1 encourage sparsity; values > 1 encourage
        more uniform weights.
    seed : int, optional
        Random seed for reproducibility.
    alpha : float, default 0.05
        Significance level for credible intervals.

    Returns
    -------
    CausalResult
        With ``.estimate`` equal to the posterior mean ATT averaged over
        all post-treatment periods, ``.ci`` giving the equal-tailed
        credible interval, and rich diagnostics in ``model_info``.

    Raises
    ------
    ValueError
        If the panel has fewer than 2 pre-treatment periods, no
        post-treatment periods, or no valid donor units.

    Examples
    --------
    >>> result = sp.bayesian_synth(
    ...     df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ...     n_iter=4000, n_warmup=2000, n_chains=4, seed=42,
    ... )
    >>> print(result.summary())

    Notes
    -----
    The sampler uses a Dirichlet proposal on the simplex (re-normalised
    perturbation) with adaptive step-size tuning during warmup targeting
    an acceptance rate of ~0.35.  Samples are thinned by a factor of 2
    to reduce autocorrelation.

    References
    ----------
    Vives, J. and Martinez, A. (2024). "Bayesian Synthetic Control Methods."
    *Journal of Computational and Graphical Statistics*.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if n_warmup >= n_iter:
        raise ValueError(
            f"n_warmup ({n_warmup}) must be strictly less than n_iter ({n_iter})"
        )
    if n_chains < 1:
        raise ValueError("n_chains must be >= 1")
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")

    # ------------------------------------------------------------------
    # Reshape to wide panel
    # ------------------------------------------------------------------
    panel = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(panel.columns)
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError("Need at least 2 pre-treatment periods")
    if len(post_times) < 1:
        raise ValueError("Need at least 1 post-treatment period")

    Y1_pre = panel.loc[treated_unit, pre_times].values.astype(np.float64)
    Y1_post = panel.loc[treated_unit, post_times].values.astype(np.float64)

    donors = [u for u in panel.index if u != treated_unit]
    if len(donors) < 2:
        raise ValueError("Need at least 2 donor units")

    Y0_pre = panel.loc[donors, pre_times].values.astype(np.float64)   # (J, T0)
    Y0_post = panel.loc[donors, post_times].values.astype(np.float64)  # (J, T1)

    # ------------------------------------------------------------------
    # Optionally append covariate features to matching objective
    #
    # We standardise each covariate block by the outcome's pre-treatment
    # residual scale so that the Gaussian likelihood sigma^2 is estimated
    # primarily from outcome residuals rather than being dominated by
    # mismatch in covariate scale. Without this, the likelihood's
    # -0.5 * (T0+K) * log(2 pi sigma^2) term would artificially inflate
    # the penalty on sigma proportional to the number of covariate rows.
    # ------------------------------------------------------------------
    n_outcome_rows = len(pre_times)
    if covariates is not None and len(covariates) > 0:
        # Reference scale: pooled pre-treatment outcome SD (treated + donors).
        outcome_sd = float(
            np.std(
                np.concatenate([Y1_pre, Y0_pre.ravel()]), ddof=1
            )
        )
        outcome_sd = max(outcome_sd, 1e-12)

        for cov in covariates:
            cov_panel = data.pivot_table(index=unit, columns=time, values=cov)
            cov_treated = cov_panel.loc[
                treated_unit, pre_times
            ].values.astype(np.float64)
            cov_donors = cov_panel.loc[
                donors, pre_times
            ].values.astype(np.float64)
            # Z-score the covariate block, then rescale to outcome's SD.
            block = np.concatenate([cov_treated, cov_donors.ravel()])
            mu = float(np.mean(block))
            sd = max(float(np.std(block, ddof=1)), 1e-12)
            cov_treated = (cov_treated - mu) / sd * outcome_sd
            cov_donors = (cov_donors - mu) / sd * outcome_sd

            Y1_pre = np.concatenate([Y1_pre, cov_treated])
            Y0_pre = np.concatenate([Y0_pre, cov_donors], axis=1)

    J = len(donors)
    T0 = len(pre_times)         # outcome pre-periods (for post-estimate bookkeeping)
    T1 = len(post_times)

    # ------------------------------------------------------------------
    # Run MCMC
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    chain_seeds = rng.integers(0, 2**31, size=n_chains)

    all_w_samples: List[np.ndarray] = []
    all_sigma_samples: List[np.ndarray] = []
    acceptance_rates: List[float] = []

    for c in range(n_chains):
        w_chain, sigma_chain, acc_rate = _mcmc_sampler(
            Y1_pre=Y1_pre,
            Y0_pre=Y0_pre,
            n_iter=n_iter,
            n_warmup=n_warmup,
            dirichlet_alpha=dirichlet_alpha,
            seed=int(chain_seeds[c]),
        )
        all_w_samples.append(w_chain)
        all_sigma_samples.append(sigma_chain)
        acceptance_rates.append(acc_rate)

    # Stack chains: each is (n_post_warmup/thin, J) or (n_post_warmup/thin,)
    w_samples = np.concatenate(all_w_samples, axis=0)      # (S, J)
    sigma_samples = np.concatenate(all_sigma_samples, axis=0)  # (S,)
    S = w_samples.shape[0]

    # ------------------------------------------------------------------
    # Posterior counterfactual and effects
    # ------------------------------------------------------------------
    # Y_synth_post[s, t] = Y0_post[:, t].T @ w_samples[s]
    posterior_counterfactual = w_samples @ Y0_post  # (S, T1)
    effects_posterior = (
        Y1_post[np.newaxis, :] - posterior_counterfactual
    )  # (S, T1)

    # Period-level summaries
    effect_mean_by_period = np.mean(effects_posterior, axis=0)  # (T1,)
    effect_sd_by_period = np.std(effects_posterior, axis=0, ddof=1)
    lo = alpha / 2
    hi = 1 - alpha / 2
    ci_lower_by_period = np.quantile(effects_posterior, lo, axis=0)
    ci_upper_by_period = np.quantile(effects_posterior, hi, axis=0)

    # ATT: average over post-treatment periods for each draw, then summarise
    att_draws = np.mean(effects_posterior, axis=1)  # (S,)
    att = float(np.mean(att_draws))
    att_se = float(np.std(att_draws, ddof=1))
    att_ci = (
        float(np.quantile(att_draws, lo)),
        float(np.quantile(att_draws, hi)),
    )

    # Posterior probability that effect > 0
    prob_positive = float(np.mean(att_draws > 0))
    # Two-sided Bayesian p-value: 2 * min(P(tau > 0), P(tau < 0))
    pvalue = float(2 * min(prob_positive, 1 - prob_positive))
    pvalue = max(pvalue, 1 / (S + 1))  # floor

    # ------------------------------------------------------------------
    # Convergence diagnostics
    # ------------------------------------------------------------------
    rhat_weights = _compute_rhat(all_w_samples)   # (J,)
    rhat_sigma = _compute_rhat_scalar(all_sigma_samples)
    n_eff_weights = _compute_neff(all_w_samples)   # (J,)

    # ------------------------------------------------------------------
    # Pre-treatment fit (posterior mean weights)
    # ------------------------------------------------------------------
    w_mean = np.mean(w_samples, axis=0)
    w_sd = np.std(w_samples, axis=0, ddof=1)
    Y1_hat_pre = Y0_pre.T @ w_mean  # (T0,) — using original T0 columns
    pre_residuals = Y1_pre[:T0] - Y1_hat_pre[:T0]
    pre_rmspe = float(np.sqrt(np.mean(pre_residuals**2)))

    # ------------------------------------------------------------------
    # Build output DataFrame
    # ------------------------------------------------------------------
    counterfactual_mean = np.mean(posterior_counterfactual, axis=0)
    effects_df = pd.DataFrame(
        {
            "time": post_times,
            "treated": Y1_post,
            "counterfactual": counterfactual_mean,
            "effect": effect_mean_by_period,
            "effect_sd": effect_sd_by_period,
            "ci_lower": ci_lower_by_period,
            "ci_upper": ci_upper_by_period,
        }
    )

    # Credible intervals by period as a list of tuples
    credible_intervals_by_period = list(
        zip(
            post_times,
            ci_lower_by_period.tolist(),
            ci_upper_by_period.tolist(),
        )
    )

    return CausalResult(
        method="Bayesian Synthetic Control",
        estimand="ATT",
        estimate=att,
        se=att_se,
        pvalue=pvalue,
        ci=att_ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info={
            "model_type": "Bayesian Synthetic Control (MCMC)",
            # Weights diagnostics
            "weights_posterior_mean": dict(zip(donors, w_mean.tolist())),
            "weights_posterior_sd": dict(zip(donors, w_sd.tolist())),
            "sigma_posterior_mean": float(np.mean(sigma_samples)),
            "sigma_posterior_sd": float(np.std(sigma_samples, ddof=1)),
            # Posterior draws (for downstream analysis)
            "posterior_draws": posterior_counterfactual,
            "effects_posterior": effects_posterior,
            "att_draws": att_draws,
            # Convergence
            "rhat": dict(zip(donors, rhat_weights.tolist())),
            "rhat_sigma": rhat_sigma,
            "n_eff": dict(zip(donors, n_eff_weights.tolist())),
            # Fit
            "pre_rmspe": pre_rmspe,
            "post_rmspe": float(
                np.sqrt(np.mean(effect_mean_by_period**2))
            ),
            # Panel dimensions
            "n_donors": J,
            "n_pre_periods": T0,
            "n_post_periods": T1,
            "n_mcmc_samples": S,
            "n_chains": n_chains,
            "acceptance_rates": acceptance_rates,
            "dirichlet_alpha": dirichlet_alpha,
            # Inference
            "posterior_prob_positive": prob_positive,
            "credible_intervals_by_period": credible_intervals_by_period,
            "effects_by_period": effects_df,
        },
    )


# ====================================================================== #
#  MCMC internals
# ====================================================================== #


def _log_posterior(
    w: np.ndarray,
    sigma: float,
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    dirichlet_alpha: float,
) -> float:
    """
    Compute the (unnormalised) log posterior for weights *w* and noise
    scale *sigma*.

    Components
    ----------
    1.  Log-likelihood: Y1_pre ~ N(Y0_pre.T @ w, sigma^2 I)
    2.  Log-prior on w: Dirichlet(alpha, ..., alpha)
    3.  Log-prior on sigma: InverseGamma(a0=1, b0=1)  [weakly informative]

    Parameters
    ----------
    w : np.ndarray, shape (J,)
        Donor weights on the simplex.
    sigma : float
        Noise standard deviation (must be > 0).
    Y1_pre : np.ndarray, shape (T0,)
        Treated unit pre-treatment outcomes.
    Y0_pre : np.ndarray, shape (J, T0)
        Donor matrix of pre-treatment outcomes.
    dirichlet_alpha : float
        Symmetric Dirichlet concentration parameter.

    Returns
    -------
    float
        Log-posterior (up to a constant).
    """
    # Guard against invalid parameter values
    if sigma <= 0 or np.any(w < 0):
        return -np.inf

    T0 = len(Y1_pre)
    J = len(w)

    # --- Log-likelihood ---
    residual = Y1_pre - Y0_pre.T @ w  # (T0,)
    ll = -0.5 * T0 * np.log(2 * np.pi * sigma**2) - np.sum(residual**2) / (
        2 * sigma**2
    )

    # --- Log-prior: Dirichlet ---
    # log p(w | alpha) = log C(alpha) + sum (alpha - 1) log(w_j)
    # We drop the normalising constant (it cancels in MH ratio)
    lp_w = np.sum((dirichlet_alpha - 1) * np.log(np.maximum(w, 1e-300)))

    # --- Log-prior: InverseGamma(a0=1, b0=1) on sigma ---
    a0, b0 = 1.0, 1.0
    lp_sigma = -(a0 + 1) * np.log(sigma) - b0 / sigma

    return ll + lp_w + lp_sigma


def _dirichlet_proposal(
    w_current: np.ndarray,
    step_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Propose new weights on the simplex by concentrating a Dirichlet
    around the current point.

    The proposal is Dirichlet(w_current * step_size + 1), which for
    large *step_size* concentrates tightly around *w_current* and for
    small *step_size* explores more broadly.

    Parameters
    ----------
    w_current : np.ndarray, shape (J,)
        Current weight vector (must be on the simplex).
    step_size : float
        Concentration multiplier — larger means smaller steps.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray, shape (J,)
        Proposed weight vector on the simplex.
    """
    conc = w_current * step_size + 1.0  # ensure > 0 everywhere
    return rng.dirichlet(conc)


def _log_dirichlet_proposal_density(
    w_proposed: np.ndarray,
    w_current: np.ndarray,
    step_size: float,
) -> float:
    """
    Log density of the Dirichlet proposal q(w_proposed | w_current).

    Needed for the Metropolis-Hastings acceptance ratio because the
    Dirichlet proposal is *not* symmetric.

    Parameters
    ----------
    w_proposed : np.ndarray, shape (J,)
    w_current : np.ndarray, shape (J,)
    step_size : float

    Returns
    -------
    float
        Log q(w_proposed | w_current).
    """
    conc = w_current * step_size + 1.0
    # Dirichlet log-density (up to normalising constant that does NOT cancel)
    log_norm = gammaln(np.sum(conc)) - np.sum(gammaln(conc))
    log_kernel = np.sum((conc - 1.0) * np.log(np.maximum(w_proposed, 1e-300)))
    return log_norm + log_kernel


def _mcmc_sampler(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    n_iter: int,
    n_warmup: int,
    dirichlet_alpha: float,
    seed: int,
    thin: int = 2,
    target_accept: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Single-chain Metropolis-Hastings sampler for (w, sigma).

    Adaptation
    ----------
    During warmup the Dirichlet concentration step_size and the sigma
    proposal scale are tuned every 50 iterations to approach
    *target_accept*.

    Parameters
    ----------
    Y1_pre : np.ndarray, shape (T0,) or (T0 + n_cov * T0,)
        Treated unit's matching vector.
    Y0_pre : np.ndarray, shape (J, T0) or (J, T0 + n_cov * T0)
        Donor matching matrix.
    n_iter : int
        Total iterations (warmup + sampling).
    n_warmup : int
        Warmup iterations.
    dirichlet_alpha : float
        Dirichlet prior concentration.
    seed : int
        RNG seed.
    thin : int, default 2
        Thinning interval for post-warmup draws.
    target_accept : float, default 0.35
        Target MH acceptance rate for step-size adaptation.

    Returns
    -------
    w_samples : np.ndarray, shape (n_kept, J)
        Post-warmup, thinned weight draws.
    sigma_samples : np.ndarray, shape (n_kept,)
        Post-warmup, thinned sigma draws.
    acceptance_rate : float
        Overall post-warmup acceptance rate.
    """
    rng = np.random.default_rng(seed)
    J = Y0_pre.shape[0]

    # --- Initialise ---
    w = np.ones(J) / J
    sigma = float(np.std(Y1_pre - Y0_pre.T @ w)) + 0.1

    # Adaptation parameters
    w_step = float(J * 10)       # initial Dirichlet concentration step
    sigma_step = 0.2 * sigma     # initial sigma proposal scale
    adapt_interval = 50

    # Storage
    n_post = n_iter - n_warmup
    max_samples = n_post // thin + 1
    w_store = np.empty((max_samples, J), dtype=np.float64)
    sigma_store = np.empty(max_samples, dtype=np.float64)

    current_lp = _log_posterior(w, sigma, Y1_pre, Y0_pre, dirichlet_alpha)

    n_accept_warmup = 0
    n_accept_post = 0
    sample_idx = 0

    for it in range(n_iter):
        is_warmup = it < n_warmup

        # ----- Propose new w -----
        w_prop = _dirichlet_proposal(w, w_step, rng)

        # ----- Propose new sigma (log-normal random walk) -----
        log_sigma_prop = np.log(sigma) + rng.normal(0, sigma_step)
        sigma_prop = np.exp(log_sigma_prop)

        # ----- MH acceptance -----
        prop_lp = _log_posterior(
            w_prop, sigma_prop, Y1_pre, Y0_pre, dirichlet_alpha
        )

        # Proposal ratio:
        # - Dirichlet is asymmetric: need explicit q ratio.
        # - Log-normal random walk on sigma: proposal is symmetric in
        #   log-space but target is on sigma, so Jacobian gives
        #   log q(sigma_prop|sigma) - log q(sigma|sigma_prop)
        #     = -log sigma_prop - (-log sigma) = log(sigma / sigma_prop),
        #   and the reverse-minus-forward contribution is
        #   log(sigma_prop / sigma).
        log_q_forward = _log_dirichlet_proposal_density(w_prop, w, w_step)
        log_q_reverse = _log_dirichlet_proposal_density(w, w_prop, w_step)
        log_jac_sigma = np.log(sigma_prop) - np.log(sigma)

        log_alpha_mh = (
            prop_lp - current_lp
            + log_q_reverse - log_q_forward
            + log_jac_sigma
        )

        if np.log(rng.uniform()) < log_alpha_mh:
            w = w_prop
            sigma = sigma_prop
            current_lp = prop_lp
            if is_warmup:
                n_accept_warmup += 1
            else:
                n_accept_post += 1

        # ----- Adaptation during warmup -----
        if is_warmup and (it + 1) % adapt_interval == 0 and it > 0:
            recent_rate = n_accept_warmup / (it + 1)
            # Increase step_size (tighter proposals) if accepting too much
            # Decrease step_size (wider proposals) if accepting too little
            if recent_rate > target_accept + 0.05:
                w_step *= 1.3
                sigma_step *= 0.85
            elif recent_rate < target_accept - 0.05:
                w_step *= 0.7
                sigma_step *= 1.15
            # Clamp to reasonable range
            w_step = np.clip(w_step, 1.0, J * 500)
            sigma_step = np.clip(sigma_step, 1e-4, 2.0 * sigma + 0.1)

        # ----- Store post-warmup samples -----
        if not is_warmup and (it - n_warmup) % thin == 0:
            if sample_idx < max_samples:
                w_store[sample_idx] = w
                sigma_store[sample_idx] = sigma
                sample_idx += 1

    w_store = w_store[:sample_idx]
    sigma_store = sigma_store[:sample_idx]
    acceptance_rate = n_accept_post / max(n_post, 1)

    return w_store, sigma_store, acceptance_rate


# ====================================================================== #
#  Convergence diagnostics
# ====================================================================== #


def _compute_rhat(chain_samples: List[np.ndarray]) -> np.ndarray:
    """
    Compute the split-R-hat convergence diagnostic for each weight
    dimension across chains.

    Uses the Gelman-Rubin potential scale reduction factor.  Values
    close to 1.0 indicate convergence; values > 1.05 suggest the
    chains have not mixed.

    Parameters
    ----------
    chain_samples : list of np.ndarray
        Each element has shape (n_samples, J).

    Returns
    -------
    np.ndarray, shape (J,)
        R-hat for each donor weight dimension.
    """
    if len(chain_samples) < 2:
        # Cannot compute R-hat with a single chain; return NaN
        J = chain_samples[0].shape[1]
        return np.full(J, np.nan)

    # Split each chain in half for split-R-hat
    splits = []
    for ch in chain_samples:
        n = ch.shape[0]
        mid = n // 2
        if mid > 0:
            splits.append(ch[:mid])
            splits.append(ch[mid:])

    M = len(splits)
    if M < 2:
        return np.full(chain_samples[0].shape[1], np.nan)

    N = min(s.shape[0] for s in splits)
    splits = [s[:N] for s in splits]

    J = splits[0].shape[1]
    rhat = np.empty(J)

    for j in range(J):
        chain_means = np.array([np.mean(s[:, j]) for s in splits])
        chain_vars = np.array([np.var(s[:, j], ddof=1) for s in splits])

        grand_mean = np.mean(chain_means)
        B = N * np.var(chain_means, ddof=1)  # between-chain variance
        W = np.mean(chain_vars)              # within-chain variance

        if W < 1e-30:
            rhat[j] = 1.0
        else:
            var_hat = (1 - 1 / N) * W + B / N
            rhat[j] = np.sqrt(var_hat / W)

    return rhat


def _compute_rhat_scalar(chain_samples: List[np.ndarray]) -> float:
    """
    R-hat for a scalar parameter (sigma) across chains.

    Parameters
    ----------
    chain_samples : list of np.ndarray
        Each element has shape (n_samples,).

    Returns
    -------
    float
        Split R-hat.
    """
    reshaped = [s.reshape(-1, 1) for s in chain_samples]
    rhat_arr = _compute_rhat(reshaped)
    return float(rhat_arr[0])


def _compute_neff(chain_samples: List[np.ndarray]) -> np.ndarray:
    """
    Estimate the effective sample size (ESS) for each weight dimension
    using the initial monotone sequence estimator.

    Parameters
    ----------
    chain_samples : list of np.ndarray
        Each element has shape (n_samples, J).

    Returns
    -------
    np.ndarray, shape (J,)
        Effective sample size per dimension.
    """
    combined = np.concatenate(chain_samples, axis=0)  # (S_total, J)
    S, J = combined.shape
    n_eff = np.empty(J)

    for j in range(J):
        x = combined[:, j]
        x_mean = np.mean(x)
        x_centered = x - x_mean
        var_x = np.var(x, ddof=1)

        if var_x < 1e-30:
            n_eff[j] = float(S)
            continue

        # Autocorrelation via FFT
        max_lag = min(S - 1, 500)
        acf = _autocorr_fft(x_centered, max_lag)

        # Initial positive sequence estimator
        # Sum pairs of consecutive autocorrelations; stop when sum < 0
        tau = 1.0
        for k in range(1, max_lag - 1, 2):
            pair_sum = acf[k] + acf[k + 1] if k + 1 < max_lag else acf[k]
            if pair_sum < 0:
                break
            tau += 2.0 * pair_sum

        n_eff[j] = max(1.0, S / tau)

    return n_eff


def _autocorr_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Normalised autocorrelation function via FFT.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Zero-mean series.
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    np.ndarray, shape (max_lag,)
        Normalised autocorrelation at lags 0 .. max_lag-1.
    """
    N = len(x)
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2

    fft_x = np.fft.rfft(x, n=fft_size)
    acf_raw = np.fft.irfft(fft_x * np.conj(fft_x), n=fft_size)[:N]

    if acf_raw[0] < 1e-30:
        return np.zeros(max_lag)

    acf_norm = acf_raw / acf_raw[0]
    return acf_norm[:max_lag]
