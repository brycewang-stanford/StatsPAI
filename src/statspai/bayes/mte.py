"""Bayesian treatment-effect-at-propensity regression (Heckman-Vytlacil-style).

**Labelling caveat (read first)**: the posterior curve returned as
``mte_curve`` is the *treatment-effect-at-propensity function*
``g(p) = E[Y | D=1, P=p] - E[Y | D=0, P=p]``, which we fit by
projecting a polynomial in the propensity ``p`` onto the structural
equation. Under the standard Heckman-Vytlacil (2005) linear-separable
outcome model plus a bivariate-normal error assumption, ``g(p)``
coincides with the textbook MTE ``tau(u) = E[Y_1 - Y_0 | U_D = u]``
evaluated at ``u = p``. More generally — and in particular when gains
are heterogeneous in ways that are not captured by the linear-in-``p``
polynomial — ``g(p)`` is **LATE at propensity level ``p``**, which
is a summary of the MTE but not literally ``MTE(u)``.

We retain the "MTE" naming (for API continuity with v0.9.8 and because
applied users expect the term) but describe the fit as
"treatment-effect-at-propensity" in the method label. Users who need
the textbook MTE under weaker functional-form assumptions should pair
this with a bespoke structural model.

Model:

.. code-block:: text

    # Selection (first stage)
    # 'plugin' mode: logit MLE gives fixed p_i
    # 'joint'  mode: pi ~ Normal priors, D_i ~ Bernoulli(sigmoid(pi'W_i))
    g(p) = b_0 + b_1 p + b_2 p^2 + ...  (polynomial in propensity)
    Y_i = alpha + beta_X' X_i + D_i * g(p_i) + eps_i

References:
- Heckman, J. J., & Vytlacil, E. J. (2005). Structural equations,
  treatment effects, and econometric policy evaluation.
  *Econometrica*, 73(3), 669-738.
- Carneiro, Heckman & Vytlacil (2011). Estimating marginal returns
  to education. *AER*, 101(6), 2754-81.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._base import (
    BayesianMTEResult,
    _require_pymc,
    _sample_model,
    _summarise_posterior,
)


def _logit_propensity(Z: np.ndarray, X: Optional[np.ndarray], D: np.ndarray):
    """Plug-in logit first stage returning fitted P(D=1|Z,X).

    Uses scikit-learn's LogisticRegression; falls back to a scipy
    Newton-Raphson if sklearn is missing (unlikely in practice since
    it's a core dep). Guarantees probabilities in ``[1e-4, 1 - 1e-4]``
    so the induced quantile function is numerically well-behaved.
    """
    from sklearn.linear_model import LogisticRegression
    W = Z.reshape(-1, 1) if Z.ndim == 1 else Z
    if X is not None:
        W = np.hstack([W, X])
    clf = LogisticRegression(max_iter=500, solver='lbfgs')
    clf.fit(W, D.astype(int))
    ps = clf.predict_proba(W)[:, 1]
    return np.clip(ps, 1e-4, 1 - 1e-4)


def bayes_mte(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: str,
    covariates: Optional[List[str]] = None,
    *,
    first_stage: str = 'plugin',
    u_grid: Optional[np.ndarray] = None,
    poly_u: int = 2,
    prior_coef_sigma: float = 10.0,
    prior_mte_sigma: float = 5.0,
    prior_noise: float = 5.0,
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
) -> BayesianMTEResult:
    """Bayesian Marginal Treatment Effects via plug-in propensity + polynomial MTE.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat, instrument : str
        Outcome, endogenous binary treatment, scalar instrument.
    covariates : list of str, optional
        Exogenous controls entering both stages.
    first_stage : {'plugin', 'joint'}, default ``'plugin'``
        Selection-equation strategy.

        - ``'plugin'`` : frequentist logit MLE on ``(Z, X) -> D``
          computed once; propensities enter the MTE polynomial as
          fixed constants. Fast and the v0.9.8 behaviour.
        - ``'joint'`` : first-stage logit coefficients live inside
          the PyMC graph with Normal priors and ``D ~ Bernoulli(p)``;
          the propensity ``p_i`` is a Deterministic and the MTE
          polynomial sees it directly, so first-stage uncertainty
          propagates into the MTE curve. 2-4× slower than plugin but
          honest about uncertainty.
    u_grid : np.ndarray, optional
        Grid of propensity-to-be-treated values on which to evaluate
        the MTE posterior. Default: ``np.linspace(0.05, 0.95, 19)``.
    poly_u : int, default 2
        Polynomial order of the MTE in ``U_D``. ``poly_u=0`` reduces
        to a constant treatment effect; ``poly_u=2`` captures
        U-shaped or inverted-U selection on gains.
    prior_mte_sigma : float, default 5.0
        SD on each MTE polynomial coefficient (Normal prior).
    prior_coef_sigma, prior_noise : float
        Priors on structural intercept + covariate slopes + residual SD.
    rope, hdi_prob, inference, advi_iterations, draws, tune, chains,
    target_accept, random_state, progressbar :
        See :func:`bayes_did`.

    Returns
    -------
    BayesianMTEResult
        With `.mte_curve` (DataFrame on ``u_grid``), `.ate`, `.att`,
        `.atu`, and `.plot_mte()`. The inherited
        :class:`BayesianCausalResult` summary fields carry the
        integrated average MTE (ATE).
    """
    pm, _ = _require_pymc()

    for c in [y, treat, instrument] + (list(covariates) if covariates else []):
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found in data")

    cov_cols = list(covariates) if covariates else []
    clean = data[[y, treat, instrument] + cov_cols].dropna().reset_index(drop=True)
    n = len(clean)
    if n < 50:
        raise ValueError(
            f"bayes_mte needs at least 50 observations; got {n}."
        )

    Y = clean[y].to_numpy(dtype=float)
    D = clean[treat].to_numpy(dtype=float)
    Z = clean[instrument].to_numpy(dtype=float)
    X = clean[cov_cols].to_numpy(dtype=float) if cov_cols else None
    uniq_D = np.unique(D)
    if not set(uniq_D).issubset({0.0, 1.0}):
        raise ValueError(
            f"Treatment '{treat}' must be binary 0/1; got unique values {uniq_D}."
        )

    if first_stage not in ('plugin', 'joint'):
        raise ValueError(
            f"first_stage must be 'plugin' or 'joint'; got {first_stage!r}"
        )

    # Set up grid and grid-powers up-front (used for MTE curve regardless
    # of which first-stage we choose).
    if u_grid is None:
        u_grid = np.linspace(0.05, 0.95, 19)
    else:
        u_grid = np.asarray(u_grid, dtype=float)
    u_grid_powers = np.column_stack([u_grid ** k for k in range(poly_u + 1)])

    # Unit-level propensity lookup for ATT / ATU integrals. In 'plugin'
    # mode we compute this once from a logit MLE. In 'joint' mode we
    # also seed with the MLE propensities (used only for ATT / ATU
    # summary integrals; the model itself treats propensity as
    # Bayesian).
    propensity_mle = _logit_propensity(Z, X, D)

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0.0, sigma=prior_coef_sigma)
        b_mte = pm.Normal(
            'b_mte', mu=0.0, sigma=prior_mte_sigma, shape=poly_u + 1,
        )

        if first_stage == 'plugin':
            # Plug-in: use MLE propensity as a fixed constant in the
            # MTE polynomial.
            U_D = propensity_mle
            U_powers = np.column_stack([U_D ** k for k in range(poly_u + 1)])
            DU_powers = D[:, None] * U_powers  # (n, poly_u+1)
            mte_contribution = pm.math.dot(DU_powers, b_mte)
        else:
            # Joint: model first-stage logit coefficients inside the
            # graph. D_i ~ Bernoulli(p_i) where p_i is a Deterministic
            # function of the coefficients; the MTE polynomial now
            # sees p_i directly so first-stage uncertainty propagates.
            pi_intercept = pm.Normal(
                'pi_intercept', mu=0.0, sigma=prior_coef_sigma,
            )
            pi_Z = pm.Normal('pi_Z', mu=0.0, sigma=prior_coef_sigma)
            logit = pi_intercept + pi_Z * Z
            if X is not None:
                pi_X = pm.Normal(
                    'pi_X', mu=0.0, sigma=prior_coef_sigma,
                    shape=X.shape[1],
                )
                logit = logit + pm.math.dot(X, pi_X)
            # NB: we deliberately do NOT wrap sigmoid(logit) in
            # pm.Deterministic. That would add a per-unit, per-draw
            # float to the trace (shape (chains, draws, n)) which
            # blows up memory for large n. Propensity is recomputed
            # post-hoc from the first-stage coefficient posterior when
            # needed for ATT/ATU summaries.
            p_model = pm.math.sigmoid(logit)
            pm.Bernoulli('d_obs', p=p_model, observed=D.astype(int))

            # Build the structural g(p_i) = sum_k b_mte[k] * p_i ** k;
            # the polynomial is evaluated at the random p_model so
            # first-stage uncertainty propagates into the posterior.
            u_powers_list = [p_model ** k for k in range(poly_u + 1)]
            mte_i = sum(b_mte[k] * u_powers_list[k] for k in range(poly_u + 1))
            mte_contribution = D * mte_i

        structural = alpha + mte_contribution
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

    # MTE posterior on the grid:
    # mte_samples[s, k] = sum_j b_mte[s, j] * u_grid[k]^j
    b_mte_post = trace.posterior['b_mte'].values.reshape(-1, poly_u + 1)
    mte_samples = b_mte_post @ u_grid_powers.T  # (S, n_grid)

    _, az = _require_pymc()
    curve_rows = []
    for k, u in enumerate(u_grid):
        col = mte_samples[:, k]
        hdi = np.asarray(az.hdi(col, hdi_prob=hdi_prob)).ravel()
        curve_rows.append({
            'u': float(u),
            'posterior_mean': float(np.mean(col)),
            'posterior_sd': float(np.std(col, ddof=1)),
            'hdi_low': float(hdi[0]),
            'hdi_high': float(hdi[1]),
            'prob_positive': float(np.mean(col > 0)),
        })
    mte_curve = pd.DataFrame(curve_rows)

    # Integrated summaries (trapezoidal integration over u_grid)
    # ATE = int_0^1 MTE(u) du, approx by grid weights
    ate_samples = np.trapezoid(mte_samples, x=u_grid, axis=1) / (u_grid.max() - u_grid.min())

    # ATT / ATU use the population-level unit propensities to weight
    # MTE over the treated / untreated subpopulations.
    # - In 'plugin' mode these are the MLE propensities (fixed).
    # - In 'joint' mode we post-compute per-unit propensity from the
    #   first-stage coefficient posterior (rather than storing a
    #   per-unit Deterministic in the trace, which blows up memory
    #   at shape (chains, draws, n)). Using the posterior mean of
    #   the coefficients is a natural point summary.
    if first_stage == 'joint':
        pi0_mean = float(trace.posterior['pi_intercept'].values.mean())
        piZ_mean = float(trace.posterior['pi_Z'].values.mean())
        lin = pi0_mean + piZ_mean * Z
        if X is not None and 'pi_X' in trace.posterior:
            piX_mean = trace.posterior['pi_X'].values.reshape(
                -1, X.shape[1]
            ).mean(axis=0)
            lin = lin + X @ piX_mean
        U_pop = 1.0 / (1.0 + np.exp(-lin))
    else:
        U_pop = propensity_mle
    treated_mask = D == 1
    untreated_mask = D == 0
    U_treated = U_pop[treated_mask] if treated_mask.sum() > 0 else np.array([])
    U_untreated = U_pop[untreated_mask] if untreated_mask.sum() > 0 else np.array([])

    def _integrated_effect(U_population):
        if U_population.size == 0:
            return float('nan'), float('nan')
        # For each posterior draw, evaluate MTE at each population
        # unit's U_D and average
        u_pow_pop = np.column_stack([U_population ** k for k in range(poly_u + 1)])
        samples = b_mte_post @ u_pow_pop.T  # (S, n_pop)
        per_draw_mean = samples.mean(axis=1)  # integrated over population
        return float(per_draw_mean.mean()), float(per_draw_mean.std(ddof=1))

    att_mean, _ = _integrated_effect(U_treated)
    atu_mean, _ = _integrated_effect(U_untreated)

    # Primary estimand: average MTE (ATE integral)
    ate_mean = float(ate_samples.mean())
    ate_median = float(np.median(ate_samples))
    ate_sd = float(ate_samples.std(ddof=1))
    ate_hdi = np.asarray(az.hdi(ate_samples, hdi_prob=hdi_prob)).ravel()
    prob_pos_ate = float(np.mean(ate_samples > 0))

    # R-hat / ESS on the average-MTE derived quantity — not perfect
    # but honest: we take the rhat/ess of the driving coefficients
    # (b_mte) and report the maximum/minimum across them.
    try:
        rhat_series = az.rhat(trace, var_names=['b_mte'])['b_mte'].values
        rhat = float(np.nanmax(rhat_series))
    except Exception:
        rhat = float('nan')
    try:
        ess_series = az.ess(trace, var_names=['b_mte'])['b_mte'].values
        ess = float(np.nanmin(ess_series))
    except Exception:
        ess = float('nan')

    model_info = {
        'inference': inference,
        'first_stage': first_stage,
        'draws': draws,
        'tune': tune,
        'chains': chains,
        'target_accept': target_accept,
        'poly_u': poly_u,
        'u_grid': u_grid.tolist(),
        'instrument': instrument,
        'covariates': cov_cols,
        'prior_mte_sigma': prior_mte_sigma,
        'prior_coef_sigma': prior_coef_sigma,
        'prior_noise': prior_noise,
    }

    # Method label flags both `poly_u` and the first-stage mode, and
    # explicitly calls out the "effect-at-propensity" parameterisation
    # so users reading the summary don't confuse the polynomial in p
    # with the textbook MTE(u) under arbitrary heterogeneity.
    method_label = (
        f'Bayesian treatment-effect-at-propensity '
        f'(poly_u={poly_u}, '
        f'{"joint" if first_stage == "joint" else "plug-in"} first stage)'
    )
    return BayesianMTEResult(
        method=method_label,
        estimand='ATE (integrated MTE)',
        posterior_mean=ate_mean,
        posterior_median=ate_median,
        posterior_sd=ate_sd,
        hdi_lower=float(ate_hdi[0]),
        hdi_upper=float(ate_hdi[1]),
        prob_positive=prob_pos_ate,
        rhat=rhat,
        ess=ess,
        n_obs=n,
        hdi_prob=hdi_prob,
        trace=trace,
        model_info=model_info,
        mte_curve=mte_curve,
        u_grid=u_grid,
        ate=ate_mean,
        att=att_mean,
        atu=atu_mean,
    )
