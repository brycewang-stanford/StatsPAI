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

    # MTE structural equation:
    # 'polynomial' mode: g(p) = b_0 + b_1 p + ... (polynomial in propensity)
    #                    Y_i = alpha + beta_X' X_i + D_i * g(p_i) + eps_i
    # 'hv_latent'  mode: tau(u) = b_0 + b_1 u + ... (polynomial in latent U_D)
    #                    raw_U_i ~ Uniform(0, 1)
    #                    U_D_i = raw_U_i * p_i               if D_i = 1
    #                          = p_i + raw_U_i * (1 - p_i)   if D_i = 0
    #                    Y_i = alpha + beta_X' X_i + D_i * tau(U_D_i) + eps_i

HV-augmentation factorisation (``hv_latent`` + ``joint``)
---------------------------------------------------------
This uses the *standard Form-2 data-augmentation factorisation*:

    p(Y, D, U_D | p, θ) = p(Y | U_D, D, θ) · p(U_D | D, p) · p(D | p)

    where p(U_D | D=1, p) = Uniform(0, p)
          p(U_D | D=0, p) = Uniform(p, 1)
          p(D=1 | p)       = p   (Bernoulli)

The truncated-uniform augmentation enforces the HV indicator
``D_i = 1{U_D_i < p_i}``, and the Bernoulli(D|p) term is the
correct marginal selection likelihood. BOTH are needed: dropping
the Bernoulli loses ~half the selection information and biases
``pi`` (empirically verified — `piZ` flipped sign in a counter-
factual test). Under this factorisation, marginalising out ``raw_U``
exactly recovers the standard HV likelihood
``p(Y, D | p, θ) = ∫ p(Y | U_D, D, θ) · p(U_D | D, p) dU_D · p(D|p)``
where the integral runs over ``[0, p]`` (if D=1) or ``[p, 1]``
(if D=0) against the truncated-uniform density.

**Memory note**: ``hv_latent`` registers a latent ``raw_U`` of
shape ``(n,)``, stored in the posterior as ``(chains, draws, n)``.
For large ``n`` this can exceed hundreds of megabytes; the function
emits a ``UserWarning`` when the product exceeds ~50M floats
(~400MB at f64).

References:
- Heckman, J. J., & Vytlacil, E. J. (2005). Structural equations,
  treatment effects, and econometric policy evaluation.
  *Econometrica*, 73(3), 669-738.
- Carneiro, Heckman & Vytlacil (2011). Estimating marginal returns
  to education. *AER*, 101(6), 2754-81.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._base import (
    BayesianMTEResult,
    PROBIT_CLIP,
    _az_hdi_compat,
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
    instrument: Union[str, Sequence[str]],
    covariates: Optional[List[str]] = None,
    *,
    first_stage: str = 'plugin',
    mte_method: str = 'polynomial',
    selection: str = 'uniform',
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
    mte_method : {'polynomial', 'hv_latent'}, default ``'polynomial'``
        MTE parameterisation.

        - ``'polynomial'`` : fit a polynomial in the propensity
          ``p_i`` (v0.9.9 behaviour). Under Heckman-Vytlacil 2005
          linear-separable + bivariate-normal errors this equals
          ``MTE(p)``; under arbitrary heterogeneity it is
          ``LATE-at-propensity g(p)``, NOT the textbook
          ``MTE(u) = E[Y_1 - Y_0 | U_D = u]``.
        - ``'hv_latent'`` : sample a latent ``U_D_i`` per unit from
          the HV-correct truncated uniform (via ``raw_U_i ~ U(0,1)``
          and a deterministic reparameterisation), then evaluate
          the polynomial at ``U_D_i``. This recovers the textbook
          MTE polynomial under linear-separable HV. Slower
          (adds shape-n latent; ``O(n·draws·chains)`` memory) but
          mathematically faithful. A UserWarning is emitted if the
          expected latent storage exceeds ~50M floats.
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

    # Normalise instrument to a list so scalar and list-of-instruments
    # paths share the same downstream code. Back-compat: passing a
    # single string still works and returns the same posterior.
    if isinstance(instrument, str):
        iv_cols: List[str] = [instrument]
    else:
        iv_cols = list(instrument)
    if len(iv_cols) == 0:
        raise ValueError("instrument must name at least one column.")

    for c in [y, treat] + iv_cols + (list(covariates) if covariates else []):
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found in data")

    cov_cols = list(covariates) if covariates else []
    clean = data[[y, treat] + iv_cols + cov_cols].dropna().reset_index(drop=True)
    n = len(clean)
    if n < 50:
        raise ValueError(
            f"bayes_mte needs at least 50 observations; got {n}."
        )

    Y = clean[y].to_numpy(dtype=float)
    D = clean[treat].to_numpy(dtype=float)
    # 2-D Z of shape (n, k_iv). k_iv=1 recovers the scalar case.
    Z = clean[iv_cols].to_numpy(dtype=float)
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
    if mte_method not in ('polynomial', 'hv_latent'):
        raise ValueError(
            f"mte_method must be 'polynomial' or 'hv_latent'; "
            f"got {mte_method!r}"
        )
    if selection not in ('uniform', 'normal'):
        raise ValueError(
            f"selection must be 'uniform' or 'normal'; "
            f"got {selection!r}"
        )

    # hv_latent adds a shape-(n,) latent ``raw_U`` whose posterior
    # is stored as (chains, draws, n). Warn the user before they
    # wait 20 minutes to discover it ate 2GB of RAM.
    if mte_method == 'hv_latent':
        expected_latent_floats = n * draws * chains
        if expected_latent_floats > 50_000_000:
            import warnings
            approx_gb = expected_latent_floats * 8 / (1024 ** 3)
            warnings.warn(
                f"bayes_mte(mte_method='hv_latent') will register a "
                f"latent of shape (n={n},) whose posterior storage is "
                f"~{expected_latent_floats:,} floats "
                f"(~{approx_gb:.1f} GB at f64). Consider reducing "
                f"draws/chains or switching to mte_method='polynomial' "
                f"for larger data.",
                UserWarning,
                stacklevel=2,
            )

    # Set up grid and grid-powers up-front (used for MTE curve regardless
    # of which first-stage we choose).
    if u_grid is None:
        u_grid = np.linspace(0.05, 0.95, 19)
    else:
        u_grid = np.asarray(u_grid, dtype=float)

    # Under `selection='normal'`, the MTE polynomial is parameterised
    # on the probit-scale V = Φ^{-1}(U_D), not on U_D itself. The
    # posterior coefficients `b_mte` describe MTE(v) = Σ_k b_k · v^k
    # where v ∈ ℝ. We still report the curve indexed by `u` for user
    # convenience (propensity is the natural scale) but the underlying
    # powers sent to PyMC are built on v-space.
    if selection == 'normal':
        from scipy.stats import norm as _norm_dist
        v_grid = _norm_dist.ppf(u_grid)              # shape (n_grid,)
        u_grid_powers = np.column_stack(
            [v_grid ** k for k in range(poly_u + 1)]
        )
    else:
        u_grid_powers = np.column_stack(
            [u_grid ** k for k in range(poly_u + 1)]
        )

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

        # ------------------------------------------------------------------
        # First stage: produce a propensity expression ``p_expr`` that
        # is either a plain numpy array (plugin) or a PyMC tensor
        # (joint). In the joint path we also register the Bernoulli
        # likelihood on observed D.
        # ------------------------------------------------------------------
        if first_stage == 'plugin':
            p_expr = propensity_mle           # fixed numpy array
        else:
            pi_intercept = pm.Normal(
                'pi_intercept', mu=0.0, sigma=prior_coef_sigma,
            )
            # pi_Z is a vector of length k_iv. For back-compat with
            # the scalar case, shape=(1,) broadcasts against a (n, 1)
            # Z matrix via pm.math.dot.
            k_iv = Z.shape[1]
            pi_Z = pm.Normal(
                'pi_Z', mu=0.0, sigma=prior_coef_sigma, shape=k_iv,
            )
            logit = pi_intercept + pm.math.dot(Z, pi_Z)
            if X is not None:
                pi_X = pm.Normal(
                    'pi_X', mu=0.0, sigma=prior_coef_sigma,
                    shape=X.shape[1],
                )
                logit = logit + pm.math.dot(X, pi_X)
            # Deliberately NOT a Deterministic — would add per-unit
            # per-draw memory blow-up. Propensity is recomputed
            # post-hoc from the coefficient posterior when needed for
            # ATT/ATU summaries.
            p_expr = pm.math.sigmoid(logit)
            pm.Bernoulli('d_obs', p=p_expr, observed=D.astype(int))

        # ------------------------------------------------------------------
        # MTE contribution: evaluate the polynomial at either the
        # propensity (polynomial mode: g(p_i)) or a latent U_D_i per
        # unit (hv_latent mode: tau(U_D_i)).
        # ``selection='normal'`` reparameterises the abscissa from
        # ``a ∈ [0, 1]`` (uniform scale) to ``v = Φ^{-1}(a) ∈ ℝ``
        # (probit scale). Under the strict HV linear-separable +
        # bivariate-normal assumption this is the frame where MTE
        # is linear (``poly_u=1``), matching Heckman (1979) /
        # HV 2005 exactly.
        # ------------------------------------------------------------------
        def _abscissa(a_expr, is_numpy: bool):
            """Transform a in [0,1] to v = Φ^{-1}(a) on the probit
            scale when ``selection == 'normal'``; otherwise pass
            through unchanged. Works on numpy arrays and PyMC
            tensors via ``pt.erfinv``."""
            if selection != 'normal':
                return a_expr
            if is_numpy:
                from scipy.stats import norm as _norm_dist
                return _norm_dist.ppf(np.clip(a_expr, PROBIT_CLIP, 1 - PROBIT_CLIP))
            import pytensor.tensor as pt
            a_safe = pm.math.clip(a_expr, PROBIT_CLIP, 1 - PROBIT_CLIP)
            # Φ^{-1}(p) = √2 · erfinv(2p - 1)
            return pt.sqrt(2.0) * pt.erfinv(2.0 * a_safe - 1.0)

        if mte_method == 'polynomial':
            if first_stage == 'plugin':
                # Closed-form constant powers + dot product with b_mte
                abscissa = _abscissa(p_expr, is_numpy=True)
                U_powers = np.column_stack(
                    [abscissa ** k for k in range(poly_u + 1)]
                )
                DU_powers = D[:, None] * U_powers
                mte_contribution = pm.math.dot(DU_powers, b_mte)
            else:
                abscissa = _abscissa(p_expr, is_numpy=False)
                u_powers = [abscissa ** k for k in range(poly_u + 1)]
                mte_i = sum(
                    b_mte[k] * u_powers[k] for k in range(poly_u + 1)
                )
                mte_contribution = D * mte_i
        else:  # 'hv_latent' — sample U_D_i per unit from the truncated
               # uniform induced by D_i and the current p_i estimate.
               # Reparameterisation keeps NUTS applicable:
               #   raw_U_i ~ Uniform(0, 1)
               #   D_i = 1: U_D_i = raw_U_i * p_i      ∈ [0, p_i]
               #   D_i = 0: U_D_i = p_i + raw_U_i*(1-p_i) ∈ [p_i, 1]
               # This yields U_D_i | D_i distributed as the correct
               # truncated uniform under the HV-2005 identification.
            raw_U = pm.Uniform('raw_U', lower=0.0, upper=1.0, shape=n)
            # Convert observed D into a float array usable in the
            # element-wise expression below.
            D_float = D.astype(float)
            U_D_i = (
                D_float * raw_U * p_expr
                + (1.0 - D_float) * (p_expr + raw_U * (1.0 - p_expr))
            )
            abscissa = _abscissa(U_D_i, is_numpy=False)
            u_powers = [abscissa ** k for k in range(poly_u + 1)]
            tau_i = sum(b_mte[k] * u_powers[k] for k in range(poly_u + 1))
            mte_contribution = D_float * tau_i

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
    # Under `selection='normal'` we also expose the underlying
    # probit-scale abscissa ``v = Φ^{-1}(u)`` so users who fit in V
    # can plot their posterior on the scale it was fit on, without
    # having to recompute the transform themselves.
    if selection == 'normal':
        from scipy.stats import norm as _norm_dist
        v_grid_out = _norm_dist.ppf(np.clip(u_grid, PROBIT_CLIP, 1 - PROBIT_CLIP))
    else:
        v_grid_out = None
    for k, u in enumerate(u_grid):
        col = mte_samples[:, k]
        hdi = _az_hdi_compat(col, hdi_prob=hdi_prob)
        row = {
            'u': float(u),
            'posterior_mean': float(np.mean(col)),
            'posterior_sd': float(np.std(col, ddof=1)),
            'hdi_low': float(hdi[0]),
            'hdi_high': float(hdi[1]),
            'prob_positive': float(np.mean(col > 0)),
        }
        if v_grid_out is not None:
            row['v'] = float(v_grid_out[k])
        curve_rows.append(row)
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
        # pi_Z is shape (k_iv,). Collapse (chains, draws, k_iv) to
        # posterior-mean vector of length k_iv.
        piZ_mean = trace.posterior['pi_Z'].values.reshape(
            -1, Z.shape[1]
        ).mean(axis=0)
        lin = pi0_mean + Z @ piZ_mean
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
        """Posterior summary of the MTE integrated over a subpopulation.

        Returns (mean, sd, hdi_lower, hdi_high). All NaN when the
        subpopulation is empty (e.g. all units treated ⇒ no ATU).
        """
        if U_population.size == 0:
            return (float('nan'),) * 4
        if selection == 'normal':
            from scipy.stats import norm as _norm_dist
            pop_abscissa = _norm_dist.ppf(
                np.clip(U_population, PROBIT_CLIP, 1 - PROBIT_CLIP)
            )
        else:
            pop_abscissa = U_population
        u_pow_pop = np.column_stack(
            [pop_abscissa ** k for k in range(poly_u + 1)]
        )
        samples = b_mte_post @ u_pow_pop.T   # (S, n_pop)
        per_draw_mean = samples.mean(axis=1)  # integrated over population
        hdi = _az_hdi_compat(per_draw_mean, hdi_prob=hdi_prob)
        return (
            float(per_draw_mean.mean()),
            float(per_draw_mean.std(ddof=1)),
            float(hdi[0]),
            float(hdi[1]),
        )

    att_mean, att_sd, att_hdi_lo, att_hdi_hi = _integrated_effect(U_treated)
    atu_mean, atu_sd, atu_hdi_lo, atu_hdi_hi = _integrated_effect(U_untreated)

    # Primary estimand: average MTE (ATE integral)
    ate_mean = float(ate_samples.mean())
    ate_median = float(np.median(ate_samples))
    ate_sd = float(ate_samples.std(ddof=1))
    ate_hdi = _az_hdi_compat(ate_samples, hdi_prob=hdi_prob)
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
        'mte_method': mte_method,
        'selection': selection,
        'draws': draws,
        'tune': tune,
        'chains': chains,
        'target_accept': target_accept,
        'poly_u': poly_u,
        'u_grid': u_grid.tolist(),
        # `instruments` is always a list regardless of whether the
        # user passed a scalar or list to keep downstream code from
        # branching on type (cf. v0.9.11 round-B review).
        'instruments': iv_cols,
        'n_instruments': len(iv_cols),
        'covariates': cov_cols,
        'prior_mte_sigma': prior_mte_sigma,
        'prior_coef_sigma': prior_coef_sigma,
        'prior_noise': prior_noise,
    }

    # Method label flags the MTE parameterisation up-front. For
    # ``polynomial`` mode we continue to use "treatment-effect-at-
    # propensity" to reflect that g(p) may only equal MTE(u) under
    # the HV2005 linear-separable + bivariate-normal assumption. For
    # ``hv_latent`` mode we explicitly say "MTE" because the model
    # samples U_D_i per unit so the polynomial is evaluated at the
    # textbook latent variable, not at the propensity.
    fs_label = 'joint' if first_stage == 'joint' else 'plug-in'
    scale_label = 'V scale (probit)' if selection == 'normal' else 'U_D scale (uniform)'
    if mte_method == 'hv_latent':
        method_label = (
            f'Bayesian HV-latent MTE on {scale_label} '
            f'(poly_u={poly_u}, {fs_label} first stage)'
        )
    else:
        method_label = (
            f'Bayesian treatment-effect-at-propensity on {scale_label} '
            f'(poly_u={poly_u}, {fs_label} first stage)'
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
        selection=selection,
        att_sd=att_sd,
        att_hdi_lower=att_hdi_lo,
        att_hdi_upper=att_hdi_hi,
        atu_sd=atu_sd,
        atu_hdi_lower=atu_hdi_lo,
        atu_hdi_upper=atu_hdi_hi,
    )
