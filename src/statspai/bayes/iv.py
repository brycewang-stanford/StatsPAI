"""Bayesian linear instrumental-variable estimation via PyMC.

Jointly models the first stage (``D ~ Z + X``) and the structural
equation (``Y ~ D + X``) with a bivariate-Normal error structure so
the posterior over the LATE prices endogeneity + weak-instrument risk
automatically.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd

from ._base import BayesianIVResult, _require_pymc, _sample_model, _summarise_posterior


def _prepare_iv_frame(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: Union[str, Sequence[str]],
    covariates: Optional[List[str]],
) -> dict:
    """Validate, drop NA, extract arrays for Bayesian IV."""
    if isinstance(instrument, str):
        iv_cols = [instrument]
    else:
        iv_cols = list(instrument)

    for c in [y, treat] + iv_cols:
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found in data")

    cov_cols = list(covariates) if covariates else []
    for c in cov_cols:
        if c not in data.columns:
            raise ValueError(f"Covariate '{c}' not found in data")

    all_cols = [y, treat] + iv_cols + cov_cols
    clean = data[all_cols].dropna().reset_index(drop=True)
    n = len(clean)
    if n < 30:
        raise ValueError(
            f"Bayesian IV needs at least 30 observations after dropping NA, "
            f"got {n}."
        )

    Y = clean[y].to_numpy(dtype=float)
    D = clean[treat].to_numpy(dtype=float)
    Z = clean[iv_cols].to_numpy(dtype=float)
    X = clean[cov_cols].to_numpy(dtype=float) if cov_cols else None

    return {
        "n": n,
        "Y": Y,
        "D": D,
        "Z": Z,
        "X": X,
        "iv_cols": iv_cols,
        "cov_cols": cov_cols,
    }


def bayes_iv(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: Union[str, Sequence[str]],
    covariates: Optional[List[str]] = None,
    *,
    per_instrument: bool = False,
    prior_late: Tuple[float, float] = (0.0, 10.0),
    prior_first_stage_sigma: float = 5.0,
    prior_coef_sigma: float = 10.0,
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
) -> BayesianIVResult:
    """Bayesian linear IV via jointly-modelled first stage + structural equation.

    The model:

    .. code-block:: text

        D_i = pi_0 + pi_Z' * Z_i + pi_X' * X_i + v_i
        Y_i = alpha + LATE * D_i + beta_X' * X_i + eps_i
        (v_i, eps_i) ~ BivariateNormal(0, Sigma)

    ``Sigma`` is parameterised through the regression of ``eps`` on
    ``v``: ``eps_i = rho * v_i + e_i`` with ``e_i ~ N(0, sigma_eps^2)``
    independent of ``v_i ~ N(0, sigma_v^2)``, a ``Normal`` prior on
    ``rho`` and ``HalfNormal`` priors on the two scales. The likelihood
    of the outcome is therefore
    normal with mean
    ``alpha + LATE * D_i + beta_X' X_i + rho * (D_i - E[D_i | Z_i, X_i])``
    and variance ``sigma_eps^2``, given ``D_i``, with the first-stage mean
    a function of the *parameters* ``pi``, so
    the posterior of the LATE carries the uncertainty of the first stage
    and of the error covariance. With a strong instrument and weak
    priors it is centred on 2SLS with the 2SLS standard deviation; with
    a weak instrument (``pi_Z`` near 0) it widens and becomes
    non-normal.

    .. note::
       Before 1.39 the residuals ``D - E[D | Z, X]`` were computed once
       by OLS and treated as data. That kept the posterior mean at 2SLS
       but made the posterior standard deviation too small by the factor
       ``sqrt(1 - corr(v, eps)^2)``: at a correlation of 0.9 the 95%
       interval covered the true effect about half the time. See
       ``MIGRATION.md``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    treat : str
        Endogenous treatment / regressor column (continuous or binary).
    instrument : str or sequence of str
        One or more instruments. Must be excluded from the structural
        equation.
    covariates : list of str, optional
        Exogenous controls entering both stages.
    per_instrument : bool, default ``False``
        When ``True`` and multiple instruments are supplied, additionally
        fits one just-identified Bayesian IV sub-model per instrument
        and populates :attr:`BayesianIVResult.instrument_summaries`,
        letting ``tidy(terms='per_instrument')`` emit one LATE row per
        ``Z_j``. The top-level pooled LATE posterior remains the joint
        over-identified fit (v0.9.15 behaviour). Each per-instrument
        sub-fit reuses the same priors and sampler controls as the
        pooled fit (draws/tune/chains/target_accept/random_state), so
        runtime scales roughly as ``(K+1)×`` the pooled fit.
    prior_late : (float, float)
        Normal prior on the structural LATE coefficient.
    prior_first_stage_sigma, prior_coef_sigma, prior_noise : float
        Priors for first-stage coefficients, structural coefficients,
        and the two residual scales.
    rope : (float, float), optional
        Region of practical equivalence.
    hdi_prob, draws, tune, chains, target_accept, random_state, progressbar :
        Sampler controls — see :func:`bayes_did`.

    Returns
    -------
    BayesianCausalResult
        Posterior summary on the LATE coefficient.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> # Bayesian IV jointly modelling first stage + structural eq.
    >>> # (requires the `bayes` extra: pip install 'statspai[bayes]').
    >>> res = sp.bayes_iv(df, y='y', treat='d', instrument='z',
    ...                   covariates=['x1'],
    ...                   draws=500, tune=500, chains=2)  # doctest: +SKIP
    >>> print(res.summary())  # doctest: +SKIP
    >>> # Multiple instruments with one LATE row per instrument:
    >>> res2 = sp.bayes_iv(df, y='y', treat='d',
    ...                    instrument=['z1', 'z2'],
    ...                    per_instrument=True)  # doctest: +SKIP
    >>> res2.tidy(terms='per_instrument')  # doctest: +SKIP
    """
    pm, _ = _require_pymc()

    prep = _prepare_iv_frame(data, y, treat, instrument, covariates)
    n = prep["n"]
    Y = prep["Y"]
    D = prep["D"]
    Z = prep["Z"]
    X = prep["X"]
    n_instr = Z.shape[1]

    mu_late, sigma_late = prior_late

    with pm.Model() as model:
        # ------------------------------------------------------------------
        # First stage: D = pi_0 + pi_Z'Z + pi_X'X + v,  v ~ N(0, sigma_v^2)
        # ------------------------------------------------------------------
        pi_intercept = pm.Normal("pi_intercept", mu=0.0, sigma=prior_first_stage_sigma)
        pi_Z = pm.Normal(
            "pi_Z",
            mu=0.0,
            sigma=prior_first_stage_sigma,
            shape=n_instr,
        )
        first_stage = pi_intercept + pm.math.dot(Z, pi_Z)
        if X is not None:
            pi_X = pm.Normal(
                "pi_X",
                mu=0.0,
                sigma=prior_first_stage_sigma,
                shape=X.shape[1],
            )
            first_stage = first_stage + pm.math.dot(X, pi_X)
        sigma_v = pm.HalfNormal("sigma_v", sigma=prior_noise)
        pm.Normal("d_obs", mu=first_stage, sigma=sigma_v, observed=D)

        # ------------------------------------------------------------------
        # Structural equation given D. With eps = rho * v + e the joint
        # normal of (v, eps) factorises into the first stage above and
        # Y | D ~ N(alpha + late * D + X'beta + rho * v, sigma_eps^2),
        # where v = D - first_stage is a function of the first-stage
        # parameters, not a pre-computed residual: treating OLS residuals
        # as data understates the posterior sd of `late` by
        # sqrt(1 - corr(v, eps)^2).
        # ------------------------------------------------------------------
        alpha = pm.Normal("alpha", mu=0.0, sigma=prior_coef_sigma)
        late = pm.Normal("late", mu=mu_late, sigma=sigma_late)
        rho = pm.Normal("rho_cf", mu=0.0, sigma=prior_coef_sigma)
        structural = alpha + late * D + rho * (D - first_stage)
        if X is not None:
            beta_X = pm.Normal(
                "beta_X",
                mu=0.0,
                sigma=prior_coef_sigma,
                shape=X.shape[1],
            )
            structural = structural + pm.math.dot(X, beta_X)
        sigma_eps = pm.HalfNormal("sigma_eps", sigma=prior_noise)
        pm.Normal("y_obs", mu=structural, sigma=sigma_eps, observed=Y)

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

    summary = _summarise_posterior(
        trace,
        "late",
        hdi_prob=hdi_prob,
        rope=rope,
    )

    model_info = {
        "inference": inference,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "target_accept": target_accept,
        "prior_late": prior_late,
        "prior_first_stage_sigma": prior_first_stage_sigma,
        "prior_coef_sigma": prior_coef_sigma,
        "prior_noise": prior_noise,
        "instruments": prep["iv_cols"],
        "covariates": prep["cov_cols"],
        "n_instruments": n_instr,
    }

    method_label = (
        f"Bayesian IV (joint 2SLS, {n_instr} instrument"
        f"{'s' if n_instr > 1 else ''})"
    )

    instrument_summaries: dict = {}
    instrument_labels: list = []
    # Per-instrument just-identified LATEs. Only meaningful with
    # multiple instruments — silently skip otherwise and return the
    # pooled result as v0.9.15 did, so passing per_instrument=True
    # with a single Z doesn't error (the single LATE IS the
    # per-instrument LATE). With K>=2 we loop the pooled fit with one
    # Z at a time; this is slow but transparent — each sub-fit is a
    # plain bayes_iv call under the hood.
    if per_instrument and n_instr >= 2:
        for z_name in prep["iv_cols"]:
            sub = bayes_iv(
                data,
                y=y,
                treat=treat,
                instrument=z_name,  # scalar path = just-identified
                covariates=covariates,
                per_instrument=False,  # guard against recursion
                prior_late=prior_late,
                prior_first_stage_sigma=prior_first_stage_sigma,
                prior_coef_sigma=prior_coef_sigma,
                prior_noise=prior_noise,
                rope=rope,
                hdi_prob=hdi_prob,
                inference=inference,
                advi_iterations=advi_iterations,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_state=random_state,
                progressbar=progressbar,
            )
            instrument_summaries[z_name] = {
                "posterior_mean": sub.posterior_mean,
                "posterior_median": sub.posterior_median,
                "posterior_sd": sub.posterior_sd,
                "hdi_lower": sub.hdi_lower,
                "hdi_upper": sub.hdi_upper,
                "prob_positive": sub.prob_positive,
            }
        instrument_labels = list(prep["iv_cols"])
        method_label += f" + {len(instrument_labels)} per-Z sub-fits"

    return BayesianIVResult(
        method=method_label,
        estimand="LATE",
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
        instrument_summaries=instrument_summaries,
        instrument_labels=instrument_labels,
    )
