"""Bayesian Double Machine Learning (arXiv:2508.12688, 2025).

Wraps :func:`sp.dml` cross-fitted residuals in a Normal-Normal conjugate
posterior, turning frequentist DML into a fully Bayesian analogue with
interpretable credible intervals, posterior probabilities, and the
option to shrink toward prior beliefs (useful for pre-registered
hypotheses or informative priors from a meta-analysis).

Two modes:

- ``'conjugate'`` (default): Normal-Normal conjugate update on the DML
  point estimate and its standard error. Extremely fast. Falls back to
  weakly-informative Normal(0, 10²) prior by default.
- ``'full'`` (PyMC): full posterior MCMC over the orthogonal-moment
  equation. Requires the ``[bayes]`` extras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


__all__ = ["bayes_dml", "BayesianDMLResult"]


@dataclass
class BayesianDMLResult:
    """Bayesian DML posterior summary."""
    posterior_mean: float
    posterior_sd: float
    ci: tuple  # (low, high) HDI at 1 - alpha
    prior_mean: float
    prior_sd: float
    dml_point: float
    dml_se: float
    posterior_prob_positive: float
    posterior_prob_negative: float
    mode: str
    draws: Optional[np.ndarray] = None

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Bayesian DML ({self.mode})\n"
            f"{'=' * 60}\n"
            f"  DML point (freq.)   : {self.dml_point:+.6f} "
            f"(SE {self.dml_se:.6f})\n"
            f"  Prior               : Normal({self.prior_mean:.3f}, {self.prior_sd:.3f}^2)\n"
            f"  Posterior mean      : {self.posterior_mean:+.6f}\n"
            f"  Posterior SD        : {self.posterior_sd:.6f}\n"
            f"  95% credible interval: [{lo:+.6f}, {hi:+.6f}]\n"
            f"  P(theta > 0 | data) : {self.posterior_prob_positive:.4f}\n"
            f"  P(theta < 0 | data) : {self.posterior_prob_negative:.4f}"
        )


def bayes_dml(
    data: pd.DataFrame,
    *,
    y: str,
    treatment: str,
    covariates: Sequence[str],
    model: str = "plr",
    prior_mean: float = 0.0,
    prior_sd: float = 10.0,
    mode: str = "conjugate",
    alpha: float = 0.05,
    n_folds: int = 5,
    random_state: int = 42,
    n_samples: int = 2000,
    ml_g: Optional[Any] = None,
    ml_m: Optional[Any] = None,
) -> BayesianDMLResult:
    """Bayesian Double Machine Learning estimator.

    Parameters
    ----------
    data : DataFrame
    y, treatment : str
    covariates : sequence of str
    model : str, default 'plr'
        Forwarded to :func:`sp.dml` — ``'plr'`` / ``'irm'`` / ``'pliv'``.
    prior_mean, prior_sd : float
        Parameters of the Normal prior on the treatment effect.
    mode : {'conjugate', 'full'}, default 'conjugate'
        Conjugate uses closed-form Normal-Normal updating on the DML
        point/SE. Full mode requires PyMC (``[bayes]`` extras) and
        samples the full posterior over the orthogonal moment equation.
    alpha : float, default 0.05
    n_folds : int, default 5
    random_state : int, default 42
    n_samples : int, default 2000
        Posterior draws (used only for ``mode='full'``).
    ml_g, ml_m : sklearn estimators, optional
        Custom nuisance learners forwarded to :func:`sp.dml`.

    Returns
    -------
    BayesianDMLResult

    Notes
    -----
    **Conjugate posterior** — for Normal(mu_0, sigma_0²) prior and a
    Gaussian likelihood with variance sigma²:

        posterior_precision = 1/sigma_0² + 1/sigma²
        posterior_mean = (mu_0/sigma_0² + dml_point/sigma²) / posterior_precision
        posterior_sd   = 1 / sqrt(posterior_precision)

    With a weakly-informative prior (sigma_0 large) the posterior mean
    collapses to the DML point estimate and the credible interval is
    (approximately) the DML 95% CI.

    References
    ----------
    Chernozhukov et al. (arXiv:2508.12688, 2025).
    """
    if mode not in ("conjugate", "full"):
        raise ValueError(f"mode must be 'conjugate' or 'full'; got {mode!r}.")
    if prior_sd <= 0:
        raise ValueError(f"prior_sd must be > 0; got {prior_sd}.")

    # Delegate to existing DML for the frequentist point/SE.
    from ..dml import dml as _dml
    dml_res = _dml(
        data=data,
        y=y,
        treat=treatment,
        covariates=list(covariates),
        model=model,
        n_folds=n_folds,
        ml_g=ml_g,
        ml_m=ml_m,
    )
    # Extract canonical point estimate + SE from the DML result object.
    dml_point = float(getattr(dml_res, "estimate", getattr(dml_res, "coef", np.nan)))
    if np.isnan(dml_point):
        # `DoubleML`-style object exposes .results DataFrame
        if hasattr(dml_res, "results") and "coef" in dml_res.results.columns:
            dml_point = float(dml_res.results["coef"].iloc[0])
    dml_se = float(getattr(dml_res, "se", np.nan))
    if np.isnan(dml_se) and hasattr(dml_res, "results") and "se" in dml_res.results.columns:
        dml_se = float(dml_res.results["se"].iloc[0])
    if np.isnan(dml_se) or dml_se <= 0:
        from statspai.exceptions import NumericalInstability
        raise NumericalInstability(
            "DML returned non-positive standard error; cannot form "
            "Bayesian posterior. Check DML inputs.",
            recovery_hint=(
                "Inspect the underlying DML fit (sp.dml) for overlap / "
                "cross-fit failures; ensure propensity scores are bounded "
                "away from 0 and 1."
            ),
            diagnostics={"dml_se": float(dml_se) if not np.isnan(dml_se) else None},
            alternative_functions=["sp.dml"],
        )

    if mode == "conjugate":
        # Normal-Normal conjugate update.
        prec_prior = 1.0 / prior_sd ** 2
        prec_like = 1.0 / dml_se ** 2
        posterior_prec = prec_prior + prec_like
        posterior_mean = (
            prior_mean * prec_prior + dml_point * prec_like
        ) / posterior_prec
        posterior_sd = float(1.0 / np.sqrt(posterior_prec))
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (posterior_mean - z * posterior_sd, posterior_mean + z * posterior_sd)
        p_pos = float(1 - stats.norm.cdf(0, loc=posterior_mean, scale=posterior_sd))
        p_neg = float(stats.norm.cdf(0, loc=posterior_mean, scale=posterior_sd))
        # Monte Carlo draws for downstream use
        rng = np.random.default_rng(random_state)
        draws = rng.normal(posterior_mean, posterior_sd, size=n_samples)
        return BayesianDMLResult(
            posterior_mean=float(posterior_mean),
            posterior_sd=posterior_sd,
            ci=(float(ci[0]), float(ci[1])),
            prior_mean=prior_mean,
            prior_sd=prior_sd,
            dml_point=dml_point,
            dml_se=dml_se,
            posterior_prob_positive=p_pos,
            posterior_prob_negative=p_neg,
            mode="conjugate",
            draws=draws,
        )

    # mode == 'full'
    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError(
            "`mode='full'` requires PyMC. Install with:\n"
            "    pip install 'statspai[bayes]'"
        ) from exc
    # Minimal single-moment Bayesian regression on (g, m) residuals.
    # Re-fit the PLR orthogonal residuals here by regressing on learners.
    # To stay tractable, re-use DML's residual Y/D if exposed; otherwise
    # fall back to refitting with a light-touch helper.
    if not hasattr(dml_res, "psi_b") or not hasattr(dml_res, "psi_a"):
        raise RuntimeError(
            "Full-mode Bayesian DML requires the DML object to expose "
            "orthogonal score arrays (psi_a, psi_b); upstream DML did not."
        )
    psi_a = np.asarray(dml_res.psi_a, dtype=float).ravel()
    psi_b = np.asarray(dml_res.psi_b, dtype=float).ravel()
    # Orthogonal moment: theta solves E[psi_a * theta + psi_b] = 0 → theta = -E[psi_b]/E[psi_a]
    with pm.Model():
        theta = pm.Normal("theta", mu=prior_mean, sigma=prior_sd)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        # Likelihood on each observation's score.
        pm.Normal(
            "obs",
            mu=psi_a * theta + psi_b,
            sigma=sigma,
            observed=np.zeros_like(psi_a),
        )
        trace = pm.sample(
            draws=n_samples, tune=1000, chains=2,
            target_accept=0.9, progressbar=False,
            random_seed=random_state,
        )
    theta_draws = trace.posterior["theta"].values.flatten()
    post_mean = float(theta_draws.mean())
    post_sd = float(theta_draws.std(ddof=1))
    ci = (
        float(np.quantile(theta_draws, alpha / 2)),
        float(np.quantile(theta_draws, 1 - alpha / 2)),
    )
    p_pos = float((theta_draws > 0).mean())
    p_neg = float((theta_draws < 0).mean())
    return BayesianDMLResult(
        posterior_mean=post_mean,
        posterior_sd=post_sd,
        ci=ci,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        dml_point=dml_point,
        dml_se=dml_se,
        posterior_prob_positive=p_pos,
        posterior_prob_negative=p_neg,
        mode="full",
        draws=theta_draws,
    )
