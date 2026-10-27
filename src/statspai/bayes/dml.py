"""Bayesian Double Machine Learning (DiTraglia & Liu, arXiv:2508.12688, 2025).

Wraps :func:`sp.dml` cross-fitted residuals in a Normal-Normal conjugate
posterior, turning frequentist DML into a fully Bayesian analogue with
interpretable credible intervals, posterior probabilities, and the
option to shrink toward prior beliefs (useful for pre-registered
hypotheses or informative priors from a meta-analysis).

Two modes:

- ``'conjugate'`` (default): Normal-Normal conjugate update on the DML
  point estimate and its standard error. Extremely fast. Falls back to
  weakly-informative Normal(0, 10²) prior by default.
- ``'full'``: deprecated (removal in 1.43) and never available. It needs
  the orthogonal score arrays of the DML fit, which :func:`sp.dml` does
  not return; it warns (``DeprecationWarning``) and raises
  ``MethodIncompatibility`` before any fitting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility

__all__ = ["bayes_dml", "BayesianDMLResult"]


@dataclass
class BayesianDMLResult(ResultProtocolMixin):
    """Bayesian DML posterior summary.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x1": rng.normal(size=300), "x2": rng.normal(size=300)})
    >>> df["d"] = df.x1 + rng.normal(size=300)
    >>> df["y"] = 0.5 * df.d + df.x2 + rng.normal(size=300)
    >>> res = sp.bayes_dml(df, y="y", treatment="d", covariates=["x1", "x2"],
    ...                    n_folds=2)
    >>> isinstance(res, sp.BayesianDMLResult)
    True
    """

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
            f"  Prior               : Normal({self.prior_mean:.3f}, "
            f"{self.prior_sd:.3f}^2)\n"
            f"  Posterior mean      : {self.posterior_mean:+.6f}\n"
            f"  Posterior SD        : {self.posterior_sd:.6f}\n"
            f"  95% credible interval: [{lo:+.6f}, {hi:+.6f}]\n"
            f"  P(theta > 0 | data) : {self.posterior_prob_positive:.4f}\n"
            f"  P(theta < 0 | data) : {self.posterior_prob_negative:.4f}"
        )


@accepts_aliases(treat="treatment", seed="random_state")
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
    mode : {'conjugate'}, default 'conjugate'
        Conjugate uses closed-form Normal-Normal updating on the DML
        point/SE. The value ``'full'`` is deprecated and will be removed in
        1.43: it was never available (it needs the orthogonal score arrays,
        which :func:`sp.dml` does not return). It warns
        (``DeprecationWarning``) and raises ``MethodIncompatibility``.
    alpha : float, default 0.05
    n_folds : int, default 5
    random_state : int, default 42
        Seed of the cross-fitting split of :func:`sp.dml` and of the
        Monte Carlo draws of the posterior.
    n_samples : int, default 2000
        Number of Monte Carlo draws from the posterior kept in ``draws``.
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
    DiTraglia, F. J., & Liu, L. (2025). "Bayesian Double Machine
    Learning for Causal Inference." arXiv:2508.12688.
    (The underlying orthogonal-moments DML construction is due to
    Chernozhukov et al. 2018, Econometrics Journal.) [@ditraglia2025bayesian]

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> # Conjugate mode (default) cross-fits sp.dml then applies a
    >>> # Normal-Normal update.
    >>> res = sp.bayes_dml(df, y='y', treatment='d',
    ...                    covariates=['x1', 'x2'],
    ...                    prior_mean=0.0, prior_sd=1.0)  # doctest: +SKIP
    >>> print(res.summary())                  # doctest: +SKIP
    >>> res.posterior_mean, res.ci            # doctest: +SKIP
    """
    if mode not in ("conjugate", "full"):
        raise ValueError(f"mode must be 'conjugate' or 'full'; got {mode!r}.")
    if prior_sd <= 0:
        raise ValueError(f"prior_sd must be > 0; got {prior_sd}.")
    if mode == "full":
        # Checked before the DML fit: the score arrays it would need are
        # not part of the sp.dml result, so the fit could only be wasted.
        warnings.warn(
            "bayes_dml(mode='full') is deprecated and will be removed in 1.43; "
            "it was never available. Use mode='conjugate'.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise MethodIncompatibility(
            "bayes_dml(mode='full') needs the orthogonal score arrays of the "
            "DML fit, which sp.dml does not return.",
            recovery_hint=(
                "Use mode='conjugate' (the Normal-Normal update of the DML "
                "point estimate and standard error)."
            ),
            diagnostics={"mode": mode},
            alternative_functions=["sp.dml"],
        )

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
        random_state=random_state,
    )
    # Extract canonical point estimate + SE from the DML result object.
    dml_point = float(getattr(dml_res, "estimate", getattr(dml_res, "coef", np.nan)))
    if np.isnan(dml_point):
        # `DoubleML`-style object exposes .results DataFrame
        if hasattr(dml_res, "results") and "coef" in dml_res.results.columns:
            dml_point = float(dml_res.results["coef"].iloc[0])
    dml_se = float(getattr(dml_res, "se", np.nan))
    if (
        np.isnan(dml_se)
        and hasattr(dml_res, "results")
        and "se" in dml_res.results.columns
    ):
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

    # Normal-Normal conjugate update.
    prec_prior = 1.0 / prior_sd**2
    prec_like = 1.0 / dml_se**2
    posterior_prec = prec_prior + prec_like
    posterior_mean = (prior_mean * prec_prior + dml_point * prec_like) / posterior_prec
    posterior_sd = float(1.0 / np.sqrt(posterior_prec))
    z = stats.norm.ppf(1 - alpha / 2)
    ci = (posterior_mean - z * posterior_sd, posterior_mean + z * posterior_sd)
    p_pos = float(stats.norm.sf(0, loc=posterior_mean, scale=posterior_sd))
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
