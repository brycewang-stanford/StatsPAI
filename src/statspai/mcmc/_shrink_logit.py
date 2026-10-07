"""
Horseshoe prior for a logistic regression, by Polya-Gamma Gibbs sampling.

The logit counterpart of the Gaussian sampler in
:mod:`statspai.mcmc.shrinkage`. Conditional on Polya-Gamma variables the
coefficients have a normal full conditional (Polson, Scott and Windle
2013), so the scale updates of the horseshoe carry over with the residual
variance set to one.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import special

from ..exceptions import ConvergenceWarning, MethodIncompatibility
from ._core import rmvnorm_prec
from ._polyagamma import rpolyagamma
from .diagnostics import mcmc_summary
from .regress import BayesRegressResult

#: latent standard deviation of a logit at probability one half; the role
#: sigma plays in the Gaussian calibration of the global scale
PSEUDO_SIGMA = 2.0


class _LogitShrinkShim:
    """What the predictive methods need to treat the fit as a logit."""

    name = "logit"
    trials = None

    def __init__(self, X: np.ndarray, names: List[str], y: np.ndarray):
        self.X = X
        self.k = X.shape[1]
        self.xnames = names
        self.y = np.asarray(y, dtype=float)

    def linear_predictor(self, draws: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.asarray(draws[:, : self.k] @ X.T)

    @staticmethod
    def expected_value(eta: np.ndarray) -> np.ndarray:
        return np.asarray(special.expit(eta))

    def log_lik(self, theta: np.ndarray) -> float:
        eta = self.X @ theta[: self.k]
        return float(self.y @ eta - np.logaddexp(0.0, eta).sum())


def shrink_logit(
    formula: str,
    y: np.ndarray,
    X: np.ndarray,
    names: List[str],
    design_info: Any,
    standardize: bool,
    global_scale: Optional[float],
    p0: Optional[float],
    slab_scale: Optional[float],
    slab_df: float,
    draws: int,
    burnin: int,
    thin: int,
    rng: np.random.Generator,
    level: float,
) -> BayesRegressResult:
    """Fit; called by :func:`statspai.bayes_shrink` with ``family='logit'``."""
    from .shrinkage import _hs_variance, _regularized_hs_step

    vals = np.unique(y)
    if not np.all(np.isin(vals, (0.0, 1.0))) or vals.size < 2:
        raise MethodIncompatibility(
            "family='logit' needs a 0 / 1 outcome with both values present."
        )
    n, p = X.shape
    xbar = X.mean(axis=0)
    sx = X.std(axis=0, ddof=1) if standardize else np.ones(p)
    if np.any(sx == 0):
        raise MethodIncompatibility("A regressor is constant.")
    Z = (X - xbar) / sx
    W = np.column_stack([np.ones(n), Z])
    kappa_y = y - 0.5
    Wk = W.T @ kappa_y
    if global_scale is not None and p0 is not None:
        raise MethodIncompatibility("Pass global_scale or p0, not both.")
    if p0 is not None:
        if not 0 < p0 < p:
            raise MethodIncompatibility(
                f"p0 must be between 0 and the number of regressors ({p})."
            )
        tau0 = float(float(p0) / (p - float(p0)) * PSEUDO_SIGMA / np.sqrt(n))
    else:
        tau0 = 1.0 if global_scale is None else float(global_scale)
    if not tau0 > 0:
        raise MethodIncompatibility("global_scale must be positive.")
    regularized = slab_scale is not None
    slab_s = float(slab_scale) if slab_scale is not None else 0.0
    if regularized and not (slab_s > 0 and slab_df > 0):
        raise MethodIncompatibility("slab_scale and slab_df must be positive.")
    slab2 = slab_s**2 if regularized else np.inf
    steps = np.array([1.0, 0.5, 1.0, 0.5])
    loc2 = np.ones(p)
    nu_aux = np.ones(p)
    glob2 = min(1.0, tau0**2)
    xi_aux = 1.0
    zvar = (Z * Z).sum(axis=0) / n
    theta = np.zeros(p + 1)
    share = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
    theta[0] = np.log(share / (1.0 - share))
    out = np.empty((draws, p + 1))
    out_t = np.empty(draws)
    out_c = np.empty(draws)
    out_k = np.zeros(p)
    m_eff = 0.0
    kept = 0
    for it in range(burnin + draws * thin):
        omega = rpolyagamma(rng, W @ theta)
        prec = (W * omega[:, None]).T @ W
        v = np.maximum(_hs_variance(loc2, glob2, slab2), 1e-300)
        prec[np.arange(1, p + 1), np.arange(1, p + 1)] += 1.0 / v
        theta, _ = rmvnorm_prec(rng, Wk, prec)
        b2 = theta[1:] ** 2
        if regularized:
            for _ in range(5):
                loc2, glob2, slab2, rates = _regularized_hs_step(
                    rng, b2, 1.0, loc2, glob2, slab2, tau0, slab_df, slab_s**2, steps
                )
            if it < burnin:
                steps *= np.exp(0.03 * (rates - 0.35))
        else:
            # Gibbs draws of the scales; a few passes per draw of the
            # coefficients, since the scales are what mixes slowly
            for _ in range(5):
                loc2 = (1.0 / nu_aux + b2 / (2.0 * glob2)) / rng.gamma(1.0, size=p)
                loc2 = np.clip(loc2, 1e-300, 1e300)
                glob2 = float(
                    (1.0 / xi_aux + float((b2 / loc2).sum()) / 2.0)
                    / rng.gamma((p + 1.0) / 2.0)
                )
                glob2 = min(max(glob2, 1e-300), 1e300)
                nu_aux = (1.0 + 1.0 / loc2) / rng.gamma(1.0, size=p)
                xi_aux = float((1.0 / tau0**2 + 1.0 / glob2) / rng.gamma(1.0))
        if it >= burnin and (it - burnin) % thin == 0:
            slopes = theta[1:] / sx
            out[kept, 1:] = slopes
            out[kept, 0] = theta[0] - xbar @ slopes
            out_t[kept] = np.sqrt(glob2)
            out_c[kept] = np.sqrt(slab2)
            kap = 1.0 / (
                1.0 + n * zvar * _hs_variance(loc2, glob2, slab2) / PSEUDO_SIGMA**2
            )
            out_k += kap
            m_eff += float((1.0 - kap).sum())
            kept += 1
    cols: Dict[str, np.ndarray] = {"Intercept": out[:, 0]}
    for j, nm in enumerate(names):
        cols[nm] = out[:, j + 1]
    cols["tau"] = out_t
    if regularized:
        cols["slab"] = out_c
    d_df = pd.DataFrame(cols)
    summ = mcmc_summary(d_df, quantiles=())
    lo = (1.0 - level) / 2.0
    table = pd.DataFrame(
        {
            "mean": summ["mean"],
            "sd": summ["sd"],
            "mcse": summ["ts_se"],
            "ess": summ["ess"],
            "lower": d_df.quantile(lo),
            "median": d_df.quantile(0.5),
            "upper": d_df.quantile(1.0 - lo),
            "prob_positive": (d_df > 0).mean(),
        }
    )
    shrink = pd.Series(np.nan, index=table.index)
    shrink.loc[names] = out_k / draws
    table["shrinkage"] = shrink
    info: Dict[str, Any] = {
        "prior": "horseshoe",
        "family": "logit",
        "standardize": bool(standardize),
        "global_scale": tau0,
        "m_eff": m_eff / draws,
    }
    if regularized:
        info.update({"slab_scale": slab_s, "slab_df": float(slab_df)})
    diag: Dict[str, Any] = {"warnings": [], "min_ess": float(table["ess"].min())}
    res = BayesRegressResult(
        model="horseshoe",
        formula=formula,
        params=table["mean"].copy(),
        std_errors=table["sd"].copy(),
        table=table,
        draws=d_df,
        chain=np.zeros(draws, dtype=int),
        n_obs=n,
        n_draws=draws,
        burnin=burnin,
        thin=thin,
        chains=1,
        sampler="Gibbs (Polya-Gamma augmentation)",
        acceptance_rate=None,
        prior={"slopes": "horseshoe", "intercept": "flat"},
        level=level,
        model_info=info,
        diagnostics_info=diag,
        _model=_LogitShrinkShim(
            np.column_stack([np.ones(n), X]), ["Intercept"] + names, y
        ),
        _design_info=design_info,
    )
    if diag["min_ess"] < 100:
        text = (
            "The chain mixes slowly: effective sample size "
            f"{diag['min_ess']:.0f}. Increase draws or thin."
        )
        diag["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=3)
    return res
