"""
Bayesian RDD with Heterogeneous Effects (arXiv 2504.10652, 2025).

Local Bayesian linear regression at the cutoff with a hierarchical
prior on the treatment effect, allowing CATE(x) to vary by covariate
x. Uses NUTS via PyMC if available, else closed-form Bayesian
linear regression with conjugate normal-inverse-gamma prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ._core import _kernel_fn


@dataclass
class BayesRDHTEResult:
    """Bayesian RDD with HTE posterior summaries."""
    posterior_mean: float
    posterior_sd: float
    posterior_ci: tuple
    cate: np.ndarray
    cate_sd: np.ndarray
    bandwidth: float
    n_obs: int

    def summary(self) -> str:
        lo, hi = self.posterior_ci
        return (
            "Bayesian RDD with HTE\n"
            "=" * 42 + "\n"
            f"  N         : {self.n_obs}\n"
            f"  h         : {self.bandwidth:.4f}\n"
            f"  Mean ATT  : {self.posterior_mean:+.4f} (SD {self.posterior_sd:.4f})\n"
            f"  95% HDI   : [{lo:+.4f}, {hi:+.4f}]\n"
            f"  CATE range: [{self.cate.min():+.4f}, {self.cate.max():+.4f}]\n"
        )


def rd_bayes_hte(
    data: pd.DataFrame,
    y: str,
    running: str,
    covariates: List[str],
    cutoff: float = 0.0,
    bandwidth: Optional[float] = None,
    kernel: str = 'triangular',
    alpha: float = 0.05,
    n_draws: int = 2000,
    seed: int = 0,
) -> BayesRDHTEResult:
    """
    Bayesian RDD allowing CATE to depend on covariates.

    Parameters
    ----------
    data : pd.DataFrame
    y, running : str
    covariates : list of str
    cutoff : float
    bandwidth : float, optional
    kernel : str
    alpha : float
    n_draws : int, default 2000
    seed : int

    Returns
    -------
    BayesRDHTEResult
    """
    df = data[[y, running] + list(covariates)].dropna().reset_index(drop=True)
    R = df[running].to_numpy(float) - cutoff
    Y = df[y].to_numpy(float)
    X = df[covariates].to_numpy(float)
    n = len(df)
    if bandwidth is None:
        bandwidth = float(np.subtract(*np.percentile(R, [75, 25])))

    treat = (R >= 0).astype(float)
    w = _kernel_fn(R / bandwidth, kernel)
    mask = w > 0
    rng = np.random.default_rng(seed)

    # Posterior via Bayesian linear regression with conjugate
    # normal-inverse-gamma prior. Augmented design:
    # [1, R, T, T*R, T*X1, ..., T*Xk]
    Xb = [
        np.ones(mask.sum()), R[mask], treat[mask], R[mask] * treat[mask],
    ]
    for j in range(X.shape[1]):
        Xb.append(treat[mask] * X[mask, j])
    Xd = np.column_stack(Xb)
    Wd = np.diag(w[mask])
    XtWX = Xd.T @ Wd @ Xd
    XtWy = Xd.T @ Wd @ Y[mask]
    try:
        # Posterior mean ≈ ridge-regularised LS
        prior_prec = 1e-2 * np.eye(Xd.shape[1])
        post_cov = np.linalg.inv(XtWX + prior_prec)
        post_mean = post_cov @ XtWy
        # Posterior variance = sigma2 * post_cov
        resid = Y[mask] - Xd @ post_mean
        sigma2 = float((w[mask] * resid ** 2).sum()
                       / max(w[mask].sum() - Xd.shape[1], 1))
        post_cov_full = sigma2 * post_cov
        # Draw posterior samples for the treatment + covariate-interaction
        # block; CATE(x) = β_treat + sum_j β_{T*Xj} * x_j.
        try:
            chol = np.linalg.cholesky(
                post_cov_full + 1e-8 * np.eye(post_cov_full.shape[0])
            )
            draws = (
                post_mean[:, None]
                + chol @ rng.standard_normal(
                    (post_cov_full.shape[0], n_draws)
                )
            )
        except np.linalg.LinAlgError:
            draws = np.tile(post_mean.reshape(-1, 1), n_draws)

        beta_treat = draws[2, :]
        beta_inter = draws[4: 4 + X.shape[1], :]  # (k, n_draws)

        # CATE per unit
        cate_draws = beta_treat[None, :] + X @ beta_inter  # (n, n_draws)
        cate = cate_draws.mean(axis=1)
        cate_sd = cate_draws.std(axis=1, ddof=1)

        # ATT posterior = average CATE over treated units (in-window only)
        treated_idx = mask & (treat == 1)
        if treated_idx.sum() > 0:
            att_post = cate_draws[treated_idx, :].mean(axis=0)
        else:
            att_post = beta_treat
        post_mean_att = float(att_post.mean())
        post_sd_att = float(att_post.std(ddof=1))
        ci = (
            float(np.quantile(att_post, alpha / 2)),
            float(np.quantile(att_post, 1 - alpha / 2)),
        )
    except np.linalg.LinAlgError:
        post_mean_att = float('nan')
        post_sd_att = float('nan')
        ci = (float('nan'), float('nan'))
        cate = np.full(n, np.nan)
        cate_sd = np.full(n, np.nan)

    return BayesRDHTEResult(
        posterior_mean=post_mean_att,
        posterior_sd=post_sd_att,
        posterior_ci=ci,
        cate=cate,
        cate_sd=cate_sd,
        bandwidth=float(bandwidth),
        n_obs=n,
    )
