"""MR-Clust: clustered causal effects via finite Gaussian mixture.

Reference
---------
Foley, C.N., Mason, A.M., Kirk, P.D.W. & Burgess, S. (2021).
"MR-Clust: clustering of genetic variants in Mendelian randomization
with similar causal estimates." *Bioinformatics*, 37(4), 531-541.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ._common import as_float_arrays, harmonize_signs


__all__ = ["MRClustResult", "mr_clust"]


@dataclass
class MRClustResult:
    """Output of :func:`mr_clust`.

    Attributes
    ----------
    cluster_estimates : pd.DataFrame
        One row per cluster, columns ``[cluster, estimate, se, ci_lower,
        ci_upper, weight, n_snps]``.  Cluster 0 is a 'null' component at
        θ=0 in the Foley et al. 2021 specification.
    assignments : np.ndarray
        Integer cluster id (0..K-1) for each SNP.
    responsibilities : np.ndarray
        (n_snps, K) matrix of posterior cluster-membership probabilities.
    K : int
        Selected number of clusters (including the null cluster).
    bic : dict
        BIC per K tried; keys are ints.
    loglik : float
        Final log-likelihood at the selected K.
    wald_ratios : np.ndarray
        Per-SNP Wald ratio (β_y / β_x).
    wald_se : np.ndarray
        Per-SNP Wald-ratio first-order SE.
    n_snps : int
    """
    cluster_estimates: pd.DataFrame
    assignments: np.ndarray
    responsibilities: np.ndarray
    K: int
    bic: dict
    loglik: float
    wald_ratios: np.ndarray
    wald_se: np.ndarray
    n_snps: int

    def summary(self) -> str:
        lines = [
            "MR-Clust (clustered causal effects)",
            "=" * 62,
            f"  n SNPs : {self.n_snps}",
            f"  Selected K (incl. null): {self.K}",
            f"  Log-likelihood: {self.loglik:.3f}",
            "",
            "BIC by K:",
        ]
        for k in sorted(self.bic):
            mark = "  <-- selected" if k == self.K else ""
            lines.append(f"  K={k}: {self.bic[k]:.3f}{mark}")
        lines.append("")
        lines.append("Cluster estimates:")
        lines.append(
            self.cluster_estimates.to_string(index=False, float_format="%.4f")
        )
        return "\n".join(lines)


def _em_gaussian_mixture(
    r: np.ndarray,
    tau: np.ndarray,
    K: int,
    *,
    include_null: bool = True,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
):
    """EM for finite-Gaussian mixture with *SNP-specific* measurement SE.

    Model:  r_i | z_i=k  ~  N(theta_k, tau_i^2)
    Component 0 is fixed at theta_0 = 0 when include_null=True
    (Foley et al. 2021 — the 'null cluster' represents SNPs with no
    causal effect on the outcome once the exposure is adjusted away).

    Returns (theta, pi, responsibilities, loglik).
    """
    rng = np.random.default_rng(seed)
    n = len(r)

    if K == 1:
        theta_init = np.array([0.0]) if include_null else np.array([np.median(r)])
    else:
        qs = np.linspace(0.1, 0.9, K)
        theta_init = np.quantile(r, qs)
        if include_null:
            theta_init[0] = 0.0
    theta = theta_init.astype(float)
    pi = np.full(K, 1.0 / K)

    prev_loglik = -np.inf
    resp = np.zeros((n, K))
    for _ in range(max_iter):
        # E-step
        log_comp = np.empty((n, K))
        for k in range(K):
            log_comp[:, k] = (
                np.log(pi[k] + 1e-300)
                - 0.5 * np.log(2 * np.pi * tau ** 2)
                - 0.5 * ((r - theta[k]) / tau) ** 2
            )
        log_row_max = np.max(log_comp, axis=1, keepdims=True)
        log_norm = log_row_max.squeeze() + np.log(
            np.sum(np.exp(log_comp - log_row_max), axis=1)
        )
        resp = np.exp(log_comp - log_norm[:, None])

        loglik = float(np.sum(log_norm))

        # M-step
        Nk = resp.sum(axis=0)
        pi = Nk / n

        for k in range(K):
            if include_null and k == 0:
                theta[k] = 0.0
            else:
                w = resp[:, k] / tau ** 2
                if w.sum() <= 1e-300:
                    theta[k] = rng.normal(0, 0.1)  # reseed empty cluster
                else:
                    theta[k] = float(np.sum(w * r) / np.sum(w))

        if abs(loglik - prev_loglik) < tol:
            break
        prev_loglik = loglik

    return theta, pi, resp, prev_loglik


def mr_clust(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    K_range: Tuple[int, int] = (1, 5),
    include_null: bool = True,
    alpha: float = 0.05,
    seed: int = 0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> MRClustResult:
    """Clustered Mendelian randomization via Gaussian mixture on Wald ratios.

    Implements the MR-Clust estimator of Foley, Mason, Kirk & Burgess
    (2021) *Bioinformatics* 37(4).  Assigns each SNP to one of K clusters
    based on its Wald ratio, where cluster 0 is a 'null' component
    (SNPs with zero effect on the outcome).  Cluster means represent
    distinct causal-effect estimates — multiple non-null clusters are
    evidence of heterogeneous / pathway-specific pleiotropy.

    Model
    -----
    For SNP i in cluster k:

    .. math::

       r_i = \\frac{\\beta_{y,i}}{\\beta_{x,i}},
       \\qquad
       r_i \\mid z_i = k \\ \\sim\\ \\mathcal{N}(\\theta_k, \\tau_i^2)

    with :math:`\\tau_i = s_{y,i} / |\\beta_{x,i}|` the first-order Wald
    SE.  When ``include_null=True`` cluster 0 is fixed at
    :math:`\\theta_0 = 0`.  Parameters fitted by EM; K selected by BIC.

    Parameters
    ----------
    beta_exposure, beta_outcome : ndarray
    se_exposure, se_outcome : ndarray
    K_range : (int, int), default (1, 5)
        Inclusive range of cluster counts to try.  K=1 with
        ``include_null=True`` is "everything null" and K=1 without is a
        single cluster at median ratio.
    include_null : bool, default True
        If True the first cluster is fixed at 0 (null).
    alpha : float, default 0.05
        For per-cluster CIs.
    seed : int, default 0
        EM initialisation rng.
    max_iter, tol : EM controls.

    Returns
    -------
    :class:`MRClustResult`

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.mr_clust(bx, by, sx, sy, K_range=(1, 4))
    >>> print(res.summary())
    """
    bx, by, sx, sy = as_float_arrays(
        beta_exposure, beta_outcome, se_exposure, se_outcome
    )
    bx, by = harmonize_signs(bx, by)

    if K_range[0] < 1 or K_range[1] < K_range[0]:
        raise ValueError(f"invalid K_range {K_range}")

    r = by / bx
    tau = sy / np.abs(bx)  # first-order Wald SE
    n = len(r)

    bic_table = {}
    fits = {}
    for K in range(K_range[0], K_range[1] + 1):
        theta, pi, resp, loglik = _em_gaussian_mixture(
            r, tau, K,
            include_null=include_null,
            max_iter=max_iter, tol=tol, seed=seed,
        )
        n_free = (K - 1) + (K - int(include_null))
        n_free = max(n_free, 0)
        bic = -2.0 * loglik + n_free * np.log(n)
        bic_table[K] = float(bic)
        fits[K] = (theta, pi, resp, loglik)

    K_best = int(min(bic_table, key=bic_table.get))
    theta, pi, resp, loglik = fits[K_best]

    assignments = np.argmax(resp, axis=1)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for k in range(K_best):
        w_k = resp[:, k] / tau ** 2
        if w_k.sum() <= 1e-300:
            se_k = np.inf
        else:
            se_k = float(np.sqrt(1.0 / w_k.sum()))
        rows.append({
            "cluster": k,
            "estimate": float(theta[k]),
            "se": se_k,
            "ci_lower": float(theta[k] - z_crit * se_k),
            "ci_upper": float(theta[k] + z_crit * se_k),
            "weight": float(pi[k]),
            "n_snps": int((assignments == k).sum()),
        })
    cluster_df = pd.DataFrame(rows)

    return MRClustResult(
        cluster_estimates=cluster_df,
        assignments=assignments,
        responsibilities=resp,
        K=K_best,
        bic=bic_table,
        loglik=float(loglik),
        wald_ratios=r,
        wald_se=tau,
        n_snps=n,
    )
