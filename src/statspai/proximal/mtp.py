"""
Proximal Causal Inference for Modified Treatment Policies
(Olivas-Martinez, Gilbert & Rotnitzky 2025, arXiv 2512.12038). [@olivas2025proximal]

Standard PCI estimates the ATE between a treatment level and its
reference. Modified Treatment Policies (MTP, Haneuse-Rotnitzky 2013)
ask: "what would happen if we *shifted* the treatment by a fixed
amount δ?" — useful for continuous-dose interventions.

This module fits a linear bridge under the MTP shift, returning the
average MTP effect E[Y(D + δ)] - E[Y(D)].
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def pci_mtp(
    data: pd.DataFrame,
    y: str,
    treat: str,
    proxy_z: List[str],
    proxy_w: List[str],
    delta: float,
    covariates: Optional[List[str]] = None,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> CausalResult:
    """
    PCI for Modified Treatment Policies (continuous-shift effect).

    Parameters
    ----------
    data : pd.DataFrame
    y, treat : str
        Outcome and continuous treatment.
    proxy_z, proxy_w : list of str
        Standard PCI proxies.
    delta : float
        MTP shift; estimand is E[Y(D + δ)] - E[Y(D)].
    covariates : list of str, optional
    alpha : float
    n_boot : int
    seed : int

    Returns
    -------
    CausalResult
    """
    cov = list(covariates or [])
    df = data[[y, treat] + list(proxy_z) + list(proxy_w) + cov] \
        .dropna().reset_index(drop=True)
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(float)
    Z = df[list(proxy_z)].to_numpy(float)
    W = df[list(proxy_w)].to_numpy(float)
    X = df[cov].to_numpy(float) if cov else np.zeros((len(df), 0))
    n = len(df)

    def _mtp(Yi, Di, Zi, Wi, Xi, delta):
        # Linear bridge h(W, D, X) = γ0 + γD D + γW W + γX X
        # via 2SLS with instruments (1, D, X, Z) for (1, D, X, W).
        const = np.ones((len(Yi), 1))
        X_exog = np.hstack([const, Di.reshape(-1, 1), Xi])
        instruments = np.hstack([X_exog, Zi])
        regressors = np.hstack([X_exog, Wi])
        try:
            ZZ_inv = np.linalg.pinv(instruments.T @ instruments)
            Pi = ZZ_inv @ instruments.T @ regressors
            X_hat = instruments @ Pi
            beta = np.linalg.pinv(X_hat.T @ X_hat) @ X_hat.T @ Yi
            gamma_d = float(beta[1])
            # MTP effect = γ_D * δ for linear bridge
            return gamma_d * delta
        except np.linalg.LinAlgError:
            return float(np.mean(Yi[Di > Di.mean()]) - np.mean(Yi[Di <= Di.mean()])) * delta

    tau = _mtp(Y, D, Z, W, X, delta)

    rng = np.random.default_rng(seed)
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot[b] = _mtp(Y[idx], D[idx], Z[idx], W[idx], X[idx], delta)
        except Exception:
            pass
    se = float(np.nanstd(boot, ddof=1)) or 1e-6

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (tau - z_crit * se, tau + z_crit * se)
    z = tau / se if se > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))

    return CausalResult(
        method="PCI for Modified Treatment Policy (shift δ)",
        estimand=f"E[Y(D+{delta})] - E[Y(D)]",
        estimate=tau,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info={
            'estimator': 'pci_mtp',
            'delta': delta,
            'reference': 'Olivas-Martinez, Gilbert & Rotnitzky (2025), arXiv 2512.12038',
        },
        _citation_key='pci_mtp',
    )


CausalResult._CITATIONS['pci_mtp'] = (
    "@article{olivas2025proximal,\n"
    "  title={Proximal Causal Inference for Modified Treatment Policies},\n"
    "  author={Olivas-Martinez, Antonio and Gilbert, Peter B. and Rotnitzky, Andrea},\n"
    "  journal={arXiv preprint arXiv:2512.12038},\n"
    "  year={2025}\n"
    "}"
)
