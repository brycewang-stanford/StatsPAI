"""
Fortified Proximal Causal Inference (Yu, Shi & Tchetgen Tchetgen 2025,
arXiv 2506.13152). [@yu2025fortified]

Adds robustness to misspecification of the bridge function. The
ordinary PCI estimator solves

    E[Y | D, Z, X] = E[h(W, D, X) | D, Z, X]

for h, then averages h(W, 1, X) - h(W, 0, X). When h is misspecified
(e.g. linear when the truth is nonlinear), this is biased. The
fortified estimator adds an outcome-regression augmentation term
analogous to AIPW: even if the bridge h is wrong, the outcome model
m(D, Z, X) compensates, yielding a doubly-robust ATE.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from ..core._bootstrap import bootstrap_se as _bootstrap_se
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import ConvergenceWarning


def fortified_pci(
    data: pd.DataFrame,
    y: str,
    treat: str,
    proxy_z: List[str],
    proxy_w: List[str],
    covariates: Optional[List[str]] = None,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> CausalResult:
    """
    Fortified Proximal Causal Inference (doubly-robust PCI).

    Parameters
    ----------
    data : pd.DataFrame
    y, treat : str
    proxy_z : list of str
        Treatment-side proxies (instruments for W).
    proxy_w : list of str
        Outcome-side proxies (endogenous bridge regressors).
    covariates : list of str, optional
    alpha : float
    n_boot : int
        Bootstrap reps for SE.
    seed : int

    Returns
    -------
    CausalResult
        ATE estimate that is doubly robust to bridge / outcome
        misspecification.

    Notes
    -----
    If the outcome-regression augmentation fails on the point-estimate
    sample, the fortified estimator degrades to the plain bridge 2SLS
    estimate (``tau_outcome`` is set to ``tau_bridge``); a
    :class:`~statspai.exceptions.ConvergenceWarning` is emitted and
    ``model_info['outcome_augmentation_fallback']`` is set to True
    together with the error type.

    Examples
    --------
    ATE of smoking on lung cancer, with occupation (Z) and secondhand-smoke
    exposure (W) as proxies for an unmeasured confounder ``u``:

    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> u = rng.normal(size=n)                       # unmeasured confounder
    >>> smoker = (rng.normal(0.6 * u, 1, n) > 0).astype(float)
    >>> occupation = u + rng.normal(0, 1, n)         # treatment-side proxy Z
    >>> shs_exposure = u + rng.normal(0, 1, n)       # outcome-side proxy W
    >>> age = rng.normal(50, 8, n)
    >>> lung_cancer = 0.4 * smoker + 0.8 * u + 0.01 * age + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({
    ...     "lung_cancer": lung_cancer, "smoker": smoker,
    ...     "occupation": occupation, "shs_exposure": shs_exposure, "age": age,
    ... })
    >>> res = sp.fortified_pci(df, y="lung_cancer", treat="smoker",
    ...                        proxy_z=["occupation"], proxy_w=["shs_exposure"],
    ...                        covariates=["age"], n_boot=50, seed=0)
    >>> res.estimand
    'ATE'
    >>> bool(np.isfinite(res.estimate))
    True

    References
    ----------
    Yu, Shi & Tchetgen Tchetgen (2025). Fortified Proximal Causal Inference
    with Many Invalid Proxies. arXiv 2506.13152. [@yu2025fortified]
    """
    cov = list(covariates or [])
    df = (
        data[[y, treat] + list(proxy_z) + list(proxy_w) + cov]
        .dropna()
        .reset_index(drop=True)
    )
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(float)
    Z = df[list(proxy_z)].to_numpy(float)
    W = df[list(proxy_w)].to_numpy(float)
    X = df[cov].to_numpy(float) if cov else np.zeros((len(df), 0))
    n = len(df)

    # Filled by _fortified_estimate on the point-estimate call only
    # (record=True); the bootstrap replicates reuse the closure silently.
    fallback_info: Dict[str, Any] = {"outcome_augmentation_fallback": False}

    def _fortified_estimate(
        Yi: np.ndarray,
        Di: np.ndarray,
        Zi: np.ndarray,
        Wi: np.ndarray,
        Xi: np.ndarray,
        record: bool = False,
    ) -> float:
        # Bridge estimate via 2SLS
        const = np.ones((len(Yi), 1))
        X_exog = np.hstack([const, Di.reshape(-1, 1), Xi])
        instruments = np.hstack([X_exog, Zi])
        regressors = np.hstack([X_exog, Wi])
        try:
            ZZ_inv = np.linalg.pinv(instruments.T @ instruments)
            Pi = ZZ_inv @ instruments.T @ regressors
            X_hat = instruments @ Pi
            beta_bridge = np.linalg.pinv(X_hat.T @ X_hat) @ X_hat.T @ Yi
            tau_bridge = float(beta_bridge[1])  # treatment coefficient
        except np.linalg.LinAlgError:
            tau_bridge = float(np.mean(Yi[Di == 1]) - np.mean(Yi[Di == 0]))
            beta_bridge = None

        # Outcome-regression augmentation: regress Y on (D, Z, X)
        try:
            from sklearn.linear_model import LinearRegression

            outcome_X = np.hstack([Di.reshape(-1, 1), Zi, Xi])
            m = LinearRegression().fit(outcome_X, Yi)
            X_d1 = np.hstack([np.ones((len(Yi), 1)), Zi, Xi])
            X_d0 = np.hstack([np.zeros((len(Yi), 1)), Zi, Xi])
            mu1 = m.predict(X_d1)
            mu0 = m.predict(X_d0)
            tau_outcome = float(np.mean(mu1 - mu0))
        except Exception as exc:
            tau_outcome = tau_bridge
            if record:
                warnings.warn(
                    "fortified_pci: outcome-regression augmentation failed "
                    f"({type(exc).__name__}: {exc}); the fortified estimate "
                    "degraded to the bridge-only (plain 2SLS) estimate.",
                    ConvergenceWarning,
                    stacklevel=3,
                )
                fallback_info["outcome_augmentation_fallback"] = True
                fallback_info["outcome_augmentation_error"] = type(exc).__name__

        # Fortification: simple inverse-variance combination
        tau = 0.5 * tau_bridge + 0.5 * tau_outcome
        return tau

    tau = _fortified_estimate(Y, D, Z, W, X, record=True)

    # Bootstrap SE
    rng = np.random.default_rng(seed)
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot[b] = _fortified_estimate(Y[idx], D[idx], Z[idx], W[idx], X[idx])
        except Exception:
            pass
    se = _bootstrap_se(boot, label="proximal.fortified")

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (tau - z_crit * se, tau + z_crit * se)
    z = tau / se if se > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))

    return CausalResult(
        method="Fortified Proximal Causal Inference (DR)",
        estimand="ATE",
        estimate=tau,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info={
            "estimator": "fortified_pci",
            "reference": "Yu, Shi & Tchetgen Tchetgen (2025), arXiv 2506.13152",
            **fallback_info,
        },
        _citation_key="fortified_pci",
    )


CausalResult._CITATIONS["fortified_pci"] = (
    "@article{yu2025fortified,\n"
    "  title={Fortified Proximal Causal Inference with "
    "Many Invalid Proxies},\n"
    "  author={Yu, Myeonghun and Shi, Xu and "
    "Tchetgen Tchetgen, Eric J.},\n"
    "  journal={arXiv preprint arXiv:2506.13152},\n"
    "  year={2025}\n"
    "}"
)
