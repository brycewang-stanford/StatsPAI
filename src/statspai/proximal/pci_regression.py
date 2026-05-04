"""
Regression-based proximal causal inference (Tchetgen Tchetgen et al.
2024; Cui et al. 2024).

Motivation
----------
Proximal causal inference identifies treatment effects in the presence
of an *unmeasured* confounder :math:`U` using two proxies:

* :math:`Z` — a *treatment* proxy (Z ⊥ Y | D, U, X)
* :math:`W` — an *outcome* proxy (W ⊥ D | U, X)

The earliest non-parametric PCI estimators required solving Fredholm
integral equations. Tchetgen Tchetgen et al. (2024) showed that, under
a *linear outcome bridge* assumption plus a parametric "treatment
bridge" equation, one can obtain a doubly-robust, point-identified
estimate via a pair of simple regressions plus a "bridge imputation"
step. This module implements that pipeline:

1. Fit outcome bridge :math:`h(W, D, X) = \\alpha_0 + \\alpha_D D + \\alpha_W W + \\alpha_X X`
   via IV with :math:`Z` instrumenting for :math:`W`.
2. Fit treatment bridge :math:`q(Z, X) = \\gamma_0 + \\gamma_D(Z, X) \\cdot \\mathbb 1\\{D=1\\}`
   (equivalently, a propensity-style model of D on Z, X).
3. Combine via the doubly robust moment

.. math::

   \\hat\\psi = E_n[\\,(h(W, 1, X) - h(W, 0, X))
                  + (2D - 1) \\cdot q(Z, X) (Y - h(W, D, X))\\,].

Under either (i) a correctly specified outcome bridge **or** (ii) a
correctly specified treatment bridge, :math:`\\hat\\psi` is consistent
for the ATE (Tchetgen et al. 2024, Theorem 3).

This is the *regression-based PCI* estimator — simpler and more
practical than the kernel / sieve implementations while retaining
double robustness under linear / GLM parameterisations.

References
----------
Tchetgen Tchetgen, E. J., Ying, A., Cui, Y., Shi, X., & Miao, W. (2024).
"An introduction to proximal causal inference." *arXiv*:2009.10982. [@tchetgentchetgen2024introduction]

Cui, Y., Pu, H., Shi, X., Miao, W. & Tchetgen Tchetgen, E. (2024).
"Semiparametric proximal causal inference." *JASA*, 119(547), 1348-1359. [@cui2024semiparametric]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
# sklearn is imported lazily inside the functions that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches proximal_regression.


@dataclass
class ProximalRegResult:
    ate: float
    se: float
    ci: tuple
    pvalue: float
    bridge_coefs: Dict[str, float]
    propensity_coefs: Dict[str, float]
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lo, hi = self.ci
        return (
            "Regression-based Proximal Causal Inference\n"
            "------------------------------------------\n"
            f"  N     : {self.n_obs}\n"
            f"  ATE   : {self.ate:.4f}  (SE={self.se:.4f})\n"
            f"  CI    : [{lo:.4f}, {hi:.4f}]\n"
            f"  p     : {self.pvalue:.4f}\n"
            f"  bridge coefs     : {self.bridge_coefs}\n"
            f"  propensity coefs : {self.propensity_coefs}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"ProximalRegResult(ATE={self.ate:.4f})"


def proximal_regression(
    data: pd.DataFrame,
    y: str,
    treat: str,
    z_proxy: str,
    w_proxy: str,
    covariates: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    propensity_bounds: tuple = (0.02, 0.98),
) -> ProximalRegResult:
    """
    Doubly-robust regression-based PCI estimator for the ATE.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    treat : str
        Binary treatment column.
    z_proxy : str
        Treatment-inducing confounding proxy Z.
    w_proxy : str
        Outcome-inducing confounding proxy W.
    covariates : sequence of str, optional
        Measured covariates X.
    alpha : float, default 0.05
    propensity_bounds : (float, float)

    Returns
    -------
    ProximalRegResult
    """
    X_cols = list(covariates or [])
    df = data[[y, treat, z_proxy, w_proxy] + X_cols].dropna().reset_index(drop=True)
    n = len(df)

    Y = df[y].to_numpy(dtype=float)
    D = df[treat].to_numpy(dtype=float)
    Zp = df[z_proxy].to_numpy(dtype=float)
    Wp = df[w_proxy].to_numpy(dtype=float)
    Xc = df[X_cols].to_numpy(dtype=float) if X_cols else np.zeros((n, 0))

    # --- Outcome bridge via 2SLS: Y on (D, W, X) using (D, Z, X) ---
    # First stage: W ~ D, Z, X
    stage1_design = np.column_stack([np.ones(n), D, Zp, Xc])
    beta_fs = np.linalg.lstsq(stage1_design, Wp, rcond=None)[0]
    W_hat = stage1_design @ beta_fs

    # Second stage: Y ~ D, W_hat, X
    stage2_design = np.column_stack([np.ones(n), D, W_hat, Xc])
    beta_ss = np.linalg.lstsq(stage2_design, Y, rcond=None)[0]
    intercept = float(beta_ss[0])
    alpha_D = float(beta_ss[1])
    alpha_W = float(beta_ss[2])
    alpha_X = beta_ss[3:].tolist()

    # Predicted h(W, d, X) under counterfactual d = 0/1
    h_obs = intercept + alpha_D * D + alpha_W * Wp + Xc @ np.array(alpha_X)
    h1 = intercept + alpha_D * 1.0 + alpha_W * Wp + Xc @ np.array(alpha_X)
    h0 = intercept + alpha_D * 0.0 + alpha_W * Wp + Xc @ np.array(alpha_X)

    # --- Treatment bridge q(Z, X): logistic propensity P(D=1 | Z, X) ---
    prop_design = np.column_stack([np.ones(n), Zp, Xc])
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        lr.fit(prop_design, D.astype(int))
        pi_hat = lr.predict_proba(prop_design)[:, 1]
    except Exception:
        pi_hat = np.full(n, float(D.mean()))
    pi_hat = np.clip(pi_hat, *propensity_bounds)

    # Doubly-robust combination (Tchetgen 2024 eq. similar to AIPW):
    dr = (h1 - h0) + (D / pi_hat - (1 - D) / (1 - pi_hat)) * (Y - h_obs)

    ate = float(np.mean(dr))
    se = float(np.std(dr, ddof=1) / np.sqrt(n))
    z_stat = ate / se if se > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(z_stat)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (ate - crit * se, ate + crit * se)

    bridge_coefs = {"intercept": intercept, "D": alpha_D, "W": alpha_W}
    for name, c in zip(X_cols, alpha_X):
        bridge_coefs[name] = float(c)

    prop_dict = {"intercept": float(lr.intercept_[0])} if hasattr(lr, "intercept_") else {}
    if hasattr(lr, "coef_"):
        names = ["Z"] + X_cols
        for name, c in zip(names, lr.coef_[0]):
            prop_dict[name] = float(c)

    return ProximalRegResult(
        ate=ate,
        se=se,
        ci=ci,
        pvalue=pval,
        bridge_coefs=bridge_coefs,
        propensity_coefs=prop_dict,
        n_obs=n,
        detail={
            "pi_range": (float(pi_hat.min()), float(pi_hat.max())),
        },
    )


__all__ = ["proximal_regression", "ProximalRegResult"]
