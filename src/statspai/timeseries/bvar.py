"""Bayesian VAR with Minnesota (Litterman) prior.

The Minnesota prior shrinks towards a random-walk model: each variable's
own first lag coefficient is centred at 1, all other coefficients at 0,
with tightness controlled by hyperparameter λ₁ (overall tightness) and
λ₂ (cross-variable shrinkage).

With the error covariance fixed at an OLS estimate (Litterman's original
treatment; Stata ``bayes, minnfixedcovprior: var``) the coefficient
posterior is normal and is computed in closed form -- no MCMC required for
the posterior mean and credible intervals.

References
----------
Litterman, R.B. (1986). "Forecasting with Bayesian Vector Autoregressions."
  *JBE&S*, 4(1), 25–38.
Doan, T., Litterman, R. & Sims, C. (1984). "Forecasting and Conditional
  Projection Using Realistic Prior Distributions."
  *ECREV*, 3(1), 1–100.
Kilian, L. & Lütkepohl, H. (2017). *Structural Vector Autoregressive
  Analysis*. Cambridge. [@litterman1986forecasting]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin


@dataclass
class BVARResult(ResultProtocolMixin):
    """Posterior summary of a Bayesian VAR with Minnesota prior.

    Produced by :func:`bvar`. Holds the posterior-mean coefficient matrix
    and residual covariance, and exposes ``.forecast()``, ``.irf()`` and
    ``.summary()``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> T = 120
    >>> y1 = np.zeros(T); y2 = np.zeros(T)
    >>> for t in range(1, T):
    ...     y1[t] = 0.5 * y1[t - 1] + 0.2 * y2[t - 1] + rng.normal()
    ...     y2[t] = 0.3 * y2[t - 1] + 0.1 * y1[t - 1] + rng.normal()
    >>> df = pd.DataFrame({"gdp": y1, "inflation": y2})
    >>> res = sp.bvar(df, lags=2)
    >>> type(res).__name__
    'BVARResult'
    >>> res.coef.shape
    (5, 2)
    >>> res.lags
    2
    >>> isinstance(res.summary(), str)
    True
    """

    coef: np.ndarray  # (K*p + 1, K)  posterior mean B
    sigma: np.ndarray  # (K, K) posterior mean of Sigma
    fitted: np.ndarray  # (T-p, K)
    residuals: np.ndarray  # (T-p, K)
    var_names: list[str]
    lags: int
    n: int
    lambda1: float
    lambda2: float
    coef_sd: Optional[np.ndarray] = None  # posterior SD, same shape as coef

    def forecast(self, horizon: int = 8) -> pd.DataFrame:
        K = self.sigma.shape[0]
        p = self.lags
        B = self.coef
        # gather last p observations
        history = np.zeros((p, K))
        T_data = self.fitted.shape[0]
        for lag in range(p):
            idx = T_data - 1 - lag
            if idx >= 0:
                history[lag] = self.fitted[idx] + self.residuals[idx]
        fc = np.empty((horizon, K))
        for h in range(horizon):
            x = np.concatenate([history.ravel(), [1.0]])
            fc[h] = x @ B
            history = np.roll(history, 1, axis=0)
            history[0] = fc[h]
        cols = self.var_names
        return pd.DataFrame(fc, columns=cols, index=np.arange(1, horizon + 1))

    def irf(self, shock_var: int = 0, horizon: int = 20) -> np.ndarray:
        """Orthogonalised impulse responses (Cholesky decomposition)."""
        K = self.sigma.shape[0]
        p = self.lags
        B = self.coef[:-1]  # exclude constant row
        # Companion form
        A_mats = [B[k * K : (k + 1) * K].T for k in range(p)]
        chol = np.linalg.cholesky(self.sigma)
        irfs = np.zeros((horizon, K))
        irfs[0] = chol[:, shock_var]
        for h in range(1, horizon):
            # Simplified: multiply lag coefficients
            contrib = np.zeros(K)
            for j in range(min(h, p)):
                contrib += A_mats[j] @ irfs[h - 1 - j]
            irfs[h] = contrib
        return irfs

    def credible_interval(
        self,
        level: float = 0.90,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior credible interval for every coefficient.

        Returns ``(lower, upper)`` matrices the same shape as ``coef``. With
        the error covariance fixed at ``Sigma0`` the coefficient posterior is
        exactly normal; ``coef_sd`` holds its marginal standard deviations.
        """
        from scipy import stats

        if self.coef_sd is None:
            raise ValueError("posterior SD unavailable for this result")
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        return self.coef - z * self.coef_sd, self.coef + z * self.coef_sd

    def summary(self) -> str:
        K = len(self.var_names)
        lines = [
            "Bayesian VAR (Minnesota prior)",
            f"  lags = {self.lags}, K = {K}, T = {self.n}",
            f"  λ₁ = {self.lambda1}, λ₂ = {self.lambda2}",
            "",
            "Posterior mean coefficients (first 5 rows):",
            str(pd.DataFrame(self.coef[:5], columns=self.var_names).round(3)),
        ]
        if self.coef_sd is not None:
            lines += [
                "",
                "Posterior SD (first 5 rows):",
                str(
                    pd.DataFrame(
                        self.coef_sd[:5],
                        columns=self.var_names,
                    ).round(3)
                ),
            ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def bvar(
    data: pd.DataFrame,
    lags: int = 4,
    lambda1: float = 0.1,
    lambda2: float = 0.5,
    lambda3: float = 1.0,
    lambda4: float = 100.0,
    sigma: str = "ar",
) -> BVARResult:
    """Bayesian VAR with the original Minnesota (Litterman) prior.

    The error covariance is fixed at an estimate ``Sigma0`` and the VAR
    coefficients get independent normal priors centred on a random walk:

    * own lag l of variable i in equation i: mean 1 if l = 1 else 0,
      variance ``(lambda1 / l**lambda3)**2``;
    * lag l of variable j in equation i (j != i): mean 0, variance
      ``(s_i^2 / s_j^2) * (lambda1 * lambda2 / l**lambda3)**2``;
    * constant of equation i: mean 0, variance ``s_i^2 * (lambda1*lambda4)**2``,

    where ``s_i^2`` is the i-th diagonal element of ``Sigma0`` (see
    ``sigma``; by default the residual variance, divisor n, of a univariate
    AR(lags) with constant for variable i). With ``Sigma0`` known the posterior of the
    coefficients is exactly normal (closed form, no MCMC). This is the
    formulation of Stata's ``bayes, minnfixedcovprior: var`` (same
    defaults); ``sigma='var'`` is its ``minnfixedcovprior(varcov)``.

    Parameters
    ----------
    data : pd.DataFrame
        Columns are the endogenous variables.
    lags : int, default 4
    lambda1 : float, default 0.1
        Overall (self) tightness (smaller = stronger shrinkage toward RW).
    lambda2 : float, default 0.5
        Cross-variable tightness relative to own lags.
    lambda3 : float, default 1.0
        Lag decay.
    lambda4 : float, default 100.0
        Tightness of the constant (exogenous) terms.
    sigma : {'ar', 'var'}, default 'ar'
        ``Sigma0``: diagonal of the univariate AR(lags) residual variances
        (Litterman's original; Stata ``arcov``), or the diagonal of the OLS
        VAR residual variances ``diag(U'U) / n`` (Stata ``varcov``). The
        prior-variance scales ``s_i^2`` follow the same choice.

    Returns
    -------
    BVARResult
        Object with ``.forecast()``, ``.irf()`` and ``.summary()``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> T = 120
    >>> y1 = np.zeros(T); y2 = np.zeros(T)
    >>> for t in range(1, T):
    ...     y1[t] = 0.5 * y1[t - 1] + 0.2 * y2[t - 1] + rng.normal()
    ...     y2[t] = 0.3 * y2[t - 1] + 0.1 * y1[t - 1] + rng.normal()
    >>> df = pd.DataFrame({"gdp": y1, "inflation": y2})
    >>> res = sp.bvar(df, lags=2)
    >>> res.sigma.shape
    (2, 2)
    >>> fc = res.forecast(horizon=4)
    >>> list(fc.columns)
    ['gdp', 'inflation']
    >>> fc.shape
    (4, 2)

    References
    ----------
    [@litterman1986forecasting]
    """
    Y_raw = data.to_numpy(dtype=float)
    var_names = list(data.columns)
    T, K = Y_raw.shape

    # Build VAR design: Y = X B + E
    Y = Y_raw[lags:]  # (T-p, K)
    n = Y.shape[0]
    X_parts = []
    for lag in range(1, lags + 1):
        X_parts.append(Y_raw[lags - lag : T - lag])
    X = np.column_stack(X_parts + [np.ones(n)])  # (n, K*p + 1)
    m = X.shape[1]

    if sigma not in ("ar", "var"):
        raise ValueError("sigma must be 'ar' or 'var'")

    # Univariate AR(lags) residual variances on the VAR estimation sample
    s2 = np.empty(K)
    for k in range(K):
        Xk = np.column_stack(
            [Y_raw[lags - l_ : T - l_, k] for l_ in range(1, lags + 1)] + [np.ones(n)]
        )
        bk = np.linalg.lstsq(Xk, Y[:, k], rcond=None)[0]
        ek = Y[:, k] - Xk @ bk
        s2[k] = float(ek @ ek) / n
    if sigma == "ar":
        Sigma0 = np.diag(s2)
    else:
        B_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
        E_ols = Y - X @ B_ols
        Sigma0 = np.diag(np.diag(E_ols.T @ E_ols) / n)

    # Prior mean / variance, equation by equation (beta = vec(B)); the
    # variance scales s_i^2 are the diagonal of Sigma0.
    s2_ar = s2
    s2 = np.diag(Sigma0).copy()
    b0 = np.zeros((m, K))
    v0 = np.empty((m, K))
    for i in range(K):
        for lag in range(1, lags + 1):
            for j in range(K):
                idx = (lag - 1) * K + j
                if j == i:
                    v0[idx, i] = (lambda1 / lag**lambda3) ** 2
                    if lag == 1:
                        b0[idx, i] = 1.0
                else:
                    v0[idx, i] = (s2[i] / s2[j]) * (
                        lambda1 * lambda2 / lag**lambda3
                    ) ** 2
        v0[m - 1, i] = s2[i] * (lambda1 * lambda4) ** 2

    # Posterior with Sigma0 known:
    #   precision = Omega0^{-1} + Sigma0^{-1} (x) X'X
    #   mean      = precision^{-1} (Omega0^{-1} beta0 + vec(X'Y Sigma0^{-1}))
    # Before 1.28.x one prior-variance vector (the LAST equation's) was used
    # for every equation -- estimates depended on the column order -- and
    # the prior was scaled by the residual variance a second time.
    Sinv = np.linalg.inv(Sigma0)
    prec = np.diag(1.0 / v0.ravel(order="F")) + np.kron(Sinv, X.T @ X)
    rhs = b0.ravel(order="F") / v0.ravel(order="F") + (X.T @ Y @ Sinv).ravel(order="F")
    cov = np.linalg.inv(prec)
    B_post = (cov @ rhs).reshape((m, K), order="F")
    coef_sd = np.sqrt(np.clip(np.diag(cov), 0.0, None)).reshape((m, K), order="F")
    E_post = Y - X @ B_post
    Sigma_post = E_post.T @ E_post / n

    _result = BVARResult(
        coef=B_post,
        sigma=Sigma_post,
        fitted=X @ B_post,
        residuals=E_post,
        var_names=var_names,
        lags=lags,
        n=n,
        lambda1=lambda1,
        lambda2=lambda2,
        coef_sd=coef_sd,
    )
    _result.sigma0 = Sigma0
    _result.ar_variances = s2_ar
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.timeseries.bvar",
            params={
                "lags": lags,
                "lambda1": lambda1,
                "lambda2": lambda2,
                "lambda3": lambda3,
                "lambda4": lambda4,
                "sigma": sigma,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
