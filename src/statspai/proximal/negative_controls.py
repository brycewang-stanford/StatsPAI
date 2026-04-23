"""
Negative control methods for unmeasured confounding.

Implements three closely related strands:

* :func:`negative_control_outcome` — Lipsitch et al. (2010) calibration
  test. Regress a *negative control outcome* (NCO, a variable that
  should NOT be caused by the treatment but shares confounders) on the
  treatment and inspect whether the coefficient is significantly
  non-zero. A large coefficient is evidence of residual confounding.

* :func:`negative_control_exposure` — complementary check using a
  negative control exposure (NCE), a pseudo-treatment assumed to have
  no effect on the outcome but sharing confounders with the true D.

* :func:`double_negative_control` — Miao, Geng & Tchetgen Tchetgen
  (2018), Shi, Miao & Tchetgen Tchetgen (2020, JASA) double-negative
  control estimator. Under a linear / index model, combining one NCE
  and one NCO point-identifies the ATE while removing latent bias.
  Implemented via a standard IV regression (NCE as instrument for the
  confounder, NCO as proxy in the outcome equation).

All routines return a :class:`CausalResult`-compatible object so they
slot into the ``sp.xxx`` surface next to :func:`evalue`, :func:`oster_bounds`
and :func:`rosenbaum_bounds`.

References
----------
Lipsitch, M., Tchetgen Tchetgen, E. J., & Cohen, T. (2010).
"Negative controls: a tool for detecting confounding and bias in
observational studies." *Epidemiology*, 21(3), 383-388.

Shi, X., Miao, W., & Tchetgen Tchetgen, E. J. (2020).
"A selective review of negative control methods in epidemiology."
*Current Epidemiology Reports*, 7, 190-202. [@shi2020selective]

Miao, W., Geng, Z., & Tchetgen Tchetgen, E. J. (2018).
"Identifying causal effects with proxy variables of an unmeasured
confounder." *Biometrika*, 105(4), 987-993. [@miao2018identifying]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class NegativeControlResult:
    """Unified result for negative-control procedures."""

    method: str
    estimate: float
    se: float
    pvalue: float
    ci: tuple
    alpha: float = 0.05
    n_obs: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""

    def summary(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Negative Control ({self.method})\n"
            f"  Estimate : {self.estimate:.4f}\n"
            f"  SE       : {self.se:.4f}\n"
            f"  p-value  : {self.pvalue:.4f}\n"
            f"  CI       : [{self.ci[0]:.4f}, {self.ci[1]:.4f}]\n"
            f"  n        : {self.n_obs}\n"
            f"  {self.interpretation}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NegativeControlResult(method='{self.method}', "
            f"est={self.estimate:.4f}, p={self.pvalue:.3g})"
        )


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _design(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Design matrix with intercept, dropping NaNs upstream."""
    n = len(df)
    if cols:
        X = df[list(cols)].to_numpy(dtype=float)
    else:
        X = np.zeros((n, 0))
    return np.column_stack([np.ones(n), X])


def _ols_with_se(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS with heteroskedasticity-robust (HC1) standard errors."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = X.T @ (X * (resid ** 2)[:, None])
    vcov = XtX_inv @ meat @ XtX_inv * (n / max(n - k, 1))
    se = np.sqrt(np.diag(vcov))
    return beta, se, vcov


# --------------------------------------------------------------------
# Lipsitch calibration (NCO)
# --------------------------------------------------------------------

def negative_control_outcome(
    data: pd.DataFrame,
    nco: str,
    treat: str,
    covariates: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> NegativeControlResult:
    """
    Lipsitch-style NCO calibration.

    Fit an OLS of the *negative-control outcome* ``nco`` on ``treat``
    and optional ``covariates``. A coefficient significantly different
    from zero signals residual confounding that the measured covariates
    failed to control for.

    Parameters
    ----------
    data : pd.DataFrame
    nco : str
        Negative-control outcome — a variable plausibly unaffected by
        the true treatment but sharing confounders with the real Y.
    treat : str
        Treatment indicator or exposure variable.
    covariates : sequence of str, optional
        Measured confounders to condition on.
    alpha : float, default 0.05

    Returns
    -------
    NegativeControlResult
    """
    cov = list(covariates or [])
    df = data[[nco, treat] + cov].dropna()
    y = df[nco].to_numpy(dtype=float)
    X = _design(df, [treat] + cov)
    beta, se, _ = _ols_with_se(y, X)
    est, sd = float(beta[1]), float(se[1])
    z = est / sd if sd > 0 else 0.0
    pval = float(2.0 * stats.norm.sf(abs(z)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (est - crit * sd, est + crit * sd)
    interp = (
        "Significant NCO coefficient => likely residual confounding."
        if pval < alpha
        else "NCO coefficient not significant => no direct evidence of residual confounding."
    )
    return NegativeControlResult(
        method="Lipsitch NCO calibration",
        estimate=est,
        se=sd,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=int(len(df)),
        detail={"beta": beta.tolist(), "covariates": cov},
        interpretation=interp,
    )


# --------------------------------------------------------------------
# Negative control exposure
# --------------------------------------------------------------------

def negative_control_exposure(
    data: pd.DataFrame,
    y: str,
    nce: str,
    covariates: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> NegativeControlResult:
    """
    Regress outcome on a *negative-control exposure*.

    A significant coefficient on ``nce`` — which by design is assumed to
    not causally affect ``y`` — indicates residual confounding along the
    exposure axis (selection, measurement error, etc.).
    """
    cov = list(covariates or [])
    df = data[[y, nce] + cov].dropna()
    yv = df[y].to_numpy(dtype=float)
    X = _design(df, [nce] + cov)
    beta, se, _ = _ols_with_se(yv, X)
    est, sd = float(beta[1]), float(se[1])
    z = est / sd if sd > 0 else 0.0
    pval = float(2.0 * stats.norm.sf(abs(z)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (est - crit * sd, est + crit * sd)
    interp = (
        "NCE coefficient is significant => possible residual confounding via exposure channel."
        if pval < alpha
        else "NCE coefficient not significant => no evidence of confounding via exposure proxy."
    )
    return NegativeControlResult(
        method="Lipsitch NCE calibration",
        estimate=est,
        se=sd,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=int(len(df)),
        detail={"beta": beta.tolist(), "covariates": cov},
        interpretation=interp,
    )


# --------------------------------------------------------------------
# Double negative control (Miao et al. 2018; Shi et al. 2020)
# --------------------------------------------------------------------

def double_negative_control(
    data: pd.DataFrame,
    y: str,
    treat: str,
    nce: str,
    nco: str,
    covariates: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> NegativeControlResult:
    """
    Double negative control estimator (Miao et al. 2018; Shi et al. 2020).

    Under the linear / index model::

        Y      = α0 + α_D D + α_U U + α_X X + ε_Y
        NCO    = β0 + β_U U + β_X X + ε_W
        E[U | NCE, X, D] linear in (NCE, X, D)

    (plus standard independence/exclusion conditions), the ATE is
    point-identified by IV-regressing Y on (D, NCO, X) using (D, NCE, X)
    as instruments: NCE instruments for the proxy NCO, breaking the
    dependence on U. The coefficient on D is the de-biased ATE.

    This is implemented as a just-identified 2SLS. The fitted ATE is
    asymptotically unbiased under the assumptions above and consistent
    with Shi et al. (2020, §3) closed-form.
    """
    cov = list(covariates or [])
    df = data[[y, treat, nce, nco] + cov].dropna()
    yv = df[y].to_numpy(dtype=float)
    D = df[treat].to_numpy(dtype=float)
    W = df[nco].to_numpy(dtype=float)
    Z = df[nce].to_numpy(dtype=float)
    Xcov = df[cov].to_numpy(dtype=float) if cov else np.zeros((len(df), 0))
    n = len(df)

    # Endogenous block: [D, W]; Exogenous covs: [1, X]; Instruments: [1, X, D, Z]
    exog = np.column_stack([np.ones(n), Xcov])
    endog = np.column_stack([D, W])
    instruments = np.column_stack([exog, D, Z])

    # First stage: project [D, W] on [exog, D, Z]
    # For D, the IV set includes D itself (identity fit). For W, Z is the instrument.
    # Use standard 2SLS: X_hat = Z(Z'Z)^{-1} Z' X_full
    X_full = np.column_stack([endog, exog])
    Z_full = np.column_stack([instruments, exog[:, 1:]]) if exog.shape[1] > 1 else instruments
    # Keep it simple — use full instrument matrix = [1, X, D, Z]
    Z_mat = instruments
    PZ = Z_mat @ np.linalg.pinv(Z_mat.T @ Z_mat) @ Z_mat.T
    X_hat = PZ @ X_full
    beta_2sls = np.linalg.pinv(X_hat.T @ X_full) @ X_hat.T @ yv
    resid = yv - X_full @ beta_2sls
    XtX_inv = np.linalg.pinv(X_hat.T @ X_full)
    meat = X_hat.T @ (X_hat * (resid ** 2)[:, None])
    vcov = XtX_inv @ meat @ XtX_inv.T * (n / max(n - X_full.shape[1], 1))
    se = np.sqrt(np.diag(vcov))

    ate = float(beta_2sls[0])  # coefficient on D
    ate_se = float(se[0])
    z_stat = ate / ate_se if ate_se > 0 else 0.0
    pval = float(2.0 * stats.norm.sf(abs(z_stat)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (ate - crit * ate_se, ate + crit * ate_se)

    interp = "Double-negative-control ATE with latent-bias correction."

    return NegativeControlResult(
        method="Double Negative Control (Miao-Geng-Tchetgen 2018)",
        estimate=ate,
        se=ate_se,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail={
            "coefficients": dict(zip(
                ["D", "NCO"] + ["const"] + cov, beta_2sls.tolist()
            )),
            "z_stat": z_stat,
        },
        interpretation=interp,
    )


__all__ = [
    "negative_control_outcome",
    "negative_control_exposure",
    "double_negative_control",
    "NegativeControlResult",
]
