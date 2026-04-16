"""Recentered Influence Function (RIF) regression and decomposition
(Firpo, Fortin & Lemieux, *Econometrica* 2009).

The RIF of a distributional statistic ``ν(F)`` evaluated at observation
``y_i`` is

    RIF(y_i; ν, F) = ν(F) + IF(y_i; ν, F)

where ``IF`` is the (classical) influence function.

**RIF-OLS** regresses RIF values on ``X`` to estimate the *unconditional
quantile partial effect* (UQPE) — how a small shift in ``X`` affects the
quantile of the unconditional distribution, not the conditional distribution.

**RIF-Oaxaca-Blinder** decomposes the between-group difference in ``ν`` into
explained (covariate endowment) and unexplained (coefficient) components at
any chosen distributional statistic, not just the mean.

Supported statistics: ``quantile(τ)``, ``variance``, ``gini``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


StatisticKind = Literal["quantile", "variance", "gini"]


# --------------------------------------------------------------------- #
#  Influence function construction
# --------------------------------------------------------------------- #

def _kernel_density_at(y: np.ndarray, point: float, bw: str = "silverman") -> float:
    """Kernel density estimate at a single point."""
    try:
        kde = sp_stats.gaussian_kde(y, bw_method=bw)
        return float(kde(point))
    except Exception:
        # fallback: histogram-based density
        h = 1.06 * y.std() * len(y) ** (-0.2)
        return float(np.mean(np.exp(-0.5 * ((y - point) / h) ** 2)) / (h * np.sqrt(2 * np.pi)))


def rif_values(y: np.ndarray, statistic: StatisticKind = "quantile",
               tau: float = 0.5) -> np.ndarray:
    """Compute the RIF of each observation.

    Parameters
    ----------
    y : (n,) array
    statistic : {"quantile", "variance", "gini"}
    tau : float
        Quantile level (only used when ``statistic="quantile"``).
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if statistic == "quantile":
        q = float(np.quantile(y, tau))
        f_q = _kernel_density_at(y, q)
        f_q = max(f_q, 1e-12)
        return q + (tau - (y <= q).astype(float)) / f_q
    if statistic == "variance":
        mu = y.mean()
        return (y - mu) ** 2
    if statistic == "gini":
        # Gini influence function (Firpo et al. 2009 Appendix A)
        mu = y.mean()
        n_obs = len(y)
        ranks = sp_stats.rankdata(y, method="average") / n_obs
        G = float(2 * np.cov(y, ranks)[0, 1] / mu)    # Gini coefficient
        rif = (
            -G
            + (2 / mu) * (ranks * y - np.cumsum(np.sort(y)) / n_obs)
            - (2 * y / mu) * (ranks - (G + 1) / 2)
        )
        return G + rif
    raise ValueError(f"unknown statistic {statistic!r}")


# --------------------------------------------------------------------- #
#  RIF-OLS
# --------------------------------------------------------------------- #

@dataclass
class RIFResult:
    params: pd.Series
    std_errors: pd.Series
    statistic: str
    tau: float
    nobs: int

    def summary(self) -> str:
        lines = [
            f"RIF-OLS — unconditional {self.statistic}"
            + (f" (τ={self.tau})" if self.statistic == "quantile" else ""),
            "-" * 50,
            f"n = {self.nobs}",
            "",
        ]
        for nm in self.params.index:
            b = self.params[nm]; se = self.std_errors[nm]
            t = b / se if np.isfinite(se) and se > 0 else np.nan
            lines.append(f"  {nm:<15s}  {b: .4f}  (SE {se: .4f}, t {t: .3f})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def rifreg(
    formula: str,
    data: pd.DataFrame,
    statistic: StatisticKind = "quantile",
    tau: float = 0.5,
) -> RIFResult:
    """RIF regression (Firpo, Fortin & Lemieux 2009).

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2"`` style.
    data : pd.DataFrame
    statistic : {"quantile", "variance", "gini"}
    tau : float
        Quantile level (default 0.5 = median UQPE).
    """
    if "~" not in formula:
        raise ValueError("formula must contain '~'")
    dep, rhs = [s.strip() for s in formula.split("~", 1)]
    indep = [v.strip() for v in rhs.split("+") if v.strip()]
    df = data[[dep] + indep].dropna()
    y = df[dep].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df))] + [df[v].to_numpy(float) for v in indep])
    names = ["Intercept"] + indep

    rif = rif_values(y, statistic=statistic, tau=tau)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ rif)
    e = rif - X @ beta
    sigma2 = float(e @ e) / (len(rif) - len(beta))
    se = np.sqrt(np.diag(sigma2 * XtX_inv))

    return RIFResult(
        params=pd.Series(beta, index=names),
        std_errors=pd.Series(se, index=names),
        statistic=statistic,
        tau=tau,
        nobs=len(df),
    )


# --------------------------------------------------------------------- #
#  RIF-Oaxaca-Blinder decomposition
# --------------------------------------------------------------------- #

@dataclass
class RIFDecompositionResult:
    total_diff: float
    explained: float
    unexplained: float
    detailed: pd.DataFrame          # per-covariate explained shares
    statistic: str
    tau: float

    def summary(self) -> str:
        lines = [
            f"RIF Oaxaca-Blinder Decomposition — {self.statistic}"
            + (f" (τ={self.tau})" if self.statistic == "quantile" else ""),
            "-" * 55,
            f"Total diff    : {self.total_diff: .4f}",
            f"Explained     : {self.explained: .4f}",
            f"Unexplained   : {self.unexplained: .4f}",
            "",
            "Detailed (explained portion):",
            self.detailed.round(4).to_string(index=False),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def rif_decomposition(
    formula: str,
    data: pd.DataFrame,
    group: str,
    statistic: StatisticKind = "quantile",
    tau: float = 0.5,
    reference: int = 0,
) -> RIFDecompositionResult:
    """RIF Oaxaca-Blinder decomposition (FFL 2009, Section 5).

    Decomposes the between-group difference in a distributional
    statistic into explained (covariate endowment) and unexplained
    (coefficient) parts at the chosen statistic.

    Parameters
    ----------
    group : str
        Binary (0/1) group indicator column.
    reference : int, default 0
        Which group's coefficients to use as the reference (0 or 1).
    """
    if "~" not in formula:
        raise ValueError("formula must contain '~'")
    dep, rhs = [s.strip() for s in formula.split("~", 1)]
    indep = [v.strip() for v in rhs.split("+") if v.strip()]
    df = data[[dep, group] + indep].dropna()
    g = df[group].to_numpy().astype(int)
    y0 = df.loc[g == 0, dep].to_numpy(float)
    y1 = df.loc[g == 1, dep].to_numpy(float)

    rif0 = rif_values(y0, statistic=statistic, tau=tau)
    rif1 = rif_values(y1, statistic=statistic, tau=tau)

    X0 = np.column_stack([np.ones(len(y0))] + [df.loc[g == 0, v].to_numpy(float) for v in indep])
    X1 = np.column_stack([np.ones(len(y1))] + [df.loc[g == 1, v].to_numpy(float) for v in indep])
    names = ["Intercept"] + indep

    beta0 = np.linalg.lstsq(X0, rif0, rcond=None)[0]
    beta1 = np.linalg.lstsq(X1, rif1, rcond=None)[0]

    total_diff = float(rif1.mean() - rif0.mean())
    if reference == 0:
        explained = float((X1.mean(axis=0) - X0.mean(axis=0)) @ beta0)
        unexplained = total_diff - explained
        detail_vals = (X1.mean(axis=0) - X0.mean(axis=0)) * beta0
    else:
        explained = float((X1.mean(axis=0) - X0.mean(axis=0)) @ beta1)
        unexplained = total_diff - explained
        detail_vals = (X1.mean(axis=0) - X0.mean(axis=0)) * beta1

    detail = pd.DataFrame({
        "variable": names,
        "explained": detail_vals,
    })

    return RIFDecompositionResult(
        total_diff=total_diff,
        explained=explained,
        unexplained=unexplained,
        detailed=detail,
        statistic=statistic,
        tau=tau,
    )
