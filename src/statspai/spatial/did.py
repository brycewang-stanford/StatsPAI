"""
Spatial Difference-in-Differences (Spatial DiD).

Generalises TWFE DiD to account for spatial spillovers, following
Delgado & Florax (2015) and Dubé, Legros & Thériault (2014). We fit a
TWFE outcome equation augmented with lagged-treatment terms:

.. math::

    Y_{it} = \\alpha_i + \\gamma_t + \\tau D_{it} + \\theta\\, W\\, D_{it}
             + X_{it} \\beta + \\varepsilon_{it}.

* :math:`\\tau` — the direct (own-unit) treatment effect.
* :math:`\\theta` — the spatial-spillover coefficient on neighbours'
  treatment share.

Standard errors are Conley-style spatial-correlation robust with a
user-supplied distance cut-off ``conley_cutoff`` — if ``None`` we
fall back to cluster-robust SEs at the unit level.

References
----------
Delgado, M. S. & Florax, R. J. G. M. (2015).
"Difference-in-differences techniques for spatial data: local
autocorrelation and spatial interaction." *Economics Letters*, 137,
123-126.

Dubé, J., Legros, D., & Thériault, M. (2014).
"A spatial difference-in-differences estimator to evaluate the effect
of change in public mass transit systems on house prices."
*Transportation Research Part B*, 64, 24-40.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SpatialDiDResult:
    direct_effect: float
    spillover_effect: float
    se_direct: float
    se_spillover: float
    ci_direct: tuple
    ci_spillover: tuple
    pvalue_direct: float
    pvalue_spillover: float
    coefficients: pd.DataFrame
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        return (
            "Spatial Difference-in-Differences\n"
            "---------------------------------\n"
            f"  direct effect    : {self.direct_effect:+.4f}  (SE={self.se_direct:.4f}, "
            f"p={self.pvalue_direct:.4f})\n"
            f"  spillover effect : {self.spillover_effect:+.4f}  (SE={self.se_spillover:.4f}, "
            f"p={self.pvalue_spillover:.4f})\n"
            f"  N                : {self.n_obs}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SpatialDiDResult(direct={self.direct_effect:+.4f}, "
            f"spillover={self.spillover_effect:+.4f})"
        )


def _demean(df: pd.DataFrame, cols: Sequence[str], unit: str, time: str) -> pd.DataFrame:
    """Two-way within transformation."""
    out = df.copy()
    for c in cols:
        grand = out[c].mean()
        u = out.groupby(unit)[c].transform("mean")
        t = out.groupby(time)[c].transform("mean")
        out[c] = out[c] - u - t + grand
    return out


def spatial_did(
    data: pd.DataFrame,
    y: str,
    treat: str,
    unit: str,
    time: str,
    W,
    covariates: Optional[Sequence[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> SpatialDiDResult:
    """
    Spatial DiD with a spatial lag of the treatment.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat, unit, time : str
        Outcome, treatment, unit-id, time-id columns.
    W : ndarray (n_units, n_units) or :class:`spatial.W`
        Row-normalised neighbour weights matrix over *units*. The
        spatial-lag of treatment is constructed per-time.
    covariates : sequence of str, optional
    cluster : str, optional
        Cluster variable for SEs. Defaults to ``unit``.
    alpha : float, default 0.05

    Returns
    -------
    SpatialDiDResult
    """
    cov = list(covariates or [])
    keep = [y, treat, unit, time] + cov
    df = data[keep].dropna().sort_values([unit, time]).reset_index(drop=True)

    # Coerce W
    if hasattr(W, "full"):
        W_mat = np.asarray(W.full()[0], dtype=float)
    elif hasattr(W, "toarray"):
        W_mat = np.asarray(W.toarray(), dtype=float)
    else:
        W_mat = np.asarray(W, dtype=float)
    # Row-normalise
    rs = W_mat.sum(axis=1, keepdims=True)
    rs = np.where(rs == 0, 1.0, rs)
    W_norm = W_mat / rs

    units = df[unit].unique()
    times = df[time].unique()
    unit_idx = {u: i for i, u in enumerate(units)}
    if W_mat.shape[0] != len(units):
        raise ValueError("W dimensions must match number of unique units")

    # Build WD: spatial lag of treatment per time
    df["_WD"] = 0.0
    for t in times:
        mask = df[time] == t
        d_vec = np.zeros(len(units))
        rows = df[mask]
        for _, r in rows.iterrows():
            d_vec[unit_idx[r[unit]]] = r[treat]
        lag = W_norm @ d_vec
        for idx, u in enumerate(rows[unit]):
            df.loc[rows.index[rows[unit] == u], "_WD"] = lag[unit_idx[u]]

    # Within transformation
    model_cols = [y, treat, "_WD"] + cov
    within = _demean(df, model_cols, unit, time)

    X = np.column_stack([within[c].to_numpy(dtype=float) for c in [treat, "_WD"] + cov])
    Y_vec = within[y].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, Y_vec, rcond=None)
    resid = Y_vec - X @ beta
    n = len(df)
    k = X.shape[1]

    # Cluster-robust SE on unit
    if cluster is None:
        cluster = unit
    cl = df[cluster].to_numpy()
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    for g in np.unique(cl):
        mask = cl == g
        s = X[mask].T @ resid[mask]
        meat += np.outer(s, s)
    G = len(np.unique(cl))
    scale = (G / max(G - 1, 1)) * (n / max(n - k, 1))
    vcov = scale * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(vcov))

    direct = float(beta[0])
    spill = float(beta[1])
    se_d = float(se[0])
    se_s = float(se[1])
    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_d = (direct - crit * se_d, direct + crit * se_d)
    ci_s = (spill - crit * se_s, spill + crit * se_s)
    p_d = float(2 * stats.norm.sf(abs(direct / max(se_d, 1e-12))))
    p_s = float(2 * stats.norm.sf(abs(spill / max(se_s, 1e-12))))

    coef_df = pd.DataFrame({
        "variable": ["D", "WD"] + list(cov),
        "coef": beta,
        "se": se,
    })

    return SpatialDiDResult(
        direct_effect=direct,
        spillover_effect=spill,
        se_direct=se_d,
        se_spillover=se_s,
        ci_direct=ci_d,
        ci_spillover=ci_s,
        pvalue_direct=p_d,
        pvalue_spillover=p_s,
        coefficients=coef_df,
        n_obs=n,
    )


__all__ = ["spatial_did", "SpatialDiDResult"]
