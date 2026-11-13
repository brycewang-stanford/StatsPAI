"""
Inference with many weak instruments (Mikusheva & Sun 2024).

When the number of instruments :math:`K` is large relative to the
sample size :math:`n` and each instrument is weak (low partial R² with
the endogenous regressor), 2SLS is severely biased and conventional
asymptotic inference fails. Mikusheva & Sun (2024, *Econometrica*)
propose a *jackknife-Anderson-Rubin* (JAR) statistic that is pivotal
under many-weak-IV asymptotics:

.. math::

   JAR(\\beta_0) = \\frac{(n - K) \\cdot \\hat\\Omega^{-1/2}\\hat\\Omega^{-1/2} \\sum_i z_i \\tilde\\varepsilon_i(\\beta_0)}
                        {\\cdots}

For simplicity we report the **leave-one-out jackknife IV estimator
(JIVE)** and a **many-weak-robust Anderson-Rubin test** by grid
inversion. The implementation is a minimal, self-contained variant of
Mikusheva-Sun's key ideas suitable for applied work:

1. **JIVE** point estimator (Angrist, Imbens & Krueger 1999; Phillips-
   Hale 2018), which has smaller finite-sample bias than 2SLS in the
   many-IV regime.
2. **Anderson-Rubin confidence set** for the structural parameter
   (:func:`many_weak_ar`): the classical statistic with its ``F``
   reference, inverted exactly. Its level does not depend on instrument
   strength; it is not the jackknife AR statistic of Mikusheva and Sun,
   which is not implemented here.

References
----------
Mikusheva, A., & Sun, L. (2024). "Inference with many weak
instruments." *Econometrica*, 92(2), forthcoming. [@mikusheva2024weak]

Angrist, J. D., Imbens, G. W., & Krueger, A. B. (1999). "Jackknife
instrumental variables estimation." *JAE*, 14(1), 57-67. [@angrist1999jackknife]

Phillips, G. D. A., & Hale, C. (2018). "The jackknife estimator: bias
and variance in the IV model." *JAE*, 33(6), 871-887.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin


@dataclass
class ManyWeakIVResult(ResultProtocolMixin):
    estimator: str
    estimate: float
    se: float
    ci: tuple
    n_obs: int
    n_instruments: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lo, hi = self.ci
        return (
            f"Many-Weak IV ({self.estimator})\n"
            "-------------------------------\n"
            f"  N               : {self.n_obs}\n"
            f"  K (instruments) : {self.n_instruments}\n"
            f"  estimate        : {self.estimate:+.4f}\n"
            f"  SE              : {self.se:.4f}\n"
            f"  95% CI          : [{lo:.4f}, {hi:.4f}]"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"ManyWeakIVResult({self.estimator}, " f"estimate={self.estimate:+.4f})"


def jive(
    data: pd.DataFrame,
    y: str,
    endog: str,
    instruments: Sequence[str],
    exog: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> ManyWeakIVResult:
    """
    Jackknife Instrumental Variables Estimator (AIK 1999; Phillips-Hale
    2018 variant). Less biased than 2SLS when K/n is not small.

    Returns
    -------
    ManyWeakIVResult
    """
    exog = list(exog or [])
    df = data[[y, endog] + list(instruments) + exog].dropna().reset_index(drop=True)
    n = len(df)
    Y = df[y].to_numpy(dtype=float)
    D = df[endog].to_numpy(dtype=float)
    Z = df[list(instruments)].to_numpy(dtype=float)
    Xc = df[exog].to_numpy(dtype=float) if exog else np.zeros((n, 0))

    # Expand instrument matrix with exogenous regressors.
    Z_full = np.column_stack([np.ones(n), Z, Xc])
    # Leave-one-out fitted D: (I - H_i) predictions
    # Projection and leverages without the n x n PZ (O(n^2) memory).
    ZtZ_pinv = np.linalg.pinv(Z_full.T @ Z_full)
    D_hat = Z_full @ (ZtZ_pinv @ (Z_full.T @ D))
    h_ii = np.einsum("ij,jk,ik->i", Z_full, ZtZ_pinv, Z_full)
    h_ii = np.clip(h_ii, a_max=0.999, a_min=None)
    D_jack = (D_hat - h_ii * D) / (1 - h_ii)

    # Second stage: D_jack is the instrument for D, not its replacement.
    # beta = (X2'X)^{-1} X2'Y with X = (D, 1, X) (Angrist-Imbens-Krueger
    # 1999); residuals are taken at the observed D. Regressing Y on D_jack
    # is attenuated toward zero under many instruments.
    X2 = np.column_stack([D_jack, np.ones(n), Xc])
    X_obs = np.column_stack([D, np.ones(n), Xc])
    bread = np.linalg.pinv(X2.T @ X_obs)
    beta = bread @ X2.T @ Y
    resid = Y - X_obs @ beta
    vcov = bread @ ((X2 * (resid**2)[:, None]).T @ X2) @ bread.T
    se = np.sqrt(np.diag(vcov))
    estimate = float(beta[0])
    se_est = float(se[0])
    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (estimate - crit * se_est, estimate + crit * se_est)

    _result = ManyWeakIVResult(
        estimator="JIVE",
        estimate=estimate,
        se=se_est,
        ci=ci,
        n_obs=n,
        n_instruments=int(len(instruments)),
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.iv.many_weak_jive",
            params={
                "y": y,
                "endog": endog,
                "instruments": list(instruments),
                "exog": list(exog) if exog else None,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def many_weak_ar(
    data: pd.DataFrame,
    y: str,
    endog: str,
    instruments: Sequence[str],
    exog: Optional[Sequence[str]] = None,
    beta_grid: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
) -> ManyWeakIVResult:
    """
    Anderson-Rubin confidence set for one endogenous regressor.

    The set of ``beta`` for which the instruments do not explain
    ``y - beta * endog``: ``AR(beta) = (e'P e / K) / (e'M e / (n - K - p))``
    compared with ``F(K, n - K - p)``, where ``e`` is the residual after
    partialling out a constant and ``exog``, ``P`` projects on the
    partialled-out instruments and ``p`` counts the constant and ``exog``.
    Its level does not depend on the strength of the instruments. It is
    the classical statistic with a homoskedastic variance, exact under
    normal errors; the jackknife AR of Mikusheva and Sun (2024), which
    also allows heteroskedasticity, is not implemented.

    Parameters
    ----------
    data : pd.DataFrame
    y, endog : str
    instruments : sequence of str
    exog : sequence of str, optional
        Included exogenous regressors. A constant is always included.
    beta_grid : sequence of float, optional
        Candidate values. Without it the set is found exactly: the
        acceptance condition is a quadratic inequality in ``beta``. With
        it, the set is the accepted grid points and ``detail`` says
        whether it touches an end of the grid.
    alpha : float, default 0.05

    Returns
    -------
    ManyWeakIVResult
        ``ci`` is the set when it is an interval. When the set is the whole
        line or the outside of an interval (weak instruments), ``ci`` is
        ``(-inf, inf)`` and ``detail['shape']`` is ``'unbounded'`` or
        ``'disjoint'`` with the two finite end points in
        ``detail['roots']``. An empty set (the over-identifying
        restrictions are rejected at every ``beta``) gives ``ci`` of
        ``(nan, nan)``. ``estimate`` minimises the statistic (LIML).

    Notes
    -----
    Before this version the default searched 101 points within 0.5 of the
    OLS estimate, left out the constant, and reported the smallest and
    largest accepted grid point. With an OLS bias near 0.5 the true value
    was outside the grid: in a design with 30 instruments and a
    concentration parameter of 60 the reported 95% set covered 70% of the
    time (``tests/reliability/se_screen/iv.py``).
    """
    exog = list(exog or [])
    df = data[[y, endog] + list(instruments) + exog].dropna().reset_index(drop=True)
    n = len(df)
    Y = df[y].to_numpy(dtype=float)
    D = df[endog].to_numpy(dtype=float)
    Z = df[list(instruments)].to_numpy(dtype=float)
    # Partial the constant and the included exogenous regressors out of
    # everything.
    Xc = np.column_stack([np.ones(n)] + [df[c].to_numpy(dtype=float) for c in exog])
    coef_y, *_ = np.linalg.lstsq(Xc, np.column_stack([Y, D, Z]), rcond=None)
    resid_all = np.column_stack([Y, D, Z]) - Xc @ coef_y
    Y, D, Z = resid_all[:, 0], resid_all[:, 1], resid_all[:, 2:]
    # Orthonormal basis of the instrument space (rank-deficient Z allowed).
    U, sv, _ = np.linalg.svd(Z, full_matrices=False)
    K = (
        int((sv > sv.max() * max(Z.shape) * np.finfo(float).eps).sum())
        if sv.size
        else 0
    )
    df_resid = n - K - Xc.shape[1]
    if K == 0 or df_resid <= 0:
        from ..exceptions import DataInsufficient

        raise DataInsufficient(
            f"many_weak_ar: {n} rows leave no residual degrees of freedom "
            f"for {K} instruments and {Xc.shape[1]} exogenous terms.",
            recovery_hint="Use fewer instruments or more observations.",
        )
    Q = U[:, :K]
    qy, qd = Q.T @ Y, Q.T @ D
    pyy, pyd, pdd = float(qy @ qy), float(qy @ qd), float(qd @ qd)
    myy, myd, mdd = float(Y @ Y) - pyy, float(Y @ D) - pyd, float(D @ D) - pdd
    f_crit = float(stats.f.ppf(1 - alpha, K, df_resid))
    c = K * f_crit / df_resid

    def ar_stat(b: float) -> float:
        num = pyy - 2 * b * pyd + b * b * pdd
        den = myy - 2 * b * myd + b * b * mdd
        return float(num / K / (den / df_resid)) if den > 0 else np.inf

    # e'Pe - c e'Me <= 0 is  qa b^2 - 2 qb b + qc <= 0.
    qa, qb, qc = pdd - c * mdd, pyd - c * myd, pyy - c * myy
    # The statistic is minimised at the LIML estimate: the smaller root of
    # the generalised eigenproblem, in closed form for one regressor.
    ea = pyd * mdd - pdd * myd
    eb = pyy * mdd - pdd * myy
    ec = pyy * myd - pyd * myy
    cands = [float(r.real) for r in np.roots([ea, -eb, ec]) if abs(r.imag) < 1e-12]
    point = min(cands, key=ar_stat) if cands else float(pyd / pdd)
    detail: Dict[str, Any] = {
        "statistic_at_estimate": ar_stat(point),
        "critical_value": f_crit,
        "df": (K, df_resid),
    }
    if beta_grid is not None:
        beta_values = np.asarray(beta_grid, dtype=float)
        stats_grid = np.array([ar_stat(float(b)) for b in beta_values])
        accepted = beta_values[stats_grid <= f_crit]
        detail["n_grid"] = int(len(beta_values))
        point = float(beta_values[int(np.argmin(stats_grid))])
        if accepted.size > 0:
            lo, hi = float(accepted.min()), float(accepted.max())
            detail["shape"] = "grid"
            detail["touches_grid_end"] = bool(
                lo <= beta_values.min() or hi >= beta_values.max()
            )
        else:
            lo = hi = float("nan")
            detail["shape"] = "empty"
    else:
        disc = qb * qb - qa * qc
        if disc < 0:
            # No real root: the inequality holds everywhere or nowhere.
            if qa < 0:
                lo, hi = -np.inf, np.inf
                detail["shape"] = "unbounded"
            else:
                lo = hi = float("nan")
                detail["shape"] = "empty"
        else:
            r1 = (qb - np.sqrt(disc)) / qa
            r2 = (qb + np.sqrt(disc)) / qa
            r_lo, r_hi = float(min(r1, r2)), float(max(r1, r2))
            if qa > 0:
                lo, hi = r_lo, r_hi
                detail["shape"] = "interval"
            else:
                lo, hi = -np.inf, np.inf
                detail["shape"] = "disjoint"
                detail["roots"] = (r_lo, r_hi)
    finite = np.isfinite(lo) and np.isfinite(hi) and hi > lo
    _result = ManyWeakIVResult(
        estimator="Anderson-Rubin confidence set",
        estimate=point,
        se=(hi - lo) / (2 * 1.96) if finite else float("nan"),
        ci=(lo, hi),
        n_obs=n,
        n_instruments=K,
        detail=detail,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.iv.many_weak_ar",
            params={
                "y": y,
                "endog": endog,
                "instruments": list(instruments),
                "exog": list(exog) if exog else None,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


__all__ = ["jive", "many_weak_ar", "ManyWeakIVResult"]
