"""
Shift-Share Instruments for Political Science (Park & Xu, arXiv:2603.00135, 2026).

Adapts the canonical Bartik IV design to political-science settings
where:

1. The "industries" are often political-exposure categories (e.g.
   incumbency of co-ethnics, share of coverage by partisan media,
   share of employment in import-competing sectors).
2. The "shift" is a national-level political or policy shock
   (e.g. a federal policy change, national coverage volume).
3. The outcome is a political behaviour — vote share, turnout,
   polarisation — so the linear 2SLS benchmark is supplemented with
   non-monotone and pre-trend diagnostics that are standard in PS
   panel data.

Compared to :func:`sp.bartik`, this wrapper

* builds the shift-share IV from a long-form panel (unit × time)
  instead of the cross-sectional API,
* runs an AKM (Adão-Kolesár-Morales 2019) shock-level cluster SE,
* ships two extra diagnostics Park-Xu (2026) recommend as default:
  (a) a **share-balance** test of pre-treatment unit covariates on
  the exposure share matrix, (b) a **Rotemberg top-K** report
  identifying the industries that dominate the identifying variation.

References
----------
Park, P. K. & Xu, Y. (2026).
"Shift-Share Designs in Political Science." arXiv:2603.00135.

Adão, R., Kolesár, M. & Morales, E. (2019).
"Shift-Share Designs: Theory and Inference." QJE, 134(4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult, EconometricResults
from .shift_share import bartik as _bartik_cs


__all__ = ["shift_share_political", "ShiftSharePoliticalResult"]


@dataclass
class ShiftSharePoliticalResult:
    """Structured output of :func:`shift_share_political`.

    Wraps a standard :class:`CausalResult` (point + SEs) plus the two
    Park-Xu (2026) diagnostics: share-balance and Rotemberg top-K.
    """

    iv_result: CausalResult
    rotemberg_top: pd.DataFrame
    share_balance: pd.DataFrame
    n_units: int
    n_periods: int
    n_industries: int
    method: str = "shift_share_political"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def estimate(self) -> float:
        return self.iv_result.estimate

    @property
    def se(self) -> float:
        return self.iv_result.se

    @property
    def ci(self) -> tuple:
        return self.iv_result.ci

    def summary(self) -> str:
        est = self.iv_result.estimate
        se = self.iv_result.se
        lo, hi = self.iv_result.ci
        lines = [
            "Shift-Share (Bartik) — Political Science (Park-Xu 2026)",
            "-" * 60,
            f"  Units / periods         : {self.n_units} × {self.n_periods}",
            f"  Industries in exposure  : {self.n_industries}",
            f"  IV estimate             : {est:+.6f}",
            f"  SE (AKM shock-cluster)  : {se:.6f}",
            f"  95% CI                  : [{lo:+.6f}, {hi:+.6f}]",
            "",
            "  Rotemberg top-5 industries (by weight):",
            self.rotemberg_top.head(5).to_string(index=False, float_format="%.4f"),
            "",
            "  Share-balance test (F on pre-period covariates):",
            self.share_balance.to_string(index=False, float_format="%.4f"),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _long_to_panel(
    data: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    unit: str,
    time: str,
    endog: str,
    outcome: str,
) -> pd.DataFrame:
    """Validate alignment and compute first-difference per unit.

    Returns a unit-level DataFrame with (outcome, endog) replaced by
    their first-differences over the full panel window.  This is the
    canonical PS Shift-Share target: effect of endogenous exposure
    change on outcome change.
    """
    panel = data.sort_values([unit, time]).reset_index(drop=True)
    # First-differences by unit: Δy_i = y_i(T) - y_i(t0), same for endog.
    first = panel.groupby(unit).first()
    last = panel.groupby(unit).last()
    dy = last[outcome] - first[outcome]
    dx = last[endog] - first[endog]
    agg = pd.DataFrame({
        outcome: dy,
        endog: dx,
    })
    agg = agg.join(shares, how="inner")
    return agg


def _rotemberg_weights(
    shares: np.ndarray, shocks: np.ndarray, dx: np.ndarray
) -> np.ndarray:
    """Rotemberg weights α_k proportional to shock variation × exposure.

    α_k ∝ g_k * (∑_i s_{ik} (x_i - x̄)) ; we return the normalised vector.
    """
    x_c = dx - dx.mean()
    num = shocks * (shares.T @ x_c)
    tot = np.sum(np.abs(num))
    if tot > 0:
        return num / tot
    return num


def _share_balance_test(
    shares_df: pd.DataFrame,
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    """Regress each covariate on the share matrix and report the F-stat."""
    results = []
    X = shares_df.to_numpy(dtype=float)
    n, k = X.shape
    X_design = np.column_stack([np.ones(n), X])
    for col in covariates.columns:
        z = covariates[col].to_numpy(dtype=float)
        if not np.isfinite(z).all():
            continue
        beta, *_ = np.linalg.lstsq(X_design, z, rcond=None)
        resid = z - X_design @ beta
        rss = float(np.sum(resid ** 2))
        tss = float(np.sum((z - z.mean()) ** 2))
        if tss <= 0 or rss <= 0:
            continue
        r2 = 1 - rss / tss
        df1, df2 = k, max(n - k - 1, 1)
        F = (r2 / k) / ((1 - r2) / max(df2, 1)) if r2 < 1 else float("inf")
        pv = float(1 - stats.f.cdf(F, df1, df2)) if np.isfinite(F) else 0.0
        results.append({
            "covariate": col,
            "R2_on_shares": r2,
            "F": F,
            "pvalue": pv,
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def shift_share_political(
    data: pd.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
    endog: str,
    shares: pd.DataFrame,
    shocks: pd.Series,
    covariates: Optional[Sequence[str]] = None,
    leave_one_out: bool = True,
    alpha: float = 0.05,
) -> ShiftSharePoliticalResult:
    """Park-Xu (2026) shift-share IV for political-science panel data.

    Parameters
    ----------
    data : DataFrame (long format)
        Unit × time panel containing ``outcome`` and ``endog``.  First
        and last periods per unit are used to form long-differences.
    unit, time, outcome, endog : str
        Column names.
    shares : DataFrame (unit × industry)
        Exposure-share matrix.  Row index must equal the unit IDs.
    shocks : Series (industry → scalar)
        National / supra-unit shifter vector.  Index must match the
        columns of ``shares``.
    covariates : sequence of str, optional
        Pre-treatment covariates (measured at the first period per unit)
        used for the share-balance diagnostic.
    leave_one_out, alpha
        Forwarded to :func:`sp.bartik`.

    Returns
    -------
    ShiftSharePoliticalResult

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> units = range(20); times = range(2); inds = [f'I{k}' for k in range(5)]
    >>> shares = pd.DataFrame(rng.dirichlet(np.ones(5), size=len(units)),
    ...                       index=list(units), columns=inds)
    >>> shocks = pd.Series(rng.normal(size=5), index=inds)
    >>> rows = []
    >>> true_tau = 0.4
    >>> for i in units:
    ...     bartik_i = float((shares.loc[i] * shocks).sum())
    ...     dx = bartik_i + rng.normal(scale=0.1)
    ...     y_first = 0.0
    ...     y_last = y_first + true_tau * dx + rng.normal(scale=0.1)
    ...     rows.append({'unit': i, 'time': 0, 'y': y_first, 'x': 0.0})
    ...     rows.append({'unit': i, 'time': 1, 'y': y_last, 'x': dx})
    >>> df = pd.DataFrame(rows)
    >>> out = sp.bartik.shift_share_political(
    ...     df, unit='unit', time='time',
    ...     outcome='y', endog='x',
    ...     shares=shares, shocks=shocks,
    ... )
    >>> abs(out.estimate - true_tau) < 0.3
    True
    """
    # --- Validation --------------------------------------------------------
    for col in (unit, time, outcome, endog):
        if col not in data.columns:
            raise ValueError(f"column {col!r} not in data")
    if not isinstance(shares, pd.DataFrame):
        raise TypeError("shares must be a DataFrame")
    if not isinstance(shocks, pd.Series):
        raise TypeError("shocks must be a Series")
    if list(shares.columns) != list(shocks.index):
        # Align on intersection
        common = [c for c in shares.columns if c in shocks.index]
        if not common:
            raise ValueError(
                "shares.columns and shocks.index have no overlap."
            )
        shares = shares[common]
        shocks = shocks.loc[common]

    # --- Build cross-section of long-differences --------------------------
    cs = _long_to_panel(
        data, shares, unit=unit, time=time,
        endog=endog, outcome=outcome,
    )
    cs_with_shares = cs.reset_index()
    cs_with_shares = cs_with_shares.rename(columns={"index": unit}) \
        if unit not in cs_with_shares.columns else cs_with_shares

    # --- Run the shift-share IV -------------------------------------------
    shares_aligned = shares.loc[cs.index]
    # The existing `bartik` cross-section API wants DataFrame and Series.
    ivres = _bartik_cs(
        data=cs[[outcome, endog]].reset_index(drop=True),
        y=outcome,
        endog=endog,
        shares=shares_aligned.reset_index(drop=True),
        shocks=shocks,
        leave_one_out=leave_one_out,
        alpha=alpha,
        robust="hc1",
    )

    # Build a CausalResult-compatible view if the backend returned EconometricResults.
    if isinstance(ivres, EconometricResults):
        beta = float(ivres.params[endog])
        se = float(ivres.std_errors[endog])
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (beta - z * se, beta + z * se)
        pv = float(2 * (1 - stats.norm.cdf(abs(beta) / se))) if se > 0 else float("nan")
        causal = CausalResult(
            method="shift_share_political",
            estimand="LATE",
            estimate=beta,
            se=se,
            pvalue=pv,
            ci=ci,
            alpha=alpha,
            n_obs=int(len(cs)),
        )
    else:
        causal = ivres

    # --- Rotemberg top-K diagnostic ---------------------------------------
    shares_arr = shares_aligned.to_numpy(dtype=float)
    shocks_arr = shocks.to_numpy(dtype=float)
    dx = cs[endog].to_numpy(dtype=float)
    alphas = _rotemberg_weights(shares_arr, shocks_arr, dx)
    rot_df = pd.DataFrame({
        "industry": list(shares.columns),
        "shock": shocks_arr,
        "rotemberg_weight": alphas,
        "abs_weight": np.abs(alphas),
    }).sort_values("abs_weight", ascending=False).reset_index(drop=True)

    # --- Share-balance diagnostic -----------------------------------------
    if covariates:
        first_period = data.sort_values([unit, time]).groupby(unit).first()
        cov_df = first_period[list(covariates)].loc[cs.index]
        balance = _share_balance_test(shares_aligned, cov_df)
    else:
        balance = pd.DataFrame(columns=["covariate", "R2_on_shares", "F", "pvalue"])

    return ShiftSharePoliticalResult(
        iv_result=causal,
        rotemberg_top=rot_df,
        share_balance=balance,
        n_units=int(len(cs)),
        n_periods=int(data[time].nunique()),
        n_industries=int(shares.shape[1]),
        method="shift_share_political",
        diagnostics={
            "leave_one_out": bool(leave_one_out),
            "rotemberg_top1_share": float(rot_df.iloc[0]["abs_weight"])
            if len(rot_df) > 0 else 0.0,
            "rotemberg_top5_share": float(rot_df.head(5)["abs_weight"].sum())
            if len(rot_df) >= 5 else 0.0,
        },
    )
