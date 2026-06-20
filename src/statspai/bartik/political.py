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
"Shift-Share Designs in Political Science." arXiv:2603.00135. [@park2026shift]

Adão, R., Kolesár, M. & Morales, E. (2019).
"Shift-Share Designs: Theory and Inference." QJE, 134(4). [@ado2019shift]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult, EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability
from .shift_share import bartik as _bartik_cs

__all__ = [
    "shift_share_political",
    "ShiftSharePoliticalResult",
    "shift_share_political_panel",
    "ShiftSharePoliticalPanelResult",
]


@dataclass
class ShiftSharePoliticalResult:
    """Structured output of :func:`shift_share_political`.

    Wraps a standard :class:`CausalResult` (point + SEs) plus the two
    Park-Xu (2026) diagnostics: share-balance and Rotemberg top-K.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> units, inds = range(20), [f"I{k}" for k in range(5)]
    >>> shares = pd.DataFrame(rng.dirichlet(np.ones(5), size=len(units)),
    ...                       index=list(units), columns=inds)
    >>> shocks = pd.Series(rng.normal(size=5), index=inds)
    >>> rows = []
    >>> for i in units:
    ...     dx = float((shares.loc[i] * shocks).sum()) + rng.normal(scale=0.1)
    ...     rows.append({"unit": i, "time": 0, "y": 0.0, "x": 0.0})
    ...     rows.append({"unit": i, "time": 1, "y": 0.4 * dx, "x": dx})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.shift_share_political(
    ...     df, unit="unit", time="time", outcome="y", endog="x",
    ...     shares=shares, shocks=shocks,
    ... )
    >>> bool(np.isfinite(res.estimate))
    True
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


def _require_dataframe(obj: Any, *, name: str, function: str) -> pd.DataFrame:
    if not isinstance(obj, pd.DataFrame):
        raise MethodIncompatibility(
            f"`{name}` must be a pandas DataFrame, got {type(obj).__name__}.",
            recovery_hint=f"Pass `{name}` as a pandas DataFrame to `{function}`.",
            diagnostics={
                "function": function,
                "argument": name,
                "type": type(obj).__name__,
            },
        )
    if obj.empty:
        raise DataInsufficient(
            f"`{name}` must contain at least one row and one column.",
            recovery_hint=f"Provide non-empty `{name}` data before calling `{function}`.",
            diagnostics={"function": function, "argument": name, "shape": obj.shape},
        )
    return obj


def _require_series(obj: Any, *, name: str, function: str) -> pd.Series:
    if not isinstance(obj, pd.Series):
        raise MethodIncompatibility(
            f"`{name}` must be a pandas Series, got {type(obj).__name__}.",
            recovery_hint=f"Pass `{name}` as a pandas Series indexed by industry.",
            diagnostics={
                "function": function,
                "argument": name,
                "type": type(obj).__name__,
            },
        )
    if obj.empty:
        raise DataInsufficient(
            f"`{name}` must contain at least one industry shock.",
            recovery_hint=f"Provide non-empty `{name}` shocks before calling `{function}`.",
            diagnostics={"function": function, "argument": name},
        )
    return obj


def _require_column_name(name: Any, *, argument: str) -> str:
    if not isinstance(name, str) or not name:
        raise MethodIncompatibility(
            f"`{argument}` must be a non-empty column name string.",
            recovery_hint=f"Pass the name of an existing DataFrame column for `{argument}`.",
            diagnostics={"argument": argument, "type": type(name).__name__},
        )
    return name


def _require_columns(
    df: pd.DataFrame, columns: Sequence[str], *, function: str
) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise MethodIncompatibility(
            f"Columns not found in data: {missing}",
            recovery_hint=f"Check the column names passed to `{function}`.",
            diagnostics={
                "function": function,
                "missing_columns": missing,
                "available_columns": list(df.columns),
            },
        )


def _coerce_optional_columns(
    columns: Optional[Sequence[str] | str], *, argument: str
) -> List[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        out = [columns]
    else:
        try:
            out = list(columns)
        except TypeError as exc:
            raise MethodIncompatibility(
                f"`{argument}` must be a column name or sequence of column names.",
                recovery_hint=f"Pass `{argument}` as 'x' or ['x1', 'x2'].",
                diagnostics={"argument": argument, "type": type(columns).__name__},
            ) from exc
    return [_require_column_name(col, argument=argument) for col in out]


def _require_alpha(alpha: Any) -> float:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise MethodIncompatibility(
            "`alpha` must be a finite number in (0, 1).",
            recovery_hint="Pass a significance level such as alpha=0.05.",
            diagnostics={"argument": "alpha", "value": alpha},
        )
    out = float(alpha)
    if not np.isfinite(out) or not (0.0 < out < 1.0):
        raise MethodIncompatibility(
            "`alpha` must be a finite number in (0, 1).",
            recovery_hint="Pass a significance level such as alpha=0.05.",
            diagnostics={"argument": "alpha", "value": alpha},
        )
    return out


def _require_bool(value: Any, *, argument: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"`{argument}` must be boolean.",
            recovery_hint=f"Pass `{argument}=True` or `{argument}=False`.",
            diagnostics={"argument": argument, "type": type(value).__name__},
        )
    return bool(value)


def _finite_frame(df: pd.DataFrame, *, name: str) -> np.ndarray:
    try:
        arr: np.ndarray = np.asarray(df.to_numpy(dtype=float), dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{name}` must contain numeric values.",
            recovery_hint=f"Coerce `{name}` to numeric columns before estimation.",
            diagnostics={"argument": name, "columns": list(df.columns)},
        ) from exc
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise DataInsufficient(
            f"`{name}` must have at least one numeric column.",
            recovery_hint=f"Provide at least one industry column in `{name}`.",
            diagnostics={"argument": name, "shape": arr.shape},
        )
    if not np.all(np.isfinite(arr)):
        raise NumericalInstability(
            f"`{name}` contains non-finite values.",
            recovery_hint=f"Drop or impute NaN/Inf values in `{name}` before estimation.",
            diagnostics={"argument": name, "shape": arr.shape},
        )
    return arr


def _finite_series(series: pd.Series, *, name: str) -> np.ndarray:
    try:
        arr: np.ndarray = np.asarray(series.to_numpy(dtype=float), dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{name}` must contain numeric values.",
            recovery_hint=f"Coerce `{name}` to numeric values before estimation.",
            diagnostics={"argument": name},
        ) from exc
    if arr.ndim != 1 or arr.size == 0:
        raise DataInsufficient(
            f"`{name}` must contain at least one value.",
            recovery_hint=f"Provide non-empty `{name}` values before estimation.",
            diagnostics={"argument": name, "shape": arr.shape},
        )
    if not np.all(np.isfinite(arr)):
        raise NumericalInstability(
            f"`{name}` contains non-finite values.",
            recovery_hint=f"Drop or impute NaN/Inf values in `{name}` before estimation.",
            diagnostics={"argument": name},
        )
    return arr


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
    agg = pd.DataFrame(
        {
            outcome: dy,
            endog: dx,
        }
    )
    agg = agg.join(shares, how="inner")
    return agg


def _rotemberg_weights(
    shares: np.ndarray, shocks: np.ndarray, dx: np.ndarray
) -> np.ndarray:
    """Rotemberg weights α_k proportional to shock variation × exposure.

    α_k ∝ g_k * (∑_i s_{ik} (x_i - x̄)) ; we return the normalised vector.
    """
    x_c = dx - dx.mean()
    num: np.ndarray = np.asarray(shocks * (shares.T @ x_c), dtype=float)
    tot = np.sum(np.abs(num))
    if tot > 0:
        weights: np.ndarray = np.asarray(num / float(tot), dtype=float)
        return weights
    return np.asarray(num, dtype=float)


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
        rss = float(np.sum(resid**2))
        tss = float(np.sum((z - z.mean()) ** 2))
        if tss <= 0 or rss <= 0:
            continue
        r2 = 1 - rss / tss
        df1, df2 = k, max(n - k - 1, 1)
        F = (r2 / k) / ((1 - r2) / max(df2, 1)) if r2 < 1 else float("inf")
        pv = float(1 - stats.f.cdf(F, df1, df2)) if np.isfinite(F) else 0.0
        results.append(
            {
                "covariate": col,
                "R2_on_shares": r2,
                "F": F,
                "pvalue": pv,
            }
        )
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
    >>> out = sp.shift_share_political(
    ...     df, unit='unit', time='time',
    ...     outcome='y', endog='x',
    ...     shares=shares, shocks=shocks,
    ... )
    >>> abs(out.estimate - true_tau) < 0.3
    True
    """
    # --- Validation --------------------------------------------------------
    data = _require_dataframe(data, name="data", function="shift_share_political")
    unit = _require_column_name(unit, argument="unit")
    time = _require_column_name(time, argument="time")
    outcome = _require_column_name(outcome, argument="outcome")
    endog = _require_column_name(endog, argument="endog")
    covariates = _coerce_optional_columns(covariates, argument="covariates")
    alpha = _require_alpha(alpha)
    leave_one_out = _require_bool(leave_one_out, argument="leave_one_out")
    _require_columns(
        data,
        (unit, time, outcome, endog, *covariates),
        function="shift_share_political",
    )
    shares = _require_dataframe(shares, name="shares", function="shift_share_political")
    shocks = _require_series(shocks, name="shocks", function="shift_share_political")
    _finite_frame(shares, name="shares")
    _finite_series(shocks, name="shocks")
    if list(shares.columns) != list(shocks.index):
        # Align on intersection
        common = [c for c in shares.columns if c in shocks.index]
        if not common:
            raise MethodIncompatibility(
                "shares.columns and shocks.index have no overlap.",
                recovery_hint="Use the same industry labels for `shares` columns and `shocks` index.",
                diagnostics={
                    "shares_columns": list(shares.columns),
                    "shocks_index": list(shocks.index),
                },
            )
        shares = shares[common]
        shocks = shocks.loc[common]
    if data[time].nunique() < 2:
        raise DataInsufficient(
            "`data` must contain at least two time periods for long-difference shift-share IV.",
            recovery_hint="Provide pre/post or multi-period panel data.",
            diagnostics={
                "function": "shift_share_political",
                "n_periods": int(data[time].nunique()),
            },
        )

    # --- Build cross-section of long-differences --------------------------
    cs = _long_to_panel(
        data,
        shares,
        unit=unit,
        time=time,
        endog=endog,
        outcome=outcome,
    )
    if cs.empty:
        raise DataInsufficient(
            "No units remain after aligning `data` with `shares`.",
            recovery_hint="Ensure `shares.index` contains the unit identifiers in `data`.",
            diagnostics={"function": "shift_share_political"},
        )
    # Guard the differenced outcome/endog: an all-NaN (or non-numeric) outcome
    # otherwise flows through the IV and returns a silent `estimate=nan`. This
    # mirrors the finite-check the panel sibling already applies.
    _finite_frame(cs[[outcome, endog]], name="shift_share_political outcome/endog")
    cs_with_shares = cs.reset_index()
    cs_with_shares = (
        cs_with_shares.rename(columns={"index": unit})
        if unit not in cs_with_shares.columns
        else cs_with_shares
    )

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
    rot_df = (
        pd.DataFrame(
            {
                "industry": list(shares.columns),
                "shock": shocks_arr,
                "rotemberg_weight": alphas,
                "abs_weight": np.abs(alphas),
            }
        )
        .sort_values("abs_weight", ascending=False)
        .reset_index(drop=True)
    )

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
            "rotemberg_top1_share": (
                float(rot_df.iloc[0]["abs_weight"]) if len(rot_df) > 0 else 0.0
            ),
            "rotemberg_top5_share": (
                float(rot_df.head(5)["abs_weight"].sum()) if len(rot_df) >= 5 else 0.0
            ),
        },
    )


# ===========================================================================
# Multi-period panel extension (Park-Xu 2026, §4.2)
# ===========================================================================


@dataclass
class ShiftSharePoliticalPanelResult:
    """Structured output of :func:`shift_share_political_panel`.

    Attributes
    ----------
    estimate : float
        Pooled 2SLS coefficient on ``endog``.
    se : float
        Panel-clustered SE (unit by default; shock-clustered available
        via the underlying AKM correction — stored in
        ``diagnostics['akm_se']``).
    ci : tuple
        ``(lower, upper)`` at ``alpha``.
    per_period : pd.DataFrame
        Per-period cross-sectional estimates (one row per ``time``),
        useful for event-study-style dynamic effects.
    rotemberg_panel : pd.DataFrame
        Rotemberg weights aggregated across periods (industries × stats).
    share_balance : pd.DataFrame
    n_units : int
    n_periods : int
    n_industries : int
    method : str
    diagnostics : dict
        Estimator-side diagnostics: ``fe`` (mode), ``cluster``, ``akm_se``,
        ``n_obs``, ``first_stage_F``.
    model_info : dict
        Output-layer metadata: ``model_type``, ``method``, ``fixed_effects``
        (column-name list as ``"unit+time"``), ``cluster``. Consumed by
        :func:`statspai.regtable` to render per-FE / cluster rows
        automatically.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> units, times, inds = list(range(30)), [0, 1, 2, 3], list("AB")
    >>> shares = pd.DataFrame(rng.dirichlet(np.ones(2), size=len(units)),
    ...                       index=units, columns=inds)
    >>> shocks = pd.DataFrame(
    ...     rng.normal(size=(len(times), 2)), index=times, columns=inds,
    ... )
    >>> rows = []
    >>> for i in units:
    ...     for t in times:
    ...         b = float((shares.loc[i] * shocks.loc[t]).sum())
    ...         x = b + rng.normal(scale=0.1)
    ...         rows.append({"u": i, "t": t, "y": 0.3 * x, "x": x})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.shift_share_political_panel(
    ...     df, unit="u", time="t", outcome="y", endog="x",
    ...     shares=shares, shocks=shocks,
    ... )
    >>> bool(np.isfinite(res.estimate))
    True
    """

    estimate: float
    se: float
    ci: tuple
    per_period: pd.DataFrame
    rotemberg_panel: pd.DataFrame
    share_balance: pd.DataFrame
    n_units: int
    n_periods: int
    n_industries: int
    alpha: float = 0.05
    method: str = "shift_share_political_panel"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lo, hi = self.ci
        cluster_label = self.diagnostics.get("cluster", "unit")
        lines = [
            "Shift-Share (Bartik) — Political Science, Panel (Park-Xu 2026 §4.2)",
            "-" * 70,
            f"  Units × periods         : {self.n_units} × {self.n_periods}",
            f"  Industries              : {self.n_industries}",
            f"  Pooled 2SLS estimate    : {self.estimate:+.6f}",
            f"  SE ({cluster_label})     : {self.se:.6f}",
            f"  {int((1 - self.alpha) * 100)}% CI                 : "
            f"[{lo:+.6f}, {hi:+.6f}]",
            "",
            "  Per-period estimates (event study):",
            self.per_period.to_string(index=False, float_format="%.4f"),
            "",
            "  Rotemberg top-5 industries (aggregate):",
            self.rotemberg_panel.head(5).to_string(index=False, float_format="%.4f"),
        ]
        if len(self.share_balance):
            lines.append("")
            lines.append("  Share-balance (F-test on shares):")
            lines.append(self.share_balance.to_string(index=False, float_format="%.4f"))
        return "\n".join(lines)


def _resolve_shares(
    shares: Any,
    times: Sequence[Any],
    units: Sequence[Any],
) -> Dict[Any, pd.DataFrame]:
    """Normalise the `shares` input into ``{time: DataFrame(unit × industry)}``.

    Accepted forms:
      * ``DataFrame`` indexed by unit — interpreted as time-invariant,
        broadcast to every period.
      * ``dict[time → DataFrame]`` — per-period share matrices (must
        share the same industry columns).
    """
    if isinstance(shares, pd.DataFrame):
        _finite_frame(shares, name="shares")
        out = {}
        for t in times:
            s = shares.loc[[u for u in units if u in shares.index]]
            out[t] = s
        return out
    if isinstance(shares, dict):
        out = {}
        cols0 = None
        for t in times:
            if t not in shares:
                raise MethodIncompatibility(
                    f"shares missing entry for time={t!r}",
                    recovery_hint="Provide a share matrix for every time period in the data.",
                    diagnostics={"time": t, "available_times": list(shares.keys())},
                )
            s = shares[t]
            if not isinstance(s, pd.DataFrame):
                raise MethodIncompatibility(
                    f"shares[{t!r}] must be DataFrame, got {type(s).__name__}",
                    recovery_hint="Use pandas DataFrames for all time-specific share matrices.",
                    diagnostics={"time": t, "type": type(s).__name__},
                )
            _finite_frame(s, name=f"shares[{t!r}]")
            if cols0 is None:
                cols0 = list(s.columns)
            elif list(s.columns) != cols0:
                raise MethodIncompatibility(
                    f"shares[{t!r}].columns != shares[{times[0]!r}].columns",
                    recovery_hint="Use the same industry columns in every time-specific share matrix.",
                    diagnostics={
                        "time": t,
                        "columns": list(s.columns),
                        "reference_columns": cols0,
                    },
                )
            out[t] = s
        return out
    raise MethodIncompatibility(
        "shares must be a DataFrame or dict[time → DataFrame]; "
        f"got {type(shares).__name__}",
        recovery_hint="Pass a time-invariant share DataFrame or a dict of time-specific DataFrames.",
        diagnostics={"argument": "shares", "type": type(shares).__name__},
    )


def _resolve_shocks(
    shocks: Any,
    times: Sequence[Any],
    industries: Sequence[Any],
) -> Dict[Any, pd.Series]:
    """Normalise the `shocks` input into ``{time: Series(industry)}``."""
    if isinstance(shocks, pd.Series):
        _finite_series(shocks, name="shocks")
        return {t: shocks for t in times}
    if isinstance(shocks, pd.DataFrame):
        # Rows = time, columns = industry
        out = {}
        for t in times:
            if t not in shocks.index:
                raise MethodIncompatibility(
                    f"shocks row missing for time={t!r}",
                    recovery_hint="Provide one shock row for every time period in the data.",
                    diagnostics={"time": t, "available_times": list(shocks.index)},
                )
            out[t] = shocks.loc[t]
            _finite_series(out[t], name=f"shocks.loc[{t!r}]")
        return out
    if isinstance(shocks, dict):
        out = {}
        for t in times:
            if t not in shocks:
                raise MethodIncompatibility(
                    f"shocks missing entry for time={t!r}",
                    recovery_hint="Provide one shock vector for every time period in the data.",
                    diagnostics={"time": t, "available_times": list(shocks.keys())},
                )
            s = shocks[t]
            if not isinstance(s, pd.Series):
                raise MethodIncompatibility(
                    f"shocks[{t!r}] must be Series, got {type(s).__name__}",
                    recovery_hint="Use pandas Series for all time-specific shock vectors.",
                    diagnostics={"time": t, "type": type(s).__name__},
                )
            _finite_series(s, name=f"shocks[{t!r}]")
            out[t] = s
        return out
    raise MethodIncompatibility(
        "shocks must be Series, DataFrame(time × industry), or dict[time → Series]",
        recovery_hint="Pass shocks as a Series, a time-by-industry DataFrame, or a dict of Series.",
        diagnostics={"argument": "shocks", "type": type(shocks).__name__},
    )


def _build_bartik_panel(
    data: pd.DataFrame,
    shares_by_t: Dict[Any, pd.DataFrame],
    shocks_by_t: Dict[Any, pd.Series],
    *,
    unit: str,
    time: str,
) -> pd.DataFrame:
    """Attach a ``bartik_iv`` column to `data` row by row."""
    out = data.copy()
    iv = np.full(len(out), np.nan)
    for t, shares_t in shares_by_t.items():
        shocks_t = shocks_by_t[t]
        # Align industries
        cols = [c for c in shares_t.columns if c in shocks_t.index]
        if not cols:
            raise MethodIncompatibility(
                f"no shared industries at time={t!r}",
                recovery_hint="Align share-matrix columns with shock-vector indexes for every period.",
                diagnostics={
                    "time": t,
                    "shares_columns": list(shares_t.columns),
                    "shocks_index": list(shocks_t.index),
                },
            )
        bart = shares_t[cols] @ shocks_t.loc[cols]
        mask = (out[time] == t) & (out[unit].isin(bart.index))
        if mask.any():
            iv[mask.values] = bart.loc[out.loc[mask, unit]].to_numpy(dtype=float)
    out["__bartik_iv__"] = iv
    return out


def shift_share_political_panel(
    data: pd.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
    endog: str,
    shares: Any,
    shocks: Any,
    covariates: Optional[Sequence[str]] = None,
    cluster: str = "unit",
    alpha: float = 0.05,
    fe: str = "two-way",
) -> ShiftSharePoliticalPanelResult:
    """Multi-period panel shift-share IV (Park-Xu 2026 §4.2).

    Pooled 2SLS with unit / time / two-way fixed effects, using the
    period-specific Bartik instrument

        Z_{it} = sum_k s_{ikt} · g_{kt}

    The share matrix can be time-invariant (``DataFrame`` indexed by
    unit) or time-varying (``dict[time → DataFrame]``); the shock
    vector can similarly be scalar-in-time (``Series``) or time-varying
    (``DataFrame`` indexed by time / ``dict[time → Series]``).

    Parameters
    ----------
    data : DataFrame (long format)
    unit, time, outcome, endog : str
    shares : DataFrame or dict[time → DataFrame]
    shocks : Series, DataFrame(time × industry), or dict[time → Series]
    covariates : sequence of str, optional
        Time-varying controls.
    cluster : {'unit', 'time', 'twoway'}, default 'unit'
        Cluster structure for the panel SE.
    alpha : float, default 0.05
    fe : {'two-way', 'unit', 'time', 'none'}, default 'two-way'
        Fixed-effect structure.  ``'two-way'`` is the Park-Xu default.

    Returns
    -------
    ShiftSharePoliticalPanelResult

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> units, times, inds = list(range(30)), [0, 1, 2, 3], list("AB")
    >>> shares = pd.DataFrame(rng.dirichlet(np.ones(2), size=len(units)),
    ...                       index=units, columns=inds)
    >>> shocks = pd.DataFrame(
    ...     rng.normal(size=(len(times), 2)), index=times, columns=inds,
    ... )
    >>> rows = []
    >>> tau = 0.3
    >>> for i in units:
    ...     for t in times:
    ...         b = float((shares.loc[i] * shocks.loc[t]).sum())
    ...         x = b + rng.normal(scale=0.1)
    ...         y = tau * x + rng.normal(scale=0.1)
    ...         rows.append({'u': i, 't': t, 'y': y, 'x': x})
    >>> df = pd.DataFrame(rows)
    >>> out = sp.shift_share_political_panel(
    ...     df, unit='u', time='t', outcome='y', endog='x',
    ...     shares=shares, shocks=shocks,
    ... )
    >>> abs(out.estimate - tau) < 0.15
    True
    """
    data = _require_dataframe(data, name="data", function="shift_share_political_panel")
    unit = _require_column_name(unit, argument="unit")
    time = _require_column_name(time, argument="time")
    outcome = _require_column_name(outcome, argument="outcome")
    endog = _require_column_name(endog, argument="endog")
    cov_cols = _coerce_optional_columns(covariates, argument="covariates")
    alpha = _require_alpha(alpha)
    if fe not in ("two-way", "unit", "time", "none"):
        raise MethodIncompatibility(
            f"fe must be one of two-way/unit/time/none; got {fe!r}",
            recovery_hint="Use fe='two-way', 'unit', 'time', or 'none'.",
            diagnostics={"argument": "fe", "value": fe},
        )
    if cluster not in ("unit", "time", "twoway", "shock"):
        raise MethodIncompatibility(
            f"cluster must be unit/time/twoway/shock; got {cluster!r}. "
            "`'shock'` invokes the Adão-Kolesár-Morales (2019) "
            "shock-clustered variance estimator — strongly recommended "
            "by Park-Xu (2026) §4.2.",
            recovery_hint="Use cluster='unit', 'time', 'twoway', or 'shock'.",
            diagnostics={"argument": "cluster", "value": cluster},
        )
    _require_columns(
        data,
        (unit, time, outcome, endog, *cov_cols),
        function="shift_share_political_panel",
    )

    data_sorted = data.sort_values([unit, time]).reset_index(drop=True)
    times = sorted(data_sorted[time].unique())
    units = sorted(data_sorted[unit].unique())
    if len(times) < 2:
        raise DataInsufficient(
            "`data` must contain at least two time periods for panel shift-share IV.",
            recovery_hint="Provide multi-period panel data.",
            diagnostics={
                "function": "shift_share_political_panel",
                "n_periods": len(times),
            },
        )
    if len(units) < 2:
        raise DataInsufficient(
            "`data` must contain at least two units for panel shift-share IV.",
            recovery_hint="Provide data for at least two units.",
            diagnostics={
                "function": "shift_share_political_panel",
                "n_units": len(units),
            },
        )
    shares_by_t = _resolve_shares(shares, times, units)
    first_t = times[0]
    industries = list(shares_by_t[first_t].columns)
    shocks_by_t = _resolve_shocks(shocks, times, industries)

    df_iv = _build_bartik_panel(
        data_sorted,
        shares_by_t,
        shocks_by_t,
        unit=unit,
        time=time,
    )
    if df_iv["__bartik_iv__"].isna().any():
        n_missing = int(df_iv["__bartik_iv__"].isna().sum())
        raise DataInsufficient(
            f"{n_missing} rows have missing Bartik IV — check that "
            "every (unit, time) is covered by shares + shocks.",
            recovery_hint="Ensure every unit and period is covered by the share and shock inputs.",
            diagnostics={
                "function": "shift_share_political_panel",
                "n_missing": n_missing,
            },
        )

    # --- Within-transformation for FE ------------------------------------
    def _demean(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        out = df.copy()
        if fe in ("two-way", "unit"):
            out[cols] = out[cols].sub(out.groupby(unit)[cols].transform("mean"))
        if fe in ("two-way", "time"):
            out[cols] = out[cols].sub(out.groupby(time)[cols].transform("mean"))
        return out

    work_cols = [outcome, endog, "__bartik_iv__"] + cov_cols
    _finite_frame(df_iv[work_cols], name="panel outcome/endog/instrument/covariates")
    df_demean = _demean(df_iv, work_cols)

    Y = df_demean[outcome].to_numpy(dtype=float)
    D = df_demean[endog].to_numpy(dtype=float)
    Z = df_demean["__bartik_iv__"].to_numpy(dtype=float)
    X_cov = (
        df_demean[cov_cols].to_numpy(dtype=float) if cov_cols else np.zeros((len(Y), 0))
    )

    # Stage 1: D ~ Z + X
    S1 = np.column_stack([np.ones(len(Y)), Z, X_cov])
    pi, *_ = np.linalg.lstsq(S1, D, rcond=None)
    D_hat = S1 @ pi

    # Stage 2: Y ~ D_hat + X
    S2 = np.column_stack([np.ones(len(Y)), D_hat, X_cov])
    b2, *_ = np.linalg.lstsq(S2, Y, rcond=None)
    beta = float(b2[1])

    # Residuals from the STRUCTURAL equation (Y ~ D, not Y ~ D_hat)
    S2_struct = np.column_stack([np.ones(len(Y)), D, X_cov])
    resid = Y - S2_struct @ b2

    # --- Standard errors --------------------------------------------------
    if cluster == "shock":
        # Adão-Kolesár-Morales (2019) shock-clustered variance for panel
        # shift-share.  For each shock k compute the stacked score
        #   u_k = sum_{i, t} s_{ikt} * Z_tilde_{it} * eps_{it}
        # and assemble var(beta) = (D_hat' D_tilde)^{-2} * sum_k u_k^2 in
        # the demeaned (within-FE) space.  D_tilde, Z_tilde, and eps here
        # are the FE-demeaned endogenous regressor, instrument, and 2SLS
        # residuals respectively.
        Z_tilde = df_demean["__bartik_iv__"].to_numpy(dtype=float)
        D_tilde = df_demean[endog].to_numpy(dtype=float)
        # Residuals of the structural equation in demeaned space (without
        # the covariates, which are demeaned too and already inside S2_struct).
        eps = resid
        # Build the score per shock k across all (i, t).
        u_k = np.zeros(len(industries), dtype=float)
        # Align rows of df_iv with shares_by_t lookup.
        unit_vals = df_iv[unit].to_numpy()
        time_vals = df_iv[time].to_numpy()
        for k_idx, ind in enumerate(industries):
            contrib = 0.0
            for t in times:
                shares_t = shares_by_t[t]
                if ind not in shares_t.columns:
                    continue
                mask = time_vals == t
                if not mask.any():
                    continue
                # shares at (units_in_t, industry=ind).  Reindex by row's
                # unit to align row-wise with Z_tilde / eps.
                s_col = (
                    shares_t[ind]
                    .reindex(unit_vals[mask])
                    .to_numpy(
                        dtype=float,
                        na_value=0.0,
                    )
                )
                contrib += float(np.sum(s_col * Z_tilde[mask] * eps[mask]))
            u_k[k_idx] = contrib
        # Denominator: (D_hat' D_tilde) under FE demeaning.
        denom = float(np.dot(D_hat, D_tilde))
        if abs(denom) < 1e-12:
            raise NumericalInstability(
                "AKM shock-cluster SE: (D_hat' D_tilde) ≈ 0 — instrument "
                "too weak after FE demeaning.",
                recovery_hint=(
                    "Check first-stage strength, simplify fixed effects, or use a stronger "
                    "shift-share instrument."
                ),
                diagnostics={
                    "function": "shift_share_political_panel",
                    "cluster": "shock",
                    "denominator": denom,
                },
            )
        var_akm = float(np.sum(u_k**2) / denom**2)
        se = float(np.sqrt(max(var_akm, 0.0)))
        akm_se = se
        cluster_label = "shock (AKM 2019)"
    else:
        # Unit / time / two-way cluster-robust sandwich on stage-2.
        cluster_col = (
            df_iv[unit].to_numpy()
            if cluster == "unit"
            else (
                df_iv[time].to_numpy()
                if cluster == "time"
                else (
                    df_iv[unit].astype(str) + "_" + df_iv[time].astype(str)
                ).to_numpy()
            )
        )
        bread = np.linalg.pinv(S2.T @ S2)
        meat = np.zeros_like(bread)
        for g in np.unique(cluster_col):
            idx = np.where(cluster_col == g)[0]
            Sg = S2[idx]
            rg = resid[idx]
            scores = Sg * rg[:, None]
            sg = scores.sum(axis=0)
            meat += np.outer(sg, sg)
        vcov = bread @ meat @ bread
        se = float(np.sqrt(max(vcov[1, 1], 0.0)))
        akm_se = None
        cluster_label = cluster
    from scipy.stats import norm as _norm

    z = _norm.ppf(1 - alpha / 2)
    ci = (beta - z * se, beta + z * se)

    # --- Per-period cross-sectional estimates ----------------------------
    per_period_rows = []
    for t in times:
        sub = df_iv[df_iv[time] == t]
        yt = sub[outcome].to_numpy(dtype=float)
        dt = sub[endog].to_numpy(dtype=float)
        zt = sub["__bartik_iv__"].to_numpy(dtype=float)
        n_t = len(sub)
        if n_t < 5 or float(np.std(zt)) < 1e-12:
            continue
        s1 = np.column_stack([np.ones(n_t), zt])
        pi_t, *_ = np.linalg.lstsq(s1, dt, rcond=None)
        dth = s1 @ pi_t
        s2 = np.column_stack([np.ones(n_t), dth])
        b_t, *_ = np.linalg.lstsq(s2, yt, rcond=None)
        r_t = yt - np.column_stack([np.ones(n_t), dt]) @ b_t
        bread_t = np.linalg.pinv(s2.T @ s2)
        meat_t = (s2 * r_t[:, None]).T @ (s2 * r_t[:, None])
        v_t = float((bread_t @ meat_t @ bread_t)[1, 1])
        per_period_rows.append(
            {
                "time": t,
                "estimate": float(b_t[1]),
                "se": float(np.sqrt(max(v_t, 0.0))),
                "n": int(n_t),
            }
        )
    per_period = pd.DataFrame(per_period_rows)

    # --- Rotemberg weights aggregated across periods ---------------------
    rot_acc = {ind: 0.0 for ind in industries}
    for t in times:
        shares_t = shares_by_t[t]
        shocks_t = shocks_by_t[t]
        sub = df_iv[df_iv[time] == t]
        d_c = sub[endog].to_numpy(dtype=float) - sub[endog].mean()
        aligned = [ind for ind in industries if ind in shocks_t.index]
        if not aligned:
            continue
        S_aligned = (
            shares_t.loc[sub[unit].astype(int), aligned].to_numpy(dtype=float)
            if all(u in shares_t.index for u in sub[unit])
            else shares_t.reindex(sub[unit])[aligned].to_numpy(dtype=float)
        )
        g_aligned = shocks_t.loc[aligned].to_numpy(dtype=float)
        alpha_t = g_aligned * (S_aligned.T @ d_c)
        for ind, w in zip(aligned, alpha_t):
            rot_acc[ind] += float(w)
    total = sum(abs(v) for v in rot_acc.values())
    rot_rows = []
    for ind, w in rot_acc.items():
        rot_rows.append(
            {
                "industry": ind,
                "rotemberg_weight": (w / total) if total > 0 else 0.0,
                "abs_weight": abs(w) / total if total > 0 else 0.0,
            }
        )
    rot_df = (
        pd.DataFrame(rot_rows)
        .sort_values("abs_weight", ascending=False)
        .reset_index(drop=True)
    )

    # --- Share-balance diagnostic (using time=first share matrix) --------
    if cov_cols:
        first_period = data_sorted[data_sorted[time] == first_t]
        cov_df = first_period.set_index(unit)[cov_cols]
        shares_first = shares_by_t[first_t].loc[
            [u for u in cov_df.index if u in shares_by_t[first_t].index]
        ]
        cov_df = cov_df.loc[shares_first.index]
        balance = _share_balance_test(shares_first, cov_df)
    else:
        balance = pd.DataFrame(columns=["covariate", "R2_on_shares", "F", "pvalue"])

    # Translate the FE *mode* into the column-name list that the output
    # layer expects under the canonical ``model_info['fixed_effects']`` key.
    # We follow the pyfixest convention — additive FEs joined with ``+`` —
    # so ``sp.regtable`` can render one row per absorbed FE without a
    # bartik-specific code path. ``diagnostics['fe']`` is kept for
    # backwards-compat with any user code that already reads the mode.
    fe_vars = {
        "none": "",
        "unit": unit,
        "time": time,
        "two-way": f"{unit}+{time}",
    }[fe]

    return ShiftSharePoliticalPanelResult(
        estimate=beta,
        se=se,
        ci=ci,
        per_period=per_period,
        rotemberg_panel=rot_df,
        share_balance=balance,
        n_units=int(len(units)),
        n_periods=int(len(times)),
        n_industries=int(len(industries)),
        alpha=float(alpha),
        method="shift_share_political_panel",
        diagnostics={
            "fe": fe,
            "cluster": cluster_label,
            "akm_se": akm_se,
            "n_obs": int(len(df_iv)),
            "first_stage_F": float(
                (D_hat.var() / max(resid.var(), 1e-12))
                if resid.var() > 0
                else float("nan")
            ),
        },
        model_info={
            "model_type": "Shift-Share IV (panel)",
            "method": "shift_share_political_panel",
            "fixed_effects": fe_vars,
            "cluster": cluster_label,
        },
    )
