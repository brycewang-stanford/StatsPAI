"""
Shift-share IV wrappers for political-science panels.

Background: Park, P. K. (2026), "Shift-Share Designs in Political Science",
arXiv preprint [@park2026shift] (single-authored; earlier versions of this
module cited it as "Park & Xu" and attributed specific recommendations and
section numbers to it that were not verified -- they have been removed).

``shift_share_political`` builds a long-difference cross-section (last minus
first period per unit) and runs :func:`sp.bartik` on it (2SLS, HC1 SE), and
adds:

* AKM (Adao-Kolesar-Morales 2019) shock-level SE and the AKM0 confidence
  interval of the same IV, computed by the shared ``_akm`` kernel that
  reproduces R ``ShiftShareSE::ivreg_ss`` [@ado2019shift];
* the Goldsmith-Pinkham, Sorkin & Swift Rotemberg weights of the Bartik IV
  (as returned by ``sp.bartik``, matching R ``bartik.weight::bw``);
* a share-balance regression of pre-period covariates on the shares.

``shift_share_political_panel`` runs pooled 2SLS with unit / time / two-way
fixed effects on the period-specific instrument ``Z_it = sum_k s_ikt g_kt``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin
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
class ShiftSharePoliticalResult(ResultProtocolMixin):
    """Structured output of :func:`shift_share_political`.

    Wraps a standard :class:`CausalResult` (2SLS point estimate, HC1 SE)
    plus the AKM shock-level SE (``diagnostics``), Rotemberg weights and a
    share-balance table.

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
            "Shift-Share (Bartik) IV — long differences",
            "-" * 60,
            f"  Units / periods         : {self.n_units} × {self.n_periods}",
            f"  Industries in exposure  : {self.n_industries}",
            f"  IV estimate             : {est:+.6f}",
            f"  SE (HC1)                : {se:.6f}",
            f"  SE (AKM shock-level)    : {self.diagnostics.get('akm_se', float('nan')):.6f}",
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
    """GPSS Rotemberg weights with only an intercept as control.

    ``alpha_k = g_k s_k' M x / sum_j g_j s_j' M x`` (signed normalisation:
    the weights sum to one and may be negative).
    """
    x_c = dx - dx.mean()
    num: np.ndarray = np.asarray(shocks * (shares.T @ x_c), dtype=float)
    tot = float(np.sum(num))
    if abs(tot) > 0:
        return np.asarray(num / tot, dtype=float)
    return np.full_like(num, np.nan)


def _share_balance_test(
    shares_df: pd.DataFrame,
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    """Regress each covariate on the share matrix and report the F-stat."""
    results = []
    X = shares_df.to_numpy(dtype=float)
    n = X.shape[0]
    X_design = np.column_stack([np.ones(n), X])
    # Shares that sum to one make the intercept collinear with them, so the
    # number of restrictions is rank([1, S]) - 1, not the number of columns
    # (the old df1 = K, df2 = n - K - 1 overstated df1 by one).
    rank = int(np.linalg.matrix_rank(X_design))
    k = rank - 1
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
        df1, df2 = k, max(n - rank, 1)
        F = (r2 / k) / ((1 - r2) / max(df2, 1)) if r2 < 1 else float("inf")
        pv = float(stats.f.sf(F, df1, df2)) if np.isfinite(F) else 0.0
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


@accepts_aliases(id="unit", y="outcome")
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
    """Long-difference shift-share IV with AKM SE and Rotemberg weights.

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
        Forwarded to :func:`sp.bartik`. With national ``shocks`` and no
        regional shocks ``sp.bartik`` cannot form a leave-one-out
        instrument; it warns and uses the plain Bartik instrument.

    Returns
    -------
    ShiftSharePoliticalResult
        ``estimate`` / ``se`` / ``ci``: 2SLS of the long-differenced outcome
        on the long-differenced ``endog`` instrumented by
        ``B_i = sum_k s_ik g_k`` (intercept only), HC1 SE with normal CI.
        ``diagnostics['akm_se']``, ``['akm_ci']``, ``['akm0_ci']``,
        ``['akm_pvalue']``: AKM / AKM0 inference for the same IV (shares
        as the ``W`` matrix, intercept as control). ``rotemberg_top``:
        ``alpha_k``, ``beta_k`` of Goldsmith-Pinkham, Sorkin & Swift
        (weights sum to one and may be negative). ``share_balance``: F-test
        of each covariate on the shares (restrictions = rank of
        ``[1, shares]`` minus one).

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
        pv = float(2 * stats.norm.sf(abs(beta) / se)) if se > 0 else float("nan")
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

    # --- AKM shock-level inference -----------------------------------------
    akm_diag: Dict[str, Any] = {}
    if isinstance(ivres, EconometricResults):
        from ._akm import _akm_fit

        ssi = ivres.data_info["_shift_share_inputs"]
        akm = _akm_fit(
            ssi["y"],
            ssi["shift_share"],
            shares_aligned.to_numpy(dtype=float),
            ssi["controls"],
            y2=ssi["endog"],
            alpha=alpha,
        )
        akm_diag = {
            "akm_se": akm["se"]["AKM"],
            "akm_ci": (akm["ci_l"]["AKM"], akm["ci_r"]["AKM"]),
            "akm_pvalue": akm["p"]["AKM"],
            "akm0_ci": (akm["ci_l"]["AKM0"], akm["ci_r"]["AKM0"]),
            "akm0_pvalue": akm["p"]["AKM0"],
            "akm0_ci_type": akm["akm0_ci_type"],
            "ehw_se_no_ssc": akm["se"]["EHW"],
        }
        rw = ivres.model_info["rotemberg_weights"]
        rot_df = pd.DataFrame(
            {
                "industry": rw["industry"].to_numpy(),
                "shock": rw["shock"].to_numpy(),
                "rotemberg_weight": rw["weight"].to_numpy(),
                "beta_k": rw["beta"].to_numpy(),
                "abs_weight": np.abs(rw["weight"].to_numpy()),
            }
        )
    else:  # pragma: no cover - sp.bartik always returns EconometricResults
        shares_arr = shares_aligned.to_numpy(dtype=float)
        shocks_arr = shocks.to_numpy(dtype=float)
        dx = cs[endog].to_numpy(dtype=float)
        alphas = _rotemberg_weights(shares_arr, shocks_arr, dx)
        rot_df = pd.DataFrame(
            {
                "industry": list(shares.columns),
                "shock": shocks_arr,
                "rotemberg_weight": alphas,
                "abs_weight": np.abs(alphas),
            }
        )
    rot_df = rot_df.sort_values("abs_weight", ascending=False).reset_index(drop=True)

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
            **akm_diag,
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
# Multi-period panel extension
# ===========================================================================


@dataclass
class ShiftSharePoliticalPanelResult(ResultProtocolMixin):
    """Structured output of :func:`shift_share_political_panel`.

    Attributes
    ----------
    estimate : float
        Pooled 2SLS coefficient on ``endog``.
    se : float
        SE of the requested ``cluster`` type (unit-clustered CR0 by
        default; with ``cluster='shock'`` the AKM SE, also stored in
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
        ``akm0_ci``, ``n_obs``, ``balanced``, ``first_stage_F`` (homoskedastic
        F of the excluded instrument, df net of the absorbed FE;
        ``fixest::fitstat(, "ivf1")``).
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
            "Shift-Share (Bartik) IV — panel with fixed effects",
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


@accepts_aliases(id="unit", y="outcome")
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
    """Multi-period panel shift-share IV with fixed effects.

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
    cluster : {'unit', 'time', 'twoway', 'shock'}, default 'unit'
        SE type. ``'unit'`` / ``'time'``: one-way cluster-robust (CR0, no
        small-sample factor; ``fixest`` with ``ssc(adj = FALSE,
        cluster.adj = FALSE)``). ``'twoway'``: Cameron-Gelbach-Miller
        two-way clustering by unit and time (CR0 components). ``'shock'``:
        AKM shock-level SE (= ``ShiftShareSE::ivreg_ss`` with the FE as
        controls; shock clusters are industries if the shocks are constant
        over time, industry x period otherwise).
    alpha : float, default 0.05
    fe : {'two-way', 'unit', 'time', 'none'}, default 'two-way'
        Fixed-effect structure (exact within transformation, also on
        unbalanced panels).

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
            "shock-level variance estimator.",
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
    unit_codes = pd.factorize(df_iv[unit])[0]
    time_codes = pd.factorize(df_iv[time])[0]
    n_u = int(unit_codes.max()) + 1
    n_t = int(time_codes.max()) + 1
    balanced = len(df_iv) == n_u * n_t and not df_iv.duplicated([unit, time]).any()

    def _demean_cols(M: np.ndarray) -> np.ndarray:
        """Exact within transformation for the chosen FE structure.

        One pass of unit then time demeaning is exact only for a balanced
        panel; otherwise alternate the two projections to convergence
        (the old code always did one pass, which is not the two-way
        within transformation on an unbalanced panel).
        """
        M = np.array(M, dtype=float, copy=True)

        def by(codes: np.ndarray, n_groups: int, A: np.ndarray) -> np.ndarray:
            sums = np.zeros((n_groups, A.shape[1]))
            np.add.at(sums, codes, A)
            cnt = np.bincount(codes, minlength=n_groups)[:, None]
            return A - (sums / cnt)[codes]

        if fe == "unit":
            return by(unit_codes, n_u, M)
        if fe == "time":
            return by(time_codes, n_t, M)
        if fe == "none":
            return M
        scale = max(1.0, float(np.abs(M).max()) if M.size else 1.0)
        for _ in range(100000):
            M_new = by(time_codes, n_t, by(unit_codes, n_u, M))
            delta = float(np.abs(M_new - M).max()) if M.size else 0.0
            M = M_new
            if balanced or delta < 1e-14 * scale:
                return M
        warnings.warn(  # pragma: no cover
            "two-way within transformation did not converge",
            RuntimeWarning,
            stacklevel=3,
        )
        return M  # pragma: no cover

    work_cols = [outcome, endog, "__bartik_iv__"] + cov_cols
    _finite_frame(df_iv[work_cols], name="panel outcome/endog/instrument/covariates")
    df_demean = df_iv.copy()
    df_demean[work_cols] = _demean_cols(df_iv[work_cols].to_numpy(dtype=float))

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

    # Number of absorbed FE parameters (connected panel) + intercept
    k_fe = {"two-way": n_u + n_t - 1, "unit": n_u, "time": n_t, "none": 1}[fe]
    n_obs = len(Y)

    # --- Standard errors --------------------------------------------------
    akm0_ci = None
    if cluster == "shock":
        # Adao-Kolesar-Morales (2019) shock-level SE via the shared kernel
        # (= R ShiftShareSE::ivreg_ss with the FE as controls). The FE are
        # partialled out first (FWL), so the controls passed are the
        # intercept and the demeaned covariates. The share matrix W has one
        # column per shock g_kt: per industry when the shocks do not vary
        # over time (Z_it = s_it' g), per (industry, period) otherwise, so
        # that the instrument is exactly W g in both cases.
        from ._akm import _akm_fit

        shocks_constant = all(
            np.array_equal(
                shocks_by_t[t].reindex(industries).to_numpy(dtype=float),
                shocks_by_t[times[0]].reindex(industries).to_numpy(dtype=float),
            )
            for t in times
        )
        W_blocks = []
        for t in times:
            mask = (df_iv[time] == t).to_numpy()
            S_t = (
                shares_by_t[t]
                .reindex(df_iv.loc[mask, unit])
                .reindex(columns=industries)
                .to_numpy(dtype=float, na_value=0.0)
            )
            block = np.zeros((len(df_iv), len(industries)))
            block[mask] = S_t
            W_blocks.append(block)
        W = sum(W_blocks) if shocks_constant else np.hstack(W_blocks)
        akm = _akm_fit(
            Y,
            Z,
            W,
            np.column_stack([np.ones(n_obs), X_cov]),
            y2=D,
            alpha=alpha,
        )
        se = float(akm["se"]["AKM"])
        akm_se = se
        akm0_ci = (akm["ci_l"]["AKM0"], akm["ci_r"]["AKM0"])
        cluster_label = "shock (AKM 2019)"
    else:
        # Cluster-robust (CR0, no small-sample factor) sandwich on stage 2:
        # V = (Dh'Dh)^-1 [sum_g (Dh_g' e_g)(Dh_g' e_g)'] (Dh'Dh)^-1.
        # 'twoway' is Cameron-Gelbach-Miller two-way clustering,
        # V_unit + V_time - V_unit x time (the old code clustered on the
        # unit x time cell, i.e. HC0, under the 'twoway' label).
        bread = np.linalg.pinv(S2.T @ S2)
        scores_all = S2 * resid[:, None]

        def _meat(codes: np.ndarray) -> np.ndarray:
            n_g = int(codes.max()) + 1
            sg = np.zeros((n_g, S2.shape[1]))
            np.add.at(sg, codes, scores_all)
            return sg.T @ sg

        if cluster == "unit":
            meat = _meat(unit_codes)
        elif cluster == "time":
            meat = _meat(time_codes)
        else:
            cell = pd.factorize(
                pd.Series(unit_codes).astype(str)
                + "_"
                + pd.Series(time_codes).astype(str)
            )[0]
            meat = _meat(unit_codes) + _meat(time_codes) - _meat(cell)
        vcov = bread @ meat @ bread
        se = float(np.sqrt(max(vcov[1, 1], 0.0)))
        akm_se = None
        cluster_label = cluster
    from scipy.stats import norm as _norm

    z = _norm.ppf(1 - alpha / 2)
    ci = (beta - z * se, beta + z * se)

    # First-stage F of the excluded instrument (homoskedastic; restricted
    # model drops Z), df = n - (FE parameters + instrument + covariates).
    rss_u = float(np.sum((D - D_hat) ** 2))
    S1r = np.column_stack([np.ones(n_obs), X_cov])
    pir, *_ = np.linalg.lstsq(S1r, D, rcond=None)
    rss_r = float(np.sum((D - S1r @ pir) ** 2))
    df_fs = n_obs - (k_fe + 1 + X_cov.shape[1])
    first_stage_F = (
        (rss_r - rss_u) / (rss_u / df_fs) if rss_u > 0 and df_fs > 0 else float("nan")
    )

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
    # GPSS weights of the pooled FE-IV: with x~ the endogenous regressor
    # after partialling out the FE and covariates,
    # alpha_kt = g_kt sum_i s_ikt x~_it / sum_{k,t} (same), summed over t per
    # industry. They sum to one and may be negative. (The old code demeaned x
    # by period only -- ignoring unit FE and covariates -- and normalised by
    # the sum of absolute values.)
    if X_cov.shape[1]:
        Xc1 = np.column_stack([np.ones(n_obs), X_cov])
        x_tilde = D - Xc1 @ np.linalg.lstsq(Xc1, D, rcond=None)[0]
    else:
        x_tilde = D - (D.mean() if fe == "none" else 0.0)
    rot_acc = {ind: 0.0 for ind in industries}
    for t in times:
        mask = (df_iv[time] == t).to_numpy()
        shocks_t = shocks_by_t[t]
        aligned = [ind for ind in industries if ind in shocks_t.index]
        S_aligned = (
            shares_by_t[t]
            .reindex(df_iv.loc[mask, unit])[aligned]
            .to_numpy(dtype=float, na_value=0.0)
        )
        g_aligned = shocks_t.loc[aligned].to_numpy(dtype=float)
        alpha_t = g_aligned * (S_aligned.T @ x_tilde[mask])
        for ind, w in zip(aligned, alpha_t):
            rot_acc[ind] += float(w)
    total = sum(rot_acc.values())
    rot_rows = []
    for ind, w in rot_acc.items():
        a_k = w / total if total != 0 else float("nan")
        rot_rows.append(
            {"industry": ind, "rotemberg_weight": a_k, "abs_weight": abs(a_k)}
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
            "akm0_ci": akm0_ci,
            "n_obs": int(len(df_iv)),
            "balanced": bool(balanced),
            "first_stage_F": float(first_stage_F),
        },
        model_info={
            "model_type": "Shift-Share IV (panel)",
            "method": "shift_share_political_panel",
            "fixed_effects": fe_vars,
            "cluster": cluster_label,
        },
    )
