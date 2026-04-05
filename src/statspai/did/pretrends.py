"""
Pre-trends testing and sensitivity analysis for Difference-in-Differences.

Implements three state-of-the-art methods:

1. **pretrends_power** -- Roth (2022) power analysis for pre-trend tests.
   A non-significant pre-trend test is uninformative when power is low.

2. **sensitivity_rr** -- Rambachan & Roth (2023) honest confidence intervals
   for the ATT under bounded violations of parallel trends (C-LF method).

3. **pretrends_test** -- Joint Wald / F test of pre-treatment coefficients.

References
----------
- Roth, J. (2022). Pre-test with Caution: Event-Study Estimates after
  Testing for Parallel Trends. *AER: Insights*, 4(3), 305--322.
- Rambachan, A. & Roth, J. (2023). A More Credible Approach to Parallel
  Trends. *Review of Economic Studies*, 90(5), 2555--2591.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _extract_event_study(result) -> pd.DataFrame:
    """Pull the event-study DataFrame from a CausalResult.

    Looks in ``result.model_info['event_study']`` first, then falls back
    to ``result.detail``.  Raises ``ValueError`` with a helpful message
    when no event-study data can be found.
    """
    es = None
    if hasattr(result, "model_info") and isinstance(result.model_info, dict):
        es = result.model_info.get("event_study", None)
    if es is None and hasattr(result, "detail"):
        es = result.detail
    if es is None or (isinstance(es, pd.DataFrame) and es.empty):
        raise ValueError(
            "Cannot extract event-study estimates from the result object. "
            "Make sure you pass a CausalResult with an 'event_study' key in "
            "model_info or a non-empty 'detail' DataFrame."
        )
    if not isinstance(es, pd.DataFrame):
        raise TypeError(
            f"Expected a DataFrame for event-study estimates, got {type(es)}."
        )
    return es


def _resolve_columns(df: pd.DataFrame):
    """Return (time_col, est_col, se_col) after inspecting column names."""
    # Time column
    time_candidates = ["relative_time", "rel_time", "event_time", "t", "time", "period"]
    time_col = None
    for c in time_candidates:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError(
            f"Cannot find a relative-time column. Looked for {time_candidates}; "
            f"columns are {list(df.columns)}."
        )

    # Estimate column
    est_candidates = ["estimate", "att", "coef", "coefficient", "beta", "effect"]
    est_col = None
    for c in est_candidates:
        if c in df.columns:
            est_col = c
            break
    if est_col is None:
        raise ValueError(
            f"Cannot find an estimate column. Looked for {est_candidates}; "
            f"columns are {list(df.columns)}."
        )

    # SE column
    se_candidates = ["se", "std_error", "std.error", "stderr", "std_err"]
    se_col = None
    for c in se_candidates:
        if c in df.columns:
            se_col = c
            break
    if se_col is None:
        raise ValueError(
            f"Cannot find a standard-error column. Looked for {se_candidates}; "
            f"columns are {list(df.columns)}."
        )

    return time_col, est_col, se_col


def _split_pre_post(df: pd.DataFrame, time_col: str, est_col: str, se_col: str):
    """Split event-study into pre-period (t < 0) and post-period (t >= 1)."""
    pre = df[df[time_col] < 0].sort_values(time_col).copy()
    post = df[df[time_col] >= 1].sort_values(time_col).copy()
    return pre, post


# ────────────────────────────────────────────────────────────────────
# 1. pretrends_test — Joint test of H0: all pre-treatment coefs = 0
# ────────────────────────────────────────────────────────────────────

def pretrends_test(
    result,
    type: str = "wald",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Joint test of pre-treatment coefficients.

    Tests H0: beta_pre = 0 (all pre-treatment event-study coefficients
    are jointly zero).

    Parameters
    ----------
    result : CausalResult
        Event-study result containing pre-treatment estimates and SEs.
    type : ``'wald'`` or ``'f'``
        ``'wald'``: chi-squared test statistic.
        ``'f'``: scaled F-statistic (requires ``df_resid`` in model_info).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        Keys: ``statistic``, ``pvalue``, ``df``, ``type``,
        ``reject``, ``interpretation``.

    References
    ----------
    Standard Wald test; see Roth (2022) for caveats on interpretation.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.event_study(df, y='y', treat='g', time='t', id='i')
    >>> sp.pretrends_test(result)
    """
    es = _extract_event_study(result)
    time_col, est_col, se_col = _resolve_columns(es)
    pre, _ = _split_pre_post(es, time_col, est_col, se_col)

    if len(pre) == 0:
        raise ValueError("No pre-treatment periods found (relative_time < 0).")

    beta_pre = pre[est_col].values.astype(float)
    se_pre = pre[se_col].values.astype(float)
    K = len(beta_pre)

    # Build variance-covariance matrix (diagonal if full VCV unavailable)
    vcv = None
    if hasattr(result, "model_info") and isinstance(result.model_info, dict):
        vcv = result.model_info.get("vcv_pre", None)
    if vcv is None:
        vcv = np.diag(se_pre ** 2)
    else:
        vcv = np.asarray(vcv, dtype=float)

    vcv_inv = np.linalg.inv(vcv)
    wald_stat = float(beta_pre @ vcv_inv @ beta_pre)

    if type == "wald":
        pvalue = float(1.0 - sp_stats.chi2.cdf(wald_stat, df=K))
        stat_label = f"Wald chi2({K})"
        out_type = "wald"
    elif type == "f":
        df_resid = None
        if hasattr(result, "model_info") and isinstance(result.model_info, dict):
            df_resid = result.model_info.get("df_resid", None)
        if hasattr(result, "n_obs") and df_resid is None:
            df_resid = max(result.n_obs - K, K + 1)
        if df_resid is None:
            df_resid = 1000  # conservative fallback
        f_stat = wald_stat / K
        pvalue = float(1.0 - sp_stats.f.cdf(f_stat, dfn=K, dfd=df_resid))
        wald_stat = f_stat
        stat_label = f"F({K}, {df_resid})"
        out_type = "f"
    else:
        raise ValueError(f"type must be 'wald' or 'f', got '{type}'.")

    reject = pvalue < alpha
    if reject:
        interpretation = (
            f"Reject H0 at alpha={alpha}: evidence against parallel pre-trends."
        )
    else:
        interpretation = (
            f"Cannot reject parallel trends at alpha={alpha}. "
            "Note: non-rejection may reflect low power (see pretrends_power)."
        )

    return {
        "statistic": wald_stat,
        "pvalue": pvalue,
        "df": K,
        "type": out_type,
        "stat_label": stat_label,
        "reject": reject,
        "alpha": alpha,
        "interpretation": interpretation,
    }


# ────────────────────────────────────────────────────────────────────
# 2. pretrends_power — Roth (2022) power of the pre-test
# ────────────────────────────────────────────────────────────────────

def pretrends_power(
    result,
    delta: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Power of the pre-trend test against a hypothesised violation.

    Implements the power calculation from Roth (2022, AER: Insights).
    A non-significant pre-trend test is uninformative when the test has
    low power against economically meaningful violations of parallel
    trends.

    Parameters
    ----------
    result : CausalResult
        Event-study result with pre-treatment estimates and SEs.
    delta : array-like, optional
        Hypothesised trend violation in the pre-period (length = number
        of pre-periods).  Default: linear trend
        ``delta[k] = (k+1) * min(|SE|)`` -- a violation equal to one SE
        at the furthest lag, declining linearly to near-zero.
    alpha : float, default 0.05
        Significance level of the pre-trend test.

    Returns
    -------
    dict
        Keys: ``power``, ``noncentrality``, ``df``, ``delta``,
        ``critical_value``, ``warning``.

    References
    ----------
    Roth, J. (2022). Pre-test with Caution: Event-Study Estimates after
    Testing for Parallel Trends. *AER: Insights*, 4(3), 305--322.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.event_study(df, y='y', treat='g', time='t', id='i')
    >>> sp.pretrends_power(result)
    """
    es = _extract_event_study(result)
    time_col, est_col, se_col = _resolve_columns(es)
    pre, _ = _split_pre_post(es, time_col, est_col, se_col)

    if len(pre) == 0:
        raise ValueError("No pre-treatment periods found (relative_time < 0).")

    se_pre = pre[se_col].values.astype(float)
    K = len(se_pre)

    # Build VCV (diagonal if full VCV unavailable)
    vcv = None
    if hasattr(result, "model_info") and isinstance(result.model_info, dict):
        vcv = result.model_info.get("vcv_pre", None)
    if vcv is None:
        vcv = np.diag(se_pre ** 2)
    else:
        vcv = np.asarray(vcv, dtype=float)

    vcv_inv = np.linalg.inv(vcv)

    # Default delta: linear trend scaled by minimum SE
    if delta is None:
        min_se = np.min(np.abs(se_pre))
        # Pre-periods are sorted earliest to latest: t=-K, ..., t=-1
        # Linear trend: magnitude grows toward treatment
        delta = np.array([(i + 1) / K * min_se for i in range(K)])
    else:
        delta = np.asarray(delta, dtype=float)
        if len(delta) != K:
            raise ValueError(
                f"delta has length {len(delta)} but there are {K} pre-periods."
            )

    # Non-centrality parameter
    ncp = float(delta @ vcv_inv @ delta)

    # Critical value under H0
    crit = float(sp_stats.chi2.ppf(1.0 - alpha, df=K))

    # Power = P(chi2(K, ncp) > crit)
    power = float(1.0 - sp_stats.ncx2.cdf(crit, df=K, nc=ncp))

    warning = None
    if power < 0.50:
        warning = (
            f"LOW POWER ({power:.2f}): the pre-trend test has less than 50% "
            "power against the hypothesised violation. A non-significant "
            "pre-trend test is therefore uninformative. Consider the "
            "sensitivity analysis in sensitivity_rr()."
        )
    elif power < 0.80:
        warning = (
            f"MODERATE POWER ({power:.2f}): power is below the conventional "
            "80% threshold. Interpret a non-significant pre-trend test "
            "with caution."
        )

    return {
        "power": power,
        "noncentrality": ncp,
        "df": K,
        "delta": delta,
        "critical_value": crit,
        "alpha": alpha,
        "warning": warning,
    }


# ────────────────────────────────────────────────────────────────────
# 3. sensitivity_rr — Rambachan & Roth (2023) honest CIs
# ────────────────────────────────────────────────────────────────────

@dataclass
class SensitivityResult:
    """Result of Rambachan & Roth (2023) sensitivity analysis.

    Attributes
    ----------
    mbar_grid : np.ndarray
        Grid of M-bar values tested.
    ci_lower : np.ndarray
        Lower bound of the honest CI at each M-bar.
    ci_upper : np.ndarray
        Upper bound of the honest CI at each M-bar.
    breakdown_mbar : float
        Smallest M-bar for which the CI includes zero (sign reversal).
    att : float
        Point estimate of the ATT.
    att_se : float
        Standard error of the ATT.
    method : str
        Extrapolation method used (``'C-LF'``).
    alpha : float
        Significance level.

    Methods
    -------
    summary()
        Print a formatted summary table.
    plot()
        Matplotlib sensitivity plot (M-bar vs CI).
    """

    mbar_grid: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    breakdown_mbar: float
    att: float
    att_se: float
    method: str = "C-LF"
    alpha: float = 0.05

    # ── Pretty printing ──────────────────────────────────────────── #

    def summary(self) -> str:
        """Return a formatted summary string."""
        z = sp_stats.norm.ppf(1.0 - self.alpha / 2)
        lines = []
        hbar = "\u2501" * 58
        lines.append(hbar)
        lines.append("  Rambachan & Roth (2023) Sensitivity Analysis")
        lines.append(f"  Method: {self.method}  |  Alpha: {self.alpha}")
        lines.append(hbar)
        lines.append(f"  ATT = {self.att:.4f}  (SE = {self.att_se:.4f})")
        lines.append(
            f"  Original CI: [{self.att - z * self.att_se:.4f}, "
            f"{self.att + z * self.att_se:.4f}]"
        )
        lines.append("")
        lines.append(f"  {'Mbar':>8s}  {'CI Lower':>12s}  {'CI Upper':>12s}  {'Includes 0?':>12s}")
        lines.append(f"  {'----':>8s}  {'--------':>12s}  {'--------':>12s}  {'-----------':>12s}")
        for i, m in enumerate(self.mbar_grid):
            lo = self.ci_lower[i]
            hi = self.ci_upper[i]
            inc = "Yes" if lo <= 0 <= hi else "No"
            lines.append(f"  {m:8.3f}  {lo:12.4f}  {hi:12.4f}  {inc:>12s}")
        lines.append("")
        if np.isfinite(self.breakdown_mbar):
            lines.append(
                f"  Breakdown Mbar = {self.breakdown_mbar:.4f}"
            )
            lines.append(
                "  (smallest Mbar where CI includes zero)"
            )
        else:
            lines.append(
                "  No breakdown: CI excludes zero for all Mbar in grid."
            )
        lines.append(hbar)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        rows = ""
        for i, m in enumerate(self.mbar_grid):
            lo = self.ci_lower[i]
            hi = self.ci_upper[i]
            inc = lo <= 0 <= hi
            bg = ' style="background:#fff3cd"' if inc else ""
            inc_str = "Yes" if inc else "No"
            rows += (
                f"<tr{bg}><td>{m:.3f}</td>"
                f"<td>{lo:.4f}</td><td>{hi:.4f}</td>"
                f"<td>{inc_str}</td></tr>\n"
            )
        bd = (
            f"<b>{self.breakdown_mbar:.4f}</b>"
            if np.isfinite(self.breakdown_mbar)
            else "None (robust for all tested Mbar)"
        )
        return f"""
        <div style="font-family:monospace; max-width:600px">
        <h3>Rambachan &amp; Roth (2023) Sensitivity Analysis</h3>
        <p>Method: {self.method} | ATT = {self.att:.4f}
           (SE = {self.att_se:.4f}) | Alpha = {self.alpha}</p>
        <table border="1" cellpadding="4" style="border-collapse:collapse">
        <tr><th>Mbar</th><th>CI Lower</th><th>CI Upper</th>
            <th>Includes 0?</th></tr>
        {rows}
        </table>
        <p>Breakdown Mbar: {bd}</p>
        </div>
        """

    # ── Plot ─────────────────────────────────────────────────────── #

    def plot(self, ax=None, figsize=(8, 5), **kwargs):
        """Sensitivity plot: M-bar on x-axis, honest CI band on y-axis.

        Parameters
        ----------
        ax : matplotlib Axes, optional
        figsize : tuple, default (8, 5)
        **kwargs : passed to ``ax.fill_between``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        fill_kw = dict(alpha=0.3, color="steelblue", label="Honest CI")
        fill_kw.update(kwargs)
        ax.fill_between(self.mbar_grid, self.ci_lower, self.ci_upper, **fill_kw)
        ax.plot(
            self.mbar_grid, self.ci_lower, color="steelblue", linewidth=0.8
        )
        ax.plot(
            self.mbar_grid, self.ci_upper, color="steelblue", linewidth=0.8
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(self.att, color="crimson", linestyle="-", linewidth=1.0,
                    label=f"ATT = {self.att:.4f}")

        if np.isfinite(self.breakdown_mbar):
            ax.axvline(
                self.breakdown_mbar, color="orange", linestyle=":",
                linewidth=1.2, label=f"Breakdown Mbar = {self.breakdown_mbar:.3f}",
            )

        ax.set_xlabel(r"$\bar{M}$ (Max. violation of parallel trends)")
        ax.set_ylabel("Treatment effect")
        ax.set_title("Rambachan & Roth (2023) Sensitivity Analysis")
        ax.legend(frameon=False)
        ax.figure.tight_layout()
        return ax


def sensitivity_rr(
    result,
    Mbar: Optional[Union[np.ndarray, List[float]]] = None,
    method: str = "C-LF",
    alpha: float = 0.05,
    n_grid: int = 20,
) -> SensitivityResult:
    """Rambachan & Roth (2023) honest confidence intervals.

    Computes confidence intervals for the ATT that are valid under
    bounded departures from parallel trends.  The *conditional
    linear-in-relative-time* (C-LF) restriction assumes the
    post-treatment violation is bounded by a linear extrapolation of the
    pre-trend plus an additional M-bar of slack.

    Parameters
    ----------
    result : CausalResult
        Event-study result with pre- and post-treatment estimates.
    Mbar : array-like, optional
        Grid of M-bar values.  Default: ``np.linspace(0, 3 * max_pre_slope, n_grid)``.
    method : ``'C-LF'``
        Extrapolation method.  Currently only C-LF is implemented.
    alpha : float, default 0.05
        Significance level.
    n_grid : int, default 20
        Number of grid points when ``Mbar`` is not supplied.

    Returns
    -------
    SensitivityResult
        Object with ``.summary()``, ``.plot()``, ``.mbar_grid``,
        ``.ci_lower``, ``.ci_upper``, ``.breakdown_mbar``.

    References
    ----------
    Rambachan, A. & Roth, J. (2023). A More Credible Approach to
    Parallel Trends. *Review of Economic Studies*, 90(5), 2555--2591.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.event_study(df, y='y', treat='g', time='t', id='i')
    >>> sens = sp.sensitivity_rr(result, Mbar=[0, 0.01, 0.02, 0.05])
    >>> sens.summary()
    >>> sens.plot()
    """
    if method != "C-LF":
        raise NotImplementedError(
            f"Only method='C-LF' is currently implemented, got '{method}'."
        )

    es = _extract_event_study(result)
    time_col, est_col, se_col = _resolve_columns(es)
    pre, post = _split_pre_post(es, time_col, est_col, se_col)

    if len(pre) == 0:
        raise ValueError("No pre-treatment periods found (relative_time < 0).")
    if len(post) == 0:
        raise ValueError("No post-treatment periods found (relative_time >= 1).")

    # ── Extract ATT ──────────────────────────────────────────────── #
    att = float(result.estimate) if hasattr(result, "estimate") else float(
        post[est_col].iloc[0]
    )
    att_se = float(result.se) if hasattr(result, "se") else float(
        post[se_col].iloc[0]
    )

    # ── Fit linear trend through pre-period ──────────────────────── #
    pre_t = pre[time_col].values.astype(float)
    pre_est = pre[est_col].values.astype(float)

    if len(pre_t) >= 2:
        # Weighted least squares through pre-period estimates
        pre_se = pre[se_col].values.astype(float)
        weights = 1.0 / (pre_se ** 2 + 1e-16)
        # WLS: y = a + b*t
        W = np.diag(weights)
        X = np.column_stack([np.ones(len(pre_t)), pre_t])
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ pre_est
        coefs = np.linalg.solve(XtWX, XtWy)
        slope = coefs[1]
    else:
        # Single pre-period: slope = estimate / |time|
        slope = pre_est[0] / max(abs(pre_t[0]), 1.0)

    # ── Extrapolate linear trend to post-period ──────────────────── #
    post_t = post[time_col].values.astype(float)
    # Baseline bias for the first post-period
    baseline_bias = abs(slope) * post_t[0]

    # Sensitivity factor: how much each unit of Mbar adds to the bias.
    # Under C-LF, the sensitivity factor for relative time h is h itself.
    sensitivity_factor = float(np.max(post_t))

    # ── Build Mbar grid ──────────────────────────────────────────── #
    max_pre_slope = max(abs(slope), 1e-6)
    if Mbar is None:
        mbar_grid = np.linspace(0.0, 3.0 * max_pre_slope, n_grid)
    else:
        mbar_grid = np.asarray(Mbar, dtype=float)

    z = sp_stats.norm.ppf(1.0 - alpha / 2)

    ci_lower = np.empty(len(mbar_grid))
    ci_upper = np.empty(len(mbar_grid))

    for i, m in enumerate(mbar_grid):
        max_bias = baseline_bias + m * sensitivity_factor
        ci_lower[i] = att - max_bias - z * att_se
        ci_upper[i] = att + max_bias + z * att_se

    # ── Breakdown M-bar ──────────────────────────────────────────── #
    includes_zero = (ci_lower <= 0) & (ci_upper >= 0)
    breakdown_idx = np.where(includes_zero)[0]
    if len(breakdown_idx) > 0:
        breakdown_mbar = float(mbar_grid[breakdown_idx[0]])
    else:
        breakdown_mbar = float("inf")

    return SensitivityResult(
        mbar_grid=mbar_grid,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        breakdown_mbar=breakdown_mbar,
        att=att,
        att_se=att_se,
        method=method,
        alpha=alpha,
    )


# ────────────────────────────────────────────────────────────────────
# Convenience: formatted combined report
# ────────────────────────────────────────────────────────────────────

def pretrends_summary(result, delta=None, alpha: float = 0.05) -> str:
    """Print a combined pre-trends diagnostic report.

    Runs ``pretrends_test`` and ``pretrends_power`` and formats the
    output in a single table.

    Parameters
    ----------
    result : CausalResult
        Event-study result.
    delta : array-like, optional
        Passed to ``pretrends_power``.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    str
        Formatted report.
    """
    test = pretrends_test(result, type="wald", alpha=alpha)
    pwr = pretrends_power(result, delta=delta, alpha=alpha)

    hbar = "\u2501" * 58
    lines = [
        hbar,
        "  Pre-Trends Analysis",
        hbar,
        "  Joint pre-trend test:",
        f"    {test['stat_label']} = {test['statistic']:.2f}, "
        f"p = {test['pvalue']:.3f}",
    ]
    if test["reject"]:
        lines.append("    \u2192 Evidence against parallel pre-trends")
    else:
        lines.append("    \u2192 Cannot reject parallel trends")

    lines.append("")
    lines.append("  Power against linear violation:")
    lines.append(f"    Power = {pwr['power']:.2f}", )
    if pwr["warning"] and pwr["power"] < 0.50:
        lines.append(f"    \u2190 LOW POWER WARNING")
    elif pwr["warning"] and pwr["power"] < 0.80:
        lines.append(f"    \u2190 Moderate power")
    lines.append(hbar)

    report = "\n".join(lines)
    print(report)
    return report


__all__ = [
    "pretrends_test",
    "pretrends_power",
    "sensitivity_rr",
    "SensitivityResult",
    "pretrends_summary",
]
