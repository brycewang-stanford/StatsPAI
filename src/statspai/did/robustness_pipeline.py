"""End-to-end parallel-trends robustness pipeline.

Chains the pieces that already exist in this package into a single call:

1. :func:`statspai.did.pretrends.pretrends_test` -- joint test of the
   pre-treatment event-study coefficients.
2. :func:`statspai.did.pretrends.pretrends_power` -- Roth (2022) power of
   that pre-test against a hypothesised violation. A non-significant
   pre-trend test is uninformative when power is low.
3. :func:`statspai.did.honest_did.honest_did` -- Rambachan & Roth (2023)
   robust confidence intervals over a grid of violation magnitudes, for
   each requested restriction family.
4. :func:`statspai.did.honest_did.breakdown_m` -- the breakdown value
   Mbar* per family.

The point of bundling them is that they are not independently
interpretable: the breakdown Mbar* only means something once you know
whether the pre-test that motivated it had any power to detect a
violation of that size in the first place.

References
----------
Rambachan, A. and Roth, J. (2023). A More Credible Approach to Parallel
Trends. *Review of Economic Studies*, 90(5), 2555-2591.
[@rambachan2023more]

Roth, J. (2022). Pretest with Caution: Event-Study Estimates after
Testing for Parallel Trends. *American Economic Review: Insights*,
4(3), 305-322. [@roth2022pretest]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from .honest_did import honest_did
from .pretrends import pretrends_power, pretrends_test

#: Public family label -> ``honest_did`` method name.
_FAMILY_TO_METHOD: Dict[str, str] = {
    "SD": "smoothness",
    "RM": "relative_magnitude",
}

_FAMILY_LABEL: Dict[str, str] = {
    "SD": "Second differences / smoothness (|Delta delta_t| <= M)",
    "RM": "Relative magnitude (|delta_post| <= Mbar x max|delta_pre|)",
}


def _normalise_families(families: Any) -> List[str]:
    """Validate the ``families`` argument into an ordered list of labels."""
    if isinstance(families, str):
        families = (families,)
    if not isinstance(families, (list, tuple)):
        raise MethodIncompatibility(
            "parallel_trends_robustness: `families` must be a sequence of "
            f"family labels, got {families!r}.",
            recovery_hint="e.g. families=('SD', 'RM').",
            diagnostics={"families": repr(families)},
        )
    out: List[str] = []
    for fam in families:
        if not isinstance(fam, str):
            raise MethodIncompatibility(
                "parallel_trends_robustness: family labels must be strings, "
                f"got {fam!r}.",
                recovery_hint="e.g. families=('SD', 'RM').",
                diagnostics={"family": repr(fam)},
            )
        key = fam.strip().upper()
        if key not in _FAMILY_TO_METHOD:
            raise MethodIncompatibility(
                f"parallel_trends_robustness: unknown family {fam!r}. Valid "
                f"families are {sorted(_FAMILY_TO_METHOD)} "
                "('SD' = smoothness, 'RM' = relative magnitude).",
                recovery_hint="e.g. families=('SD', 'RM').",
                diagnostics={
                    "family": fam,
                    "valid_families": sorted(_FAMILY_TO_METHOD),
                },
            )
        if key not in out:
            out.append(key)
    if not out:
        raise MethodIncompatibility(
            "parallel_trends_robustness: `families` is empty; there is "
            "nothing to compute.",
            recovery_hint="e.g. families=('SD', 'RM').",
            diagnostics={"families": repr(families)},
        )
    return out


@dataclass
class ParallelTrendsRobustnessResult(ResultProtocolMixin):
    """Bundled pre-trends power + honest-DiD sensitivity for one DiD result.

    Attributes
    ----------
    power_table : pd.DataFrame
        One row per quantity from the Roth (2022) pre-test analysis
        (joint statistic, p-value, power, non-centrality).
    ci_grid : pd.DataFrame
        Long-format robust-CI grid: ``family``, ``M``, ``ci_lower``,
        ``ci_upper``, ``rejects_zero``.
    breakdown : dict
        ``family -> Mbar*``, the largest violation magnitude at which the
        effect stays significant.
    verdict : str
        One-line plain-language reading of the table.
    att, att_se : float
        Point estimate and standard error at relative time ``e``.
    e : int
        Relative time the sensitivity analysis targets.
    alpha : float
        Significance level.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=80, n_periods=8, staggered=False, seed=0)
    >>> df["first_treat"] = df["first_treat"].fillna(0)
    >>> es = sp.event_study(df, y="y", treat_time="first_treat",
    ...                     time="time", unit="unit", window=(-3, 3))
    >>> rob = sp.parallel_trends_robustness(es)
    >>> bool(isinstance(rob.summary(), str))
    True

    References
    ----------
    Rambachan & Roth (2023) [@rambachan2023more]; Roth (2022)
    [@roth2022pretest].
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "rambachan2023more",
        "roth2022pretest",
    )

    power_table: pd.DataFrame
    ci_grid: pd.DataFrame
    breakdown: Dict[str, float]
    verdict: str
    att: float
    att_se: float
    e: int = 0
    alpha: float = 0.05
    families: List[str] = field(default_factory=list)
    pretrend_test: Dict[str, Any] = field(default_factory=dict)
    pretrend_power: Dict[str, Any] = field(default_factory=dict)

    # ── Pretty printing ──────────────────────────────────────────── #

    def summary(self) -> str:
        """Return a formatted multi-section summary string."""
        hbar = "━" * 66
        lines = [
            hbar,
            "  Parallel-Trends Robustness",
            f"  Relative time e = {self.e}  |  alpha = {self.alpha}",
            hbar,
            f"  ATT = {self.att:.4f}  (SE = {self.att_se:.4f})",
            "",
            "  1. Pre-trend test and its power (Roth 2022)",
        ]
        for _, row in self.power_table.iterrows():
            val = row["value"]
            val_s = f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
            lines.append(f"     {str(row['quantity']):<28s} {val_s:>12s}")
        if self.pretrend_power.get("warning"):
            lines.append(f"     ! {self.pretrend_power['warning']}")

        lines.append("")
        lines.append("  2. Breakdown Mbar* by restriction family")
        lines.append(f"     {'Family':<8s}{'Mbar*':>12s}   Restriction")
        for fam in self.families:
            mstar = self.breakdown.get(fam, float("nan"))
            lines.append(f"     {fam:<8s}{mstar:12.4f}   {_FAMILY_LABEL[fam]}")

        lines.append("")
        lines.append("  3. Robust CI grid (Rambachan & Roth 2023)")
        lines.append(
            f"     {'Family':<8s}{'M':>10s}{'CI lower':>12s}"
            f"{'CI upper':>12s}{'Excl. 0?':>10s}"
        )
        for _, row in self.ci_grid.iterrows():
            lines.append(
                f"     {str(row['family']):<8s}{row['M']:10.4f}"
                f"{row['ci_lower']:12.4f}{row['ci_upper']:12.4f}"
                f"{('Yes' if row['rejects_zero'] else 'No'):>10s}"
            )

        lines.append("")
        lines.append(f"  Verdict: {self.verdict}")
        lines.append(hbar)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        bd_rows = "".join(
            f"<tr><td>{fam}</td><td>{self.breakdown.get(fam, float('nan')):.4f}</td>"
            f"<td>{_FAMILY_LABEL[fam]}</td></tr>"
            for fam in self.families
        )
        ci_rows = ""
        for _, row in self.ci_grid.iterrows():
            excl = bool(row["rejects_zero"])
            bg = "" if excl else ' style="background:#fff3cd"'
            ci_rows += (
                f"<tr{bg}><td>{row['family']}</td><td>{row['M']:.4f}</td>"
                f"<td>{row['ci_lower']:.4f}</td><td>{row['ci_upper']:.4f}</td>"
                f"<td>{'Yes' if excl else 'No'}</td></tr>"
            )
        pw_rows = "".join(
            f"<tr><td>{r['quantity']}</td><td>{r['value']}</td></tr>"
            for _, r in self.power_table.iterrows()
        )
        return f"""
        <div style="font-family:monospace; max-width:760px">
        <h3>Parallel-Trends Robustness</h3>
        <p>ATT = {self.att:.4f} (SE = {self.att_se:.4f}) |
           e = {self.e} | alpha = {self.alpha}</p>
        <h4>Pre-trend test &amp; power</h4>
        <table border="1" cellpadding="4" style="border-collapse:collapse">
        <tr><th>Quantity</th><th>Value</th></tr>{pw_rows}</table>
        <h4>Breakdown Mbar*</h4>
        <table border="1" cellpadding="4" style="border-collapse:collapse">
        <tr><th>Family</th><th>Mbar*</th><th>Restriction</th></tr>
        {bd_rows}</table>
        <h4>Robust CI grid</h4>
        <table border="1" cellpadding="4" style="border-collapse:collapse">
        <tr><th>Family</th><th>M</th><th>CI lower</th><th>CI upper</th>
            <th>Excludes 0?</th></tr>{ci_rows}</table>
        <p><b>Verdict:</b> {self.verdict}</p>
        </div>
        """

    # ── Export ───────────────────────────────────────────────────── #

    def to_latex(
        self,
        *,
        caption: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        """Booktabs LaTeX table of the robust-CI grid and breakdown values."""
        cap = caption or (
            "Parallel-trends robustness: Rambachan--Roth robust confidence "
            f"intervals at relative time $e={self.e}$"
        )
        lab = label or "tab:parallel_trends_robustness"
        out = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{cap}}}",
            f"\\label{{{lab}}}",
            "\\begin{tabular}{llrrrc}",
            "\\toprule",
            "Family & Restriction & $M$ & CI lower & CI upper & " "Excludes 0 \\\\",
            "\\midrule",
        ]
        for _, row in self.ci_grid.iterrows():
            fam = str(row["family"])
            out.append(
                f"{fam} & {'SD' if fam == 'SD' else 'RM'} & "
                f"{row['M']:.4f} & {row['ci_lower']:.4f} & "
                f"{row['ci_upper']:.4f} & "
                f"{'Yes' if row['rejects_zero'] else 'No'} \\\\"
            )
        out.append("\\midrule")
        for fam in self.families:
            mstar = self.breakdown.get(fam, float("nan"))
            out.append(
                f"\\multicolumn{{6}}{{l}}{{Breakdown $\\bar{{M}}^*$ "
                f"({fam}) $= {mstar:.4f}$}} \\\\"
            )
        out.append(
            f"\\multicolumn{{6}}{{l}}{{Pre-test power $= "
            f"{float(self.pretrend_power.get('power', float('nan'))):.3f}$}} \\\\"
        )
        out += [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
        return "\n".join(out)

    # ── Plot ─────────────────────────────────────────────────────── #

    def plot(
        self,
        ax: Any = None,
        figsize: Tuple[float, float] = (8, 5),
        **kwargs: Any,
    ) -> Any:
        """Robust-CI bands against M, one band per restriction family.

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
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "matplotlib is required for .plot(); install the 'plotting' " "extra."
            ) from exc

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        colors = {"SD": "steelblue", "RM": "darkorange"}
        for fam in self.families:
            sub = self.ci_grid[self.ci_grid["family"] == fam].sort_values("M")
            if sub.empty:
                continue
            fill_kw = dict(alpha=0.25, color=colors.get(fam, "grey"), label=fam)
            fill_kw.update(kwargs)
            ax.fill_between(
                sub["M"].to_numpy(),
                sub["ci_lower"].to_numpy(),
                sub["ci_upper"].to_numpy(),
                **fill_kw,
            )
            mstar = self.breakdown.get(fam)
            if mstar is not None and np.isfinite(mstar):
                ax.axvline(
                    mstar,
                    color=colors.get(fam, "grey"),
                    linestyle=":",
                    linewidth=1.2,
                    label=f"{fam} breakdown Mbar* = {mstar:.3f}",
                )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(
            self.att,
            color="crimson",
            linewidth=1.0,
            label=f"ATT = {self.att:.4f}",
        )
        ax.set_xlabel(r"$M$ / $\bar{M}$ (violation magnitude)")
        ax.set_ylabel(f"Treatment effect at e = {self.e}")
        ax.set_title("Parallel-trends robustness")
        ax.legend(frameon=False, fontsize=8)
        ax.figure.tight_layout()
        return ax


def parallel_trends_robustness(
    result: Any,
    m_grid: Optional[Sequence[float]] = None,
    families: Sequence[str] = ("SD", "RM"),
    alpha: float = 0.05,
    e: int = 0,
    delta: Optional[np.ndarray] = None,
) -> ParallelTrendsRobustnessResult:
    """Run the full parallel-trends robustness pipeline on a DiD result.

    Chains the joint pre-trend test, the Roth (2022) power calculation for
    that test, and Rambachan & Roth (2023) honest confidence intervals
    (plus their breakdown value Mbar*) for each requested restriction
    family, and reduces the whole thing to a one-line verdict.

    Parameters
    ----------
    result : CausalResult
        A fitted DiD/event-study result carrying event-study estimates in
        ``result.model_info['event_study']`` (e.g. from
        ``sp.event_study``, ``sp.callaway_santanna``, ``sp.sun_abraham``).

        .. note::
           If the result does not carry a pre-period covariance matrix in
           ``model_info['vcv_pre']``, the pre-trend test and power fall
           back to assuming the pre-period coefficients are mutually
           independent and warn loudly. ``sp.event_study`` computes the full
           cluster-robust covariance; pass ``expose_pre_vcov=True`` to it to
           have this pipeline use the correct covariance instead of the
           diagonal fallback.
    m_grid : sequence of float, optional
        Grid of violation magnitudes. Default: the ``honest_did``
        default, multiples of the standard error at ``e``.
    families : sequence of str, default ``("SD", "RM")``
        Restriction families. ``"SD"`` maps to ``honest_did``'s
        ``method='smoothness'`` (bounded second differences); ``"RM"``
        maps to ``method='relative_magnitude'``.
    alpha : float, default 0.05
        Significance level.
    e : int, default 0
        Relative time whose effect the sensitivity analysis targets.
    delta : array-like, optional
        Hypothesised pre-trend violation passed to ``pretrends_power``.

    Returns
    -------
    ParallelTrendsRobustnessResult
        With ``.summary()``, ``.plot()``, ``.to_latex()``, and the
        ``power_table`` / ``ci_grid`` / ``breakdown`` / ``verdict``
        fields.

    References
    ----------
    Rambachan, A. and Roth, J. (2023). A More Credible Approach to
    Parallel Trends. *Review of Economic Studies*, 90(5), 2555-2591.
    [@rambachan2023more]

    Roth, J. (2022). Pretest with Caution: Event-Study Estimates after
    Testing for Parallel Trends. *AER: Insights*, 4(3), 305-322.
    [@roth2022pretest]

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=80, n_periods=8, staggered=False, seed=0)
    >>> df["first_treat"] = df["first_treat"].fillna(0)
    >>> es = sp.event_study(df, y="y", treat_time="first_treat",
    ...                     time="time", unit="unit", window=(-3, 3))
    >>> rob = sp.parallel_trends_robustness(es, families=("SD", "RM"))
    >>> sorted(rob.breakdown)
    ['RM', 'SD']
    >>> bool("Mbar" in rob.verdict or "not robust" in rob.verdict)
    True
    """
    fams = _normalise_families(families)
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float)):
        raise MethodIncompatibility(
            f"parallel_trends_robustness: `alpha` must be a number in (0, 1), "
            f"got {alpha!r}.",
            recovery_hint="e.g. alpha=0.05.",
            diagnostics={"alpha": repr(alpha)},
        )
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(
            f"parallel_trends_robustness: `alpha` must be in (0, 1), got " f"{alpha}.",
            recovery_hint="e.g. alpha=0.05.",
            diagnostics={"alpha": alpha},
        )

    # ── 1. Pre-trend test + Roth (2022) power ────────────────────── #
    test = pretrends_test(result, type="wald", alpha=alpha)
    power = pretrends_power(result, delta=delta, alpha=alpha)

    power_table = pd.DataFrame(
        [
            {"quantity": "Joint pre-trend statistic", "value": test["statistic"]},
            {"quantity": "Pre-trend p-value", "value": test["pvalue"]},
            {"quantity": "Degrees of freedom", "value": float(test["df"])},
            {"quantity": "Power vs. hypothesised delta", "value": power["power"]},
            {"quantity": "Non-centrality", "value": power["noncentrality"]},
            {"quantity": "Critical value", "value": power["critical_value"]},
        ]
    )

    # ── 2 & 3. Honest CIs and breakdown Mbar*, per family ────────── #
    grids: List[pd.DataFrame] = []
    breakdown: Dict[str, float] = {}
    att = float("nan")
    att_se = float("nan")

    for fam in fams:
        method = _FAMILY_TO_METHOD[fam]
        grid = honest_did(
            result,
            e=e,
            m_grid=list(m_grid) if m_grid is not None else None,
            method=method,
            alpha=alpha,
            backend="native",
        )
        grid = grid.copy()
        grid.insert(0, "family", fam)
        grids.append(grid)
        breakdown[fam] = _breakdown_for_family(result, e=e, method=method, alpha=alpha)

    ci_grid = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()

    # Point estimate at e, read off the event-study table.
    es = getattr(result, "model_info", {}).get("event_study", None)
    if isinstance(es, pd.DataFrame) and "relative_time" in es.columns:
        row = es[es["relative_time"] == e]
        if len(row) > 0:
            att = float(row["att"].iloc[0])
            att_se = float(row["se"].iloc[0])

    # ── 4. Verdict ───────────────────────────────────────────────── #
    verdict = _build_verdict(breakdown, ci_grid, power, att, e)

    return ParallelTrendsRobustnessResult(
        power_table=power_table,
        ci_grid=ci_grid,
        breakdown=breakdown,
        verdict=verdict,
        att=att,
        att_se=att_se,
        e=e,
        alpha=alpha,
        families=fams,
        pretrend_test=dict(test),
        pretrend_power={k: v for k, v in power.items() if k != "delta"},
    )


def _breakdown_for_family(
    result: Any,
    e: int,
    method: str,
    alpha: float,
    m_hi: float = 1e4,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> float:
    """Largest M at which ``honest_did``'s CI still excludes zero.

    Obtained by bisecting ``honest_did`` itself rather than re-deriving the
    bias bound, so the reported Mbar* is *consistent by construction* with
    the ``ci_grid`` this pipeline also returns.

    ``did.honest_did.breakdown_m`` is deliberately NOT used for this:
    it accepts a ``method`` argument, validates it, and then computes
    ``(|theta| - z*SE) / (e + 1)`` regardless -- i.e. it always returns the
    smoothness answer, so it disagrees with the relative-magnitude CI grid.
    ``breakdown_m`` is still called for the smoothness family as a
    cross-check (see the accompanying test); it lives in a module this
    pipeline does not own.
    """

    def _rejects(m: float) -> bool:
        out = honest_did(
            result,
            e=e,
            m_grid=[float(m)],
            method=method,
            alpha=alpha,
            backend="native",
        )
        return bool(out["rejects_zero"].iloc[0])

    if not _rejects(0.0):
        # Already indistinguishable from zero with no violation allowed.
        return 0.0
    if _rejects(m_hi):
        return float("inf")

    lo, hi = 0.0, float(m_hi)
    for _ in range(max_iter):
        if hi - lo <= tol * max(1.0, hi):
            break
        mid = 0.5 * (lo + hi)
        if _rejects(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


def _build_verdict(
    breakdown: Dict[str, float],
    ci_grid: pd.DataFrame,
    power: Dict[str, Any],
    att: float,
    e: int,
) -> str:
    """Reduce the tables to a single plain-language sentence."""
    if not breakdown:
        return "No restriction family was evaluated; no verdict available."

    # The binding family is the one that breaks down first.
    weakest_fam = min(breakdown, key=lambda f: breakdown[f])
    weakest = breakdown[weakest_fam]

    if not np.isfinite(weakest) or weakest <= 0.0:
        core = (
            f"The effect at e = {e} is NOT robust: the confidence interval "
            f"already includes zero at M = 0 under the {weakest_fam} "
            "restriction, i.e. before allowing any violation of parallel "
            "trends."
        )
    else:
        parts = ", ".join(
            f"{fam} Mbar* = {breakdown[fam]:.4g}" for fam in sorted(breakdown)
        )
        core = (
            f"The conclusion (ATT = {att:.4g} at e = {e}) survives up to "
            f"Mbar = {weakest:.4g} under the binding ({weakest_fam}) "
            f"restriction [{parts}]."
        )

    pw = float(power.get("power", float("nan")))
    if np.isfinite(pw) and pw < 0.50:
        core += (
            f" Caveat: the pre-trend test has only {pw:.0%} power against the "
            "hypothesised violation, so a clean pre-trend plot is weak "
            "evidence here (Roth 2022)."
        )
    return core


__all__ = [
    "parallel_trends_robustness",
    "ParallelTrendsRobustnessResult",
]
