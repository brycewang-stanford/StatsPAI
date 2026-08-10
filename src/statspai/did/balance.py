"""Covariate balance for DiD designs — levels *and* changes.

Baker, Callaway, Cunningham, Goodman-Bacon and Sant'Anna (2026, §4.1)
make covariate balance the standard piece of empirical evidence about
whether *unconditional* parallel trends is plausible, and their Table 4
reports it in a specific shape that no general-purpose balance table
produces:

1. **Two panels, not one.** Baseline covariate *levels* (X at g-1) and
   covariate *changes* (ΔX between the base and comparison period). The
   change panel is the informative half — in their Medicaid application
   several imbalances flip sign between levels and changes, because DiD
   identifies off trends and "areas that are poor are not the same as
   areas that are becoming poor."
2. **Normalized difference**, not the t-statistic. Imbens and Rubin
   (2015, ch. 14) scale the gap by the pooled standard deviation, so it
   does not shrink mechanically as n grows; the |value| > 0.25 rule of
   thumb is theirs (p. 277). A t-test answers "is the imbalance
   detectable", which is not the question.
3. **Weighted and unweighted side by side**, because ω changes the
   comparison the same way it changes the estimand.

The result also flags the covariate-versus-mechanism problem the paper
raises: differential movement in ΔX only signals a parallel-trends
violation if X cannot itself be affected by the treatment. That is a
judgement about institutions, not a data question, so this module states
it rather than deciding it.

References
----------
baker2026difference, imbens2015causal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["did_balance", "DiDBalanceResult"]


# Imbens & Rubin (2015, p. 277): normalized differences above this in
# absolute value indicate potentially problematic imbalance. Not a law —
# the paper's own footnote 12 notes 0.1 is sometimes already worrying.
_IR_THRESHOLD = 0.25


# ----------------------------------------------------------------------
# Result object
# ----------------------------------------------------------------------


@dataclass
class DiDBalanceResult:
    """Balance table for a DiD design.

    Attributes
    ----------
    table : pandas.DataFrame
        Long form, one row per (panel, covariate). Columns:
        ``panel`` (``'levels'`` / ``'changes'``), ``covariate``,
        ``mean_treated``, ``mean_comparison``, ``norm_diff``,
        ``abs_norm_diff``, ``flagged``, and — when ω is supplied — the
        same three statistics again with a ``w_`` prefix.
    levels, changes : pandas.DataFrame
        The two panels, split out for convenience.
    threshold : float
        Absolute normalized-difference cutoff used for ``flagged``.
    n_treated, n_comparison : int
    base_period, comparison_period : Any
    weighted : bool
    diagnostics : dict
    """

    table: pd.DataFrame
    levels: pd.DataFrame
    changes: pd.DataFrame
    threshold: float
    n_treated: int
    n_comparison: int
    base_period: Any
    comparison_period: Any
    weighted: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------
    @property
    def flagged(self) -> List[str]:
        """Covariates breaching the threshold in either panel."""
        col = "w_abs_norm_diff" if self.weighted else "abs_norm_diff"
        hit = self.table[self.table[col] > self.threshold]
        return sorted(set(hit["covariate"].tolist()))

    @property
    def max_abs_norm_diff(self) -> float:
        col = "w_abs_norm_diff" if self.weighted else "abs_norm_diff"
        vals = self.table[col].to_numpy(dtype=float)
        return float(np.nanmax(vals)) if vals.size else float("nan")

    def summary(self) -> str:
        """Fixed-width report in the shape of the paper's Table 4."""
        lines: List[str] = []
        lines.append("=" * 78)
        lines.append("DiD covariate balance (Baker et al. 2026, Table 4 shape)")
        lines.append("=" * 78)
        lines.append(
            f"treated n = {self.n_treated}   comparison n = {self.n_comparison}"
            f"   base period = {self.base_period}"
            f"   comparison period = {self.comparison_period}"
        )
        lines.append(
            f"normalized difference = (Xbar_T - Xbar_C) / "
            f"sqrt((S2_T + S2_C)/2);  flag at |.| > {self.threshold}"
        )
        lines.append("")

        wtd = self.weighted
        for panel_name, panel_df, blurb in (
            ("levels", self.levels, f"Covariate LEVELS at t = {self.base_period}"),
            (
                "changes",
                self.changes,
                f"Covariate CHANGES, {self.base_period} -> "
                f"{self.comparison_period}",
            ),
        ):
            if panel_df.empty:
                continue
            lines.append(blurb)
            if wtd:
                head = (
                    f"  {'variable':<22}{'compar.':>10}{'treated':>10}"
                    f"{'n.diff':>9}   {'w-comp.':>10}{'w-treat':>10}{'w-n.diff':>10}"
                )
            else:
                head = f"  {'variable':<22}{'compar.':>10}{'treated':>10}{'n.diff':>9}"
            lines.append(head)
            lines.append("  " + "-" * (len(head) - 2))
            for _, r in panel_df.iterrows():
                mark = "  *" if bool(r["flagged"]) else "   "
                row = (
                    f"  {str(r['covariate'])[:22]:<22}"
                    f"{r['mean_comparison']:>10.3f}{r['mean_treated']:>10.3f}"
                    f"{_fmt_nd(r['norm_diff']):>9}"
                )
                if wtd:
                    row += (
                        f"   {r['w_mean_comparison']:>10.3f}"
                        f"{r['w_mean_treated']:>10.3f}"
                        f"{_fmt_nd(r['w_norm_diff']):>10}"
                    )
                lines.append(row + mark)
            lines.append("")

        flagged = self.flagged
        lines.append("-" * 78)
        if flagged:
            lines.append(
                f"IMBALANCED (|norm. diff| > {self.threshold}): " + ", ".join(flagged)
            )
            lines.append(
                "  Unconditional parallel trends is questionable. Move to a "
                "conditional design:"
            )
            lines.append(
                "    sp.callaway_santanna(..., x=[...], estimator='dr')   "
                "# doubly robust"
            )
            lines.append(
                "  The paper recommends DR over regression-adjustment or IPW "
                "alone (§4.4)."
            )
        else:
            lines.append(
                f"No covariate breaches |norm. diff| > {self.threshold} in "
                "either panel."
            )
            lines.append(
                "  Balance is evidence *for* unconditional PT, not proof: PT "
                "restricts unobserved"
            )
            lines.append(
                "  potential-outcome trends, which no covariate table can see."
            )
        lines.append("")
        lines.append(
            "Reading the CHANGES panel: differential movement in X signals a "
            "PT violation ONLY"
        )
        lines.append(
            "if X cannot be affected by the treatment. If it can, the same "
            "gap may be a causal"
        )
        lines.append(
            "effect running through X (a 'bad control'), and conditioning on "
            "it biases the ATT."
        )
        lines.append(
            "That is a question about institutions, not about data "
            "(Caetano et al. 2024)."
        )
        lines.append("=" * 78)
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"DiDBalanceResult(n_treated={self.n_treated}, "
            f"n_comparison={self.n_comparison}, "
            f"flagged={self.flagged}, weighted={self.weighted})"
        )

    def to_latex(self, caption: Optional[str] = None) -> str:
        """booktabs fragment mirroring the paper's Table 4 layout."""
        cap = caption or "Covariate balance statistics"
        ncol = 7 if self.weighted else 4
        out = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{cap}}}",
            "\\begin{tabular}{l" + "r" * (ncol - 1) + "}",
            "\\toprule",
        ]
        if self.weighted:
            out.append(
                "Variable & Comparison & Treated & Norm. diff. "
                "& Comparison & Treated & Norm. diff. \\\\"
            )
            out.append(
                "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}"
                " & \\multicolumn{3}{c}{Unweighted} "
                "& \\multicolumn{3}{c}{Weighted} \\\\"
            )
        else:
            out.append("Variable & Comparison & Treated & Norm. diff. \\\\")
        out.append("\\midrule")
        for panel_name, panel_df in (
            ("Covariate levels", self.levels),
            ("Covariate differences", self.changes),
        ):
            if panel_df.empty:
                continue
            out.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{panel_name}}}}} \\\\")
            for _, r in panel_df.iterrows():
                cells = [
                    str(r["covariate"]).replace("_", "\\_"),
                    f"{r['mean_comparison']:.2f}",
                    f"{r['mean_treated']:.2f}",
                    f"{r['norm_diff']:.2f}",
                ]
                if self.weighted:
                    cells += [
                        f"{r['w_mean_comparison']:.2f}",
                        f"{r['w_mean_treated']:.2f}",
                        f"{r['w_norm_diff']:.2f}",
                    ]
                out.append(" & ".join(cells) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        return "\n".join(out)


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------


def _fmt_nd(x: float) -> str:
    """Render a normalized difference, keeping +/-inf readable."""
    if not np.isfinite(x):
        return "nan" if np.isnan(x) else ("+inf" if x > 0 else "-inf")
    return f"{x:.3f}"


def _wmean(v: np.ndarray, w: np.ndarray) -> float:
    tot = float(w.sum())
    return float(np.dot(w, v) / tot) if tot > 0 else float("nan")


def _wvar(v: np.ndarray, w: np.ndarray) -> float:
    """Frequency-weighted variance with the (n-1)-style bias correction.

    Uses the reliability-weight form ``sum(w (v - vbar)^2) / (sum w -
    sum(w^2)/sum w)``, which reduces to the usual ``ddof=1`` sample
    variance when every weight is 1.
    """
    tot = float(w.sum())
    if tot <= 0:
        return float("nan")
    mu = _wmean(v, w)
    num = float(np.dot(w, (v - mu) ** 2))
    denom = tot - float(np.dot(w, w)) / tot
    return num / denom if denom > 0 else float("nan")


def _norm_diff(
    v_t: np.ndarray,
    v_c: np.ndarray,
    w_t: np.ndarray,
    w_c: np.ndarray,
) -> Dict[str, float]:
    """Imbens-Rubin normalized difference and the two group means.

    Degenerate pooled variance needs care. A covariate that moves
    deterministically within group — common in the *changes* panel, where
    ΔX can be a fixed policy step — has S2_T = S2_C = 0 up to rounding.
    Dividing by that pooled SD turns a 1e-16 rounding gap into a
    normalized difference of 1e15 and flags a perfectly balanced
    covariate as catastrophically imbalanced. Detect the degenerate case
    on a *relative* scale and resolve it honestly instead:

    - means also equal  -> 0.0   (balanced, no dispersion)
    - means differ      -> +/-inf (complete separation; genuinely
      imbalanced, and the magnitude is not meaningfully finite)
    """
    m_t, m_c = _wmean(v_t, w_t), _wmean(v_c, w_c)
    s2_t, s2_c = _wvar(v_t, w_t), _wvar(v_c, w_c)
    pooled = float(np.sqrt((s2_t + s2_c) / 2.0))
    gap = m_t - m_c

    # Scale against the magnitude of the data, not an absolute epsilon:
    # a covariate measured in millions has a "small" SD of 1e-6 * 1e6.
    scale = max(abs(m_t), abs(m_c), 1.0)
    degenerate = not np.isfinite(pooled) or pooled <= 1e-10 * scale

    if degenerate:
        nd = 0.0 if abs(gap) <= 1e-10 * scale else np.sign(gap) * np.inf
    else:
        nd = gap / pooled
    return {
        "mean_treated": m_t,
        "mean_comparison": m_c,
        "norm_diff": float(nd),
    }


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def did_balance(
    data: pd.DataFrame,
    covariates: Sequence[str],
    *,
    g: str,
    t: str,
    i: str,
    weights: Optional[str] = None,
    base_period: Optional[Any] = None,
    comparison_period: Optional[Any] = None,
    cohort: Optional[Any] = None,
    control_group: str = "nevertreated",
    threshold: float = _IR_THRESHOLD,
) -> DiDBalanceResult:
    """Covariate balance in levels **and** changes for a DiD design.

    Reproduces the shape of Table 4 in Baker et al. (2026): for each
    covariate, the treated and comparison group means plus the
    Imbens-Rubin normalized difference, computed once on baseline
    *levels* and once on *changes* across the treatment date, optionally
    both weighted and unweighted.

    Parameters
    ----------
    data : DataFrame
        Long-format panel.
    covariates : sequence of str
        Columns to audit. May be time-varying; the changes panel is only
        meaningful for those that are.
    g : str
        First-treatment period per unit; ``0`` (or ``inf``) marks
        never-treated.
    t : str
        Time period column.
    i : str
        Unit identifier.
    weights : str, optional
        Unit weights ω. When given, the weighted statistics are reported
        alongside the unweighted ones, exactly as the paper does — the
        two answer different questions and neither is a check on the
        other.
    base_period : optional
        Pre-treatment period for the levels panel. Defaults to ``g-1``
        for the cohort under audit.
    comparison_period : optional
        Second period for the changes panel. Defaults to the cohort's
        treatment date ``g``.
    cohort : optional
        Which treated cohort to audit. Defaults to the largest one.
        Balance is a two-group statistic, so a staggered design has one
        table per cohort rather than one overall.
    control_group : {'nevertreated', 'notyettreated'}, default
        'nevertreated'
        Which units form the comparison group. Should match the
        comparison group of the estimator you intend to run — balance
        against a group you will not use is not evidence about your
        design.
    threshold : float, default 0.25
        Absolute normalized difference above which a covariate is
        flagged (Imbens and Rubin 2015, p. 277).

    Returns
    -------
    DiDBalanceResult

    Examples
    --------
    >>> import statspai as sp
    >>> mpdta = sp.datasets.mpdta()                    # doctest: +SKIP
    >>> bal = sp.did_balance(                          # doctest: +SKIP
    ...     mpdta, ["lpop"], g="first_treat", t="year", i="countyreal"
    ... )
    >>> print(bal.summary())                           # doctest: +SKIP

    References
    ----------
    baker2026difference, imbens2015causal
    """
    covariates = list(covariates)
    if not covariates:
        raise MethodIncompatibility(
            "did_balance requires at least one covariate.",
            recovery_hint="Pass covariates=['x1', 'x2', ...].",
            diagnostics={"covariates": covariates},
        )
    missing = [c for c in [g, t, i, *covariates] if c not in data.columns]
    if weights is not None and weights not in data.columns:
        missing.append(weights)
    if missing:
        raise MethodIncompatibility(
            f"Column(s) not found in data: {missing}",
            recovery_hint="Check column names before calling `did_balance`.",
            diagnostics={"missing": missing},
        )
    if control_group not in ("nevertreated", "notyettreated"):
        raise MethodIncompatibility(
            f"control_group must be 'nevertreated' or 'notyettreated', "
            f"got {control_group!r}.",
            recovery_hint="Use control_group='nevertreated'.",
            diagnostics={"control_group": control_group},
        )

    df = data.copy()
    # Normalise the never-treated marker to 0 the way callaway_santanna does.
    df[g] = (
        pd.to_numeric(df[g], errors="coerce")
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .astype(int)
    )

    unit_g = df.groupby(i)[g].first()
    treated_cohorts = sorted(v for v in unit_g.unique() if v > 0)
    if not treated_cohorts:
        raise DataInsufficient(
            "No treated cohorts found; balance needs a treated group.",
            recovery_hint="Encode first-treatment periods in `g` (0 = never).",
            diagnostics={"function": "did_balance"},
        )

    if cohort is None:
        counts = unit_g[unit_g > 0].value_counts()
        cohort = int(counts.idxmax())
    if isinstance(cohort, (list, tuple, set, np.ndarray, pd.Series)):
        # Pooling cohorts would build a treated arm that mixes different
        # treatment dates, so the "changes" panel would span different
        # windows per unit and the normalized difference would not
        # describe any single 2x2 comparison. Refuse rather than pool.
        raise MethodIncompatibility(
            f"cohort must be a single treated cohort, got {cohort!r}. The "
            "normalized difference is a two-group statistic, so balance is "
            "audited one cohort at a time.",
            recovery_hint=(
                "Call did_balance once per cohort and read the tables side " "by side."
            ),
            diagnostics={"cohort": repr(cohort)},
        )
    cohort = int(cohort)
    if cohort not in treated_cohorts:
        raise MethodIncompatibility(
            f"cohort={cohort} is not a treated cohort; available: "
            f"{treated_cohorts}.",
            recovery_hint="Pass one of the listed cohorts.",
            diagnostics={"cohort": cohort, "available": treated_cohorts},
        )

    periods = sorted(df[t].unique())
    if base_period is None:
        earlier = [p for p in periods if p < cohort]
        if not earlier:
            raise DataInsufficient(
                f"Cohort {cohort} has no pre-treatment period, so there is "
                "no baseline to balance on.",
                recovery_hint="Audit a later-treated cohort.",
                diagnostics={"cohort": cohort, "periods": periods},
            )
        base_period = earlier[-1]
    if comparison_period is None:
        later = [p for p in periods if p > base_period]
        if not later:
            raise DataInsufficient(
                "No period after the base period, so covariate changes "
                "cannot be formed.",
                recovery_hint="Provide comparison_period explicitly.",
                diagnostics={"base_period": base_period, "periods": periods},
            )
        comparison_period = later[0]

    # Group membership.
    is_treated = unit_g == cohort
    if control_group == "nevertreated":
        is_comparison = unit_g == 0
    else:
        is_comparison = (unit_g == 0) | (unit_g > comparison_period)
    is_comparison = is_comparison & ~is_treated

    if not is_treated.any() or not is_comparison.any():
        raise DataInsufficient(
            f"Empty treated ({int(is_treated.sum())}) or comparison "
            f"({int(is_comparison.sum())}) group for cohort {cohort}.",
            recovery_hint="Check the cohort encoding and control_group.",
            diagnostics={"cohort": cohort, "control_group": control_group},
        )

    # Wide covariate slices at the two periods.
    base_slice = df[df[t] == base_period].set_index(i)
    comp_slice = df[df[t] == comparison_period].set_index(i)
    units = unit_g.index

    w_units = np.ones(len(units), dtype=float)
    if weights is not None:
        wser = df.groupby(i)[weights].first().reindex(units)
        if wser.isna().any() or not np.isfinite(wser.to_numpy(dtype=float)).all():
            raise MethodIncompatibility(
                f"weights column {weights!r} has missing/non-finite values.",
                recovery_hint="Drop or impute those units first.",
                diagnostics={"weights": weights},
            )
        w_units = wser.to_numpy(dtype=float)
        if np.any(w_units < 0):
            raise MethodIncompatibility(
                f"weights column {weights!r} contains negative values.",
                recovery_hint="Sampling/population weights must be >= 0.",
                diagnostics={"weights": weights},
            )

    t_mask = is_treated.reindex(units).to_numpy()
    c_mask = is_comparison.reindex(units).to_numpy()

    rows: List[Dict[str, Any]] = []
    constant_in_changes: List[str] = []

    for panel in ("levels", "changes"):
        for cov in covariates:
            base_v = base_slice[cov].reindex(units).to_numpy(dtype=float)
            if panel == "levels":
                v = base_v
            else:
                comp_v = comp_slice[cov].reindex(units).to_numpy(dtype=float)
                v = comp_v - base_v

            ok = np.isfinite(v)
            tm, cm = t_mask & ok, c_mask & ok
            if tm.sum() < 2 or cm.sum() < 2:
                continue
            if panel == "changes" and np.allclose(v[ok], 0.0, atol=1e-12):
                constant_in_changes.append(cov)

            unw = _norm_diff(
                v[tm], v[cm], np.ones(int(tm.sum())), np.ones(int(cm.sum()))
            )
            row: Dict[str, Any] = {
                "panel": panel,
                "covariate": cov,
                **unw,
                "abs_norm_diff": abs(unw["norm_diff"]),
            }
            if weights is not None:
                wtd = _norm_diff(v[tm], v[cm], w_units[tm], w_units[cm])
                row.update(
                    {
                        "w_mean_treated": wtd["mean_treated"],
                        "w_mean_comparison": wtd["mean_comparison"],
                        "w_norm_diff": wtd["norm_diff"],
                        "w_abs_norm_diff": abs(wtd["norm_diff"]),
                    }
                )
            flag_col = "w_abs_norm_diff" if weights is not None else "abs_norm_diff"
            row["flagged"] = bool(row[flag_col] > threshold)
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        raise DataInsufficient(
            "No covariate had enough non-missing observations in both "
            "groups to compute balance.",
            recovery_hint="Check for missing covariate values at the base "
            "and comparison periods.",
            diagnostics={"cohort": cohort, "base_period": base_period},
        )

    diagnostics: Dict[str, Any] = {
        "cohort": cohort,
        "control_group": control_group,
        "constant_in_changes": constant_in_changes,
        "threshold_source": "Imbens and Rubin (2015, p. 277)",
    }

    return DiDBalanceResult(
        table=table,
        levels=table[table["panel"] == "levels"].reset_index(drop=True),
        changes=table[table["panel"] == "changes"].reset_index(drop=True),
        threshold=float(threshold),
        n_treated=int(t_mask.sum()),
        n_comparison=int(c_mask.sum()),
        base_period=base_period,
        comparison_period=comparison_period,
        weighted=weights is not None,
        diagnostics=diagnostics,
    )
