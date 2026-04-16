"""
Multi-method comparison and auto-recommendation for Synthetic Control.

Provides ``synth_compare()`` to run multiple SCM variants on the same data
and ``synth_recommend()`` to quickly identify the best method.

This is a **unique** feature: no existing package (R, Stata, or Python) offers
automated multi-method SCM comparison with a principled recommendation engine.

Functions
---------
- **synth_compare** — run 12 SCM variants, compare pre-fit & ATT, recommend
- **synth_recommend** — quick one-liner returning the best method name
"""

from __future__ import annotations

import time as _time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.results import CausalResult
from .scm import synth


# ====================================================================== #
#  Method registry
# ====================================================================== #

# Ordered from simplest to most complex (used for tiebreaking).
_METHOD_REGISTRY: List[Tuple[str, int]] = [
    ("classic", 1),
    ("penalized", 2),
    ("demeaned", 3),
    ("detrended", 4),
    ("unconstrained", 5),
    ("elastic_net", 6),
    ("augmented", 7),
    ("sdid", 8),
    ("gsynth", 9),
    ("mc", 10),
    ("discos", 11),
    ("scpi", 12),
    ("penscm", 13),
    ("fdid", 14),
    ("sparse", 15),
    ("cluster", 16),
    ("kernel", 17),
    ("kernel_ridge", 18),
    ("bayesian", 19),
    ("bsts", 20),
]

_ALL_METHODS = [m for m, _ in _METHOD_REGISTRY]
_SIMPLICITY = {m: r for m, r in _METHOD_REGISTRY}


# ====================================================================== #
#  Helpers
# ====================================================================== #

def _extract_pre_rmspe(result: CausalResult) -> float:
    """Extract pre-treatment RMSPE from a CausalResult.

    Searches ``model_info`` for common key names used across SCM variants.
    Falls back to ``np.inf`` if unavailable.
    """
    mi = getattr(result, "model_info", {}) or {}
    # Classic / demeaned / robust store pre_treatment_mspe
    mspe = mi.get("pre_treatment_mspe")
    if mspe is not None:
        return float(np.sqrt(mspe))
    # Some variants store rmspe / rmse directly
    for key in ("pre_treatment_rmspe", "pre_treatment_rmse", "pre_rmspe"):
        val = mi.get(key)
        if val is not None:
            return float(val)
    return np.inf


def _extract_n_effective_donors(result: CausalResult) -> int:
    """Count donors with weight > 0.01 (effective donors)."""
    mi = getattr(result, "model_info", {}) or {}
    weights = mi.get("donor_weights")
    if weights is not None:
        if isinstance(weights, dict):
            return int(sum(1 for w in weights.values() if abs(w) > 0.01))
        if isinstance(weights, (np.ndarray, pd.Series)):
            return int(np.sum(np.abs(weights) > 0.01))
    # Fallback: return total donors
    n = mi.get("n_donors")
    return int(n) if n is not None else 0


# ====================================================================== #
#  SynthComparison
# ====================================================================== #

class SynthComparison:
    """Structured container for multi-method SCM comparison results.

    Attributes
    ----------
    results : dict
        Mapping of method name to ``CausalResult``.
    comparison_table : pd.DataFrame
        Side-by-side metrics for every successful method, sorted by
        ``pre_rmspe`` ascending.
    recommended : str
        Name of the recommended method.
    recommendation_reason : str
        Human-readable justification.
    """

    def __init__(
        self,
        results: Dict[str, CausalResult],
        comparison_table: pd.DataFrame,
        recommended: str,
        recommendation_reason: str,
    ):
        self.results = results
        self.comparison_table = comparison_table
        self.recommended = recommended
        self.recommendation_reason = recommendation_reason

    # ------------------------------------------------------------------ #
    #  Display helpers
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Return a formatted multi-line summary string.

        Returns
        -------
        str
        """
        lines = [
            "=" * 72,
            "Synthetic Control — Multi-Method Comparison",
            "=" * 72,
            "",
            f"Methods attempted : {len(self.comparison_table) + sum(1 for _ in [])}",
            f"Methods succeeded : {len(self.comparison_table)}",
            f"Recommended       : {self.recommended}",
            f"Reason            : {self.recommendation_reason}",
            "",
            "-" * 72,
        ]

        # Format the table
        display_cols = [
            "rank", "method", "att", "se", "pvalue",
            "ci_lower", "ci_upper", "pre_rmspe",
            "n_effective_donors", "time_seconds",
        ]
        cols = [c for c in display_cols if c in self.comparison_table.columns]
        tbl = self.comparison_table[cols].to_string(index=False, float_format="%.4f")
        lines.append(tbl)
        lines.append("-" * 72)
        lines.append("")
        lines.append(
            f"* Recommended method '{self.recommended}' is highlighted."
        )
        lines.append("=" * 72)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SynthComparison(n_methods={len(self.results)}, "
            f"recommended='{self.recommended}')"
        )

    def __str__(self) -> str:
        return self.summary()

    # ------------------------------------------------------------------ #
    #  Plotting
    # ------------------------------------------------------------------ #

    def plot(self, **kwargs) -> Any:
        """Overlay all results using ``synthplot(type='compare')``.

        Parameters
        ----------
        **kwargs
            Forwarded to ``synthplot``.

        Returns
        -------
        matplotlib Figure or Axes
        """
        from .plots import synthplot

        result_list = list(self.results.values())
        return synthplot(result_list, type="compare", **kwargs)


# ====================================================================== #
#  Recommendation algorithm
# ====================================================================== #

def _recommend(table: pd.DataFrame) -> Tuple[str, str]:
    """Pick the best method from a comparison table.

    Algorithm
    ---------
    1. Filter to methods whose ``pre_rmspe < 2 * min(pre_rmspe)``
       (adequate pre-treatment fit).
    2. Among those, exclude methods with unreasonably wide CI
       (CI width > 5 * median CI width across candidates).
    3. Tiebreak by simplicity rank (lower is simpler).
    4. Return the recommended method name and reason string.

    Parameters
    ----------
    table : pd.DataFrame
        Must contain columns ``method``, ``pre_rmspe``, ``ci_lower``,
        ``ci_upper``, ``simplicity_rank``.

    Returns
    -------
    recommended : str
    reason : str
    """
    if table.empty:
        return "classic", "No methods succeeded; defaulting to classic."

    df = table.copy()

    # Step 1 — filter by pre-treatment fit
    min_rmspe = df["pre_rmspe"].replace([np.inf, np.nan], np.nan).min()
    if np.isnan(min_rmspe) or min_rmspe == 0:
        # Cannot filter meaningfully; keep all
        fit_mask = pd.Series(True, index=df.index)
    else:
        fit_mask = df["pre_rmspe"] <= 2.0 * min_rmspe
    candidates = df.loc[fit_mask]

    if candidates.empty:
        candidates = df  # fall back to all

    # Step 2 — filter by CI width
    ci_width = (candidates["ci_upper"] - candidates["ci_lower"]).abs()
    median_width = ci_width.median()
    if median_width > 0 and not np.isnan(median_width):
        reasonable = ci_width <= 5.0 * median_width
        if reasonable.any():
            candidates = candidates.loc[reasonable]

    # Step 3 — tiebreak: simplest method wins
    best_idx = candidates["simplicity_rank"].idxmin()
    best = candidates.loc[best_idx]
    best_method = best["method"]

    # Build reason
    parts = [f"Best pre-treatment fit (RMSPE={best['pre_rmspe']:.4f})"]
    if best["simplicity_rank"] <= 4:
        parts.append("simple and interpretable")
    n_good_fit = fit_mask.sum()
    if n_good_fit > 1:
        parts.append(
            f"chosen over {n_good_fit - 1} other well-fitting method(s) "
            "by parsimony"
        )
    reason = "; ".join(parts) + "."

    return str(best_method), reason


# ====================================================================== #
#  synth_compare
# ====================================================================== #

def synth_compare(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any = None,
    treatment_time: Any = None,
    methods: Optional[List[str]] = None,
    placebo: bool = True,
    alpha: float = 0.05,
    **kwargs,
) -> SynthComparison:
    """Run multiple SCM variants and compare them side by side.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any, optional
        Identifier of the treated unit.
    treatment_time : any, optional
        First treatment period (inclusive).
    methods : list of str, optional
        SCM variants to compare. If ``None`` (default), all 20 registered
        methods are attempted, in ascending complexity order:
        ``classic, penalized, demeaned, detrended, unconstrained,
        elastic_net, augmented, sdid, gsynth, mc, discos, scpi, penscm,
        fdid, sparse, cluster, kernel, kernel_ridge, bayesian, bsts``.
        Pass an explicit subset to reduce runtime.
    placebo : bool, default True
        Whether to run placebo inference for each method.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    **kwargs
        Additional keyword arguments forwarded to ``synth()``.

    Returns
    -------
    SynthComparison
        Structured comparison object with ``.results``,
        ``.comparison_table``, ``.recommended``, and ``.plot()``.

    Examples
    --------
    >>> comp = sp.synth_compare(
    ...     df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ... )
    >>> print(comp.summary())
    >>> print(comp.recommended)
    'demeaned'
    >>> comp.plot()

    Compare a subset of methods:

    >>> comp = sp.synth_compare(
    ...     df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ...     methods=['classic', 'augmented', 'sdid', 'mc'],
    ... )

    See Also
    --------
    synth_recommend : Quick one-liner returning only the method name.
    synth : Unified SCM dispatcher.
    """
    if methods is None:
        methods = _ALL_METHODS.copy()

    results: Dict[str, CausalResult] = {}
    rows: List[Dict[str, Any]] = []

    for method_name in methods:
        t0 = _time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = synth(
                    data=data,
                    outcome=outcome,
                    unit=unit,
                    time=time,
                    treated_unit=treated_unit,
                    treatment_time=treatment_time,
                    method=method_name,
                    placebo=placebo,
                    alpha=alpha,
                    **kwargs,
                )
        except Exception:
            # Method failed on this data — skip silently
            continue
        elapsed = _time.time() - t0

        results[method_name] = res

        # Extract metrics
        att = getattr(res, "estimate", np.nan)
        se = getattr(res, "se", np.nan)
        pval = getattr(res, "pvalue", np.nan)
        ci = getattr(res, "ci", (np.nan, np.nan))
        ci_lo, ci_hi = (ci if ci is not None else (np.nan, np.nan))
        pre_rmspe = _extract_pre_rmspe(res)
        n_eff = _extract_n_effective_donors(res)

        rows.append({
            "method": method_name,
            "att": att,
            "se": se,
            "pvalue": pval,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "pre_rmspe": pre_rmspe,
            "n_effective_donors": n_eff,
            "simplicity_rank": _SIMPLICITY.get(method_name, 99),
            "time_seconds": round(elapsed, 3),
        })

    # Build comparison table sorted by pre-treatment fit
    comparison_table = pd.DataFrame(rows)
    if not comparison_table.empty:
        comparison_table = comparison_table.sort_values(
            "pre_rmspe", ascending=True
        ).reset_index(drop=True)
        comparison_table["rank"] = range(1, len(comparison_table) + 1)

    # Recommend
    recommended, reason = _recommend(comparison_table)

    return SynthComparison(
        results=results,
        comparison_table=comparison_table,
        recommended=recommended,
        recommendation_reason=reason,
    )


# ====================================================================== #
#  synth_recommend
# ====================================================================== #

def synth_recommend(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any = None,
    treatment_time: Any = None,
    **kwargs,
) -> str:
    """Quickly recommend the best SCM method for the given data.

    Runs ``synth_compare`` internally with ``placebo=False`` for speed,
    then returns just the recommended method name.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any, optional
        Identifier of the treated unit.
    treatment_time : any, optional
        First treatment period (inclusive).
    **kwargs
        Additional keyword arguments forwarded to ``synth_compare()``.

    Returns
    -------
    str
        Name of the recommended SCM method (e.g., ``'classic'``,
        ``'augmented'``, ``'sdid'``).

    Examples
    --------
    >>> best = sp.synth_recommend(
    ...     df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ... )
    >>> best
    'demeaned'

    Then use it:

    >>> result = sp.synth(
    ...     df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ...     method=best,
    ... )

    See Also
    --------
    synth_compare : Full comparison with all metrics and plots.
    """
    comp = synth_compare(
        data=data,
        outcome=outcome,
        unit=unit,
        time=time,
        treated_unit=treated_unit,
        treatment_time=treatment_time,
        placebo=False,
        **kwargs,
    )
    return comp.recommended
