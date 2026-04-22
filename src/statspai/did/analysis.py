"""
DID Analysis Workflow — one-call comprehensive DID analysis.

Automates the standard DID workflow from Cunningham (2021, *Causal
Inference: The Mixtape*):

1. Design detection — 2×2, staggered, or panel
2. Bacon decomposition — diagnose TWFE issues (staggered only)
3. Method selection — auto-pick best estimator or use user's choice
4. Main estimation — ATT with chosen estimator
5. Event study — dynamic treatment effects + pre-trend test
6. Sensitivity — honest_did parallel trends sensitivity (optional)

Returns a ``DIDAnalysis`` result object that bundles everything.

References
----------
Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University
Press. Ch. 9: Difference-in-Differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.results import CausalResult


@dataclass
class DIDAnalysis:
    """Bundled results from a full DID analysis workflow."""

    design: str
    method_used: str
    main_result: CausalResult
    event_study_result: Optional[CausalResult] = None
    bacon: Optional[Dict[str, Any]] = None
    sensitivity: Optional[pd.DataFrame] = None
    diagnostics: Optional[Dict[str, Any]] = None
    steps_log: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Print comprehensive analysis summary."""
        lines = []
        lines.append("=" * 65)
        lines.append("DID Analysis Report")
        lines.append("=" * 65)
        lines.append(f"Design detected : {self.design}")
        lines.append(f"Method used     : {self.method_used}")
        lines.append("")

        # Main result
        r = self.main_result
        lines.append("--- Main Estimate ---")
        lines.append(f"  ATT      = {r.estimate:.6f}")
        lines.append(f"  SE       = {r.se:.6f}")
        lines.append(f"  p-value  = {r.pvalue:.4f}")
        lines.append(f"  95% CI   = [{r.ci[0]:.6f}, {r.ci[1]:.6f}]")
        lines.append(f"  N        = {r.n_obs}")
        lines.append("")

        # Bacon decomposition
        if self.bacon is not None:
            lines.append("--- Bacon Decomposition ---")
            neg_wt = self.bacon.get('negative_weight_share', 0)
            lines.append(f"  TWFE estimate          = {self.bacon.get('beta_twfe', 'N/A'):.6f}")
            lines.append(f"  Negative weight share  = {neg_wt:.1%}")
            if neg_wt > 0.1:
                lines.append("  ⚠ Substantial negative weights — TWFE may be biased.")
                lines.append("    Recommended: Callaway-Sant'Anna or Sun-Abraham.")
            else:
                lines.append("  ✓ Negative weight share is small — TWFE likely reliable.")
            lines.append("")

        # Event study
        if self.event_study_result is not None:
            mi = self.event_study_result.model_info or {}
            lines.append("--- Event Study ---")
            pretrend_p = mi.get('pretrend_pvalue')
            if pretrend_p is not None:
                lines.append(f"  Pre-trend test p-value = {pretrend_p:.4f}")
                if pretrend_p < 0.05:
                    lines.append("  ⚠ Pre-trend test rejects at 5% — parallel trends concern.")
                else:
                    lines.append("  ✓ No evidence of pre-trend violation.")
            lines.append("")

        # Sensitivity
        if self.sensitivity is not None and len(self.sensitivity) > 0:
            lines.append("--- Sensitivity (Honest DID) ---")
            # Find breakdown M
            breakdown = self.sensitivity[self.sensitivity['rejects_zero']].tail(1)
            if len(breakdown) > 0:
                m_star = breakdown.iloc[0]['M']
                lines.append(f"  Breakdown M* = {m_star:.4f}")
                lines.append("  (Largest violation magnitude where effect remains significant)")
            else:
                lines.append("  Effect not significant even at M=0.")
            lines.append("")

        # Steps log
        if self.steps_log:
            lines.append("--- Analysis Steps ---")
            for i, step in enumerate(self.steps_log, 1):
                lines.append(f"  {i}. {step}")

        lines.append("=" * 65)
        return "\n".join(lines)

    def plot(self, **kwargs):
        """Plot event study if available, else main result."""
        if self.event_study_result is not None:
            return self.event_study_result.plot(**kwargs)
        return self.main_result.plot(**kwargs)


def did_analysis(
    data: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    method: str = 'auto',
    estimator: str = 'dr',
    control_group: str = 'nevertreated',
    run_bacon: bool = True,
    run_event_study: bool = True,
    run_sensitivity: bool = True,
    event_window: Optional[tuple] = None,
    cluster: Optional[str] = None,
    robust: bool = True,
    alpha: float = 0.05,
    **kwargs,
) -> DIDAnalysis:
    """
    Comprehensive DID analysis workflow.

    Runs the full DID analysis pipeline in one call: design detection,
    Bacon decomposition (staggered), estimation, event study, and
    honest_did sensitivity analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    treat : str
        Treatment variable. For 2×2: binary (0/1).
        For staggered: first treatment period (0 = never treated).
    time : str
        Time period variable.
    id : str, optional
        Unit identifier. Required for staggered designs.
    covariates : list of str, optional
        Control variables.
    method : str, default 'auto'
        Estimation method: 'auto', '2x2', 'cs', 'sa', 'sdid'.
    estimator : str, default 'dr'
        For CS: 'dr', 'ipw', or 'reg'.
    control_group : str, default 'nevertreated'
        For CS/SA: 'nevertreated' or 'notyettreated'.
    run_bacon : bool, default True
        Run Bacon decomposition for staggered designs.
    run_event_study : bool, default True
        Run event study for dynamic effects + pre-trend test.
    run_sensitivity : bool, default True
        Run honest_did sensitivity analysis.
    event_window : tuple of (int, int), optional
        Event study window, e.g. (-5, 5). Auto-detected if None.
    cluster : str, optional
        Cluster variable for standard errors.
    robust : bool, default True
        HC1 robust standard errors.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    DIDAnalysis
        Bundled results with ``.summary()``, ``.plot()`` methods.

    Examples
    --------
    Classic 2×2:

    >>> report = did_analysis(df, y='wage', treat='policy', time='post')
    >>> print(report.summary())

    Staggered — full pipeline:

    >>> report = did_analysis(df, y='earnings', treat='first_treat',
    ...                       time='year', id='worker')
    >>> print(report.summary())
    >>> report.plot()

    Quick estimate only (skip diagnostics):

    >>> report = did_analysis(df, y='y', treat='g', time='t', id='i',
    ...                       run_bacon=False, run_sensitivity=False)
    """
    from .did_2x2 import did_2x2
    from .callaway_santanna import callaway_santanna
    from .sun_abraham import sun_abraham
    from .bacon import bacon_decomposition
    from .honest_did import honest_did
    from .event_study import event_study

    steps = []
    bacon_result = None
    es_result = None
    sens_result = None
    diag = {}

    # ── Step 1: Detect design ──────────────────────────────────────── #
    if method == 'auto':
        if id is None:
            treat_vals = set(data[treat].dropna().unique())
            if treat_vals <= {0, 1, True, False}:
                design = '2x2'
                method = '2x2'
            else:
                raise ValueError(
                    f"Cannot auto-detect design. Treatment '{treat}' has "
                    f"values {sorted(treat_vals)}. Provide 'id' for staggered."
                )
        else:
            design = 'staggered'
            method = 'cs'
        steps.append(f"Design detected: {design}")
    else:
        design = '2x2' if method == '2x2' else 'staggered'
        steps.append(f"Design set by user: {design} (method={method})")

    # ── Step 2: Bacon decomposition (staggered only) ───────────────── #
    if run_bacon and design == 'staggered' and id is not None:
        try:
            bacon_result = bacon_decomposition(
                data, y=y, treat=treat, time=time, id=id, alpha=alpha,
            )
            neg_wt = bacon_result.get('negative_weight_share', 0)
            steps.append(
                f"Bacon decomposition: {bacon_result.get('n_comparisons', '?')} "
                f"sub-comparisons, negative weight = {neg_wt:.1%}"
            )

            # Auto-recommend method based on Bacon
            if method == 'cs' and neg_wt > 0.1:
                steps.append(
                    "⚠ High negative weights — Callaway-Sant'Anna recommended "
                    "(already selected)."
                )
            diag['bacon'] = bacon_result
        except Exception as e:
            steps.append(f"Bacon decomposition skipped: {e}")

    # ── Step 3: Main estimation ────────────────────────────────────── #
    if method == '2x2':
        main_result = did_2x2(
            data, y=y, treat=treat, time=time,
            covariates=covariates, cluster=cluster,
            robust=robust, alpha=alpha,
        )
        method_label = 'DID 2×2 (OLS)'
    elif method in ('cs', 'callaway_santanna'):
        if id is None:
            from statspai.exceptions import MethodIncompatibility
            raise MethodIncompatibility(
                "'id' is required for Callaway-Sant'Anna.",
                recovery_hint=(
                    "Pass the unit identifier column via id='...', or use "
                    "method='2x2' if you only have two periods."
                ),
                diagnostics={"method": "callaway_santanna", "missing": "id"},
                alternative_functions=["sp.did"],
            )
        main_result = callaway_santanna(
            data, y=y, g=treat, t=time, i=id,
            x=covariates, estimator=estimator,
            control_group=control_group, alpha=alpha,
        )
        method_label = f'Callaway-Sant\'Anna ({estimator.upper()})'
    elif method in ('sa', 'sun_abraham', 'sunab'):
        if id is None:
            from statspai.exceptions import MethodIncompatibility
            raise MethodIncompatibility(
                "'id' is required for Sun-Abraham.",
                recovery_hint=(
                    "Pass the unit identifier column via id='...', or use "
                    "method='2x2' if you only have two periods."
                ),
                diagnostics={"method": "sun_abraham", "missing": "id"},
                alternative_functions=["sp.did"],
            )
        main_result = sun_abraham(
            data, y=y, g=treat, t=time, i=id,
            covariates=covariates, cluster=cluster, alpha=alpha,
        )
        method_label = 'Sun-Abraham (IW)'
    elif method == 'sdid':
        from ..synth.sdid import sdid as _sdid
        # SDID requires different parameter mapping
        treat_time_vals = sorted(data.loc[data[treat] > 0, treat].unique())
        treat_time_val = treat_time_vals[0] if treat_time_vals else None
        treat_units = data.loc[data[treat] > 0, id].unique().tolist() if id else None
        main_result = _sdid(
            data, y=y, unit=id, time=time,
            treat_unit=treat_units, treat_time=treat_time_val,
            method='sdid', alpha=alpha, **kwargs,
        )
        method_label = 'Synthetic DID (Arkhangelsky et al.)'
    else:
        raise ValueError(f"Unknown method: '{method}'")

    steps.append(f"Main estimation: {method_label}")
    steps.append(
        f"ATT = {main_result.estimate:.6f} "
        f"(SE = {main_result.se:.6f}, p = {main_result.pvalue:.4f})"
    )

    # ── Step 4: Event study ────────────────────────────────────────── #
    if run_event_study and design == 'staggered' and id is not None:
        try:
            window = event_window or (-4, 4)
            es_result = event_study(
                data, y=y, treat_time=treat, time=time, unit=id,
                window=window, cluster=cluster, alpha=alpha,
            )
            mi = es_result.model_info or {}
            pretrend_p = mi.get('pretrend_pvalue')
            if pretrend_p is not None:
                steps.append(
                    f"Event study: pre-trend test p = {pretrend_p:.4f}"
                    + (" ✓" if pretrend_p >= 0.05 else " ⚠ VIOLATION")
                )
            else:
                steps.append("Event study: computed (no pre-trend p-value)")
            diag['pretrend_pvalue'] = pretrend_p
        except Exception as e:
            steps.append(f"Event study skipped: {e}")

    # ── Step 5: Honest DID sensitivity ─────────────────────────────── #
    if run_sensitivity and es_result is not None:
        try:
            sens_result = honest_did(es_result, e=0, method='smoothness', alpha=alpha)
            steps.append("Honest DID sensitivity analysis: computed")
        except Exception as e:
            steps.append(f"Sensitivity analysis skipped: {e}")

    return DIDAnalysis(
        design=design,
        method_used=method_label,
        main_result=main_result,
        event_study_result=es_result,
        bacon=bacon_result,
        sensitivity=sens_result,
        diagnostics=diag,
        steps_log=steps,
    )
