"""
Estimator Recommendation Engine.

Given a research question, data structure, and optionally a causal DAG,
this registered workflow helper recommends estimator candidates with
reasoning. The recommendation is planning support; users remain
responsible for the identification argument and validation status.

Usage
-----
>>> import statspai as sp
>>> rec = sp.recommend(
...     data=df, y='wage', treatment='training',
...     design='observational',  # or 'rct', 'panel', 'iv', 'rd', 'did'
...     dag=my_dag,  # optional
... )
>>> print(rec.summary())
>>> result = rec.run()  # execute the recommended estimator
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class RecommendationResult:
    """Result from the estimator recommendation engine.

    Returned by :func:`recommend`. Holds a ranked list of estimator
    ``recommendations`` (each a dict with ``method`` / ``function`` /
    ``reason`` / ``assumptions``), the detected ``design``, the ``data_profile``
    (outcome / treatment types, panel shape, missingness), and any ``warnings``.
    Call :meth:`summary` for a ranked, human-readable report.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> rec = sp.recommend(df, y="log_wage", treatment="union",
    ...                    covariates=["education", "experience"])
    >>> type(rec).__name__
    'RecommendationResult'
    >>> bool(len(rec.recommendations) > 0)
    True
    >>> bool("method" in rec.recommendations[0])
    True
    """

    def __init__(
        self,
        recommendations: List[Dict[str, Any]],
        data_profile: Dict[str, Any],
        design: str,
        warnings: List[str],
        data: pd.DataFrame,
        y: str,
        treatment: Optional[str],
    ) -> None:
        self.recommendations = recommendations  # list of dicts
        self.data_profile = data_profile
        self.design = design
        self.warnings = warnings
        self._data = data
        self._y = y
        self._treatment = treatment

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "StatsPAI Estimator Recommendation",
            "=" * 70,
            "",
            "DATA PROFILE",
            f"  N obs:       {self.data_profile['n_obs']:,}",
            f"  Variables:   {self.data_profile['n_vars']}",
            f"  Outcome:     {self._y} ({self.data_profile['y_type']})",
            f"  Treatment:   {self._treatment} ({self.data_profile['treat_type']})",
        ]
        if self.data_profile.get("panel"):
            lines.append(
                f"  Panel:       {self.data_profile['n_units']} units × "
                f"{self.data_profile['n_periods']} periods"
            )
        if self.data_profile.get("missing_pct", 0) > 0:
            lines.append(f"  Missing:     {self.data_profile['missing_pct']:.1%}")

        lines.append(f"\n  Design:      {self.design.upper()}")

        if self.warnings:
            lines.append("\n⚠ WARNINGS")
            for w in self.warnings:
                lines.append(f"  • {w}")

        lines.append(f"\n{'─' * 70}")
        lines.append("RECOMMENDED ESTIMATORS (ranked by appropriateness)")
        lines.append(f"{'─' * 70}")

        for i, rec in enumerate(self.recommendations):
            star = "★" if i == 0 else "○"
            lines.append(f"\n  {star} #{i+1}: {rec['method']}")
            lines.append(f"    Function: sp.{rec['function']}()")
            lines.append(f"    Why: {rec['reason']}")
            if rec.get("assumptions"):
                lines.append(f"    Assumptions: {', '.join(rec['assumptions'])}")
            if rec.get("robustness"):
                lines.append(f"    Robustness: {rec['robustness']}")
            if rec.get("code"):
                lines.append(f"    Code: {rec['code']}")
            v = rec.get("verify")
            if v:
                if v.get("error"):
                    lines.append(f"    Stability: skipped ({v['error']})")
                elif np.isfinite(v.get("score", np.nan)):
                    stab = v.get("stability", {}).get("score", np.nan)
                    plac = v.get("placebo", {}).get("score", np.nan)
                    subs = v.get("subsample", {}).get("score", np.nan)
                    lines.append(
                        f"    Stability: score={v['score']:.0f}/100  "
                        f"(resample={stab:.0f}, placebo={plac:.0f}, "
                        f"subsample={subs:.0f}, B={v.get('B_used','?')}, "
                        f"{v.get('elapsed_s',0):.1f}s)  "
                        f"[measures resampling stability, NOT identification validity]"
                    )

        lines.append(f"\n{'─' * 70}")
        lines.append("SUGGESTED WORKFLOW")
        lines.append(f"{'─' * 70}")
        for i, step in enumerate(self._workflow_steps()):
            lines.append(f"  {i+1}. {step}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def to_latex(
        self, caption: Optional[str] = None, label: str = "tab:recommendation"
    ) -> str:
        r"""Export recommendations as a booktabs LaTeX table.

        If ``verify=True`` was used when calling ``recommend()``, the
        table includes the stability-check columns (composite score,
        bootstrap stability, placebo pass-rate, subsample agreement).

        IMPORTANT CAVEAT FOR AUTHORS:
        The stability score measures whether a method gives **consistent**
        estimates under resampling on the observed data — it does NOT
        establish identification validity or protect against unobserved
        confounding. A biased OLS on observational data will typically
        score high because the bias is stable across resamples.  Do not
        cite this score as evidence that a method is "correct" for a
        given design; use it only to compare the stability of methods
        that already satisfy the design's identification assumptions.

        Parameters
        ----------
        caption : str, optional
            Table caption. Defaults to the detected design.
        label : str
            LaTeX label for cross-referencing.

        Returns
        -------
        str
            LaTeX source (booktabs + threeparttable).
        """
        has_verify = any(
            isinstance(r.get("verify"), dict)
            and np.isfinite(r["verify"].get("score", np.nan))
            for r in self.recommendations
        )
        if caption is None:
            caption = (
                f"StatsPAI recommended estimators for "
                f"{self.design.replace('_', ' ')} design"
                + (" (with empirical verification)" if has_verify else "")
            )

        def _esc(s: str) -> str:
            return (
                str(s)
                .replace("\\", r"\textbackslash{}")
                .replace("_", r"\_")
                .replace("&", r"\&")
                .replace("%", r"\%")
                .replace("#", r"\#")
            )

        if has_verify:
            header = (
                r"Rank & Method & Function & "
                r"Stab.\ Score & Resample & Plac. & Subs. \\"
            )
            col_spec = "rllrrrr"
        else:
            header = r"Rank & Method & Function & Reason \\"
            col_spec = "rllp{6cm}"

        lines = [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\begin{threeparttable}",
            rf"\caption{{{_esc(caption)}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header,
            r"\midrule",
        ]

        for i, rec in enumerate(self.recommendations, 1):
            method = _esc(rec["method"])
            func = rf"\texttt{{sp.{_esc(rec['function'])}()}}"
            if has_verify:
                v = rec.get("verify") or {}
                score = v.get("score", np.nan)
                stab = (v.get("stability") or {}).get("score", np.nan)
                plac = (v.get("placebo") or {}).get("score", np.nan)
                subs = (v.get("subsample") or {}).get("score", np.nan)
                err = v.get("error")

                def _fmt(x: Any) -> str:
                    return (
                        f"{x:.0f}" if isinstance(x, float) and np.isfinite(x) else "--"
                    )

                marker = f" {{\\tiny\\textit{{({_esc(err)})}}}}" if err else ""
                lines.append(
                    f"{i} & {method}{marker} & {func} & "
                    f"{_fmt(score)} & {_fmt(stab)} & {_fmt(plac)} & {_fmt(subs)} \\\\"
                )
            else:
                reason = _esc(rec.get("reason", ""))
                lines.append(f"{i} & {method} & {func} & {reason} \\\\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\begin{tablenotes}",
                r"\footnotesize",
            ]
        )
        if has_verify:
            lines.extend(
                [
                    r"\item \textbf{Stab.\ Score}: weighted composite in [0, 100]"
                    r" measuring estimate \emph{stability} under resampling;"
                    r" \emph{not} a measure of identification validity.",
                    r"\item \textbf{Resample}: bootstrap stability "
                    r"($100 \times (1 - \text{CV})$ of point estimate across $B$ resamples).",
                    r"\item \textbf{Plac.}: permutation placebo pass rate "
                    r"(\% of permuted-treatment runs with $p > 0.10$). "
                    r"Note: unconditional permutation destroys confounder "
                    r"structure and thus has limited power for "
                    r"selection-on-observables designs.",
                    r"\item \textbf{Subs.}: sign agreement across 50\% subsamples.",
                    r"\item \textbf{Caveat}: high stability is necessary but not"
                    r" sufficient; a biased estimator can be perfectly stable.",
                    r"\item Data: "
                    + _esc(f"N={self.data_profile['n_obs']:,}")
                    + (
                        f", {self.data_profile.get('n_units', '?')} units "
                        f"$\\times$ {self.data_profile.get('n_periods', '?')} periods"
                        if self.data_profile.get("panel")
                        else ""
                    )
                    + ".",
                ]
            )
        else:
            lines.append(
                r"\item Rankings are rule-based; call with "
                r"\texttt{verify=True} for empirical scores."
            )
        lines.extend(
            [
                r"\end{tablenotes}",
                r"\end{threeparttable}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines)

    def _workflow_steps(self) -> List[str]:
        """Generate a recommended workflow."""
        steps: List[str] = []
        rec = self.recommendations[0] if self.recommendations else None
        if not rec:
            return ["Insufficient information to recommend"]

        # Pre-estimation
        steps.append("Run sp.sumstats(df) to check data quality")
        if self.data_profile.get("missing_pct", 0) > 5:
            steps.append(
                f"Handle missing data: sp.mice(df, m=5) — {self.data_profile['missing_pct']:.0%} missing"
            )

        if self._treatment:
            steps.append(
                f"Check balance: sp.balance_check(df, treatment='{self._treatment}', "
                f"covariates=[...])"
            )

        # Main estimation
        steps.append(f"Estimate: result = {rec['code']}")

        # Post-estimation
        steps.append("Diagnostics: sp.diagnose_result(result)")

        if rec["function"] in ["regress", "iv", "panel"]:
            steps.append("Sensitivity: sp.sensemakr(result) or sp.oster_bounds(result)")
        if rec["function"] in ["did", "callaway_santanna"]:
            steps.append("Pre-trends: sp.pretrends_test(result)")
            steps.append("Event study: sp.event_study(df, ...)")
        if rec["function"] == "rdrobust":
            steps.append("McCrary test: sp.rddensity(df, x='running_var')")

        steps.append("Robustness: sp.robustness_report(result)")
        steps.append("Export: sp.outreg2(result, filename='results.xlsx')")

        return steps

    def run(self, which: int = 0, **kwargs: Any) -> Any:
        """Execute the recommended estimator.

        Parameters
        ----------
        which : int, default 0
            Which recommendation to run (0 = top recommendation).
        **kwargs
            Override any parameters.
        """
        import statspai as sp

        rec = self.recommendations[which]
        func = getattr(sp, rec["function"])
        params = rec.get("params", {})
        params.update(kwargs)
        return func(**params)

    def run_all(self, **kwargs: Any) -> Dict[str, Any]:
        """Run all recommended estimators and return a comparison."""
        results = {}
        for i, rec in enumerate(self.recommendations):
            try:
                results[rec["method"]] = self.run(which=i, **kwargs)
            except Exception as e:
                results[rec["method"]] = f"Error: {e}"
        return results


def _profile_data(
    data: pd.DataFrame,
    y: str,
    treatment: Optional[str],
    id_col: Optional[str],
    time_col: Optional[str],
) -> Dict[str, Any]:
    """Profile the dataset to understand its structure."""
    profile: Dict[str, Any] = {
        "n_obs": len(data),
        "n_vars": len(data.columns),
    }

    # Outcome type
    y_data = data[y].dropna()
    if y_data.nunique() == 2:
        profile["y_type"] = "binary"
    elif pd.api.types.is_integer_dtype(y_data) and y_data.min() >= 0:
        if y_data.max() <= 50:
            profile["y_type"] = "count"
        else:
            profile["y_type"] = "continuous"
    elif len(y_data) > 0 and y_data.min() >= 0 and y_data.max() <= 1:
        profile["y_type"] = "fractional"
    else:
        profile["y_type"] = "continuous"

    # Treatment type
    if treatment:
        t_data = data[treatment]
        if t_data.nunique() == 2:
            profile["treat_type"] = "binary"
        elif t_data.nunique() <= 10:
            profile["treat_type"] = "categorical"
        else:
            profile["treat_type"] = "continuous"
    else:
        profile["treat_type"] = "none"

    # Panel structure
    if id_col and time_col:
        profile["panel"] = True
        profile["n_units"] = data[id_col].nunique()
        profile["n_periods"] = data[time_col].nunique()
        profile["balanced"] = data.groupby(id_col).size().nunique() == 1
    else:
        profile["panel"] = False

    # Missing data
    profile["missing_pct"] = data.isna().any(axis=1).mean()

    return profile


# Synthetic-control detection thresholds. The SCM signature is a single (or
# very few) ever-treated unit against a substantial donor pool over a
# multi-period panel. The bounds are deliberately tight: with this few treated
# units the staggered-DID estimators (Callaway-Sant'Anna / Sun-Abraham) are
# degenerate, while genuine staggered DID (hundreds of treated units) is never
# misrouted. See benchmarks/recommend_hit_rate F-001.
_SYNTH_MAX_TREATED = 2  # at most this many ever-treated units to call it synth
_SYNTH_MIN_DONORS = 5  # need a donor pool of at least this many controls
_SYNTH_MIN_PERIODS = 4  # need enough periods to fit donor weights pre/post


def _detect_design(
    data: pd.DataFrame,
    y: str,
    treatment: Optional[str],
    id_col: Optional[str],
    time_col: Optional[str],
    running_var: Optional[str],
    instrument: Optional[str],
    profile: Dict[str, Any],
) -> str:
    """Auto-detect the likely research design."""
    if running_var:
        return "rd"
    if instrument:
        return "iv"
    if id_col and time_col and treatment:
        # Check if treatment varies over time → DID / synthetic control
        treat_varies = data.groupby(id_col)[treatment].nunique().max() > 1
        if treat_varies:
            # Synthetic control vs staggered DID: a comparative case study has
            # one (or a handful of) treated unit(s) and a donor pool, where a
            # synthetic counterfactual is the right tool. Staggered DID needs
            # many treated units with variation in adoption timing.
            ever_treated = data.groupby(id_col)[treatment].max()
            n_treated = int((ever_treated > 0).sum())
            n_control = int((ever_treated <= 0).sum())
            n_periods = int(data[time_col].nunique())
            if (
                n_treated <= _SYNTH_MAX_TREATED
                and n_control >= _SYNTH_MIN_DONORS
                and n_periods >= _SYNTH_MIN_PERIODS
            ):
                return "synth"
            return "did"
        else:
            return "panel"
    if id_col and time_col:
        return "panel"
    if treatment:
        return "observational"
    return "cross-section"


def recommend(
    data: pd.DataFrame,
    y: str,
    treatment: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    id: Optional[str] = None,
    time: Optional[str] = None,
    running_var: Optional[str] = None,
    instrument: Optional[str] = None,
    cutoff: Optional[float] = None,
    design: Optional[str] = None,
    dag: Any = None,
    # --- Sprint-B / 0.9.6 causal-method extensions (all opt-in) ---
    mediator: Optional[str] = None,
    tv_confounders: Optional[List[str]] = None,
    proxy_z: Optional[List[str]] = None,
    proxy_w: Optional[List[str]] = None,
    post_treat_strata: Optional[str] = None,
    # --- verification (pre-existing) ---
    verify: bool = False,
    verify_B: int = 50,
    verify_budget_s: float = 30.0,
    verify_top_k: int = 3,
    # --- v1.13 stability gating (agent-safe by default) ---
    allow_experimental: bool = False,
) -> RecommendationResult:
    """
    Recommend the appropriate estimator(s) for your research question.

    Given your data, outcome, treatment, and research design, this function:
    1. Profiles your data (type, structure, missing patterns)
    2. Detects your research design (RCT, DID, RD, IV, observational)
    3. Recommends ranked estimators with reasoning
    4. Generates a complete workflow (pre-estimation → estimation → robustness)
    5. Provides executable code via `.run()`

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treatment : str, optional
        Treatment/exposure variable.
    covariates : list of str, optional
        Control variables.
    id : str, optional
        Unit identifier (for panel data).
    time : str, optional
        Time variable (for panel/DID).
    running_var : str, optional
        Running variable (for RD designs).
    instrument : str, optional
        Instrumental variable.
    cutoff : float, optional
        RD cutoff value.
    design : str, optional
        Override design detection: 'rct', 'did', 'rd', 'iv',
        'observational', 'panel', 'cross-section'.
    dag : DAG, optional
        Causal DAG for identification analysis.
    mediator : str, optional
        Mediator variable name. Triggers mediation / front-door
        recommendations (Imai-Keele-Tingley 2010,
        VanderWeele et al. 2014, Pearl 1995).
    tv_confounders : list of str, optional
        Time-varying confounders (must be pre-treatment at each period).
        Triggers `sp.msm` — Marginal Structural Model via IPTW
        (Robins, Hernán & Brumback 2000).
    proxy_z : list of str, optional
        Treatment-side proxy variables. Triggers `sp.proximal` when
        `proxy_w` is also supplied (Tchetgen Tchetgen et al. 2020).
    proxy_w : list of str, optional
        Outcome-side proxy variables (see `proxy_z`).
    post_treat_strata : str, optional
        Binary post-treatment variable defining principal strata
        (take-up, survival, employment, …). Triggers
        `sp.principal_strat` (Frangakis & Rubin 2002).
    verify : bool, default False
        If True, run *resampling-stability* checks on the top-k
        recommendations (bootstrap CV, permutation placebo, 50%-subsample
        sign agreement) and attach a composite ``score`` to each
        recommendation's ``verify`` dict. The score is used to re-rank
        the top-k. Opt-in because it costs extra compute.

        IMPORTANT: this score measures whether estimates are **stable**
        under resampling, NOT whether the method satisfies the design's
        identification assumptions. A biased method on observational
        data can score near 100. See ``statspai.smart.verify`` docstring.
    verify_B : int, default 50
        Bootstrap replications per recommendation (auto-capped by budget).
    verify_budget_s : float, default 30.0
        Wall-clock budget (seconds) per verified recommendation.
    verify_top_k : int, default 3
        Number of top recommendations to verify.
    allow_experimental : bool, default False
        Whether to include estimators registered as
        ``stability='experimental'`` (or ``'deprecated'``) in the
        ranked output. The default ``False`` is the agent-safe choice
        — an LLM agent or pipeline that asks for an estimator
        recommendation should not silently land on a frontier MVP. Set
        ``True`` when you are explicitly exploring frontier methods
        (e.g. ``causal_text``, ``did_multiplegt_dyn``); dropped names
        are listed in ``RecommendationResult.warnings`` either way.
        See ``docs/guides/stability.md`` for the full contract.

    Returns
    -------
    RecommendationResult
        With .summary(), .run(), .run_all() methods.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> rec = sp.recommend(df, y="log_wage", treatment="union",
    ...                    covariates=["education", "experience"])
    >>> type(rec).__name__
    'RecommendationResult'
    >>> bool(len(rec.recommendations) > 0)
    True
    >>> bool("method" in rec.recommendations[0])
    True
    """
    if covariates is None:
        covariates = [
            c
            for c in data.columns
            if c not in [y, treatment, id, time, running_var, instrument]
            and pd.api.types.is_numeric_dtype(data[c])
        ]

    profile = _profile_data(data, y, treatment, id, time)

    if design is None:
        design = _detect_design(
            data, y, treatment, id, time, running_var, instrument, profile
        )

    warnings_list: List[str] = []
    recommendations: List[Dict[str, Any]] = []

    # DAG-based recommendations
    dag_adjustment: Optional[List[str]] = None
    if dag is not None:
        try:
            adj_sets = dag.adjustment_sets(treatment, y)
            if adj_sets:
                dag_adjustment = list(adj_sets[0])
                bad = dag.bad_controls(treatment, y)
                if bad:
                    warnings_list.append(
                        f"DAG flags bad controls (do NOT include): {bad}"
                    )
        except Exception as e:
            warnings_list.append(f"DAG analysis failed: {e}")

    controls = dag_adjustment if dag_adjustment else covariates
    ctrl_str = ", ".join(f"'{c}'" for c in controls[:5])
    if len(controls) > 5:
        ctrl_str += ", ..."

    # Missing data warning
    if profile["missing_pct"] > 0.1:
        warnings_list.append(
            f"{profile['missing_pct']:.0%} observations have missing values. "
            f"Consider sp.mice() before estimation."
        )

    # ===== DESIGN-SPECIFIC RECOMMENDATIONS =====

    if design == "rct":
        recommendations.append(
            {
                "method": "OLS with robust SE (primary)",
                "function": "regress",
                "reason": "RCT: simple difference in means is unbiased. "
                "Add covariates for precision.",
                "assumptions": ["Random assignment", "SUTVA", "No attrition bias"],
                "robustness": "Check sp.balance_check() and sp.attrition_test()",
                "code": f"sp.regress('{y} ~ {treatment} + {ctrl_str}', data=df, robust='hc1')",
                "params": {
                    "formula": f"{y} ~ {treatment}",
                    "data": data,
                    "robust": "hc1",
                },
            }
        )
        if profile["y_type"] == "binary":
            recommendations.append(
                {
                    "method": "Logit (for binary outcome)",
                    "function": "logit",
                    "reason": "Binary outcome → logit for correct functional form.",
                    "code": f"sp.logit(data=df, y='{y}', x=['{treatment}'] + controls)",
                    "params": {"data": data, "y": y, "x": [treatment] + controls[:5]},
                }
            )

    elif design == "did":
        # Staggered (panel, multi-period) needs a unit id to build cohorts.
        # Without an id — e.g. repeated cross-sections (different individuals
        # each period) — fall through to the pooled / classic DiD card, which
        # does not require a panel id.
        if time and id and data[time].nunique() > 2:
            cohort_col = f"_cohort_{treatment}"

            def _derive_cohort(
                df_in: pd.DataFrame,
                _treat: Any = treatment,
                _id: Any = id,
                _time: Any = time,
                _col: str = cohort_col,
            ) -> pd.DataFrame:
                """Attach cohort column = first treated period per unit."""
                out = df_in.copy()
                if _id and _id in out.columns:
                    treated = out[out[_treat] == 1]
                    cmap = treated.groupby(_id)[_time].min()
                    out[_col] = out[_id].map(cmap).fillna(0).astype(int)
                else:
                    out[_col] = 0
                return out

            did_data = _derive_cohort(data)
            recommendations.append(
                {
                    "method": "Callaway-Sant'Anna (2021) — staggered DID",
                    "function": "callaway_santanna",
                    "reason": "Multiple time periods with staggered treatment adoption. "
                    "Robust to heterogeneous treatment effects (unlike TWFE).",
                    "assumptions": [
                        "Parallel trends",
                        "No anticipation",
                        "Staggered adoption",
                    ],
                    "robustness": "Run sp.pretrends_test(), sp.honest_did(), sp.event_study()",
                    "code": f"# Derived cohort column = first period treated\n"
                    f"sp.callaway_santanna(df, y='{y}', g='{cohort_col}', "
                    f"t='{time}', i='{id}')",
                    "params": {
                        "data": did_data,
                        "y": y,
                        "g": cohort_col,
                        "t": time,
                        "i": id,
                    },
                    "prep": _derive_cohort,
                    "raw_treat": treatment,
                }
            )
            recommendations.append(
                {
                    "method": "Sun-Abraham (2021) — interaction-weighted",
                    "function": "sun_abraham",
                    "reason": "Alternative heterogeneity-robust DID estimator.",
                    "code": f"sp.sun_abraham(df, y='{y}', g='{cohort_col}', "
                    f"t='{time}', i='{id}')",
                    "params": {
                        "data": did_data,
                        "y": y,
                        "g": cohort_col,
                        "t": time,
                        "i": id,
                    },
                    "prep": _derive_cohort,
                    "raw_treat": treatment,
                }
            )
            recommendations.append(
                {
                    "method": "Event-study (dynamic treatment effects)",
                    "function": "event_study",
                    "reason": "Trace the effect by event time: pre-periods test "
                    "parallel trends, post-periods show the dynamic path.",
                    "assumptions": ["Parallel trends", "No anticipation"],
                    "robustness": "Joint pre-trend test; honest-DiD bands on the "
                    "event-study path (sp.honest_did).",
                    "code": f"sp.event_study(df, y='{y}', treat_time='{cohort_col}', "
                    f"time='{time}', unit='{id}')",
                    "params": {
                        "data": did_data,
                        "y": y,
                        "treat_time": cohort_col,
                        "time": time,
                        "unit": id,
                    },
                    "prep": _derive_cohort,
                }
            )
        else:
            _rcs = bool(time and not id and data[time].nunique() > 2)
            recommendations.append(
                {
                    "method": (
                        "Classic 2×2 DID"
                        if not _rcs
                        else "Pooled DID (repeated cross-sections)"
                    ),
                    "function": "did",
                    "reason": (
                        "Two groups, two periods — classic DID is appropriate."
                        if not _rcs
                        else "Repeated cross-sections (a treated/control group "
                        "observed over time, no panel id) — pooled DID on the "
                        "group × post interaction."
                    ),
                    "assumptions": ["Parallel trends", "No anticipation", "SUTVA"],
                    "code": f"sp.did(df, y='{y}', treat='{treatment}', time='{time}')",
                    "params": {"data": data, "y": y, "treat": treatment, "time": time},
                }
            )

    elif design == "synth":
        # Single-treated-unit comparative case study → synthetic control. Derive
        # the treated unit's name and adoption period from the treatment column
        # so the recommendation is directly runnable via .run().
        treated_unit_val: Any = None
        treat_time_val: Any = None
        if id and id in data.columns:
            _ever = data.groupby(id)[treatment].max()
            _treated_units = [u for u, v in _ever.items() if v and v > 0]
            if _treated_units:
                treated_unit_val = _treated_units[0]
                if time and time in data.columns:
                    _mask = (data[id] == treated_unit_val) & (data[treatment] > 0)
                    if _mask.any():
                        treat_time_val = data.loc[_mask, time].min()
        n_donors = int((data.groupby(id)[treatment].max() <= 0).sum()) if id else 0
        recommendations.append(
            {
                "method": "Synthetic Control (Abadie, Diamond & Hainmueller 2010)",
                "function": "synth",
                "reason": (
                    "Single treated unit with a long pre-period and a donor pool "
                    f"of {n_donors} controls — a comparative case study. "
                    "Staggered-DID estimators are degenerate with one treated "
                    "unit; build a synthetic counterfactual from the donor pool."
                ),
                "assumptions": [
                    "Good pre-treatment fit (low RMSPE)",
                    "No interference / spillover onto donor units",
                    "Treated unit in the donor convex hull",
                ],
                "robustness": (
                    "Check pre-treatment RMSPE, run in-space placebo inference "
                    "(sp.synth_placebo / sp.synth_sensitivity), and "
                    "sp.synth_time_placebo()."
                ),
                "code": (
                    f"sp.synth(df, outcome='{y}', unit='{id}', time='{time}', "
                    f"treated_unit={treated_unit_val!r}, "
                    f"treatment_time={treat_time_val!r})"
                ),
                "params": {
                    "data": data,
                    "outcome": y,
                    "unit": id,
                    "time": time,
                    "treated_unit": treated_unit_val,
                    "treatment_time": treat_time_val,
                    "method": "classic",
                },
            }
        )
        recommendations.append(
            {
                "method": "Augmented SCM (Ben-Michael, Feller & Rothstein 2021)",
                "function": "augsynth",
                "reason": (
                    "Ridge-augmented SCM corrects bias when the treated unit "
                    "falls outside the donor convex hull (imperfect pre-fit)."
                ),
                "code": (
                    f"sp.augsynth(df, outcome='{y}', unit='{id}', time='{time}', "
                    f"treated_unit={treated_unit_val!r}, "
                    f"treatment_time={treat_time_val!r})"
                ),
                "params": {
                    "data": data,
                    "outcome": y,
                    "unit": id,
                    "time": time,
                    "treated_unit": treated_unit_val,
                    "treatment_time": treat_time_val,
                },
            }
        )
        recommendations.append(
            {
                "method": "Synthetic DiD (Arkhangelsky et al. 2021)",
                "function": "synthdid_estimate",
                "reason": (
                    "Combines synthetic-control unit weights with DID time "
                    "weights; doubly robust to imperfect pre-fit and mild "
                    "parallel-trends violations."
                ),
                "code": (
                    f"sp.synthdid_estimate(df, y='{y}', unit='{id}', "
                    f"time='{time}', treat_unit={treated_unit_val!r}, "
                    f"treat_time={treat_time_val!r})"
                ),
                "params": {
                    "data": data,
                    "y": y,
                    "unit": id,
                    "time": time,
                    "treat_unit": treated_unit_val,
                    "treat_time": treat_time_val,
                },
            }
        )

    elif design == "rd":
        rv = running_var or "running_var"
        cutoff_value = cutoff or 0
        # High-confidence sharp-vs-fuzzy auto-detection: if a treatment column
        # is given and is (nearly) a deterministic step at the cutoff, it is a
        # SHARP RD; otherwise the treatment *probability* jumps → FUZZY RD
        # (identified like IV at the cutoff), which needs a first-stage
        # compliance check. Safe by construction — it only refines an already
        # detected RD, so there is no false-positive risk on non-RD designs.
        _fuzzy = False
        if treatment and treatment in data.columns and rv in data.columns:
            try:
                _above = (data[rv] >= cutoff_value).astype(int)
                _match = float((data[treatment] == _above).mean())
                _fuzzy = _match < 0.95  # not a clean step → fuzzy
            except (KeyError, TypeError, ValueError):
                _fuzzy = False
        if _fuzzy:
            recommendations.append(
                {
                    "method": "Fuzzy RD (local polynomial, CCT 2014)",
                    "function": "rdrobust",
                    "reason": "Treatment probability (not treatment itself) jumps "
                    "at the cutoff — a fuzzy RD, identified like IV at the cutoff.",
                    "assumptions": [
                        "Continuity of potential outcomes at cutoff",
                        "No manipulation of the running variable",
                        "First-stage compliance jump (monotonicity)",
                    ],
                    "robustness": "Check the first-stage compliance jump (a weak "
                    "jump = weak IV at the cutoff); sp.rddensity(), "
                    "sp.rdbwsensitivity(), sp.rdplacebo().",
                    "code": f"sp.rdrobust(df, y='{y}', x='{rv}', c={cutoff_value}, "
                    f"fuzzy='{treatment}')",
                    "params": {
                        "data": data,
                        "y": y,
                        "x": rv,
                        "c": cutoff_value,
                        "fuzzy": treatment,
                    },
                }
            )
        else:
            recommendations.append(
                {
                    "method": "Local polynomial RD (CCT 2014)",
                    "function": "rdrobust",
                    "reason": "Sharp RD with MSE-optimal bandwidth and bias correction.",
                    "assumptions": [
                        "Continuity of potential outcomes at cutoff",
                        "No manipulation of running variable",
                    ],
                    "robustness": "Run sp.rddensity(), sp.rdbwsensitivity(), sp.rdplacebo()",
                    "code": f"sp.rdrobust(df, y='{y}', x='{rv}', c={cutoff_value})",
                    "params": {"data": data, "y": y, "x": rv, "c": cutoff_value},
                }
            )

    elif design == "iv":
        z = instrument or "instrument"
        exog_controls = [c for c in controls if c not in (treatment, z)]
        exog_str = " + ".join(exog_controls[:5]) if exog_controls else ""
        iv_formula = (
            f"{y} ~ {exog_str} + ({treatment} ~ {z})"
            if exog_str
            else f"{y} ~ ({treatment} ~ {z})"
        )

        # Compute the first-stage F live so the ranking + reasons can
        # adapt to weak instruments (Staiger-Stock 1997 rule of thumb
        # F=10; Stock-Yogo 2005 10% max-size critical value F=16.38
        # for one endogenous variable / one instrument).
        first_stage_F: Optional[float] = None
        weak_iv = False
        very_weak_iv = False
        if treatment and z and treatment in data.columns and z in data.columns:
            try:
                d_vec = data[treatment].astype(float).to_numpy()
                z_vec = data[z].astype(float).to_numpy()
                exog_arrays = []
                for c in exog_controls:
                    if c in data.columns:
                        try:
                            exog_arrays.append(data[c].astype(float).to_numpy())
                        except (TypeError, ValueError):
                            pass
                n_obs = len(d_vec)
                ones = np.ones(n_obs)
                X_full = np.column_stack([ones, z_vec] + exog_arrays)
                X_restricted = np.column_stack([ones] + exog_arrays)
                beta_full, *_ = np.linalg.lstsq(X_full, d_vec, rcond=None)
                beta_rest, *_ = np.linalg.lstsq(X_restricted, d_vec, rcond=None)
                rss_full = float(np.sum((d_vec - X_full @ beta_full) ** 2))
                rss_rest = float(np.sum((d_vec - X_restricted @ beta_rest) ** 2))
                df_denom = n_obs - X_full.shape[1]
                if rss_full > 0 and df_denom > 0:
                    first_stage_F = ((rss_rest - rss_full) / 1) / (rss_full / df_denom)
                    very_weak_iv = first_stage_F < 10.0
                    weak_iv = first_stage_F < 16.38
            except (np.linalg.LinAlgError, ValueError, KeyError, TypeError):
                first_stage_F = None
                weak_iv = False
                very_weak_iv = False

        # Build the 2SLS recommendation with adaptive reason.
        twoSLS_reason = "Standard IV estimator for endogenous treatment."
        twoSLS_assumptions = [
            "Instrument relevance (F > 10)",
            "Exclusion restriction",
            "Monotonicity (for LATE)",
        ]
        if first_stage_F is not None:
            twoSLS_reason += f" First-stage F = {first_stage_F:.2f}"
            if very_weak_iv:
                twoSLS_reason += (
                    " < 10 (Staiger-Stock 1997 rule of thumb): 2SLS "
                    "biased toward OLS, HC1 SEs ignore weak-IV bias. "
                    "Prefer LIML and Anderson-Rubin inference below."
                )
            elif weak_iv:
                twoSLS_reason += (
                    " < 16.38 (Stock-Yogo 2005 10% max size for 1 "
                    "endog/1 IV): consider LIML or AR inference."
                )
            else:
                twoSLS_reason += " (clears Stock-Yogo 10% max size)."

        twoSLS_rec: Dict[str, Any] = {
            "method": "2SLS (two-stage least squares)",
            "function": "ivreg",
            "reason": twoSLS_reason,
            "assumptions": twoSLS_assumptions,
            "robustness": (
                "Check first-stage F, sp.anderson_rubin_test(), " "sp.kitagawa_test()"
            ),
            "code": f"sp.ivreg('{iv_formula}', data=df, robust='hc1')",
            "params": {"formula": iv_formula, "data": data, "robust": "hc1"},
        }
        if first_stage_F is not None:
            twoSLS_rec["first_stage_F"] = float(first_stage_F)
            twoSLS_rec["weak_iv"] = bool(weak_iv)
            twoSLS_rec["very_weak_iv"] = bool(very_weak_iv)

        liml_reason = (
            "Limited Information Maximum Likelihood; less biased "
            "than 2SLS under weak instruments."
        )
        if very_weak_iv:
            liml_reason = (
                f"First-stage F = {first_stage_F:.2f} < 10 "
                "(Staiger-Stock rule of thumb): LIML is the preferred "
                "point-estimator under weak IV, with better "
                "small-sample bias than 2SLS."
            )
        elif weak_iv:
            liml_reason = (
                f"First-stage F = {first_stage_F:.2f} < 16.38 "
                "(Stock-Yogo 10% max size): LIML reduces 2SLS "
                "weak-instrument bias."
            )
        liml_rec: Dict[str, Any] = {
            "method": "LIML (robust to weak instruments)",
            "function": "liml",
            "reason": liml_reason,
            "code": (
                f"sp.liml(data=df, y='{y}', x_endog=['{treatment}'], " f"z=['{z}'])"
            ),
            "params": {"data": data, "y": y, "x_endog": [treatment], "z": [z]},
        }

        # An Anderson-Rubin confidence interval is robust to weak
        # instruments by construction; surface it as a third row when
        # weak IV is detected so the agent can use AR for inference
        # while LIML provides the point estimate.
        ar_rec = None
        if very_weak_iv or weak_iv:
            ar_rec = {
                "method": (
                    "Anderson-Rubin confidence interval " "(weak-IV robust inference)"
                ),
                "function": "anderson_rubin_ci",
                "reason": (
                    "AR confidence intervals are valid even when the "
                    "first-stage F is small; recommended whenever "
                    "2SLS HC1 SEs cannot be trusted."
                ),
                "code": (
                    f"sp.anderson_rubin_ci(data=df, y='{y}', "
                    f"d='{treatment}', z=['{z}'])"
                ),
                "params": {"data": data, "y": y, "d": treatment, "z": [z]},
            }

        # Ranking: under (very) weak IV, lift LIML and AR above 2SLS so
        # the top-of-list recommendation matches the inference that
        # actually has good calibration on the given data.  Under
        # strong IV, keep the historical 2SLS-first ordering.
        if very_weak_iv:
            recommendations.append(liml_rec)
            if ar_rec is not None:
                recommendations.append(ar_rec)
            recommendations.append(twoSLS_rec)
        else:
            recommendations.append(twoSLS_rec)
            recommendations.append(liml_rec)
            if ar_rec is not None:
                recommendations.append(ar_rec)

        # Surface a top-level warning so the human-readable
        # ``summary()`` and the agent-facing ``warnings`` field both
        # carry the weak-IV signal — duplicate of the per-row
        # rationale, but at the place where downstream
        # workflow-orchestration code reads ``RecommendationResult.warnings``.
        if very_weak_iv:
            warnings_list.append(
                f"First-stage F = {first_stage_F:.2f} < 10 "
                f"(Staiger-Stock 1997): 2SLS HC1 SEs are biased "
                f"toward OLS. LIML promoted to #1; "
                f"sp.anderson_rubin_ci(...) added. Mirrors "
                f"`sp.preflight(data, 'ivreg', formula=...)` "
                f"first_stage_strength check."
            )
        elif weak_iv:
            warnings_list.append(
                f"First-stage F = {first_stage_F:.2f} < 16.38 "
                f"(Stock-Yogo 2005, 10% max size for 1 endog/1 IV): "
                f"consider method='liml' or sp.anderson_rubin_ci(...)."
            )

    elif design == "observational":
        # Lead with a confounding-adjusting estimator, NOT bare OLS. On a
        # selection-on-observables design, naive OLS is biased under
        # confounding (the canonical Dehejia-Wahba 1999 result); leading with
        # it produces a plausible-but-wrong headline whose audit only asks
        # regression checks and misses overlap / balance. See
        # benchmarks/recommend_hit_rate F-002.
        #
        # Propensity-score matching is only defined for a BINARY treatment; for
        # a continuous / categorical dose lead with DML (which handles
        # continuous treatment) and treat OLS-with-controls as a legitimate
        # linear adjustment rather than a naive baseline.
        treat_is_binary = profile.get("treat_type") == "binary"
        # Adjusting estimators (PSM / DML) require covariates to adjust on. With
        # no observables, selection-on-observables is vacuous and OLS /
        # difference-in-means is all that is identified — so only then does OLS
        # lead. PSM additionally requires a BINARY treatment; for a continuous
        # dose, DML leads.
        adjust_controls = [c for c in controls if c not in (treatment, y)]
        has_controls = len(adjust_controls) > 0
        if treat_is_binary and has_controls:
            recommendations.append(
                {
                    "method": "Propensity Score Matching (selection on observables)",
                    "function": "match",
                    "reason": "Nonparametric causal effect under unconfoundedness; "
                    "enforces common support and lets you check covariate balance.",
                    "assumptions": [
                        "Unconfoundedness (CIA)",
                        "Common support (overlap)",
                    ],
                    "robustness": "Check sp.overlap_plot() / common support, "
                    "covariate balance after matching (sp.love_plot), and "
                    "sp.sensemakr() / sp.oster_bounds() for unobserved confounding.",
                    "code": f"sp.match(df, y='{y}', treat='{treatment}', "
                    f"covariates=[{ctrl_str}])",
                    "params": {
                        "data": data,
                        "y": y,
                        "treat": treatment,
                        "covariates": controls[:10],
                    },
                }
            )
        if has_controls:
            recommendations.append(
                {
                    "method": "Double ML (high-dimensional controls)",
                    "function": "dml",
                    "reason": "Doubly-robust causal effect; handles many controls "
                    "without overfitting via cross-fitting"
                    + ("." if treat_is_binary else " (and a continuous treatment)."),
                    "assumptions": [
                        "Unconfoundedness (CIA)",
                        "Overlap",
                        "Correct nuisance learners",
                    ],
                    "robustness": "Check overlap of estimated propensity scores and "
                    "sp.dml_sensitivity() for unobserved confounding.",
                    "code": f"sp.dml(df, y='{y}', treat='{treatment}', "
                    f"covariates=[{ctrl_str}])",
                    "params": {
                        "data": data,
                        "y": y,
                        "treat": treatment,
                        "covariates": controls[:20],
                    },
                }
            )
        if not has_controls:
            _ols_method = "OLS with robust SE"
            _ols_reason = (
                "No covariates were detected to adjust on, so a "
                "selection-on-observables adjustment is not identified; OLS / "
                "difference-in-means is the available estimate."
            )
            _ols_assumptions = ["E[ε|X]=0 (exogeneity)"]
            _ols_robust = "Run sp.sensemakr(), sp.oster_bounds(), sp.spec_curve()."
        elif treat_is_binary:
            _ols_method = (
                "OLS with robust SE (naive baseline — biased under confounding)"
            )
            _ols_reason = (
                "Reference point only, NOT the causal estimate: OLS is biased "
                "when treatment is confounded. Compare against the matching / "
                "DML estimate to gauge the selection bias."
            )
            _ols_assumptions = ["E[ε|X]=0 (exogeneity — implausible under selection)"]
            _ols_robust = (
                "Run sp.sensemakr(), sp.oster_bounds(), sp.spec_curve(); check "
                "overlap / common support before trusting any adjustment."
            )
        else:
            _ols_method = "OLS with robust SE (linear-adjustment baseline)"
            _ols_reason = (
                "Linear adjustment for the controls; consistent under "
                "unconfoundedness AND linearity. Compare against DML, which "
                "relaxes the linear functional form."
            )
            _ols_assumptions = [
                "Unconfoundedness (CIA)",
                "Correct (linear) functional form",
            ]
            _ols_robust = "Run sp.sensemakr(), sp.oster_bounds(), sp.spec_curve()."
        recommendations.append(
            {
                "method": _ols_method,
                "function": "regress",
                "reason": _ols_reason,
                "assumptions": _ols_assumptions,
                "robustness": _ols_robust,
                "code": f"sp.regress('{y} ~ {treatment} + {ctrl_str}', data=df, robust='hc1')",
                "params": {
                    "formula": f"{y} ~ {treatment}",
                    "data": data,
                    "robust": "hc1",
                },
            }
        )

    elif design == "panel":
        panel_rhs = treatment if treatment else "1"
        panel_controls = [c for c in controls if c != treatment][:5]
        if panel_controls:
            panel_rhs += " + " + " + ".join(panel_controls)
        panel_formula = f"{y} ~ {panel_rhs}"
        recommendations.append(
            {
                "method": "Panel FE (within estimator)",
                "function": "panel",
                "reason": "Controls for time-invariant unobservables.",
                "assumptions": ["Strict exogeneity", "No time-varying confounders"],
                "code": f"sp.panel(df, '{panel_formula}', "
                f"entity='{id}', time='{time}', method='fe')",
                "params": {
                    "data": data,
                    "formula": panel_formula,
                    "entity": id,
                    "time": time,
                    "method": "fe",
                },
            }
        )
        recommendations.append(
            {
                "method": "Correlated Random Effects (Mundlak)",
                "function": "panel",
                "reason": "Mundlak projection allows RE efficiency with FE consistency.",
                "code": f"sp.panel(df, '{panel_formula}', "
                f"entity='{id}', time='{time}', method='mundlak')",
                "params": {
                    "data": data,
                    "formula": panel_formula,
                    "entity": id,
                    "time": time,
                    "method": "mundlak",
                },
            }
        )

    elif design == "bunching":
        # Mass-point / kink design: the behavioural response shows up as
        # excess mass at a policy threshold (Saez 2010; Kleven 2016).
        _thr = cutoff if cutoff is not None else 0.0
        recommendations.append(
            {
                "method": "Bunching estimator (Saez 2010; Kleven 2016)",
                "function": "bunching",
                "reason": "Excess mass at a kink/notch in the running variable "
                "identifies the behavioural elasticity; estimate the "
                "counterfactual density and the bunching mass.",
                "assumptions": [
                    "Smooth counterfactual density absent the threshold",
                    "No other policy discontinuity at the threshold",
                ],
                "robustness": "Vary the excluded bunching window and the "
                "polynomial order; check for round-number / heaping bunching.",
                "code": f"sp.bunching(df, running_var='{running_var}', "
                f"threshold={_thr})",
                "params": {
                    "data": data,
                    "running_var": running_var,
                    "threshold": _thr,
                },
            }
        )

    elif design == "rkd":
        # Regression kink design: the SLOPE of the outcome in the running
        # variable changes at the kink (Card, Lee, Pei & Weber 2015).
        recommendations.append(
            {
                "method": "Regression Kink Design (Card, Lee, Pei & Weber 2015)",
                "function": "rkd",
                "reason": "A kink (not a jump) in the policy rule at the "
                "threshold identifies the effect from the change in slope.",
                "assumptions": [
                    "Smooth density at the kink",
                    "Continuous slope of confounders at the kink",
                ],
                "robustness": "Bandwidth sensitivity; placebo kinks; "
                "McCrary-style density continuity.",
                "code": f"sp.rkd(df, y='{y}', x='{running_var}', "
                f"c={cutoff if cutoff is not None else 0})",
                "params": {
                    "data": data,
                    "y": y,
                    "x": running_var,
                    "c": cutoff if cutoff is not None else 0,
                },
            }
        )

    elif design == "ddd":
        # Triple-difference: a second comparison group nets out a
        # confound that ordinary DiD cannot (Gruber 1994; Olden & Møen 2022).
        recommendations.append(
            {
                "method": "Triple Difference (DDD)",
                "function": "ddd",
                "reason": "A second (placebo) group differences out a "
                "group-specific shock that violates DiD parallel trends.",
                "assumptions": [
                    "Parallel trends in the *difference* across subgroups",
                    "No subgroup-specific treatment-timing confounder",
                ],
                "robustness": "Inspect each constituent 2x2; event-study the "
                "triple difference; honest-DiD on the DDD estimand.",
                "code": f"sp.ddd(df, y='{y}', treat='{treatment}', "
                f"time='{time}', subgroup='<subgroup>')",
                "params": {
                    "data": data,
                    "y": y,
                    "treat": treatment,
                    "time": time,
                },
            }
        )

    elif design == "bartik":
        # Shift-share / Bartik IV: identification from exposure shares
        # interacted with common shocks (Goldsmith-Pinkham, Sorkin & Swift 2020).
        recommendations.append(
            {
                "method": "Bartik / shift-share IV (Goldsmith-Pinkham et al. 2020)",
                "function": "bartik",
                "reason": "Exposure shares × common shocks instrument an "
                "endogenous regressor; identification is the share-weighted "
                "sum of just-identified IVs.",
                "assumptions": [
                    "Exogenous shares (or exogenous shocks)",
                    "Relevance of the shift-share instrument",
                ],
                "robustness": "Rotemberg weights to find influential shares; "
                "over-identification across shocks; pre-trend balance on shares.",
                "code": f"sp.bartik(df, y='{y}', endog='{treatment}', "
                f"shares=<shares_df>, shocks=<shocks>)",
                "params": {"data": data, "y": y, "endog": treatment},
            }
        )

    elif design == "decomposition":
        # Between-group gap decomposition (Oaxaca 1973; Blinder 1973;
        # DiNardo, Fortin & Lemieux 1996 for the distributional version).
        _grp = treatment
        _x = [c for c in controls if c != treatment][:10]
        recommendations.append(
            {
                "method": "Oaxaca-Blinder decomposition (Oaxaca 1973; Blinder 1973)",
                "function": "oaxaca",
                "reason": "Split a between-group mean gap into an "
                "explained (composition) and unexplained (structure) part.",
                "assumptions": [
                    "Correct (linear) conditional mean per group",
                    "Common support of covariates across groups",
                ],
                "robustness": "Check the detailed contributions; for the full "
                "distribution use sp.decompose(method='dfl'/'ffl'/'rif').",
                "code": f"sp.oaxaca(df, y='{y}', group='{_grp}', " f"x={_x})",
                "params": {"data": data, "y": y, "group": _grp, "x": _x},
            }
        )

    else:
        # Cross-section
        if treatment:
            formula = f"{y} ~ {treatment}"
        elif controls:
            formula = f"{y} ~ {controls[0]}"
        else:
            formula = f"{y} ~ 1"
        recommendations.append(
            {
                "method": "OLS with robust SE",
                "function": "regress",
                "reason": "Cross-sectional data with continuous outcome.",
                "code": f"sp.regress('{y} ~ {ctrl_str or '...'}', data=df, robust='hc1')",
                "params": {"formula": formula, "data": data, "robust": "hc1"},
            }
        )

    # ====================================================================== #
    #  Sprint-B causal extensions (0.9.6): opt-in via the new kwargs.        #
    #  These append to the candidate list — primary design-based             #
    #  recommendations still drive the top slot.                             #
    # ====================================================================== #

    # Proximal Causal Inference — unobserved confounding with twin proxies
    if proxy_z and proxy_w and treatment:
        exog = [c for c in (covariates or []) if c not in proxy_z + proxy_w]
        recommendations.append(
            {
                "method": "Proximal Causal Inference (linear bridge 2SLS)",
                "function": "proximal",
                "reason": "Unmeasured confounder U with a treatment-side "
                "proxy Z and outcome-side proxy W available; "
                "linear bridge 2SLS identifies ATE under the "
                "proxy completeness conditions.",
                "assumptions": [
                    "Z ⊥ Y | (D, U, X)  — treatment proxy",
                    "W ⊥ D | (U, X)  — outcome proxy",
                    "Linear outcome bridge h(W, D, X) (current release)",
                ],
                "robustness": "Inspect first_stage_F in r.model_info; "
                "compare to sp.dml/sp.aipw for sensitivity.",
                "code": f"sp.proximal(df, y='{y}', treat='{treatment}', "
                f"proxy_z={proxy_z!r}, proxy_w={proxy_w!r})",
                "params": {
                    "data": data,
                    "y": y,
                    "treat": treatment,
                    "proxy_z": list(proxy_z),
                    "proxy_w": list(proxy_w),
                    "covariates": exog,
                },
            }
        )

    # Marginal Structural Model — time-varying treatment + tv confounders
    if tv_confounders and treatment and id and time:
        baseline = [c for c in (covariates or []) if c not in tv_confounders]
        recommendations.append(
            {
                "method": "Marginal Structural Model (stabilized IPTW)",
                "function": "msm",
                "reason": "Time-varying treatment with time-varying confounders "
                "that are themselves affected by past treatment. "
                "Standard panel regression blocks a causal path and "
                "opens a collider; MSM with stabilized weights "
                "recovers the marginal causal parameter.",
                "assumptions": [
                    "Sequential exchangeability",
                    "Positivity at every period",
                    "Consistency / no interference",
                ],
                "robustness": "Check sw_mean ≈ 1 and sw_max in model_info; "
                "try trim_per_period=True if weights blow up.",
                "code": (
                    f"sp.msm(panel, y='{y}', treat='{treatment}', "
                    f"id='{id}', time='{time}', "
                    f"time_varying={tv_confounders!r}, "
                    f"baseline={baseline[:3]!r})"
                ),
                "params": {
                    "data": data,
                    "y": y,
                    "treat": treatment,
                    "id": id,
                    "time": time,
                    "time_varying": list(tv_confounders),
                    "baseline": baseline,
                },
            }
        )

    # Principal Stratification — post-treatment strata variable
    if post_treat_strata and treatment:
        assumps = ["Monotonicity S(1) ≥ S(0)", "Exclusion restriction"]
        rec_args = {
            "data": data,
            "y": y,
            "treat": treatment,
            "strata": post_treat_strata,
        }
        if covariates:
            rec_args["covariates"] = covariates
            rec_args["method"] = "principal_score"
            method_label = "Principal stratification (principal score weighting)"
            function_reason = (
                "Covariates available — Ding & Lu (2017) "
                "principal score point-identifies "
                "always-taker / complier / never-taker PCEs "
                "under principal ignorability."
            )
            assumps.append("Principal ignorability Y(d) ⊥ stratum | X within D=d")
            code_tail = f", covariates={covariates[:3]!r}, method='principal_score'"
        else:
            rec_args["method"] = "monotonicity"
            method_label = (
                "Principal stratification (monotonicity + Zhang-Rubin bounds)"
            )
            function_reason = (
                "Post-treatment stratum variable present; "
                "monotonicity + Zhang-Rubin (2003) sharp "
                "bounds on SACE plus complier LATE."
            )
            code_tail = ""
        recommendations.append(
            {
                "method": method_label,
                "function": "principal_strat",
                "reason": function_reason,
                "assumptions": assumps,
                "robustness": "Inspect mono_violation_frac in model_info; "
                "pair with a sensitivity analysis.",
                "code": (
                    f"sp.principal_strat(df, y='{y}', treat='{treatment}', "
                    f"strata='{post_treat_strata}'{code_tail})"
                ),
                "params": rec_args,
            }
        )

    # Mediation recommendations (natural + interventional + front-door)
    if mediator and treatment:
        # Natural effects (Imai-Keele-Tingley)
        recommendations.append(
            {
                "method": "Causal mediation — natural direct/indirect effects",
                "function": "mediate",
                "reason": "Decomposes total effect into ACME (indirect via M) "
                "and ADE (direct). Uses the product / quasi-Bayesian "
                "simulation approach.",
                "assumptions": [
                    "No unobserved D-Y confounder",
                    "No unobserved M-Y confounder",
                    "No treatment-induced M-Y confounder",
                    "Cross-world independence (natural effects)",
                ],
                "code": (
                    f"sp.mediate(df, y='{y}', treat='{treatment}', "
                    f"mediator='{mediator}')"
                ),
                "params": {
                    "data": data,
                    "y": y,
                    "treat": treatment,
                    "mediator": mediator,
                    "covariates": covariates,
                },
            }
        )
        # Interventional effects — appropriate when tv_confounders present
        if tv_confounders:
            recommendations.append(
                {
                    "method": "Interventional mediation (VanderWeele 2014)",
                    "function": "mediate_interventional",
                    "reason": "Treatment-induced mediator-outcome confounder "
                    "present — natural effects are not identified; "
                    "interventional effects are.",
                    "assumptions": [
                        "No unobserved baseline D-Y confounder",
                        "No unobserved M-Y confounder (given L)",
                    ],
                    "code": (
                        f"sp.mediate_interventional(df, y='{y}', "
                        f"treat='{treatment}', mediator='{mediator}', "
                        f"tv_confounders={tv_confounders!r})"
                    ),
                    "params": {
                        "data": data,
                        "y": y,
                        "treat": treatment,
                        "mediator": mediator,
                        "covariates": covariates,
                        "tv_confounders": list(tv_confounders),
                    },
                }
            )
        # Front-door — when the mediator is claimed to fully transmit D→Y
        recommendations.append(
            {
                "method": "Front-door adjustment (Pearl 1995)",
                "function": "front_door",
                "reason": "If an unobserved back-door confounder U blocks the "
                "standard adjustment but the mediator M fully "
                "transmits D's effect on Y, Pearl's front-door "
                "formula identifies the ATE.",
                "assumptions": [
                    "No direct D→Y path (all effect via M)",
                    "No unobserved M-Y confounder",
                    "Positivity on M | D",
                ],
                "robustness": "Verify the DAG assumption with sp.dag; "
                "compare to sp.mediate / sp.mediate_interventional.",
                "code": (
                    f"sp.front_door(df, y='{y}', treat='{treatment}', "
                    f"mediator='{mediator}')"
                ),
                "params": {
                    "data": data,
                    "y": y,
                    "treat": treatment,
                    "mediator": mediator,
                    "covariates": covariates,
                },
            }
        )

    # G-computation as a baseline companion for observational designs
    if (
        design == "observational"
        and treatment
        and profile["treat_type"] in ("binary", "continuous")
    ):
        if profile["treat_type"] == "binary":
            estimand_kw = "ATE"
            gcomp_reason = (
                "Parametric g-formula (standardization) — "
                "complements matching/DML with a pure-outcome-"
                "model baseline; easy-to-audit dose-response "
                "slices."
            )
        else:
            estimand_kw = "dose_response"
            gcomp_reason = (
                "Continuous treatment → g-formula dose-response "
                "curve is a natural summary under "
                "unconfoundedness."
            )
        recommendations.append(
            {
                "method": f"G-computation ({estimand_kw})",
                "function": "g_computation",
                "reason": gcomp_reason,
                "assumptions": [
                    "Unconfoundedness (CIA)",
                    "Correctly-specified outcome model",
                ],
                "code": (
                    f"sp.g_computation(df, y='{y}', treat='{treatment}', "
                    f"covariates={(covariates or [])[:3]!r}, "
                    f"estimand={estimand_kw!r})"
                ),
                "params": {
                    "data": data,
                    "y": y,
                    "treat": treatment,
                    "covariates": covariates or [],
                    "estimand": estimand_kw,
                },
            }
        )

    # Outcome-type-specific additions
    if profile["y_type"] == "binary" and design not in ["rd", "did"]:
        recommendations.append(
            {
                "method": "Logit (binary outcome)",
                "function": "logit",
                "reason": "Binary dependent variable → logit for correct likelihood.",
                "code": f"sp.logit(data=df, y='{y}', x=['{treatment}'] + controls[:5])",
                "params": {
                    "data": data,
                    "y": y,
                    "x": [treatment] + controls[:5] if treatment else controls[:5],
                },
            }
        )
    elif profile["y_type"] == "count":
        recommendations.append(
            {
                "method": "Poisson regression (count outcome)",
                "function": "poisson",
                "reason": "Count outcome → Poisson with robust SE is consistent.",
                "code": f"sp.poisson(data=df, y='{y}', x=['{treatment}'] + controls[:5])",
                "params": {
                    "data": data,
                    "y": y,
                    "x": [treatment] + controls[:5] if treatment else controls[:5],
                },
            }
        )
    elif profile["y_type"] == "fractional":
        recommendations.append(
            {
                "method": "Fractional logit (outcome in [0,1])",
                "function": "fracreg",
                "reason": "Proportional outcome → fractional logit (Papke-Wooldridge).",
                "code": f"sp.fracreg(data=df, y='{y}', x=['{treatment}'] + controls[:5])",
                "params": {
                    "data": data,
                    "y": y,
                    "x": [treatment] + controls[:5] if treatment else controls[:5],
                },
            }
        )

    # Optional empirical verification (Plan 3: rule prior + empirical posterior)
    if verify and recommendations:
        from .verify import verify_recommendation

        k = min(verify_top_k, len(recommendations))
        for rec in recommendations[:k]:
            try:
                rec["verify"] = verify_recommendation(
                    rec,
                    data,
                    B=verify_B,
                    budget_s=verify_budget_s,
                )
            except Exception as e:
                rec["verify"] = {"score": np.nan, "error": str(e)}

        # Re-rank top-k by verify score (stable sort; rest of list untouched).
        # Sort descending on score; NaN/error scores fall to the BOTTOM of
        # the head (keyed to +inf), not the middle — this preserves
        # determinism and makes a score=0.0 method (runnable but unstable)
        # strictly outrank a NaN one (not runnable).
        head = recommendations[:k]
        tail = recommendations[k:]

        def _sort_key(rec: Dict[str, Any]) -> float:
            v = rec.get("verify") or {}
            s = v.get("score")
            if s is None or not np.isfinite(s):
                return float("inf")  # push NaN / missing to the end
            return -float(s)  # primary: descending score

        head.sort(key=_sort_key)
        recommendations = head + tail

    # ------------------------------------------------------------------
    # Agent-native enrichment: pull structured metadata from the registry
    # so each recommendation carries the canonical assumptions / failure
    # modes / alternatives / typical_n_min.  Single source of truth — if
    # an agent card exists, use its fields even when the hardcoded ones
    # here fall out of date.
    # ------------------------------------------------------------------
    _enrich_with_agent_cards(
        recommendations,
        n_obs=int(profile.get("n_obs", 0) or 0),
        warnings_list=warnings_list,
    )

    # ------------------------------------------------------------------
    # Stability gating (v1.13): by default, drop recommendations that
    # point at a function whose registry entry is
    # ``stability='experimental'`` or ``'deprecated'``, so an agent
    # asking ``sp.recommend(...)`` for an applied analysis
    # never silently lands on a frontier MVP.  Pass
    # ``allow_experimental=True`` to opt back in (e.g. when the user
    # is explicitly exploring frontier methods).  See
    # ``docs/guides/stability.md``.
    # ------------------------------------------------------------------
    if not allow_experimental:
        recommendations, dropped = _filter_unstable_recommendations(recommendations)
        if dropped:
            warnings_list.append(
                "Dropped {n} experimental recommendation(s) "
                "({names}) — pass allow_experimental=True to include them.".format(
                    n=len(dropped),
                    names=", ".join(f"sp.{name}" for name in dropped),
                )
            )

    return RecommendationResult(
        recommendations=recommendations,
        data_profile=profile,
        design=design,
        warnings=warnings_list,
        data=data,
        y=y,
        treatment=treatment,
    )


def _filter_unstable_recommendations(
    recommendations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Drop recommendations whose function is experimental/deprecated.

    Returns ``(filtered_recommendations, dropped_function_names)``.
    Stability lookup goes through the registry; if a recommendation's
    ``function`` is not in the registry (or is missing entirely), it
    is preserved — this preserves backward compatibility for any
    custom recommendation a downstream caller may have appended.
    """
    from ..registry import _REGISTRY, _ensure_full_registry  # local: avoid cycle

    _ensure_full_registry()
    keep: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for rec in recommendations:
        fn = rec.get("function")
        if not isinstance(fn, str):
            keep.append(rec)
            continue
        spec = _REGISTRY.get(fn) if fn else None
        if spec is not None and spec.stability in {"experimental", "deprecated"}:
            dropped.append(fn)
            continue
        keep.append(rec)
    return keep, dropped


def _enrich_with_agent_cards(
    recommendations: List[Dict[str, Any]],
    *,
    n_obs: int,
    warnings_list: List[str],
) -> None:
    """Merge registry agent-card metadata into each recommendation in place.

    Adds keys ``agent_card`` (full card), ``pre_conditions``,
    ``failure_modes``, ``alternatives``, ``typical_n_min``.  Preserves
    the hand-written ``assumptions`` / ``reason`` fields unless the
    recommendation left them empty, in which case it promotes the card
    values.  Appends an ``n_obs < typical_n_min`` warning to
    ``warnings_list`` for the top recommendation when applicable.

    Quiet if the named function has no agent card — everything else
    continues to work.
    """
    try:
        from statspai.registry import agent_card as _card
    except ImportError:
        return

    first_flagged = False
    for rec in recommendations:
        name = rec.get("function")
        if not name:
            continue
        try:
            card = _card(name)
        except KeyError:
            continue

        # Only attach an agent_card view if the entry is actually
        # populated with agent-native fields.  Plain auto-registered
        # entries contribute nothing useful here.
        card_has_content = (
            card.get("assumptions")
            or card.get("failure_modes")
            or card.get("alternatives")
            or card.get("pre_conditions")
            or card.get("typical_n_min")
        )
        if not card_has_content:
            continue

        rec.setdefault("agent_card", card)
        rec.setdefault("pre_conditions", card["pre_conditions"])
        rec.setdefault("failure_modes", card["failure_modes"])
        rec.setdefault("alternatives", card["alternatives"])
        rec.setdefault("typical_n_min", card["typical_n_min"])

        # Promote card assumptions when the rec didn't set any.
        if not rec.get("assumptions") and card["assumptions"]:
            rec["assumptions"] = list(card["assumptions"])

        # First rec only: flag n < typical_n_min once in the
        # top-level warnings.
        n_min = card.get("typical_n_min")
        if (
            not first_flagged
            and n_min is not None
            and isinstance(n_min, int)
            and n_obs
            and n_obs < n_min
        ):
            warnings_list.append(
                f"Sample size n={n_obs} is below the typical minimum "
                f"({n_min}) for sp.{name}; interpret cautiously "
                f"(see sp.agent_card('{name}'))."
            )
            first_flagged = True
