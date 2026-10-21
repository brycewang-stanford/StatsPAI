"""
One-call DID robustness summary across multiple staggered-DID estimators.

``did_summary()`` fits a common set of modern staggered-DID estimators to
the same data and returns a tidy comparison table with overall ATT, SE,
95 % CI, p-value, and a short note for each method. The goal is to make
method-robustness checks a single function call instead of hand-wiring
five different estimators.

Supported methods (all use the shared ``(y, group, time, first_treat)``
interface — ``group`` is the unit identifier, ``first_treat`` the cohort
variable):

================  =====================================================
Key               Underlying estimator
================  =====================================================
``'cs'``          Callaway & Sant'Anna (2021) + ``aggte(type='simple')``
``'sa'``          Sun & Abraham (2021) interaction-weighted
``'bjs'``         Borusyak, Jaravel & Spiess (2024) imputation
``'etwfe'``       Wooldridge (2021) extended TWFE
``'stacked'``     Cengiz et al. (2019) stacked event study
================  =====================================================

Example
-------
>>> import statspai as sp
>>> df = sp.dgp_did(n_units=200, n_periods=10, staggered=True, seed=0)
>>> summary = sp.did_summary(df, y='y', time='time',
...                          first_treat='first_treat', group='unit')
>>> summary.detail["method"].tolist()
['cs', 'sa', 'bjs', 'etwfe', 'stacked']
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._input_validation import missing_named_columns
from ..core._call_record import records_call
from ..core._vcov_spec import accepts_vce
from ..core.results import CausalResult
from ..exceptions import ColumnNotFound


def _stars(p: float) -> str:
    """Significance stars for a p-value."""
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


class DIDSummaryResult(CausalResult):
    """
    CausalResult subclass returned by :func:`did_summary`.

    Defines ``.summary()`` as a real method (rather than a closure-bound
    instance attribute) so the result object serialises cleanly for
    caching (joblib / multiprocessing).

    Notes
    -----
    The ``se`` attribute stores the **cross-method dispersion** (SD of
    estimates across successfully-fit methods), *not* a sampling SE.
    Read :attr:`model_info` ``['dispersion']`` for the same value.
    """

    _DID_SUMMARY_MARKER: bool = True

    @property
    def event_study_paths(self) -> pd.DataFrame:
        """Event-study paths of every method, one row per (method, k).

        Columns ``method, relative_time, att, se, ci_lower, ci_upper,
        pvalue, lead_definition, reference``. Present when the summary was
        fitted with ``event_study=True``; see :func:`did_summary`.
        """
        paths = (self.model_info or {}).get("event_study_paths")
        if not isinstance(paths, pd.DataFrame):
            from ..core._result_access import ResultAccessorUnavailable

            raise ResultAccessorUnavailable(
                "This did_summary result holds no event-study paths.",
                recovery_hint="Refit with sp.did_summary(..., event_study=True).",
            )
        return paths

    # Own layout without the base class's digits= / fmt= precision options;
    # passing them raises TypeError rather than being ignored.
    def summary(  # type: ignore[override]  # noqa: ARG002
        self, alpha: Optional[float] = None
    ) -> str:
        mi = self.model_info or {}
        fit = mi.get("methods_fit", [])
        failed = mi.get("methods_failed", {})
        disp = mi.get("dispersion")
        bd = mi.get("breakdown_m")

        lines = ["=" * 78, "  DID Method-Robustness Summary", "=" * 78, ""]
        if fit:
            lines.append(f"  Fitted methods: {', '.join(fit)}")
        if failed:
            lines.append(f"  Failed:         {', '.join(failed.keys())}")
        if len(fit) >= 2 and not np.isnan(self.estimate):
            lines.append(f"  Mean ATT:       {self.estimate:+.4f}")
            if disp is not None and not np.isnan(disp):
                lines.append(f"  Cross-method SD: {disp:+.4f}")
        elif len(fit) == 1 and not np.isnan(self.estimate):
            lines.append(f"  ATT:            {self.estimate:+.4f}")
        if bd is not None:
            lines.append(f"  Breakdown M*:   {bd:.4f} (CS row)")
        lines.append("")
        lines.append(did_summary_to_markdown(self))
        return "\n".join(lines)


_DEFAULT_METHODS: List[str] = ["cs", "sa", "bjs", "etwfe", "stacked"]

_Runner = Callable[
    [
        pd.DataFrame,
        str,
        str,
        str,
        str,
        Optional[List[str]],
        Optional[str],
        float,
    ],
    CausalResult,
]

_METHOD_LABELS = {
    "cs": "Callaway & Sant'Anna (2021)",
    "sa": "Sun & Abraham (2021)",
    "bjs": "Borusyak, Jaravel & Spiess (2024)",
    "etwfe": "Wooldridge (2021) ETWFE",
    "stacked": "Stacked DID (Cengiz et al. 2019)",
}

#: What each row's overall estimate averages. All five are ATTs over the
#: treated post-treatment observations; they differ in how the cells are
#: estimated, not in what is averaged.
_ESTIMANDS = {
    "cs": "ATT, treated-obs weighted (aggte simple)",
    "sa": "ATT, treated-obs weighted (fixest agg='att')",
    "bjs": "ATT, equal weight per treated obs",
    "etwfe": "ATT, treated-obs weighted",
    "stacked": "ATT, treated-obs weighted over the window",
}

#: How each method's pre-treatment coefficients are defined.
#: ``reference_period``: relative to k = -1, which is 0 by construction.
#: ``untreated_regression``: BJS leads from a TWFE regression on the
#: untreated observations, relative to the periods before the earliest
#: lead (Stata ``did_imputation, pretrends()``); k = -1 is estimated.
#: ``post_only``: no leads reported.
_LEAD_DEFINITIONS = {
    "cs": "reference_period",
    "sa": "reference_period",
    "stacked": "reference_period",
    "bjs": "untreated_regression",
    "etwfe": "post_only",
}


def _run_cs(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]],
    cluster: Optional[str],
    alpha: float,
) -> CausalResult:
    return _aggte_simple(
        _fit_cs(data, y, group, time, first_treat, controls, cluster, alpha),
        cluster,
        group,
        alpha,
    )


def _fit_cs(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]],
    cluster: Optional[str],
    alpha: float,
) -> CausalResult:
    """CS fit honouring ``cluster=``.

    CS analytic SEs cluster on the unit only; clustering on a coarser
    variable needs the multiplier bootstrap (R ``did::att_gt(clustervars=)``
    has the same restriction). Fixed seed so the summary is reproducible.
    """
    from .callaway_santanna import callaway_santanna

    extra: Dict[str, Any] = {}
    if cluster is not None and cluster != group:
        extra = dict(clustervars=[group, cluster], bstrap=True, random_state=0)
    return callaway_santanna(
        data,
        y=y,
        g=first_treat,
        t=time,
        i=group,
        x=controls,
        alpha=alpha,
        **extra,
    )


def _aggte_simple(
    cs: CausalResult, cluster: Optional[str], group: str, alpha: float
) -> CausalResult:
    from .aggte import aggte

    if cluster is not None and cluster != group:
        return aggte(cs, type="simple", alpha=alpha, bstrap=True, random_state=0)
    return aggte(cs, type="simple", alpha=alpha, bstrap=False)


def _run_sa(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]],
    cluster: Optional[str],
    alpha: float,
) -> CausalResult:
    from .sun_abraham import sun_abraham

    return sun_abraham(
        data,
        y=y,
        g=first_treat,
        t=time,
        i=group,
        covariates=controls,
        cluster=cluster,
        alpha=alpha,
        # The cell-weighted ATT (fixest summary(agg = 'att')), the estimand
        # of the other rows; 'event_time' averages the event times equally.
        aggregation="fixest_att",
        crit_dist="normal",
    )


def _run_bjs(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]],
    cluster: Optional[str],
    alpha: float,
) -> CausalResult:
    from .did_imputation import did_imputation

    return did_imputation(
        data,
        y=y,
        group=group,
        time=time,
        first_treat=first_treat,
        controls=controls,
        cluster=cluster,
        alpha=alpha,
    )


def _run_etwfe(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]],
    cluster: Optional[str],
    alpha: float,
) -> CausalResult:
    from .wooldridge_did import etwfe

    return etwfe(
        data,
        y=y,
        group=group,
        time=time,
        first_treat=first_treat,
        controls=controls,
        cluster=cluster,
        alpha=alpha,
    )


def _run_stacked(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]],
    cluster: Optional[str],
    alpha: float,
) -> CausalResult:
    from .stacked_did import stacked_did

    return stacked_did(
        data,
        y=y,
        group=group,
        time=time,
        first_treat=first_treat,
        controls=controls,
        cluster=cluster,
        alpha=alpha,
        aggregation="treated_obs",
    )


_DISPATCH: dict[str, _Runner] = {
    "cs": _run_cs,
    "sa": _run_sa,
    "bjs": _run_bjs,
    "etwfe": _run_etwfe,
    "stacked": _run_stacked,
}


def _extract(res: CausalResult) -> dict:
    """Pull (estimate, se, pvalue, ci_low, ci_high) from a CausalResult."""
    est = float(res.estimate) if res.estimate is not None else np.nan
    se = float(res.se) if res.se is not None else np.nan
    p = float(res.pvalue) if res.pvalue is not None else np.nan
    ci = res.ci if res.ci is not None else (np.nan, np.nan)
    ci_lo = float(ci[0]) if ci[0] is not None else np.nan
    ci_hi = float(ci[1]) if ci[1] is not None else np.nan
    n = int(res.n_obs) if getattr(res, "n_obs", None) else np.nan
    return dict(estimate=est, se=se, pvalue=p, ci_low=ci_lo, ci_high=ci_hi, n_obs=n)


_PATH_COLUMNS = [
    "method",
    "relative_time",
    "att",
    "se",
    "ci_lower",
    "ci_upper",
    "pvalue",
    "lead_definition",
    "reference",
]


def _bjs_path(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]],
    cluster: Optional[str],
    alpha: float,
    window: Any,
) -> CausalResult:
    """BJS refitted with horizons and Stata ``pretrends()`` leads.

    The leads need an omitted category of earlier untreated periods, so at
    most ``max_pre - 1`` of them, ``max_pre`` being the longest
    pre-treatment span of any cohort.
    """
    from .did_imputation import did_imputation

    t = pd.to_numeric(data[time], errors="coerce")
    g = pd.to_numeric(data[first_treat], errors="coerce")
    treated = g.notna() & np.isfinite(g) & (g > 0)
    max_pre = int((g[treated] - t.min()).max()) if treated.any() else 0
    max_post = int((t.max() - g[treated]).max()) if treated.any() else 0
    lo, hi = int(window[0]), int(window[1])
    pre = min(-lo, max_pre - 1) if lo < 0 else 0
    return did_imputation(
        data,
        y=y,
        group=group,
        time=time,
        first_treat=first_treat,
        controls=controls,
        cluster=cluster,
        alpha=alpha,
        horizon=list(range(0, min(hi, max_post) + 1)),
        pretrends=pre if pre >= 1 else None,
    )


def _event_study_path(name: str, res: CausalResult, window: Any) -> pd.DataFrame:
    """One method's event study on the common layout, inside ``window``."""
    if name == "etwfe":
        from .wooldridge_did import etwfe_emfx

        res = etwfe_emfx(res, type="event", alpha=res.alpha)
    es = res.event_study
    lead = _LEAD_DEFINITIONS[name]
    lo, hi = int(window[0]), int(window[1])
    rt = pd.to_numeric(es["relative_time"], errors="coerce")
    out = es.loc[(rt >= lo) & (rt <= hi)].copy()
    out["relative_time"] = rt.loc[out.index].astype(int)
    if lead == "post_only":
        out = out.loc[out["relative_time"] >= 0]
    out["reference"] = False
    if lead == "reference_period" and lo <= -1 <= hi:
        # k = -1 is the normalisation: one explicit zero row, so every
        # reference-period path is drawn through the same point.
        out = out.loc[out["relative_time"] != -1]
        ref = pd.DataFrame(
            [dict(relative_time=-1, att=0.0, se=0.0, ci_lower=0.0, ci_upper=0.0)]
        )
        ref["pvalue"] = np.nan
        ref["reference"] = True
        out = pd.concat([out, ref], ignore_index=True)
    out["method"] = name
    out["lead_definition"] = lead
    return out[_PATH_COLUMNS].sort_values("relative_time").reset_index(drop=True)


@records_call("did_summary")
@accepts_aliases(covariates="controls", cohort="first_treat", g="first_treat")
@accepts_vce(default="cluster")
def did_summary(
    data: pd.DataFrame,
    y: str,
    time: str,
    first_treat: str,
    group: str,
    methods: Union[str, List[str]] = "auto",
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    include_sensitivity: bool = False,
    verbose: bool = False,
    event_study: bool = False,
    event_window: Tuple[int, int] = (-5, 5),
) -> CausalResult:
    """
    One-call method-robustness comparison for staggered DID.

    Fits every requested estimator to the same data and returns a single
    :class:`CausalResult` whose ``detail`` attribute is a tidy comparison
    table — one row per method, columns ``(method, estimator, estimand,
    estimate, se, pvalue, ci_low, ci_high, n_obs, breakdown_m, note)``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset (long format).
    y : str
        Outcome variable.
    time : str
        Time / period variable (integer-valued).
    first_treat : str
        First-treatment period per unit; NaN (or 0) for never-treated.
    group : str
        Unit identifier.
    methods : str or list of str, default ``'auto'``
        Methods to run. Valid keys: ``'cs'``, ``'sa'``, ``'bjs'``,
        ``'etwfe'``, ``'stacked'``, or ``'all'`` / ``'auto'`` for all.
    controls : list of str, optional
        Time-varying covariates passed to methods that support them.
    cluster : str, optional
        Cluster variable for SE (defaults to ``group`` in each sub-method).
    alpha : float, default 0.05
        Significance level for confidence intervals.
    include_sensitivity : bool, default False
        If ``True`` and ``'cs'`` is among the methods fit, compute the
        Rambachan–Roth (2023) *breakdown M\\** — the largest relative
        violation of parallel trends under which the treatment effect
        is still significantly different from zero. The value is added
        to ``model_info['breakdown_m']`` and to the ``breakdown_m``
        column of ``detail`` (CS row only; other methods leave ``NaN``).
    verbose : bool, default False
        Print progress for each method.
    event_study : bool, default False
        Also collect every method's event-study path into
        ``result.event_study_paths`` (also ``model_info
        ['event_study_paths']``): columns ``method, relative_time, att,
        se, ci_lower, ci_upper, pvalue, lead_definition, reference``, and
        draw them with ``sp.did_summary_plot(result, kind='event_study')``.
        Post-treatment coefficients are comparable across methods; leads
        are not. ``lead_definition`` says how each method defines them:
        ``'reference_period'`` (CS with its universal base, SA, stacked:
        relative to k = -1, stored as an explicit zero row with
        ``reference=True``), ``'untreated_regression'`` (BJS: Stata
        ``did_imputation, pretrends()`` leads from a regression on the
        untreated observations, relative to the periods before the
        earliest lead, so k = -1 is estimated) and ``'post_only'`` (ETWFE).
        A dCDH placebo ``l`` (not among these methods) would sit at
        ``k = -(l + 1)`` on this axis. BJS is refitted with horizons for
        this, so the option costs one more imputation fit; the other paths
        are read from the fits already made.
    event_window : (int, int), default (-5, 5)
        Relative times kept in ``event_study_paths``.

    Returns
    -------
    CausalResult
        ``estimate`` : mean of successfully-fit overall ATTs.
        ``se``       : standard deviation across methods (not a standard
                       error — a crude dispersion measure).
        ``detail``   : comparison DataFrame described above.
        ``model_info`` : ``{'methods_requested': [...], 'methods_fit':
                        [...], 'methods_failed': {name: error_msg, ...}}``.

    Notes
    -----
    Every row is an ATT averaged over the treated post-treatment
    observations (the ``estimand`` column), so the rows answer one
    question and differ only in how the cells are estimated:

    - CS ``aggte(type='simple')`` weights ATT(g, t) by cohort size, i.e.
      by treated observations on a balanced panel.
    - SA is fitted with ``aggregation='fixest_att'`` (fixest
      ``summary(agg = 'att')``), BJS gives each treated observation equal
      weight, ETWFE reports the treated-observation-weighted simple ATT.
    - Stacked is fitted with ``aggregation='treated_obs'`` and averages
      over its event window (``stacked_did``'s default ``(-5, 5)``).

    Up to StatsPAI 1.40.0 the SA and stacked rows averaged the event times with
    equal weight instead. That puts a large weight on late event times
    carried by a few early cohorts, so those rows could differ from the
    others because they targeted a different average, not because the
    estimators disagreed.

    With a common estimand, differences across rows come from the
    comparisons each estimator makes (never- vs not-yet-treated controls,
    imputation vs 2x2 cells) and from sampling noise; ``se`` is the
    dispersion of the point estimates, not a test of their equality. A
    large spread is worth following up (e.g. with
    ``sp.bacon_decomposition`` or ``sp.honest_did``); agreement says
    nothing about parallel trends, which every row assumes.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=10, staggered=True, seed=0)
    >>> out = sp.did_summary(df, y='y', time='time',
    ...                      first_treat='first_treat', group='unit')
    >>> text = out.summary()  # formatted comparison table (str)
    >>> out.detail["method"].tolist()
    ['cs', 'sa', 'bjs', 'etwfe', 'stacked']
    >>> list(out.detail.columns[:5])
    ['method', 'estimator', 'estimate', 'se', 'pvalue']
    """
    if methods in ("auto", "all"):
        methods_list = list(_DEFAULT_METHODS)
    elif isinstance(methods, str):
        methods_list = [methods]
    else:
        methods_list = list(methods)

    unknown = [m for m in methods_list if m not in _DISPATCH]
    if unknown:
        raise ValueError(
            f"Unknown method(s): {unknown}. "
            f"Valid keys: {sorted(_DISPATCH)} (or 'auto' / 'all')."
        )

    # C4: validate column names up front so user typos surface as one
    # crisp ColumnNotFound (with did-you-mean) rather than 5 "method FAILED"
    # rows.
    required = [y, time, first_treat, group]
    optional = list(controls or [])
    if cluster:
        optional.append(cluster)
    missing = [c for c in required + optional if c not in data.columns]
    if missing:
        raise ColumnNotFound.from_missing(
            missing_named_columns(
                {
                    "y": y,
                    "time": time,
                    "first_treat": first_treat,
                    "group": group,
                    "controls": list(controls or []),
                    "cluster": cluster or None,
                },
                data.columns,
            ),
            list(data.columns),
            message=f"columns not found in data: {missing}. "
            f"Available columns: {list(data.columns)[:15]}"
            + (" ..." if len(data.columns) > 15 else ""),
        )

    rows: List[dict] = []
    failed: dict = {}
    fit: List[str] = []
    cs_raw = None  # raw CS result for sensitivity analysis
    fits: Dict[str, CausalResult] = {}

    # Narrow exceptions that represent "the estimator couldn't fit this
    # data" rather than a bug in user input. Other exceptions propagate.
    _expected_exc = (
        np.linalg.LinAlgError,
        ValueError,
        RuntimeError,
        ZeroDivisionError,
    )

    for name in methods_list:
        label = _METHOD_LABELS[name]
        if verbose:
            print(f"  running {name} ({label})...", flush=True)
        try:
            if name == "cs" and (include_sensitivity or event_study):
                cs_raw = _fit_cs(
                    data, y, group, time, first_treat, controls, cluster, alpha
                )
                res = _aggte_simple(cs_raw, cluster, group, alpha)
            else:
                res = _DISPATCH[name](
                    data,
                    y,
                    group,
                    time,
                    first_treat,
                    controls,
                    cluster,
                    alpha,
                )
            vals = _extract(res)
            rows.append(
                dict(
                    method=name,
                    estimator=label,
                    estimand=_ESTIMANDS[name],
                    note="",
                    **vals,
                )
            )
            fit.append(name)
            fits[name] = cs_raw if name == "cs" and cs_raw is not None else res
        except _expected_exc as exc:
            failed[name] = type(exc).__name__ + ": " + str(exc)[:160]
            rows.append(
                dict(
                    method=name,
                    estimator=label,
                    estimand=_ESTIMANDS[name],
                    estimate=np.nan,
                    se=np.nan,
                    pvalue=np.nan,
                    ci_low=np.nan,
                    ci_high=np.nan,
                    n_obs=np.nan,
                    note=f"FAILED: {type(exc).__name__}",
                )
            )

    # Optional Rambachan–Roth breakdown M*
    breakdown_m_value: Optional[float] = None
    breakdown_m_col: List[float] = [np.nan] * len(rows)
    if include_sensitivity and cs_raw is not None:
        try:
            from .honest_did import breakdown_m

            breakdown_m_value = float(breakdown_m(cs_raw, e=0, alpha=alpha))
            # Fill the CS row
            for i, r in enumerate(rows):
                if r["method"] == "cs":
                    breakdown_m_col[i] = breakdown_m_value
                    if r["note"] == "":
                        r["note"] = f"breakdown M* = {breakdown_m_value:.3f}"
                    break
        except Exception as exc:
            failed["__sensitivity__"] = (
                f"breakdown_m failed: {type(exc).__name__}: {str(exc)[:120]}"
            )

    for i, r in enumerate(rows):
        r["breakdown_m"] = breakdown_m_col[i]

    # Optional event-study paths on one relative-time axis
    paths: Optional[pd.DataFrame] = None
    if event_study:
        frames: List[pd.DataFrame] = []
        for name in fit:
            try:
                res_es = fits[name]
                if name == "bjs":
                    res_es = _bjs_path(
                        data,
                        y,
                        group,
                        time,
                        first_treat,
                        controls,
                        cluster,
                        alpha,
                        event_window,
                    )
                frames.append(_event_study_path(name, res_es, event_window))
            except _expected_exc + (AttributeError, KeyError) as exc:
                failed[f"__event_study__{name}"] = (
                    f"{type(exc).__name__}: {str(exc)[:120]}"
                )
        paths = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=_PATH_COLUMNS)
        )

    detail = pd.DataFrame(
        rows,
        columns=[
            "method",
            "estimator",
            "estimand",
            "estimate",
            "se",
            "pvalue",
            "ci_low",
            "ci_high",
            "n_obs",
            "breakdown_m",
            "note",
        ],
    )

    ests = detail.loc[detail["estimate"].notna(), "estimate"].values
    if len(ests) > 0:
        avg_est = float(np.mean(ests))
        disp = float(np.std(ests, ddof=1)) if len(ests) > 1 else np.nan
    else:
        avg_est, disp = np.nan, np.nan

    result = DIDSummaryResult(
        method="DID Method-Robustness Summary",
        estimand="Overall ATT (mean across methods)",
        estimate=avg_est,
        se=disp,
        pvalue=np.nan,
        ci=(np.nan, np.nan),
        alpha=alpha,
        n_obs=int(len(data)),
        detail=detail,
        model_info={
            "methods_requested": methods_list,
            "methods_fit": fit,
            "methods_failed": failed,
            "dispersion": disp,
            "breakdown_m": breakdown_m_value,
            "estimands": {m: _ESTIMANDS[m] for m in methods_list},
            "event_study_paths": paths,
            # H5: stable sentinel — export helpers and plots match on this
            "_did_summary_marker": True,
        },
        _citation_key="did_summary",
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Export helpers: Markdown / LaTeX from a did_summary
# ═══════════════════════════════════════════════════════════════════════


def _ensure_did_summary(result: CausalResult) -> pd.DataFrame:
    """Validate that result came from did_summary() and return its detail."""
    mi = getattr(result, "model_info", None) or {}
    if not mi.get("_did_summary_marker", False):
        raise ValueError(
            "did_summary_to_markdown / _to_latex require a CausalResult "
            "produced by sp.did_summary() (missing '_did_summary_marker' "
            "in model_info)."
        )
    det = getattr(result, "detail", None)
    if not isinstance(det, pd.DataFrame) or "estimator" not in det.columns:
        raise ValueError(
            "did_summary result has malformed detail; expected an "
            "'estimator' column."
        )
    return det


def did_summary_to_markdown(
    result: CausalResult,
    digits: int = 4,
    include_ci: bool = True,
    include_breakdown: bool = True,
) -> str:
    """
    Render a :func:`did_summary` result as a GitHub-Flavoured Markdown table.

    Columns shown (in order):
    ``Method``, ``Estimate``, ``SE``, ``95 % CI``, ``p-value``, and
    optionally ``Breakdown M*`` (when sensitivity was requested).

    Parameters
    ----------
    result : CausalResult
        Output of :func:`did_summary`.
    digits : int, default 4
        Decimal precision for numeric columns.
    include_ci : bool, default True
        Include the 95 % CI column.
    include_breakdown : bool, default True
        Include the Rambachan-Roth breakdown M* column (CS row only,
        blank for others). Ignored if sensitivity was not requested.

    Returns
    -------
    str
        Multi-line Markdown table, ready to paste into notebooks or PRs.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=120, n_periods=8, staggered=True,
    ...                 seed=0)
    >>> out = sp.did_summary(df, y='y', time='time',
    ...                      first_treat='first_treat', group='unit',
    ...                      methods=['cs', 'bjs'])
    >>> md = sp.did_summary_to_markdown(out)
    >>> isinstance(md, str)
    True
    >>> md.splitlines()[0].startswith('| Method')
    True
    """
    det = _ensure_did_summary(result)

    has_bd = include_breakdown and det["breakdown_m"].notna().any()
    lines = []
    header_cells = ["Method", "Estimate", "SE"]
    if include_ci:
        header_cells.append("95% CI")
    header_cells += ["p"]
    if has_bd:
        header_cells.append("Breakdown M*")
    header_cells.append("Notes")
    lines.append("| " + " | ".join(header_cells) + " |")
    align_cells = [":---"] + ["---:"] * (len(header_cells) - 2) + [":---"]
    lines.append("|" + "|".join(align_cells) + "|")

    fmt = f"{{:.{digits}f}}"
    for _, row in det.iterrows():
        cells = [row["estimator"]]
        if pd.isna(row["estimate"]):
            cells.append("—")
            cells.append("—")
            if include_ci:
                cells.append("—")
            cells.append("—")
            if has_bd:
                cells.append("—")
        else:
            est_str = fmt.format(row["estimate"]) + _stars(row["pvalue"])
            cells.append(est_str)
            cells.append(fmt.format(row["se"]))
            if include_ci:
                cells.append(
                    f"[{fmt.format(row['ci_low'])}, {fmt.format(row['ci_high'])}]"
                )
            cells.append(fmt.format(row["pvalue"]))
            if has_bd:
                bd = row.get("breakdown_m", np.nan)
                cells.append(fmt.format(bd) if pd.notna(bd) else "—")
        cells.append(str(row.get("note", "") or ""))
        lines.append("| " + " | ".join(cells) + " |")

    # Footer
    lines.append("")
    if getattr(result, "estimate", None) is not None and not pd.isna(result.estimate):
        lines.append(
            f"*Mean across methods = {fmt.format(result.estimate)}, "
            f"SD across methods = {fmt.format(result.se)}.*  "
            "\\* p<0.1, \\*\\* p<0.05, \\*\\*\\* p<0.01."
        )
    return "\n".join(lines)


def did_summary_to_latex(
    result: CausalResult,
    digits: int = 4,
    include_ci: bool = True,
    include_breakdown: bool = True,
    label: str = "tab:did_summary",
    caption: str = "DID method-robustness summary.",
) -> str:
    """
    Render a :func:`did_summary` result as a LaTeX
    ``booktabs`` table.

    Parameters
    ----------
    result : CausalResult
        Output of :func:`did_summary`.
    digits : int, default 4
        Decimal precision.
    include_ci : bool, default True
        Include the 95 % CI column.
    include_breakdown : bool, default True
        Include the Rambachan-Roth breakdown M* column when sensitivity
        was requested.
    label : str, default ``'tab:did_summary'``
        LaTeX label for the table.
    caption : str, default ``'DID method-robustness summary.'``
        LaTeX caption.

    Returns
    -------
    str
        Full ``\\begin{table} ... \\end{table}`` block using the
        ``booktabs`` package (``\\toprule``, ``\\midrule``, ``\\bottomrule``).

    Notes
    -----
    Requires ``\\usepackage{booktabs}`` in the LaTeX preamble.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=120, n_periods=8, staggered=True,
    ...                 seed=0)
    >>> out = sp.did_summary(df, y='y', time='time',
    ...                      first_treat='first_treat', group='unit',
    ...                      methods=['cs', 'bjs'])
    >>> tex = sp.did_summary_to_latex(out)
    >>> isinstance(tex, str)
    True
    >>> tex.splitlines()[0].startswith('\\\\begin{table}')
    True
    """
    det = _ensure_did_summary(result)
    has_bd = include_breakdown and det["breakdown_m"].notna().any()

    cols = ["l", "c", "c"]
    header = ["Method", "Estimate", "SE"]
    if include_ci:
        cols.append("c")
        header.append("95\\% CI")
    cols.append("c")
    header.append("$p$")
    if has_bd:
        cols.append("c")
        header.append("Breakdown $M^*$")

    fmt = f"{{:.{digits}f}}"
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "".join(cols) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]

    for _, row in det.iterrows():
        cells = [str(row["estimator"]).replace("&", "\\&")]
        if pd.isna(row["estimate"]):
            cells += ["---"] * (len(header) - 1)
        else:
            est_str = fmt.format(row["estimate"]) + _stars(row["pvalue"])
            cells.append(est_str)
            cells.append(fmt.format(row["se"]))
            if include_ci:
                cells.append(
                    f"[{fmt.format(row['ci_low'])}, {fmt.format(row['ci_high'])}]"
                )
            cells.append(fmt.format(row["pvalue"]))
            if has_bd:
                bd = row.get("breakdown_m", np.nan)
                cells.append(fmt.format(bd) if pd.notna(bd) else "---")
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]

    # Footnote row
    note_parts = []
    if getattr(result, "estimate", None) is not None and not pd.isna(result.estimate):
        note_parts.append(
            f"Mean across methods = {fmt.format(result.estimate)}, "
            f"SD across methods = {fmt.format(result.se)}."
        )
    note_parts.append("$^*p<0.1,\\,^{**}p<0.05,\\,^{***}p<0.01$.")
    if note_parts:
        lines.append("\\vspace{0.5ex}\\footnotesize " + " ".join(note_parts))
    lines.append("\\end{table}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  One-call DID bundle report — txt + md + tex + png in a single directory
# ═══════════════════════════════════════════════════════════════════════


@records_call("did_report")
@accepts_aliases(covariates="controls", cohort="first_treat", g="first_treat")
@accepts_vce(default="cluster")
def did_report(
    data: pd.DataFrame,
    y: str,
    time: str,
    first_treat: str,
    group: str,
    save_to: str,
    methods: Union[str, List[str]] = "auto",
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    include_sensitivity: bool = True,
    plot_sort_by: Optional[str] = "estimate",
    verbose: bool = False,
) -> CausalResult:
    """
    DID report bundle: fits selected methods and writes report artifacts.

    Writes the following files to ``save_to``:

    - ``did_summary.txt`` : text dump of ``result.summary()``.
    - ``did_summary.md``  : GitHub-Flavoured Markdown table.
    - ``did_summary.tex`` : LaTeX ``booktabs`` fragment.
    - ``did_summary.png`` : forest plot (requires matplotlib).
    - ``did_summary.json`` : detail table + model_info in JSON.

    Parameters
    ----------
    data, y, time, first_treat, group, methods, controls, cluster, alpha
        Same as :func:`did_summary`.
    save_to : str
        Directory path. Created if it does not exist.
    include_sensitivity : bool, default True
        Whether to run Rambachan-Roth breakdown M*. Defaults to ``True``
        in ``did_report`` (vs ``False`` in ``did_summary``) because a
        report is expected to be comprehensive.
    plot_sort_by : {'estimate', None}, default 'estimate'
        Sort the forest plot by estimate ascending.
    verbose : bool, default False
        Print progress lines.

    Returns
    -------
    CausalResult
        The underlying :func:`did_summary` output. All side-effect files
        are written to ``save_to`` as a bundle.

    Examples
    --------
    >>> import tempfile
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=120, n_periods=8, staggered=True,
    ...                 seed=0)
    >>> out_dir = tempfile.mkdtemp()
    >>> res = sp.did_report(df, y='y', time='time',
    ...                     first_treat='first_treat', group='unit',
    ...                     save_to=out_dir, methods=['cs', 'bjs'],
    ...                     include_sensitivity=False)
    >>> import os
    >>> 'did_summary.md' in os.listdir(out_dir)
    True
    """
    import json
    from pathlib import Path

    out_dir = Path(save_to)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[did_report] fitting methods into {out_dir} ...", flush=True)

    result = did_summary(
        data,
        y=y,
        time=time,
        first_treat=first_treat,
        group=group,
        methods=methods,
        controls=controls,
        cluster=cluster,
        alpha=alpha,
        include_sensitivity=include_sensitivity,
        verbose=verbose,
    )
    detail = _ensure_did_summary(result)

    # TXT
    (out_dir / "did_summary.txt").write_text(result.summary(), encoding="utf-8")

    # Markdown
    (out_dir / "did_summary.md").write_text(
        did_summary_to_markdown(result),
        encoding="utf-8",
    )

    # LaTeX
    (out_dir / "did_summary.tex").write_text(
        did_summary_to_latex(result),
        encoding="utf-8",
    )

    # PNG (best-effort — skip if matplotlib unavailable)
    try:
        from .plots import did_summary_plot

        fig, _ = did_summary_plot(result, sort_by=plot_sort_by)
        fig.savefig(out_dir / "did_summary.png", dpi=150, bbox_inches="tight")
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
    except ImportError:
        if verbose:
            print("[did_report] matplotlib missing — skipping PNG", flush=True)

    # JSON (detail + model_info)
    payload = {
        "method": result.method,
        "estimate": None if pd.isna(result.estimate) else float(result.estimate),
        "cross_method_sd": None if pd.isna(result.se) else float(result.se),
        "detail": detail.replace({np.nan: None}).to_dict(orient="records"),
        "model_info": {
            k: (list(v) if isinstance(v, (set, tuple)) else v)
            for k, v in result.model_info.items()
        },
    }
    (out_dir / "did_summary.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )

    if verbose:
        print(f"[did_report] wrote bundle to {out_dir}", flush=True)

    return result
