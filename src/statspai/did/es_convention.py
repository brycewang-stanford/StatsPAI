"""Event-study reference conventions across DiD estimators.

Two DiD implementations can agree on every post-treatment coefficient and
still plot different event studies, because the *pre*-treatment
coefficients are a separate construction and the construction is not
shared.  Roth (2026) shows the consequence: applied to the same
non-staggered panel with no treatment effect and a linear violation of
parallel trends, a dynamic TWFE event study is a straight line, the
Callaway--Sant'Anna default in R shows a kink at the treatment date, and
the Borusyak--Jaravel--Spiess Stata default shows a jump.  None of the
three is wrong; they answer with different reference periods.  The
practical consequence is that the visual heuristic applied researchers
learned on TWFE plots -- "a break at zero is evidence of a real effect" --
does not transfer, and neither do sensitivity analyses that compare
pre-treatment with post-treatment coefficients.

This module makes the convention an inspectable object instead of a
footnote.  :func:`event_study_convention` reports what a given estimator
constructs; :func:`compare_event_study_conventions` runs the estimators on
one panel and measures how far each path departs from the dynamic TWFE
benchmark, separating a common vertical shift (harmless) from an
asymmetry between the two halves of the path (not harmless).

References
----------
Roth, J. (2026).  "Interpreting Event-Studies from Recent
Difference-in-Differences Methods."  arXiv:2401.12309.
[@roth2026interpreting]

Callaway, B. and Sant'Anna, P. H. C. (2021).  "Difference-in-Differences
with Multiple Time Periods."  *Journal of Econometrics*, 225(2), 200-230.
[@callaway2021difference]

Borusyak, K., Jaravel, X. and Spiess, J. (2024).  "Revisiting Event-Study
Designs: Robust and Efficient Estimation."  *Review of Economic Studies*,
91(6), 3253-3285.  [@borusyak2024revisiting]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility

__all__ = [
    "EventStudyConventionResult",
    "compare_event_study_conventions",
    "event_study_convention",
]


# --------------------------------------------------------------------- #
# Static registry
# --------------------------------------------------------------------- #
# ``twfe_comparable`` means: in a NON-STAGGERED design, does this path
# coincide with the dynamic TWFE event study up to one common vertical
# shift?  Every entry marked True or False here is checked numerically by
# tests/test_did_event_study_conventions.py on Roth's (2026) design, so
# the column is evidence rather than documentation.
_REGISTRY: Dict[str, Dict[str, Any]] = {
    "event_study": {
        "estimator": "event_study",
        "label": "dynamic TWFE",
        "pre_construction": "difference against the omitted period (default -1)",
        "post_construction": "difference against the omitted period (default -1)",
        "symmetric": True,
        "twfe_comparable": True,
        "reference_implementation": "the TWFE benchmark itself",
    },
    "callaway_santanna[base_period=universal]": {
        "estimator": "callaway_santanna",
        "label": "Callaway--Sant'Anna, universal base period",
        "pre_construction": "long difference against the period before treatment",
        "post_construction": "long difference against the period before treatment",
        "symmetric": True,
        "twfe_comparable": True,
        "reference_implementation": "R did::att_gt(base_period='universal')",
        "note": (
            "StatsPAI's default. R did defaults to base_period='varying', so "
            "the two packages plot different pre-trends out of the box even "
            "though their post-treatment coefficients agree."
        ),
    },
    "callaway_santanna[base_period=varying]": {
        "estimator": "callaway_santanna",
        "label": "Callaway--Sant'Anna, varying base period",
        "pre_construction": "short difference between consecutive periods",
        "post_construction": "long difference against the period before treatment",
        "symmetric": False,
        "twfe_comparable": False,
        "reference_implementation": "R did::att_gt() default; Stata csdid default",
        "note": (
            "Produces the kink at the treatment date documented by Roth "
            "(2026). Better at exposing anticipation, worse for judging a "
            "long-run parallel-trends violation."
        ),
    },
    "sun_abraham": {
        "estimator": "sun_abraham",
        "label": "Sun--Abraham interaction-weighted",
        "pre_construction": "saturated TWFE interactions against the omitted period",
        "post_construction": "saturated TWFE interactions against the omitted period",
        "symmetric": True,
        "twfe_comparable": True,
        "reference_implementation": "R fixest::sunab",
    },
    "wooldridge_did": {
        "estimator": "wooldridge_did",
        "label": "extended TWFE (Wooldridge)",
        "pre_construction": "saturated TWFE interactions against the omitted period",
        "post_construction": "saturated TWFE interactions against the omitted period",
        "symmetric": True,
        "twfe_comparable": True,
        "reference_implementation": "R etwfe; Stata jwdid",
    },
    "did_imputation[pretrend_method=bjs]": {
        "estimator": "did_imputation",
        "label": "imputation, BJS pre-trend convention",
        "pre_construction": (
            "auxiliary dynamic TWFE on untreated observations, referenced to "
            "the pooled earlier pre-periods"
        ),
        "post_construction": (
            "imputation residuals, referenced to the average of all "
            "pre-treatment periods"
        ),
        "symmetric": False,
        "twfe_comparable": False,
        "reference_implementation": "Stata did_imputation, pretrends(k)",
        "note": (
            "Produces the jump at the treatment date documented by Roth "
            "(2026). The leads remain a valid test of parallel pre-trends; "
            "they are not a plot to eyeball next to the lags."
        ),
    },
    "did_imputation[pretrend_method=in-sample]": {
        "estimator": "did_imputation",
        "label": "imputation, in-sample residual pre-trends",
        "pre_construction": "mean in-sample imputation residual at each lead",
        "post_construction": (
            "imputation residuals, referenced to the average of all "
            "pre-treatment periods"
        ),
        "symmetric": False,
        "twfe_comparable": False,
        "reference_implementation": "R fect; R did2s",
        "note": (
            "Attenuated by the untreated unit share N0/N in the "
            "non-staggered case, so it understates pre-trends."
        ),
    },
    "did_imputation[pretrend_method=symmetric]": {
        "estimator": "did_imputation",
        "label": "imputation, Roth symmetric pre-trends",
        "pre_construction": "pre-treatment mean as the common reference",
        "post_construction": "pre-treatment mean as the common reference",
        "symmetric": True,
        "twfe_comparable": True,
        "reference_implementation": "Roth (2026) beta-hat^{BJS,new}",
        "note": (
            "Pre-treatment coefficients average to zero by construction; "
            "read their movement, not their level."
        ),
    },
    "gardner_did": {
        "estimator": "gardner_did",
        "label": "two-stage DiD (Gardner)",
        "pre_construction": "mean in-sample imputation residual at each lead",
        "post_construction": (
            "imputation residuals, referenced to the average of all "
            "pre-treatment periods"
        ),
        "symmetric": False,
        "twfe_comparable": False,
        "reference_implementation": "R did2s",
    },
}

_REGISTRY_COLUMNS = [
    "key",
    "estimator",
    "label",
    "pre_construction",
    "post_construction",
    "symmetric",
    "twfe_comparable",
    "reference_implementation",
    "note",
]


def event_study_convention(
    estimator: Optional[str] = None,
) -> "pd.DataFrame | Dict[str, Any]":
    """Report how an estimator builds its event-study reference periods.

    Parameters
    ----------
    estimator : str, optional
        Either a registry key such as
        ``'callaway_santanna[base_period=varying]'`` or a bare estimator
        name such as ``'did_imputation'``.  A bare name returns every
        option-specific row for that estimator.  ``None`` returns the whole
        registry.

    Returns
    -------
    DataFrame or dict
        A dict for an exact key, otherwise a DataFrame with one row per
        convention.

    Examples
    --------
    >>> import statspai as sp
    >>> table = sp.event_study_convention()
    >>> bool((~table["twfe_comparable"]).any())
    True
    >>> sp.event_study_convention(
    ...     "callaway_santanna[base_period=varying]"
    ... )["symmetric"]
    False
    """
    if estimator is None:
        rows = [{"key": k, **v} for k, v in _REGISTRY.items()]
        return pd.DataFrame(rows).reindex(columns=_REGISTRY_COLUMNS)
    if estimator in _REGISTRY:
        return {"key": estimator, **_REGISTRY[estimator]}
    matches = [
        {"key": k, **v} for k, v in _REGISTRY.items() if v["estimator"] == estimator
    ]
    if not matches:
        raise KeyError(
            f"No event-study convention recorded for {estimator!r}. Known "
            f"estimators: {sorted({v['estimator'] for v in _REGISTRY.values()})}. "
            "Call sp.event_study_convention() with no argument for the full "
            "registry."
        )
    return pd.DataFrame(matches).reindex(columns=_REGISTRY_COLUMNS)


# --------------------------------------------------------------------- #
# Empirical comparison
# --------------------------------------------------------------------- #
@dataclass
class EventStudyConventionResult:
    """Paths and convention diagnostics for one panel.

    Attributes
    ----------
    paths : DataFrame
        Long format: ``key``, ``relative_time``, ``att``.
    table : DataFrame
        One row per estimator with the registry fields plus ``shift_pre``,
        ``shift_post``, ``asymmetry``, ``shape_gap`` (split into
        ``shape_gap_pre`` / ``shape_gap_post``) and ``matches_twfe``.
    reference : str
        Registry key of the benchmark path (dynamic TWFE).
    tolerance : float
        Scale-aware threshold used for ``matches_twfe``.
    """

    paths: pd.DataFrame
    table: pd.DataFrame
    reference: str = "event_study"
    tolerance: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable report."""
        lines = [
            "Event-study reference conventions",
            "=" * 62,
            f"Benchmark: {self.reference} (dynamic TWFE), "
            f"tolerance {self.tolerance:.2e}",
            "",
            f"{'estimator':<46}{'asym':>9}{'shape':>9}  ok",
        ]
        for _, row in self.table.iterrows():
            lines.append(
                f"{row['key']:<46}{row['asymmetry']:>9.3f}"
                f"{row['shape_gap']:>9.3f}  {'y' if row['matches_twfe'] else 'n'}"
            )
        lines += [
            "",
            "asym  = mean(post gap) - mean(pre gap) vs TWFE; a jump at zero.",
            "shape = worst within-half departure from TWFE after removing",
            "        that half's shift; a kink or a rescaling.",
            "A row with ok=n plots a break at the treatment date that the",
            "TWFE path does not, on data where both are estimating the same",
            "thing. See Roth (2026).",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "tolerance": self.tolerance,
            "table": self.table.to_dict(orient="records"),
            "paths": self.paths.to_dict(orient="records"),
            "diagnostics": dict(self.diagnostics),
        }


def _normalise_path(obj: Any) -> pd.DataFrame:
    """Pull ``(relative_time, att)`` out of the shapes estimators return."""
    if isinstance(obj, pd.DataFrame):
        frame = obj
    elif isinstance(obj, dict):
        frame = pd.DataFrame(obj)
    else:  # pragma: no cover - guarded by the callers
        raise TypeError(f"Cannot read an event-study path from {type(obj)!r}.")

    time_col = next(
        (
            c
            for c in ("relative_time", "rel_time", "event_time", "horizon", "e")
            if c in frame.columns
        ),
        None,
    )
    att_col = next(
        (c for c in ("att", "estimate", "coef", "coefficient") if c in frame.columns),
        None,
    )
    if time_col is None or att_col is None:
        raise KeyError(
            "Event-study frame lacks a recognised relative-time / estimate "
            f"column pair; saw {list(frame.columns)}."
        )
    out = frame[[time_col, att_col]].copy()
    out.columns = ["relative_time", "att"]
    out["relative_time"] = out["relative_time"].astype(int)
    out["att"] = out["att"].astype(float)
    return out.groupby("relative_time", as_index=False)["att"].mean()


def _default_runners(
    data: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    first_treat: str,
    window: Tuple[int, int],
    cluster: Optional[str],
) -> Dict[str, Callable[[], pd.DataFrame]]:
    from . import callaway_santanna, did_imputation, event_study, sun_abraham
    from .aggte import aggte

    horizon = list(range(window[0], window[1] + 1))
    nan_treat = "__es_conv_first_treat_nan__"
    frame = data.copy()
    frame[nan_treat] = frame[first_treat].where(frame[first_treat] > 0, np.nan)

    def _twfe() -> pd.DataFrame:
        res = event_study(
            frame,
            y=y,
            treat_time=nan_treat,
            time=time,
            unit=unit,
            window=window,
            cluster=cluster or unit,
        )
        return _normalise_path(res.model_info["event_study"])

    def _cs(base_period: str) -> pd.DataFrame:
        res = callaway_santanna(
            data,
            y=y,
            g=first_treat,
            t=time,
            i=unit,
            estimator="reg",
            base_period=base_period,
        )
        tidy = aggte(res, type="dynamic").tidy()
        rows = tidy[tidy["type"] == "event_study"].copy()
        rows["relative_time"] = (
            rows["term"].str.replace("event_", "", regex=False).astype(int)
        )
        return _normalise_path(rows.rename(columns={"estimate": "att"}))

    def _sunab() -> pd.DataFrame:
        res = sun_abraham(
            data,
            y=y,
            g=first_treat,
            t=time,
            i=unit,
            event_window=window,
            cluster=cluster,
        )
        return _normalise_path(res.model_info["event_study"])

    # The BJS lead regression needs one pre-treatment period left over as
    # the omitted category, so it cannot be asked for the earliest lead
    # the panel supports. Dropping it here keeps the comparison runnable;
    # the row's n_relative_times records that it is one shorter.
    rel_available = pd.to_numeric(data[time], errors="coerce") - pd.to_numeric(
        data[first_treat], errors="coerce"
    ).where(pd.to_numeric(data[first_treat], errors="coerce") > 0)
    rel_min = (
        int(np.floor(rel_available.min())) if rel_available.notna().any() else None
    )

    def _imputation(method: str) -> pd.DataFrame:
        this = horizon
        if method == "bjs" and rel_min is not None:
            this = [k for k in horizon if k != rel_min]
        res = did_imputation(
            data,
            y=y,
            group=unit,
            time=time,
            first_treat=first_treat,
            horizon=this,
            cluster=cluster,
            pretrend_method=method,
        )
        return _normalise_path(res.model_info["event_study"])

    return {
        "event_study": _twfe,
        "callaway_santanna[base_period=universal]": lambda: _cs("universal"),
        "callaway_santanna[base_period=varying]": lambda: _cs("varying"),
        "sun_abraham": _sunab,
        "did_imputation[pretrend_method=bjs]": lambda: _imputation("bjs"),
        "did_imputation[pretrend_method=in-sample]": lambda: _imputation("in-sample"),
        "did_imputation[pretrend_method=symmetric]": lambda: _imputation("symmetric"),
    }


def compare_event_study_conventions(
    data: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    first_treat: str,
    *,
    estimators: Optional[Sequence[str]] = None,
    window: Optional[Tuple[int, int]] = None,
    cluster: Optional[str] = None,
    tolerance: Optional[float] = None,
) -> EventStudyConventionResult:
    """Measure how each estimator's event-study path departs from TWFE.

    Runs the requested estimators on one panel, aligns their event-study
    paths on shared relative times, and decomposes the difference from the
    dynamic TWFE benchmark into a common vertical shift within each half
    of the path and a residual.  A symmetric estimator differs from TWFE
    by one shift and nothing else; an asymmetric one shows a different
    shift before and after the treatment date, which is the jump or kink
    Roth (2026) documents.

    The diagnostic is only interpretable in a **non-staggered** design,
    because that is where every estimator here targets the same object and
    Roth's analysis applies, so a staggered panel raises rather than
    returning a number that cannot be read.

    Parameters
    ----------
    data : DataFrame
        Panel with unit, time, outcome and cohort columns.
    y, unit, time, first_treat : str
        Column names.  ``first_treat`` holds the first treated period, with
        ``0`` (or a non-positive value) for never-treated units.
    estimators : sequence of str, optional
        Registry keys to run.  Defaults to every estimator with a runner.
    window : (int, int), optional
        Relative-time window.  Defaults to the widest the panel supports.
    cluster : str, optional
        Cluster column; defaults to ``unit``.
    tolerance : float, optional
        Threshold for ``matches_twfe``.  Defaults to
        ``1e-6 * max(1, max|beta_twfe|)``, which keeps the verdict
        scale-free.

    Returns
    -------
    EventStudyConventionResult

    Raises
    ------
    MethodIncompatibility
        If the panel is staggered, or has no never-treated units.

    Examples
    --------
    >>> import statspai as sp
    >>> panel = sp.datasets  # doctest: +SKIP
    >>> res = sp.compare_event_study_conventions(  # doctest: +SKIP
    ...     df, y="y", unit="unit", time="time", first_treat="g"
    ... )
    >>> print(res.summary())  # doctest: +SKIP
    """
    cohorts = pd.to_numeric(data[first_treat], errors="coerce")
    treated_cohorts = sorted({float(g) for g in cohorts.dropna().unique() if g > 0})
    if len(treated_cohorts) == 0:
        raise MethodIncompatibility(
            f"No treated cohort found in '{first_treat}'.",
            recovery_hint=(
                "Encode the first treated period in first_treat; " "0 = never treated."
            ),
        )
    if len(treated_cohorts) > 1:
        raise MethodIncompatibility(
            "compare_event_study_conventions is defined for non-staggered "
            f"designs; '{first_treat}' has {len(treated_cohorts)} treated "
            "cohorts. With staggered timing the estimators no longer target "
            "the same object, so a gap against the TWFE path mixes the "
            "reference-period convention with the forbidden-comparison "
            "problem and cannot be read as either.",
            recovery_hint=(
                "Restrict to one cohort plus the never-treated units, or "
                "inspect the conventions without running them via "
                "sp.event_study_convention()."
            ),
        )
    if not (cohorts.isna() | (cohorts <= 0)).any():
        raise MethodIncompatibility(
            "No never-treated units: the TWFE benchmark path is not defined.",
            recovery_hint="Include never-treated units in the panel.",
        )

    g0 = treated_cohorts[0]
    periods = pd.to_numeric(data[time], errors="coerce")
    if window is None:
        window = (
            int(np.floor(periods.min() - g0)),
            int(np.floor(periods.max() - g0)),
        )

    runners = _default_runners(data, y, unit, time, first_treat, window, cluster)
    keys = list(estimators) if estimators is not None else list(runners)
    unknown = [k for k in keys if k not in runners]
    if unknown:
        raise KeyError(
            f"No runner for {unknown}. Available: {sorted(runners)}. "
            "sp.event_study_convention() documents estimators without a "
            "runner here."
        )
    if "event_study" not in keys:
        keys = ["event_study"] + keys

    paths: Dict[str, pd.DataFrame] = {k: runners[k]() for k in keys}
    bench = paths["event_study"].set_index("relative_time")["att"]
    scale = float(np.max(np.abs(bench.to_numpy()))) if len(bench) else 1.0
    tol = float(tolerance) if tolerance is not None else 1e-6 * max(1.0, scale)

    long_rows: List[pd.DataFrame] = []
    table_rows: List[Dict[str, Any]] = []
    for key in keys:
        path = paths[key].copy()
        path.insert(0, "key", key)
        long_rows.append(path)

        series = paths[key].set_index("relative_time")["att"]
        shared = series.index.intersection(bench.index)
        gap = (series.loc[shared] - bench.loc[shared]).astype(float)
        pre = gap[gap.index < 0]
        post = gap[gap.index >= 0]
        shift_pre = float(pre.mean()) if len(pre) else float("nan")
        shift_post = float(post.mean()) if len(post) else float("nan")
        halves = {}
        for name, half, shift in (("pre", pre, shift_pre), ("post", post, shift_post)):
            halves[name] = (
                float(np.max(np.abs(half.to_numpy() - shift))) if len(half) else 0.0
            )
        shape = max(halves["pre"], halves["post"])
        asymmetry = shift_post - shift_pre
        record = dict(_REGISTRY.get(key, {"estimator": key, "label": key}))
        record.update(
            {
                "key": key,
                "n_relative_times": int(len(shared)),
                "shift_pre": shift_pre,
                "shift_post": shift_post,
                "asymmetry": asymmetry,
                "shape_gap": shape,
                "shape_gap_pre": halves["pre"],
                "shape_gap_post": halves["post"],
                "matches_twfe": bool(
                    np.isfinite(asymmetry) and abs(asymmetry) <= tol and shape <= tol
                ),
            }
        )
        table_rows.append(record)

    table = pd.DataFrame(table_rows)
    ordered = _REGISTRY_COLUMNS + [
        "n_relative_times",
        "shift_pre",
        "shift_post",
        "asymmetry",
        "shape_gap",
        "shape_gap_pre",
        "shape_gap_post",
        "matches_twfe",
    ]
    table = table.reindex(columns=[c for c in ordered if c in table.columns])

    # A registry row that claims TWFE comparability and then fails the
    # numerical check is a documentation bug, and the point of running
    # this is to catch it rather than to reprint it.
    for _, row in table.iterrows():
        claimed = row.get("twfe_comparable")
        if claimed is not None and not pd.isna(claimed):
            if bool(claimed) != bool(row["matches_twfe"]):
                warnings.warn(
                    f"event-study convention registry says "
                    f"twfe_comparable={bool(claimed)} for {row['key']!r}, but "
                    f"on this panel asymmetry={row['asymmetry']:.3g} and "
                    f"shape_gap={row['shape_gap']:.3g} against a tolerance of "
                    f"{tol:.3g}. Treat the registry entry as unverified here.",
                    UserWarning,
                    stacklevel=2,
                )

    return EventStudyConventionResult(
        paths=pd.concat(long_rows, ignore_index=True),
        table=table,
        reference="event_study",
        tolerance=tol,
        diagnostics={
            "treated_cohort": g0,
            "window": tuple(window),
            "benchmark_scale": scale,
        },
    )
