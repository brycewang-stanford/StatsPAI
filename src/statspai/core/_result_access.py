"""Named accessors for the tables a DiD fit stores in ``model_info``.

The staggered DiD estimators keep their ATT(g,t) cells, event study and
aggregations in different places: CS puts ATT(g,t) in ``detail`` and the
event study in ``model_info['event_study']``; ``gardner_did`` stores the
event study as a dict of ``'D_k+h'`` labels; ``etwfe`` keeps cohort by
relative-time cells there instead; ``aggte`` puts its table in ``detail``.
Scripts had to guess (``model_info.get(..., res.detail)``, DiD replication
report 2026-10-21, item 8).

:class:`DiDAccessorsMixin` gives :class:`~statspai.core.results.CausalResult`
three read-only properties that resolve those places once and return one
column layout:

* ``result.attgt``: ATT(g,t) as ``group, time, relative_time, att, se,
  ci_lower, ci_upper, pvalue``.
* ``result.event_study``: one row per relative time, ``relative_time, att,
  se, ci_lower, ci_upper, pvalue`` first.
* ``result.aggregations``: ``{'overall', 'dynamic', 'group', 'calendar'}``
  (the ones the result holds) mapped to tables.

They only read what the fit stored; nothing is refitted. A missing table
raises :class:`ResultAccessorUnavailable`, which is an ``AttributeError``
(so ``hasattr`` and ``getattr(result, name, None)`` keep working) and a
:class:`~statspai.exceptions.MethodIncompatibility` carrying a
``recovery_hint``. The keys each estimator writes are listed in
``docs/reference/did_result_contents.md``.

The mixin lives in its own module so the estimation path does not run it;
the properties execute only when read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility

__all__ = ["ResultAccessorUnavailable", "DiDAccessorsMixin"]

#: Leading columns of every table the accessors return.
EVENT_STUDY_LEAD = ("relative_time", "att", "se", "ci_lower", "ci_upper", "pvalue")
ATTGT_LEAD = ("group", "time") + EVENT_STUDY_LEAD

#: Spellings in use across estimators, mapped to the accessor's names.
_RENAME = {
    "rel_time": "relative_time",
    "event_time": "relative_time",
    "estimate": "att",
    "ci_low": "ci_lower",
    "ci_high": "ci_upper",
}

_ATTGT_SOURCES = (
    "sp.callaway_santanna (and sp.did with a staggered design), sp.etwfe, "
    "sp.wooldridge_did"
)
_EVENT_STUDY_SOURCES = (
    "sp.callaway_santanna, sp.aggte(type='dynamic'), sp.sun_abraham, "
    "sp.did_imputation(horizon=...), sp.gardner_did(event_study=True), "
    "sp.stacked_did, sp.event_study, sp.etwfe_emfx(type='event')"
)


class ResultAccessorUnavailable(MethodIncompatibility, AttributeError):
    """A named table (``attgt`` / ``event_study`` / ``aggregations``) the
    result does not hold.

    Subclasses ``AttributeError`` so ``hasattr(result, 'attgt')`` is
    ``False`` and ``getattr(result, 'attgt', None)`` returns ``None``.
    """

    code = "result_accessor_unavailable"


def _normal_fill(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Add ``ci_lower`` / ``ci_upper`` / ``pvalue`` where a table lacks them.

    Normal reference, the convention of every estimator whose table is
    stored without them (``gardner_did``'s own intervals are normal; the
    etwfe cells are what ``etwfe_emfx`` reports with a normal interval).
    Existing columns are never overwritten.
    """
    from scipy import stats

    att = pd.to_numeric(frame["att"], errors="coerce").to_numpy(dtype=float)
    se = pd.to_numeric(frame["se"], errors="coerce").to_numpy(dtype=float)
    if not {"ci_lower", "ci_upper"} <= set(frame.columns):
        z = float(stats.norm.ppf(1.0 - float(alpha) / 2.0))
        frame["ci_lower"] = att - z * se
        frame["ci_upper"] = att + z * se
    if "pvalue" not in frame.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, att / se, np.nan)
        frame["pvalue"] = 2.0 * stats.norm.sf(np.abs(t))
    return frame


def _ordered(frame: pd.DataFrame, lead: tuple) -> pd.DataFrame:
    head = [c for c in lead if c in frame.columns]
    rest = [c for c in frame.columns if c not in head and not str(c).startswith("_")]
    return frame[head + rest].reset_index(drop=True)


def _standardise(frame: pd.DataFrame, alpha: float, lead: tuple) -> pd.DataFrame:
    out = frame.rename(
        columns={k: v for k, v in _RENAME.items() if v not in frame.columns}
    ).copy()
    return _ordered(_normal_fill(out, alpha), lead)


class DiDAccessorsMixin:
    """``attgt`` / ``event_study`` / ``aggregations`` on a result object.

    Expects ``model_info`` (dict), ``detail`` (DataFrame or None),
    ``alpha``, ``estimand``, ``estimate``, ``se``, ``ci``, ``pvalue``.
    """

    if TYPE_CHECKING:  # not runtime annotations: the result schema reads those
        model_info: Dict[str, Any]
        detail: Optional[pd.DataFrame]
        alpha: float

    # -- resolution helpers (no side effects) ----------------------------
    def _attgt_or_none(self) -> Optional[pd.DataFrame]:
        det = self.detail
        if isinstance(det, pd.DataFrame) and {"group", "time", "att", "se"} <= set(
            det.columns
        ):
            out = det.copy()
            if "relative_time" not in out.columns:
                try:
                    out["relative_time"] = out["time"] - out["group"]
                except TypeError:
                    pass
            return _standardise(out, self.alpha, ATTGT_LEAD)
        es = (self.model_info or {}).get("event_study")
        if isinstance(es, pd.DataFrame) and "cohort" in es.columns:
            # etwfe / wooldridge_did: one row per cohort x relative time.
            out = es.rename(columns={"cohort": "group"}).copy()
            if "period" in out.columns:
                out = out.rename(columns={"period": "time"})
            elif "rel_time" in out.columns:
                out["time"] = out["group"] + out["rel_time"]
            return _standardise(out, self.alpha, ATTGT_LEAD)
        return None

    def _event_study_or_none(self) -> Optional[pd.DataFrame]:
        es = (self.model_info or {}).get("event_study")
        if isinstance(es, dict):
            # gardner_did: {'horizon', 'coef', 'se', 'ci', ...} keyed 'D_k+h'.
            from ..did.es_inference import _es_frame

            frame = _es_frame(self)
            if frame is None or frame.empty:
                return None
            ci = es.get("ci") or {}
            labels = frame["_label"].tolist()
            if all(lab in ci for lab in labels):
                frame["ci_lower"] = [float(ci[lab][0]) for lab in labels]
                frame["ci_upper"] = [float(ci[lab][1]) for lab in labels]
            return _standardise(frame, self.alpha, EVENT_STUDY_LEAD)
        if isinstance(es, pd.DataFrame) and "cohort" not in es.columns:
            cols = set(es.columns)
            if cols & {"relative_time", "rel_time", "event_time"} and cols & {
                "att",
                "estimate",
            }:
                if "se" in cols:
                    return _standardise(es, self.alpha, EVENT_STUDY_LEAD)
        det = self.detail
        if (
            isinstance(det, pd.DataFrame)
            and {"relative_time", "att", "se"} <= set(det.columns)
            and not {"group", "time"} <= set(det.columns)
        ):
            # aggte(type='dynamic') keeps the event study in ``detail``.
            return _standardise(det, self.alpha, EVENT_STUDY_LEAD)
        return None

    # -- public accessors ------------------------------------------------
    @property
    def attgt(self) -> pd.DataFrame:
        """ATT(g,t) cells: ``group, time, relative_time, att, se, ci_lower,
        ci_upper, pvalue`` (then any estimator-specific columns).

        Available on fits of ``sp.callaway_santanna`` (and ``sp.did`` when
        it routes there), ``sp.etwfe`` and ``sp.wooldridge_did``. An
        aggregated result (``sp.aggte``) no longer carries the cells; read
        them from the fit it was built from.

        Raises
        ------
        ResultAccessorUnavailable
            (an ``AttributeError``) when the result holds no ATT(g,t) table.
        """
        out = self._attgt_or_none()
        if out is None:
            raise ResultAccessorUnavailable(
                f"This {getattr(self, 'method', 'result')!r} result has no "
                "ATT(g,t) table.",
                recovery_hint=(
                    f"ATT(g,t) cells come from {_ATTGT_SOURCES}. After "
                    "sp.aggte, read .attgt on the fit passed to aggte."
                ),
                diagnostics={"accessor": "attgt"},
                alternative_functions=["sp.callaway_santanna", "sp.etwfe"],
            )
        return out

    @property
    def event_study(self) -> pd.DataFrame:
        """Event-study table, one row per relative time: ``relative_time,
        att, se, ci_lower, ci_upper, pvalue`` (then estimator extras).

        Read from ``model_info['event_study']`` (a frame, or
        ``gardner_did``'s ``'D_k+h'`` dict) or, for ``sp.aggte(type=
        'dynamic')``, from ``detail``. Intervals and p-values are the
        estimator's own; where a table was stored without them they are
        filled with the normal reference at the fit's ``alpha``.

        ``etwfe`` stores cohort-by-relative-time cells, not an event study;
        aggregate them with ``sp.etwfe_emfx(result, type='event')``.

        Raises
        ------
        ResultAccessorUnavailable
            (an ``AttributeError``) when the result holds no event study.
        """
        out = self._event_study_or_none()
        if out is None:
            es = (self.model_info or {}).get("event_study")
            if isinstance(es, pd.DataFrame) and "cohort" in es.columns:
                hint = (
                    "This fit stores cohort x relative-time cells (see "
                    ".attgt); aggregate them with "
                    "sp.etwfe_emfx(result, type='event')."
                )
            else:
                hint = (
                    f"An event study comes from {_EVENT_STUDY_SOURCES}; "
                    "some need a horizon / event_study= argument."
                )
            raise ResultAccessorUnavailable(
                f"This {getattr(self, 'method', 'result')!r} result has no "
                "event-study table.",
                recovery_hint=hint,
                diagnostics={"accessor": "event_study"},
                alternative_functions=["sp.aggte", "sp.etwfe_emfx"],
            )
        return out

    @property
    def aggregations(self) -> Dict[str, pd.DataFrame]:
        """The aggregations this result holds, as tables.

        Keys, when present:

        * ``'overall'``: the headline (one row: ``estimand, att, se,
          ci_lower, ci_upper, pvalue``). For a raw CS fit this is the
          simple aggregation; for ``sp.aggte`` it is that call's overall
          summary (``model_info['aggregation']`` names the type).
        * ``'dynamic'``: the event study (same table as ``.event_study``).
        * ``'group'``: one row per cohort (``sp.aggte(type='group')``, or
          the per-cohort ATTs of ``etwfe`` / ``wooldridge_did``).
        * ``'calendar'``: one row per period (``sp.aggte(type='calendar')``).

        Aggregations the fit did not compute are absent, not computed
        here: call ``sp.aggte(result, type=...)`` for the rest.

        Raises
        ------
        ResultAccessorUnavailable
            (an ``AttributeError``) when the result holds no table beyond
            its headline (a 2x2 fit, a non-DiD result).
        """
        info = self.model_info or {}
        out: Dict[str, pd.DataFrame] = {}
        es = self._event_study_or_none()
        if es is not None:
            out["dynamic"] = es
        det = self.detail if isinstance(self.detail, pd.DataFrame) else pd.DataFrame()
        cols = set(det.columns)
        agg_type = info.get("aggregation")
        if {"att", "se"} <= cols:
            if "group" in cols and "time" not in cols:
                out["group"] = _standardise(
                    det, self.alpha, ("group",) + EVENT_STUDY_LEAD[1:]
                )
            elif (
                "cohort" in cols and "time" not in cols and "relative_time" not in cols
            ):
                g = det.rename(columns={"cohort": "group"})
                out["group"] = _standardise(
                    g, self.alpha, ("group",) + EVENT_STUDY_LEAD[1:]
                )
            elif agg_type == "calendar" and "time" in cols and "group" not in cols:
                out["calendar"] = _standardise(
                    det, self.alpha, ("time",) + EVENT_STUDY_LEAD[1:]
                )
        if not out and self._attgt_or_none() is None:
            raise ResultAccessorUnavailable(
                f"This {getattr(self, 'method', 'result')!r} result holds no "
                "aggregation beyond its headline estimate.",
                recovery_hint=(
                    "Read .estimate / .tidy() for the headline. Aggregations "
                    "of ATT(g,t) come from sp.aggte(result, type='dynamic' | "
                    "'group' | 'calendar') on a sp.callaway_santanna fit, or "
                    "sp.etwfe_emfx on a sp.etwfe fit."
                ),
                diagnostics={"accessor": "aggregations"},
                alternative_functions=["sp.aggte", "sp.etwfe_emfx"],
            )
        overall = {
            "estimand": getattr(self, "estimand", None),
            "att": getattr(self, "estimate", np.nan),
            "se": getattr(self, "se", np.nan),
            "ci_lower": (getattr(self, "ci", None) or (np.nan, np.nan))[0],
            "ci_upper": (getattr(self, "ci", None) or (np.nan, np.nan))[1],
            "pvalue": getattr(self, "pvalue", np.nan),
        }
        if agg_type is not None:
            overall["aggregation"] = agg_type
        result: Dict[str, pd.DataFrame] = {"overall": pd.DataFrame([overall])}
        result.update(out)
        return result
