"""Heuristic study-design detection from a raw DataFrame.

``sp.detect_design(data, **hints)`` answers the agent's first question
on receiving unfamiliar data: *what kind of dataset is this?* —
cross-section, panel, or something with an obvious RD running variable.

Distinct from siblings:

* :func:`sp.recommend` — needs a research question (outcome, treatment)
  and recommends an *estimator*; ``detect_design`` only inspects shape.
* :func:`sp.check_identification` — diagnoses a *specific* declared
  design; ``detect_design`` decides which design is plausible at all.

The function is intentionally heuristic — it reports a confidence,
ranks alternatives, and surfaces every column-role candidate it
considered, so an agent can override with hints (``unit=...`` /
``time=...`` / ``running_var=...``) when the heuristic is wrong.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


# Heuristic thresholds (literature-free — these are pragmatic
# defaults, deliberately permissive to avoid silently rejecting real
# panels with mild imbalance / unusual shapes).

#: Minimum (unit_count × period_count) / n_obs ratio for a candidate
#: (unit, time) pair to count as a panel. Balanced panels have
#: ratio ≈ 1; we allow some imbalance.
_PANEL_FILL_MIN = 0.30

#: A column with > this fraction of unique values is too granular to
#: be a unit identifier (probably a continuous outcome).
_UNIT_UNIQUE_FRAC_MAX = 0.95

#: A column with ≤ this many unique values is too coarse to be a unit
#: identifier (probably a categorical covariate).
_UNIT_UNIQUE_MIN = 2

#: A panel time dimension typically has 2–500 distinct periods.
_TIME_UNIQUE_MIN = 2
_TIME_UNIQUE_MAX = 500

#: For RD running-variable detection: the candidate must be numeric,
#: continuous (≥ this many unique values), and concentrated near the
#: candidate cutoff with a non-trivial mass on each side.
_RD_RUNNING_UNIQUE_MIN = 30
_RD_SIDE_MASS_MIN = 0.10  # ≥ 10 % of obs on each side of cutoff

#: Cap on candidate columns considered for unit / time roles. On a
#: 50-column DataFrame an unbounded sweep is O(n_cols² × n_rows) on
#: the duplicate-pair check inside ``_score_panel_pair`` — slow on
#: large data. The top-K cap keeps the heuristic O(K² × n_rows)
#: regardless of input width.
_PAIR_CANDIDATE_CAP = 10


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _is_id_like(series: pd.Series, n: int) -> bool:
    """Return True if a column's value distribution looks like a
    discrete unit / cohort identifier (not a continuous regressor).

    NaN handling: ``dropna=True`` so NaN values are *not* counted as
    a distinct level. A column with 50 firm IDs and many missing
    rows still has cardinality 50, not 51 — keeps it consistent with
    the pair-scoring counts below.
    """
    n_unique = series.nunique(dropna=True)
    if n_unique < _UNIT_UNIQUE_MIN:
        return False
    if n_unique / max(n, 1) > _UNIT_UNIQUE_FRAC_MAX:
        return False
    return True


def _is_time_like(series: pd.Series) -> bool:
    """Return True if the column could plausibly be a panel-time
    dimension. NaN dropped from cardinality count for the same
    reason as :func:`_is_id_like`."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    n_unique = series.nunique(dropna=True)
    if n_unique < _TIME_UNIQUE_MIN or n_unique > _TIME_UNIQUE_MAX:
        return False
    # Numeric / integer / string time codes all admissible.
    return True


def _score_panel_pair(data: pd.DataFrame, unit_col: str, time_col: str
                       ) -> Optional[Dict[str, Any]]:
    """Score a candidate ``(unit, time)`` pair as a panel.

    Returns a dict with the score and shape stats, or ``None`` if the
    pair is incompatible with a panel layout (zero coverage, all
    duplicate ``(unit, time)`` rows, etc.).
    """
    n = len(data)
    if n == 0:
        return None
    unit = data[unit_col]
    time = data[time_col]
    # Drop NaN consistently with ``_is_id_like`` / ``_is_time_like`` —
    # otherwise NaN-polluted columns inflate ``expected_balanced`` and
    # depress the fill ratio, classifying real panels as cross-sections.
    n_units = unit.nunique(dropna=True)
    n_periods = time.nunique(dropna=True)
    if n_units < _UNIT_UNIQUE_MIN or n_periods < _TIME_UNIQUE_MIN:
        return None

    # Fill ratio: how close to a balanced panel of n_units × n_periods.
    expected_balanced = n_units * n_periods
    fill_ratio = n / expected_balanced if expected_balanced else 0.0
    fill_ratio = min(fill_ratio, 1.0)  # clip overcounted (duplicate
                                       # (unit,time) rows)

    # Duplicate-row penalty: a clean panel has at most one row per
    # (unit, time). Multiple rows usually mean the columns aren't
    # really an id pair (or the data is long-format with multiple
    # outcomes per cell — also fine, but with weaker confidence).
    dup_pairs = data.duplicated(subset=[unit_col, time_col]).sum()
    dup_penalty = dup_pairs / n

    # Score: blend fill ratio (closer to 1 → higher) with low duplicate
    # penalty. Range roughly [0, 1]; thresholded by _PANEL_FILL_MIN.
    score = fill_ratio * (1 - dup_penalty)

    return {
        "unit": unit_col,
        "time": time_col,
        "n_units": int(n_units),
        "n_periods": int(n_periods),
        "n_obs": int(n),
        "fill_ratio": round(float(fill_ratio), 3),
        "duplicate_pair_share": round(float(dup_penalty), 3),
        "is_balanced": bool(n == expected_balanced and dup_pairs == 0),
        "score": round(float(score), 3),
    }


def _detect_panel_candidates(data: pd.DataFrame,
                              hint_unit: Optional[str] = None,
                              hint_time: Optional[str] = None,
                              ) -> List[Dict[str, Any]]:
    """Enumerate plausible ``(unit, time)`` pairs ranked by score.

    Skips columns that look like outcomes (too unique) or constants
    (too coarse). Time columns either have a datetime dtype or fall
    in the [2, 500]-distinct range.
    """
    n = len(data)
    cols = list(data.columns)

    if hint_unit is not None and hint_unit not in cols:
        hint_unit = None
    if hint_time is not None and hint_time not in cols:
        hint_time = None

    # Restrict candidate sets when hints are supplied. Without hints
    # we cap each side at ``_PAIR_CANDIDATE_CAP`` so a 50-column
    # DataFrame doesn't trigger 2 500 ``data.duplicated()`` passes —
    # ranking by cardinality picks the most plausible candidates
    # first (distinct ID-like values for units, fewer distinct levels
    # for time).
    if hint_unit:
        unit_candidates = [hint_unit]
    else:
        ids = [c for c in cols if _is_id_like(data[c], n)]
        # Prefer columns with cardinality typical of a unit: prefer
        # mid-range cardinality over extreme values. A simple ranking
        # by descending nunique gets us closest to "50-firm panels"
        # vs "thousands-of-row IDs that are really row indices".
        ids.sort(key=lambda c: data[c].nunique(dropna=True),
                 reverse=True)
        unit_candidates = ids[:_PAIR_CANDIDATE_CAP]

    if hint_time:
        time_candidates = [hint_time]
    else:
        times = [c for c in cols if _is_time_like(data[c])]
        # Time columns typically have few distinct values; rank
        # ascending by cardinality so 5-period panels beat 200-level
        # categoricals.
        times.sort(key=lambda c: data[c].nunique(dropna=True))
        time_candidates = times[:_PAIR_CANDIDATE_CAP]

    out: List[Dict[str, Any]] = []
    for u in unit_candidates:
        for t in time_candidates:
            if u == t:
                continue
            scored = _score_panel_pair(data, u, t)
            if scored is None:
                continue
            if scored["score"] < _PANEL_FILL_MIN:
                continue
            out.append(scored)

    # Sort: highest score first; on ties, prefer the pair with more
    # units than periods (panels conventionally have N >> T).
    out.sort(key=lambda r: (-r["score"],
                              -(r["n_units"] - r["n_periods"])))

    # Deduplicate symmetric pairs: ``(firm_id, year)`` and
    # ``(year, firm_id)`` shape-score identically when both columns
    # pass the id / time filter, but they encode different role
    # assignments. Keep only the highest-scoring orientation per
    # column-pair.
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for r in out:
        key = frozenset((r["unit"], r["time"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def _detect_rd_running_var(data: pd.DataFrame,
                            hint_running: Optional[str] = None,
                            hint_cutoff: Optional[float] = None,
                            ) -> Optional[Dict[str, Any]]:
    """Look for a numeric column that could be an RD running variable.

    Heuristic: continuous numeric column (≥ ``_RD_RUNNING_UNIQUE_MIN``
    distinct values), with the candidate cutoff lying near the median
    or — if hinted — at ``hint_cutoff``. We DO NOT auto-discover the
    cutoff value; reviewers expect it to be a specific institutional
    threshold (age 65, income $50K, …) and guessing is worse than
    saying "you'll need to specify".
    """
    n = len(data)
    if n == 0:
        return None

    cols = [hint_running] if (
        hint_running and hint_running in data.columns
    ) else [
        c for c in data.columns
        if pd.api.types.is_numeric_dtype(data[c])
        and data[c].nunique(dropna=False) >= _RD_RUNNING_UNIQUE_MIN
    ]

    best: Optional[Dict[str, Any]] = None
    for c in cols:
        s = data[c].dropna()
        if len(s) == 0:
            continue
        cutoff = (hint_cutoff
                  if hint_cutoff is not None
                  else float(s.median()))
        left = (s < cutoff).mean()
        right = (s >= cutoff).mean()
        if left < _RD_SIDE_MASS_MIN or right < _RD_SIDE_MASS_MIN:
            continue
        # Score is the smaller of the two side masses — closer to 0.5
        # is more "RD-shaped". Prefer the column whose mass is most
        # evenly split when no cutoff is hinted.
        score = float(min(left, right))
        cand = {
            "running_var": c,
            "cutoff": float(cutoff),
            "left_share": round(float(left), 3),
            "right_share": round(float(right), 3),
            "n_unique": int(data[c].nunique(dropna=False)),
            "score": round(score, 3),
            "cutoff_inferred": hint_cutoff is None,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def detect_design(data: pd.DataFrame,
                   *,
                   unit: Optional[str] = None,
                   time: Optional[str] = None,
                   running_var: Optional[str] = None,
                   cutoff: Optional[float] = None,
                   ) -> Dict[str, Any]:
    """Detect the most plausible study design from a DataFrame.

    Heuristic — never definitive. Returns ranked candidates so the
    agent can override with hints (``unit=...`` / ``time=...`` /
    ``running_var=...`` / ``cutoff=...``) when the auto-detection is
    wrong.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to inspect. Must have ≥ 1 row.
    unit : str, optional
        Column the caller has already identified as the unit ID. Skips
        unit-detection and pins this column.
    time : str, optional
        Column the caller has already identified as the time dimension.
    running_var : str, optional
        Force this numeric column to be evaluated as an RD running
        variable.
    cutoff : float, optional
        RD cutoff value (the heuristic does NOT auto-discover this).

    Returns
    -------
    dict
        JSON-safe payload with keys:

        - ``design`` (str) — ``"panel"`` / ``"rd"`` / ``"cross_section"``
        - ``confidence`` (float in [0, 1])
        - ``identified`` (dict[str, str|float]) — column-role
          assignments for the chosen design
        - ``candidates`` (list[dict]) — every alternative considered,
          each with its own ``design`` / ``confidence`` / details. Use
          this to override when the top pick is wrong.
        - ``n_obs`` (int) — sample size
        - ``columns`` (list[str]) — input column names

    Examples
    --------
    Panel data:

    >>> df = pd.DataFrame({
    ...     'firm_id': np.repeat(range(50), 10),
    ...     'year': np.tile(range(2010, 2020), 50),
    ...     'sales': np.random.randn(500),
    ... })
    >>> sp.detect_design(df)['design']
    'panel'

    Cross-section:

    >>> df = pd.DataFrame({'x': np.random.randn(200),
    ...                    'y': np.random.randn(200)})
    >>> sp.detect_design(df)['design']
    'cross_section'

    See Also
    --------
    sp.recommend :
        Method advisor — pair this with a declared research question
        (outcome / treatment) and it recommends an estimator.
    sp.check_identification :
        Design-level diagnostics for an *already declared* design.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data must be a pandas DataFrame; got {type(data).__name__}")
    n = len(data)
    if n == 0:
        return {
            "design": "cross_section",
            "confidence": 0.0,
            "identified": {},
            "candidates": [],
            "n_obs": 0,
            "columns": list(data.columns),
        }

    panel_candidates = _detect_panel_candidates(
        data, hint_unit=unit, hint_time=time)
    rd_candidate = _detect_rd_running_var(
        data, hint_running=running_var, hint_cutoff=cutoff)

    candidates: List[Dict[str, Any]] = []

    # RD candidate. Without an explicit hint (``running_var`` or
    # ``cutoff``), median-split detection alone is far too noisy:
    # ANY continuous variable on a roughly-symmetric distribution
    # passes the side-mass check, so unhinted RD would beat
    # cross-section on pure noise. Cap the unhinted confidence
    # at 0.30 so it never wins the top slot — but keep the
    # candidate in the list so an agent who knows the cutoff can
    # promote it via a follow-up call.
    rd_hinted = bool(running_var) or cutoff is not None
    rd_score = 0.0
    if rd_candidate is not None:
        if rd_hinted:
            rd_score = min(1.0, rd_candidate["score"] + 0.4)
        else:
            rd_score = min(0.30, rd_candidate["score"])
        candidates.append({
            "design": "rd",
            "confidence": round(rd_score, 3),
            **rd_candidate,
        })

    panel_score = 0.0
    if panel_candidates:
        top = panel_candidates[0]
        panel_score = top["score"]
        if unit or time:
            panel_score = min(1.0, panel_score + 0.2)
        for p in panel_candidates:
            candidates.append({
                "design": "panel",
                "confidence": round(min(1.0, p["score"]
                                          + (0.2 if (unit or time)
                                             else 0.0)), 3),
                **p,
            })

    # Cross-section is always a fallback candidate.
    cs_score = max(0.0, 1.0 - max(panel_score, rd_score))
    candidates.append({
        "design": "cross_section",
        "confidence": round(cs_score, 3),
        "n_obs": n,
        "n_columns": len(data.columns),
    })

    # Pick the winner: highest confidence score across all candidates.
    candidates.sort(key=lambda c: -c["confidence"])
    winner = candidates[0]

    identified: Dict[str, Any] = {}
    if winner["design"] == "panel":
        identified["unit"] = winner["unit"]
        identified["time"] = winner["time"]
    elif winner["design"] == "rd":
        identified["running_var"] = winner["running_var"]
        identified["cutoff"] = winner["cutoff"]

    return {
        "design": winner["design"],
        "confidence": winner["confidence"],
        "identified": identified,
        "candidates": candidates,
        "n_obs": n,
        "columns": list(data.columns),
    }


__all__ = ["detect_design"]
