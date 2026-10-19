"""Stata's ``collapse``: a dataset of summary statistics by group.

This module is the one implementation: ``sp.collapse`` and the
``collapse`` data step of ``sp.stata`` both call :func:`collapse_frame`.
The statistics follow ``[D] collapse``: missing values are skipped, a group
whose values are all missing gets missing (``count`` and ``sum`` give 0),
the rows with a missing by-value form a group of their own, listed last.
With weights, see :func:`collapse_frame`.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..exceptions import ColumnNotFound, MethodIncompatibility

__all__ = ["COLLAPSE_STATS", "collapse", "collapse_frame", "frequency_rows"]

#: collapse statistics -> the pandas reduction (missing values are skipped,
#: and an all-missing group gives missing, as in Stata)
COLLAPSE_STATS = {
    "mean": "mean",
    "median": "median",
    "p50": "median",
    "sd": "std",
    "sum": "sum",
    "rawsum": "rawsum",
    "count": "count",
    "max": "max",
    "min": "min",
    # pandas' first / last skip missing values, which is Stata's firstnm /
    # lastnm; Stata's first / last take the row whatever it holds
    "firstnm": "first",
    "lastnm": "last",
    "first": "_row_first",
    "last": "_row_last",
}

_WEIGHT_KINDS = {
    "aweight": "aw",
    "aw": "aw",
    "fweight": "fw",
    "fw": "fw",
    "pweight": "pw",
    "pw": "pw",
    "iweight": "iw",
    "iw": "iw",
}


def _refuse(message: str, hint: str) -> MethodIncompatibility:
    return MethodIncompatibility(message, recovery_hint=hint)


def frequency_rows(
    data: pd.DataFrame, keep: np.ndarray, w: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The rows a frequency weight stands for, and the rows themselves
    (those with a positive weight)."""
    held = keep & ~np.isnan(w)
    if np.any(w[held] < 0):
        raise _refuse("collapse: negative weights", "Weights must be >= 0.")
    if np.any(w[held] != np.round(w[held])):
        raise _refuse(
            "may not use noninteger frequency weights",
            "Use weight_type='aweight' for non-integer weights.",
        )
    held &= w > 0
    rows = data.loc[held]
    counts = w[held].astype(np.int64)
    return rows.loc[rows.index.repeat(counts)], rows


def _row_end(how: str) -> Any:
    position = 0 if how == "_row_first" else -1
    return lambda s: s.iloc[position] if len(s) else np.nan


def collapse_frame(
    data: pd.DataFrame,
    targets: List[Tuple[str, str, str]],
    by: List[str],
    keep: np.ndarray,
    weight: Optional[Tuple[str, np.ndarray]] = None,
    check_groups: Optional[Callable[[pd.DataFrame], None]] = None,
) -> pd.DataFrame:
    """The collapsed data, in double precision.

    Parameters
    ----------
    data : DataFrame
    targets : list of (new, source, how)
        ``how`` is a value of :data:`COLLAPSE_STATS`.
    by : list of str
    keep : bool array
        The rows that take part (Stata's ``if`` / ``in``).
    weight : (kind, values), optional
        ``kind`` is ``'aw'`` / ``'fw'`` / ``'pw'`` / ``'iw'``. A frequency
        weight stands for that many identical rows (rows with a zero or
        missing weight drop out; ``rawsum`` sums the rows themselves).
        Otherwise, with weights w on the rows where the variable is
        observed (n of them): the mean is ``sum(w x) / sum(w)``; ``sum`` is
        ``sum(w x)``, for analytic weights after scaling them to add up to
        n; ``count`` is ``sum(w)``, and n for analytic weights; ``sd`` uses
        n - 1 with the analytic-weight variance; the median is the weighted
        one.
    check_groups : callable, optional
        Called with the rows that are grouped before grouping them (the
        session refuses by-variables with extended missing values).
    """
    raw = None
    if weight is not None and weight[0] == "fw":
        frame, raw = frequency_rows(data, keep, weight[1])
    elif weight is not None:
        return _collapse_weighted(data, targets, by, keep, weight, check_groups)
    else:
        frame = data.loc[keep]
    if raw is None:
        raw = frame
    if by:
        # the rows with a missing by-value are a group too, listed last
        if check_groups is not None:
            check_groups(frame)
        grouped = frame.groupby(by, sort=True, dropna=False)
        cols = {}
        for new, source, how in targets:
            if how == "rawsum":
                cols[new] = raw.groupby(by, sort=True, dropna=False)[source].sum(
                    min_count=0
                )
            elif how == "sum":
                cols[new] = grouped[source].sum(min_count=0)
            elif how.startswith("_row_"):
                cols[new] = grouped[source].agg(_row_end(how))
            else:
                cols[new] = getattr(grouped[source], how)()
        return pd.DataFrame(cols).reset_index()

    def whole(source: str, how: str) -> Any:
        if how == "sum":
            return frame[source].sum()
        if how == "rawsum":
            return raw[source].sum()
        if how.startswith("_row_"):
            return _row_end(how)(frame[source])
        if how in ("first", "last"):
            end = "_row_first" if how == "first" else "_row_last"
            return _row_end(end)(frame[source].dropna())
        return getattr(frame[source], how)()

    return pd.DataFrame({new: [whole(source, how)] for new, source, how in targets})


def _collapse_weighted(
    data: pd.DataFrame,
    targets: List[Tuple[str, str, str]],
    by: List[str],
    keep: np.ndarray,
    weight: Tuple[str, np.ndarray],
    check_groups: Optional[Callable[[pd.DataFrame], None]],
) -> pd.DataFrame:
    from ..output.sumstats import weighted_percentile

    kind, w_all = weight
    ok = keep & ~np.isnan(w_all) & (w_all != 0)
    if np.any(w_all[ok] < 0):
        raise _refuse("collapse: negative weights", "Weights must be >= 0.")
    frame = data.loc[ok].copy()
    frame["__w"] = w_all[ok]
    if by and check_groups is not None:
        check_groups(frame)

    def one(part: pd.DataFrame, source: str, how: str) -> float:
        x = part[source].to_numpy(dtype=float, na_value=np.nan)
        w = part["__w"].to_numpy(dtype=float)
        held = ~np.isnan(x)
        x, w = x[held], w[held]
        n = float(x.size)
        if how == "count":
            return n if kind == "aw" else float(w.sum())
        if n == 0:
            return np.nan
        if how in ("min", "max"):
            return float(x.min() if how == "min" else x.max())
        total = float(w.sum())
        mean = float(np.sum(w * x) / total)
        if how == "mean":
            return mean
        if how == "sum":
            return mean * n if kind == "aw" else float(np.sum(w * x))
        if how == "median":
            return weighted_percentile(x, w, 50.0)
        if how == "std":
            if kind in ("pw", "iw"):
                raise _refuse(
                    "collapse: (sd) is not allowed with pweights",
                    "Use aweights or fweights for a standard deviation.",
                )
            size = total if kind == "fw" else n
            if size <= 1:
                return np.nan
            return float(
                np.sqrt(np.sum(w * (x - mean) ** 2) / total * size / (size - 1))
            )
        raise _refuse(
            f"collapse: ({how}) with weights is not implemented",
            "Weighted collapse supports mean, median, sd, sum, count, min " "and max.",
        )

    if by:
        rows = []
        for level, part in frame.groupby(by, sort=True, dropna=False):
            level = level if isinstance(level, tuple) else (level,)
            row = dict(zip(by, level))
            for new, source, how in targets:
                row[new] = one(part, source, how)
            rows.append(row)
        return pd.DataFrame(rows, columns=by + [t[0] for t in targets])
    return pd.DataFrame(
        {new: [one(frame, source, how)] for new, source, how in targets}
    )


_NAME = re.compile(r"[A-Za-z_]\w*\Z")


def _stat_name(stat: str) -> str:
    key = str(stat).strip().lower().strip("()")
    if key not in COLLAPSE_STATS:
        raise _refuse(
            f"sp.collapse: statistic {stat!r} is not implemented",
            "Use one of " + ", ".join(sorted(COLLAPSE_STATS)) + ".",
        )
    return key


def _targets_from_text(text: str) -> List[Tuple[str, str, str]]:
    """``"(mean) x y (sum) tot=z"`` -> targets, as Stata reads it."""
    stat = "mean"
    out = []
    for tok in re.sub(r"\s*=\s*", "=", text).split():
        m = re.fullmatch(r"\((\w+)\)", tok)
        if m:
            stat = _stat_name(m.group(1))
            continue
        new, _, source = tok.rpartition("=")
        new = new or source
        if not _NAME.match(new) or not source:
            raise _refuse(
                f"sp.collapse: cannot read {tok!r}",
                "Write the statistics as '(mean) x y (sum) total=z'.",
            )
        out.append((new, source, stat))
    return out


def collapse(
    data: pd.DataFrame,
    stats: Union[str, Mapping[str, Union[str, Sequence[str]]]],
    by: Optional[Union[str, Sequence[str]]] = None,
    weights: Optional[str] = None,
    weight_type: str = "aweight",
    cw: bool = False,
) -> pd.DataFrame:
    """Make a dataset of summary statistics, as Stata's ``collapse`` does.

    ``sp.collapse(df, "(mean) wage (sum) hours (count) n=wage",
    by="state")`` is ``collapse (mean) wage (sum) hours (count) n=wage,
    by(state)``.

    Parameters
    ----------
    data : DataFrame
        Not modified. Filter rows first for Stata's ``if`` / ``in``.
    stats : str or dict
        Either Stata's syntax, ``"(mean) x y (sd) sdx=x"`` (the statistic
        applies to the names after it until the next one; the default is
        ``mean``; ``new=old`` names the result), or a dict
        ``{new_name: stat}`` (the source is ``new_name``) or ``{new_name:
        (stat, source)}``. Statistics: ``mean``, ``median`` / ``p50``,
        ``sd``, ``sum``, ``rawsum``, ``count``, ``min``, ``max``,
        ``first`` / ``last`` (the first / last row of the group, missing or
        not), ``firstnm`` / ``lastnm`` (the first / last non-missing value).
    by : str or list of str, optional
        The groups. The rows with a missing value of a by-variable form a
        group of their own, listed last. Without ``by``: one row.
    weights : str, optional
        A weight column.
    weight_type : {'aweight', 'fweight', 'pweight', 'iweight'}, default 'aweight'
        Stata's weight types. Frequency weights stand for that many
        identical rows (non-integer ones are refused); with the others the
        mean is ``sum(w x) / sum(w)``, ``sum`` is ``sum(w x)`` (for
        analytic weights after scaling them to sum to the number of
        observations), ``count`` is ``sum(w)`` (the number of observations
        for analytic weights). Rows with a zero or missing weight drop out.
    cw : bool, default False
        Stata's ``cw``: use only the rows where every source variable is
        observed. Otherwise each statistic uses all rows where its own
        variable is observed.

    Returns
    -------
    DataFrame
        The by-variables (sorted), then one column per statistic, in the
        order given. Values are double precision (Stata stores a mean of a
        ``float`` variable in single precision; nothing here does).

    Raises
    ------
    MethodIncompatibility
        An unknown statistic; a result name used twice or equal to a
        by-variable; negative weights; non-integer frequency weights;
        ``sd`` with pweights / iweights (Stata refuses it); ``rawsum`` with
        weights other than frequency weights, or ``first`` / ``last`` with
        weights (not implemented).

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"g": [1, 1, 2, 2], "x": [1.0, 3.0, 5.0, None]})
    >>> sp.collapse(df, "(mean) x (count) n=x", by="g").values.tolist()
    [[1.0, 2.0, 2.0], [2.0, 5.0, 1.0]]
    >>> sp.collapse(df, {"x_sd": ("sd", "x")}, by="g")["x_sd"].tolist()
    [1.4142135623730951, nan]

    References
    ----------
    StataCorp. Stata 18 Data Management Reference Manual, ``[D] collapse``.
    """
    if isinstance(stats, str):
        targets = _targets_from_text(stats)
    elif isinstance(stats, Mapping):
        targets = []
        for new, spec in stats.items():
            if isinstance(spec, str):
                stat, source = spec, str(new)
            else:
                spec = list(spec)
                if len(spec) != 2:
                    raise _refuse(
                        f"sp.collapse: cannot read {new!r}: {spec!r}",
                        "Give {new_name: stat} or {new_name: (stat, source)}.",
                    )
                stat, source = spec[0], spec[1]
            targets.append((str(new), str(source), _stat_name(stat)))
    else:
        raise _refuse(
            "sp.collapse: stats must be a string or a dict",
            "e.g. stats='(mean) x (sum) total=y' or {'total': ('sum', 'y')}.",
        )
    if not targets:
        raise _refuse(
            "sp.collapse needs at least one variable",
            "e.g. stats='(mean) x'.",
        )
    by_list = [by] if isinstance(by, str) else list(by or [])
    sources = [t[1] for t in targets]
    missing = [c for c in dict.fromkeys(sources + by_list) if c not in data.columns]
    if weights is not None and weights not in data.columns:
        missing.append(weights)
    if missing:
        raise ColumnNotFound.from_missing(
            missing, list(data.columns), context="sp.collapse"
        )
    names = [t[0] for t in targets]
    if len(set(names)) != len(names) or set(names) & set(by_list):
        raise _refuse(
            "collapse: a result name is used twice",
            "Name each result differently from the others and from the "
            "by-variables (new=old).",
        )
    weight = None
    if weights is not None:
        kind = _WEIGHT_KINDS.get(str(weight_type).lower())
        if kind is None:
            raise _refuse(
                f"sp.collapse: weight_type {weight_type!r} is not one of "
                "aweight, fweight, pweight, iweight",
                "Pass weight_type='aweight' (Stata's default for collapse).",
            )
        if not pd.api.types.is_numeric_dtype(data[weights]):
            raise _refuse(
                "collapse: a weight must be numeric", "Pass a numeric column."
            )
        if kind != "fw" and any(t[2] == "rawsum" for t in targets):
            raise _refuse(
                "collapse: (rawsum) with weights is not implemented",
                "Use (rawsum) with frequency weights or without weights.",
            )
        weight = (kind, data[weights].to_numpy(dtype=float, na_value=np.nan))
    keep = np.ones(len(data), dtype=bool)
    if cw:
        keep &= data[list(dict.fromkeys(sources))].notna().all(axis=1).to_numpy()
    targets = [(new, source, COLLAPSE_STATS[stat]) for new, source, stat in targets]
    return collapse_frame(data, targets, by_list, keep, weight)
