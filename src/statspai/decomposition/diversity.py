"""Species-diversity indices for record-level or count data.

Empirical work on biodiversity outcomes — the effect of a policy on the
bird community of a county-month, say — starts by turning raw sighting
records into a panel outcome. That step is a modelling choice, not
bookkeeping: Shannon entropy, species richness and Pielou evenness answer
different questions, and a treatment that leaves Shannon flat can still
move richness and evenness in opposite directions.

``sp.diversity_index`` computes them together, from either long-format
records or a site-by-species count matrix, and returns them as a tidy
frame ready to merge onto a panel.

References
----------
Shannon, C. E. (1948). "A Mathematical Theory of Communication."
*Bell System Technical Journal*, 27(3), 379-423.
doi:10.1002/j.1538-7305.1948.tb01338.x [@shannon1948mathematical]

Simpson, E. H. (1949). "Measurement of Diversity." *Nature*, 163, 688.
doi:10.1038/163688a0 [@simpson1949measurement]

Pielou, E. C. (1966). "The measurement of diversity in different types of
biological collections." *Journal of Theoretical Biology*, 13, 131-144.
doi:10.1016/0022-5193(66)90013-0 [@pielou1966measurement]

Hill, M. O. (1973). "Diversity and Evenness: A Unifying Notation and Its
Consequences." *Ecology*, 54(2), 427-432. doi:10.2307/1934352
[@hill1973diversity]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["diversity_index", "DIVERSITY_INDICES"]

#: Indices ``sp.diversity_index`` can return.
DIVERSITY_INDICES = (
    "shannon",
    "richness",
    "pielou",
    "simpson",
    "gini_simpson",
    "inv_simpson",
    "hill",
)


def _index_from_counts(
    counts: np.ndarray, index: str, q: float, base: Optional[float]
) -> float:
    """Compute one diversity index from a vector of species abundances."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts / total
    S = int(counts.size)

    if index == "richness":
        return float(S)
    if index == "shannon":
        h = -float(np.sum(p * np.log(p)))
        return h if base is None else h / float(np.log(base))
    if index == "pielou":
        # Shannon divided by its maximum, log(S). A one-species site has
        # no evenness to speak of; NaN is the honest answer, not 1.0.
        if S <= 1:
            return np.nan
        return float(-np.sum(p * np.log(p)) / np.log(S))
    if index == "simpson":
        # Simpson's concentration: probability two draws share a species.
        return float(np.sum(p**2))
    if index == "gini_simpson":
        return float(1.0 - np.sum(p**2))
    if index == "inv_simpson":
        conc = float(np.sum(p**2))
        return float(1.0 / conc) if conc > 0 else np.nan
    if index == "hill":
        # Hill number of order q: the "effective number of species".
        if np.isclose(q, 1.0):
            return float(np.exp(-np.sum(p * np.log(p))))
        if np.isclose(q, 0.0):
            return float(S)
        return float(np.sum(p**q) ** (1.0 / (1.0 - q)))
    raise MethodIncompatibility(  # pragma: no cover - guarded by caller
        f"Unknown diversity index {index!r}.",
        diagnostics={"supported": list(DIVERSITY_INDICES)},
    )


def diversity_index(
    data: Union[pd.DataFrame, np.ndarray],
    species: Optional[str] = None,
    count: Optional[str] = None,
    by: Optional[Union[str, Sequence[str]]] = None,
    index: Union[str, Sequence[str]] = "shannon",
    q: float = 1.0,
    base: Optional[float] = None,
    min_records: int = 0,
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Species-diversity indices from record-level or count data.

    Computes Shannon, richness, Pielou evenness, the Simpson family and
    Hill numbers, grouped onto a panel index.

    Parameters
    ----------
    data : DataFrame or ndarray
        Either long-format records (one row per sighting, or one row per
        species-site with a ``count`` column) or a site-by-species matrix
        of abundances. A 1-D array is treated as one site's abundances.
    species : str, optional
        Species-identifier column. Required for long-format input.
    count : str, optional
        Abundance column. Omit when each row is one individual/record,
        in which case rows are counted.
    by : str or list of str, optional
        Grouping keys — typically the panel index, e.g.
        ``by=["county", "ym"]``. Without it the whole frame is one site.
    index : str or list of str, default ``"shannon"``
        Any of ``shannon``, ``richness``, ``pielou``, ``simpson``,
        ``gini_simpson``, ``inv_simpson``, ``hill``, or ``"all"``.

        - ``shannon`` — ``-sum p_i log p_i`` (Shannon 1948).
        - ``richness`` — number of species observed.
        - ``pielou`` — Shannon / log(S), evenness in ``[0, 1]``
          (Pielou 1966); ``NaN`` for single-species sites.
        - ``simpson`` — concentration ``sum p_i^2`` (Simpson 1949);
          ``gini_simpson`` is ``1 - sum p_i^2`` and ``inv_simpson`` is
          its reciprocal.
        - ``hill`` — Hill number of order ``q`` (Hill 1973), the
          effective number of species; ``q=0`` is richness, ``q=1`` is
          ``exp(shannon)``, ``q=2`` is ``inv_simpson``.
    q : float, default 1.0
        Order of the Hill number. Ignored by the other indices.
    base : float, optional
        Logarithm base for Shannon. ``None`` (default) uses natural logs,
        the convention in the ecological literature; pass ``2`` for bits.
    min_records : int, default 0
        Groups with fewer than this many records return ``NaN`` instead
        of an estimate. Diversity indices are badly biased downward in
        small samples, so thin site-periods are usually dropped rather
        than trusted — this makes that filter explicit and auditable
        instead of silent.

    Returns
    -------
    float, Series, or DataFrame
        A scalar for one index and no grouping; a Series indexed by group
        for one index with ``by``; a DataFrame otherwise, with one column
        per requested index plus an ``n_records`` column when grouped.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> records = pd.DataFrame({
    ...     "county": ["A", "A", "A", "A", "B", "B", "B", "B"],
    ...     "species": ["s1", "s1", "s2", "s3", "s1", "s1", "s1", "s2"],
    ... })
    >>> out = sp.diversity_index(
    ...     records, species="species", by="county", index=["shannon", "richness"]
    ... )
    >>> bool(out.loc["A", "shannon"] > out.loc["B", "shannon"])
    True
    >>> int(out.loc["A", "richness"])
    3

    Notes
    -----
    Every index here is a *sample* quantity: it describes the individuals
    actually recorded, not the community they were drawn from. With
    citizen-science data, observation effort therefore enters the outcome
    directly, which is why effort controls (records per observer-hour,
    say) belong in the regression even after the index is computed.

    References
    ----------
    [@shannon1948mathematical], [@simpson1949measurement],
    [@pielou1966measurement], [@hill1973diversity]
    """
    if isinstance(index, str):
        requested: List[str] = list(DIVERSITY_INDICES) if index == "all" else [index]
        single = index != "all"
    else:
        requested = [str(i) for i in index]
        single = False
    unknown = [i for i in requested if i not in DIVERSITY_INDICES]
    if unknown:
        raise MethodIncompatibility(
            f"Unknown diversity index/indices: {unknown}.",
            recovery_hint=f"Choose from {list(DIVERSITY_INDICES)} or 'all'.",
            diagnostics={"unknown": unknown, "supported": list(DIVERSITY_INDICES)},
        )

    # ── Matrix input: rows are sites, columns are species ──────────────
    if not isinstance(data, pd.DataFrame) or species is None:
        arr = np.asarray(
            data.to_numpy() if isinstance(data, pd.DataFrame) else data, dtype=float
        )
        if arr.ndim == 1:
            vals = {i: _index_from_counts(arr, i, q, base) for i in requested}
            return float(vals[requested[0]]) if single else pd.Series(vals)
        if arr.ndim != 2:
            raise MethodIncompatibility(
                "Matrix input must be 1-D (one site) or 2-D (sites x species).",
                recovery_hint=(
                    "Pass long-format records with species= instead, or "
                    "reshape to a site-by-species matrix."
                ),
                diagnostics={"ndim": int(arr.ndim)},
            )
        rows = {
            i: [_index_from_counts(arr[r], i, q, base) for r in range(arr.shape[0])]
            for i in requested
        }
        idx = data.index if isinstance(data, pd.DataFrame) else None
        out = pd.DataFrame(rows, index=idx)
        n_rec = arr.sum(axis=1)
        out.loc[n_rec < min_records, requested] = np.nan
        out["n_records"] = n_rec
        return out[requested[0]] if single else out

    # ── Long-format records ────────────────────────────────────────────
    if species not in data.columns:
        raise MethodIncompatibility(
            f"species column {species!r} is not in `data`.",
            diagnostics={"columns": list(data.columns)[:20]},
        )
    by_l = [by] if isinstance(by, str) else list(by or [])
    missing = [c for c in by_l + ([count] if count else []) if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"Columns not found in `data`: {missing}.",
            diagnostics={"missing": missing},
        )

    work = data[by_l + [species] + ([count] if count else [])].dropna(
        subset=by_l + [species]
    )
    if work.empty:
        raise DataInsufficient(
            "No records remain after dropping rows with missing group or "
            "species values.",
            recovery_hint="Check the species and grouping columns for NaNs.",
            diagnostics={"n_input_rows": int(len(data))},
        )
    if count is None:
        work = work.assign(__count=1.0)
        count_col = "__count"
    else:
        count_col = count

    if not by_l:
        counts = work.groupby(species, observed=True)[count_col].sum().to_numpy()
        if counts.sum() < min_records:
            vals = {i: np.nan for i in requested}
        else:
            vals = {i: _index_from_counts(counts, i, q, base) for i in requested}
        return float(vals[requested[0]]) if single else pd.Series(vals)

    grouped = work.groupby(by_l + [species], observed=True)[count_col].sum()

    def _per_group(s: pd.Series) -> pd.Series:
        counts = s.to_numpy(dtype=float)
        n_rec = float(counts.sum())
        if n_rec < min_records:
            vals: Dict[str, Any] = {i: np.nan for i in requested}
        else:
            vals = {i: _index_from_counts(counts, i, q, base) for i in requested}
        vals["n_records"] = n_rec
        return pd.Series(vals)

    out = grouped.groupby(level=list(range(len(by_l))), observed=True).apply(_per_group)
    out = out.unstack(-1) if isinstance(out.index, pd.MultiIndex) else out.to_frame().T
    out = out[requested + ["n_records"]]
    out.index.names = by_l
    return out[requested[0]] if single else out
