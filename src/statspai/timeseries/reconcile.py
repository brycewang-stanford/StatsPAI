"""Hierarchical and grouped time series: building the aggregation
structure and reconciling forecasts so that they add up.

Forecasts made series by series do not respect the accounting
identities of the data (states do not sum to the country). Reconciliation
maps the base forecasts ``y_hat`` of all series to coherent ones,
``S G y_hat``, with ``S`` the summing matrix. The trace-minimising
choice of ``G`` (MinT) is due to Wickramasuriya, Athanasopoulos and
Hyndman (2019); ordinary least squares reconciliation to Hyndman, Ahmed,
Athanasopoulos and Shang (2011).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

_METHODS = {
    "bottom_up": "bottom_up",
    "bu": "bottom_up",
    "bottomup": "bottom_up",
    "top_down": "top_down",
    "td": "top_down",
    "topdown": "top_down",
    "ols": "ols",
    "wls_struct": "wls_struct",
    "structural": "wls_struct",
    "wls_var": "wls_var",
    "wls": "wls_var",
    "mint_cov": "mint_cov",
    "mint_sample": "mint_cov",
    "mint_shrink": "mint_shrink",
    "mint": "mint_shrink",
}


@dataclass
class Hierarchy:
    """An aggregation structure returned by :func:`statspai.hierarchy`.

    Attributes
    ----------
    Y : pd.DataFrame
        Every series of the structure, aggregates first and the bottom
        level last; rows are time periods, columns series identifiers
        (the key values joined by ``sep``).
    S : pd.DataFrame
        Summing matrix: one row per series, one column per bottom-level
        series; ``Y.T = S @ Y_bottom.T``.
    tags : dict
        Level name (the key columns joined by ``sep``) to the list of
        its series.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({
    ...     "country": "AU", "state": ["A", "A", "B", "B"],
    ...     "t": [1, 2, 1, 2], "y": [1.0, 2.0, 3.0, 4.0]})
    >>> h = sp.hierarchy(df, [["country"], ["country", "state"]], time="t", value="y")
    >>> h.Y.columns.tolist()
    ['AU', 'AU/A', 'AU/B']
    >>> h.S.values.tolist()
    [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    """

    Y: pd.DataFrame
    S: pd.DataFrame
    tags: Dict[str, List[str]]

    @property
    def bottom(self) -> List[str]:
        """Identifiers of the bottom-level series."""
        return list(self.S.columns)

    def __repr__(self) -> str:
        lv = ", ".join(f"{k} ({len(v)})" for k, v in self.tags.items())
        return (
            f"Hierarchy: {self.Y.shape[1]} series over {self.Y.shape[0]} periods, "
            f"{self.S.shape[1]} at the bottom\n  levels: {lv}"
        )


def hierarchy(
    data: pd.DataFrame,
    spec: Sequence[Sequence[str]],
    *,
    time: str,
    value: str,
    sep: str = "/",
) -> Hierarchy:
    """Aggregate bottom-level series into a hierarchical or grouped
    structure and build its summing matrix.

    Parameters
    ----------
    data : pd.DataFrame
        Long table: one row per bottom-level series and period, with the
        key columns that define the aggregation.
    spec : list of list of str
        One list of key columns per level, e.g. ``[["country"],
        ["country", "state"], ["country", "state", "region"]]``. The last
        entry is the bottom level and must identify the most
        disaggregated series; the others may nest (a hierarchy) or cross
        (a grouped structure, e.g. by state and by purpose).
    time, value : str
        Period and value columns.
    sep : str, default "/"
        Separator of the key values in the series identifiers.

    Returns
    -------
    Hierarchy
        ``Y`` (all series, wide), ``S`` (summing matrix), ``tags``.

    Notes
    -----
    The bottom level must be balanced: every bottom series observed in
    every period. Missing cells would silently bias every aggregate, so
    they are refused.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({
    ...     "all": "Total",
    ...     "state": ["A", "A", "A", "A", "B", "B", "B", "B"],
    ...     "kind": ["x", "x", "y", "y", "x", "x", "y", "y"],
    ...     "t": [1, 2] * 4,
    ...     "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    >>> spec = [["all"], ["all", "state"], ["all", "kind"], ["all", "state", "kind"]]
    >>> h = sp.hierarchy(df, spec, time="t", value="y")
    >>> h.S.shape
    (9, 4)
    >>> h.Y["Total"].tolist()
    [16.0, 20.0]

    References
    ----------
    hyndman2011optimal, hyndman2026fpppy
    """
    if not spec or any(len(s) == 0 for s in spec):
        raise MethodIncompatibility(
            "hierarchy: spec must be a non-empty list of non-empty key lists.",
            recovery_hint='E.g. spec=[["country"], ["country", "state"]].',
        )
    levels = [list(s) for s in spec]
    needed = sorted({c for s in levels for c in s} | {time, value})
    absent = [c for c in needed if c not in data.columns]
    if absent:
        raise MethodIncompatibility(
            f"hierarchy: columns {absent} are not in data.",
            recovery_hint="Check spec, time= and value=.",
        )
    bottom_keys = levels[-1]
    for lv in levels[:-1]:
        extra = [c for c in lv if c not in bottom_keys]
        if extra:
            raise MethodIncompatibility(
                f"hierarchy: level {lv} uses {extra}, which the bottom level "
                f"{bottom_keys} does not contain.",
                recovery_hint="The last entry of spec must contain every key.",
            )
    work = data[needed].copy()
    if work[value].isna().any():
        raise MethodIncompatibility(
            f"hierarchy: {int(work[value].isna().sum())} missing values in "
            f"{value!r}.",
            recovery_hint="Fill or drop them before aggregating.",
        )

    def ident(frame: pd.DataFrame, cols: List[str]) -> pd.Series:
        out = frame[cols[0]].astype(str)
        for c in cols[1:]:
            out = out + sep + frame[c].astype(str)
        return out

    work["_bottom"] = ident(work, bottom_keys)
    dup = work.duplicated(["_bottom", time])
    if dup.any():
        raise MethodIncompatibility(
            f"hierarchy: {int(dup.sum())} duplicated (series, period) rows at "
            f"the bottom level {bottom_keys}.",
            recovery_hint="Add the key that distinguishes them to the last "
            "entry of spec, or aggregate first.",
        )
    wide_bottom = work.pivot(index=time, columns="_bottom", values=value).sort_index()
    if wide_bottom.isna().any().any():
        n_bad = int(wide_bottom.isna().sum().sum())
        raise MethodIncompatibility(
            f"hierarchy: the bottom level is unbalanced ({n_bad} missing "
            "series-period cells).",
            recovery_hint="Fill the gaps (e.g. with zeros for counts) first.",
            diagnostics={"n_missing_cells": n_bad},
        )
    bottom_ids = list(wide_bottom.columns)
    meta = work.drop_duplicates("_bottom").set_index("_bottom").loc[bottom_ids]
    blocks: List[pd.DataFrame] = []
    s_rows: List[np.ndarray] = []
    ids: List[str] = []
    tags: Dict[str, List[str]] = {}
    for lv in levels:
        lab = sep.join(lv)
        if lv == bottom_keys:
            blocks.append(wide_bottom)
            s_rows.append(np.eye(len(bottom_ids)))
            ids.extend(bottom_ids)
            tags[lab] = list(bottom_ids)
            continue
        member = ident(meta, lv)
        groups = sorted(member.unique())
        ind = np.zeros((len(groups), len(bottom_ids)))
        for i, g in enumerate(groups):
            ind[i] = (member.to_numpy() == g).astype(float)
        agg = pd.DataFrame(
            wide_bottom.to_numpy() @ ind.T, index=wide_bottom.index, columns=groups
        )
        blocks.append(agg)
        s_rows.append(ind)
        ids.extend(groups)
        tags[lab] = list(groups)
    if len(set(ids)) != len(ids):
        raise MethodIncompatibility(
            "hierarchy: two levels produce the same series identifiers.",
            recovery_hint="Give each level a distinct set of key columns.",
        )
    Y = pd.concat(blocks, axis=1)
    Y.columns = ids
    Y.columns.name = None
    S = pd.DataFrame(np.vstack(s_rows), index=ids, columns=bottom_ids)
    return Hierarchy(Y=Y, S=S, tags=tags)


@dataclass
class ReconcileResult(ResultProtocolMixin):
    """Reconciled forecasts returned by :func:`statspai.reconcile`.

    Attributes
    ----------
    forecasts : pd.DataFrame
        Coherent forecasts, laid out like the base forecasts.
    method : str
    G : pd.DataFrame
        The matrix that maps base forecasts of all series to bottom-level
        forecasts; ``forecasts = base @ (S @ G).T``.
    weights : np.ndarray or None
        The matrix ``W`` of the generalised least squares projection.
    shrinkage : float or None
        Shrinkage intensity of ``mint_shrink``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> S = pd.DataFrame([[1, 1], [1, 0], [0, 1]], index=["T", "A", "B"],
    ...                  columns=["A", "B"], dtype=float)
    >>> base = pd.DataFrame([[10.0, 4.0, 5.0]], columns=["T", "A", "B"])
    >>> rec = sp.reconcile(base, S, method="ols")
    >>> rec.forecasts.round(4).values.tolist()
    [[9.6667, 4.3333, 5.3333]]
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "wickramasuriya2019optimal",
        "hyndman2011optimal",
    )

    forecasts: pd.DataFrame
    method: str
    G: pd.DataFrame
    weights: Optional[np.ndarray] = None
    shrinkage: Optional[float] = None
    adjustment: Optional[pd.DataFrame] = field(default=None, repr=False)
    sd: Optional[pd.DataFrame] = field(default=None, repr=False)

    def intervals(self, level: float = 95) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Lower and upper bounds of the reconciled normal prediction
        intervals, laid out like ``forecasts``.

        Available when ``sd=`` was given to :func:`statspai.reconcile`.
        ``level`` is the coverage in percent.
        """
        if self.sd is None:
            raise MethodIncompatibility(
                "intervals: no forecast standard deviations were reconciled.",
                recovery_hint="Pass sd= (the base forecasts' standard "
                "deviations) to sp.reconcile.",
            )
        from scipy import stats

        pct = float(level) * 100.0 if 0.0 < float(level) < 1.0 else float(level)
        if not 0.0 < pct < 100.0:
            raise MethodIncompatibility(
                f"intervals: level={level!r} is not a coverage in percent.",
                recovery_hint="Use e.g. level=95.",
            )
        z = float(stats.norm.ppf(0.5 + pct / 200.0))
        return self.forecasts - z * self.sd, self.forecasts + z * self.sd

    def summary(self) -> str:
        n, nb = self.G.shape[1], self.G.shape[0]
        lines = [
            f"Forecast reconciliation: {self.method}",
            "-" * 46,
            f"series     : {n} ({nb} at the bottom level)",
            f"periods    : {self.forecasts.shape[0]}",
        ]
        if self.shrinkage is not None:
            lines.append(f"shrinkage  : {self.shrinkage:.4f}")
        if self.adjustment is not None:
            adj = self.adjustment.abs().to_numpy()
            lines.append(f"largest change to a base forecast: {np.nanmax(adj):.6g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _shrunk_covariance(res: np.ndarray) -> Tuple[np.ndarray, float]:
    """Covariance of the residuals shrunk towards its diagonal, with the
    intensity of Schafer and Strimmer (2005) computed on correlations."""
    T, n = res.shape
    cov = res.T @ res / T
    sd = np.sqrt(np.diag(cov))
    if not (sd > 0).all():
        raise DataInsufficient(
            "reconcile: a series has residuals that are identically zero.",
            recovery_hint="Use method='wls_struct' or 'ols', which need no "
            "residuals.",
        )
    corr = cov / np.outer(sd, sd)
    xs = res / sd
    v = (1.0 / (T * (T - 1.0))) * ((xs**2).T @ (xs**2) - (xs.T @ xs) ** 2 / T)
    np.fill_diagonal(v, 0.0)
    d = (corr - np.eye(n)) ** 2
    den = float(d.sum())
    lam = float(v.sum()) / den if den > 0 else 1.0
    lam = max(min(lam, 1.0), 0.0)
    return lam * np.diag(np.diag(cov)) + (1.0 - lam) * cov, lam


def reconcile(
    base: Any,
    S: Union[pd.DataFrame, Hierarchy],
    *,
    method: str = "mint_shrink",
    residuals: Any = None,
    history: Any = None,
    proportions: str = "average",
    sd: Any = None,
) -> ReconcileResult:
    """Reconcile base forecasts of a hierarchical or grouped structure.

    Parameters
    ----------
    base : pd.DataFrame or array-like
        Base forecasts: one row per forecast period, one column per
        series, named as the rows of ``S`` (an array is taken in the row
        order of ``S``).
    S : pd.DataFrame or Hierarchy
        Summing matrix (rows: all series, columns: bottom-level series),
        or the :class:`Hierarchy` returned by :func:`statspai.hierarchy`.
    method : str, default "mint_shrink"
        - ``"bottom_up"``: sum the bottom-level forecasts.
        - ``"top_down"``: split the forecast of the total by historical
          proportions (needs ``history``).
        - ``"ols"``: least squares projection on the coherent subspace.
        - ``"wls_struct"``: weights equal to the number of bottom series
          each series aggregates.
        - ``"wls_var"``: weights equal to the variance of each series'
          in-sample one-step residuals (needs ``residuals``).
        - ``"mint_shrink"``: minimum trace with the residual covariance
          shrunk towards its diagonal (needs ``residuals``).
        - ``"mint_cov"``: minimum trace with the sample covariance; needs
          more residual periods than series.
    residuals : pd.DataFrame or array-like, optional
        In-sample one-step forecast errors of every series (periods by
        series), for the variance-based methods. Rows with a missing
        value are dropped.
    history : pd.DataFrame or array-like, optional
        The observed series (periods by series), for ``"top_down"``.
    proportions : {"average", "of_averages"}, default "average"
        ``"average"``: mean over time of each bottom series' share of
        the total. ``"of_averages"``: its mean divided by the mean of
        the total.

    sd : pd.DataFrame or array-like, optional
        Standard deviations of the base forecasts, laid out like ``base``
        (for a forecast table, ``(upper - lower) / (2 z)``). When given,
        the result carries the standard deviations of the reconciled
        forecasts in ``sd`` and ``intervals(level)`` returns their normal
        prediction intervals.

    Returns
    -------
    ReconcileResult
        ``forecasts`` (coherent), ``G``, ``weights``, ``shrinkage``,
        ``adjustment`` (reconciled minus base), and with ``sd=`` the
        reconciled ``sd`` and ``intervals(level)``.

    Notes
    -----
    The reconciled forecasts are ``S G y_hat`` with
    ``G = (S' W^{-1} S)^{-1} S' W^{-1}``; the methods differ in ``W``.
    With unbiased base forecasts every least squares variant is
    unbiased, and MinT has the smallest total forecast variance when
    ``W`` is proportional to the covariance of the base forecast errors.

    The covariance estimates divide the cross-products of the residuals
    by the number of periods and do not centre them, as R's
    ``hts::MinT`` does.

    Intervals. Under normal base forecasts with covariance ``W_h`` the
    reconciled forecasts are normal with covariance ``S G W_h G' S'``
    (Panagiotelis et al., 2023). With ``sd=``, ``W_h`` is built from the
    base forecasts' standard deviations at each horizon and the
    correlations of the residual covariance the method estimated
    (``mint_shrink``, ``mint_cov``); the other methods estimate no
    correlations and the base forecast errors are taken as uncorrelated.
    Reconciled intervals are usually narrower than the base ones at the
    aggregate levels. This is the ``Normality`` method of
    hierarchicalforecast.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> S = pd.DataFrame([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ...                  index=["T", "A", "B", "C"], columns=["A", "B", "C"],
    ...                  dtype=float)
    >>> res = pd.DataFrame(rng.normal(size=(60, 4)), columns=S.index)
    >>> base = pd.DataFrame([[31.0, 10.0, 9.0, 11.0]], columns=S.index)
    >>> rec = sp.reconcile(base, S, method="mint_shrink", residuals=res)
    >>> f = rec.forecasts.iloc[0]
    >>> bool(abs(f["T"] - f[["A", "B", "C"]].sum()) < 1e-10)
    True

    References
    ----------
    wickramasuriya2019optimal, hyndman2011optimal, schafer2005shrinkage,
    panagiotelis2023probabilistic, hyndman2026fpppy
    """
    if isinstance(S, Hierarchy):
        S = S.S
    if not isinstance(S, pd.DataFrame):
        raise MethodIncompatibility(
            "reconcile: S must be a DataFrame (rows: all series, columns: "
            "bottom series) or a Hierarchy.",
            recovery_hint="Build it with sp.hierarchy(...).",
        )
    key = _METHODS.get(str(method).lower().replace("-", "_"))
    if key is None:
        raise MethodIncompatibility(
            f"reconcile: method={method!r} is not available.",
            recovery_hint=(
                "Use 'bottom_up', 'top_down', 'ols', 'wls_struct', 'wls_var', "
                "'mint_shrink' or 'mint_cov'."
            ),
        )
    ids = list(S.index)
    bottom = list(S.columns)
    Sm = S.to_numpy(dtype=float)
    n, nb = Sm.shape
    if nb >= n or np.linalg.matrix_rank(Sm) < nb:
        raise MethodIncompatibility(
            f"reconcile: S is {n} by {nb}; it needs more rows than columns "
            "and full column rank.",
            recovery_hint="Rows are all series, columns the bottom series.",
        )

    def align(obj: Any, what: str) -> np.ndarray:
        if isinstance(obj, pd.DataFrame):
            absent = [c for c in ids if c not in obj.columns]
            if absent:
                raise MethodIncompatibility(
                    f"reconcile: {what} lacks the series {absent[:5]}"
                    + (" ..." if len(absent) > 5 else ""),
                    recovery_hint="Its columns must be the row labels of S.",
                )
            return np.asarray(obj[ids].to_numpy(dtype=float), dtype=float)
        arr = np.asarray(obj, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != n:
            raise MethodIncompatibility(
                f"reconcile: {what} has {arr.shape[1]} columns, S has {n} rows.",
                recovery_hint="One column per series, in the row order of S.",
            )
        return arr

    yhat = align(base, "base")
    if not np.isfinite(yhat).all():
        raise MethodIncompatibility(
            "reconcile: the base forecasts have missing values.",
            recovery_hint="Every series needs a forecast for every period.",
        )
    W: Optional[np.ndarray] = None
    lam: Optional[float] = None

    def bottom_rows() -> List[int]:
        """Row of S of each bottom series: by label, else the last row
        that is a unit vector on it (an aggregate with a single child has
        the same row and comes first)."""
        if all(b in ids for b in bottom) and len(set(ids)) == len(ids):
            return [ids.index(b) for b in bottom]
        rows_: List[int] = []
        unit = np.abs(Sm).sum(axis=1) == 1.0
        for j in range(nb):
            hit = np.flatnonzero((Sm[:, j] == 1.0) & unit)
            if hit.size == 0:
                raise MethodIncompatibility(
                    f"reconcile: S has no row for the bottom series {bottom[j]!r}.",
                    recovery_hint="Each bottom series needs its own row of S.",
                )
            rows_.append(int(hit[-1]))
        return rows_

    if key == "bottom_up":
        rows = bottom_rows()
        G = np.zeros((nb, n))
        G[np.arange(nb), rows] = 1.0
    elif key == "top_down":
        if history is None:
            raise MethodIncompatibility(
                "reconcile: method='top_down' needs history= to compute the "
                "proportions.",
                recovery_hint="Pass the observed series (Hierarchy.Y).",
            )
        top = np.flatnonzero((Sm == 1.0).all(axis=1))
        if top.size == 0:
            raise MethodIncompatibility(
                "reconcile: S has no series that is the total of every bottom "
                "series.",
                recovery_hint="Top-down needs a single top level.",
            )
        hist = align(history, "history")
        hist = hist[np.isfinite(hist).all(axis=1)]
        rows = bottom_rows()
        tot = hist[:, top[0]]
        bot = hist[:, rows]
        which = proportions.lower()
        if which in ("average", "average_proportions"):
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.nanmean(bot / tot[:, None], axis=0)
        elif which in ("of_averages", "proportion_averages"):
            p = bot.mean(axis=0) / tot.mean()
        else:
            raise MethodIncompatibility(
                f"reconcile: proportions={proportions!r} is not 'average' or "
                "'of_averages'.",
                recovery_hint="Use proportions='average'.",
            )
        G = np.zeros((nb, n))
        G[:, top[0]] = p
    else:
        if key == "ols":
            W = np.eye(n)
        elif key == "wls_struct":
            W = np.diag(Sm.sum(axis=1))
        else:
            if residuals is None:
                raise MethodIncompatibility(
                    f"reconcile: method={key!r} needs residuals=, the in-sample "
                    "one-step forecast errors of every series.",
                    recovery_hint="E.g. a frame of each model's .residuals; or "
                    "use method='wls_struct'.",
                )
            res = align(residuals, "residuals")
            res = res[np.isfinite(res).all(axis=1)]
            T = res.shape[0]
            if T < 3:
                raise DataInsufficient(
                    f"reconcile: {T} complete residual periods are too few.",
                    recovery_hint="Use method='wls_struct' or 'ols'.",
                )
            cov = res.T @ res / T
            if key == "wls_var":
                W = np.diag(np.diag(cov))
            elif key == "mint_cov":
                if T <= n:
                    raise DataInsufficient(
                        f"reconcile: the sample covariance of {n} series from "
                        f"{T} periods is singular.",
                        recovery_hint="Use method='mint_shrink'.",
                        diagnostics={"n_series": n, "n_periods": T},
                    )
                W = cov
            else:
                W, lam = _shrunk_covariance(res)
        if np.any(np.diag(W) <= 0):
            raise DataInsufficient(
                "reconcile: a series has zero weight (zero residual variance).",
                recovery_hint="Use method='wls_struct' or 'ols'.",
            )
        try:
            Winv_S = np.linalg.solve(W, Sm)
            G = np.linalg.solve(Sm.T @ Winv_S, Winv_S.T)
        except np.linalg.LinAlgError as exc:
            raise DataInsufficient(
                f"reconcile: the weight matrix of {key!r} is singular.",
                recovery_hint="Use method='mint_shrink' or 'wls_struct'.",
            ) from exc
    rec = yhat @ (Sm @ G).T
    if isinstance(base, pd.DataFrame):
        out = pd.DataFrame(rec, index=base.index, columns=ids)
        adj = out - base[ids]
    else:
        out = pd.DataFrame(rec, columns=ids)
        adj = pd.DataFrame(rec - yhat, columns=ids)
    sd_rec: Optional[pd.DataFrame] = None
    if sd is not None:
        sig = align(sd, "sd")
        if sig.shape != yhat.shape:
            raise MethodIncompatibility(
                f"reconcile: sd has shape {sig.shape}, base {yhat.shape}.",
                recovery_hint="One standard deviation per base forecast.",
            )
        if not np.isfinite(sig).all() or (sig < 0).any():
            raise MethodIncompatibility(
                "reconcile: sd has missing or negative values.",
                recovery_hint="Give the standard deviation of every base forecast.",
            )
        corr = np.eye(n)
        if W is not None and key in ("mint_shrink", "mint_cov"):
            d = np.sqrt(np.diag(W))
            corr = W / np.outer(d, d)
        SG = Sm @ G
        var = np.empty_like(sig)
        for r in range(sig.shape[0]):
            cov_h = corr * np.outer(sig[r], sig[r])
            var[r] = np.einsum("ij,jk,ik->i", SG, cov_h, SG)
        sd_rec = pd.DataFrame(
            np.sqrt(np.maximum(var, 0.0)), index=out.index, columns=ids
        )
    return ReconcileResult(
        forecasts=out,
        method=key,
        G=pd.DataFrame(G, index=bottom, columns=ids),
        weights=W,
        shrinkage=lam,
        adjustment=adj,
        sd=sd_rec,
    )
