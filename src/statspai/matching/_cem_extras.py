"""Coarsened exact matching extras: k-to-k matching and the L1 imbalance.

``k2k`` follows R ``MatchIt``'s ``method = "cem", k2k = TRUE`` (GPL; used as
a black box, documentation and outputs only): after the coarsening, 1:1
nearest-neighbour matching without replacement on the Mahalanobis distance
of the original covariates, inside each coarsened cell, in data order --
which MatchIt documents as equivalent to nearest-neighbour matching with
the cell as an ``exact`` variable. Unmatched units of either arm drop out.

:func:`l1_imbalance` is the multivariate L1 statistic of Iacus, King and
Porro [@iacus2011multivariate]: half the sum, over the cells of a
coarsening of the covariates, of the absolute difference between the
(weighted) relative frequencies of the treated and the controls. It is 0
when the two coarsened distributions coincide and 1 when they share no
cell.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def match_cem_k2k(data: pd.DataFrame, spec: Dict[str, Any], fit: Any) -> Any:
    """``sp.match(method='cem', k2k=True)``."""
    est = str(spec.get("estimand", "ATT")).upper()
    if est not in ("ATT", "ATC"):
        raise MethodIncompatibility(
            "sp.match: k2k=True pairs units 1:1 and supports estimand='ATT' "
            "or 'ATC' (the arm whose units are matched in data order).",
            recovery_hint="Use estimand='ATT', or k2k=False for the ATE.",
        )
    if spec.get("exact") or spec.get("mahvars") or spec.get("covariate_calipers"):
        raise MethodIncompatibility(
            "sp.match: k2k=True already matches within the coarsened cells; "
            "exact= / mahvars= / covariate_calipers= are not combined with it.",
        )
    if not data.index.is_unique:
        data = data.reset_index(drop=True)
    cem = fit(data, dict(spec))
    md = cem.model_info["matched_data"]
    work = data.copy()
    col = "__cem_cell__"
    while col in work.columns:  # pragma: no cover - name clash
        col += "_"
    # object dtype: the cells are strings (a float column would warn now and
    # refuse the assignment in a later pandas)
    work[col] = pd.Series(np.nan, index=work.index, dtype=object)
    work.loc[md.index, col] = md["_stratum"].astype(str).to_numpy()
    nn_spec = dict(spec)
    nn_spec.update(
        method="nearest",
        distance="mahalanobis",
        exact=[col],
        replace=False,
        m_order="data",
        n_matches=1,
        n_bins=None,
    )
    res = fit(work, nn_spec)
    res.model_info["k2k"] = True
    res.model_info["cem_cells"] = int(md["_stratum"].nunique())
    res.model_info["n_matched_pairs"] = int(res.model_info.get("n_matched_treated", 0))
    res.method = "Coarsened exact matching, k2k (Mahalanobis within cells)"
    return res


def l1_imbalance(
    data: pd.DataFrame,
    treat: str,
    covariates: Sequence[str],
    weights: Optional[Any] = None,
    n_bins: Optional[int] = None,
) -> float:
    """Multivariate L1 imbalance of Iacus, King and Porro (2011).

    Each numeric covariate with more than ``n_bins`` distinct values is cut
    into ``n_bins`` equal-width bins between its minimum and maximum (bins
    closed on the right, the first on both sides, as ``sp.match(method=
    'cem')`` coarsens); other covariates keep their values. ``L1 = 1/2
    sum_cells |f_cell - g_cell|`` with ``f`` and ``g`` the relative
    (weighted) frequencies of the treated and the controls.

    Parameters
    ----------
    data : pd.DataFrame
    treat : str
        0/1 treatment.
    covariates : sequence of str
    weights : array-like or str, optional
        Matching or balancing weights (``NaN`` = unmatched, weight 0); the
        default is the unweighted sample. The bins are set on every row with
        complete covariates whatever its weight, so a weighted (after
        matching) and an unweighted (before matching) L1 use the same
        cells. Negative or infinite weights are refused.
    n_bins : int, optional
        Bins per covariate; default Sturges' rule ``ceil(log2(n) + 1)``
        (the default of R ``cem``'s ``imbalance`` is a different,
        histogram-based coarsening, so the numbers are comparable only
        under the same coarsening).

    Returns
    -------
    float
        In ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'d': rng.integers(0, 2, 500), 'x': rng.normal(size=500)})
    >>> bool(0 <= sp.l1_imbalance(df, 'd', ['x']) < 0.3)
    True

    References
    ----------
    [@iacus2011multivariate]
    """
    cols = list(covariates)
    missing = [c for c in [treat] + cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"l1_imbalance: columns {missing} not found.")
    if isinstance(weights, str):
        w = data[weights].to_numpy(dtype=float)
    elif weights is None:
        w = np.ones(len(data))
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if len(w) != len(data):
            raise MethodIncompatibility("l1_imbalance: weights have the wrong length.")
    # a missing weight is an unmatched row (matched_data's _weight): mass 0
    w = np.where(np.isnan(w), 0.0, w)
    if weights is not None and (np.any(w < 0) or not np.isfinite(w).all()):
        raise MethodIncompatibility(
            "l1_imbalance: weights must be finite and non-negative.",
            recovery_hint="Drop or fix rows with negative or infinite weights.",
        )
    # The coarsening (bin count and edges) is fixed on every complete row,
    # zero-weight rows included, as cem's imbalance() does; weights only
    # move mass between the cells. Dropping zero weights first gave the
    # after-matching L1 a different coarsening from the before-matching one
    # (0.150 instead of 0.104 after k2k, 2026-10-20 probe).
    keep = data[[treat] + cols].notna().all(axis=1).to_numpy()
    df = data.loc[keep]
    w = w[keep]
    try:
        t = df[treat].to_numpy(dtype=float)
    except (TypeError, ValueError):
        t = np.array([np.nan])
    if not np.isin(t, (0.0, 1.0)).all() or t.min() == t.max():
        raise MethodIncompatibility(
            "l1_imbalance: the treatment must be 0/1 with both arms present.",
            recovery_hint="Recode the treatment as 0/1 (e.g. "
            "(df[treat] == 'treated').astype(int)).",
        )
    if w[t == 1].sum() <= 0 or w[t == 0].sum() <= 0:
        raise MethodIncompatibility(
            "l1_imbalance: an arm has zero total weight.",
            recovery_hint="Each arm needs some positive weight.",
        )
    n = len(df)
    k = int(n_bins) if n_bins is not None else int(np.ceil(np.log2(n) + 1))
    if k < 1:
        raise MethodIncompatibility("l1_imbalance: n_bins must be at least 1.")
    codes = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > k:
            x = s.to_numpy(dtype=float)
            edges = np.linspace(x.min(), x.max(), k + 1)
            codes.append(
                (x >= edges[0]).astype(int) + np.searchsorted(edges[1:], x, side="left")
            )
        else:
            codes.append(pd.factorize(s)[0])
    cell = pd.Series(list(zip(*codes))) if codes else pd.Series(np.zeros(n))
    cell_id = pd.factorize(cell)[0]
    m = int(cell_id.max()) + 1
    f = np.bincount(cell_id[t == 1], weights=w[t == 1], minlength=m)
    g = np.bincount(cell_id[t == 0], weights=w[t == 0], minlength=m)
    return float(0.5 * np.abs(f / f.sum() - g / g.sum()).sum())
