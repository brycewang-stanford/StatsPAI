"""
DiD using Double Negative Controls and GNNs for Unmeasured Network
Confounding (Zhang, Fu & Wang 2026, arXiv 2601.00603).

Combines double negative controls (DNC) with graph neural network
features that summarise unit-level network position. The full
algorithm learns those features with a GNN (heavy dependency); this
lightweight version takes them as given through the ``embedding``
argument (one row per unit, e.g. a GNN embedding or structural features
such as degree and clustering) and adjusts for them linearly. Without an
embedding no network feature enters the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility


@dataclass
class DNCGNNDiDResult(ResultProtocolMixin):
    """Output of DNC + GNN + DiD.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for uid in range(40):
    ...     treat_time = 3 if uid < 20 else 0
    ...     u = rng.normal()
    ...     for t in range(1, 6):
    ...         post = 1 if (treat_time > 0 and t >= treat_time) else 0
    ...         rows.append({
    ...             "id": uid, "time": t, "treat": treat_time,
    ...             "y": u + 0.5 * t + 1.2 * post + rng.normal(0, 0.3),
    ...             "nc_y": u + 0.4 * t + rng.normal(0, 0.3),
    ...             "nc_d": u + rng.normal(0, 0.3),
    ...         })
    >>> df = pd.DataFrame(rows)
    >>> res = sp.dnc_gnn_did(
    ...     df, y="y", treat="treat", time="time", id="id",
    ...     nc_outcome=["nc_y"], nc_exposure=["nc_d"], n_boot=50, seed=1)
    >>> isinstance(res, sp.DNCGNNDiDResult)
    True
    >>> bool(res.se > 0 and res.ci[0] < res.estimate < res.ci[1])
    True
    """

    estimate: float
    se: float
    ci: tuple
    n_obs: int
    method: str = "DNC + GNN-aware DiD"
    diagnostics: Optional[dict] = None

    def summary(self) -> str:
        diag = self.diagnostics or {}
        rows = [
            self.method,
            "=" * 42,
            f"  N         : {self.n_obs}",
            f"  Estimate  : {self.estimate:+.4f} (SE {self.se:.4f})",
            f"  95% CI    : [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}]",
        ]
        for k, v in diag.items():
            rows.append(f"  {k}: {v}")
        return "\n".join(rows)


def _embedding_frame(
    embedding: Any, unit_col: pd.Series, units: pd.Index
) -> pd.DataFrame:
    """Network features as a frame indexed by unit, one row per ``units``.

    A DataFrame is aligned on its index (the unit ids). An array is read in
    sorted-unit order, the order the estimator's unit table uses, and is
    refused when the units do not first appear in that order, since its
    rows could then be meant in either order.
    """
    if isinstance(embedding, pd.Series):
        embedding = embedding.to_frame()
    if isinstance(embedding, pd.DataFrame):
        if embedding.index.has_duplicates:
            raise MethodIncompatibility(
                "dnc_gnn_did: the embedding index repeats a unit id.",
                recovery_hint="Pass one row of network features per unit, "
                "indexed by the id column.",
            )
        missing = units.difference(embedding.index)
        if len(missing):
            raise MethodIncompatibility(
                f"dnc_gnn_did: the embedding has no row for {len(missing)} "
                f"unit(s), e.g. {list(missing[:5])}.",
                recovery_hint="Index the embedding by the values of the id "
                "column and give a row for every unit.",
                diagnostics={"n_missing": int(len(missing))},
            )
        values = embedding.loc[units].to_numpy(dtype=float)
    else:
        arr = np.asarray(embedding, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        all_units = pd.Index(pd.unique(unit_col.dropna()))
        sorted_units = all_units.sort_values()
        if not all_units.equals(sorted_units):
            raise MethodIncompatibility(
                "dnc_gnn_did: an embedding array has no unit labels, and the "
                "units do not appear in sorted order in data, so its rows "
                "cannot be matched to units unambiguously.",
                recovery_hint="Pass the embedding as a pandas DataFrame "
                "indexed by the id values.",
            )
        if arr.shape[0] == len(sorted_units):
            values = pd.DataFrame(arr, index=sorted_units).loc[units].to_numpy(float)
        elif arr.shape[0] == len(units):
            values = arr
        else:
            raise MethodIncompatibility(
                f"dnc_gnn_did: the embedding has {arr.shape[0]} rows, but "
                f"data has {len(sorted_units)} units ({len(units)} with "
                "pre- and post-period outcomes).",
                recovery_hint="Pass one row per unit in sorted id order, or "
                "a DataFrame indexed by id.",
            )
    if not np.all(np.isfinite(values)):
        raise MethodIncompatibility(
            "dnc_gnn_did: the embedding has missing or infinite values.",
            recovery_hint="Impute or drop those units' features first.",
        )
    cols = [f"_emb{i}" for i in range(values.shape[1])]
    return pd.DataFrame(values, index=units, columns=cols)


def dnc_gnn_did(
    data: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id: str,
    nc_outcome: List[str],
    nc_exposure: List[str],
    embedding: Optional[Any] = None,
    alpha: float = 0.05,
    n_boot: int = 100,
    seed: int = 0,
) -> DNCGNNDiDResult:
    """
    DiD with double negative controls + (optional) GNN embedding for
    network-level unmeasured confounding.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    y : str
        Outcome.
    treat : str
        First-treatment-period column (0 = never-treated).
    time : str
    id : str
    nc_outcome : list of str
        Negative-control outcomes — variables expected NOT affected
        by the treatment but sharing the same confounding structure.
    nc_exposure : list of str
        Negative-control exposures — pseudo-treatments expected NOT
        to cause the outcome.
    embedding : pandas.DataFrame or np.ndarray, optional
        Network features of each unit (a GNN embedding, degree,
        clustering ...), entered linearly in the adjustment regression.
        A DataFrame is matched to the units by its index, which must hold
        the ``id`` values (row order does not matter; every estimated unit
        must be present). An array has no unit labels: its rows are read
        in sorted-``id`` order, and it is refused when that order differs
        from the order in which units first appear in ``data`` (pass a
        DataFrame indexed by ``id`` then). ``None`` (default) uses no
        network feature: the estimate is the negative-control-adjusted
        DiD, and ``diagnostics['network_features']`` is ``None``.

        .. versionchanged:: 1.42
           DataFrames are aligned by unit id (a DataFrame used to become
           all-NaN features), and arrays whose row order is ambiguous are
           refused. The docstring used to promise a lagged-outcome stand-in
           feature that was never built.
    alpha : float
    n_boot : int
    seed : int

    Returns
    -------
    DNCGNNDiDResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for uid in range(40):
    ...     treat_time = 3 if uid < 20 else 0
    ...     u = rng.normal()
    ...     for t in range(1, 6):
    ...         post = 1 if (treat_time > 0 and t >= treat_time) else 0
    ...         rows.append({
    ...             "id": uid, "time": t, "treat": treat_time,
    ...             "y": u + 0.5 * t + 1.2 * post + rng.normal(0, 0.3),
    ...             "nc_y": u + 0.4 * t + rng.normal(0, 0.3),
    ...             "nc_d": u + rng.normal(0, 0.3),
    ...         })
    >>> df = pd.DataFrame(rows)
    >>> res = sp.dnc_gnn_did(
    ...     df, y="y", treat="treat", time="time", id="id",
    ...     nc_outcome=["nc_y"], nc_exposure=["nc_d"], n_boot=50, seed=1)
    >>> bool(res.se > 0)
    True
    >>> print(res.summary())  # doctest: +SKIP
    """
    df = (
        data[[y, treat, time, id] + list(nc_outcome) + list(nc_exposure)]
        .dropna()
        .reset_index(drop=True)
    )
    treated_mask = df[treat] > 0
    # Post indicator: for treated units, post = time >= first_treat;
    # for never-treated, use the median first-treatment period across the
    # treated cohorts so they too contribute pre/post means.
    if treated_mask.any():
        ref_time = float(df.loc[treated_mask, treat].median())
    else:
        ref_time = float(df[time].median())
    df["_post"] = np.where(
        treated_mask,
        (df[time] >= df[treat]).astype(int),
        (df[time] >= ref_time).astype(int),
    )
    df["_treat"] = treated_mask.astype(int)

    # Per-unit pre/post means
    unit_means = df.groupby([id, "_treat", "_post"])[y].mean().reset_index()
    if unit_means.empty:
        raise ValueError("Could not compute unit-level pre/post means.")

    # Naive 2x2 DID
    pivot = unit_means.pivot_table(index=id, columns="_post", values=y).dropna()
    pivot.columns = ["y_pre", "y_post"]
    pivot["delta"] = pivot["y_post"] - pivot["y_pre"]
    treat_per_unit = (df.groupby(id)["_treat"].max() > 0).astype(int)
    pivot = pivot.join(treat_per_unit.rename("_treat_unit"))
    naive_dim = float(
        pivot.loc[pivot["_treat_unit"] == 1, "delta"].mean()
        - pivot.loc[pivot["_treat_unit"] == 0, "delta"].mean()
    )

    # Negative-control adjustment: regress delta on (treat, NC outcome means,
    # NC exposure means, embedding); coefficient on treat is corrected ATT.
    nc_o_means = df.groupby(id)[list(nc_outcome)].mean()
    nc_e_means = df.groupby(id)[list(nc_exposure)].mean()
    pivot = pivot.join(nc_o_means).join(nc_e_means).dropna()

    emb_cols: List[str] = []
    if embedding is not None:
        emb_df = _embedding_frame(embedding, data[id], pivot.index)
        emb_cols = list(emb_df.columns)
        pivot = pivot.join(emb_df)

    Yd = pivot["delta"].to_numpy(float)
    D = pivot["_treat_unit"].to_numpy(float)
    extra_cols = list(nc_outcome) + list(nc_exposure) + emb_cols
    Xextra = pivot[extra_cols].to_numpy(float)
    X = np.column_stack([np.ones_like(D), D, Xextra])
    try:
        beta = np.linalg.solve(X.T @ X, X.T @ Yd)
        resid = Yd - X @ beta
        sigma2 = float(np.sum(resid**2) / max(len(Yd) - X.shape[1], 1))
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        att = float(beta[1])
        se_closed = float(np.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        att = naive_dim
        se_closed = float(pivot["delta"].std(ddof=1) / np.sqrt(max(len(pivot), 1)))

    # Cluster bootstrap on units for SE
    rng = np.random.default_rng(seed)
    boot = np.full(n_boot, np.nan)
    units = pivot.index.to_numpy()
    for b in range(n_boot):
        idx = rng.choice(len(units), size=len(units), replace=True)
        try:
            sub = pivot.iloc[idx]
            X_b = np.column_stack(
                [
                    np.ones(len(sub)),
                    sub["_treat_unit"].to_numpy(float),
                    sub[extra_cols].to_numpy(float),
                ]
            )
            beta_b = np.linalg.solve(X_b.T @ X_b, X_b.T @ sub["delta"].to_numpy(float))
            boot[b] = float(beta_b[1])
        except np.linalg.LinAlgError:
            continue  # replicate stays NaN; bootstrap_se tracks the failure
    from ..core._bootstrap import bootstrap_se

    se = bootstrap_se(boot, label="interference.dnc_gnn_did") or se_closed
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (att - z_crit * se, att + z_crit * se)

    return DNCGNNDiDResult(
        estimate=att,
        se=se,
        ci=ci,
        n_obs=len(pivot),
        diagnostics={
            "naive_dim": naive_dim,
            "embedding_used": embedding is not None,
            "network_features": emb_cols or None,
            "reference": "Zhang, Fu & Wang (2026), arXiv 2601.00603",
        },
    )
