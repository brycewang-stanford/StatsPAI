"""The Rotemberg-weight summary table of Goldsmith-Pinkham, Sorkin & Swift.

``sp.rotemberg_summary`` reproduces the table their replication code builds
for a Bartik instrument (``make_rotemberg_summary_ADH.do``)
[@goldsmithpinkham2020bartik]:

* Rotemberg weights ``alpha_kt`` and just-identified estimates ``beta_kt``
  for every industry x period instrument, as their ``bartik_weight``;
* Panel A -- sum, mean and share of the negative and positive weights of the
  industry aggregates ``alpha_k = sum_t alpha_kt``;
* Panel B -- correlations across industries of ``alpha_k``, ``g_k``,
  ``beta_k``, the first-stage ``F_k`` and the dispersion of the industry
  share (the column GPSS label ``Var(z_k)`` holds its alpha-weighted
  standard deviation);
* Panel C -- sum and mean of ``alpha_kt`` by period;
* Panel D -- the top industries by ``alpha_k`` with ``g_k``, ``beta_k``, the
  weak-instrument-robust interval of their ``ch_weak`` (clustered
  Anderson-Rubin tests inverted over a grid) and the industry share;
* Panel E -- the alpha-weighted sum of ``beta_k``, its share of the overall
  estimate and the mean ``beta_k``, by sign of the weight.

Industry aggregates are alpha-weighted averages across periods
(``beta_k = sum_t alpha_kt beta_kt / alpha_k``), as in the GPSS code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..exceptions import MethodIncompatibility


def _stata_tstats(
    X: np.ndarray, Y: np.ndarray, w: np.ndarray, groups: Optional[np.ndarray]
) -> np.ndarray:
    """t statistics of column 0 of ``X`` for each column of ``Y``.

    Stata ``regress y X [aw=w], cluster(g)`` (or ``robust`` without
    ``groups``), aweights rescaled to sum to N: CR1 with
    ``G/(G-1) (N-1)/(N-K)``, HC1 with ``N/(N-K)``.
    """
    n, k = X.shape
    wn = w * n / w.sum()
    bread = np.linalg.inv((X * wn[:, None]).T @ X)
    B = bread @ ((X * wn[:, None]).T @ Y)
    E = Y - X @ B
    score = (X @ bread[0] * wn)[:, None] * E  # n x m, the column-0 score
    if groups is None:
        var = (n / (n - k)) * np.sum(score**2, axis=0)
    else:
        G = int(groups.max()) + 1
        S = np.zeros((G, Y.shape[1]))
        np.add.at(S, groups, score)
        var = (G / (G - 1)) * ((n - 1) / (n - k)) * np.sum(S**2, axis=0)
    return B[0] / np.sqrt(var)


@accepts_aliases(controls="covariates", period="time")
def rotemberg_summary(
    data: pd.DataFrame,
    y: str,
    x: str,
    shares: List[str],
    shocks: pd.DataFrame,
    *,
    time: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    top: int = 5,
    ci_grid: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """GPSS Rotemberg-weight summary of a Bartik instrument.

    Parameters
    ----------
    data : pd.DataFrame
        One row per unit (x period): outcome, endogenous regressor, the
        industry share columns and the controls.
    y, x : str
        Outcome and endogenous regressor.
    shares : list of str
        Share columns, one per industry (the row's own-period shares).
    shocks : pd.DataFrame
        Columns ``industry`` (the share column names), ``g`` and, with
        ``time``, ``period``.
    time : str, optional
        Period column (``period=`` is an alias); each industry x period is its own instrument, as in
        GPSS's ``t<year>_<share>`` construction.
    covariates : list of str, optional
        Controls; an intercept is added (``controls=`` is an alias).
    weights : str, optional
        Analytic weights.
    cluster : str, optional
        Cluster variable for ``F_k`` and the intervals (heteroskedasticity-
        robust when omitted, as ``ch_weak``).
    top : int, default 5
        Industries listed in panel D.
    ci_grid : sequence of float, optional
        Grid over which the Anderson-Rubin test is inverted; default
        ``-10, -9.9, ..., 10`` as in GPSS. An interval that reaches an end
        of the grid is reported as unbounded (their "N/A").
    alpha : float, default 0.05
        Level of the Anderson-Rubin test.

    Returns
    -------
    dict
        ``cells`` (industry x period alpha, beta, g), ``industries`` (the
        aggregates, sorted by alpha), ``panel_a`` .. ``panel_e`` as
        DataFrames, and ``beta`` (the alpha-weighted sum of beta_kt, which
        is the 2SLS estimate).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> S = rng.dirichlet(np.ones(6), size=200) * 0.5
    >>> g = rng.normal(1, 1, 6)
    >>> df = pd.DataFrame(S, columns=[f"s{k}" for k in range(6)])
    >>> df["x"] = S @ (g * rng.normal(1, 0.5, 6)) + rng.normal(scale=0.2, size=200)
    >>> df["y"] = -0.5 * df["x"] + rng.normal(size=200)
    >>> shocks = pd.DataFrame({"industry": df.columns[:6], "g": g})
    >>> out = sp.rotemberg_summary(df, y="y", x="x", shares=list(df.columns[:6]),
    ...                            shocks=shocks)
    >>> round(float(out["industries"]["alpha"].sum()), 12)
    1.0

    References
    ----------
    [@goldsmithpinkham2020bartik]
    """
    period = time
    controls = list(covariates or [])
    need = [y, x] + list(shares) + controls
    need += [c for c in (weights, cluster, period) if c]
    missing = [c for c in need if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"rotemberg_summary: columns not found: {missing}")
    if not {"industry", "g"} <= set(shocks.columns) or (
        period is not None and "period" not in shocks.columns
    ):
        raise MethodIncompatibility(
            "rotemberg_summary: shocks needs columns industry, g"
            + (" and period" if period else "")
        )
    df = data[list(dict.fromkeys(need))].dropna().reset_index(drop=True)
    n = len(df)
    w = df[weights].to_numpy(float) if weights else np.ones(n)
    groups = pd.factorize(df[cluster])[0] if cluster else None
    periods = sorted(df[period].unique()) if period else [None]
    Wc = np.column_stack([df[c].to_numpy(float) for c in controls] + [np.ones(n)])
    sw = np.sqrt(w)

    def resid(v: np.ndarray) -> np.ndarray:
        coef = np.linalg.lstsq(Wc * sw[:, None], v * sw, rcond=None)[0]
        return v - Wc @ coef

    xt = resid(df[x].to_numpy(float))
    yt = resid(df[y].to_numpy(float))

    # Industry x period instruments and shocks (GPSS t<year>_share / _g).
    shock_key = (
        shocks.set_index(["period", "industry"])["g"]
        if period
        else shocks.set_index("industry")["g"]
    )
    cells = []
    Zcols = []
    for t in periods:
        in_t = (df[period] == t).to_numpy() if period else np.ones(n, bool)
        for k in shares:
            key = (t, k) if period else k
            if key not in shock_key.index:
                raise MethodIncompatibility(f"rotemberg_summary: no shock for {key!r}.")
            Zcols.append(np.where(in_t, df[k].to_numpy(float), 0.0))
            cells.append(dict(industry=k, period=t, g=float(shock_key.loc[key])))
    Z = np.column_stack(Zcols)
    G = np.array([c["g"] for c in cells])
    zx = Z.T @ (w * xt)
    zy = Z.T @ (w * yt)
    denom = float(G @ zx)
    if abs(denom) < 1e-14:
        raise MethodIncompatibility(
            "rotemberg_summary: the Bartik first stage is zero; weights undefined."
        )
    a_kt = G * zx / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        b_kt = np.where(zx != 0, zy / zx, np.nan)
    cell_df = pd.DataFrame(cells).assign(alpha=a_kt, beta=b_kt)

    # Industry-share mean and (aweighted) sd by industry x period, as
    # ``collapse (sd) (rawsum) [aweight=w], by(ind year)``.
    share_stats = []
    for t in periods:
        in_t = (df[period] == t).to_numpy() if period else np.ones(n, bool)
        wt = w[in_t]
        m = int(in_t.sum())
        for k in shares:
            v = df.loc[in_t, k].to_numpy(float)
            mu = float(wt @ v / wt.sum())
            sd = float(np.sqrt(m / (m - 1) * (wt @ (v - mu) ** 2) / wt.sum()))
            share_stats.append(dict(industry=k, period=t, share_mean=mu, share_sd=sd))
    cell_df = cell_df.merge(pd.DataFrame(share_stats), on=["industry", "period"])

    # First-stage F of each industry's own instrument share_k * g_k.
    X0 = df[x].to_numpy(float)
    F = {}
    for k in shares:
        if period:
            gk = (
                df[period]
                .map({t: shock_key.loc[(t, k)] for t in periods})
                .to_numpy(float)
            )
        else:
            gk = np.full(n, float(shock_key.loc[k]))
        zk = df[k].to_numpy(float) * gk
        tstat = _stata_tstats(np.column_stack([zk, Wc]), X0[:, None], w, groups)[0]
        F[k] = float(tstat**2)

    agg = (
        cell_df.assign(
            _ab=cell_df.alpha * cell_df.beta,
            _ag=cell_df.alpha * cell_df.g,
            _as=cell_df.alpha * cell_df.share_mean,
            _asd=cell_df.alpha * cell_df.share_sd,
        )
        .groupby("industry", sort=False)[["alpha", "_ab", "_ag", "_as", "_asd"]]
        .sum()
    )
    ind = pd.DataFrame(
        {
            "alpha": agg["alpha"],
            "g": agg["_ag"] / agg["alpha"],
            "beta": agg["_ab"] / agg["alpha"],
            "F": pd.Series(F),
            "share": agg["_as"] / agg["alpha"],
            "share_sd": agg["_asd"] / agg["alpha"],
        }
    ).sort_values("alpha", ascending=False)

    a = ind["alpha"]
    sum_p, sum_n = float(a[a > 0].sum()), float(a[a < 0].sum())
    tot = abs(sum_p) + abs(sum_n)
    panel_a = pd.DataFrame(
        {
            "sum": [sum_n, sum_p],
            "mean": [
                float(a[a < 0].mean()) if (a < 0).any() else np.nan,
                float(a[a > 0].mean()) if (a > 0).any() else np.nan,
            ],
            "share": [abs(sum_n) / tot, abs(sum_p) / tot],
        },
        index=["negative", "positive"],
    )
    panel_b = ind[["alpha", "g", "beta", "F", "share_sd"]].corr()
    panel_c = (
        cell_df.groupby("period")["alpha"].agg(["sum", "mean"])
        if period
        else pd.DataFrame({"sum": [float(a_kt.sum())], "mean": [float(a_kt.mean())]})
    )
    pos = ind["alpha"] > 0
    ab = ind["alpha"] * ind["beta"]
    total_ab = float(ab.sum())
    panel_e = pd.DataFrame(
        {
            "alpha_weighted_sum": [float(ab[~pos].sum()), float(ab[pos].sum())],
            "share_of_beta": [
                float(ab[~pos].sum()) / total_ab,
                float(ab[pos].sum()) / total_ab,
            ],
            "mean_beta": [
                float(ind.loc[~pos, "beta"].mean()),
                float(ind.loc[pos, "beta"].mean()),
            ],
        },
        index=["negative", "positive"],
    )

    grid = np.round(
        (
            np.asarray(ci_grid, float)
            if ci_grid is not None
            else np.arange(-100, 101) / 10.0
        ),
        12,
    )
    Y0 = df[y].to_numpy(float)
    rows = []
    for k in list(ind.index[:top]):
        if period:
            gk = (
                df[period]
                .map({t: shock_key.loc[(t, k)] for t in periods})
                .to_numpy(float)
            )
        else:
            gk = np.full(n, float(shock_key.loc[k]))
        zk = df[k].to_numpy(float) * gk
        V = Y0[:, None] - np.outer(X0, grid)
        tstats = _stata_tstats(np.column_stack([zk, Wc]), V, w, groups)
        df_t = int(groups.max()) if groups is not None else n - Wc.shape[1] - 1
        p = 2 * stats.t.sf(np.abs(tstats), df_t)
        acc = grid[p > alpha]
        lo = float(acc.min()) if acc.size else np.nan
        hi = float(acc.max()) if acc.size else np.nan
        bounded = acc.size > 0 and lo > grid.min() and hi < grid.max()
        rows.append(
            dict(
                industry=k,
                alpha=float(ind.loc[k, "alpha"]),
                g=float(ind.loc[k, "g"]),
                beta=float(ind.loc[k, "beta"]),
                ci_lower=lo,
                ci_upper=hi,
                ci_bounded=bool(bounded),
                share=float(ind.loc[k, "share"]),
                # GPSS print the share in percent ("Ind Share")
                share_pct=100.0 * float(ind.loc[k, "share"]),
            )
        )
    panel_d = pd.DataFrame(rows).set_index("industry")
    return dict(
        cells=cell_df,
        industries=ind,
        panel_a=panel_a,
        panel_b=panel_b,
        panel_c=panel_c,
        panel_d=panel_d,
        panel_e=panel_e,
        beta=float(np.nansum(a_kt * b_kt)),
    )
