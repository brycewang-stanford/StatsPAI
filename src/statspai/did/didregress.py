"""Two-way fixed-effects DID as Stata ``didregress`` / ``xtdidregress``.

The model is

    y = group effects + time effects + covariates + ATET * D + error,

with ``D`` the 0/1 treatment-status indicator (it switches on when a group
is treated). The two Stata commands fit the same coefficient and differ in
how the variance counts the absorbed effects:

* ``didregress`` (repeated cross-sections, effects for ``group``) counts
  the group dummies among the regressors, as ``regress`` with dummies or
  ``areg`` does.
* ``xtdidregress`` (panel, effects for the panel unit) uses the ``xtreg, fe``
  convention, which does not.

Both cluster on ``group`` and refer statistics to a t with ``G - 1`` degrees
of freedom.

The two post-estimation tests of ``estat ptrends`` and ``estat granger`` are
fitted with the model and returned in ``model_info``; ``sp.estat(result,
'ptrends')`` and ``sp.estat(result, 'granger')`` read them. Like Stata they
need a single treatment date.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["didregress"]

# Verbatim from paper.bib (wooldridge2010econometric).
CausalResult._CITATIONS["twfe_didregress"] = (
    "@book{wooldridge2010econometric,\n"
    "  title={Econometric Analysis of Cross Section and Panel Data},\n"
    "  author={Wooldridge, Jeffrey M.},\n"
    "  publisher={MIT Press},\n"
    "  edition={2nd},\n"
    "  year={2010},\n"
    "  isbn={978-0-262-23258-6}\n"
    "}"
)


def _within(columns: np.ndarray, effects: pd.DataFrame) -> np.ndarray:
    """Sweep the fixed effects out of each column."""
    from ..fast import demean

    out, keep = demean(columns, effects, drop_singletons=False, tol=1e-13)
    if not bool(np.all(keep)):  # pragma: no cover - singletons are kept
        raise DataInsufficient("didregress: fixed-effect sweep dropped rows.")
    return np.asarray(out, dtype=float)


def _cluster_fit(
    y: np.ndarray,
    X: np.ndarray,
    cluster: np.ndarray,
    n_absorbed: int,
    context: str,
) -> Dict[str, Any]:
    """OLS on swept data with the cluster-robust variance of the command.

    ``n_absorbed`` is the number of parameters the variance counts besides
    the columns of ``X``: the time effects and the constant, plus the group
    dummies for ``didregress``.
    """
    n, k = X.shape
    if np.linalg.matrix_rank(X) < k:
        raise DataInsufficient(
            f"{context}: the regressors are collinear after the group and "
            "time effects are removed.",
            recovery_hint=(
                "A covariate that is constant within group or within period "
                "is absorbed by the effects; drop it."
            ),
        )
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    codes, _ = pd.factorize(cluster)
    n_clusters = int(codes.max()) + 1
    scores = np.zeros((n_clusters, k))
    np.add.at(scores, codes, X * resid[:, None])
    dof = n - k - n_absorbed
    if dof <= 0 or n_clusters < 2:
        raise DataInsufficient(
            f"{context}: not enough observations or clusters for the "
            "cluster-robust variance.",
            diagnostics={"n": n, "parameters": k + n_absorbed, "clusters": n_clusters},
        )
    factor = n_clusters / (n_clusters - 1) * (n - 1) / dof
    vcov = factor * bread @ (scores.T @ scores) @ bread
    return {"beta": beta, "vcov": vcov, "n_clusters": n_clusters}


def _wald_f(fit: Dict[str, Any], positions: Sequence[int]) -> Dict[str, float]:
    idx = list(positions)
    b = fit["beta"][idx]
    V = fit["vcov"][np.ix_(idx, idx)]
    f_stat = float(b @ np.linalg.solve(V, b) / len(idx))
    df_denom = fit["n_clusters"] - 1
    return {
        "F": f_stat,
        "df": float(len(idx)),
        "df_denom": float(df_denom),
        "pvalue": float(stats.f.sf(f_stat, len(idx), df_denom)),
    }


def didregress(
    data: pd.DataFrame,
    y: str,
    treat: str,
    group: str,
    time: str,
    *,
    covariates: Optional[List[str]] = None,
    id: Optional[str] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """Difference-in-differences by two-way fixed effects, as Stata
    ``didregress`` and ``xtdidregress``.

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
        Outcome.
    treat : str
        Treatment status, 0/1, equal to 1 in the periods a group is treated.
        This is the second equation of the Stata command, not a
        treated-group indicator.
    group : str
        The level at which treatment is assigned. Group effects are fitted
        for it and it is the default cluster.
    time : str
        Time period.
    covariates : list of str, optional
        Controls of the outcome equation.
    id : str, optional
        Panel unit. Given, the model has unit effects and the variance
        follows ``xtreg, fe`` (``xtdidregress``). Omitted, the model has
        group effects counted as regressors (``didregress``).
    cluster : str, optional
        Cluster variable, ``group`` by default.
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        ``estimate`` is the ATET, with a t interval on ``G - 1`` degrees of
        freedom. ``detail`` lists the ATET and covariate coefficients.
        ``model_info['ptrends']`` and ``model_info['granger']`` hold the
        parallel-trends and anticipation tests (``F``, ``df``, ``df_denom``,
        ``pvalue``), or ``{'unavailable': reason}``.

    Notes
    -----
    This is one coefficient for all treated periods. With staggered
    adoption and effects that vary across cohorts or over time it is a
    weighted average with weights that can be negative; see
    ``sp.bacon_decomposition`` for the weights and ``sp.callaway_santanna``
    or ``sp.did_imputation`` for estimators that do not have the problem.

    ``ptrends`` adds a linear time trend for the treated groups, separately
    before and after treatment, and tests the pre-treatment slope. Time is
    centred at the last pre-treatment period. Stata's ``estat ptrends``
    agrees to about 3e-6: three ways of fitting this model in Stata
    (``estat ptrends``, ``areg`` with the trend on raw years, ``areg`` with
    centred time) differ from each other by that much, and the centred one
    equals the value here to 1e-12. ``granger`` adds a treated-group dummy
    for every pre-treatment period but the last and tests them jointly.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.mpdta()
    >>> df = df[df.first_treat.isin([0, 2006])].copy()
    >>> df["d"] = ((df.first_treat > 0) & (df.year >= df.first_treat)).astype(int)
    >>> res = sp.didregress(df, "lemp", "d", group="countyreal", time="year")
    >>> round(res.estimate, 4), round(res.se, 4)
    (-0.03, 0.0103)
    >>> round(sp.estat(res, "granger", print_results=False)["statistic"], 4)
    0.8619

    References
    ----------
    wooldridge2010econometric, goodmanbacon2021difference
    """
    context = "didregress" if id is None else "xtdidregress"
    covariates = list(covariates or [])
    cluster_col = cluster or group
    needed = [y, treat, group, time, *covariates]
    for extra in (id, cluster_col):
        if extra is not None and extra not in needed:
            needed.append(extra)
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"{context}: columns not in data: {missing}.")
    if not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility(f"{context}: alpha must lie in (0, 1).")

    clean = data[needed].dropna()
    n_dropped = len(data) - len(clean)
    if n_dropped:
        warnings.warn(
            f"{context}: dropped {n_dropped} rows with missing values.",
            UserWarning,
            stacklevel=2,
        )
    d = clean[treat].to_numpy(dtype=float)
    if not np.isin(d, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"{context}: treat must be a 0/1 treatment-status indicator.",
            recovery_hint="Build it as (treated group) * (period >= adoption).",
        )
    if d.min() == d.max():
        raise DataInsufficient(f"{context}: treat does not vary.")
    varies = clean.groupby([group, time])[treat].nunique()
    if (varies > 1).any():
        raise MethodIncompatibility(
            f"{context}: treat varies within {int((varies > 1).sum())} "
            "group-period cells; treatment is assigned by group.",
            recovery_hint="Pass the assignment level as group=.",
        )

    unit = id if id is not None else group
    if id is not None:
        if clean.duplicated([id, time]).any():
            raise MethodIncompatibility(
                f"{context}: repeated {id!r}-{time!r} observations; a panel "
                "has one row per unit and period.",
                recovery_hint="Drop id= for repeated cross-sections.",
            )
        if (clean.groupby(id)[group].nunique() > 1).any():
            raise MethodIncompatibility(
                f"{context}: a panel unit appears in more than one group."
            )
    effects = pd.DataFrame(
        {
            "unit": pd.factorize(clean[unit])[0],
            "time": pd.factorize(clean[time])[0],
        }
    )
    n_units = int(effects["unit"].max()) + 1
    n_periods = int(effects["time"].max()) + 1
    # parameters the variance counts besides the regressors: the constant
    # and T - 1 time effects, plus G - 1 group dummies for didregress
    n_absorbed = n_periods + (0 if id is not None else n_units - 1)
    cluster_values = clean[cluster_col].to_numpy()
    yv = clean[y].to_numpy(dtype=float)
    cov = clean[covariates].to_numpy(dtype=float) if covariates else None

    def fit(extra: Optional[np.ndarray] = None) -> Dict[str, Any]:
        cols = [d[:, None]]
        if extra is not None:
            cols.append(extra)
        if cov is not None:
            cols.append(cov)
        swept = _within(np.column_stack([yv[:, None], *cols]), effects)
        return _cluster_fit(
            swept[:, 0], swept[:, 1:], cluster_values, n_absorbed, context
        )

    main = fit()
    est = float(main["beta"][0])
    se = float(np.sqrt(main["vcov"][0, 0]))
    df_t = main["n_clusters"] - 1
    crit = float(stats.t.ppf(1 - alpha / 2, df_t))
    names = ["ATET"] + covariates
    ses = np.sqrt(np.diag(main["vcov"]))
    tstats = main["beta"] / ses
    detail = pd.DataFrame(
        {
            "term": names,
            "estimate": main["beta"],
            "se": ses,
            "t": tstats,
            "pvalue": 2 * stats.t.sf(np.abs(tstats), df_t),
            "ci_lower": main["beta"] - crit * ses,
            "ci_upper": main["beta"] + crit * ses,
        }
    )

    # ── estat ptrends / estat granger ─────────────────────────────────
    periods = np.sort(clean[time].unique())
    first_treated = clean.loc[d == 1].groupby(group)[time].min()
    ever = clean[group].isin(first_treated.index).to_numpy(dtype=float)
    tests: Dict[str, Dict[str, Any]] = {}
    reason = None
    if first_treated.nunique() > 1:
        reason = "treatment assignment times vary"
    else:
        start = first_treated.iloc[0]
        pre_periods = periods[periods < start]
        if len(pre_periods) < 2:
            reason = "fewer than two pre-treatment periods"
    if reason is not None:
        tests["ptrends"] = {"unavailable": reason}
        tests["granger"] = {"unavailable": reason}
    else:
        tv = clean[time].to_numpy()
        try:
            clock = tv.astype(float) - float(pre_periods[-1])
        except (TypeError, ValueError):
            clock = np.searchsorted(periods, tv).astype(float) - (len(pre_periods) - 1)
        post = (tv >= start).astype(float)
        trend = np.column_stack([ever * (1 - post) * clock, ever * post * clock])
        try:
            tests["ptrends"] = _wald_f(fit(trend), [1])
        except DataInsufficient:
            # one post-treatment period: its trend is the ATET itself
            tests["ptrends"] = _wald_f(fit(trend[:, :1]), [1])
        leads = np.column_stack([ever * (tv == p) for p in pre_periods[:-1]])
        tests["granger"] = _wald_f(fit(leads), range(1, 1 + leads.shape[1]))
        tests["ptrends"]["H0"] = "Linear trends are parallel"
        tests["granger"]["H0"] = "No effect in anticipation of treatment"

    tstat = est / se
    return CausalResult(
        method=f"Difference in differences (TWFE, {context})",
        estimand="ATET",
        estimate=est,
        se=se,
        pvalue=float(2 * stats.t.sf(abs(tstat), df_t)),
        ci=(est - crit * se, est + crit * se),
        alpha=alpha,
        n_obs=len(clean),
        detail=detail,
        model_info={
            "command": context,
            "n_groups": int(clean[group].nunique()),
            "n_clusters": int(main["n_clusters"]),
            "n_periods": n_periods,
            "df": float(df_t),
            "df_inference": float(df_t),
            "cluster": cluster_col,
            "n_dropped": int(n_dropped),
            "treatment_times": sorted(first_treated.unique().tolist()),
            "ptrends": tests["ptrends"],
            "granger": tests["granger"],
        },
        _citation_key="twfe_didregress",
    )
