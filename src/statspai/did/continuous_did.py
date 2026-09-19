"""
Difference-in-Differences with Continuous Treatment (heuristic modes).

Provides three heuristic DiD estimators for settings where treatment
intensity varies continuously across units:

- ``method='twfe'`` — TWFE OLS with a ``dose × post`` interaction.
- ``method='att_gt'`` — dose-quantile binned 2×2 DID versus the ``dose=0``
  arm (or the lowest-dose quantile as fallback), with bootstrap SE.
- ``method='dose_response'`` — local-linear regression of
  ``ΔY = Y_post − Y_pre`` on baseline dose, reporting the average
  derivative.

**Not** the Callaway, Goodman-Bacon & Sant'Anna (2024) ATT(d|g,t) /
ACRT(d|g,t) estimator. That one is :func:`statspai.cgs_continuous_did`,
which is pinned against the authors' own ``contdid`` package. The default
``method='att_gt'`` here is a dose-bin heuristic, not the CGS group-time
estimand.

.. deprecated:: 1.21.0
   ``method='cgs'`` is superseded by :func:`statspai.cgs_continuous_did`.
   It was an outcome-regression MVP with a bootstrap standard error and
   unresolved ``[待核验]`` formula details; the replacement is the real
   estimator with parity evidence. It warns and will be removed after one
   minor release.

References
----------
Callaway, B., Goodman-Bacon, A. & Sant'Anna, P.H.C. (2024).
"Difference-in-Differences with a Continuous Treatment."
[@callaway2024difference] — the target of the MVP ``method='cgs'``.

de Chaisemartin, C. & D'Haultfœuille, X. (2018).
"Fuzzy Differences-in-Differences." [@dechaisemartin2018fuzzy].
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core._bootstrap import bootstrap_se as _bootstrap_se
from ..core.results import CausalResult


@accepts_aliases(_strict=True, unit="id", covariates="controls")
def continuous_did(
    data: pd.DataFrame,
    y: str,
    dose: str,
    time: str,
    id: str,
    post: Optional[str] = None,
    t_pre: Optional[int] = None,
    t_post: Optional[int] = None,
    method: str = "att_gt",
    n_quantiles: int = 5,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """
    Difference-in-Differences with continuous treatment (heuristic).

    Three heuristic DiD estimators for continuous-dose panels. This is
    **not** a faithful implementation of Callaway, Goodman-Bacon &
    Sant'Anna (2024); ``method='att_gt'`` is a dose-bin 2×2 DID rollup,
    not the CGS ATT(d|g,t) group-time estimand. A paper-faithful
    ``method='cgs'`` is tracked in ``docs/rfc/continuous_did_cgs.md``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    y : str
        Outcome variable.
    dose : str
        Continuous treatment/dose variable.
    time : str
        Time period variable.
    id : str
        Unit identifier.
    post : str, optional
        Binary post-treatment indicator. If None, inferred from t_pre/t_post.
    t_pre : int, optional
        Last pre-treatment period.
    t_post : int, optional
        First post-treatment period.
    method : str, default 'att_gt'
        Estimation method:

        - ``'att_gt'``: Dose-quantile 2×2 DID rollup (heuristic). SEs from
          a unit bootstrap drawn jointly for every bin and the pooled
          estimate.
        - ``'twfe'``: TWFE OLS with ``dose × post`` interaction; unit and
          period effects absorbed exactly (also on unbalanced panels), SEs
          with ``fixest::feols`` degrees of freedom (pinned in
          ``tests/reference_parity/test_did_synth_didvar_parity.py``).
        - ``'dose_response'``: local-linear regression of ΔY on baseline
          dose; the SE is a unit bootstrap of the average derivative.
        - ``'cgs'``: Callaway-Goodman-Bacon-Sant'Anna (2024) ATT(d|g,t) /
          ACRT(d|g,t) MVP with ``[待核验]`` markers on paper formulas;
          see ``docs/rfc/continuous_did_cgs.md``. **Not yet reference-
          parity** with R ``contdid``; OR only, no DR/IPW, bootstrap SE.
    n_quantiles : int, default 5
        Number of dose quantiles for discretization.
    controls : list of str, optional
        Control variables. Used by ``method='twfe'`` (and ``'cgs'``); the
        ``'att_gt'`` and ``'dose_response'`` heuristics ignore them and warn.
    cluster : str, optional
        Cluster variable for SE (``method='twfe'`` only).
    n_boot : int, default 500
        Bootstrap replications for SE.
    alpha : float, default 0.05
    seed : int, optional

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> # Two-period panel: half the workers get zero training (control
    >>> # arm), the rest receive a positive, continuously varying dose.
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> dose = np.where(np.arange(n) < n // 2, 0.0,
    ...                 rng.uniform(1, 10, size=n))
    >>> rows = []
    >>> for i in range(n):
    ...     for year in (2019, 2020):
    ...         post = 1 if year == 2020 else 0
    ...         wage = 20 + 0.3 * dose[i] * post + rng.normal(0, 1)
    ...         rows.append({'worker_id': i, 'year': year,
    ...                      'training_hours': dose[i], 'wage': wage})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.continuous_did(
    ...     df, y='wage', dose='training_hours',
    ...     time='year', id='worker_id', t_pre=2019, t_post=2020,
    ...     n_boot=200, seed=0,
    ... )
    >>> bool(np.isfinite(result.estimate))
    True

    References
    ----------
    callaway2024difference, dechaisemartin2018fuzzy
    """
    rng = np.random.default_rng(seed)
    df = data.copy()

    # Determine pre/post
    if post is None:
        if t_pre is not None and t_post is not None:
            df["_post"] = (df[time] >= t_post).astype(int)
        else:
            times = sorted(df[time].unique())
            mid = len(times) // 2
            df["_post"] = (df[time] >= times[mid]).astype(int)
        post_col = "_post"
    else:
        post_col = post

    if method == "twfe":
        return _continuous_did_twfe(
            df,
            y,
            dose,
            time,
            id,
            post_col,
            controls,
            cluster,
            alpha,
        )
    elif method == "cgs":
        warnings.warn(
            "continuous_did(method='cgs') is deprecated in favour of "
            "sp.cgs_continuous_did, which implements the same Callaway, "
            "Goodman-Bacon & Sant'Anna estimator and IS pinned against the "
            "authors' contdid package (curves and both overall quantities at "
            "1e-12). This mode is an outcome-regression MVP with a bootstrap "
            "SE and unresolved [待核验] formula details; it will be removed "
            "after one minor release. See MIGRATION.md.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _continuous_did_cgs(
            df,
            y=y,
            dose=dose,
            time=time,
            unit=id,
            t_pre=t_pre,
            t_post=t_post,
            controls=controls,
            n_boot=n_boot,
            alpha=alpha,
            rng=rng,
        )
    elif method == "dose_response":
        return _continuous_did_dose_response(
            df,
            y,
            dose,
            time,
            id,
            post_col,
            controls,
            n_quantiles,
            n_boot,
            alpha,
            rng,
        )
    else:
        return _continuous_did_att_gt(
            df,
            y,
            dose,
            time,
            id,
            post_col,
            controls,
            n_quantiles,
            n_boot,
            alpha,
            rng,
        )


def _twfe_fe_dof(df: pd.DataFrame, id: str, time: str) -> int:
    """Rank of the unit + period dummy block: ``N + T - c``.

    ``c`` is the number of connected components of the bipartite unit-period
    graph (1 for any panel in which every unit is observed in a period that
    links it to the rest), matching how ``fixest`` / ``reghdfe`` count the
    absorbed parameters of two fixed effects.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    u_codes, u_uniq = pd.factorize(df[id], sort=False)
    t_codes, t_uniq = pd.factorize(df[time], sort=False)
    n_u, n_t = len(u_uniq), len(t_uniq)
    graph = coo_matrix(
        (np.ones(len(df)), (u_codes, n_u + t_codes)), shape=(n_u + n_t, n_u + n_t)
    )
    n_comp, _ = connected_components(graph, directed=False)
    return int(n_u + n_t - n_comp)


def _continuous_did_twfe(
    df: pd.DataFrame,
    y: str,
    dose: str,
    time: str,
    id: str,
    post: str,
    controls: Optional[List[str]],
    cluster: Optional[str],
    alpha: float,
) -> CausalResult:
    """TWFE: ``y_it = a_i + l_t + b * dose_it * post_t (+ controls) + e_it``.

    Unit and period effects are absorbed by alternating projections, which is
    exact on unbalanced panels (a one-pass ``y - ybar_i - ybar_t + ybar``
    transform is only exact on balanced ones). Degrees of freedom follow
    ``fixest::feols`` defaults: the iid variance divides by ``n - K`` with
    ``K`` counting the slopes and every absorbed fixed-effect parameter; the
    cluster-robust variance uses ``G/(G-1) * (n-1)/(n-K)`` with ``K`` counting
    only fixed effects not nested in the cluster (``ssc(fixef.K = "nested")``).
    Inference uses the normal distribution.
    """
    from ..fast.demean import demean as _hdfe_demean
    from ._core import fe_dof_not_nested as _fe_dof_not_nested

    df = df.copy()
    df["_dose_post"] = df[dose] * df[post]
    x_cols = ["_dose_post"] + list(controls or [])
    needed = [y, id, time] + x_cols + ([cluster] if cluster else [])
    valid = df[needed].notna().all(axis=1).to_numpy()
    for c in [y] + x_cols:
        valid &= np.isfinite(df[c].to_numpy(dtype=float))
    dfv = df.loc[valid].reset_index(drop=True)

    mat = dfv[[y] + x_cols].to_numpy(dtype=float)
    dem, info = _hdfe_demean(
        mat,
        [dfv[id].to_numpy(), dfv[time].to_numpy()],
        drop_singletons=False,
        tol=1e-14,
        max_iter=100_000,
        backend="numpy",
    )
    if not info.converged:
        warnings.warn(
            "continuous_did(method='twfe'): the two-way within transform did "
            "not converge; the slope may be inaccurate.",
            RuntimeWarning,
            stacklevel=3,
        )
    y_v = dem[:, 0]
    X_v = dem[:, 1:]
    n, k = X_v.shape

    beta = np.linalg.lstsq(X_v, y_v, rcond=None)[0]
    resid = y_v - X_v @ beta
    XtX_inv = np.linalg.inv(X_v.T @ X_v)

    if cluster:
        clusters = dfv[cluster].to_numpy()
        unique_clusters = np.unique(clusters)
        n_cl = len(unique_clusters)
        meat = np.zeros((k, k))
        for cl in unique_clusters:
            cl_mask = clusters == cl
            score = X_v[cl_mask].T @ resid[cl_mask]
            meat += np.outer(score, score)
        K = k + _fe_dof_not_nested(dfv, [id, time], cluster)
        correction = n_cl / (n_cl - 1) * (n - 1) / (n - K)
        var_cov = correction * XtX_inv @ meat @ XtX_inv
    else:
        K = k + _twfe_fe_dof(dfv, id, time)
        sigma2 = np.sum(resid**2) / (n - K)
        var_cov = sigma2 * XtX_inv

    tau = float(beta[0])
    se = float(np.sqrt(var_cov[0, 0]))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    p_val = 2 * stats.norm.sf(abs(tau / se)) if se > 0 else np.nan

    return CausalResult(
        method="Continuous DID (TWFE)",
        estimand="Dose-response coefficient",
        estimate=tau,
        se=se,
        pvalue=p_val,
        ci=(tau - z_crit * se, tau + z_crit * se),
        alpha=alpha,
        n_obs=int(n),
        model_info={
            "dose_variable": dose,
            "n_units": int(dfv[id].nunique()),
            "n_periods": int(dfv[time].nunique()),
            "dof_K": int(K),
            "cluster_var": cluster,
        },
    )


def _continuous_did_att_gt(
    df: pd.DataFrame,
    y: str,
    dose: str,
    time: str,
    id: str,
    post: str,
    controls: Optional[List[str]],
    n_quantiles: int,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> CausalResult:
    """Dose-bin 2x2 DID rollup.

    Units with positive baseline dose are cut into dose quantiles; each bin is
    compared with the ``dose == 0`` units (or, when there are none, with the
    lowest bin) by a 2x2 difference of row means. The pooled estimate is the
    treated-count-weighted mean of the bin DIDs.

    Standard errors come from a unit (cluster) bootstrap that resamples the
    analysis units **with multiplicity** and recomputes every bin and the
    pooled estimate on the same draw, so the pooled SE carries the covariance
    induced by the shared comparison group.
    """
    if controls:
        warnings.warn(
            "continuous_did(method='att_gt'): `controls` are not used by the "
            "dose-bin 2x2 rollup and are ignored. Use method='twfe' for a "
            "covariate-adjusted slope.",
            UserWarning,
            stacklevel=3,
        )
    # Get dose at baseline (pre-period)
    pre_data = df[df[post] == 0]
    dose_baseline = pre_data.groupby(id)[dose].mean()

    # Define dose groups (quantiles)
    quantile_edges = np.quantile(
        dose_baseline[dose_baseline > 0].values,
        np.linspace(0, 1, n_quantiles + 1),
    )
    quantile_edges = np.unique(quantile_edges)
    if len(quantile_edges) < 2:
        quantile_edges = np.array([dose_baseline.min(), dose_baseline.max()])

    dose_groups = pd.cut(
        dose_baseline,
        bins=quantile_edges,
        labels=False,
        include_lowest=True,
    )

    # Comparison arm: dose == 0, else the lowest dose bin.
    untreated_ids = dose_baseline[dose_baseline == 0].index
    fallback_control = len(untreated_ids) == 0
    if fallback_control:
        lowest_g = dose_groups.dropna().min()
        control_ids = dose_groups[dose_groups == lowest_g].index
    else:
        lowest_g = None
        control_ids = untreated_ids

    bins: List[tuple] = []
    for g in sorted(dose_groups.dropna().unique()):
        group_ids = dose_groups[dose_groups == g].index
        if len(group_ids) < 2:
            continue
        if fallback_control and g == lowest_g:
            continue
        bins.append((g, group_ids))

    # Per-unit sums / counts of the outcome in each period block; a 2x2 DID
    # of row means is a ratio of these, so a resample with multiplicities is
    # just a weighted ratio.
    units = pd.Index(dose_baseline.index)
    yv = df[y].astype(float)
    is_post = df[post] == 1
    sums_post = (
        yv[is_post].groupby(df.loc[is_post, id]).sum().reindex(units, fill_value=0.0)
    )
    cnt_post = (
        yv[is_post].groupby(df.loc[is_post, id]).count().reindex(units, fill_value=0)
    )
    sums_pre = (
        yv[~is_post].groupby(df.loc[~is_post, id]).sum().reindex(units, fill_value=0.0)
    )
    cnt_pre = (
        yv[~is_post].groupby(df.loc[~is_post, id]).count().reindex(units, fill_value=0)
    )
    S1, C1 = sums_post.to_numpy(float), cnt_post.to_numpy(float)
    S0, C0 = sums_pre.to_numpy(float), cnt_pre.to_numpy(float)
    ctrl_mask = units.isin(control_ids)
    bin_masks = [units.isin(ids) for _, ids in bins]

    def _did(mult: np.ndarray, tmask: np.ndarray) -> float:
        with np.errstate(invalid="ignore", divide="ignore"):
            t1 = (mult[tmask] @ S1[tmask]) / (mult[tmask] @ C1[tmask])
            t0 = (mult[tmask] @ S0[tmask]) / (mult[tmask] @ C0[tmask])
            c1 = (mult[ctrl_mask] @ S1[ctrl_mask]) / (mult[ctrl_mask] @ C1[ctrl_mask])
            c0 = (mult[ctrl_mask] @ S0[ctrl_mask]) / (mult[ctrl_mask] @ C0[ctrl_mask])
        return float((t1 - t0) - (c1 - c0))

    def _all(mult: np.ndarray) -> np.ndarray:
        atts = np.array([_did(mult, m) for m in bin_masks], dtype=float)
        n_tr = np.array([mult[m].sum() for m in bin_masks], dtype=float)
        ok = np.isfinite(atts) & (n_tr > 0)
        pooled = (
            float(np.sum(n_tr[ok] * atts[ok]) / n_tr[ok].sum()) if ok.any() else np.nan
        )
        return np.append(atts, pooled)

    ones = np.ones(len(units))
    point = _all(ones)

    in_sample = ctrl_mask.copy()
    for m in bin_masks:
        in_sample |= m
    pos = np.flatnonzero(in_sample)
    boot = np.full((n_boot, len(bins) + 1), np.nan)
    if len(bins) > 0 and len(pos) > 0:
        for b in range(n_boot):
            draw = rng.choice(pos, size=len(pos), replace=True)
            mult = np.bincount(draw, minlength=len(units)).astype(float)
            boot[b] = _all(mult)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    results_rows = []
    for j, (g, group_ids) in enumerate(bins):
        att = float(point[j])
        se = _bootstrap_se(boot[:, j], label="did.continuous.dose_att")
        results_rows.append(
            {
                "dose_group": int(g),
                "dose_midpoint": dose_baseline[group_ids].mean(),
                "att": att,
                "se": se,
                "ci_lower": att - z_crit * se,
                "ci_upper": att + z_crit * se,
                "p_value": (2 * stats.norm.sf(abs(att / se)) if se > 0 else np.nan),
                "n_treated": len(group_ids),
                "n_control": len(control_ids),
            }
        )

    results_df = pd.DataFrame(results_rows)

    if len(results_df) > 0:
        pooled_att = float(point[-1])
        pooled_se = _bootstrap_se(boot[:, -1], label="did.continuous.pooled_att")
    else:
        pooled_att, pooled_se = np.nan, np.nan

    p_val = 2 * stats.norm.sf(abs(pooled_att / pooled_se)) if pooled_se > 0 else np.nan

    return CausalResult(
        method="Continuous DID (dose-bin heuristic)",
        estimand="Sample-weighted mean of dose-bin 2x2 DIDs (not CGS 2024 ATT(d|g,t))",
        estimate=pooled_att,
        se=pooled_se,
        pvalue=p_val,
        ci=(pooled_att - z_crit * pooled_se, pooled_att + z_crit * pooled_se),
        alpha=alpha,
        n_obs=len(df),
        detail=results_df if len(results_df) > 0 else None,
        model_info={
            "dose_variable": dose,
            "n_dose_groups": len(results_df),
            "n_units": df[id].nunique(),
            "control_arm": "lowest_dose_bin" if fallback_control else "zero_dose",
            "n_boot": n_boot,
        },
    )


def _continuous_did_dose_response(
    df: pd.DataFrame,
    y: str,
    dose: str,
    time: str,
    id: str,
    post: str,
    controls: Optional[List[str]],
    n_quantiles: int,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> CausalResult:
    """Average derivative of a local-linear fit of ``dY_i`` on baseline dose.

    ``dY_i`` is the unit's mean post outcome minus its mean pre outcome. The
    headline is the grid-average of the finite-difference slope of the fitted
    curve. Its standard error is a unit bootstrap of that same statistic (grid
    and bandwidth held at their full-sample values), not the pointwise SE of
    the fitted *level*, which measures a different quantity.
    """
    if controls:
        warnings.warn(
            "continuous_did(method='dose_response'): `controls` are not used "
            "by the local-linear dose-response fit and are ignored.",
            UserWarning,
            stacklevel=3,
        )
    # Compute unit-level DID: dY = Y_i,post - Y_i,pre
    pre_y = df[df[post] == 0].groupby(id)[y].mean()
    post_y = df[df[post] == 1].groupby(id)[y].mean()
    common_ids = pre_y.index.intersection(post_y.index)

    delta_y = post_y[common_ids] - pre_y[common_ids]
    dose_vals = df[df[post] == 0].groupby(id)[dose].mean()[common_ids]

    from ..nonparametric.lpoly import lpoly as _lpoly

    temp_df = pd.DataFrame({"delta_y": delta_y.values, "dose": dose_vals.values})
    temp_df = temp_df.dropna().reset_index(drop=True)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    try:
        lp_result = _lpoly(temp_df, y="delta_y", x="dose", degree=1, n_grid=50)
    except (ValueError, np.linalg.LinAlgError) as exc:
        warnings.warn(
            "continuous_did(method='dose_response'): the local-linear fit "
            f"failed ({exc}); falling back to a global linear slope of dY on "
            "dose with its OLS standard error.",
            RuntimeWarning,
            stacklevel=3,
        )
        from scipy.stats import linregress

        slope, intercept, _, p_val, se_slope = linregress(
            temp_df["dose"].to_numpy(), temp_df["delta_y"].to_numpy()
        )
        return CausalResult(
            method="Continuous DID (Dose-Response)",
            estimand="Average marginal effect",
            estimate=slope,
            se=se_slope,
            pvalue=p_val,
            ci=(slope - z_crit * se_slope, slope + z_crit * se_slope),
            alpha=alpha,
            n_obs=len(temp_df),
            model_info={"fallback": "linregress", "fallback_reason": str(exc)},
        )

    grid = lp_result.grid
    bw = lp_result.bandwidth

    def _avg_slope(fitted: np.ndarray) -> float:
        with np.errstate(invalid="ignore"):
            return float(np.nanmean(np.diff(fitted) / np.diff(grid)))

    avg_effect = _avg_slope(lp_result.fitted)

    boot = np.full(n_boot, np.nan)
    n_units = len(temp_df)
    for b in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        bdf = temp_df.iloc[idx]
        try:
            bfit = _lpoly(bdf, y="delta_y", x="dose", degree=1, grid=grid, bandwidth=bw)
        except (ValueError, np.linalg.LinAlgError):
            continue  # stays NaN; bootstrap_se reports the failure count
        boot[b] = _avg_slope(bfit.fitted)
    avg_se = _bootstrap_se(boot, label="did.continuous.dose_response")

    p_val = 2 * stats.norm.sf(abs(avg_effect / avg_se)) if avg_se > 0 else np.nan

    return CausalResult(
        method="Continuous DID (Dose-Response)",
        estimand="Average marginal effect",
        estimate=avg_effect,
        se=avg_se,
        pvalue=p_val,
        ci=(avg_effect - z_crit * avg_se, avg_effect + z_crit * avg_se),
        alpha=alpha,
        n_obs=len(temp_df),
        model_info={
            "dose_response_grid": grid.tolist(),
            "dose_response_fitted": lp_result.fitted.tolist(),
            "dose_response_pointwise_se": lp_result.se.tolist(),
            "bandwidth": float(bw),
            "n_units": len(common_ids),
            "n_boot": n_boot,
            "se_method": "unit bootstrap of the average derivative",
        },
    )


# ----------------------------------------------------------------------
# method='cgs' — Callaway-Goodman-Bacon-Sant'Anna (2024) MVP
# ----------------------------------------------------------------------


def _continuous_did_cgs(
    df: pd.DataFrame,
    *,
    y: str,
    dose: str,
    time: str,
    unit: str,
    t_pre: Optional[int],
    t_post: Optional[int],
    controls: Optional[List[str]],
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> CausalResult:
    """Callaway-Goodman-Bacon-Sant'Anna (2024) continuous DiD — MVP.

    **This is an MVP, not paper-parity.** See
    ``docs/rfc/continuous_did_cgs.md`` for the full design + test plan.
    All paper-specific identification formulas are flagged [待核验]
    below.

    MVP identification [待核验 — CGS 2024 §3-§4]
    -------------------------------------------
    Assumes a 2-period design (pre, post) with continuous dose d ∈ [0, ∞)
    on the treated arm and d = 0 on the untreated arm. Under strong
    parallel trends:

        ATT(d | post) ≈ E[Y_post − Y_pre | D = d]
                         − E[Y_post − Y_pre | D = 0]

    ATT(d) is estimated as a function of d by local-linear smoother over
    the unit-level long differences. ACRT(d) = d ATT(d) / dd is the
    derivative, approximated by first-differencing the smoother.

    MVP limitations
    ---------------
    - Only 2-period (pre, post) designs. Full CGS group-time ATT(d|g,t)
      across multiple cohorts / periods is not in this MVP.
    - OR only (no IPW / DR / influence-function variance). SE via
      pairs bootstrap.
    - No explicit strong-parallel-trends diagnostic yet.
    """
    # 2-period design: pre is everything with time < t_post (or == t_pre);
    # post is everything with time >= t_post (or == t_post if given).
    times = sorted(df[time].unique())
    if t_pre is None:
        t_pre_val = times[0]
    else:
        t_pre_val = t_pre
    if t_post is None:
        t_post_val = times[-1]
    else:
        t_post_val = t_post

    pre = df[df[time] == t_pre_val]
    post = df[df[time] == t_post_val]
    merged = post[[unit, y, dose] + (controls or [])].merge(
        pre[[unit, y]].rename(columns={y: f"{y}_pre"}),
        on=unit,
        how="inner",
    )
    if merged.empty:
        return CausalResult(
            method="Continuous DID (CGS 2024 MVP) [待核验]",
            estimand="ATT(d) averaged over treated dose support",
            estimate=np.nan,
            se=np.nan,
            pvalue=np.nan,
            ci=(np.nan, np.nan),
            alpha=alpha,
            n_obs=0,
        )
    merged["_dy"] = merged[y] - merged[f"{y}_pre"]

    untreated = merged[merged[dose] == 0]
    treated = merged[merged[dose] > 0]

    if len(untreated) < 2:
        return CausalResult(
            method="Continuous DID (CGS 2024 MVP) [待核验]",
            estimand="ATT(d) averaged over treated dose support",
            estimate=np.nan,
            se=np.nan,
            pvalue=np.nan,
            ci=(np.nan, np.nan),
            alpha=alpha,
            n_obs=int(len(merged)),
            model_info={
                "warning": "No dose=0 control units; paper requires never-treated.",
            },
        )
    if len(treated) < 3:
        return CausalResult(
            method="Continuous DID (CGS 2024 MVP) [待核验]",
            estimand="ATT(d) averaged over treated dose support",
            estimate=np.nan,
            se=np.nan,
            pvalue=np.nan,
            ci=(np.nan, np.nan),
            alpha=alpha,
            n_obs=int(len(merged)),
            model_info={
                "warning": "Too few treated units with dose>0 to fit ATT(d).",
            },
        )

    # Control baseline: average ΔY among dose=0 units.
    control_dy = untreated["_dy"].mean()

    # ATT(d) on the treated support: local-linear smoother of (Δy - control_dy)
    # on d. Use the Nadaraya-Watson-ish average for a robust MVP.
    treated_sorted = treated.sort_values(dose).reset_index(drop=True)
    treated_sorted["_att_raw"] = treated_sorted["_dy"] - control_dy

    # Grid over the treated dose support (quartile-spaced for evenness)
    n_grid = 50
    d_vals = treated_sorted[dose].values.astype(float)
    d_lo, d_hi = np.percentile(d_vals, [5, 95])
    grid = np.linspace(d_lo, d_hi, n_grid)

    att_curve = _local_linear_curve(
        x=d_vals,
        y=treated_sorted["_att_raw"].values.astype(float),
        grid=grid,
    )

    # ACRT(d) = derivative of att_curve
    acrt_curve = np.gradient(att_curve, grid)

    # Summary scalars
    att_overall = float(np.nanmean(treated_sorted["_att_raw"].values))
    acrt_overall = float(np.nanmean(acrt_curve))

    # Bootstrap SE for ATT_overall and ACRT_overall
    boot_att = np.full(n_boot, np.nan)
    boot_acrt = np.full(n_boot, np.nan)
    n_merged = len(merged)
    for b in range(n_boot):
        idx = rng.integers(0, n_merged, size=n_merged)
        sample = merged.iloc[idx]
        u_b = sample[sample[dose] == 0]
        t_b = sample[sample[dose] > 0]
        if len(u_b) < 2 or len(t_b) < 3:
            continue
        ctrl_b = u_b["_dy"].mean()
        ta_b = t_b["_dy"].values - ctrl_b
        boot_att[b] = float(np.nanmean(ta_b))
        # ACRT: fit local linear on bootstrap sample and take mean slope
        d_b = t_b[dose].values.astype(float)
        order = np.argsort(d_b)
        try:
            curve_b = _local_linear_curve(
                x=d_b[order],
                y=ta_b[order],
                grid=grid,
            )
            boot_acrt[b] = float(np.nanmean(np.gradient(curve_b, grid)))
        except Exception:
            continue  # replicate stays NaN; bootstrap_se tracks the failure

    se_att = _bootstrap_se(boot_att, label="did.continuous.att_overall")
    se_acrt = _bootstrap_se(boot_acrt, label="did.continuous.acrt_overall")
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    p_att = (
        float(2 * stats.norm.sf(abs(att_overall / se_att)))
        if (se_att > 0 and np.isfinite(se_att))
        else np.nan
    )
    ci_att = (
        (att_overall - z_crit * se_att, att_overall + z_crit * se_att)
        if se_att > 0
        else (np.nan, np.nan)
    )

    detail_df = pd.DataFrame(
        {
            "dose": grid,
            "att_d": att_curve,
            "acrt_d": acrt_curve,
        }
    )

    return CausalResult(
        method="Continuous DID (CGS 2024 MVP) [待核验 — OR only, 2-period design]",
        estimand="ATT(d) averaged over treated support",
        estimate=att_overall,
        se=se_att,
        pvalue=p_att,
        ci=ci_att,
        alpha=alpha,
        n_obs=int(len(merged)),
        detail=detail_df,
        model_info={
            "acrt_overall": acrt_overall,
            "acrt_se": se_acrt,
            "n_treated": int(len(treated)),
            "n_control_d0": int(len(untreated)),
            "grid_range": (float(d_lo), float(d_hi)),
            "warning": (
                "MVP: OR only, 2-period design. Full CGS (2024) ATT(d|g,t) "
                "across cohorts + IPW/DR + analytical IF variance pending. "
                "See docs/rfc/continuous_did_cgs.md for the roadmap."
            ),
        },
        _citation_key="callaway2024difference",
    )


def _local_linear_curve(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    bandwidth: Optional[float] = None,
) -> np.ndarray:
    """Local-linear smoother on x, y evaluated at grid points.

    Gaussian kernel, Silverman rule-of-thumb bandwidth by default. Used
    for the ATT(d) curve in the CGS MVP.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if bandwidth is None:
        n = len(x)
        std = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        scale = min(std, iqr / 1.349) if iqr > 0 else std
        bandwidth = 1.06 * scale * n ** (-0.2) if scale > 0 else 0.1

    out = np.empty(len(grid))
    for i, g in enumerate(grid):
        u = (x - g) / bandwidth
        w = np.exp(-0.5 * u * u)
        if w.sum() < 1e-10:
            out[i] = np.nan
            continue
        # Weighted local linear: fit y = a + b*(x - g) with weights w.
        X = np.column_stack([np.ones_like(x), x - g])
        W = w[:, None]
        try:
            XtWX = X.T @ (W * X)
            XtWy = X.T @ (W[:, 0] * y)
            beta = np.linalg.solve(XtWX, XtWy)
            out[i] = float(beta[0])
        except np.linalg.LinAlgError:
            out[i] = float(np.average(y, weights=w)) if w.sum() > 0 else np.nan
    return out
