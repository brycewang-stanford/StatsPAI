"""
de Chaisemartin & D'Haultfœuille (2020) DID_M estimator.

Estimates the effect of a binary treatment that can switch on AND off,
robust to heterogeneous treatment effects across groups and time periods.
Unlike Callaway & Sant'Anna (which assumes staggered adoption / no
treatment reversal), this estimator handles general treatment paths
where units can enter and exit treatment.

The DID_M estimator is a **pair-rollup**: for each consecutive period
pair ``(t-1, t)`` it computes a cell-level DID comparing "switchers"
(units whose treatment changed) to "stayers" (units whose treatment
did not change), then aggregates to an overall estimate using the
switcher count per cell as the weight.

**Scope note** — the ``dynamic=`` argument here is the pair-rollup
event-study extension (long differences relative to stayers at each
pair). It is **not** the dCDH (2024) ``did_multiplegt_dyn`` estimator,
which is a separate long-difference event-study with its own influence
function and a "not-yet-treated-at-horizon-l" control construction.
That estimator is tracked in ``docs/rfc/multiplegt_dyn.md`` and will
land as a separate ``sp.did_multiplegt_dyn`` function.

References
----------
de Chaisemartin, C. and D'Haultfoeuille, X. (2020).
"Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects."
*American Economic Review*, 110(9), 2964-2996. [@dechaisemartin2020two]

de Chaisemartin, C. and D'Haultfoeuille, X. (2022).
"Two-way fixed effects and differences-in-differences with heterogeneous
treatment effects: A survey."  *The Econometrics Journal*, 26(3), C1-C30.
[@dechaisemartin2022fixed]

de Chaisemartin, C. and D'Haultfoeuille, X. (2024).
"Difference-in-Differences Estimators of Intertemporal Treatment Effects."
*Review of Economics and Statistics*, forthcoming.  (Joint placebo test
and average cumulative effect, Section 3.) [@dechaisemartin2024difference]
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult

# ======================================================================
# Public API
# ======================================================================


def did_multiplegt(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]] = None,
    placebo: int = 0,
    dynamic: int = 0,
    cluster: Optional[str] = None,
    n_boot: int = 100,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    de Chaisemartin & D'Haultfœuille (2020) DID_M estimator.

    Estimates the effect of a binary treatment that can switch on AND off,
    robust to heterogeneous treatment effects across groups and time.

    **Note** — ``dynamic=H`` here extends the DID_M pair rollup to H
    horizons; it is not equivalent to the dCDH (2024) ``_dyn``
    event-study estimator. For the 2024 long-difference event-study with
    not-yet-treated controls per horizon, see ``sp.did_multiplegt_dyn``
    (tracked in ``docs/rfc/multiplegt_dyn.md``).

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format (one row per unit-period).
    y : str
        Outcome variable name.
    group : str
        Group (unit) identifier.
    time : str
        Time period variable.
    treatment : str
        Binary treatment indicator (0/1). Unlike Callaway-Sant'Anna, this
        is the *current* treatment status, not the first-treatment period.
        Units may switch treatment on and off.
    controls : list of str, optional
        Control variables. When provided, first-differences of the outcome
        are residualized on first-differences of the controls.
    placebo : int, default 0
        Number of placebo (pre-treatment) tests.  Placebo *l* checks
        whether switchers at *t* already differed from stayers between
        *t-l-1* and *t-l*.
    dynamic : int, default 0
        Number of dynamic (post-treatment) effect horizons to estimate.
        Dynamic effect *l* measures the long difference Y_{t+l} - Y_{t-1}
        for switchers at *t* relative to stayers.
    cluster : str, optional
        Variable for cluster bootstrap.  Defaults to ``group``.
    n_boot : int, default 100
        Number of bootstrap replications for standard errors.
    seed : int, optional
        Random seed for reproducibility.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        Result object with ``.summary()`` and ``.plot()`` methods.
        The ``detail`` DataFrame contains all (group, time)-level DID
        estimates; ``model_info`` stores event-study coefficients
        (placebo and dynamic effects) for plotting.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for county in range(12):
    ...     state = county % 3
    ...     g = int(rng.choice([4, 6, 0]))  # cohort; 0 = never treated
    ...     for year in range(1, 9):
    ...         on = 1 if (g != 0 and year >= g) else 0
    ...         rows.append({'county': county, 'state': state, 'year': year,
    ...                      'treated': on,
    ...                      'wage': county + 0.2 * year + 1.5 * on
    ...                              + rng.normal(0, 0.5)})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.did_multiplegt(
    ...     data=df, y="wage", group="county", time="year",
    ...     treatment="treated", placebo=1, dynamic=2,
    ...     cluster="state", n_boot=50, seed=42,
    ... )
    >>> bool(np.isfinite(result.estimate))
    True

    Notes
    -----
    The estimator proceeds as follows for each consecutive pair of
    periods (t-1, t):

    1. **Switchers**: units whose treatment status changed between t-1
       and t.
    2. **Stayers**: units whose treatment status did *not* change.
    3. DID_{g,t} = mean(DeltaY_switchers) - mean(DeltaY_stayers),
       where DeltaY = Y_t - Y_{t-1}.
    4. DID_M = weighted average of DID_{g,t} with weights proportional
       to the number of switchers in each cell.

    Standard errors are computed via cluster bootstrap (resampling
    clusters with replacement).
    """
    # ── Validate inputs ──────────────────────────────────────────── #
    df = data.copy()
    _required = [y, group, time, treatment]
    for col in _required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if controls is not None:
        for col in controls:
            if col not in df.columns:
                raise ValueError(f"Control column '{col}' not found in data.")

    cluster_var = cluster if cluster is not None else group

    # ── Sort and compute main estimate ───────────────────────────── #
    df = df.sort_values([group, time]).reset_index(drop=True)

    main = _estimate_did_m(df, y, group, time, treatment, controls)

    # ── Placebo effects ──────────────────────────────────────────── #
    placebo_results = []
    for lag_idx in range(1, placebo + 1):
        plac = _estimate_placebo(
            df,
            y,
            group,
            time,
            treatment,
            controls,
            lag=lag_idx,
        )
        placebo_results.append(plac)

    # ── Dynamic effects ──────────────────────────────────────────── #
    dynamic_results = []
    for horizon_idx in range(0, dynamic + 1):
        dyn = _estimate_dynamic(
            df,
            y,
            group,
            time,
            treatment,
            controls,
            horizon=horizon_idx,
        )
        dynamic_results.append(dyn)

    # ── Bootstrap standard errors ────────────────────────────────── #
    rng = np.random.default_rng(seed)
    clusters = df[cluster_var].unique()
    n_clusters = len(clusters)

    boot_main = np.full(n_boot, np.nan)
    boot_placebo = np.full((n_boot, max(placebo, 0)), np.nan) if placebo > 0 else None
    boot_dynamic = np.full((n_boot, dynamic + 1), np.nan) if dynamic >= 0 else None

    for b in range(n_boot):
        sampled = rng.choice(clusters, size=n_clusters, replace=True)
        # Build bootstrap sample preserving panel structure
        frames = []
        for j, c in enumerate(sampled):
            chunk = df[df[cluster_var] == c].copy()
            # Relabel group to avoid collisions from sampling with replacement
            chunk[group] = chunk[group].astype(str) + f"_b{j}"
            frames.append(chunk)
        bdf = pd.concat(frames, ignore_index=True)

        b_main = _estimate_did_m(bdf, y, group, time, treatment, controls)
        boot_main[b] = b_main["did_m"]

        if placebo > 0:
            assert boot_placebo is not None
            for lag_idx in range(1, placebo + 1):
                plac = _estimate_placebo(
                    bdf,
                    y,
                    group,
                    time,
                    treatment,
                    controls,
                    lag=lag_idx,
                )
                boot_placebo[b, lag_idx - 1] = plac["estimate"]

        if dynamic >= 0:
            assert boot_dynamic is not None
            for horizon_idx in range(0, dynamic + 1):
                dyn = _estimate_dynamic(
                    bdf,
                    y,
                    group,
                    time,
                    treatment,
                    controls,
                    horizon=horizon_idx,
                )
                boot_dynamic[b, horizon_idx] = dyn["estimate"]

    # ── Compute SEs, p-values, CIs ───────────────────────────────── #
    se_main = np.nanstd(boot_main, ddof=1)
    z_main = main["did_m"] / se_main if se_main > 0 else 0.0
    p_main = 2 * (1 - stats.norm.cdf(abs(z_main)))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_main = (main["did_m"] - z_crit * se_main, main["did_m"] + z_crit * se_main)

    # Placebo SEs
    placebo_out = []
    for lag_idx in range(placebo):
        est = placebo_results[lag_idx]["estimate"]
        se = (
            np.nanstd(boot_placebo[:, lag_idx], ddof=1)
            if boot_placebo is not None
            else 0.0
        )
        z = est / se if se > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        ci = (est - z_crit * se, est + z_crit * se)
        placebo_out.append(
            {
                "lag": -(lag_idx + 1),
                "estimate": est,
                "se": se,
                "pvalue": p,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }
        )

    # Dynamic SEs
    dynamic_out = []
    for horizon_idx in range(dynamic + 1):
        est = dynamic_results[horizon_idx]["estimate"]
        se = (
            np.nanstd(boot_dynamic[:, horizon_idx], ddof=1)
            if boot_dynamic is not None
            else 0.0
        )
        z = est / se if se > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        ci = (est - z_crit * se, est + z_crit * se)
        dynamic_out.append(
            {
                "horizon": horizon_idx,
                "estimate": est,
                "se": se,
                "pvalue": p,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }
        )

    # ── Build detail DataFrame ───────────────────────────────────── #
    detail_rows = main.get("cell_estimates", [])
    detail_df = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()

    # ── Event-study DataFrame for plotting ───────────────────────── #
    # CausalResult.summary() and .plot() expect columns:
    #   relative_time, att, se, pvalue, ci_lower, ci_upper
    es_rows = []
    for p_row in placebo_out:
        es_rows.append(
            {
                "relative_time": p_row["lag"],
                "att": p_row["estimate"],
                "se": p_row["se"],
                "pvalue": p_row["pvalue"],
                "ci_lower": p_row["ci_lower"],
                "ci_upper": p_row["ci_upper"],
                "type": "placebo",
            }
        )
    for d_row in dynamic_out:
        es_rows.append(
            {
                "relative_time": d_row["horizon"],
                "att": d_row["estimate"],
                "se": d_row["se"],
                "pvalue": d_row["pvalue"],
                "ci_lower": d_row["ci_lower"],
                "ci_upper": d_row["ci_upper"],
                "type": "dynamic",
            }
        )
    es_df = pd.DataFrame(es_rows) if es_rows else pd.DataFrame()

    # ── Joint placebo test + avg cumulative dynamic effect ───────── #
    joint_placebo = _joint_placebo_test(placebo_results, boot_placebo)
    avg_cumulative = _avg_cumulative_effect(
        dynamic_results,
        boot_dynamic,
        alpha,
    )

    # ── Assemble model_info ──────────────────────────────────────── #
    model_info: Dict[str, Any] = {
        "estimator": "de Chaisemartin-D'Haultfoeuille (2020)",
        "n_switching_cells": main["n_switching_cells"],
        "n_switchers": main["n_switchers"],
        "n_boot": n_boot,
        "seed": seed,
        "cluster_var": cluster_var,
        "controls": controls,
        "placebo": placebo_out,
        "dynamic": dynamic_out,
        "event_study": es_df,
        "joint_placebo_test": joint_placebo,
        "avg_cumulative_effect": avg_cumulative,
    }

    _result = CausalResult(
        method="de Chaisemartin-D'Haultfoeuille (2020)",
        estimand="ATT",
        estimate=float(main["did_m"]),
        se=float(se_main),
        pvalue=float(p_main),
        ci=ci_main,
        alpha=alpha,
        n_obs=len(data),
        detail=detail_df,
        model_info=model_info,
        _citation_key="did_multiplegt",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.did_multiplegt",
            params={
                "y": y,
                "group": group,
                "time": time,
                "treatment": treatment,
                "controls": controls,
                "placebo": placebo,
                "dynamic": dynamic,
                "cluster": cluster,
                "n_boot": n_boot,
                "seed": seed,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ======================================================================
# Internal helpers
# ======================================================================


def _residualize(dy: np.ndarray, dX: np.ndarray) -> np.ndarray:
    """Residualize dy on dX via OLS."""
    if dX.shape[1] == 0:
        return dy
    # Add intercept
    ones = np.ones((dX.shape[0], 1))
    X = np.hstack([ones, dX])
    try:
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        return np.asarray(dy - X @ beta, dtype=float)
    except np.linalg.LinAlgError:
        return dy


def _estimate_did_m(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Core DID_M estimator: weighted average of cell-level DID estimates.

    For each period t, identify switchers and stayers, compute DID_gt.
    """
    periods = sorted(df[time].unique())
    cell_estimates = []
    total_switchers = 0

    for idx in range(1, len(periods)):
        t_prev, t_curr = periods[idx - 1], periods[idx]

        df_prev = df[df[time] == t_prev][
            [group, y, treatment] + (controls or [])
        ].copy()
        df_curr = df[df[time] == t_curr][
            [group, y, treatment] + (controls or [])
        ].copy()

        df_prev = df_prev.rename(
            columns={y: f"{y}_prev", treatment: f"{treatment}_prev"},
        )
        if controls:
            df_prev = df_prev.rename(columns={c: f"{c}_prev" for c in controls})

        merged = df_curr.merge(df_prev, on=group, how="inner")
        if merged.empty:
            continue

        merged["_switched"] = merged[treatment] != merged[f"{treatment}_prev"]
        merged["_dy"] = merged[y] - merged[f"{y}_prev"]

        # de Chaisemartin--D'Haultfoeuille (2020): the cell-level DID must
        # condition on the BASELINE treatment d_{t-1}. Switchers are compared
        # only to stayers that shared the same period-(t-1) treatment value, and
        # switch-OFF cells (baseline 1 -> 0) enter with a flipped sign so the
        # estimand is the effect of *gaining* treatment. Pooling across baselines
        # (i) contaminates the control trend with already-treated stayers and
        # (ii) mixes switch-on / switch-off effects under a single sign -- both
        # bias DID_M (see Paper-DiD-JAE/replication/did_multiplegt/).
        for base in sorted(merged[f"{treatment}_prev"].unique()):
            grp = merged[merged[f"{treatment}_prev"] == base]
            switchers = grp[grp["_switched"]]
            stayers = grp[~grp["_switched"]]

            n_switch = len(switchers)
            if n_switch == 0 or len(stayers) == 0:
                continue

            dy_switch = switchers["_dy"].values.copy()
            dy_stay = stayers["_dy"].values.copy()

            # Residualize on controls within the baseline cell
            if controls:
                dX_switch = np.column_stack(
                    [
                        switchers[c].values - switchers[f"{c}_prev"].values
                        for c in controls
                    ]
                )
                dX_stay = np.column_stack(
                    [stayers[c].values - stayers[f"{c}_prev"].values for c in controls]
                )
                dX_all = np.vstack([dX_switch, dX_stay])
                dy_all = np.concatenate([dy_switch, dy_stay])
                resid = _residualize(dy_all, dX_all)
                dy_switch = resid[:n_switch]
                dy_stay = resid[n_switch:]

            # Binary treatment: baseline 0 -> switch ON (+); baseline 1 -> OFF (-).
            turned_on = base == 0
            sign = 1.0 if turned_on else -1.0

            did_gt = sign * (np.mean(dy_switch) - np.mean(dy_stay))

            cell_estimates.append(
                {
                    "time": t_curr,
                    "time_prev": t_prev,
                    "baseline_treatment": float(base),
                    "n_switchers": n_switch,
                    "n_stayers": len(stayers),
                    "did_gt": did_gt,
                    "mean_dy_switch": float(np.mean(dy_switch)),
                    "mean_dy_stay": float(np.mean(dy_stay)),
                    "direction": "on" if turned_on else "off",
                }
            )
            total_switchers += n_switch

    if total_switchers == 0:
        return {
            "did_m": 0.0,
            "n_switching_cells": 0,
            "n_switchers": 0,
            "cell_estimates": [],
        }

    # Weighted average
    did_m = sum(
        ce["did_gt"] * ce["n_switchers"] / total_switchers for ce in cell_estimates
    )

    return {
        "did_m": float(did_m),
        "n_switching_cells": len(cell_estimates),
        "n_switchers": total_switchers,
        "cell_estimates": cell_estimates,
    }


def _estimate_placebo(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    lag: int,
) -> Dict[str, Any]:
    """
    Placebo test at lag *l* (de Chaisemartin--D'Haultfoeuille 2020).

    For switchers at time t, the first-difference placebo compares the
    pre-treatment outcome change over (t-l-1, t-l) for switchers vs stayers,
    CONDITIONING on the baseline treatment d_{t-1} (switchers are compared only
    to same-baseline stayers; switch-off cells enter with a flipped sign). The
    sign convention mirrors Stata ``did_multiplegt_old``.
    """
    periods = sorted(df[time].unique())
    dpiv = df.pivot_table(index=group, columns=time, values=treatment, aggfunc="first")
    ypiv = df.pivot_table(index=group, columns=time, values=y, aggfunc="first")

    estimates: List[float] = []
    weights: List[int] = []

    for i in range(1, len(periods)):
        t_prev, t_curr = periods[i - 1], periods[i]  # switching step (t-1 -> t)
        e_idx = i - lag  # placebo "end" period index
        b_idx = e_idx - 1  # placebo "start" period index
        if b_idx < 0:
            continue
        t_end, t_base = periods[e_idx], periods[b_idx]

        common = dpiv.index[dpiv[t_prev].notna() & dpiv[t_curr].notna()]
        for base in sorted(set(dpiv.loc[common, t_prev].dropna().unique())):
            in_base = common[dpiv.loc[common, t_prev] == base]
            sw = in_base[dpiv.loc[in_base, t_curr] != base]  # switchers from baseline
            st = in_base[dpiv.loc[in_base, t_curr] == base]  # stayers at baseline
            sw = sw[ypiv.loc[sw, t_end].notna() & ypiv.loc[sw, t_base].notna()]
            st = st[ypiv.loc[st, t_end].notna() & ypiv.loc[st, t_base].notna()]
            if len(sw) == 0 or len(st) == 0:
                continue
            ps = float((ypiv.loc[sw, t_end] - ypiv.loc[sw, t_base]).mean())
            pt = float((ypiv.loc[st, t_end] - ypiv.loc[st, t_base]).mean())
            sign = 1.0 if base == 0 else -1.0
            estimates.append(
                -sign * (ps - pt)
            )  # -sign matches Stata placebo convention
            weights.append(len(sw))

    if not estimates:
        return {"estimate": 0.0, "n_cells": 0}

    total_w = sum(weights)
    weighted_est = sum(e * w / total_w for e, w in zip(estimates, weights))

    return {"estimate": float(weighted_est), "n_cells": len(estimates)}


def _estimate_dynamic(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    horizon: int,
) -> Dict[str, Any]:
    """
    Dynamic effect at horizon *l* (de Chaisemartin--D'Haultfoeuille 2020,
    dynamic-robust).

    For switchers at time t (relative to baseline d_{t-1}), the long-difference
    Y_{t+l} - Y_{t-1} is compared to that of ROBUST stayers: units that share the
    baseline treatment AND keep it unchanged over the whole window [t, t+l] (so
    the control trend is not contaminated by units treated during the horizon).
    Switch-off cells enter with a flipped sign.
    """
    periods = sorted(df[time].unique())
    dpiv = df.pivot_table(index=group, columns=time, values=treatment, aggfunc="first")
    ypiv = df.pivot_table(index=group, columns=time, values=y, aggfunc="first")

    estimates: List[float] = []
    weights: List[int] = []

    for i in range(1, len(periods)):
        t_prev, t_curr = periods[i - 1], periods[i]
        f_idx = i + horizon
        if f_idx >= len(periods):
            continue
        t_future = periods[f_idx]
        window = periods[i : f_idx + 1]  # t_curr .. t_future

        common = dpiv.index[dpiv[t_prev].notna() & dpiv[t_curr].notna()]
        for base in sorted(set(dpiv.loc[common, t_prev].dropna().unique())):
            in_base = common[dpiv.loc[common, t_prev] == base]
            sw = in_base[dpiv.loc[in_base, t_curr] != base]
            # robust stayers: keep the baseline treatment over the whole window
            stable = (dpiv.loc[in_base, window] == base).all(axis=1)
            st = in_base[stable]
            sw = sw[ypiv.loc[sw, t_future].notna() & ypiv.loc[sw, t_prev].notna()]
            st = st[ypiv.loc[st, t_future].notna() & ypiv.loc[st, t_prev].notna()]
            if len(sw) == 0 or len(st) == 0:
                continue
            lds = float((ypiv.loc[sw, t_future] - ypiv.loc[sw, t_prev]).mean())
            ldt = float((ypiv.loc[st, t_future] - ypiv.loc[st, t_prev]).mean())
            sign = 1.0 if base == 0 else -1.0
            estimates.append(sign * (lds - ldt))
            weights.append(len(sw))

    if not estimates:
        return {"estimate": 0.0, "n_cells": 0}

    total_w = sum(weights)
    weighted_est = sum(e * w / total_w for e, w in zip(estimates, weights))

    return {"estimate": float(weighted_est), "n_cells": len(estimates)}


# ======================================================================
# Joint inference
# ======================================================================


def _joint_placebo_test(
    placebo_results: List[Dict[str, Any]],
    boot_placebo: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """Joint Wald test of H0: all placebo lags equal zero.

    Uses the bootstrap draws (cluster-level) to estimate the covariance
    across lags, so the test accounts for correlation between consecutive
    pre-periods.  Follows dCDH (2024, §3.3) "joint placebo" recommendation.
    """
    if not placebo_results or boot_placebo is None:
        return None

    est = np.array([r["estimate"] for r in placebo_results], dtype=float)
    boot = np.asarray(boot_placebo, dtype=float)  # (n_boot, k)
    # Drop all-NaN columns / rows before covariance.
    mask_cols = ~np.all(np.isnan(boot), axis=0)
    if not mask_cols.any():
        return None
    est = est[mask_cols]
    boot = boot[:, mask_cols]
    valid_rows = ~np.any(np.isnan(boot), axis=1)
    if valid_rows.sum() < boot.shape[1] + 1:
        return None
    boot = boot[valid_rows]

    cov = np.cov(boot, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    k = est.shape[0]
    cov_reg = cov + np.eye(k) * 1e-10

    try:
        W = float(est @ np.linalg.solve(cov_reg, est))
    except np.linalg.LinAlgError:
        W = float(est @ np.linalg.pinv(cov_reg) @ est)

    pval = float(1 - stats.chi2.cdf(W, k))
    return {"statistic": W, "df": int(k), "pvalue": pval}


def _avg_cumulative_effect(
    dynamic_results: List[Dict[str, Any]],
    boot_dynamic: Optional[np.ndarray],
    alpha: float,
) -> Optional[Dict[str, Any]]:
    """Average cumulative dynamic effect: mean of dynamic[0..L].

    Bootstrap SE is computed over the per-draw average across horizons,
    preserving the cross-horizon covariance (dCDH 2024 §3.4).
    """
    if not dynamic_results or boot_dynamic is None:
        return None
    est_vec = np.array([r["estimate"] for r in dynamic_results], dtype=float)
    avg_est = float(np.mean(est_vec))

    boot = np.asarray(boot_dynamic, dtype=float)
    per_draw_avg = np.nanmean(boot, axis=1)
    per_draw_avg = per_draw_avg[np.isfinite(per_draw_avg)]
    if per_draw_avg.size < 2:
        return {
            "estimate": avg_est,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "pvalue": np.nan,
            "n_horizons": len(est_vec),
        }
    se = float(np.std(per_draw_avg, ddof=1))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = avg_est / se if se > 0 else 0.0
    return {
        "estimate": avg_est,
        "se": se,
        "ci_lower": avg_est - z_crit * se,
        "ci_upper": avg_est + z_crit * se,
        "pvalue": float(2 * (1 - stats.norm.cdf(abs(z)))),
        "n_horizons": int(len(est_vec)),
    }
