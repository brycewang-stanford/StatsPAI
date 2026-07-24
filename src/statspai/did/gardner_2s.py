r"""
Gardner (2021) two-stage DID estimator (a.k.a. ``did2s``).

The **two-stage DID** method of Gardner (2021) recovers the ATT under staggered
treatment adoption by a two-step regression:

1. **Stage 1 — Fit FE model on untreated rows only.**
   Using observations where the unit is *not yet* treated, regress the outcome
   on unit and time fixed effects (plus any covariates):

       Y_it = alpha_i + lambda_t + X_it' beta + e_it    for (i, t) untreated.

2. **Stage 2 — Residualise + regress on treatment.**
   Construct the residualised outcome  Y_tilde_it = Y_it - (predicted from
   Stage 1), and fit a pooled regression on treatment dummies (either a single
   ATT or an event-study by relative time):

       Y_tilde_it = tau * D_it + u_it.

The default standard errors (``vce='analytic'``) cluster the Stage-2 residuals
by unit but do **not** account for the variance from estimating the Stage-1
fixed effects, so they understate uncertainty (≈0.78 coverage at a nominal 95%
level in simulations). Pass ``vce='bootstrap'`` to resample whole clusters and
re-run both stages for valid inference. The estimator closely parallels the
Borusyak-Jaravel-Spiess (2024) imputation estimator numerically, but the
two-step regression framing makes event studies and covariate interactions
trivial.

References
----------
Gardner, J. (2021).  "Two-stage differences in differences."
    *Working paper*, University of Mississippi.  arXiv:2207.05943.
Butts, K. and Gardner, J. (2022).  "did2s: Two-Stage Difference-in-Differences."
    *R Journal*, 14(3), 162-173. [@gardner2022stage]
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core._bootstrap import bootstrap_se as _bootstrap_se
from ..core.results import CausalResult

__all__ = ["gardner_did", "did_2stage"]


def _gardner_cluster_bootstrap(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: List[str],
    event_study: bool,
    horizon: Optional[List[int]],
    cluster: str,
    names: List[str],
    n_boot: int,
    seed: int,
) -> Tuple[Dict[str, float], float]:
    """Pairs-cluster bootstrap of the *full* Gardner two-step procedure.

    The analytic Stage-2 SE clusters the second-stage residuals as if the
    imputed counterfactual were known data; it therefore omits the variance
    contributed by estimating the Stage-1 fixed effects. Resampling whole
    clusters and re-running both stages restores valid coverage (the SE the
    module docstring refers to). Returns ``(per_coef_se, overall_att_se)``.
    """
    rng = np.random.default_rng(seed)
    clusters = pd.unique(df[cluster])
    n_g = len(clusters)
    rows_by_cluster = {c: df[df[cluster] == c] for c in clusters}
    ctrl = controls if controls else None

    n_boot_int = int(n_boot)
    # Pre-size to the full attempt count so failed replicates stay NaN and
    # bootstrap_se can surface the failure rate loudly (CLAUDE.md §3.7)
    # instead of silently computing the SD over survivors only.
    boot_overall = np.full(n_boot_int, np.nan, dtype=float)
    boot_coefs: Dict[str, np.ndarray] = {
        nm: np.full(n_boot_int, np.nan, dtype=float) for nm in names
    }
    for b in range(n_boot_int):
        drawn = rng.choice(n_g, size=n_g, replace=True)
        parts = []
        for j, ci in enumerate(drawn):
            sub = rows_by_cluster[clusters[ci]].copy()
            # Fresh, globally-unique ids so a cluster drawn twice (and the
            # units inside it) do not collide in the re-estimation.
            sub[group] = sub[group].astype(str) + f"__b{j}"
            sub["__bcl"] = j
            parts.append(sub)
        bd = pd.concat(parts, ignore_index=True)
        try:
            r = gardner_did(
                bd,
                y=y,
                group=group,
                time=time,
                first_treat=first_treat,
                controls=ctrl,
                event_study=event_study,
                horizon=horizon,
                cluster="__bcl",
                vce="none",
            )
        except Exception:
            continue  # replicate stays NaN; bootstrap_se tracks the failure
        if np.isfinite(r.estimate):
            boot_overall[b] = float(r.estimate)
        es = r.model_info.get("event_study") if event_study else None
        if es:
            for nm in names:
                boot_coefs[nm][b] = float(es["coef"].get(nm, np.nan))

    overall_se = _bootstrap_se(boot_overall, label="did.gardner.overall")
    se_dict: Dict[str, float] = {}
    for nm in names:
        se_dict[nm] = _bootstrap_se(boot_coefs[nm], label=f"did.gardner[{nm}]")
    if not event_study and names:
        se_dict[names[0]] = overall_se
    return se_dict, overall_se


def _cluster_vcov(
    X: np.ndarray,
    resid: np.ndarray,
    cluster: np.ndarray,
) -> np.ndarray:
    """Liang-Zeger cluster-robust variance for an OLS coefficient vector."""
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    clusters = np.unique(cluster)
    G = len(clusters)
    meat = np.zeros((k, k))
    for g in clusters:
        mask = cluster == g
        xg = X[mask]
        eg = resid[mask]
        s = xg.T @ eg
        meat += np.outer(s, s)
    if G > 1 and n > k:
        dof = G / (G - 1) * (n - 1) / (n - k)
    else:
        dof = 1.0
    return np.asarray(dof * xtx_inv @ meat @ xtx_inv, dtype=float)


def _build_fe_design(
    unit: np.ndarray,
    time: np.ndarray,
    X: Optional[np.ndarray],
    *,
    u_levels: Optional[np.ndarray] = None,
    t_levels: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (intercept, unit dummies minus one, time dummies minus one, X).

    When ``u_levels``/``t_levels`` are provided, builds the design against that
    reference set of levels (useful for prediction on a different sample).
    """
    if u_levels is None:
        u_levels = np.unique(unit)
    if t_levels is None:
        t_levels = np.unique(time)

    n = len(unit)
    # Intercept
    intercept = np.ones((n, 1))
    # Unit dummies drop first level
    D_u = np.zeros((n, max(len(u_levels) - 1, 0)))
    for j, lvl in enumerate(u_levels[1:]):
        D_u[:, j] = (unit == lvl).astype(float)
    # Time dummies drop first level
    D_t = np.zeros((n, max(len(t_levels) - 1, 0)))
    for j, lvl in enumerate(t_levels[1:]):
        D_t[:, j] = (time == lvl).astype(float)

    parts = [intercept, D_u, D_t]
    if X is not None and X.size > 0:
        parts.append(X)
    A = np.hstack(parts)
    return A, u_levels, t_levels


def gardner_did(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]] = None,
    event_study: bool = False,
    horizon: Optional[List[int]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    vce: str = "analytic",
    n_boot: int = 199,
    boot_seed: int = 0,
) -> CausalResult:
    """Gardner (2021) two-stage DID estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    y : str
        Outcome column name.
    group : str
        Unit (panel-id) column.
    time : str
        Time column.
    first_treat : str
        First-treatment-period column.  Never-treated units should be encoded
        as ``0``, ``NaN``, or ``+inf``.
    controls : list of str, optional
        Additional covariates included in both stages.
    event_study : bool, default False
        If True, Stage 2 reports coefficients by relative time
        ``k = t - first_treat_i``.
    horizon : list of int, optional
        Relative-time leads/lags to report when ``event_study=True``;
        defaults to ``range(-5, 6)`` intersected with available support.
    cluster : str, optional
        Cluster variable for Stage-2 SEs.  Defaults to ``group``.
    alpha : float, default 0.05
        Two-sided CI level.
    vce : {'analytic', 'bootstrap'}, default 'analytic'
        Standard-error mode. ``'analytic'`` clusters the Stage-2 residuals
        (fast) but ignores the variance from estimating the Stage-1 fixed
        effects and is **anti-conservative** (empirically ~0.78 coverage at a
        nominal 95% level); a ``UserWarning`` recommends ``'bootstrap'``.
        ``'bootstrap'`` resamples whole clusters and re-runs the full two-step
        procedure (Gardner 2021 / ``did2s``), substantially improving coverage
        (≈0.90 vs ≈0.78 in simulations; it approaches nominal as the number of
        clusters grows). Point estimates are identical either way.
    n_boot : int, default 199
        Number of cluster-bootstrap replications when ``vce='bootstrap'``.
    boot_seed : int, default 0
        Seed for the cluster bootstrap (deterministic results).

    Returns
    -------
    CausalResult
        ``.estimate`` is the overall ATT; ``.model_info['event_study']``
        carries the event-study dict when requested.  Supplies ``.summary()``,
        ``.cite()``, and is compatible with ``sp.outreg2()``.

    Notes
    -----
    Identification requires the usual staggered-DID conditions (parallel
    trends, no anticipation) plus a linear two-way FE + additive covariate
    structure for the untreated potential outcome.  Stage-2 standard errors
    cluster by unit — bootstrapping the whole two-step procedure gives a
    conservative covariance when covariate models are heavy.

    References
    ----------
    Gardner, J. (2022). Two-stage differences in differences. Working paper.
    [@gardner2022stage]

    Examples
    --------
    Staggered panel with a never-treated group (``first_treat = 0``).
    ``sp.did_2stage`` is an alias of this function.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n_units, n_periods = 40, 8
    >>> unit = np.repeat(np.arange(n_units), n_periods)
    >>> time = np.tile(np.arange(1, n_periods + 1), n_units)
    >>> first = np.where(unit < 20, 5, 0)  # 0 = never treated
    >>> d = ((first > 0) & (time >= first)).astype(float)
    >>> y = (0.5 * unit + 0.3 * time + 2.0 * d
    ...      + rng.normal(0, 0.5, unit.size))
    >>> df = pd.DataFrame(
    ...     {"y": y, "unit": unit, "time": time, "g": first}
    ... )
    >>> res = sp.gardner_did(
    ...     df, y="y", group="unit", time="time", first_treat="g"
    ... )
    >>> round(res.estimate, 2)  # true ATT = 2.0
    2.03
    """
    if vce not in ("analytic", "bootstrap", "none"):
        raise ValueError(f"vce must be 'analytic', 'bootstrap', or 'none'; got {vce!r}")
    if controls is None:
        controls = []
    df = data.copy()
    for col in [y, group, time, first_treat] + controls:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data")

    df[y] = pd.to_numeric(df[y], errors="coerce")
    df = df.dropna(subset=[y, group, time, first_treat]).reset_index(drop=True)

    if cluster is None:
        cluster = group
    elif cluster not in df.columns:
        raise ValueError(f"cluster column '{cluster}' not found")

    ft = df[first_treat].to_numpy(dtype=float)
    t_arr = df[time].to_numpy(dtype=float)
    treated_now = np.array(
        [(np.isfinite(fi) and fi > 0 and ti >= fi) for fi, ti in zip(ft, t_arr)]
    )
    df["_D"] = treated_now.astype(float)

    # ── Stage 1: FE + covariate regression on untreated rows ────── #
    untreated_mask = ~treated_now
    if untreated_mask.sum() < 10:
        raise ValueError("Not enough untreated observations for Stage 1 (<10).")

    unit_all = df[group].to_numpy()
    time_all = df[time].to_numpy()
    X_all = df[controls].to_numpy(dtype=float) if controls else np.zeros((len(df), 0))

    # Stage-1 fit: design against ALL units/times seen in the data; rows from
    # untreated subset only.  Units that appear only in treated rows simply
    # get a zero dummy — they contribute nothing to the untreated fit but can
    # still be predicted (via intercept + time FE only).
    u_levels = np.unique(unit_all)
    t_levels = np.unique(time_all)

    A_un, _, _ = _build_fe_design(
        unit_all[untreated_mask],
        time_all[untreated_mask],
        X_all[untreated_mask] if controls else None,
        u_levels=u_levels,
        t_levels=t_levels,
    )
    y_un = df.loc[untreated_mask, y].to_numpy(dtype=float)

    coefs, *_ = np.linalg.lstsq(A_un, y_un, rcond=None)

    # Predict counterfactual Y(0) for all rows using Stage-1 coefficients.
    A_full, _, _ = _build_fe_design(
        unit_all,
        time_all,
        X_all if controls else None,
        u_levels=u_levels,
        t_levels=t_levels,
    )
    y_all_arr = df[y].to_numpy(dtype=float)
    y_hat_0 = A_full @ coefs
    y_tilde = y_all_arr - y_hat_0

    # ── Stage 2: recover treatment effects from the imputed gap ──── #
    # Overall ATT: clustered OLS of ỹ on the treatment indicator, with
    # an intercept to absorb any mean residual in the untreated rows.
    # Event study: direct within-(cohort × relative-time) averaging of ỹ
    # — the Borusyak-Jaravel-Spiess style — to avoid the reference-
    # category contamination bias that a Stage-2 dummy regression would
    # introduce (the "baseline" in a dummy regression lumps never-treated
    # units together with treated units outside the event-study horizon,
    # pulling every coefficient toward the residual mean).
    cl = df[cluster].to_numpy()
    if event_study:
        rel_time = np.where(
            np.isfinite(ft) & (ft > 0),
            t_arr - ft,
            np.inf,  # never-treated → excluded
        )
        if horizon is None:
            support = np.unique(rel_time[np.isfinite(rel_time)])
            horizon = [int(k) for k in support if -5 <= int(k) <= 5]
            if 0 not in horizon:
                horizon.append(0)
            horizon = sorted(set(horizon))

        names, est_list, se_list, count_list = [], [], [], []
        for k in horizon:
            key = f"D_k{int(k):+d}"
            names.append(key)
            mask = rel_time == k
            n_k = int(mask.sum())
            count_list.append(n_k)
            if n_k == 0:
                est_list.append(float("nan"))
                se_list.append(float("nan"))
                continue
            y_k = y_tilde[mask]
            coef_k = float(np.mean(y_k))
            # Cluster-robust SE of the within-bin mean
            cl_k = cl[mask]
            uniq = np.unique(cl_k)
            G = len(uniq)
            if G > 1:
                # SE of the unweighted mean over n rows, allowing cluster
                # correlation: Var(mean) ≈ (1/n²) Σ_g (Σ_{i∈g} (y_ki - coef))²
                sq = 0.0
                for g in uniq:
                    idx = cl_k == g
                    sq += float(np.sum(y_k[idx] - coef_k)) ** 2
                var_k = sq / (len(y_k) ** 2)
                se_k = float(np.sqrt(max(var_k, 0.0)))
            else:
                se_k = float(np.std(y_k, ddof=1) / np.sqrt(len(y_k)))
            est_list.append(coef_k)
            se_list.append(se_k)
        est = np.array(est_list)
        se = np.array(se_list)
        coef_dict = dict(zip(names, est))
        se_dict = dict(zip(names, se))
        count_dict = dict(zip(names, count_list))
    else:
        X2 = df["_D"].to_numpy(dtype=float).reshape(-1, 1)
        names = ["ATT"]
        design2 = np.column_stack([np.ones(len(y_tilde)), X2])
        coef2, *_ = np.linalg.lstsq(design2, y_tilde, rcond=None)
        resid2 = y_tilde - design2 @ coef2
        V = _cluster_vcov(design2, resid2, cl)
        se_full = np.sqrt(np.clip(np.diag(V), 0, None))
        est = coef2[1:]
        se = se_full[1:]
        coef_dict = dict(zip(names, est))
        se_dict = dict(zip(names, se))

    # Standard-error mode. 'analytic' (default) clusters the Stage-2 residuals
    # only; 'bootstrap' resamples whole clusters and re-runs both stages;
    # 'none' skips the warning (used internally by the bootstrap to avoid
    # recursion).
    boot_overall_se: Optional[float] = None
    if vce == "bootstrap":
        se_dict, boot_overall_se = _gardner_cluster_bootstrap(
            df,
            y,
            group,
            time,
            first_treat,
            controls,
            event_study,
            horizon if event_study else None,
            cluster,
            names,
            n_boot,
            boot_seed,
        )
    elif vce == "analytic":
        warnings.warn(
            "gardner_did: the default analytic standard error clusters the "
            "Stage-2 residuals and ignores the variance from estimating the "
            "Stage-1 fixed effects, so it understates uncertainty "
            "(empirically ~0.78 coverage at a nominal 95% level). For valid "
            "inference pass vce='bootstrap' (a cluster bootstrap of the full "
            "two-step procedure).",
            UserWarning,
            stacklevel=2,
        )

    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci = {
        k: (coef_dict[k] - z * se_dict[k], coef_dict[k] + z * se_dict[k]) for k in names
    }

    if event_study:
        post_keys = [
            k
            for k in names
            if int(k.split("k")[1]) >= 0
            and np.isfinite(coef_dict[k])
            and count_dict.get(k, 0) > 0
        ]
        if post_keys:
            # Treated-observation-weighted mean of the post-treatment coefs
            # (the did2s aggregated-ATT convention). An *unweighted* mean
            # disagrees with the non-event-study ATT whenever the horizons have
            # unbalanced support / heterogeneous effects.
            w = np.array([count_dict[k] for k in post_keys], dtype=float)
            wn = w / w.sum()
            coefs_post = np.array([coef_dict[k] for k in post_keys], dtype=float)
            att_overall = float(np.dot(wn, coefs_post))
            if vce == "bootstrap" and boot_overall_se is not None:
                # Bootstrapping the overall ATT directly accounts for the
                # cross-horizon correlation the analytic aggregation ignores.
                att_se = float(boot_overall_se)
            else:
                # SE of the weighted average (horizons treated as independent;
                # use vce='bootstrap' for the correlation-aware SE).
                ses_post = np.array([se_dict[k] for k in post_keys], dtype=float)
                att_se = float(np.sqrt(np.sum((wn * ses_post) ** 2)))
        else:
            att_overall, att_se = float("nan"), float("nan")
    else:
        att_overall = float(coef_dict["ATT"])
        att_se = float(se_dict["ATT"])

    pvalue = (
        float(2 * (1 - sp_stats.norm.cdf(abs(att_overall / att_se))))
        if att_se > 0
        else float("nan")
    )

    n_units = int(df[group].nunique())
    n_treated_units = int(df.loc[treated_now, group].nunique())

    model_info = {
        "method": "Gardner 2021 two-stage DID",
        "vce": vce,
        "n_obs": int(len(df)),
        "n_units": n_units,
        "n_treated_units": n_treated_units,
        "alpha": alpha,
        "stage1_n": int(untreated_mask.sum()),
        "event_study": (
            {
                "horizon": names,
                "coef": coef_dict,
                "se": se_dict,
                "ci": ci,
            }
            if event_study
            else None
        ),
        "citation": (
            "Gardner, J. (2021). Two-stage differences in differences. "
            "arXiv:2207.05943. Butts & Gardner (2022), R Journal 14(3)."
        ),
    }

    _result = CausalResult(
        method="Gardner 2021 two-stage DID (did2s)",
        estimand="ATT",
        estimate=att_overall,
        se=att_se,
        pvalue=pvalue,
        ci=(att_overall - z * att_se, att_overall + z * att_se),
        alpha=alpha,
        n_obs=int(len(df)),
        model_info=model_info,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.gardner_did",
            params={
                "y": y,
                "group": group,
                "time": time,
                "first_treat": first_treat,
                "controls": controls,
                "event_study": event_study,
                "horizon": horizon,
                "cluster": cluster,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# Convenience alias aligned with the R package ``did2s``.
did_2stage = gardner_did
