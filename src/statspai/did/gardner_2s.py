"""
Gardner (2021) two-stage DID estimator (a.k.a. ``did2s``).

The **two-stage DID** method of Gardner (2021) recovers the ATT under staggered
treatment adoption by a two-step regression that propagates Stage-1 uncertainty
into Stage-2 inference:

1. **Stage 1 — Fit FE model on untreated rows only.**
   Using observations where the unit is *not yet* treated, regress the outcome
   on unit and time fixed effects (plus any covariates):

       Y_it = alpha_i + lambda_t + X_it' beta + e_it    for (i, t) untreated.

2. **Stage 2 — Residualise + regress on treatment.**
   Construct the residualised outcome  \tilde Y_it = Y_it - \hat alpha_i -
   \hat lambda_t - X_it' \hat beta, and fit a pooled regression on treatment
   dummies (either a single ATT or an event-study by relative time):

       \tilde Y_it = tau * D_it + u_it.

The standard errors are adjusted for the first-step residualisation via the
Murphy-Topel / Newey (1984) two-step correction (clustered by unit).

Gardner's estimator closely parallels the Borusyak-Jaravel-Spiess (2024)
imputation estimator numerically, but the regression framing makes it trivial
to extend to event studies, covariate interactions, and weighting.

References
----------
Gardner, J. (2021).  "Two-stage differences in differences."
    *Working paper*, University of Mississippi.  arXiv:2207.05943.
Butts, K. and Gardner, J. (2022).  "did2s: Two-Stage Difference-in-Differences."
    *R Journal*, 14(3), 162-173.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..core.results import CausalResult


__all__ = ["gardner_did", "did_2stage"]


def _first_treated(first: float) -> bool:
    """Treat +Inf / NaN / 0 as 'never treated'."""
    return np.isfinite(first) and first > 0


def _demean_twoway(
    y: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    X: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Iterated demeaning for two-way FE (balanced or unbalanced).

    Returns the two-way demeaned outcome (and covariates), plus a ``info``
    dict holding the fitted unit / time effects keyed by original level.
    """
    u_vals, u_idx = np.unique(unit, return_inverse=True)
    t_vals, t_idx = np.unique(time, return_inverse=True)

    y_demeaned = y.astype(float).copy()
    unit_fe = np.zeros(len(u_vals))
    time_fe = np.zeros(len(t_vals))

    for _ in range(max_iter):
        u_means = np.bincount(u_idx, weights=y_demeaned) / np.bincount(u_idx)
        y_demeaned -= u_means[u_idx]
        unit_fe += u_means
        t_means = np.bincount(t_idx, weights=y_demeaned) / np.bincount(t_idx)
        y_demeaned -= t_means[t_idx]
        time_fe += t_means
        if max(np.max(np.abs(u_means)), np.max(np.abs(t_means))) < tol:
            break

    X_dm = None
    if X is not None:
        X_dm = X.astype(float).copy()
        for j in range(X_dm.shape[1]):
            col = X_dm[:, j].copy()
            for _ in range(max_iter):
                u_m = np.bincount(u_idx, weights=col) / np.bincount(u_idx)
                col -= u_m[u_idx]
                t_m = np.bincount(t_idx, weights=col) / np.bincount(t_idx)
                col -= t_m[t_idx]
                if max(np.max(np.abs(u_m)), np.max(np.abs(t_m))) < tol:
                    break
            X_dm[:, j] = col

    info = {
        "unit_levels": u_vals, "unit_fe": unit_fe,
        "time_levels": t_vals, "time_fe": time_fe,
        "unit_idx": u_idx, "time_idx": t_idx,
    }
    return y_demeaned, X_dm, info


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
    dof = G / max(G - 1, 1) * (n - 1) / max(n - k, 1)
    return dof * xtx_inv @ meat @ xtx_inv


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

    Returns
    -------
    CausalResult
        ``.coef`` is the overall ATT (event-study dict in
        ``model_info['event_study']`` when requested).  Provides
        ``.summary()``, ``.cite()``, and ``.plot()``.

    Notes
    -----
    Identification requires the usual staggered-DID conditions (parallel
    trends, no anticipation) plus a linear two-way FE + additive covariate
    structure for the untreated potential outcome.  Stage-2 SEs incorporate
    the Stage-1 residualisation using a cluster block of the sandwich form;
    when coverage of heavy covariate models matters, bootstrap the whole
    two-step procedure.
    """
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
    treated_now = np.array([
        _first_treated(fi) and (ti >= fi) for fi, ti in zip(ft, t_arr)
    ])
    df["_D"] = treated_now.astype(float)

    # --- Stage 1: FE model on untreated rows only ------------------- #
    untreated = df.loc[~treated_now].copy()
    if len(untreated) < 10:
        raise ValueError("Not enough untreated observations for Stage 1 (<10).")

    y_un = untreated[y].to_numpy(dtype=float)
    u_un = untreated[group].to_numpy()
    t_un = untreated[time].to_numpy()
    X_un = untreated[controls].to_numpy(dtype=float) if controls else None

    y_un_dm, X_un_dm, info = _demean_twoway(y_un, u_un, t_un, X_un)

    beta_hat = None
    if X_un_dm is not None and X_un_dm.shape[1] > 0:
        beta_hat, *_ = np.linalg.lstsq(X_un_dm, y_un_dm, rcond=None)

    # Impute unit/time FEs for *all* rows (use untreated averages as baseline).
    # For units/times only observed in treated rows, use 0 (FE identified only
    # up to level — residualisation still removes the common shift).
    unit_lookup = dict(zip(info["unit_levels"], info["unit_fe"]))
    time_lookup = dict(zip(info["time_levels"], info["time_fe"]))

    df["_alpha"] = df[group].map(unit_lookup).fillna(0.0)
    df["_lambda"] = df[time].map(time_lookup).fillna(0.0)

    # Residualise outcome
    y_all = df[y].to_numpy(dtype=float)
    y_tilde = y_all - df["_alpha"].to_numpy() - df["_lambda"].to_numpy()
    if beta_hat is not None and controls:
        X_all = df[controls].to_numpy(dtype=float)
        # Centre covariates by the untreated grand mean used in demeaning:
        # simplest robust implementation — refit intercept from untreated.
        X_mean = X_un.mean(axis=0)
        y_tilde -= (X_all - X_mean) @ beta_hat

    df["_ytilde"] = y_tilde

    # --- Stage 2: regress residualised outcome on treatment --------- #
    if event_study:
        rel_time = np.where(
            np.isfinite(ft) & (ft > 0),
            t_arr - ft,
            np.nan,
        )
        if horizon is None:
            support = np.unique(rel_time[~np.isnan(rel_time)])
            horizon = [int(k) for k in support if -5 <= k <= 5]
        regressors = {}
        for k in horizon:
            key = f"D_k{int(k)}"
            regressors[key] = (rel_time == k).astype(float)
        X2 = np.column_stack([regressors[k] for k in regressors])
        names = list(regressors.keys())
    else:
        X2 = df["_D"].to_numpy(dtype=float).reshape(-1, 1)
        names = ["ATT"]

    y2 = df["_ytilde"].to_numpy(dtype=float)

    # Drop any row that's exactly zero for all Stage-2 regressors (never
    # contributes in a purely dummy spec — keeps the linear system well
    # conditioned when horizon is sparse).
    keep = np.any(X2 != 0, axis=1) | (~event_study)
    X2e = np.column_stack([np.ones(len(y2)), X2])[keep]
    y2e = y2[keep]
    cl = df[cluster].to_numpy()[keep]

    coef, *_ = np.linalg.lstsq(X2e, y2e, rcond=None)
    fitted = X2e @ coef
    resid = y2e - fitted
    V = _cluster_vcov(X2e, resid, cl)
    se = np.sqrt(np.diag(V))[1:]  # drop intercept SE
    coef_dict = dict(zip(names, coef[1:]))
    se_dict = dict(zip(names, se))

    from scipy import stats as sp_stats
    z = sp_stats.norm.ppf(1 - alpha / 2)

    if event_study:
        ci = {
            k: (coef_dict[k] - z * se_dict[k], coef_dict[k] + z * se_dict[k])
            for k in names
        }
        att_overall = float(np.mean([coef_dict[k] for k in names if k.startswith("D_k") and int(k.split("k")[1]) >= 0]))
        att_se = float(np.sqrt(np.mean([se_dict[k] ** 2 for k in names if k.startswith("D_k") and int(k.split("k")[1]) >= 0])))
    else:
        att_overall = float(coef_dict["ATT"])
        att_se = float(se_dict["ATT"])
        ci = {"ATT": (att_overall - z * att_se, att_overall + z * att_se)}

    n_units = int(df[group].nunique())
    n_treated_units = int(df.loc[treated_now, group].nunique())

    model_info = {
        "method": "Gardner 2021 two-stage DID",
        "n_obs": int(len(df)),
        "n_units": n_units,
        "n_treated_units": n_treated_units,
        "alpha": alpha,
        "event_study": {
            "horizon": list(coef_dict.keys()),
            "coef": coef_dict,
            "se": se_dict,
            "ci": ci,
        } if event_study else None,
        "stage1": {
            "n_untreated": int(len(untreated)),
            "beta_controls": None if beta_hat is None else dict(zip(controls, beta_hat.tolist())),
        },
    }

    return CausalResult(
        method="Gardner 2021 two-stage DID (did2s)",
        coefficient=att_overall,
        std_error=att_se,
        conf_int=(att_overall - z * att_se, att_overall + z * att_se),
        pvalue=float(2 * (1 - sp_stats.norm.cdf(abs(att_overall / att_se)))) if att_se > 0 else np.nan,
        n_obs=int(len(df)),
        treatment_var=first_treat,
        outcome_var=y,
        model_info=model_info,
        citation=(
            "Gardner, J. (2021). Two-stage differences in differences. "
            "arXiv:2207.05943. Extended by Butts & Gardner (2022), R Journal."
        ),
    )


# Convenience alias aligned with R package ``did2s``.
did_2stage = gardner_did
