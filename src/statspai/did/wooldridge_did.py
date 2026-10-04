"""
Modern staggered DID estimators: Wooldridge (2021), DR-DID, and TWFE
decomposition.

Implements three cutting-edge methods for DID with staggered treatment adoption:

1. **wooldridge_did()** — Wooldridge (2021) extended TWFE with cohort × time
   interactions.
   Shows that a properly saturated TWFE regression recovers valid ATT even with
   heterogeneous treatment effects, without specialised estimators.

2. **drdid()** — Sant'Anna & Zhao (2020) doubly robust DID for 2×2 designs with
   covariates.  Combines outcome regression and inverse probability weighting,
   consistent if *either* model is correctly specified.

3. **twfe_decomposition()** — Enhanced Goodman-Bacon (2021) decomposition with
   de Chaisemartin–D'Haultfoeuille (2020) weights diagnostic.

References
----------
Wooldridge, J.M. (2021).
    "Two-Way Fixed Effects, the Two-Way Mundlak Regression, and
     Difference-in-Differences Estimators."
    Working paper, Michigan State University. [@wooldridge2021two]

Sant'Anna, P.H.C. and Zhao, J. (2020).
    "Doubly Robust Difference-in-Differences Estimators."
    *Journal of Econometrics*, 219(1), 101–122.

Goodman-Bacon, A. (2021).
    "Difference-in-Differences with Variation in Treatment Timing."
    *Journal of Econometrics*, 225(2), 254–277. [@goodmanbacon2021difference]

de Chaisemartin, C. and D'Haultfoeuille, X. (2020).
    "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects."
    *American Economic Review*, 110(9), 2964–2996. [@dechaisemartin2020two]
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._core import drop_unusable_rows as _drop_unusable_rows
from ._core import fe_dof_not_nested as _fe_dof_not_nested
from ._etwfe_glm_design import normalise_hettype as _normalise_hettype
from ._etwfe_nonlinear import _ETWFE_GLM_FAMILIES
from ._etwfe_nonlinear import etwfe_glm as _etwfe_glm_impl
from ._etwfe_nonlinear import etwfe_glm_emfx as _etwfe_glm_emfx_impl
from ._etwfe_nonlinear import normalise_scale as _normalise_etwfe_scale

# ═══════════════════════════════════════════════════════════════════════
#  Helper: cluster-robust OLS
# ═══════════════════════════════════════════════════════════════════════


def _ols_fit(
    X: np.ndarray,
    y: np.ndarray,
    cluster: Optional[np.ndarray] = None,
    dof_k: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS / WLS with optional cluster-robust (CR1) standard errors.

    ``weights`` are observation weights in the fixest ``weights=`` / Stata
    ``[pw=]`` sense: the estimator minimises ``sum_i w_i (y_i - x_i b)^2``
    and the sandwich uses the scores ``x_i w_i e_i``.  The small-sample
    factors count observations (``n``), not the weight total, as both
    reference implementations do.

    ``dof_k`` overrides the parameter count used in the CR1 finite-sample
    factor ``(N-1)/(N-K)``. Callers that absorb fixed effects by demeaning
    must pass it: the absorbed effects are real parameters, and
    ``fixest``/``reghdfe`` count every one that is not nested inside the
    cluster variable (see ``did._core.fe_dof_not_nested``). Leaving it at
    ``None`` counts only the explicit design columns, which understates
    ``K`` — and therefore the SE — whenever fixed effects were absorbed.

    Returns (beta, se, vcov).
    """
    n, k = X.shape
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        sw = np.sqrt(w)
        Xw = X * sw[:, np.newaxis]
        yw = y * sw
    else:
        w = None
        Xw, yw = X, y
    # QR solve (X = QR): beta = R^{-1} Q'y and (X'X)^{-1} = R^{-1} R^{-ᵀ}.
    # Avoids squaring cond(X) the way forming inv(X'X) does — same numerical
    # hardening as the core OLS kernel (cf. NIST StRD certification under
    # tests/numerical_accuracy/). Well-conditioned DiD designs are unchanged
    # to ~1e-12; ill-conditioned (many group×period dummies) gain accuracy.
    try:
        Q, R = np.linalg.qr(Xw)
        R_inv = np.linalg.solve(R, np.eye(k))
        XtX_inv = R_inv @ R_inv.T
        beta = R_inv @ (Q.T @ yw)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)
        beta = XtX_inv @ (Xw.T @ yw)
    resid = y - X @ beta
    # Score contribution per observation: x_i * w_i * e_i (w_i = 1 unweighted).
    u = resid if w is None else w * resid

    if cluster is not None:
        _, cluster_inverse = np.unique(cluster, return_inverse=True)
        n_cl = int(cluster_inverse.max()) + 1
        if n_cl < 2:
            raise DataInsufficient(
                "Cluster-robust Wooldridge DID inference requires at least "
                "two clusters."
            )
        scores = np.zeros((n_cl, k), dtype=float)
        np.add.at(scores, cluster_inverse, X * u[:, np.newaxis])
        meat = scores.T @ scores
        k_eff = int(dof_k) if dof_k is not None else k
        k_eff = min(max(k_eff, 1), n - 1)
        correction = (n_cl / (n_cl - 1)) * ((n - 1) / (n - k_eff))
        vcov = correction * XtX_inv @ meat @ XtX_inv
    else:
        # HC1 robust
        hc1 = (n / (n - k)) * u**2
        meat = X.T @ (X * hc1[:, np.newaxis])
        vcov = XtX_inv @ meat @ XtX_inv

    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return beta, se, vcov


_ETWFE_AGG_WEIGHTS = ("estimation", "unit")


def _validate_agg_weights(agg_weights: str) -> str:
    if agg_weights not in _ETWFE_AGG_WEIGHTS:
        raise MethodIncompatibility(
            f"agg_weights must be one of {list(_ETWFE_AGG_WEIGHTS)}; "
            f"got {agg_weights!r}",
            recovery_hint=(
                "Use agg_weights='estimation' (Stata jwdid, estat) or "
                "agg_weights='unit' (R etwfe::emfx)."
            ),
            diagnostics={"agg_weights": agg_weights},
        )
    return agg_weights


def _validate_etwfe_weights(data: pd.DataFrame, weights: str) -> pd.DataFrame:
    """Check an ETWFE ``weights`` column and drop zero-weight rows.

    Weights must be a numeric column with finite, non-negative entries and a
    positive total (NaN or negative weights raise -- they cannot be given a
    meaning silently).  Rows with weight exactly 0 are removed from the
    estimation sample with a warning, which is what fixest ``weights=`` and
    Stata ``[pw=]`` do, so ``n_obs`` and the small-sample factors count the
    same observations as the references.
    """
    if weights not in data.columns:
        raise MethodIncompatibility(
            f"weights column {weights!r} is not in the data.",
            recovery_hint="Pass the name of an existing numeric column.",
            diagnostics={"weights": weights},
        )
    w = pd.to_numeric(data[weights], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(w)):
        raise MethodIncompatibility(
            f"weights column {weights!r} contains NaN, non-numeric or "
            f"infinite values in {int((~np.isfinite(w)).sum())} rows.",
            recovery_hint="Drop or impute those rows before estimating.",
            diagnostics={"weights": weights},
        )
    if np.any(w < 0):
        raise MethodIncompatibility(
            f"weights column {weights!r} contains negative values.",
            recovery_hint="Sampling/population weights must be >= 0.",
            diagnostics={"weights": weights, "min": float(w.min())},
        )
    total = float(w.sum())
    if total <= 0:
        raise DataInsufficient(
            f"weights column {weights!r} sums to {total}; at least one "
            "observation must carry positive weight.",
            recovery_hint="Check the weights column for an all-zero slice.",
            diagnostics={"weights": weights},
        )
    zero = w == 0
    if zero.any():
        warnings.warn(
            f"etwfe: {int(zero.sum())} observation(s) with weight 0 in "
            f"{weights!r} were dropped from the estimation sample (fixest "
            "weights= / Stata [pw=] semantics).",
            UserWarning,
            stacklevel=3,
        )
        data = data.loc[~zero].reset_index(drop=True)
    return data


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ═══════════════════════════════════════════════════════════════════════
#  1. Wooldridge (2021) Extended TWFE
# ═══════════════════════════════════════════════════════════════════════


def _cohort_atts_from_cells(
    event_study: pd.DataFrame,
    event_vcov: Optional[np.ndarray],
    cohorts: List[Any],
    cohort_sizes: Dict[int, int],
    df_resid: int,
    alpha: float,
) -> Tuple[pd.DataFrame, np.ndarray, float, float, float, Tuple[float, float]]:
    """Aggregate saturated cohort x period cells into cohort-level ATTs.

    Wooldridge (2021) identifies one coefficient per *treated* cell
    ``(g, t)``, ``t >= g``; every pre-treatment cell is absorbed by the
    cohort and period fixed effects. ``ATT(g)`` is then the
    treated-observation-weighted mean of cohort ``g``'s cells -- the
    aggregation R ``etwfe::emfx(type='group')`` reports.

    Reading ``ATT(g)`` off a *separate* regression carrying a single post
    dummy per cohort is not equivalent. That design is not saturated in
    cohort x period, so under dynamic treatment effects the already-treated
    cohorts enter the period fixed effects and contaminate every treatment
    coefficient -- the forbidden comparison of Goodman-Bacon (2021), which
    is precisely what the extended TWFE design exists to remove. StatsPAI
    read the cohort ATTs off that unsaturated design through 1.26.0; see
    the 1.27.0 CHANGELOG entry "ETWFE cohort-level ATT".

    Returns ``(detail, cohort_vcov, att_overall, att_se, pvalue, ci)`` where
    the headline is the cohort-size-weighted average of ``ATT(g)`` and its
    SE is the delta-method value implied by ``cohort_vcov``.
    """
    post = event_study.loc[event_study["rel_time"] >= 0]
    if len(post) == 0:
        raise DataInsufficient(
            "No post-treatment cohort x period cell survived estimation, so "
            "no cohort ATT is identified."
        )
    V = np.asarray(event_vcov, dtype=float) if event_vcov is not None else None
    n_cells = int(V.shape[0]) if V is not None else 0
    W = np.zeros((len(cohorts), n_cells)) if n_cells else None

    rows: List[Dict[str, Any]] = []
    atts: List[float] = []
    ses: List[float] = []
    for i, g in enumerate(cohorts):
        cells = post.loc[post["cohort"] == int(g)]
        if len(cells) == 0:
            raise DataInsufficient(
                f"Cohort {int(g)} has no post-treatment cohort x period cell; "
                "its ATT is not identified by the ETWFE design."
            )
        w_raw = cells["n_treated_obs"].astype(float).to_numpy()
        total = float(w_raw.sum())
        w_g = (
            w_raw / total
            if np.isfinite(total) and total > 0
            else np.full(len(cells), 1.0 / len(cells))
        )
        att_g = float(w_g @ cells["estimate"].astype(float).to_numpy())

        idx = None
        if W is not None and "_vcov_idx" in cells.columns:
            cand = cells["_vcov_idx"].astype(int).to_numpy() - 1
            if bool(((cand >= 0) & (cand < n_cells)).all()):
                idx = cand
        if idx is not None:
            se_g = float(np.sqrt(max(w_g @ V[np.ix_(idx, idx)] @ w_g, 0.0)))
            W[i, idx] = w_g
        else:
            # No usable cell covariance: fall back to the independent-cell
            # approximation and say so rather than inventing a covariance.
            se_g = float(
                np.sqrt(np.sum((w_g * cells["se"].astype(float).to_numpy()) ** 2))
            )
        t_g = att_g / se_g if se_g > 0 else np.nan
        p_g = (
            float(2 * stats.t.sf(abs(t_g), max(df_resid, 1)))
            if np.isfinite(t_g)
            else np.nan
        )
        rows.append(
            {
                "cohort": int(g),
                "att": att_g,
                "se": se_g,
                "tstat": t_g,
                "pvalue": p_g,
                "n_obs": int(cohort_sizes.get(int(g), 0)),
                "n_treated_obs": int(total) if np.isfinite(total) else 0,
            }
        )
        atts.append(att_g)
        ses.append(se_g)

    detail = pd.DataFrame(rows)
    att_vec = np.asarray(atts, dtype=float)
    if W is not None and V is not None and np.any(W):
        cohort_vcov = W @ V @ W.T
    else:
        cohort_vcov = np.diag(np.asarray(ses, dtype=float) ** 2)

    sizes = detail["n_obs"].to_numpy(dtype=float)
    weights = (
        sizes / sizes.sum()
        if np.isfinite(sizes).all() and sizes.sum() > 0
        else np.full(len(sizes), 1.0 / max(len(sizes), 1))
    )
    att_overall = float(weights @ att_vec)
    att_se = float(np.sqrt(max(weights @ cohort_vcov @ weights, 0.0)))
    t_overall = att_overall / att_se if att_se > 0 else np.nan
    p_overall = (
        float(2 * stats.t.sf(abs(t_overall), max(df_resid, 1)))
        if np.isfinite(t_overall)
        else np.nan
    )
    t_crit = stats.t.ppf(1 - alpha / 2, max(df_resid, 1))
    ci = (att_overall - t_crit * att_se, att_overall + t_crit * att_se)
    return detail, cohort_vcov, att_overall, att_se, p_overall, ci


@accepts_aliases(_strict=True, id="group", unit="group", covariates="controls")
def wooldridge_did(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Wooldridge (2021) extended TWFE estimator for staggered DID.

    Estimates the TWFE regression *saturated in cohort x period* — one
    coefficient per cohort-period cell, unit and period fixed effects
    absorbed by two-way demeaning, and each cohort's ``rel_time = -1``
    cell held out as the reference. Saturation is what makes the design
    valid under heterogeneous and dynamic treatment effects: a single
    post dummy per cohort is *not* saturated and is contaminated by
    already-treated cohorts through the period fixed effects.

    ``ATT(g)`` is the treated-observation-weighted mean of cohort ``g``'s
    post-treatment cells, and ``estimate`` is the **cohort-size-weighted**
    average of those ``ATT(g)``. This reproduces R
    ``etwfe::etwfe(..., cgroup = "never")`` + ``emfx(type = "group")``
    cell for cell (parity module ``17_etwfe``).

    .. versionchanged:: 1.27.0
       ⚠️ Correctness fix. Through 1.26.0 ``estimate`` and ``detail`` were
       read off a *separate*, unsaturated cohort x post regression while
       only the event-study output used the saturated design. On a
       deterministic DGP with true cohort ATTs 3.0 and 2.5 the old path
       returned 2.67 and 1.63 (headline 2.15 against a true 2.78); it now
       returns 2.99 and 2.44. See MIGRATION.md.

    See Also
    --------
    etwfe : The same estimator with the treated-observation-weighted
        simple ATT that R ``emfx(type='simple')`` and Stata
        ``jwdid, estat simple`` report as the headline, plus the
        ``cgroup='notyet'`` not-yet-treated comparison group.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset (long format, one row per unit-period).
    y : str
        Outcome variable.
    group : str
        Unit identifier (e.g. county, individual).
    time : str
        Time period variable (integer-valued).
    first_treat : str
        Column indicating when the unit is first treated.
        Use ``np.nan`` (or 0) for never-treated units.
    controls : list of str, optional
        Time-varying covariates to include.
    cluster : str, optional
        Cluster variable for standard errors.  Defaults to *group*
        (unit-level clustering).
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        ``estimate`` is the cohort-size-weighted average of ``ATT(g)``.
        ``detail`` holds one row per cohort with ``ATT(g)`` and its
        delta-method SE, aggregated from the saturated cells.
        ``model_info['event_study']`` holds the cohort x relative-time
        cells themselves (leads included, for pre-trend inspection) and
        ``model_info['event_vcov']`` their cluster-robust covariance.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=10, staggered=True, seed=42)
    >>> result = sp.wooldridge_did(df, y='y', group='unit',
    ...                            time='time', first_treat='first_treat')
    >>> result.estimand
    'ATT'
    >>> bool(result.detail is not None)  # cohort-specific ATTs
    True
    """
    df = data.copy()

    # ── Normalise first_treat ────────────────────────────────────────
    ft = df[first_treat].copy()
    # Treat 0 and NaN as never-treated → sentinel
    ft = ft.replace(0, np.nan)
    df["_ft"] = ft

    periods = sorted(df[time].unique())
    cohorts = sorted(df.loc[df["_ft"].notna(), "_ft"].unique())

    if len(cohorts) == 0:
        raise DataInsufficient("No treated cohorts found. Check 'first_treat' column.")

    # ── Unit and time FE via demeaning ──────────────────────────────
    # Within-group (unit) demeaning
    df["_y"] = df[y].astype(float)
    unit_mean = df.groupby(group)["_y"].transform("mean")
    time_mean = df.groupby(time)["_y"].transform("mean")
    grand_mean = df["_y"].mean()
    df["_y_dm"] = df["_y"] - unit_mean - time_mean + grand_mean

    # ── Build the cohort × relative-time design ─────────────────────
    # One dummy per cohort × period cell, with each cohort's rel = -1 cell
    # left as the omitted reference. Every treated cell (rel >= 0) therefore
    # carries its own coefficient: this is the saturated design Wooldridge
    # (2021) requires, and it reproduces R ``etwfe(cgroup = "never")`` cell
    # for cell (see tests/r_parity/17_etwfe.py).
    #
    # A *single* post dummy per cohort is NOT this design. It is not
    # saturated in cohort x period, so under dynamic treatment effects the
    # already-treated cohorts enter the period fixed effects and contaminate
    # every treatment coefficient — the forbidden comparison of
    # Goodman-Bacon (2021), which is exactly what extended TWFE removes.
    # StatsPAI read `detail` and the headline off that unsaturated design
    # through 1.26.0; both now come from the cells below.
    event_cols: List[str] = []
    event_meta: List[Tuple[int, int]] = []  # (cohort, rel_time)
    for g in cohorts:
        for t_val in periods:
            rel = int(t_val - g)
            if rel == -1:
                continue  # reference / omitted period
            col = (
                f"_coh{int(g)}_rel{rel}"
                if rel >= 0
                else f"_coh{int(g)}_rel_neg{abs(rel)}"
            )
            df[col] = ((df["_ft"] == g) & (df[time] == t_val)).astype(float)
            event_cols.append(col)
            event_meta.append((int(g), rel))

    if len(event_cols) == 0:
        raise DataInsufficient(
            "No cohort × period cell could be created; check "
            f"{first_treat!r} against the observed periods."
        )

    # ── Demean the design (same two-way FE projection as the outcome) ─
    dummy_values = df[event_cols]
    unit_means = dummy_values.groupby(df[group]).transform("mean")
    time_means = dummy_values.groupby(df[time]).transform("mean")
    dummy_dm = dummy_values - unit_means - time_means + dummy_values.mean()
    dummy_dm.columns = [f"{col}_dm" for col in event_cols]
    df[dummy_dm.columns] = dummy_dm

    # ── Demean controls ─────────────────────────────────────────────
    ctrl_dm_cols: List[str] = []
    if controls:
        ctrl_cols = []
        for c in controls:
            col = f"_ctrl_{c}"
            df[col] = df[c].astype(float)
            ctrl_cols.append(col)
            ctrl_dm_cols.append(f"{col}_dm")
        ctrl_values = df[ctrl_cols]
        unit_means = ctrl_values.groupby(df[group]).transform("mean")
        time_means = ctrl_values.groupby(df[time]).transform("mean")
        ctrl_dm = ctrl_values - unit_means - time_means + ctrl_values.mean()
        ctrl_dm.columns = ctrl_dm_cols
        df[ctrl_dm_cols] = ctrl_dm

    # ── Drop NaN rows ───────────────────────────────────────────────
    keep_cols = ["_y_dm"] + [f"{c}_dm" for c in event_cols] + ctrl_dm_cols
    valid = df[keep_cols].notna().all(axis=1)
    df_valid = df.loc[valid].reset_index(drop=True)

    cl_arr = df_valid[cluster].values if cluster is not None else df_valid[group].values

    # ── ETWFE regression on the saturated cells ─────────────────────
    y_vec = df_valid["_y_dm"].values
    X_cols = [f"{c}_dm" for c in event_cols] + ctrl_dm_cols
    X = np.column_stack([np.ones(len(y_vec))] + [df_valid[c].values for c in X_cols])
    if len(y_vec) <= X.shape[1]:
        raise DataInsufficient(
            f"The saturated ETWFE design needs more than {X.shape[1]} usable "
            "rows (one coefficient per cohort × period cell) but only "
            f"{len(y_vec)} remain. Pool cohorts or periods, or use "
            "sp.callaway_santanna / sp.did_imputation on this design."
        )
    # Identification. sp.wooldridge_did estimates the *never-treated-control*
    # ETWFE: every cohort x period cell except each cohort's rel = -1
    # reference carries a dummy, so the untreated comparison has to come from
    # never-treated units. With none, the cells span the panel and nothing
    # pins the period effects — `pinv` still returns numbers (observed:
    # 1e13-scale coefficients on a two-cohort panel), and a silently absurd
    # estimate is the worst possible output.
    if not bool(df_valid["_ft"].isna().any()):
        raise DataInsufficient(
            "sp.wooldridge_did needs at least one never-treated unit "
            f"({first_treat!r} NaN or 0): it estimates the ETWFE identified "
            "against a never-treated comparison group, and with every unit "
            "treated the saturated cohort × period design is not identified. "
            "Use sp.etwfe(..., cgroup='notyet') for the not-yet-treated "
            "comparison group, or sp.callaway_santanna / sp.did_imputation."
        )
    # Residual rank deficiency is usually benign — a redundant cell column or
    # a collinear control, which fixest and reghdfe simply drop and which
    # leaves the reported aggregates estimable under the minimum-norm
    # solution. Warn rather than raise, matching _etwfe_repeated_cs.
    _rank = int(np.linalg.matrix_rank(X))
    if _rank < X.shape[1]:
        warnings.warn(
            f"wooldridge_did: the saturated ETWFE design is rank-deficient by "
            f"{X.shape[1] - _rank} column(s) (rank {_rank}, ncol "
            f"{X.shape[1]}). Individual cell coefficients are then an "
            "arbitrary member of the solution set; cohort ATTs and the "
            "headline remain estimable. Check for collinear controls or for "
            "a cohort with no usable pre-period.",
            RuntimeWarning,
            stacklevel=2,
        )

    # The unit and period effects are absorbed by the two-way demeaning
    # above but are still parameters. fixest's ssc(fixef.K="nested") and
    # reghdfe's default count every absorbed effect that is not nested in
    # the cluster; the explicit constant is collinear with them and drops
    # out. sp.event_study and sp.sun_abraham already follow this rule
    # (parity module 85); wooldridge_did did not until 1.27.0, which left
    # its clustered SEs a uniform ~0.08% below R etwfe's.
    _cluster_col = cluster if cluster is not None else group
    k_fe = _fe_dof_not_nested(df_valid, [group, time], _cluster_col)
    dof_k = X.shape[1] - 1 + k_fe
    beta, se, vcov = _ols_fit(X, y_vec, cluster=cl_arr, dof_k=dof_k)

    n_obs = len(y_vec)
    df_resid = max(n_obs - dof_k, 1)
    n_event = len(event_cols)
    event_vcov = vcov[1 : 1 + n_event, 1 : 1 + n_event]

    ev_rows: List[Dict[str, Any]] = []
    for j, (coh_val, rel_val) in enumerate(event_meta):
        n_treated_ev = int(
            (
                (df_valid["_ft"] == coh_val)
                & (df_valid[time] == coh_val + rel_val)
                & (df_valid[time] >= coh_val)
            ).sum()
        )
        ev_rows.append(
            {
                "cohort": coh_val,
                "rel_time": rel_val,
                "estimate": float(beta[j + 1]),
                "se": float(se[j + 1]),
                "_vcov_idx": j + 1,
                "n_treated_obs": n_treated_ev,
            }
        )
    event_study_df = pd.DataFrame(ev_rows)

    # ── Cohort ATTs and the cohort-size-weighted headline ───────────
    cohort_sizes = {int(g): int((df_valid["_ft"] == g).sum()) for g in cohorts}
    (
        detail,
        cohort_vcov,
        att_overall,
        att_se_overall,
        p_overall,
        ci,
    ) = _cohort_atts_from_cells(
        event_study_df,
        event_vcov,
        [int(g) for g in cohorts],
        cohort_sizes,
        df_resid,
        alpha,
    )
    weights = detail["n_obs"].to_numpy(dtype=float)
    weights = (
        weights / weights.sum()
        if weights.sum() > 0
        else np.full(len(weights), 1.0 / max(len(weights), 1))
    )

    # ── Model info ──────────────────────────────────────────────────
    model_info: Dict[str, Any] = {
        "n_cohorts": len(cohorts),
        "cohorts": [int(g) for g in cohorts],
        "n_periods": len(periods),
        "n_units": df[group].nunique(),
        "controls": controls or [],
        "cluster_var": cluster or group,
        "n_clusters": len(np.unique(cl_arr)) if cl_arr is not None else None,
        "cohort_weights": {int(g): float(w) for g, w in zip(cohorts, weights)},
        "cohort_weighting": "cohort",
        "cohort_vcov": cohort_vcov,
    }
    if event_study_df is not None:
        model_info["event_study"] = event_study_df
        model_info["event_vcov"] = event_vcov  # H1: proper aggregation SE

    _result = CausalResult(
        method="Wooldridge (2021) Extended TWFE",
        estimand="ATT",
        estimate=att_overall,
        se=att_se_overall,
        pvalue=p_overall,
        ci=ci,
        alpha=alpha,
        n_obs=n_obs,
        detail=detail,
        model_info=model_info,
        _citation_key="wooldridge_twfe",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.wooldridge_did",
            params={
                "y": y,
                "group": group,
                "time": time,
                "id": id,
                "first_treat": first_treat,
                "controls": controls,
                "cluster": cluster,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _etwfe_glm(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    family: str,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    cgroup: str = "notyet",
    fe: Optional[str] = None,
    scale: str = "response",
    hettype: Optional[str] = None,
    xvar: Optional[Any] = None,
    separated: str = "keep",
    response_se: str = "profile",
) -> CausalResult:
    """Nonlinear ETWFE; implementation in :mod:`._etwfe_nonlinear`."""
    return _etwfe_glm_impl(
        data=data,
        y=y,
        group=group,
        time=time,
        first_treat=first_treat,
        family=family,
        controls=controls,
        cluster=cluster,
        alpha=alpha,
        cgroup=cgroup,
        fe=fe,
        scale=scale,
        hettype=hettype,
        xvar=xvar,
        separated=separated,
        response_se=response_se,
    )


@accepts_aliases(_strict=True, id="group", unit="group", covariates="controls")
def etwfe(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    xvar: Optional[Any] = None,
    panel: bool = True,
    cgroup: str = "notyet",
    family: Optional[str] = None,
    weights: Optional[str] = None,
    agg_weights: str = "estimation",
    fe: Optional[str] = None,
    scale: str = "response",
    hettype: Optional[str] = None,
    separated: str = "keep",
    response_se: str = "profile",
) -> CausalResult:
    """Public ``sp.etwfe`` entry point — see ``_dispatch_etwfe_impl`` for
    the full docstring on options and behaviour.

    ``weights`` names a column of non-negative observation weights: the
    cohort-by-period regression becomes weighted least squares with the
    semantics of R ``fixest`` ``weights=`` / Stata ``reghdfe [pw=]``, with
    the same cluster-robust (CR1) small-sample convention as the unweighted
    fit.  ``agg_weights`` picks how the estimated cells are averaged into
    the ``emfx`` aggregates: ``'estimation'`` (default; Stata ``jwdid,
    estat``) weights each cell by the sum of the estimation weights over its
    treated observations, so the never-treated ``simple`` aggregate is the
    weighted Callaway--Sant'Anna simple ATT; ``'unit'`` (R
    ``etwfe::emfx``) gives every treated observation weight one regardless
    of its estimation weight.  Without ``weights`` the two rules coincide.

    ``family`` selects the outcome model. ``None``/``'gaussian'`` (default)
    is the historical linear ETWFE and is unchanged. ``'poisson'`` and
    ``'logit'`` fit Wooldridge (2023) nonlinear ETWFE by maximum likelihood
    and report the **average marginal effect on the response scale**,
    matching R ``etwfe::emfx(type='simple')`` — so a Poisson fit returns an
    effect in counts, not log points. The nonlinear branch supports
    ``cgroup='notyet'`` and ``cgroup='nevertreated'`` (the latter also
    estimates the pre-treatment cells, i.e. event-study leads, relative to
    ``g - 1``); it does not currently accept ``panel=False`` or
    ``weights``.  Its ``controls`` enter additively without interactions
    (Stata ``jwdid``'s ``exovar()``).  With ``fe='unit'`` its ``xvar``
    follows Stata ``jwdid y x``: each treatment effect is moderated by the
    covariate demeaned within the ``hettype`` cells, with
    covariate-by-period (and, when the covariate varies within units,
    covariate-by-cohort) terms.  With ``fe='cohort'`` it is R ``etwfe``'s
    design: the covariate demeaned within cohort x period cells,
    interacted with the cells and with the periods (one period slope per
    level of a categorical covariate, where ``etwfe`` 0.6.2 fits a single
    summed slope).  A categorical column (pandas ``category`` /
    ``object`` / ``bool``) enters as level dummies, and
    ``sp.etwfe_emfx(..., by_xvar=True)`` reports the ATT per level
    (``estat ..., over()``).

    ``hettype`` pools the cohort x period cells as Stata ``jwdid,
    hettype()``: ``'timecohort'`` (default, saturated), ``'time'`` (one
    effect per calendar period), ``'cohort'`` (one per cohort), ``'event'``
    (one per event time) or ``'twfe'`` (a single effect).  The restricted
    designs are the robustness checks of an ETWFE table, not
    heterogeneity-robust estimators themselves.  On the linear model a
    non-default ``hettype`` -- or ``fe='unit'`` -- fits Stata ``jwdid``
    without ``method()``: ``reghdfe`` with unit and period effects,
    clustered on the unit with its small-sample factor, ``jwdid``'s
    covariate design for ``xvar``, and the aggregations of the nonlinear
    branch (``scale='link'``; ``panel=True`` and no ``weights``).

    ``separated`` (``family='poisson', fe='unit'``) decides what happens to
    separated rows (all-zero units, perfectly predicted zeros) after they
    leave IRLS.  ``'keep'`` (default) keeps them in ``N``, the cluster
    count and the aggregation weights: the ATT averages over every treated
    observation, and a separated row enters with its exact-zero marginal
    effect.  ``'drop'`` removes them from all three, which is what Stata
    ``jwdid`` reports whenever ``ppmlhdfe`` flags the separation (its
    ``estat`` averages over ``e(sample)``); ``ppmlhdfe`` does not always
    flag it on large panels, and then ``'keep'`` is the matching choice.

    ``response_se`` (nonlinear families) picks the delta-method convention
    for response-scale standard errors; it matters only for
    ``family='poisson', fe='unit'``, where the unit effects are absorbed.
    ``'profile'`` (default) profiles the unit effect out -- the Poisson
    first-order condition ties each unit's fitted total to its observed
    total -- and differentiates through it.  ``'margins'`` reproduces
    Stata ``jwdid ..., method(ppmlhdfe)`` + ``estat``: ``margins`` holds the
    absorbed effects fixed and differentiates through ``ppmlhdfe``'s
    ``_cons``, i.e. around the sample-weighted mean of the regressors
    instead of each unit's own.  Point estimates and link-scale SEs do
    not depend on it; with cohort dummies (``fe='cohort'``) or the linear
    model the two conventions coincide.  ``'unconditional'`` (nonlinear
    families, either ``fe``) targets the population ATT, as Stata
    ``margins, vce(unconditional)``: it adds the sampling variability of
    the units averaged over to the delta-method term through the
    aggregates' influence function, and agrees with the delete-one-unit
    jackknife.  Use it when the estimand is the population ATT rather than
    the ATT of the sample's own units; it is well above ``'profile'``
    under ``fe='unit'`` when unit effects are heterogeneous.
    ``model_info['response_se']`` records the choice.

    ``scale`` (nonlinear families only) picks the headline scale:
    ``'response'`` (default) is the average marginal effect above;
    ``'link'`` is the treated-observation-weighted average of the
    cohort x period coefficients, :math:`\\sum N_{gt}\\hat\\beta_{gt} /
    \\sum N_{gt}` -- log points for Poisson -- which is what Stata
    ``jwdid ..., method(ppmlhdfe)`` followed by ``estat simple,
    predict(xb)`` prints.  Both scales are always computed and stored in
    ``model_info['aggregations']``; ``sp.etwfe_emfx(res, type=...,
    scale=...)`` serves either.

    ``fe`` (linear model and ``family='poisson'``) picks the heterogeneity
    control: ``'cohort'`` (default) is R ``etwfe``'s cohort-dummy (Mundlak)
    design; ``'unit'`` absorbs unit fixed effects like Stata ``jwdid``
    (``reghdfe`` / ``ppmlhdfe``).  On a balanced panel the two give identical
    coefficients; on an unbalanced one they do not, and ``fe='cohort'``
    warns.  With ``fe='unit'`` units whose outcome is always zero are
    separated (their fitted mean is exactly zero) and are left out of
    IRLS, but stay in ``N`` and in the cluster count as ``jwdid`` reports
    them; ``model_info['n_separated']`` records how many rows that was.

    ``controls`` and ``xvar`` also take Stata factor-variable terms:
    ``controls=["i.year#i.nodecity"]`` adds the year-by-node-city dummies
    (base levels omitted), i.e. Stata ``jwdid ..., exovar(i.year#i.nodecity)``;
    ``xvar="i.region"`` treats ``region`` as categorical (``jwdid y
    i.region``).  :func:`sp.jwdid` takes Stata's option names directly.

    Thin wrapper around the 4-branch dispatcher (panel-with-xvar /
    panel-never-only / panel-notyet / repeated-cross-section) that
    attaches a :class:`Provenance` record to the returned result so
    downstream replication_pack / Quarto appendix / table footers
    can pick up the call without each branch having to opt in.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=120, n_periods=8, staggered=True, seed=42)
    >>> res = sp.etwfe(df, y='y', group='unit', time='time',
    ...                first_treat='first_treat')
    >>> res.estimate > 0  # R/Stata simple ATT (true effect 0.5)
    True
    >>> res.detail is not None  # cohort-specific ATTs
    True

    See Also
    --------
    wooldridge_did : The same saturated cohort x period regression reported
        under a different headline aggregation — the **cohort-size-weighted**
        average of ``ATT(g)`` under a never-treated comparison group, i.e. R
        ``etwfe::etwfe(cgroup='never')`` + ``emfx(type='group')``. ``sp.etwfe``
        instead reports the **treated-observation-weighted** simple ATT that R
        ``emfx(type='simple')`` and Stata ``jwdid, estat simple`` print, under
        the not-yet-treated comparison group by default. The two are separate
        documented aggregations of one estimator, not two estimators, and they
        differ materially on the ``17_etwfe`` parity bytes: the ``sp.etwfe``
        default is 15.9% from the ``sp.wooldridge_did`` headline, and
        ``sp.etwfe(cgroup='nevertreated')`` — same comparison group, different
        weights — is still 10.5% from it. Pick the one your write-up claims.
    """
    # Stata factor terms (``controls=["i.year#i.nodecity"]``,
    # ``xvar="i.region"``) become columns first; plain column names pass
    # through untouched.
    from ._factor_terms import expand_factor_terms, factor_xvar

    data, controls = expand_factor_terms(data, controls, context="etwfe")
    data, xvar = factor_xvar(data, xvar, context="etwfe")

    # Drop rows no branch can use before dispatch, so a wiped outcome surfaces
    # as an error rather than an ATT of exactly 0.0 (or a raw ValueError out of
    # an empty-array reduction further down).
    data = _drop_unusable_rows(
        data,
        # `first_treat` is deliberately absent: NaN there encodes
        # never-treated, so requiring it would silently delete the control
        # group — the exact failure this guard exists to prevent.
        columns=[y, time, group, *(controls or [])],
        function="etwfe",
    )

    # Every branch identifies the ATT off cohort × post cells. With no treated
    # post-treatment observation left — cohorts that adopt after the last
    # observed period, or a post-period outcome wiped by a bad merge — those
    # dummies are identically zero, the regression is degenerate, and the
    # aggregate comes back as exactly 0.0 rather than as an error. Check once
    # here so all four dispatch branches inherit the guard. (§7: fail loudly.)
    # A panel with no cohorts at all already raises "No treated cohorts found"
    # in each branch; leave that contract alone and guard only the case that
    # previously slipped through — cohorts present, but no post cell.
    _validate_agg_weights(agg_weights)
    if weights is not None:
        data = _validate_etwfe_weights(data, weights)

    _ft = data[first_treat].replace(0, np.nan)
    _cohorts = sorted(_ft.dropna().unique().tolist())
    _n_treated_post = int((_ft.notna() & (data[time] >= _ft)).sum())
    if _cohorts and _n_treated_post == 0:
        raise DataInsufficient(
            "No treated post-treatment observations: every unit is either "
            "never-treated or first treated after the last observed period, "
            "so there is no cohort × post cell to identify the ATT from.",
            recovery_hint=(
                "Check that `first_treat` is on the same scale as `time` and "
                "that post-treatment outcomes survived any upstream merge."
            ),
            diagnostics={
                "function": "etwfe",
                "n_treated_post": 0,
                "max_time": None if data.empty else data[time].max(),
                "cohorts": _cohorts,
            },
        )

    data, _last_cohort_ref = _trim_to_notyet_support(
        data, time=time, first_treat=first_treat, cgroup=cgroup
    )

    fam_key = None if family is None else str(family).strip().lower()
    if fam_key in _ETWFE_GLM_FAMILIES:
        # Fail loudly rather than silently ignoring an option the nonlinear
        # branch does not implement — a quietly-dropped xvar/cgroup would
        # change the estimand without telling anyone.
        for arg_name, arg_val, bad in (
            ("panel", panel, not panel),
            ("weights", weights, weights is not None),
        ):
            if bad:
                raise MethodIncompatibility(
                    f"etwfe(family={family!r}) does not support "
                    f"{arg_name}={arg_val!r} yet.",
                    recovery_hint=(
                        "Drop the option, or use family=None for the linear "
                        "ETWFE which supports it."
                    ),
                    diagnostics={"family": family, arg_name: arg_val},
                )
        _result = _etwfe_glm(
            data=data,
            y=y,
            group=group,
            time=time,
            first_treat=first_treat,
            family=fam_key,
            controls=controls,
            cluster=cluster,
            alpha=alpha,
            cgroup=cgroup,
            fe=fe,
            scale=scale,
            hettype=hettype,
            xvar=xvar,
            separated=separated,
            response_se=response_se,
        )
    elif fam_key not in (None, "gaussian", "normal"):
        raise MethodIncompatibility(
            f"family={family!r} is not supported; use one of "
            f"{sorted(set(_ETWFE_GLM_FAMILIES) | {'gaussian'})} or None.",
            recovery_hint="Pass family='poisson', 'logit', or 'gaussian'.",
            diagnostics={"family": family},
        )
    else:
        # The linear model has a single scale.
        _normalise_etwfe_scale(scale)
        if str(response_se).strip().lower() not in {"profile", "margins"}:
            raise MethodIncompatibility(
                f"etwfe(response_se={response_se!r}) applies to the "
                "nonlinear families (family='poisson' / 'logit'); the linear "
                "ETWFE accepts only 'profile' or 'margins', which coincide "
                "on it.",
                recovery_hint="Drop response_se=; the linear ETWFE has one " "scale.",
                diagnostics={"response_se": response_se, "family": family},
            )
        if str(separated).strip().lower() != "keep":
            raise MethodIncompatibility(
                f"etwfe(separated={separated!r}) applies to family='poisson' "
                "with fe='unit'.",
                recovery_hint="Drop separated= for the linear ETWFE.",
                diagnostics={"separated": separated, "family": family},
            )
        fe_key = None if fe is None else str(fe).strip().lower()
        if fe_key not in (None, "cohort", "unit", "ivar", "id", "individual"):
            raise MethodIncompatibility(
                f"etwfe(fe={fe!r}) is not recognised; use 'cohort' or 'unit'.",
                recovery_hint="Drop fe= for R etwfe's linear design.",
                diagnostics={"fe": fe, "family": family},
            )
        jwdid_linear = _normalise_hettype(hettype) != "timecohort" or fe_key in (
            "unit",
            "ivar",
            "id",
            "individual",
        )
        if jwdid_linear:
            # Stata jwdid without method(): reghdfe with unit and period
            # effects, any hettype, jwdid's covariate design -- served by
            # the design / aggregation code of the nonlinear branch.
            if fe_key == "cohort":
                raise MethodIncompatibility(
                    f"etwfe(hettype={hettype!r}) on the linear model absorbs "
                    "unit fixed effects (Stata jwdid); fe='cohort' is only "
                    "available with the saturated default design.",
                    recovery_hint="Drop fe= (or pass fe='unit') with hettype=.",
                    diagnostics={"fe": fe, "hettype": hettype},
                )
            for arg_name, arg_val, bad in (
                ("panel", panel, not panel),
                ("weights", weights, weights is not None),
            ):
                if bad:
                    raise MethodIncompatibility(
                        "etwfe(hettype= / fe='unit') on the linear model does "
                        f"not support {arg_name}={arg_val!r} yet.",
                        recovery_hint="Drop the option, or use the saturated "
                        "default design (no hettype=, no fe=).",
                        diagnostics={arg_name: arg_val, "hettype": hettype},
                    )
            _result = _etwfe_glm(
                data=data,
                y=y,
                group=group,
                time=time,
                first_treat=first_treat,
                family="gaussian",
                controls=controls,
                cluster=cluster,
                alpha=alpha,
                cgroup=cgroup,
                fe="unit",
                scale="link",
                hettype=hettype,
                xvar=xvar,
            )
        else:
            _result = _dispatch_etwfe_impl(
                data=data,
                y=y,
                group=group,
                time=time,
                first_treat=first_treat,
                controls=controls,
                cluster=cluster,
                alpha=alpha,
                xvar=xvar,
                panel=panel,
                cgroup=cgroup,
                weights=weights,
                agg_weights=agg_weights,
            )
    if _last_cohort_ref is not None:
        _result.model_info["last_cohort_reference"] = _last_cohort_ref
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.etwfe",
            params={
                "y": y,
                "group": group,
                "time": time,
                "first_treat": first_treat,
                "controls": controls,
                "cluster": cluster,
                "alpha": alpha,
                "xvar": list(xvar) if isinstance(xvar, (list, tuple)) else xvar,
                "panel": panel,
                "cgroup": cgroup,
                "weights": weights,
                "agg_weights": agg_weights,
                "fe": fe,
                "scale": scale,
                "hettype": hettype,
                "separated": separated,
                "response_se": response_se,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _trim_to_notyet_support(
    data: pd.DataFrame,
    *,
    time: str,
    first_treat: str,
    cgroup: str,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Restrict a panel in which every unit is eventually treated to the
    periods that still have a not-yet-treated comparison.

    With no never-treated unit and no cohort adopting after the last observed
    period, nothing is untreated from the last cohort's adoption date on, so
    the period effects of those dates are not identified: the saturated
    design is rank-deficient there and ``pinv`` returns arbitrary numbers
    (observed: an overall ATT of 4.7e11 on a panel whose true ATT is 68).
    R ``etwfe`` handles the case by taking the last cohort as the reference
    group and dropping the periods from its adoption onwards
    (``.Dtreat = NA`` for ``t >= gref``); the same sample is what
    ``did_imputation, autosample`` and Callaway-Sant'Anna's not-yet-treated
    comparison keep. This does that, and says so.

    Returns the (possibly trimmed) data and, when it trimmed, a record for
    ``model_info['last_cohort_reference']``.
    """
    if cgroup != "notyet" or data.empty:
        return data, None
    ft = data[first_treat].replace(0, np.nan)
    max_time = data[time].max()
    # A never-treated unit, or a cohort adopting after the panel ends, is an
    # untreated comparison in every observed period: nothing to trim.
    if ft.isna().any() or (ft > max_time).any():
        return data, None
    cohorts = sorted(ft.unique().tolist())
    g_ref = cohorts[-1]
    keep = (data[time] < g_ref).to_numpy()
    trimmed = data.loc[keep].copy()
    # Inside the retained window the last cohort is never treated.
    trimmed[first_treat] = trimmed[first_treat].where(ft.loc[keep] != g_ref, np.nan)
    n_dropped = int((~keep).sum())
    ft_kept = trimmed[first_treat].replace(0, np.nan)
    if not (ft_kept.notna() & (trimmed[time] >= ft_kept)).any():
        raise DataInsufficient(
            "Every unit is eventually treated and no treated observation has "
            "a not-yet-treated comparison: once the periods from the last "
            f"cohort's adoption ({g_ref}) are set aside, no cohort × post "
            "cell is left to identify an ATT from.",
            recovery_hint=(
                "A single adoption date with no untreated group identifies "
                "nothing by difference-in-differences; add never-treated "
                "units or a later-treated cohort."
            ),
            diagnostics={
                "function": "etwfe",
                "cohorts": cohorts,
                "reference_cohort": g_ref,
            },
        )
    warnings.warn(
        "etwfe: every unit is eventually treated, so no untreated comparison "
        f"exists from period {g_ref} on. The last cohort ({g_ref}) is used as "
        f"the not-yet-treated reference and the {n_dropped} observation(s) "
        f"with {time} >= {g_ref} are dropped, as R etwfe does; effects for "
        "that cohort and those periods are not identified.",
        UserWarning,
        stacklevel=3,
    )
    return trimmed, {
        "reference_cohort": g_ref,
        "n_obs_dropped": n_dropped,
        "last_period_kept": trimmed[time].max(),
    }


def _etwfe_with_simple_headline(
    result: CausalResult,
    *,
    cgroup: str,
    panel: bool,
    alpha: float,
    method: str,
    source_branch: str,
    agg_weights: str = "estimation",
) -> CausalResult:
    """Return an ETWFE result whose headline matches R ``emfx(type='simple')``.

    The low-level ETWFE branches keep cohort/event coefficients in
    ``detail``/``model_info``.  R ``etwfe`` and Stata ``jwdid`` report the
    simple ATT as a treated-observation-weighted average over post-treatment
    cohort-time marginal effects, so the public ``sp.etwfe`` headline should
    use the same aggregation instead of the older cohort-size average.
    """
    simple = etwfe_emfx(
        result,
        type="simple",
        alpha=alpha,
        weighting="treated",
        agg_weights=agg_weights,
    )
    model_info = dict(result.model_info or {})
    model_info.setdefault("weights", None)
    model_info.update(
        {
            "cgroup": cgroup,
            "panel": panel,
            "agg_weights": agg_weights,
            "headline_aggregation": "emfx_simple",
            "headline_weighting": "treated_observations",
            "headline_source_branch": source_branch,
            # Recorded for sp.validation_scope; the GLM branch writes its own.
            "family": "linear",
        }
    )
    return CausalResult(
        method=method,
        estimand="Overall ATT (treated-observation weighted)",
        estimate=float(simple.estimate),
        se=float(simple.se),
        pvalue=float(simple.pvalue) if simple.pvalue is not None else np.nan,
        ci=simple.ci,
        alpha=alpha,
        n_obs=int(result.n_obs),
        detail=result.detail,
        model_info=model_info,
        _citation_key="wooldridge_twfe",
    )


def _dispatch_etwfe_impl(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    xvar: Optional[Any] = None,  # Union[str, List[str]]
    panel: bool = True,
    cgroup: str = "notyet",
    weights: Optional[str] = None,
    agg_weights: str = "estimation",
) -> CausalResult:
    """
    Extended Two-Way Fixed Effects (ETWFE) — Wooldridge (2021).

    Explicit API matching the R package ``etwfe`` (McDermott, 2023).
    The public headline matches R ``etwfe::emfx(type='simple')`` /
    Stata ``jwdid, estat simple``: a treated-observation-weighted simple
    ATT.  The ``cgroup`` argument governs the identifying comparison
    group (``'notyet'`` by default, ``'nevertreated'`` for never-treated
    controls).

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset (long format).
    y : str
        Outcome variable.
    group : str
        Unit identifier.
    time : str
        Time period variable.
    first_treat : str
        Column with first-treatment period; NaN or 0 for never-treated.
    controls : list of str, optional
        Time-varying covariates, entering additively (Stata ``jwdid``'s
        ``exovar()``; R ``etwfe`` interacts demeaned controls with the
        treatment cells instead, a different model). With ``fe='unit'``
        the fit reproduces ``jwdid, exovar()``. The default design uses
        cohort rather than unit effects, which gives the same cell
        coefficients without controls but not with a time-varying one,
        and has no external reference in that case.
    cluster : str, optional
        Cluster variable for SE (defaults to ``group``).
    alpha : float, default 0.05
        Significance level.
    weights : str, optional
        Column of non-negative observation weights.  The cohort-by-period
        regression is then weighted least squares with R ``fixest``
        ``weights=`` / Stata ``reghdfe [pw=]`` semantics (rows with weight
        0 are dropped from the estimation sample, as both do), and the
        cluster-robust covariance keeps the unweighted fit's small-sample
        convention ``G / (G - 1) * (n - 1) / (n - K)`` with ``n`` counting
        observations.  Not available together with ``xvar`` or the
        nonlinear ``family`` branch (both raise).
    agg_weights : {'estimation', 'unit'}, default 'estimation'
        How the cohort-by-period cells are averaged into the ``emfx``
        aggregates (``simple`` / ``group`` / ``event`` / ``calendar``)
        when ``weights`` is set; the two rules coincide otherwise.

        ``'estimation'`` -- Stata ``jwdid, estat``: cell ``(g, t)`` enters
        with the sum of the estimation weights over its treated
        observations, ``W_{g,t} = sum_{i in g} w_{i,t}``.  Under
        never-treated controls without covariates this makes the
        ``simple`` aggregate identical to the weighted Callaway--Sant'Anna
        simple ATT (an estimand-level identity), which is why it is the
        default.

        ``'unit'`` -- R ``etwfe::emfx``: every treated observation carries
        weight one, ``W_{g,t} = n_{g,t}``, and the estimation weights only
        enter the regression.  With weights that are constant within unit
        on a balanced panel the two rules agree on the ``group`` rows (all
        post cells of a cohort share one weight) and differ on ``simple``
        / ``event`` / ``calendar``, which mix cohorts.

    Returns
    -------
    CausalResult
        Treated-observation-weighted simple ATT with cohort-level detail
        and event-study coefficients in ``model_info['event_study']``.

    Notes
    -----
    Naming map to the R ``etwfe`` package:

    ============================  ========================================
    R ``etwfe`` argument          ``sp.etwfe`` argument
    ============================  ========================================
    ``fml = y ~ 1``               ``y='y'``
    ``tvar = time``               ``time='time'``
    ``gvar = first_treat``        ``first_treat='first_treat'``
    ``ivar = unit``               ``group='unit'``
    ``xvar`` (covariate het.)     ``xvar='x1'`` or ``xvar=['x1','x2']``
    ``vcov = ~cluster``           ``cluster='cluster'``
    ``weights = ~w``              ``weights='w'``
    ``emfx`` (N = 1 per obs)      ``agg_weights='unit'``
    ============================  ========================================

    For aggregated marginal effects (R ``emfx`` equivalents), call
    :func:`statspai.did.etwfe_emfx` on the returned result.

    References
    ----------
    Wooldridge, J.M. (2021). "Two-Way Fixed Effects, the Two-Way Mundlak
    Regression, and Difference-in-Differences Estimators." [@wooldridge2021two]

    McDermott, G. (2023). ``etwfe``: Extended Two-Way Fixed Effects.
    https://grantmcdermott.com/etwfe/ [@mcdermott2022etwfe]

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=10, staggered=True, seed=0)
    >>> res = sp.etwfe(df, y='y', group='unit',
    ...                time='time', first_treat='first_treat')
    >>> type(res).__name__
    'CausalResult'
    >>> text = res.summary()

    See Also
    --------
    wooldridge_did : Historical saturated TWFE helper.
    callaway_santanna : CS (2021) group-time ATT estimator.
    aggte : Aggregation of group-time ATTs (event/group/calendar/simple).
    """
    # Validate cgroup
    if cgroup not in ("notyet", "nevertreated"):
        raise MethodIncompatibility(
            f"cgroup must be 'notyet' or 'nevertreated'; got {cgroup!r}"
        )
    _validate_agg_weights(agg_weights)

    # Normalise xvar to a list (or None)
    xvar_list: Optional[List[str]] = None
    if xvar is not None:
        xvar_list = [xvar] if isinstance(xvar, str) else list(xvar)
        if weights is not None:
            # The covariate-moderated branch runs on a two-way within
            # transformation that has no weighted closed form; refuse rather
            # than silently fit an unweighted regression.
            raise MethodIncompatibility(
                "etwfe(weights=...) is not yet supported together with xvar.",
                recovery_hint="Drop weights, or drop xvar and use controls=.",
                diagnostics={"weights": weights, "xvar": xvar_list},
            )
        # C1/C2: fail fast on missing or constant xvars
        for xv in xvar_list:
            if xv not in data.columns:
                raise KeyError(f"xvar {xv!r} not found in data.columns")
            col = pd.to_numeric(data[xv], errors="coerce")
            finite = col.dropna()
            if len(finite) < 2:
                raise DataInsufficient(
                    f"xvar {xv!r} has fewer than 2 non-NaN rows "
                    f"(found {len(finite)}); cannot estimate heterogeneity."
                )
            if float(finite.std()) < 1e-12:
                raise DataInsufficient(
                    f"xvar {xv!r} is (near-)constant — no heterogeneity "
                    "slope can be identified. Drop it or choose another column."
                )

    # C3: explicit guard for the unimplemented combination
    if not panel and cgroup == "nevertreated":
        raise NotImplementedError(
            "cgroup='nevertreated' with panel=False is not yet supported. "
            "Use either panel=True + cgroup='nevertreated', or "
            "panel=False + cgroup='notyet'."
        )

    # Dispatch to the right implementation.
    if not panel:
        # Repeated cross-section: no unit FE, replace with cohort dummies.
        base = _etwfe_repeated_cs(
            data=data,
            y=y,
            time=time,
            first_treat=first_treat,
            xvar=xvar_list,
            controls=controls,
            cluster=cluster,
            alpha=alpha,
            cgroup=cgroup,
            weights=weights,
            agg_weights=agg_weights,
        )
        return _etwfe_with_simple_headline(
            base,
            cgroup=cgroup,
            panel=False,
            alpha=alpha,
            method="Wooldridge (2021) ETWFE — repeated cross-section",
            source_branch="repeated_cross_section",
            agg_weights=agg_weights,
        )
    if cgroup == "nevertreated":
        ft_local = data[first_treat].replace(0, np.nan)
        if not ft_local.isna().any():
            raise DataInsufficient(
                "cgroup='nevertreated' requires at least one never-treated "
                "unit (first_treat NaN / 0), but none were found."
            )
        # R etwfe(cgroup='never') / Stata jwdid never-control semantics: the
        # cohort-by-period cell design with every period of every cohort
        # dummied except g - 1 (R `.Dtreat = t != g - 1`), so only the
        # never-treated units identify the period effects.  Same cohort +
        # period fixed-effect basis as the not-yet-treated branch below, so
        # the two control groups differ only in their cells and every
        # `etwfe_emfx` aggregation honours the control group automatically.
        if xvar_list is not None:
            base = _etwfe_with_xvar(
                data=data,
                y=y,
                group=group,
                time=time,
                first_treat=first_treat,
                xvar=xvar_list,
                controls=controls,
                cluster=cluster,
                alpha=alpha,
            )
        else:
            base = _etwfe_repeated_cs(
                data=data,
                y=y,
                time=time,
                first_treat=first_treat,
                xvar=None,
                controls=controls,
                cluster=cluster or group,
                alpha=alpha,
                cgroup="never",
                weights=weights,
                agg_weights=agg_weights,
            )
        return _etwfe_with_simple_headline(
            base,
            cgroup="nevertreated",
            panel=True,
            alpha=alpha,
            method="Wooldridge (2021) ETWFE — never-treated control",
            source_branch="never_treated_event_cells",
            agg_weights=agg_weights,
        )
    if xvar_list is None:
        # R etwfe's default cgroup='notyet' and Stata jwdid identify the
        # simple marginal effect from cohort-time event cells, with treated
        # observations as aggregation weights.  The repeated-CS design branch
        # implements that event-cell basis; for panel calls, default inference
        # still clusters on the unit id unless the user supplied cluster=.
        base = _etwfe_repeated_cs(
            data=data,
            y=y,
            time=time,
            first_treat=first_treat,
            xvar=None,
            controls=controls,
            cluster=cluster or group,
            alpha=alpha,
            cgroup="notyet",
            weights=weights,
            agg_weights=agg_weights,
        )
        return _etwfe_with_simple_headline(
            base,
            cgroup="notyet",
            panel=True,
            alpha=alpha,
            method="Wooldridge (2021) ETWFE — not-yet-treated control",
            source_branch="notyet_event_cells",
            agg_weights=agg_weights,
        )
    # Covariate-moderated ETWFE preserves the historical xvar branch (including
    # slope columns in result.detail), then reports the same treated-observation
    # weighted simple headline as R emfx.
    base = _etwfe_with_xvar(
        data=data,
        y=y,
        group=group,
        time=time,
        first_treat=first_treat,
        xvar=xvar_list,
        controls=controls,
        cluster=cluster,
        alpha=alpha,
    )
    return _etwfe_with_simple_headline(
        base,
        cgroup="notyet",
        panel=True,
        alpha=alpha,
        method="Wooldridge (2021) ETWFE — not-yet-treated control",
        source_branch="notyet_covariate_branch",
        agg_weights=agg_weights,
    )


def _etwfe_with_xvar(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    xvar: List[str],
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """ETWFE with covariate-moderated heterogeneity (R etwfe's ``xvar``).

    Supports single or multiple covariates. For each cohort ``g`` and
    each xvar ``x_j``, adds an interaction between the cohort × post
    dummy and ``(x_j - mean(x_j))``. The main cohort coefficient is
    ATT(g) evaluated at the sample mean of every xvar; each slope
    measures how ATT(g) shifts per unit of ``x_j``.
    """
    df = data.copy()
    ft = df[first_treat].replace(0, np.nan)
    df["_ft"] = ft

    periods = sorted(df[time].unique())
    cohorts = sorted(df.loc[df["_ft"].notna(), "_ft"].unique())
    if len(cohorts) == 0:
        raise DataInsufficient("No treated cohorts found. Check 'first_treat' column.")

    # Demean outcome
    df["_y"] = df[y].astype(float)
    unit_mean = df.groupby(group)["_y"].transform("mean")
    time_mean = df.groupby(time)["_y"].transform("mean")
    grand_mean = df["_y"].mean()
    df["_y_dm"] = df["_y"] - unit_mean - time_mean + grand_mean

    # Center every xvar by its grand mean so baseline ATT is evaluated
    # at (x1=mean, x2=mean, …).
    xc_cols: List[str] = []
    x_centers: Dict[str, float] = {}
    for x in xvar:
        raw = df[x].astype(float)
        ctr = float(raw.mean())
        x_centers[x] = ctr
        df[f"_xc_{x}"] = raw - ctr
        xc_cols.append(f"_xc_{x}")

    # Build cohort × post dummies AND cohort × post × xc_j interactions.
    base_cols: List[str] = []
    slope_cols_by_cohort: Dict[int, List[str]] = {}
    for g in cohorts:
        mask_post = (df["_ft"] == g) & (df[time] >= g)
        b = f"_coh{int(g)}_post"
        df[b] = mask_post.astype(float)
        base_cols.append(b)
        slope_cols_by_cohort[int(g)] = []
        for x in xvar:
            s = f"_coh{int(g)}_post_x_{x}"
            df[s] = df[b] * df[f"_xc_{x}"]
            slope_cols_by_cohort[int(g)].append(s)

    all_slope = [c for v in slope_cols_by_cohort.values() for c in v]
    all_inter = base_cols + all_slope

    # H4 fix: explicit name-to-index map so slope lookups never rely on
    # implicit ordering. Column 0 is the constant; columns [1:] are
    # X_cols = [f"{c}_dm" for c in all_inter] + ctrl_dm_cols in order.
    coef_index: Dict[str, int] = {"_const": 0}
    for i, col in enumerate(all_inter, start=1):
        coef_index[col] = i

    # Demean interactions via FE projection
    for col in all_inter:
        u_m = df.groupby(group)[col].transform("mean")
        t_m = df.groupby(time)[col].transform("mean")
        g_m = df[col].mean()
        df[f"{col}_dm"] = df[col] - u_m - t_m + g_m

    # Demean controls
    ctrl_dm_cols: List[str] = []
    if controls:
        for c in controls:
            df[f"_ctrl_{c}"] = df[c].astype(float)
            u_m = df.groupby(group)[f"_ctrl_{c}"].transform("mean")
            t_m = df.groupby(time)[f"_ctrl_{c}"].transform("mean")
            g_m = df[f"_ctrl_{c}"].mean()
            df[f"_ctrl_{c}_dm"] = df[f"_ctrl_{c}"] - u_m - t_m + g_m
            ctrl_dm_cols.append(f"_ctrl_{c}_dm")

    keep = ["_y_dm"] + [f"{c}_dm" for c in all_inter] + ctrl_dm_cols
    valid = df[keep].notna().all(axis=1)
    dfv = df.loc[valid].reset_index(drop=True)

    y_vec = dfv["_y_dm"].values
    X_cols = [f"{c}_dm" for c in all_inter] + ctrl_dm_cols
    X = dfv[X_cols].values
    X = np.column_stack([np.ones(len(y_vec)), X])

    cl_arr = dfv[cluster].values if cluster else dfv[group].values
    beta, se, vcov = _ols_fit(X, y_vec, cluster=cl_arr)
    n_obs = len(y_vec)
    df_resid = max(n_obs - X.shape[1], 1)

    k = len(cohorts)
    p_x = len(xvar)  # retained for the single-xvar backward-compat block

    cohort_results = []
    for g in cohorts:
        # H4 fix: look up baseline and slope indices by coefficient name
        # rather than by arithmetic position, so future column-order
        # changes cannot silently mis-attribute.
        b_idx = coef_index[f"_coh{int(g)}_post"]
        att = float(beta[b_idx])
        att_se = float(se[b_idx])
        p = float(2 * stats.t.sf(abs(att / att_se) if att_se > 0 else 0, df_resid))
        row: Dict[str, Any] = {
            "cohort": int(g),
            "att_at_xmean": att,
            "att_se": att_se,
            "att_pvalue": p,
        }
        for x in xvar:
            s_idx = coef_index[f"_coh{int(g)}_post_x_{x}"]
            slope = float(beta[s_idx])
            slope_se = float(se[s_idx])
            p_s = float(
                2 * (stats.t.sf(abs(slope / slope_se) if slope_se > 0 else 0, df_resid))
            )
            row[f"slope_{x}"] = slope
            row[f"slope_{x}_se"] = slope_se
            row[f"slope_{x}_pvalue"] = p_s
        # Backward-compat: if exactly one xvar, alias to the older name
        if p_x == 1:
            only = xvar[0]
            row["slope_wrt_x"] = row[f"slope_{only}"]
            row["slope_se"] = row[f"slope_{only}_se"]
            row["slope_pvalue"] = row[f"slope_{only}_pvalue"]
        row["n_obs"] = int((dfv["_ft"] == g).sum())
        row["n_treated_obs"] = int(((dfv["_ft"] == g) & (dfv[time] >= g)).sum())
        cohort_results.append(row)

    detail = pd.DataFrame(cohort_results)
    sizes = detail["n_obs"].values.astype(float)
    weights_vec = sizes / sizes.sum() if sizes.sum() > 0 else np.ones(k) / k
    att_overall = float(weights_vec @ detail["att_at_xmean"].values)
    # H4 fix: extract baseline-coefficient vcov by explicit index lookup
    base_idx = np.array([coef_index[f"_coh{int(g)}_post"] for g in cohorts])
    base_vcov = vcov[np.ix_(base_idx, base_idx)]
    att_se_overall = float(np.sqrt(weights_vec @ base_vcov @ weights_vec))
    t_stat = att_overall / att_se_overall if att_se_overall > 0 else np.nan
    p_overall = float(2 * stats.t.sf(abs(t_stat), df_resid))
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    ci = (att_overall - t_crit * att_se_overall, att_overall + t_crit * att_se_overall)

    model_info = {
        "n_cohorts": k,
        "cohorts": [int(g) for g in cohorts],
        "n_periods": len(periods),
        "n_units": df[group].nunique(),
        "controls": controls or [],
        "cluster_var": cluster or group,
        "xvar": list(xvar),
        "xvar_means": x_centers,
        "heterogeneity": (
            "ATT(g) = baseline(g) + Σ_j slope_j(g) * " "(x_j - mean(x_j))"
        ),
        "cohort_weighting": "cohort",
        "cohort_vcov": base_vcov,
    }

    x_label = ", ".join(f"{x}={x_centers[x]:.4g}" for x in xvar)
    return CausalResult(
        method="Wooldridge (2021) ETWFE with covariate heterogeneity",
        estimand=f"ATT at [{x_label}] (sample means)",
        estimate=att_overall,
        se=att_se_overall,
        pvalue=p_overall,
        ci=ci,
        alpha=alpha,
        n_obs=n_obs,
        detail=detail,
        model_info=model_info,
        _citation_key="wooldridge_twfe",
    )


# ═══════════════════════════════════════════════════════════════════════
#  1a. ETWFE — repeated cross-section (ivar=NULL in R etwfe)
# ═══════════════════════════════════════════════════════════════════════


def _warn_if_rank_deficient(X: np.ndarray) -> None:
    """H6: the cohort/period-dummy design is more collinearity-prone than the
    within-demeaned panel path.  Say so loudly; results still come from
    ``pinv`` so the caller can inspect them."""
    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        deficit = X.shape[1] - rank
        warnings.warn(
            f"etwfe: design matrix is rank-deficient by "
            f"{deficit} column(s) (rank={rank}, ncol={X.shape[1]}). "
            "Falling back to pseudoinverse; some coefficients are "
            "arbitrary linear combinations. Consider dropping collinear "
            "controls or shortening the event window.",
            RuntimeWarning,
            stacklevel=4,
        )


def _aggregate_cells(
    es: pd.DataFrame,
    event_vcov: Optional[np.ndarray],
    key_col: str,
    weight_col: str,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], bool]:
    """Average cohort-by-period cell coefficients within each level of ``key_col``.

    This is R ``etwfe::emfx``'s aggregation: every treated observation carries
    weight ``N = 1`` and its marginal effect is the coefficient of its
    cohort-by-period cell, so a level of ``key_col`` (cohort, event time or
    calendar time) reports the cell coefficients averaged with the cells' own
    observation counts (``weight_col``).  Standard errors are the delta method
    through the cell covariance ``event_vcov`` (``_vcov_idx`` is 1-based into
    it); when that block is unavailable the cells are treated as independent.

    Returns ``(rows, vcov, vcov_based)``: ``rows`` has ``key_col``,
    ``estimate``, ``se``, ``n_cells`` and ``n_obs`` (the summed weights),
    ``vcov`` is the covariance of the ``rows`` estimates (``None`` in the
    independent-cell fallback).
    """
    if es.empty:
        raise DataInsufficient(
            f"No cohort × period cells to aggregate by {key_col!r}.",
            recovery_hint="Check the treated cohorts and the event window.",
        )
    keys = sorted(es[key_col].unique().tolist())
    key_arr = es[key_col].to_numpy()
    est_cells = es["estimate"].to_numpy(dtype=float)
    if weight_col in es.columns:
        wts = es[weight_col].to_numpy(dtype=float)
        wts = np.where(np.isfinite(wts), wts, 0.0)
    else:
        wts = np.ones(len(es), dtype=float)
    W = np.zeros((len(keys), len(es)), dtype=float)
    raw_sum = np.zeros(len(keys), dtype=float)
    n_cells = np.zeros(len(keys), dtype=int)
    for r, k in enumerate(keys):
        mask = key_arr == k
        w = wts * mask
        raw_sum[r] = float(w.sum())
        n_cells[r] = int(mask.sum())
        if raw_sum[r] <= 0:
            # Pre-treatment leads carry no treated observations: fall back
            # to an unweighted mean of the cells at that level.
            w = mask.astype(float)
        W[r] = w / w.sum()
    est = W @ est_cells
    vcov_based = event_vcov is not None and "_vcov_idx" in es.columns
    if vcov_based:
        idx = es["_vcov_idx"].astype(int).to_numpy() - 1
        V_sub = np.asarray(event_vcov, dtype=float)[np.ix_(idx, idx)]
        V_full = np.asarray(W @ V_sub @ W.T, dtype=float)
        V_agg: Optional[np.ndarray] = V_full
        se = np.sqrt(np.maximum(np.diag(V_full), 0.0))
    else:
        se_cells = es["se"].to_numpy(dtype=float)
        se = np.sqrt(np.sum((W * se_cells) ** 2, axis=1))
        V_agg = None
    rows = pd.DataFrame(
        {
            key_col: keys,
            "estimate": est,
            "se": se,
            "n_cells": n_cells,
            "n_obs": raw_sum,
        }
    )
    return rows, V_agg, vcov_based


def _etwfe_repeated_cs(
    data: pd.DataFrame,
    y: str,
    time: str,
    first_treat: str,
    xvar: Optional[List[str]] = None,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    cgroup: str = "notyet",
    weights: Optional[str] = None,
    agg_weights: str = "estimation",
) -> CausalResult:
    """ETWFE with cohort and period fixed effects (no unit fixed effects).

    This is the design R ``etwfe::etwfe(ivar = NULL)`` fits (``fe = "feo"``):
    cohort dummies, period dummies and cohort-by-period treatment cells, with
    fixest's default small-sample factor ``(n - 1) / (n - K)`` counting every
    column of that design (``K`` = cells + cohort levels + period levels - 1
    + constant).  It serves ``panel=False`` and, without ``xvar``, the panel
    entry points too: in a balanced panel the cell coefficients equal those
    of the unit-fixed-effects regression and the cluster-robust "meat" is
    identical, so only ``K`` separates it from Stata ``jwdid`` (see
    :func:`etwfe_emfx` Notes).

    ``cgroup='notyet'``
        Cells for ``t >= g`` (R ``.Dtreat = t >= g`` with ``ref2 = tref``):
        the pre-treatment rows of every cohort act as controls.
    ``cgroup='never'``
        Cells for every period of every cohort except the period before
        adoption (R ``.Dtreat = t != g - 1``, no ``ref2``): only never-treated
        units identify the period effects and each cell is the 2x2 comparison
        against ``g - 1``, so without covariates the aggregates coincide with
        Callaway-Sant'Anna's never-treated aggregates.

    ``result.detail`` is the R ``emfx(type = "group")`` table -- per-cohort
    averages of the post-treatment cells weighted by their observation
    counts, with delta-method standard errors -- and
    ``model_info['event_study']`` / ``['event_vcov']`` hold the cells and
    their covariance for :func:`etwfe_emfx`.  With ``xvar`` the historical
    pooled cohort x post design with centred slopes is kept (no cells).

    ``weights`` turns the fit into weighted least squares (fixest
    ``weights=`` / reghdfe ``[pw=]``).  Every cell then carries both its
    observation count (``n_cell_obs``) and its estimation-weight total
    (``w_cell_obs``); ``agg_weights`` selects which of the two averages the
    post cells into the cohort ``detail`` rows (``'estimation'`` = Stata
    ``jwdid, estat``, ``'unit'`` = R ``emfx``).
    """
    if cgroup == "nevertreated":
        cgroup = "never"
    _validate_agg_weights(agg_weights)
    if cgroup not in ("notyet", "never"):
        raise MethodIncompatibility(
            f"cgroup must be 'notyet' or 'never'; got {cgroup!r}"
        )
    df = data.copy()
    ft = df[first_treat].replace(0, np.nan)
    df["_ft"] = ft

    periods = sorted(df[time].unique())
    tref = periods[0]
    cohorts = sorted(df.loc[df["_ft"].notna(), "_ft"].unique())
    if len(cohorts) == 0:
        raise DataInsufficient("No treated cohorts found. Check 'first_treat' column.")
    if cgroup == "never" and not df["_ft"].isna().any():
        raise DataInsufficient(
            "cgroup='nevertreated' requires at least one never-treated "
            "unit (first_treat NaN / 0), but none were found."
        )

    df["_y"] = df[y].astype(float)
    if weights is not None:
        df["_w"] = df[weights].astype(float)

    # Cohort dummies (leave never-treated as baseline), time dummies
    coh_dummies: List[str] = []
    for g in cohorts:
        col = f"_CG_{int(g)}"
        df[col] = (df["_ft"] == g).astype(float)
        coh_dummies.append(col)
    time_dummies: List[str] = []
    for tt in periods[1:]:  # first period as baseline
        col = f"_T_{int(tt)}"
        df[col] = (df[time] == tt).astype(float)
        time_dummies.append(col)

    ctrl_cols: List[str] = []
    if controls:
        for c in controls:
            df[f"_ctrl_{c}"] = df[c].astype(float)
            ctrl_cols.append(f"_ctrl_{c}")

    xvar = xvar or []
    if xvar:
        if cgroup == "never":
            raise MethodIncompatibility(
                "xvar with cgroup='nevertreated' is not available in the "
                "cohort/period fixed-effect design.",
                recovery_hint="Use panel=True (unit fixed effects) or drop xvar.",
            )
        if weights is not None:
            raise MethodIncompatibility(
                "etwfe(weights=...) is not yet supported together with xvar.",
                recovery_hint="Drop weights, or drop xvar and use controls=.",
                diagnostics={"weights": weights, "xvar": list(xvar)},
            )
        # ── historical covariate-moderated design: cohort × post + slopes ──
        base_cols: List[str] = []
        for g in cohorts:
            col = f"_coh{int(g)}_post"
            df[col] = ((df["_ft"] == g) & (df[time] >= g)).astype(float)
            base_cols.append(col)
        slope_cols: List[str] = []
        x_centers: Dict[str, float] = {}
        for x in xvar:
            raw = df[x].astype(float)
            ctr = float(raw.mean())
            x_centers[x] = ctr
            df[f"_xc_{x}"] = raw - ctr
        for g in cohorts:
            for x in xvar:
                col = f"_coh{int(g)}_post_x_{x}"
                df[col] = df[f"_coh{int(g)}_post"] * df[f"_xc_{x}"]
                slope_cols.append(col)

        design_cols = coh_dummies + time_dummies + base_cols + slope_cols + ctrl_cols
        keep = ["_y"] + design_cols
        valid = df[keep].notna().all(axis=1)
        dfv = df.loc[valid].reset_index(drop=True)

        y_vec = dfv["_y"].values
        X = np.column_stack(
            [np.ones(len(y_vec))] + [dfv[c].values for c in design_cols]
        )
        cl_arr = dfv[cluster].values if cluster else None
        _warn_if_rank_deficient(X)
        beta, se, vcov = _ols_fit(X, y_vec, cluster=cl_arr)
        n_obs = len(y_vec)
        df_resid = max(n_obs - X.shape[1], 1)

        # ATT(g) coefficients — their index in X:
        # 0: const, [cohort dummies], [time dummies], [base cols], [slopes], [ctrl]
        base_start = 1 + len(coh_dummies) + len(time_dummies)
        k = len(cohorts)
        cohort_rows = []
        for i, g in enumerate(cohorts):
            idx = base_start + i
            att = float(beta[idx])
            s_ = float(se[idx])
            p = float(2 * stats.t.sf(abs(att / s_) if s_ > 0 else 0, df_resid))
            n_g = int((dfv["_ft"] == g).sum())
            n_treated_g = int(((dfv["_ft"] == g) & (dfv[time] >= g)).sum())
            cohort_rows.append(
                {
                    "cohort": int(g),
                    "att": att,
                    "se": s_,
                    "tstat": att / s_ if s_ > 0 else np.nan,
                    "pvalue": p,
                    "n_obs": n_g,
                    "n_treated_obs": n_treated_g,
                }
            )
        detail = pd.DataFrame(cohort_rows)
        sizes = detail["n_obs"].values.astype(float)
        w = sizes / sizes.sum() if sizes.sum() > 0 else np.ones(k) / k
        att_overall = float(w @ detail["att"].values)
        base_vcov = vcov[base_start : base_start + k, base_start : base_start + k]
        att_se = float(np.sqrt(w @ base_vcov @ w))
        t_stat = att_overall / att_se if att_se > 0 else np.nan
        p_overall = float(2 * stats.t.sf(abs(t_stat), df_resid))
        t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
        ci = (att_overall - t_crit * att_se, att_overall + t_crit * att_se)

        return CausalResult(
            method="Wooldridge (2021) ETWFE — repeated cross-section",
            estimand="Overall ATT (no unit FE)",
            estimate=att_overall,
            se=att_se,
            pvalue=p_overall,
            ci=ci,
            alpha=alpha,
            n_obs=n_obs,
            detail=detail,
            model_info={
                "n_cohorts": k,
                "cohorts": [int(g) for g in cohorts],
                "n_periods": len(periods),
                "panel": False,
                "controls": controls or [],
                "xvar": list(xvar),
                "xvar_means": x_centers,
                "cgroup": cgroup,
                "cohort_weighting": "cohort",
                "cohort_vcov": base_vcov,
                "event_study": None,
                "event_vcov": None,
            },
            _citation_key="wooldridge_twfe",
        )

    # ── cohort-by-period cells: R etwfe's `.Dtreat : i(gvar, i.tvar)` ──
    event_cols: List[str] = []
    event_meta: List[Tuple[int, int, int]] = []
    ref_period: Dict[int, int] = {}
    for g in cohorts:
        if cgroup == "never":
            # R: `.Dtreat = t != g - 1`.  The latest pre-adoption period is
            # the reference (identical for consecutive periods); a cohort
            # observed only from adoption onwards falls back to the first
            # period so the design stays full rank.
            pre = [tt for tt in periods if tt < g]
            ref_g = max(pre) if pre else tref
        else:
            # R: `.Dtreat = t >= g` with `ref2 = tref`.
            ref_g = tref
        ref_period[int(g)] = int(ref_g)
        for tt in periods:
            rel = int(tt - g)
            if cgroup == "notyet" and rel < 0:
                continue
            if tt == ref_g:
                continue
            col = (
                f"_coh{int(g)}_rel{rel}"
                if rel >= 0
                else f"_coh{int(g)}_rel_neg{abs(rel)}"
            )
            df[col] = ((df["_ft"] == g) & (df[time] == tt)).astype(float)
            event_cols.append(col)
            event_meta.append((int(g), rel, int(tt)))
    if not event_cols:
        raise DataInsufficient("No cohort × period cells could be created.")

    design_cols = coh_dummies + time_dummies + event_cols + ctrl_cols
    keep = ["_y"] + design_cols + (["_w"] if weights is not None else [])
    valid = df[keep].notna().all(axis=1)
    dfv = df.loc[valid].reset_index(drop=True)

    y_vec = dfv["_y"].values
    X = np.column_stack([np.ones(len(y_vec))] + [dfv[c].values for c in design_cols])
    cl_arr = dfv[cluster].values if cluster else None
    w_vec = dfv["_w"].to_numpy(dtype=float) if weights is not None else None
    _warn_if_rank_deficient(X)
    beta, se, vcov = _ols_fit(X, y_vec, cluster=cl_arr, weights=w_vec)
    n_obs = len(y_vec)
    df_resid = max(n_obs - X.shape[1], 1)
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)

    event_start = 1 + len(coh_dummies) + len(time_dummies)
    ev_rows = []
    for j, (col, (coh_val, rel_val, time_val)) in enumerate(
        zip(event_cols, event_meta)
    ):
        idx_j = event_start + j
        in_cell = ((dfv["_ft"] == coh_val) & (dfv[time] == time_val)).to_numpy()
        n_cell = int(in_cell.sum())
        # Estimation-weight total of the cell (= the count when unweighted):
        # Stata `jwdid, estat` aggregates the cells with these totals.
        w_cell = float(w_vec[in_cell].sum()) if w_vec is not None else float(n_cell)
        ev_rows.append(
            {
                "cohort": coh_val,
                "rel_time": rel_val,
                "period": time_val,
                "estimate": float(beta[idx_j]),
                "se": float(se[idx_j]),
                "_vcov_idx": j + 1,
                "n_cell_obs": n_cell,
                "n_treated_obs": n_cell if rel_val >= 0 else 0,
                "w_cell_obs": w_cell,
                "w_treated_obs": w_cell if rel_val >= 0 else 0.0,
            }
        )
    event_study_df = pd.DataFrame(ev_rows)
    event_vcov = vcov[
        event_start : event_start + len(event_cols),
        event_start : event_start + len(event_cols),
    ]

    # ── R emfx(type = "group"): per-cohort averages of the post cells ──
    post = event_study_df.loc[event_study_df["rel_time"] >= 0]
    if post.empty:
        raise DataInsufficient(
            "No treated post-treatment cells: every cohort adopts after the "
            "last observed period."
        )
    # Which cell total averages the post cells: the estimation-weight sums
    # (Stata `jwdid, estat`) or the observation counts (R `emfx`).  The two
    # columns are identical when the fit is unweighted.
    cell_weight_col = (
        "w_cell_obs"
        if (agg_weights == "estimation" and w_vec is not None)
        else "n_cell_obs"
    )
    grp, grp_vcov, _ = _aggregate_cells(
        post, event_vcov, key_col="cohort", weight_col=cell_weight_col
    )
    t_np = dfv[time].to_numpy()
    cohort_rows = []
    for _, r in grp.iterrows():
        att = float(r["estimate"])
        s_ = float(r["se"])
        t_ = att / s_ if s_ > 0 else np.nan
        p = float(2 * stats.t.sf(abs(t_), df_resid)) if np.isfinite(t_) else np.nan
        coh_mask = (dfv["_ft"] == r["cohort"]).to_numpy()
        post_mask = coh_mask & (t_np >= r["cohort"])
        cohort_rows.append(
            {
                "cohort": int(r["cohort"]),
                "att": att,
                "se": s_,
                "tstat": t_,
                "pvalue": p,
                "n_obs": int(coh_mask.sum()),
                "n_treated_obs": int(post_mask.sum()),
                "w_obs": (
                    float(w_vec[coh_mask].sum())
                    if w_vec is not None
                    else float(coh_mask.sum())
                ),
                "w_treated_obs": (
                    float(w_vec[post_mask].sum())
                    if w_vec is not None
                    else float(post_mask.sum())
                ),
            }
        )
    detail = pd.DataFrame(cohort_rows)
    k = len(detail)
    sizes = detail["n_obs"].values.astype(float)
    w = sizes / sizes.sum() if sizes.sum() > 0 else np.ones(k) / k
    att_overall = float(w @ detail["att"].values)
    base_vcov = np.asarray(grp_vcov, dtype=float)
    att_se = float(np.sqrt(max(w @ base_vcov @ w, 0.0)))
    t_stat = att_overall / att_se if att_se > 0 else np.nan
    p_overall = (
        float(2 * stats.t.sf(abs(t_stat), df_resid)) if np.isfinite(t_stat) else np.nan
    )
    ci = (att_overall - t_crit * att_se, att_overall + t_crit * att_se)
    n_clusters = int(len(np.unique(cl_arr))) if cl_arr is not None else None

    return CausalResult(
        method="Wooldridge (2021) ETWFE — repeated cross-section",
        estimand="Overall ATT (no unit FE)",
        estimate=att_overall,
        se=att_se,
        pvalue=p_overall,
        ci=ci,
        alpha=alpha,
        n_obs=n_obs,
        detail=detail,
        model_info={
            "n_cohorts": k,
            "cohorts": [int(c) for c in detail["cohort"]],
            "n_periods": len(periods),
            "panel": False,
            "controls": controls or [],
            "xvar": [],
            "xvar_means": {},
            "cgroup": cgroup,
            "cohort_weighting": "cohort",
            "cohort_vcov": base_vcov,
            "event_study": event_study_df,
            "event_vcov": event_vcov,
            "fixed_effects": "cohort + period",
            "reference_period": ref_period,
            "cluster_var": cluster,
            "weights": weights,
            "agg_weights": agg_weights,
            "cell_weight_column": cell_weight_col,
            "ssc": {
                "n": n_obs,
                "K": int(X.shape[1]),
                "n_clusters": n_clusters,
                "sum_weights": (
                    float(w_vec.sum()) if w_vec is not None else float(n_obs)
                ),
                "adjustment": (
                    "(n - 1) / (n - K) × G / (G - 1)"
                    if cl_arr is not None
                    else "HC1: n / (n - K)"
                ),
            },
        },
        _citation_key="wooldridge_twfe",
    )


# ═══════════════════════════════════════════════════════════════════════
#  1b. ETWFE — never-treated-only control (cgroup='nevertreated')
# ═══════════════════════════════════════════════════════════════════════


def _etwfe_never_only(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    xvar: Optional[List[str]] = None,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """ETWFE where each cohort is identified against never-treated only.

    Runs a separate ETWFE regression per cohort, each using only
    (units in cohort g) ∪ (never-treated units). Combines cohort ATTs
    with cohort-size weighting. Matches R ``etwfe(cgroup='never')``.

    Notes
    -----
    Per-cohort regressions each run on a different subset (cohort g +
    never-treated), so the cluster small-sample correction
    ``(n_cl/(n_cl-1))`` is evaluated cohort-by-cohort. Cohort SEs may
    therefore be slightly larger than a single full-sample regression
    would produce. The aggregated SE assumes cross-cohort independence,
    which is exact under this per-cohort design.
    """
    # H3 fix: compute the first-treat series locally rather than
    # writing a helper column back to the outer frame. Prevents
    # accidental column leakage when callers re-use `data`.
    df = data.copy()
    ft_local = df[first_treat].replace(0, np.nan)
    cohorts = sorted(ft_local.dropna().unique())
    if len(cohorts) == 0:
        raise DataInsufficient("No treated cohorts found. Check 'first_treat' column.")
    never_ids = df.loc[ft_local.isna(), group].unique()
    if len(never_ids) == 0:
        raise DataInsufficient(
            "cgroup='nevertreated' requires at least one never-treated "
            "unit (first_treat NaN / 0), but none were found."
        )

    rows: List[Dict[str, Any]] = []
    ses: List[float] = []
    for g in cohorts:
        coh_ids = df.loc[ft_local == g, group].unique()
        keep = np.concatenate([coh_ids, never_ids])
        sub = df.loc[df[group].isin(keep)].copy()
        if xvar:
            r = _etwfe_with_xvar(
                sub,
                y=y,
                group=group,
                time=time,
                first_treat=first_treat,
                xvar=xvar,
                controls=controls,
                cluster=cluster,
                alpha=alpha,
            )
        else:
            r = wooldridge_did(
                sub,
                y=y,
                group=group,
                time=time,
                first_treat=first_treat,
                controls=controls,
                cluster=cluster,
                alpha=alpha,
            )
        rows.append(
            {
                "cohort": int(g),
                "att": float(r.estimate),
                "se": float(r.se),
                "pvalue": float(r.pvalue) if r.pvalue is not None else np.nan,
                "n_obs": int(r.n_obs),
                "n_treated_obs": (
                    int(r.detail["n_treated_obs"].sum())
                    if (
                        isinstance(r.detail, pd.DataFrame)
                        and "n_treated_obs" in r.detail.columns
                    )
                    else int(r.n_obs)
                ),
            }
        )
        ses.append(float(r.se))

    detail = pd.DataFrame(rows)
    sizes = detail["n_obs"].values.astype(float)
    w = sizes / sizes.sum() if sizes.sum() > 0 else np.ones(len(rows)) / len(rows)
    att_overall = float(w @ detail["att"].values)
    # SE of a weighted sum of independent estimates (conservative — assumes
    # independent per-cohort regressions, which is exactly what we ran).
    att_se = float(np.sqrt(np.sum((w * np.array(ses)) ** 2)))
    t_stat = att_overall / att_se if att_se > 0 else np.nan
    df_resid = max(int(detail["n_obs"].sum()) - len(cohorts), 1)
    p_overall = float(2 * stats.t.sf(abs(t_stat), df_resid))
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    ci = (att_overall - t_crit * att_se, att_overall + t_crit * att_se)

    return CausalResult(
        method="Wooldridge (2021) ETWFE — never-treated control",
        estimand="Cohort-size-weighted ATT",
        estimate=att_overall,
        se=att_se,
        pvalue=p_overall,
        ci=ci,
        alpha=alpha,
        n_obs=int(detail["n_obs"].sum()),
        detail=detail,
        model_info={
            "n_cohorts": len(cohorts),
            "cohorts": [int(g) for g in cohorts],
            "cgroup": "nevertreated",
            "controls": controls or [],
            "xvar": list(xvar) if xvar else None,
            "cohort_weighting": "cohort",
            "cohort_vcov": np.diag(np.array(ses, dtype=float) ** 2),
        },
        _citation_key="wooldridge_twfe",
    )


# ═══════════════════════════════════════════════════════════════════════
#  2. Doubly Robust DID — Sant'Anna & Zhao (2020)
# ═══════════════════════════════════════════════════════════════════════


@accepts_aliases(_strict=True, unit="id", treat="group", controls="covariates")
def drdid(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    covariates: Optional[List[str]] = None,
    method: str = "imp",
    alpha: float = 0.05,
    n_boot: int = 500,
    random_state: Optional[int] = None,
    seed: Optional[int] = None,
    id: Optional[str] = None,
    *,
    est_method: str = "dr",
    normalized: bool = True,
    locally_efficient: bool = True,
    weights: Optional[str] = None,
    trim_level: float = 0.995,
) -> CausalResult:
    """
    Doubly Robust Difference-in-Differences (Sant'Anna & Zhao 2020).

    Combines outcome regression with inverse probability weighting for
    2×2 DID with covariates.  Consistent if *either* the outcome model
    *or* the propensity score model is correctly specified.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with one row per unit-period in 2x2 design.
    y : str
        Outcome variable.
    group : str
        Binary treatment-group indicator (1 = treated, 0 = control).
    time : str
        Binary time indicator (1 = post, 0 = pre).
    covariates : list of str, optional
        Covariate names.  If ``None``, runs a simple (un-adjusted) DID.
    method : str, default ``'imp'``
        Which nuisance estimators the DR variants use, following R
        ``DRDID::drdid(estMethod=)``:

        - ``'imp'`` — "improved": inverse probability tilting for the
          propensity score and propensity-odds-weighted least squares for
          the outcome model, so the DR moment is Neyman-orthogonal by
          construction and the influence function carries no
          nuisance-estimation terms.
        - ``'trad'`` — plain logit + OLS, with the estimation effects
          propagated explicitly.

        Only consulted when ``est_method='dr'``.

        .. versionchanged:: 1.23.0
           On repeated cross-sections this argument used to do nothing:
           both settings returned ``DRDID::drdid_rc1`` regardless. It now
           selects ``drdid_imp_rc`` / ``drdid_rc`` as documented.
    est_method : {'dr', 'ipw', 'reg', 'twfe'}, default ``'dr'``
        Which estimator family to use. With ``method`` and the two flags
        below this reaches all 14 estimators of R ``DRDID`` 1.2.3:

        ============  =======================  ==========================
        est_method    panel (``id=`` given)    repeated cross-sections
        ============  =======================  ==========================
        ``'dr'``      ``drdid_[imp_]panel``    ``drdid_[imp_]rc[1]``
        ``'ipw'``     ``[std_]ipw_did_panel``  ``[std_]ipw_did_rc``
        ``'reg'``     ``reg_did_panel``        ``reg_did_rc``
        ``'twfe'``    ``twfe_did_panel``       ``twfe_did_rc``
        ============  =======================  ==========================

        ``'twfe'`` is included for comparison, not as a recommendation:
        with covariates it is exactly the specification Sant'Anna & Zhao
        (2020) and Caetano & Callaway (2024) warn about.

        .. versionadded:: 1.23.0
    normalized : bool, default True
        Only for ``est_method='ipw'``. ``True`` gives the Hájek-normalised
        estimator (``std_ipw_did_*``), where the control arm is divided by
        its own realised weight mass. ``False`` gives Abadie (2005)
        (``ipw_did_*``), where both arms share the denominator ``E[D]``.

        .. versionadded:: 1.23.0
    locally_efficient : bool, default True
        Only for ``est_method='dr'`` on repeated cross-sections. ``False``
        drops the terms that attain the semiparametric efficiency bound,
        giving ``drdid_rc1`` / ``drdid_imp_rc1``. Both are consistent;
        the ``rc1`` variants avoid fitting outcome regressions on the
        treated cells.

        .. versionadded:: 1.23.0
    weights : str, optional
        Column of observation weights (R ``DRDID``'s ``i.weights``),
        renormalised to mean one.

        .. versionadded:: 1.23.0
    trim_level : float, default 0.995
        Drop control units whose estimated propensity score reaches this
        cutoff, matching ``DRDID``'s ``trim.level``. Pass ``1.0`` to
        disable.

        .. versionadded:: 1.23.0
    alpha : float, default 0.05
        Significance level.
    n_boot : int, default 500
        Number of bootstrap replications for inference.
    random_state : int, optional
        Seed for bootstrap reproducibility.
    id : str, optional
        Unit identifier for a true two-period panel. When supplied, the
        improved estimator uses the Sant'Anna-Zhao panel formula with
        calibrated propensity scores and influence-function standard errors,
        matching ``DRDID::drdid_imp_panel`` and Stata ``drdid, drimp``.

    Returns
    -------
    CausalResult
        ``estimate`` is the DR-DID ATT.
        ``detail`` contains influence-function diagnostics.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> G = rng.integers(0, 2, n)
    >>> T = rng.integers(0, 2, n)
    >>> x = rng.normal(0, 1, n)
    >>> y_val = 1 + 0.5*x + 2*G + 3*T + 4*G*T + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({'y': y_val, 'treated': G, 'post': T, 'x': x})
    >>> result = sp.drdid(df, y='y', group='treated', time='post',
    ...                   covariates=['x'])
    >>> abs(result.estimate - 4.0) < 1.0
    True
    """
    df = data.copy()

    # R-style covariate formula (DRDID::drdid(xformla = ~ x1 + I(x1**2))).
    if isinstance(covariates, str) and "~" in covariates:
        from ._core import covariates_from_formula as _covariates_from_formula

        df, covariates = _covariates_from_formula(df, covariates, function="drdid")
        covariates = covariates or None
        data = df

    # ── Validate method ─────────────────────────────────────────────
    # Only the improved (locally-efficient) and traditional DR-DID
    # estimators are implemented. Previously any other string silently
    # fell through to the traditional branch (a §7 violation); fail loud.
    if method not in ("imp", "trad"):
        raise MethodIncompatibility(
            f"method must be 'imp' (improved, locally efficient) or 'trad' "
            f"(traditional DR-DID); got {method!r}."
        )
    if est_method not in ("dr", "ipw", "reg", "twfe"):
        raise MethodIncompatibility(
            f"est_method must be 'dr', 'ipw', 'reg' or 'twfe'; got " f"{est_method!r}."
        )
    if not (0.0 < float(trim_level) <= 1.0):
        raise MethodIncompatibility(
            f"trim_level must be in (0, 1]; got {trim_level!r}."
        )
    # Say so when a flag cannot bite, rather than letting the caller
    # believe it took effect (§7).
    if est_method != "ipw" and not normalized:
        warnings.warn(
            f"drdid: normalized=False only affects est_method='ipw' "
            f"(it selects the Abadie 2005 estimator over the Hajek-"
            f"normalised one), but est_method={est_method!r} was "
            "requested, so it has no effect.",
            UserWarning,
            stacklevel=2,
        )
    if est_method != "dr" and not locally_efficient:
        warnings.warn(
            f"drdid: locally_efficient=False only affects est_method='dr', "
            f"but est_method={est_method!r} was requested, so it has no "
            "effect.",
            UserWarning,
            stacklevel=2,
        )

    # ── Observation weights (DRDID's i.weights) ─────────────────────
    if weights is not None:
        if weights not in df.columns:
            raise MethodIncompatibility(
                f"weights column {weights!r} is not in data.",
                recovery_hint="Pass an existing column name, or weights=None.",
                diagnostics={"weights": weights},
            )
        if (df[weights] < 0).any():
            raise MethodIncompatibility(
                "weights must be non-negative.",
                recovery_hint="Repair or drop the negative weights.",
                diagnostics={"weights": weights},
            )

    # ── Validate 2×2 design ─────────────────────────────────────────
    g_vals = sorted(df[group].dropna().unique())
    t_vals = sorted(df[time].dropna().unique())
    if len(g_vals) != 2:
        raise MethodIncompatibility(f"'{group}' must be binary, got values: {g_vals}")
    if len(t_vals) != 2:
        raise MethodIncompatibility(f"'{time}' must be binary, got values: {t_vals}")

    if id is not None:
        if id not in df.columns:
            raise MethodIncompatibility(f"'{id}' must be a column in data")
        covariates_list = covariates or []
        needed = [id, y, group, time] + covariates_list
        if weights is not None:
            needed.append(weights)
        missing = [col for col in needed if col not in df.columns]
        if missing:
            raise MethodIncompatibility(f"Missing columns for panel DR-DID: {missing}")
        panel_df = df[needed].dropna().copy()
        pre_df = panel_df[panel_df[time] == t_vals[0]][[id, y, group] + covariates_list]
        post_df = panel_df[panel_df[time] == t_vals[1]][[id, y, group]]
        if pre_df[id].duplicated().any() or post_df[id].duplicated().any():
            raise MethodIncompatibility(
                "id/time must identify at most one row per unit-period"
            )

        wide = pre_df.merge(post_df, on=id, suffixes=("_pre", "_post"))
        if wide.empty:
            raise DataInsufficient("No complete pre/post unit pairs for panel DR-DID")
        if not np.all(
            wide[f"{group}_pre"].to_numpy() == wide[f"{group}_post"].to_numpy()
        ):
            raise MethodIncompatibility(f"'{group}' must be time-invariant within id")

        D_panel = (wide[f"{group}_pre"] == g_vals[1]).astype(float).to_numpy()
        y0 = wide[f"{y}_pre"].astype(float).to_numpy()
        y1 = wide[f"{y}_post"].astype(float).to_numpy()
        delta_y = y1 - y0
        if covariates_list:
            X_panel = wide[covariates_list].astype(float).to_numpy()
            X_panel = np.column_stack([np.ones(len(X_panel)), X_panel])
        else:
            X_panel = np.ones((len(wide), 1))

        w_panel = (
            wide[weights].astype(float).to_numpy()
            if weights is not None and weights in wide.columns
            else np.ones(len(wide), dtype=float)
        )
        w_panel = w_panel / w_panel.mean()

        z_crit_panel = stats.norm.ppf(1 - alpha / 2)
        if est_method == "dr" and method == "imp":
            att_hat, att_se, ci, ps_fit, ps_flag = _drdid_imp_panel_core(
                delta_y,
                D_panel,
                X_panel,
                alpha=alpha,
                trim_level=trim_level,
            )
            engine_name = "drdid_imp_panel"
        else:
            # The remaining panel estimators are exactly the 2x2 engines
            # `callaway_santanna` runs per (g, t) cell, verified against
            # DRDID 1.2.3 to machine precision. Reuse them rather than
            # keeping a second implementation of the same algebra.
            from .callaway_santanna import _dr_att, _ipw_abadie_att, _ipw_att, _reg_att

            x_panel = X_panel[:, 1:] if X_panel.shape[1] > 1 else None
            if est_method == "dr":
                att_hat, att_se, _inf = _dr_att(
                    delta_y, D_panel, x_panel, trim_level, w_panel
                )
                engine_name = "drdid_panel"
            elif est_method == "ipw":
                fn = _ipw_att if normalized else _ipw_abadie_att
                att_hat, att_se, _inf = fn(
                    delta_y, D_panel, x_panel, trim_level, w_panel
                )
                engine_name = "std_ipw_did_panel" if normalized else "ipw_did_panel"
            elif est_method == "reg":
                att_hat, att_se, _inf = _reg_att(delta_y, D_panel, x_panel, w_panel)
                engine_name = "reg_did_panel"
            else:  # twfe
                from ._rcs import twfe_did_rc as _twfe

                # DRDID's twfe_did_panel stacks the two periods and runs
                # the same regression, so the RC engine on the long form
                # is the identical estimator (verified: both give
                # 1.696871155 / 0.11277800 on the reference fixture).
                _res = _twfe(
                    np.concatenate([y0, y1]),
                    np.concatenate([np.zeros(len(y0)), np.ones(len(y1))]),
                    np.concatenate([D_panel, D_panel]),
                    (
                        np.concatenate([x_panel, x_panel])
                        if x_panel is not None
                        else None
                    ),
                    weights=np.concatenate([w_panel, w_panel]),
                )
                att_hat, att_se = float(_res.att), float(_res.se)
                engine_name = "twfe_did_panel"
            ci = (att_hat - z_crit_panel * att_se, att_hat + z_crit_panel * att_se)
            ps_fit = _logistic_fit(X_panel, D_panel)
            ps_flag = 0

        t_stat = att_hat / att_se if att_se > 0 else np.nan
        pvalue = float(2 * stats.norm.sf(abs(t_stat)))
        detail = pd.DataFrame(
            {
                "statistic": [
                    "ATT",
                    "SE (influence function)",
                    "z-stat",
                    "p-value",
                    "CI lower",
                    "CI upper",
                    "N units",
                ],
                "value": [att_hat, att_se, t_stat, pvalue, ci[0], ci[1], len(wide)],
            }
        )
        panel_model_info: Dict[str, Any] = {
            "method": "improved" if method == "imp" else "traditional",
            "est_method": est_method,
            "engine": engine_name,
            "drdid_reference": f"DRDID::{engine_name}",
            "panel": True,
            "id": id,
            "n_units": int(len(wide)),
            "n_treated": int(D_panel.sum()),
            "n_control": int((1 - D_panel).sum()),
            "n_post": int((panel_df[time] == t_vals[1]).sum()),
            "n_pre": int((panel_df[time] == t_vals[0]).sum()),
            "ps_mean_treated": float(ps_fit[D_panel == 1].mean()),
            "ps_mean_control": float(ps_fit[D_panel == 0].mean()),
            "ps_flag": int(ps_flag),
            "se_method": "influence_function",
            "n_boot": n_boot,
            "covariates": covariates_list,
        }

        _result = CausalResult(
            method=(f"DID panel via DRDID::{engine_name} " "(Sant'Anna & Zhao 2020)"),
            estimand="ATT",
            estimate=att_hat,
            se=att_se,
            pvalue=pvalue,
            ci=ci,
            alpha=alpha,
            n_obs=int(len(panel_df)),
            detail=detail,
            model_info=panel_model_info,
            _citation_key="drdid",
        )
        try:
            from ..output._lineage import attach_provenance as _attach_prov

            _attach_prov(
                _result,
                function="sp.did.drdid",
                params={
                    "y": y,
                    "group": group,
                    "time": time,
                    "id": id,
                    "covariates": covariates,
                    "method": method,
                    "alpha": alpha,
                    "n_boot": n_boot,
                    "random_state": random_state,
                    "seed": seed,
                },
                data=data,
                overwrite=False,
            )
        except Exception:  # pragma: no cover
            pass
        return _result

    G = (df[group] == g_vals[1]).astype(float).values
    T = (df[time] == t_vals[1]).astype(float).values
    Y = df[y].astype(float).values
    w_vec = (
        df[weights].astype(float).to_numpy()
        if weights is not None
        else np.ones(len(df), dtype=float)
    )

    # Covariates
    if covariates and len(covariates) > 0:
        X = df[covariates].values.astype(float)
        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])
    else:
        X = np.ones((len(Y), 1))

    # Drop NaN rows
    valid = np.isfinite(Y) & np.isfinite(w_vec)
    for j in range(X.shape[1]):
        valid &= np.isfinite(X[:, j])
    G, T, Y, X, w_vec = G[valid], T[valid], Y[valid], X[valid], w_vec[valid]
    n = len(Y)

    def _estimate_att(
        G_b: np.ndarray,
        T_b: np.ndarray,
        Y_b: np.ndarray,
        X_b: np.ndarray,
    ) -> float:
        """Core DR-DID estimator for one sample."""
        # Share treated
        p_hat = G_b.mean()
        if p_hat <= 0 or p_hat >= 1:
            return np.nan

        # ── Propensity score: P(G=1 | X) via logistic regression ────
        # Use IRLS for logistic regression (no sklearn dependency)
        ps = _logistic_fit(X_b, G_b)
        ps = np.clip(ps, 1e-6, 1 - 1e-6)

        # ── Outcome regression for controls: E[DeltaY | X, G=0] ────
        # Compute DeltaY for each unit that appears in both periods
        # In repeated cross-section / 2×2 stacked data, compute change
        # We treat the data as pooled; for controls in post vs pre:
        ctrl_post = (G_b == 0) & (T_b == 1)
        ctrl_pre = (G_b == 0) & (T_b == 0)

        # For the outcome model, regress Y on X separately for
        # control-post and control-pre
        if ctrl_post.sum() < X_b.shape[1] or ctrl_pre.sum() < X_b.shape[1]:
            # Not enough data; fall back to simple DID
            return float(
                (
                    Y_b[(G_b == 1) & (T_b == 1)].mean()
                    - Y_b[(G_b == 1) & (T_b == 0)].mean()
                    - Y_b[(G_b == 0) & (T_b == 1)].mean()
                    + Y_b[(G_b == 0) & (T_b == 0)].mean()
                )
            )

        # OLS for E[Y|X, G=0, T=1]
        try:
            beta_post = np.linalg.lstsq(X_b[ctrl_post], Y_b[ctrl_post], rcond=None)[0]
        except np.linalg.LinAlgError:
            beta_post = np.linalg.pinv(X_b[ctrl_post]) @ Y_b[ctrl_post]

        # OLS for E[Y|X, G=0, T=0]
        try:
            beta_pre = np.linalg.lstsq(X_b[ctrl_pre], Y_b[ctrl_pre], rcond=None)[0]
        except np.linalg.LinAlgError:
            beta_pre = np.linalg.pinv(X_b[ctrl_pre]) @ Y_b[ctrl_pre]

        m1_x = X_b @ beta_post  # predicted E[Y|X, G=0, T=1]
        m0_x = X_b @ beta_pre  # predicted E[Y|X, G=0, T=0]
        # ── DR-DID estimator ────────────────────────────────────────
        if method == "imp":
            # Improved (locally efficient) DR-DID
            # Weight construction
            w_treat_post = G_b * T_b
            w_treat_pre = G_b * (1 - T_b)
            w_ctrl_post = ps / (1 - ps) * (1 - G_b) * T_b
            w_ctrl_pre = ps / (1 - ps) * (1 - G_b) * (1 - T_b)

            # Normalise weights
            eta_1 = w_treat_post.mean()
            eta_0 = w_treat_pre.mean()
            if eta_1 == 0 or eta_0 == 0:
                return np.nan

            att = (
                (w_treat_post * (Y_b - m1_x)).sum() / (w_treat_post.sum() + 1e-10)
                - (w_treat_pre * (Y_b - m0_x)).sum() / (w_treat_pre.sum() + 1e-10)
                - (w_ctrl_post * (Y_b - m1_x)).sum() / (w_ctrl_post.sum() + 1e-10)
                + (w_ctrl_pre * (Y_b - m0_x)).sum() / (w_ctrl_pre.sum() + 1e-10)
            )
        else:
            # Traditional DR-DID (Sant'Anna & Zhao 2020), repeated-cross-
            # section form. Each of the four cell terms is a *weighted
            # average* of the outcome-regression residual over the units
            # selected by its weight, so it must be normalised by that
            # weight's total mass — NOT by the full sample size ``n_b``.
            # ⚠️ correctness fix (2026-06-05): the previous code divided
            # every term by ``n_b``, which multiplied each term by the
            # cell's sample share (~0.25 per cell on a balanced 2×2) and
            # so biased the ATT toward zero by roughly 50%. method='imp'
            # was unaffected (it already normalised by the weight mass).
            w1 = G_b / p_hat
            w0 = ps * (1 - G_b) / ((1 - ps) * p_hat)

            w_tp = w1 * T_b  # treated, post
            w_t0 = w1 * (1 - T_b)  # treated, pre
            w_cp = w0 * T_b  # control, post (ps-reweighted)
            w_c0 = w0 * (1 - T_b)  # control, pre  (ps-reweighted)

            att_1 = (w_tp * (Y_b - m1_x)).sum() / (w_tp.sum() + 1e-10)
            att_0 = (w_t0 * (Y_b - m0_x)).sum() / (w_t0.sum() + 1e-10)
            ctrl_1 = (w_cp * (Y_b - m1_x)).sum() / (w_cp.sum() + 1e-10)
            ctrl_0 = (w_c0 * (Y_b - m0_x)).sum() / (w_c0.sum() + 1e-10)

            att = (att_1 - att_0) - (ctrl_1 - ctrl_0)

        return float(att)

    # ── Estimation ──────────────────────────────────────────────────
    #
    # ⚠️ correctness fix. This branch used to run a bespoke DR estimator
    # with a nonparametric bootstrap standard error, and two things were
    # wrong with it:
    #
    #   1. ``method='imp'`` and ``method='trad'`` computed *the same
    #      number* — the argument was silently inert here (they agreed to
    #      3e-13, i.e. floating-point reassociation). Whatever the caller
    #      asked for, they got ``DRDID::drdid_rc1``, labelled "improved"
    #      or "traditional" according to an argument that did nothing.
    #   2. The SE came from resampling rather than from the Sant'Anna-Zhao
    #      influence function, so it was random, ``n_boot``-dependent, and
    #      off the reference by 1.5-5.5%.
    #
    # Both are now routed to the same verified engines the repeated
    # cross-section path in ``callaway_santanna`` uses, which reproduce
    # ``DRDID`` 1.2.3 to machine precision. The R wrapper's own mapping is
    # followed exactly: ``estMethod='imp' -> drdid_imp_rc``,
    # ``'trad' -> drdid_rc``.
    from ._rcs import drdid_imp_rc as _drdid_imp_rc
    from ._rcs import drdid_rc as _drdid_rc
    from ._rcs import ipw_did_rc as _ipw_did_rc
    from ._rcs import reg_did_rc as _reg_did_rc
    from ._rcs import std_ipw_did_rc as _std_ipw_did_rc
    from ._rcs import twfe_did_rc as _twfe_did_rc

    x_rc = X[:, 1:] if X.shape[1] > 1 else None
    if est_method == "dr":
        engine = _drdid_imp_rc if method == "imp" else _drdid_rc
        res = engine(
            Y,
            T,
            G,
            x_rc,
            weights=w_vec,
            trim_level=trim_level,
            locally_efficient=locally_efficient,
        )
        _eff = "" if locally_efficient else "1"
        engine_name = f"drdid_imp_rc{_eff}" if method == "imp" else f"drdid_rc{_eff}"
    elif est_method == "ipw":
        engine = _std_ipw_did_rc if normalized else _ipw_did_rc
        res = engine(Y, T, G, x_rc, weights=w_vec, trim_level=trim_level)
        engine_name = "std_ipw_did_rc" if normalized else "ipw_did_rc"
    elif est_method == "reg":
        res = _reg_did_rc(Y, T, G, x_rc, weights=w_vec)
        engine_name = "reg_did_rc"
    else:  # "twfe"
        res = _twfe_did_rc(Y, T, G, x_rc, weights=w_vec)
        engine_name = "twfe_did_rc"

    att_hat, att_se = float(res.att), float(res.se)
    influence = res.influence

    t_stat = att_hat / att_se if att_se > 0 else np.nan
    pvalue = float(2 * stats.norm.sf(abs(t_stat)))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (att_hat - z_crit * att_se, att_hat + z_crit * att_se)

    # ── Detail DataFrame ────────────────────────────────────────────
    detail = pd.DataFrame(
        {
            "statistic": [
                "ATT",
                "SE (influence function)",
                "z-stat",
                "p-value",
                "CI lower",
                "CI upper",
                "N obs",
            ],
            "value": [att_hat, att_se, t_stat, pvalue, ci[0], ci[1], n],
        }
    )

    # ── Diagnostics ─────────────────────────────────────────────────
    ps_full = _logistic_fit(X, G)
    n_treated = int(G.sum())
    n_control = int((1 - G).sum())

    rcs_model_info: Dict[str, Any] = {
        "method": "improved" if method == "imp" else "traditional",
        "est_method": est_method,
        "engine": engine_name,
        "drdid_reference": f"DRDID::{engine_name}",
        "panel": False,
        "se_method": "influence_function",
        "n_treated": n_treated,
        "n_control": n_control,
        "n_post": int(T.sum()),
        "n_pre": int((1 - T).sum()),
        "ps_mean_treated": float(ps_full[G == 1].mean()),
        "ps_mean_control": float(ps_full[G == 0].mean()),
        "covariates": covariates or [],
    }

    _result = CausalResult(
        method=(
            f"DID repeated cross-sections via DRDID::{engine_name} "
            "(Sant'Anna & Zhao 2020)"
        ),
        estimand="ATT",
        estimate=att_hat,
        se=att_se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=rcs_model_info,
        _influence_funcs=influence[:, None],
        _citation_key="drdid",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.drdid",
            params={
                "y": y,
                "group": group,
                "time": time,
                "covariates": covariates,
                "method": method,
                "alpha": alpha,
                "n_boot": n_boot,
                "random_state": random_state,
                "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _logistic_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
) -> np.ndarray:
    """Fit logistic regression via IRLS, return coefficients."""
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        z = X @ beta
        z = np.clip(z, -20, 20)
        mu = 1.0 / (1.0 + np.exp(-z))
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        w = mu * (1 - mu)
        Xw = X * w[:, np.newaxis]
        try:
            H = np.linalg.inv(Xw.T @ X)
        except np.linalg.LinAlgError:
            H = np.linalg.pinv(Xw.T @ X)
        grad = X.T @ (y - mu)
        delta = H @ grad
        beta += delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    return beta


def _logistic_fit(X: np.ndarray, y: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Fit logistic regression via IRLS, return predicted probabilities."""
    beta = _logistic_coefficients(X, y, max_iter=max_iter)
    z = X @ beta
    z = np.clip(z, -20, 20)
    return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)


def _weighted_lstsq(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted least-squares coefficients using the same objective as R lm."""
    weights = np.asarray(weights, dtype=float)
    sw = np.sqrt(np.clip(weights, 0.0, np.inf))
    try:
        return np.asarray(
            np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0],
            dtype=float,
        )
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.pinv(X * sw[:, None]) @ (y * sw), dtype=float)


def _calibrated_pscore(
    X: np.ndarray,
    D: np.ndarray,
    i_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, int]:
    """DRDID::pscore.cal translation for the improved panel estimator."""
    n = len(D)
    iw: np.ndarray
    if i_weights is None:
        iw = np.ones(n, dtype=float)
    else:
        iw = np.asarray(i_weights, dtype=float)
        if np.any(iw < 0):
            raise MethodIncompatibility("i_weights must be non-negative")
        iw = iw / iw.mean()

    init = _logistic_coefficients(X, D, max_iter=100)

    def _eta(gamma: np.ndarray) -> np.ndarray:
        return np.asarray(np.clip(X @ gamma, -700, 700), dtype=float)

    def objective(gamma: np.ndarray) -> float:
        eta = _eta(gamma)
        return float(np.mean(iw * ((1.0 - D) * np.exp(eta) - D * eta)))

    def gradient(gamma: np.ndarray) -> np.ndarray:
        eta = _eta(gamma)
        return np.asarray((iw * ((1.0 - D) * np.exp(eta) - D)) @ X / n, dtype=float)

    opt = optimize.minimize(
        objective,
        init,
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    gamma = opt.x if opt.success else init
    flag = 0 if opt.success else 2

    # Newton refinement. The tilting loss is convex with a closed-form
    # Hessian, and R's `pscore.cal` solves it with a trust-region Newton
    # method; quasi-Newton stops a few orders of magnitude short of that.
    #
    # The extra precision is not cosmetic. Inverse probability tilting
    # earns its name by making the covariate-balance conditions hold
    # *exactly* at the optimum, which is what kills the first-order
    # nuisance terms in the improved DR estimators. A γ that is merely
    # close leaves those terms alive: a 1.5e-9 discrepancy in p̂(X) moved
    # `drdid_imp_rc` by 7e-4 in the ATT and 2.9% in the SE — five orders
    # of magnitude of amplification, because the odds weights p/(1-p)
    # multiply the imbalance back in.
    def _hessian(g: np.ndarray) -> np.ndarray:
        eta = _eta(g)
        h_w = iw * (1.0 - D) * np.exp(eta)
        return np.asarray((X.T * h_w) @ X / n, dtype=float)

    for _ in range(50):
        grad = gradient(gamma)
        if not np.all(np.isfinite(grad)) or np.max(np.abs(grad)) < 1e-13:
            break
        hess = _hessian(gamma)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        candidate = gamma - step
        # Convex problem, but guard the step anyway: a bad Hessian near a
        # boundary should not move us to a worse point.
        if objective(candidate) > objective(gamma):
            break
        gamma = candidate

    ps = 1.0 / (1.0 + np.exp(-np.clip(X @ gamma, -700, 700)))
    return np.minimum(ps, 1.0 - 1e-6), flag


def _drdid_imp_panel_core(
    delta_y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    *,
    alpha: float = 0.05,
    trim_level: float = 0.995,
) -> tuple[float, float, tuple[float, float], np.ndarray, int]:
    """Sant'Anna-Zhao improved panel DR-DID core matching DRDID::drdid_imp_panel."""
    n = len(D)
    if n == 0:
        raise DataInsufficient("panel DR-DID requires at least one complete unit")
    p_treat = D.mean()
    if p_treat <= 0 or p_treat >= 1:
        raise DataInsufficient("panel DR-DID requires treated and control units")

    ps, ps_flag = _calibrated_pscore(X, D)
    trim_ps = np.ones(n, dtype=float)
    trim_ps[D == 0] = (ps[D == 0] < trim_level).astype(float)

    control = D == 0
    if control.sum() < X.shape[1]:
        raise DataInsufficient(
            "Not enough control units for panel DR-DID outcome regression"
        )
    odds = ps / (1.0 - ps)
    beta = _weighted_lstsq(X[control], delta_y[control], odds[control])
    out_delta = X @ beta

    summand = trim_ps * (1.0 - (1.0 - D) / (1.0 - ps)) * (delta_y - out_delta)
    att = float(np.mean(summand) / p_treat)
    inf_func = trim_ps * (summand - D * att) / p_treat
    se = float(np.std(inf_func, ddof=1) * np.sqrt(n - 1) / n)
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci = (att - z_crit * se, att + z_crit * se)
    return att, se, (float(ci[0]), float(ci[1])), ps, ps_flag


# ═══════════════════════════════════════════════════════════════════════
#  3. Enhanced TWFE Decomposition (Bacon + dCDH weights)
# ═══════════════════════════════════════════════════════════════════════


def twfe_decomposition(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    alpha: float = 0.05,
) -> CausalResult:
    """
    TWFE decomposition: Goodman-Bacon (2021) + de Chaisemartin–D'Haultfoeuille weights.

    The coefficient of the two-way fixed effects regression of ``y`` on a
    treatment indicator, taken apart two ways. ``detail`` holds the
    Goodman-Bacon decomposition: every 2×2 difference in differences
    between timing groups with its weight, which sum to the coefficient
    exactly (the same rows as :func:`statspai.bacon_decomposition`).
    ``model_info['dcdh_weights']`` holds the de Chaisemartin–D'Haultfoeuille
    (2020) weights the coefficient puts on the treated cells, some of which
    can be negative.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset in long format.
    y : str
        Outcome variable.
    group : str
        Unit identifier.
    time : str
        Time period variable.
    first_treat : str
        Treatment timing column (NaN or 0 for never-treated).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        ``estimate`` is the TWFE coefficient and ``se`` its standard error
        clustered by unit (the small-sample factor of Stata ``xtreg, fe
        vce(cluster)``). ``detail`` has columns ``type``,
        ``treated_cohort``, ``control_cohort``, ``estimate``, ``weight``,
        ``weighted_est``. ``model_info`` has ``twfe_beta``, ``bacon_att``
        (the weighted sum of the 2×2 estimates), ``dcdh_weights`` (one row
        per cohort and treated period: the sum of the unit-level weights,
        the number of units and how many of their weights are negative),
        ``n_negative_weights_dcdh`` and ``sum_negative_weights_dcdh``
        (counted over unit × period cells, as R ``twowayfeweights``
        does).

    Notes
    -----
    The Goodman-Bacon theorem needs a balanced panel. On an unbalanced one
    the coefficient and the dCDH weights are still reported, ``detail`` is
    empty and a warning says so.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=8, staggered=True, seed=0)
    >>> result = sp.twfe_decomposition(df, y='y', group='unit',
    ...                                time='time',
    ...                                first_treat='first_treat')
    >>> bool('weight' in result.detail.columns)  # 2x2 decomposition weights
    True
    >>> bool(abs(result.detail['weighted_est'].sum() - result.estimate) < 1e-8)
    True
    """
    from ._twfe_weights import _ZERO_BELOW, _two_way_residual
    from .bacon import bacon_decomposition

    missing = [c for c in (y, group, time, first_treat) if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"twfe_decomposition: columns not found in data: {missing}",
            diagnostics={"missing": missing},
        )
    df = data[[y, group, time, first_treat]].copy()
    df["_ft"] = df[first_treat].replace(0, np.nan)
    df = df.dropna(subset=[y, group, time]).reset_index(drop=True)

    periods = sorted(df[time].unique())
    n_periods = len(periods)
    cohorts = sorted(df.loc[df["_ft"].notna(), "_ft"].unique())
    has_never = bool(df["_ft"].isna().any())
    if not cohorts:
        raise DataInsufficient("No valid 2×2 comparisons found. Check data structure.")

    df["_treated"] = ((df["_ft"].notna()) & (df[time] >= df["_ft"])).astype(float)
    df["_y"] = df[y].astype(float)
    n = len(df)
    gi = pd.factorize(df[group], sort=True)[0]
    ti = pd.factorize(df[time], sort=True)[0]
    n_units = int(gi.max() + 1)
    counts = np.bincount(gi * n_periods + ti, minlength=n_units * n_periods)
    balanced = bool(np.all(counts == 1))
    if not balanced and counts.max() > 1:
        raise MethodIncompatibility(
            "twfe_decomposition: more than one row for a unit in a period.",
            recovery_hint="Aggregate to one row per unit and period first.",
        )

    def _residual(v: np.ndarray) -> np.ndarray:
        if balanced:
            # exact on a balanced panel: v - unit mean - period mean + mean
            unit_mean = np.bincount(gi, weights=v) / n_periods
            time_mean = np.bincount(ti, weights=v) / n_units
            return np.asarray(v - unit_mean[gi] - time_mean[ti] + v.mean())
        if n_units + n_periods > 5000:
            raise MethodIncompatibility(
                "twfe_decomposition: the panel is unbalanced and too large "
                "for the exact two-way projection used here.",
                recovery_hint="Balance the panel (keep the units observed in "
                "every period) and run it again.",
            )
        return _two_way_residual(v, gi, ti, np.ones(n))

    D = df["_treated"].to_numpy(dtype=float)
    eps = _residual(D)
    y_res = _residual(df["_y"].to_numpy(dtype=float))
    denom = float(eps @ eps)
    if not denom > 0:
        raise DataInsufficient(
            "twfe_decomposition: treatment does not vary within units and "
            "periods, so the TWFE coefficient is not identified."
        )
    twfe_beta = float(eps @ y_res / denom)

    # ── standard error of the coefficient, clustered by unit ────────
    resid = y_res - twfe_beta * eps
    score = np.bincount(gi, weights=eps * resid, minlength=n_units)
    k_slopes = n_periods + 1  # treatment, period dummies, constant
    if n_units > 1 and n > k_slopes:
        factor = (n_units / (n_units - 1)) * ((n - 1) / (n - k_slopes))
        att_se = float(np.sqrt(factor * float(score @ score)) / denom)
    else:
        att_se = float("nan")

    # ── Goodman-Bacon decomposition ─────────────────────────────────
    detail_cols = [
        "type", "treated_cohort", "control_cohort", "estimate", "weight",
        "weighted_est",
    ]  # fmt: skip
    comp_df = pd.DataFrame(columns=detail_cols)
    bacon_att = float("nan")
    if balanced:
        bacon = bacon_decomposition(
            df, y="_y", treat="_treated", time=time, id=group, alpha=alpha
        )
        rows = bacon["decomposition"]
        labels = {
            "Earlier vs Later Treated": "Earlier vs Later",
            "Later vs Earlier Treated": "Later vs Earlier",
            "Treated vs Untreated": "Treated vs Never",
        }

        def _cohort(value: Any) -> Any:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return value
            return int(number) if number == int(number) else number

        comp_df = pd.DataFrame(
            {
                "type": [labels.get(t, t) for t in rows["type"]],
                "treated_cohort": [_cohort(v) for v in rows["treated"]],
                "control_cohort": [_cohort(v) for v in rows["control"]],
                "estimate": rows["estimate"].to_numpy(dtype=float),
                "weight": rows["weight"].to_numpy(dtype=float),
            }
        )
        comp_df["weighted_est"] = comp_df["weight"] * comp_df["estimate"]
        bacon_att = float(comp_df["weighted_est"].sum())
    else:
        warnings.warn(
            "twfe_decomposition: the panel is unbalanced, so the "
            "Goodman-Bacon decomposition (which holds for a balanced panel) "
            "is not computed; the TWFE coefficient and the dCDH weights are.",
            UserWarning,
            stacklevel=2,
        )

    # ── de Chaisemartin–D'Haultfoeuille weights ─────────────────────
    # beta_TWFE = sum over treated cells of w_gt * ATT_gt with
    # w_gt = D_gt eps_gt / sum(D eps); here a cell is a unit in a period.
    weight = D * eps / float(D @ eps)
    weight = np.where(np.abs(weight) < _ZERO_BELOW, 0.0, weight)
    treated_rows = D > 0
    cells = pd.DataFrame(
        {
            "cohort": df.loc[treated_rows, "_ft"].to_numpy(),
            "period": df.loc[treated_rows, time].to_numpy(),
            "w": weight[treated_rows],
        }
    )
    dcdh_df = (
        cells.groupby(["cohort", "period"], sort=True)["w"]
        .agg(dcdh_weight="sum", n_cell="size", n_negative=lambda v: int((v < 0).sum()))
        .reset_index()
    )
    dcdh_df["cohort"] = [int(c) if float(c) == int(c) else c for c in dcdh_df["cohort"]]
    n_negative_dcdh = int((cells["w"] < 0).sum())

    model_info: Dict[str, Any] = {
        "twfe_beta": twfe_beta,
        "bacon_att": bacon_att,
        "balanced": balanced,
        "n_comparisons": len(comp_df),
        "n_negative_weights_bacon": int((comp_df["weight"] < -1e-10).sum()),
        "n_negative_weights_dcdh": n_negative_dcdh,
        "sum_negative_weights_dcdh": float(cells.loc[cells["w"] < 0, "w"].sum()),
        "n_cohorts": len(cohorts),
        "cohorts": [int(g) if float(g) == int(g) else g for g in cohorts],
        "has_never_treated": has_never,
        "n_units": n_units,
        "n_periods": n_periods,
        "se_type": "cluster (unit)",
        "dcdh_weights": dcdh_df,
    }

    usable = np.isfinite(att_se) and att_se > 0
    pvalue = float(2 * stats.norm.sf(abs(twfe_beta / att_se))) if usable else np.nan
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (
        (twfe_beta - z_crit * att_se, twfe_beta + z_crit * att_se)
        if usable
        else (np.nan, np.nan)
    )

    return CausalResult(
        method="TWFE Decomposition (Bacon 2021 + dCDH 2020)",
        estimand="ATT (TWFE composite)",
        estimate=twfe_beta,
        se=att_se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=comp_df,
        model_info=model_info,
        _citation_key="twfe_decomposition",
    )


# ═══════════════════════════════════════════════════════════════════════
#  4. etwfe_emfx — R etwfe-style marginal-effects aggregations
# ═══════════════════════════════════════════════════════════════════════


def _etwfe_glm_emfx(
    result: CausalResult,
    type: str,
    alpha: float,
    scale: Optional[str] = None,
    include_leads: bool = False,
    by_xvar: bool = False,
) -> CausalResult:
    """Nonlinear-fit aggregations; implementation in :mod:`._etwfe_nonlinear`."""
    return _etwfe_glm_emfx_impl(
        result,
        type=type,
        alpha=alpha,
        scale=scale,
        include_leads=include_leads,
        by_xvar=by_xvar,
    )


def etwfe_emfx(
    result: CausalResult,
    type: str = "simple",
    alpha: float = 0.05,
    include_leads: bool = False,
    weighting: str = "treated",
    agg_weights: Optional[str] = None,
    scale: Optional[str] = None,
    by_xvar: bool = False,
) -> CausalResult:
    """
    R ``etwfe::emfx``-style aggregated marginal effects for an ETWFE fit.

    Takes the result of :func:`etwfe` / :func:`wooldridge_did` and returns
    one of four aggregations used in applied work:

    ================  ========================================================
    ``type``          Aggregation
    ================  ========================================================
    ``'simple'``      Overall treated-observation-weighted ATT (same as
                      ``result.estimate`` for current ``sp.etwfe`` results).
    ``'group'``       ATT per treatment cohort ``g``: the cohort's post-
                      treatment cells averaged over its treated observations.
    ``'event'``       ATT per event time ``e = t - g``: the cells at that event
                      time averaged over their treated observations.
    ``'calendar'``    ATT per calendar time ``t``: the cells of every cohort
                      with ``g <= t`` averaged over their treated observations.
    ================  ========================================================

    Parameters
    ----------
    result : CausalResult
        Output of :func:`etwfe` or :func:`wooldridge_did`.
    type : {'simple', 'group', 'event', 'calendar'}, default 'simple'
        Aggregation type.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    include_leads : bool, default False
        For ``type='event'`` and ``type='calendar'``, whether to include
        pre-treatment relative times (``rel_time < 0``) in the output.
        These coefficients identify pre-trends and are informative for
        parallel-trends inspection. Default ``False`` for backward
        compatibility with earlier versions; set ``True`` for full
        event-study output matching the R ``etwfe::emfx(type='event')``
        default. ``rel_time = -1`` is always the reference category
        and is excluded.
    weighting : {'cohort', 'treated'}, default 'treated'
        Aggregation weights for cohort-level marginal effects. ``'treated'``
        uses the number of treated post-period observations, matching R
        ``etwfe::emfx(type='simple')`` and Stata ``jwdid, estat simple``.
        ``'cohort'`` preserves the historical StatsPAI cohort-share weighting.
    agg_weights : {'estimation', 'unit'}, optional
        For a fit estimated with ``weights=``: whether each cohort-by-period
        cell enters an aggregate with the sum of the estimation weights over
        its treated observations (``'estimation'``, Stata ``jwdid, estat``)
        or with its observation count (``'unit'``, R ``etwfe::emfx``).
        ``None`` (default) reuses the rule the fit was called with
        (``result.model_info['agg_weights']``).  Ignored for unweighted
        fits, where the two rules coincide.
    scale : {'response', 'link'}, optional
        Nonlinear (``family='poisson'``/``'logit'``) fits only: aggregate
        the average marginal effects (``'response'``, R ``emfx`` default)
        or the cohort x period coefficients themselves (``'link'``, Stata
        ``estat ..., predict(xb)`` / R ``emfx(predict='link')``).
        ``None`` (default) keeps the scale the fit was reported on.  The
        linear model has one scale and accepts either value.
    by_xvar : bool, default False
        Nonlinear fits with one categorical ``xvar``: report the
        aggregation per level of the covariate (R ``emfx(by_xvar =
        TRUE)``, Stata ``estat ..., over()``); ``detail`` gains a
        ``level`` column.

    Returns
    -------
    CausalResult
        ``estimate`` is the overall ATT (for ``type='simple'``) or the
        mean of the sub-aggregation (for the other types). ``detail``
        contains one row per group/event-time/calendar-time with
        (estimate, se, pvalue, ci_low, ci_high).

    Notes
    -----
    Every aggregation is R ``emfx``'s: each treated observation carries
    weight ``N = 1`` and its marginal effect is the coefficient of its
    cohort-by-period cell, so a level of the aggregation averages the cell
    coefficients with the cells' own observation counts.  Standard errors
    are the delta method through the cluster-robust cell covariance in
    ``model_info['event_vcov']``.  ``weighting`` only governs how cohorts
    enter the simple headline; the ``'group'`` / ``'event'`` /
    ``'calendar'`` rows are defined by the cells alone.

    For a weighted fit the cell totals depend on ``agg_weights``.  Writing
    ``W_{g,t}`` for the weight of cell ``(g, t)`` in every aggregate,

    * ``'estimation'`` (Stata ``jwdid, estat``, which runs ``margins`` on
      the ``[pw=]`` estimation sample): ``W_{g,t} = sum_{i in g} w_{i,t}``;
    * ``'unit'`` (R ``emfx``, which evaluates ``marginaleffects::slopes``
      with ``wts = 1`` per treated row): ``W_{g,t} = n_{g,t}``.

    Under never-treated controls without covariates the ``'estimation'``
    ``simple`` aggregate equals the weighted Callaway--Sant'Anna simple
    ATT.  With weights constant within unit on a balanced panel the two
    rules agree on ``'group'`` (a cohort's post cells all carry the same
    weight) and differ on ``'simple'`` / ``'event'`` / ``'calendar'``.

    The linear ``sp.etwfe`` fit uses R's default design (cohort and period
    fixed effects, ``ivar = NULL``) and fixest's default small-sample factor
    ``(n - 1) / (n - K) * G / (G - 1)`` with ``K`` counting every column of
    that design.  Stata ``jwdid`` (``reghdfe`` with unit and period fixed
    effects) reports the same estimates and a covariance that differs only
    through ``K``: it drops the unit effects as nested in the cluster and
    keeps the period effects plus a constant, so
    ``se_Stata = se * sqrt((n - K) / (n - K_Stata))`` with
    ``K - K_Stata`` equal to the number of treated cohorts (the cohort-effect
    parameters fixest counts and reghdfe does not; ``model_info['ssc']``
    records ``n``, ``K`` and ``G``).

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=10, staggered=True, seed=0)
    >>> fit = sp.etwfe(df, y='y', time='time',
    ...                first_treat='first_treat', group='unit')
    >>> evt = sp.etwfe_emfx(fit, type='event')
    >>> evt.detail["event_time"].tolist()   # ATT by event time
    [0, 1, 2, 3, 4, 5]
    >>> grp = sp.etwfe_emfx(fit, type='group')
    >>> list(grp.detail.columns[:3])
    ['cohort', 'estimate', 'se']
    >>> cal = sp.etwfe_emfx(fit, type='calendar')
    """
    valid = {"simple", "group", "event", "calendar"}
    if type not in valid:
        raise MethodIncompatibility(
            f"type must be one of {sorted(valid)}; got {type!r}"
        )
    valid_weighting = {"cohort", "treated", "treated_observations"}
    if weighting not in valid_weighting:
        raise MethodIncompatibility(
            "weighting must be one of " f"{sorted(valid_weighting)}; got {weighting!r}"
        )
    weighting = "treated" if weighting == "treated_observations" else weighting

    if (
        isinstance(result.model_info, dict)
        and result.model_info.get("estimator") == "etwfe_glm"
    ):
        return _etwfe_glm_emfx(
            result,
            type=type,
            alpha=alpha,
            scale=scale,
            include_leads=include_leads,
            by_xvar=by_xvar,
        )
    if by_xvar:
        raise MethodIncompatibility(
            "etwfe_emfx(by_xvar=True) is implemented for nonlinear "
            "(family='poisson' / 'logit') fits with a categorical xvar.",
            recovery_hint="Refit with family='poisson' and a categorical "
            "xvar, or read the moderator slopes from model_info for the "
            "linear fit.",
            diagnostics={"by_xvar": by_xvar},
        )
    if scale is not None:
        _normalise_etwfe_scale(scale)

    if not isinstance(result.model_info, dict) or "cohorts" not in result.model_info:
        raise MethodIncompatibility(
            "etwfe_emfx requires a result produced by sp.etwfe / "
            "sp.wooldridge_did — missing 'cohorts' in model_info."
        )

    mi = result.model_info
    cohorts = mi["cohorts"]
    event_study_raw = mi.get("event_study")
    event_study = event_study_raw if isinstance(event_study_raw, pd.DataFrame) else None
    if agg_weights is None:
        agg_weights = mi.get("agg_weights") or "estimation"
    _validate_agg_weights(agg_weights)
    # Estimation-weight totals only differ from the counts on a weighted
    # fit; unweighted / legacy results keep their count-based columns.
    use_estimation_totals = agg_weights == "estimation" and (
        mi.get("weights") is not None
    )

    def _detail_frame() -> pd.DataFrame:
        if not isinstance(result.detail, pd.DataFrame):
            raise MethodIncompatibility(
                "etwfe_emfx requires a result with cohort-level detail."
            )
        return result.detail.copy()

    def _weighted_headline(
        use_event_cells: bool = False,
    ) -> Tuple[float, float, float, Tuple[float, float], Dict[str, Any]]:
        if use_event_cells and weighting == "treated" and event_study is not None:
            es = event_study.copy()
            es = es.loc[es["rel_time"] >= 0].copy()
            cell_col = (
                "w_treated_obs"
                if use_estimation_totals and "w_treated_obs" in es.columns
                else "n_treated_obs"
            )
            if len(es) > 0 and cell_col in es.columns:
                w_raw = es[cell_col].astype(float).to_numpy()
                w = (
                    w_raw / float(w_raw.sum())
                    if float(w_raw.sum()) > 0
                    else np.full(len(es), 1.0 / len(es))
                )
                est_vec = es["estimate"].astype(float).to_numpy()
                est = float(w @ est_vec)
                event_vcov = mi.get("event_vcov")
                has_vcov = event_vcov is not None and "_vcov_idx" in es.columns
                if has_vcov:
                    idx = es["_vcov_idx"].astype(int).values - 1
                    V_sub = np.asarray(event_vcov, dtype=float)[np.ix_(idx, idx)]
                    se = float(np.sqrt(max(w @ V_sub @ w, 0.0)))
                    se_method = "event-cell vcov-based (delta method)"
                else:
                    se_vec = es["se"].astype(float).to_numpy()
                    se = float(np.sqrt(np.sum((w * se_vec) ** 2)))
                    se_method = "event-cell independent-coefficient approximation"
                df_resid = max(result.n_obs - len(cohorts), 1)
                t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
                t_stat = est / se if se > 0 else np.nan
                p = (
                    float(2 * stats.t.sf(abs(t_stat), df_resid))
                    if not np.isnan(t_stat)
                    else np.nan
                )
                ci = (
                    (est - t_crit * se, est + t_crit * se)
                    if np.isfinite(se)
                    else (np.nan, np.nan)
                )
                info = {
                    "weighting": weighting,
                    "weight_column": f"event_{cell_col}",
                    "aggregation_unit": "cohort_time",
                    "agg_weights": agg_weights,
                    "se_method": se_method,
                }
                return est, se, p, ci, info

        det = _detail_frame()
        if "att_at_xmean" in det.columns:
            est_col = "att_at_xmean"
        elif "att" in det.columns:
            est_col = "att"
        else:
            raise MethodIncompatibility(
                "ETWFE detail must contain 'att' or 'att_at_xmean'."
            )

        weight_col = "n_obs" if weighting == "cohort" else "n_treated_obs"
        if use_estimation_totals and f"w_{weight_col[2:]}" in det.columns:
            weight_col = f"w_{weight_col[2:]}"
        if weight_col not in det.columns:
            raise MethodIncompatibility(
                f"weighting={weighting!r} requires '{weight_col}' in result.detail; "
                "refit with a current StatsPAI ETWFE result."
            )

        w_raw = det[weight_col].astype(float).to_numpy()
        if not np.isfinite(w_raw).all() or float(w_raw.sum()) <= 0:
            w = np.full(len(det), 1.0 / max(len(det), 1))
        else:
            w = w_raw / float(w_raw.sum())
        est_vec = det[est_col].astype(float).to_numpy()
        est = float(w @ est_vec)

        vcov = mi.get("cohort_vcov")
        V = np.asarray(vcov, dtype=float) if vcov is not None else None
        if V is not None and V.shape == (len(w), len(w)):
            se = float(np.sqrt(max(w @ V @ w, 0.0)))
            se_method = "vcov-based (delta method)"
        else:
            if "se" in det.columns:
                se_vec = det["se"].astype(float).to_numpy()
            elif "att_se" in det.columns:
                se_vec = det["att_se"].astype(float).to_numpy()
            else:
                se_vec = np.full(len(w), np.nan)
            se = float(np.sqrt(np.sum((w * se_vec) ** 2)))
            se_method = "independent-coefficient approximation (fallback)"

        df_resid = max(result.n_obs - len(cohorts), 1)
        t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
        t_stat = est / se if se > 0 else np.nan
        p = (
            float(2 * stats.t.sf(abs(t_stat), df_resid))
            if not np.isnan(t_stat)
            else np.nan
        )
        ci = (
            (est - t_crit * se, est + t_crit * se)
            if np.isfinite(se)
            else (np.nan, np.nan)
        )
        weight_map = (
            {
                int(c): float(w_i)
                for c, w_i in zip(det["cohort"].astype(int).tolist(), w.tolist())
            }
            if "cohort" in det.columns
            else {}
        )
        detail_info: Dict[str, Any] = {
            "weighting": weighting,
            "weight_column": weight_col,
            "weights": weight_map,
            "agg_weights": agg_weights,
            "se_method": se_method,
        }
        return est, se, p, ci, detail_info

    # ── simple ──
    if type == "simple":
        est, se, p, ci, weight_info = _weighted_headline(use_event_cells=True)
        detail = pd.DataFrame(
            [
                {
                    "aggregation": "simple",
                    "estimate": est,
                    "se": se,
                    "pvalue": p,
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "n_cohorts": len(cohorts),
                    "weighting": weighting,
                }
            ]
        )
        return CausalResult(
            method="ETWFE — simple aggregation (overall ATT)",
            estimand="Overall ATT",
            estimate=est,
            se=se,
            pvalue=p,
            ci=ci,
            alpha=alpha,
            n_obs=int(result.n_obs),
            detail=detail,
            model_info={
                "type": "simple",
                "source_method": result.method,
                **weight_info,
            },
            _citation_key="wooldridge_twfe",
        )

    # ── group / event / calendar: averages of cohort-by-period cells ──
    # R emfx: every treated observation carries weight N = 1 and its marginal
    # effect is its cohort-by-period cell coefficient, so each level of the
    # `by` variable is the cell coefficients averaged with the cells' own
    # observation counts.  `weighting` governs only how cohorts enter the
    # simple headline; the per-level rows are defined by the cells alone.
    df_resid = max(result.n_obs - len(cohorts), 1)
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    has_cells = event_study is not None and len(event_study) > 0

    if type == "group" and not has_cells:
        # Covariate-moderated fits (xvar) carry no cells: report the pooled
        # cohort coefficients evaluated at the covariate means.
        det = _detail_frame()
        if "att_at_xmean" in det.columns:
            est_col = "att_at_xmean"
            se_col = "att_se"
            p_col = "att_pvalue"
        else:
            est_col = "att"
            se_col = "se"
            p_col = "pvalue"
        rows = []
        for _, r in det.iterrows():
            est = float(r[est_col])
            se = float(r[se_col])
            rows.append(
                {
                    "cohort": int(r["cohort"]),
                    "estimate": est,
                    "se": se,
                    "pvalue": float(r[p_col]) if p_col in r.index else np.nan,
                    "ci_low": est - t_crit * se,
                    "ci_high": est + t_crit * se,
                    "n_obs": int(r["n_obs"]),
                }
            )
        out_det = pd.DataFrame(rows)
        headline_est, headline_se, p_head, ci_head, weight_info = _weighted_headline(
            use_event_cells=True
        )
        return CausalResult(
            method="ETWFE — group aggregation (ATT per cohort)",
            estimand="ATT(g) per cohort",
            estimate=headline_est,
            se=headline_se,
            pvalue=p_head,
            ci=ci_head,
            alpha=alpha,
            n_obs=int(result.n_obs),
            detail=out_det,
            model_info={
                "type": "group",
                "source_method": result.method,
                "rows_source": "cohort_detail",
                **weight_info,
            },
            _citation_key="wooldridge_twfe",
        )

    if not has_cells:
        raise DataInsufficient(
            "type='event'/'calendar' requires event_study coefficients "
            "in result.model_info['event_study']."
        )
    assert event_study is not None
    es = event_study.copy()
    # Cell totals: `w_cell_obs` (estimation-weight sums) on a weighted fit
    # under the Stata rule; otherwise the observation counts -- `n_cell_obs`
    # covers leads too, older results only carry the post-period count.
    if use_estimation_totals and "w_cell_obs" in es.columns:
        weight_col = "w_cell_obs"
    else:
        weight_col = "n_cell_obs" if "n_cell_obs" in es.columns else "n_treated_obs"

    # H7: default post-only; pre-treatment leads are available for
    # type='event' via include_leads=True (pre-trend inspection).
    if type != "event" or not include_leads:
        es = es.loc[es["rel_time"] >= 0].copy()

    if type == "group":
        key_col = "cohort"
        label_col = "cohort"
    elif type == "event":
        key_col = "rel_time"
        label_col = "event_time"
    else:
        es["calendar_time"] = es["cohort"].astype(int) + es["rel_time"].astype(int)
        key_col = "calendar_time"
        label_col = "calendar_time"

    # H1: use the stored cell vcov (delta method) when available.
    event_vcov_raw = mi.get("event_vcov")
    event_vcov = (
        np.asarray(event_vcov_raw, dtype=float) if event_vcov_raw is not None else None
    )
    agg, _agg_vcov, vcov_based = _aggregate_cells(
        es, event_vcov, key_col=key_col, weight_col=weight_col
    )
    se_method = (
        "vcov-based (delta method)"
        if vcov_based
        else "independent-coefficient approximation (fallback — vcov unavailable)"
    )

    cohort_n_obs: Dict[int, int] = {}
    if (
        type == "group"
        and isinstance(result.detail, pd.DataFrame)
        and {"cohort", "n_obs"} <= set(result.detail.columns)
    ):
        cohort_n_obs = dict(
            zip(result.detail["cohort"].astype(int), result.detail["n_obs"].astype(int))
        )

    rows = []
    for _, r in agg.iterrows():
        est = float(r["estimate"])
        se = float(r["se"])
        t_stat = est / se if se > 0 else np.nan
        p = (
            float(2 * stats.t.sf(abs(t_stat), df_resid))
            if not np.isnan(t_stat)
            else np.nan
        )
        row: Dict[str, Any] = {
            label_col: int(r[key_col]),
            "estimate": est,
            "se": se,
            "pvalue": p,
            "ci_low": est - t_crit * se,
            "ci_high": est + t_crit * se,
        }
        if type == "group":
            row["n_obs"] = cohort_n_obs.get(int(r[key_col]), int(r["n_obs"]))
            row["n_treated_obs"] = int(r["n_obs"])
        else:
            row["n_cohorts_used"] = int(r["n_cells"])
        rows.append(row)
    out_det = pd.DataFrame(rows).sort_values(label_col).reset_index(drop=True)

    if type == "group":
        # The group view's headline is the simple overall ATT under the
        # caller-selected cohort weighting (H2).
        headline_est, headline_se, p_head, ci_head, weight_info = _weighted_headline(
            use_event_cells=True
        )
        return CausalResult(
            method="ETWFE — group aggregation (ATT per cohort)",
            estimand="ATT(g) per cohort",
            estimate=headline_est,
            se=headline_se,
            pvalue=p_head,
            ci=ci_head,
            alpha=alpha,
            n_obs=int(result.n_obs),
            detail=out_det,
            model_info={
                "type": "group",
                "source_method": result.method,
                "rows_source": "event_cells",
                "rows_se_method": se_method,
                "cell_weight_column": weight_col,
                **weight_info,
                "agg_weights": agg_weights,
            },
            _citation_key="wooldridge_twfe",
        )

    # Joint covariance of the reported rows (aligned with ``out_det``), so
    # event-time output can feed uniform bands and HonestDiD.
    agg_labels = [int(v) for v in agg[key_col]]
    rows_vcov = None
    if _agg_vcov is not None:
        rows_vcov = pd.DataFrame(
            np.asarray(_agg_vcov, dtype=float), index=agg_labels, columns=agg_labels
        ).loc[out_det[label_col].tolist(), out_det[label_col].tolist()]

    # Headline: the unweighted mean of the reported rows. Its SE is the
    # delta method through ``rows_vcov`` (1' V 1 / k^2); without the joint
    # covariance it is not identified and is reported as NaN with a warning.
    head_rows = out_det
    if type == "event" and include_leads:
        head_rows = out_det.loc[out_det[label_col] >= 0]
    mean_est = float(head_rows["estimate"].mean())
    if rows_vcov is not None and len(head_rows):
        lab = head_rows[label_col].tolist()
        wv = np.full(len(lab), 1.0 / len(lab))
        mean_se = float(np.sqrt(max(wv @ rows_vcov.loc[lab, lab].to_numpy() @ wv, 0.0)))
    else:
        mean_se = np.nan
        warnings.warn(
            f"etwfe_emfx(type={type!r}): no cell covariance on the fit, so the "
            "SE of the headline (mean of the reported rows) is not available.",
            UserWarning,
            stacklevel=2,
        )
    if np.isfinite(mean_se) and mean_se > 0:
        t_head = mean_est / mean_se
        p_head_m = float(2 * stats.t.sf(abs(t_head), df_resid))
        ci_head_m = (mean_est - t_crit * mean_se, mean_est + t_crit * mean_se)
    else:
        p_head_m, ci_head_m = np.nan, (np.nan, np.nan)
    event_study_frame = None
    if type == "event":
        event_study_frame = pd.DataFrame(
            {
                "relative_time": out_det["event_time"].astype(int),
                "att": out_det["estimate"],
                "se": out_det["se"],
                "ci_lower": out_det["ci_low"],
                "ci_upper": out_det["ci_high"],
                "pvalue": out_det["pvalue"],
            }
        )
    return CausalResult(
        method=f"ETWFE — {type} aggregation",
        estimand=f"ATT by {label_col.replace('_', ' ')}",
        estimate=mean_est,
        se=mean_se,
        pvalue=p_head_m,
        ci=ci_head_m,
        alpha=alpha,
        n_obs=int(result.n_obs),
        detail=out_det,
        model_info={
            "type": type,
            "source_method": result.method,
            "se_method": se_method,
            "weighting": weighting,
            "weight_column": weight_col,
            "cell_weight_column": weight_col,
            "agg_weights": agg_weights,
            "vcov": rows_vcov,
            "event_study": event_study_frame,
            "event_study_vcov": rows_vcov if type == "event" else None,
            "headline": (
                "unweighted mean of the post-treatment event-time rows"
                if type == "event"
                else "unweighted mean of the calendar-time rows"
            ),
        },
        _citation_key="wooldridge_twfe",
    )
