"""Nonlinear ETWFE (Wooldridge 2023): Poisson / logit staggered DiD.

Backs ``sp.etwfe(family='poisson' | 'logit')``.  The saturated design is

.. math::

    h(E[Y_{it}]) = c_i + \\lambda_t
                   + \\sum_{g}\\sum_{t \\in \\mathcal{T}_g}
                     \\beta_{gt} 1\\{G_i = g, T = t\\}

where ``c_i`` is either a cohort effect (``fe='cohort'``, the Mundlak form
R ``etwfe`` fits) or a unit effect (``fe='unit'``, Stata ``jwdid ...,
method(ppmlhdfe)`` which absorbs ``ivar``), and :math:`\\mathcal{T}_g` is
``t >= g`` for not-yet-treated comparisons or every ``t != g - 1`` for
never-treated comparisons (the pre-period cells then carry the event-study
leads).

Two scales are reported for every aggregation (simple / group / event /
calendar):

* ``'response'`` -- the average marginal effect over the treated
  observations of a cell set, :math:`\\bar{h^{-1}(\\hat\\eta)} -
  \\bar{h^{-1}(\\hat\\eta - \\hat\\beta_{gt})}`; R ``etwfe::emfx`` and
  Stata ``jwdid, estat`` default.  A Poisson fit gives counts.
* ``'link'`` -- the treated-observation-weighted average of the cell
  coefficients :math:`\\sum N_{gt}\\hat\\beta_{gt} / \\sum N_{gt}` (log points
  for Poisson, log-odds for logit); Stata ``estat simple, predict(xb)`` and
  R ``emfx(..., predict = "link")``.

Every aggregate is a function of per-cell sums (count, sum of marginal
effects, sum of their gradients), so the fit is summarised once per cell and
each aggregation is a small weighted sum; standard errors are the delta
method through the cluster-robust coefficient covariance.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility

#: Non-identity families supported by ``sp.etwfe(family=...)``, mapped to
#: their ``statsmodels`` family constructor.  ``None``/``'gaussian'`` keeps
#: the historical linear OLS path untouched.
_ETWFE_GLM_FAMILIES = {
    "poisson": "Poisson",
    "logit": "Binomial",
    "binomial": "Binomial",
}

_SCALES = ("response", "link")


def normalise_scale(scale: Optional[str]) -> str:
    """Validate ``scale=`` ('response' | 'link'; 'xb'/'eta' alias 'link')."""
    if scale is None:
        return "response"
    key = str(scale).strip().lower()
    if key in {"xb", "eta", "linear_predictor", "log"}:
        key = "link"
    if key in {"mu", "ame", "count", "probability"}:
        key = "response"
    if key not in _SCALES:
        raise MethodIncompatibility(
            f"scale={scale!r} is not recognised; use 'response' or 'link'.",
            recovery_hint="scale='link' reports the treated-observation-"
            "weighted average of the cohort x period coefficients (Stata "
            "estat simple, predict(xb)); 'response' the average marginal "
            "effect.",
            diagnostics={"scale": scale},
        )
    return key


def normalise_glm_fe(fe: Optional[str], fam_key: str) -> str:
    """Validate ``fe=`` for the nonlinear branch ('cohort' | 'unit')."""
    key = "cohort" if fe is None else str(fe).strip().lower()
    if key in {"ivar", "id", "individual"}:
        key = "unit"
    if key in {"gvar", "group", "mundlak"}:
        key = "cohort"
    if key not in {"cohort", "unit"}:
        raise MethodIncompatibility(
            f"fe={fe!r} is not recognised; use 'cohort' or 'unit'.",
            recovery_hint="fe='cohort' is R etwfe's design; fe='unit' "
            "absorbs unit fixed effects like Stata jwdid ..., "
            "method(ppmlhdfe).",
            diagnostics={"fe": fe},
        )
    if key == "unit" and fam_key != "poisson":
        raise MethodIncompatibility(
            f"fe='unit' is only available for family='poisson'; a {fam_key} "
            "model with unit effects suffers the incidental-parameters bias.",
            recovery_hint="Use fe='cohort' (Wooldridge's pooled Mundlak form) "
            "for binary outcomes.",
            diagnostics={"family": fam_key, "fe": fe},
        )
    return key


def _normalise_glm_cgroup(cgroup: str) -> str:
    key = str(cgroup).strip().lower()
    if key in {"notyet", "notyettreated", "not_yet"}:
        return "notyet"
    if key in {"never", "nevertreated", "never_treated"}:
        return "never"
    raise MethodIncompatibility(
        f"cgroup={cgroup!r} is not recognised; use 'notyet' or 'nevertreated'.",
        recovery_hint="cgroup='notyet' (default) or cgroup='nevertreated'.",
        diagnostics={"cgroup": cgroup},
    )


def _cluster_sandwich(
    scores: np.ndarray,
    bread_inv: np.ndarray,
    cl_codes: np.ndarray,
    n_clusters: int,
    factor: float,
) -> np.ndarray:
    """``factor * A^{-1} (sum_g s_g s_g') A^{-1}`` with per-cluster sums."""
    G = int(cl_codes.max()) + 1 if cl_codes.size else 0
    S = np.column_stack(
        [
            np.bincount(cl_codes, weights=scores[:, j], minlength=G)
            for j in range(scores.shape[1])
        ]
    )
    meat = S.T @ S
    return factor * (bread_inv @ meat @ bread_inv)


def _fit_poisson_unit_fe(
    y: np.ndarray,
    X: np.ndarray,
    unit_codes: np.ndarray,
    cl_codes: np.ndarray,
    n_clusters_full: int,
    n_full: int,
) -> Dict[str, Any]:
    """PPML with the unit effect absorbed; ``fe='unit'``.

    Separated rows (units whose outcome is zero in every period, and rows a
    single regressor predicts to be zero) are removed before IRLS, as in
    Stata ``ppmlhdfe``; they carry no information about the slopes and their
    fitted mean is exactly zero.  They stay in ``N`` and in the cluster count
    that enters the ``G/(G-1)`` factor, which is how ``jwdid`` reports its
    ``ppmlhdfe`` fits (the replication-package tables print the full-sample
    ``N``).
    """
    from ..regression.count import _ppml_hdfe_irls, _ppml_separation_mask

    keep, sep_counts = _ppml_separation_mask(y, X, [unit_codes])
    if not keep.any():
        raise DataInsufficient(
            "etwfe(family='poisson', fe='unit'): every unit has an all-zero "
            "outcome; nothing to estimate.",
            recovery_hint="Check the outcome column.",
            diagnostics={"n_separated": int(len(y))},
        )
    Xk = X[keep]
    live = np.flatnonzero(np.any(Xk != 0, axis=0))
    beta_live, mu, converged, n_iter, X_dm = _ppml_hdfe_irls(
        y[keep],
        Xk[:, live],
        fe_indices_list=[unit_codes[keep]],
        maxiter=1000,
        tol=1e-10,
    )
    if not converged:
        warnings.warn(
            "etwfe(family='poisson', fe='unit'): PPML did not converge; "
            "estimates may be unreliable.",
            stacklevel=3,
        )
    yk = y[keep]
    bread = X_dm.T @ (mu[:, None] * X_dm)
    try:
        bread_inv = np.linalg.inv(bread)
    except np.linalg.LinAlgError as exc:
        raise ConvergenceFailure(
            "etwfe(family='poisson', fe='unit'): singular information matrix "
            "(collinear cohort x period cells or controls).",
            recovery_hint="Drop collinear controls or check the cohort " "timing.",
            diagnostics={"n_obs": int(keep.sum())},
        ) from exc
    k = len(live)
    G = int(n_clusters_full)
    factor = (G / (G - 1.0) if G > 1 else 1.0) * ((n_full - 1.0) / max(n_full - k, 1.0))
    vcov = _cluster_sandwich(
        X_dm * (yk - mu)[:, None], bread_inv, cl_codes[keep], G, factor
    )
    return {
        "keep": keep,
        "live": live,
        "beta": beta_live,
        "vcov": vcov,
        "mu": mu,
        "converged": bool(converged),
        "n_iter": int(n_iter),
        "sep_counts": sep_counts,
        "ssc": {"n": int(n_full), "k": int(k), "G": G, "factor": float(factor)},
    }


def etwfe_glm(
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
) -> CausalResult:
    """Nonlinear ETWFE — Wooldridge (2023) staggered DiD with a link function.

    See the module docstring for the design.  ``controls`` enter the linear
    index additively, without cohort / period / treatment interactions --
    the role of Stata ``jwdid``'s ``exovar()``.

    Verified against R ``etwfe`` 0.6.2 (``fe='cohort'``) on a simulated
    count panel: simple AME matches to 1e-10 and every event-time AME to
    1e-6; see ``tests/reference_parity/test_etwfe_glm_parity.py``.  The
    ``fe='unit'`` link-scale coefficients match an explicit
    ``sp.fepois`` / ``sp.ppmlhdfe`` fit of the same cells with the unit
    effect absorbed (``tests/test_etwfe_nonlinear.py``).
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover - statsmodels is a core dep
        raise MethodIncompatibility(
            "etwfe(family=...) requires statsmodels.",
            recovery_hint="pip install statsmodels",
            diagnostics={"family": family},
        ) from exc

    fam_key = str(family).strip().lower()
    if fam_key not in _ETWFE_GLM_FAMILIES:
        raise MethodIncompatibility(
            f"family={family!r} is not supported; use one of "
            f"{sorted(set(_ETWFE_GLM_FAMILIES) | {'gaussian'})}.",
            recovery_hint="Pass family='poisson', 'logit', or 'gaussian'.",
            diagnostics={"family": family},
        )
    scale = normalise_scale(scale)
    fe_mode = normalise_glm_fe(fe, fam_key)
    cg = _normalise_glm_cgroup(cgroup)

    df = data.copy()
    df["_ft"] = df[first_treat].replace(0, np.nan)
    df["_y"] = df[y].astype(float)

    periods = sorted(df[time].unique())
    cohorts = sorted(df.loc[df["_ft"].notna(), "_ft"].unique())
    if not cohorts:
        raise DataInsufficient(
            "No treated cohorts found. Check 'first_treat' column.",
            recovery_hint="Ensure first_treat holds the first treated period "
            "(0 or NaN for never-treated).",
            diagnostics={"first_treat": first_treat},
        )
    if cg == "never" and not df["_ft"].isna().any():
        raise DataInsufficient(
            "cgroup='nevertreated' needs never-treated units (first_treat "
            "0 or NaN); none were found.",
            recovery_hint="Use cgroup='notyet'.",
            diagnostics={"cgroup": cgroup},
        )

    if fam_key in {"logit", "binomial"}:
        bad = df["_y"].notna() & ~df["_y"].isin([0.0, 1.0])
        if bad.any():
            raise MethodIncompatibility(
                f"family={family!r} needs a 0/1 outcome; column {y!r} has "
                f"{int(bad.sum())} value(s) outside {{0, 1}}.",
                recovery_hint="Recode the outcome to 0/1 or use "
                "family='poisson'/'gaussian'.",
                diagnostics={"family": family, "n_invalid": int(bad.sum())},
            )
    elif (df["_y"] < 0).any():
        raise MethodIncompatibility(
            f"family='poisson' needs a non-negative outcome; column {y!r} "
            "contains negative values.",
            recovery_hint="Use family='gaussian' for outcomes that can be "
            "negative (e.g. logged or differenced variables).",
            diagnostics={"family": family},
        )

    # ── Design ──────────────────────────────────────────────────────────
    # fe='cohort': intercept + cohort dummies (never-treated omitted) +
    # period dummies (first omitted) + cells, mirroring R etwfe's
    #   ~ .Dtreat:i(gvar, i.tvar, ref=0, ref2=<first>) + i(gvar, ref=0)
    #     + i(tvar, ref=<first>)
    # fe='unit': period dummies + cells; the unit effect is absorbed.
    cols: List[np.ndarray] = []
    names: List[str] = []
    if fe_mode == "cohort":
        cols.append(np.ones(len(df)))
        names.append("const")
        for g_val in cohorts:
            cols.append((df["_ft"] == g_val).to_numpy(dtype=float))
            names.append(f"cohort[{int(g_val)}]")
    for t_val in periods[1:]:
        cols.append((df[time] == t_val).to_numpy(dtype=float))
        names.append(f"period[{int(t_val)}]")

    interaction_idx: List[int] = []
    interaction_cell: List[Tuple[int, int]] = []
    cell_post: List[bool] = []
    for g_val in cohorts:
        pre = [p for p in periods if p < g_val]
        ref = pre[-1] if pre else None
        for t_val in periods:
            if cg == "notyet":
                if t_val < g_val:
                    continue
            elif t_val == ref:
                continue
            col = ((df["_ft"] == g_val) & (df[time] == t_val)).to_numpy(dtype=float)
            if col.sum() <= 0:
                continue
            cols.append(col)
            names.append(f"treat[{int(g_val)},{int(t_val)}]")
            interaction_idx.append(len(names) - 1)
            interaction_cell.append((int(g_val), int(t_val)))
            cell_post.append(bool(t_val >= g_val))

    if not any(cell_post):
        raise DataInsufficient(
            "No post-treatment cohort x period cells — nothing to estimate.",
            recovery_hint="Check that treated cohorts have observed periods "
            "at or after their first_treat value.",
            diagnostics={"cohorts": [int(c) for c in cohorts]},
        )

    ctrl_names: List[str] = []
    for c in controls or []:
        cols.append(df[c].astype(float).to_numpy())
        names.append(f"control[{c}]")
        ctrl_names.append(c)

    X = np.column_stack(cols)
    y_vec = df["_y"].to_numpy(dtype=float)
    keep_rows = np.isfinite(X).all(axis=1) & np.isfinite(y_vec)
    X, y_vec = X[keep_rows], y_vec[keep_rows]
    df_keep = df.loc[keep_rows].reset_index(drop=True)
    n_obs = int(len(y_vec))

    cluster_col = cluster or group
    cl_codes = pd.factorize(df_keep[cluster_col])[0].astype(np.intp)
    n_clusters = int(cl_codes.max()) + 1 if n_obs else 0

    # A balanced-panel identity (Wooldridge 2023) makes pooled Poisson with
    # cohort dummies reproduce Poisson with unit effects; with missing
    # outcomes it no longer does, and the two designs give different
    # coefficients.  Say so instead of letting the gap pass silently.
    per_unit = df_keep.groupby(group)[time].nunique()
    balanced = bool(
        per_unit.nunique() <= 1 and per_unit.iloc[0] == df_keep[time].nunique()
    )
    if fam_key == "poisson" and fe_mode == "cohort" and not balanced:
        warnings.warn(
            "etwfe(family='poisson'): the estimation sample is an unbalanced "
            "panel, where cohort dummies (fe='cohort', R etwfe's design) and "
            "unit fixed effects (fe='unit', Stata jwdid ..., "
            "method(ppmlhdfe)) no longer give the same coefficients. Pass "
            "fe='unit' to reproduce jwdid / ppmlhdfe.",
            UserWarning,
            stacklevel=3,
        )

    cells_arr = np.asarray(interaction_idx, dtype=int)
    cell_of = np.full(n_obs, -1, dtype=int)
    Xc = X[:, cells_arr]
    in_cell = Xc.sum(axis=1) > 0
    cell_of[in_cell] = np.argmax(Xc[in_cell], axis=1)
    n_cells = len(interaction_cell)
    cell_n = np.bincount(cell_of[in_cell], minlength=n_cells).astype(float)

    sep_info: Dict[str, Any] = {"n_separated": 0}
    omitted: List[str] = []

    if fe_mode == "cohort":
        sm_family = getattr(sm.families, _ETWFE_GLM_FAMILIES[fam_key])()
        model = sm.GLM(y_vec, X, family=sm_family)
        try:
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": cl_codes})
        except Exception as exc:
            raise ConvergenceFailure(
                f"etwfe(family={family!r}) failed to converge: {exc}",
                recovery_hint="Check for separation / collinear cohort-period "
                "cells, or fall back to family='gaussian'.",
                diagnostics={"family": family, "n_obs": n_obs},
            ) from exc
        beta = np.asarray(fit.params, dtype=float)
        vcov = np.asarray(fit.cov_params(), dtype=float)
        converged = bool(getattr(fit, "converged", True))
        link = sm_family.link
        X0 = X.copy()
        X0[:, cells_arr] = 0.0
        mu1 = np.asarray(link.inverse(X @ beta), dtype=float)
        mu0 = np.asarray(link.inverse(X0 @ beta), dtype=float)
        dmu1 = np.asarray(link.inverse_deriv(X @ beta), dtype=float)
        dmu0 = np.asarray(link.inverse_deriv(X0 @ beta), dtype=float)
        rows = np.flatnonzero(in_cell)
        me_rows = (mu1 - mu0)[rows]
        grad_rows = X[rows] * dmu1[rows, None] - X0[rows] * dmu0[rows, None]
        cell_rows = cell_of[rows]
        link_name = type(link).__name__
        coef_names = list(names)
        ssc = None
    else:
        unit_codes = pd.factorize(df_keep[group])[0].astype(np.intp)
        res = _fit_poisson_unit_fe(y_vec, X, unit_codes, cl_codes, n_clusters, n_obs)
        live = res["live"]
        keep = res["keep"]
        omitted = [names[j] for j in range(len(names)) if j not in set(live.tolist())]
        n_sep = int((~keep).sum())
        sep_info = {
            "n_separated": n_sep,
            "n_separated_by_rule": dict(res["sep_counts"]),
        }
        if n_sep:
            warnings.warn(
                f"etwfe(family='poisson', fe='unit'): {n_sep} separated "
                "observation(s) (all-zero units / perfectly predicted zeros) "
                "have a fitted mean of exactly zero and were left out of "
                "IRLS; they stay in N and in the cluster count, as jwdid "
                "reports them.",
                UserWarning,
                stacklevel=3,
            )
        beta = res["beta"]
        vcov = res["vcov"]
        converged = res["converged"]
        coef_names = [names[j] for j in live]
        Xl = X[keep][:, live]
        pos_in_live = {int(j): i for i, j in enumerate(live)}
        live_cells = [pos_in_live.get(int(j), -1) for j in cells_arr]
        cmask = np.zeros(len(live), dtype=bool)
        cmask[[c for c in live_cells if c >= 0]] = True
        mu = res["mu"]
        mu0 = mu * np.exp(-(Xl[:, cmask] @ beta[cmask]))
        # Profile the unit effect: exp(c_i) = sum_t y_it / sum_t exp(x_it b),
        # so mu_it = Y_i * p_it with p_it the within-unit softmax of x_it b
        # and d mu_it / d b = mu_it (x_it - xbar_i), xbar_i = sum_s p_is x_is.
        uk = unit_codes[keep]
        U = int(uk.max()) + 1
        musum = np.bincount(uk, weights=mu, minlength=U)
        inv = np.zeros(U)
        inv[musum > 0] = 1.0 / musum[musum > 0]
        xbar = np.column_stack(
            [
                np.bincount(uk, weights=mu * Xl[:, j], minlength=U) * inv
                for j in range(Xl.shape[1])
            ]
        )[uk]
        cell_k = cell_of[keep]
        rows = np.flatnonzero(cell_k >= 0)
        X0l = Xl[rows].copy()
        X0l[:, cmask] = 0.0
        me_rows = (mu - mu0)[rows]
        grad_rows = mu[rows, None] * (Xl[rows] - xbar[rows]) - mu0[rows, None] * (
            X0l - xbar[rows]
        )
        cell_rows = cell_k[rows]
        link_name = "Log"
        ssc = res["ssc"]
        if omitted:
            dropped_cells = [c for c in omitted if c.startswith("treat[")]
            if dropped_cells:
                warnings.warn(
                    f"etwfe(family='poisson', fe='unit'): cell(s) "
                    f"{dropped_cells} contain only separated observations and "
                    "are omitted from every aggregate.",
                    UserWarning,
                    stacklevel=3,
                )
        # Cells that were omitted drop out of the aggregation weights.
        cell_live = np.array([c >= 0 for c in live_cells])
        cell_n = np.where(cell_live, cell_n, 0.0)
        cells_arr = np.array(live_cells, dtype=int)

        # **CRITICAL FIX**: When fe='unit', cell_n was computed from all observations
        # (including separated ones), but me_sum and grad_sum are computed only from
        # kept (non-separated) observations. This caused the response-scale SE to be
        # underestimated (denominator too large). Recompute cell_n from kept rows only.
        cell_n_kept = np.bincount(cell_k[cell_k >= 0], minlength=n_cells).astype(float)
        # For omitted cells (cell_live=False), set to 0; otherwise use kept count
        cell_n = np.where(cell_live, cell_n_kept, 0.0)

    K = len(beta)
    # Per-cell summaries: the response-scale aggregates only need these.
    me_sum = np.bincount(cell_rows, weights=me_rows, minlength=n_cells)
    grad_sum = (
        np.column_stack(
            [
                np.bincount(cell_rows, weights=grad_rows[:, j], minlength=n_cells)
                for j in range(K)
            ]
        )
        if K
        else np.zeros((n_cells, 0))
    )
    cell_beta = np.array([beta[c] if c >= 0 else np.nan for c in cells_arr])
    cell_se = np.array(
        [np.sqrt(max(vcov[c, c], 0.0)) if c >= 0 else np.nan for c in cells_arr]
    )

    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    def _agg(sel: np.ndarray, sc: str) -> Tuple[float, float, int]:
        sel = sel & (cell_n > 0)
        n_sel = float(cell_n[sel].sum())
        if n_sel <= 0:
            return np.nan, np.nan, 0
        if sc == "response":
            est = float(me_sum[sel].sum() / n_sel)
            grad = grad_sum[sel].sum(axis=0) / n_sel
        else:
            w = cell_n[sel] / n_sel
            est = float(w @ cell_beta[sel])
            grad = np.zeros(K)
            np.add.at(grad, cells_arr[sel], w)
        var = float(grad @ vcov @ grad)
        return est, float(np.sqrt(max(var, 0.0))), int(n_sel)

    g_arr = np.array([c[0] for c in interaction_cell], dtype=float)
    t_arr = np.array([c[1] for c in interaction_cell], dtype=float)
    post_arr = np.array(cell_post, dtype=bool)
    e_arr = t_arr - g_arr

    def _row(label: str, key: Any, est: float, se: float, n: int) -> Dict[str, Any]:
        z = est / se if se and se > 0 else np.nan
        return {
            label: key,
            "att": est,
            "se": se,
            "pvalue": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "ci_lower": est - z_crit * se,
            "ci_upper": est + z_crit * se,
            "n_treated": n,
        }

    aggregations: Dict[str, Dict[str, Any]] = {}
    for sc in _SCALES:
        simple = _agg(post_arr, sc)
        ev_rows = []
        for e_val in sorted(set(e_arr.tolist())):
            est, se, n = _agg(e_arr == e_val, sc)
            if n:
                ev_rows.append(_row("relative_time", int(e_val), est, se, n))
        ev = pd.DataFrame(ev_rows)
        if not ev.empty:
            ev["post"] = ev["relative_time"] >= 0
        grp_rows = []
        for g_val in cohorts:
            est, se, n = _agg(post_arr & (g_arr == g_val), sc)
            if n:
                grp_rows.append(_row("cohort", int(g_val), est, se, n))
        cal_rows = []
        for t_val in sorted(set(t_arr[post_arr].tolist())):
            est, se, n = _agg(post_arr & (t_arr == t_val), sc)
            if n:
                cal_rows.append(_row("period", int(t_val), est, se, n))
        aggregations[sc] = {
            "simple": {"att": simple[0], "se": simple[1], "n_treated": simple[2]},
            "event": ev,
            "group": pd.DataFrame(grp_rows),
            "calendar": pd.DataFrame(cal_rows),
        }

    cells_df = pd.DataFrame(
        {
            "cohort": [c[0] for c in interaction_cell],
            "period": [c[1] for c in interaction_cell],
            "relative_time": e_arr.astype(int),
            "post": post_arr,
            "n_treated": cell_n.astype(int),
            "coef": cell_beta,
            "se": cell_se,
            "ame": np.where(cell_n > 0, me_sum / np.maximum(cell_n, 1), np.nan),
        }
    )

    # Build event-study covariance matrix for use by pretrends_test/honest_did
    # The cell coefficients' covariance is a submatrix of vcov indexed by
    # the live cell positions (for fe='unit' with separated observations).
    event_times = sorted(set(e_arr[cell_n > 0].astype(int).tolist()))
    if event_times and K > 0:
        # Identify which cells correspond to each event time
        event_vcov_map = {}  # event_time -> list of (cell_idx, coef_idx)
        for i, (coh, period) in enumerate(interaction_cell):
            if (
                cell_n[i] > 0 and cells_arr[i] >= 0
            ):  # cell has observations and coef was estimated
                e = int(t_arr[i] - g_arr[i])
                if e not in event_vcov_map:
                    event_vcov_map[e] = []
                event_vcov_map[e].append((i, cells_arr[i]))

        if event_vcov_map:
            # For each event time, aggregate the covariance of its cell coefficients
            # using the same delta-method aggregation as the point estimates
            event_vcov_df = pd.DataFrame(
                index=event_times, columns=event_times, dtype=float
            )
            for e1 in event_times:
                for e2 in event_times:
                    if e1 in event_vcov_map and e2 in event_vcov_map:
                        # Covariance between aggregates at e1 and e2
                        cov = 0.0
                        for i1, c1 in event_vcov_map[e1]:
                            n1 = cell_n[i1]
                            for i2, c2 in event_vcov_map[e2]:
                                n2 = cell_n[i2]
                                if c1 >= 0 and c2 >= 0 and c1 < K and c2 < K:
                                    # Weight by cell sizes and scale to per-treated-obs
                                    w = 1.0 / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
                                    cov += w * vcov[c1, c2]
                        event_vcov_df.loc[e1, e2] = cov

            event_vcov_df = pd.DataFrame(
                event_vcov_df.astype(float).values,
                index=event_times,
                columns=event_times,
            )
            # Mark as block diagonal if pre and post periods are from separate regressions
            has_pre = any(e < 0 for e in event_times)
            has_post = any(e >= 0 for e in event_times)
            if has_pre and has_post and cg == "nevertreated":
                # With never-treated, leads may be from separate auxiliary regression
                # Check if the off-diagonal cross terms are near-zero (separate estimation)
                max_cross = 0.0
                for e1 in event_times:
                    if e1 < 0:
                        for e2 in event_times:
                            if e2 >= 0:
                                max_cross = max(
                                    max_cross, abs(event_vcov_df.loc[e1, e2])
                                )
                if max_cross < 1e-10:
                    event_vcov_df.attrs["block_diagonal"] = True

    head = aggregations[scale]
    att, se_att = head["simple"]["att"], head["simple"]["se"]
    z_stat = att / se_att if se_att > 0 else 0.0
    if scale == "response":
        estimand = "ATT (average marginal effect, response scale)"
    else:
        estimand = (
            "ATT (link scale: treated-observation-weighted mean of the "
            + ("log-point" if fam_key == "poisson" else "log-odds")
            + " cohort x period coefficients)"
        )

    # Keep the historical public tables: event study without leads unless
    # the never-treated design estimated them.
    ev_head = head["event"]
    event_study = ev_head.drop(columns=["ci_lower", "ci_upper"], errors="ignore")
    group_tbl = head["group"]
    calendar_tbl = head["calendar"]

    return CausalResult(
        method=f"Wooldridge (2023) nonlinear ETWFE — family={fam_key}",
        estimand=estimand,
        estimate=att,
        se=se_att,
        pvalue=float(2 * stats.norm.sf(abs(z_stat))),
        ci=(att - z_crit * se_att, att + z_crit * se_att),
        alpha=alpha,
        n_obs=n_obs,
        detail=(
            group_tbl[["cohort", "att", "se", "n_treated"]].copy()
            if not group_tbl.empty
            else group_tbl
        ),
        model_info={
            "estimator": "etwfe_glm",
            "family": fam_key,
            "link": link_name,
            "cgroup": cg,
            "fe": fe_mode,
            "scale": scale,
            "event_study": event_study,
            "event_study_vcov": (
                event_vcov_df
                if "event_vcov_df" in locals()
                and isinstance(event_vcov_df, pd.DataFrame)
                else None
            ),
            "calendar": (
                calendar_tbl[["period", "att", "se", "n_treated"]].copy()
                if not calendar_tbl.empty
                else calendar_tbl
            ),
            "coef_names": coef_names,
            "coefficients": beta,
            "vcov": vcov,
            "interaction_cells": interaction_cell,
            "cells": cells_df,
            "aggregations": aggregations,
            "att_response": aggregations["response"]["simple"]["att"],
            "se_response": aggregations["response"]["simple"]["se"],
            "att_link": aggregations["link"]["simple"]["att"],
            "se_link": aggregations["link"]["simple"]["se"],
            "n_treated_obs": int(cell_n[post_arr].sum()),
            "n_clusters": n_clusters,
            "se_type": f"cluster-robust on {cluster_col}",
            "controls": ctrl_names,
            "balanced_panel": balanced,
            "omitted": omitted,
            "ssc": ssc,
            "converged": converged,
            **sep_info,
        },
        _citation_key="wooldridge2021two",
    )


def etwfe_glm_emfx(
    result: CausalResult,
    type: str,
    alpha: float,
    scale: Optional[str] = None,
    include_leads: bool = False,
) -> CausalResult:
    """Serve the aggregations a nonlinear ``sp.etwfe`` fit already computed.

    ``scale=None`` keeps the scale the fit was reported on.  For
    ``type='simple'`` the estimate and SE are that scale's overall ATT.  For
    the other types ``detail`` holds one row per cohort / event time /
    period with its own delta-method SE; the headline ``estimate`` is the
    unweighted mean of those rows and ``se`` the overall ATT's SE (the rows
    share coefficients, so averaging their SEs would understate).
    """
    mi = result.model_info or {}
    sc = normalise_scale(scale if scale is not None else mi.get("scale", "response"))
    aggs = mi.get("aggregations")
    if not isinstance(aggs, dict) or sc not in aggs:
        raise MethodIncompatibility(
            "This nonlinear etwfe result predates scale-aware aggregation; "
            "re-fit it with the current sp.etwfe.",
            recovery_hint="Call sp.etwfe(..., family=...) again.",
            diagnostics={"scale": sc},
        )
    agg = aggs[sc]
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    simple = agg["simple"]
    se_all = float(simple["se"])

    if type == "simple":
        if sc == mi.get("scale", "response") and alpha == result.alpha:
            return result
        est = float(simple["att"])
        z = est / se_all if se_all > 0 else 0.0
        return CausalResult(
            method=f"{result.method} — emfx[simple, {sc}]",
            estimand=(
                "ATT (average marginal effect, response scale)"
                if sc == "response"
                else "ATT (link scale)"
            ),
            estimate=est,
            se=se_all,
            pvalue=float(2 * stats.norm.sf(abs(z))),
            ci=(est - z_crit * se_all, est + z_crit * se_all),
            alpha=alpha,
            n_obs=result.n_obs,
            detail=None,
            model_info={**mi, "emfx_type": "simple", "emfx_scale": sc},
            _citation_key="wooldridge2021two",
        )

    label = {"event": "relative_time", "group": "cohort", "calendar": "period"}[type]
    frame = agg[type]
    if isinstance(frame, pd.DataFrame) and type == "event" and not include_leads:
        if "relative_time" in frame.columns:
            frame = frame[frame["relative_time"] >= 0]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataInsufficient(
            f"etwfe_emfx(type={type!r}) has no cells to report.",
            recovery_hint="Check the treated cohorts and event window.",
            diagnostics={"type": type},
        )
    frame = frame.copy()
    frame["ci_lower"] = frame["att"] - z_crit * frame["se"]
    frame["ci_upper"] = frame["att"] + z_crit * frame["se"]
    est = float(np.average(frame["att"].to_numpy(dtype=float)))
    z_stat = est / se_all if se_all > 0 else 0.0
    return CausalResult(
        method=f"{result.method} — emfx[{type}]",
        estimand=result.estimand if sc == mi.get("scale") else f"ATT ({sc} scale)",
        estimate=est,
        se=se_all,
        pvalue=float(2 * stats.norm.sf(abs(z_stat))),
        ci=(est - z_crit * se_all, est + z_crit * se_all),
        alpha=alpha,
        n_obs=result.n_obs,
        detail=frame.reset_index(drop=True),
        model_info={
            **mi,
            "emfx_type": type,
            "emfx_label": label,
            "emfx_scale": sc,
            "emfx_note": (
                "estimate is the unweighted mean of the reported cells; "
                "se is the overall delta-method SE from the fit; each row "
                "carries its own delta-method se"
            ),
        },
        _citation_key="wooldridge2021two",
    )
