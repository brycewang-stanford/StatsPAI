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
method through the cluster-robust coefficient covariance.  The event-time
aggregates also carry their joint covariance ``G V G'`` (one regression, so
leads and horizons covary); it is ``model_info['event_study_vcov']``, the
matrix ``sp.event_study_vcov``, ``sp.pretrends_test`` and ``sp.honest_did``
consume.

Aggregation weights are the treated observations of each cell.  With
``fe='unit'`` separated (all-zero) rows are kept in them by default
(``separated='keep'``): a separated row's fitted mean and marginal effect
are exactly zero, so on the response scale it enters the denominator only.
``separated='drop'`` removes them from the weights, ``N`` and the cluster
count, which is what ``jwdid`` reports whenever ``ppmlhdfe`` flags the
separation (``estat`` averages over ``e(sample)``).  ``hettype`` and
``xvar`` are built in :mod:`._etwfe_glm_design`; the fit helpers live in
:mod:`._etwfe_glm_fit` and the aggregation-serving code in
:mod:`._etwfe_glm_emfx`.

Response-scale SEs under ``fe='unit'`` are a documented convention
difference from ``jwdid``.  StatsPAI profiles the unit effect (the Poisson
first-order condition gives ``sum_t mu_it = sum_t y_it``, so ``mu`` moves
with ``beta`` only through within-unit shares) and differentiates through
it.  Stata's ``estat`` runs ``margins`` after ``ppmlhdfe``, which holds the
absorbed effects fixed and differentiates through ``ppmlhdfe``'s ``_cons``
normalisation instead; on a 44k-unit patent panel that gives 0.0318 against
StatsPAI's 0.0262 (rebuilt from Stata's ``e(V)`` to 8e-5).  Point estimates
and every link-scale SE agree.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility
from ._etwfe_glm_design import (  # noqa: F401 - re-exported
    _ETWFE_GLM_FAMILIES,
    _SCALES,
    _normalise_glm_cgroup,
    build_glm_design,
)
from ._etwfe_glm_design import is_categorical as _is_categorical
from ._etwfe_glm_design import (  # noqa: F401 - re-exported
    normalise_glm_fe,
    normalise_hettype,
    normalise_scale,
)
from ._etwfe_glm_emfx import etwfe_glm_emfx  # noqa: F401 - re-exported
from ._etwfe_glm_fit import (
    _cluster_sandwich,
    _fit_poisson_unit_fe,
    _independent_columns,
)


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
    hettype: Optional[str] = None,
    xvar: Optional[List[str]] = None,
    separated: str = "keep",
) -> CausalResult:
    """Nonlinear ETWFE — Wooldridge (2023) staggered DiD with a link function.

    See the module docstring for the design and :mod:`._etwfe_glm_design`
    for ``hettype`` / ``xvar``.  ``controls`` enter the linear index
    additively, without cohort / period / treatment interactions -- the
    role of Stata ``jwdid``'s ``exovar()``.

    Verified against R ``etwfe`` 0.6.2 (``fe='cohort'``) on a simulated
    count panel: simple AME matches to 1e-10 and every event-time AME to
    1e-6; see ``tests/reference_parity/test_etwfe_glm_parity.py``.  With
    ``fe='unit'``, every ``hettype`` and categorical / continuous ``xvar``
    matches Stata ``jwdid ..., method(ppmlhdfe)`` (``estat simple`` on both
    scales, ``estat event``, ``estat simple, over()``); see
    ``tests/reference_parity/test_etwfe_poisson_jwdid_parity.py``.
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
    if fam_key in {"normal", "linear"}:
        fam_key = "gaussian"
    if fam_key not in _ETWFE_GLM_FAMILIES and fam_key != "gaussian":
        raise MethodIncompatibility(
            f"family={family!r} is not supported; use one of "
            f"{sorted(set(_ETWFE_GLM_FAMILIES) | {'gaussian'})}.",
            recovery_hint="Pass family='poisson', 'logit', or 'gaussian'.",
            diagnostics={"family": family},
        )
    scale = normalise_scale(scale)
    fe_mode = normalise_glm_fe(fe, fam_key)
    cg = _normalise_glm_cgroup(cgroup)
    het = normalise_hettype(hettype)
    sep_mode = str(separated).strip().lower()
    if sep_mode not in {"keep", "drop"}:
        raise MethodIncompatibility(
            f"separated={separated!r} is not recognised; use 'keep' or 'drop'.",
            recovery_hint="'keep' (default) keeps separated rows in the "
            "aggregation weights, N and the cluster count; 'drop' removes "
            "them, as jwdid does whenever ppmlhdfe flags the separation.",
            diagnostics={"separated": separated},
        )
    if fam_key == "gaussian" and fe_mode != "unit":
        raise MethodIncompatibility(
            "The linear branch of etwfe_glm absorbs unit effects (Stata "
            "jwdid / reghdfe); pass fe='unit'.",
            recovery_hint="sp.etwfe without family= is R etwfe's linear design.",
            diagnostics={"fe": fe_mode},
        )
    if sep_mode == "drop" and (fe_mode != "unit" or fam_key != "poisson"):
        raise MethodIncompatibility(
            "separated='drop' applies to fe='unit' (the only design that "
            "separates units).",
            recovery_hint="Pass fe='unit', or drop separated=.",
            diagnostics={"separated": separated, "fe": fe_mode},
        )
    xvar_list: List[str] = []
    if xvar is not None:
        xvar_list = [xvar] if isinstance(xvar, str) else list(xvar)
        for xv in xvar_list:
            if xv not in data.columns:
                raise KeyError(f"xvar {xv!r} not found in data.columns")

    df = data.copy()
    df["_ft"] = df[first_treat].replace(0, np.nan)
    df["_y"] = df[y].astype(float)

    # Estimation sample first: covariates are demeaned over it (jwdid's
    # touse), so rows a regressor cannot use must go before the design.
    usable = np.isfinite(df["_y"].to_numpy(dtype=float))
    for c in list(controls or []) + xvar_list:
        col = df[c]
        usable &= col.notna().to_numpy()
        if not _is_categorical(col):
            usable &= np.isfinite(col.to_numpy(dtype=float))
    df = df.loc[usable].reset_index(drop=True)

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
    elif fam_key == "poisson" and (df["_y"] < 0).any():
        raise MethodIncompatibility(
            f"family='poisson' needs a non-negative outcome; column {y!r} "
            "contains negative values.",
            recovery_hint="Use family='gaussian' for outcomes that can be "
            "negative (e.g. logged or differenced variables).",
            diagnostics={"family": family},
        )

    des = build_glm_design(
        df,
        group=group,
        time=time,
        cohorts=cohorts,
        periods=periods,
        cgroup=cg,
        hettype=het,
        fe_mode=fe_mode,
        xvar=xvar_list,
        controls=controls,
    )
    X, names = des["X"], des["names"]
    interaction_cell: List[Tuple[int, int]] = des["cells"]
    post_arr = des["cell_post"]
    if not post_arr.any():
        raise DataInsufficient(
            "No post-treatment cohort x period cells — nothing to estimate.",
            recovery_hint="Check that treated cohorts have observed periods "
            "at or after their first_treat value.",
            diagnostics={"cohorts": [int(c) for c in cohorts]},
        )
    y_vec = df["_y"].to_numpy(dtype=float)
    n_obs = int(len(y_vec))
    n_cells = len(interaction_cell)
    row_cell = des["row_cell"]
    levels = des["level_codes"]
    n_levels = max(len(des["level_labels"]), 1)

    cluster_col = cluster or group
    cl_codes = pd.factorize(df[cluster_col])[0].astype(np.intp)
    n_clusters = int(cl_codes.max()) + 1 if n_obs else 0

    # A balanced-panel identity (Wooldridge 2023) makes pooled Poisson with
    # cohort dummies reproduce Poisson with unit effects; with missing
    # outcomes it no longer does, and the two designs give different
    # coefficients.  Say so instead of letting the gap pass silently.
    per_unit = df.groupby(group)[time].nunique()
    balanced = bool(per_unit.nunique() <= 1 and per_unit.iloc[0] == df[time].nunique())
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

    # Aggregation units: (cell, covariate level).  Every treated row of a
    # cell counts in the weights, separated rows included (their fitted mean
    # and marginal effect are exactly zero -- see the module docstring).
    in_cell = row_cell >= 0
    unit_of = np.full(n_obs, -1, dtype=int)
    unit_of[in_cell] = row_cell[in_cell] * n_levels + levels[in_cell]
    n_units = n_cells * n_levels
    unit_n = np.bincount(unit_of[in_cell], minlength=n_units).astype(float)
    unit_cell = np.repeat(np.arange(n_cells), n_levels)
    unit_level = np.tile(np.arange(n_levels), n_cells)

    sep_info: Dict[str, Any] = {"n_separated": 0}
    omitted: List[str] = []
    treat_cols = des["treat_cols"]

    if fam_key == "gaussian":
        # Stata jwdid without method(): reghdfe absorbing ivar and tvar
        # (the period effects are explicit columns here).  OLS on the
        # within-unit transform; the link and response scales coincide.
        unit_codes = pd.factorize(df[group])[0].astype(np.intp)
        live = _independent_columns(X, groups=unit_codes)
        omitted = [names[j] for j in range(len(names)) if j not in set(live.tolist())]
        Xl = X[:, live]
        U = int(unit_codes.max()) + 1
        cnt = np.bincount(unit_codes, minlength=U).astype(float)

        def _within(a: np.ndarray) -> np.ndarray:
            return (
                a - (np.bincount(unit_codes, weights=a, minlength=U) / cnt)[unit_codes]
            )

        Xd = np.column_stack([_within(Xl[:, j]) for j in range(Xl.shape[1])])
        yd = _within(y_vec)
        bread = Xd.T @ Xd
        bread_inv = np.linalg.inv(bread)
        beta = bread_inv @ (Xd.T @ yd)
        resid = yd - Xd @ beta
        k_reg = int(Xl.shape[1]) + 1  # reghdfe counts the constant
        G = n_clusters
        factor = ((n_obs - 1.0) / max(n_obs - k_reg, 1.0)) * (
            G / (G - 1.0) if G > 1 else 1.0
        )
        vcov = _cluster_sandwich(Xd * resid[:, None], bread_inv, cl_codes, G, factor)
        converged = True
        pos_in_live = {int(j): i for i, j in enumerate(live)}
        treat_k = np.array(
            [pos_in_live[int(j)] for j in treat_cols if int(j) in pos_in_live],
            dtype=int,
        )
        rows = np.flatnonzero(in_cell)
        Zt_rows = np.zeros((len(rows), Xl.shape[1]))
        Zt_rows[:, treat_k] = Xl[rows][:, treat_k]
        me_rows = Zt_rows @ beta
        grad_rows = Zt_rows
        resp_units = unit_of[rows]
        Z_all = Xl
        link_name = "Identity"
        coef_names = [names[j] for j in live]
        ssc = {"n": int(n_obs), "k": k_reg, "G": int(G), "factor": float(factor)}
        param_live = np.array([int(c) in pos_in_live for c in des["param_col"]])
    elif fe_mode == "cohort":
        sm_family = getattr(sm.families, _ETWFE_GLM_FAMILIES[fam_key])()
        live = _independent_columns(X)
        omitted = [names[j] for j in range(len(names)) if j not in set(live.tolist())]
        pos_in_live = {int(j): i for i, j in enumerate(live)}
        treat_k = np.array(
            [pos_in_live[int(j)] for j in treat_cols if int(j) in pos_in_live],
            dtype=int,
        )
        X = X[:, live]
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
        X0[:, treat_k] = 0.0
        eta1, eta0 = X @ beta, X0 @ beta
        mu1 = np.asarray(link.inverse(eta1), dtype=float)
        mu0 = np.asarray(link.inverse(eta0), dtype=float)
        dmu1 = np.asarray(link.inverse_deriv(eta1), dtype=float)
        dmu0 = np.asarray(link.inverse_deriv(eta0), dtype=float)
        rows = np.flatnonzero(in_cell)
        me_rows = (mu1 - mu0)[rows]
        grad_rows = X[rows] * dmu1[rows, None] - X0[rows] * dmu0[rows, None]
        resp_units = unit_of[rows]
        Z_all = X  # link-scale effect of row i is Z_i[treat] @ beta[treat]
        link_name = type(link).__name__
        coef_names = [names[j] for j in live]
        ssc = None
        param_live = np.array([int(c) in pos_in_live for c in des["param_col"]])
    else:
        unit_codes = pd.factorize(df[group])[0].astype(np.intp)
        res = _fit_poisson_unit_fe(
            y_vec,
            X,
            unit_codes,
            cl_codes,
            n_clusters,
            n_obs,
            count_separated=(sep_mode == "keep"),
        )
        live = res["live"]
        keep = res["keep"]
        live_set = set(live.tolist())
        omitted = [names[j] for j in range(len(names)) if j not in live_set]
        n_sep = int((~keep).sum())
        sep_info = {
            "n_separated": n_sep,
            "n_separated_by_rule": dict(res["sep_counts"]),
        }
        if n_sep:
            where = (
                "they stay in N, the cluster count and the aggregation "
                "weights (separated='keep')"
                if sep_mode == "keep"
                else "they are dropped from N, the cluster count and the "
                "aggregation weights (separated='drop')"
            )
            warnings.warn(
                f"etwfe(family='poisson', fe='unit'): {n_sep} separated "
                "observation(s) (all-zero units / perfectly predicted zeros) "
                "have a fitted mean of exactly zero and were left out of "
                f"IRLS; {where}.",
                UserWarning,
                stacklevel=3,
            )
            if sep_mode == "drop":
                in_cell = in_cell & keep
                unit_of = np.where(keep, unit_of, -1)
                unit_n = np.bincount(unit_of[in_cell], minlength=n_units).astype(float)
                n_obs = int(keep.sum())
                n_clusters = int(res["ssc"]["G"])
        beta = res["beta"]
        vcov = res["vcov"]
        converged = res["converged"]
        coef_names = [names[j] for j in live]
        pos_in_live = {int(j): i for i, j in enumerate(live)}
        treat_k = np.array(
            [pos_in_live[int(j)] for j in treat_cols if int(j) in pos_in_live],
            dtype=int,
        )
        Z_all = X[:, live]
        Xl = Z_all[keep]
        cmask = np.zeros(len(live), dtype=bool)
        cmask[treat_k] = True
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
        unit_k = unit_of[keep]
        rows = np.flatnonzero(unit_k >= 0)
        X0l = Xl[rows].copy()
        X0l[:, cmask] = 0.0
        me_rows = (mu - mu0)[rows]
        grad_rows = mu[rows, None] * (Xl[rows] - xbar[rows]) - mu0[rows, None] * (
            X0l - xbar[rows]
        )
        resp_units = unit_k[rows]
        link_name = "Log"
        ssc = res["ssc"]
        param_live = np.array([int(c) in pos_in_live for c in des["param_col"]])
        dropped = [lab for lab, ok in zip(des["param_labels"], param_live) if not ok]
        if dropped:
            warnings.warn(
                f"etwfe(family='poisson', fe='unit'): treatment parameter(s) "
                f"{dropped} have only separated observations and are omitted "
                "from every aggregate.",
                UserWarning,
                stacklevel=3,
            )

    # Cells whose treatment parameter was not estimable leave the weights.
    cell_live = param_live[des["cell_param"]]
    unit_n = np.where(cell_live[unit_cell], unit_n, 0.0)

    K = len(beta)
    # Per-unit sums: link-scale effect sum_i z_i' beta and its gradient
    # sum_i z_i (z_i = the treatment columns of row i), and the
    # response-scale marginal effects and their gradients.
    rows_all = np.flatnonzero(in_cell)
    u_all = unit_of[rows_all]
    Zt = np.asarray(Z_all[rows_all][:, treat_k], dtype=float)
    link_grad = np.zeros((n_units, K))
    for j, k in enumerate(treat_k):
        link_grad[:, k] = np.bincount(u_all, weights=Zt[:, j], minlength=n_units)
    link_sum = link_grad @ beta
    me_sum = np.bincount(resp_units, weights=me_rows, minlength=n_units)
    grad_sum = (
        np.column_stack(
            [
                np.bincount(resp_units, weights=grad_rows[:, j], minlength=n_units)
                for j in range(K)
            ]
        )
        if K
        else np.zeros((n_units, 0))
    )

    z_crit = float(stats.norm.ppf(1 - alpha / 2))

    def _agg(sel: np.ndarray, sc: str) -> Tuple[float, float, int, np.ndarray]:
        """Aggregate over the (cell, level) units in ``sel``; the gradient
        is returned so joint covariances of several aggregates are
        ``G V G'``."""
        sel = sel & (unit_n > 0)
        n_sel = float(unit_n[sel].sum())
        if n_sel <= 0:
            return np.nan, np.nan, 0, np.zeros(K)
        if sc == "response":
            est = float(me_sum[sel].sum() / n_sel)
            grad = grad_sum[sel].sum(axis=0) / n_sel
        else:
            est = float(link_sum[sel].sum() / n_sel)
            grad = link_grad[sel].sum(axis=0) / n_sel
        var = float(grad @ vcov @ grad)
        return est, float(np.sqrt(max(var, 0.0))), int(n_sel), grad

    g_arr = np.array([c[0] for c in interaction_cell], dtype=float)
    t_arr = np.array([c[1] for c in interaction_cell], dtype=float)
    e_arr = t_arr - g_arr
    ug, ut, ue = g_arr[unit_cell], t_arr[unit_cell], e_arr[unit_cell]
    upost = post_arr[unit_cell]

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

    def _tables(sc: str, lev: Optional[int] = None) -> Dict[str, Any]:
        base = np.ones(n_units, dtype=bool) if lev is None else unit_level == lev
        simple = _agg(base & upost, sc)
        ev_rows, ev_times, ev_grads = [], [], []
        for e_val in sorted(set(e_arr.tolist())):
            est, se, n, grad = _agg(base & (ue == e_val), sc)
            if n:
                ev_rows.append(_row("relative_time", int(e_val), est, se, n))
                ev_times.append(int(e_val))
                ev_grads.append(grad)
        ev = pd.DataFrame(ev_rows)
        if not ev.empty:
            ev["post"] = ev["relative_time"] >= 0
        # Joint covariance of the event-time aggregates.  Leads and horizons
        # are cells of one regression, so the matrix is joint (not block
        # diagonal) and its diagonal is exactly the table's se**2 --
        # sp.event_study_vcov / pretrends_test / honest_did read it.
        Gm = np.asarray(ev_grads, dtype=float).reshape(len(ev_times), K)
        ev_vcov = pd.DataFrame(Gm @ vcov @ Gm.T, index=ev_times, columns=ev_times)
        grp_rows = []
        for g_val in cohorts:
            est, se, n, _ = _agg(base & upost & (ug == g_val), sc)
            if n:
                grp_rows.append(_row("cohort", int(g_val), est, se, n))
        cal_rows = []
        for t_val in sorted(set(t_arr[post_arr].tolist())):
            est, se, n, _ = _agg(base & upost & (ut == t_val), sc)
            if n:
                cal_rows.append(_row("period", int(t_val), est, se, n))
        return {
            "simple": {"att": simple[0], "se": simple[1], "n_treated": simple[2]},
            "event": ev,
            "event_vcov": ev_vcov,
            "group": pd.DataFrame(grp_rows),
            "calendar": pd.DataFrame(cal_rows),
        }

    aggregations: Dict[str, Dict[str, Any]] = {}
    for sc in _SCALES:
        aggregations[sc] = _tables(sc)
        if des["level_var"] is not None:
            by = {}
            for li, lab in enumerate(des["level_labels"]):
                by[lab] = _tables(sc, li)
            aggregations[sc]["by_xvar"] = by

    # Per-cell table: link-scale cell ATT (the cell coefficient without
    # covariates; its covariate-averaged value with them), its SE, the AME.
    cell_rows = []
    for c in range(n_cells):
        sel = unit_cell == c
        est, se, n, _ = _agg(sel, "link")
        n_c = float(unit_n[sel].sum())
        cell_rows.append(
            (
                est,
                se,
                int(n_c),
                float(me_sum[sel].sum() / n_c) if n_c > 0 else np.nan,
            )
        )
    cell_n = np.array([r[2] for r in cell_rows], dtype=float)
    cells_df = pd.DataFrame(
        {
            "cohort": [c[0] for c in interaction_cell],
            "period": [c[1] for c in interaction_cell],
            "relative_time": e_arr.astype(int),
            "post": post_arr,
            "param": [des["param_labels"][p] for p in des["cell_param"]],
            "n_treated": cell_n.astype(int),
            "coef": [r[0] for r in cell_rows],
            "se": [r[1] for r in cell_rows],
            "ame": [r[3] for r in cell_rows],
        }
    )
    by_xvar_simple = None
    if des["level_var"] is not None:
        by_xvar_simple = pd.DataFrame(
            [
                _row(
                    "level",
                    lab,
                    aggregations[scale]["by_xvar"][lab]["simple"]["att"],
                    aggregations[scale]["by_xvar"][lab]["simple"]["se"],
                    aggregations[scale]["by_xvar"][lab]["simple"]["n_treated"],
                )
                for lab in des["level_labels"]
            ]
        )

    head = aggregations[scale]
    att, se_att = head["simple"]["att"], head["simple"]["se"]
    z_stat = att / se_att if se_att > 0 else 0.0
    if fam_key == "gaussian":
        estimand = "ATT (linear ETWFE, treated-observation-weighted)"
    elif scale == "response":
        estimand = "ATT (average marginal effect, response scale)"
    else:
        estimand = (
            "ATT (link scale: treated-observation-weighted mean of the "
            + ("log-point" if fam_key == "poisson" else "log-odds")
            + " cohort x period effects)"
        )

    ev_head = head["event"]
    event_study = ev_head.drop(columns=["ci_lower", "ci_upper"], errors="ignore")
    group_tbl = head["group"]
    calendar_tbl = head["calendar"]

    return CausalResult(
        method=(
            f"Wooldridge (2021) ETWFE — linear, unit FE, hettype={het}"
            if fam_key == "gaussian"
            else f"Wooldridge (2023) nonlinear ETWFE — family={fam_key}"
        ),
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
            "hettype": het,
            "scale": scale,
            "event_study": event_study,
            "event_study_vcov": head["event_vcov"],
            "calendar": (
                calendar_tbl[["period", "att", "se", "n_treated"]].copy()
                if not calendar_tbl.empty
                else calendar_tbl
            ),
            "coef_names": coef_names,
            "coefficients": beta,
            "vcov": vcov,
            "interaction_cells": interaction_cell,
            "treatment_params": list(des["param_labels"]),
            "cells": cells_df,
            "aggregations": aggregations,
            "xvar": xvar_list,
            "xvar_columns": des["x_names"],
            "by_xvar_var": des["level_var"],
            "by_xvar": by_xvar_simple,
            "att_response": aggregations["response"]["simple"]["att"],
            "se_response": aggregations["response"]["simple"]["se"],
            "att_link": aggregations["link"]["simple"]["att"],
            "se_link": aggregations["link"]["simple"]["se"],
            "n_treated_obs": int(cell_n[post_arr].sum()),
            "n_clusters": n_clusters,
            "se_type": f"cluster-robust on {cluster_col}",
            "controls": des["ctrl_names"],
            "balanced_panel": balanced,
            "omitted": omitted,
            "ssc": ssc,
            "converged": converged,
            **sep_info,
        },
        _citation_key="wooldridge2021two",
    )
