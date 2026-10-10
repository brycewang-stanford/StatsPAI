"""Heterogeneity analysis after a causal forest: groups, support, pre-trends.

Three questions come after the forest has produced ``tau(x)``, and each
needs its own inference rather than a summary of the fitted predictions:

* :func:`forest_group_effects` -- how large is the effect for a country
  pair, a country, a period, or a quantile of the forest's prediction?
  Averages of unbiased scores with valid standard errors (imputation scores
  for forests with fixed effects, AIPW scores otherwise), including
  membership groups for dyadic data and a dyadic-robust variance.
* :func:`forest_support` -- is a counterfactual prediction for a unit that
  was never treated an interpolation or an extrapolation?
* :func:`cate_pretrend_test` -- were the units the forest ranks as high-
  effect already on a different trajectory before treatment?

The worked example in ``docs/guides/heterogeneity_panel_forests.md``
reproduces the workflow of [aytug2026euro] (causal forests with fixed
effects [kattenberg2023causal] on a dyadic trade panel) on simulated data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)
from . import _fe_imputation as fi
from . import _grf_inference as gi

__all__ = [
    "forest_group_effects",
    "forest_support",
    "cate_pretrend_test",
    "rate_split",
    "forest_policy_tree",
]


def _require_grf(forest: Any, context: str) -> None:
    if not getattr(forest, "fitted_", False) or not gi.is_grf_forest(forest):
        raise MethodIncompatibility(
            f"{context} needs a causal forest fitted with the GRF engine.",
            recovery_hint=(
                "Fit with sp.causal_forest(...) (default split_rule='grf') "
                "before calling this function."
            ),
        )


# --------------------------------------------------------------------------- #
#  Group effects
# --------------------------------------------------------------------------- #


@accepts_aliases(_strict=True, controls="covariates")
def forest_group_effects(
    forest: Any,
    by: Any = None,
    *,
    members: Any = None,
    n_groups: int = 4,
    cluster: Any = None,
    variance: str = "forest",
    alpha: float = 0.05,
    scale: str = "level",
    min_rows: int = 1,
    covariates: Any = "none",
) -> pd.DataFrame:
    """Average treatment effects by group, with valid standard errors.

    A causal forest's predictions are regularised: averaging them over a
    group gives a number that is shrunk toward the overall mean and whose
    spread says nothing about sampling error.  This function averages an
    *unbiased* per-row signal instead and uses the forest only to form
    groups when asked to:

    * **Forests with two-way fixed effects** (``fe="twoway"``): the
      imputation score ``Y - alpha_hat_i - gamma_hat_t`` of each treated
      cell, with unit and period effects fitted on the untreated cells
      [borusyak2024revisiting].  Each group estimate is the ATT over the
      group's treated cells.  Standard errors use the exact linear weights
      of the imputation estimator; treated residuals are centred on the
      out-of-bag forest prediction and then on cohort x event-time means
      (``variance="forest"``) or on those means only (``variance="bjs"``,
      the conservative convention of :func:`statspai.did_imputation`).
    * **Pooled forests**: the AIPW score of every row; each estimate is the
      group ATE.

    Rows are weighted by the forest's observation weights, so with
    ``equalize_cluster_weights=True`` every cluster gets equal weight in each
    group average (a different estimand from :func:`statspai.did_imputation`,
    which weights treated cells equally).

    Parameters
    ----------
    forest : CausalForest
        Fitted with the GRF engine.
    by : None, "cate_quantile" or array-like
        ``None`` pools all eligible rows (or, with ``members``, forms one
        group per member).  ``"cate_quantile"`` bins eligible rows into
        ``n_groups`` quantiles of the out-of-bag forest prediction (sorted
        group average treatment effects, [chernozhukov2025generic]).  An
        array gives one label per training row (a pair id, a period band,
        ``df["year"] >= 2009`` ...).
    members : (n, 2) array-like, optional
        The two members of each row of dyadic data (exporter and importer,
        the two countries of a pair).  With ``by=None`` each member is a
        group and a row counts for both of its members -- the average
        effect over all pairs a country belongs to.
    n_groups : int, default 4
        Number of quantile groups for ``by="cate_quantile"``.
    cluster : None, "dyadic" or array-like
        ``None`` clusters by the forest's clusters (default: the panel
        unit).  ``"dyadic"`` (requires ``members``) allows any two rows
        that share a member to be correlated [aronow2015cluster].  An array
        gives other cluster ids.
    variance : {"forest", "bjs"}, default "forest"
        Forests with fixed effects only; see above.
    alpha : float, default 0.05
    scale : {"level", "percent"}, default "level"
        ``"percent"`` adds ``100 * (exp(x) - 1)`` columns for log outcomes
        (interval endpoints are transformed, not re-derived).
    min_rows : int, default 1
        Drop groups with fewer eligible rows.
    covariates : "none", "auto", list of str or array, default "none"
        Forests with fixed effects only: covariates entering the untreated
        outcome model linearly, ``Y(0) = alpha_i + gamma_t + C' beta``.
        ``"none"`` is the pure two-way model of
        :func:`statspai.did_imputation`; ``"auto"`` adds the forest's effect
        modifiers and covariates that vary within units (time-invariant ones
        are absorbed by the unit effect).  Controls must not be affected by
        the treatment.

    Returns
    -------
    pandas.DataFrame
        One row per group: ``n_rows``, ``n_units``, ``estimate``, ``se``,
        ``z``, ``p``, ``ci_low``, ``ci_high`` and ``forest_mean`` (the mean
        out-of-bag prediction over the same rows, for comparison).
        ``attrs`` holds ``method``, ``estimand``, the full ``vcov`` and
        ``tests``: a Wald test that all group effects are equal and the
        last-minus-first difference with its standard error.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.currency_union_panel(seed=0)
    >>> cf = sp.causal_forest(
    ...     data=df, y="log_trade", d="euro",
    ...     x=["pre_trade", "log_gdp_prod", "log_gdppc"],
    ...     id="pair", time="year", fe="twoway",
    ...     n_estimators=400, random_state=0,
    ... )
    >>> by_country = sp.forest_group_effects(
    ...     cf, members=df[["country_i", "country_j"]].to_numpy(),
    ...     scale="percent",
    ... )
    >>> by_country.index.name
    'group'

    References
    ----------
    [borusyak2024revisiting], [chernozhukov2025generic], [aronow2015cluster],
    [kattenberg2023causal]
    """
    _require_grf(forest, "forest_group_effects()")
    return fi.group_effects(
        forest,
        by,
        members=members,
        n_groups=n_groups,
        cluster=cluster,
        variance=variance,
        alpha=alpha,
        scale=scale,
        min_rows=min_rows,
        covariates=covariates,
    )


# --------------------------------------------------------------------------- #
#  Support of counterfactual predictions
# --------------------------------------------------------------------------- #


def _reference_rows(forest: Any) -> np.ndarray:
    """Rows that inform the effect: switching units for FE forests."""
    n = int(len(forest._Y_original))
    if not gi.is_fe_forest(forest):
        return np.ones(n, dtype=bool)
    unit = np.asarray(forest._fe_unit, dtype=np.int64)
    T = np.asarray(forest._T_original, dtype=float)
    n_u = int(unit.max()) + 1
    lo = np.full(n_u, np.inf)
    hi = np.full(n_u, -np.inf)
    np.minimum.at(lo, unit, T)
    np.maximum.at(hi, unit, T)
    return np.asarray((hi > lo)[unit])


def _kth_distance(ref: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(ref).query(query, k=k)
    dist = np.asarray(dist)
    return dist if dist.ndim == 1 else dist[:, -1]


def _kth_distance_other_units(
    ref: np.ndarray, groups: np.ndarray, k: int
) -> np.ndarray:
    """k-th nearest-neighbour distance of each reference row to rows of
    *other* units, so repeated rows of a unit cannot shrink the benchmark
    that a new unit is compared with."""
    from scipy.spatial import cKDTree

    max_rows = int(np.bincount(groups).max())
    kk = min(k + max_rows, ref.shape[0])
    dist, idx = cKDTree(ref).query(ref, k=kk)
    dist = np.asarray(dist).reshape(ref.shape[0], -1)
    idx = np.asarray(idx).reshape(ref.shape[0], -1)
    other = groups[idx] != groups[:, None]
    out = np.full(ref.shape[0], np.nan)
    for i in range(ref.shape[0]):
        d = dist[i, other[i]]
        if d.size >= k:
            out[i] = d[k - 1]
    return out


def forest_support(
    forest: Any,
    X_new: Any,
    *,
    k: int = 10,
    quantile: float = 0.95,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Check whether new rows lie inside the support that informs ``tau(x)``.

    Predicting the effect a unit *would* have had (a country that did not
    adopt a currency, a region that was never treated) applies the fitted
    ``tau(x)`` outside the treated sample.  That is a model-based
    extrapolation, credible only where the forest has seen similar units
    switch treatment.  For every row of ``X_new`` this reports

    * ``cate``, its little-bag ``se`` and a ``(1 - alpha)`` interval;
    * ``n_outside_range`` -- effect modifiers outside the ``[min, max]`` of
      the reference rows;
    * ``knn_ratio`` -- the distance (on reference-standardised modifiers)
      to the ``k``-th nearest reference row, divided by the ``quantile``
      of the reference rows' distance to their ``k``-th nearest row from a
      *different* unit (rows of the same unit are excluded, so a unit's
      other periods do not shrink the benchmark).  Above 1 the row is
      farther from the data than all but ``1 - quantile`` of the reference
      rows are from other units;
    * ``supported`` -- ``n_outside_range == 0`` and ``knn_ratio <= 1``.

    The reference rows are the rows that inform the effect: for a forest
    with fixed effects, the rows of units whose treatment varies (only they
    identify the within effect); for a pooled forest, all training rows.

    Parameters
    ----------
    forest : CausalForest
        Fitted with the GRF engine.
    X_new : array-like or DataFrame
        Effect modifiers of the rows to predict (DataFrame columns are
        matched to the fitted feature names).
    k : int, default 10
    quantile : float, default 0.95
    alpha : float, default 0.05

    Returns
    -------
    pandas.DataFrame
        One row per row of ``X_new``; ``attrs["summary"]`` holds the share
        of supported rows and the reference-sample size.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.currency_union_panel(seed=0)
    >>> x = ["pre_trade", "log_gdp_prod", "log_gdppc"]
    >>> cf = sp.causal_forest(
    ...     data=df, y="log_trade", d="euro", x=x,
    ...     id="pair", time="year", fe="twoway",
    ...     n_estimators=400, random_state=0,
    ... )
    >>> outs = df[df["ever_euro"] == 0]
    >>> sup = sp.forest_support(cf, outs[x])
    >>> bool(sup["supported"].mean() > 0)
    True
    """
    _require_grf(forest, "forest_support()")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise MethodIncompatibility(
            "forest_support(): k must be a positive integer.",
            recovery_hint="Use k=10.",
        )
    if not 0.0 < float(quantile) < 1.0 or not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility(
            "forest_support(): quantile and alpha must lie in (0, 1).",
            recovery_hint="Use quantile=0.95, alpha=0.05.",
        )
    Xn = forest._prepare_effect_matrix(X_new, context="forest_support()")
    ref_mask = _reference_rows(forest)
    ref = np.asarray(forest._X_original, dtype=float)[ref_mask]
    if gi.is_fe_forest(forest):
        ref_units = np.asarray(forest._fe_unit, dtype=np.int64)[ref_mask]
    elif getattr(forest, "_clusters", None) is not None:
        ref_units = np.asarray(forest._clusters, dtype=np.int64)[ref_mask]
    else:
        ref_units = np.arange(ref.shape[0], dtype=np.int64)
    ref_units = np.asarray(pd.factorize(pd.Series(ref_units))[0], dtype=np.int64)
    if ref.shape[0] <= k:
        raise DataInsufficient(
            "forest_support(): fewer reference rows than k.",
            recovery_hint="Lower k.",
            diagnostics={"n_reference_rows": int(ref.shape[0]), "k": int(k)},
        )
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    outside = (Xn < lo) | (Xn > hi)
    scale = ref.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    ref_s = (ref - ref.mean(axis=0)) / scale
    new_s = (Xn - ref.mean(axis=0)) / scale
    if int(ref_units.max()) + 1 <= k:
        raise DataInsufficient(
            "forest_support(): the reference rows come from k or fewer units.",
            recovery_hint="Lower k.",
            diagnostics={"n_reference_units": int(ref_units.max()) + 1, "k": int(k)},
        )
    ref_dist = _kth_distance_other_units(ref_s, ref_units, int(k))
    cutoff = float(np.nanquantile(ref_dist, float(quantile)))
    new_dist = _kth_distance(ref_s, new_s, int(k))
    ratio = new_dist / cutoff if cutoff > 0 else np.where(new_dist > 0, np.inf, 0.0)
    cate = np.asarray(forest.effect(Xn), dtype=float).ravel()
    try:
        var = np.asarray(forest.effect_variance(Xn), dtype=float).ravel()
        se = np.sqrt(np.clip(var, 0.0, None))
    except MethodIncompatibility:
        se = np.full(cate.size, np.nan)
    z = float(stats.norm.ppf(1 - float(alpha) / 2))
    names = list(getattr(forest, "_feature_names", None) or [])
    out = pd.DataFrame(
        {
            "cate": cate,
            "se": se,
            "ci_low": cate - z * se,
            "ci_high": cate + z * se,
            "n_outside_range": outside.sum(axis=1).astype(int),
            "knn_distance": new_dist,
            "knn_ratio": ratio,
        }
    )
    out["supported"] = (out["n_outside_range"] == 0) & (out["knn_ratio"] <= 1.0)
    if isinstance(X_new, pd.DataFrame):
        out.index = X_new.index
    per_feature = {
        (names[j] if j < len(names) else f"X{j}"): float(outside[:, j].mean())
        for j in range(Xn.shape[1])
    }
    out.attrs["summary"] = {
        "share_supported": float(out["supported"].mean()),
        "share_outside_range_by_feature": per_feature,
        "n_reference_rows": int(ref.shape[0]),
        "reference": (
            "rows of units whose treatment varies"
            if gi.is_fe_forest(forest)
            else "all training rows"
        ),
        "knn_cutoff": cutoff,
        "k": int(k),
        "quantile": float(quantile),
    }
    return out


# --------------------------------------------------------------------------- #
#  Pre-trends by predicted-effect group
# --------------------------------------------------------------------------- #


def _two_way_demean(V: np.ndarray, unit: np.ndarray, time: np.ndarray) -> np.ndarray:
    from . import _grf_engine as engine

    n = unit.size
    idx = np.arange(n, dtype=np.int64)
    G = np.ones(n)
    n_u = int(unit.max()) + 1
    n_t = int(time.max()) + 1
    cols = []
    for j in range(V.shape[1]):
        out = np.zeros(n)
        sweeps = engine._fe_residualize(
            idx,
            np.ascontiguousarray(V[:, j], dtype=np.float64),
            G,
            unit,
            time,
            np.zeros(n_u),
            np.zeros(n_u),
            np.zeros(n_t),
            np.zeros(n_t),
            out,
            100000,
            1e-13,
        )
        if sweeps < 0:
            raise DataInsufficient(
                "cate_pretrend_test(): the two-way within transformation did "
                "not converge.",
                recovery_hint=(
                    "Check the untreated panel for disconnected unit-period " "sets."
                ),
            )
        cols.append(out)
    return np.column_stack(cols)


@accepts_aliases(_strict=True, controls="covariates")
def cate_pretrend_test(
    forest: Any,
    *,
    n_groups: int = 2,
    leads: Optional[int] = None,
    groups: Any = None,
    time_effects: str = "common",
    alpha: float = 0.05,
    covariates: Any = "none",
) -> Dict[str, Any]:
    """Test whether predicted-effect groups had different pre-treatment trends.

    Heterogeneity found by a forest can be spurious if the units it ranks
    high were already diverging before treatment [aytug2026euro].  This
    test sorts units into ``n_groups`` quantiles of their mean out-of-bag
    prediction (or uses ``groups``) and, on the **untreated** cells only,
    regresses the outcome on unit and period effects and on lead indicators
    ``1[t - g = k] x 1[group]`` for ``k = -1, ..., -leads`` -- the
    pre-trend regression of [borusyak2024revisiting], interacted with the
    groups.  Periods earlier than ``-leads`` and never-treated units are the
    reference.  Using untreated cells only keeps post-treatment effect
    heterogeneity out of the test.

    Two Wald tests are reported: all leads are zero, and the leads are
    equal across groups (for each ``k``, every group's lead equals the
    first group's).  The second is the test that bears on heterogeneity.
    Standard errors are clustered by the forest's clusters with the CR1
    factor ``G/(G-1) (n-1)/(n-K)``, where ``K`` counts the leads and the
    period effects (unit effects are nested in unit clusters, fixest's
    ``fixef.K = "nested"``).

    Parameters
    ----------
    forest : CausalForest
        Fitted with ``fe="twoway"`` and a binary absorbing treatment.
    n_groups : int, default 2
        Quantile groups of the unit-level mean OOB prediction.
    leads : int, optional
        Number of pre-treatment leads (default: ``min(4, L - 1)`` where
        ``L`` is the longest pre-treatment spell; with ``L`` leads every
        pre-period of the earliest-observed units is a lead and the leads
        are collinear with their unit effects).
    groups : array-like, optional
        One label per training row, constant within units; overrides
        ``n_groups``.
    time_effects : {"common", "by_group"}, default "common"
        ``"common"`` matches the forest and the imputation scores, which
        use one set of period effects; ``"by_group"`` gives each group its
        own period effects (separate event studies per group).
    alpha : float, default 0.05
    covariates : "none", "auto", list of str or array, default "none"
        Time-varying covariates added linearly, as in the imputation model
        of :func:`forest_group_effects`.

    Returns
    -------
    dict
        ``coefficients`` (DataFrame indexed by group and lead), ``joint_zero``
        and ``equal_across_groups`` (``stat``, ``df``, ``p``), ``group_of_unit``
        (Series), ``n_obs``, ``n_clusters`` and ``method``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.currency_union_panel(seed=0)
    >>> cf = sp.causal_forest(
    ...     data=df, y="log_trade", d="euro",
    ...     x=["pre_trade", "log_gdp_prod", "log_gdppc"],
    ...     id="pair", time="year", fe="twoway",
    ...     n_estimators=400, random_state=0,
    ... )
    >>> res = sp.cate_pretrend_test(cf, n_groups=2, leads=3)
    >>> sorted(res["equal_across_groups"])
    ['df', 'p', 'stat']

    References
    ----------
    [borusyak2024revisiting], [aytug2026euro]
    """
    context = "cate_pretrend_test()"
    _require_grf(forest, context)
    from ._grf_family import validate_alpha

    alpha = validate_alpha(alpha, context)
    if not gi.is_fe_forest(forest):
        raise MethodIncompatibility(
            f"{context} needs a causal forest with fixed effects (fe=...).",
            recovery_hint=(
                "Fit sp.causal_forest(..., id=, time=, fe='twoway'); for "
                "staggered designs sp.did_forest() reports its own pre-trend test."
            ),
            alternative_functions=["sp.did_forest"],
        )
    if time_effects not in ("common", "by_group"):
        raise MethodIncompatibility(
            f"{context}: time_effects must be 'common' or 'by_group'.",
            recovery_hint="Use time_effects='common'.",
        )
    design = fi.imputation_design(forest, context, covariates)
    if not design.absorbing:
        raise MethodIncompatibility(
            f"{context} needs an absorbing (staggered) treatment so that "
            "leads relative to the adoption period are defined.",
            recovery_hint="Use a design in which units stay treated once treated.",
        )
    unit, time = design.unit, design.time
    n = design.n
    n_u = int(unit.max()) + 1

    # Unit-level groups.
    if groups is None:
        if not isinstance(n_groups, (int, np.integer)) or n_groups < 2:
            raise MethodIncompatibility(
                f"{context}: n_groups must be an integer >= 2.",
                recovery_hint="Use n_groups=2 (above / below the median).",
            )
        tau = np.asarray(forest._oob_tau, dtype=float)
        ok = np.isfinite(tau)
        s = np.bincount(unit[ok], weights=tau[ok], minlength=n_u)
        c = np.bincount(unit[ok], minlength=n_u)
        unit_tau = np.divide(s, c, out=np.full(n_u, np.nan), where=c > 0)
        valid = np.isfinite(unit_tau)
        ranks = pd.Series(unit_tau[valid]).rank(method="first").to_numpy()
        code = np.full(n_u, -1, dtype=np.int64)
        code[valid] = np.minimum(
            (ranks - 1) * n_groups // valid.sum(), n_groups - 1
        ).astype(np.int64)
        labels: List[Any] = [f"Q{g + 1}" for g in range(int(n_groups))]
        row_group = code[unit]
    else:
        arr = np.asarray(groups, dtype=object)
        if arr.ndim != 1 or arr.size != n or pd.isna(pd.Series(arr)).any():
            raise MethodIncompatibility(
                f"{context}: groups must give one non-missing label per training row.",
                recovery_hint="Pass df['group'].to_numpy() aligned with the fit rows.",
            )
        codes, uniq = pd.factorize(pd.Series(arr), sort=True)
        per_unit = pd.Series(codes).groupby(unit).nunique()
        if (per_unit > 1).any():
            raise MethodIncompatibility(
                f"{context}: groups must be constant within units.",
                recovery_hint="Assign each unit to one group.",
            )
        row_group = codes.astype(np.int64)
        labels = list(uniq)
    G = len(labels)
    if G < 2:
        raise DataInsufficient(
            f"{context}: need at least two groups.",
            recovery_hint="Use n_groups >= 2 or pass two or more labels.",
        )

    untreated = ~design.treated & (row_group >= 0)
    ever = design.cohort >= 0
    rel = design.rel
    max_leads = int(-rel[untreated & ever].min()) if np.any(untreated & ever) else 0
    if max_leads < 2 and leads is None:
        raise DataInsufficient(
            f"{context}: treated units have at most one untreated period "
            "before adoption, so leads cannot be separated from unit effects.",
            recovery_hint=(
                "Pass leads=1 explicitly only if some units have more " "pre-periods."
            ),
            diagnostics={"max_leads": max_leads},
        )
    if leads is None:
        leads = min(4, max_leads - 1)
    if not isinstance(leads, (int, np.integer)) or leads < 1 or leads > max_leads:
        raise DataInsufficient(
            f"{context}: leads must be between 1 and {max_leads} for this panel.",
            recovery_hint=(
                "Treated units need that many untreated periods before " "adoption."
            ),
            diagnostics={"max_leads": max_leads},
        )
    rows = np.flatnonzero(untreated)
    cols: List[np.ndarray] = []
    index: List[Any] = []
    for g in range(G):
        for kk in range(1, int(leads) + 1):
            col = ((row_group[rows] == g) & ever[rows] & (rel[rows] == -kk)).astype(
                float
            )
            if col.sum() == 0:
                continue
            cols.append(col)
            index.append((labels[g], -kk))
    if not cols:
        raise DataInsufficient(
            f"{context}: no untreated pre-treatment cells within the leads.",
            recovery_hint="Use fewer leads or a panel with pre-periods.",
        )
    n_leads_cols = len(cols)
    if design.control_names:
        C, cnames = fi._candidate_covariates(forest, covariates)
        C = C[:, [cnames.index(c) for c in design.control_names]]
        cols.extend(C[rows].T)
    L = np.column_stack(cols)
    y = design.y[rows]
    u = np.asarray(pd.factorize(pd.Series(unit[rows]))[0], dtype=np.int64)
    if time_effects == "common":
        tt = np.asarray(pd.factorize(pd.Series(time[rows]))[0], dtype=np.int64)
    else:
        key = pd.Series(time[rows].astype(np.int64) * (G + 1) + row_group[rows])
        tt = np.asarray(pd.factorize(key)[0], dtype=np.int64)
    V = _two_way_demean(np.column_stack([y, L]), u, tt)
    yd, Ld = V[:, 0], V[:, 1:]
    XtX = Ld.T @ Ld
    if np.linalg.matrix_rank(XtX) < Ld.shape[1]:
        raise DataInsufficient(
            f"{context}: the lead indicators are collinear with the fixed effects.",
            recovery_hint="Use fewer leads or time_effects='common'.",
        )
    XtX_inv = np.linalg.inv(XtX)
    beta_all = XtX_inv @ (Ld.T @ yd)
    resid = yd - Ld @ beta_all
    cl = design.clusters[rows]
    cl = np.asarray(pd.factorize(pd.Series(cl))[0], dtype=np.int64)
    n_cl = int(cl.max()) + 1
    S = np.zeros((n_cl, Ld.shape[1]))
    np.add.at(S, cl, Ld * resid[:, None])
    n_obs = rows.size
    # Period effects: one redundancy with the unit effects per group of
    # units sharing a set of period dummies (all units for common period
    # effects, each group for group-specific ones).
    n_period_sets = 1 if time_effects == "common" else G
    k_fe_time = int(tt.max()) + 1 - n_period_sets
    unit_nested = bool(pd.Series(cl).groupby(u).nunique().max() == 1)
    K = Ld.shape[1] + k_fe_time + (0 if unit_nested else int(u.max()) + 1)
    factor = n_cl / (n_cl - 1) * (n_obs - 1) / max(n_obs - K, 1)
    vcov_all = factor * XtX_inv @ (S.T @ S) @ XtX_inv
    beta = beta_all[:n_leads_cols]
    vcov = vcov_all[:n_leads_cols, :n_leads_cols]
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
    z = float(stats.norm.ppf(1 - alpha / 2))
    coef = pd.DataFrame(
        {
            "coef": beta,
            "se": se,
            "z": beta / np.where(se > 0, se, np.nan),
            "ci_low": beta - z * se,
            "ci_high": beta + z * se,
        },
        index=pd.MultiIndex.from_tuples(index, names=["group", "lead"]),
    )
    coef["p"] = 2 * stats.norm.sf(np.abs(coef["z"]))

    def _wald(R: np.ndarray) -> Dict[str, float]:
        d = R @ beta
        M = R @ vcov @ R.T
        stat = float(d @ np.linalg.pinv(M) @ d)
        dof = int(np.linalg.matrix_rank(M))
        return {"stat": stat, "df": dof, "p": float(stats.chi2.sf(stat, dof))}

    joint = _wald(np.eye(beta.size))
    rows_R = []
    pos = {key: i for i, key in enumerate(index)}
    for kk in range(1, int(leads) + 1):
        base = (labels[0], -kk)
        if base not in pos:
            continue
        for g in range(1, G):
            key = (labels[g], -kk)
            if key in pos:
                r = np.zeros(beta.size)
                r[pos[key]] = 1.0
                r[pos[base]] = -1.0
                rows_R.append(r)
    equal = (
        _wald(np.vstack(rows_R)) if rows_R else {"stat": np.nan, "df": 0, "p": np.nan}
    )
    group_of_unit = pd.Series(
        [
            labels[c] if c >= 0 else None
            for c in pd.Series(row_group).groupby(unit).first()
        ],
        name="group",
    )
    return {
        "coefficients": coef,
        "joint_zero": joint,
        "equal_across_groups": equal,
        "group_of_unit": group_of_unit,
        "leads": int(leads),
        "n_obs": int(n_obs),
        "n_clusters": n_cl,
        "time_effects": time_effects,
        "controls": list(design.control_names),
        "method": (
            "pre-trend regression on untreated cells with unit and "
            f"{'common' if time_effects == 'common' else 'group-specific'} "
            "period effects and group x lead indicators; CR1 cluster-robust "
            "Wald tests"
        ),
    }


# --------------------------------------------------------------------------- #
#  Honest evaluation of a targeting rule
# --------------------------------------------------------------------------- #

# Below this many units (or dyadic members) a side, the split is measuring
# itself. Not a theorem: the Monte Carlo behind sp.rate_split had 75 units a
# side and behaved; the 15-country trade panel of the guide has 7 and does
# not. Warn, do not refuse -- a user who wants the number can have it.
_THIN_HALF = 30

_CTOR_PARAMS = (
    "n_estimators",
    "min_samples_leaf",
    "max_depth",
    "max_samples",
    "model_y",
    "model_t",
    "discrete_treatment",
    "honest",
    "bootstrap",
    "random_state",
    "n_jobs",
    "verbose",
    "split_rule",
    "mtry",
    "honesty_fraction",
    "honesty_prune_leaves",
    "alpha",
    "imbalance_penalty",
    "stabilize_splits",
    "ci_group_size",
    "equalize_cluster_weights",
    "nuisance_folds",
    "fe",
)


def _split_keys(
    forest: Any, members: Any, train_frac: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Row masks for the training and evaluation halves.

    Units (or, for dyadic data, members) are split, never rows: two cells of
    one unit share its fixed effect, and two trade flows of one country
    share its shocks, so a row split would leak the evaluation outcomes into
    the rule being evaluated -- the very thing the split is for.
    """
    n = int(len(forest._Y_original))
    rng = np.random.default_rng(random_state)
    if members is not None:
        i_code, j_code, _ = fi.dyad_codes(members, n)
        nodes = np.unique(np.concatenate([i_code, j_code]))
        train_nodes = rng.choice(
            nodes,
            size=max(1, int(round(train_frac * nodes.size))),
            replace=False,
        )
        i_train, j_train = np.isin(i_code, train_nodes), np.isin(j_code, train_nodes)
        return i_train & j_train, ~i_train & ~j_train, "members"
    keys = getattr(forest, "_clusters", None)
    if keys is None:
        keys = getattr(forest, "_fe_unit", None)
    if keys is None:
        keys = np.arange(n)
        label = "rows"
    else:
        label = "units"
    codes = pd.factorize(np.asarray(keys).ravel())[0]
    groups = np.unique(codes)
    train_groups = rng.choice(
        groups, size=max(1, int(round(train_frac * groups.size))), replace=False
    )
    in_train = np.isin(codes, train_groups)
    return in_train, ~in_train, label


@dataclass
class _Halves:
    """Two forests fitted on disjoint units, and the masks that made them."""

    train: Any
    evaluate: Any
    in_train: np.ndarray
    in_eval: np.ndarray
    split_by: str
    dropped: int
    n_rows: int

    def n_groups(self, mask: np.ndarray, forest: Any, members: Any) -> int:
        """However many of the thing that was split, not of rows."""
        if self.split_by == "members":
            i_code, j_code, _ = fi.dyad_codes(members, self.n_rows)
            return int(np.unique(np.concatenate([i_code[mask], j_code[mask]])).size)
        keys = getattr(forest, "_clusters", None)
        if keys is None:
            keys = getattr(forest, "_fe_unit", None)
        if keys is None:
            return int(mask.sum())
        return int(np.unique(np.asarray(keys)[mask]).size)


def _vein(
    per_split: List[Dict[str, float]], alpha: float, n_splits: int
) -> Dict[str, Any]:
    r"""Aggregate split-conditional results the way [chernozhukov2025generic] do.

    One split is not an estimator, it is a draw. Their variational
    estimation and inference methods (VEIN) account for the splitting
    uncertainty on top of the conditional uncertainty: the point estimate
    is the **median** over splits, the interval is the **median of the
    conditional intervals**, and the p-value is the **median of the
    conditional p-values, doubled**. A median of intervals built at level
    ``1 - a`` covers at ``1 - 2a``, so to deliver the ``1 - alpha`` the
    caller asked for, the conditional intervals are built at
    ``1 - alpha / 2`` -- which the callers of this helper do.

    Their warning is the reason this is the default rather than an option:
    with a single split "empiricists may unintentionally look for a 'good'
    data split, which supports their prior beliefs about the likely
    results, thereby invalidating inference".
    """
    est = np.array([r["estimate"] for r in per_split], dtype=float)
    lo = np.array([r["ci_low"] for r in per_split], dtype=float)
    hi = np.array([r["ci_high"] for r in per_split], dtype=float)
    se = np.array([r["se"] for r in per_split], dtype=float)
    pv = np.array([r["p"] for r in per_split], dtype=float)
    if n_splits == 1:
        out = dict(per_split[0])
        out.update(
            n_splits=1,
            aggregation="single split (conditional inference, not VEIN)",
            estimate_min=float(est[0]),
            estimate_max=float(est[0]),
        )
        return out
    point = float(np.median(est))
    # A split whose variance was not estimable (the dyadic estimator can go
    # non-positive) still has a usable point estimate; it just contributes
    # no interval. Taking a plain median would let one of them turn the
    # whole interval into NaN.
    n_no_interval = int(np.sum(~np.isfinite(lo) | ~np.isfinite(hi)))
    with np.errstate(invalid="ignore"):
        pooled_p = float(min(1.0, 2.0 * np.nanmedian(pv)))
        med_se = float(np.nanmedian(se)) if np.any(np.isfinite(se)) else float("nan")
        med_lo = float(np.nanmedian(lo)) if np.any(np.isfinite(lo)) else float("nan")
        med_hi = float(np.nanmedian(hi)) if np.any(np.isfinite(hi)) else float("nan")
    return {
        "estimate": point,
        "se": med_se,
        "ci_low": med_lo,
        "ci_high": med_hi,
        "z": float(point / med_se) if med_se > 0 else float("nan"),
        "p": pooled_p,
        "n_splits": int(n_splits),
        "n_splits_without_interval": n_no_interval,
        "aggregation": (
            "VEIN: median over splits; the interval is the median of "
            f"{100 * (1 - alpha / 2):.4g}% conditional intervals and covers "
            f"at {100 * (1 - alpha):.4g}%; p is twice the median conditional p"
        ),
        "estimate_min": float(est.min()),
        "estimate_max": float(est.max()),
        "estimate_iqr": float(np.percentile(est, 75) - np.percentile(est, 25)),
    }


def _warn_single_split(context: str) -> None:
    warnings.warn(
        f"{context}: n_splits=1 reports one draw, not an estimate. The split "
        "moves the answer -- on the guide's 15-country trade panel six splits "
        "ranged over +/-0.08 -- and with random_state in reach it is easy to "
        "keep the one that agrees with you, which is exactly the practice "
        "Chernozhukov et al. (2025) show invalidates inference. Use the "
        "default n_splits=21, or 100 as they do.",
        AssumptionWarning,
        stacklevel=3,
    )


def _run_splits(
    run_one: Any, n_splits: int, context: str
) -> Tuple[List[Dict[str, Any]], int]:
    """Run every split, tolerating the ones the panel cannot support.

    A split that leaves the evaluation half with no imputable treated cell
    is not a draw from the estimator's distribution, it is an inadmissible
    partition -- and on a small panel at least one of twenty-one will be.
    Failing the whole call on it would make ``n_splits`` a liability
    exactly where the split matters most, so those are skipped and counted;
    the estimand becomes "over admissible partitions", which the warning
    says. Half of them failing, or fewer than three surviving, means the
    panel is too small to split at all, and that is an error.
    """
    runs: List[Dict[str, Any]] = []
    failures: List[str] = []
    for b in range(n_splits):
        try:
            runs.append(run_one(b))
        except (DataInsufficient, NumericalInstability) as exc:
            failures.append(str(exc).splitlines()[0])
    n_failed = len(failures)
    # Enough must survive to take a median of, but "enough" is relative to
    # what was asked for: n_splits=2 must not trip a floor of three.
    needed = min(3, n_splits)
    if len(runs) < needed or (n_splits > 1 and n_failed * 2 > n_splits):
        raise DataInsufficient(
            f"{context}: only {len(runs)} of {n_splits} splits could be "
            "evaluated, so the panel is too small to split at all. First "
            f"reason: {failures[0] if failures else 'unknown'}",
            recovery_hint=(
                "Bring a rule fitted outside this sample and pass it to "
                "sp.rate(..., priorities=), or report sp.rate as a "
                "diagnostic and say so."
            ),
            diagnostics={"n_splits": n_splits, "n_failed": n_failed},
        )
    if n_failed:
        warnings.warn(
            f"{context}: {n_failed} of {n_splits} splits left an evaluation "
            "half with nothing to impute and were skipped, so the estimate "
            "is a median over the admissible partitions rather than over all "
            f"of them. First reason: {failures[0]}",
            AssumptionWarning,
            stacklevel=3,
        )
    return runs, n_failed


def _refit_halves(
    forest: Any,
    members: Any,
    train_frac: float,
    random_state: int,
    context: str,
) -> _Halves:
    """Split the units and refit the same forest on each half.

    Shared by :func:`rate_split` and :func:`forest_policy_tree`, which need
    the same thing for the same reason: a rule read off the scores it is
    then graded on is graded on its own noise.
    """
    from .causal_forest import CausalForest

    _require_grf(forest, context)
    if not 0.0 < float(train_frac) < 1.0:
        raise MethodIncompatibility(
            f"{context}: train_frac must be strictly between 0 and 1.",
            recovery_hint="Use train_frac=0.5.",
            diagnostics={"train_frac": train_frac},
        )
    in_train, in_eval, split_by = _split_keys(
        forest, members, float(train_frac), int(random_state)
    )
    n_rows = int(len(forest._Y_original))
    dropped = int(n_rows - in_train.sum() - in_eval.sum())
    params = {p: getattr(forest, p) for p in _CTOR_PARAMS}
    # The constructor fills model_y / model_t with scikit-learn defaults that
    # the GRF path never uses; handing those back would make them count as
    # user models and switch the halves to cross-fitted scikit-learn
    # nuisances (class labels as W_hat under fe=). Only models the caller
    # actually supplied are carried.
    if not getattr(forest, "_user_model_y", False):
        params["model_y"] = None
    if not getattr(forest, "_user_model_t", False):
        params["model_t"] = None
    Y = np.asarray(forest._Y_original, dtype=float)
    T = np.asarray(forest._T_original, dtype=float)
    X = np.asarray(forest._X_original, dtype=float)
    # Controls of the forest being split (they enter the nuisance models and,
    # under fe=, the imputation design).
    controls = getattr(forest, "_W_original", None)
    if controls is None:
        controls = getattr(forest, "_fe_W", None)
    if (
        controls is None
        and int(getattr(forest, "data_info", {}).get("n_controls", 0)) > 0
    ):
        raise MethodIncompatibility(
            f"{context}: this forest was fitted with controls but does not "
            "carry them (it was fitted by an older release), so the halves "
            "cannot be refitted the way it was.",
            recovery_hint="Refit the forest with this release and call again.",
        )
    # Nuisances the caller supplied are data, not estimates: each half keeps
    # its own rows of them instead of re-estimating.
    source = getattr(forest, "diagnostics", {}).get("nuisance_source", {}) or {}
    supplied = {
        name: np.asarray(getattr(forest, attr), dtype=float)
        for name, attr in (("Y_hat", "_m_insample"), ("W_hat", "_e_insample"))
        if source.get(name) == "user-supplied"
    }
    unit = getattr(forest, "_fe_unit", None)
    time = getattr(forest, "_fe_time", None)
    clusters = getattr(forest, "_clusters", None)

    def _fit(mask: np.ndarray, side: str) -> Any:
        if int(mask.sum()) < 2 or len(np.unique(T[mask])) < 2:
            raise DataInsufficient(
                f"{context}: the {side} half has no usable variation "
                f"({int(mask.sum())} rows). Lower train_frac, or the panel is "
                "too small to split.",
                recovery_hint=(
                    "Bring a rule fitted outside this sample instead of " "splitting."
                ),
                diagnostics={"side": side, "n_rows": int(mask.sum())},
            )
        sub = CausalForest(**params)
        try:
            sub.fit(
                Y=Y[mask],
                T=T[mask],
                X=X[mask],
                W=None if controls is None else np.asarray(controls)[mask],
                Y_hat=supplied["Y_hat"][mask] if "Y_hat" in supplied else None,
                W_hat=supplied["W_hat"][mask] if "W_hat" in supplied else None,
                clusters=None if clusters is None else np.asarray(clusters)[mask],
                id=None if unit is None else np.asarray(unit)[mask],
                time=None if time is None else np.asarray(time)[mask],
            )
        except (DataInsufficient, MethodIncompatibility) as exc:
            raise type(exc)(
                f"{context}: refitting the forest on the {side} half failed -- "
                f"{exc}",
                recovery_hint=(
                    "Each half has to support a forest of its own. Move rows "
                    "with train_frac, or bring a rule fitted outside this "
                    "sample."
                ),
                diagnostics={"side": side, "n_rows": int(mask.sum())},
            ) from exc
        # Refitting goes through arrays, which would leave the halves with
        # X0, X1, ... and make every diagnostic read in a vocabulary the
        # caller never used.
        names = getattr(forest, "_feature_names", None)
        if names:
            sub._feature_names = list(names)
        return sub

    return _Halves(
        train=_fit(in_train, "training"),
        evaluate=_fit(in_eval, "evaluation"),
        in_train=in_train,
        in_eval=in_eval,
        split_by=split_by,
        dropped=dropped,
        n_rows=n_rows,
    )


def _rate_split_once(
    forest: Any,
    target: str = "AUTOC",
    *,
    train_frac: float = 0.5,
    random_state: int = 0,
    members: Any = None,
    alpha: float = 0.05,
    variance: str = "bjs",
    cluster: Any = None,
    covariates: Any = "none",
    se_method: str = "auto",
    q_grid: int = 100,
    _warn: bool = True,
) -> Dict[str, Any]:
    r"""One split's RATE. :func:`rate_split` aggregates many of these."""

    from .forest_inference import rate as _rate

    context = "rate_split()"
    halves = _refit_halves(forest, members, train_frac, random_state, context)
    train, evaluate = halves.train, halves.evaluate
    in_train, in_eval = halves.in_train, halves.in_eval
    split_by, dropped = halves.split_by, halves.dropped
    X = np.asarray(forest._X_original, dtype=float)
    prio = np.asarray(train.effect(X[in_eval]), dtype=float).ravel()
    kwargs: Dict[str, Any] = dict(
        target=target,
        priorities=prio,
        alpha=alpha,
        q_grid=q_grid,
        se_method=se_method,
        covariates=covariates,
    )
    if gi.is_fe_forest(forest):
        kwargs.update(variance=variance, cluster=cluster)
        if members is not None:
            mem = np.asarray(members)
            kwargs.update(cluster="dyadic", members=mem[in_eval])
    try:
        out = dict(_rate(evaluate, **kwargs))
    except (DataInsufficient, NumericalInstability) as exc:
        raise type(exc)(
            f"{context}: evaluating the rule on the held-out half failed -- " f"{exc}",
            recovery_hint=(
                "The half-sample is thinner than the full one, so overlap or "
                "imputability that held on all the data can fail on it. Raise "
                "train_frac (a smaller evaluation half is not the fix -- lower "
                "it to move rows the other way), or run sp.rate with "
                "priorities= from a rule fitted outside this sample."
            ),
        ) from exc

    n_train = halves.n_groups(in_train, forest, members)
    n_eval = halves.n_groups(in_eval, forest, members)
    if _warn and min(n_train, n_eval) < _THIN_HALF:
        warnings.warn(
            f"rate_split(): the halves hold {n_train} and {n_eval} "
            f"{split_by}. A rule fitted on that many is close to noise, and "
            "the RATE it earns measures the split as much as the "
            "heterogeneity -- on the 15-country trade panel of the guide it "
            f"ranged over {chr(177)}0.08 across six splits, with the "
            "dyadic variance going non-positive on two of them, while the "
            "true-tau ranking on the whole sample gave +0.081 (se 0.031). "
            f"The design this was calibrated on had 75 units a side. Below "
            f"{_THIN_HALF} report sp.rate as a diagnostic, or bring an "
            "external rule to sp.rate(..., priorities=), and do not read a "
            "single split as a test.",
            AssumptionWarning,
            stacklevel=2,
        )
    out.update(
        priority_source="held_out_forest",
        n_train_units=n_train,
        n_eval_units=n_eval,
        n_rows_dropped=dropped,
        split_by=split_by,
        split_random_state=int(random_state),
        method=out.get("method", "RATE") + ", split-sample",
    )
    return out


@accepts_aliases(_strict=True, controls="covariates")
def rate_split(
    forest: Any,
    target: str = "AUTOC",
    *,
    n_splits: int = 21,
    train_frac: float = 0.5,
    random_state: int = 0,
    members: Any = None,
    alpha: float = 0.05,
    variance: str = "bjs",
    cluster: Any = None,
    covariates: Any = "none",
    se_method: str = "auto",
    q_grid: int = 100,
) -> Dict[str, Any]:
    r"""RATE of the forest's targeting rule, fitted and evaluated on disjoint units.

    :func:`statspai.rate` ranks the rows it also scores. For a forest with
    fixed effects that is not a valid test even though the predictions are
    out-of-bag: every imputation score carries :math:`-\hat\gamma_t`,
    estimated from the same periods the forest was trained on, so the
    ranking and the scores stay correlated. Measured on a design with **no
    heterogeneity whatsoever** (200 replications, N = 150 units, T = 8,
    staggered adoption selected on the unit effect, ``tau = 0.3``): AUTOC
    averaged **-0.025** instead of 0 and a nominal 5% test rejected
    **17.5%** of the time; QINI averaged -0.006 and rejected 13.5%.

    This function removes the overlap. The units (or, with ``members=``, the
    nodes of a dyadic panel) are split in two; a forest with the same
    hyper-parameters is refitted on each half; the training half's forest
    ranks the evaluation half's treated cells, whose imputation scores come
    from an untreated two-way model fitted on the evaluation half alone.
    Nothing the rule saw enters the score it is graded on. On the same null
    design AUTOC then averaged **+0.0008** and rejected **7.5%** of the time
    (QINI +0.0004 and 4.0%); against ``tau = 0.3 + 0.5 z`` it kept 99.5% and
    100% power. ``variance='forest'`` rejected 5.0% and 4.0% under the null
    but understates the spread under the alternative (mean standard error
    0.060 against a Monte Carlo 0.083), which is why ``'bjs'`` is the
    default here.

    The price is sample: each forest sees half the units, so the rule is
    noisier than the one fitted on everything and the RATE it earns is a
    *lower bound* on what the full-sample rule is worth. Report
    :func:`statspai.rate` for the point estimate of the full rule if you
    like, but report this one when the claim is that targeting pays.

    Parameters
    ----------
    forest : fitted CausalForest
        Supplies the data and the hyper-parameters; it is not itself used to
        rank or to score, and is left untouched.
    target : {'AUTOC', 'QINI'}
    train_frac : float
        Share of units (members) that fit the rule. The rest evaluate it.
    random_state : int
        Seed for the split. The refitted forests keep the original's seed.
    members : array-like, optional
        The two members of each dyadic row. Splitting is then by member and
        rows that straddle the two halves are dropped (counted in the
        result), because a flow between a training and an evaluation country
        belongs to neither.
    alpha, variance, cluster, covariates, se_method, q_grid
        Passed to :func:`statspai.rate` on the evaluation half.

    Returns
    -------
    dict
        As :func:`statspai.rate`, plus ``n_train_units`` / ``n_eval_units``,
        ``n_rows_dropped``, ``split_by`` and ``split_random_state``.

    See Also
    --------
    statspai.rate : the same curve without the split (a diagnostic).
    statspai.forest_group_effects : effects for named groups rather than a curve.

    References
    ----------
    [@yadlowsky2025evaluating]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(80):
    ...     a, z = rng.normal(), rng.normal()
    ...     g = 4 if i % 3 else 10**6
    ...     for t in range(1, 7):
    ...         d = 1.0 * (t >= g)
    ...         y = a + 0.2 * t + (0.3 + 0.5 * z) * d + rng.normal(0, 0.5)
    ...         rows.append((i, t, y, d, z))
    >>> df = pd.DataFrame(rows, columns=["id", "t", "y", "d", "z"])
    >>> cf = sp.causal_forest(
    ...     "y ~ d | z", data=df, fe="twoway", unit="id", time="t",
    ...     clusters=df["id"].values, n_estimators=100, random_state=0,
    ... )
    >>> res = sp.rate_split(cf, target="AUTOC")
    >>> res["split_by"], res["n_eval_units"] > 0
    ('units', True)


    **Many splits, not one.** A single split is a draw, not an estimator:
    on the guide's 15-country trade panel six of them ranged over +/-0.08,
    and with ``random_state`` in reach it is easy to keep the one that
    agrees with you -- the practice [chernozhukov2025generic] show
    invalidates inference. Their variational estimation and inference
    (VEIN) is what ``n_splits`` does: the estimate is the median over
    splits, the interval is the median of the conditional intervals built
    at ``1 - alpha / 2`` (whose median covers at ``1 - alpha``), and the
    p-value is twice the median conditional p-value. ``n_splits=21`` costs
    about twelve seconds on a 150-unit panel; 100, as they use, costs about
    a minute. ``n_splits=1`` reproduces a single conditional split and
    warns. ``estimate_min`` / ``estimate_max`` / ``estimate_iqr`` in the
    result show how much the split was moving the answer.

    Parameters
    ----------
    forest, target, train_frac, members, alpha, variance, cluster,
    covariates, se_method, q_grid
        As described above.
    n_splits : int, default 21
        Splits to aggregate by VEIN. Odd, so the median is an order
        statistic.
    random_state : int, default 0
        Seeds the *sequence* of splits; split ``b`` uses
        ``random_state + b``.
    """
    context = "rate_split()"
    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, (int, np.integer))
        or n_splits < 1
    ):
        raise MethodIncompatibility(
            f"{context}: n_splits must be a positive integer.",
            recovery_hint="Use n_splits=21 (the default) or 100.",
            diagnostics={"n_splits": n_splits},
        )
    n_splits = int(n_splits)
    if n_splits == 1:
        _warn_single_split(context)
    conditional_alpha = float(alpha) if n_splits == 1 else float(alpha) / 2.0
    runs, n_failed = _run_splits(
        lambda b: _rate_split_once(
            forest,
            target,
            train_frac=train_frac,
            random_state=int(random_state) + b,
            members=members,
            alpha=conditional_alpha,
            variance=variance,
            cluster=cluster,
            covariates=covariates,
            se_method=se_method,
            q_grid=q_grid,
            _warn=(b == 0),
        ),
        n_splits,
        context,
    )

    def _conditional(r: Dict[str, Any]) -> Dict[str, float]:
        # sp.rate reports no p-value; derive the conditional one so the
        # median-and-double rule has something to work on.
        se = float(r["se"])
        z = float(r["estimate"]) / se if se > 0 else float("nan")
        return {
            "estimate": float(r["estimate"]),
            "se": se,
            "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"]),
            "p": float(2 * stats.norm.sf(abs(z))) if se > 0 else float("nan"),
        }

    pooled = _vein([_conditional(r) for r in runs], float(alpha), len(runs))
    out = dict(runs[0])
    out.update(pooled)
    out["alpha"] = float(alpha)
    out["toc_curve"] = np.column_stack(
        [
            runs[0]["toc_curve"][:, 0],
            np.median(np.stack([r["toc_curve"][:, 1] for r in runs]), axis=0),
        ]
    )
    out["method"] = runs[0]["method"] + (
        f", VEIN over {len(runs)} splits" if n_splits > 1 else ""
    )
    out["split_random_state"] = int(random_state)
    out["n_splits_skipped"] = n_failed
    return out


# --------------------------------------------------------------------------- #
#  Policy learning on the cells the design identifies
# --------------------------------------------------------------------------- #


def _policy_functional(
    forest: Any,
    policy: np.ndarray,
    cost: float,
    variance: str,
    cluster: Any,
    members: Any,
    covariates: Any,
    alpha: float,
    context: str,
) -> Dict[str, Any]:
    r"""Value of a given policy on one forest's treated cells, with its SE.

    ``V(pi) = mean over treated cells of (tau - cost) * pi``. Both that and
    the gain over treating every cell, ``V(pi) - V(1)``, are linear in
    ``y`` once ``pi`` is fixed, so each gets the exact, cluster- or
    dyad-robust variance the ATT gets, from one design and one call.

    The gain's *estimate* is exactly the difference of the other two. Its
    *standard error* is not the difference of their variances, and should
    not be: the BJS centring subtracts a cohort x event-time mean weighted
    by that functional's own ``v^2``, so the gain -- whose weights vanish
    on every cell the rule treats -- is centred on the withheld cells
    alone. Measured against the true value of the fitted rule, at two
    operating points, 150 replications each:

    ========================  ==========  =============  ==============
    share of cells treated    MC error    dedicated      ``c'Vc``
    ========================  ==========  =============  ==============
    0.50                      0.047       0.063 / 98.7%  0.096 / 99.3%
    0.83                      0.025       0.025 / 94.0%  0.065 / 100%
    ========================  ==========  =============  ==============

    The dedicated column is exact where the rule withholds little and
    conservative where it withholds a lot; it never undercovered, and the
    worry that centring on few withheld cells would deflate it did not
    materialise. It is what ``gain_over_treat_all['se']`` reports.
    ``diagnostics['gain_se_contrast']`` carries the ``c'Vc`` reading, so a
    caller who rebuilds it from ``vcov`` can see why the two differ instead
    of finding a discrepancy.
    """
    design = fi.imputation_design(forest, context, covariates)
    fi._require_target_oob(forest, design, context, needed=variance == "forest")
    rows = np.flatnonzero(design.target)
    m = rows.size
    pi = np.asarray(policy, dtype=float).ravel()
    if pi.size != m:
        raise MethodIncompatibility(
            f"{context}: the policy must give one action per treated cell "
            f"({m}), got {pi.size}.",
            recovery_hint="Predict the policy on the evaluation half's cells.",
        )
    if not np.all(np.isin(pi, (0.0, 1.0))):
        raise MethodIncompatibility(
            f"{context}: the policy must be 0/1.",
            recovery_hint="Threshold a continuous rule before evaluating it.",
        )
    # Three functionals at once so their covariance comes out: the policy's
    # value, treating everyone, and the difference.
    W = np.zeros((design.n, 3))
    W[rows, 0] = pi / m
    W[rows, 1] = 1.0 / m
    W[rows, 2] = (pi - 1.0) / m
    mem = None if members is None else fi.dyad_codes(members, design.n)[:2]
    est, V, label = fi._estimate(forest, design, W, variance, cluster, mem)
    # The cost is known, so it shifts the estimate and not its variance.
    share = float(pi.mean())
    est = np.asarray(est, dtype=float) - cost * np.array([share, 1.0, share - 1.0])
    # A rule that treats every cell makes the gain a functional with
    # identically zero weights: its value is exactly 0 with variance exactly
    # 0, which is not the same thing as a variance that could not be
    # estimated, and must not be reported as NaN.
    degenerate = ~np.any(W != 0.0, axis=0)
    se = np.zeros(3)
    live = np.flatnonzero(~degenerate)
    if live.size:
        se[live] = fi._safe_se(V[np.ix_(live, live)], context)
    est = np.where(degenerate, 0.0, est)
    z = float(stats.norm.ppf(1 - alpha / 2))
    keys = ("policy", "treat_all", "gain_over_treat_all")
    out: Dict[str, Any] = {}
    for k, name in enumerate(keys):
        with np.errstate(divide="ignore", invalid="ignore"):
            zstat = est[k] / se[k] if se[k] > 0 else np.nan
        if degenerate[k]:  # exactly 0, exactly certain: the rule is treat-all
            pval: float = 1.0
        elif se[k] > 0:
            pval = float(2 * stats.norm.sf(abs(zstat)))
        else:
            pval = float("nan")
        out[name] = {
            "estimate": float(est[k]),
            "se": float(se[k]),
            "ci_low": float(est[k] - z * se[k]),
            "ci_high": float(est[k] + z * se[k]),
            "z": float(zstat),
            "p": pval,
        }
    contrast = np.array([1.0, -1.0, 0.0])
    out["_gain_se_contrast"] = float(np.sqrt(max(contrast @ V @ contrast, 0.0)))
    out["_vcov"] = V
    out["_label"] = label
    out["_n_cells"] = int(m)
    out["_share_treated"] = share
    out["_design"] = design
    out["_rows"] = rows
    return out


def _forest_policy_tree_once(
    forest: Any,
    *,
    depth: int = 2,
    cost: float = 0.0,
    x: Any = None,
    min_leaf_size: Optional[int] = None,
    train_frac: float = 0.5,
    random_state: int = 0,
    members: Any = None,
    alpha: float = 0.05,
    variance: str = "bjs",
    cluster: Any = None,
    covariates: Any = "none",
    _warn: bool = True,
) -> Dict[str, Any]:
    r"""One split's rule and price. :func:`forest_policy_tree` aggregates many."""

    from ..policy_learning._exact_tree import exact_policy_tree
    from ..policy_learning.policy_tree import PolicyTree

    context = "forest_policy_tree()"
    _require_grf(forest, context)
    if not gi.is_fe_forest(forest):
        raise MethodIncompatibility(
            f"{context} is for forests with fixed effects, whose scores are "
            "imputation scores on treated cells. A pooled forest has a "
            "propensity and a population-wide estimand, so it gets the "
            "doubly-robust policy tree instead.",
            recovery_hint="Use sp.policy_tree(data, y, d, X, ...).",
            alternative_functions=["sp.policy_tree"],
        )
    if not isinstance(depth, (int, np.integer)) or isinstance(depth, bool) or depth < 1:
        raise MethodIncompatibility(
            f"{context}: depth must be a positive integer.",
            recovery_hint="Use depth=2 (exactly searched) or depth=1.",
            diagnostics={"depth": depth},
        )
    halves = _refit_halves(forest, members, train_frac, random_state, context)

    names = list(getattr(forest, "_feature_names", []) or [])
    x_all = np.asarray(forest._X_original, dtype=float)
    if x is None:
        cols = list(range(x_all.shape[1]))
    elif all(isinstance(c, str) for c in np.atleast_1d(x)):
        wanted = list(np.atleast_1d(x))
        missing = [c for c in wanted if c not in names]
        if missing:
            raise MethodIncompatibility(
                f"{context}: effect modifier(s) {missing} are not in the "
                f"forest's features {names}.",
                recovery_hint="Pass names the forest was fitted on, or x=None.",
            )
        cols = [names.index(c) for c in wanted]
    else:
        supplied = np.asarray(x, dtype=float)
        if supplied.shape[0] != x_all.shape[0]:
            raise MethodIncompatibility(
                f"{context}: x must have one row per training row "
                f"({x_all.shape[0]}), got {supplied.shape[0]}.",
                recovery_hint="Pass column names instead of an array.",
            )
        x_all = supplied
        cols = list(range(x_all.shape[1]))
        names = [f"x{k}" for k in cols]
    policy_names = [names[c] if c < len(names) else f"x{c}" for c in cols]

    common = dict(
        cost=float(cost),
        variance=variance,
        cluster=cluster,
        covariates=covariates,
        alpha=float(alpha),
        context=context,
    )
    mem_train = None if members is None else np.asarray(members)[halves.in_train]
    mem_eval = None if members is None else np.asarray(members)[halves.in_eval]

    # Fit on the training half's scores.
    design_tr = fi.imputation_design(halves.train, context, covariates)
    rows_tr = np.flatnonzero(design_tr.target)
    scores_tr = design_tr.gamma[rows_tr] - float(cost)
    x_tr = x_all[halves.in_train][:, cols][rows_tr]
    design_ev = fi.imputation_design(halves.evaluate, context, covariates)
    rows_ev = np.flatnonzero(design_ev.target)
    x_ev = x_all[halves.in_eval][:, cols][rows_ev]
    leaf = (
        int(min_leaf_size)
        if min_leaf_size is not None
        else max(10, int(0.05 * rows_ev.size))
    )
    if rows_tr.size < 2 * leaf or rows_ev.size < 2:
        raise DataInsufficient(
            f"{context}: {rows_tr.size} treated cell(s) to fit the rule and "
            f"{rows_ev.size} to price it is not enough for a leaf of {leaf}.",
            recovery_hint="Lower min_leaf_size or depth, or widen the panel.",
            diagnostics={"n_fit": int(rows_tr.size), "n_price": int(rows_ev.size)},
        )
    # PolicyTree owns the tree grower, the predictor and the rule printer;
    # reuse them rather than growing a second implementation of a policy
    # tree in the forest module.
    helper = PolicyTree.__new__(PolicyTree)  # its __init__ wants a dataset
    helper.max_depth = int(depth)
    helper.min_leaf_size = leaf
    helper.split_step = 1
    if int(depth) <= 2:
        tree = exact_policy_tree(
            x_tr, scores_tr, max_depth=int(depth), min_leaf_size=leaf
        )
        method = f"exact search, depth {int(depth)}"
    else:
        tree = helper._grow_tree(x_tr, scores_tr, depth=0)  # type: ignore[attr-defined]
        method = f"greedy search, depth {int(depth)}"

    policy = helper._predict_tree(tree, x_ev)
    priced = _policy_functional(
        halves.evaluate, policy, members=mem_eval, **common  # type: ignore[arg-type]
    )
    rules = helper._tree_to_rules(tree, policy_names)
    n_train = halves.n_groups(halves.in_train, forest, members)
    n_eval = halves.n_groups(halves.in_eval, forest, members)
    if _warn and min(n_train, n_eval) < _THIN_HALF:
        warnings.warn(
            f"{context}: the halves hold {n_train} and {n_eval} "
            f"{halves.split_by}. A rule fitted on that many is close to "
            "noise, and the value it earns measures the split as much as the "
            "heterogeneity. Report it as exploratory.",
            AssumptionWarning,
            stacklevel=2,
        )
    del mem_train
    return {
        "rules": rules,
        "tree": tree,
        "policy": np.asarray(policy, dtype=int),
        "policy_covariates": policy_names,
        "value": priced["policy"],
        "value_treat_all": priced["treat_all"],
        "gain_over_treat_all": priced["gain_over_treat_all"],
        "share_treated": priced["_share_treated"],
        "cost": float(cost),
        "n_cells_priced": priced["_n_cells"],
        "n_cells_fitted": int(rows_tr.size),
        "n_train_units": n_train,
        "n_eval_units": n_eval,
        "n_rows_dropped": halves.dropped,
        "split_by": halves.split_by,
        "split_random_state": int(random_state),
        "method": method,
        "estimand": (
            "retrospective: value per treated cell of a rule applied to the "
            "cells that were treated; effects on untreated cells are not "
            "identified"
        ),
        "alpha": float(alpha),
        "diagnostics": {
            "variance": variance,
            "vcov": priced["_vcov"],
            # The same gain read as a contrast of the first two functionals.
            # It differs because the BJS centring is weight-dependent; see
            # the note in the docstring before treating either as the other's
            # mistake.
            "gain_se_contrast": priced["_gain_se_contrast"],
            "se_label": priced["_label"],
            "min_leaf_size": leaf,
            "imputation_covariates": list(design_ev.control_names),
        },
    }


@accepts_aliases(_strict=True, controls="covariates")
def forest_policy_tree(
    forest: Any,
    *,
    depth: int = 2,
    cost: float = 0.0,
    x: Any = None,
    min_leaf_size: Optional[int] = None,
    n_splits: int = 21,
    train_frac: float = 0.5,
    random_state: int = 0,
    members: Any = None,
    alpha: float = 0.05,
    variance: str = "bjs",
    cluster: Any = None,
    covariates: Any = "none",
) -> Dict[str, Any]:
    r"""A treatment rule for a fixed-effects forest, and what it was worth.

    Policy learning [athey2021policy] maximises
    :math:`V(\pi) = E[\Gamma_i \pi(X_i)]` over a class of rules, with
    :math:`\Gamma` a score unbiased for the individual effect. A
    within-unit design has no propensity, so the doubly-robust score
    :func:`statspai.policy_tree` uses does not exist; the imputation scores
    that give the ATT do, and the rule is learned on those.

    **Read the population it applies to.** Those scores live on treated
    cells, so this is a *retrospective* rule: "of the cells that were
    treated, which should have been". It does not say whether an untreated
    unit should be treated -- that is an extrapolation of ``tau(x)``, which
    :func:`statspai.average_treatment_effect` also refuses for these
    forests. Check :func:`statspai.forest_support` before carrying the rule
    to units the design never switched.

    With ``cost=0`` and effects that are positive everywhere, treating
    every cell is optimal and the tree has nothing to find. ``cost`` is
    what usually makes targeting a real question: the rule then treats
    where ``tau > cost``.

    **The rule is fitted and priced on disjoint units.** A tree chosen to
    maximise the value of the very scores it is then priced against is
    priced on its own noise. Measured on a design where *every rule is
    worth exactly the same* -- no heterogeneity, and ``cost`` equal to the
    constant effect, so the true gain over treating everyone is exactly 0
    (200 replications, N = 200 units, T = 8) -- the same-sample gain
    averaged **+0.048**, larger than its own standard error of 0.037, and
    **13.0%** of runs reported a significant benefit from targeting where
    none existed. Fitted and priced on disjoint halves: **+0.005** and
    **3.5%**. The split costs almost nothing when the heterogeneity is
    real: with ``tau = 0.3 + 0.8 z`` and ``cost = 0.3`` the oracle gain is
    ``0.8 * phi(0) = 0.3191``, and the split-sample estimate averaged
    **0.3189** at 99.5% power, against a same-sample **0.3316** that
    overshoots by 0.012. The split is therefore not optional here;
    ``train_frac`` moves it, nothing switches it off.

    Parameters
    ----------
    forest : fitted CausalForest with ``fe="twoway"``
        Supplies the data and the hyper-parameters; it is not itself used
        to fit or to price the rule, and is left untouched.
    depth : int, default 2
        Tree depth. ``<= 2`` is searched exactly (the ``policytree``
        guarantee), deeper is greedy and says so in ``method``.
    cost : float, default 0
        Cost of treating one cell, in the outcome's units. The rule treats
        where the effect exceeds it.
    x : list of str or array, optional
        Policy covariates. Defaults to the forest's effect modifiers.
        Keep this list short and interpretable -- it is the rule a
        programme would actually be written in.
    min_leaf_size : int, optional
        Smallest leaf; default ``max(10, 5% of the evaluation cells)``.
    train_frac : float, default 0.5
        Share of units (or, with ``members=``, dyadic nodes) that fit the
        rule. The rest price it.
    random_state : int, default 0
    members : array-like, optional
        The two members of each dyadic row; splitting is then by member.
    alpha, variance, cluster, covariates
        As :func:`statspai.forest_group_effects`. ``variance`` defaults to
        ``'bjs'`` here for the reason it does in :func:`statspai.rate`.

    Returns
    -------
    dict
        ``rules`` (the tree, printable), ``tree`` (nested dict),
        ``policy`` (0/1 per evaluation cell), ``value`` and
        ``value_treat_all`` and ``gain_over_treat_all`` (each with
        ``estimate``, ``se``, ``ci_low``, ``ci_high``, ``p``),
        ``share_treated``, ``n_train_units`` / ``n_eval_units``,
        ``split_by``, ``method``, ``diagnostics``.

    See Also
    --------
    statspai.policy_tree : the doubly-robust version, for designs with a
        propensity.
    statspai.rate_split : is *any* ranking worth targeting on?
    statspai.forest_support : is a rule safe to carry to untreated units?

    References
    ----------
    [@athey2021policy], [@borusyak2024revisiting], [@kattenberg2023causal]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(120):
    ...     a, z = rng.normal(), rng.normal()
    ...     g = 4 if i % 3 else 10**6
    ...     for t in range(1, 8):
    ...         d = 1.0 * (t >= g)
    ...         y = a + 0.2 * t + (0.3 + 0.8 * z) * d + rng.normal(0, 0.5)
    ...         rows.append((i, t, y, d, z))
    >>> df = pd.DataFrame(rows, columns=["id", "t", "y", "d", "z"])
    >>> cf = sp.causal_forest(
    ...     "y ~ d | z", data=df, fe="twoway", unit="id", time="t",
    ...     clusters=df["id"].to_numpy(), n_estimators=100, random_state=0,
    ... )
    >>> res = sp.forest_policy_tree(cf, depth=1, cost=0.3, n_splits=3)
    >>> 0.0 <= res["share_treated"] <= 1.0
    True
    >>> gain = res["gain_over_treat_all"]
    >>> sorted(k for k in gain if k in {"estimate", "se", "ci_low", "ci_high"})
    ['ci_high', 'ci_low', 'estimate', 'se']
    >>> gain["n_splits"], res["n_splits"]
    (3, 3)
    >>> sorted(res["diagnostics"]["split_stability"])[:2]
    ['root_covariate_counts', 'root_covariate_modal_share']


    **Many splits, not one.** One split is a draw: it fixes both the rule
    and its price, and with ``random_state`` in reach it is easy to keep
    the pair that agrees with you -- the practice
    [chernozhukov2025generic] show invalidates inference. ``n_splits``
    aggregates by their variational estimation and inference (VEIN): the
    value, the treat-all value and the gain are each the median over
    splits, their intervals are medians of conditional intervals built at
    ``1 - alpha / 2`` (whose median covers at ``1 - alpha``), and the
    p-values are twice the median conditional p-value. Each is aggregated
    on its own, as [chernozhukov2025generic] do, so with ``n_splits > 1``
    the three medians **do not** satisfy
    ``gain = value - value_treat_all`` -- a median of differences is not a
    difference of medians. That identity holds within each split, and
    ``n_splits=1`` reports it directly. A *rule* cannot be
    averaged either, so the one reported is the rule from the split whose
    gain is the median, and ``diagnostics['split_stability']`` says how often each
    covariate was chosen at the root and how far the thresholds and treated
    shares moved -- if the root covariate changes from split to split, the
    rule is not identified by this sample however tight the interval on its
    value looks. ``n_splits=21`` costs about twelve seconds on a 150-unit
    panel; ``n_splits=1`` reproduces one conditional split and warns.

    Parameters
    ----------
    n_splits : int, default 21
        Splits to aggregate by VEIN. Odd, so the median is an order
        statistic.
    random_state : int, default 0
        Seeds the *sequence* of splits; split ``b`` uses
        ``random_state + b``.
    """
    context = "forest_policy_tree()"
    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, (int, np.integer))
        or n_splits < 1
    ):
        raise MethodIncompatibility(
            f"{context}: n_splits must be a positive integer.",
            recovery_hint="Use n_splits=21 (the default) or 100.",
            diagnostics={"n_splits": n_splits},
        )
    n_splits = int(n_splits)
    if n_splits == 1:
        _warn_single_split(context)
    conditional_alpha = float(alpha) if n_splits == 1 else float(alpha) / 2.0
    runs, n_failed = _run_splits(
        lambda b: _forest_policy_tree_once(
            forest,
            depth=depth,
            cost=cost,
            x=x,
            min_leaf_size=min_leaf_size,
            train_frac=train_frac,
            random_state=int(random_state) + b,
            members=members,
            alpha=conditional_alpha,
            variance=variance,
            cluster=cluster,
            covariates=covariates,
            _warn=(b == 0),
        ),
        n_splits,
        context,
    )
    gains = np.array([r["gain_over_treat_all"]["estimate"] for r in runs])
    # A rule is not a number, so the reported tree is the one from the split
    # whose gain is the median -- a representative draw, not an average.
    representative = int(np.argsort(gains)[len(gains) // 2])
    out = dict(runs[representative])
    for key in ("value", "value_treat_all", "gain_over_treat_all"):
        out[key] = _vein([r[key] for r in runs], float(alpha), len(runs))
    roots = [
        (
            r["policy_covariates"][r["tree"]["feature"]]
            if r["tree"].get("type") == "split"
            else "(no split)"
        )
        for r in runs
    ]
    thresholds = [
        float(r["tree"]["threshold"]) if r["tree"].get("type") == "split" else np.nan
        for r in runs
    ]
    shares = np.array([r["share_treated"] for r in runs], dtype=float)
    counts: Dict[str, int] = {}
    for name in roots:
        counts[name] = counts.get(name, 0) + 1
    out["n_splits"] = len(runs)
    out["n_splits_skipped"] = n_failed
    out["alpha"] = float(alpha)
    out["split_random_state"] = int(random_state)
    out["representative_split"] = representative
    out["method"] = runs[representative]["method"] + (
        f", VEIN over {len(runs)} splits" if n_splits > 1 else ""
    )
    out["diagnostics"] = dict(runs[representative]["diagnostics"])
    out["diagnostics"]["split_stability"] = {
        "root_covariate_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "root_covariate_modal_share": float(max(counts.values()) / len(runs)),
        "threshold_min": float(np.nanmin(thresholds)) if runs else float("nan"),
        "threshold_max": float(np.nanmax(thresholds)) if runs else float("nan"),
        "share_treated_min": float(shares.min()),
        "share_treated_max": float(shares.max()),
    }
    return out
