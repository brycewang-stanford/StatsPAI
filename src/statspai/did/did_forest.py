"""Difference-in-differences causal forests with clean group-time comparisons.

``sp.did_forest`` estimates heterogeneous effects in a staggered-adoption
panel without the "forbidden comparisons" of a pooled forest.  For every
treatment cohort ``g`` and period ``t`` it builds one cross-section of units:

* treated: units first treated in ``g``;
* comparison: never-treated units, plus (``control_group="notyettreated"``)
  units not yet treated by ``max(t, base)``;
* outcome: the long difference ``Y_t - Y_base`` with the universal base
  period ``base`` the last period before ``g`` (net of ``anticipation``);
* covariates: time-invariant unit characteristics.

Under conditional parallel trends the conditional ATT ``tau_{g,t}(x)`` is
the conditional mean difference of that long difference between treated and
comparison units, so a causal forest on the block estimates it directly --
the difference-in-difference causal forest of Gavrilova, Langorgen and
Zoutman (2025), which runs one forest per post-period on ``Y_t - Y_base``,
applied to each Callaway-Sant'Anna group-time comparison.  On a two-period
block, removing unit and period effects is algebraically the same long
difference, so this is also the block construction of fixed-effects causal
forests for staggered adoption (Aytug 2026).

Aggregates
----------
``ATT(g, t)`` is the forest's doubly-robust average effect on the treated
(plug-in mean of the out-of-bag CATE over cohort ``g`` plus the
inverse-propensity-weighted residual correction), i.e. a doubly-robust DiD
with a forest outcome model.  Each cell's estimate has a per-unit influence
function; event-study, cohort and overall averages sum those influence
functions across cells unit by unit (cluster by cluster when ``clusters`` is
given), so their standard errors account for units that appear in several
cells.  Aggregation weights (cohort sizes) are treated as known.  A joint
Wald test of the pre-period event-study coefficients is reported.

References
----------
[@gavrilova2025difference], [@callaway2021difference],
[@athey2019generalized], [@aytug2026fixed]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility
from ._core import calendar_time_aware as _calendar_time_aware

__all__ = ["did_forest", "DIDForestResult"]


@dataclass
class DIDForestResult(ResultProtocolMixin):
    """Result of :func:`did_forest`.

    Attributes
    ----------
    att_gt : DataFrame
        One row per estimated group-time cell: ``group``, ``time``,
        ``event_time``, ``att``, ``se``, ``ci_low``, ``ci_high``, ``pvalue``,
        ``n_treated``, ``n_control``, ``calibration_slope`` and
        ``heterogeneity_p`` (one-sided test that the forest's CATE ranking
        carries heterogeneity).
    event_study : DataFrame
        Cohort-size-weighted averages of ``att_gt`` by event time.
    overall : dict
        Average of post-treatment cells weighted by treated-unit counts.
    group_effects : DataFrame
        Average post-treatment effect for each cohort.
    pretrend_test : dict
        Joint Wald test that all pre-period event-study effects are zero.
    unit_cate : DataFrame
        Out-of-bag CATEs of each treated unit in its own cohort's
        post-treatment cells, with pointwise standard errors.
    dropped_cells : DataFrame
        Cells that could not be estimated and why.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> N = 300
    >>> x1 = rng.normal(size=N)
    >>> g = rng.choice([3, 0], size=N)
    >>> df = pd.DataFrame([dict(id=i, t=t, g=g[i], x1=x1[i],
    ...                         y=0.2 * t + (g[i] > 0 and t >= g[i]) * (1 + x1[i])
    ...                           + rng.normal())
    ...                    for i in range(N) for t in range(1, 5)])
    >>> res = sp.did_forest(df, y="y", id="id", time="t", cohort="g",
    ...                     x="x1", n_estimators=100)
    >>> isinstance(res, sp.DIDForestResult)
    True
    >>> list(res.att_gt.columns[:4])
    ['group', 'time', 'event_time', 'att']
    """

    att_gt: pd.DataFrame
    event_study: pd.DataFrame
    overall: Dict[str, float]
    group_effects: pd.DataFrame
    pretrend_test: Dict[str, Any]
    unit_cate: pd.DataFrame
    dropped_cells: pd.DataFrame
    n_units: int
    n_clusters: Optional[int]
    settings: Dict[str, Any] = field(default_factory=dict)

    _citation_keys = (
        "gavrilova2025difference",
        "callaway2021difference",
        "athey2019generalized",
    )

    def summary(self) -> str:
        o = self.overall
        lines = [
            "Difference-in-differences causal forest",
            "=" * 60,
            f"Units: {self.n_units}   Clusters: {self.n_clusters or 'units'}   "
            f"Cells: {len(self.att_gt)} estimated, {len(self.dropped_cells)} dropped",
            f"Comparison group: {self.settings.get('control_group')}   "
            f"Base period: universal (g - 1 - anticipation)",
            "",
            f"Overall ATT: {o['estimate']:.4f} (SE {o['se']:.4f}, "
            f"95% CI [{o['ci_low']:.4f}, {o['ci_high']:.4f}])",
        ]
        pt = self.pretrend_test
        if pt.get("df"):
            lines.append(
                f"Pre-trend joint test: chi2({pt['df']}) = {pt['statistic']:.3f}, "
                f"p = {pt['pvalue']:.4f}"
            )
        lines += ["", "Event study:", self.event_study.to_string(index=False)]
        if len(self.att_gt):
            post = self.att_gt[self.att_gt["event_time"] >= 0]
            if len(post):
                share = float((post["heterogeneity_p"] < 0.05).mean())
                lines += [
                    "",
                    f"Heterogeneity detected (calibration p < 0.05) in "
                    f"{share:.0%} of post-treatment cells.",
                ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience
        o = self.overall
        return (
            f"DIDForestResult(overall ATT={o['estimate']:.4f}, SE={o['se']:.4f}, "
            f"cells={len(self.att_gt)})"
        )

    def forest(self, group: Any, time: Any) -> Any:
        """The fitted :class:`~statspai.CausalForest` of cell ``(group, time)``."""
        forests = getattr(self, "_forests", {})
        key = (group, time)
        if key not in forests:
            raise MethodIncompatibility(
                f"did_forest: no forest for cell {key}.",
                recovery_hint="Pick a (group, time) pair listed in result.att_gt.",
            )
        return forests[key]

    def predict_cate(self, X: Any, event_time: Any) -> np.ndarray:
        """Event-time CATE surface at covariates ``X``.

        ``tau_e(x) = sum_g w_g tau_{g, g+e}(x)`` with cohort-size weights over
        the cohorts observed at event time ``e`` (Aytug 2026, eq. 4).
        """
        cells = self.att_gt[self.att_gt["event_time"] == event_time]
        if cells.empty:
            raise MethodIncompatibility(
                f"did_forest: no estimated cells at event time {event_time}.",
                recovery_hint="Use an event time listed in result.event_study.",
            )
        weights = cells["n_treated"].to_numpy(dtype=float)
        weights = weights / weights.sum()
        out = None
        for w, (_, row) in zip(weights, cells.iterrows()):
            pred = w * np.asarray(self.forest(row["group"], row["time"]).effect(X))
            out = pred if out is None else out + pred
        return np.asarray(out, dtype=float)

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Event-study plot with pointwise 95% intervals."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 4)))
        es = self.event_study
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.axvline(-0.5, color="grey", lw=0.8, ls="--")
        ax.errorbar(
            es["event_time"],
            es["att"],
            yerr=[es["att"] - es["ci_low"], es["ci_high"] - es["att"]],
            fmt="o",
            capsize=3,
            **kwargs,
        )
        ax.set_xlabel("Event time")
        ax.set_ylabel("ATT")
        ax.set_title("DiD causal forest: event study")
        return ax


def _cohort_codes(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = np.where(np.isnan(arr) | (arr == 0), np.inf, arr)
    return arr


def _normal_row(estimate: float, var: float, alpha: float) -> Dict[str, float]:
    from scipy import stats

    se = float(np.sqrt(max(var, 0.0)))
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = float(2 * stats.norm.sf(abs(estimate / se))) if se > 0 else float("nan")
    return {
        "estimate": float(estimate),
        "se": se,
        "ci_low": float(estimate - z * se),
        "ci_high": float(estimate + z * se),
        "pvalue": p,
    }


def _att_with_influence(
    cf: Any, z: np.ndarray, W: np.ndarray, clip: float
) -> Tuple[float, np.ndarray, int]:
    """Doubly-robust ATT of one cell and its per-row influence function."""
    tau = np.asarray(cf._oob_tau, dtype=float)
    m = np.asarray(cf._m_insample, dtype=float)
    e_raw = np.asarray(cf._e_insample, dtype=float)
    e = np.clip(e_raw, 0.0, 1.0 - clip)
    n_clipped = int(np.sum(e_raw > 1.0 - clip))
    n = z.size
    treated = W == 1
    control = ~treated
    n1 = int(treated.sum())
    tau_raw = float(tau[treated].mean())
    g_c = e[control] / (1.0 - e[control])
    gamma = np.zeros(n)
    gamma[control] = g_c / g_c.sum() * n if g_c.sum() > 0 else 0.0
    gamma[treated] = 1.0 / n1 * n
    mu0 = m - e * tau
    mu1 = m + (1.0 - e) * tau
    corr = W * gamma * (z - mu1) - (1.0 - W) * gamma * (z - mu0)
    dr = float(corr.mean())
    att = tau_raw + dr
    influence = (n / n1) * W * (tau - tau_raw) + (corr - dr)
    return att, influence, n_clipped


def _aggregate_variance(
    contributions: np.ndarray, unit_cluster: Optional[np.ndarray]
) -> np.ndarray:
    """Covariance of aggregates from per-unit contribution rows.

    ``contributions`` has shape (n_units, k): row u holds the sum over cells
    of ``w_c * IF_{c,u} / n_c`` for each of k aggregates.  No finite-sample
    factor is applied (the convention of R ``did``'s analytical standard
    errors), so a single-cell aggregate reproduces the cell's own SE.
    """
    if unit_cluster is None:
        return contributions.T @ contributions
    G = int(unit_cluster.max()) + 1
    summed = np.zeros((G, contributions.shape[1]))
    np.add.at(summed, unit_cluster, contributions)
    return summed.T @ summed


@accepts_aliases(unit="id")
@_calendar_time_aware(time="time", cohort="cohort")
def did_forest(
    data: pd.DataFrame,
    y: str,
    id: str,
    time: str,
    cohort: str,
    x: Union[str, Sequence[str]],
    *,
    control_group: str = "notyettreated",
    anticipation: int = 0,
    clusters: Optional[str] = None,
    event_window: Optional[Tuple[int, int]] = None,
    min_group_size: int = 20,
    n_estimators: int = 1000,
    random_state: int = 0,
    n_jobs: int = 1,
    alpha: float = 0.05,
    propensity_clip: float = 0.001,
    forest_kwargs: Optional[Dict[str, Any]] = None,
) -> DIDForestResult:
    """Heterogeneous difference-in-differences with causal forests.

    Estimates group-time conditional ATTs ``tau_{g,t}(x)`` in a panel with
    staggered (absorbing) adoption by fitting one honest causal forest per
    clean group-time comparison on the long-differenced outcome, then
    aggregates doubly-robust cell ATTs into an event study and an overall
    ATT with influence-function standard errors.

    Parameters
    ----------
    data : DataFrame
        Long panel, one row per unit-period.
    y, id, time : str
        Outcome, unit id and (numeric) period columns (``unit=`` is
        accepted as an alias of ``id``).
    cohort : str
        First treatment period of each unit; ``0``, ``NaN`` or ``inf``
        mark never-treated units.  Must be constant within unit.
    x : str or list of str
        Time-invariant unit covariates along which effects may vary.  Use
        pre-treatment (baseline) values; time-varying columns are refused.
    control_group : {"notyettreated", "nevertreated"}
        Comparison units for each cell (as in Callaway and Sant'Anna).
    anticipation : int, default 0
        Periods of anticipation; the base period is ``g - 1 - anticipation``.
    clusters : str, optional
        Column with a cluster id constant within unit (e.g. state).  Forests
        sample and cross-fit by cluster and all standard errors are
        cluster-robust.  By default units are independent.
    event_window : (int, int), optional
        Only estimate cells with ``lo <= t - g <= hi``.
    min_group_size : int, default 20
        Minimum treated and comparison units per cell; smaller cells are
        dropped and listed in ``dropped_cells``.
    n_estimators : int, default 1000
        Trees per cell forest.
    random_state, n_jobs :
        Seed (cell ``k`` uses ``random_state + k``) and threads per forest.
    alpha : float, default 0.05
    propensity_clip : float, default 0.001
        Upper clip for the estimated probability of belonging to the cohort,
        which enters the control weights ``e / (1 - e)``.
    forest_kwargs : dict, optional
        Extra arguments for :class:`~statspai.CausalForest`
        (``min_samples_leaf``, ``mtry``, ``honesty_fraction``, ...).

    Returns
    -------
    DIDForestResult

    Notes
    -----
    Identification: conditional parallel trends given ``x`` for the chosen
    comparison group, no anticipation beyond ``anticipation``, and overlap
    (every covariate profile has comparison units).  Pre-period cells are
    placebo estimates; their joint test is in ``pretrend_test``.  The cell
    standard error is the variance of the cell's influence function, which
    includes the covariance between the plug-in and correction terms (grf's
    ``average_treatment_effect`` reports the sum of the two variances).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> N, T = 400, 6
    >>> x1 = rng.normal(size=N)
    >>> g = rng.choice([3, 5, 0], size=N)
    >>> rows = []
    >>> for i in range(N):
    ...     for t in range(1, T + 1):
    ...         d = g[i] > 0 and t >= g[i]
    ...         rows.append(dict(id=i, t=t, g=g[i], x1=x1[i],
    ...                          y=x1[i] + 0.3 * t + d * (1 + x1[i]) + rng.normal()))
    >>> df = pd.DataFrame(rows)
    >>> res = sp.did_forest(df, y="y", id="id", time="t", cohort="g",
    ...                     x="x1", n_estimators=200)
    >>> res.event_study["event_time"].tolist()
    [-4, -3, -2, 0, 1, 2, 3]

    References
    ----------
    [@gavrilova2025difference], [@callaway2021difference],
    [@athey2019generalized], [@aytug2026fixed]
    """
    from scipy import stats

    from ..forest.causal_forest import CausalForest

    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "did_forest(): data must be a pandas DataFrame.",
            recovery_hint="Pass a long panel DataFrame.",
        )
    x_cols = [x] if isinstance(x, str) else list(x)
    if not x_cols:
        raise MethodIncompatibility(
            "did_forest(): at least one covariate is required.",
            recovery_hint=(
                "Pass x=[...] with baseline characteristics; without "
                "covariates use sp.callaway_santanna()."
            ),
            alternative_functions=["sp.callaway_santanna"],
        )
    needed = [y, id, time, cohort, *x_cols] + ([clusters] if clusters else [])
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"did_forest(): column(s) not in data: {missing}.",
            recovery_hint="Check the column names against data.columns.",
        )
    if control_group not in ("notyettreated", "nevertreated"):
        raise MethodIncompatibility(
            "did_forest(): control_group must be 'notyettreated' or 'nevertreated'.",
            recovery_hint="Use control_group='notyettreated'.",
        )
    if (
        isinstance(anticipation, bool)
        or not isinstance(anticipation, (int, np.integer))
        or anticipation < 0
    ):
        raise MethodIncompatibility(
            "did_forest(): anticipation must be a non-negative integer.",
            recovery_hint="Use anticipation=0.",
        )
    if (
        isinstance(min_group_size, bool)
        or not isinstance(min_group_size, (int, np.integer))
        or min_group_size < 2
    ):
        raise MethodIncompatibility(
            "did_forest(): min_group_size must be an integer >= 2.",
            recovery_hint="Use min_group_size=20.",
        )

    df = data[needed].copy()
    df = df.dropna(subset=[y, id, time, *x_cols])
    if df.duplicated([id, time]).any():
        raise MethodIncompatibility(
            "did_forest(): (unit, time) pairs must be unique.",
            recovery_hint="Aggregate to one row per unit-period.",
        )
    try:
        df[time] = pd.to_numeric(df[time])
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "did_forest(): the time column must be numeric.",
            recovery_hint="Encode periods as integers (e.g. years).",
        ) from exc
    df["_cohort"] = _cohort_codes(df[cohort])

    units = df.groupby(id, sort=True)
    if (units["_cohort"].nunique(dropna=False) > 1).any():
        raise MethodIncompatibility(
            "did_forest(): the cohort column varies within unit.",
            recovery_hint="Code each unit's first treatment period on every row.",
        )
    x_var = units[x_cols].agg(lambda s: float(np.nanmax(s) - np.nanmin(s)))
    if (x_var.to_numpy() > 1e-9).any():
        bad = list(x_var.columns[(x_var.to_numpy() > 1e-9).any(axis=0)])
        raise MethodIncompatibility(
            f"did_forest(): covariate(s) {bad} vary within unit.",
            recovery_hint=(
                "Use baseline (pre-treatment) values of time-varying "
                "characteristics; effect modifiers must be fixed per unit."
            ),
        )
    unit_ids = np.array(sorted(df[id].unique()), dtype=object)
    n_units = unit_ids.size
    first = units.first()
    unit_cohort = first["_cohort"].reindex(unit_ids).to_numpy(dtype=float)
    unit_X = first[x_cols].reindex(unit_ids).to_numpy(dtype=float)
    unit_cluster: Optional[np.ndarray] = None
    if clusters is not None:
        if (units[clusters].nunique(dropna=False) > 1).any():
            raise MethodIncompatibility(
                "did_forest(): the cluster column varies within unit.",
                recovery_hint="Cluster at a level that contains whole units.",
            )
        _, unit_cluster = np.unique(
            first[clusters].reindex(unit_ids).astype(str).to_numpy(),
            return_inverse=True,
        )
        unit_cluster = unit_cluster.astype(np.int64)
        if unit_cluster.max() + 1 < 2:
            raise DataInsufficient(
                "did_forest(): at least two clusters are required.",
                recovery_hint="Use a cluster variable with more than one value.",
            )

    periods = np.array(sorted(df[time].unique()), dtype=float)
    integral_time = bool(np.all(np.mod(periods, 1.0) == 0.0))
    wide = (
        df.pivot(index=id, columns=time, values=y)
        .reindex(index=unit_ids, columns=periods)
        .to_numpy(dtype=float)
    )
    cohorts = sorted(c for c in np.unique(unit_cohort) if np.isfinite(c))
    if not cohorts:
        raise DataInsufficient(
            "did_forest(): no treated cohort found.",
            recovery_hint="Code first treatment periods in the cohort column.",
        )
    if control_group == "nevertreated" and not np.any(~np.isfinite(unit_cohort)):
        raise DataInsufficient(
            "did_forest(): control_group='nevertreated' but no unit is never treated.",
            recovery_hint="Use control_group='notyettreated'.",
        )

    kwargs = dict(forest_kwargs or {})
    for reserved in ("clusters", "random_state", "n_estimators", "n_jobs", "fe"):
        if reserved in kwargs:
            raise MethodIncompatibility(
                f"did_forest(): pass {reserved!r} directly, not in forest_kwargs.",
                recovery_hint=f"Use did_forest(..., {reserved}=...).",
            )

    rows: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    influence: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
    forests: Dict[Tuple[Any, Any], Any] = {}
    unit_cate_rows: List[pd.DataFrame] = []
    cell_k = 0

    for g in cohorts:
        pos_g = int(np.searchsorted(periods, g))
        base_pos = pos_g - 1 - int(anticipation)
        if base_pos < 0:
            dropped.append(
                {
                    "group": int(g) if integral_time else g,
                    "time": np.nan,
                    "reason": "no pre-treatment base period",
                }
            )
            continue
        base = periods[base_pos]
        for t in periods:
            if t == base:
                continue
            event = int(t - g) if integral_time and float(g).is_integer() else t - g
            if integral_time:
                t = int(t)
                g_label = int(g)
            else:
                g_label = g
            if event_window is not None and not (
                event_window[0] <= event <= event_window[1]
            ):
                continue
            treated = unit_cohort == g
            if control_group == "nevertreated":
                control = ~np.isfinite(unit_cohort)
            else:
                horizon = max(t, base) + anticipation
                control = (~np.isfinite(unit_cohort)) | (unit_cohort > horizon)
            control &= ~treated
            observed = np.isfinite(
                wide[:, int(np.searchsorted(periods, t))]
            ) & np.isfinite(wide[:, base_pos])
            in_cell = (treated | control) & observed
            n1 = int((treated & in_cell).sum())
            n0 = int((control & in_cell).sum())
            if n1 < min_group_size or n0 < min_group_size:
                dropped.append(
                    {
                        "group": g_label,
                        "time": t,
                        "reason": f"too few units (treated {n1}, comparison {n0}; "
                        f"min_group_size={min_group_size})",
                    }
                )
                continue
            idx = np.flatnonzero(in_cell)
            z = wide[idx, int(np.searchsorted(periods, t))] - wide[idx, base_pos]
            W = treated[idx].astype(float)
            cf = CausalForest(
                n_estimators=n_estimators,
                random_state=int(random_state) + cell_k,
                n_jobs=n_jobs,
                **kwargs,
            )
            cf.fit(
                Y=z,
                T=W,
                X=unit_X[idx],
                clusters=None if unit_cluster is None else unit_cluster[idx],
            )
            cell_k += 1
            if np.any(~np.isfinite(cf._oob_tau)):
                dropped.append(
                    {
                        "group": g_label,
                        "time": t,
                        "reason": (
                            "rows without out-of-bag prediction; "
                            "increase n_estimators"
                        ),
                    }
                )
                continue
            att, inf, n_clip = _att_with_influence(cf, z, W, propensity_clip)
            n_cell = idx.size
            var = _aggregate_variance(
                (inf / n_cell)[:, None],
                None if unit_cluster is None else unit_cluster[idx],
            )[0, 0]
            stats_row = _normal_row(att, var, alpha)
            from ..forest._grf_inference import calibration_blp

            try:
                cal = calibration_blp(cf)
                slope = float(cal.loc["differential_forest_prediction", "coef"])
                het_p = float(cal.loc["differential_forest_prediction", "p_one_sided"])
            except DataInsufficient:
                slope, het_p = float("nan"), float("nan")
            rows.append(
                {
                    "group": g_label,
                    "time": t,
                    "event_time": event,
                    "att": stats_row["estimate"],
                    "se": stats_row["se"],
                    "ci_low": stats_row["ci_low"],
                    "ci_high": stats_row["ci_high"],
                    "pvalue": stats_row["pvalue"],
                    "n_treated": n1,
                    "n_control": n0,
                    "calibration_slope": slope,
                    "heterogeneity_p": het_p,
                    "n_propensity_clipped": n_clip,
                }
            )
            influence[(g_label, t)] = (idx, inf)
            forests[(g_label, t)] = cf
            if event >= 0:
                tr = W == 1
                var_oob = cf._oob_var
                unit_cate_rows.append(
                    pd.DataFrame(
                        {
                            "unit": unit_ids[idx[tr]],
                            "group": g_label,
                            "time": t,
                            "event_time": event,
                            "cate": cf._oob_tau[tr],
                            "cate_se": (
                                np.sqrt(var_oob[tr]) if var_oob is not None else np.nan
                            ),
                        }
                    )
                )

    att_gt = pd.DataFrame(rows)
    dropped_df = pd.DataFrame(dropped, columns=["group", "time", "reason"])
    if att_gt.empty:
        raise DataInsufficient(
            "did_forest(): no group-time cell could be estimated.",
            recovery_hint="Lower min_group_size or check cohort/period coding.",
            diagnostics={"dropped_cells": dropped[:10]},
        )
    if len(dropped_df):
        import warnings

        warnings.warn(
            f"did_forest(): {len(dropped_df)} group-time cell(s) were dropped; "
            "see result.dropped_cells.",
            AssumptionWarning,
            stacklevel=2,
        )

    # ---- aggregation via stacked per-unit influence contributions ----------
    def contributions(weight_sets: List[Dict[Tuple[Any, Any], float]]) -> np.ndarray:
        C = np.zeros((n_units, len(weight_sets)))
        for k, weights in enumerate(weight_sets):
            for key, w in weights.items():
                idx, inf = influence[key]
                C[idx, k] += w * inf / idx.size
        return C

    def estimate(weights: Dict[Tuple[Any, Any], float]) -> float:
        lookup = att_gt.set_index(["group", "time"])["att"]
        return float(sum(w * lookup.loc[key] for key, w in weights.items()))

    events = sorted(att_gt["event_time"].unique())
    es_weights: List[Dict[Tuple[Any, Any], float]] = []
    for e in events:
        cells = att_gt[att_gt["event_time"] == e]
        tot = cells["n_treated"].sum()
        es_weights.append(
            {(r.group, r.time): r.n_treated / tot for r in cells.itertuples()}
        )
    es_cov = _aggregate_variance(contributions(es_weights), unit_cluster)
    es_rows = []
    for k, e in enumerate(events):
        s = _normal_row(estimate(es_weights[k]), es_cov[k, k], alpha)
        es_rows.append(
            {
                "event_time": e,
                "att": s["estimate"],
                "se": s["se"],
                "ci_low": s["ci_low"],
                "ci_high": s["ci_high"],
                "pvalue": s["pvalue"],
                "n_cells": len(es_weights[k]),
            }
        )
    event_study = pd.DataFrame(es_rows)

    pre_idx = [k for k, e in enumerate(events) if e < 0]
    if pre_idx:
        b = event_study.loc[pre_idx, "att"].to_numpy()
        V = es_cov[np.ix_(pre_idx, pre_idx)]
        stat = float(b @ np.linalg.pinv(V) @ b)
        dof = int(np.linalg.matrix_rank(V))
        pretrend = {
            "statistic": stat,
            "df": dof,
            "pvalue": float(stats.chi2.sf(stat, dof)) if dof > 0 else float("nan"),
            "null": "all pre-period event-study effects are zero",
        }
    else:
        pretrend = {
            "statistic": float("nan"),
            "df": 0,
            "pvalue": float("nan"),
            "null": "no pre-periods estimated",
        }

    post = att_gt[att_gt["event_time"] >= 0]
    if post.empty:
        raise DataInsufficient(
            "did_forest(): no post-treatment cell could be estimated.",
            recovery_hint="Widen event_window or lower min_group_size.",
        )
    tot = post["n_treated"].sum()
    overall_w = {(r.group, r.time): r.n_treated / tot for r in post.itertuples()}
    group_ws = []
    group_labels = []
    for g, cells in post.groupby("group"):
        group_labels.append(g)
        group_ws.append(
            {(r.group, r.time): 1.0 / len(cells) for r in cells.itertuples()}
        )
    agg_cov = _aggregate_variance(contributions([overall_w] + group_ws), unit_cluster)
    overall = _normal_row(estimate(overall_w), agg_cov[0, 0], alpha)
    group_effects = pd.DataFrame(
        [
            {
                "group": g,
                **{
                    ("att" if k == "estimate" else k): v
                    for k, v in _normal_row(
                        estimate(w), agg_cov[j + 1, j + 1], alpha
                    ).items()
                },
                "n_post_periods": len(w),
            }
            for j, (g, w) in enumerate(zip(group_labels, group_ws))
        ]
    )

    unit_cate = (
        pd.concat(unit_cate_rows, ignore_index=True)
        if unit_cate_rows
        else pd.DataFrame(
            columns=["unit", "group", "time", "event_time", "cate", "cate_se"]
        )
    )
    result = DIDForestResult(
        att_gt=att_gt,
        event_study=event_study,
        overall=overall,
        group_effects=group_effects,
        pretrend_test=pretrend,
        unit_cate=unit_cate,
        dropped_cells=dropped_df,
        n_units=int(n_units),
        n_clusters=None if unit_cluster is None else int(unit_cluster.max()) + 1,
        settings={
            "control_group": control_group,
            "anticipation": int(anticipation),
            "covariates": x_cols,
            "n_estimators": int(n_estimators),
            "min_group_size": int(min_group_size),
            "event_window": event_window,
            "clusters": clusters,
            "aggregation_weights": "cohort sizes (treated as known)",
        },
    )
    result._forests = forests  # type: ignore[attr-defined]
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            result,
            function="sp.did_forest",
            params={
                "y": y,
                "id": id,
                "time": time,
                "cohort": cohort,
                "x": x_cols,
                "control_group": control_group,
                "anticipation": anticipation,
                "clusters": clusters,
                "n_estimators": n_estimators,
                "random_state": random_state,
            },
            data=data,
            overwrite=False,
        )
    except (ImportError, AttributeError, TypeError) as exc:  # pragma: no cover
        result.settings["provenance_error"] = f"{type(exc).__name__}: {exc}"
    return result
