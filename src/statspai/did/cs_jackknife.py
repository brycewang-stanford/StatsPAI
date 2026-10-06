"""
Cluster-jackknife (CV3) inference for Callaway--Sant'Anna aggregates.

The analytic influence-function variance and the multiplier bootstrap of
:func:`callaway_santanna` / :func:`aggte` both rest on the same asymptotic
approximation, and both over-reject when there are few clusters or few
treated clusters. Karim, Nielsen, MacKinnon and Webb [@karim2026improved]
show that the delete-one-cluster jackknife repairs most of that. This
module reproduces their reference implementations, R ``didjack`` and Stata
``csdidjack``: delete cluster ``h``, re-estimate every ATT(g, t) and the
aggregation weights, re-aggregate, and repeat for every cluster.

    CV3 = (R - 1) / R * sum_h (ATT_(-h) - ATT)^2

centred on the full-sample estimate, with ``R`` the number of replicates and
inference from ``t(R - 1)``. When fewer than two clusters contain a treated
unit, the treated cluster is skipped (deleting it would leave nothing to
estimate), exactly as both references do.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility
from .aggte import aggte
from .callaway_santanna import callaway_santanna, few_never_treated_cutoff

_AGG_TYPES = ("simple", "dynamic", "group", "calendar")
# Arguments that set the inference the jackknife replaces, or the data it
# deletes from; accepting them silently would report a different object.
_FORBIDDEN_KWARGS = ("bstrap", "cband", "se_method", "biters", "clustervars")


@accepts_aliases(_strict=True, t="time", i="id")
def cs_jackknife(
    data: pd.DataFrame,
    y: str,
    g: str,
    time: str,
    id: str,
    *,
    type: str = "simple",
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    **cs_kwargs: Any,
) -> CausalResult:
    """Delete-one-cluster jackknife (CV3) for a Callaway--Sant'Anna aggregate.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format balanced panel, as for :func:`callaway_santanna`.
    y, g, time, id : str
        Outcome, first-treatment period (0 = never treated), time, and unit
        columns.
    type : {'simple', 'dynamic', 'group', 'calendar'}, default 'simple'
        Aggregation passed to :func:`aggte`; the jackknifed object is its
        overall summary (``aggte(...).estimate``), the same object R
        ``didjack`` and Stata ``csdidjack`` report.
    cluster : str, optional
        Cluster variable; defaults to the unit ``id``. Must be constant within
        unit, since deleting a cluster must delete whole units.
    alpha : float, default 0.05
        Level for the ``t(R - 1)`` confidence interval.
    **cs_kwargs
        Forwarded unchanged to every :func:`callaway_santanna` fit
        (``control_group``, ``estimator``, ``base_period``, ``weights``,
        ``x``, ...). Inference options (``bstrap``, ``cband``, ``se_method``,
        ``biters``, ``clustervars``) are rejected: the jackknife is the
        inference.

    Returns
    -------
    CausalResult
        ``estimate`` is the full-sample aggregate, ``se`` the CV3 standard
        error, ``pvalue`` and ``ci`` from ``t(R - 1)``. ``detail`` holds one
        row per replicate (deleted cluster and its estimate);
        ``model_info`` records ``n_clusters``, ``n_replicates``, ``df``,
        ``skipped_clusters``, the analytic influence-function SE of the same
        aggregate, and ``se_method='cluster_jackknife'``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for u in range(24):
    ...     g = [0, 3, 4][u % 3]
    ...     a = rng.normal()
    ...     for tt in range(1, 6):
    ...         d = 1.0 if g and tt >= g else 0.0
    ...         rows.append((u, tt, g, a + 0.2 * tt + 0.5 * d + rng.normal(0, 0.3)))
    >>> df = pd.DataFrame(rows, columns=["unit", "time", "g", "y"])
    >>> jk = sp.cs_jackknife(
    ...     df, y="y", g="g", time="time", id="unit", estimator="reg"
    ... )
    >>> jk.model_info["n_replicates"]
    24
    >>> bool(jk.se > 0)
    True

    References
    ----------
    @karim2026improved; @callaway2021difference
    """
    if type not in _AGG_TYPES:
        raise MethodIncompatibility(
            f"cs_jackknife: type must be one of {list(_AGG_TYPES)}, got {type!r}.",
            recovery_hint="Use type='simple' for the overall ATT.",
            diagnostics={"type": type},
        )
    bad = [k for k in _FORBIDDEN_KWARGS if k in cs_kwargs]
    if bad:
        raise MethodIncompatibility(
            f"cs_jackknife: {bad} set the inference the jackknife replaces.",
            recovery_hint=("Drop them; pass cluster= to choose the jackknife cluster."),
            diagnostics={"rejected": bad},
        )
    if not (isinstance(alpha, float) and 0.0 < alpha < 1.0):
        raise MethodIncompatibility(
            f"cs_jackknife: alpha must be a float in (0, 1), got {alpha!r}.",
            recovery_hint="Use alpha=0.05 for a 95 percent interval.",
            diagnostics={"alpha": alpha},
        )
    cluster = id if cluster is None else cluster
    for col in (y, g, time, id, cluster):
        if col not in data.columns:
            raise MethodIncompatibility(
                f"cs_jackknife: column {col!r} not found in data.",
                recovery_hint="Check the column names passed to cs_jackknife.",
                diagnostics={"missing": col},
            )
    if cluster != id:
        varying = data.groupby(id)[cluster].nunique(dropna=False)
        if (varying > 1).any():
            raise MethodIncompatibility(
                f"cs_jackknife: cluster {cluster!r} varies within unit, so "
                "deleting a cluster would cut units in half and unbalance "
                "the panel.",
                recovery_hint="Use a time-invariant cluster variable.",
                diagnostics={"n_time_varying_units": int((varying > 1).sum())},
            )

    def _fit(frame: pd.DataFrame, *, quiet: bool = False) -> CausalResult:
        with warnings.catch_warnings():
            if quiet:
                # The delete-one refits repeat whatever the full fit already
                # said (few treated clusters, trimming, ...) once per
                # replicate; the full fit below is not silenced.
                warnings.simplefilter("ignore", AssumptionWarning)
            fit = callaway_santanna(frame, y=y, g=g, t=time, i=id, **cs_kwargs)
        return aggte(fit, type=type, bstrap=False, cband=False, alpha=alpha)

    full = _fit(data)
    att = float(full.estimate)

    # Never-treated units carry g = 0 (callaway_santanna also reads NaN / inf
    # as never treated), so a cluster is treated if any of its rows is not.
    g_vals = pd.to_numeric(data[g], errors="coerce").replace([np.inf, -np.inf], np.nan)
    treated_rows = g_vals.fillna(0).to_numpy() != 0
    clusters = np.sort(pd.unique(data[cluster]))
    treated_clusters = pd.unique(data.loc[treated_rows, cluster])
    single_treated = len(treated_clusters) < 2
    skipped = list(treated_clusters) if single_treated else []

    # The refits are silenced for what the full fit already said. A
    # never-treated group that only falls below R `did`'s minimum once a
    # cluster is deleted is something the full fit never said, so say it here.
    if cs_kwargs.get("control_group", "nevertreated") == "nevertreated":
        never = pd.DataFrame(
            {
                "unit": data.loc[~treated_rows, id].to_numpy(),
                "cluster": data.loc[~treated_rows, cluster].to_numpy(),
            }
        ).drop_duplicates()
        n_never = int(never["unit"].nunique())
        cutoff = few_never_treated_cutoff(cs_kwargs.get("x"))
        if len(never) and n_never >= cutoff:
            fewest = n_never - int(never.groupby("cluster")["unit"].nunique().max())
            if fewest < cutoff:
                warnings.warn(
                    f"cs_jackknife: the {n_never} never-treated units fall to "
                    f"{fewest} in some delete-one samples, below the {cutoff} "
                    "at which R `did` refuses control_group='nevertreated'. "
                    "The replicates are computed; R `didjack` on did >= 2.5.0 "
                    "stops on this design.",
                    AssumptionWarning,
                    stacklevel=2,
                )

    labels: List[Any] = []
    estimates: List[float] = []
    seen_warnings: Dict[str, int] = {}
    cluster_col = data[cluster].to_numpy()
    for c in clusters:
        if c in skipped:
            continue
        sub = data.loc[cluster_col != c]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                rep = _fit(sub, quiet=True)
            except (DataInsufficient, MethodIncompatibility, ValueError) as exc:
                raise DataInsufficient(
                    f"cs_jackknife: deleting cluster {c!r} leaves an aggregate "
                    f"that cannot be estimated ({exc}).",
                    recovery_hint=(
                        "The jackknife needs every delete-one sample to "
                        "identify the aggregate; with a comparison group or "
                        "cohort carried by one cluster, report the analytic "
                        "or bootstrap SE and state why."
                    ),
                    diagnostics={
                        "cluster": repr(c),
                        "error_type": exc.__class__.__name__,
                    },
                ) from exc
        for w in caught:
            key = f"{w.category.__name__}: {w.message}"
            seen_warnings[key] = seen_warnings.get(key, 0) + 1
        labels.append(c)
        estimates.append(float(rep.estimate))

    reps = len(estimates)
    if reps < 2:
        raise DataInsufficient(
            f"cs_jackknife: {reps} jackknife replicate(s); CV3 needs at least two.",
            recovery_hint="The jackknife is undefined with fewer than three clusters.",
            diagnostics={"n_clusters": int(len(clusters)), "skipped": skipped},
        )
    for msg, count in seen_warnings.items():
        warnings.warn(
            f"cs_jackknife: in {count} of {reps} replicates -- {msg}",
            UserWarning,
            stacklevel=2,
        )

    atts = np.asarray(estimates, dtype=float)
    df = reps - 1
    se = float(np.sqrt(df / reps * np.sum((atts - att) ** 2)))
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    if se > 0:
        tstat = att / se
        pvalue = float(2.0 * stats.t.sf(abs(tstat), df))
    else:
        tstat, pvalue = np.nan, np.nan
        warnings.warn(
            "cs_jackknife: every replicate returned the full-sample estimate, "
            "so the CV3 standard error is zero and no test is reported.",
            UserWarning,
            stacklevel=2,
        )

    detail = pd.DataFrame(
        {"deleted_cluster": labels, "att": atts, "deviation": atts - att}
    )
    model_info: Dict[str, Any] = {
        "se_method": "cluster_jackknife",
        "variance": "CV3 = (R-1)/R * sum_h (ATT_(-h) - ATT)^2, t(R-1) inference",
        "aggregation": type,
        "cluster": cluster,
        "n_clusters": int(len(clusters)),
        "n_treated_clusters": int(len(treated_clusters)),
        "n_replicates": reps,
        "df": df,
        "crit_val": crit,
        "t_stat": float(tstat),
        "skipped_clusters": skipped,
        "single_treated_cluster": bool(single_treated),
        "analytic_se": float(full.se),
        "cs_kwargs": {k: v for k, v in cs_kwargs.items() if k != "x"},
    }
    return CausalResult(
        method=(
            f"Callaway and Sant'Anna (2021) — aggte[{type}], " "cluster jackknife (CV3)"
        ),
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=(att - crit * se, att + crit * se),
        alpha=alpha,
        n_obs=full.n_obs,
        detail=detail,
        model_info=model_info,
        _citation_key="cs_jackknife",
    )
