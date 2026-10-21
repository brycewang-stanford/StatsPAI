"""Two-stage randomized (saturation) designs under partial interference.

Groups are first randomized to a *saturation* (the share of the group to be
treated), then units inside each group are randomized to treatment with
that share treated: Hudgens and Halloran's (2008) two-stage design with
complete randomization at both stages. Units interfere only within their
group. The estimands are Hudgens and Halloran's group-averaged potential
outcomes ``Ybar(z; a)``: the mean outcome of a unit with own treatment
``z`` when its group is at saturation ``a``, averaged over the within-group
assignments of the design, then over the units of a group, then over
groups (``weighting='group'``) or over units (``weighting='individual'``).

The estimators are group-level means: ``Yhat_i(z)`` is the mean outcome of
the units of group ``i`` with treatment ``z`` and ``Yhat(z; a)`` the mean of
``Yhat_i(z)`` over the groups at saturation ``a``. The variances are the
conservative group-level ("ultimate cluster") estimators: the sample
variance across groups of the group-level estimate, divided by the number
of groups (one term per saturation in a cross-saturation contrast). Their
expectation exceeds the randomization variance by exactly ``S^2 / N``, the
between-group variance of the group-level effect over all ``N`` groups, so
they are unbiased when that effect is the same in every group (Basse and
Feller 2018, Theorem 5.1 with household weights, which they note recovers
Hudgens and Halloran's results). ``tests/test_two_stage_interference.py``
checks both statements by enumerating every assignment of small designs.

Individual weighting rescales each outcome by ``n_i / nbar`` (group size
over the mean group size) and applies the same estimators, as in Basse and
Feller (2018, Section 5).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility

__all__ = ["two_stage_interference"]

# Verbatim from paper.bib (hudgens2008toward).
CausalResult._CITATIONS["two_stage_interference"] = (
    "@article{hudgens2008toward,\n"
    "  title={Toward causal inference with interference},\n"
    "  author={Hudgens, Michael G. and Halloran, M. Elizabeth},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={103},\n"
    "  number={482},\n"
    "  pages={832--842},\n"
    "  year={2008},\n"
    "  publisher={Taylor \\& Francis},\n"
    "  doi={10.1198/016214508000000292}\n"
    "}"
)

_WEIGHTINGS = ("group", "individual")


def _groups(
    data: pd.DataFrame, y: str, treat: str, group: str, saturation: str, weighting: str
) -> pd.DataFrame:
    """One row per group: size, treated count, saturation and the group
    means of the (weighted) outcome by own treatment."""
    frame = data[[y, treat, group, saturation]]
    bad = frame.columns[frame.isna().any()].tolist()
    if bad:
        raise MethodIncompatibility(
            f"two_stage_interference: missing values in {bad}.",
            recovery_hint="The design-based estimators need every unit of every "
            "group: impute or drop whole groups before calling.",
            diagnostics={"columns": bad},
        )
    d = pd.to_numeric(frame[treat], errors="coerce")
    if d.isna().any() or not set(np.unique(d)) <= {0.0, 1.0}:
        raise MethodIncompatibility(
            f"two_stage_interference: {treat!r} must be coded 0/1.",
            recovery_hint="Recode the unit-level treatment as 0 (control) / 1 "
            "(treated).",
        )
    sat = pd.to_numeric(frame[saturation], errors="coerce")
    if sat.isna().any() or (sat < 0).any() or (sat > 1).any():
        raise MethodIncompatibility(
            f"two_stage_interference: {saturation!r} must be the share of the "
            "group assigned to treatment, a number in [0, 1].",
            recovery_hint="Pass the group's randomized saturation (e.g. 0, 0.25, "
            "0.75), not a label.",
        )
    work = pd.DataFrame(
        {
            "g": frame[group].to_numpy(),
            "d": d.to_numpy(dtype=float),
            "a": sat.to_numpy(dtype=float),
            "y": pd.to_numeric(frame[y], errors="raise").to_numpy(dtype=float),
        }
    )
    varying = work.groupby("g", sort=True)["a"].nunique()
    if (varying > 1).any():
        raise MethodIncompatibility(
            f"two_stage_interference: {saturation!r} varies within "
            f"{int((varying > 1).sum())} group(s).",
            recovery_hint="The saturation is assigned to the whole group in the "
            "first stage; it must be constant within each group.",
            diagnostics={"groups": varying[varying > 1].index.tolist()[:10]},
        )
    size = work.groupby("g", sort=True)["y"].transform("size").to_numpy(float)
    if weighting == "individual":
        work["y"] = work["y"] * size / work.groupby("g")["y"].size().mean()
    g = work.groupby("g", sort=True)
    out = pd.DataFrame(
        {
            "n": g.size(),
            "m": g["d"].sum(),
            "a": g["a"].first(),
            "ybar": g["y"].mean(),
            "ybar1": work[work["d"] == 1].groupby("g")["y"].mean(),
            "ybar0": work[work["d"] == 0].groupby("g")["y"].mean(),
        }
    )
    return out


def _check_design(tab: pd.DataFrame) -> List[Dict[str, Any]]:
    """Refuse groups whose treated count contradicts their saturation's
    support; return groups whose count is off by one unit or more."""
    zero = tab[(tab["a"] == 0) & (tab["m"] > 0)]
    full = tab[(tab["a"] == 1) & (tab["m"] < tab["n"])]
    inner = tab[(tab["a"] > 0) & (tab["a"] < 1)]
    empty = inner[(inner["m"] == 0) | (inner["m"] == inner["n"])]
    for bad, what in [
        (zero, "saturation 0 but treated units"),
        (full, "saturation 1 but untreated units"),
        (empty, "a saturation strictly between 0 and 1 but only one arm"),
    ]:
        if len(bad):
            raise MethodIncompatibility(
                f"two_stage_interference: {len(bad)} group(s) with {what} "
                f"(e.g. {bad.index.tolist()[:5]}).",
                recovery_hint="Check the saturation column against the "
                "realized assignment; each group at 0 < a < 1 needs treated "
                "and untreated units.",
                diagnostics={"groups": bad.index.tolist()[:20]},
            )
    off = inner[np.abs(inner["m"] - inner["a"] * inner["n"]) >= 1]
    return [
        {"group": k, "n": int(r.n), "treated": int(r.m), "saturation": float(r.a)}
        for k, r in off.iterrows()
    ]


def _mean_se(v: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))


def two_stage_interference(
    data: pd.DataFrame,
    y: str,
    treat: str,
    group: str,
    saturation: str,
    reference: Optional[float] = None,
    weighting: str = "group",
    alpha: float = 0.05,
) -> CausalResult:
    """Direct, indirect, total and overall effects of a two-stage randomized
    (saturation) design under partial interference.

    Groups are randomized to saturations, then units within each group are
    randomized with that share treated (complete randomization at both
    stages). Units may affect others in their own group but not in other
    groups.

    Parameters
    ----------
    data : DataFrame
        One row per unit.
    y : str
        Outcome.
    treat : str
        Unit-level treatment, 0/1.
    group : str
        Group (cluster) identifier: the unit of the first-stage
        randomization and the boundary of interference.
    saturation : str
        The saturation each group was randomized to, the share of its units
        to be treated, constant within the group (0 for pure-control
        groups).
    reference : float, optional
        The saturation the cross-saturation effects are measured against.
        Default: the lowest saturation in the data.
    weighting : {'group', 'individual'}, default 'group'
        ``'group'`` averages over groups (Hudgens and Halloran's
        estimands); ``'individual'`` over units, through the outcomes
        rescaled by ``n_i / nbar``.
    alpha : float, default 0.05
        Level of the normal confidence intervals.

    Returns
    -------
    CausalResult
        Headline: the total effect of the highest saturation, ``Ybar(1;
        high) - Ybar(0; reference)``. ``detail`` has one row per effect:
        ``direct`` at each saturation strictly between 0 and 1 (``Ybar(1;
        a) - Ybar(0; a)``), and for each other saturation ``a``
        ``indirect`` (``Ybar(0; a) - Ybar(0; reference)``), ``total``
        (``Ybar(1; a) - Ybar(0; reference)``) and ``overall`` (``Ybar(a) -
        Ybar(reference)``), with the conservative standard error, normal
        interval and p-value. ``model_info['means']`` has each
        ``Ybar(z; a)`` and ``Ybar(a)`` with its standard error;
        ``model_info['count_mismatch']`` lists groups whose treated count is
        one or more units away from ``saturation * size``.

    Notes
    -----
    Signs are treated minus untreated and higher minus reference
    saturation; Hudgens and Halloran (2008) write each effect the other way
    round (untreated minus treated, low minus high coverage). The identity
    ``direct(a) + indirect(a) = total(a)`` holds exactly for the estimates.

    Each standard error is the sample standard deviation across groups of
    the group-level estimate over the square root of the number of groups,
    with one such term per saturation in a cross-saturation contrast. It
    overstates the randomization variance by the between-group variance of
    the group-level effect divided by the number of groups, and is exact
    when the effect is constant across groups (Basse and Feller 2018,
    Theorem 5.1). The intervals rely on the number of groups being large.

    Unbiasedness requires that the second stage fixes the number treated
    in each group; with Bernoulli assignment inside groups the group means
    are ratio estimators. A count far from ``saturation * size`` is
    reported (and warned about), not corrected.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for g, a in enumerate(np.repeat([0.0, 0.25, 0.75], 20)):
    ...     d = np.zeros(8, int); d[rng.choice(8, int(a * 8), replace=False)] = 1
    ...     y = 1 + 2 * d + 1.5 * a + rng.normal(size=8)
    ...     rows.append(pd.DataFrame({"g": g, "a": a, "d": d, "y": y}))
    >>> df = pd.concat(rows, ignore_index=True)
    >>> r = sp.two_stage_interference(df, "y", "d", "g", "a")
    >>> sorted(set(r.detail["effect"]))
    ['direct', 'indirect', 'overall', 'total']

    References
    ----------
    [@hudgens2008toward], [@basse2018analyzing]
    """
    if weighting not in _WEIGHTINGS:
        raise MethodIncompatibility(
            f"two_stage_interference: weighting={weighting!r}.",
            recovery_hint="Use weighting='group' or 'individual'.",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"two_stage_interference: alpha={alpha} is not in (0, 1).",
            recovery_hint="Pass e.g. alpha=0.05.",
        )
    check_columns(
        data,
        {"y": y, "treat": treat, "group": group, "saturation": saturation},
        function="two_stage_interference",
    )
    tab = _groups(data, y, treat, group, saturation, weighting)
    mismatch = _check_design(tab)
    if mismatch:
        warnings.warn(
            f"two_stage_interference: {len(mismatch)} group(s) have a treated "
            "count one or more units away from saturation x size; the "
            "estimators assume the design fixed the count at the saturation.",
            AssumptionWarning,
            stacklevel=2,
        )
    levels = sorted(tab["a"].unique())
    if len(levels) < 2:
        raise DataInsufficient(
            "two_stage_interference: only one saturation in the data.",
            recovery_hint="The cross-saturation effects need groups at two or "
            "more saturations; with one, use sp.difference_in_means(cluster=).",
        )
    counts = tab.groupby("a").size()
    thin = counts[counts < 2]
    if len(thin):
        raise DataInsufficient(
            "two_stage_interference: fewer than two groups at saturation(s) "
            f"{thin.index.tolist()}; the group-level variance needs two.",
            recovery_hint="Pool or drop the saturation levels with a single " "group.",
            diagnostics={"groups_per_saturation": counts.to_dict()},
        )
    ref = levels[0] if reference is None else float(reference)
    if ref not in levels:
        raise MethodIncompatibility(
            f"two_stage_interference: reference={reference} is not a saturation "
            f"in the data ({levels}).",
            recovery_hint="Pass one of the saturations, or omit it for the " "lowest.",
        )

    def cells(a: float) -> pd.DataFrame:
        return tab[tab["a"] == a]

    def column(a: float, z: Optional[int]) -> Optional[np.ndarray]:
        key = "ybar" if z is None else f"ybar{z}"
        if (z == 1 and a == 0) or (z == 0 and a == 1):
            return None
        return cells(a)[key].to_numpy(float)

    means = []
    for a in levels:
        for z in (0, 1, None):
            v = column(a, z)
            if v is None:
                continue
            est, se = _mean_se(v)
            means.append(
                {
                    "saturation": a,
                    "treat": "all" if z is None else z,
                    "estimate": est,
                    "se": se,
                    "n_groups": len(v),
                }
            )

    zc = stats.norm.ppf(1 - alpha / 2)
    rows: List[Dict[str, Any]] = []

    def add(effect: str, a: float, est: float, var: float, n_groups: int) -> None:
        se = float(np.sqrt(var))
        rows.append(
            {
                "effect": effect,
                "saturation": a,
                "reference": np.nan if effect == "direct" else ref,
                "estimate": est,
                "se": se,
                "ci_lower": est - zc * se,
                "ci_upper": est + zc * se,
                "pvalue": float(2 * stats.norm.sf(abs(est) / se)) if se else np.nan,
                "n_groups": n_groups,
            }
        )

    for a in levels:
        if 0 < a < 1:
            diff = column(a, 1) - column(a, 0)  # type: ignore[operator]
            est, se = _mean_se(diff)
            add("direct", a, est, se**2, len(diff))
    for a in levels:
        if a == ref:
            continue
        for effect, z_a, z_ref in [
            ("indirect", 0, 0),
            ("total", 1, 0),
            ("overall", None, None),
        ]:
            va, vr = column(a, z_a), column(ref, z_ref)
            if va is None or vr is None:
                continue
            (ma, sa), (mr, sr) = _mean_se(va), _mean_se(vr)
            add(effect, a, ma - mr, sa**2 + sr**2, len(va) + len(vr))
    detail = pd.DataFrame(rows)
    scale = float(np.nanmax(np.abs(tab[["ybar", "ybar0", "ybar1"]].to_numpy())))
    if (detail["se"] <= 1e-12 * max(scale, 1e-300)).any():
        raise DataInsufficient(
            "two_stage_interference: a group-level estimate does not vary "
            "across groups, so its standard error is zero.",
            recovery_hint="Check for a constant outcome within a saturation.",
        )
    headline_a = max(a for a in levels if a != ref)
    pick = detail[(detail["effect"] == "total") & (detail["saturation"] == headline_a)]
    if pick.empty:  # reference saturation 1: no untreated reference units
        pick = detail[
            (detail["effect"] == "overall") & (detail["saturation"] == headline_a)
        ]
    head = pick.iloc[0]
    info = {
        "means": pd.DataFrame(means),
        "reference": ref,
        "saturations": levels,
        "groups_per_saturation": {float(k): int(v) for k, v in counts.items()},
        "weighting": weighting,
        "headline": f"{head['effect']} effect, saturation {headline_a} vs {ref}",
        "count_mismatch": mismatch,
        "sign_convention": "treated minus untreated; saturation minus reference "
        "(Hudgens and Halloran 2008 write each effect with the opposite sign)",
        "variance": "group-level sample variance / number of groups; "
        "conservative by S^2/N, exact for constant group-level effects",
    }
    return CausalResult(
        method="Two-stage randomized design (Hudgens-Halloran)",
        estimand=f"{head['effect']} effect",
        estimate=float(head["estimate"]),
        se=float(head["se"]),
        pvalue=float(head["pvalue"]),
        ci=(float(head["ci_lower"]), float(head["ci_upper"])),
        alpha=alpha,
        n_obs=int(tab["n"].sum()),
        detail=detail,
        model_info=info,
        _citation_key="two_stage_interference",
    )
