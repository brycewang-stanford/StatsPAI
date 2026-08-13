"""Design-level audits for DiD studies: the contract and the cluster count.

Two diagnostics that run on the design rather than on the estimate.

:func:`did_design_contract` operationalises the forward-engineering recipe
of Baker, Callaway, Cunningham, Goodman-Bacon and Sant'Anna (2026): fix
the target parameter, state the identifying assumption, choose the
estimation method, and declare the inference frame *before* reading a
coefficient, rather than reverse-engineering a causal reading out of a
familiar regression.  The contract reports which of those a fitted result
actually pins down and which it leaves implicit.  It is written to be
unflattering: a slot the result cannot determine is reported as
undetermined, never as a default, because the whole point of the recipe
is that an unstated choice is still a choice.

:func:`did_cluster_diagnostics` reports the number of clusters at which
treatment is assigned, and grades it against the simulation grid of
Ulloa-Perez, Bair, Navathe and Linn (2025).  Their finding is blunt: at
30 clusters *every* modern staggered estimator they evaluated
under-covered a nominal 95 percent interval, and coverage improved as
clusters accumulated.  Thirty is also the smallest cell they ran, so a
design with fewer clusters is outside their evidence rather than
reassured by it, and this function says so instead of extrapolating.

References
----------
Baker, A., Callaway, B., Cunningham, S., Goodman-Bacon, A. and
Sant'Anna, P. H. C. (2026).  "Difference-in-Differences Designs: A
Practitioner's Guide."  *Journal of Economic Literature*, 64(2), 498-557.
[@baker2026difference]

Ulloa-Perez, E., Bair, E. F., Navathe, A. S. and Linn, K. A. (2025).
"Comparative Evaluation of Difference in Differences Methods for
Staggered Adoption Interventions."  arXiv:2508.14365.
[@ulloaperez2025comparative]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient

__all__ = [
    "DiDClusterDiagnostics",
    "DiDDesignContract",
    "did_cluster_diagnostics",
    "did_design_contract",
]


# --------------------------------------------------------------------- #
# Cluster count
# --------------------------------------------------------------------- #
#: The cluster counts Ulloa-Perez et al. (2025) actually ran.  Anything
#: below the smallest is extrapolation, and is labelled as such.
_UP_GRID = (30, 50, 100)

_UP_VERDICTS = {
    "below-evidence": (
        "Fewer clusters than the smallest cell of the Ulloa-Perez et al. "
        "(2025) grid (30). Their result -- that every evaluated staggered "
        "estimator under-covered at 30 clusters -- is the closest evidence "
        "available and it does not extend downward. Treat interval coverage "
        "as unknown, not as merely degraded."
    ),
    "weakest-cell": (
        "At the weakest cell of the Ulloa-Perez et al. (2025) grid. There, "
        "every evaluated estimator under-covered a nominal 95 percent "
        "interval; two-way Mundlak ranged from 60 to 85 percent, and the "
        "doubly-robust and IPW estimators also under-covered."
    ),
    "improving": (
        "Between the weakest and largest cells of the Ulloa-Perez et al. "
        "(2025) grid. Coverage improved with cluster count throughout their "
        "study but several estimators had still not reached nominal."
    ),
    "largest-cell-or-above": (
        "At or above the largest cell of the Ulloa-Perez et al. (2025) grid "
        "(100). Coverage was closest to nominal there, though two-way "
        "Mundlak did not reach it in any setting they ran."
    ),
}


@dataclass
class DiDClusterDiagnostics:
    """Cluster counts for a staggered design, graded against known evidence."""

    n_clusters: int
    n_treated_clusters: int
    n_control_clusters: int
    n_cohorts: int
    clusters_per_cohort: Dict[Any, int]
    cluster_column: str
    cluster_is_unit: bool
    verdict: str
    interpretation: str
    reference_grid: tuple = _UP_GRID

    def summary(self) -> str:
        lines = [
            "DiD cluster diagnostics",
            "=" * 52,
            f"Clustering variable : {self.cluster_column}"
            + ("  (defaulted to the unit id)" if self.cluster_is_unit else ""),
            f"Clusters            : {self.n_clusters} "
            f"({self.n_treated_clusters} ever treated, "
            f"{self.n_control_clusters} never treated)",
            f"Treated cohorts     : {self.n_cohorts}",
            f"Verdict             : {self.verdict}",
            "",
            self.interpretation,
        ]
        if self.clusters_per_cohort:
            thin = {g: n for g, n in self.clusters_per_cohort.items() if n < 5}
            if thin:
                lines += [
                    "",
                    f"Cohorts with fewer than five clusters: {thin}. A "
                    "group-time effect for these rests on very few "
                    "independent units regardless of the total.",
                ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "n_treated_clusters": self.n_treated_clusters,
            "n_control_clusters": self.n_control_clusters,
            "n_cohorts": self.n_cohorts,
            "clusters_per_cohort": dict(self.clusters_per_cohort),
            "cluster_column": self.cluster_column,
            "cluster_is_unit": self.cluster_is_unit,
            "verdict": self.verdict,
            "interpretation": self.interpretation,
            "reference_grid": list(self.reference_grid),
        }


def did_cluster_diagnostics(
    data: pd.DataFrame,
    unit: str,
    first_treat: str,
    cluster: Optional[str] = None,
    *,
    warn: bool = True,
) -> DiDClusterDiagnostics:
    """Count the clusters treatment is assigned at, and grade the count.

    Parameters
    ----------
    data : DataFrame
        Long panel.
    unit : str
        Unit identifier.
    first_treat : str
        First treated period; ``0``, ``NaN`` or ``inf`` for never-treated.
    cluster : str, optional
        The level treatment is assigned at -- a state, a provider
        organisation, a school district.  Defaults to ``unit`` with a
        warning, because the two coincide only when each unit is
        independently assigned, and assuming so is the optimistic error.
    warn : bool, default True
        Emit a warning when the design sits in or below the weakest cell
        of the reference grid.

    Returns
    -------
    DiDClusterDiagnostics

    Notes
    -----
    The grading is a statement about the evidence that exists, not a
    power calculation for this design.  A "largest-cell-or-above" verdict
    means the closest published simulation found coverage near nominal at
    that cluster count for the estimators it evaluated; it is not a
    guarantee about the estimator or the data at hand.

    Examples
    --------
    >>> import statspai as sp
    >>> mpdta = sp.datasets.mpdta()
    >>> diag = sp.did_cluster_diagnostics(
    ...     mpdta, unit="countyreal", first_treat="first_treat", warn=False
    ... )
    >>> diag.n_cohorts
    3
    """
    for col in (unit, first_treat) + ((cluster,) if cluster else ()):
        if col not in data.columns:
            raise ValueError(
                f"Column {col!r} not found in data. Available: "
                f"{sorted(data.columns)[:12]}..."
            )

    cluster_is_unit = cluster is None
    cluster_col = cluster or unit
    if cluster_is_unit and warn:
        warnings.warn(
            "did_cluster_diagnostics: no cluster= given, so the unit id is "
            "used as the clustering level. That is only right when each "
            "unit's treatment is assigned independently. If treatment was "
            "assigned at a coarser level (state, provider group, district), "
            "pass that column -- the coarser count is the one that governs "
            "inference, and it is always the smaller number.",
            UserWarning,
            stacklevel=2,
        )

    # dict.fromkeys keeps order and drops the duplicate when the cluster
    # level defaulted to the unit id; selecting a column twice yields a
    # 2-D grouper and a confusing pandas error.
    frame = data[list(dict.fromkeys([unit, cluster_col, first_treat]))].copy()
    ft = pd.to_numeric(frame[first_treat], errors="coerce")
    treated = ft.notna() & np.isfinite(ft) & (ft > 0)
    frame["_treated"] = treated

    per_cluster = frame.groupby(cluster_col)["_treated"].any()
    n_clusters = int(len(per_cluster))
    if n_clusters == 0:
        raise DataInsufficient(
            f"No clusters found in column {cluster_col!r}.",
            recovery_hint="Check that the cluster column is populated.",
        )
    n_treated_clusters = int(per_cluster.sum())
    n_control_clusters = n_clusters - n_treated_clusters

    treated_rows = frame[frame["_treated"]]
    cohorts = pd.to_numeric(treated_rows[first_treat], errors="coerce")
    clusters_per_cohort = {
        (int(g) if float(g).is_integer() else float(g)): int(
            treated_rows.loc[cohorts == g, cluster_col].nunique()
        )
        for g in sorted(cohorts.dropna().unique())
    }

    if n_clusters < _UP_GRID[0]:
        verdict = "below-evidence"
    elif n_clusters < _UP_GRID[1]:
        verdict = "weakest-cell"
    elif n_clusters < _UP_GRID[2]:
        verdict = "improving"
    else:
        verdict = "largest-cell-or-above"

    diag = DiDClusterDiagnostics(
        n_clusters=n_clusters,
        n_treated_clusters=n_treated_clusters,
        n_control_clusters=n_control_clusters,
        n_cohorts=len(clusters_per_cohort),
        clusters_per_cohort=clusters_per_cohort,
        cluster_column=cluster_col,
        cluster_is_unit=cluster_is_unit,
        verdict=verdict,
        interpretation=_UP_VERDICTS[verdict],
    )

    if warn and verdict in ("below-evidence", "weakest-cell"):
        warnings.warn(
            f"did_cluster_diagnostics: {n_clusters} clusters "
            f"({verdict}). {_UP_VERDICTS[verdict]} Consider a "
            "cluster-robust bootstrap or randomisation inference rather "
            "than an analytic interval.",
            UserWarning,
            stacklevel=2,
        )
    return diag


# --------------------------------------------------------------------- #
# Forward-engineering contract
# --------------------------------------------------------------------- #
#: Baker et al. (2026), section 6.  Steps 5 and 8 are process rather than
#: anything a result object can evidence, and are reported as such rather
#: than being given a fabricated status.
_STEPS: tuple = (
    (1, "target_parameter", "Define target parameters"),
    (2, "identification", "State the identification assumptions formally"),
    (3, "estimation", "Determine the appropriate estimation method"),
    (4, "uncertainty", "Discuss sources of uncertainty"),
    (5, "estimate", "Estimate"),
    (6, "sensitivity", "Conduct sensitivity analysis"),
    (7, "heterogeneity", "Conduct heterogeneity analysis"),
    (8, "keep_learning", "Keep learning"),
)


@dataclass
class DiDDesignContract:
    """What a fitted DiD result pins down, against Baker et al.'s recipe."""

    steps: List[Dict[str, Any]] = field(default_factory=list)
    method: str = ""
    undetermined: List[str] = field(default_factory=list)

    @property
    def n_determined(self) -> int:
        return sum(1 for s in self.steps if s["status"] == "determined")

    def summary(self) -> str:
        lines = [
            "DiD forward-engineering contract (Baker et al. 2026, sec. 6)",
            "=" * 62,
            f"Estimator: {self.method or 'unknown'}",
            "",
        ]
        for step in self.steps:
            mark = {
                "determined": "[x]",
                "undetermined": "[ ]",
                "not-evidenced": "[-]",
            }[step["status"]]
            lines.append(f"{mark} {step['step']}. {step['title']}")
            lines.append(f"      {step['detail']}")
        lines += [
            "",
            "[x] the result determines this; [ ] it does not, and the choice "
            "is therefore implicit;",
            "[-] not something a fitted object can evidence.",
            "",
            "An unstated choice is still a choice: every [ ] is a decision "
            "the write-up has to make explicit, not a step that was skipped "
            "harmlessly.",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "steps": list(self.steps),
            "undetermined": list(self.undetermined),
            "n_determined": self.n_determined,
        }


def _mi(result: Any) -> Dict[str, Any]:
    return getattr(result, "model_info", None) or {}


def did_design_contract(result: Any) -> DiDDesignContract:
    """Report which forward-engineering steps a fitted DiD result pins down.

    Baker et al. (2026, section 6) argue that a DiD study should be
    *forward* engineered -- target parameter, identifying assumption,
    estimator, inference frame, in that order -- rather than reverse
    engineered from whichever regression is familiar.  This reads a fitted
    result and reports which of those the call actually determined.

    The function never invents a status.  A slot that cannot be read off
    the result is ``undetermined``, which is the informative answer: it
    means the choice was made by a default rather than by the analyst, and
    that the write-up still owes the reader a statement of it.

    Parameters
    ----------
    result : CausalResult
        Output of a DiD estimator.

    Returns
    -------
    DiDDesignContract

    Examples
    --------
    >>> import statspai as sp
    >>> mpdta = sp.datasets.mpdta()
    >>> fit = sp.callaway_santanna(
    ...     mpdta, y="lemp", g="first_treat", t="year", i="countyreal"
    ... )
    >>> contract = sp.did_design_contract(fit)
    >>> contract.steps[0]["status"] in {"determined", "undetermined"}
    True
    """
    info = _mi(result)
    method = str(
        info.get("estimator")
        or info.get("method")
        or info.get("model_type")
        or getattr(result, "method", "")
        or ""
    )

    steps: List[Dict[str, Any]] = []
    undetermined: List[str] = []

    def add(step: int, key: str, title: str, status: str, detail: str) -> None:
        steps.append(
            {
                "step": step,
                "key": key,
                "title": title,
                "status": status,
                "detail": detail,
            }
        )
        if status == "undetermined":
            undetermined.append(key)

    # Step 1 -- target parameter.
    estimand = info.get("aggregation") or info.get("estimand") or info.get("type")
    if estimand:
        add(
            1,
            "target_parameter",
            _STEPS[0][2],
            "determined",
            f"Reported aggregation: {estimand}.",
        )
    else:
        add(
            1,
            "target_parameter",
            _STEPS[0][2],
            "undetermined",
            "The result does not record which summary of ATT(g,t) is "
            "reported. State whether the target is the simple, dynamic, "
            "group or calendar aggregate.",
        )

    # Step 2 -- identification. The comparison group IS the assumption.
    control = info.get("control_group")
    base = info.get("base_period")
    anticipation = info.get("anticipation")
    if control:
        pt = {
            "nevertreated": "parallel trends against never-treated units",
            "notyettreated": "parallel trends against not-yet-treated units",
        }.get(str(control), f"comparison group {control!r}")
        bits = [f"Imposes {pt}"]
        if base:
            bits.append(f"base period {base!r}")
        if anticipation is not None:
            bits.append(f"anticipation window {anticipation}")
        add(2, "identification", _STEPS[1][2], "determined", "; ".join(bits) + ".")
    else:
        add(
            2,
            "identification",
            _STEPS[1][2],
            "undetermined",
            "No comparison group recorded. Never-treated, not-yet-treated "
            "and all-periods parallel trends are three different "
            "assumptions with different pre-trend implications; the paper "
            "must say which one it imposes.",
        )

    # Step 3 -- estimation strategy.
    strategy = info.get("est_method") or info.get("estimator") or method
    if strategy:
        add(
            3,
            "estimation",
            _STEPS[2][2],
            "determined",
            f"Estimation strategy: {strategy}.",
        )
    else:
        add(
            3,
            "estimation",
            _STEPS[2][2],
            "undetermined",
            "The nuisance strategy (regression adjustment, IPW, doubly "
            "robust) is not recorded.",
        )

    # Step 4 -- uncertainty. Sampling vs design-based, and the cluster level.
    vce = info.get("vce") or info.get("se_method") or info.get("inference")
    cluster_var = info.get("cluster_var") or info.get("clustervars")
    n_clusters = info.get("n_clusters")
    if vce or cluster_var:
        bits = []
        if vce:
            bits.append(f"variance route {vce!r}")
        if cluster_var:
            bits.append(f"clustered on {cluster_var!r}")
        if n_clusters:
            bits.append(f"{n_clusters} clusters")
        add(
            4,
            "uncertainty",
            _STEPS[3][2],
            "determined",
            "; ".join(bits) + ". Sampling-based unless stated otherwise.",
        )
    else:
        add(
            4,
            "uncertainty",
            _STEPS[3][2],
            "undetermined",
            "Neither the variance route nor the clustering level is "
            "recorded. State what is treated as random.",
        )

    # Step 5 -- estimation happened; that is what the object is.
    est = getattr(result, "estimate", None)
    add(
        5,
        "estimate",
        _STEPS[4][2],
        "determined" if est is not None else "undetermined",
        (
            "A point estimate is present."
            if est is not None
            else "No point estimate on the result object."
        ),
    )

    # Step 6 -- sensitivity.
    has_sensitivity = any(
        k in info for k in ("honest_did", "sensitivity", "breakdown_m", "rambachan")
    )
    if has_sensitivity:
        add(
            6,
            "sensitivity",
            _STEPS[5][2],
            "determined",
            "Sensitivity output is attached to the result.",
        )
    else:
        add(
            6,
            "sensitivity",
            _STEPS[5][2],
            "undetermined",
            "No sensitivity analysis attached. A pre-trend test that fails "
            "to reject is not one; see sp.honest_did_from_result.",
        )

    # Step 7 -- heterogeneity.
    has_het = any(k in info for k in ("hetby", "group", "cate", "subgroup"))
    if has_het:
        add(
            7,
            "heterogeneity",
            _STEPS[6][2],
            "determined",
            "Subgroup or group-specific effects are reported.",
        )
    else:
        add(
            7,
            "heterogeneity",
            _STEPS[6][2],
            "undetermined",
            "Only aggregate effects are reported; an aggregate can mask "
            "heterogeneity that is itself the question.",
        )

    # Step 8 -- process, not evidence.
    add(
        8,
        "keep_learning",
        _STEPS[7][2],
        "not-evidenced",
        "Whether DiD is the right design here is a judgement about the "
        "setting, not a property of this object.",
    )

    return DiDDesignContract(steps=steps, method=method, undetermined=undetermined)
