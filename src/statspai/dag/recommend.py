"""
Estimator recommendation from a declared DAG.

Given a DAG and the (exposure, outcome) pair, inspect the graph
structure and suggest an estimator from the statspai API, citing the
identification assumption being relied upon.

Rules (checked in priority order):
  1. No unblocked path / no confounders  -> OLS (sp.regress)
  2. Unobserved confounder U + valid IV Z available  -> IV (sp.iv)
  3. Mediator M on the causal path X -> M -> Y  -> mediate (sp.mediate)
  4. Backdoor path blockable by observed set S  -> IPW or matching
  5. Otherwise -> report non-identifiable and refer to sp.dag.identify
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, List, Optional, Sequence, Set, Tuple

from .._aliases import accepts_aliases

__all__ = ["EstimatorRecommendation", "recommend_estimator"]


@dataclass
class EstimatorRecommendation:
    estimator: str  # statspai function name
    sp_call: str  # example sp.xxx(...) string
    identification: str  # identification assumption in plain English
    adjustment_set: Optional[Set[str]]  # what to condition on, if any
    instrument: Optional[str] = None
    mediators: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "DAG -> Estimator Recommendation",
            "=" * 55,
            f"  Recommended estimator : sp.{self.estimator}",
            f"  Example call          : {self.sp_call}",
            f"  Identification        : {self.identification}",
        ]
        if self.adjustment_set is not None:
            lines.append(
                f"  Adjustment set        : "
                f"{{{', '.join(sorted(self.adjustment_set)) or '(empty)'}}}"
            )
        if self.instrument:
            lines.append(f"  Instrument            : {self.instrument}")
        if self.mediators:
            lines.append(f"  Mediators on path     : {', '.join(self.mediators)}")
        if self.alternatives:
            lines.append("  Alternatives          :")
            for alt in self.alternatives:
                lines.append(f"      - {alt}")
        if self.warnings:
            lines.append("  Warnings              :")
            for w in self.warnings:
                lines.append(f"      ! {w}")
        return "\n".join(lines)


@accepts_aliases(y="outcome")
def recommend_estimator(
    dag: Any,
    exposure: str,
    outcome: str,
    candidate_instruments: Optional[Sequence[str]] = None,
) -> EstimatorRecommendation:
    """Inspect a DAG and recommend a statspai estimator.

    Parameters
    ----------
    dag : statspai.dag.DAG
    exposure, outcome : str
    candidate_instruments : list of str, optional
        Variable names to check as potential IVs.  If omitted, all
        observed nodes other than exposure/outcome are considered.

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.dag("Z -> X; Z -> Y; X -> Y")  # Z confounds X -> Y
    >>> rec = sp.dag_recommend_estimator(g, exposure="X", outcome="Y")
    >>> rec.estimator
    'regress'
    >>> "Z" in rec.adjustment_set  # backdoor path blocked by conditioning on Z
    True
    """
    if exposure not in dag.nodes or outcome not in dag.nodes:
        raise KeyError("exposure / outcome must be nodes in the DAG.")

    alternatives: List[str] = []
    warnings: List[str] = []

    # Check for mediators on the causal path
    mediators = _causal_mediators(dag, exposure, outcome)
    if mediators:
        alternatives.append(
            f"sp.mediate(...): mediators detected ({', '.join(mediators)}); "
            "if you want the total vs. direct effect."
        )

    # Try backdoor
    adjustment_sets = dag.adjustment_sets(exposure, outcome, minimal=True)
    if adjustment_sets:
        s = adjustment_sets[0]
        if not s:
            return EstimatorRecommendation(
                estimator="regress",
                sp_call=f"sp.regress('{outcome} ~ {exposure}', data=df)",
                identification="No open backdoor path — direct regression OK.",
                adjustment_set=set(),
                mediators=mediators,
                alternatives=alternatives,
            )
        s_str = " + ".join(sorted(s))
        return EstimatorRecommendation(
            estimator="regress",
            sp_call=(
                f"sp.regress('{outcome} ~ {exposure} + {s_str}', data=df)"
                f"  # or sp.ipw(df, y='{outcome}', treat='{exposure}', "
                f"covariates={sorted(s)!r})"
            ),
            identification=(
                f"Backdoor paths blocked by conditioning on {sorted(s)}. "
                "OLS with these controls is consistent under conditional "
                "exchangeability (= selection-on-observables = ignorability)."
            ),
            adjustment_set=set(s),
            mediators=mediators,
            alternatives=[
                f"sp.ipw(df, y='{outcome}', treat='{exposure}', "
                f"covariates={sorted(s)!r})",
                ("sp.aipw(...): doubly-robust combination of IPW + " "outcome model"),
                (
                    f"sp.match(..., covariates={sorted(s)!r}): "
                    "propensity-score matching"
                ),
            ]
            + alternatives,
        )

    # No adjustment set found — try IV
    iv_found = _find_instrument(
        dag,
        exposure,
        outcome,
        candidate_instruments,
    )
    if iv_found is not None:
        iv_candidate, iv_controls = iv_found
        controls_str = "".join(f"{c} + " for c in sorted(iv_controls))
        given = f" given {sorted(iv_controls)}" if iv_controls else ""
        return EstimatorRecommendation(
            estimator="iv",
            sp_call=(
                f"sp.iv('{outcome} ~ {controls_str}"
                f"({exposure} ~ {iv_candidate})', data=df)"
            ),
            identification=(
                f"Unobserved confounding blocks backdoor adjustment, but "
                f"{iv_candidate} is an instrument{given}: it is associated "
                f"with {exposure}, and every path from it to {outcome} runs "
                f"through {exposure}."
            ),
            adjustment_set=set(iv_controls) if iv_controls else None,
            instrument=iv_candidate,
            mediators=mediators,
            alternatives=[
                "sp.liml(...): weak-IV-robust LIML",
                "sp.anderson_rubin_ci(...): weak-IV-robust CI",
                (
                    f"sp.bartik(...): shift-share IV if {iv_candidate} "
                    "is a shift-share"
                ),
            ]
            + alternatives,
        )

    # Fall back to front-door if available
    fd_set = _frontdoor_set(dag, exposure, outcome)
    if fd_set:
        fd_sorted = sorted(fd_set)
        fd_warnings: List[str] = []
        if len(fd_sorted) > 1:
            fd_warnings.append(
                "sp.front_door takes one mediator; the front-door set "
                f"{fd_sorted} has {len(fd_sorted)}, so the call below uses "
                f"{fd_sorted[0]!r} only and does not identify the effect on "
                "its own."
            )
        return EstimatorRecommendation(
            estimator="front_door",
            sp_call=(
                f"sp.front_door(df, y='{outcome}', treat='{exposure}', "
                f"mediator='{fd_sorted[0]}')"
            ),
            identification=(
                f"Front-door criterion: mediators {fd_sorted} intercept "
                f"every directed path from {exposure} to {outcome}, have no "
                f"open backdoor from {exposure}, and their backdoor paths to "
                f"{outcome} are blocked by {exposure}."
            ),
            adjustment_set=None,
            mediators=fd_sorted,
            alternatives=alternatives,
            warnings=fd_warnings,
        )

    warnings.append(
        "No valid backdoor / IV / frontdoor found; the effect is NOT "
        "identifiable under the declared DAG. Consider adding a proxy "
        "(sp.proximal), bounds (sp.bounds), or sensitivity analysis."
    )
    return EstimatorRecommendation(
        estimator="identify",
        sp_call=(f"sp.identify(dag, '{exposure}', '{outcome}')  # to see why"),
        identification="Not identifiable under the declared DAG.",
        adjustment_set=None,
        mediators=mediators,
        alternatives=[
            "sp.proximal(...): use proxies for unmeasured confounding",
            ("sp.lee_bounds(...) / sp.manski_bounds(...): partial " "identification"),
            "sp.sensemakr(...): sensitivity to unobserved confounding",
        ]
        + alternatives,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _causal_mediators(dag: Any, x: str, y: str) -> List[str]:
    """Nodes lying on a directed path from X to Y (excluding endpoints)."""
    descendants_x = dag.descendants(x)
    ancestors_y = dag.ancestors(y)
    mediators = (descendants_x & ancestors_y) - {x, y}
    return sorted(m for m in mediators if not m.startswith("_L_"))


def _find_instrument(
    dag: Any,
    exposure: str,
    outcome: str,
    candidates: Optional[Sequence[str]],
    max_conditioning: int = 2,
) -> Optional[Tuple[str, Set[str]]]:
    """Search for an instrument ``Z`` and a conditioning set ``S``.

    ``Z`` qualifies given ``S`` when, with ``S`` made of observed variables
    that the exposure does not affect,

    - ``Z`` and the exposure are d-connected given ``S`` (relevance), and
    - ``Z`` and the outcome are d-separated given ``S`` in the graph with
      every arrow into the exposure removed (exclusion and exogeneity: any
      remaining open path from ``Z`` to the outcome bypasses the exposure).

    The second condition is checked in the mutilated graph on purpose.
    Conditioning on the exposure in the original graph opens the collider
    ``Z -> exposure <- U`` and would reject every instrument in exactly the
    confounded graphs an instrument is for.

    Returns ``(Z, S)`` for the first candidate that qualifies with the
    smallest ``S`` (at most ``max_conditioning`` variables), or ``None``.
    """
    desc_x = dag.descendants(exposure)
    observed = [n for n in sorted(dag.observed_nodes) if n not in (exposure, outcome)]
    if candidates is None:
        candidates = observed
    mutilated = dag.do(exposure)
    for z in candidates:
        if z in desc_x or z not in dag.nodes or z in (exposure, outcome):
            continue
        pool = [n for n in observed if n != z and n not in desc_x]
        for size in range(0, min(max_conditioning, len(pool)) + 1):
            for combo in combinations(pool, size):
                cond = set(combo)
                if dag.d_separated(z, exposure, cond):
                    continue
                if mutilated.d_separated(z, outcome, cond):
                    return z, cond
    return None


def _frontdoor_set(dag: Any, x: str, y: str) -> Set[str]:
    """A minimal set meeting Pearl's front-door criterion, or the empty set.

    Delegates to :meth:`DAG.frontdoor_sets`, which checks the three
    conditions path by path. A direct edge ``x -> y`` therefore rules the
    front door out, as does a latent parent shared with a mediator.
    """
    sets = dag.frontdoor_sets(x, y)
    return set(sets[0]) if sets else set()
