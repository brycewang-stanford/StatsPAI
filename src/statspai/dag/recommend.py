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
from typing import List, Optional, Set


__all__ = ["EstimatorRecommendation", "recommend_estimator"]


@dataclass
class EstimatorRecommendation:
    estimator: str                       # statspai function name
    sp_call: str                         # example sp.xxx(...) string
    identification: str                  # identification assumption in plain English
    adjustment_set: Optional[Set[str]]   # what to condition on, if any
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


def recommend_estimator(
    dag,
    exposure: str,
    outcome: str,
    candidate_instruments: Optional[List[str]] = None,
) -> EstimatorRecommendation:
    """Inspect a DAG and recommend a statspai estimator.

    Parameters
    ----------
    dag : statspai.dag.DAG
    exposure, outcome : str
    candidate_instruments : list of str, optional
        Variable names to check as potential IVs.  If omitted, all
        observed nodes other than exposure/outcome are considered.
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
                f"  # or sp.ipw(df, treat='{exposure}', outcome='{outcome}', "
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
                f"sp.ipw(treat='{exposure}', outcome='{outcome}', "
                f"covariates={sorted(s)!r})",
                "sp.aipw(...): doubly-robust combination of IPW + outcome model",
                f"sp.match(..., covariates={sorted(s)!r}): propensity-score matching",
            ] + alternatives,
        )

    # No adjustment set found — try IV
    iv_candidate = _find_instrument(
        dag, exposure, outcome, candidate_instruments,
    )
    if iv_candidate is not None:
        return EstimatorRecommendation(
            estimator="iv",
            sp_call=f"sp.iv('{outcome} ~ [{exposure} ~ {iv_candidate}]', data=df)",
            identification=(
                f"Unobserved confounding blocks backdoor adjustment, but "
                f"{iv_candidate} satisfies the exclusion restriction "
                f"(associates with {exposure}, no direct path to {outcome})."
            ),
            adjustment_set=None,
            instrument=iv_candidate,
            mediators=mediators,
            alternatives=[
                "sp.liml(...): weak-IV-robust LIML",
                "sp.anderson_rubin_ci(...): weak-IV-robust CI",
                f"sp.bartik(...): shift-share IV if {iv_candidate} is a shift-share",
            ] + alternatives,
        )

    # Fall back to front-door if available
    fd_set = _frontdoor_set(dag, exposure, outcome, mediators)
    if fd_set:
        return EstimatorRecommendation(
            estimator="front_door",
            sp_call=(
                f"sp.front_door(df, exposure='{exposure}', outcome='{outcome}', "
                f"mediators={sorted(fd_set)!r})"
            ),
            identification=(
                f"Front-door criterion: mediators {sorted(fd_set)} capture "
                f"the full causal pathway and have no open backdoor to {outcome}."
            ),
            adjustment_set=None,
            mediators=list(fd_set),
            alternatives=alternatives,
        )

    warnings.append(
        "No valid backdoor / IV / frontdoor found; the effect is NOT "
        "identifiable under the declared DAG. Consider adding a proxy "
        "(sp.proximal), bounds (sp.bounds), or sensitivity analysis."
    )
    return EstimatorRecommendation(
        estimator="identify",
        sp_call=f"sp.dag.identify(dag, '{exposure}', '{outcome}')  # to see why",
        identification="Not identifiable under the declared DAG.",
        adjustment_set=None,
        mediators=mediators,
        alternatives=[
            "sp.proximal(...): use proxies for unmeasured confounding",
            "sp.lee_bounds(...) / sp.manski_bounds(...): partial identification",
            "sp.sensemakr(...): sensitivity to unobserved confounding",
        ] + alternatives,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _causal_mediators(dag, x: str, y: str) -> List[str]:
    """Nodes lying on a directed path from X to Y (excluding endpoints)."""
    descendants_x = dag.descendants(x)
    ancestors_y = dag.ancestors(y)
    mediators = (descendants_x & ancestors_y) - {x, y}
    return sorted(m for m in mediators if not m.startswith("_L_"))


def _find_instrument(
    dag,
    exposure: str,
    outcome: str,
    candidates: Optional[List[str]],
) -> Optional[str]:
    """Search for a node Z that:
      - has a directed path to ``exposure``,
      - has no directed path to ``outcome`` other than through ``exposure``,
      - is d-separated from ``outcome`` given ``exposure`` and the unconfounded set.

    This is a heuristic IV finder — not a full identification check.
    """
    if candidates is None:
        candidates = [n for n in dag.observed_nodes
                      if n not in (exposure, outcome)]

    desc_x = dag.descendants(exposure)
    for z in candidates:
        # relevance
        if exposure not in dag.descendants(z):
            continue
        # exclusion: no descendant of Z reaches Y except through X
        desc_z = dag.descendants(z)
        if outcome in desc_z - (desc_x | {exposure}):
            # Z has a path to Y that doesn't go through X
            continue
        # exogeneity: d-separated from Y given X
        if dag.d_separated(z, outcome, {exposure}):
            return z
    return None


def _frontdoor_set(dag, x: str, y: str, mediators: List[str]) -> Set[str]:
    """Identify a frontdoor set: mediators such that
      - all directed paths X -> Y pass through them,
      - they have no unblocked backdoor to Y,
      - no backdoor from X to them.
    """
    if not mediators:
        return set()
    # Very simple heuristic: accept the full mediator set if there are
    # no latent parents of X among the mediators' parents or children.
    fd = set(mediators)
    # Reject if any latent has both X and a mediator as children
    for p, children in dag._edges.items():
        if p.startswith("_L_"):
            if x in children and any(m in children for m in fd):
                return set()
            if y in children and any(m in children for m in fd):
                return set()
    return fd
