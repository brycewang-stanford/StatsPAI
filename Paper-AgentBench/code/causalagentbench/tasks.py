"""Task generation for CausalAgentBench.

Tasks are built on top of StatsPAI's *known-truth* data-generating
processes (``sp.dgp_rct``, ``sp.dgp_did``, ``sp.dgp_rd``, ``sp.dgp_iv``,
``sp.dgp_observational``) and its canonical datasets. Because the DGP
fixes the true effect, the gold answer is the *true* causal effect — not
another paper's estimate. This is the methodological wedge that
distinguishes a known-truth agent benchmark from real-world replication
benchmarks, where the "gold" is itself an estimate and cannot certify
estimation accuracy.

This module produces a *demonstration task pack* that is schema-identical
to, and expandable to, the pre-registered 50-prompt frozen set
(L1 x 20, L2 x 20, L3 x 10). The frozen 50 prompts and their gold files
live in the OSF replication archive; this generator is what you use to
develop, smoke-test, and extend the harness.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .schema import Design, Difficulty, Gold, Task

# ---------------------------------------------------------------------------
# Per-design knowledge: how to frame the question, which diagnostics a
# correct L3 workflow must run, and which methods count as correct (M2).
# ---------------------------------------------------------------------------

_DESIGN_INFO: Dict[Design, Dict] = {
    Design.RCT: dict(
        named_method="a difference-in-means / OLS estimator",
        id_story="treatment was randomly assigned",
        diagnostics=["balance", "attrition"],
        accepted=["regress", "ols", "difference_in_means", "ttest", "rct"],
    ),
    Design.DID: dict(
        named_method="difference-in-differences",
        id_story="a policy switched on for a treated group at a known date "
        "while a comparison group was never treated",
        diagnostics=["parallel_trends", "pretrends", "event_study"],
        accepted=["did", "twfe", "callaway_santanna", "drdid", "did_2x2"],
    ),
    Design.STAGGERED_DID: dict(
        named_method="a heterogeneity-robust staggered DiD estimator",
        id_story="units adopt treatment at different dates (staggered "
        "rollout) and effects may be heterogeneous across cohorts",
        diagnostics=["parallel_trends", "pretrends", "event_study",
                     "negative_weights_check"],
        # NB: plain TWFE is intentionally NOT accepted here — under
        # staggered timing + heterogeneous effects it is biased
        # (Goodman-Bacon). This is the "robust-estimator" trap (M2).
        accepted=["callaway_santanna", "sun_abraham", "did_imputation",
                  "etwfe", "bjs", "gardner_did", "stacked_did"],
    ),
    Design.RD: dict(
        named_method="a local-linear regression discontinuity estimator",
        id_story="treatment is assigned by whether a running variable "
        "crosses a known cutoff",
        diagnostics=["mccrary_density", "bandwidth_sensitivity",
                     "covariate_continuity"],
        accepted=["rdrobust", "rd", "local_linear", "rdd"],
    ),
    Design.IV: dict(
        named_method="two-stage least squares (2SLS)",
        id_story="an instrument shifts treatment but is excludable from the "
        "outcome equation",
        diagnostics=["first_stage_F", "weak_iv"],
        accepted=["ivreg", "iv", "2sls", "tsls", "liml"],
    ),
    Design.OBSERVATIONAL: dict(
        named_method="a selection-on-observables estimator",
        id_story="treatment is not randomised but is unconfounded given the "
        "recorded covariates",
        diagnostics=["overlap", "balance", "sensitivity_unobserved"],
        accepted=["psm", "matching", "ipw", "dml", "aipw",
                  "regression_adjustment", "causal_forest", "drlearner"],
    ),
}

# Synthetic scenario specs mirror statspai.smart.benchmark._SCENARIOS so the
# harness stays in lock-step with the package's own known-truth DGPs.
_SCENARIOS: Dict[Design, Dict] = {
    # Sample sizes follow Track B's N in {500, 5000}; the demo pack uses the
    # larger end so the no-LLM oracle reads as a clean "right method recovers
    # the truth" reference rather than being swamped by finite-sample noise
    # against the tight +/-5% band.
    Design.RCT: dict(dgp="dgp_rct",
                     kwargs=dict(n=4000, effect=0.5, n_covariates=2),
                     roles=dict(y="y", treatment="treatment"),
                     true_effect=0.5, estimand="ATE"),
    Design.DID: dict(dgp="dgp_did",
                     kwargs=dict(n_units=600, n_periods=2, effect=0.5),
                     roles=dict(y="y", treatment="treated", id="unit", time="time"),
                     true_effect=0.5, estimand="ATT"),
    Design.STAGGERED_DID: dict(dgp="dgp_did",
                               kwargs=dict(n_units=600, n_periods=5, effect=0.5,
                                           staggered=True),
                               roles=dict(y="y", treatment="treated", id="unit", time="time"),
                               true_effect=0.5, estimand="ATT"),
    Design.RD: dict(dgp="dgp_rd",
                    kwargs=dict(n=5000, effect=0.5),
                    roles=dict(y="y", running_var="x", cutoff=0.0),
                    true_effect=0.5, estimand="LATE"),
    Design.IV: dict(dgp="dgp_iv",
                    kwargs=dict(n=4000, effect=0.5, first_stage=0.7),
                    roles=dict(y="y", treatment="treatment", instrument="instrument"),
                    true_effect=0.5, estimand="LATE"),
    Design.OBSERVATIONAL: dict(dgp="dgp_observational",
                               kwargs=dict(n=4000, effect=0.5, confounding=0.2),
                               roles=dict(y="y", treatment="treatment",
                                          covariates=["x1", "x2"]),
                               true_effect=0.5, estimand="ATE"),
}


def _question(design: Design, diff: Difficulty, roles: Dict) -> str:
    info = _DESIGN_INFO[design]
    y = roles.get("y", "the outcome")
    t = roles.get("treatment", roles.get("running_var", "the treatment"))
    if diff is Difficulty.L1:
        return (
            f"Estimate the average causal effect of `{t}` on `{y}` using "
            f"{info['named_method']}. The data are attached as a CSV. "
            f"Report a single point estimate."
        )
    if diff is Difficulty.L2:
        return (
            f"In the attached data, {info['id_story']}. Estimate the causal "
            f"effect of `{t}` on `{y}`. Choose and justify an appropriate "
            f"identification strategy, then report a single point estimate."
        )
    # L3
    return (
        f"In the attached data, {info['id_story']}. Produce a complete causal "
        f"analysis of the effect of `{t}` on `{y}`: (1) state and justify your "
        f"identification strategy; (2) run the standard diagnostic checks for "
        f"that design; (3) report the point estimate and its standard error; "
        f"(4) assess robustness. If the effect is not identifiable from these "
        f"data, say so explicitly instead of reporting a number."
    )


def load_tasks(
    n_l1: int = 12,
    n_l2: int = 12,
    n_l3: int = 6,
    designs: Optional[List[Design]] = None,
) -> List[Task]:
    """Build a demonstration task pack.

    Defaults give a 30-task pack (12/12/6). Set ``n_l1=20, n_l2=20,
    n_l3=10`` to match the pre-registered 50-prompt shape once the frozen
    gold files are wired in. Tasks are deterministic given their seed.
    """
    designs = designs or list(_SCENARIOS.keys())
    tasks: List[Task] = []

    def _make(diff: Difficulty, k: int) -> List[Task]:
        out = []
        for i in range(k):
            design = designs[i % len(designs)]
            spec = _SCENARIOS[design]
            info = _DESIGN_INFO[design]
            roles = dict(spec["roles"])
            seed = 1000 * list(Difficulty).index(diff) + i
            diags = info["diagnostics"] if diff is Difficulty.L3 else []
            gold = Gold(
                point_estimate=float(spec["true_effect"]),
                design=design,
                estimand=spec["estimand"],
                source="dgp_true_effect",
                required_diagnostics=diags,
                accepted_methods=info["accepted"],
                notes=f"Synthetic {design.value} DGP; true effect is exact.",
            )
            tasks_id = f"{design.value}-{diff.value}-{i:02d}"
            out.append(Task(
                task_id=tasks_id,
                difficulty=diff,
                design=design,
                question=_question(design, diff, roles),
                roles=roles,
                gold=gold,
                data_spec=dict(kind="dgp", dgp=spec["dgp"],
                               kwargs=spec["kwargs"], seed=seed),
                tags=[design.value, diff.value, "synthetic"],
            ))
        return out

    tasks += _make(Difficulty.L1, n_l1)
    tasks += _make(Difficulty.L2, n_l2)
    tasks += _make(Difficulty.L3, n_l3)
    return tasks


def materialize(task: Task, seed_offset: int = 0) -> pd.DataFrame:
    """Realise a task's data. Deterministic given (task seed, seed_offset).

    ``seed_offset`` is the per-replication seed (Track D runs 3 reps per
    cell). It is *added* to the task's base seed so reps differ but stay
    reproducible.
    """
    import statspai as sp

    spec = task.data_spec
    if spec.get("kind") != "dgp":
        raise NotImplementedError(
            f"Only synthetic DGP tasks are materialised here; got "
            f"{spec.get('kind')!r}. Canonical-dataset tasks load via "
            f"statspai.datasets and freeze their gold separately."
        )
    dgp_fn = getattr(sp, spec["dgp"], None)
    if dgp_fn is None:  # pragma: no cover - guards against API drift
        raise AttributeError(
            f"statspai has no DGP '{spec['dgp']}'. Available dgp_* fns "
            f"changed; update tasks._SCENARIOS."
        )
    seed = int(spec.get("seed", 0)) + int(seed_offset)
    try:
        return dgp_fn(seed=seed, **spec["kwargs"])
    except TypeError:
        # Some DGPs may use ``random_state`` rather than ``seed``.
        return dgp_fn(random_state=seed, **spec["kwargs"])
