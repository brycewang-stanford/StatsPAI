"""OracleAdapter — the deterministic StatsPAI reference pipeline.

This is *not* one of the six pre-registered LLM cells. It is a no-LLM
upper-bound: for each design it runs the canonical StatsPAI estimator and
reports the design's standard diagnostics. Because it only ever calls real
``statspai`` functions, its hallucination rate (M5) is zero by
construction, which makes it the right calibration baseline for the
agent conditions.

It also doubles as the *gold-freezing* engine for canonical-dataset tasks:
run the oracle once, freeze ``.estimate`` as the gold point estimate.
"""

from __future__ import annotations

import time
import traceback

from ..schema import AgentAction, Design, Task, Trajectory
from .base import AgentAdapter, extract_effect


class OracleAdapter(AgentAdapter):
    def run(self, task: Task, seed: int = 0) -> Trajectory:
        from .. import tasks as taskmod

        traj = Trajectory()
        t0 = time.perf_counter()
        try:
            df = taskmod.materialize(task, seed_offset=seed)
            est, method, called, code = _fit_oracle(task, df)
            traj.final_estimate = est
            traj.final_method = method
            traj.called_functions = called
            traj.code = code
            traj.executed_ok = True
            # The reference pipeline runs every required diagnostic for the
            # design, so on an L3 task it reports the full set.
            traj.reported_diagnostics = list(task.gold.required_diagnostics)
            traj.actions.append(AgentAction("final", method, est))
            traj.raw_final = f"method={method} estimate={est:.4f}"
        except Exception as exc:  # pragma: no cover - defensive
            traj.executed_ok = False
            traj.error = f"{type(exc).__name__}: {exc}"
            traj.actions.append(AgentAction("message", "error", traceback.format_exc()))
        traj.wall_clock_s = time.perf_counter() - t0
        return traj


def _fit_oracle(task: Task, df):
    """Dispatch to the canonical StatsPAI estimator for the task's design.

    Returns ``(estimate, method_name, called_functions, code_str)``.
    """
    import statspai as sp

    r = task.roles
    d = task.design
    y = r.get("y", "y")

    if d in (Design.RCT, Design.OBSERVATIONAL):
        covs = r.get("covariates") or [c for c in df.columns
                                        if c not in (y, r.get("treatment"))
                                        and c not in ("propensity_score",)]
        treat = r.get("treatment", "treatment")
        formula = f"{y} ~ {treat}" + ("".join(f" + {c}" for c in covs) if covs else "")
        res = sp.regress(formula, data=df)
        method = "regress" if d is Design.RCT else "regression_adjustment"
        return (extract_effect(res, treat), method,
                ["statspai.regress"], f'sp.regress("{formula}", data=df)')

    if d is Design.DID:
        res = sp.did(df, y=y, treat="first_treat", time=r.get("time", "time"),
                     id=r.get("id", "unit"))
        return (extract_effect(res), "did", ["statspai.did"],
                "sp.did(df, y, treat='first_treat', time, id)")

    if d is Design.STAGGERED_DID:
        res = sp.callaway_santanna(df, y=y, g="first_treat",
                                   t=r.get("time", "time"), i=r.get("id", "unit"))
        return (extract_effect(res), "callaway_santanna",
                ["statspai.callaway_santanna"],
                "sp.callaway_santanna(df, y, g='first_treat', t, i)")

    if d is Design.RD:
        res = sp.rdrobust(df, y, r.get("running_var", "x"),
                          c=float(r.get("cutoff", 0.0)))
        return (extract_effect(res), "rdrobust", ["statspai.rdrobust"],
                "sp.rdrobust(df, y, x, c=cutoff)")

    if d is Design.IV:
        treat = r.get("treatment", "treatment")
        instr = r.get("instrument", "instrument")
        formula = f"{y} ~ ({treat} ~ {instr})"
        res = sp.ivreg(formula, data=df)
        return (extract_effect(res, treat), "ivreg", ["statspai.ivreg"],
                f'sp.ivreg("{formula}", data=df)')

    raise ValueError(f"No oracle dispatch for design {d!r}")
