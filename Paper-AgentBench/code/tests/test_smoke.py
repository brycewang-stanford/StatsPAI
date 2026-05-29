"""Smoke tests — run with `pytest`. No API key needed (oracle path only)."""

from __future__ import annotations

import warnings

import causalagentbench as cab
from causalagentbench.schema import Difficulty, Design

warnings.filterwarnings("ignore")


def test_task_pack_shape():
    tasks = cab.load_tasks(n_l1=6, n_l2=6, n_l3=6)
    assert len(tasks) == 18
    assert {t.difficulty for t in tasks} == {Difficulty.L1, Difficulty.L2, Difficulty.L3}
    assert {t.design for t in tasks} == set(Design)
    # L3 tasks carry required diagnostics; L1 do not.
    l3 = [t for t in tasks if t.difficulty is Difficulty.L3]
    assert all(t.gold.required_diagnostics for t in l3)


def test_oracle_runs_and_grades():
    tasks = cab.load_tasks(n_l1=3, n_l2=3, n_l3=3)
    results = cab.run_suite(tasks, conditions=["oracle"], seeds=[0, 1], progress=False)
    assert len(results) == len(tasks) * 2
    # Oracle calls only real statspai functions -> never hallucinates,
    # always executes, always picks the gold method.
    assert all(r.m3_exec_ok for r in results)
    assert all(r.m5_hallucinated is False for r in results)
    assert all(r.m2_method_correct for r in results)


def test_materialize_deterministic():
    tasks = cab.load_tasks(n_l1=1, n_l2=0, n_l3=0)
    a = cab.materialize(tasks[0], seed_offset=0)
    b = cab.materialize(tasks[0], seed_offset=0)
    c = cab.materialize(tasks[0], seed_offset=1)
    assert a.equals(b)        # same seed -> identical data
    assert not a.equals(c)    # different rep seed -> different data


def test_staggered_did_rubric_excludes_twfe():
    """The robust-estimator trap: plain TWFE is NOT accepted on staggered DiD."""
    from causalagentbench import metrics
    from causalagentbench.schema import Trajectory
    task = [t for t in cab.load_tasks(n_l1=0, n_l2=0, n_l3=6)
            if t.design is Design.STAGGERED_DID][0]
    twfe = Trajectory(final_method="twfe", final_estimate=0.5)
    cs = Trajectory(final_method="callaway_santanna", final_estimate=0.5)
    assert metrics.m2_method_correct(task, twfe) is False
    assert metrics.m2_method_correct(task, cs) is True


def test_hypothesis_machinery_runs_without_llm_cells():
    tasks = cab.load_tasks(n_l1=2, n_l2=2, n_l3=2)
    results = cab.run_suite(tasks, conditions=["oracle"], seeds=[0, 1], progress=False)
    H = cab.test_hypotheses(results, B=99)
    assert H["H1"]["evaluable"] is False  # needs C1,C2
    assert H["_meta"]["conditions_present"] == ["oracle"]
