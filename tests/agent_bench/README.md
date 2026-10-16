# CausalAgentBench v2

A pre-registered benchmark of LLM agents on 50 applied-econometrics tasks,
in six cells: {StatsPAI, Python stack, R} x {Anthropic, OpenAI}. The
protocol is `prompts/_protocol.md`; the production run is gated on its OSF
deposit (`BUDGET_RUNBOOK.md`).

## Layout

```text
prompts/_protocol.md        pre-registration text
prompts/prompts.json        50 tasks with named report fields (build_tasks.py)
data/                       frozen data files + MANIFEST.json (build_data.py)
golds/reference.R           R reference computations -> reference_values.json
golds/golds.json            acceptable variants and tolerances (build_golds.py)
runners/cells.py            toolsets, system prompt, tool schemas
runners/sandbox.py          persistent Python / R sessions per trial
runners/agent_loop.py       one trial: model <-> tools until submit_answer
runners/models.py           ScriptedModel, OracleModel, Anthropic, OpenAI
runners/grader.py           per-trial scores, H1-H5 cluster-bootstrap tests
runners/prereg.py           pre-registration lock (required for API runs)
runners/runner.py           (cell, task, rep) -> results/trials.jsonl
runners/oracle_statspai.py  StatsPAI reference solutions to all 50 tasks
```

## Rebuild and check (no API)

```bash
cd tests/agent_bench
python runners/build_data.py --check        # data bytes match the manifest
Rscript golds/reference.R                    # reference values (R 4.5)
python runners/build_golds.py
python runners/runner.py --model oracle --cells C1 --tasks all
python runners/grader.py --in results/trials.jsonl
pytest ../agent_eval/test_agent_bench_harness.py
```

The oracle run replays StatsPAI's reference solutions through the full loop;
all 50 score full marks. It is the harness check, not a result.

## Production run

Only after the OSF deposit and `python runners/prereg.py lock ...`; see
`BUDGET_RUNBOOK.md`.
