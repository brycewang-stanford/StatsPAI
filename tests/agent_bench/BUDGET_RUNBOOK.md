# CausalAgentBench production run: runbook

This runbook does not authorise spending. Every step below that touches an
API or OSF is Bryce's.

## Decisions to make before the deposit

1. **Models.** One Anthropic and one OpenAI release, written as exact IDs
   into the lock. A current Anthropic choice is `claude-opus-5-5` (the
   default Claude model); the OpenAI ID must be copied from OpenAI's model
   documentation on the deposit day.
2. **Anthropic effort.** Pinned in the lock (default in `prereg.py`:
   `high`). Claude Opus 5.5 defaults to `medium` if unset, so it is never
   left implicit.
3. **H4 and the MCP tool definitions.** The StatsPAI cells carry about
   17,500 tokens of MCP tool definitions per turn, the others about 450.
   Cached, they cost little, but H4 as written counts every token processed.
   Options: keep H4 on all tokens (most conservative; likely not
   supported), move H4 to `tokens_uncached`, or add cells that use StatsPAI
   from Python without the MCP tools. Recommendation: keep the primary H4
   on all tokens and pre-register `tokens_uncached` as a named secondary
   measure, so the result is reported either way.
4. **Repetitions and turn limit.** Defaults: 3 reps, 20 turns, 300 s per
   code call.

## Steps

1. Estimate the bill:
   `python runners/estimate_budget.py --openai-input <$> --openai-output <$> --openai-cached <$>`
   (prices per million tokens from the provider pages). Set hard caps in
   both billing dashboards.
2. Check the design is complete:
   `python runners/build_data.py --check`, `pytest ../agent_eval/test_agent_bench_harness.py`,
   `python runners/runner.py --model oracle --cells C1` (50/50).
3. `python runners/prereg.py fingerprint`; deposit `prompts/_protocol.md`
   and the fingerprints on OSF; note the registration DOI.
4. `python runners/prereg.py lock --doi <DOI> --anthropic-model <id> --openai-model <id> --anthropic-effort <level>`,
   then commit `prompts/preregistration.lock.json`.
5. Export `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`; run
   `python runners/runner.py --model api --cells C1,C2,C3,C4,C5,C6 --tasks all`.
   The runner verifies the lock before every model is built and refuses if
   any fingerprinted file changed.
6. `python runners/grader.py --in results/trials.jsonl` writes
   `results/scores.csv` and `results/summary.json` (H1-H5).
