# CausalAgentBench

**A behavioural benchmark for LLM agents that *do* causal inference — not just talk about it.**

CausalAgentBench puts an LLM agent in front of a real dataset and a causal
research question, lets it choose an identification strategy, run diagnostics,
write and execute estimation code, and report a number — then grades the whole
trajectory against a **known true effect**. It is the code-level implementation
of **Track D** of the [StatsPAI](https://github.com/brycewang-stanford/StatsPAI)
JSS programme, and is designed to live as a sub-module of StatsPAI under
[`Paper-AgentBench/`](https://github.com/brycewang-stanford/StatsPAI/tree/main/Paper-AgentBench).

> Status: research scaffold (v0.1). The 50-prompt frozen task set and the
> 900-trial production run are **pre-registered on OSF**; the methodology in
> this README follows that pre-registration verbatim
> (`StatsPAI/Paper-AgentBench/manuscript/notes/osf-preregistration.md`).
> The code here is the harness you use to develop, dry-run, and extend it.

---

## Table of contents

1. [Why another causal benchmark?](#1-why-another-causal-benchmark)
2. [Landscape: what already exists](#2-landscape-what-already-exists)
3. [Where CausalAgentBench sits (the delta)](#3-where-causalagentbench-sits-the-delta)
4. [Benchmark design (Track D)](#4-benchmark-design-track-d)
5. [Install & quick start](#5-install--quick-start)
6. [Architecture](#6-architecture)
7. [How it integrates with StatsPAI](#7-how-it-integrates-with-statspai)
8. [Mounting into `StatsPAI/Paper-AgentBench`](#8-mounting-into-statspaipaper-agentbench)
9. [Roadmap](#9-roadmap)
10. [References](#10-references)

---

## 1. Why another causal benchmark?

Most "causal benchmarks" test one of two things:

- **Can a model *reason* about causality in text?** (CLadder, Corr2Cause,
  CausalBench, CausalPitfalls …) — no data, no code, no estimate.
- **Can a model *produce the right number* in one shot?** (QRData,
  Econometrics-Agent, CausalReasoningBenchmark) — static, single-turn.

Neither measures what an analyst actually does: **pick a design → test its
assumptions on the data → revise when they fail → estimate → report how fragile
the conclusion is → and know when to refuse**. CausalAgentBench measures that
*process*, on data with a **known ground-truth effect**, across a controlled
3×2 grid of tool stacks and LLM providers.

---

## 2. Landscape: what already exists

A condensed map of the field (full citations in [`references.md`](references.md)).

### A. Text-only causal *reasoning* (no data, no estimation)

| Benchmark | What it tests | Note |
|---|---|---|
| **CLadder** | Pearl's ladder (assoc./intervention/counterfactual) in NL | 10K Q; most complete reasoning bench |
| **Corr2Cause** | infer causal graph from correlations | GPT-4 ≈ F1 0.29 |
| **CausalBench** (SIGHAN'24) | text + math + code, 4 query views | name collides with a gene-network bench |
| **CausalPitfalls** ("Ice Cream…") | detect confounding / collider / bad-control / Simpson | text scenarios, MCQ-style |
| **causAIign** | collider reasoning vs human baseline | human comparison |
| **ReCITE** | extract causal relations from real papers | real-world text |

### B. Data-driven statistical / causal QA (data attached, must compute)

| Benchmark | Design | Scale / result |
|---|---|---|
| **QRData** | data sheet + stat/causal Q; tests CoT/PoT/ReAct/code-interp | 411 Q; GPT-4 58% vs human 76%; causal hardest |
| **Causal Agent (CausalTQA)** | 4 levels: variable/edge/graph/effect; agent uses causal-learn + EconML | ~1.4K Q; exact-match |
| **BLADE** | data + research question → variables/transforms/models vs expert ground truth | analysis-decision quality |
| **StatQA** | statistical method selection + applicability | 11.6K Q; GPT-4o 64.8% |
| **CauSciBench** | full pipeline: formulation→variables→method→implement→interpret | o3 real-data MRE 53%; finds LLMs over-use OLS |
| **InterveneBench** | identification + study design with NO given causal graph (744 social-science studies) | GPT-5.1 method-selection 49% |

> `references.md` has the exhaustive, annotated list (B–H categories) incl.
> DiscoveryBench, CORE-Bench, ReplicatorBench, Mining-Causality/IV, the CATE
> half-synthetic tradition (IHDP/Twins/ACIC), and the Curth et al. critique of
> CATE benchmarking — read it before finalising the scoring protocol.

### C. Econometrics / economics agents

| Benchmark | Design | Scale / scoring |
|---|---|---|
| **Econometrics-Agent** | replicate expert tasks (OLS/PSM/IV/DiD/RDD) | 63 tasks; 1%/5%/sign scoring |
| **EconEvals** | procurement / scheduling / pricing decisions | in-context economic decision-making |

### D. Autonomous causal-analysis **systems** (reference baselines, not benches)

**Causal-Copilot** (end-to-end discovery→inference→HPO, 20+ methods),
**Auto-Bench** (interactive causal-graph discovery), **CAMO** (multi-agent
discovery), **MRAgent** (Mendelian-randomisation discovery).

### E. Neighbouring data-science agent benches

**DSBench**, **DABStep** (450+ tasks, best agent 14.6% on hard), **InfiAgent-DABench**,
**IDA-Bench**, **DataSciBench** — multi-step execution harnesses worth borrowing
from, but not causal-specific.

### F. The closest prior work — and our reference point

**CausalReasoningBenchmark** (CRB; Sawarni, Tan & Syrgkanis, Stanford, 2026,
[arXiv:2602.20571](https://arxiv.org/abs/2602.20571)). 173 queries over 138
real datasets from 85 papers + 4 textbooks. Its key, excellent idea:
**disentangle identification from estimation** and score them separately
(strategy ≈ 84% correct, full identification spec only ≈ 30%). Designs: IV, RD,
DiD, selection-on-observables, RCT.

---

## 3. Where CausalAgentBench sits (the delta)

CRB is the state of the art for **static, single-shot, real-world** causal
evaluation. CausalAgentBench is deliberately **orthogonal** on four axes that
CRB's design cannot reach (per CRB §Limitations and our read of the paper):

| Axis | CRB (static) | **CausalAgentBench** |
|---|---|---|
| **Form** | one-shot: prompt → spec + number | **agentic**: multi-turn, code sandbox, tool use, revise-on-failure |
| **Ground truth** | gold = another paper's estimate (no true ATE in real data) | **known-truth DGP** → score against the *true* effect; contamination-proof, scalable |
| **Assumptions** | scores whether the *named* design is right | scores whether the agent **runs and interprets the diagnostics** (parallel trends, weak-IV F, McCrary, overlap) — M6 |
| **Refusal** | every query has a gold solution | includes **unidentifiable traps**: correct answer is "not identifiable / refuse" (planned L4) |
| **Robust-estimator trap** | DiD treated as one design | **staggered DiD rubric rejects plain TWFE** — must reach for Callaway–Sant'Anna / Sun–Abraham (M2) |

**Positioning:** CausalAgentBench is the *dynamic / agentic / known-truth
companion* to CRB — we cite it, reuse its identification-vs-estimation split as
two of our metrics (M1 estimation, M2 method), and add the process, ground-truth,
and tool-stack axes on top. We do **not** compete on real-world replication
scale.

---

## 4. Benchmark design (Track D)

Frozen in the OSF pre-registration. The package implements all of it; the
*frozen 50 prompts + gold files* are deposited separately on OSF, while the
[`tasks`](causalagentbench/tasks.py) module ships a schema-identical,
**expandable demonstration pack** built from StatsPAI's known-truth DGPs.

### Task set — 50 prompts × 3 difficulty tiers
- **L1 (direct, ×20)** — the method is named; pure estimation.
- **L2 (indirect, ×20)** — identification described, agent picks the method.
- **L3 (workflow, ×10)** — full pipeline incl. diagnostics + robustness;
  may be *unidentifiable* (refusal is the correct answer).

### Conditions — 3×2 factorial (C1…C6)

|                | Claude | GPT |
|----------------|:------:|:---:|
| StatsPAI + MCP | C1 | C2 |
| Pythonic stack (statsmodels, linearmodels, DoubleML, EconML, …) | C3 | C4 |
| R via MCP (MatchIt, did, fixest, rdrobust, synthdid, HonestDiD, …) | C5 | C6 |

Plus a non-pre-registered **`oracle`** cell: a deterministic StatsPAI pipeline
(no LLM) — the upper-bound calibration baseline (hallucination = 0 by
construction). **50 × 6 × 3 seeds = 900 trials**, `temperature = 0`.

### Metrics

| | metric | definition |
|---|---|---|
| **M1** | task success | final estimate within ±5% of gold |
| **M2** | method correctness | gold-rubric estimator (LLM-judge + 20% human) |
| **M3** | code-execution success | code runs without unhandled exception |
| **M4** | token efficiency | median input+output tokens / trial |
| M5 | hallucination | calls a non-existent function (machine-checkable for StatsPAI stack) |
| M6 | diagnostic completeness | share of required diagnostics reported |
| M7 | reproducibility | across-seed variance of the estimate |
| M8 | time-to-result | median wall-clock / trial |

### Pre-registered hypotheses (cluster bootstrap on prompt, B=9,999, Bonferroni α=0.01)

- **H1** StatsPAI success ≥ 90% (L1) / 70% (L2) / 50% (L3)
- **H2** StatsPAI − Pythonic success ≥ 15 pp on L2–L3
- **H3** StatsPAI hallucination < 5%; Pythonic > 15%
- **H4** StatsPAI uses ≤ 60% of Pythonic tokens
- **H5** R-via-MCP ≈ StatsPAI on success but ≥ 1.5× tokens

All implemented in [`stats.test_hypotheses`](causalagentbench/stats.py); cells
that need an absent condition report `evaluable=False` instead of erroring.

---

## 5. Install & quick start

```bash
cd CausalAgentBench
pip install -e .            # core: runs the no-LLM oracle path
pip install -e ".[llm]"     # + anthropic/openai for the C1..C6 agent cells
```

Run the whole pipeline with **no API key** (StatsPAI reference oracle):

```bash
cab tasks                                   # list the demo task pack
cab run --conditions oracle --out runs/oracle.jsonl
cab analyze runs/oracle.jsonl               # summary table + H1..H5
```

```python
import causalagentbench as cab

tasks   = cab.load_tasks(n_l1=4, n_l2=4, n_l3=4)
results = cab.run_suite(tasks, conditions=["oracle"], seeds=[0, 1, 2])
print(cab.summarize(results))
print(cab.test_hypotheses(results))
```

Run real agents (needs keys):

```bash
export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...
cab run --conditions C1 C3 C5 --seeds 0 1 2 --out runs/claude.jsonl
cab analyze runs/claude.jsonl
```

### Verified oracle dry-run (this repo, no keys)

```
cell             n  M1 succ  M2 meth  M3 exec  M5 hal  M6 diag   M4 tok    M8 s
-------------------------------------------------------------------------------
oracle/L1       18     0.44     1.00     1.00    0.00     1.00        0    0.00
oracle/L2       18     0.61     1.00     1.00    0.00     1.00        0    0.00
oracle/L3       18     0.33     1.00     1.00    0.00     1.00        0    0.00
```

The oracle always picks the gold method (M2=1.00), always executes (M3=1.00),
never hallucinates (M5=0.00). Its M1 < 1.0 is **honest finite-sample
behaviour**: the ±5% band is strict, and high-variance designs (RD, DiD) miss
it even with the correct estimator at moderate N — a realistic ceiling for the
LLM cells, not a scoring bug. (All six estimators converge to the true 0.5 as
N→∞; verified.)

---

## 6. Architecture

```
causalagentbench/
├── schema.py        Task / Gold / Trajectory / TrialResult  (+ Difficulty, Design enums)
├── tasks.py         known-truth task generator (statspai dgp_* + canonical datasets)
├── conditions.py    C1..C6 + oracle; Pythonic/R stack allow-lists
├── adapters/
│   ├── base.py            AgentAdapter ABC + uniform estimate extractor
│   ├── statspai_oracle.py deterministic StatsPAI pipeline (no LLM)
│   └── llm.py             Claude/GPT driver + subprocess code sandbox
├── metrics.py       M1..M8
├── scoring.py       grade(trajectory) -> TrialResult
├── stats.py         cluster bootstrap, Bonferroni, H1..H5, summary table
├── runner.py        run_suite (task × condition × seed) -> JSONL
└── cli.py           `cab tasks | run | analyze`
```

Every layer is pure-Python and import-cheap; the LLM SDKs load lazily so the
oracle path needs only `statspai + numpy + pandas`.

---

## 7. How it integrates with StatsPAI

CausalAgentBench is **built on `statspai`, not merely adjacent to it.** Three
hard dependencies:

1. **Known-truth data.** Tasks materialise from StatsPAI's own DGPs
   (`sp.dgp_rct/dgp_did/dgp_rd/dgp_iv/dgp_observational`, the same six scenarios
   behind `sp.verify_benchmark`) and canonical datasets (`mpdta`, `card_1995`,
   `nsw_lalonde`, `lee_2008_senate`, …). Because the DGP fixes the effect, gold
   is the *true* causal parameter.
2. **The C1/C2 tool surface.** The StatsPAI condition exposes the agent-native
   API — `sp.detect_design`, `sp.recommend`, `sp.preflight`, `sp.audit`,
   `sp.sensitivity` — via `statspai.agent.mcp_server`. The oracle and the M5
   hallucination check both validate calls against the live `statspai` namespace.
3. **The reference estimators.** The oracle dispatches to `sp.regress`,
   `sp.did`, `sp.callaway_santanna`, `sp.rdrobust`, `sp.ivreg` and reads the
   uniform `CausalResult.estimate` / `EconometricResults.params`. The same path
   freezes gold for canonical-dataset tasks.

This is the experiment that turns StatsPAI's "agent-native" claim into a
falsifiable measurement (hypotheses H1–H5).

---

## 8. Mounting into `StatsPAI/Paper-AgentBench`

This repo is structured to drop straight in as the **code home** beside the
existing companion-paper workspace (`README.md`, `archive-from-jss/`,
`manuscript/`). Recommended layout:

```
StatsPAI/Paper-AgentBench/
├── README.md                 (existing paper-workspace readme)
├── manuscript/               (existing)
├── archive-from-jss/         (existing)
└── causalagentbench/         ← this package  (+ pyproject.toml, tests/, examples/)
```

```bash
# from the StatsPAI repo root
cp -r /path/to/CausalAgentBench/{causalagentbench,tests,examples,pyproject.toml,README.md,references.md} \
      Paper-AgentBench/code/      # or keep at Paper-AgentBench/ root
cd Paper-AgentBench/code && pip install -e ".[dev]" && pytest -q
```

> ⚠️ The pre-registration treats CausalAgentBench's protocol as **frozen on
> OSF**. Keep code changes (estimator dispatch, harness, new DGPs) separate from
> protocol changes (task counts, metrics, hypotheses); the latter require an OSF
> amendment and a disclosed deviation in the JSS manuscript. The frozen 50
> prompts + gold files belong in the StatsPAI-JSS-replication archive at the SHA
> recorded in the OSF deposit, not inlined here.

---

## 9. Roadmap

- [ ] **L4 unidentifiable traps** — tasks with no valid design; score refusal calibration / over-confidence.
- [ ] **Real MCP tool-loop** for C1/C2 (wire `statspai.agent.mcp_server`) instead of the single-turn code protocol.
- [ ] **Counterfactual-perturbation pairs** — same task ± staggered timing, to test whether design choice is causal-reasoned vs pattern-matched.
- [ ] **Sensitivity-as-output** — require + score a sensitivity statistic (Oster δ / E-value / `HonestDiD`) against the DGP's known fragility.
- [ ] Freeze the 50-prompt set + gold files; deposit on OSF; wire canonical-dataset gold via `freeze_reference_gold()`.
- [ ] R-via-MCP executor (`Rscript` path is stubbed in `adapters/llm.py`).

---

## 10. References

See [`references.md`](references.md) for the full annotated bibliography
(arXiv IDs, GitHub repos, HF datasets) of every benchmark cited above.

## License & citation

Code: MIT. Task prompts (when frozen): CC-BY-4.0; gold files: MIT.
Author: Biaoyue (Bryce) Wang — Stanford REAP / CoPaper.AI.
Cite via the StatsPAI `CITATION.cff` and the OSF pre-registration DOI.
