# LLM-Assisted Causal Discovery — Family Guide

> Closed-loop LLM × CI-test causal discovery in StatsPAI. Combines an LLM
> "domain-knowledge oracle" with constraint-based discovery (PC) and
> data-driven validation, so prior knowledge does not silently override
> what the data say.

## When to use this family

- You have **variable-name + description level** domain knowledge that an
  LLM can leverage (e.g. medical concepts, business processes,
  political-science variables).
- You have **observational data** with enough samples (rule of thumb:
  `n >> p`, at least 200 rows for ~10 variables).
- You want **per-edge confidence** combining LLM prior and data evidence
  — not just a single estimated DAG.

If you only have data → use `sp.pc_algorithm` directly. If you only have
LLM-derived knowledge with no data → use `sp.llm_dag_propose`.

## Functions

| Function | Use case |
|----------|----------|
| `sp.llm_dag_propose` | Single-shot LLM proposal from variable descriptions (no data validation). |
| `sp.llm_dag_constrained` | **Closed loop**: propose → constrained PC → CI-validate → demote → iterate. |
| `sp.llm_dag_validate` | Per-edge CI-test audit of any declared DAG. |
| `sp.pc_algorithm(..., forbidden=, required=)` | Constrained PC with background knowledge as hard edge constraints. |
| `sp.causal_mas` | Multi-agent LLM consensus (proposer / critic / domain-expert). |
| `sp.llm_causal_assess` | Benchmark an LLM's causal-reasoning accuracy. |
| `sp.pairwise_causal_benchmark` | Pairwise causal-direction benchmark. |

## Decision tree

```
Have variable descriptions?
├── No  → sp.pc_algorithm (data-only) or sp.notears
└── Yes → Have data?
         ├── No  → sp.llm_dag_propose
         └── Yes → sp.llm_dag_constrained   ← recommended
```

## Quick start: closed loop

```python
import statspai as sp
import pandas as pd

# Domain-described variables, observational data
df = pd.read_csv("clinical.csv")
descriptions = {
    "age": "patient age in years",
    "smoking": "smoking status (0/1)",
    "lung_cancer": "lung cancer diagnosis (0/1)",
    "tar": "tar deposition level (mcg)",
}

# Wire your favourite LLM as an oracle. Any callable that returns
# [(parent, child, confidence), ...] works. See docs/guides/llm_oracles.md
# for OpenAI / Anthropic / echo wrappers.
def oracle(variables, descriptions):
    # Calls an LLM and parses to (parent, child, confidence) tuples.
    return your_llm_client.propose_edges(variables, descriptions)

result = sp.llm_dag_constrained(
    df,
    variables=["age", "smoking", "lung_cancer", "tar"],
    descriptions=descriptions,
    oracle=oracle,
    alpha=0.05,
    max_iter=3,
    high_conf_threshold=0.7,
)

print(result.summary())
print(result.edge_confidence)  # per-edge LLM score + CI p-value
```

## How the loop works

```
  ┌───────────────────┐
  │ Oracle / LLM call │
  └─────────┬─────────┘
            ▼
  ┌─────────────────────────────────┐
  │ Split: high-conf -> required    │
  │        low-conf  -> forbidden*  │
  └─────────┬───────────────────────┘
            ▼
  ┌─────────────────────────────────┐
  │ pc_algorithm(forbidden, required│
  │ = background knowledge)         │
  └─────────┬───────────────────────┘
            ▼
  ┌─────────────────────────────────┐
  │ For each required edge a -> b:  │
  │   partial-corr CI test          │
  │   p > alpha → demote            │
  └─────────┬───────────────────────┘
            ▼
       any demoted?
       ├── yes → re-run constrained PC
       └── no  → converged ✓
```

`*` `forbid_low_conf=False` by default; only enable when your oracle
emits explicit non-edge claims (most LLMs only return positive edges).

## Reading the result

`LLMConstrainedDAGResult` exposes:

| Field | What it tells you |
|-------|-------------------|
| `final_edges` | Directed edges in the final CPDAG. |
| `edge_confidence` | DataFrame: `edge`, `llm_score`, `ci_pvalue`, `retained`, `source`. The `source` column distinguishes `required` (LLM-asserted, data-supported), `demoted` (LLM-asserted, data-rejected), `ci-test` (data-discovered, no LLM input), `forbidden` (LLM said no, kept out). |
| `iteration_log` | Per-iteration trace with edges demoted at each step. Use to spot oscillations. |
| `converged` | True if no demotions in the last iteration. |
| `to_dag()` | Convert to a `statspai.dag.DAG` for downstream `recommend_estimator`. |

## Combining with the rest of StatsPAI

```python
# Get a validated DAG from the loop
dag = result.to_dag()

# Use it to pick an estimator
rec = dag.recommend_estimator(exposure="smoking", outcome="lung_cancer")
print(rec.summary())        # tells you whether backdoor / IV / frontdoor

# Or feed straight into sp.causal()
w = sp.causal(df, y="lung_cancer", treatment="smoking", dag=dag)
w.report("analysis.html")
```

## Validating an existing DAG

```python
g = sp.dag("smoking -> lung_cancer; tar -> lung_cancer; age -> smoking")
v = sp.llm_dag_validate(g, df, alpha=0.05)
print(v.summary())
# DAG Edge Validation
# ============================================================
#   Alpha            : 0.05
#   Edges supported  : 2
#   Edges unsupported: 1
#     smoking -> lung_cancer  p=0.000  [OK]
#     tar -> lung_cancer      p=0.000  [OK]
#     age -> smoking          p=0.314  [REJECT]
```

`supported=True` means the data did **not** provide evidence to remove
the edge (CI test failed to reject); `supported=False` means the
implied conditional independence is consistent with the data — i.e. the
edge looks spurious.

## For Agents

- **Pre-conditions**: at least 2 numeric columns intersecting `variables`; `n_obs >> p`.
- **Common failure**: ValueError "Variable X not in data.columns" → pass only existing columns.
- **Recovery on non-convergence** (`converged=False`): inspect `iteration_log` for an oscillating edge; raise `alpha` (looser CI rejection) or lower `high_conf_threshold` (fewer forced edges). Fall back to `sp.llm_dag_propose` if the oracle is consistent enough that you don't need the loop.
- **Determinism**: with a deterministic oracle (e.g. `echo_client`), the result is bit-for-bit reproducible.
- **Cost**: one oracle call per loop iteration; budget accordingly. Default `max_iter=3` is usually sufficient — convergence in 1–2 iterations is the common case.

## Caveats

1. **Faithfulness** — partial-correlation CI tests assume linear-Gaussian
   relationships. For nonlinear DGPs, validate with caution; future
   versions will support kernel-based CI tests (HSIC).
2. **Causal sufficiency** — PC assumes no unmeasured confounders among
   `variables`. If you suspect latents, use `sp.fci` instead.
3. **Sample size** — CI tests have low power at small `n` and may fail
   to reject any edge. Prefer `n >= 500` for stable results.
4. **LLM quality matters** — a noisy oracle that emits many spurious
   edges will trigger many demotions and may not converge within
   `max_iter`. Benchmark your oracle first with `sp.llm_causal_assess`.

## References

- Spirtes, Glymour & Scheines (2000). *Causation, Prediction, and Search.*
- Kıcıman, Ness, Sharma & Tan (2023). "Causal reasoning and large language models." [arXiv:2305.00050](https://arxiv.org/abs/2305.00050).
- Long, Piché, Zantedeschi, Schuster & Drouin (2023). "Causal discovery with language models as imperfect experts." [arXiv:2307.02390](https://arxiv.org/abs/2307.02390).
- Jiralerspong, Chen, More, Shah & Bengio (2024). "Efficient Causal Graph Discovery Using Large Language Models." [arXiv:2402.01207](https://arxiv.org/abs/2402.01207).

<!-- AGENT-BLOCK-START: llm_dag_constrained -->

## For Agents

**Pre-conditions**
- data has at least 2 numeric columns intersecting `variables`
- n_obs >> number of variables (PC unstable when p ~ n)

**Identifying assumptions**
- Faithfulness (PC's CI tests reflect d-separation)
- Causal sufficiency (no unmeasured confounder among `variables`)
- Linear/Gaussian relationships (Fisher-Z partial correlation)

**Failure modes → recovery**

| Symptom | Exception | Remedy | Try next |
| --- | --- | --- | --- |
| ValueError 'Variable X not in data.columns' | `ValueError` | Pass only column names that exist in data |  |
| Loop never converges (max_iter reached) | `(none — returns converged=False)` | Inspect iteration_log for oscillating edges; raise alpha or lower high_conf_threshold | `sp.llm_dag_propose (single-shot)` |

**Alternatives (ranked)**
- `sp.sp.llm_dag_propose: single-shot LLM proposal without CI loop`
- `sp.sp.pc_algorithm: data-only PC (no LLM)`
- `sp.sp.causal_mas: multi-agent LLM consensus`

**Typical minimum N**: 200

<!-- AGENT-BLOCK-END -->

<!-- AGENT-BLOCK-START: llm_dag_validate -->

## For Agents

**Identifying assumptions**
- Faithfulness
- Linear/Gaussian (Fisher-Z)

**Failure modes → recovery**

| Symptom | Exception | Remedy | Try next |
| --- | --- | --- | --- |
| Many supported=False edges | `(none — informational)` | DAG may be misspecified; rerun discovery or check for nonlinearity / unmeasured confounders | `sp.llm_dag_constrained` |

**Typical minimum N**: 200

<!-- AGENT-BLOCK-END -->
