# Causal MAS — Multi-Agent LLM Causal Discovery

> arXiv:2509.00987 (September 2025): *Causal MAS — A Survey of Large
> Language Model Architectures for Discovery and Effect Estimation.*

## 1. Why multi-agent?

Single-prompt LLM causal-discovery pipelines (Kiciman-Sharma 2025) are
brittle: one model run produces one DAG, and you have no audit trail
to tell a confident but wrong edge from a cautious correct one.

Causal MAS flips the architecture:

```
  Proposer  ──► list of candidate edges
  Critic    ──► rejects edges violating time-order / exclusion / DAG axioms
  Domain    ──► endorses edges consistent with substantive knowledge
  Synthesiser ─► accumulates per-edge votes over N rounds
```

After `N` rounds every edge has a confidence in `[0, 1]`: the fraction
of rounds in which it survived critic rejection plus a half-point per
domain endorsement.  The final DAG keeps edges above
`final_threshold`.

## 2. Quick-start (offline, no API key)

```python
import statspai as sp

res = sp.causal_llm.causal_mas(
    variables=['age', 'sex', 'income', 'treatment', 'mortality'],
    domain='observational mortality cohort',
    rounds=3,
    final_threshold=0.5,
)
print(res.summary())
print(res.to_dag_string())
```

The default backend is the **deterministic heuristic** shipped with
StatsPAI — it uses the same variable-name pattern matcher as
`sp.causal_llm.llm_dag_propose`.  You get reproducible output with no
API key, no network, and no surprises.

## 3. Plugging in a real LLM

```python
# OpenAI-compatible (default gpt-4o-mini, set OPENAI_API_KEY)
openai_cli = sp.causal_llm.openai_client(
    model='gpt-4o-mini',
    temperature=0.0,     # deterministic for reproducibility
)

# Anthropic (default claude-opus-4-7, set ANTHROPIC_API_KEY)
claude_cli = sp.causal_llm.anthropic_client(
    model='claude-opus-4-7',
    temperature=0.0,
)

res = sp.causal_llm.causal_mas(
    variables=['age', 'sex', 'treatment', 'ldl', 'mi'],
    domain='cardiovascular prevention in type-2 diabetes',
    client=claude_cli,
    rounds=4,
)
```

Both clients expose the `LLMClient.chat(role, prompt) -> str`
interface, so you can supply any other model by wrapping it yourself:

```python
class MyClient(sp.causal_llm.LLMClient):
    name = 'my-local-llm'
    def chat(self, role, prompt):
        return my_inference_server(role, prompt)  # whatever you have
```

## 4. Claude extended thinking

Claude 4.5 / Opus 4.7 support extended thinking — the model reasons
privately for a configurable token budget before producing its public
answer.  For causal discovery this frequently lifts edge accuracy on
ambiguous domains (Kiciman-Sharma 2025 §5).

```python
claude_thinking = sp.causal_llm.anthropic_client(
    model='claude-opus-4-7',
    thinking_budget=4096,     # reasoning tokens
    max_tokens=8192,          # total (must exceed thinking_budget)
)
res = sp.causal_llm.causal_mas(
    variables=[...], client=claude_thinking, rounds=3,
)

# Inspect the private reasoning trace for audit:
for entry in claude_thinking.history:
    if 'thinking' in entry:
        print(f"[{entry['role']}] thought for "
              f"{len(entry['thinking'])} chars before answering.")
```

The reasoning text is stored on `client.history[-1]['thinking']`; it is
**never** included in the public answer that `causal_mas` parses — so
your DAG construction is unaffected by the thinking text, you just get
optional auditability.

## 5. The debate transcript

Every MAS run produces a full `transcript` for auditability:

```python
for entry in res.transcript[:8]:
    print(f"[round={entry['round']}] {entry['agent']}:{entry['action']}: "
          f"{entry['payload']}")
```

Four entries per round: `propose` (proposer), `reject` (critic),
`endorse` (domain expert), `score` (synthesiser).  The `score` entry
records the running edge-count map, so you can replay how confidence
accumulated.

## 6. Reading the result

```python
res.edges          # -> list[(parent, child)] surviving final_threshold
res.confidence     # -> dict{edge -> float in [0, 1]}
res.roles          # -> dict{variable -> 'treatment' | 'outcome' | 'confounder' | ...}
res.transcript     # -> list of round-by-round debate events
res.backend        # -> 'heuristic' | 'openai' | 'anthropic' | 'echo' | user's name
res.to_dag_string()  # -> 'A -> B; C -> D' for sp.dag(...)
```

## 7. Pipe into `sp.dag` for identification

```python
dag_str = res.to_dag_string()
dag = sp.dag(dag_str)

# Identify the causal effect of 'treatment' on 'mortality':
ident = sp.identify(dag, treatment='treatment', outcome='mortality')
print(ident.summary())
```

If the MAS DAG identifies the effect, `ident` will include an
adjustment set.  If the DAG has cycles or incomplete roles, the
`identify` call will tell you which edges to challenge — typically a
useful signal that the critic round didn't reject enough.

## 8. Role overrides vs. the heuristic

The heuristic variable-name classifier is a starting guess.  Override
it whenever you know better:

```python
res = sp.causal_llm.causal_mas(
    variables=['A', 'B', 'C', 'D'],   # all opaque names
    treatment='B',
    outcome='D',
    confounders=['A'],
    instruments=['C'],
    client=my_client,
    rounds=3,
)
```

Explicit kwargs always win over the name heuristic.

## 9. Integration with the rest of StatsPAI

| Use case                                            | Pipeline                                                         |
|-----------------------------------------------------|------------------------------------------------------------------|
| Propose a DAG, identify effects, estimate, robustness | `causal_mas` → `sp.dag` → `sp.identify` → `sp.smart.recommend` → `sp.dml` |
| Build an E-value under unobserved confounding       | `causal_mas` → `sp.causal_llm.llm_unobserved_confounders` → `sp.evalue` |
| Cross-examine against a domain DAG                  | `causal_mas` → diff `res.edges` with the experts' DAG             |
| Streaming update of the DAG as new data arrive       | Run `causal_mas` per batch, combine via `sp.dag` edge intersection |

## 10. When *not* to use this

- **High-stakes clinical or legal decisions.**  The heuristic backend
  is not a substitute for a domain-expert DAG review.  Treat the
  output as a *proposal* to be audited, not a ground truth.
- **Very large variable sets (≥ 50 nodes).**  Token budgets and
  combinatorial explosion of edge proposals degrade quality; cluster
  the variables first or use a structured DAG-learning algorithm
  like `sp.pcalg` or `sp.ges`.
- **Quantitative effect estimation.**  Causal MAS only produces a DAG.
  For effect magnitudes pipe the DAG into `sp.dml`, `sp.causal_forest`,
  `sp.metalearner`, etc.

## 11. References

- arXiv:2509.00987 (2025/09).
  *Causal MAS: A Survey of Large Language Model Architectures for
  Discovery and Effect Estimation.*
- Wan, G., Lu, Y., Wu, Y., Hu, M. & Li, S. (2024).
  *Enhancing Causal Discovery with Large Language Models.*  arXiv:2402.11068.
- Anthropic (2025).
  *Extended Thinking in Claude 4.5 / Opus 4.7.*  Technical report.
