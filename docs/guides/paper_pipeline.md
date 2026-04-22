# `sp.paper()` — Data → Publication-Draft Pipeline

> One call from a DataFrame + a natural-language question to a
> publication-quality draft (markdown, LaTeX, or Word) with EDA,
> identification check, recommended estimator, fitted result, and
> robustness suite — pre-assembled.

## When to use

- You have data and a research question; you want **a draft you can
  edit**, not a slide deck.
- You're orchestrating an **autonomous agent** that needs to produce
  empirical reports without holding the analyst's hand through each
  stage.
- You want a single artifact that captures *why* the estimator was
  chosen (identification + assumptions), not just the point estimate.

If you only need the fitted model → use the matched estimator directly
(`sp.regress`, `sp.did`, `sp.iv`, ...). If you want the full workflow
without rendering → use `sp.causal()`.

## Quick start

```python
import statspai as sp
import pandas as pd

df = pd.read_stata("nlsw88.dta")

draft = sp.paper(
    df,
    "effect of training on wage",
    treatment="training",
    y="wage",
    covariates=["education", "experience"],
)

print(draft.to_markdown())   # human readable
draft.write("nlsw88_draft.tex")  # LaTeX
draft.write("nlsw88_draft.docx") # Word (requires python-docx)
```

## What you get

A `PaperDraft` with **seven sections**:

| Section | Contents |
|---------|----------|
| Question | The natural-language question + parsed outcome / treatment / design. |
| Data | Sample size, missingness top-5, outcome moments, treatment distribution, mean covariates by arm, std-diff. |
| Identification | Verdict from `sp.check_identification` plus per-finding severity / fix. |
| Estimator | Recommended method + `sp.<fn>()` call + rationale + key assumptions. |
| Results | Estimate, SE, 95% CI, p-value, n_obs (or coefficient table for regressions). |
| Robustness | Method-specific suite (e.g. parallel trends + e-value for DiD; first-stage F + AR-CI for IV). |
| References | BibTeX-style entries pulled from `result.cite()`. |

Plus an extra **Pipeline notes** section if any stage failed (so the
draft never silently hides errors).

## The question parser

`sp.paper()` runs a lightweight parser to extract column hints from the
natural-language question — **but explicit kwargs always win**.

| Pattern | Example | Hints |
|---------|---------|-------|
| "effect of X on Y" | "effect of training on wage" | treatment=training, y=wage |
| "Y ~ X" | "wage ~ training" | y=wage, treatment=training |
| "difference-in-differences" / "did" | "DiD effect of ..." | design='did' |
| "regression discontinuity" / "rd" | "RD with running variable score" | design='rd', running_var='score' |
| "instrument(ing)? <Z>" | "using distance as an instrument" | design='iv', instrument='distance' |
| "discontinuity at <c>" / "threshold <c>" | "threshold at 0.5" | cutoff=0.5, design='rd' |

The parser **never overrides** explicit args — it only fills in what
you didn't pass. To bypass it entirely, pass everything explicitly.

```python
draft = sp.paper(
    df,
    question="(any string)",  # parser ignored when args complete
    y="wage", treatment="training", design="did",
    time="year", id="worker_id",
)
```

## Rendering formats

### Markdown
The default. Easy to render in Jupyter, GitHub, or as a starting point
for a PR description.

```python
draft.to_markdown()
```

### LaTeX
Wraps the same content in a complete `article` document with a proper
`thebibliography`. Inline markdown markers (`**bold**`, ``` ` ``` code,
bullet lists, code fences) are translated to LaTeX equivalents.

```python
tex = draft.to_tex()
draft.write("paper.tex")
```

### Word
Requires `pip install python-docx`. Falls back to a markdown file with
a notice header if the dependency is missing — never crashes silently.

```python
draft.to_docx("paper.docx")
```

## Combining with the rest of StatsPAI

### With LLM-DAG

```python
# Step 1: build a validated DAG (P1-A)
dag = sp.llm_dag_constrained(df, variables=[...],
                              oracle=...).to_dag()

# Step 2: feed it into the paper pipeline
draft = sp.paper(df, "effect of X on Y", dag=dag)
```

### With agent tool-use

`sp.paper` is registered with full agent metadata
(`assumptions`, `pre_conditions`, `failure_modes`, `alternatives`):

```python
schema = sp.function_schema('paper')
# → JSON-schema usable as an OpenAI / Anthropic tool definition
```

The `PaperDraft.to_dict()` method returns a JSON-serializable payload
suitable for round-tripping through agent tool calls.

## For Agents

- **Pre-conditions**: `data` must contain the outcome (either passed via `y=` or parsed from "effect of X on Y"). The pipeline never silently invents a column.
- **Recovery**: failures in any single stage are captured in a `Pipeline notes` section rather than raising; agents should check `len(draft.workflow.diagnostics.findings)` and the presence of "Pipeline notes" before accepting the draft as complete.
- **Determinism**: with the same `data` and arguments, output is bit-for-bit reproducible (no LLM is invoked unless the user passes `dag=` from a prior LLM-DAG run).
- **Cost**: O(1) workflow runs (one diagnose + one estimate + one robustness). For sensitivity sweeps, use `sp.spec_curve` separately.

## Caveats

1. **Heuristic parser** — if your question phrasing is unusual, pass
   `y` and `treatment` explicitly. The parser is conservative on
   purpose: it would rather miss a hint than mis-fill one.
2. **Robustness suite is design-specific** — DiD gets parallel-trends +
   e-value; IV gets first-stage F. Custom robustness checks should be
   added to the underlying `CausalWorkflow` (`sp.causal(...)`).
3. **Citations** are pulled from each result's `cite()` method when
   available; not every estimator implements one yet.
4. **No LLM is called by default** — `sp.paper` is purely deterministic
   unless you wire LLM-DAG via `dag=`.

## References

- `CausalWorkflow` (`sp.causal(...)`): the underlying orchestrator.
- `sp.check_identification`: identification-stage diagnostics.
- `sp.recommend`: estimator-selection rule engine.
- `sp.evalue_from_result`, `sp.honest_did`, `sp.anderson_rubin_ci`,
  `sp.rdsensitivity`: per-design robustness primitives.

<!-- AGENT-BLOCK-START: paper -->

## For Agents

**Pre-conditions**
- data must contain the outcome column (`y` or parsed)
- If treatment given, it must be a column

**Identifying assumptions**
- Question parser is heuristic — explicit kwargs always win
- Underlying sp.causal() determines design when not specified

**Failure modes → recovery**

| Symptom | Exception | Remedy | Try next |
| --- | --- | --- | --- |
| ValueError 'Could not determine the outcome y' | `ValueError` | Pass `y=...` explicitly or include 'effect of X on Y' in the question text |  |
| Pipeline notes section appears in draft | `(none — informational)` | One pipeline stage failed; inspect `draft.workflow.diagnostics` and pipeline_errors |  |

**Alternatives (ranked)**
- `sp.causal`
- `sp.recommend`

<!-- AGENT-BLOCK-END -->
