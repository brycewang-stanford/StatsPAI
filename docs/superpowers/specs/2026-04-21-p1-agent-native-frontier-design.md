# P1 — Agent-Native × Methodological Frontier (2026-04-21)

> Goal: push StatsPAI's two competitive axes — **Agent-native** and
> **methodological frontier** — by closing the LLM-DAG loop, shipping an
> end-to-end `sp.paper()` orchestrator, and opening the `causal_text`
> frontier (MVP).

## Context (探测事实)

- P0 (`2026-04-21-p0-agent-native-design.md`) already shipped:
  - MCP server (`agent/mcp_server.py`, 355 LOC, full JSON-RPC stdio)
  - Tool manifest + `execute_tool` (`agent/tools.py`, 576 LOC)
  - Remediation registry (`agent/remediation.py`, 270 LOC)
  - `recommend` / `verify_recommendation` rule engine
  - LLM single-shot DAG propose (`causal_llm/llm_dag.py`, in
    `llm_dag_propose`) + LLM evaluator (`dag/llm_evaluator.py`)
  - `sp.causal()` workflow object (`workflow/causal_workflow.py`, 1075 LOC,
    diagnose → recommend → estimate → robustness → report)
- 9 causal-discovery algorithms exist in `causal_discovery/`
  (PC, GES, FCI, NOTEARS, DyNOTEARS, LiNGAM, ICP, PCMCI, LPCMCI), but
  none accept *prior knowledge* (forbidden / required edges) as a
  constraint
- `dag/llm_dag.py` only does *single-shot* merge of an oracle with a CI
  skeleton — no iteration, no per-edge confidence
- `sp.causal()` produces an HTML/MD report but **does not** produce a
  publication-ready TeX/Word section, and is not callable from a single
  user-facing entry point that takes data + a natural-language question
- No `causal_text` module exists (text/embedding as treatment / outcome /
  confounder / measurement-error correction is unaddressed)

## Scope

Three subprojects under one P1 umbrella, sized for one release:

| ID | Item | Depth | Why |
|----|------|-------|-----|
| **P1-A** | LLM-DAG closed loop | full | shared infra: `sp.paper()` and `recommend()` both consume DAGs |
| **P1-C** | `sp.paper(data, question)` end-to-end | full | flagship demo of agent-native value |
| **P1-B** | `sp.causal_text` MVP | MVP | greenfield — open the door, ship 2 methods, mark `experimental` |

Out of scope (deferred):
- `text-as-confounder` (Roberts-Stewart-Nielsen STM-style) — needs topic-
  model dependency story
- `text-as-outcome` (Egami et al. 2018) — needs annotation pipeline
- LLM ensemble DAG aggregation (multiple LLMs voting) — depends on stable
  closed-loop first

## Deliverables

### D1 — P1-A: LLM-DAG closed loop

**New module**: `src/statspai/causal_llm/llm_dag_loop.py`

**New public functions** (registered):

1. `sp.llm_dag_constrained(data, variables, descriptions, oracle=None,
   algo='pc', alpha=0.05, max_iter=3, ci_validate=True,
   verbose=False) -> LLMConstrainedDAGResult`
   - **Loop**: (a) call `oracle` (or `llm_dag_propose`) to get candidate
     directed edges + confidences; (b) split into `forbidden` (LLM says
     no edge) and `required` (LLM says edge); (c) run `algo` (PC variant
     that respects forbidden/required edges as background knowledge);
     (d) for each LLM-required edge, run a CI test on data — if rejected
     at level `alpha`, demote to candidate; (e) iterate until converged
     or `max_iter`.
   - Returns: `final_dag` (statspai DAG), `edge_confidence`
     (DataFrame: edge, llm_score, ci_pvalue, retained, source),
     `iteration_log`, `provenance`.

2. `sp.llm_dag_validate(dag, data, alpha=0.05, ci_test='fisherz') ->
   DAGValidationResult`
   - Standalone: take any DAG (LLM-produced or otherwise), run a CI test
     for each present edge (parents-only conditioning) and each absent
     edge (full marginal correlation), return per-edge support evidence.

**Constrained PC**: extend `causal_discovery/pc.py` with
`pc_algorithm(..., forbidden=None, required=None)` parameters. The
constraints inject before the orientation step:
- `required` edges are never removed during the skeleton phase
- `forbidden` edges are removed regardless of CI test result

**Result classes**:

```python
@dataclass
class LLMConstrainedDAGResult:
    final_dag: DAG
    edge_confidence: pd.DataFrame  # edge, llm_score, ci_pvalue, retained, source
    iteration_log: list[dict]
    llm_proposal: LLMDAGProposal | None
    skeleton: pd.DataFrame
    provenance: dict
    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
```

**Tests** (`tests/test_llm_dag_loop.py`):
- `test_constrained_pc_respects_forbidden`: synthetic DGP with
  X→Y→Z, forbid X→Z, check X→Z absent
- `test_constrained_pc_respects_required`: same DGP, force a known-true
  edge, check it survives
- `test_loop_demotes_ci_rejected_edge`: synthetic where LLM proposes
  X→Z but data CI test rejects → final DAG drops it
- `test_validate_returns_per_edge_support`: declared DAG → per-edge
  p-values match a manual partial-correlation test
- `test_loop_converges_in_one_iter_when_llm_correct`: deterministic
  echo oracle returning the true edges → loop converges immediately
- `test_loop_works_without_oracle`: `oracle=None` → falls back to plain
  PC (data-only) and returns a sensible result

**Family guide**: `docs/guides/llm_dag_family.md` — when to use closed
loop vs single-shot, how to wire an oracle (Anthropic / OpenAI / echo),
how to read `edge_confidence`, how to combine with `recommend_estimator`.

### D2 — P1-C: `sp.paper(data, question)` end-to-end

**New module**: `src/statspai/workflow/paper.py`

**New public function** (registered):

```python
sp.paper(
    data: pd.DataFrame,
    question: str,
    *,
    y: str | None = None,
    treatment: str | None = None,
    covariates: list[str] | None = None,
    id: str | None = None,
    time: str | None = None,
    running_var: str | None = None,
    instrument: str | None = None,
    cutoff: float | None = None,
    cohort: str | None = None,
    design: str | None = None,
    dag=None,
    fmt: str = 'markdown',  # 'markdown' | 'tex' | 'docx'
    output_path: str | None = None,
    include_eda: bool = True,
    include_robustness: bool = True,
    cite: bool = True,
) -> PaperDraft
```

**Pipeline** (orchestrator — does NOT re-implement primitives):
1. Parse `question` (lightweight regex / heuristic: detect "effect of X on Y", "DiD", "RD around C", "IV using Z") to fill missing y/treatment when not passed explicitly
2. Run `sp.causal()` workflow → get diagnostics + recommendation + estimate + robustness
3. EDA section: descriptives via `sp.sumstats` (sample size, treatment balance, missingness)
4. Robustness section: existing workflow.robustness() output + add
   `sp.evalue_from_result` (already used) + design-specific suite
   (e.g. for DiD: `pretrends_test` + `honest_did(M=0,...)`; for IV:
   first-stage F + `anderson_rubin_ci`; for RD: `rdbalance` +
   `rdsensitivity`)
5. Render to chosen fmt:
   - `markdown`: sections (Question, Data, Identification, Estimator,
     Results, Robustness, References)
   - `tex`: same content via `to_latex` on each result + jinja-style
     template
   - `docx`: same via `to_docx` if available

**Result class**:

```python
@dataclass
class PaperDraft:
    question: str
    sections: dict[str, str]  # title -> body
    workflow: CausalWorkflow
    fmt: str
    citations: list[str]
    def to_markdown(self) -> str: ...
    def to_tex(self) -> str: ...
    def to_docx(self, path: str) -> None: ...
    def write(self, path: str) -> None: ...
    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
```

**Tests** (`tests/test_paper_pipeline.py`):
- `test_paper_basic_observational`: synthetic IID data + simple question
  → `PaperDraft` with all sections, markdown contains key terms
- `test_paper_did_pipeline`: DiD synthetic → markdown includes
  "parallel trends" + "Callaway" or appropriate estimator
- `test_paper_rd_pipeline`: synthetic RD with a known threshold → output
  includes McCrary / rdrobust mention + estimate close to truth
- `test_paper_question_parser`: parser maps key strings to design hints
- `test_paper_tex_renders`: tex output contains `\section` and the
  estimate numerically
- `test_paper_writes_to_disk`: round-trip write + file exists + contains
  marker substring
- `test_paper_handles_missing_design`: design auto-detected by
  `check_identification` when not passed

**Family guide**: `docs/guides/paper_pipeline.md`.

### D3 — P1-B: `sp.causal_text` MVP

**New module**: `src/statspai/causal_text/`

```
causal_text/
    __init__.py
    text_treatment.py      # Veitch et al. (2020) text-as-treatment
    llm_annotator.py       # Egami et al. (2024) LLM-annotator MEC
    _common.py             # shared embedding + utility helpers
```

**New public functions** (registered, marked `experimental`):

1. `sp.text_treatment_effect(data, text_col, outcome, embedding=None,
   embedder='hash', n_components=20, covariates=None,
   estimator='dml', random_state=None) ->
   TextTreatmentResult`
   - Implements Veitch-Wang-Blei (2020) "Adapting text embeddings for
     causal inference": treat text as treatment, use embedding-projected
     features as confounders (or as adjustment), estimate ATE via DML
     with the projected representation.
   - `embedder='hash'`: fall-back deterministic hashing-vectorizer (no
     external deps); `embedder='sbert'` lazy-imports
     `sentence-transformers`; `embedder='custom'` accepts a callable
     `text -> np.ndarray`.
   - Returns `CausalResult` with `estimate`, `se`, `ci`, `method`,
     `diagnostics={'n_text_components': k, 'embedder': name}`.

2. `sp.llm_annotator_correct(annotations_llm, annotations_human=None,
   outcome=None, treatment=None, covariates=None, data=None,
   method='egami2024', alpha=0.05) -> LLMAnnotatorResult`
   - Implements Egami-Hinck-Stewart-Wei (2024) "Using imperfect
     surrogates for downstream inference": when an LLM-annotated label
     is used as a treatment / covariate / outcome, correct downstream
     estimands for measurement error using a small human-validation
     subset.
   - Returns coefficient, SE, and CI under the corrected estimator;
     reports the validation-set size and an estimate of the LLM-human
     agreement rate.

**Result classes**: `TextTreatmentResult`, `LLMAnnotatorResult` —
subclass `CausalResult` with extra `diagnostics` keys.

**Tests** (`tests/test_causal_text.py`):
- `test_text_treatment_recovers_synthetic_ate`: synthetic DGP where the
  true text-derived treatment is a known function of token counts → DML
  estimate within 2 SEs of the truth
- `test_text_treatment_hash_embedder_deterministic`: same data + seed →
  identical estimate
- `test_text_treatment_custom_embedder`: callable embedder is honored
- `test_llm_annotator_corrects_known_bias`: simulate biased LLM label
  with known confusion matrix + human-validation subset → corrected
  estimate within 2 SEs of the truth, naive estimate is biased
- `test_llm_annotator_requires_validation_subset`: missing
  `annotations_human` → raises a clear `DataInsufficient`
- `test_text_treatment_marks_experimental`:
  result.diagnostics["status"] == "experimental"

**Family guide**: `docs/guides/causal_text_family.md` — when to use,
caveats, how to plug a sentence-transformer model.

### D4 — Wiring & cross-cutting

- Register all 5 new public functions in `src/statspai/registry.py`
- Re-export from `src/statspai/__init__.py` so `sp.<fn>` works
- Update `src/statspai/agent/tools.py` (and MCP catalog if needed) so
  the new functions are agent-callable
- Bump `__version__` and `pyproject.toml` to **`1.6.0`**
- Add CHANGELOG entry under `[1.6.0]` with three sections (Added /
  Changed / ⚠️ Correctness if applicable)
- Update `mkdocs.yml` to surface the 3 new family guides

## Non-goals (explicit deferrals)

- Will **not** implement text-as-confounder / text-as-outcome (deferred
  to v1.7 along with topic-model integration)
- Will **not** rewrite `causal_discovery/pc.py` from scratch — only add
  optional `forbidden` / `required` parameters
- Will **not** add a hard dependency on `sentence-transformers`,
  `transformers`, or `torch` — all heavy embedders lazy-loaded
- `sp.paper()` will **not** call an LLM by default — orchestration is
  deterministic (LLM-DAG only triggers if the user passes an oracle
  to the underlying `sp.causal()` flow)

## Acceptance criteria

- All new public functions appear in `sp.list_functions()` after
  `import statspai as sp`
- `pytest tests/test_llm_dag_loop.py tests/test_paper_pipeline.py
  tests/test_causal_text.py -v` → all green
- Full `pytest -q` suite remains green (no regressions)
- `pytest tests/reference_parity/ -q` remains green
- `python -c "import statspai as sp; r =
  sp.paper(<synthetic_did>, 'effect of treatment on y'); print(r.to_markdown()[:500])"`
  produces a non-empty draft with at least 4 named sections
- `python -c "import statspai as sp; r = sp.llm_dag_constrained(<df>,
  variables=['X','M','Y'], descriptions={...},
  oracle=<echo>); print(r.summary())"` shows iteration log + per-edge
  confidence

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Constrained PC is non-trivial to add without regressing existing PC behavior | Add `forbidden`/`required` as optional kwargs only; default `None` preserves exact prior behavior; new tests pin the contract |
| `sp.paper()` becomes a brittle template that breaks on edge cases | Render via `try/except` per section; missing section → fallback short note, never crash the draft |
| LLM-annotator MEC has many algorithmic variants | Pick the simplest defensible one (Egami 2024 plug-in correction with paired validation), ship MVP, document caveats, leave room for `method=` extension |
| Heavy text embedders (sentence-transformers / torch) inflate install size | Lazy-import inside function; `embedder='hash'` is the always-available default; document `pip install statspai[text]` for sbert |
| Adding 5 new public functions risks namespace collision | Confirm via `sp.list_functions()` no name overlap before registering |

## Ordering / dependencies

P1-A → P1-C (paper pipeline does *not* require LLM-DAG, but they share
the agent-native registration plumbing — A first to get patterns right)
→ P1-B (independent module, can be done last; lowest cross-coupling)

## Out of session

- Code review by `oracle` agent (correctness audit)
- Self review (line-by-line) before commit
- Push to `main` (no PR per `feedback_no_pr.md`)
- Notify user for acceptance with summary
