# P0 — Agent-Native Coverage + Recovery (2026-04-21)

> Goal: take StatsPAI from "agent-native in principle" to "agent-native with
> measurable coverage" along two axes — **what's reachable** (MCP surface)
> and **what's recoverable** (failure → fix loop).

## Context (探测事实)

- `sp.list_functions()` = **889 registered functions**
- `sp.agent.tool_manifest()` = **8 hand-curated tools** → 99.1% coverage gap
- `sp.function_schema(name)` already emits OpenAI-format schemas for any
  registered function (field-complete — name/description/parameters with
  enum + required)
- `sp.all_schemas()` already bulk-exports all 889 schemas
- `CausalResult` / `EconometricResults` have `.summary / .tidy / .glance /
  .cite / .to_latex / .to_docx / .next_steps` but **no `to_dict()` /
  `to_json()` / `for_agent()`** — serialization sits externally in
  `agent/tools.py::_default_serializer` and is fragile
- `REMEDIATIONS` in `agent/remediation.py` = 11 entries, mostly generic
  Python errors — no causal-specific recovery paths
- `mcp_server.py::SERVER_VERSION = "0.9.17"` — version drift vs project
  v1.5.0 (release flow doesn't propagate)

## Scope

Approach **A+B hybrid**:

- **A (coverage lever)**: drive MCP tool count from 8 → ~150 via auto-
  generation on top of the existing `all_schemas()` surface. Use a
  category-based whitelist so we don't expose every utility function.
- **B (depth)**: add agent-readable serialization to the result classes,
  expand remediation registry to cover causal-specific failures, sync
  server version, and make `sp.recommend` / `sp.compare_estimators` first-
  class MCP tools.

## Deliverables

### D1 — `to_dict()` / `for_agent()` on result classes

- `CausalResult.to_dict()` → flat JSON-safe dict with `method`, `estimand`,
  `estimate`, `se`, `pvalue`, `ci`, `alpha`, `n_obs`, `diagnostics`,
  `citation_key`, optional `detail_head` (first N rows of detail DataFrame)
- `CausalResult.for_agent()` → adds `warnings`, `next_steps` (compact),
  `assumption_checks` (from model_info), and `suggested_functions` (other
  `sp.xxx` to call next)
- `EconometricResults.to_dict()` → adds `coefficients` keyed by term with
  estimate/se/t/p/CI
- `EconometricResults.for_agent()` → same + glance()-style model summary
- Existing `_default_serializer` in `agent/tools.py` reduced to a
  one-liner that calls `result.to_dict()` with backward-compat fallback

### D2 — `auto_tool_manifest()` + merged `tool_manifest()`

- New `agent/auto_tools.py`:
  - `auto_tool_manifest(categories=None, exclude=None)` → walks the
    registry, returns MCP tool specs for every function whose `category`
    is in the whitelist AND which is not explicitly excluded
  - Default whitelist: `{causal, regression, panel, inference, diagnostics,
    smart, decomposition, robustness, postestimation, survival, survey}`
  - Default exclude: plotting, I/O, CLI helpers, private underscore names,
    known deep-learning estimators that need torch (excluded to keep
    manifest loadable without optional deps)
- `tool_manifest()` (in `agent/tools.py`) becomes a merger: hand-curated
  8 tools win on name collision; auto-generated tools fill the rest
- Target: manifest size ≥ 100 tools, ≤ 250 tools (deliberate ceiling so
  the payload stays under Claude/GPT tool-list limits)

### D3 — Remediation expansion

Add ~15 entries to `REMEDIATIONS` keyed to causal-specific failure
modes. Each carries `category`, `diagnosis`, `fix` (concrete next sp
function), and a regex `match`:

| category | typical message | suggested fix |
|---|---|---|
| `overlap_violation` | "propensity score.*extreme", "no overlap" | trim / use `sp.ebalance` / `sp.sbw` |
| `parallel_trends_fail` | "pre-trend.*reject", "parallel trends violated" | `sp.honest_did` / `sp.callaway_santanna` |
| `mccrary_reject` | "McCrary.*reject", "density discontinuity" | `sp.rd_manipulation_test`; consider donut RD |
| `dml_ortho_fail` | "orthogonality.*failed", "score mean ≠ 0" | increase folds / stronger nuisance learners |
| `bayes_convergence` | "rhat.*>", "low ess", "divergences" | increase draws/tune; reparameterize; `target_accept=0.95` |
| `negative_weights_twfe` | "negative weights", "forbidden comparisons" | `sp.bacon_decomposition`; switch to CS / dCdH |
| `small_cohort` | "cohort size.*<", "thin cohort" | aggregate cohorts or report simple ATT |
| `hausman_reject` | "Hausman.*reject" | use FE over RE; inspect cluster structure |
| `ci_coverage_fail` | "conformal.*coverage", "empirical coverage.*<" | increase calibration set; `sp.jackknife_plus` |
| `sbw_infeasible` | "SBW infeasible", "balance constraints.*tight" | relax tolerance; drop extreme-propensity units |
| `placebo_fail` | "placebo.*significant", "pre-period placebo rejects" | investigate anticipation effects or bad controls |
| `iv_exclusion_fail` | "Hansen J.*reject", "over-id.*reject" | drop weakest instrument; consider local identification |
| `matching_unbalanced` | "SMD.*>.*0.1", "covariates unbalanced after match" | tighten caliper; try `sp.cem` / `sp.ebalance` |
| `synth_no_pretrend_fit` | "pre-treatment RMSE.*high", "poor donor fit" | expand donor pool; try `sp.synthdid` / MSCM |
| `identification_unknown` | "design could not be inferred" | call `sp.check_identification` explicitly; pass `design=` |

### D4 — Housekeeping

- `mcp_server.py::SERVER_VERSION` reads from `statspai.__version__` at
  import time (single source of truth)
- Add to curated `TOOL_REGISTRY`: `recommend`, `compare_estimators`,
  `bacon_decomposition`, `honest_did`, `sensitivity`, `spec_curve`
  (high-value agent tools currently missing)

## Non-goals (defer to P1)

- DataFrame passthrough transport (MCP stdio is text-only by design —
  the CSV round-trip is fine for v1)
- LLM-in-the-loop causal discovery
- `sp.causal_text` module
- Lighthouse Claude Desktop demo (can land after P0 is green)

## Test plan

New/expanded tests in `tests/test_agent.py`:

1. `TestResultSerialization`:
   - `CausalResult.to_dict()` round-trips through `json.dumps`
   - Keys present: method, estimand, estimate, se, pvalue, ci, n_obs
   - `for_agent()` includes warnings/next_steps/suggested_functions
   - `EconometricResults.to_dict()` contains coefficients map with
     estimate/se/p_value for each term
2. `TestAutoManifest`:
   - `auto_tool_manifest()` returns ≥ 100 tools
   - Every auto tool has valid JSON schema (type/properties/required)
   - Manifest is JSON-serializable end to end
   - `tool_manifest()` preserves all 8 canonical names (backward compat)
   - No duplicate tool names after merge
3. `TestRemediationCoverage`:
   - Each new remediation category matches its sample error message
   - `remediate()` returns `matched=True` for all ≥ 25 patterns
4. `TestVersionSync`:
   - `SERVER_VERSION == statspai.__version__`

## Rollout

1. D1 + D1 tests → commit `feat(agent): result.to_dict() + for_agent()`
2. D2 + D2 tests → commit `feat(agent): auto-generated MCP tool manifest`
3. D3 + D3 tests → commit `feat(agent): causal-specific remediation`
4. D4 + D4 tests → commit `chore(agent): version sync + curated tools`
5. Full `pytest -q` + `pytest tests/test_agent.py -q` → commit any fixes
6. Notify user for acceptance

## Risks

- **Auto-generated schemas may have missing enum values** for parameters
  where the docstring doesn't clearly declare choices. Mitigation: the
  merge lets hand-curated schemas override, so anywhere we notice poor
  agent UX we can promote to curated without a breaking change.
- **Torch-dependent estimators in whitelist** may ImportError when the
  manifest is built. Mitigation: explicit deny list includes
  `dragonnet`, `deepiv`, `neural_causal_*` etc.; manifest build is
  wrapped in per-function try/except.
- **Registry autopopulation side-effects**: `all_schemas()` triggers
  `_ensure_full_registry()` which imports every submodule. This already
  happens on first `sp.help()` call, so no new side effect.
