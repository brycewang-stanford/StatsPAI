# RFC: Output module consolidation (PR-B)

**Status**: draft, ready for review
**Author**: Claude (under Bryce's supervision)
**Date**: 2026-04-29
**Companion**: PR-A is already merged (commits `c4bbae0`...`930c9e3`).

## 0. Why this RFC exists

PR-A took every safe, no-deprecation cut on the output module:

| Module | Change | Outcome |
|---|---|---|
| 1 | Lazy-load `openpyxl` in `outreg2.py` (drags in PIL transitively) | `import statspai` cold-start: **2,066 ms → 1,306 ms** (-37%); zero heavy modules eagerly loaded at top level |
| 3 | Promote canonical formatters to `output/_format.py`, delegate from `outreg2` / `modelsummary` / `estimates` | Killed ~80 lines of duplicate `_format_stars` / `_fmt_val` / `_fmt_int`; fixed dead-code bug in `modelsummary._stars_str` |
| 4 | Split `MeanComparisonResult` into `output/mean_comparison.py` | `regression_table.py`: **3,335 → 2,831 lines** (-15%); back-compat re-export preserved |
| 5 | Group `output/__init__.py` by purpose, document canonical entry | Doc clarity; `sp.list_functions()` unchanged at 973 |

Net: **522 passing tests** unchanged, no behavior regressions, zero deprecations.

PR-B contains the cuts that need a deprecation cycle and careful design — items that risk subtly changing rendered output (numbers / formatting / row ordering) if rushed. Per CLAUDE.md §12: *"不要悄悄改现有估计器的数值输出"*.

## 1. The two pieces of bloat that remain

### 1.1 Four parallel regression-table backends

| Backend | Public entry | Lines | Real usage (docs+tests grep) |
|---|---|---|---|
| `regtable` (RegtableResult) | `sp.regtable(*models)` | ~1,989 | **163** |
| `esttab` (EstimateTable) | `sp.eststo(r); sp.esttab()` | ~1,025 | 6 |
| `modelsummary` | `sp.modelsummary([r])` | ~845 | 7 |
| `outreg2` (OutReg2) | `sp.outreg2(*r, filename=...)` | ~811 | 4 |

Each one carries its own:

- coef / SE / pvalue extractor (`_extract_model_data`, `_extract_coefs`, `OutReg2._create_regression_table`),
- significance / number formatters — partly fixed in PR-A,
- text / LaTeX / HTML / Markdown / Excel / Word renderers.

Bug surface multiplies by 4. The user research is done — `regtable` is the de facto standard. The other three are **Stata/R muscle-memory facades** worth keeping for ergonomics, but they should be ~50-line wrappers, not 800-line reimplementations.

### 1.2 `RegtableResult` is a 1,989-line god class

[`output/regression_table.py:368-2208`](../../src/statspai/output/regression_table.py#L368-L2208) houses:

- `_extract_model_data` (coef extraction)
- `_resolve_vars` / `_resolve_stat_keys` (row/column resolution)
- `_format_marker` / `_apply_coef_transform` / `_build_cell_context` (cell shaping)
- `to_text` + `_to_text_transposed`
- `to_html` + `_to_html_transposed`
- `to_latex`
- `to_markdown` + `to_quarto`
- `to_dataframe`
- `to_excel`
- `to_word`
- `_render` / `_repr_html_`

Any time we add a feature (e.g., transposed mode, multi-row SE, eform), every renderer grows. This is the textbook case for **separating model from view**.

## 2. Proposed design

### 2.1 Introduce `output/_table_model.py`

Internal data shape that captures *what* the table contains, not *how* it renders:

```python
@dataclass(frozen=True)
class TableCell:
    text: str                     # already formatted ("0.123***")
    align: Literal["l","c","r"] = "c"
    raw: Any = None               # underlying numeric (for to_dataframe / to_excel)
    meta: dict = field(default_factory=dict)  # { "is_se": True, ... }

@dataclass(frozen=True)
class TableRow:
    cells: tuple[TableCell, ...]
    kind: Literal["coef","se","fe","stat","header","sep","title","note"] = "coef"
    label: str = ""

@dataclass(frozen=True)
class TableModel:
    title: str
    column_labels: tuple[str, ...]
    rows: tuple[TableRow, ...]
    notes: tuple[str, ...]
    meta: dict   # star_levels, se_label, eform_note, repro, ...
```

**Invariants**:

- All numeric formatting happens at *model build time* via `output/_format.py`.
- Renderers see only strings + alignment hints + row-kind tags. Renderers cannot make business decisions ("should I print this stat?") — only style decisions ("how do I draw a horizontal rule?").
- `to_dataframe` reads `cell.raw`; everything else reads `cell.text`.

### 2.2 Move renderers to `output/_renderers/`

```
output/_renderers/
    __init__.py
    text.py      # ~200 lines
    html.py      # ~200 lines  (incl. _repr_html_ helper)
    latex.py     # ~200 lines
    markdown.py  # ~150 lines  (also produces Quarto via flag)
    excel.py     # ~150 lines  (delegates styling to _excel_style.py)
    docx.py      # ~150 lines  (delegates styling to _aer_style.py)
```

Each renderer has signature `def render(model: TableModel, **kwargs) -> str | None`. Excel / docx return `None` and write to a path.

### 2.3 Slim down `RegtableResult` to a thin facade

```python
class RegtableResult:
    def __init__(self, model: TableModel): ...

    def to_text(self) -> str:        return text.render(self._model)
    def to_html(self) -> str:        return html.render(self._model)
    def to_latex(self) -> str:       return latex.render(self._model)
    def to_markdown(self, *, quarto=False): return markdown.render(self._model, quarto=quarto)
    def to_dataframe(self): ...      # reads cell.raw; ~30 lines
    def to_excel(self, path):        excel.render(self._model, path)
    def to_word(self, path):         docx.render(self._model, path)
    # ... save / _render / __str__ / __repr__ / _repr_html_

class _RegtableBuilder:
    """Holds all the option resolution + coef extraction logic;
    output is a TableModel."""
    def build(self) -> TableModel: ...
```

**Predicted line counts**:
- `RegtableResult` class: ~120 lines (was 1,989)
- `_RegtableBuilder`: ~600 lines (the actual business logic — coef resolve, vcov dispatch, FE rows, multi-SE, etc.)
- `_renderers/`: ~1,050 lines combined
- `_table_model.py`: ~80 lines
- `regression_table.py` overall: ~800 lines (was 2,831)
- **Net delta**: -780 lines on `regression_table.py`, +1,200 lines split across 7 small files. Total package size roughly the same — the win is **single point of fix** for renderer bugs.

### 2.4 Convert `esttab` / `modelsummary` / `outreg2` to facades

Each becomes a ~50-line dialect translator:

```python
# output/esttab.py (~50 lines)
def esttab(*results, **kwargs):
    """Stata esttab → builds a TableModel via _RegtableBuilder, with esttab defaults."""
    builder = _RegtableBuilder.from_esttab_kwargs(**kwargs)
    for r in results:
        builder.add_model(r)
    return EstimateTableResult(builder.build())  # subclass of RegtableResult for repr identity
```

The `EstimateTableResult` / `OutReg2` / `modelsummary`-returned object types are preserved for `isinstance()` callers, but they all wrap `TableModel` and reuse renderers.

**Net deletion**: `esttab.py` 1,025 → ~50, `modelsummary.py` 845 → ~80, `outreg2.py` 811 → ~80. About **2,470 lines removed**.

### 2.5 Apply the same backend to `MeanComparisonResult` and `CausalResult.to_latex`

Currently `MeanComparisonResult` carries 510 lines of its own renderer code (PR-A moved it but didn't dedupe). And [`core/results.py:1522`](../../src/statspai/core/results.py#L1522) hand-rolls a LaTeX template for `CausalResult` that doesn't share any code with `regtable`.

Both should build a `TableModel` and delegate. Estimated savings: ~400 + ~70 lines. More importantly: **a single bug fix in star formatting / SE display propagates everywhere**.

## 3. Migration plan (semver-respecting)

| Step | Action | Risk | Verification |
|---|---|---|---|
| B-1 | Land `_table_model.py` + empty `_renderers/` skeleton; no behavior change | none | unit tests on TableModel construction |
| B-2 | Implement `_renderers/text.py`, port `RegtableResult.to_text` to delegate. Keep old method as `_to_text_legacy`. CI matrix: byte-diff `to_text` vs `_to_text_legacy` on a fixture corpus | low | snapshot tests on ≥20 fixture tables |
| B-3 | Repeat B-2 for HTML, LaTeX, Markdown, Excel, Docx — one renderer per PR | low (per PR) | snapshot tests; legacy method retained behind `_legacy=True` flag for one minor version |
| B-4 | Build `_RegtableBuilder` and route `RegtableResult.__init__` through it | medium — extraction logic is the most fragile | snapshot tests + parity tests against R `fixest`/`modelsummary` outputs |
| B-5 | Convert `esttab` / `modelsummary` / `outreg2` to facades with `DeprecationWarning("Direct use of OutReg2 internals is deprecated; use sp.regtable(...).to_excel(...) for the same output")`. Keep facades in place for **2 minor versions** | medium | full snapshot suite + ensure all 4 entry points produce identical-looking tables on a shared test set |
| B-6 | Apply TableModel to `MeanComparisonResult` and `CausalResult.to_latex` | low | snapshot tests |
| B-7 | Remove `_to_text_legacy` etc. once snapshots have been stable for 2 minor versions | low | regression tests |

**Deprecation policy** (per CLAUDE.md §3.8):

- Each facade gets a `DeprecationWarning` on first call per session.
- `MIGRATION.md` entry pointing each removed/changed parameter to its `regtable` equivalent.
- Two minor-version cushion before any facade is removed.

## 4. Out of scope for PR-B

- **`sklearn` eager-load** at top of `causal/causal_forest.py` (still costs ~200 ms on `import statspai`). Belongs in a separate "estimator import discipline" PR — too many call sites to bundle here.
- **`Collection` and `paper_tables` Excel/docx writers** that currently bypass `_excel_style.py` / `_aer_style.py`. Once renderers are centralised, audit these to ensure they go through the same path.
- **`fastest`-class table rendering** (e.g., `polars`/`pyarrow` outputs). Premature.

## 5. Success metrics (when PR-B is done)

- `output/` total lines: 11,319 → ≤ **6,500** (-43%)
- `regression_table.py`: 2,831 → ≤ **800**
- Single-source bug fixes: changing star format propagates to all 4 backends + `MeanComparisonResult` + `CausalResult.to_latex`
- All snapshot tests stable across deprecation period
- Public API of `sp.regtable` / `sp.esttab` / `sp.modelsummary` / `sp.outreg2` unchanged for users (rendered output may differ in unimportant whitespace, captured in snapshots)

## 6. What I want feedback on

1. **Do you want PR-B at all, or live with the current state?** PR-A already cut the worst items (cold-start time, dead code, format duplication). PR-B is "right shape for the next 5 years" work, not "stop the bleeding" work.
2. **Snapshot-test corpus**: which estimators / option combinations should we lock down? My instinct: 1 OLS, 1 fixest-FE, 1 IV, 1 CausalResult, 1 multi-row SE — 5 fixtures × 6 renderers = 30 snapshots, easy to maintain.
3. **Facade deprecation horizon**: 2 minor versions feels right given StatsPAI's current cadence. Confirm?
4. **`mean_comparison` reuse**: currently `MeanComparisonResult` carries its own to_text/to_html/to_latex. After B-6 it would build a TableModel — but balance tables are structurally different from regression tables (no SE row, etc.). Worth designing a shared model, or keep them separate? I lean toward shared model with a `kind="balance"` discriminator.

— Claude
