# Exporting regression results

A one-page map of how StatsPAI turns a fitted model into a
publication-quality table — LaTeX, Word, Excel, HTML, Markdown, Quarto — or
an agent-native JSON payload. Everything below is reachable from a single
import:

```python
import statspai as sp
```

The design goal is symmetry: **any** result object exports the same way, and
a single canonical builder (`sp.regtable`) drives every format so the numbers
and styling never drift between LaTeX and Word.

---

## 1. The 30-second version

```python
r = sp.regress("y ~ x + z", data=df)          # an EconometricResults
r.to_latex(caption="Main results", label="tab:main")   # -> LaTeX string
r.to_markdown()                                # -> GitHub-flavoured Markdown
r.to_html(path="table.html")                   # -> writes + returns the HTML
r.to_excel("table.xlsx")                       # -> styled .xlsx
r.to_word("table.docx")                        # -> AER/QJE-styled .docx
```

For two or more models side by side, use `sp.regtable`:

```python
m_ols = sp.regress("wage ~ educ + exper", data=df)
m_iv  = sp.iv("wage ~ (educ ~ dist) + exper", data=df)

tbl = sp.regtable(m_ols, m_iv,
                  model_labels=["OLS", "IV"],
                  template="aer")
print(tbl)                  # text in the terminal
tbl.to_latex()              # LaTeX source
tbl.save("table1.docx")     # format inferred from the extension
```

---

## 2. Single-model exports

Both result families expose the same export surface, so estimator choice
never changes how you write the table out:

| Method            | Returns | Notes                                              |
| ----------------- | ------- | -------------------------------------------------- |
| `.to_latex()`     | `str`   | `booktabs` `\begin{table}` float; `caption=`, `label=` |
| `.to_html()`      | `str`   | inline-styled `<table>`                            |
| `.to_markdown()`  | `str`   | GFM; `quarto=True` for Quarto cross-refs           |
| `.to_excel(path)` | `str`   | styled `.xlsx` (needs `openpyxl`)                  |
| `.to_word(path)`  | `str`   | AER/QJE `.docx` (needs `python-docx`)              |
| `.to_dict()`      | `dict`  | JSON-safe agent payload                            |
| `.to_json()`      | `str`   | `json.dumps` of `to_dict()`                        |

The string-returning methods (`to_latex` / `to_html` / `to_markdown`) take an
optional `path=` and write the file *and* return the string. `to_excel` /
`to_word` take the path as their first argument and return it.

### Passing table options through

Single-model exports on `EconometricResults` are thin wrappers over
`sp.regtable`, so every `regtable` keyword passes straight through:

```python
r.to_latex(
    coef_labels={"educ": "Years of schooling", "exper": "Experience"},
    keep=["educ", "exper"],          # hide the intercept and controls
    order=["exper", "educ"],         # reorder rows
    se_type="t",                     # t-stats instead of SE in parentheses
    stars=True,
    fmt="%.4f",
    template="qje",
    notes=["Robust standard errors."],
)
```

> `CausalResult` (DiD / RD / IV-effect / synth …) exports are tuned to the
> design — `to_latex()` on a `sp.callaway_santanna(...)` result renders the
> event-study / group-time grid rather than a plain coefficient column. The
> method names are identical; the layout is design-aware.

---

## 3. Multi-model tables: `sp.regtable`

`sp.regtable(*models, ...)` is the canonical builder. It accepts any mix of
`EconometricResults`, `CausalResult`, and duck-typed objects exposing
`params` / `std_errors`.

```python
tbl = sp.regtable(
    m1, m2, m3,
    model_labels=["(1)", "(2)", "(3)"],
    dep_var_labels=["Wage", "Wage", "Hours"],
    coef_labels={"educ": "Schooling"},
    keep=["educ", "exper"],
    stats=["N", "R2", "adj_R2", "F"],
    se_type="se",                 # "se" | "t" | "p" | "ci"
    star_levels=(0.10, 0.05, 0.01),
    template="aer",
    title="Returns to schooling",
    notes=["Standard errors in parentheses."],
)
```

Highlights (see `sp.describe_function("regtable")` for the full list):

- **Journal presets** — `template=` one of `"aer"`, `"qje"`, `"econometrica"`,
  `"restat"`, `"jf"`, `"aeja"`, `"jpe"`, `"restud"`. Each sets star levels,
  the SE-row label, default stats, and notes.
- **Auto-diagnostic rows** — fixed-effect / cluster indicators, IV first-stage
  F, DiD pre-trend p, RD bandwidth/kernel/poly are extracted automatically
  (`diagnostics="auto"`).
- **Fixed-effect level counts** — `fixef_sizes=True` emits `# Firm`, `# Year`
  rows.
- **Robust SE at print time** — `vcov="HC1"` (`"HC0"/"HC2"/"HC3"/"robust"`)
  recomputes SE/t/p/CI for OLS without re-fitting.
- **Stacked SE specs** — `multi_se={"Bootstrap SE": [se1, se2]}` prints extra
  bracketed SE rows beneath the primary one.
- **Column spanners** — `column_spanners=[("OLS", 2), ("IV", 2)]` renders a
  grouping header (`\multicolumn` / `colspan` / `cmidrule`).
- **eform** — `eform=True` reports `exp(b)` (odds ratio / IRR / HR) with
  delta-method SE; pass a per-model list to mix columns.

### Decimal places

Economics journals do not mandate a decimal count. What they do in practice is
scale precision to the estimate, and print a coefficient and its standard error
at the **same** decimal place. That is what `fmt="auto"` — the default — does:

```python
sp.regtable(m)                 # adaptive precision (default)
sp.regtable(m, fmt="auto")     # the same thing, spelled out
```

Each coefficient row gets one decimal count, chosen so that neither the
estimate nor its standard error loses a significant digit, and both halves
print at that count. A single table can then carry rows on completely
different scales without either being rounded into meaninglessness:

```text
treat                   2,108***
                           (472)

age                        -26.7
                          (21.6)

re75                    0.392***
                         (0.049)
```

A fixed `fmt="%.3f"` would have written the treatment effect as
`2108.412 (471.938)` — seven digits of false precision on a dollar
amount — and `fmt="%.0f"` would have flattened `re75` to `0***`.

To pin precision yourself:

| call                | effect                                                |
| ------------------- | ----------------------------------------------------- |
| `fmt=3`             | `"%.3f"` everywhere — the int is shorthand            |
| `digits=3`          | identical to `fmt=3`; the R/Stata-flavoured spelling  |
| `fmt="%.4f"`        | any printf template                                   |
| `se_fmt="%.2f"`     | standard-error row only, deliberately unpaired        |
| `stats_fmt="%.4f"`  | R² / adj. R² / F rows and `tests=` footer statistics  |

`fmt` and `digits` are the same knob — passing both raises. Summary statistics
never follow `fmt`, because they sit on their own scale; `N` is always a
thousands-separated integer.

Two edges worth knowing:

- **`siunitx`** decimal-aligns a whole LaTeX column, so it cannot use per-row
  precision. `to_latex(siunitx=True)` collapses adaptive precision to the
  finest count any row asked for and drops thousands separators.
- **Values below the decimal ceiling** escape to scientific notation rather
  than printing `0.000000`, so a nonzero estimate is never displayed as an
  exact zero.

### Saving and the format matrix

```python
tbl.to_text()        tbl.to_latex()      tbl.to_html()
tbl.to_markdown()    tbl.to_quarto()     tbl.to_dataframe()
tbl.to_excel("t.xlsx")   tbl.to_word("t.docx")
tbl.save("t.tex")    # extension dispatch (below)
```

`save(path)` and `regtable(..., filename=path)` pick the writer from the
extension:

| Extension | Writer                                      |
| --------- | ------------------------------------------- |
| `.tex`    | `to_latex`                                  |
| `.html`   | `to_html`                                   |
| `.md`     | `to_markdown`                               |
| `.qmd`    | `to_quarto`                                 |
| `.docx`   | `to_word`                                   |
| `.xlsx`   | `to_excel`                                  |
| `.csv`    | `to_dataframe().to_csv`                     |
| `.json`   | `to_json` (agent payload, below)            |

---

## 4. Agent-native serialization

The package is agent-native: a rendered table is a first-class artifact an
LLM tool loop can serialise, cache, and reason over without re-rendering.

```python
payload = tbl.to_dict()
```

`to_dict()` carries three layers:

1. **metadata** — `model_labels`, `dep_var_labels`, `panel_labels`, `title`,
   `notes`, `template`, `se_type`, `stars` / `star_levels`,
   `requested_stats`, `coef_labels`.
2. **`table`** — the *rendered* cell grid (the formatted `"2.067***"` /
   `"(0.074)"` strings), as a list of
   `{"term": ..., "<model label>": "<cell>"}` records, with `columns`.
3. **`models`** — the *numeric* truth per model: coefficient `estimate`,
   `std_error`, `t_statistic`, `p_value`, `conf_low`, `conf_high`, plus
   summary `stats` and `depvar`.

Use the `models` layer for machine reasoning, the `table` layer for faithful
re-display:

```python
beta = payload["models"][0]["coefficients"]["educ"]["estimate"]   # a float

tbl.to_json(indent=2)                       # str, round-trips through json
tbl.to_dict(renders=["latex"])              # also embed rendered strings
tbl.to_dict(renders=True)                   # latex + html + markdown + text
```

NaN / Inf are coerced to `null`, so `json.dumps(tbl.to_dict())` is always
strict-valid JSON.

The payload is a faithful cache, not just a snapshot — `from_dict` rebuilds a
table that re-renders byte-identically (for tables built without `multi_se` /
`eform` / `column_spanners` / `tests`):

```python
from statspai.output.regression_table import RegtableResult

cached = tbl.to_json()                       # store anywhere
again = RegtableResult.from_dict(json.loads(cached))
assert again.to_latex() == tbl.to_latex()    # exact round-trip
```

Single results carry the same idea: `sp.regress(...).to_dict(detail="agent")`
adds `violations` / `next_steps` / `suggested_functions` for tool loops.

---

## 5. Multi-panel tables and document containers

For a stacked multi-panel layout (Main / Heterogeneity / Robustness /
Placebo) — each argument is a list of models:

```python
pt = sp.paper_tables(
    main=[m1, m2],
    robustness=[m3, m4],
    template="aer",
)
pt.to_latex("paper_tables.tex")
```

To assemble several tables (and free text) into one document and export it as
a single `.docx` / `.xlsx` / `.tex` / `.md` / `.html`:

```python
c = sp.collect()
c.add_regression(m1, m2, title="Table 1")
c.add_summary(df, title="Table 2: Descriptives")
c.save("appendix.docx")
```

See also [the replication-pack guide](replication_workflow.md) for bundling
tables with provenance, Quarto cross-refs, `great_tables`, and CSL
bibliographies.

---

## 6. Coming from Stata or R

| Stata                              | R                                       | StatsPAI                                       |
| ---------------------------------- | --------------------------------------- | ---------------------------------------------- |
| `esttab m1 m2 using t.tex`         | `modelsummary(list(m1, m2))`            | `sp.regtable(m1, m2).save("t.tex")`            |
| `estout, cells(b se)`              | `stargazer(m1, m2)`                     | `sp.regtable(m1, m2, se_type="se")`            |
| `outreg2 using t.xls`              | `texreg(list(m1, m2))`                  | `sp.regtable(m1, m2).to_excel("t.xlsx")`       |
| `eststo` / `esttab`                | `etable(m1, m2)` (`fixest`)             | `sp.regtable(m1, m2)`                          |
| `esttab, star(* .1 ** .05)`        | `modelsummary(stars = c(...))`          | `sp.regtable(..., star_levels=(0.1, 0.05, 0.01))` |
| `esttab, label`                    | `coef_map = c(...)`                     | `sp.regtable(..., coef_labels={...})`          |
| `esttab, keep(x)`                  | `coef_omit` / `coef_map`                | `sp.regtable(..., keep=["x"])`                 |
| `esttab, indicate("FE = ...")`     | `gof_map`                               | `sp.regtable(..., fixef_sizes=True)`           |
| `eststo` store + `esttab`          | `gt()` / `kableExtra`                   | `sp.regtable(...)` + `.to_dict()` for agents   |

Stata-flavoured (`sp.eststo` / `sp.esttab`) and R-flavoured
(`sp.modelsummary`) surfaces exist for muscle memory, but they now forward to
`sp.regtable` and emit a `DeprecationWarning`. New code should call
`sp.regtable` directly for full control over labels, templates, and SE
formats.

---

## 7. Coefficient plots

`sp.coefplot(*models)` returns a Matplotlib `(fig, ax)` forest plot — PNG / PDF
come free via `fig.savefig`:

```python
fig, ax = sp.coefplot(m1, m2, model_names=["OLS", "IV"])
fig.savefig("coefplot.pdf")     # or .png, .svg — any Matplotlib backend
```

For a LaTeX-native, vector, editable plot, `sp.coefplot_tikz(*models)` emits
`pgfplots` source (the same forest layout — one series per model, horizontal CI
error bars, zero reference line):

```python
tikz = sp.coefplot_tikz(m1, m2, coef_labels={"x": "Treatment"},
                        level=0.95, standalone=True)
open("coefplot.tex", "w").write(tikz)   # standalone=True → compiles as-is
```

---

## 8. What exports — and what doesn't

`sp.regtable(result)` is the **universal exporter**: it duck-types any result
exposing `params` / `std_errors` (multi-coefficient) or `estimate` / `se`
(single effect), so the same call works regardless of estimator. Verified
across the result families:

| Result class           | Returned by (e.g.)                       | `sp.regtable(r)` |
| ---------------------- | ---------------------------------------- | ---------------- |
| `EconometricResults`   | `regress` / `ols` / `iv` / `feols` / `logit` / `glm` | ✅ |
| `CausalResult`         | `did` / `rd` / `dml` / `tmle` / `qreg`   | ✅ |
| `PanelResults`         | `panel`                                  | ✅ |
| `FrontierResult`       | `frontier`                               | ✅ |

So you never learn a per-estimator export API — fit anything, then
`sp.regtable(r).to_latex()` / `.to_dict()` / `.save(...)`.

**Out of scope (by design).** Some results are not coefficient tables and
carry their own visualisations or summaries instead: causal-forest
*heterogeneous* effects (`.plot()` of the CATE distribution), partial-
identification *bounds* (intervals, not point ± SE), and density / dose-
response *curves*. Forcing these into a coefficient column would misrepresent
them, so they are intentionally excluded from the table exporters. The
`tests/test_export_surface_contract.py` suite pins the universality claim for
the table-shaped results above.

---

## See also

- `sp.describe_function("regtable")` — every parameter, inline.
- [Replication workflow](replication_workflow.md) — provenance + Quarto + CSL.
- [Migrating from R](migration-from-r.md) — the full `fixest` / `did` map.
