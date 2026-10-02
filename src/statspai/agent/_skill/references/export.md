# Export cookbook: Word / Excel / LaTeX, regtable recipes, figure factory

> Reference file of the `statspai-analysis` skill. Read the section you need; `validate_api_claims.py` checks it against the installed StatsPAI: every `sp.*` name resolves, every `sp.*(...)` call in a code block binds to the real signature (keywords of functions that take `**kwargs` cannot be checked), and the result attributes on the gate's smoke-fit list exist.

## Export cookbook — Word / Excel / LaTeX in one line

StatsPAI's export stack is the agent-native equivalent of Stata's `outreg2` / `esttab` / `collect` and R's `modelsummary` / `gtsummary`. Three tiers, picked by **scope** of what you're exporting:

| Tier | Use when | API | Hot kwargs |
|---|---|---|---|
| **1. Single multi-column table** (the outreg2 / `summary_col` equivalent) | Exporting *one* Table 2 / Table 3 / Table A1 with progressive columns | `rt = sp.regtable(M1, M2, ..., template="aer", title=...)`  *(default: all coefs incl. intercept)*<br>`rt.to_word("table2.docx")`<br>`rt.to_excel("table2.xlsx")`<br>`rt.to_latex()` · `rt.to_markdown()` | `template`, `coef_labels`, `model_labels`, `panel_labels`, `dep_var_labels`, `stats`, `stars`, `add_rows`; opt-in filters: `drop=["Intercept"]` (suppress constant), `keep=[focal]` (focal-only) |
| **2. Multi-panel paper format** (Tables 2 + 3 + A1 + A2 in one file) | Producing the *paper-tables block* — main + heterogeneity + robustness + placebo as a single document | `pt = sp.paper_tables(main=[M1...M5], heterogeneity=[H1,H2,H3], robustness=[R1...Rn], placebo=[P1,P2], template="aer")`<br>`pt.to_docx("paper_tables.docx")`<br>`pt.to_xlsx("paper_tables.xlsx")`<br>`pt.to_latex(...)` | `main`, `heterogeneity`, `robustness`, `placebo`, `template`, `coef_labels`, `model_labels_<panel>`, `keep` |
| **3. Full session bundle** (Stata 15 `collect` equivalent) | Replication appendix that mixes summary stats + balance + multiple regression tables + headings + prose in **one** file | `c = sp.collect("Paper title", template="aer")`<br>`c.add_heading("§1. Descriptives")`<br>`c.add_summary(df, vars=...)`<br>`c.add_balance(df, treatment=, variables=...)`<br>`c.add_regression(M1, M2, ..., title="Table 2")`<br>`c.add_text("Notes ...")`<br>`c.save("paper.docx")` (auto-detect by extension; `.xlsx`/`.tex`/`.md`/`.html`/`.txt` all work) | `add_heading(level=)`, `add_summary(stats=, labels=)`, `add_balance(weights=, test=)`, `add_regression(**regtable_kwargs)`, `add_table(result)`, `add_text(...)` |

**Journal templates** (apply the right SE label, star levels, and notes automatically):

```python
sp.list_journal_templates()
# → ('aer', 'qje', 'econometrica', 'restat', 'jf', 'aeja', 'jpe', 'restud')

rt = sp.regtable(M1, M2, M3, template="qje")    # QJE styling; default = full coef list (incl. intercept)
rt.to_word("table2_qje.docx")
# Opt-in filters:
#   • drop the constant only:    sp.regtable(M1, M2, M3, template="qje", drop=["Intercept"])
#   • focal-coefficient only:    sp.regtable(M1, M2, M3, template="qje", keep=["x"])

sp.get_journal_template("aer")                                 # inspect a preset
# → {'label': 'American Economic Review', 'star_levels': (0.1, 0.05, 0.01),
#    'se_label': 'Standard errors', 'stats': ('N', 'R-squared'),
#    'notes_default': ('Standard errors in parentheses.', '*** p<0.01, ** p<0.05, * p<0.10.'),
#    'font_name': 'Times New Roman'}    # note: tuples, not lists
```

**Inline citations in prose** (drop a coefficient straight into a sentence):

```python
sp.cite(M3, "training")                  # → "1.239*** (0.153)"
sp.cite(M3, "training", output="latex")  # → "1.239^{***}~(0.153)"  (wrap in $...$ yourself)
```

> **Naming gotcha**: `sp.regtable(..., output="docx")` is invalid — the enum is `{"text", "latex", "tex", "html", "markdown", "md", "qmd", "quarto", "word", "excel"}`. Use `output="word"` / `"excel"`, or — simpler — drop `output=` and call `.to_word(filename)` / `.to_excel(filename)` on the result.

---

## Notebook setup — CJK fonts + retina DPI

Run **once at the top of every analysis script / notebook**, *before* any matplotlib-backed plot (`sp.regtable.to_*` exporters do not need this — only `.savefig` / `sp.coefplot` / `sp.binscatter` / `sp.cate_plot` / etc.). Two failures it fixes in one shot:

1. **CJK labels render as ▢▢▢ tofu** — the matplotlib default `DejaVu Sans` carries no Chinese / Japanese / Korean glyphs, so `ax.set_title("教育回报")` silently degrades into squares.
2. **Plots look fuzzy on hi-DPI displays** — matplotlib's default `figure.dpi=100` is half the density of a Retina / 4K screen.

### Drop-in snippet

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

def setup_plot(retina: bool = True) -> None:
    """One-shot matplotlib boilerplate: CJK font fallback + retina DPI.

    Idempotent — safe to call multiple times. Call BEFORE any plotting.
    """
    # 1. CJK font fallback chain — covers macOS / Windows / Linux in one list.
    #    matplotlib uses the first available font; later names are fallbacks,
    #    so listing all three platforms is harmless on any single host.
    mpl.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Hiragino Sans GB",   # macOS
        "Microsoft YaHei", "SimHei", "SimSun",           # Windows
        "Noto Sans CJK SC", "Source Han Sans SC",        # Linux / Adobe
        "WenQuanYi Micro Hei",                           # Linux fallback
        "Arial Unicode MS",                              # universal fallback
        "DejaVu Sans",                                   # last-resort Latin
    ]
    mpl.rcParams["axes.unicode_minus"] = False          # 修复中文字体下负号渲染成 □

    # 2. Retina-grade DPI. figure.dpi controls on-screen / inline rendering;
    #    savefig.dpi controls .png exports. Set both — they are independent.
    if retina:
        mpl.rcParams["figure.dpi"]  = 144   # 2× default — sharp on Retina/HiDPI
        mpl.rcParams["savefig.dpi"] = 300   # manuscript/export PNG (AER house norm)
        # Jupyter inline retina backend (no-op outside IPython):
        try:
            from IPython import get_ipython
            ipy = get_ipython()
            if ipy is not None:
                ipy.run_line_magic("config", "InlineBackend.figure_format = 'retina'")
        except Exception:
            pass

setup_plot()                                            # call once at the top
```

### Smoke test (5 seconds, run once after `setup_plot()`)

```python
fig, ax = plt.subplots(figsize=(4, 2.5))
ax.plot([0, 1, 2], [-1, 0, 1])
ax.set_title("中文标题测试 — Card (1995) 教育回报")
ax.set_xlabel("受教育年数 (years)")
fig.tight_layout()
fig.savefig("figures/_font_smoke_test.png", dpi=300)    # delete after verifying
```

If the saved PNG shows Chinese characters cleanly *and* the y-axis tick `-1` is a real minus sign (not a square), the setup is good. Otherwise see troubleshooting below.

### Saving figures — the `(fig, ax)` idiom (READ THIS)

> **Every StatsPAI plotter and every result `.plot()` returns a `(fig, ax)` tuple — NOT a bare Figure.** So `sp.parallel_trends_plot(...).savefig(...)` raises `AttributeError: 'tuple' object has no attribute 'savefig'`. Always unpack, then save the figure:
>
> ```python
> fig, ax = sp.parallel_trends_plot(df, y="wage", time="year", treat="training", treat_time=2015)
> fig.savefig("figures/fig1.png", dpi=300)
> ```
>
> Two exceptions to memorize:
> - **`sp.binscatter(...)` returns a 3-tuple** `(fig, ax, binned_df)` — `fig, ax, _ = sp.binscatter(...)`.
> - **`sp.kaplan_meier(...).plot()` returns a bare `Axes`** (it is a `KMResult`, not a `CausalResult`) — save via `ax = km.plot(); ax.figure.savefig(...)`.
>
> This applies uniformly to `coefplot`, `binscatter`, `rdplot`, `rddensity().plot()`, `bacon_plot`, `enhanced_event_study_plot`, `did_summary_plot`/`ggdid`/`group_time_plot`, `synthdid_plot`, `cate_plot`, `cate_group_plot`, `dose_response().plot()`, `sensitivity_plot`, `match().plot()`, `synth().plot()`, and a generic `result.plot()`. The code blocks below all use the unpack-then-save form.

### Troubleshooting

| Symptom | Fix |
|---|---|
| Title still shows ▢▢▢ tofu after `setup_plot()` | Host has none of the listed fonts. Install one — **macOS**: pre-installed (no action). **Linux**: `sudo apt install fonts-noto-cjk` (Debian/Ubuntu) or `sudo dnf install google-noto-sans-cjk-fonts` (Fedora/RHEL). **Windows**: pre-installed. Then clear matplotlib's font cache: `rm -rf ~/.cache/matplotlib` (Linux/macOS) / `%LOCALAPPDATA%\matplotlib` (Windows), and restart the Python / Jupyter kernel. |
| Negative numbers render as ▢ | `axes.unicode_minus = False` was overridden by a later `plt.style.use(...)` or `mpl.rcParams.update(...)`. Re-call `setup_plot()` after any style change. |
| Plot blurry inside VSCode `.ipynb` | VSCode's notebook UI ignores `figure.dpi` for inline rendering. Either switch the cell output to "Open in Image Viewer", or use `%matplotlib inline` *before* `setup_plot()`. The saved `.png` (driven by `savefig.dpi=300`) is sharp regardless. |
| `sp.<plot>(...)` output still shows tofu | The `sp.*` plotters honor global `rcParams`, so this only happens when `setup_plot()` was called *after* the plot was drawn. Move the call to the very top of the script. |
| Need to verify which font matplotlib picked | `mpl.font_manager.findfont(mpl.font_manager.FontProperties(family=mpl.rcParams["font.sans-serif"]))` returns the resolved file path — if it ends in `DejaVuSans.ttf` despite Chinese labels, no CJK font is installed. |

### Persist as project default (optional)

Drop the same rcParams into a project-level `matplotlibrc` next to `pyproject.toml` so co-authors and CI runners pick it up without calling `setup_plot()`:

```
# matplotlibrc — committed to the repo
font.sans-serif: PingFang SC, Heiti SC, Microsoft YaHei, SimHei, Noto Sans CJK SC, Arial Unicode MS, DejaVu Sans
axes.unicode_minus: False
figure.dpi: 144
savefig.dpi: 300
```

The `setup_plot()` function above is the in-script fallback when a project `matplotlibrc` is not present.

---

## Regtable cookbook (one-page recipe index)

`sp.regtable(*models, ...)` is the single primitive behind every multi-regression table in an AER paper. The eight patterns above map to:

| Pattern | What varies across columns | Step |
|---|---|---|
| **A. Progressive controls** | covariate set / FE depth | 4.1 — Table 2 |
| **B. Design horse race** | identification strategy (OLS / 2SLS / DID / DML / PSM) | 4.2 — Table 2-bis |
| **C. Multi-outcome** | dependent variable Y | 4.3 — Table 2-ter |
| **D. Stacked Panel A / B** | horizon / sample (panel rows × spec columns) | 4.4 — Table 2-quater |
| **E. IV reporting triplet** | first stage / reduced form / 2SLS | 4.5 — Table 2-quinto |
| **F. `sp.causal(...)` orchestrator** | 1 column, full diagnostics | 4.6 |
| **G. Subgroup table** | subsample (full / female / male / Q1…Q4) | 5.1 — Table 3 |
| **H. Robustness master** | every robustness check stacked | 7.11 — Table A1 |

Default `sp.regtable` settings for AER house style — and the export pipeline
(produce `.docx` + `.xlsx` + `.tex` from the same `RegtableResult`):

```python
rt = sp.regtable(*models,
                 template="aer",                  # journal preset: aer/qje/econometrica/restat/jf/aeja/jpe/restud
                 # AER convention: pass NEITHER `keep=` NOR `drop=` —
                 # `regtable` will then surface every estimated parameter
                 # (controls AND the intercept). Add `drop=["Intercept"]`
                 # only if you want the constant suppressed; add
                 # `keep=[focal]` only for an intentional focal-only table.
                 coef_labels={"training": "Training"},
                 model_labels=[...],              # column labels
                 stats=["N", "R2", "Cluster", "FE", "DV mean"],
                 title="Table N. ...")

# One-call exports — never hand-roll Word/Excel from pandas:
rt.to_word ("tables/tableN.docx")                  # editable Word, AER book-tab borders
rt.to_excel("tables/tableN.xlsx")                  # editable Excel, one sheet
open("tables/tableN.tex", "w").write(rt.to_latex()) # LaTeX for the build
print(rt.to_text())                                 # quick terminal preview
```

For pyfixest-style native output, `sp.etable(*models, ...)` is the alternative; for stacking many tables in one `.docx`, use `sp.paper_tables(...)` (Tier 2) or `sp.collect()` (Tier 3) — see Step 8.

## Figure factory (the 12 standard AER figures)

| # | Figure | StatsPAI call | Section |
|---|---|---|---|
| 1a | Raw trends (DID Figure 1) | `sp.parallel_trends_plot(df, y, time, treat, treat_time, ci=True)` | §1 |
| 1b | Treatment rollout heatmap | `sp.treatment_rollout_plot(df, time, treat, id)` | §1 |
| 2a | Event-study coefficients | `sp.enhanced_event_study_plot(cs)` *(cs = `sp.callaway_santanna(...)`; **not** `event_study()` output)* | §3 |
| 2a' | Bacon weights | `sp.bacon_plot(sp.bacon_decomposition(...))` | §3 |
| 2a'' | CS-DID dynamic effects | `cs.plot()` · `sp.ggdid(cs)` · `sp.group_time_plot(cs)` | §3 |
| 2b | RD canonical plot | `sp.rdplot(df, y, x, c)` | §3 |
| 2b' | McCrary density | `sp.rddensity(df, x, c).plot()` | §3 |
| 2c | Matching love plot | `sp.match(...).plot()` | §3 |
| 2d | SCM trajectory | `sp.synth(...).plot()` · `sp.synthdid_plot(sp.sdid(...))` | §3 |
| 3 | Coefficient plot of main specs | `sp.coefplot(M1...M5, variables=["x"])` | §4 |
| 4a | Dose-response | `sp.dose_response(...).plot()` | §5 |
| 4b | CATE histogram | `sp.cate_plot(ml, kind="hist")`  *(ml = `sp.metalearner(..., learner='dr')`)* | §5 |
| 4c | CATE by group bar | `g = sp.cate_by_group(ml, df, by=..., n_groups=4); sp.cate_group_plot(g)` | §5 |
| 5 | Robustness forest plot | `sp.coefplot(*rob.values(), variables=["x"])` | §7 |
| 5b | Specification curve | `sp.spec_curve(...).plot()` | §7 |
| 6 | Sensitivity (text dashboard + honest-DID figure) | `print(sp.sensitivity_dashboard(result).summary())` · `sp.sensitivity_plot(sp.honest_did(cs, ...))` | §7 |
| 7 | Final result.plot() | `result.plot()` (estimator-specific) | §8 |

> Every plotting function above accepts `ax=` so panels can be combined with matplotlib subplots, and **returns a `(fig, ax)` tuple** — unpack it and call `fig.savefig(path, dpi=300)` for publication output (see "Saving figures — the `(fig, ax)` idiom" above). `sp.binscatter` returns `(fig, ax, binned_df)`; `sp.kaplan_meier(...).plot()` returns a bare `Axes` (use `ax.figure.savefig(...)`).

---
