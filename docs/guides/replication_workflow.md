# Replication Workflow — Question → Estimate → Paper → Archive

> One call to bundle data, code, environment, paper, citations, and
> per-number provenance into a journal-ready replication archive.
> Built for the AEA / AEJ data-editor checklist out of the box.

This guide ties together the v1.7.2 export trinity:

- `sp.paper(...)` / `q.paper()` — **data → draft** pipeline (Markdown,
  LaTeX, Quarto `.qmd`, Word).
- `sp.replication_pack(...)` — **draft → journal-ready zip** with
  manifest, hashes, environment lock, and lineage.
- `sp.Provenance` / `sp.attach_provenance()` — **per-number traceability**
  back to the call that produced it.

Plus the surrounding glue:

- `sp.gt(result)` — `great_tables` adapter for publication-grade HTML
  / LaTeX tables.
- `sp.csl_url(...)` / `sp.write_bib(...)` — CSL hub + `paper.bib`
  writer for Quarto citation rendering.
- `sp.paper(..., llm='auto')` — auto-propose a Causal DAG via LLM
  (see [LLM-DAG setup guide](llm_dag_setup.md)).

## When to use

- You're submitting to AER / AEJ / Econometrica / QJE / RestStat /
  RestUd / JF / JPE and the data editor wants a self-contained
  replication archive.
- You're an agent that needs to produce **audit-grade** empirical
  reports — every number traceable to a function call + parameter
  set + input data hash.
- You want **one source** that compiles to PDF / HTML / DOCX / Beamer
  via Quarto, with auto-generated citations and an embedded
  Reproducibility appendix.

## Quickstart — full pipeline in 4 lines

```python
import statspai as sp
import pandas as pd

df = pd.read_csv("training_panel.csv")

# 1. Question → estimate → draft
draft = sp.paper(df, "effect of trained on wage",
                 treatment="trained", y="wage",
                 fmt="qmd")  # Quarto-native output

# 2. Draft → journal-ready replication archive
sp.replication_pack(draft, "submission.zip",
                    code="analysis.py")
```

Open `submission.zip` and you'll find:

```text
submission.zip
├── MANIFEST.json          versions, timestamp, git SHA, per-file SHA-256
├── README.md              replication instructions
├── data/
│   ├── dataset.csv        the analysis frame
│   └── manifest.json      shape + dtypes + SHA-256
├── code/
│   └── script.py          your analysis script
├── env/
│   └── requirements.txt   from pip freeze (or importlib.metadata fallback)
├── paper/
│   ├── paper.qmd          Quarto source — `quarto render paper.qmd`
│   └── paper.bib          auto-emitted from estimator citations
└── lineage.json           per-result Provenance (function + params + data hash)
```

Hand the zip to a co-author or upload it to the journal's data
repository — `quarto render paper/paper.qmd` reproduces your draft
verbatim, and `MANIFEST.json` lets anyone verify the data is byte-
identical to what you analyzed.

## Two entry points

### A. Natural-language path

When you want StatsPAI to infer the design from prose:

```python
draft = sp.paper(df,
    "effect of training on wages, controlling for education",
    treatment="trained", y="wage",
    covariates=["edu", "experience"],
    fmt="qmd",
)
```

The question parser fills in any column hints you didn't pass
explicitly (`treatment` / `y` / `design`); explicit kwargs always
win.

### B. Estimand-first (`sp.causal_question`) path

When you've **pre-registered** the analysis (Target Trial Protocol /
PICOTS rubric) and want the paper to match the declaration verbatim:

```python
q = sp.causal_question(
    treatment="trained",
    outcome="wage",
    data=df,
    population="manufacturing workers, 2018-2019",
    estimand="ATT",
    design="did",
    time="year", id="worker_id",
    covariates=["edu"],
    notes="Pre-registered 2026-04-15.",
)

# Method-style:
draft = q.paper(fmt="qmd")

# Or function-style dispatch:
draft = sp.paper(q, fmt="qmd")
```

The Question / Identification / Estimator / Results sections come
straight from your declaration + `q.identify()` + `q.estimate()`,
not from natural-language inference. Use this path when you want the
draft's identification claims to match what was pre-registered with
your IRB / journal preregistration.

## Output formats

`PaperDraft` exposes four renderers (route via `.write()` extension or
explicit method):

| Format   | Method                       | When                                    |
| ---      | ---                          | ---                                     |
| Markdown | `to_markdown()` / `.md`      | Quick review, GitHub gist               |
| Quarto   | `to_qmd()` / `.qmd`          | Publication-grade pipeline (recommended)|
| LaTeX    | `to_tex()` / `.tex`          | Direct overleaf submission              |
| Word     | `to_docx(path)` / `.docx`    | Co-authors who only edit in Word        |

```python
draft.write("paper.qmd")    # → quarto render paper.qmd
draft.write("paper.tex")    # → pdflatex paper.tex
draft.write("paper.docx")   # → opens in Word
```

The Quarto path is the **strongest** — one source compiles to
PDF / HTML / DOCX / Beamer with cross-refs, citations, and a
machine-readable provenance block in the YAML header.

## Quarto integration

`draft.to_qmd()` emits:

```yaml
---
title: "Causal Analysis Draft"
date: "2026-04-27"
subtitle: "effect of trained on wage"
format:
  pdf: default
  html: default
  docx: default
bibliography: "paper.bib"
csl: "american-economic-association.csl"
statspai:
  version: "1.7.2"
  run_id: "9c3aa1bf"
  data_hash: "5c64c6e6b67c"
---

## Question
...
```

Notable bits:

- **`format:` block** lists every Quarto output you want (`pdf`,
  `html`, `docx`, `beamer`, ...). Override via
  `draft.to_qmd(formats=["pdf", "beamer"])`.
- **`bibliography:`** auto-emits when `draft.citations` is non-empty;
  `replication_pack` writes the actual `paper.bib` next to the qmd.
- **`csl:`** accepts short names — `csl='aer'` resolves to
  `american-economic-association.csl`. See the
  [CSL section below](#cite-style-csl-and-bibliography).
- **`statspai:` block** carries `version` / `run_id` / `data_hash` so
  any reader can audit "is this paper running on the same code +
  data I have?".

When the underlying `result` carries a `_provenance` (any of the 9
instrumented estimators — see
[provenance scorecard](#provenance-scorecard)), the qmd auto-appends:

````markdown
## Reproducibility {.appendix}

```text
Provenance
  function   : sp.did.callaway_santanna
  run_id     : 9c3aa1bf
  ...
  data       : SHA256:5c64c6e6b67c 1200×7
  params     :
    - y = 'wage'
    - g = 'first_treat'
    ...
```
````

## Causal DAG appendix

Pass a DAG and the draft gains a **Causal DAG** section with edges,
adjustment sets, back-door paths, and bad controls — rendered as
text-art for markdown / LaTeX, **mermaid** for Quarto:

```python
from statspai.dag.graph import DAG

g = DAG("trained -> wage; edu -> wage; edu -> trained")
draft = sp.paper(df, "effect of trained on wage",
                 treatment="trained", y="wage",
                 dag=g, fmt="qmd")
```

The qmd renders the DAG as a Quarto-native mermaid block:

````markdown
## Causal DAG

```{mermaid}
%%| fig-cap: Declared causal DAG
graph LR
  trained --> wage
  edu --> wage
  edu --> trained
```

**Adjustment sets** (back-door criterion for `trained` → `wage`):
- {`edu`}

**Back-door paths** from `trained` to `wage`:
- `trained` — `edu` — `wage`
````

## LLM-DAG auto-propose

When you don't have a hand-built DAG, ask an LLM to propose one:

```python
# Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your environment, then:
draft = sp.paper(df, "effect of trained on wage",
                 treatment="trained", y="wage",
                 llm="auto",                # opt-in
                 llm_domain="labor economics, training programmes",
                 fmt="qmd")
```

`llm="auto"` resolves a credential via the layered fallback (env var
→ explicit param → config file → terminal prompt → fail with concrete
remediation), calls `llm_dag_propose`, and attaches the resulting DAG.
Failures (no key, network error, malformed JSON) silently fall back
to a no-DAG paper — auto-DAG never breaks the pipeline.

See the [LLM-DAG setup guide](llm_dag_setup.md) for credential setup,
provider choice, and `configure_llm()` persistence.

To pin the offline heuristic backend (no API call):

```python
draft = sp.paper(..., llm="heuristic")
```

## Cite style (CSL) and bibliography

StatsPAI auto-emits `paper/paper.bib` from estimator `cite()` strings
inside `replication_pack`. To pick a journal style, pass `csl=` to
`to_qmd()`:

```python
draft = q.paper(fmt="qmd")
qmd = draft.to_qmd(csl="aer")  # → american-economic-association.csl
```

Short names supported: `aer`, `aeja`, `aejmac`, `aejmicro`, `aejpol`,
`qje`, `econometrica`, `restat`, `restud`, `jpe`, `jf`,
`chicago-author-date`, `apa`. See `sp.list_csl_styles()` for the full
list.

`.csl` files themselves are **not bundled** with StatsPAI (Zotero
styles are CC-BY-SA-3.0, incompatible with our MIT license). Download
once at project setup:

```bash
curl -O $(python -c "import statspai as sp; print(sp.csl_url('aer'))")
# → american-economic-association.csl in the current directory
```

Quarto resolves `csl: "american-economic-association.csl"` against
that local copy.

For finer control, build the bib yourself:

```python
sp.write_bib([
    "Callaway B, Sant'Anna PHC. (2021). DiD with multiple time periods. JoE.",
    "Imbens GW (2004). Nonparametric estimation of ATEs.",
], "paper.bib")
```

## Numerical lineage / Provenance

Every result from an instrumented estimator carries a `_provenance`
dataclass:

```python
r = sp.callaway_santanna(df, y="y", g="g", t="t", i="i")
prov = sp.get_provenance(r)
print(prov.short())
# → sp.did.callaway_santanna · data:48b58dd2b436 · run:c8bdcc04
print(prov.params)
# → {'y': 'y', 'g': 'g', 't': 't', 'i': 'i', 'estimator': 'dr',
#    'control_group': 'nevertreated', 'base_period': 'universal', ...}
```

Provenance flows into `replication_pack` automatically:

```python
synth_r = sp.synth(df, ...)
rd_r = sp.rdrobust(df_rd, y="y", x="x", c=0)

rp = sp.replication_pack([synth_r, rd_r], "out.zip",
                         data=df, code="analysis.py")
# rp.output_path / lineage.json now contains both runs.
```

`lineage.json` shape:

```json
{
  "n_runs": 2,
  "runs": {
    "9c3aa1bf...": {
      "function": "sp.synth",
      "params": {"outcome": "gdp", "method": "augmented", ...},
      "data_hash": "5c64c6e6b67c",
      "run_id": "9c3aa1bf...",
      "statspai_version": "1.7.2",
      "python_version": "3.11.5",
      "timestamp": "2026-04-27T15:34:55"
    },
    "1874e42d...": {...}
  },
  "data_inputs": [
    {"hash": "5c64c6e6b67c",
     "consumers": [{"function": "sp.synth", "run_id": "9c3aa1bf..."}]}
  ],
  "statspai_version": "1.7.2",
  "python_version": "3.11.5"
}
```

### Aggregation chain (DiD `aggte`)

`sp.did.aggte()` is **chain-aware** — its `Provenance.params` records
both the aggregation choice (`type='simple'` / `'dynamic'` / ...) and
the upstream Callaway-Sant'Anna run that produced its input ATTs:

```python
cs = sp.callaway_santanna(df, y="y", g="g", t="t", i="i")
agg = sp.did.aggte(cs, type="dynamic")
prov = sp.get_provenance(agg)
print(prov.params["upstream_run_id"])     # → '9c3aa1bf'
print(prov.params["upstream_function"])   # → 'sp.did.callaway_santanna'
```

So `lineage.json` traces the full chain: aggregate → producing CS run
→ input data hash.

## Provenance scorecard

As of v1.7.2, **21 estimators** are instrumented:

| Estimator                                                | Phase    |
|---                                                       |---       |
| `sp.regress`                                             | P3       |
| `sp.callaway_santanna`                                   | P3       |
| `sp.did_2x2`                                             | P3       |
| `statspai.regression.iv.iv`                              | P3       |
| `sp.synth` (13-method dispatcher)                        | P4       |
| `sp.did.did_imputation`                                  | P4       |
| `sp.did.aggte` (chain-aware)                             | P4       |
| `sp.did.did_multiplegt`                                  | P4       |
| `sp.rd.rdrobust`                                         | P4       |
| `sp.cic` (Athey-Imbens 2006)                             | **P7**   |
| `sp.cohort_anchored_event_study` (arXiv:2509.01829)      | **P7**   |
| `sp.design_robust_event_study` (Wright 2026, 2601.18801) | **P7**   |
| `sp.gardner_did` / `sp.did_2stage`                       | **P7**   |
| `sp.harvest_did` (Borusyak et al. 2025)                  | **P7**   |
| `sp.did_misclassified` (arXiv:2507.20415)                | **P7**   |
| `sp.stacked_did` (Cengiz et al. 2019)                    | **P7**   |
| `sp.wooldridge_did` (Wooldridge 2021 ETWFE)              | **P7**   |
| `sp.etwfe` (4-branch dispatcher, wrap pattern)           | **P7**   |
| `sp.drdid` (Sant'Anna-Zhao 2020 DR)                      | **P7**   |
| `sp.rd_honest` (Armstrong-Kolesar 2018, 2020)            | **P7**   |
| `sp.rkd` (Card et al. 2015 Regression Kink)              | **P7**   |

The remaining ~904 estimators are scheduled for v1.7.3+ rollouts. To
check whether a specific estimator is instrumented:

```python
r = sp.your_estimator(df, ...)
print(sp.get_provenance(r))   # None if not yet instrumented
```

## Tables — `sp.gt(result)` great_tables adapter

For publication-grade HTML / LaTeX tables, pipe a `RegtableResult`
through Posit's `great_tables`:

```python
import statspai as sp

m = sp.feols("wage ~ trained + edu | year + worker_id", df)
rt = sp.regtable(m, template="aer", title="Returns to Training")

g = sp.gt(rt)            # great_tables.GT instance
g.as_raw_html()          # → embed in Quarto / HTML
g.as_latex()             # → \begin{table}...\end{table}
```

`sp.gt()` accepts:

- `RegtableResult` — full-fidelity (title / notes / journal preset →
  gt theme).
- `PaperTables` — multi-panel with row groups.
- `MeanComparisonResult` — flattens via `to_dataframe()`.
- `DataFrame` — wraps verbatim with optional `rowname_col=`.
- Any object with `to_dataframe()` — duck-typed.

`great_tables` is an **optional** dependency. Install with
`pip install great_tables` — the wider StatsPAI stack imports cleanly
without it; only `sp.gt(...)` requires it at call time.

## Recipes

### AEA submission

```python
import statspai as sp
import pandas as pd

df = pd.read_stata("nlsw88.dta")

q = sp.causal_question(
    treatment="union", outcome="wage",
    data=df, design="did",
    time="year", id="idcode",
    covariates=["age", "edu"],
    estimand="ATT",
    notes="Pre-registered for AER replication review.",
)
draft = q.paper(fmt="qmd")

# Use the AER CSL style:
draft.to_qmd(csl="aer")  # already wired by replication_pack below

sp.replication_pack(
    draft,
    "aer-submission.zip",
    code="analysis.py",
    title="Returns to Union Membership",
    paper_format="qmd",
)
```

Then locally:

```bash
unzip aer-submission.zip -d aer-submission/
cd aer-submission
curl -O $(python -c "import statspai as sp; print(sp.csl_url('aer'))")
quarto render paper/paper.qmd
```

### AEJ: Applied submission with DAG

Same as AER but with an explicit DAG and AEJ CSL:

```python
from statspai.dag.graph import DAG

g = DAG("union -> wage; age -> wage; age -> union; edu -> wage")
draft = q.paper(fmt="qmd", dag=g)
draft.to_qmd(csl="aeja")  # AEJ uses the AER style file

sp.replication_pack(draft, "aeja-submission.zip",
                    code="analysis.py")
```

### Auditable agent run

For an autonomous-agent context where every number must be traceable:

```python
# Agent: 50-line script.
draft = sp.paper(df, query, treatment=t, y=y, fmt="qmd")
rp = sp.replication_pack(
    draft, f"runs/{run_id}.zip",
    code=__file__,           # capture this script verbatim
    title=f"Run {run_id}",
)
print(rp.summary())
# ReplicationPack
# ===============
#   Path     : /runs/abc123.zip
#   Files    : 8
#   StatsPAI : v1.7.2
#   Created  : 2026-04-27T15:34:55
```

Each `lineage.json` then ties any reported number back to the exact
function / params / data hash that produced it — auditable months
later, by a different reviewer, with no shared session state.

## What `sp.paper()` does NOT do

- It does **not** run a hyperparameter sweep — it picks the
  recommendation from `sp.recommend(...)`. Use `sp.spec_curve(...)`
  for multiverse analysis and pass the resulting summary into
  `extra_files=` on `replication_pack`.
- It does **not** call any LLM by default. Pass `llm="auto"` to
  opt in; without it, no network call ever fires.
- It does **not** verify your CSL file exists — Quarto reports the
  error at render time. Run `quarto render` once locally before
  shipping.

## See also

- [`sp.paper()` data → publication-draft pipeline](paper_pipeline.md)
- [LLM-DAG setup](llm_dag_setup.md) — provider, credentials,
  `configure_llm()`
- [Choosing a DiD estimator](choosing_did_estimator.md)
- [Robustness workflow](robustness_workflow.md)
