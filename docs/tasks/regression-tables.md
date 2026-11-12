---
description: Make publication-ready regression tables in Python and export them to Word, Excel, LaTeX and Markdown, the equivalent of Stata esttab / outreg2 and R modelsummary.
---

# Publication-ready regression tables in Python

**Short answer.** Pass a list of fitted models to `sp.regtable` and save the
result: the file extension picks the format (`.docx`, `.xlsx`, `.tex`, `.md`,
`.html`). It plays the role of Stata's `esttab` / `outreg2` and R's
`modelsummary`, and it accepts OLS, IV, fixed-effects, GLM and most causal
estimators from StatsPAI in the same table. Standard errors are whatever each
model was fitted with, so robust or clustered errors carry through unchanged.

## Example

```python
import statspai as sp

card = sp.datasets.card_1995()
controls = "exper + expersq + black + south + smsa"
m1 = sp.regress("lwage ~ educ", data=card, vce="robust")
m2 = sp.regress(f"lwage ~ educ + {controls}", data=card, vce="robust")
m3 = sp.ivreg(f"lwage ~ (educ ~ nearc4) + {controls}", data=card, robust="hc1")

tab = sp.regtable([m1, m2, m3], model_labels=["OLS", "OLS + controls", "2SLS"],
                  keep=["educ"])
print(tab.to_latex())
```

```text
& OLS & OLS + controls & 2SLS \\
educ & 0.0521*** & 0.0740*** & 0.1323*** \\
& (0.0029) & (0.0036) & (0.0486) \\
First-stage F & & & 17.51 \\
N & 3,010 & 3,010 & 3,010 \\
```

## Save to Word, Excel, LaTeX

```python
tab.save("table1.docx")   # journal-style Word table
tab.save("table1.xlsx")   # styled Excel sheet
tab.save("table1.tex")    # LaTeX float
```

The same object also has `.to_markdown()`, `.to_html()`, `.to_dataframe()` and
`.to_json()`. Labels, notes, star cut-offs, which statistics to show and
column groups are all keyword arguments; the
[export guide](../guides/exporting-regression-tables.md) lists them and maps
each one to its `esttab` and `modelsummary` counterpart.

## Single results

Every result object exports itself as well, which is handy for causal
estimators with their own layout:

```py
fit.to_word("main.docx")
fit.to_latex(caption="Main results", label="tab:main")
cs_result.to_excel("event_study.xlsx")
```

## Stata and R equivalents

| Task | Stata | R | StatsPAI |
| --- | --- | --- | --- |
| Side-by-side table | `esttab m1 m2 m3` / `outreg2` | `modelsummary(list(m1, m2, m3))` | `sp.regtable([m1, m2, m3])` |
| Word | `esttab using t.rtf` / `outreg2 using t.doc` | `modelsummary(..., output = "t.docx")` | `tab.save("t.docx")` |
| LaTeX | `esttab using t.tex, booktabs` | `modelsummary(..., output = "t.tex")` | `tab.save("t.tex")` |

`sp.esttab`, `sp.outreg2` and `sp.modelsummary` still exist as deprecated thin
wrappers over `sp.regtable`.

## Other Python options

`pyfixest.etable` builds tables for pyfixest models (Word output added in
0.70.0); statsmodels has `summary_col` and LaTeX/HTML summaries for single
models. See the [library comparison](python-econometrics-libraries.md).

---

*Tested with StatsPAI 1.43.1 on 2026-11-12. The code and numbers on this page
are run by `tests/test_docs_task_pages.py`.
[Report a problem](https://github.com/brycewang-stanford/StatsPAI/issues).*
