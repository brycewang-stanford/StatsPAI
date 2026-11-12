---
description: Move a Stata .do-file workflow to Python - read .dta files with labels, map Stata commands to Python calls, run existing Stata lines, and get the same numbers.
---

# From Stata to Python: a migration guide

**Short answer.** Read your `.dta` files with `sp.read_data`, keep the same
estimators under Python names (`regress` → `sp.regress`, `reghdfe` →
`sp.feols`, `ivregress 2sls` → `sp.ivreg`, `csdid` → `sp.callaway_santanna`,
`rdrobust` → `sp.rdrobust`, `esttab` → `sp.regtable`), and check numbers
against your Stata logs as you go. StatsPAI follows Stata's conventions where
they change results: `vce(robust)` and `vce(cluster id)` small-sample factors,
t versus z reference distributions, singleton and collinear-variable
dropping. Existing do-file lines can be run as they are with `sp.stata(...)`
or translated to Python code with `sp.from_stata(...)`.

## 1. Translate one command

```python
import statspai as sp

t = sp.from_stata("reghdfe l_homicide post, absorb(sid year) vce(cluster sid)")
print(t["python_code"])
```

```text
sp.hdfe_ols('l_homicide ~ post | sid + year', data=df, cluster='sid')
```

The translation also reports options it could not carry over
(`untranslated_options`) instead of dropping them silently.

## 2. Run Stata lines on a DataFrame

```python
df = sp.load_dataset("castle_doctrine")
fit = sp.stata("reghdfe l_homicide post, absorb(sid year) vce(cluster sid)", data=df)
print(fit.summary())
```

```text
post 0.0694 0.0559 1.24 0.2200
```

Stata 18 reports the same coefficient and clustered standard error on this
panel (0.069398, 0.055860). `sp.stata` accepts whole snippets: comments,
`///` continuations, `global` / `local` macros, `xtset`, loops and `if`
blocks. Anything only Stata can evaluate is refused with a message rather than
guessed; see [live translators](../guides/translator.md).

## 3. Command map

| Stata | StatsPAI |
| --- | --- |
| `use file.dta` | `df = sp.read_data("file.dta")` (variable and value labels kept) |
| `reg y x, vce(robust)` | `sp.regress("y ~ x", data=df, vce="robust")` |
| `reg y x, vce(cluster firm)` | `sp.regress("y ~ x", data=df, vce="cluster firm")` |
| `reghdfe y x, absorb(id year) vce(cluster id)` | `sp.feols("y ~ x \| id + year", data=df, cluster="id")` |
| `ivregress 2sls y (d = z) x, small` | `sp.ivreg("y ~ (d ~ z) + x", data=df)` |
| `xtreg y x, fe` / `, re` | `sp.panel(df, "y ~ x", entity="id", time="t", method="fe")` / `method="re"` |
| `logit y x` + `margins, dydx(x)` | `fit = sp.logit("y ~ x", data=df)` + `sp.margins(fit)` |
| `csdid y, ivar(id) time(t) gvar(g)` | `sp.callaway_santanna(df, y="y", i="id", t="t", g="g")` |
| `rdrobust y x, c(0)` | `sp.rdrobust(data=df, y="y", x="x", c=0)` |
| `synth` | `sp.synth()` |
| `psmatch2` | `sp.psmatch2()` |
| `test x = 0`, `lincom` | `fit.test(...)`, `fit.lincom(...)` |
| `esttab` / `outreg2` | `sp.regtable([...]).save("table.docx")` |

## 4. Checking that the numbers agree

- Defaults that differ between Stata and other Python libraries (degrees of
  freedom, small-sample factors, which variables are dropped) are listed in
  [reproducing Stata defaults](../guides/stata_defaults.md).
- `sp.parity_status("feols")` shows which Stata and R references a function
  has been compared with, the tolerance and the test file. Only rows marked
  `bit-exact` or `aligned` are claims of agreement.
- Data preparation (`xtset`, lags with gaps, `reshape`, `collapse`, `egen`)
  is covered in [preparing data the Stata way](../guides/data_layer.md).

## What does not carry over

Mata, `ml` programs you wrote yourself and user-written commands without a
StatsPAI counterpart need rewriting. `sp.from_stata` says which part of a
line it could not translate. The companion
[stata-code](https://github.com/brycewang-stanford/stata-code/) MCP server can
run the original do-file next to the Python version when you have a Stata
licence.

---

*Tested with StatsPAI 1.43.1 on 2026-11-12. The code and numbers on this page
are run by `tests/test_docs_task_pages.py`.
[Report a problem](https://github.com/brycewang-stanford/StatsPAI/issues).*
