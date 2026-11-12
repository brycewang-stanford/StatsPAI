---
description: Estimate two-way and high-dimensional fixed-effects regressions in Python with clustered standard errors, matching Stata reghdfe and R fixest.
---

# Fixed effects and high-dimensional fixed effects in Python

**Short answer.** `sp.feols("y ~ x | unit + year", data=df, cluster="unit")`
absorbs any number of fixed effects, clusters standard errors, and reports
within R². It is the Python counterpart of Stata's `reghdfe` and R's
`fixest::feols`, and on StatsPAI's cross-language test suite its coefficients
and standard errors agree with both to within 1e-6 (`sp.parity_status("feols")`).
For classical one-way panels with random effects and a Hausman test, use
`sp.panel`.

## When this applies

- **What FE buys you**: control for anything constant within a unit (or
  within a year, an industry-year, …). Identification comes from variation
  within those groups.
- **Data**: a long panel or any data with group identifiers. Singletons are
  dropped as in `reghdfe`; collinear regressors are dropped and reported.
- **Careful with** staggered treatment dummies: a TWFE coefficient on a
  treatment that switches on at different dates is not an average treatment
  effect when effects vary. See [difference-in-differences](difference-in-differences.md).

## Two-way fixed effects with clustered SEs

```python
import statspai as sp

df = sp.load_dataset("castle_doctrine")      # 50 states x 11 years, real data

twfe = sp.feols("l_homicide ~ post | sid + year", data=df, cluster="sid")
wtd = sp.feols("l_homicide ~ post | sid + year", data=df, weights="popwt",
               vcov={"CRV1": "sid"})
print(sp.regtable([twfe, wtd], model_labels=["TWFE", "population-weighted"]))
```

```text
TWFE population-weighted
post 0.069 0.076**
(0.056) (0.033)
Sid FE Yes Yes
Year FE Yes Yes
Cluster SE sid sid
N 550 550
```

The same two specifications in Stata 18 (`reghdfe l_homicide post, absorb(sid
year) vce(cluster sid)`, with and without `[aw=popwt]`) give identical
coefficients and standard errors (0.069398 / 0.055860 and 0.075533 /
0.033194); see the [castle-doctrine replication](../guides/mixtape_castle_replication.md).

While fitting the weighted model StatsPAI also warns that the weights
concentrate on a few states (weight-effective number of clusters 22.6 of 50),
so cluster-robust intervals may be too short. Warnings like this are written
to the result's diagnostics rather than swallowed.

## More than two dimensions, and other options

```py
sp.feols("y ~ x | firm + year + industry", data=df, cluster="firm")    # three sets of FE
sp.feols("y ~ x | firm + year", data=df, vcov={"CRV1": "firm + year"})  # two-way clustering
sp.fepois("count ~ x | firm + year", df)                                # Poisson (PPML) with FE
sp.panel(df, "y ~ x", entity="firm", time="year", method="re")          # random effects
```

Interacted fixed effects such as `industry^year` currently go through the
optional pyfixest backend (`pip install "statspai[fixest]"`).

The optional Rust kernels (`pip install statspai-hdfe`, installed
automatically where a wheel exists) speed up demeaning on large panels;
`sp.fast.backend_info()` says whether they are in use. Timings against
`fixest` and `pyfixest` are in the [performance guide](../guides/performance.md).

## Stata and R equivalents

| Task | Stata | R | StatsPAI |
| --- | --- | --- | --- |
| Absorb FE, cluster | `reghdfe y x, absorb(id year) vce(cluster id)` | `feols(y ~ x \| id + year, cluster = ~id)` | `sp.feols("y ~ x \| id + year", data=df, cluster="id")` |
| Poisson with FE | `ppmlhdfe` | `fepois()` | `sp.fepois()` / `sp.ppmlhdfe()` |
| FE vs RE | `xtreg, fe` / `xtreg, re` + `hausman` | `plm` + `phtest` | `sp.panel()` |
| Translate a command | — | — | `sp.from_stata("reghdfe y x, absorb(id year)")` |

## Other Python options

`pyfixest` mirrors `fixest` closely and is a strong choice if HDFE regression
is all you need. `linearmodels` offers `PanelOLS` (up to two effects),
`AbsorbingLS` and random effects. See the
[library comparison](python-econometrics-libraries.md).

---

*Tested with StatsPAI 1.43.1 on 2026-11-12. The code and numbers on this page
are run by `tests/test_docs_task_pages.py`.
[Report a problem](https://github.com/brycewang-stanford/StatsPAI/issues).*
