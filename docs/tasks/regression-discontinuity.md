---
description: Run a regression discontinuity design in Python with rdrobust-style bias-corrected inference, a density (manipulation) test and an RD plot.
---

# Regression discontinuity in Python

**Short answer.** `sp.rdrobust(data=df, y="y", x="running", c=0)` estimates a
sharp RD with MSE-optimal bandwidths and robust bias-corrected confidence
intervals, following the `rdrobust` packages for Stata and R. Add
`sp.rddensity` to test for manipulation of the running variable and
`sp.rdplot` for the binned scatter. On the U.S. Senate data below, bandwidths,
estimates and standard errors match R `rdrobust::rdrobust()` and Stata
`rdrobust` (`sp.parity_status("rdrobust")`). A fuzzy RD takes `fuzzy="d"`.

## When this applies

- **Estimand**: the average effect for units at the cutoff.
- **Assumptions**: potential outcomes are continuous at the cutoff, and units
  cannot sort precisely around it (check the density and covariate balance).
- **Data**: a running variable with enough observations on both sides near the
  cutoff. With a discrete running variable, see the
  [RD chooser](../guides/choosing_rd_estimator.md).

## Example: incumbency advantage in U.S. Senate elections

`x` is the party's vote-share margin at election *t*; `y` its vote share at
*t*+2. The data ship with R's `rdrobust`.

```python
import statspai as sp

senate = sp.datasets.lee_2008_senate()
rd = sp.rdrobust(data=senate, y="y", x="x", c=0)
print(rd.summary())
```

```text
RD Effect: 7.51 ***
Std. Error: (1.74)
[95% CI]: [4.09, 10.92]
...
Conventional 7.4141 1.4587 5.0826 0.0000 4.5551 10.2732
Robust 7.5065 1.7413 4.3110 0.0000 4.0937 10.9193
```

Barely winning raises the vote share two years later by about 7.4 percentage
points (7.5 with bias correction). The headline line reports the robust
bias-corrected estimate and interval, which is what `rdrobust` recommends for
inference.

## Manipulation test and plot

```python
dens = sp.rddensity(data=senate, x="x", c=0)
print(dens.summary())
sp.rdplot(data=senate, y="y", x="x", c=0)
```

```text
P-value: 0.381
```

No evidence of bunching at the cutoff (Cattaneo–Jansson–Ma density test,
p = 0.38).

## Check before you report

- Density test (`sp.rddensity`) and covariate balance at the cutoff
  (`sp.rdbalance`).
- Sensitivity to the bandwidth (`sp.rdbwsensitivity`) and placebo cutoffs
  (`sp.rdplacebo`).
- `sp.rd_honest` gives bias-aware intervals in the Armstrong–Kolesár style;
  local randomization inference is in `sp.rdrandinf`.

## Stata and R equivalents

| Task | Stata | R | StatsPAI |
| --- | --- | --- | --- |
| Estimate | `rdrobust y x, c(0)` | `rdrobust::rdrobust(y, x, c = 0)` | `sp.rdrobust(data=df, y="y", x="x", c=0)` |
| Bandwidths | `rdbwselect` | `rdbwselect()` | `sp.rdbwselect()` |
| Density | `rddensity` | `rddensity()` | `sp.rddensity()` |
| Plot | `rdplot` | `rdplot()` | `sp.rdplot()` |

## Other Python options

The official `rdrobust` Python port (GPL-3.0) from the same authors covers
estimation, bandwidth selection and plots. DoubleML's `RDFlex` adds
machine-learning covariate adjustment. See the
[library comparison](python-econometrics-libraries.md).

---

*Tested with StatsPAI 1.43.1 on 2026-11-12. The code and numbers on this page
are run by `tests/test_docs_task_pages.py`.
[Report a problem](https://github.com/brycewang-stanford/StatsPAI/issues).*
