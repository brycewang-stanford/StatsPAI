# The Mixtape replication suite

Four worked examples from Cunningham's *Causal Inference: The Mixtape*
ship with StatsPAI as **real data plus checked numbers** — not simulated
stand-ins. Every value is pinned in CI against **Stata 18 MP** and, where
a reference exists, **R `did`**.

```python
import statspai as sp

sp.list_replications()                 # what is available
data, guide = sp.replicate('castle_2013')
print(guide)                           # code + expected numbers + caveats
```

---

## What is in the suite

| key | chapter | dataset | what it teaches |
| --- | --- | --- | --- |
| [`castle_2013`](mixtape_castle_replication.md) | 9 — DiD | `castle_doctrine()` | staggered adoption, Bacon decomposition, Callaway–Sant'Anna |
| `texas_1993` | 10 — synthetic control | `texas_prison()` | classic SCM's non-convex V–W problem, and what survives it |
| `sasp_within` | 8 — panel | `sasp_panel()` | the within transformation, three equivalent ways |
| `thornton_2008` | 4 — potential outcomes | `thornton_hiv()` | randomization inference on a real experiment |

Alongside them the older entries — `card_1995`, `abadie_2010`,
`lalonde_1986`, `lee_2008` — cover IV, synthetic control, matching and RD.

---

## Verification status

Read this column before quoting anything.

| replication | agreement with reference software |
| --- | --- |
| `castle_2013` | **bit-parity.** Four TWFE specs ≤1e-6 vs Stata (including `aweight` and 19 collinear drops); `bacon_decomposition` matches `bacondecomp` across all 25 cells; `callaway_santanna` matches R `did` and `csdid` to 1e-9 on estimates *and* standard errors. |
| `sasp_within` | **bit-parity.** Pooled OLS, within, and manual demeaning all ~5e-10 vs Stata. |
| `thornton_2008` | **bit-parity** on the estimate: group means, the simple difference and the HC1 standard error match Stata exactly. The RI p-value is Monte Carlo and is not pinned to a draw. |
| `texas_1993` | **deliberately not parity.** See below. |

### Why `texas_1993` is shipped as a non-parity case

The book's SCM recipe puts four lagged outcomes among the predictors,
which leaves the predictor-weight matrix V weakly identified (Kaul et al.
2015) and makes the nested V–W problem non-convex. Stata `synth` and
`sp.synth` converge to different local optima:

| | donors | mean 1994–2000 gap | pre-RMSE |
| --- | --- | ---: | ---: |
| Stata `synth` | CA .408 IL .360 LA .122 FL .109 | 23,074 | 1227 |
| `sp.synth` | FL .436 NY .311 IL .253 | 23,779 | **865** |

StatsPAI reaches the *lower* pre-treatment RMSE and returns the identical
optimum at 4 and 40 random starts, so neither is wrong on its own
objective. Across five routes — including outcome-only classic SCM
(21,482) and synthdid (19,479) — every estimate is a large positive
effect within a ~30% band, on entirely disjoint donor sets.

**The effect is identified far more robustly than the weights are.**
Report the effect; do not interpret the donor weights, and do not tune
the recipe until the weights look familiar. If a number has to reproduce
across software, use synthdid or the outcome-only recipe, where V is
fixed to the identity and the donor-weight problem is convex with a
unique solution.

---

## What the suite caught

Replications earn their keep by finding things unit tests cannot, because
a package can be perfectly consistent with itself and still wrong.

**`sp.aggte` standard errors were up to 8% too small.** The
Callaway–Sant'Anna aggregation weights are *estimated* cohort shares, and
the variance treated them as fixed — dropping R's `did:::wif` term. Point
estimates were always correct, which is exactly why no internal check
caught it: only a cross-implementation comparison could. Single-cohort
aggregates were unaffected (the shares cancel), so the defect hid in
cross-cohort aggregates. Fixed; see
[`MIGRATION.md`](https://github.com/brycewang-stanford/StatsPAI/blob/main/MIGRATION.md).

**`sp.ri_test` dropped missing rows silently.** Passing `cluster=` where
the cluster id is itself missing shrank the sample without saying so. On
Thornton, four of 2834 rows have no village id, moving the statistic from
0.450552 to 0.451982 — which reads as a discrepancy rather than a smaller
sample. It now warns with the counts.

**Four shipped replication guides contained code that could not run.**
They called `sp.regtable(..., column_labels=...)`, which is not a
parameter. Every guide's code block is now executed in CI.

---

## Two traps the data itself sets

**Castle doctrine: `post` is not `1{year >= effyear}`.** Cheng & Hoekstra
code `post = 1{year > effyear}` because the law was in force for only
part of the adoption year; that year's fractional exposure lives in
`cdl`. The obvious reconstruction silently changes 21 of 550
observations, and it moves the Callaway–Sant'Anna ATT from 0.110 to
0.019 depending on how you then code the cohort.

**SASP: the within transformation destroys twelve of the controls.**
Provider age, race, schooling, BMI and marital status do not vary across
a provider's four sessions. Stata's `xtreg, fe` omits them silently;
StatsPAI raises `NumericalInstability` naming the column, because a
regressor with no identifying variation should be named rather than
quietly discarded.

---

## Running them

```python
# Every guide's code is executable as printed.
data, guide = sp.replicate('sasp_within')
print(guide)

# The datasets stand alone too.
df = sp.datasets.sasp_panel(analytic_sample=True)   # book's 1028-row extract
df = sp.datasets.castle_doctrine(event_time=True)   # adds time_til / gvar
df = sp.datasets.thornton_hiv(complete_case=True)   # n = 2834
df = sp.datasets.texas_prison()
```

The parity tests live in `tests/reference_parity/` — `test_castle_stata_parity.py`,
`test_sasp_within_parity.py`, `test_thornton_ri_parity.py`,
`test_texas_synth_parity.py`, and `test_aggte_r_did_parity.py`. Each records
the exact Stata or R command that produced its reference values, so they can
be regenerated rather than trusted.

Data redistributed from the MIT-licensed
[mixtape repository](https://github.com/scunning1975/mixtape).
