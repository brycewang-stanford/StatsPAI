# Castle doctrine — a real-data staggered DiD replication

This guide replicates Cheng & Hoekstra (2013) — the castle-doctrine
dataset behind Chapter 9 of Cunningham's *Causal Inference: The
Mixtape* — on the **real panel**, not a simulated stand-in.

Every number below is checked in CI against **Stata 18 MP** and **R
`did`**. The checks live in
`tests/reference_parity/test_castle_stata_parity.py`.

> **Why this dataset?** It is the cleanest teaching case of staggered
> adoption in economics: 50 states, 11 years, 21 staggered adopters, and
> a genuine never-treated group. It is also the case where the naive
> answer and the modern answer differ enough to matter.

---

## 0. The data

```python
import statspai as sp

df = sp.datasets.castle_doctrine()
df.shape          # (550, 29)
```

50 US states × 11 years (2000–2010). Between 2005 and 2009, 21 states
expanded "castle doctrine" self-defence law; 29 never did.

| column | meaning |
| --- | --- |
| `l_homicide` | log homicide rate per 100,000 — the outcome |
| `post` | the paper's treatment dummy (**read §1**) |
| `cdl` | fractional castle-doctrine exposure within the year |
| `effyear` | year the law took effect (`NaN` = never treated) |
| `popwt` | state population weight (Stata `aweight`) |
| `sid`, `year` | panel identifiers |

The published extract also carries 44 region × year dummies and 51 state
linear trends. Those are pure design-matrix columns, so StatsPAI
regenerates them on request rather than shipping 95 columns of zeros and
ones:

```python
df = sp.datasets.castle_doctrine(
    region_year_fe=True,   # r20001 … r20104   (44)
    state_trends=True,     # trend_1 … trend_51 (51)
    event_time=True,       # adds time_til and gvar
)
```

---

## 1. The trap: `post` is not `year >= effyear`

Cheng & Hoekstra code

$$\texttt{post} = \mathbf{1}\{\texttt{year} > \texttt{effyear}\}$$

The **adoption year itself is coded untreated**, because the law was in
force for only part of it. That year's fractional exposure lives in
`cdl` — Alabama 2006 is `0.5808`, meaning the law was live for about 58%
of the year.

```python
eff = df["effyear"].fillna(9999)
naive = (df["year"] >= eff).astype(float)
(naive != df["post"]).sum()      # 21 observations flip
```

Rebuilding the treatment dummy the "obvious" way silently changes 21 of
550 observations. This is the single most common way to fail to
reproduce the paper.

---

## 2. The classic answer: two-way fixed effects

The paper's ladder, weighted by state population and clustered on state:

```python
xvar = ['l_police', 'unemployrt', 'poverty', 'l_income',
        'l_prisoner', 'l_lagprisoner', 'blackm_15_24',
        'whitem_15_24', 'blackm_25_44', 'whitem_25_44',
        'l_exp_subsidy', 'l_exp_pubwelfare']
region = [c for c in df.columns if c.startswith('r20')]
trends = [c for c in df.columns if c.startswith('trend_')]

bare = sp.feols('l_homicide ~ post | sid + year',
                data=df, vcov={'CRV1': 'sid'})
wtd  = sp.feols('l_homicide ~ post | sid + year', data=df,
                weights='popwt', vcov={'CRV1': 'sid'})
full = sp.feols(
    'l_homicide ~ post + ' + ' + '.join(xvar + region + trends)
    + ' | sid + year',
    data=df, weights='popwt', vcov={'CRV1': 'sid'})

sp.regtable([bare, wtd, full],
            model_labels=['TWFE', '+ weights', '+ full controls'])
```

| specification | β(post) | cluster SE | Stata 18 MP |
| --- | --- | --- | --- |
| TWFE, unweighted | 0.069398 | 0.055860 | ✅ identical |
| TWFE, `aweight=popwt` | 0.075533 | 0.033194 | ✅ identical |
| + time-varying controls | 0.079635 | 0.030876 | ✅ identical |
| + region × year + state trends | 0.076949 | 0.033938 | ✅ identical |

The last row is the strongest check in the suite: 19 of the 95 extra
regressors are collinear, and it only reproduces if StatsPAI drops
exactly the columns Stata drops.

Read plainly: strengthening self-defence law is associated with roughly
an **8 log-point increase** in homicide — the opposite of deterrence.

---

## 3. Why is TWFE that number? Decompose it

Under staggered adoption, TWFE is a weighted average of every possible
2×2 DiD — including "forbidden" comparisons that use already-treated
states as controls (Goodman-Bacon 2021).

```python
bacon = sp.bacon_decomposition(
    df, y='l_homicide', treat='post', time='year', id='sid')

dec = bacon['decomposition']
clean = dec[dec['type'] == 'Treated vs Untreated']['weight'].sum()
print(f"TWFE = {bacon['beta_twfe']:.4f}, never-treated weight = {clean:.1%}")
# TWFE = 0.0694, never-treated weight = 89.9%
```

All 25 comparisons — estimates *and* weights — match Stata's
`bacondecomp` cell by cell.

**89.9% of the weight sits on clean never-treated comparisons.** That is
unusually benign; many staggered designs are far worse. But the residual
10.1% still bites: several early-vs-late cells are strongly negative
(−0.218, −0.154), dragging the average down.

---

## 4. The modern answer: Callaway–Sant'Anna

```python
df = sp.datasets.castle_doctrine(event_time=True)   # adds `gvar`

cs = sp.callaway_santanna(
    df, y='l_homicide', g='gvar', t='year', i='sid',
    control_group='nevertreated')

sp.aggte(cs, type='simple').estimate     # 0.110383
sp.aggte(cs, type='dynamic').plot()
```

**ATT = 0.1104**, against TWFE's 0.0694 — the heterogeneity-robust
estimate is roughly 60% larger. This matches R `did::aggte` and Stata
`csdid` to 1e-9.

### 4.1 The cohort-coding decision changes the answer

Because the adoption year is only partially treated, there is no
unambiguous cohort variable:

| cohort coding | simple ATT | what it does |
| --- | --- | --- |
| `gvar = effyear` | **0.1104** | clean pre-period base; adoption year counted as fully treated |
| `gvar = effyear + 1` | **0.0194** | consistent with `post`; but the partially treated year becomes the base period |

A factor of 5.7, and the second is indistinguishable from zero. Neither
is simply "right":

- `effyear` keeps the **base period clean** but overstates exposure at
  event time 0.
- `effyear + 1` matches `post` but **contaminates the baseline** with a
  partially treated year, which biases toward zero.

The defensible options are to report both, or to drop the adoption year
entirely and estimate on unambiguous periods. What you should *not* do
is pick one silently.

---

## 5. What this replication buys you

| claim | evidence |
| --- | --- |
| StatsPAI's `feols` matches Stata `xtreg, fe` including `aweight` and collinear-drop behaviour | 4 specifications, ≤1e-6 |
| `bacon_decomposition` matches Stata `bacondecomp` | all 25 cells, estimates and weights |
| `callaway_santanna` matches R `did` and Stata `csdid` | point estimates **and** SEs to 1e-9 |

### A bug this replication caught

Building this page surfaced a real defect. StatsPAI's Callaway–Sant'Anna
*standard errors* were 0.3–8% smaller than R `did` and Stata `csdid`
(which agreed with each other) — point estimates were always correct.

The aggregation weights in Callaway–Sant'Anna are estimated cohort
shares $\hat p_g$, not constants, and `sp.aggte` was treating them as
fixed. That drops a term from the variance (R's `did:::wif`) and makes
the reported SE **anti-conservative**. It is fixed as of the current
release; see `MIGRATION.md`. Cells drawn from a single cohort were never
affected, which is why the discrepancy hid in cross-cohort aggregates
only.

This is the argument for replication suites over unit tests: no
self-consistent test would have caught it, because StatsPAI was
consistent with itself.

---

## References

- Cheng, C. & Hoekstra, M. (2013). Does Strengthening Self-Defense Law
  Deter Crime or Escalate Violence?: Evidence from Expansions to Castle
  Doctrine. *Journal of Human Resources* 48(3), 821–853.
  [doi:10.1353/jhr.2013.0023](https://doi.org/10.1353/jhr.2013.0023)
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in
  treatment timing. *Journal of Econometrics* 225(2), 254–277.
- Callaway, B. & Sant'Anna, P. H. C. (2021). Difference-in-differences
  with multiple time periods. *Journal of Econometrics* 225(2), 200–230.
- Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale
  University Press.

Data redistributed from the MIT-licensed
[mixtape repository](https://github.com/scunning1975/mixtape).
