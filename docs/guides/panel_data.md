# Panel data: fixed effects, random effects, and HDFE

Panel (longitudinal) data — the same units observed repeatedly over
time — is where StatsPAI's `sp.panel` family lives. One dispatcher
covers the classical estimators (FE / RE / between / first-difference /
pooled / two-way / correlated random effects / Arellano-Bond GMM), and
a dedicated high-dimensional-fixed-effects (HDFE) toolkit
(`sp.feols`, `sp.hdfe_ols`, `sp.absorb_ols`, `sp.demean`) mirrors
Stata's `reghdfe` and R's `fixest`. This guide walks the full surface:
when to use what, how to read the output, and the pitfalls that bite
in practice.

## 0. TL;DR flowchart

```text
Do you have repeated observations per unit (unit x time)?
  NO  -> cross-section tools: sp.regress / sp.dml / sp.metalearner
  YES -> Is the regressor of interest a treatment that switches on
         at different times across units (staggered adoption)?
          YES -> this is a DID problem, not a plain panel regression.
                 See docs/guides/choosing_did_estimator.md
          NO  -> How many fixed-effect dimensions?
                  1-2 small (unit, time)   -> sp.panel(method='fe'/'twoway')
                  2+ large (worker, firm,
                  product, market, ...)    -> sp.hdfe_ols / sp.feols
         Is the regressor of interest time-invariant (e.g. gender)?
          YES -> FE cannot estimate it. Use method='re' (after a
                 Hausman test) or method='mundlak' (CRE).
         Is a lag of the outcome on the right-hand side?
          YES -> FE is Nickell-biased. Use sp.xtabond / method='ab'.
```

## 1. Why panel methods: the within transformation

The canonical panel model is

```text
y_it = x_it' b + a_i + e_it
```

where `a_i` is a unit-specific, time-invariant unobservable (worker
ability, firm culture, county geography). If `a_i` is correlated with
`x_it`, pooled OLS is biased — it attributes the effect of `a_i` to
`x`. The **within transformation** subtracts each unit's time mean:

```text
(y_it - ybar_i) = (x_it - xbar_i)' b + (e_it - ebar_i)
```

`a_i` is constant within a unit, so it differences out *exactly* —
no matter how strongly it is correlated with `x`. That is the entire
value proposition of fixed effects: it buys you robustness to **any**
time-invariant confounder, observed or not [@wooldridge2010econometric].

Watch it happen on simulated wage data where unobserved ability drives
both training take-up and wages:

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(42)
n, T = 100, 6
worker = np.repeat(np.arange(n), T)
year = np.tile(np.arange(2010, 2016), n)

ability = rng.normal(0, 1.0, n)                     # unobserved, time-invariant
training = 0.8 * ability[worker] + rng.normal(0, 1.0, n * T)   # confounded
female = rng.integers(0, 2, n)[worker]              # time-invariant covariate
log_wage = (2.0 + 0.30 * training - 0.10 * female
            + ability[worker] + rng.normal(0, 1.0, n * T))

df = pd.DataFrame({"worker": worker, "year": year, "training": training,
                   "female": female, "log_wage": log_wage})

pooled = sp.panel(df, "log_wage ~ training", entity="worker", time="year",
                  method="pooled")
re = sp.panel(df, "log_wage ~ training", entity="worker", time="year",
              method="re")
fe = sp.panel(df, "log_wage ~ training", entity="worker", time="year",
              method="fe")

print(f"pooled OLS : {pooled.params['training']:.3f}")   # 0.641 — badly biased
print(f"random eff.: {re.params['training']:.3f}")       # 0.466 — still biased
print(f"fixed eff. : {fe.params['training']:.3f}")       # 0.303 — truth is 0.30
```

Pooled OLS more than doubles the true effect; RE (a weighted average of
within and between variation) shrinks but does not remove the bias; FE
recovers the truth. Whenever the unit effect is correlated with the
regressor, only the within estimator is consistent.

## 2. The estimator family: `sp.panel(method=...)`

One dispatcher, ten estimators. Method names are case-insensitive and
accept the obvious aliases:

| `method=`     | Aliases                              | What it does                                                  | Use when                                                              |
|---------------|--------------------------------------|---------------------------------------------------------------|-----------------------------------------------------------------------|
| `'fe'`        | `'fixed'`, `'within'`                | Within (entity-demeaned) OLS                                  | Unit effect correlated with regressors — the default                  |
| `'re'`        | `'random'`                           | GLS random effects                                            | Unit effect uncorrelated with regressors (verify via Hausman)         |
| `'be'`        | `'between'`                          | OLS on unit time-means                                        | Long-run cross-sectional relationships                                |
| `'fd'`        | `'first_difference'`                 | OLS on first differences                                      | Unit-root-ish outcomes; serial correlation near random walk           |
| `'pooled'`    | `'pooled_ols'`, `'ols'`              | Plain OLS ignoring panel structure                            | Benchmark only                                                        |
| `'twoway'`    | `'two_way'`, `'2way'`                | Entity + time FE                                              | Common shocks (recessions, policy years) hit all units                |
| `'mundlak'`   | `'mundlak_cre'`                      | Correlated random effects [@mundlak1978pooling]               | Want FE-consistent slopes *and* time-invariant regressors             |
| `'chamberlain'` | `'chamberlain_cre'`                | Chamberlain projection CRE [@chamberlain1982multivariate]     | CRE with period-specific projections                                  |
| `'ab'`        | `'arellano_bond'`, `'gmm'`           | Difference GMM [@arellano1991some]                            | Lagged dependent variable on the RHS (see §7)                         |
| `'hdfe'`      | `'feols'`, `'reghdfe'`               | High-dimensional FE absorption (routes to `sp.hdfe_ols`)      | Many / large FE dimensions (see §5)                                   |

`fd` vs `fe` is a serial-correlation call: both remove `a_i`, but FE is
efficient when `e_it` is i.i.d. while FD is efficient when `e_it` is a
random walk. If the two give materially different answers, that is
itself a diagnostic worth reporting.

To estimate the same model under several methods side-by-side:

```python
tab = sp.panel_compare(df, "log_wage ~ training",
                       entity="worker", time="year",
                       methods=["pooled", "be", "fd", "re", "fe"])
print(tab[["Pooled OLS", "Panel Between", "Panel First Difference",
           "Panel RE (GLS)", "Panel FE (Within)"]])
```

## 3. Reading the output

Every classical method returns a `PanelResults` (a panel-aware
`EconometricResults`) with the standard StatsPAI interface:

```python
fe = sp.panel(df, "log_wage ~ training", entity="worker", time="year",
              method="fe")

print(fe.summary())          # formatted table + diagnostics block
print(fe.params)             # coefficient Series
print(fe.std_errors)         # SE Series
print(fe.conf_int())         # 95% CI DataFrame
print(fe.tidy())             # broom-style long table
print(fe.glance())           # one-row model summary (nobs, R2, F, ...)
```

The diagnostics block of `summary()` reports three R-squareds — read
them carefully:

- **R-squared (within)** — fit of the demeaned regression. This is the
  R² that corresponds to the variation FE actually uses.
- **R-squared (between)** — fit across unit means.
- **R-squared** — for `method='fe'` this is the within R², matching
  Stata's `xtreg, fe` convention. Do not compare it to a pooled-OLS R²:
  the FE specification "explains" all between-unit variation by
  construction via the unit dummies, and the within R² deliberately
  excludes that.

Like every StatsPAI result, `PanelResults` also supports `.to_latex()`,
`.to_word()`, `.to_excel()`, `.plot()`, and `.next_steps()`.

## 4. FE vs RE: the Hausman test and friends

RE is more efficient than FE (it uses between *and* within variation,
and it can estimate time-invariant regressors), but it is only
consistent if `a_i` is uncorrelated with the regressors. The Hausman
test [@hausman1978specification] compares the two coefficient vectors:
under the RE null they should agree.

```python
h = fe.hausman_test()                # FE vs RE
print(h["interpretation"])
# chi2(1) = 130.6310, p = 0.0000. Reject H0: use Fixed Effects.

bp = re.bp_lm_test()                 # pooled OLS vs RE
print(bp["interpretation"])
# LM = 112.0486, p = 0.0000. Reject H0: use Random Effects.

f = fe.f_test_effects()              # joint significance of the unit FE
print(f["interpretation"])
# F(99, 499) = 4.3079, p = 0.0000. Reject H0: entity effects are significant — use FE.

cd = fe.pesaran_cd_test()            # cross-sectional dependence
print(cd["interpretation"])
```

The same tests exist as standalone functions when you have not fitted a
model yet:

```python
h = sp.hausman_test(df, y="log_wage", x=["training"],
                    id="worker", time="year")
print(h["recommendation"])           # 'FE'
print(h["beta_fe"].round(3).to_dict(), h["beta_re"].round(3).to_dict())
```

The full diagnostic battery: Hausman (FE vs RE)
[@hausman1978specification], Breusch-Pagan LM (pooled vs RE)
[@breusch1980lagrange], F-test on the unit effects, and the Pesaran CD
test for cross-sectional dependence [@pesaran2004general].

**Two practical caveats:**

1. The classical Hausman statistic uses the unadjusted FE-RE covariance
   difference, which is not guaranteed positive definite in finite
   samples. When the quadratic form goes negative, StatsPAI clips the
   statistic at 0 (so p = 1). If you see `statistic: 0` while
   `beta_fe` and `beta_re` differ visibly, do not read it as evidence
   for RE — inspect the coefficient vectors directly, or use the
   Mundlak regression below.
2. The classical test is invalid under heteroskedasticity / clustering.

The **Mundlak correlated-random-effects** regression
[@mundlak1978pooling] fixes both: add unit means of the regressors to
an RE model. The coefficient on `x` reproduces the FE estimate, and a
test on the `_mean_x` coefficients is a regression-based,
cluster-robust-friendly Hausman alternative
[@wooldridge2010econometric]:

```python
m = sp.panel(df, "log_wage ~ training", entity="worker", time="year",
             method="mundlak")
print(m.params.round(4))
# training       0.3026   <- identical to the FE slope
# _mean_training 0.8077   <- significant => a_i correlated with x => FE-style
```

As a bonus, the CRE framework lets you keep time-invariant regressors
(like `female` above) in the model — something FE itself cannot do
(see §8.3).

## 5. High-dimensional fixed effects: `sp.feols`, `sp.hdfe_ols`, `sp.absorb_ols`

`sp.panel(method='fe')` materialises the panel structure via
`linearmodels`-style entity/time effects — fine for one entity
dimension plus time. Once you need worker FE **and** firm FE **and**
year FE (or origin x destination, judge x court, ...), you want FE
*absorption*: iteratively demean rather than ever forming dummy
matrices [@gaure2013multiple; @correia2017linear]. The formula grammar
is the `fixest` / `reghdfe` pipe:

```text
"y ~ x1 + x2 | fe1 + fe2 + fe3"
```

StatsPAI ships three layers; pick by dependency budget and scale:

| Tool           | Backend                                  | When it is the right tool                                                  |
|----------------|-------------------------------------------|----------------------------------------------------------------------------|
| `sp.hdfe_ols`  | Native (NumPy, Numba-accelerated if installed) | Default. Zero extra dependencies, multi-way FE, multi-way cluster, built-in wild bootstrap |
| `sp.feols`     | pyfixest (`pip install statspai[fixest]`) | You want the full `fixest` feature set: IV in-formula, `csw()` multiple estimation, `i()` interactions, `etable` |
| `sp.absorb_ols` / `sp.demean` | Same engine as `hdfe_ols`, array-level | Building your own estimator on top of FE absorption                        |
| `sp.fast.feols`| Rust / JAX accelerated path               | Very large N; GPU bootstrap (see the [GPU guide](gpu_acceleration.md))     |

> **Backend note.** `sp.hdfe_ols` runs everywhere: the within
> transformation uses Numba-compiled kernels when Numba is available
> and transparently falls back to pure NumPy otherwise. The optional
> Rust extension (`statspai_hdfe`, built with maturin) and the JAX path
> are exposed under `sp.fast.*`; nothing in this section requires them.

A worker-firm wage regression with three-way FE:

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(0)
n_workers, n_firms, T = 200, 30, 5
N = n_workers * T
worker = np.repeat(np.arange(n_workers), T)
year = np.tile(np.arange(2015, 2020), n_workers)
firm = rng.integers(0, n_firms, N)                  # workers switch firms
w_fe = rng.normal(0, 0.5, n_workers)[worker]
f_fe = rng.normal(0, 0.5, n_firms)[firm]
tenure = rng.uniform(0, 10, N)
log_wage = 2.0 + 0.04 * tenure + w_fe + f_fe + rng.normal(0, 0.3, N)
jobs = pd.DataFrame({"worker": worker, "firm": firm, "year": year,
                     "tenure": tenure, "log_wage": log_wage})

r = sp.hdfe_ols("log_wage ~ tenure | worker + firm + year",
                data=jobs, cluster="firm")
print(f"tenure: {r.params['tenure']:.4f} (SE {r.std_errors['tenure']:.4f})")
print("groups per FE:", r.n_fe)                 # [200, 30, 5]
print("within R2:", round(r.r2_within, 3))
print("singletons dropped:", r.n_singletons_dropped)
print("SE type:", r.se_type)                    # 'cluster'
```

The result (`FEOLSResult`) reports `r2_within`, the degrees of freedom
absorbed by the FEs (`dof_fe`), the number of groups per FE dimension
(`n_fe`), and keeps the fitted `absorber` so you can re-project new
columns onto the same FE structure.

The same model through the pyfixest backend — handy when you want
`fixest`-style multiple estimation or IV-in-formula syntax:

```python
r_pf = sp.feols("log_wage ~ tenure | worker + firm + year",
                data=jobs, vcov={"CRV1": "firm"})
print(r_pf.params.round(4))      # tenure  0.0405 — matches sp.hdfe_ols
```

And the array-level primitives, for building custom estimators:

```python
# Within transformation only: residualise tenure on worker + firm FE
tenure_dm, keep_mask = sp.demean(jobs["tenure"].to_numpy(),
                                 jobs[["worker", "firm"]])
print(abs(tenure_dm.mean()) < 1e-8)              # True — demeaned

# Full absorbed OLS on raw arrays
out = sp.absorb_ols(jobs["log_wage"].to_numpy(),
                    jobs[["tenure"]].to_numpy(),
                    fe=jobs[["worker", "firm", "year"]],
                    cluster=[jobs["firm"].to_numpy()])
print(np.round(out["coef"], 4), np.round(out["se"], 4))
```

`sp.panel(..., method='hdfe')` routes to `sp.hdfe_ols`, so the
dispatcher covers this case too — if the formula has no `|`, the
`entity` / `time` columns are bolted on as absorbed FEs for you.

For count outcomes (patents, trade flows) the Poisson analogues are
`sp.fepois` (pyfixest) and `sp.ppmlhdfe` (native, mirroring Stata's
`ppmlhdfe` [@correia2020fast]) — same pipe syntax, same clustering
options.

## 6. Cluster-robust inference

Panel residuals are serially correlated within units almost by
definition (`a_i` may be gone, but `e_it` rarely is i.i.d.). Default
OLS standard errors are therefore generally too small. The standard
fix is clustering.

**Which level?** Cluster at the level where you believe errors are
correlated — at minimum the level at which the regressor of interest
varies or treatment was assigned. In a unit x time panel that usually
means clustering by **unit** (which handles arbitrary within-unit
serial correlation). For state-level policies in individual-level data,
cluster by **state**, not person. When in doubt, the practitioner's
guide of Cameron-Miller-style advice updated for the modern toolkit is
[@mackinnon2023cluster].

```python
r_iid = sp.hdfe_ols("log_wage ~ tenure | worker + firm", data=jobs)
r_cl1 = sp.hdfe_ols("log_wage ~ tenure | worker + firm", data=jobs,
                    cluster="firm")                      # one-way
r_cl2 = sp.hdfe_ols("log_wage ~ tenure | worker + firm", data=jobs,
                    cluster=["firm", "year"])            # two-way CGM
r_wild = sp.hdfe_ols("log_wage ~ tenure | worker + firm", data=jobs,
                     cluster="firm", wild=True,
                     wild_n_boot=499, wild_seed=1)       # wild cluster bootstrap

for name, r in [("iid", r_iid), ("cluster(firm)", r_cl1),
                ("two-way", r_cl2), ("wild bootstrap", r_wild)]:
    print(f"{name:15s} SE = {r.std_errors['tenure']:.4f}  [{r.se_type}]")
```

- **One-way** clustering is Liang-Zeger CR1.
- **Two-way / N-way** clustering uses the Cameron-Gelbach-Miller
  inclusion-exclusion estimator [@cameron2011robust] — pass a list to
  `cluster=`. With a fitted `EconometricResults` you can also apply it
  post hoc via `sp.twoway_cluster(result, data, "firm", "year")`.
- **Few clusters** (rule of thumb: fewer than ~30-50, or very unequal
  cluster sizes) makes CR1 unreliable. Use the **wild cluster
  bootstrap** [@cameron2008bootstrap; @mackinnon2017wild] — built into
  `sp.hdfe_ols(wild=True)` with Webb weights by default, or standalone
  via `sp.wild_cluster_bootstrap(...)` (which defaults to Rademacher
  weights — pass `weight_type="webb"` when clusters are very few). A
  CR2 small-sample correction is available as
  `sp.cr2_se(result, data, cluster=...)`.

For `sp.panel` classical methods, pass `cluster="worker"` (one-way) the
same way; for `sp.feols`, use the pyfixest vcov spec, e.g.
`vcov={"CRV1": "firm"}`.

## 7. Dynamic panels: lagged outcomes and Nickell bias

Putting `y_{i,t-1}` on the right-hand side breaks FE: the demeaned lag
is mechanically correlated with the demeaned error, biasing the AR
coefficient downward by O(1/T) — severe in short panels. The standard
remedy is Arellano-Bond difference GMM, which first-differences out
`a_i` and instruments the differenced lag with deeper lags in levels
[@arellano1991some]:

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(5)
n, T, burn = 200, 10, 50
a = rng.normal(0, 1, n)
rows = []
for i in range(n):                       # AR(1) panel, rho=0.5, beta=0.3
    y_prev = a[i] / 0.5
    for s in range(burn + T):
        x = rng.normal()
        y_new = 0.5 * y_prev + 0.3 * x + a[i] + rng.normal(0, 0.5)
        if s >= burn:
            rows.append((i, s - burn, y_new, x))
        y_prev = y_new
dyn = pd.DataFrame(rows, columns=["id", "t", "y", "x"])

# Hand-made lag + FE: the Nickell-biased benchmark
dyn["y_lag"] = dyn.groupby("id")["y"].shift(1)
fe_dyn = sp.panel(dyn.dropna(), "y ~ y_lag + x",
                  entity="id", time="t", method="fe")
print("FE+lag :", fe_dyn.params.round(3).to_dict())
# {'y_lag': 0.369, 'x': 0.293}  <- AR coefficient biased down from 0.5

ab = sp.panel(dyn, "y ~ x", entity="id", time="t", method="ab", lags=1)
print("AB GMM :", ab.params.round(3).to_dict())
# {'L1.y': 0.464, 'x': 0.282}   <- close to truth (0.5, 0.3)
```

The hand-made lag in the FE line exists only to exhibit the bias. For
`method='ab'` do **not** create the lag column yourself — `method='ab'`
(and the standalone `sp.xtabond`, validated against Stata's `xtabond`)
adds `L1.y` automatically; supplying your own lag as a regressor would
treat it as exogenous and double-count it. For instrument-count and
specification guidance (too many instruments overfit the endogenous
lag), see [@roodman2009xtabond]. Blundell-Bond system GMM
[@blundell1998initial] is recognised by the dispatcher but deliberately
raises `NotImplementedError` until it has a Stata-parity reference —
StatsPAI fails loudly rather than shipping an unvalidated estimator.

If your outcome may be nonstationary (long macro panels), test before
running levels regressions:

```python
ur = sp.panel_unitroot(dyn, variable="y", id="id", time="t", test="ips")
print(ur.summary())     # Im-Pesaran-Shin panel unit root test
```

The IPS test [@im2003testing] averages unit-by-unit ADF statistics;
rejection means the panels are stationary. Mind the power: on this
`dyn` panel (T = 10, AR coefficient 0.5) IPS *fails to reject* the
unit-root null even though the process is stationary — persistent
series in short panels routinely defeat unit-root tests. Treat a
non-rejection as "cannot tell", not "unit root confirmed".

## 8. Common pitfalls

### 8.1 Unbalanced panels

FE and the HDFE tools handle unbalanced panels natively — but *check
how* unbalanced you are before trusting the answer, and ask why
observations are missing (attrition correlated with the outcome breaks
FE just like any other selection problem).

```python
from statspai.datasets import mpdta
panel_df = mpdta()          # simulated replica of a county x year panel (§9)
balanced = sp.balance_panel(panel_df, entity="countyreal", time="year")
print(len(panel_df), "->", len(balanced))   # 2500 -> 2500 (already balanced)
```

`sp.balance_panel` keeps only entities observed in every period — use
it as a robustness check (re-estimate on the balanced subsample), not
as a default.

### 8.2 Singleton groups

A unit observed once (or a group with one member after nesting) is
"explained" perfectly by its own fixed effect. Leaving singletons in
deflates clustered standard errors [@correia2015singletons].
`sp.hdfe_ols` drops them by default and tells you:

```python
toy = pd.DataFrame({
    "y": rng.normal(size=8),
    "x": rng.normal(size=8),
    "g": [1, 1, 2, 2, 3, 4, 5, 5],       # groups 3 and 4 are singletons
})
r = sp.hdfe_ols("y ~ x | g", data=toy)
print(r.n_singletons_dropped, r.n_obs)   # 2 dropped, 6 used
```

Watch `n_singletons_dropped` in any multi-way FE regression — losing a
large share of the sample to singletons is a sign your FE structure is
too fine for the data.

### 8.3 Time-invariant regressors

The within transformation annihilates anything constant within a unit.
Asking FE for the coefficient on `female` fails loudly:

```python
try:
    sp.panel(df, "log_wage ~ training + female",
             entity="worker", time="year", method="fe")
except Exception as e:
    print(type(e).__name__)              # AbsorbingEffectError
```

Options, in order of preference: (1) if Hausman does not reject, use
`method='re'`; (2) use `method='mundlak'` — CRE keeps the
time-invariant coefficient while reproducing FE slopes on time-varying
regressors (§4); (3) accept that the question is cross-sectional and
report the between estimate for that variable separately.

### 8.4 Within R² vs overall R²

A within R² of 0.09 next to a pooled R² of 0.29 (as in our §1 example)
is not a worse model — it is a different denominator. The within R²
only scores variation *inside* units, which is the only variation FE
uses for identification. Never compare R² across `method='pooled'` and
`method='fe'` to "choose" a model; use the Hausman/BP/F battery of §4.

### 8.5 Serial correlation

Within-unit serial correlation in `e_it` is the rule, not the
exception, and it inflates t-statistics dramatically if ignored.
Minimum hygiene: cluster at the unit level (§6). If T is long and
errors are near a random walk, prefer `method='fd'`. The Pesaran CD
test (§4) flags *cross-sectional* dependence — common shocks across
units — which clustering by unit does not fix (add time FE, or
cluster two-way on unit and time).

### 8.6 TWFE is not a DID estimator under staggered adoption

A two-way FE regression of `y` on a treatment dummy is *numerically*
fine but *causally* treacherous when units adopt treatment at
different times and effects are heterogeneous: already-treated units
serve as controls with possibly negative weights
[@goodmanbacon2021difference]. Diagnose with
`sp.bacon_decomposition`, and estimate with
`sp.callaway_santanna` [@callaway2021difference] or the two-way
Mundlak / ETWFE route [@wooldridge2021two]. Full decision tree:
[Choosing a DID estimator](choosing_did_estimator.md).

## 9. Worked example: county teen employment (`mpdta`)

The bundled `mpdta` dataset is a **simulated replica** of the `mpdta`
county-year panel from R's `did` package (the example data of
[@callaway2021difference]) — not the original data. It preserves the
original's structure: 500 counties x 5 years of log teen employment,
three staggered minimum-wage cohorts (2004 / 2006 / 2007) plus a
never-treated group, county-clustered residuals, and a built-in ATT of
about -0.04. Numbers below describe this simulated panel, not a
real-world minimum-wage finding. It ties the pieces together:

```python
import statspai as sp
from statspai.datasets import mpdta

panel_df = mpdta()
print(panel_df.columns.tolist())
# ['countyreal', 'year', 'lemp', 'first_treat', 'treat']

# 1. Structure check: is the panel balanced?
balanced = sp.balance_panel(panel_df, entity="countyreal", time="year")
assert len(balanced) == len(panel_df)            # 500 x 5, fully balanced

# 2. Two-way FE regression, clustered by county — three equivalent routes
tw = sp.panel(panel_df, "lemp ~ treat", entity="countyreal", time="year",
              method="twoway", cluster="countyreal")
hd = sp.hdfe_ols("lemp ~ treat | countyreal + year",
                 data=panel_df, cluster="countyreal")
via_dispatch = sp.panel(panel_df, "lemp ~ treat", entity="countyreal",
                        time="year", method="hdfe", cluster="countyreal")

print(f"twoway : {tw.params['treat']:.4f} (SE {tw.std_errors['treat']:.4f})")
print(f"hdfe   : {hd.params['treat']:.4f} (SE {hd.std_errors['treat']:.4f})")
print(f"dispatch: {via_dispatch.params['treat']:.4f}")
# all three: -0.0375 — close to the simulated ATT of about -0.04
```

All three routes agree to machine precision — `method='twoway'`
materialises entity/time effects, while the HDFE routes absorb them
iteratively; same estimator, different plumbing.

**But stop before putting that -0.0375 in a paper.** Treatment here is
*staggered* (`first_treat` is 2004, 2006, or 2007, with `first_treat = 0`
marking the never-treated cohort), so this static TWFE coefficient is
exactly the object §8.6 warns about. The honest next
step is heterogeneity-robust DID:

```python
cs = sp.callaway_santanna(panel_df, y="lemp", g="first_treat",
                          t="year", i="countyreal")
print(f"CS overall ATT: {cs.estimate:.4f} (SE {cs.se:.4f})")
```

The panel toolkit got you a clean, correctly-clustered TWFE baseline in
one line each; the [DID guide](choosing_did_estimator.md) takes over
from here.

## 10. Stata / R command mapping

| Task                          | Stata                               | R                                          | StatsPAI                                                                 |
|-------------------------------|--------------------------------------|---------------------------------------------|---------------------------------------------------------------------------|
| Declare panel                 | `xtset id year`                      | `pdata.frame(df, c("id","year"))`           | not needed — pass `entity=` / `time=` per call                            |
| Fixed effects (within)        | `xtreg y x, fe`                      | `plm(y ~ x, model="within")`                | `sp.panel(df, "y ~ x", entity="id", time="year", method="fe")`            |
| Random effects                | `xtreg y x, re`                      | `plm(y ~ x, model="random")`                | `sp.panel(..., method="re")`                                              |
| Between                       | `xtreg y x, be`                      | `plm(y ~ x, model="between")`               | `sp.panel(..., method="be")`                                              |
| First difference              | `reg d.y d.x`                        | `plm(y ~ x, model="fd")`                    | `sp.panel(..., method="fd")`                                              |
| Hausman test                  | `hausman fe re`                      | `phtest(fe, re)`                            | `fe.hausman_test()` or `sp.hausman_test(df, y=, x=, id=, time=)`          |
| BP LM test                    | `xttest0`                            | `plmtest(pooled, type="bp")`                | `re.bp_lm_test()`                                                         |
| Multi-way FE absorption       | `reghdfe y x, absorb(i j t)`         | `feols(y ~ x \| i + j + t)`                 | `sp.hdfe_ols("y ~ x \| i + j + t", data=df)` or `sp.feols(...)`           |
| Clustered SE                  | `, vce(cluster id)`                  | `cluster = ~id`                             | `cluster="id"`  (`vcov={"CRV1": "id"}` for `sp.feols`)                    |
| Two-way clustered SE          | `reghdfe ..., vce(cluster id year)`  | `cluster = ~id + year`                      | `cluster=["id", "year"]`                                                  |
| Wild cluster bootstrap        | `boottest`                           | `fwildclusterboot`                          | `sp.hdfe_ols(..., wild=True)` / `sp.wild_cluster_bootstrap(...)`          |
| FE Poisson (PPML)             | `ppmlhdfe y x, absorb(i t)`          | `fepois(y ~ x \| i + t)`                    | `sp.ppmlhdfe("y ~ x \| i + t", data=df)` or `sp.fepois(...)`              |
| Dynamic panel GMM             | `xtabond y x`                        | `pgmm(...)`                                 | `sp.xtabond(df, y="y", x=["x"], id="id", time="t")`                       |
| Panel unit root               | `xtunitroot ips y`                   | `purtest(y, test="ips")`                    | `sp.panel_unitroot(df, variable="y", id="id", time="t", test="ips")`      |

More R-to-StatsPAI translations (including `fixest` IV syntax and
multiple-estimation sugar): [Migrating from R](migration-from-r.md).

## 11. Choosing an estimator: decision summary

1. **Default to FE** (`method='fe'`, or `'twoway'` when common time
   shocks are plausible — which is almost always). Cluster by unit.
2. **Run the §4 battery** — Hausman, BP-LM, F-test. Switch to RE only
   when Hausman does not reject *and* you need its efficiency or a
   time-invariant coefficient; prefer `method='mundlak'` over a raw RE
   when in doubt.
3. **Escalate to HDFE** (`sp.hdfe_ols` / `sp.feols`) as soon as you
   have a second large FE dimension. Watch `n_singletons_dropped`.
4. **Lagged outcome on the RHS?** `sp.xtabond` / `method='ab'`, never
   FE with a hand-made lag.
5. **Treatment that switches on over time?** You are doing DID —
   leave this guide: [Choosing a DID estimator](choosing_did_estimator.md).
6. **Few clusters?** Wild bootstrap (`wild=True`) before you trust any
   p-value.

Related guides:

- [Choosing a DID estimator](choosing_did_estimator.md) — staggered
  adoption, event studies, Bacon decomposition.
- [Robustness workflow](robustness_workflow.md) — the layered
  identification / specification / sensitivity checklist.
- [Migrating from R](migration-from-r.md) — `fixest` / `plm` / `did`
  command translations.
- [GPU acceleration](gpu_acceleration.md) — `sp.fast.feols` and the
  JAX bootstrap path for very large panels.
- [Exporting regression tables](exporting-regression-tables.md) —
  `.to_latex()` / `.to_word()` for the final paper.
