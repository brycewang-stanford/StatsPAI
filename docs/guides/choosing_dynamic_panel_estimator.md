# Choosing a dynamic panel estimator

When the lagged dependent variable is on the right-hand side —

$$y_{it} = \rho\,y_{i,t-1} + x_{it}'\beta + \alpha_i + \varepsilon_{it}$$

— the usual panel toolkit breaks. Pooled OLS is inconsistent because
$y_{i,t-1}$ contains $\alpha_i$. The within (fixed-effects) estimator is
*also* inconsistent, and not just in small samples: demeaning correlates
$y_{i,t-1}$ with the demeaned error, producing the **Nickell (1981) bias**
of order $1/T$ that does not vanish as $N\to\infty$. With $T=6$ and
$\rho=0.5$ the within estimator lands around $0.33$.

This guide covers the GMM family that fixes it.

```python
import statspai as sp
```

---

## 1. The one-minute answer

| Situation | Use |
| --- | --- |
| Short $T$, moderately persistent series ($\rho$ well below 1) | `sp.xtabond(...)` — difference GMM |
| Persistent series ($\rho$ near 1), or difference GMM returns $\hat\rho \ge 1$ | `sp.xtdpdsys(...)` — system GMM |
| Panel with interior holes | add `orthogonal=True` |
| $T$ large enough that instruments outnumber units | add `collapse=True` |
| Heteroskedasticity suspected (i.e. almost always) | `twostep=True, robust=True` |
| Anything with a time dimension shared across units | `time_dummies=True` |
| Firms nested in industries / regions | `cluster='ind'` |
| Sanity check that instruments aren't driving the result | `method='ah'` |

A defensible default for a typical short panel:

```python
res = sp.xtdpdsys(
    df, y="y", x=["x1", "x2"], id="firm", time="year",
    twostep=True,        # efficient weight
    robust=True,         # Windmeijer-corrected SEs
    collapse=True,       # keep the instrument count sane
    time_dummies=True,   # absorb common shocks
)
print(res.summary())
```

---

## 2. Difference GMM — `sp.xtabond`

Arellano & Bond (1991) first-difference the equation, which removes
$\alpha_i$:

$$\Delta y_{it} = \rho\,\Delta y_{i,t-1} + \Delta x_{it}'\beta + \Delta\varepsilon_{it}$$

$\Delta y_{i,t-1}$ is still endogenous ($y_{i,t-1}$ appears in
$\Delta\varepsilon_{it}$), but $y_{i,t-2}, y_{i,t-3}, \dots$ are not. Each
available deeper lag becomes a separate moment condition for that period's
equation — the block-diagonal instrument matrix.

```python
sp.xtabond(df, y="n", x=["l(0/1).w", "l(0/2).k"], id="id", time="year", lags=2)
```

Lag-operator syntax (`l(0/2).k`, `L2.k`, `L.w`) is accepted directly, so the
canonical Arellano-Bond (1991) employment equation is one call.

**Equivalent to** Stata `xtabond ..., noconstant`, and matched to machine
precision — coefficients, standard errors, AR(1)/AR(2), Sargan and Hansen —
on the `abdata` panel.

### When difference GMM fails

As $\rho \to 1$ the level $y_{i,t-2}$ carries almost no information about
$\Delta y_{i,t-1}$: the instruments are weak, and the estimate is biased and
imprecise. The symptom is unmistakable — $\hat\rho$ drifts toward or past
1.0. On a simulated panel with $\rho = 0.9$, $T = 6$, $N = 200$, difference
GMM averages $0.61$ across draws (bias $-0.29$) while system GMM averages
$0.97$.

---

## 3. System GMM — `sp.xtdpdsys` / `method='system'`

Blundell & Bond (1998) keep the differenced equation *and* stack the level
equation, instrumenting the latter with lagged **differences**:

$$\mathbb{E}\!\left[\Delta y_{i,t-1}\,(\alpha_i + \varepsilon_{it})\right] = 0.$$

This is far more informative when the series is persistent, and it
identifies an intercept.

```python
sp.xtdpdsys(df, y="n", x=["w", "k"], id="id", time="year", twostep=True)
```

**The price** is an extra assumption: each unit's deviation from its
long-run mean must be uncorrelated with $\alpha_i$ — roughly, the process
must be in steady state, with no systematic relationship between initial
conditions and fixed effects. That assumption is testable, and you should
test it:

```python
res.model_info["difference_in_hansen"]["GMM instruments for levels"]
# {'hansen_excluding': 56.15, 'df_excluding': 27,
#  'statistic': 5.32, 'df': 7, 'pvalue': 0.62}
```

A small p-value there is evidence against the level moments — fall back to
difference GMM rather than keeping the more precise but invalid estimate.

**Equivalent to** `xtabond2 ... robust` (Stata), matched to machine precision
in one-step, two-step Windmeijer and collapsed configurations.

---

## 4. The three knobs that matter most

### `collapse=True` — instrument proliferation

The uncollapsed instrument count grows as $O(T^2)$. Too many moments and
three things go wrong at once: the two-step weight matrix becomes
near-singular, the estimate is biased *toward* the (Nickell-biased) within
estimator, and the Hansen test loses all power — its p-value is pushed
toward 1.0, which reads as reassurance while being uninformative.

`collapse=True` (Roodman 2009) uses one instrument per lag *distance*
instead of one per (period, distance) pair. On `abdata` that is 7
instruments instead of 28.

StatsPAI warns when the instrument count reaches the number of units, and
records the count in `model_info["n_instruments"]`. **Report it.** A paper
that reports a Hansen p-value without the instrument count has not reported
the Hansen test.

### `orthogonal=True` — gaps

Forward orthogonal deviations (Arellano & Bover 1995) replace each
observation by its deviation from the mean of its *available future*
observations, scaled to keep the errors serially uncorrelated. First
differencing destroys the equations on **both** sides of a hole; forward
deviations lose only one. On gappy panels this is a real efficiency gain.

### `twostep=True, robust=True` — inference

One-step SEs assume homoskedasticity. Two-step is efficient under arbitrary
heteroskedasticity but its conventional SEs are severely downward biased in
short panels — the Windmeijer (2005) correction fixes that, and StatsPAI
applies it automatically when `robust=True`. `twostep=True, robust=False`
returns the biased SEs and warns.

---

## 5. Instrument classes

By default every variable in `x=` is treated as **strictly exogenous** and
enters as a single $\Delta x$ instrument column. Most applied specifications
need more:

| Class | Assumption | Argument |
| --- | --- | --- |
| Strictly exogenous | $\mathbb{E}[x_{is}\varepsilon_{it}]=0$ for all $s,t$ | `x=[...]` |
| Predetermined | $=0$ for $s \le t$ (feedback from past shocks allowed) | `predetermined=[...]`, lags 1+ |
| Endogenous | correlated with $\varepsilon_{it}$ | `endogenous=[...]`, lags 2+ |

```python
sp.xtabond(
    df, y="n", x=["l(0/2).k"],          # capital: strictly exogenous
    predetermined=["l(0/1).w"],          # wages respond to past shocks
    id="id", time="year", lags=2,
)
```

Lag windows are **absolute** — counted from the equation period, matching
`xtabond2`'s `gmm(x, lag(a b))`. Stata's older `xtabond` counts *further*
lags beyond the deepest regressor lag, so
`xtabond ..., pre(w, lagstruct(p, .))` corresponds to
`predetermined=['l(0/p).w'], predetermined_lags=(p + 1, None)`.

---

## 5b. Beyond two steps

`steps=` generalises `twostep`:

| `steps` | What it does |
| --- | --- |
| `1`, `2` | one- and two-step (`steps=2` is exactly `twostep=True`) |
| `3`, `4`, … | repeat the recursion — re-estimate the weight at the current residuals, re-solve |
| `'iterated'` | run the recursion to a fixed point, where the coefficients and the weight they imply are mutually consistent |
| `'cue'` | continuously-updated: re-evaluate the weight *inside* the objective, so the estimate never depends on preliminary residuals at all |

Two-step's dependence on the first-step residuals is exactly what the
Windmeijer correction exists to patch; iterated and CUE remove it instead.
The price is that CUE optimises a non-convex objective numerically —
`model_info['converged']` tells you whether it got there.

On a heavily over-identified fit these can sit well away from the two-step
estimate. That is information about the instrument set, not noise: try
`collapse=True` and see whether they converge on each other.

## 5c. Anderson–Hsiao as a robustness check

```python
sp.xtabond(df, y="n", x=["w", "k"], id="id", time="year", method="ah")
```

Anderson & Hsiao (1981) use a *single* pooled instrument for the differenced
lagged dependent variable — `y_{t-2}` in levels, or `Δy_{t-2}` with
`ah_instrument='differences'` — rather than the block-diagonal set. It is
consistent but inefficient, and that is the point: with one instrument,
instrument proliferation cannot be driving the answer. A large gap between
`method='ah'` and `sp.xtabond`'s default is evidence about the instrument
set rather than about the data.

## 5d. Clustering

```python
sp.xtabond(df, y="n", x=["w", "k"], id="id", time="year", cluster="ind")
```

The moment conditions are summed within a unit by construction, so the
cluster must be at least as **coarse** as the unit — industry, region,
cohort. A finer variable raises rather than silently producing
anti-conservative standard errors. Only the meat of the sandwich re-groups;
the one-step weight stays a within-unit object.

Two-step estimation with fewer clusters than moment conditions is refused:
the efficient weight is the inverse of a covariance whose rank cannot exceed
the cluster count, so the estimate would be an artefact of whichever
generalized inverse ran. Use `collapse=True` to get the moment count under
the cluster count, or stay with one-step (whose cluster-robust standard
errors are valid at any cluster count).

---

## 6. Reading the diagnostics

```python
mi = res.model_info
mi["ar1_z"], mi["ar1_p"]        # should reject  — MA(1) is mechanical
mi["ar2_z"], mi["ar2_p"]        # should NOT reject
mi["hansen_stat"], mi["hansen_p"]
mi["n_instruments"], mi["n_units"]
mi["difference_in_hansen"]
```

- **AR(1)** rejecting is expected: differencing induces MA(1) by
  construction. AR(1) *not* rejecting is the surprising outcome.
- **AR(2)** rejecting means the level errors are serially correlated, which
  invalidates $y_{i,t-2}$ as an instrument. Fix it by deepening the lag
  window (`gmm_lags=(3, None)`) rather than by ignoring it.
- **Hansen J** is the over-identification test to report; it is robust to
  heteroskedasticity and is now computed for one-step fits too. The Sargan
  statistic is reported alongside but is not robust.
- A Hansen p-value of exactly 1.00, or anything above ~0.9 with a large
  instrument count, is a red flag rather than a clean bill of health.

All of this is also available through the Stata-style postestimation
surface, which formats it rather than making you read `model_info`:

```python
sp.estat(res, "abond")      # AR(1) / AR(2)
sp.estat(res, "sargan")     # Sargan and Hansen J side by side
sp.estat(res, "difhansen")  # difference-in-Hansen, per instrument subset
sp.estat(res, "all")        # all three
```

---

## 7. Known limits

- **Sargan scale.** StatsPAI follows `xtabond`
  ($\hat\sigma^2 = \hat e^{*\prime}\hat e^{*}/(2(N^{*}-k))$ over transformed
  rows); `xtabond2` divides by $2N^{*}$, so its Sargan sits a factor
  $N^{*}/(N^{*}-k)$ higher. The Hansen J has no such free scale and matches
  both.

Coefficients, standard errors, the Arellano-Bond AR(1) and AR(2)
statistics, Sargan/Hansen and the difference-in-Hansen block match Stata 18
and `xtabond2` to ~1e-11 across all 39 reference specifications, for every
transform, instrument class, step count and variance grouping — **including
panels with interior gaps**.

### A note if you are cross-checking gapped panels in Stata

Write the instrument set on the **level**:

```stata
xtabond2 n L.n, gmm(n, lag(2 .)) noleveleq noconstant robust
```

not on the lagged expression:

```stata
xtabond2 n L.n, gmm(L.n, lag(1 .)) noleveleq noconstant robust
```

On a gap-free panel these are the same moment set. On a panel with holes
they are not, and the second one will *look* like a StatsPAI bug. Stata
materialises `L.n` row by row, so it is missing wherever the preceding row
is absent, and `xtabond2` then lags that already-holed series — the
instrument ends up requiring both period $t-k-1$ **and** period $t-k$ to
exist, where the level form requires only $t-k-1$. `gmm_lags=(a, b)` names
the level form.

This cost StatsPAI a documented "known limitation" that never existed: the
gap was written into the reference fixture, and a mis-specified reference
reads exactly like a broken estimator. Both moment sets are valid; they are
simply different, and only one of them is the one `gmm_lags` asks for.

A warning still fires on any fit with interior gaps, but it is now an
efficiency advisory: first differencing loses two equations per hole and
`orthogonal=True` loses one, so forward orthogonal deviations remain the
better transform on gappy panels.

## References

- Arellano, M. and Bond, S. (1991). Some tests of specification for panel
  data: Monte Carlo evidence and an application to employment equations.
  *Review of Economic Studies* 58(2), 277–297.
- Arellano, M. and Bover, O. (1995). Another look at the instrumental
  variable estimation of error-components models. *Journal of Econometrics*
  68(1), 29–51.
- Anderson, T.W. and Hsiao, C. (1981). Estimation of dynamic models with
  error components. *Journal of the American Statistical Association*
  76(375), 598–606.
- Blundell, R. and Bond, S. (1998). Initial conditions and moment
  restrictions in dynamic panel data models. *Journal of Econometrics*
  87(1), 115–143.
- Hansen, L.P., Heaton, J. and Yaron, A. (1996). Finite-sample properties of
  some alternative GMM estimators. *Journal of Business & Economic
  Statistics* 14(3), 262–280.
- Nickell, S. (1981). Biases in dynamic models with fixed effects.
  *Econometrica* 49(6), 1417–1426.
- Roodman, D. (2009). How to do xtabond2: An introduction to difference and
  system GMM in Stata. *Stata Journal* 9(1), 86–136.
- Windmeijer, F. (2005). A finite sample correction for the variance of
  linear efficient two-step GMM estimators. *Journal of Econometrics*
  126(1), 25–51.
