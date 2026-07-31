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

---

## 7. Known limits

- **Gapped panels.** The design (sample, equations, instruments) matches
  Stata exactly, and a just-identified fit reproduces Stata to 2e-15, but
  the one-step weight matrix uses a different gap convention, so
  coefficients differ from Stata by roughly 2–6%. Both estimators remain
  consistent; only finite-sample efficiency differs. A warning fires, and
  `orthogonal=True` is the better answer on such panels anyway.
- **Two-step AR test.** The Windmeijer correction is applied to the
  coefficient SEs but not yet to the AR-test variance, so the two-step AR
  $z$ differs from Stata's by ~0.1% (difference GMM). One-step AR statistics
  are exact.
- **Sargan scale.** StatsPAI follows `xtabond`
  ($\hat\sigma^2 = \hat e^{*\prime}\hat e^{*}/(2(N^{*}-k))$ over transformed
  rows); `xtabond2` divides by $2N^{*}$, so its Sargan sits a factor
  $N^{*}/(N^{*}-k)$ higher. The Hansen J has no such free scale and matches
  both.

---

## References

- Arellano, M. and Bond, S. (1991). Some tests of specification for panel
  data: Monte Carlo evidence and an application to employment equations.
  *Review of Economic Studies* 58(2), 277–297.
- Arellano, M. and Bover, O. (1995). Another look at the instrumental
  variable estimation of error-components models. *Journal of Econometrics*
  68(1), 29–51.
- Blundell, R. and Bond, S. (1998). Initial conditions and moment
  restrictions in dynamic panel data models. *Journal of Econometrics*
  87(1), 115–143.
- Nickell, S. (1981). Biases in dynamic models with fixed effects.
  *Econometrica* 49(6), 1417–1426.
- Roodman, D. (2009). How to do xtabond2: An introduction to difference and
  system GMM in Stata. *Stata Journal* 9(1), 86–136.
- Windmeijer, F. (2005). A finite sample correction for the variance of
  linear efficient two-step GMM estimators. *Journal of Econometrics*
  126(1), 25–51.
