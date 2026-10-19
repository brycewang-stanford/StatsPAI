# Experiments that look early: group sequential designs, always-valid tests, CUPED

Three questions come up whenever an experiment is analysed before it ends.

| Question | Tool | Evidence |
| --- | --- | --- |
| I will analyse at a few planned looks. Where are the stopping lines, and how much larger must the trial be? | `sp.group_sequential` | Stata 18 `gsbounds`, 117 designs, 3e-7 |
| I want to watch the dashboard after every user and stop whenever it is significant | `sp.msprt` | closed form = defining integral; Monte Carlo of the guarantee |
| Can pre-experiment data shrink the standard error? | `sp.cuped` | formula identities; known-truth Monte Carlo |
| What power does *my* estimator have on *my* design? | `sp.power("simulation", ...)` | agrees with exact t power within Monte Carlo error |

## 1. Group sequential designs

```python
import statspai as sp

d = sp.group_sequential(n_looks=3, efficacy="errobfleming",
                        futility="errpocock", side="upper")
print(d.summary())
d.boundaries[["info_frac", "efficacy_upper", "futility_lower"]]
d.info_ratio          # 1.2779: the trial needs 28% more information
d.decision([1.1, 2.3]) # continue at look 1, stop for efficacy at look 2
```

The boundary families are those of Stata's `gsbounds` (classical
O'Brien-Fleming, Pocock and Wang-Tsiatis; Lan-DeMets error spending of
O'Brien-Fleming and Pocock type, Kim-DeMets, Hwang-Shih-DeCani). Futility bounds
are nonbinding by default: the efficacy bounds ignore them, so overruling a
futility stop does not inflate the type I error. `binding=True` lowers the
efficacy bounds and requires that every futility stop be honoured.

For sample sizes, compute the fixed-design size and pass it on (this is what
Stata's `gsdesign` does in one command):

```python
n_fix = sp.power_ttest(delta=10, sd=35, power=0.8).params["n_exact"]
sp.group_sequential(n_looks=3, n_fixed=n_fix).n_max   # 393.25
```

With error-spending bounds the interim schedule need not be fixed in advance:
recompute the design at the information actually observed, with the planned
maximum as the last value, e.g. `information=[0.27, 0.61, 1]`.

## 2. Always-valid inference (mixture SPRT)

A fixed-horizon test checked after every observation rejects a true null far
more often than its nominal level. `sp.msprt` reports a p-value and a
confidence sequence that stay valid at every look:

```python
r = sp.msprt(df, "conversion", "variant", tau=0.02)
r.path.tail()       # estimate, se, likelihood ratio, p-value, CS at each look
r.stop_look         # first look with p <= alpha
```

`tau` is the standard deviation of the effect sizes the test should be most
sensitive to, in outcome units. It changes power, never validity. Pass `sigma=`
when the outcome standard deviation is known (then the test is exact for
Gaussian data); otherwise running sample variances are plugged in, which is
valid asymptotically.

## 3. CUPED

```python
r = sp.cuped(df, "revenue", "variant", ["revenue_pre"])
r.model_info["variance_reduction"]   # share of the variance removed
sp.cuped(df, "revenue", "variant", ["revenue_pre", "visits_pre"], method="cupac")
sp.cuped(df, "revenue", "variant", ["revenue_pre"], method="lin")  # = sp.lm_lin
```

CUPED subtracts the pooled linear projection on pre-experiment covariates and
compares arms; with one covariate correlated `rho` with the outcome the
variance falls by `rho^2`. With unequal arms and effects that vary with the
covariate, `method="lin"` (separate slopes) is never less precise.

## 4. Simulation power for any estimator

```python
import numpy as np, pandas as pd

def dgp(n, rng, effect_size=0.0):
    d = rng.permutation(np.repeat([0, 1], n // 2))
    x = rng.normal(size=n)
    return pd.DataFrame({"d": d, "x": x, "y": effect_size * d + x + rng.normal(size=n)})

r = sp.power("simulation", n=[100, 200, 400], effect_size=0.3, dgp=dgp,
             estimator=lambda df: sp.lm_lin(df, "y", "d", ["x"]),
             n_sims=500, seed=1)
r.params["table"]   # power, bias, RMSE, coverage, each with its Monte Carlo SE
```

The estimator may return any StatsPAI result with `.estimate` and `.se`, a
regression result (pass `term=`), or a tuple `(estimate, se)`.

## References

- `lan1983discrete`, `obrien1979multiple`, `pocock1977group`,
  `wang1987approximately`, `kim1987design`, `hwang1990group` (boundaries)
- `johari2022always` (mixture SPRT)
- `deng2013improving` (CUPED), `lin2013agnostic` (interacted adjustment)
