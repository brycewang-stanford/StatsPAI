# Quasi-experiments, one contract: counterfactuals, decisions, and a Bayesian/OLS switch

StatsPAI's quasi-experimental designs share three things so that you — and an
agent — can read any of them the same way:

1. **One counterfactual contract.** Any design that produces an
   observed-vs-counterfactual series exposes it through `sp.counterfactual_data`
   / `sp.counterfactual_plot`.
2. **One decision layer.** Any causal result answers "did it matter?" through
   `sp.effect_summary` (frequentist verdicts / ROPE, or Bayesian directional
   posterior probabilities).
3. **One inference switch.** Where a Bayesian counterpart exists, the same design
   call selects the backend with `engine=`.

This guide ties the pieces together.

## 1. The counterfactual contract

`sp.counterfactual_data(result)` normalises a fitted result into a tidy frame
with `time, observed, counterfactual, point_effect`, plus `cf_lower/cf_upper`
(counterfactual band), `post`, and `cumulative_effect` when available. It works
for causal impact, the synthetic-control family, interrupted time series, and the
Bayesian time-series designs.

```python
import statspai as sp

res = sp.its(df, y="sales", time="week", intervention=30)
cf = sp.counterfactual_data(res)        # tidy observed vs counterfactual
fig = sp.counterfactual_plot(res)       # observed/counterfactual + effect panel
```

`intervention` is the row position of the first treated period. A numeric
`time` column is the clock: the trend and the slope change are per unit of
it. A date or period column only orders and labels the rows, and the clock
is then one unit per row. `seasonality_period` counts observations (12 for
monthly data) whatever the unit of `time`.

The same two calls work on `sp.synth(...)`, `sp.causal_impact(...)`,
`sp.bayes_its(...)` and `sp.bayes_synth(...)` results — the Bayesian ones carry
genuine credible bands from the posterior.

## 2. The decision layer

```python
res = sp.did(df, y="y", treat="d", time="t", id="unit")
print(sp.effect_summary(res, rope=0.5))     # verdict + ROPE comparison, table + prose
```

For a Bayesian result, `effect_summary` reports a directional posterior
probability (`P(effect > 0)`), the HDI, and posterior ROPE mass when available:

```python
bres = sp.bayes_rd(df, y="y", running="x", cutoff=0.0)
print(sp.effect_summary(bres, direction="increase"))
```

## 3. The inference switch

Regression discontinuity runs frequentist (CCT local polynomial) or Bayesian from
one call:

```python
ols   = sp.rdrobust(df, y="y", x="run", c=0.5)                 # engine="ols" (default)
bayes = sp.rdrobust(df, y="y", x="run", c=0.5, engine="bayes") # -> sp.bayes_rd
```

`engine="bayes"` routes to `sp.bayes_rd` with the shared arguments mapped across;
options with no Bayesian analogue (fuzzy, RKD, kernel/bandwidth selection,
clustering, …) raise rather than being silently dropped. For full prior control,
call `sp.bayes_rd` directly.

## 4. Lightweight pre/post designs

When you have a treated and a non-randomised comparison group:

```python
# Covariate-adjusted comparison of group means
sp.ancova(df, outcome="post", group="treated", covariates=["baseline", "age"])

# Pre/post non-equivalent group design (ANCOVA on baseline, or change-score)
sp.negd(df, group="treated", pre="score0", post="score1")               # ANCOVA (default)
sp.negd(df, group="treated", pre="score0", post="score1",
        method="change_score")                                          # gain score (warns on RTM)
```

## 5. Geo experiments (geo-lift)

Measure incremental lift in treated markets against a synthetic counterfactual
built from untreated markets:

```python
res = sp.geolift(df, outcome="sales", geo="dma", time="week",
                 treated_geos=["NYC", "LA"], treatment_time=40)
res.estimate, res.model_info["relative_lift_pct"]
sp.counterfactual_plot(res)             # treated markets vs synthetic counterfactual
```

## 6. Bayesian counterparts

| Frequentist | Bayesian | Estimand |
| --- | --- | --- |
| `sp.rdrobust` | `sp.bayes_rd` / `sp.rdrobust(..., engine="bayes")` | LATE at the cutoff |
| `sp.its` | `sp.bayes_its` | level change |
| `sp.synth` | `sp.bayes_synth` | ATT |
| `sp.did` | `sp.bayes_did` | ATT |

The Bayesian time-series designs (`bayes_its`, `bayes_synth`) feed the same
`counterfactual_plot` with posterior credible bands — the small-sample honesty
that motivates a Bayesian fit when there are few pre-periods or donors.

> Bayesian designs need the optional extra: `pip install "statspai[bayes]"`.
