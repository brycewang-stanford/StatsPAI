# Counterfactual estimators for panel data: `sp.fect`

`sp.fect` is a native port of the estimation core of Liu, Wang and Xu's
**fect** (R; Stata port `fect_stata`): impute the untreated potential
outcome of every treated unit-period from a model fitted on the
*untreated* cells only, and average `Y - Y(0)` over the treated cells.
It handles staggered adoption, many treated units, unbalanced panels and
treatment reversals, and offers three outcome models.

```python
import statspai as sp

fe  = sp.fect(df, y="y", treat="d", unit="id", time="t")                     # two-way FE (imputation)
ife = sp.fect(df, y="y", treat="d", unit="id", time="t", method="ife", r=2)  # interactive FE, 2 factors
mc  = sp.fect(df, y="y", treat="d", unit="id", time="t", method="mc", lam=0.01)  # matrix completion

ife.estimate                 # ATT over treated observations
ife.detail                   # by relative period: fect_time, relative_time, att, count
ife.model_info["pre_treatment_rmse"]   # fit on the untreated periods of treated units
ife_se = sp.fect(df, y="y", treat="d", unit="id", time="t", method="ife", r=2,
                 vce="bootstrap", n_boot=200, seed=0)   # unit bootstrap SEs
```

`fect_time` follows fect's coding (0 = last untreated period, 1 = first
treated period); `relative_time = fect_time - 1` is the StatsPAI
convention shared with `sp.callaway_santanna` and `sp.sun_abraham`.

## Choosing the outcome model

| Model | When | Tuning |
|---|---|---|
| `fe` | Parallel trends in the two-way sense are credible; equals `sp.did_imputation` on a staggered panel without reversals. | none (`min_t0 = 1`) |
| `ife` | Units respond differently to common shocks (a factor structure); the pre-treatment ATT path under `fe` is not flat. | `r` factors (`min_t0 = 5`) |
| `mc` | Many treated cells, low-rank counterfactual with a soft penalty rather than a fixed rank. | `lam` on fect's raw scale; `model_info["lambda_norm"]` reports it relative to the largest singular value, and a value above 1 collapses to `fe` |

Cross-validation of `r` and `lam` is not ported; compare
`pre_treatment_rmse` across candidates and read the pre-period ATT path.

## Diagnostic tests

fect's three out-of-sample checks are options of `sp.fect`; each needs
resampling standard errors (`vce='jackknife'` is deterministic) and runs
one at a time:

```python
import statspai as sp

kw = dict(y="y", treat="d", unit="id", time="t", vce="jackknife")
pl = sp.fect(df, placebo_period=(-2, 0), **kw)   # last three pre-periods
pl.model_info["placebo_test"]     # estimate, se, pvalue, equiv_pvalue
co = sp.fect(df, carryover_period=(1, 2), **kw)  # treatments that switch off
lo = sp.fect(df, loo=True, **kw)
lo.model_info["loo"]["table"]     # out-of-sample pre-period effects
lo.model_info["pretrend_test"]    # F test, equivalence F, TOST
```

Periods use fect's coding: `0` is the last untreated period, `1` the
first treated period (or, for the carryover test, the first period after
the treatment switches off). A small placebo or carryover p-value, or a
large TOST equivalence p-value, is evidence against the identifying
assumptions. The equivalence bound defaults to fect's
`0.36 * sqrt(sigma2)`; set `tost_threshold=` to a bound that is
substantively negligible in your units.

With the jackknife, fect's own F statistic is built from the plain
covariance of the leave-one-unit-out replicates, which understates the
covariance of the estimates by `(n - 1)^2 / n`; StatsPAI uses the
jackknife covariance, so its F is that factor smaller than fect's. Every
other number matches fect 2.4.1.

## Conventions and parity

The port runs fect's EM map step for step (fixest two-way initial fit,
E-step fill of the treated cells, two-way demeaning, `panel_factor` SVD
with the `sqrt(T)`/`sqrt(N)` normalisation or the soft-threshold on
`E/(T*N)`, relative convergence on the fitted surface and on the
interactive component). Track A module 86 pins fe / ife / mc on one
staggered two-factor panel against R `fect` at `1e-10` (same iteration
count on both sides) and against the authors' Stata port at `1e-9` for
fe / mc and `1.5e-7` for ife, where the Stata port's own stopping rule
sets the floor.

Reference: Liu, Wang and Xu (2024), *American Journal of Political
Science* 68(1), 160--176, `liu2024practical`.
