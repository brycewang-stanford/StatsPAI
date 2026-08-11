# Survival Analysis

`statspai.survival` — Kaplan-Meier, Cox PH with time-varying
covariates, accelerated failure time (AFT), frailty, and competing
risks.

## Non-parametric

```python
# Kaplan-Meier
km = sp.kaplan_meier(df, duration='t', event='d', group='arm')
km.plot(ci=True, at_risk_table=True)
km.median_survival

# Log-rank test
sp.logrank_test(df, duration='t', event='d', group='arm')
```

## Cox proportional hazards

```python
r = sp.cox(
    data=df, duration='t', event='d',
    x=['age','sex','treatment'],
    strata='centre',
    ties='efron',                    # or 'breslow' | 'exact'
    robust='hc0',                    # sandwich SE
    cluster='patient_id',
)
r.hazard_ratios()                    # HR + 95% CI
r.proportional_hazards_test()        # Schoenfeld residuals test
r.plot(kind='survival')              # adjusted survival curves
r.predict_survival(new_df, times=[30, 60, 90])
```

## Accelerated Failure Time

```python
sp.aft('t + d ~ age + sex', df,
       family='weibull')            # 'weibull' | 'exponential' | 'lognormal' | 'loglogistic'
```

## Frailty models

```python
sp.cox_frailty('t + d ~ age + sex', df,
               cluster='family_id')      # shared gamma frailty
```

## Competing risks

```python
# Fine-Gray subdistribution hazard
sp.finegray(df, duration='t', event='d_type', x=['age', 'sex'], cause=1)

# Cumulative incidence by group
sp.cuminc(df, duration='t', event='d_type', group='arm')
```

## Validation and diagnostics

Every result exposes:

```python
r.summary(); r.to_latex(); r.cite()
r.schoenfeld()                       # PH-assumption residuals
r.martingale()
r.plot(kind='log_minus_log')         # proportional-hazards check
```
