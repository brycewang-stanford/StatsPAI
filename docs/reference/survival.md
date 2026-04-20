# Survival Analysis

`statspai.survival` — Kaplan-Meier, Cox PH with time-varying
covariates, accelerated failure time (AFT), frailty, and competing
risks.

## Non-parametric

```python
# Kaplan-Meier
km = sp.kaplan_meier(df, time='t', event='d', group='arm')
km.plot(ci=True, at_risk_table=True)
km.median_survival

# Nelson-Aalen cumulative hazard
sp.nelson_aalen(df, time='t', event='d')

# Log-rank and stratified log-rank
sp.log_rank_test(df, time='t', event='d', group='arm')
sp.log_rank_test(..., strata='centre')
```

## Cox proportional hazards

```python
r = sp.cox(
    df, time='t', event='d',
    x=['age','sex','treatment'],
    strata='centre',
    ties='efron',                    # or 'breslow' | 'exact'
    vce='robust',                    # or 'cluster'
    cluster='patient_id',
)
r.hazard_ratios()                    # HR + 95% CI
r.proportional_hazards_test()        # Schoenfeld residuals test
r.plot(kind='survival')              # adjusted survival curves
r.predict_survival(new_df, times=[30, 60, 90])
```

### Time-varying covariates

```python
sp.cox(df, time_start='t0', time_stop='t1', event='d', x=[...])
```

## Accelerated Failure Time

```python
sp.aft(df, time='t', event='d', x=[...],
       dist='weibull')              # 'weibull' | 'exponential' | 'lognormal' | 'loglogistic' | 'gamma'
```

## Frailty models

```python
sp.cox_frailty(df, time='t', event='d', x=[...],
               cluster='family_id',
               frailty_dist='gamma')     # shared gamma frailty
sp.aft_frailty(df, ..., frailty_dist='log_normal')
```

## Competing risks

```python
# Fine-Gray subdistribution hazard
sp.competing_risks(df, time='t', event='d_type',
                   x=[...],
                   event_of_interest=1,
                   method='fine_gray')

# Cause-specific hazard
sp.competing_risks(..., method='cause_specific')
```

## Validation and diagnostics

Every result exposes:

```python
r.summary(); r.to_latex(); r.cite()
r.schoenfeld()                       # PH-assumption residuals
r.martingale()
r.plot(kind='log_minus_log')         # proportional-hazards check
```
