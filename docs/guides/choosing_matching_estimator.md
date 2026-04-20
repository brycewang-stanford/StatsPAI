# Choosing a matching / weighting estimator

When your design relies on selection-on-observables (CIA / unconfoundedness)
and you have a binary treatment, you have 7+ estimators in StatsPAI.
Here's how to choose.

## 0. TL;DR flowchart

```
Is your covariate set high-dimensional (p > 20)?
  YES -> Double ML (sp.dml), meta-learners (sp.S_Learner, etc.)
  NO  -> continue

Is your target the ATT (effect on the treated)?
  YES -> sp.ebalance (entropy balancing) or sp.match(estimand='ATT')
  NO  -> continue

Is your target the ATE (population average)?
  YES -> sp.cbps(estimand='ATE') or sp.aipw
  NO  -> continue

Do you need OVERLAP-weighted effect (avoiding extrapolation)?
  YES -> sp.overlap_weights (ATO)
  NO  -> rethink — what estimand do you actually want?
```

## 1. Entropy balancing (ebal) — the "just works" default for ATT

Hainmueller (2012). Exact covariate balance by reweighting, no
propensity-score modelling needed.

```python
r = sp.ebalance(df, y='y', treat='d',
                covariates=['X1', 'X2', 'X3'],
                moments=1)  # balance means; moments=2 adds variances
```

**Pros:** no PSM model specification; exact balance by construction;
no King-Nielsen issue.
**Cons:** targets ATT only; can be sensitive to extreme weights.

## 2. Nearest-neighbor matching

Beware: King & Nielsen (2019) show that PSM-based nearest-neighbor
matching can **increase** imbalance. Prefer Mahalanobis or coarsened
exact matching (CEM):

```python
r = sp.match(df, y='y', treat='d', covariates=[...],
             distance='mahalanobis',  # NOT 'propensity'
             method='nearest', n_matches=3)
```

## 3. Covariate Balancing Propensity Score (CBPS)

Imai-Ratkovic (2014). Fits the propensity score to balance covariates
directly, not to maximise likelihood.

```python
r = sp.cbps(df, y='y', treat='d', covariates=[...],
            estimand='ATE',  # or 'ATT'
            variant='over')   # 'over' (overidentified) is preferred
```

More robust to PS misspecification than IPW.

## 4. Overlap weights (ATO)

Li-Morgan-Zaslavsky (2018). Weights each unit by its propensity of
receiving the "other" treatment, yielding effects on the **overlap
population** — the subpopulation where both treatments are plausible.

```python
r = sp.overlap_weights(df, y='y', treat='d', covariates=[...],
                       estimand='ATO')
```

Avoids extreme weights from near-zero / near-one propensities.

## 5. Doubly-robust estimators

AIPW combines an outcome model and a propensity-score model — correct
if **either** is right.

```python
r = sp.aipw(df, y='y', treat='d', covariates=[...])
```

For high-dimensional covariates, use Double ML (Chernozhukov et al. 2018):

```python
r = sp.dml(df, y='y', treat='d', covariates=[...],
           ml_model='lasso',         # or 'rf', 'xgb'
           cross_fitting_folds=5)
```

DML is the state of the art for observational ATE with many controls.

## 6. Meta-learners (for heterogeneous effects)

If you want not just the ATE but a CATE function τ(X):

```python
from statspai.metalearners import S_Learner, T_Learner, X_Learner, DR_Learner

dr = DR_Learner(outcome_model='rf', ps_model='lr')
dr.fit(df[cov_cols], df['d'], df['y'])
cate = dr.predict(df_new[cov_cols])
```

See the [meta-learner guide](../reference/causal.md) for
diagnostics (CATE calibration, policy value).

## 7. Common mistakes

| Mistake                                          | Fix                                           |
|--------------------------------------------------|-----------------------------------------------|
| Including post-treatment variables in covariates | Drop them — never condition on consequences   |
| Including colliders as covariates               | Use a DAG (`sp.DAG`) to check adjustment sets |
| Reporting results without checking overlap      | Always plot PS distributions (`sp.psplot`)    |
| Reporting ATE when you computed ATT              | Check `estimand` in the call / result         |
| Using PSM nearest-neighbor (King-Nielsen 2019)   | Use `distance='mahalanobis'` or `method='cem'` |
| Not trimming extreme weights                    | Use `trim=0.01` or overlap weights            |

## 8. Mandatory diagnostics

```python
r = sp.ebalance(df, y='y', treat='d', covariates=[...])

# 1. Balance before/after
sp.love_plot(r)      # SMDs before and after weighting
sp.ps_balance(r)     # formal balance statistics

# 2. Overlap / common support
sp.overlap_plot(r)
sp.trimming(r, threshold=0.01)

# 3. Sensitivity to unobserved confounding
sp.sensemakr(r, benchmark_covariates=['X1'])  # Cinelli-Hazlett
sp.oster_bounds(r)                             # Oster 2019
sp.evalue(r)                                   # VanderWeele-Ding E-value
```

## 9. Reading the output

```python
r.estimate           # Point estimate (ATT / ATE / ATO)
r.se                 # Bootstrap or analytical SE
r.ci                 # CI
r.tidy()             # Main row + per-unit weights if detail available
r.glance()           # method, nobs, estimand, ESS (effective sample size)
r.detail             # If present: balance table with SMDs
```

## 10. Estimand cheat sheet

| Estimand | What it is                          | Recommended estimator      |
|----------|-------------------------------------|----------------------------|
| ATT      | Average effect on the treated       | `ebalance`, `match(ATT)`   |
| ATE      | Average effect on the population    | `cbps(ATE)`, `aipw`, `dml` |
| ATO      | Effect on the overlap population    | `overlap_weights`          |
| ATC      | Average effect on the controls      | `match(estimand='ATC')`    |
| CATE(x)  | Conditional on covariates X=x       | Meta-learners, causal forest |
| LATE     | Effect on compliers                 | IV (not matching)          |
