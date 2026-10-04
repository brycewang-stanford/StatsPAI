# Choosing a matching / weighting estimator

When your design relies on selection-on-observables (CIA / unconfoundedness)
and you have a binary treatment, you have 7+ estimators in StatsPAI.
Here's how to choose.

## 0. TL;DR flowchart

```
Is your covariate set high-dimensional (p > 20)?
  YES -> Double ML (sp.dml), meta-learners (sp.metalearner, sp.xlearner)
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

`moments` also takes one order per covariate, which is Stata's
`targets(2 2 1)`. An indicator needs its mean only: its square is itself,
and a moment that repeats another one is left out of the problem.

```python
r = sp.ebalance(df, y='y', treat='d',
                covariates=['income', 'share_black', 'democrat'],
                moments=[2, 2, 1],
                dof_adjust=True)   # the scaling of Stata ebalance
```

With `dof_adjust=True` the weights agree with Stata `ebalance,
tolerance(1e-10)` to 1e-10. Stata's default `tolerance(.015)` stops before
the moments are balanced, so a default Stata run agrees to about three
digits. Without `dof_adjust` the raw moments are matched, as R `ebal` and
`WeightIt` do.

**Pros:** no PSM model specification; exact balance by construction;
no King-Nielsen issue.
**Cons:** targets ATT only; can be sensitive to extreme weights.

### Coarsened exact matching

```python
r = sp.match(df, y='y', treat='d',
             covariates=['income', 'share_black', 'democrat'],
             method='cem',
             n_bins={'income': [2.5, 3.5, 5.0],   # cut edges
                     'share_black': 6,            # six equal-width bins
                     'democrat': 2})              # an indicator: two bins
```

Without `n_bins` every covariate is cut by Sturges' rule, as in the `cem`
packages for R and Stata. That is rarely what you want for an indicator,
so name it. Stata's `cem x(#k)` counts cut points: `x(#k)` is
`n_bins=k - 1`, and `x(#2)` does not split `x` at all.

## 2. Nearest-neighbor matching

Beware: King & Nielsen (2019) show that PSM-based nearest-neighbor
matching can **increase** imbalance. Prefer Mahalanobis or coarsened
exact matching (CEM):

```python
r = sp.match(df, y='y', treat='d', covariates=[...],
             distance='mahalanobis',  # NOT 'propensity'
             method='nearest', n_matches=3)
```

If a paper or a Stata do-file matches on the propensity score and you
need its numbers, these are the counterparts. The score is a logit unless
`ps_model='probit'` (Stata `psmatch2` fits a probit by default, `teffects`
a logit).

```python
# teffects psmatch (y) (d x1 x2, probit), atet
r = sp.match(df, y='y', treat='d', covariates=['x1', 'x2'],
             distance='propensity', estimand='ATT', ties='all',
             se_method='abadie_imbens_2016', ps_model='probit')
# teffects psmatch (y) (d x1 x2), ate   -- same call with estimand='ATE'
# psmatch2 d x1 x2, outcome(y)
r = sp.psmatch2(df, treat='d', covariates=['x1', 'x2'], outcome='y',
                ps_model='probit')
# teffects ipw (y) (d x1 x2, probit), atet
r = sp.ipw(df, y='y', treat='d', covariates=['x1', 'x2'],
           estimand='ATT', se_method='sandwich', ps_model='probit')
```

With `ties='all'` every control at the smallest score distance is kept.
Units with the same covariates have the same score, so they tie.

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

DML is designed for observational ATE estimation with many controls.

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
sp.trimming(df, treatment='d', covariates=['X1', 'X2'], method='crump')

# 3. Sensitivity to unobserved confounding
sp.sensemakr(df, y='y', treat='d', controls=['X1', 'X2'],
             benchmark=['X1'])                 # Cinelli-Hazlett
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

<!-- AGENT-BLOCK-START: match -->

## For Agents

**Pre-conditions**
- binary treatment 0/1
- covariates are pre-treatment (temporally prior to D)
- enough control units for each treated unit under the chosen method (k:1 matching)
- covariates numeric; categoricals one-hot or handled by caliper/mahalanobis

**Identifying assumptions**
- Unconfoundedness / CIA: Y(d) ⊥ D | X
- Overlap / common support: treated X-values are in the control X-support
- SUTVA: no interference between matched units
- Covariates are selected before looking at outcomes (no post-treatment conditioning)

**Failure modes → recovery**

| Symptom | Exception | Remedy | Try next |
| --- | --- | --- | --- |
| Covariate imbalance after matching (max \|SMD\| > 0.1) | `statspai.AssumptionViolation` | Re-match with stricter caliper, add interactions, or switch to sp.ebalance (entropy balancing). | `sp.ebalance` |
| Poor propensity score overlap (density plots, treated mass where controls are sparse) | `statspai.AssumptionViolation` | Apply sp.trimming (Crump 2009) or redefine the estimand to the overlap region. | `sp.trimming` |
| Too few matched controls per treated unit | `statspai.DataInsufficient` | Relax caliper, allow with-replacement, or use entropy balancing / overlap weights. | `sp.ebalance` |
| Results highly sensitive to match specification | `statspai.AssumptionWarning` | Report sp.rosenbaum_bounds (sensitivity to unobserved confounding) and compare multiple matching methods. | `sp.rosenbaum_bounds` |

**Alternatives (ranked)**
- `sp.ebalance`
- `sp.cbps`
- `sp.optimal_match`
- `sp.sbw`
- `sp.ipw`

**Typical minimum N**: 200

<!-- AGENT-BLOCK-END -->
