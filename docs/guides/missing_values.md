# Missing values, infinite values and degenerate input

This page says what the core estimators do with incomplete or degenerate
data, and how to ask for a hard failure instead of row deletion.

## Missing values: `missing="drop"` or `missing="raise"`

By default an estimator deletes every row that has a missing value (`NaN`,
`None`, `pd.NA`, `NaT`) in a column the call uses. That is listwise
deletion, as in Stata and R. The estimator says how many rows went, by
column, in an `AssumptionWarning`, and the fit reports the smaller N.

Replication work often needs the opposite. If the data should be complete,
a missing value is a bug upstream, and a fit on fewer rows hides it. Pass
`missing="raise"` and the call is refused instead:

```python
import statspai as sp

fit = sp.regress("wage ~ educ + exper", df, cluster="firm", missing="raise")
# DataInsufficient: regress: missing='raise' and the data hold 3 missing
# value(s) in the columns this call uses, in 3 row(s) -- by column:
# 'exper' (2), 'firm' (1).
```

The error is a `DataInsufficient`. Its `diagnostics` carry
`missing_by_column`, `n_missing` and `n_rows_missing`, so an agent can
repair the call without parsing the message.

The columns counted are the ones the call uses. They are the outcome, the
regressors and covariates (formula terms included), the weights column, the
cluster column (also when it is spelled `vce="cluster firm"` or
`vcov={"CRV1": "firm"}`), and any other column named in the call. A missing
value in a column the call does not use is ignored.

Two kinds of missing value carry meaning and are not counted:

- a missing cohort in a staggered DiD (`g` / `first_treat` / `treat_time`,
  or `treat` in `sp.did`) means the unit is never treated;
- a missing outcome in `sp.heckman` marks an unselected row.

`missing=` is available on `regress`, `logit`, `probit`, `poisson`, `glm`,
`ivreg`, `iv`, `feols`, `fepois`, `feglm`, `qreg`, `tobit`, `heckman`,
`did`, `callaway_santanna`, `sun_abraham`, `did_imputation`,
`event_study`, `rdrobust`, `synth`, `dml`, `panel`, `match`, `ipw` and
`aipw`. With complete data both settings give the same numbers.

## Infinite values

`inf` is not a missing value. It almost always comes from a transformation
that failed, such as `log(0)` or a division by zero. A used column holding
`inf` is refused with a `DataInsufficient` that names each column and its
count (`diagnostics["infinite_by_column"]`), whatever `missing=` says. To
have those rows dropped like missing ones, recode them first with
`data.replace([np.inf, -np.inf], np.nan)`. The one exception is the
cohort column of a staggered DiD, where `g = inf` means never treated.

## Other input the core estimators refuse

Before any fitting, the same shared check refuses these inputs with a
typed error and a recovery hint.

| Input | Error |
| --- | --- |
| A frame with no rows | `DataInsufficient`, `diagnostics["n_rows"] == 0` |
| A used column with no non-missing value | `DataInsufficient`, `diagnostics["all_missing_columns"]` |
| A text outcome where numbers are needed | `MethodIncompatibility`, `diagnostics["column"]` |
| A cluster variable with one value | `DataInsufficient`, `diagnostics["n_clusters"] == 1` |

A number column stored as `object` (for example `Decimal` values from a SQL
`NUMERIC` column) is read as numbers, not refused.

## Constant and collinear regressors

A regressor that never varies, or one that is a linear combination of the
others, is omitted with a note that names it, as Stata does. The
remaining coefficients are then the ones of the fit without that
regressor. `sp.regress(collinear="raise")` refuses instead. Estimators that
do not report regressor coefficients (the DiD family, `sp.dml`,
`sp.match`) give the fit without the redundant covariate. Two cases are
refused because nothing is left to estimate. One is `sp.panel(method="fe")`
when every regressor is absorbed by the fixed effects. The other is a
`sp.rdrobust` covariate that is collinear with the running-variable
polynomial.

## Where this is tested

`tests/test_edge_case_contract.py` runs each of these inputs over the 25
estimators above. It also checks that noise of relative size 1e-13 on the
outcome leaves every estimate unchanged and that exact fits recover the
coefficients that generated the data.
