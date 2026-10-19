# Predictions, comparisons and slopes after a model

`sp.predictions`, `sp.comparisons`, `sp.slopes` and their averaged
versions, plus `sp.hypotheses`, use the vocabulary of R's
`marginaleffects` package. A user (or an agent) who knows that package
gets the same quantities, under the same names and the same defaults.
`sp.margins` stays as the Stata-style entry point. Both are built on one
design rebuilt from the fitted formula, so interactions and terms such as
`I(x**2)` are differentiated through.

They work after `sp.regress`, `sp.logit`, `sp.probit`, `sp.cloglog`,
`sp.poisson`, `sp.nbreg` and `sp.glm`.

```python
import statspai as sp

fit = sp.logit("y ~ treat * x + z + C(region)", data=df)

sp.avg_predictions(fit, by="region")       # average Pr(y = 1) by region
sp.avg_comparisons(fit, "treat")           # model-implied ATE (g-formula)
sp.avg_slopes(fit)                         # average marginal effects
sp.slopes(fit, "x", slope="eyex")          # unit-level elasticities
sp.avg_comparisons(fit, {"x": "iqr"})      # third against first quartile
```

## What each function returns

Each call returns a DataFrame with `estimate`, `std_error`, `statistic`,
`p_value`, `conf_low` and `conf_high`. Unit-level calls keep the data's
columns, one row per row. Averaged calls keep only `term`, `contrast` and
any `by=` columns.

| Function | Quantity |
| --- | --- |
| `predictions` / `avg_predictions` | Fitted value on the response (or `type="link"`) scale |
| `comparisons` / `avg_comparisons` | Prediction at a high value minus (or over) a low value |
| `slopes` / `avg_slopes` | Derivative of the prediction; `eyex`, `eydx`, `dyex` elasticities |
| `hypotheses` | Any smooth function of coefficients or of a table's rows |

## Which change is contrasted

* A factor (`C(g)`, strings, categories): every level against the first.
* A numeric 0/1 variable: 1 against 0, in `slopes` as well.
* Another numeric variable: an increase of 1 from each row's own value.
  `variables={"x": 5}` changes the step. `"sd"` and `"2sd"` contrast
  mean + sd/2 against mean - sd/2 (and +/- sd) for every row, which is
  what marginaleffects computes although its label reads `x + sd/2`.
  `"iqr"` is the third against the first quartile, `"minmax"` the
  maximum against the minimum, and `(low, high)` two fixed values.

`comparison="ratio"` and `"lnratio"` are per row. Once averaged (the
`avg_` functions or `by=`), they become the ratio of the average
predictions (`"ratioavg"`, `"lnratioavg"`), as in marginaleffects.

## Defaults that differ from `sp.margins`

Averages are unweighted and inference uses the normal distribution,
which are marginaleffects' defaults. `sp.margins` follows Stata, using the
fit's weights and the t distribution after `regress`. Pass `wts=True` and
`df="model"` to get Stata's numbers. In that case `sp.avg_slopes` equals
`sp.margins` to rounding, and a test checks it.

## Testing functions of estimates

```python
sp.hypotheses(fit, "b2 = b3")                  # coefficients by position
sp.hypotheses(fit, "x = 2 * z")                # or by name

tab = sp.avg_comparisons(fit, "region")
sp.hypotheses(tab, "b1 - b2 = 0")              # rows of a table
```

Each table carries the Jacobian of its estimates with respect to the
coefficients (`tab.attrs["jacobian"]`, read it with `np.asarray`). That is
how a test of two rows uses their joint covariance; a subset of the rows
(`tab.iloc[[0, 2]]`) keeps the link to its own rows. The expression is
parsed with the same restricted grammar as `sp.nlcom` and is never run as
code.

A list of hypotheses gives one row each. `joint=` runs one Wald test of
several restrictions instead, linear or not (Stata `testnl`):

```python
sp.hypotheses(fit, ["x = 0", "z = 0"])                 # two rows
sp.hypotheses(fit, joint=["x", "z"])                    # F on (2, n - k)
sp.hypotheses(fit, ["exp(x) = 2", "x * z = 0.1"], joint=True, joint_test="chisq")
```

With `joint_test="f"` (the default) the statistic is `W / q` on `q` and
the fit's residual degrees of freedom, also after a logit, as
marginaleffects reports it.

## Fixed-effects models

After `sp.feols` / `sp.fepois` / `sp.feglm` with absorbed effects
(`y ~ x | firm + year`) the absorbed effects are held at their estimates
and enter the linear index like an offset, as marginaleffects does on a
fixest model. `fit.fixef()` returns them (normalised as fixest reports
them) and `fit.predict(newdata=...)` adds them; a level the fit never saw
predicts NaN with a warning.

Standard errors follow marginaleffects. A slope or a difference of a
linear model does not depend on the effects and keeps its standard
error. A prediction, or any quantity of a Poisson or logit model, does.
The coefficient covariance does not carry the effects' uncertainty, so
those calls need `vcov=False` (as marginaleffects' `vcov = FALSE`) and
return estimates only. `sp.margins`
reports delta-method standard errors that are conditional on the
estimated effects.

## Evidence

On a shared CSV, `tests/reference_parity/test_marginaleffects_r_parity.py`
compares every function with R `marginaleffects` 0.32.0 across `lm`,
logit, probit and Poisson fits with interactions, factors and squared
terms. Estimates agree to 1e-8 or better. Standard errors agree to 2e-5,
the precision of marginaleffects' numerical Jacobian; ours is analytic.
The R package is GPL-3 and is used only as a black box. Two reference
details matter when you compare by hand:

* R `glm` takes the covariance from the expected information. For a
  probit (a non-canonical link) this differs from `sp.probit`, which like
  Stata uses the observed information (1.7% in standard errors here).
  `sp.glm(..., information="expected")` reproduces R.
* At its default `epsilon = 1e-8`, R `glm` reports a covariance about 1e-5
  away from the inverse information at its own estimates. The fixtures
  are generated with `epsilon = 1e-14`.
