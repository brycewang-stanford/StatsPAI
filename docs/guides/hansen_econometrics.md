# Hansen's *Econometrics*: the book's computations in StatsPAI

Bruce Hansen's *Econometrics* (Princeton University Press, 2022) ships with
the Stata, R and MATLAB programs behind its tables. This page maps the
empirical work of each chapter to a StatsPAI call. The Stata programs were
run in Stata 18 and replayed through `sp.stata`; where a number below is
compared with Stata, that replay or the test
`tests/reference_parity/test_hansen_methods_stata_parity.py` is the
evidence.

The book's data are on the author's page and are not redistributed here.
The code blocks run on simulated data so that they can be copied as they
are.

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(2022)
n = 500
df = pd.DataFrame({
    "educ": rng.integers(8, 21, n).astype(float),
    "exper": rng.uniform(0, 40, n),
    "firm": rng.integers(0, 50, n),
})
df["exper2"] = df.exper**2 / 100
df["lwage"] = (
    0.5 + 0.10 * df.educ + 0.04 * df.exper - 0.07 * df.exper2
    + rng.normal(scale=0.5, size=n)
)
```

## Chapter by chapter

| Chapter | In the book | StatsPAI |
| --- | --- | --- |
| 3, 4 | Least squares, leverage, HC0 to HC3, clustered standard errors | `sp.regress(..., vce='hc2')`, `cluster=`, `sp.estat(fit, 'leverage')` |
| 8 | Constrained least squares, efficient minimum distance | `sp.cnsreg` |
| 9 | Wald tests, functions of coefficients | `sp.test`, `sp.lincom`, `sp.nlcom` |
| 10 | Jackknife and bootstrap | `sp.jackknife`, `sp.bootstrap` |
| 11 | Principal components, factor models | `sp.pca`, `sp.factor` |
| 12 | 2SLS, LIML, overidentification, control function | `sp.iv`, `sp.estat(fit, 'overid')`, `'endogenous'`, `'firststage'` |
| 13 | GMM | `sp.iv(method='gmm')`, `sp.gmm` |
| 14 | Autoregressions, Newey-West | `sp.ardl`, `sp.regress(vce='hac', hac_lags=)` |
| 15 | VAR, impulse responses, structural VAR | `sp.var`, `sp.varsoc`, `sp.svar`, `fit.irf()`, `fit.fevd()` |
| 16 | Unit roots, KPSS, cointegration | `sp.unitroot`, `sp.vec`, `sp.johansen` |
| 17 | Fixed and random effects, Hausman-Taylor, dynamic panels | `sp.panel`, `sp.xthtaylor`, `sp.xtabond`, `sp.xtdpdsys` |
| 18 | Difference in differences | `sp.did`, `sp.panel(method='fe')` |
| 19 to 21 | Kernel regression, series, regression discontinuity | `sp.lpoly`, `sp.rdrobust` |
| 23 | Nonlinear least squares, threshold and kink models | `sp.nls`, `sp.threshold` |
| 24 | Quantile regression | `sp.qreg`, `sp.sqreg` |
| 25, 26 | Binary and multiple choice | `sp.logit`, `sp.probit`, `sp.margins`, `sp.mlogit`, `sp.clogit`, `sp.nlogit`, `sp.mixlogit` |
| 27 | Censoring and selection | `sp.tobit`, `sp.heckman` |
| 28 | Model selection and averaging | `sp.model_average` |
| 29 | Lasso, ridge and their relatives | `sp.rlasso`, `sp.lasso_select` |

Not available yet: multinomial probit (chapter 26).

## Constrained regression (chapter 8)

The book restricts the Mankiw-Romer-Weil growth regression so that three
coefficients sum to zero. The same idea with a wage equation:

```python
free = sp.regress("lwage ~ educ + exper + exper2", data=df, vce="robust")
cls = sp.cnsreg("lwage ~ educ + exper + exper2", df, "exper + exper2 = 0",
                vce="robust")
emd = sp.cnsreg("lwage ~ educ + exper + exper2", df, "exper + exper2 = 0",
                method="emd")
print(cls.params.round(4).to_dict())
print(cls.model_info["constraint_test"])
```

`method='cls'` is Stata's `cnsreg`. `method='emd'` is the efficient minimum
distance estimator, which has the smaller variance under
heteroskedasticity. `constraint_test` is the Wald test of the restriction
at the unrestricted estimate. Here it rejects, as it should: the data were
not generated under the restriction, and both restricted estimators are
then inconsistent for the unrestricted coefficients.

## A function of the coefficients three ways (chapters 9 and 10)

The experience level at which the wage profile peaks is
`-50 * b_exper / b_exper2`. The delta method, the jackknife and the
bootstrap each give a standard error.

```python
peak = "-50 * _b[exper] / _b[exper2]"
delta = sp.nlcom(free, peak)

def statistic(d):
    b = sp.regress("lwage ~ educ + exper + exper2", data=d).params
    return -50 * b["exper"] / b["exper2"]

jack = sp.jackknife(df, statistic)
boot = sp.bootstrap(df, statistic, n_boot=500, seed=1)
print(round(delta["estimate"], 2), round(delta["se"], 2),
      round(jack.se, 2), round(boot.se, 2))
```

The jackknife involves no random number, so it reproduces Stata's
`jackknife` prefix digit for digit. It is consistent for smooth functions
such as this one and not for quantiles. The bootstrap depends on the draws:
the same procedure in two programs gives standard errors that agree up to
simulation error.

## Principal components and factors (chapter 11)

```python
tests = pd.DataFrame(
    rng.normal(size=(n, 1)) * [0.8, 0.7, 0.6, 0.5]
    + rng.normal(size=(n, 4)) * 0.6,
    columns=["word", "sentence", "letter", "spelling"],
)
pc = sp.pca(tests)
print(pc.eigenvalues.round(3))
fa = sp.factor(tests, method="ml", n_factors=1)
print(fa.loadings.round(3))
index = pc.scores(tests)["Comp1"]
```

Eigenvectors are defined up to sign. StatsPAI signs each one so that its
elements sum to a positive number, which is what Stata prints; R may show
the opposite sign. Loadings are unrotated.

## Overidentification after 2SLS (chapter 12)

```python
z = rng.normal(size=(n, 3))
df["z1"], df["z2"], df["z3"] = z.T
v = rng.normal(size=n)
df["school"] = 12 + z @ [0.8, 0.5, 0.3] + v
df["y"] = 1 + 0.1 * df.school + 0.5 * v + rng.normal(size=n)
fit = sp.iv("y ~ exper + (school ~ z1 + z2 + z3)", data=df, robust="robust")
over = sp.estat(fit, "overid", print_results=False)
print(round(over["score"], 3), {k: round(v, 3) for k, v in over["iid_errors"].items()})
```

After a robust fit the headline statistic is Hansen's J at the 2SLS
residuals, which is also the robust score statistic Stata reports.
`iid_errors` holds Sargan's and Basmann's statistics, which Stata shows
with `estat overid, forcenonrobust`.

If an excluded instrument is a linear combination of the others and of the
exogenous regressors, `sp.iv` drops it, warns, and lists it in
`model_info['omitted_instruments']`.

## Time-invariant regressors in a panel (chapter 17)

```python
units, periods = 150, 6
effect = rng.normal(size=units)
pan = pd.DataFrame({"firm": np.repeat(np.arange(units), periods)})
pan["sales"] = rng.normal(size=len(pan))
pan["debt"] = rng.normal(size=len(pan)) + effect[pan.firm]
pan["sector"] = rng.integers(0, 2, units)[pan.firm].astype(float)
pan["listed"] = ((rng.normal(size=units) + effect) > 0)[pan.firm].astype(float)
pan["invest"] = (
    1 + 0.5 * pan.sales - 0.3 * pan.debt + 0.2 * pan.sector + 0.4 * pan.listed
    + effect[pan.firm] + rng.normal(size=len(pan))
)
ht = sp.xthtaylor("invest ~ sales + debt + sector + listed", pan, id="firm",
                  endog=["debt", "listed"])
print(ht.params.round(3).to_dict())
print(ht.model_info["ti_endogenous"], round(ht.model_info["rho"], 3))
```

Fixed effects cannot estimate the coefficients of `sector` and `listed`,
which do not vary within firm. Random effects can, but assumes `debt` and
`listed` are unrelated to the firm effect. `sp.xthtaylor` lets the
regressors named in `endog=` be correlated with it and instruments them
with the within variation of the time-varying regressors and the firm means
of the exogenous ones. It needs at least as many exogenous time-varying
regressors as endogenous time-invariant ones.

## Nonlinear least squares (chapter 23)

```python
df["x"] = np.exp(rng.normal(size=n) / 2)
df["q"] = 2 + 3 * df.x**0.5 + rng.normal(scale=0.3, size=n)
nl = sp.nls("q ~ {a} + {b} * x^{c}", df, start={"a": 1, "b": 1, "c": 1},
            vce="robust")
print(nl.params.round(3).to_dict())
print(sp.test(nl, "c = 0.5")["pvalue"] > 0.01)
```

The formula is Stata's `nl` syntax with the parameters in braces. A Python
function `f(params, data)` works too. The sum of squares of a nonlinear
model can have several local minima, so try more than one set of starting
values and keep the fit with the smallest `diagnostics['Residual SS']`.
When a parameter is a threshold at which the function jumps, the estimate
is usable and its normal standard error is not; use `sp.threshold`.

## Threshold and kink models (chapter 23)

```python
df["tip"] = rng.uniform(0, 1, n)
df["move"] = (
    1 + 0.5 * df.x + (df.tip > 0.4) * (1.0 + 0.8 * df.x)
    + rng.normal(scale=0.5, size=n)
)
thr = sp.threshold("move ~ x", df, "tip", n_boot=200, seed=1)
print(round(thr.model_info["threshold"], 3), thr.model_info["threshold_ci"])
print(thr.model_info["regimes"].round(3))
print(thr.model_info["linearity_test"]["pvalue"])
```

The threshold is estimated by least squares over the sample values of
`tip`. Its interval is the set of values a likelihood-ratio test does not
reject, because the estimate is not normal. The standard errors of the
other coefficients take the threshold as known. Whether there is a
threshold at all is a separate question, and the usual F test does not
answer it: under the null the threshold is not identified. `n_boot=` runs
the bootstrap version.

`kink=True` fits a continuous function whose slope changes at an unknown
point. There the point is estimated at the usual rate and gets a standard
error:

```python
df["growth"] = (
    2 - 1.0 * np.minimum(df.tip - 0.4, 0) + 2.0 * np.maximum(df.tip - 0.4, 0)
    + rng.normal(scale=0.3, size=n)
)
kink = sp.threshold("growth ~ 1", df, "tip", kink=True)
print(kink.params.round(3).to_dict(), round(kink.std_errors["threshold"], 3))
```

This is the model the book fits with `nl` to the Reinhart-Rogoff data.
`sp.threshold` searches a grid first, so it does not depend on starting
values.

## Model selection and averaging (chapter 28)

```python
candidates = [
    "lwage ~ educ + exper",
    "lwage ~ educ + exper + exper2",
    "lwage ~ educ + exper + exper2 + I(exper**3)",
]
avg = sp.model_average(candidates, df)          # jackknife weights
print(avg.table[["k", "aic", "bic", "cv", "w_mma", "w_jma"]].round(3))
print(avg.selected)
return_at_10 = avg.average(
    lambda fit: fit.predict(pd.DataFrame({"educ": [12.0], "exper": [11.0],
                                          "exper2": [1.21]}))[0]
    - fit.predict(pd.DataFrame({"educ": [12.0], "exper": [10.0],
                                "exper2": [1.0]}))[0]
)
```

`table` holds every criterion and all four sets of weights; `method=`
chooses which one defines the averaged coefficients and predictions.
Jackknife weights are the default because they stay optimal under
heteroskedasticity. No standard errors are reported: an estimator that was
selected or averaged is not normal around the coefficient of one model.

## Running the book's do-files

`sp.stata` runs the chapters' Stata code on a DataFrame.

```python
out = sp.stata(
    """
    constraint define 1 exper + exper2 = 0
    cnsreg lwage educ exper exper2, constraints(1) r
    """,
    data=df,
)
assert np.allclose(out.params, cls.params)
```

What it still declines in these chapters is listed in
`docs/dev/2026-10-05-hansen-econometrics-review.md`.
