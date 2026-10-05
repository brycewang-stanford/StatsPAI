# Coming from statsmodels and linearmodels

Most Python econometrics teaching code is written for `statsmodels`,
`linearmodels`, `scikit-learn` and `arch`. This page maps those calls to
StatsPAI and lists the places where the same request returns a different
number, with the reason. It was written by rerunning every example of
Dogan's *Introduction to Econometrics with Python*, a companion to Stock &
Watson, on the book's data.

The formulas carry over as they are. `I(x**2)`, `np.log(x)`, `C(g)`,
`a*b`, `a:b` and `x.shift(1)` mean what they mean in `statsmodels`.

## The map

| You wrote | Write |
| --- | --- |
| `smf.ols(f, df).fit()` | `sp.regress(f, df)` |
| `.fit(cov_type="HC1")` | `sp.regress(f, df, robust="hc1")` |
| `.fit(cov_type="cluster", cov_kwds={"groups": df["g"]})` | `sp.regress(f, df, cluster="g")` |
| `.fit(cov_type="HAC", cov_kwds={"maxlags": 7})` | `sp.regress(f, df, robust="hac", hac_lags=7)` |
| `res.bse`, `res.rsquared`, `res.resid` | `res.std_errors`, `res.r2`, `res.residuals()` |
| `res.f_test("x1 = 0, x2 = 0")` | `sp.test(res, "x1 = 0, x2 = 0")` |
| `res.get_prediction(new).summary_frame()` | `res.predict(new, what="confidence")` |
| `smf.probit(f, df).fit()`, `smf.logit` | `sp.probit(f, df)`, `sp.logit(f, df)` |
| `res.get_margeff(at="overall")` | `sp.margins(res, df, method="ame")` |
| `res.get_margeff(at="mean")` | `sp.margins(res, df, method="mem")` |
| `PanelOLS.from_formula("y ~ x + EntityEffects", df)` | `sp.panel(df, "y ~ x", entity="id", time="t", method="fe")` |
| `... + EntityEffects + TimeEffects` | `sp.panel(..., method="twoway")` |
| `RandomEffects`, `PooledOLS` | `sp.panel(..., method="re")`, `method="pooled"` |
| `IV2SLS.from_formula("y ~ 1 + w + [x ~ z]", df)` | `sp.ivreg("y ~ w + [x ~ z]", df)` |
| `rdrobust(y=Y, x=X, c=c)` | `sp.rdrobust(df, y="y", x="x", c=c)` |
| `ARIMA(y, order=(2, 0, 0)).fit()` | `sp.arima(y, order=(2, 0, 0))` |
| `adfuller(y, regression="ct")` | `sp.unitroot(y, trend="ct")` |
| `VAR(df).fit(2)`, `.irf(10)`, `.test_causality` | `sp.var(df, lags=2)`, `sp.irf`, `sp.granger_causality` |
| `coint_johansen(df, 0, 3)` | `sp.johansen(df, lags=3)` |
| `arch_model(r, vol="Garch", p=1, q=1).fit()` | `sp.garch(r, p=1, q=1, vce="robust")` |
| `RidgeCV`, `LassoCV`, `PCA` + `LinearRegression` | `sp.shrinkage(df, y, x, method="ridge" / "lasso" / "pcr")` |
| `summary_col([r1, r2])`, `Stargazer([r1, r2])` | `sp.regtable(r1, r2)` |

## Where the numbers differ, and why

Coefficients agree to machine precision in every row above. These are the
differences in what is reported around them.

**Confidence intervals and p-values after robust errors.** `statsmodels`
switches to the normal distribution when `cov_type` is not `"nonrobust"`.
StatsPAI keeps the t distribution with the residual degrees of freedom, as
Stata does. With 420 observations the interval is wider by 0.3%.

**`cov_type="HC1"` on probit and logit.** `statsmodels` accepts the
argument and returns the sandwich without the `N / (N - K)` factor, which
is HC0. `sp.probit(..., robust="hc0")` reproduces it to 1e-14.
`robust="hc1"` applies the factor the name promises, and
`robust="robust"` gives Stata's `N / (N - 1)`.

**Clustered errors in a fixed-effects panel.** The usual `linearmodels`
formula spells the intercept, `y ~ 1 + x + EntityEffects`, and counts it
as a regressor in the small-sample factor. `sp.panel` has no intercept to
count. The standard error differs in the fourth digit (0.28923 against
0.28880 on the traffic-fatality panel). Without the `1 +` the two agree,
and with entity and time effects they agree either way. `ssc="stata"`
gives the factor of `xtreg, fe`.

**`cov_type="robust"` in `IV2SLS`.** It is HC0. Use `robust="hc0"` to
match it; `robust="robust"` is HC1, as in Stata's `ivregress ..., small`.

**VAR.** `statsmodels` divides the residual covariance by `T - Kp - 1`.
`sp.var` follows Stata's `var` and divides by `T`, so standard errors,
orthogonalised impulse responses and information criteria differ by that
factor. Coefficients and forecasts are identical. The information
criteria are also on Stata's scale (they include the constant of the
Gaussian likelihood), and the lag they select is the same.

**Johansen critical values.** `statsmodels` uses the MacKinnon, Haug and
Michelis values (15.49 and 3.84 for two series). `sp.johansen` prints the
Osterwald-Lenum values Stata's `vecrank` uses (15.41 and 3.76). The trace
statistics are identical.

**GARCH.** `arch` starts the variance recursion from an exponentially
weighted backcast and StatsPAI from the sample variance, as Stata does.
On 7,055 daily returns the estimates differ in the third or fourth
digit. `arch`
reports Bollerslev-Wooldridge standard errors by default, which is
`vce="robust"` here.

**Cross-validation.** Textbook code often standardises the predictors
once and then splits into folds. `sp.shrinkage` standardises inside each
training fold. Its penalty multiplies the sum of squared residuals, so
ridge is `Ridge(alpha=penalty)` and the lasso is
`Lasso(alpha=penalty / (2 * n))`.

**`fit(vcov="HC1")`.** `statsmodels` ignores keyword arguments it does
not know, so this misspelling of `cov_type` returns classical standard
errors without a word (the book's chapter 12 has five such calls).
`sp.regress(f, df, vcov="HC1")` computes HC1, and an option StatsPAI does
not know raises.

**`sp.acf`.** It is the Ackerberg-Caves-Frazer production-function
estimator. Autocorrelations and the Ljung-Box statistics are `sp.corrgram`.

## A short session

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(0)
n = 500
z, w, u = rng.normal(size=(3, n))
x = 0.8 * z + 0.4 * w + 0.6 * u + rng.normal(size=n)
inc = np.exp(rng.normal(size=n))
df = pd.DataFrame({
    "y": 1 + 0.7 * x - 0.5 * w + 0.3 * inc - 0.02 * inc**2 + u,
    "x": x, "w": w, "z": z, "inc": inc,
})

ols = sp.regress("y ~ x + w + inc + I(inc**2)", df, robust="hc1")
print(sp.test(ols, "inc = 0, I(inc**2) = 0")["pvalue"])   # joint F test
print(ols.predict(df.head(3), what="confidence"))          # with intervals

iv = sp.ivreg("y ~ 1 + w + inc + [x ~ z]", df, robust="hc0")
print(sp.regtable(ols, iv))

cols = ["x", "w", "z", "inc"]
fit = sp.shrinkage(df.iloc[:400], "y", cols, method="lasso")
print(fit.summary())
print(fit.rmspe(df.iloc[400:]))                            # hold-out RMSPE
```
