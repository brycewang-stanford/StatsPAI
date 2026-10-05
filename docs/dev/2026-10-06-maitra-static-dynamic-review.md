# Maitra, *A Practical Guide to Static and Dynamic Econometric Modelling*: what it showed about StatsPAI

2026-10-06. Worktree `.claude/worktrees/maitra-textbook`, branch
`wt/maitra-textbook`.

## What was examined

The book (Springer, <https://link.springer.com/book/10.1007/978-3-031-86862-7>)
teaches regression diagnostics, asset-pricing regressions and time-series
models in Python with `statsmodels` and `scipy`. Its repository,
`github.com/saritmaitra/PracticalGuideEconometrics`, holds four notebooks
covering chapters 1, 2, 5, 6 and 7 (153 numbered listings, about 2,300
lines of code) and no data files. Every series is downloaded when the
notebook runs, from FRED and from Yahoo Finance. The clone sits in
`改进建议-收集整理/25-Maitra-StaticDynamicEconometricModelling/`, which is
gitignored.

Chapters 3 and 4 have no notebook. Chapter 1 is mostly plotting.

| Chapter | Content |
| --- | --- |
| 1 and 2 | OLS on simulated data; Shapiro-Wilk, t test, Bartlett, Breusch-Pagan, Durbin-Watson, RESET, Cook's distance, Chow, CUSUM, BDS, ARCH LM |
| 5 | Market beta, rolling beta, hedge ratios, F tests, quantile regression, Fama-French, a macro factor model with VIF, HC3, Newey-West, influence, RESET, CUSUM, rolling coefficients, BDS, Chow |
| 6 | ADF, two-stage least squares by hand, Granger causality, AR order selection, the geometric lag, ARDL, automatic ARIMA, Ljung-Box and Box-Pierce, rolling forecasts |
| 7 | VAR order selection, Johansen, VAR, impulse responses, variance decomposition, forecasts, Engle-Granger, VECM |

## Method

The book prints its output, but the downloaded series have been revised
since, and Yahoo's adjusted prices change with every dividend. The printed
numbers cannot be reproduced from today's downloads. So each computation
was run twice on the data in hand, once with the library the book uses and
once with the StatsPAI call a user would write first. Where the two
disagreed the difference was traced to a convention or to a defect. R
4.5.2 and Stata 18 were the further references. The simulated data of
chapters 1 and 2 come from a fixed seed, and there the printed numbers
were reproduced directly.

Nineteen FRED series and the Yahoo prices were downloaded on 2026-10-06
into `改进建议-收集整理/25-Maitra-StaticDynamicEconometricModelling/data/`.

```bash
STATSPAI_MAITRA_DIR=改进建议-收集整理/25-Maitra-StaticDynamicEconometricModelling \
  pytest tests/external_parity/test_maitra_static_dynamic.py
pytest tests/reference_parity/test_dynamic_modelling_parity.py
```

The second file needs no download. It runs on a committed synthetic
dataset against frozen R output, Stata output quoted in the file, and
statsmodels run in the test.

## What agreed

Agreement is relative difference.

| Chapter | Computation | Reference | Agreement |
| --- | --- | --- | --- |
| 1 | OLS on the book's simulated data | printed table | every printed digit |
| 2 | Shapiro-Wilk, Breusch-Pagan, White, Durbin-Watson, Breusch-Godfrey, RESET, ARCH LM | `scipy`, `statsmodels` | 1e-12 |
| 2 | Cook's distance and the three flagged rows | `statsmodels` | 1e-12, same rows |
| 2 | One-sample t test, Bartlett's test | `scipy`, printed | printed digits |
| 5 | Market beta, factor model, HC3, Newey-West with 6 lags | `scipy`, `statsmodels` | 1e-13 |
| 5 | F tests written as `a = b = 0` and `x = Intercept = 1` | `statsmodels` | 1e-12 |
| 5 | `np.power(x, 2)` inside a formula | `statsmodels` | 1e-13 |
| 6 | ADF statistic, lag and p-value, given the same largest lag | `statsmodels` | identical |
| 6 | AR order by BIC, AR coefficients and forecasts | `statsmodels` | 1e-12 |
| 6 | ARDL coefficients | `statsmodels` | 1e-9 |
| 6 | Ljung-Box statistics | `statsmodels` | 1e-12 |
| 7 | VAR order by four criteria, coefficients, forecasts, impulse responses, variance decomposition | `statsmodels` | 1e-8 |
| 7 | Johansen trace and maximum-eigenvalue statistics | `statsmodels` | 1e-10 |
| 7 | Engle-Granger statistic at a given lag | `statsmodels` | 1e-14 |
| 7 | VECM loadings, cointegrating vector, log-likelihood | `statsmodels` | 1e-10 |

## What was wrong, and what changed

| # | Finding | Kind | Status |
| --- | --- | --- | --- |
| 1 | `sp.arima` on a differenced series depended on the unit of measurement | correctness, silent | fixed |
| 2 | `sp.arima` reported a log-likelihood too low for a differenced model with MA terms, under either method | correctness, silent | fixed |
| 3 | `sp.arima(auto=True)` never considered a drift | correctness, silent | fixed |
| 4 | `sp.arima(auto=True)` could select a model fitted on the unit circle | correctness, silent | fixed |
| 5 | The search for the ARIMA maximum stopped short on badly scaled series | accuracy | fixed |
| 6 | ADF p-value far above the fitted range of MacKinnon's surface | correctness, silent | found here too; the Hansen pass fixed it first |
| 7 | `sp.johansen(test="maxeig").summary()` labelled its column "Trace stat" | mislabel | fixed |
| 8 | No Chow test for a known date | gap | `sp.chow_test` |
| 9 | No CUSUM test on OLS residuals | gap | `sp.cusum_test(method="ols")` |
| 10 | No BDS test | gap | `sp.bds` |
| 11 | No rolling or recursive regression | gap | `sp.rolling` |
| 12 | No Box-Pierce statistic; no degrees-of-freedom correction for ARMA residuals | gap | `sp.corrgram(boxpierce=, model_df=)` |
| 13 | ARDL could not search the lags of each regressor | gap | `sp.ardl(x_lags="bic")` |
| 14 | ARDL forecast one step only, and not at all with a contemporaneous regressor | gap | `fit.forecast(steps=, exog=)` |
| 15 | No long-run effect from an ARDL | gap | `fit.long_run()` |
| 16 | VECM had no forecasts, impulse responses or variance decomposition | gap | `fit.forecast`, `fit.irf`, `fit.fevd` |
| 17 | Engle-Granger gave critical values and no p-value; the result printed as an object address | gap | `pvalue`, `repr` |
| 18 | `sp.swilk(series)` and `sp.ttest(array, mu=)` refused anything but a DataFrame | usability | accepted |
| 19 | `sp.qreg(quantile=[...])` failed with a comparison `TypeError` | usability | says to use `sp.sqreg` |
| 20 | `result.get_influence()` raised a bare `AttributeError` | usability | names `sp.influence_measures` |

### 1 and 2. ARIMA on a differenced series

The book fits `ARIMA(gdp, order=(1, 1, 1))` to nominal GDP in billions of
dollars and prints an AR coefficient of 1.0000 with a standard error of
0.002 and a variance estimate with a standard error of 4.8e-07. The
numbers looked like a failed fit, and they are.

A state-space ARIMA carries the level of an integrated series as a state.
Before any data that level is unknown, so it is given a diffuse prior.
`statsmodels` approximates the diffuse prior by a normal with variance
1e6. That is diffuse when the innovations have a variance near 1. GDP's
quarterly innovations have a variance of 1.5e5. Against that the prior is
informative, and the likelihood of every later observation is distorted
by an amount that depends on the parameters.

`sp.arima` wrapped the same model, so it had the same defect.

| GDP measured in | `statsmodels` AR(1) of ΔGDP | `sp.arima` before | `sp.arima` now | R, Stata |
| --- | --- | --- | --- | --- |
| trillions | 0.2159 | 0.2159 | 0.2159 | 0.2159 |
| billions | 0.1696 | 0.1696 | 0.2159 | 0.2159 |
| millions | 0.0248 | 0.0248 | 0.2159 | 0.2159 |

The fix is the exact diffuse initialisation of the filter, which has no
tuning constant. The `d + sD` observations that pin down the integration
states are left out of the likelihood, which is then the likelihood of
the differenced series. The log-likelihood now equals R's
`arima(method="ML")` and Stata's `arima` to seven digits, in any unit.

`method='innovations_mle'` estimated the coefficients on the differenced
series and so had them right. But it reported a log-likelihood computed
by the same filter with the same prior. For ARIMA(0, 2, 1) on GDP it was
-408.62 where R gives -400.13. Its AIC was off by 17.

AIC, BIC and AICc are now computed here from the log-likelihood, the
parameter count and the number of observations after differencing.

### 3 and 4. Automatic ARIMA

After differencing once, the search fitted models with no constant only.
A series that grows, which is why it was differenced, then has no
adequate model at `d = 1`. The search compensated with an AR root at one
or with a second difference. R's `auto.arima` treats the drift as a
candidate. So does `sp.arima(auto=True)` now, and the mean of an
undifferenced series likewise, unless `trend=` fixes the choice.

R also discards a candidate whose AR or MA polynomial has a root within
1% of the unit circle. Such a fit sits on the boundary of the parameter
space and its AICc is not comparable. On a stationary series that the
KPSS test differenced by mistake, the model with a drift had its MA root
at -1 and the lowest AICc of all. That rule is now applied.

On thirteen series (seven macroeconomic, six in the fixture) the order,
the differencing, the presence of a constant or drift and the AICc match
`forecast::auto.arima(stepwise=FALSE, approximation=FALSE)`.

### 5. A search that stopped early

The likelihood does not depend on the unit, but the quasi-Newton search
does. With GDP in billions the drift of ARIMA(1, 1, 0) came out as 249.09
where the maximum is at 248.90, and the AR coefficients of
ARIMA(2, 1, 1) were off in the third digit. A series whose differences
have a standard deviation outside 0.1 to 10, or whose mean is more than
ten standard deviations from zero, is now searched on a standardised
copy. The estimates are mapped back and the reported fit is evaluated on
the series as given. A series already of order one is fitted as before,
so the Track A ARIMA rows are unchanged to the last bit.

On log GDP growth, where the scale is 0.02, the constant of ARIMA(0, 0, 0)
is now the sample mean to five digits. `statsmodels` stops 4e-4 short.

### 6. The ADF p-value of an explosive series

MacKinnon's surface for the regression with a trend was fitted up to a
statistic of 0.70. Above that the cubic turns down. A statistic of 2.85,
which an explosive series gives, had a p-value of 0.41, and 3.5 had 0.005,
a rejection of the unit root in the wrong direction. Stata prints 1.0000.
This was found here on GDP in levels and, on rebasing, found already
fixed on main by the Hansen pass a few hours earlier. A test with Stata's
numbers is added.

### 8 to 12. Tests the book uses and StatsPAI lacked

`sp.chow_test` is the F test at a date fixed in advance, with a subset of
coefficients, several breaks, or a robust covariance if wanted. It agrees
with `strucchange::sctest(type="Chow")` to 12 digits and its Wald form
with Stata's `estat sbknown`. `sp.structural_break(method="chow")` stays
what it was, the supremum over dates with the sup-F reference, and its
documentation already said the two are different procedures.

`sp.cusum_test(method="ols")` is the Ploberger-Krämer test. It agrees
with `strucchange` to 12 digits, with Stata's `estat sbcusum, ols` to 7
and with `breaks_cusumolsresid(ddof=k)` exactly.

`sp.bds` reproduces `statsmodels.tsa.stattools.bds` exactly, written from
the published formulas.

`sp.rolling` agrees with `RollingOLS` to 1e-13 and adds recursive
estimation, a step and four robust covariances. A window in which the
regressors are collinear returns missing values and a warning that counts
them.

`sp.corrgram(model_df=2, boxpierce=True)` agrees with R's `Box.test` and
`acorr_ljungbox`.

### 13 to 17. ARDL and VECM

The lag search matches `ardl_select_order` in eight configurations. The
dynamic forecast matches `ARDLResults.predict` and `AutoReg.predict` to
1e-12. The long-run effects and their delta-method standard errors match
`UECM.ci_params` and `ci_bse`.

VECM forecasts and impulse responses match `statsmodels` for the five
deterministic cases and two ranks to 1e-8, and `vars::vec2var` on
`urca::ca.jo` to 1e-7. The Engle-Granger p-value is MacKinnon's, the one
`coint` reports.

## What the book gets wrong

These are noted because a user comparing against the book will meet them.

- **Table 27, Chow test.** The statistic is computed as
  `(RSS1 + RSS2) / RSS * (n - 2) / 2`, which is 46.0 on data with no break
  and would be near 49 for any stable relation with 100 observations. The
  Chow statistic is 3.10. Table 94 has the right formula.
- **Table 41, Scholes-Williams beta.** The lagged and the lead regression
  are the same regression, and the three betas are averaged where the
  estimator divides their sum by one plus twice the market's
  autocorrelation.
- **Table 53, excess returns.** A monthly log return is reduced by the
  annual Treasury yield in percent. The regression that follows has an
  R-squared of 0.998 because both sides are dominated by the yield.
- **Table 70, VIF.** `variance_inflation_factor` is called on the
  regressors without a constant, which gives uncentred factors.
- **Table 91.** The "CUSUMSQ" line calls the CUSUM function again with
  `ddof=0`, its default, and prints the same number.
- **Table 108.** The lag order is selected on prices and the model is
  fitted to returns.
- **Table 127.** The ARIMA(1, 1, 1) with an AR coefficient of 1.0000
  discussed above.
- **Table 134.** The Johansen test is applied to differenced series.
- **Table 152.** The VECM is fitted with rank 3 for 3 series, which is a
  VAR in levels.

## Differences that are conventions

| Where | `statsmodels` | StatsPAI | To reproduce |
| --- | --- | --- | --- |
| Largest lag in the ADF search | `ceil(12 (T/100)^0.25)` | the floor | `max_lags=` |
| AutoReg / ARDL classical standard errors | RSS / n | RSS / (n - k) | multiply by `sqrt((n - k) / n)` |
| OLS-CUSUM scale | RSS / n | RSS / (n - k) | `ddof=k` in statsmodels |
| Granger F | residual variance with divisor `T - m` | Wald / df, as Stata `var, small` | `se_df="r"` |
| VAR standard errors, orthogonalised responses | divisor `T - Kp - 1` | divisor `T`, as Stata | `se_df="r"`, `sigma_df=` |
| Quantile regression | iterated least squares, kernel sandwich | linear programme, `vce(iid)` | `vce="kernel"` is close, not equal |
| Engle-Granger lag | chosen by AIC | fixed rule | `lags=` |
| Johansen critical values | MacKinnon-Haug-Michelis | Osterwald-Lenum, as Stata | the statistics are identical |

Stata 18 confirmed the Granger row. After `var c1 c2 c3, lags(1/2) small`,
`vargranger` prints F = 20.412 where the Wald statistic is 40.825 with 2
degrees of freedom.

## Left open

- **ARDL bounds test.** `fit.long_run()` gives the long-run effects. The
  Pesaran-Shin-Smith test for whether a levels relation exists needs its
  critical-value tables and is not written.
- **Confidence bands for VECM impulse responses.** `fit.irf` returns the
  point responses only.
- **Scholes-Williams and Dimson betas.** One line each from regressions on
  leads and lags. No function was added for a single listing.
- **Seasonal automatic ARIMA.** The search runs over `(p, d, q)` and the
  constant. Seasonal orders are taken as given.
- **`sp.johansen` prints "5% CV" when `alpha=0.01`**, and reports the
  number of rows read as `N` where Stata reports the observations used.
- **The book's chapters 3 and 4** have no notebook and were not examined.

## Evidence

- `tests/reference_parity/test_dynamic_modelling_parity.py`: 83 tests on
  `_fixtures/dynamic_modelling.csv` against
  `_fixtures/dynamic_modelling_R.json` (R 4.5.2 with forecast 9.0.2,
  strucchange 1.5.4, urca 1.3.4, vars 1.6.1), Stata 18 MP values from
  `_fixtures/_generate_dynamic_modelling_Stata.do`, and statsmodels.
- `tests/external_parity/test_maitra_static_dynamic.py`: 14 tests on the
  book's own computations, 5 of which need no download.
