# Neusser, *Time Series Econometrics*: what it showed about StatsPAI

2026-10-06. Worktree `.claude/worktrees/neusser-tse`, branch
`wt/neusser-tse`.

## What was examined

The book (Springer 2016, DOI `10.1007/978-3-319-32862-1`) is a graduate
text on univariate and multivariate time series. Its companion page offers
ten data sets in Excel, EViews and MATLAB format and one piece of code, the
MATLAB Kalman filter of section 17.4. The download sits in
`改进建议-收集整理/28-Neusser-TimeSeriesEconometrics/`, which is gitignored.
The book text itself was not available, so the examples were rebuilt from
the data, the section titles on the companion page and the stored MATLAB
objects.

| Section | Data | Content |
| --- | --- | --- |
| 5.6 | Swiss real GDP, 1980Q1 to 2003Q3 | ACF, PACF, ARMA order selection, forecasts |
| 8.4 | Swiss Market Index, 3,798 trading days | ARCH tests, GARCH, heavy tails, value at risk |
| 11.3 | Swiss GDP growth and consumer sentiment | Prewhitened cross-correlations, leading indicator |
| 14.4 | US output, prices, money, interest rate | VAR order selection and forecasts |
| 15.4.4 | Advertising and sales, 1907 to 1960 | Recursive SVAR, impulse responses with bands |
| 15.4.5 | Blanchard (1989), five US series | AB model |
| 15.5.2 | Blanchard and Quah, growth and unemployment | Long-run restrictions |
| 16.5 | US consumption, investment, output, real rate | Johansen rank, restrictions on the cointegration space |
| 17.4 | Swiss annual GDP and two quarterly indicators | Mixed-frequency state space model, Kalman smoother |

Chapters without a data set were covered as a syllabus: spectral analysis
and filters (6), integrated processes (7), long-run variance (4.4).

The book is from 2016 and its code is older (MATLAB System Identification
objects dated 2003 and 2004). It was used as a list of what a time-series
reader expects, not as a numerical reference. Every number was computed
three ways on the same file: R 4.5.2 (`stats`, `vars` 1.6-1, `urca` 1.3-4,
`sandwich` 3.1-1, `KFAS` 1.6.0, `rugarch`), Stata 18, and `sp`.

Reading the data needed two workarounds. Three `.xlsx` files were written
by EViews with backslashes in the archive member names, which pandas and
openpyxl refuse. The `.xls` files need `xlrd`. Both are handled by
`tests/external_parity/neusser_convert.py`.

## What was wrong

### `sp.garch` with more than one lagged variance

`sp.garch(y, p=2, q=1)` on the SMI returns stopped at a log-likelihood of
-5600.552 with the second lagged-variance coefficient at zero and a
standard error of zero. The nested GARCH(1,1) reaches -5599.372, so the
larger model fitted worse than the model it contains. Stata's `arch`
reaches -5598.548 with coefficients 0.371 and 0.374 on the two lags.

The cause was the search. One simplex from an equal split of the
persistence walked into the boundary, where the objective is a constant
penalty, and stayed. The search now runs a bounded quasi-Newton from
several starting splits and keeps the best, and the old path is kept for
the final polish, so GARCH(1,1) results are unchanged to the last digit.
An estimate that does end on the boundary raises a warning, because its
standard error means nothing there.

Evidence. Stata 18 on a committed simulated series
(`test_garch_extensions_stata_parity.py`): coefficients to 5e-5, OPG and
OIM standard errors to 5e-4, log-likelihood to 1e-8. Our likelihood
evaluated at Stata's estimates equals Stata's to 1.4e-9 relative, which
separates the objective from the optimum.

### `sp.johansen` printed the wrong sample size and level

The summary printed `N: 236` for statistics scaled by 234 observations, and
`5% CV` when `alpha=0.01` was asked for. Both are fixed, and the result
carries `n_used`, `alpha`, `trend` and `var_names`. The statistics were
right and are unchanged (`urca` to 1e-8 in three deterministic cases).

## What was missing

| Topic | Before | Now |
| --- | --- | --- |
| AR terms and heavy tails in GARCH (8.4) | constant mean, normal errors | `sp.garch(ar=, dist='t')`, `forecast_mean`, `value_at_risk` |
| Bands for impulse responses (15.4) | point estimates only | `sp.irf(ci='asymptotic' / 'bootstrap')`, `SVARResult.irf(ci='bootstrap')` |
| Restrictions on the cointegration space (16.5) | rank test only | `sp.johansen_lrtest` |
| Unit root with a break (7) | none | `sp.zivot_andrews` |
| Cross-correlation and prewhitening (11) | none | `sp.xcorr` |
| Long-run variance (4.4) | inside `sp.regress(robust='hac')` only | `sp.lrvar` |
| Spectral density (6) | none | `sp.periodogram`, `sp.cumulative_periodogram_test` |
| Linear filters (6.5) | none | `sp.tsfilter` (HP, Baxter-King, Christiano-Fitzgerald, Butterworth, Hamilton) |
| Beveridge-Nelson decomposition (7.1) | none | `sp.beveridge_nelson` |
| State space models (17) | `sp.dlm` (random-walk regression only) | `sp.kalman_filter`, `sp.statespace` |

Evidence for each, all on committed simulated data with a generator script
next to the reference file:

| Function | Reference | Agreement |
| --- | --- | --- |
| `sp.garch(ar=, dist='t')` | Stata 18 `arch, ar() distribution(t)` | coefficients 5e-5, standard errors 5e-4, log-likelihood 1e-8 |
| `sp.irf(ci='asymptotic')` | Stata 18 `irf create` | simple, orthogonalised and cumulative responses and standard errors to 1e-6, the precision of Stata's single-precision `.irf` file |
| `sp.irf(ci='bootstrap')` | known truth | 90% bands cover between 80% and 97% over 150 samples; spread within a fifth of the delta method at T = 600 |
| `sp.johansen_lrtest` | `urca` `blrtest`, `bh5lrtest`, `alrtest` | statistic 1e-8, degrees of freedom equal, three deterministic cases |
| `sp.zivot_andrews` | `urca::ur.za` | statistic, path over break dates and regression to 1e-7, break date equal |
| `sp.xcorr` | R `ccf` and `ar`, Stata `xcorr` | 1e-10 |
| `sp.lrvar` | `sandwich` `kernHAC`, `NeweyWest`, `lrvar` | 1e-9 over 104 cases |
| `sp.periodogram` | R `spec.pgram`, `spec.ar`, Stata `pergram` | 1e-9 |
| `sp.cumulative_periodogram_test` | Stata `wntestb` | statistic 1e-12, p-value 2e-8 (Stata truncates the series) |
| `sp.tsfilter` | Stata `tsfilter`, statsmodels | 1e-9 |
| `sp.beveridge_nelson` | the long-horizon forecast that defines the trend | 1e-8; no package reference found |
| `sp.kalman_filter`, `sp.statespace` | `KFAS`, statsmodels | filter and smoother 1e-9, estimates 1e-6 |

## The book's examples

Opt-in tests: `tests/external_parity/test_neusser_time_series.py` and
`test_neusser_quarterly_gdp.py`.

- **5.6, ARMA for Swiss GDP growth.** ARMA(1,3) and AR(2) agree with R
  `arima` (log-likelihood -107.0496 and -112.9774). Over the 6 x 6 grid of
  orders `sp.arima` had the higher likelihood in ten cells and the lower
  in one, ARMA(5,2), where R reaches -107.03 and we stopped at -110.13. A
  third start, from zero ARMA coefficients (where R begins), is now tried
  for a final fit; ARMA(5,2) reaches R's value and no other cell moves.
- **8.4, Swiss Market Index.** The GARCH fix above. AR(1)-GARCH(1,1) with
  t errors gives 7.29 degrees of freedom (Stata 7.2867) and a likelihood
  177 points above the normal model. `rugarch` with its hybrid solver
  returned a mean of 3.64 for ARCH(1) on returns whose mean is 0.04; that
  is a failure of the reference, noted so that nobody compares against it.
- **11.3, consumer sentiment as a leading indicator.** The book's residual
  columns are reproduced by an AR(8) on each series (3.5e-13). Their
  cross-correlation is outside the 95% band at one lag only, sentiment one
  quarter ahead of GDP growth (0.323). The raw cross-correlations run from
  0.05 to 0.67 and peak on the other side of zero, which is the point of
  prewhitening.
- **15.4.4, advertising and sales.** Cholesky impact matrix equals R to
  1e-10, and the delta-method standard errors equal Stata's to the six
  digits it prints.
- **15.4.5, Blanchard (1989).** The book's exact restrictions could not be
  recovered without the text. A just-counted pattern that is not locally
  identified made `sp.svar` raise `IdentificationFailure`, while
  `vars::SVAR` returned coefficients of 130 and 217 with NaN standard
  errors and no error. Nothing to change.
- **15.5.2, Blanchard and Quah.** Impact matrix equals `vars::BQ` after the
  documented difference in the covariance divisor (T against T - Kp - 1).
  Bootstrap bands now come from `res.irf(ci='bootstrap')`; the band of the
  restricted long-run response collapses to zero in every replicate, as it
  must.
- **16.5, cointegration.** Trace statistics equal `urca`. With rank 2, the
  hypothesis that consumption and investment enter only relative to output
  is not rejected (LR 2.84, p 0.24); that the consumption-output ratio
  alone is a cointegrating vector is rejected (22.8); weak exogeneity of
  the real rate is rejected (27.2).
- **17.4, quarterly GDP.** `sp.statespace` reproduces the model in a few
  lines. The MATLAB filter agrees with ours to 7e-16. Its likelihood
  differs from ours by exactly `0.5 log(2 pi)` per missing annual value,
  because it scores each one as a standard normal draw of zero (48.7037
  both ways, to 1e-12). All optimisers and seven starting points reach the
  same optimum. **The book's smoother has an error in the variance line.**
  `KalmanSmootherTVP.m` writes `Pt*F*inv(Ptp1)*(...)*inv(Ptp1)*Pt` where
  the recursion needs `Pt*F'*inv(Ptp1)*(...)*inv(Ptp1)*F*Pt`. Smoothed
  means are unaffected. Smoothed variances are too large by up to 0.134,
  so the confidence band in the book's second figure is too wide. The
  corrected recursion agrees with ours, KFAS and statsmodels to 1e-16.

## Conventions confirmed, not changed

- `sp.johansen(lags=)` and `sp.vec(lags=)` count lagged differences; Stata
  and `urca` count lags of the levels. Documented already.
- `sp.svar` scales the residual covariance by T, as Stata does; `vars`
  scales by T - Kp - 1.
- `sp.corrgram` computes partial autocorrelations by regression (Stata);
  R uses Durbin-Levinson. `pac=` switches.
- `sp.arima` reports OPG standard errors (Stata); R reports the Hessian.
- `sp.xcorr(x, y)` at lag h is the correlation of `x[t + h]` with `y[t]`,
  which is R's `ccf(x, y)`. Stata's `xcorr x y` is the mirror image.
- `sp.lrvar` defaults to Bartlett with Andrews' bandwidth and no
  prewhitening. `sandwich::lrvar` defaults to the quadratic spectral
  kernel with prewhitening. The docstring lists the equivalent calls.
- `sp.periodogram` keeps the Fourier frequencies of the sample; R pads to
  a length with factors 2, 3 and 5 unless told not to. `fast=True`
  reproduces R.
- `sp.tsfilter(method='cf')` removes a drift by default, as the authors
  and statsmodels do; Stata does not. `drift=False` reproduces Stata.

## Open items

| Item | Why it is open |
| --- | --- |
| EGARCH, threshold GARCH, ARCH in mean | Chapter 8 names them; `sp.garch` has none. `sp.from_stata` reports `tarch()` and `archm` as untranslated and refuses `earch()`. |
| Standard errors of the variance decomposition | Stata's `irf table fevd` has them; `fevd()` returns shares only. |
| Bias-corrected bootstrap for impulse responses | Kilian's correction is the usual choice for persistent VARs. `boot='hall'` is the only alternative to the percentile band. |
| Exact diffuse initialisation in `sp.kalman_filter` | The large-variance approximation is used and said so. `burn=` removes the affected likelihood terms. |
| Beveridge-Nelson from an ARMA model | AR(p) only. |
| `tsfilter cf, stationary` and `smaorder()` | Stata's weights for the stationary variant did not match any published formula we could find, so they were not copied. |
| Zivot-Andrews lag choice at every break date | The order is chosen once, without the break. Stata's `zandrews` (SSC) was not installed and not compared. |
| Markov switching, time-varying VARs | Chapter 18, a survey chapter. Stochastic volatility arrived the same day from another pass (`sp.stochvol`); regime switching and time-varying VARs have nothing. |
| A reader for EViews-written `.xlsx` | Only the test converter handles it. |

## How to rerun

```bash
python tests/external_parity/neusser_convert.py <book folder>     # needs xlrd
STATSPAI_NEUSSER_DIR=<book folder> pytest tests/external_parity/test_neusser_time_series.py \
    tests/external_parity/test_neusser_quarterly_gdp.py
pytest tests/reference_parity/test_garch_extensions_stata_parity.py \
    tests/reference_parity/test_irf_bands_stata_parity.py \
    tests/reference_parity/test_johansen_lrtest_parity.py \
    tests/reference_parity/test_zivot_andrews_parity.py \
    tests/reference_parity/test_xcorr_parity.py tests/reference_parity/test_lrvar_parity.py \
    tests/reference_parity/test_spectral_parity.py tests/reference_parity/test_tsfilter_parity.py \
    tests/reference_parity/test_beveridge_nelson_parity.py \
    tests/reference_parity/test_statespace_parity.py
```

Reference fixtures are regenerated by the `_generate_*` scripts in
`tests/reference_parity/_fixtures/` (R 4.5.2 with `urca`, `sandwich`,
`KFAS`; Stata 18 for the `.do` files).
