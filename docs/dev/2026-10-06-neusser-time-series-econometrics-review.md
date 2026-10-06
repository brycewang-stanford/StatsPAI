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

A second edition appeared in 2025 (Springer, DOI
`10.1007/978-3-031-88838-0`, Crossref record checked). This pass used the
companion data of the first edition; whether the second changes the
examples was not checked.

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
| Asymmetric volatility, variance in the mean (8.1) | none | `sp.garch(model='gjr' / 'egarch', in_mean=True, threshold=)` |
| Bands for impulse responses (15.4) | point estimates only | `sp.irf(ci='asymptotic' / 'bootstrap')`, `SVARResult.irf(ci='bootstrap')`, `VARResult.fevd(ci=)`, `SVARResult.fevd(ci=)`, `boot='kilian'` |
| Restrictions on the cointegration space (16.5) | rank test only | `sp.johansen_lrtest` |
| Unit root with a break (7) | none | `sp.zivot_andrews` |
| Cross-correlation and prewhitening (11) | none | `sp.xcorr` |
| Long-run variance (4.4) | inside `sp.regress(robust='hac')` only | `sp.lrvar` |
| Spectral density (6) | none | `sp.periodogram`, `sp.cumulative_periodogram_test` |
| Linear filters (6.5) | none | `sp.tsfilter` (HP and one-sided HP, Baxter-King, Christiano-Fitzgerald in three variants, Butterworth, Hamilton) |
| Beveridge-Nelson decomposition (7.1) | none | `sp.beveridge_nelson`, from an AR or an ARMA model |
| State space models (17) | `sp.dlm` (random-walk regression only) | `sp.kalman_filter`, `sp.statespace`, exact diffuse start |
| Time-varying coefficients (18) | none | `sp.tvp_var` |
| Regime switching (18) | none | `sp.mswitch`, `sp.mswitch_lrtest` |
| Drifting volatilities (18) | none | `sp.tvp_var_sv` |
| Reading the book's EViews workbooks | pandas error | `sp.read_data` repairs them in memory |

Evidence for each, all on committed simulated data with a generator script
next to the reference file:

| Function | Reference | Agreement |
| --- | --- | --- |
| `sp.garch(ar=, dist='t')` | Stata 18 `arch, ar() distribution(t)` | coefficients 5e-5, standard errors 5e-4, log-likelihood 1e-8 |
| `sp.garch(model='gjr' / 'egarch')` | Stata 18 `arch, tarch()` and `arch, earch() egarch()` | log-likelihood 1e-8, coefficients 2e-4, standard errors 2e-3, with normal and t errors and an AR term |
| `sp.irf(ci='asymptotic')`, `fevd(ci='asymptotic')` | Stata 18 `irf create` | simple, orthogonalised and cumulative responses, variance shares and their standard errors to 1e-6, the precision of Stata's single-precision `.irf` file |
| `sp.garch(in_mean=True)` | Stata 18 `arch, archm` | log-likelihood 1e-8, coefficients 1e-4, standard errors 2e-3 |
| `sp.garch(in_mean='sd' / 'log', in_mean_lags=)` | Stata 18 `archmexp()`, `archmlags()` | log-likelihood 1e-8, coefficients 3e-4, standard errors 3e-3 (six specifications) |
| `sp.irf(ci='bootstrap')` | known truth | 90% bands cover between 80% and 97% over 150 samples; spread within a fifth of the delta method at T = 600 |
| `boot='kilian'` | known truth | root 0.92, T = 60, nominal 90%: percentile 38%, Hall 56%, delta method 62%, bias-corrected 86% (300 samples) |
| `sp.johansen_lrtest` | `urca` `blrtest`, `bh5lrtest`, `alrtest` | statistic 1e-8, degrees of freedom equal, three deterministic cases |
| `sp.zivot_andrews` | `urca::ur.za` | statistic, path over break dates and regression to 1e-7, break date equal |
| `sp.xcorr` | R `ccf` and `ar`, Stata `xcorr` | 1e-10 |
| `sp.lrvar` | `sandwich` `kernHAC`, `NeweyWest`, `lrvar` | 1e-9 over 104 cases |
| `sp.periodogram` | R `spec.pgram`, `spec.ar`, Stata `pergram` | 1e-9 |
| `sp.cumulative_periodogram_test` | Stata `wntestb` | statistic 1e-12, p-value 2e-8 (Stata truncates the series) |
| `sp.tsfilter` | Stata `tsfilter`, statsmodels | 1e-9 |
| `sp.beveridge_nelson` | the long-horizon forecast that defines the trend | 1e-8; no package reference found |
| `sp.kalman_filter`, `sp.statespace` | `KFAS`, statsmodels | filter and smoother 1e-9, estimates 1e-6 |
| exact diffuse start | `KFAS` exact diffuse, statsmodels | smoothed states 5e-15, covariances 1e-12, diffuse likelihood 1e-12 (ours keeps the `2 pi` constant KFAS drops for the diffuse observations) |
| `sp.beveridge_nelson(order=(p, q))` | long-horizon forecast of the same fitted model, R `arima` | cycle 5e-13 at common estimates |
| `sp.tsfilter(sma_order=, one_sided=)` | Stata `tsfilter cf, smaorder()`; expanding-window HP | 1e-9; 1e-10 |
| `sp.zivot_andrews` with `zandrews` options | Stata `zandrews` 1.0.5 (SSC) | statistic 5e-13, break date and lag equal, 48 cases |
| `sp.mswitch` | Stata 18 `mswitch dr` / `ar`; statsmodels | likelihood and regime probabilities at Stata's estimates 1e-11, covariance 5e-6, 18 models |
| `sp.mswitch_lrtest` | known truth | rejects 4.5% at the 5% level under one regime (se 1.1), 95% under two well-separated regimes; a full double bootstrap agrees with the one-draw shortcut within Monte Carlo error |
| `sp.tvp_var_sv` | exact conditionals; joint-distribution test; `bvarsv` as a screen | each Gibbs block within Monte Carlo error of its exact counterpart; volatility medians 1 to 4% from `bvarsv` |
| `sp.tvp_var(method='kalman')` | `sp.dlm` per equation; R `KFAS` | 2e-11; 1e-12 under a proper prior, 2e-6 under the diffuse one |
| `sp.tvp_var(method='forgetting')` | discounted least squares; independent implementation | 1e-8; 1e-10 |

## The book's examples

Opt-in tests: `tests/external_parity/test_neusser_time_series.py` and
`test_neusser_quarterly_gdp.py`.

- **5.6, ARMA for Swiss GDP growth.** ARMA(1,3) and AR(2) agree with R
  `arima` (log-likelihood -107.0496 and -112.9774). Over the 6 x 6 grid of
  orders `sp.arima` had the higher likelihood in ten cells and the lower
  in one, ARMA(5,2), where R reaches -107.03 and we stopped at -110.13. A
  third start, from zero ARMA coefficients (where R begins), is now tried
  for a final fit; ARMA(5,2) reaches R's value and no other cell moves.
  A second case came from the level series. For the quarterly change of
  log GDP, MA(3) stopped at -249.22 with a root on the unit circle, below
  the MA(2) it nests (-244.90); R stops at the same point. A start from
  the estimates of each model with one term fewer now gives -243.82.
- **8.4, Swiss Market Index.** The GARCH fix above. AR(1)-GARCH(1,1) with
  t errors gives 7.29 degrees of freedom (Stata 7.2867) and a likelihood
  177 points above the normal model. `rugarch` with its hybrid solver
  returned a mean of 3.64 for ARCH(1) on returns whose mean is 0.04; that
  is a failure of the reference, noted so that nobody compares against it.
  The leverage effect is large on this index. The threshold model reaches
  -5545.146 and EGARCH -5538.915, both more than 50 log-likelihood points
  above GARCH(1,1), and both equal to Stata.
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
  corrected recursion agrees with `sp.kalman_filter` to 2e-16 (recomputed
  for this note), and that smoother agrees with KFAS on six other models
  to 5e-15. A draft note to the author is in
  `docs/dev/2026-10-06-neusser-kalman-smoother-note-draft.md`; it has not
  been sent.

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
- `sp.garch(model='gjr')` writes the threshold term on negative shocks
  (`gamma > 0` is leverage). Stata's `tarch` is on positive shocks, so
  `gamma = -tarch` and `alpha = arch + tarch`. The likelihood is the same.
  Stata starts the threshold recursion with the full pre-sample value and
  centres EGARCH's `|z|` at `sqrt(2/pi)` under t errors too; both were
  identified by evaluating our likelihood at Stata's estimates.
- `sp.garch(in_mean=True)` follows Stata's estimator. The pre-sample
  variance is the mean squared innovation at the estimates, and Stata
  treats it as a constant while climbing the likelihood. Treating it as a
  function of the parameters gives a point 0.008 standard errors away
  with a likelihood 8e-5 higher. Both are defensible; the default is the
  reference's, and `test_presample_value_is_held_fixed_as_in_stata` shows
  the mechanism.
- With `archmexp()` and `archmlags()` together, Stata starts the lagged
  term from the untransformed pre-sample variance (the variance where the
  standard deviation belongs). `sp.garch` transforms it. The likelihoods
  then differ in the second decimal; the test rebuilds Stata's value from
  its rule to 1e-9. Without lags, or without a transform, the two agree.
- Under a diffuse prior `sp.tvp_var` and KFAS looked 2e-6 apart. A
  Kalman filter in 60-digit arithmetic shows both within 2e-8 of the
  exact answer relative to `max(|value|, 1)`; the larger figure was pure
  relative error on coefficients near zero. Neither side is wrong.
- Stata does not restrict the sign of `arch + tarch`. With `arch(1/2)
  tarch(1)` on the test series its estimate has the variance falling
  after a positive shock at lag one. `sp.garch` keeps `alpha >= 0`, stops
  on the boundary, warns, and fits 0.15 log-likelihood points lower.
- The default bootstrap band stays the percentile one, as in Stata and
  `vars`, with a warning above a root of 0.9. The coverage numbers argue
  for `boot='kilian'` as the default; that is a decision for Bryce.
- Stata's `tsfilter cf, stationary` does not compute the formula of its
  manual: at interior dates the weight on the first and on the last
  observation is the ideal weight of the neighbouring lag. `sp.tsfilter`
  implements the manual's formula; the test rebuilds Stata's numbers from
  the rule above.
- Stata's `zandrews` chooses the lag order once, from the regression
  without a break, ignores `maxlags()` unless `lagmethod(input)`, and trims
  with `int(trim T + 0.49)`. `lag_rule='zandrews'` and `trim_rule='zandrews'`
  reproduce it; the defaults are unchanged.
- Three findings about the Markov-switching references, each with
  independent evidence in `test_mswitch_parity.py`. After `mswitch ar ...,
  switch(z)` Stata's `predict` disagrees with Stata's own likelihood (our
  likelihood equals `e(ll)` to 6e-14 and our probabilities equal
  statsmodels to 1e-15; Stata's are off by up to 0.04). Stata's
  three-state AR(1) on the test series stops unconverged, 22
  log-likelihood points below the maximum. statsmodels with a switching
  variance and two or more AR lags disagrees with both Stata and us.
  Stata's default `predict, yhat` applies the transition matrix once more
  than the one-step prediction; ours equals `yhat smethod(filter)`.
- On the US data the three time-varying VARs now give a consistent
  reading once volatility is modelled. The interest-rate shock
  volatility peaks in 1980Q4 (standard deviation 2.2 against 0.2 in
  1996). The response of output to a unit interest-rate shock did not
  change detectably (posterior probability of a weaker response 0.57 to
  0.60). The forgetting-factor filter's "weaker over time" was smaller
  shocks, not weaker transmission.
- `sp.tsfilter(method='cf')` removes a drift by default, as the authors
  and statsmodels do; Stata does not. `drift=False` reproduces Stata.

## Open items

Rounds two and three closed most of the first list. What is left:

| Item | Why it is open |
| --- | --- |
| Section 15.4.5 again | A web search for the book's restrictions found nothing usable; the paper itself (Blanchard, AER 1989) was not read. Left as it was. |
| The default bootstrap band | Still the percentile band of Stata and `vars`, with a warning above a root of 0.9. The coverage numbers argue for `boot='kilian'`; that is a decision for Bryce. |
| Section 15.4.5, Blanchard (1989) | The book's AB restrictions could not be recovered without the text. |
| `lag_selection='break'` in `sp.zivot_andrews` | No reference implementation chooses the lag order at every break date (Stata's `zandrews` chooses it once, without the break). Checked against an independent recomputation only. |
| The forgetting-factor TVP-VAR | No package reference. The one CRAN implementation found (`ConnectednessApproach::TVPVAR`) computes a different recursion from its second step on. Checked against discounted least squares and an independent implementation in the test file. |
| `sp.mswitch_lrtest` beyond one regime against two | Size evidence exists for one against two regimes with a switching constant. Two against three, and switching variances (where the likelihood is unbounded), have mechanical tests only. Garcia's asymptotic critical values were not implemented: the paper was not read. |
| `sp.tvp_var_sv` against `bvarsv` | Volatility medians differ by 1 to 4% on average (up to 11% at the first date), 1 to 3 Monte Carlo standard errors, with a smooth tilt over time. Matching the estimation window and rescaling the volatility prior did not close it. Reported as a screen, not an equivalence. |
| Mixture reweighting in `sp.tvp_var_sv` | The seven-normal approximation is used without the reweighting step; the joint-distribution test shows a bias of about 0.5% in the volatility innovation variance. |
| Three notes to third parties | `docs/dev/2026-10-06-neusser-kalman-smoother-note-draft.md` (the book's smoother), `docs/dev/2026-10-06-stata-tsfilter-cf-stationary-note-draft.md` (Stata's `tsfilter cf, stationary`) and `docs/dev/2026-10-06-stata-mswitch-predict-note-draft.md` (Stata's `predict` after `mswitch ar, switch()`). Drafts; not sent. |

## What the full test suite showed

The pushes of rounds one to three ran the tests of the files they
touched and every pre-push gate, not the whole suite. A full run on main
after round three (31,369 passed, 7 failed, 89 minutes) found four
failures from round one: `ZivotAndrewsResult` did not inherit the result
protocol, which two audits check. Fixed in round four. Two more came
from another pass (`alternatives` naming functions that do not exist),
and one is a timeout test that passes on an idle machine. The lesson is
in CLAUDE.md: the result-protocol and agent-contract audits belong to
the set run before a push that adds a result class.

## How to rerun

```bash
python tests/external_parity/neusser_convert.py <book folder>     # needs xlrd
STATSPAI_NEUSSER_DIR=<book folder> pytest tests/external_parity/test_neusser_time_series.py \
    tests/external_parity/test_neusser_quarterly_gdp.py
pytest tests/reference_parity/test_garch_extensions_stata_parity.py \
    tests/reference_parity/test_garch_asymmetric_stata_parity.py \
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
