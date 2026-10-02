# Stock & Watson 4E replication files: what they showed about StatsPAI

Started 2026-10-02. Worktree `.claude/worktrees/sw-textbook`, branch
`wt/sw-textbook`.

## What was examined

The replication files for *Introduction to Econometrics*, 4th edition, from
<https://www.princeton.edu/~mwatson/Stock-Watson_4E/>. Chapters 1 to 7, 9 to
13 and 15 and the dataset bundle came from the Wayback Machine. The zips for
chapters 8, 14, 16 and 17 are not archived and the origin is behind a
Cloudflare challenge, so those four have to be downloaded by hand. The files
sit in `改进建议-收集整理/Stock/`, which is gitignored.

Three kinds of material are in there.

| Material | Chapters | Use |
| --- | --- | --- |
| Stata do-files with the logs they produced | 2 to 7, 9 to 13 | Answer key for `sp.stata` |
| RATS programs with their output | 15 | Reference for AR / ADL, BIC, ADF, QLR, pseudo out-of-sample forecasts |
| Datasets (`.dta`, `.xlsx`) | all | Inputs |

## Method

A Stata log holds the commands that ran and what they printed. Replaying each
command through one `StataSession` and comparing every printed number is a
stricter test than asking whether a command is translated.
`scripts/stata_log_replay.py` does this for any log. `scripts/stata_corpus_scan.py`
had reported 97.6% of this corpus as translated faithfully before any of the
problems below were found.

Two details matter when reading a replay.

- Stata's `generate` stores single precision. A replay that computes in
  double differs from the log at about 1e-5 in the coefficients. The session
  follows Stata.
- A number matches when it is within two units of the last digit Stata
  printed.

Current state on the 13 logs: 1,186 numbers reproduced, none different. Run
it with

```bash
python scripts/stata_log_replay.py <files> --data <files>/SW_4E_Replication_Data
STATSPAI_SW4E_DIR=<files> pytest tests/external_parity/test_stock_watson_4e_logs.py
```

## What was wrong, and what changed

| # | Finding | Kind | Status |
| --- | --- | --- | --- |
| 1 | `probit` / `logit` / `poisson` / `nbreg, vce(robust)` translated to HC1 (`N/(N-K)`); Stata uses `N/(N-1)`. 93 standard errors and every Wald test after them were off in chapter 11 | translation, silent | fixed |
| 2 | `ivreg2` / `ivregress, robust` without `small` translated to HC1; Stata applies no factor | translation, silent | fixed |
| 3 | `regress ...; summarize x; test x` ran `test` on the summary table | runner | fixed |
| 4 | Legacy `ivreg` was unknown, and the corpus scan left unknown commands out of its denominator | coverage | fixed |
| 5 | `ttest` had no counterpart in the package | missing function | `sp.ttest` |
| 6 | `correlate`, `pwcorr`, `newey`, `dfuller`, command abbreviations (`summ`, `regr`), `tobit, vce()` | coverage | translated |
| 7 | `if` / `in` qualifiers made `sp.stata` refuse the line | coverage | applied, with Stata's missing-value rules |
| 8 | No data steps, so a do-file had to be rewritten before it ran | coverage | `generate`, `replace`, `keep`, `drop`, `sort`, `mvdecode`, `encode`, `preserve`, `restore`, `predict`, `scalar`, `display` |
| 9 | `sp.regress(robust='hac')` had no way to set the lag length, and could not reproduce Stata's `newey` (it scales by `N/(N-K)`) | missing option | `hac_lags=`, `hac_small=` |
| 10 | No unit-root test for a single series | missing function | `sp.unitroot` (ADF, DF-GLS) |
| 11 | AR / ADL forecasting, lag selection and pseudo out-of-sample evaluation had to be assembled by hand | missing function | `sp.ardl` |
| 12 | `sp.structural_break(method='sup-f')` tested every coefficient with a homoskedastic F; the textbook's QLR holds the lags of `y` fixed and is robust | missing option | `break_vars=`, `vce=` |
| 13 | Time-series operators (`L.x`, `D.x`, `L(1/4).x`) were refused | coverage | resolved against the `tsset` / `xtset` time variable, within panel |

Item 9 deserves a note. Track A module `51_newey` passed against Stata with
a 1e-2 tolerance. The gap was the documented `N/(N-K)` factor. With
`hac_small=True` the same golden file is matched to 3e-16.

Item 10: the ADF statistic, p-value, critical values and chosen lag agree
with `statsmodels.adfuller`, and the regression reproduces the RATS output
in chapter 15 (coefficient -0.019306733, t = -1.95436). The DF-GLS statistic
agrees with the `arch` package to 1e-14. Its critical values are a response
surface in the sample size and the lag order, fitted to a simulated null by
`scripts/simulate_dfgls_critical_values.py`. The asymptotic values are too
lenient at the sample sizes macro data come in: with a constant the 5% point
is -1.95 in the limit and about -2.27 at 50 observations.

Items 11 and 12 are checked against the RATS output of chapter 15, number
for number: the AR(1), AR(2), ADL(2,1) and ADL(2,2) coefficients and robust
standard errors, their forecasts of 2017:Q4, the Granger statistic, the two
BIC / AIC tables, the pseudo out-of-sample bias and mean squared error, and
the QLR statistic with its date. RATS `linreg(robust)` is HC0; its QLR
scales the HC0 Wald by `ndf/nobs`, which is the HC1 statistic.

```bash
STATSPAI_SW4E_DIR=<files> pytest tests/external_parity/test_stock_watson_4e_ch15.py
```

## Where the textbook's practice has moved since 2018

This is a reading of the book's chapters against what applied work now
expects, and where StatsPAI stands on each. It is a judgement, not a
citation list; the references for each method are on the function's own
docstring.

| Chapter | The book does | Common practice now | In StatsPAI |
| --- | --- | --- | --- |
| 3, 5 to 7 | Heteroskedasticity-robust SEs by default, F tests | Unchanged | `sp.regress(robust='hc1')`, `sp.test` |
| 8 | Polynomials, logs and interactions by hand | Binned scatter and flexible interaction diagnostics before choosing a functional form | `sp.binscatter`, `sp.interflex` |
| 9 | Internal and external validity as a checklist | Formal sensitivity to an omitted variable | `sp.sensemakr`, `sp.oster_bounds`, `sp.spec_curve` |
| 10 | State and time fixed effects, SEs clustered by state | Same estimator for a static regressor. With few clusters, wild cluster bootstrap or CR2. With a policy adopted at different dates, two-way fixed effects is no longer the default | `sp.feols`, `sp.wild_cluster_bootstrap`, `sp.cr2_se`, `sp.callaway_santanna`, `sp.bacon_decomposition` |
| 11 | Probit and logit, effects computed by plugging sample means into the index by hand | Average marginal effects with delta-method SEs | `sp.margins` |
| 12 | 2SLS, first-stage F above 10, J test | Effective F, Anderson-Rubin intervals, tF critical values; the F above 10 rule does not control size with robust errors | `sp.iv_diag`, `sp.effective_f_test`, `sp.anderson_rubin_ci`, `sp.tF_adjustment` |
| 13 | Differences estimator, difference-in-differences, sharp and fuzzy regression discontinuity | Event studies with heterogeneity-robust estimators and sensitivity to pre-trends; local polynomial RD with robust bias-corrected intervals and a density test | `sp.did`, `sp.event_study`, `sp.honest_did`, `sp.rdrobust`, `sp.rddensity` |
| 14 | Ridge, lasso and principal components for prediction, tuned by cross-validation | The same tools as nuisance learners inside a causal estimator | `sp.lasso_select`, `sp.rlasso`, `sp.dml`; ridge and principal-components prediction are not offered as standalone functions |
| 15 | AR and ADL forecasts, BIC, QLR break test, pseudo out-of-sample RMSFE | Unchanged as teaching material | `sp.ardl` (`.forecast()`, `.granger()`, `.poos()`), `sp.structural_break(method='sup-f', break_vars=, vce='hc1')`, `sp.unitroot` |
| 16 | Distributed lags with HAC errors, truncation `m = 0.75 T^(1/3)` | Local projections; larger HAC bandwidths with fixed-b critical values | `sp.local_projections`, `sp.regress(robust='hac', hac_lags=)`; fixed-b and EWC inference are not implemented |
| 17 | VAR, DF-GLS, cointegration, GARCH | Unchanged | `sp.var`, `sp.unitroot`, `sp.engle_granger`, `sp.johansen`, `sp.garch` |

## Open items

| Item | Why it matters | Size |
| --- | --- | --- |
| `xtreg, fe` prints `_cons` (mean of the fixed effects); the translation has slopes only | 20 numbers in chapters 10 and 13 have no counterpart | small |
| `tin()`, `tsset` dates, `pctile` | chapter 2 and 4 boxes do not replay | small |
| Fixed-b / EWC inference for HAC | the textbook authors' own later recommendation | medium |
| Ridge and principal-components prediction with cross-validated MSPE | chapter 14 | medium |
| Chapters 8, 14, 16, 17 files | not downloaded | needs Bryce |
| `summarize, detail` percentiles follow pandas' interpolation, not Stata's rule | differs between observations; stated in `semantics`, not fixed | small |
