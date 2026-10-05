# Xu and Lan, *Causal-Inference Econometrics*: what its syllabus showed about StatsPAI

Done 2026-10-05. Worktree `.claude/worktrees/xuxiaojun-textbook`, branch
`wt/xuxiaojun-textbook`.

## What was examined

徐小君、蓝嘉俊《因果推断计量经济学》(清华大学出版社, 2025, ISBN
978-7-302-69328-4). The publisher's page offers no data and no code. What
is public is a 32-page sample (preface, table of contents and chapter 1),
kept in `改进建议-收集整理/徐小君-因果推断计量经济学/`, which is gitignored.

So there was nothing to replicate. The book was used as a syllabus instead.
Its twelve chapters are a complete undergraduate course that puts classical
econometrics and causal inference side by side.

| Chapter | Content | Commands the audit ran |
| --- | --- | --- |
| 1 | probability and statistics review, tests on a mean and on a variance | `summarize`, `ttest`, `sdtest`, `ztest`, `ci`, `sktest`, `prtest`, `bitest` |
| 2 | simple regression, OLS / ML / moments, quantile regression | `regress`, `predict`, `qreg`, `sqreg` |
| 3 | multiple regression, selection criteria, multicollinearity, heteroskedasticity, serial correlation | `estat ic / vif / hettest / imtest / ovtest / dwatson / durbinalt / bgodfrey / archlm`, `prais`, `newey`, `stepwise`, weights |
| 4 | functional form, dummies, restriction tests, binary outcomes, truncation, censoring, selection | `test`, `testparm`, `lincom`, `nlcom`, `probit`, `logit`, `margins`, `tobit`, `truncreg`, `heckman` |
| 5 | causal graphs, selection bias, Simpson's paradox | `sp.dag` (no Stata counterpart) |
| 6 | instrumental variables, endogeneity tests, LATE | `ivregress 2sls / liml / gmm`, `estat endogenous / firststage / overid`, `hausman` |
| 7 | Heckman selection, endogenous treatment | `heckman`, `etregress` |
| 8 | potential outcomes, experiments, matching | `teffects ra / nnmatch / psmatch / ipw / ipwra / aipw`, `tebalance` |
| 9 | regression discontinuity, parametric and nonparametric, fuzzy | polynomial `regress`, `rdrobust`, `rdbwselect`, `rddensity` |
| 10 | panel data, fixed and random effects, Hausman, DiD, DDD | `xtreg fe / re / be / mle`, `hausman`, `xttest0`, `areg`, factor interactions |
| 11 | AR, MA, ARIMA, ARCH, unit roots, cointegration, error correction | `corrgram`, `dfuller`, `pperron`, `dfgls`, `kpss`, `arima`, `arch`, `varsoc`, `egranger`, `vec` |
| 12 | distributed lags, partial adjustment, Almon lags, VAR, SVAR, local projections | lag operators, `nlcom`, `var`, `varstable`, `vargranger`, `irf`, `svar`, `lpirf` |

## Method

One do-file per chapter was written for this audit
(`tests/external_parity/xu_lan_syllabus/`). Each runs the commands the
chapter's methods are taught with, on datasets that ship with Stata
(`auto`, `nlsw88`, `mroz`, `womenwk`, `union3`, `cattaneo2`, `nlswork`,
`grunfeld`, `klein`, `wpi1`, `lutkepohl2`) and on the Senate RD data bundled
with StatsPAI. They were run in Stata 18 with a text log per chapter. Each
log was replayed through one `sp.stata` session by
`scripts/stata_log_replay.py`, which compares every number Stata printed.

A command the translator refuses hides what the function behind it would
have returned. So every refused command whose method StatsPAI implements
was also called directly and compared with the log by hand (`sp.heckman`,
`sp.etregress`, `sp.truncreg`, `sp.tobit`, `sp.arima`, `sp.garch`,
`sp.engle_granger`, `sp.var` impulse responses).

Chapter 5 has no Stata counterpart. The textbook graphs (confounder,
collider, M-bias, mediator, instrument, front door) were put through
`sp.dag`: adjustment sets, d-separation, bad controls and `sp.identify` all
give the textbook answers.

State before and after. At the start the chapter 10 log did not finish
replaying (item 7 below) and stopped on an error when it did (item 6), so
the first column covers the other ten logs.

| | Before (10 logs) | After (10 logs) | After (all 11) |
| --- | --- | --- | --- |
| Numbers compared and reproduced | 1,174 | 1,219 | 1,609 |
| Numbers different | 26 | 17 | 19, none of them a StatsPAI error (see below) |
| Commands `sp.stata` refused | 84 | 43 | 44 |

The counts understate the change. The replay script does not yet read the
output of `arima`, `arch`, `pperron`, `kpss`, `sdtest`, `ztest`, `nlcom`
or `svar`, so those commands now run but add nothing to the first row.
Their numbers were compared by hand and are pinned in
`tests/reference_parity/test_textbook_syllabus_stata_parity.py`.

## What was wrong

Each item below was traced to its cause before anything was changed. The
tests pin the fix on committed synthetic data, with Stata 18 output written
by `_fixtures/_generate_textbook_syllabus_stata.do`.

1. **`sp.arima` had no constant.** The default fitted a zero-mean model.
   An AR(1) on the differenced log wholesale price index gave 0.763 where
   Stata gives 0.615. With a mean of 10 and a true coefficient of 0.5 the
   old fit returns more than 0.95. The innovations path already estimated
   the mean, so the two methods of one function disagreed about the model.
   `trend=` was added. Track A module 39 did not catch it because its
   series has mean zero.
2. **`ivregress gmm` translated to 2SLS.** Stata's default weight matrix
   is robust and `sp.ivreg`'s is not. Nine numbers of chapter 6 differed,
   including Hansen's J. Another session found the same thing on Card's
   data the same night and fixed it on main first (`21398457`:
   `sp.iv(method='gmm', robust='hc1', small=False)` is `ivregress gmm`).
   The fix written here was dropped in favour of that one. The Stata
   numbers of this audit's own dataset are kept as a second check, and
   they agree.
3. **`heckman` translated to the wrong estimator.** Stata defaults to
   maximum likelihood and `sp.heckman` to the two-step, and the
   translation did not write the method out. The Wooldridge pass fixed
   that on main the same night. What this pass adds is `select(z1 z2)`
   without a selection variable, the form the Stata manual leads with,
   which was refused (`sp.heckman(select=None)`).
4. **Two-step Heckman standard errors used the expected information of the
   probit.** Stata and R `sampleSelection` use the observed information.
   The gap of 3e-4 had been sitting inside the tolerance of Track A module
   43 since June. Its budget is now 1e-6.
5. **`sp.garch(p=1, q=0)` returned estimates of an unidentified model.**
6. **`xtreg, fe` with year dummies failed on a short list of firms.** The
   statistics of the fixed-effects fit went through the between regression
   of the random-effects components.
7. **`sp.panel(method='mle')` did not finish on the NLS panel.** More than
   150 seconds against 0.3 after the likelihood was concentrated.
8. **A regression with 4,100 dummy columns spent more than 400 seconds
   checking for collinearity.** Two loops took one dot product per pair of
   columns.

## What was missing

Added because the syllabus needs them and nothing else in the package does
the job.

- `sp.sdtest`, `sp.ztest`: the two worked examples of chapter 1.
- `sp.estat(..., 'durbinalt' | 'archlm')`: chapters 3 and 11. The
  Wooldridge pass added both to main the same night; the version written
  here was dropped and its Stata numbers kept as a second check.
- `sp.unitroot(test='pp' | 'kpss')`: chapter 11.
- `sp.nlcom`: the long-run multiplier of chapters 4 and 12.
- `sp.svar` with short-run, long-run and sign restrictions, and
  `VARResult.fevd()`: section 12.3.
- Stata factor-variable names in `sp.test` / `sp.lincom`, and `testparm`.
- Eleven commands in `sp.stata` (see the changelog).

Stata agrees with all of these to the tolerances stated in the test file.
For the over-identified SVAR and for `arima`, Stata's default convergence
stops short of the optimum. The generator iterates Stata to tight
tolerances, at which point its log-likelihood equals ours to 13 digits.

## Second round, the same day

Bryce delegated the open decisions. Four were closed.

- **`tobit` translated with a lower limit Stata did not ask for.** Found
  while closing the bare `ll` item: `tobit y x, ul(2)` kept `sp.tobit`'s
  default `ll=0`. Both limits are now written out and `sp.tobit` takes
  `ll=None`. Stata 18 on a sample censored from above gives 0.998381
  (0.045687) with 107 censored observations; so does the translation now.
- **Variable-name abbreviations.** The Wooldridge pass had added them for
  the main varlist of the regression commands. This round extends the
  same rule to `heckman`, `truncreg`, `etregress` and `prais`, and to the
  options that hold a varlist.
- **`ttesti`.** `sp.ttest` takes `n=`, `mean=`, `sd=`.
- **A stale test on main.** `test_stale_covariance_is_not_used` still
  required a joint test under CR2 to be refused, after `vce='cr2'` began
  storing its full covariance. It now checks that the test uses that
  matrix.

The dummy-variable regression behind `areg` lost another ten seconds
(the bread matrix was assembled entry by entry). It is still slow.

## Differences that are not errors

- **`tobit`, 14 numbers of chapter 4.** Stata's default convergence rule
  stops at a relative error of about 1e-6. With `nrtolerance(1e-12)` Stata
  prints our coefficients and standard errors to every digit.
- **`teffects ra` / `ipwra`, the sample size (3 numbers).** The replay
  script reads the wrong field; `result.n_obs` is 4,642 as in Stata.
- **`areg`, the constant (2 numbers).** `areg` reports the intercept at the
  average absorbed effect; the dummy-variable regression it is translated
  to reports the first group's level. The translation says so in a note.
  Slopes and their standard errors agree.
- **`arima` and `arch` to four or five digits** when Stata runs at its
  default tolerance.

## Left open

| Item | Why it is open |
| --- | --- |
| Variable-name abbreviations outside the commands and options listed above | Each remaining command needs its own decision about which words are variables. |
| `ci means` / `ci variances`, `sktest`, `swilk`, `prtest`, `tabulate, chi2` | Chapter 1 commands with no `sp` counterpart yet. Each is small; none was added without a second use. |
| `stepwise:` prefix, `estat szroeter`, `vwls` | `sp.stepwise` exists with different entry and exit rules; the other two have no counterpart. |
| `margins, predict(ystar(0,.))` after `tobit`; predictive margins | `sp.margins` has no censored-outcome predictions. |
| `sqreg`, `bsqreg`, `iqreg` | Bootstrap standard errors; only the point estimates could be compared. |
| `dfuller, drift`; `dfgls` table; `L.D.x`; `arima, ma(1 4)`; `arch` with an ARMA mean; `predict` after `arima` | Not translated. `sp.unitroot` and `sp.arima` do not have these variants. |
| `egranger` | `sp.engle_granger` reproduces its statistic (-3.978, -1.799); the command, its `ecm` option and the regression table are not translated. |
| `irf table`, `svar` in `sp.stata`, `varnorm` | `sp.svar` and `VARResult.fevd` give the numbers; the session does not yet pass `matrix` definitions to them. |
| `lpirf` | `sp.local_projections` is a single-equation estimator with a different specification. Not compared. |
| `heckman, mills()`; `etregress, poutcomes` | Options with no counterpart; reported as untranslated. |
| `areg` with thousands of groups | Translated to dummy variables on purpose (degrees of freedom). Now about 3 minutes on 4,134 groups, dominated by a dense QR. An absorbing path with `areg`'s degrees of freedom would make it instant. |
| `tobit y x, ll` with no value | Stata censors at the observed minimum. Now refused with that explanation; running it needs the data. |
| `regress D.(y x1 x2)` | An operator applied to a parenthesised varlist is not expanded. |

## Reproducing

```bash
# Stata 18, in tests/external_parity/xu_lan_syllabus/ (see its README)
do _master.do
# then
python scripts/stata_log_replay.py tests/external_parity/xu_lan_syllabus/*.log \
    --data tests/external_parity/xu_lan_syllabus
pytest tests/reference_parity/test_textbook_syllabus_stata_parity.py -q
```
