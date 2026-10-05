# Wooldridge, *Introductory Econometrics*: what the examples showed about StatsPAI

Started 2026-10-05. Worktree `.claude/worktrees/wooldridge-textbook`, branch
`wt/wooldridge-textbook`.

## What was examined

The material sits in `改进建议-收集整理/Wooldridge-8e/`, which is gitignored.

| Material | Use |
| --- | --- |
| Heiss, *Using R / Python / Julia for Introductory Econometrics*: one script per example, chapters 2 to 19 | The list of what a reader of the book computes |
| The 112 datasets of the book as `.dta` | Inputs |
| The R `wooldridge` package and the Python `wooldridge` package | Data documentation |

There are no Stata logs in this material, so the log replay used for Stock
and Watson does not apply. The method here was to write each example with
StatsPAI and compare it with a reference run on the same file.

- `statsmodels` and `linearmodels`, which the Python scripts use.
- Stata 18 MP, run for this review, where the book's own numbers are Stata's
  (`prais`, `heckman`, `reg3`, `truncreg`, `glm`, `estat`, `dfbeta`).
- R (`censReg`, `survival`, `quantreg`, `lmtest`) for the examples the R
  scripts solve with a package.

More than a hundred quantities were compared across chapters 2 to 18.

## What agreed without any change

| Chapter | Quantity | Reference | Agreement |
| --- | --- | --- | --- |
| 2 to 7 | OLS coefficients, standard errors, R-squared, regression through the origin, confidence intervals, F tests, `lincom`, factor and interaction terms | statsmodels | 1e-12 or better |
| 8 | HC0 to HC3 standard errors, robust F test, Breusch-Pagan (LM and F), White, WLS, feasible GLS | statsmodels | 1e-13; WLS 1e-8 (single-precision weights) |
| 9 | RESET, listwise deletion, leverage, Cook's distance | statsmodels, R | 1e-13 |
| 10 to 12 | Distributed lags, long-run propensity, HAC standard errors, Durbin-Watson, Breusch-Godfrey | statsmodels, Stata | 1e-11 |
| 12 | Cochrane-Orcutt and Prais-Winsten | Stata `prais` | 4e-11 |
| 13, 14 | First differences, fixed effects, random effects | linearmodels | exact |
| 15 | 2SLS, Durbin-Wu-Hausman, Sargan, first-stage F | linearmodels and the manual regressions of the book | 1e-12 |
| 16 | 3SLS | Stata `reg3` | printed precision |
| 17 | Logit, probit, Wald tests, Poisson, Tobit, two-step Heckman, censored duration regression | statsmodels, R `censReg`, `survreg`, Stata | 1e-6 or better |
| 18 | Augmented Dickey-Fuller | statsmodels | 1e-12 |

Two references turned out to be the weaker side. The iteratively reweighted
median regression of `statsmodels` stops at an intercept of 1.620759 on
`rdchem`; `sp.qreg` and R `quantreg` agree on 1.6207404. `statsmodels`
iterates Cochrane-Orcutt with a Yule-Walker estimate of rho and gets 0.2959
on `barium`; StatsPAI and Stata `prais, corc` get 0.2934.

## What was wrong, and what changed

| # | Finding | Kind | Status |
| --- | --- | --- | --- |
| 1 | `sp.estat(result, 'leverage')` returned DFBETAS without the division by the square root of the diagonal of `(X'X)^-1`. The values were on the scale of the regressor and the `2/sqrt(n)` rule was applied to the wrong quantity | correctness, silent | fixed; `rstudent` and `rstandard` added |
| 2 | `result.predict(new_data)` after `sp.glm` returned the linear index, while `result.predict()` returned the mean. After `sp.poisson` and `sp.nbreg` it raised on the `_cons` column | correctness, silent for `sp.glm` | fixed |
| 3 | `sp.from_stata("heckman ..., select(...)")` produced the two-step estimator. Stata's default is maximum likelihood. The `twostep` option was reported as untranslated | translation, silent | fixed |
| 4 | `sp.estat(result, 'endogenous')` and `'overid'` closed with "p-value not available", next to the p-value they had just reported | reporting | fixed |
| 5 | A transformed outcome (`np.log(wage) ~ ...`) was refused by `sp.ivreg` and `sp.panel`; a transformed endogenous regressor was refused by `sp.ivreg`; `sp.qreg` took no transformed term at all | coverage | fixed |
| 6 | `log(wage)`, as R and Stata users write it, ended in a bare `KeyError` | coverage, bad failure | `log`, `log2`, `log10`, `log1p`, `exp`, `sqrt` read without the `np.` prefix; an unknown name is reported by name |
| 7 | `sp.panel(method='fd')` with the time variable among the regressors (Example 13.9 has a `year` trend) raised `KeyError` | bug | fixed |
| 8 | `sp.panel(method='fe')` raised when a regressor was absorbed by the unit effects (`educ` in Example 14.2). Stata and R omit it and say so | bad failure | omitted with a warning, listed in `model_info['omitted']` |
| 9 | `sp.panel(method='mundlak')` raised on a time-invariant regressor or on period dummies, the standard specification of section 14.3 | bad failure | the means that add no rank are left out, listed in `model_info['cre_means_omitted']` |
| 10 | `sp.test` could not name a coefficient such as `I(exper ** 2)` or `np.log(sales)`, and did not take a list of restrictions | coverage | fixed, also in `sp.lincom` |
| 11 | `sp.lrtest` raised `AttributeError` on logit, probit, Poisson and Tobit fits (Example 17.1) | coverage, bad failure | accepts maximum-likelihood regressions; refuses different samples, different models, reversed order |
| 12 | Results of `sp.regress` had no interval for a prediction and refused new data when the formula had a transform (sections 6.4 and 18.5) | missing option | `predict(data, what='confidence' / 'prediction' / 'link')` |
| 13 | Logit and probit `predict` took a design matrix only and failed on a DataFrame | coverage | takes a DataFrame |
| 14 | No test for ARCH (Example 12.9) | missing option | `sp.estat(result, 'archlm', lags=)` |
| 15 | The F test of the lagged residuals that the book reports for AR(q) serial correlation (Example 12.4) was not available | missing option | `sp.estat(result, 'bgodfrey', version='fstat')`, `sp.estat(result, 'durbinalt')` |
| 16 | The special form of White's test, on the fitted values and their squares (Example 8.5) | missing option | `sp.estat(result, 'white', variables='fitted')` |
| 17 | The quasi-Poisson standard errors of Example 17.3 | missing option | `sp.glm(..., scale='x2')` |
| 18 | `estat archlm`, `estat durbinalt`, `truncreg`, `glm` were not translated from Stata | coverage | translated |
| 19 | Standardized coefficients (`regress, beta`, section 6.1) were not available | missing option | `sp.estat(result, 'beta')` |
| 20 | `sp.tobit` and `sp.truncreg` took `y=` and `x=` lists only, so the square of Example 17.2 had to be a column | coverage | `formula=` |
| 21 | `sp.survreg(data, ...)` with the data first failed with `AttributeError` | bad failure | fixed |

Items 1 to 4 change numbers or text that StatsPAI used to return. They are
in `CHANGELOG.md` under Correctness and in `MIGRATION.md`.

Item 15 deserves a note, because three programs use three names. The
auxiliary regression is the same in all of them: residuals on the regressors
and `q` lags of the residuals.

| Statistic | Stata | R `lmtest` | statsmodels | StatsPAI |
| --- | --- | --- | --- | --- |
| `N R^2` | `estat bgodfrey` | `bgtest()` | `acorr_breusch_godfrey` (first two values) | `'bgodfrey'` |
| F of the lagged residuals | `estat durbinalt, small` | `bgtest(type="F")` | last two values | `'bgodfrey', version='fstat'` or `'durbinalt', version='fstat'` |
| `q` times that F | `estat durbinalt` | | | `'durbinalt'` |
| `N R^2 / q` | `estat bgodfrey, small` | | | not offered |

On `barium` with three lags Stata prints 14.768, 5.125, 15.374 and 4.923.
The scripts for Example 12.4 compute the second. Stata's `estat bgodfrey,
small` is left untranslated for that reason: it is a different number from
the one a reader of the book expects under the name F.

## Tests

```bash
pytest tests/test_wooldridge_review_fixes.py
STATSPAI_WOOLDRIDGE_DIR=<folder of .dta files> \
    pytest tests/test_wooldridge_examples.py
```

The first file needs no data. It pins every item above on simulated data
against statsmodels, linearmodels or an identity. The second holds the Stata
18 and R numbers on the book's datasets and is skipped without them.

## Where the book's practice has moved

The book is a sound first course, and most of it is unchanged practice. This
table reads its chapters against what applied work now expects and where
StatsPAI stands. It is a judgement. The references for each method are on
the docstring of the function.

| Chapter | The book does | Common practice now | In StatsPAI |
| --- | --- | --- | --- |
| 3 to 7 | Homoskedastic standard errors first, robust ones in chapter 8 | Robust standard errors from the start | `sp.regress(robust='hc1')`; the default stays classical, as in Stata and R |
| 8 | Breusch-Pagan and White tests, then WLS or feasible GLS | Robust standard errors whatever the test says; WLS for efficiency with robust errors on top | `sp.estat`, `sp.regress(weights=, robust=)` |
| 9 | RESET, proxy variables, outliers by studentized residuals | Sensitivity to an omitted variable stated as a number | `sp.sensemakr`, `sp.oster_bounds` |
| 10 to 12 | Durbin-Watson, Cochrane-Orcutt and Prais-Winsten as remedies | OLS with HAC standard errors; the AR(1) transformations need strict exogeneity | `sp.regress(robust='hac', hac_lags=)`, `robust='ewc'`; `sp.prais` is there for replication |
| 12 | ARCH as a test on squared residuals | The same test; a GARCH model when the variance is the object | `sp.estat(..., 'archlm')`, `sp.garch` |
| 13 | Two-period difference-in-differences by an interaction term | Event studies, estimators robust to heterogeneous effects under staggered adoption, sensitivity to pre-trends | `sp.did`, `sp.callaway_santanna`, `sp.honest_did` |
| 14 | Fixed against random effects by a Hausman test with classical variances | Fixed effects with clustered standard errors; the comparison by the cluster-robust test of the Mundlak terms | `sp.panel(method='mundlak', cluster=)` reports that Wald test |
| 15 | 2SLS, first-stage F, overidentification test | Effective F, Anderson-Rubin intervals, inference that survives weak instruments | `sp.iv_diag`, `sp.effective_f_test`, `sp.anderson_rubin_test` |
| 17 | Partial effects at the average, scale factors by hand | Average marginal effects with delta-method standard errors | `sp.margins` |
| 17 | Poisson with the variance scaled by the Pearson statistic | Poisson with robust standard errors, which does not assume the variance is proportional to the mean | `sp.poisson(robust='robust')`; `sp.glm(scale='x2')` reproduces the book |
| 17 | Heckman's two-step with an exclusion restriction | The same, with the fragility without an exclusion restriction said aloud | `sp.heckman` |
| 18 | Dickey-Fuller tests, Engle-Granger cointegration, spurious regression | DF-GLS for power; finite-sample critical values | `sp.unitroot(test='dfgls')`, `sp.engle_granger` |

## Left open

- **Tests of non-nested models.** The Davidson-MacKinnon J test and the
  encompassing F test of section 9.1 have to be assembled from `sp.regress`
  and `sp.test`.
- **A formula for `sp.heckman`.** It has two equations and takes `x=` and
  `z=` lists; `sp.tobit` and `sp.truncreg` now take a formula.
- **Names of built terms.** `sp.regress` names a term `I(exper ** 2)`;
  `sp.ivreg` and `sp.panel` name the same term `I[exper ** 2]`, and a factor
  level `year[1981.0]` where `sp.regress` has `C(year)[T.1981.0]`. The
  intercept is `Intercept`, `const` or `_cons` depending on the estimator.
  Unifying them changes result objects users index by name and belongs in a
  release of its own.
- **Prediction with a factor.** `predict(new_data)` rebuilds the coding from
  the levels named by the coefficients. When the new rows lack the reference
  level and every other level too, the coding cannot be recovered and the
  call is refused with a message.
- **Stata translations not attempted.** `hausman fe re` (needs stored
  estimates), `reg3` and `sureg` (multi-equation syntax), `cnreg` and
  `intreg`, `arch`.
- **A discrepancy not traced.** On one simulated series
  `statsmodels.acorr_breusch_godfrey` returns 43.127 where the auxiliary
  regression written out by hand, and StatsPAI, give 43.175. On `barium`
  the two agree to 1e-13 and both agree with Stata. The unit test uses the
  explicit regression.
