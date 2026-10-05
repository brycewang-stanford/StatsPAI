# Hansen, *Econometrics*: what the book's programs showed about StatsPAI

Started 2026-10-05. Worktree `.claude/worktrees/hansen-textbook`, branch
`wt/hansen-textbook`.

## What was examined

The material sits in `改进建议-收集整理/14-Hansen-Econometrics -----/`, which is
gitignored. It is the archive the author distributes with the book
(Princeton University Press, 2022).

| Material | Use |
| --- | --- |
| 16 Stata do-files, chapters 3 to 26 | Run in Stata 18, replayed through `sp.stata` |
| 22 datasets (`.dta`) | Inputs |
| 128 R scripts, 5 MATLAB scripts | Read. Most draw the book's figures from simulated or closed-form curves. The ones that estimate something repeat the do-files |
| 5 `.log` files | Simulated critical values for the Dickey-Fuller, Engle-Granger, KPSS and Johansen tests |

The do-files come without output. The answer key is a log made by running
them (`tests/external_parity/hansen_econometrics_prepare.py` writes the
run folder). `scripts/stata_log_replay.py` then runs every logged command
through one `sp.stata` session and compares each printed number with what
StatsPAI returns, to the precision Stata printed it.

The book is from 2022 and its programs are older in places (`xi:`, a Stata
7 data file, `use ..., replace`). It was used as a probe. Nothing was
changed to imitate an old convention.

## The count

| | Reproduced | Different | Not run |
| --- | --- | --- | --- |
| First replay | 1,925 | 112 | 146 |
| Now | 3,876 | 10 | 54 |

Thirty more numbers are bootstrap standard errors, which the replay marks
`random`. Of the 54 commands not run, 33 are the choice-model commands of
chapter 26. The opt-in test
`tests/external_parity/test_hansen_econometrics_logs.py` holds the
per-chapter numbers and the two ledgers (documented differences, declined
commands).

## What was wrong, and what changed

| # | Finding | Kind | Status |
| --- | --- | --- | --- |
| 1 | `xtreg, re` with industry dummies (Table 17.2): coefficients differed from Stata in the fourth digit. The within residual variance was divided by `N - G - K`, counting regressors that are constant within firm as within parameters | correctness, silent | fixed in `panel/xt_tools._swamy_arora`, which `panel/_cre.py` now calls instead of keeping a second copy of the arithmetic |
| 2 | `dfuller` on the labour force participation rate (chapter 16): statistic +1.145 with a trend, Stata's p-value 1.0000, ours 0.9959. MacKinnon's cubic was evaluated above its fitted range, where it bends down: p = 0.06 at +3.25 and 0.00 at +4.0 | correctness, silent | p-value is 1 above the range (2.74 with a constant, 0.70 with a trend) |
| 3 | 2SLS with a cubic in class size (chapter 20): coefficients 34.203155 against Stata's 34.203130. A 50-digit solve of the same system gives 34.2031291, Stata's number | numerical precision | projection by QR, second stage by least squares on the projected regressors; error 3e-13 |
| 4 | `bys store: gen nperiods = [_N]` (chapter 18) was refused, and the 82 numbers after it differed because the sample was never restricted | translator | square brackets group like parentheses, checked in Stata (`display [2+3]*2` is 10) |
| 5 | `ivregress ... (edu = i.qob#i.yob)` (Angrist and Krueger, chapter 12) was refused. Run directly, `sp.iv` stopped with numpy's `Singular matrix` | bad failure | collinear instruments are dropped with a warning; the formula no longer duplicates dummies; 180 instruments on 329,509 rows run in 27 seconds and agree with Stata |
| 6 | `sp.clogit` on the travel-mode data (chapter 26) took 22 seconds | performance | 0.13 seconds |
| 7 | `Koppelman.dta` is a Stata 7 file. `sp.read_data` passed on pandas' refusal | coverage | format 110 is format 111 with the storage types spelled as letters; read through a converted copy |
| 8 | No constrained regression (`cnsreg`, chapter 8), no nonlinear least squares (`nl`, chapter 23), no principal components or factor analysis (chapter 11), no general jackknife (chapter 10) | missing | `sp.cnsreg`, `sp.nls`, `sp.pca`, `sp.factor`, `sp.jackknife` |
| 9 | `var` / `varsoc` / `svar` with `exog()` (Blanchard and Perotti, Blanchard and Quah, chapter 15) | missing option | `sp.var(exog=)`, `sp.varsoc(exog=)` |
| 11 | Model selection and averaging (chapter 28) existed only for double machine learning | missing | `sp.model_average`: AIC, BIC, cross-validation; Mallows, jackknife and smoothed weights. Checked against `figure28_5.R` and against `quadprog` on committed data |
| 10 | `irf table`, `estimates stats`, `jackknife:` / `bootstrap:`, `vce(jackknife)`, `L(1/3).D.x`, `lag()` for `lags()`, `perfect`, `forcenonrobust`, `r(sargan)`, `e(rank)`, `nlcom (a)/(b)`, `lincom x + z/5` | translator | run |

## What agreed without any change

Chapters 3 and 4 (OLS, leverage, HC0 to HC3, clustered standard errors),
the first-stage and reduced-form regressions and every 2SLS and LIML fit
of chapter 12 on the Card data, all the autoregressions and Newey-West
standard errors of chapter 14, the unit-root regressions, `vec` and
`vecrank` of chapter 16, fixed effects and the difference-in-differences
regressions of chapters 17 and 18, the quantile regressions of chapter 24
and the logit and probit marginal effects of chapter 25.

## Differences that stay, and why

**`nl` with a kink (Reinhart and Rogoff, chapter 23).** The regression
function has a threshold parameter, so the sum of squares is not smooth.
`sp.nls` stops at a residual sum of squares of 3738.2673; Stata stops at
3738.271. The threshold agrees to five digits (43.86059 against 43.86067),
the slopes to three. Both are local stopping points of a search on a
surface with flat stretches; ours is lower. The two smooth examples of the
chapter agree to the digits Stata's own convergence rule leaves.

**The robust test of overidentifying restrictions with `perfect`.** For
one specification Stata prints a score statistic of 5.379. Reordering the
instruments in Stata gives 6.467 and 7.866. Wooldridge's statistic is
built from a subset of the instruments and does not depend on the subset
unless one of them is an exact combination of the fitted regressors, which
is the situation `perfect` allows (experience is age minus education minus
six, and age is an instrument). StatsPAI returns 7.866, which is also
Hansen's J at the 2SLS residuals and what Stata gives for the order that
avoids the degenerate instrument. Without the exact dependence the two
programs agree to nine digits and Stata's number does not move with the
order (checked: 6.37301854 both ways).

**Mallows weights of the chapter 28 program.** On the nine wage
regressions of `figure28_5.R` the information criteria, the
cross-validation criterion, the smoothed weights and the jackknife weights
agree with the program's output to every printed digit. The Mallows
weights do not (ours put 0.58 on model 5, the program's 0.57 on model 9).
The cause is in the program: its averaging block comes after a section
that reloads the data for another subsample and reassigns `n`, so the
error variance of the Mallows penalty is divided by the wrong sample size.
With the same lines moved above that section, R's `solve.QP` returns our
weights, and our weights give the lower value of the criterion as the
program defines it (383.481 against 383.951).

**Standard errors of iterative factor methods.** Stata's `factor, ml` and
`factor, ipf` stop at a loose tolerance. With `ltolerance(1e-14)` Stata
prints our loadings to seven digits and our log likelihood to nine.

**Jackknife standard errors agree to seven digits, not twelve.** Stata
keeps the leave-one-out values in single precision unless `double` is
given.

**Bootstrap standard errors** come from numpy's random numbers here and
from Stata's in the log. The replay marks them `random` and compares the
point estimate only.

## Declined, with the reason

| Command | Reason |
| --- | --- |
| `mata { ... }` (chapter 8) | Not translated. The block computes the efficient minimum distance estimator, which is `sp.cnsreg(method='emd')`; its numbers are reproduced by a direct call |
| `xthtaylor` (chapter 17) | No Hausman-Taylor estimator yet |
| `xtdpd` (chapter 17) | `sp.xtabond` / `sp.xtdpdsys` exist; the `dgmmiv()` / `lgmmiv()` grammar is not translated |
| `cmset`, `cmclogit`, `nlogit`, `cmmprobit`, `cmmixlogit` and their `margins` (chapter 26) | Not translated. `sp.clogit` called directly reproduces `cmclogit` to the last printed digit. There is no multinomial probit |
| `estat bootstrap, all` | Percentile and BCa intervals of the last bootstrap; the draws are not Stata's |
| `matrix list e(Sigma)` | Display only |
| `reg ... i.qob#i.yob` with `testparm` | A product of factors whose main effects are absent is coded by Stata as one indicator per cell; the formula language codes it differently. Refused on purpose, as before |
| `disp c, pc` | Two expressions in one `display` |

## Open items

1. Hausman-Taylor (`xthtaylor`). Stata's output for Table 17.2 is in the
   run folder as the reference.
2. Translation of the choice-model commands of chapter 26, and a
   multinomial probit.
3. `xtdpd` grammar.
4. Threshold regression with its non-standard inference (chapter 23) and
   series regression with cross-validated order (chapter 20). The R
   scripts `figure23_3.R` and `figure20_*.R` are written-out reference
   implementations.
5. Rotation of factor loadings (`rotate`).
6. The simulated critical-value tables shipped with the book (`df.log`,
   `eg.log`, `kpss.log`, `Johansen.log`) could be compared with the tables
   behind `sp.unitroot`, `sp.engle_granger` and `sp.johansen`. The KPSS
   and Dickey-Fuller 5% values were read against them and agree to two
   decimals.

## How to rerun

```bash
python tests/external_parity/hansen_econometrics_prepare.py "<folder>"
# in Stata 18, from <folder>/run:  do _master.do     (about 50 minutes;
# chapter 26 with 3,000 integration points takes most of it; kpss from SSC)
STATSPAI_HANSEN_DIR="<folder>/run" \
    pytest tests/external_parity/test_hansen_econometrics_logs.py
```

Chapters 12 and 24 make the replay slow (a jackknife and bootstraps of
instrumental-variable and quantile regressions): about fifteen minutes.
