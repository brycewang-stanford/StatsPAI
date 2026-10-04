# Clarke, *Applied Microeconometrics*: what its Stata chapters showed about StatsPAI

Started 2026-10-04. Worktree `.claude/worktrees/clarke-textbook`, branch
`wt/clarke-textbook`.

## What was examined

The companion site of Damian Clarke's *Applied Microeconometrics*, a Quarto
project that gives every code call-out of the book three times, in R, Stata
and Python. The files sit in
`改进建议-收集整理/Clarke-AppliedMicroeconometrics/`, which is gitignored.
Neither the programs nor the data are redistributed.

| Chapter | Call-outs | Stata commands | Data |
| --- | --- | --- | --- |
| 2 | simulation, an exact p-value, randomisation inference, the bootstrap | `regress`, `summarize`, `bsample`, `_pctile`, `program`, loops | Banerjee et al. (2021) |
| 3 | matching, overlap and balance, weighting | `teffects psmatch`, `teffects ipw`, `psmatch2`, `logit`, `predict` | Dehejia and Wahba, birth weights |
| 4 | the wild cluster bootstrap by hand and with `boottest`, the two-way fixed effects estimator taken apart (Goodman-Bacon, de Chaisemartin and D'Haultfoeuille), event studies, synthetic control and synthetic difference in differences | `regress, cluster()`, `boottest`, `vce(bootstrap)`, `synth`, `sdid`, frames, matrices, loops | Porter and Serra (2020), a three-unit panel built in the chapter, Proposition 99 |
| 9 (Python only) | double machine learning, causal forests | not replayed | Oreopoulos (2011) |

The site is a work in progress and parts of it are dated. Chapter 4 writes
the wild bootstrap as a loop over frames where one command now does it.
Chapter 3 calls `psmatch2` with the outcome inside the covariate list and
no `outcome()`. The synthetic-control block lists `cigsale(1985)` twice and
omits 1975. None of that matters for the purpose here, which is to ask
whether StatsPAI gives Stata's numbers on what the chapters run.

## Method

The blocks were extracted into one do-file per chapter
(`tests/external_parity/clarke_microeconometrics_prepare.py`) and run once
in Stata 18 with a log. Each log was replayed through one `sp.stata`
session by `scripts/stata_log_replay.py`, which compares every number
Stata printed with what StatsPAI returns, to the precision Stata printed
it. Where a difference appeared, Stata was asked for more (the matches of
each treated unit, the propensity scores in full precision, `boottest` with
99,999 replications) until the first point of divergence was located.

Reference numbers for the tests were then produced on a dataset the package
ships (`sp.datasets.nsw_lalonde`), so the tests run without the book.

State before and after, on the three logs. (The first count was taken on a
log in which `quietly { }` hid the data steps of chapters 2 and 4, so part
of the 378 was the replay not seeing the data.)

| | Before | After |
| --- | --- | --- |
| Numbers compared and reproduced | 103 | 663 |
| Numbers different | 21 | 21, one command, documented below |
| Numbers with no counterpart | 0 | 0 |
| Commands `sp.stata` refused | 378 | 107, listed below |
| Numbers on simulated or resampled data (not comparable) | 11 | 20 |

## What was wrong, and is fixed

### 1. Units with the same covariates did not tie in propensity-score matching

`teffects psmatch (re78) (treat age education black hispanic married
nodegree), atet` on the 445 observations of Dehejia and Wahba gives
2686.832 (691.0373). StatsPAI gave 2698.270 (691.2444).

The matches Stata lists (`generate()`) were compared with ours row by row.
Seven treated units differed, always the same way. Stata matched a treated
unit to two or three controls and StatsPAI to one of them. The controls in
question have the same age, education and indicators as each other, so
their propensity scores are equal. Ours were not. They differed by 5.6e-17
or 1.1e-16, one unit in the last place, because a matrix product rounds
each row on its own and the result depends on where the row falls in the
vectorised loop. Ties are decided by equality of the score, so that
rounding decided which of several identical controls was kept. It also
made the estimate depend on the order of the rows.

This is our bug, not a convention. The index is now evaluated once per
distinct covariate row (`matching/match.py::_index_by_row`), so identical
units get identical scores and no tolerance is needed. Two alternatives
were tried and rejected. Holding the score in single precision, which is
what Stata's `predict` returns, reproduces this sample but moves the
standard error on a sample of 16,437. An absolute tolerance on the
distance reproduces both at 1e-12 and breaks the larger one at 1e-10,
which is a tolerance tuned to two datasets.

After the fix, to every digit Stata prints.

| Sample | Model | Stata | StatsPAI before | StatsPAI now |
| --- | --- | --- | --- | --- |
| DW, 445 | logit, ATET | 2686.832 (691.0373) | 2698.270 (691.2444) | 2686.832 (691.0373) |
| DW, 445 | probit, ATET | 2375.279 (672.2979) | not available | 2375.279 (672.2979) |
| DW with PSID, 16,437 | logit, ATET | 1477.592 (789.5247) | 1477.592 (789.5207) | 1477.592 (789.5231) |
| DW with PSID, 16,437 | probit, ATET | 1823.478 (723.5223) | not available | 1823.478 (723.5223) |
| Lalonde, 614, discrete covariates | logit, ATET | 1209.718 (1261.376) | 1199.666 (1266.607) | 1209.718 (1261.376) |

The third row's standard error is 2e-6 away. The regressors there are
earnings in dollars and Stata's own `teffects ipw` reports "convergence not
achieved" on them, so the two logits stop at slightly different places.

### 2. `sp.twfe_decomposition` did not decompose the coefficient

Call-out 4.2 builds a panel of three units over ten years with no noise and
works out every piece of the two decompositions by hand. On it
`sp.bacon_decomposition` returns the book's numbers exactly: the
coefficient 27/11 and the weights 7/22, 8/22, 3/22 and 4/22.
`sp.twfe_decomposition`, which names the same theorem, returned 1.839 with
four weights of 0.25.

The source said why. Weights were "Simplified: proportional to n_units",
the timing comparisons used all periods instead of the window in which each
is a clean 2×2, the dCDH weights came from a "Simplified formula" that did
not sum to one, and the standard error was the spread of the rows. The
defect had been found before, against R `bacondecomp` and `twowayfeweights`
on `mpdta`, and pinned as a strict `xfail` with a note that the fix
belonged to another line of work
(`docs/dev/campaign_phase3/did_synth.md`, D4). Nobody had picked it up.

The function now takes its rows from `sp.bacon_decomposition`, computes the
dCDH weights `w = D eps / sum(D eps)` on unit × period cells, reports the
TWFE coefficient as its estimate and the unit-clustered standard error of
that coefficient. On the book's panel every number is the book's
(`tests/reference_parity/test_twfe_decomposition_known_truth.py`), the
standard error equals `xtreg, fe vce(cluster)` to 1e-10, and the `mpdta`
comparison passes, including the 125 negative weights. On an unbalanced
panel, where the theorem does not hold, the coefficient and the weights are
returned and the 2×2 table is left empty with a warning.

### 3. Three translations ran something other than the Stata command

- `teffects ipw` became `sp.ipw` with its default, a bootstrap standard
  error. `sp.ipw(se_method='sandwich')` is the stacked M-estimation
  variance Stata reports and matches it to seven digits. The translation
  now asks for it.
- `teffects aipw` became `sp.aipw` with its default, five-fold
  cross-fitting, which is another estimator and a random one. The
  translation now runs `cross_fit=False, se_method='sandwich'`, which
  matches Stata (316.2065 (858.4882) on the Lalonde data). `teffects aipw`
  has no `atet`; the translation used to pass `estimand='ATT'` and now
  refuses.
- `boottest` became a call of `sp.wild_cluster_bootstrap` with a fitted
  result and a `hypothesis=` argument. That function takes neither, so
  every `boottest` raised a `TypeError`. It now runs `sp.wild_cluster_boot`
  on the regression before it.

## What was missing, and is added

**A probit propensity score.** The book's `teffects ipw (..., probit)` was
refused, and `psmatch2`, whose default in Stata is a probit, was run as a
logit with a note. `sp.ipw`, `sp.match` and `sp.psmatch2` now take
`ps_model='probit'`. For `sp.ipw` the sandwich variance uses the probit's
generalised residual and observed Hessian. For `sp.match` the
Abadie-Imbens (2016) correction uses the normal density at the index where
the logit has `p(1-p)`. The book's own number, 1177.529 (616.7019), is
reproduced, and so are Stata's on the bundled data
(`tests/reference_parity/test_teffects_probit_stata.py`, seven digits for
`teffects`, 1e-7 for `psmatch2`).

**`boottest` options.** `weight(rademacher | webb | mammen)`, `reps()`,
`seed()`, `level()`, a null value (`boottest x = c`, through the new
`sp.wild_cluster_boot(h0=)`), and `bootcluster()` when it names the error
cluster. On the book's regression (12 clusters) the Rademacher draws are
enumerated on both sides and the p-value is 0.09179688 on both. With Webb
weights it is a random number: 0.0896 from Stata at 99,999 replications,
0.088 to 0.094 from StatsPAI over three seeds at 9,999.

**Do-file constructs.** A do-file from any applied paper uses these, and
`sp.stata` refused each of them.

| Construct | Now |
| --- | --- |
| `quietly { ... }`, `capture { ... }`, `noisily { ... }` | the body is run |
| `capture cmd` | an error Stata would raise is swallowed (a missing variable, `restore` without `preserve`); a command that is not translated still stops |
| `by g:` / `bysort g (t):` before `generate` / `replace` | run by group, with `_n`, `_N` and `x[_n-1]` counted within the group; plain `by` needs sorted data, as in Stata |
| `count [if] [in]` | `r(N)` |
| `_pctile x, percentiles()`; `summarize, detail` | `r(r1)` ...; `r(p1)` ... `r(p99)`, `r(skewness)`, `r(kurtosis)`, with Stata's percentile definition |
| `scalar(name)` in an expression; `_b["x"]`; `scalar` with no data in memory | evaluated |
| `duplicates drop [varlist, force]` | run |
| `bsample [, cluster()]` | run with numpy's draws; the data are marked random, so later numbers are not compared |
| `predict` after a regression with `i.` variables | indicators and their products are rebuilt from the coefficient names |
| `psmatch2` | leaves `_pscore`, `_treated`, `_support`, `_weight`, `_id`, `_n1`, `_nn`, `_pdif` in the data |

Each was written from the Stata manual and tested on synthetic commands
(`tests/test_stata_session_constructs.py`).

## What agrees and needed nothing

- **Synthetic difference in differences** on Proposition 99. `sdid` with
  `method(sdid)`, `method(sc)` and `method(did)` gives −15.60383,
  −19.61966 and −27.34911, and so does `sp.sdid`, to the last digit. The
  placebo standard errors are random on both sides.
- **`sp.bacon_decomposition`** on the three-unit panel (see above).
- **Every regression** of chapters 2 to 4, with clustered and robust
  standard errors, and the hand-built 2×2 comparisons.
- **`sp.wild_cluster_bootstrap`**, the t statistic and the enumerated
  p-value.

## Differences that remain, with their reasons

**`synth`, 21 numbers.** The chapter's call uses the regression-based
predictor weights (no `nested`) with one predictor listed twice. Stata
reports a pre-treatment RMSPE of 1.657121. StatsPAI's solution has 1.657031,
lower, with the weight of Utah at 0.393 for Stata's 0.396. The predictor
weights are not unique when two predictors are the same variable, and the
quadratic programme is solved to different tolerances. The same class of
difference as chapter 18 of Chen Qiang's textbook; no change.

**The confidence set of `boottest`.** Stata prints the set of null values
not rejected at 5%. `sp.wild_cluster_ci_inv` computes the same set and
`sp.wild_cluster_boot` reports the percentile-t interval instead, which the
translation says. On the enumerated case the two sets are
[−0.017439, 0.162265] (Stata) and [−0.017350, 0.161589]. The bootstrap
p-value is a step function of the null value with steps of 2/4096, and the
p-value at each of the four endpoints is within two steps of 0.05, so both
are a valid reading of where the function crosses. The difference is how
each programme interpolates across a step.


## What `sp.stata` still declines on these logs

| Commands | Count | Why |
| --- | --- | --- |
| `frame`, `frlink`, `frget` | 49 | Stata 16 frames. The chapter uses them to hand-write a bootstrap that `boottest` does in one line. |
| loops (`forvalues`, `foreach`), an `if` block, the `program` the chapter defines and its two calls | 15 | Declined by design: write the loop in Python around the call. |
| `svmat`, `tempfile`, `save` | 8 | Matrices are not held and files are not written. |
| lines that depend on one of the above (a variable a loop or a matrix would have made) | 33 | They follow from the rows above. |
| `reg ..., vce(bootstrap, reps() cluster())` | 1 | `sp.regress` has no bootstrap variance option; `sp.bootstrap` is the generic tool. The numbers would be random in any case. |
| a comment the log wraps over two lines | 1 | Not a command. |

All but the bootstrap variance were closed in the third round, below.

## Second round: the three items the first round left open

**`egen`.** Not used by the book, and the most common remaining gap in
other do-files. `sp.stata` now runs `egen [type] newvar = fcn(...) [if]
[in] [, by()]`, also behind `by g:` / `bysort g:`, for `count`, `mean`,
`median`, `sd`, `min`, `max`, `total`, `pctile`, `iqr`, `std`, `group`,
`tag`, `rowtotal`, `rowmean`, `rowmin`, `rowmax`, `rowsd`, `rownonmiss` and
`rowmiss` (`_stata_egen.py`). Each follows the missing-value rule its entry
in `[D] egen` states: `total` counts missing as zero, `count` of nothing is
0, `tag` is never missing, a missing value of a `by()` variable is a group
of its own. Thirty commands were run beside Stata 18 on 614 rows with
missing values planted in them. Twenty-seven agree to 1e-15 on every row.
The other three depend on the order of rows within a group, which Stata's
sort does not fix; `tag` marks one row per group on both sides.

**The confidence set of `boottest`.** `sp.wild_cluster_boot(confidence_set=True)`
returns `ci_inverted`, the null values the test does not reject, found by
bracketing each side on a grid and bisecting to the jump of the p-value.
The translated `boottest` asks for it unless the line says `noci`. The
difference from Stata's endpoints described above remains and is stated in
the docstring: about 0.2% and 1.9% of a standard error on the book's
regression.

**The ATE standard error of `teffects psmatch`.** `sp.match(estimand='ATE',
se_method='abadie_imbens_2016')` now reports it, from the formulas of
"PSM, ATE, and ATET variance adjustment" in `[CAUSAL] teffects nnmatch`.
One thing in that section is not what Stata computes. The manual prints
the adjusted variance as the base variance *plus* `c'Vc`. Stata's `e(V)`
is the base variance *minus* `c'Vc`, which is also what Abadie and Imbens
(2016) prove: estimating the score can only lower the variance of the ATE.
With the plus sign the five reference fits are off by 1% to 5%. With the
minus sign all five agree to the seven digits `teffects` prints (logit and
probit, 445 and 614 observations).

| Fit | Stata | plus | minus |
| --- | --- | --- | --- |
| DW 445, probit | 651.6872 | 663.2612 | 651.6872 |
| DW 445, logit | 662.6717 | 675.0007 | 662.6717 |
| Lalonde, discrete covariates, logit | 973.734 | 990.9515 | 973.7340 |
| Lalonde, all covariates, logit | 1076.527 | 1125.7525 | 1076.5271 |
| Lalonde, all covariates, probit | 1029.85 | 1090.8848 | 1029.8499 |

`teffects psmatch, ate` is translated with it. With a caliper the ATE
variance is not implemented and the translation says so.

## Third round: what the first two declined

The first round left 107 commands of the three logs unrun, and the second
closed none of them. They were of five kinds: loops and the program the
book defines, frames, matrices, temporary files, and lines that needed one
of those to have run. All five are now run. One command is left.

| | Round 1 | Round 3 |
| --- | --- | --- |
| Numbers compared and reproduced | 663 | 663 |
| Numbers different | 21 (`synth`) | 21 (`synth`) |
| Commands `sp.stata` refused | 107 | 1 |
| Numbers on simulated or resampled data (not comparable) | 20 | 153 |

The 133 numbers that moved to "not comparable" are the ones the book
computes on bootstrap and permutation draws. They are now computed, on
numpy's draws instead of Stata's. Where the book's hand-written wild
cluster bootstrap ends in a p-value, Stata's run gave 0.084, `boottest`
0.0951, and the replay about 0.10.

**Loops, branches, macros, programs** (`_stata_flow.py`). `forvalues`,
`foreach`, `while`, `if { } else { }` and `continue` are run as Stata runs
them: the body is collected up to its closing brace, the loop macro is
set, and the lines go back to the session one at a time. A command inside
a loop is therefore translated and refused by the same rules as outside
one. The macros a loop needs are evaluated (`local x = exp`, `local ++i`,
`` `=exp' ``, `` `r(N)' ``, `tempvar`, `tempfile`), and a program can take
arguments. This reverses a rule the translator had ("loops are refused").
The reason for the rule was that a loop cannot be translated into one
`sp.*` call. It does not have to be: it can be run. What stays refused is
what cannot be read off the text, namely extended macro functions,
`syntax` and `mata`.

**More than one dataset** (`_stata_multi.py`). `save` keeps a copy in the
session and `use`, `append` and `merge ... using` find it there or in
`sp.stata(..., files={...})`. No file is read or written. `merge` (1:1,
m:1, 1:m, on `_n`), `append` and the frame commands the
chapter uses (`frame put`, `frame change`, `frlink`, `frget`, and `frame
create` / `post` / `copy` / `drop`) were each run beside Stata 18 on small
frames. Two things had to be learned from Stata's output. After a merge
the master's rows are sorted by the key and the rows found only in the
using data come last. And a variable present on both sides keeps the
master's value on the master's rows but has the using data's value on a
using-only row. `frget ag, from(cluster)` works when the link variable is
`clusters`, since Stata accepts an unambiguous abbreviation.

**Matrices** (`_stata_matrix.py`). A matrix as a table of results: `J()`,
`e(b)` (constant last, as Stata stores it), `e(V)`, a cell at a time,
`A[i, j]` in an expression, `svmat`, `mkmat`, and matrix expressions with
sums, products, a transpose and `inv()` (a Stock and Watson log computes the
variance of a linear combination as `d * V * d'`, and the result equals
`lincom`'s). Other matrix functions are refused. A
matrix filled from resampled data is marked, so that the numbers computed
after `svmat` are reported as random and not as differences.

**The rest of `egen`.** `skew`, `kurt`, `mad`, `mdev`, `mode`, `rank`,
`seq`, `anycount`, `anymatch`, `cut` with `at()`, `rowfirst`, `rowlast`.
Nineteen commands beside Stata 18, all equal on every row.

**The ATE of `teffects psmatch` with a caliper.** In Stata a caliper is a
condition and not a change of estimator. Either every unit has `m`
opposite-arm matches and `h` other same-arm units within it, and the
estimate and standard error are the ones without a caliper, or the command
stops. `sp.match` does the same: at `caliper(0.3)` it returns 320.1955
(973.734), at `caliper(0.05)` it stops because one observation (number 58,
the one Stata names) has a single same-arm unit in range, at
`caliper(0.02)` it stops for want of matches. `nneighbor(3)` and
`vce(robust, nn(3))` were added to the reference fits (120.075 (820.524)
and 320.1955 (1226.906)) and agree.

**The log replay** reads a block as one command. Stata echoes the body of
a loop as numbered lines and the replay hands them to the session. A body
that ran quietly is not in the log at all, so it is taken from the do-file
next to the log. The session now keeps its macros, scalars, matrices and
saved datasets when a new dataset is loaded, as Stata does.

### Chapter 9, which exists only in Python

The site's last chapter has two examples and no Stata or R output.

*Double machine learning on the 401(k) data*, with `DoubleMLPLR` and random
forests in three folds. The book's call gives 8884 (1308). `sp.dml` with
the same learners gives 8915 (1312), 8675 (1304) and 9024 (1305) for three
fold assignments. That spread is the fold assignment, and the comparison
at fixed folds is `tests/external_parity/test_dml_python_parity.py`.

*A causal forest on a correspondence study* (Oreopoulos 2011: résumés with
randomly assigned names, 10,184 of them). This is where the chapter is
dated. It fits `econml`'s `CausalForestDML`, predicts the effect for every
row it trained on, and reports the mean of those predictions as "the ATE",
0.0645, with no standard error. Names were randomised, so the difference
in callback rates is an unbiased benchmark: 0.0563 (0.0072).
`sp.causal_forest` reports the doubly robust average of out-of-bag scores,
0.0575 (0.0071), which is the benchmark to within a fifth of a standard
error. The two forests agree on the direction of the heterogeneity the
chapter looks at (a larger effect for the lower-quality degree: 0.073
against 0.043 here, 0.085 against 0.047 in `econml`).

One thing turned up on the way. `sp.causal_forest` took `d=` and `x=` for
the column names and refused the names the rest of the package uses. It
now also accepts `treat=` and `covariates=`.

## What is still declined, and why

- **`regress ..., vce(bootstrap, reps() cluster())`**, the one command
  left. `sp.regress` has no bootstrap variance, and giving it one is a new
  option on the package's central estimator, with its own validation
  entries, for a standard error that is random on both sides.
  `sp.bootstrap` resamples any statistic, by cluster if asked.
- Extended macro functions, `syntax`, `mata`, matrix functions beyond
  `inv` / `diag` / `trace`, `merge m:m`,
  `frame name { }` blocks, `egen` with `fill`, `cut(, group())`,
  `rank(, unique)` and the string functions. Each is refused with its
  reason.
- String variables are not generated (`gen s = "a"`), so `string()`,
  `substr()` and their relatives are not implemented.

## How to run it again

```bash
python tests/external_parity/clarke_microeconometrics_prepare.py <Quarto project>
# in Stata 18, from <Quarto project>/run:  do _master.do
python scripts/stata_log_replay.py <Quarto project>/run --data <Quarto project>/run
STATSPAI_CLARKE_DIR=<Quarto project>/run \
    pytest tests/external_parity/test_clarke_microeconometrics_logs.py
```
