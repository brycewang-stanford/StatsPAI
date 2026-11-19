# Python Polars edition of *Empirical Research in Accounting*, run through StatsPAI (2026-11-19)

Ian Gow and Martien Lubberink's Python edition
(<https://iangow.github.io/era_pl_book/>) replaces the R edition's
`fixest` / `farr` code with Polars, statsmodels, pyfixest, linearmodels,
rdrobust 2.1.1 (the Python port), causalml 0.18 and the helper package
`era_pl` (github.com/iangow/era_py, `src/era_pl`). The R edition was
covered in `2026-10-06-gow-ding-accounting-review.md`. No new API was
added in this pass.

## Method

The Python code blocks of the book's 29 HTML pages were extracted (926
blocks).
Every chapter whose statistical calls can run without WRDS was run on the
book's packaged data (`era_pl.load_data`, the `era_data` parquet files) or
its own simulations; the WRDS chapters were run on synthetic frames with
the same columns (and, for chapter 19, the real `fhk_pilot` /
`fhk_firm_years` keys). Each book call was run next to the StatsPAI call
on the same frame. Where the two differed, R (fixest 0.14, rdrobust
4.0.0) was run as a third reference. The harness lived in the session
scratchpad and is not committed; the fixes are pinned by
`tests/test_era_pl_book_pass.py`.

| chapter | calls compared | result |
| --- | --- | --- |
| 3 Regression fundamentals | `smf.ols`, interactions, `pf.feols(... \| grade + id)`, the FWL model with `C(fyear) * (...)` on `comp` | exact |
| 4 Causal inference | `smf.ols(subset=)`, boolean regressors and interactions | exact |
| 5 Statistical inference | OLS, HC1, one-way cluster, `PanelOLS` two-way, `FamaMacBeth` (plain and Bartlett bandwidth 1) | exact except the two-way convention below |
| 13 to 16 | `I(abs(ret))`, `I(vol / vol.mean())`, `C(acc_decile) - 1`, Jones model fit on a subset and `predict` on all rows | exact |
| 19 Natural experiments revisited | 24 `pf.feols` specs with `ssc=pf.ssc(G_df="conventional")`, one- and two-way clusters, firm / year FE; by-year `C(year)*pilot - pilot - 1` | exact; by-year two-way differs from pyfixest and equals R fixest (below) |
| 20 Instrumental variables | `pf.feols` IV, statsmodels `IV2SLS`, the book's Sargan n R^2, first-stage F | exact |
| 21 Panel data | `get_test_scores` + `pf.feols` with `iid` and CRV1 on `grade`, `id`, `grade + id`; `era_pl.fit_test_score_panel`; event-time dummies | exact after fix 2 |
| 22 RDD | `rdrobust(..., masspoints="off")` sharp and fuzzy on `iliev_2010` floats | equal to R rdrobust; the Python port differs (below) |
| 23 Beyond OLS | probit / logit / Poisson coefficients, SEs and the book's average marginal effect; Poisson GLM HC1 | exact |
| 24 Extreme values | `.era.winsorize` / `.era.truncate`, `pf.feols` raw / winsorized / truncated, Cook's D, `RLM` bisquare, `cmsw_2018` Poisson HC1 and OLS HC1, ITCV | exact after fix 1, except the RLM constant (below) |
| 25 Matching | `pf.feols` with `I(mkt_cap / 1e9)`; causalml `NearestNeighborMatch` without replacement | identical matched set |
| 26 Prediction | scikit-learn classifiers, RUSBoost | not compared (no StatsPAI counterpart) |

Chapters 1, 2, 6 to 12 and 17 are data manipulation, linking or plotting.

## Fixed

1. **⚠️ Text regressors in `sp.poisson`, `sp.nbreg`, `sp.ologit`,
   `sp.oprobit`, `sp.qreg`.** Chapter 24 passes `ff12` as text
   (`pl.col("ff12").cast(pl.Utf8)`). These estimators built the design
   with `data[v].astype(float)` when the formula was plain names, so the
   digit labels became one numeric regressor (`ff12`, coefficient on the
   industry code) and every coefficient changed; letters raised. The
   formula path is now taken for a text / object / category column
   (`core/utils.py::label_columns`), giving statsmodels' coefficients.
   `sp.zinb`, `sp.zip_model`, `sp.hurdle`, `sp.clogit` and the `y=` /
   `x=` entries refuse such a column (`refuse_label_columns`). Other
   column-list estimators with the same `astype(float)` pattern
   (`heckman`, `fracreg`, `advanced_iv`, `mixed_logit`) were not changed
   in this pass; they take explicit column lists and are not reached by
   the book.
2. **`-inf` as a factor level.** Chapter 21 codes never-treated firms'
   event time as `-inf` inside `C(t, Treatment(reference=-1))`.
   `check_estimator_input` and `fixest/_input_checks.py` refused it as an
   infinite value. `_input_validation.py::label_only_columns` now exempts
   a column that a formula reads only as labels (first argument of
   `C()` / `i()` / `factor()`, or a fixed effect after `|`) and that no
   other argument reads as a number.
3. **`sp.regress` silently dropped unknown keywords** on the two-way
   cluster, CR2 / CR3, Conley, jackknife and `dfadjust` paths. Found by
   trying `cluster=["firm", "year"], ssc=...` to reproduce linearmodels'
   `group_debias=False`. Rejected before the dispatch now.

## Second pass (same day): what a full rewrite in `sp.*` would hit

Ian Gow plans to implement the whole book with StatsPAI, so the second pass
looked at what the first one did not exercise: the 35 regression tables,
the RD plots, the simulation loops, Polars dtypes, and every estimator a
rewrite could reach with a text column.

4. **⚠️ The text-regressor bug was wider.** A sweep of every registered
   function that takes `(formula, data)`, plus the column-list entries,
   found the same silent cast in `sp.ivreg`, `sp.iv`, `sp.cox` (now
   expanded, equal to pyfixest and statsmodels `PHReg`) and in `sp.xtnbreg`,
   `sp.hdfe_ols`, `sp.rifreg`, `sp.heckman`, `sp.fracreg`, `sp.betareg`,
   `sp.liml`, `sp.jive`, `sp.lasso_iv`, `sp.survreg`, `sp.mixed`,
   `sp.meglm`, and the `y=` / `x=` entries of logit / probit and tobit (now
   refused). Every `(formula, data)` estimator now either expands such a
   column or refuses it.
5. **`keep=` / `drop=` in `sp.regtable` / `sp.esttab` matched exact names
   only**, so the book's `ETable(..., drop=["fyear", "ppe", ...])` idiom
   hid nothing, without a word. Entries are now exact names, else globs,
   else regular expressions (fixest, pyfixest, maketables, Stata `esttab`),
   and an entry that matches nothing warns.
6. **`sp.rdplot` had no `masspoints=`**; the book passes
   `masspoints="off"`. The bin engine already implemented it; it is now
   passed through. Bins equal R `rdplot` on the chapter's floats.

Checked and fine: Polars `Categorical`, `Enum`, `Int8`, `UInt32`, `Date`
columns and `LazyFrame` input; `sp.bitest` against `stats.binomtest`
(1e-14); per-firm `da ~ part_i` on samples of 3 to 10 rows against the
book's hand-coded t-test; `sp.auc` against `roc_auc_score` with ties.
Speed inside the book's simulation loops: `sp.regress` and `sp.ivreg` as
fast as statsmodels / pyfixest, `sp.feols` 1.3x pyfixest, `sp.probit` 6x
statsmodels (30 ms a fit; 2,000 fits take a minute).

Notes for the rewrite (not bugs):

- A Polars `Categorical` takes its base level from the order of first
  appearance in pyfixest, statsmodels and StatsPAI alike; R sorts levels.
  `era_pl.load_data` casts R factors to `Categorical` without their level
  order (`cmsw_2018$wbtype`, `wbsource`). Use `pl.Enum([...])` or
  `C(x, Treatment(reference=...))` to fix the base.
- Chapter 26's "logit" is scikit-learn's ridge-penalised fit (`C = 1` on
  standardised features); the R edition used `glm`. `sp.logit` is the
  unpenalised MLE; on a rare outcome the two predicted probabilities differ
  in the third decimal.
- `sp.regress` names a boolean regressor `flag` alone and `flag[T.True]`
  inside an interaction (patsy); `sp.feols` names it `flag` and
  `treat:post`, as pyfixest does. Chapter 19's `coef()["treat:post"]`
  works with `sp.feols`.
- `sp.esttab` is deprecated in favour of `sp.regtable`.

## Documented differences (not bugs)

- **Two-way clustering, chapter 5.** linearmodels `PanelOLS(...,
  debiased=True, group_debias=False)` omits G / (G - 1); StatsPAI, fixest
  and Stata include it with G = min(G1, G2): sqrt(10 / 9) = 5.4% on the
  book's 10-year panel. `sp.feols(..., ssc=pf.ssc(G_adj=False))` gives the
  book's number to 1e-4; the rest is n / (n - k) in linearmodels against
  (n - 1) / (n - k) in fixest. `sp.regress` has no switch for it.
- **statsmodels' clustered p-values** with `use_t=False` are normal;
  StatsPAI uses t(G - 1).
- **Non-PSD two-way covariance, chapter 19 by-year figure.** With 13 year
  dummies and clusters on `fyear` and `gvkey` the matrix has 10 negative
  eigenvalues. `sp.feols` and `sp.regress` apply fixest's eigenvalue fix
  (equal to R fixest 0.14 at every printed digit) and warn; pyfixest
  0.70 does not, so its SEs are up to 5x different.
- **Python rdrobust 2.1.1 against R rdrobust 4.0.0.** With ties in the
  running variable and `masspoints="off"` (default `vce="nn"`), the
  Python port's bandwidth differs from R's by about 1e-4 and its
  estimates in the fifth digit. `sp.rdrobust` equals R. Not reported
  upstream yet.
- **RLM's MAD constant.** statsmodels normalises by
  `norm.ppf(0.75) = 0.6744898`; `sp.robreg(method="m")` and R's `rlm` by
  0.6745, so coefficients differ by about 3e-5. `vce="huber"` reproduces
  RLM's default `H1` covariance; the default `vce="robust"` is the
  sandwich and differs by much more.
- **GLM HC1.** statsmodels' GLM `HC1` has no n / (n - k). Chapter 24
  multiplies it in, chapter 23 does not. `sp.poisson(robust="hc1")` and
  `sp.glm(..., robust="hc1")` include it (R `sandwich`, Stata).
- **Quasi-separation in chapter 24.** `mobflag` takes 1 for 13 cases, all
  with zero `firmpenalty`, so its Poisson coefficient is not finite.
  statsmodels returns -27, StatsPAI -22 with a warning and
  `model_info["separated_terms"] == ["mobflag"]`. `wbflag` agrees to 2e-13.
- Naming. A boolean regressor alone is `female`; inside an interaction
  patsy names it `admitted[T.True]`. statsmodels uses `[T.True]` in both
  places. Not changed.

## For Ian Gow

Items worth passing on, if he files StatsPAI issues from his fork:
the rdrobust Python-port gap, the chapter 23 / 24 HC1 inconsistency, the
chapter 19 pyfixest PSD gap, and the unidentified `mobflag` coefficient.
