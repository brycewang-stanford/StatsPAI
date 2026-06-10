# Migrating between StatsPAI versions + from PyStataR

Internal version-to-version migrations are at the top; the long-form
`PyStataR → StatsPAI` migration follows below.

---

<a id="evalue-hr-ci-parity"></a>

## Unreleased — ⚠️ `sp.evalue` HR / CI E-value parity with R `EValue`

**What changed.** Two numerical behaviours of `sp.evalue` (and
`sp.evalue_from_result`) changed so that StatsPAI now reproduces the R
`EValue` package exactly (#21):

1. **Hazard ratios.** `measure='HR'` was always treated as a *rare-outcome*
   ratio (`OR ≈ RR ≈ HR`). It now uses the exact common-outcome conversion
   `(1 − 0.5^√HR)/(1 − 0.5^√(1/HR))` by default, matching `EValue::evalues.HR`.
   HR E-values change for non-rare outcomes.
2. **Confidence intervals that cross the null.** The CI E-value is now exactly
   `1` whenever the interval already contains the null (or a user-supplied
   `true` value), instead of a spurious value > 1 computed from the limit.

**How to get the old numbers.** Pass `rare=True` for the rare-HR
approximation. There is **no** flag to restore the un-clamped CI E-value — the
old value was incorrect (it claimed "confounding needed" for a result already
compatible with the null).

**Parameter rename.** `rare_outcome` → `rare`. The old name still works (emits
`DeprecationWarning`) and will be removed no earlier than the next minor.

**Who is affected.** Anyone computing an E-value from a hazard ratio with a
non-rare outcome, or reading the CI E-value of a non-significant result. RR-
and OR-based point E-values are unchanged.

**JOSS / JSS.** This is parity module `23_evalue` in the JSS cross-language
table (`Paper-JSS/manuscript/tables/appendix_b_parity.tex`); the change
*increases* agreement with R `EValue` and the row remains a machine-precision
**PASS** (worst relative difference 5.8e-14 over 26 rows). No JOSS (#10604)
numeric figure uses an HR or CI E-value.

---

<a id="matching-nearest-tie-break"></a>

## Unreleased — ⚠️ `sp.match` nearest-neighbor tie-breaking stabilised

**What changed.** `sp.match(method='nearest')` now resolves exact equal-distance
nearest-neighbor ties by the source DataFrame index. Previously the
Euclidean/propensity nearest-neighbor path delegated tie selection to
`argpartition` and incidental row order, so ties on discrete or binary
covariates could move the ATT across environments. Lower-index control units
are now selected first; when matching without replacement and multiple treated
units have the same best distance, lower-index treated units are assigned first.

**Who is affected.** Only users whose matching data contain exact
equal-distance ties. Continuous covariates without exact ties are unchanged.
For tied designs, results are now deterministic across row order and backend as
long as the DataFrame index preserves unit identity. One caveat: distances that
are merely *near*-equal (differing at the ~1e-13 ULP level because the BLAS
build computes propensity scores slightly differently) are still resolved by
strict comparison, so a residual backend sensitivity of that magnitude remains
— on LaLonde it amounts to ~$4.5 (vs. ~$150 before the fix), and all GitHub CI
platforms (ubuntu/windows/macos) now agree bitwise.

**Action required.** None for code. If you previously recorded a nearest-match
estimate on tied discrete covariates, re-run it once and treat the new value as
the stable pin. The bundled LaLonde 1:1 NN PSM guard now pins the two observed
fixed points exactly (`1967.94` on GitHub CI, `1963.43` under Accelerate on
macOS 26) instead of allowing the old ~$300 cross-backend tie band.

---

<a id="blp-maxiter-fix"></a>

## Unreleased — ⚠️ `sp.blp` functionality fix (was non-functional)

**What changed.** `sp.blp` (BLP random-coefficients logit demand) now runs.
Previously its GMM inner loop called `_gmm_objective(..., maxiter=1000)` while
the parameter is named `maxiter_inner`, so **every** `sp.blp` call raised
`TypeError: _gmm_objective() got an unexpected keyword argument 'maxiter'` as
soon as the outer optimiser evaluated the objective — i.e. on every estimation
path (`contraction`, `mpec`, `gmm`).

**Who is affected.** Anyone who tried to call `sp.blp`. Because the function
produced *no* output before (it crashed), this fix cannot move any
previously-correct number — JOSS (#10604) / JSS dossier figures are unaffected.
`sp.blp` (and `BLPResult`) appears in the JSS manuscript only as a
function-inventory catalog row (`function_inventory_full.tex`), never in any
numeric or parity table, and the fix does not change that row. **Note the name
collision:** the "BLP" entries in the JSS parity-change log (`05-parity.tex`,
`05-parity-compact.tex`) refer to the **Best Linear Projection of CATE**
(`best_linear_projection` / `blp_test` / `test_calibration`,
Chernozhukov-Demirer et al.) — a different feature, not this
Berry-Levinsohn-Pakes demand estimator. A regression guard now recovers the
known linear price/characteristic coefficients on a pure-logit DGP
(`tests/test_tierD_structural_analytic.py::TestBLPAnalytic`).

**Action required.** None, beyond noting that `sp.blp` is now usable. Found by
the Tier D analytic special-case test campaign (CLAUDE.md §5).

<a id="dag-dseparation-fix"></a>

## Unreleased — ⚠️ d-separation corrected (forks & colliders)

**What changed.** `statspai.dag`'s d-separation engine (`_d_separated`, behind
`DAG.d_separated`, `adjustment_sets`, `backdoor_paths`, `do_rule1/2/3`,
`do_calculus_apply`, `swig`, `dag_recommend_estimator`) moralised the ancestral
graph incorrectly — it married *siblings* instead of *co-parents*. So the two
non-trivial d-separation rules were backwards: conditioning on a common cause
did **not** block a fork (`A ⊥ C | M` on `M→A, M→C` wrongly returned `False`),
and conditioning on a collider did **not** open it (`A ⊥ C | K` on `A→K←C`
wrongly returned `True`). Chains were unaffected. Moralisation now connects
every pair of a node's parents, so all three canonical structures and the
adjustment-set / do-calculus routines built on them are correct.

**Who is affected.** Anyone who used `DAG.d_separated`, `adjustment_sets`,
`backdoor_paths`, the do-calculus rule checkers, `swig`, or
`dag_recommend_estimator`. **Re-derive any adjustment sets / identification
conclusions** obtained from these — previous fork/collider answers were
unreliable. No API change. None of the package's reference-parity or
JOSS/JSS-dossier numbers come from these graph routines, and all nine
dag-touching test files pass unchanged (none had encoded the broken behaviour).

**Action required.** None code-wise; re-check any DAG-derived adjustment sets.
Found by the Tier D analytic special-case campaign (CLAUDE.md §5); guard:
`tests/test_tierD_p2_dag_dsep_analytic.py`.

<a id="granger-wald-variance-fix"></a>

## Unreleased — ⚠️ `sp.granger_causality` test statistic corrected

**What changed.** `sp.granger_causality` now computes the correct Wald
statistic. The coefficient covariance used in the test was a placeholder
`V = sigma2 * I` (the caused equation's residual variance, not its coefficient
covariance), which omitted the design-matrix factor `(X'X)⁻¹`. The reported
F-statistic was too small by a factor of roughly `T·Var(regressors)`, so the
test essentially never rejected — even a textbook-strong lagged link went
undetected (true F≈326 reported as ≈0.36). `VARResult` now stores `(X'X)⁻¹` and
the test forms `Var(β̂_caused) = σ²_caused·(X'X)⁻¹`; the F now equals the
standard restricted-vs-unrestricted OLS F-test.

**Who is affected.** Anyone who called `sp.granger_causality` (directly or via
`VARResult.granger_test`). **Re-run any Granger conclusions** — prior runs
almost certainly failed to reject and were not trustworthy. There is no API
change. No JOSS (#10604) / JSS table uses this function, and the previous output
was statistically meaningless, so no valid published result is invalidated.

**Action required.** None code-wise; re-run Granger tests and expect them to
detect real causal directions now. Found by the Tier D analytic special-case
campaign (CLAUDE.md §5); guard: `tests/test_tierD_p2_timeseries_analytic.py`.

<a id="ols-qr-kernel"></a>

## Unreleased — ⚠️ OLS kernel switched to a QR solve (numerical accuracy)

**What changed.** The core OLS kernel — `ols_fit` for coefficients and
`OLSEstimator.estimate` for the variance-covariance matrix, both in
`src/statspai/` — now solves least squares via the **QR factorisation** of the
design matrix `X = QR` (`b = R⁻¹Qᵀy`, `(X'X)⁻¹ = R⁻¹R⁻ᵀ`). The previous
implementation solved the normal equations `(X'X) b = X'y` and formed
`inv(X'X)` directly. Forming `X'X` squares the condition number of `X`, so on
ill-conditioned designs roughly half of the available digits are lost — and on
the worst cases the result is meaningless.

**Why.** The new NIST StRD certification suite
(`tests/numerical_accuracy/test_nist_strd_ols.py`) showed the normal-equations
path produced **0 correct digits** on the NIST Filippelli dataset (a degree-10
polynomial fit, `cond(X) ≈ 1e10`) and only ~6 digits on several Wampler
polynomials. The QR path tracks `cond(X)` rather than `cond(X)²` and lifts
those to ~7 and ~9–13 digits respectively, matching the published certified
values.

**Who is affected.**

- **Well-conditioned regressions (the overwhelming majority): no action.**
  Coefficients, standard errors, R², F all match the old output to ≈1e-12 —
  far below any reporting precision. The full `reference_parity` and
  `external_parity` (JOSS reproduction) suites pass unchanged.
- **Regressions on near-collinear or high-degree-polynomial designs:** you will
  now get **different — and correct — numbers**. If you previously fit, say, a
  high-order polynomial trend or a strongly collinear specification directly
  (without centring/orthogonalising) and recorded the coefficients, re-run and
  expect them to move. The old numbers were the unstable ones.

There is **no API change** and nothing to rewrite; this note exists only so the
numerical shift on ill-conditioned designs is on the record.

Separately, exact-fit OLS (`R² == 1`) now reports the F-statistic as `inf`
(matching NIST's certified "Infinity") instead of emitting a divide-by-zero
`RuntimeWarning`; non-exact fits are unaffected.

Separately, `sp.regress` now fits intercept models in mean-centred
(Frisch-Waugh-Lovell) coordinates and reconstructs the intercept afterward.
This is algebraically identical to the raw fit in exact arithmetic, but it
avoids catastrophic cancellation when `y` or a regressor has a very large
constant offset. Well-conditioned fits remain unchanged to machine precision;
the visible effect is on pathological offset designs such as the NIST StRD
ANOVA `SmLs07/08/09` cases, where F/R² accuracy improves down to the float64
input-representation floor.

---

<a id="regress-collinearity-guard"></a>

## Unreleased — ⚠️ `sp.regress` raises on perfect collinearity; `sp.logit`/`sp.probit` warn on separation

**What changed.** Two silent-failure corners now fail loudly:

- **Perfect collinearity** in `sp.regress` — duplicate or proportional
  regressors, the dummy-variable trap (complementary 0/1 dummies plus an
  intercept), or a constant non-intercept regressor — previously returned
  enormous unidentified coefficients (e.g. `~1e14`) with no warning. It now
  raises `statspai.exceptions.NumericalInstability`, with the offending columns
  in `error.diagnostics`.
- **Perfect / quasi-complete separation** in `sp.logit` / `sp.probit` —
  where the outcome is perfectly predicted and the maximum-likelihood estimate
  does not exist — previously returned large finite coefficients with no
  signal. It now emits a `statspai.exceptions.ConvergenceWarning`.

**Why.** The project rule is "fail loudly": returning wrong numbers silently is
the cheapest way to hide a correctness problem. Neither case has a meaningful
answer to return.

**What you need to do.**

- If you *intended* collinear regressors, drop one (or remove the intercept for
  a single constant regressor). The exception names them.
- For separation, use penalized (Firth) logistic regression, drop the
  separating predictor, or pool sparse categories.

**Scope / non-goals.** Collinearity detection is deliberately *structural*
(duplicate/proportional columns, zero-variance regressors), not based on the
condition number or matrix rank. A rank tolerance loose enough to catch real
collinearity also flags legitimately ill-conditioned but full-rank designs —
the NIST StRD Filippelli benchmark is numerically *more* singular
(`s_min/s_max ~ 6e-16`) than an exactly duplicated column yet must fit. So a
general exact linear dependence among 3+ columns that is not reducible to a
pairwise duplicate or a constant column is **not** auto-detected; inspect the
design's condition number if you suspect one.

---

<a id="drdid-traditional-normalisation"></a>

## 1.17.0 — ⚠️ `sp.drdid(method='trad')` ATT correctness fix

**What changed.** The traditional doubly-robust DiD branch of `sp.drdid`
(Sant'Anna & Zhao 2020) divided each of its four cell terms — treated/control ×
post/pre, each a weighted average of the outcome-regression residual — by the
**full sample size** `n` rather than by that cell's weight mass. On a balanced
2×2 each cell holds ~¼ of the sample, so every term was scaled down by its
sample share and the ATT was biased toward zero by ~50%. Concretely, on a 2×2
with true ATT 2.0 (raw DiD 1.96) `method='trad'` returned ≈1.04. Each term is
now normalised by its own weight total. The traditional estimator therefore now
reduces **exactly** to the raw 2×2 DiD when no covariates are supplied, and
recovers the true ATT with covariates.

Separately, `sp.drdid` now **raises `ValueError`** when `method` is neither
`'imp'` nor `'trad'`. Previously any other string (e.g. `'ipw'`, `'reg'`,
`'dr'`) fell through silently to the traditional branch; such calls were never
distinct estimators and now fail loudly.

**Who is affected.** Callers of `sp.drdid(..., method='trad')` (or any non-`imp`
string, which silently ran the traditional branch). The **default**
`method='imp'` (improved, locally efficient) already normalised correctly and
is **unchanged** — its point estimates, standard errors, the
source-audit / R-`DRDID` parity numbers (which pin
`drdid_imp_panel`), and every other default-path result are **not** affected.

**Action.** If you relied on `method='trad'` output, re-run; the corrected ATT
is ~2× the previously reported (downward-biased) value and now matches the raw
DiD / `method='imp'`. Replace any `method` value other than `'imp'`/`'trad'`
with one of those two.

---

<a id="multiway-cluster-intersection"></a>

## 1.17.0 — ⚠️ `sp.multiway_cluster_vcov` multiway-cluster SE correctness fix

**What changed.** `sp.multiway_cluster_vcov` forms the Cameron-Gelbach-Miller
(2011) variance by inclusion-exclusion over the clustering dimensions, which
requires an *intersection* cluster: the unique combinations of the dimensions'
levels (e.g. the distinct `(firm, year)` pairs). The intersection key was built
by joining the dimensions into one string with a `"\0"` separator, but NumPy
fixed-width unicode strips the embedded NUL byte, so `(1, 23)` and `(12, 3)`
both collapsed to `"123"`. On a 40×50 crossed-cluster DGP this merged 1733 true
intersection clusters into 1639, which inflated the `G/(G-1)` finite-sample
factor on the subtracted intersection term and biased the multiway SE by ~0.2%
(two-way) to ~0.5% (three-way) away from the canonical estimator. The
intersection key is now built collision-free via `np.unique(axis=0)` on
per-dimension integer codes. `sp.multiway_cluster_vcov` now reproduces
`sandwich::vcovCL(cluster = ~ g1 + g2 + ...)` and `sp.twoway_cluster` to machine
precision (two-way exact; three-way relative error ~4e-7).

**Who is affected.** Callers of `sp.multiway_cluster_vcov` with **two or more**
clustering dimensions, and the multiway-clustered standard errors of
`did.harvest` and `panel.feols`. One-way clustering is unaffected.
`sp.twoway_cluster` is **not** affected — it used a separate, collision-free
intersection key and already matched `sandwich::vcovCL` to machine precision.
Point estimates are unchanged; only multiway-cluster SEs/CIs/p-values move
(typically by tenths of a percent, always toward the canonical value).

**Action.** Re-run any analysis that reported `sp.multiway_cluster_vcov`-based
multiway-cluster SEs with ≥2 dimensions; the corrected SEs now agree with
`sandwich::vcovCL` (R) and Stata multiway-cluster conventions.

---

<a id="structural-break-supf-null"></a>

## 1.17.0 — ⚠️ `sp.structural_break` sup-F p-value null distribution correctness fix

**What changed.** The sup-F / Chow statistic in `sp.structural_break(...)` is a
*supremum* of the Chow F statistic over all candidate break points. Under the
null of no break it therefore follows the Andrews (1993) sup-F limiting law,
**not** the ordinary `F(k, n-2k)` distribution. The previous code referred the
maximised statistic to the F CDF (`1 - scipy.stats.f.cdf(best_f, k, n-2k)`),
which ignored the search over break points and produced p-values that were far
too small. Measured false-positive rate on Gaussian white noise at the 5%
level: **33–37%** (n ∈ {100, 200, 400}) — a roughly 7× inflation. P-values are
now drawn from the Andrews (1993) null (a q-vector Brownian-bridge functional,
sampled by a deterministic seeded simulation cached per `(q, grid, trimming)`),
which restores **nominal size (~5%)** with no material loss of power. The same
correct critical value now governs the Bai-Perron sequential `supF(l+1|l)`
stopping rule, so `method='bai-perron'` stops over-segmenting noise.

**Who is affected.** Anyone who relied on the `p_values` / `break_dates` of
`sp.structural_break` with `method` in `{'sup-f', 'chow', 'bai-perron'}`.
Previously-reported breaks (and their tiny p-values) were anti-conservative;
some "significant" breaks were spurious. The point estimate of the *location*
of the most likely break (`break_dates[0]` for sup-F) is unchanged — only its
significance and the break **count** change.

**What to do.** Re-run any structural-break tests and re-check significance
against the corrected p-values. A break that was marginal under the old
(inflated) test may no longer reject. `method='bai-perron'` may now return
fewer breaks. The result object additionally gained populated `f_stats` /
`p_values` for the Bai-Perron path (one entry per detected break, sorted by
date), where it previously returned `None`.

**Reference.** Andrews, D.W.K. (1993). "Tests for Parameter Instability and
Structural Change with Unknown Change Point." *Econometrica*, 61(4), 821-856.
doi:10.2307/2951764 (verified via Crossref, the Econometric Society, and
RePEc).

---

<a id="msm-singleperiod-iptw"></a>

## 1.17.0 — ⚠️ `sp.stabilized_weights` / `sp.msm` single-period IPTW correctness fix

**What changed.** On a **single-period (point-treatment) panel**,
`sp.stabilized_weights(...)` (and therefore `sp.msm(...)`) previously returned
stabilized weights that were all exactly `1.0`. The within-unit lagged-
treatment column is all-zero in that setting, which made the logistic
treatment-model design singular; the failure was silently caught and the
weights fell back to the marginal mean for both the numerator and denominator,
cancelling to `1.0`. The MSM then silently reduced to an unweighted,
**confounded** regression. The fix drops zero-variance columns before fitting,
so the confounders are now used and the weights are computed correctly.

**Who is affected.** Anyone who called `sp.stabilized_weights` / `sp.msm` on a
panel with **one period per unit** (point treatment). Multi-period panels —
the intended MSM use case — are **unaffected** (their weights already varied
correctly and are numerically identical before and after).

**What to do.** Re-run any single-period MSM analyses: the previous output was
equivalent to an unadjusted regression and should not be relied on. The fixed
weights match a textbook stabilized-IPTW computation to machine precision. If
the treatment model genuinely cannot be fit (e.g. perfect separation), you now
get a `RuntimeWarning` instead of a silent fallback.

---

<a id="sp-synth-default-classic"></a>

## 1.16.1 — ⚠️ `sp.synth()` default method restored to `'classic'`

**What changed.** A bare `sp.synth(...)` call (no `method=`) now runs
`method='classic'` — canonical Abadie–Diamond–Hainmueller (2010) synthetic
control with convex, non-negative, sum-to-one donor weights. The signature
default had silently drifted to `method='augmented'` (Augmented SCM,
Ben-Michael, Feller & Rothstein 2021), which deliberately allows negative
donor weights by extrapolating outside the donor convex hull. That
contradicted the documented default (`sp.synth` docstring: `method : str,
default 'classic'`), the migration-from-R mapping (`Synth::synth` is
classic), and the canonical Prop99 examples shipped in the docs.

**Who is affected.** Anyone calling `sp.synth(...)` **without** an explicit
`method=`. Estimated effects, donor weights, and synthetic trajectories
revert from ASCM to classic SCM. Every call that already passes `method=`
(including `method='augmented'` / `'ascm'`) is **unchanged**.

**What to do.** To keep Augmented SCM, pass it explicitly:

```python
res = sp.synth(df, outcome=..., unit=..., time=...,
               treated_unit=..., treatment_time=..., method='augmented')
```

Otherwise no action is needed — the default now matches the documentation and
the R `Synth` reference implementation.

Guarded by
`tests/test_synth.py::TestSyntheticControl::test_weights_non_negative`.

---

<a id="sp-causal-forest-aipw-fix"></a>

## 1.16.0+source.20260531 — ⚠️ Causal-forest ATE/ATT now doubly-robust (AIPW)

**What changed.** `CausalForest.average_treatment_effect(...)` previously
returned a plug-in average of the forest's CATE predictions. Forest
regularisation shrinks those predictions, so the plug-in mean is biased
(≈ 15 % high on a clean-overlap design) and is *not* the estimand
`grf::average_treatment_effect` reports. It now returns the doubly-robust
AIPW influence-function mean built from the forest's own cross-fitted
nuisances (`Γ_i = τ̂ + (T−ê)/(ê(1−ê))·(Y − m̂ − (T−ê)τ̂)`), with the
influence-function standard error `sd(Γ)/√n`.

**Who is affected.** Anyone reading
`cf.average_treatment_effect(...)['estimate']` or `['se']` (any
`target_sample`: `all`/`treated`/`control`/`overlap`). The plug-in
convenience methods `cf.ate()` / `cf.att()` are **unchanged**.

**What to do.** Re-run any analysis that reported a causal-forest ATE/ATT
from `average_treatment_effect`. The new estimate is closer to truth and
agrees with `grf` within combined Monte Carlo error.

```python
ate = cf.average_treatment_effect(target_sample="all")  # ['method']=='aipw'
ate_plugin = cf.ate()                                   # still available, plug-in
```

Guarded by `tests/reference_parity/test_causal_forest_aipw_recovery.py`
and `tests/reference_parity/test_grf_parity.py`.

---

## 1.16.0 — ⚠️ `sp.xtabond` Arellano-Bond GMM correctness fix

**What broke.** `sp.xtabond` (and `sp.panel(method='ab')`) used a flat,
fixed block of lagged-level instrument columns and then dropped every
row that was missing any of them — on a short panel this discards most
of the sample — and weighted with `W = (Z'Z)⁻¹`. The correct
Arellano-Bond estimator uses a **block-diagonal** GMM instrument matrix
(each available deeper lag `Y_{i,s}`, `s ≤ t-2`, is a period-specific
moment; missing lags are zero-filled, no rows dropped) and the one-step
weight `W = (Σᵢ Zᵢ'H Zᵢ)⁻¹`, with `H` the first-difference MA(1)
structure (2 on the diagonal, −1 on the first off-diagonals). The old
code returned `β_{y₋₁}=0.264 (se 0.224)` where Stata returns
`0.391 (se 0.046)` — a 48 % estimate gap and an 80 % SE gap.

**Who is affected.** Anyone who called `sp.xtabond(...)` or
`sp.panel(..., method='ab'|'system')` on an earlier release. **Both the
point estimates and the standard errors change** — point estimates are
*not* preserved here (unlike the qreg fix).

**What to do.**

| Surface | Pre-fix | Action |
| --- | --- | --- |
| `res.estimate`, `detail["coefficient"]` | biased (instrument set wrong) | Rerun |
| `res.se`, `detail["se"]`, `res.ci`, `res.pvalue` | wrong | Rerun |
| `gmm_lags` default | `(2, 5)` | now `(2, None)` = all deeper lags (Stata default); pass an explicit max to cap |
| `method='system'` | returned a number | now raises `NotImplementedError`; use `method='difference'` |
| `twostep=True` SEs | uncorrected | now Windmeijer (2005)-corrected when `robust=True` |

**Verification.** One-step robust `sp.xtabond` now matches Stata
`xtabond y x, lags(1) vce(robust)` to machine precision on the parity
DGP (`tests/r_parity/50_xtabond`, rel ≈ 1e-15 on both β and SE);
guarded by `tests/test_gmm.py::TestArellanoBond::test_parity_matches_stata_xtabond`.

---

<a id="sp-qreg-se-fix"></a>

## 1.16.0 — ⚠️ `sp.qreg` Powell sandwich SE correctness fix

**What broke.** The Powell (1991) kernel sandwich for quantile
regression standard errors was implemented with an extra factor of
`n` in the denominator: `V = τ(1−τ) / (n · f̂(0)²) · (X'X)⁻¹`. The
textbook formula (Koenker 2005, eq. 3.7) is
`V = τ(1−τ) / f̂(0)² · (X'X)⁻¹` — no `n`. The reported SE was
therefore the correct SE divided by √n. On the parity dataset with
n = 500 (`tests/r_parity/40_qreg`), the bug under-reported SE by
~20× and produced z-statistics in the 6–30 range for null
covariates.

**Who is affected.** Anyone who used the `se`, `pvalue`, `ci`, or
`z` columns of `sp.qreg(...).detail` (or the top-level `res.se` /
`res.pvalue` / `res.ci`) on an earlier release. Point estimates
(`res.estimate`, `detail["coefficient"]`) are **unchanged at machine
precision** and do not need to be rerun.

**What to do.** Pull the patch, then rerun any analysis that
referenced an `sp.qreg` standard error. Concretely:

| Surface | Pre-fix value | Action |
| --- | --- | --- |
| `res.se`                                       | SE / √n   | Multiply by √n to recover, or just rerun |
| `res.pvalue`                                   | ~0        | Rerun — most pre-fix p-values were spuriously zero |
| `res.ci`                                       | too narrow | Rerun |
| `res.detail["se" / "z" / "pvalue"]`            | as above  | Rerun |
| `res.estimate`, `res.detail["coefficient"]`    | correct   | No change needed |

**Verification.** The cross-language parity table in
`tests/r_parity/results/parity_table_3way.md` for module `40_qreg`
shows the post-fix SE matching `quantreg::rq` (Powell `nid` kernel)
within 1.4–6.8 % and Stata `qreg` (Koenker-Bassett) within 2.9 %.
This is the expected residual gap between three different
implementations of the same sandwich.

**Why was it not caught earlier.** No 3-way Stata parity test
existed for quantile regression before the 2026-05-28 session, and
the unit tests in `tests/test_quantile.py` checked only point
estimates and that SEs were finite — never against an external
reference value.

---

<a id="sp-rdrobust-bwselect-cct-r-parity-opt-in"></a>

## v1.15.2 → v1.15.3 — doc-only PyPI hero-banner fix

**No code changes, no migration step.** The v1.15.2 PyPI project page
rendered the hero banner as a broken image because the `<img>` tag in
`README.md` / `README_CN.md` used a repo-relative path
(`docs/logo/readme-1.png`) that PyPI's long-description renderer
cannot resolve. v1.15.3 swaps the path for the absolute raw GitHub
URL so the banner loads on PyPI / TestPyPI / off-GitHub mirrors.
Module hashes match v1.15.2 bit-for-bit; only the long-description
metadata baked into the wheel + sdist changes.

---

## v1.15.1 → v1.15.2 — strict-JSON MCP wire, dual-track replicate, packaging

**No estimator numerical path changes.** Three classes of consumers
should take note:

- **`sp.agent.mcp_server` clients** (Claude Desktop / Codex / any
  RFC 8259-strict JSON parser). v1.15.1 could leak the non-standard
  literals `NaN` / `Infinity` / `-Infinity` into responses whenever an
  estimator surfaced a degenerate float (`np.nan` standard errors on a
  singular covariate, `inf` log-likelihood on a saturated model, etc).
  v1.15.2 walks all containers before `json.dumps` and serialises with
  `allow_nan=False`, replacing those values with `null`. **Action**:
  none — strict parsers that previously failed now succeed; lenient
  parsers see `null` where they used to see `NaN`. Update your
  downstream JSON Schema if it explicitly typed those fields as
  `number` (they should be `["number", "null"]`).

- **`sp.causal_text` users.** The MVP relied on a soft import of
  `sentence-transformers`. v1.15.2 adds an explicit
  `pip install statspai[text]` extra. The lazy import path is
  preserved, but the `ImportError` message now points at the extra
  instead of suggesting a bare `pip install sentence-transformers`.

- **`sp.replicate` users.** Entries for Card (1995), Abadie-Diamond-
  Hainmueller (2010), Lalonde (1986) / DW (1999), and Lee (2008) now
  return classic + modern recipes computed on the bundled real CSVs
  instead of single-track simulated stubs. If you were pinning to the
  v1.15.1 simulated numbers in CI, switch to the published-paper
  benchmarks now exposed via `df.attrs['paper_original']` (see
  `sp.datasets.nsw_lalonde(simulated=False)` and
  `sp.datasets.lee_2008_senate(simulated=False)`).

Existing `sp.rdrobust` / `sp.nbreg` / `sp.xtnbreg` / `sp.menbreg`
call sites carry over unchanged from v1.15.1.

---

## v1.15.0 → v1.15.1 — `sp.rdrobust(bwselect='cct')` R-parity opt-in

**No breaking change.** `sp.rdrobust` keeps `bwselect='mserd'` (StatsPAI's
own MSE-optimal recipe) as the default — every existing call returns the
same numbers. A new opt-in value `bwselect='cct'` is added for users who
need bit-equal R `rdrobust::rdrobust` parity.

`sp.nbreg`, `sp.xtnbreg`, and `sp.menbreg` also get clearer README /
release-note documentation in v1.15.1. Their call signatures and
numerical paths are unchanged, so there is no migration step for
negative-binomial regression users.

### When to switch from `'mserd'` to `'cct'`

Use `bwselect='cct'` when **any** of these apply:

- You're replicating a CCT 2014 / Cattaneo-Idrobo-Titiunik (2018, 2020)
  paper and need the published numbers to the 4th decimal.
- A reviewer asks for "the same number R `rdrobust` gives".
- Your data has features that stress StatsPAI's internal pilot bandwidth
  (heavy tails, small `n`, mass points). On the canonical Lee/CCT Senate
  replication, `'mserd'` gives `Conv = 12.62 / h = 4.6` while `'cct'`
  gives `Conv = 7.41 / h = 17.75` — the latter matches R bit-equal.

Keep the default `bwselect='mserd'` when:

- You don't need exact R parity, **and**
- You don't want a soft dependency on the `rdrobust` package, **and**
- Your downstream tests / pipelines have already been calibrated against
  StatsPAI's `'mserd'` numbers.

### How to switch

```python
import statspai as sp

# Before — StatsPAI internal MSE-optimal (kept stable)
res = sp.rdrobust(data=df, y='y', x='x', c=0)
# After — R-bit-equal via official rdrobust delegation
res = sp.rdrobust(data=df, y='y', x='x', c=0, bwselect='cct')
```

Install the optional dependency once:

```bash
pip install statspai[rd-cct]   # adds rdrobust>=1.3
```

Calling `bwselect='cct'` without it raises a clear `ImportError` that
points you to the install command — no silent fallback.

### Why we didn't change `'mserd'` itself

Aligning the internal `'mserd'` to R `rdbwselect`'s recursive 3-step
recipe would shift point estimates on every dataset that exercises
StatsPAI's RD path (5+ test classes, `r_parity` scripts, downstream
docs / notebooks). The additive `'cct'` route gives anyone who wants R
parity an immediate path **and** preserves the 1.x line's numerical
stability. A future major version may flip the default.

---

## v1.11 → v1.12 — DML module hardening

`sp.dml`, `sp.dml_panel`, `sp.dml_model_averaging` keep all of their
existing call signatures (every old script imports the same way and
runs without code changes), but several internal numerical behaviours
shift on the boundaries of the input space. The full release-note
discussion lives in [`CHANGELOG.md`](CHANGELOG.md) under
`[1.12.0]`; the breaking points are summarised here.

### What can change in your numbers

| Estimator | What changed | When you'll notice |
| --- | --- | --- |
| `sp.dml(model='irm')` | `KFold` → `StratifiedKFold` (stratified by D). Empty subgroup folds were silently filled with zeros for `g(1, X)` / `g(0, X)`; they now raise `IdentificationFailure`. | Small N, imbalanced D, or small `n_folds` may give point estimates a hair different from before — folds are no longer drawn from the un-stratified KFold sequence. |
| `sp.dml(model='iivm')` | Same — `StratifiedKFold` on Z, plus empty-subgroup `IdentificationFailure`. | Small N or imbalanced Z. |
| `sp.dml(model='pliv')` | Weak-IV floor on the ML-residualised partial correlation: `1e-6 → 1e-3`. | When your instrument's first-stage corr after ML residualisation is in `[1e-6, 1e-3]`, the call now raises `RuntimeError` with a clear hint to consult `sp.weakrobust` / `sp.anderson_rubin_test`. |
| `sp.dml_model_averaging` | Default `weight_rule="inverse_risk"` → `"short_stacking"`. | Different default point estimate. To preserve the v1.11 number, pass `weight_rule="inverse_risk"` explicitly. |
| `sp.dml_model_averaging` | NaN rows in `y` / `treat` / `covariates` are now dropped instead of being passed to sklearn. | If your data had NaNs you may have been getting `RuntimeError("No candidate produced a finite estimate")` or, worse, NaN θ̂; now you'll silently lose those rows but the estimate will be finite. The dropped count is reported in `model_info["n_dropped_missing"]`. |
| `sp.dml_panel(binary_treatment=True)` | Now a deprecated no-op — the previous classifier path was incorrect. The estimator runs as `binary_treatment=False` (regressor on D̃) regardless. | Different θ̂ when you used `binary_treatment=True`; a `DeprecationWarning` fires so you see it. |

### Recovering the v1.11 default for `dml_model_averaging`

```python
# v1.11 default behaviour (inverse-MSE-weighted average of per-candidate θ̂)
result = sp.dml_model_averaging(
    df, y="y", treat="d", covariates=cov_list,
    weight_rule="inverse_risk",   # v1.12 default is "short_stacking"
)

# v1.12 default — Ahrens et al. (2025, JAE) eq. 7 short-stacking
result = sp.dml_model_averaging(
    df, y="y", treat="d", covariates=cov_list,
    # weight_rule="short_stacking" (now the default)
)
result.model_info["weights_g"]   # CLS stacking weights for E[Y|X]
result.model_info["weights_m"]   # CLS stacking weights for E[D|X]
```

### Recovering the v1.11 `dml_panel(binary_treatment=True)` semantics

There is no recovery — the v1.11 path was incorrect (classifier on
within-demeaned features but raw {0,1} labels). For DR-style ATE on
binary D in panels, prefer one of:

```python
# (a) sp.dml IRM with unit dummies as covariates
import pandas as pd
unit_dummies = pd.get_dummies(df["unit"], drop_first=True)
df_aug = pd.concat([df, unit_dummies], axis=1)
sp.dml(df_aug, y="y", treat="d",
       covariates=[*cov_list, *unit_dummies.columns.tolist()],
       model="irm")

# (b) sp.etwfe (extended TWFE for staggered binary treatment in panels)
sp.etwfe(df, yname="y", tname="t", gname="treatment_cohort",
         idname="unit", covariates=cov_list)

# (c) sp.callaway_santanna (staggered DR-DiD)
sp.callaway_santanna(df, yname="y", tname="t",
                     gname="treatment_cohort", idname="unit")
```

### New capabilities (no migration needed — purely additive)

- `sample_weight=` is now accepted on `sp.dml(model='plr' | 'irm')`,
  `sp.dml_panel`, and `sp.dml_model_averaging`. Pass a 1-D array, a
  pandas Series, or a column name. The weighted estimator uses a
  Z-estimator sandwich variance throughout. `sp.dml(model='pliv' | 'iivm')`
  raise `NotImplementedError` if a non-trivial weight is supplied.
- `random_state=` (default 42) on every `sp.dml(model=...)` call
  controls fold assignment deterministically.
- `model_info["diagnostics"]` is populated on every variant — propensity
  distribution, n clipped, subgroup-fallback counts, partial correlation,
  approximate first-stage F, etc.
- String learner aliases (already shipped in 1.11.4) still work:
  `sp.dml(..., ml_g='rf', ml_m='lasso')`.

---

## v1.11 → v1.12 — `esttab` becomes a thin facade over `regtable`

The Stata-style `esttab()` previously shipped a ~500-line
`EstimateTable` class that re-implemented the full renderer pipeline.
PR-B/5c in v1.12 collapses it to a thin facade that translates
Stata-flavoured kwargs and forwards to `sp.regtable`.

**API is unchanged**, including `eststo()` / `estclear()` global store,
`isinstance(x, EstimateTableResult)` type identity, and all
`esttab(*results, se=, t=, p=, ci=, stats=, output=, ...)` keyword
spellings. Rendered output now matches `regtable`'s book-tab style.
A `DeprecationWarning` is emitted on first use; plan to migrate to
`sp.regtable(...)` directly within the next two minor releases.

### Behaviour changes

| Old | New |
| --- | --- |
| `se=True/t=True/p=True/ci=True` exclusive flags | translated to `regtable(se_type='se' \| 't' \| 'p' \| 'ci')`. Priority `ci > p > t > se` if multiple are passed (matches legacy). |
| `output='csv'` | implemented via `result.to_dataframe().to_csv()`. |
| `output='markdown'` / `'md'` / `'tex'` aliases | unchanged, all forward to the corresponding regtable renderer. |
| `filename=` extension auto-detect | unchanged (`.tex` → latex, `.html` → html, `.md` → markdown, `.csv` → csv). |

### Side-by-side migration

```python
# Before — Stata-style stateful workflow
sp.eststo(m1, name="(1)")
sp.eststo(m2, name="(2)")
sp.esttab(stats=["N", "R2", "adj_R2"], output="latex",
          filename="table1.tex")
sp.estclear()

# After — direct regtable call (same LaTeX, no global state)
sp.regtable(
    [m1, m2],
    model_labels=["(1)", "(2)"],
    stats=["N", "R2", "adj_R2"],
    filename="table1.tex",
)
```

---

## v1.11 → v1.12 — `modelsummary` becomes a thin facade over `regtable`

The R-style `modelsummary()` previously shipped a ~700-line renderer
pipeline that re-implemented coefficient extraction, star formatting,
three-line table styling and every export format. PR-B/5b in v1.12
collapses it to a thin facade that translates R-flavoured kwargs and
forwards to `sp.regtable`.

**API is unchanged**, but rendered output now matches `regtable` (book-tab
three-line, publication-quality star legend). A `DeprecationWarning` is
emitted on first use; plan to migrate to `sp.regtable(...)` directly
within the next two minor releases.

### Behaviour changes

| Old | New |
| --- | --- |
| `stars={"*": 0.10, "**": 0.05, "***": 0.01}` | only the threshold *values* are kept; the symbol overrides are dropped (regtable's ladder is `*/**/***` by convention; use `regtable(notation='symbols')` for `†/‡/§`) |
| `se_type='brackets'` | downgraded to parens with `UserWarning`; use `show_ci=True` for `[lo, hi]` if you want brackets to convey actual information |
| `se_type='none'` | downgraded to parens with `UserWarning`; the SE row stays |
| Stat keys `nobs/r_squared/adj_r_squared/f_stat` | translated to regtable canonical (`N`/`r2`/`adj_r2`/`F`) |
| Stat keys `method`/`bandwidth`/`estimand` | silently dropped (modelsummary-only; build a custom `add_rows={}` if needed) |

`coefplot` is unchanged — independent of the table renderer.

### Side-by-side migration

```python
# Before — R-style functional API
sp.modelsummary(m1, m2, m3,
                model_names=["Base", "Mid", "Full"],
                stats=["nobs", "r_squared", "adj_r_squared"],
                output="latex")

# After — direct regtable call (same LaTeX output, full control)
sp.regtable(
    [m1, m2, m3],
    model_labels=["Base", "Mid", "Full"],
    stats=["N", "r2", "adj_r2"],
).to_latex()
```

---

## v1.11 → v1.12 — `outreg2` becomes a thin facade over `regtable`

The Stata-style `OutReg2` class and `outreg2()` function previously
shipped a bespoke 800-line renderer that re-implemented coefficient
extraction, star formatting, three-line table styling, and Excel /
Word / LaTeX export. PR-B in v1.12 collapses that to ~150 lines of
glue that translates Stata-flavoured kwargs and forwards to
`sp.regtable`.

**API is unchanged**, but rendered output now matches `regtable`'s
canonical book-tab style. The visible label changes are listed below.
A `DeprecationWarning` is emitted on first use; plan to migrate to
`sp.regtable(...)` directly within the next two minor releases.

### Label / format changes

| Legacy outreg2 output | New (regtable canonical) |
| --- | --- |
| `Variables` column header | blank (book-tab convention) |
| `R-squared` | `R²` |
| `Adj. R-squared` | `Adj. R²` |
| `Observations` | `N` |
| `F-statistic / Trees` | `F` *(bug fix: "/ Trees" only applied to causal-forest results)* |
| LaTeX missing star legend | proper `\multicolumn` legend below the rule |
| LaTeX `& None & None \\` junk row | gone *(bug fix: spurious empty ATE row)* |

### Removed parameter

| Old | New |
| --- | --- |
| `show_se=False` | no longer supported. Emits `UserWarning`; the SE row stays. Use `sp.regtable(..., se_type='t' \| 'p' \| 'ci')` directly if you need a different cell. |

### Side-by-side migration

```python
# Before — Stata-style stateful builder
o = sp.OutReg2()
o.set_title("Wage Regressions")
o.add_model(m1, "Baseline")
o.add_model(m2, "Full")
o.add_note("Robust SE in parentheses")
o.to_excel("table1.xlsx")

# After — direct regtable call (same Excel output, full control)
sp.regtable(
    [m1, m2],
    title="Wage Regressions",
    model_labels=["Baseline", "Full"],
    notes=["Robust SE in parentheses"],
).to_excel("table1.xlsx")
```

---

## Migrating from `pyreghdfe`

`pyreghdfe` (`pip install pyreghdfe`) is a Python port of Stata's
`reghdfe` maintained as a standalone package. Its scope — multi-way FE
OLS with robust / multi-way cluster SEs, singleton dropping, weighted
regression — is now a strict subset of `sp.hdfe_ols` / `sp.absorb_ols`
in StatsPAI.

### API mapping (pyreghdfe → StatsPAI)

| `pyreghdfe` | StatsPAI (`import statspai as sp`) |
| --- | --- |
| `reghdfe(data=df, y='y', x=['x'], fe=['firm','year'], cluster=['firm'])` | `sp.absorb_ols(y=df['y'].values, X=df[['x']].values, fe=df[['firm','year']], cluster=df['firm'].values, solver='lsmr')` |
| Stata-style formula via pyreghdfe is not supported | `sp.hdfe_ols("y ~ x \| firm + year", data=df, cluster="firm")` (formula interface via pyfixest backend) |
| `solver='lsmr'` / `'lsqr'` | `solver='lsmr'` / `'lsqr'` — same Krylov paths (scipy.sparse.linalg) |
| Krylov-based solvers (LSMR/LSQR) | default `solver='map'` — alternating projections + Irons-Tuck acceleration, typically faster on well-conditioned panels. LSMR/LSQR remain opt-in for pathological FE structures. |
| weighted regression | `weights=` kwarg; LSMR path uses the standard √w transformation on both the sparse design and the response |
| singleton drop | `drop_singletons=True` (default) |
| multi-way cluster SE | `cluster=[firm_arr, year_arr]` (inclusion-exclusion CGM with PSD correction) |

### What you also get

- `sp.ppmlhdfe` — Poisson pseudo-ML with HDFE (not available in `pyreghdfe`).
- Rust-accelerated mean-sweep kernel ([rust/statspai_hdfe/](rust/statspai_hdfe/)).
- Formula interface and unified result object (`summary()`, `to_latex()`, `to_excel()`).
- One-line cross-solver parity check (all three solvers exposed under the
  same API — see `tests/test_hdfe_native.py::test_demean_alt_solver_matches_map_two_way`).

### Numerical parity

Default MAP and `solver='lsmr'` / `'lsqr'` agree on identical data to
`atol=1e-6` on two-way FE OLS (with and without weights, with and
without clustering). See the cross-solver parity suite in
`tests/test_hdfe_native.py`. We do not take a runtime dependency on
`pyreghdfe`; correctness is anchored to scipy's well-established
`scipy.sparse.linalg.lsmr` / `lsqr` plus the internal MAP baseline.

### When to prefer which solver

- **Default (`solver='map'`)**: almost everything. MAP + Aitken is
  typically 2–5× faster than LSMR on canonical firm × year panels.
- **`solver='lsmr'`**: ill-conditioned / highly nested FE structures
  where MAP shows slow convergence (`converged=False`,
  `iters==maxiter`). LSMR is more robust to near-redundancy between FE
  dimensions.
- **`solver='lsqr'`**: exposed for users migrating from code that
  explicitly requested LSQR. For new work prefer LSMR, which scipy
  implements on the same interface and generally offers better
  numerical stability on sparse least-squares.

---

## v1.8.0 → v1.9.0 — Agent-native API surface (no breaking changes)

**Strictly additive release.** Twelve new agent-shaped APIs land
under ``sp.``: ``audit``, ``bib_for``, ``brief``, ``detect_design``,
``examples``, ``preflight``, ``session`` (the seven new top-level
functions), plus ``result.brief()`` / ``result.cite(format=...)``
methods, plus three MCP-server features (``statspai-mcp`` console
script, ``prompts/list``, per-function ``statspai://function/{name}``
resources). **No estimator numerical paths changed**; every
coefficient / SE / CI / p-value is byte-identical to v1.8.0. See
the v1.9.0 [CHANGELOG](CHANGELOG.md#190--agent-native-api-surface-12-modules-across-4-phases)
entry for the full surface.

### Backward-compat invariants the test suite pins

The 422 new tests include explicit regression guards on these
contracts. If your code depended on any of them, nothing changes.

- ``CausalResult.to_dict()`` with no kwargs is **byte-identical**
  to ``to_dict(detail="standard")`` — the legacy default. The new
  ``detail`` parameter is keyword-only and adds three documented
  levels (``"minimal"`` / ``"standard"`` / ``"agent"``).
- ``CausalResult.cite()`` with no kwargs still returns a BibTeX
  string. The new ``format=`` keyword adds ``"apa"`` / ``"json"``
  options without changing the default.
- ``result.for_agent()`` is now a thin alias for
  ``result.to_dict(detail="agent")`` and produces the same dict.
  Existing callers see no change; new code should prefer the
  explicit form for readability.
- ``result.to_agent_summary()`` is unchanged. Its docstring now
  cross-references ``to_dict(detail="agent")`` so future readers
  know the distinction (``to_agent_summary`` is the *nested*
  schema with a ``point`` sub-dict; ``to_dict(detail="agent")`` is
  the *flat* schema). Both round-trip through ``json.dumps``.
- ``execute_tool``'s exception envelope still carries the legacy
  ``error`` / ``tool`` / ``arguments`` / ``remediation`` fields
  unchanged. Two new fields — ``error_kind`` and ``error_payload``
  — are added **only** when the caught exception is a
  ``StatsPAIError`` subclass, so any agent that previously branched
  on ``"error_kind" in out`` to detect structured errors gets a
  clean signal.

### One subtle widening to be aware of

- ``sp.agent.execute_tool``'s default serializer now invokes
  ``r.to_dict(detail="agent")`` instead of ``r.to_dict()``. The
  result dict is a strict superset of the previous shape — every
  pre-1.9 key is still present at the same path; ``violations``,
  ``warnings``, ``next_steps``, and ``suggested_functions`` are
  added. The MCP ``tools/call`` payload is therefore ~3× larger by
  default. Agents that need the smaller form should pass
  ``detail="standard"`` (or ``"minimal"``) in the ``tools/call``
  arguments — the MCP input schema documents this.

### New entry points worth knowing about

- Agents handed unfamiliar data → ``sp.detect_design(df)``.
- Before an expensive call → ``sp.preflight(df, "did", y=..., ...)``.
- After fitting → ``result.brief()`` for dashboards,
  ``sp.audit(result)`` for the missing-evidence checklist,
  ``result.cite(format="apa")`` for prose citations.
- Reproducible RNG → ``with sp.session(seed=42): ...``.
- One-shot install for MCP clients → ``pip install statspai`` now
  exposes ``statspai-mcp`` on PATH (Claude Desktop /
  ``claude_desktop_config.json`` example in
  [agent/mcp_server.py](src/statspai/agent/mcp_server.py)).

---

## v1.6.5 → v1.6.6 — ⚠️ Heckman two-step SE correctness fix (+ HDFE solver option)

**Two-part release.** (1) Correctness fix for `sp.heckman` standard
errors — point estimates unchanged, **SE / t / p / CI change**.
(2) Additive HDFE LSMR/LSQR solver option — all HDFE MAP output is
byte-identical to v1.6.5.

### What changed numerically (Heckman two-step)

`sp.heckman(...)` previously reported an HC1-style sandwich that the
source code itself flagged as
`"Heckman SEs are complex; robust is conservative"`. This was a known
limitation, not a secret bug — but it meant reported SEs, t-stats,
p-values and CIs were off by an amount that depended on (a) how
strongly selection induced heteroskedasticity `σ²(1 − ρ² δ_i)` and
(b) how uncertain the probit first-stage estimate γ̂ was.

v1.6.6 replaces it with the textbook Heckman (1979) / Greene (2003, eq.
22-22) / Wooldridge (2010, §19.6) analytical two-step variance:

```text
V(β̂) = σ̂² (X*'X*)⁻¹ [ X*'(I − ρ̂² D_δ) X* + ρ̂² F V̂_γ F' ] (X*'X*)⁻¹
```

- `X*`: second-stage design matrix including λ̂ as its last column.
- `δ_i = λ̂_i (λ̂_i + Z_iγ̂) ≥ 0` (Mills' ratio inequality).
- `D_δ = diag(δ_i)`; `F = X*' D_δ Z` (`k × q`).
- `V̂_γ = (Z' diag(w_i) Z)⁻¹` with probit information weights
  `w_i = φ(Z_iγ̂)² / [Φ(Z_iγ̂)(1 − Φ(Z_iγ̂))]`.
- `σ̂² = RSS / n_sel + β̂_λ² · mean(δ_i)` (Greene 22-21) —
  replaces the old naive `RSS / (n_sel − k)`.
- `ρ̂² = β̂_λ² / σ̂²`.

`model_info['sigma']` / `model_info['rho']` now also use this
consistent σ̂², so downstream code reading those fields will see
slightly different numbers.

### Who is affected

- Any caller of `sp.heckman(...)` — SEs, t-stats, p-values, CIs change.
- Point estimates `β̂` **do not change** (OLS of y on [X, λ̂]
  is unaffected by the variance formula).
- Callers that pin SE values in their own test suites against a
  pre-v1.6.6 StatsPAI will need to re-baseline.

### What you should do

1. **If you cited a Heckman SE / t / p / CI from StatsPAI ≤ 1.6.5**,
   re-run and update. The direction of change depends on whether
   selection-induced heteroskedasticity (reduces SE) or
   generated-regressor uncertainty (increases SE) dominates.
2. **Cross-validation**: compare the new output against Stata
   `heckman y x, select(z) twostep` or R
   `sampleSelection::heckit(...)`. Both implement the same Heckman
   (1979) formula; agreement should be to the documented precision.
3. **If you want the old conservative HC1 sandwich** for any reason
   (e.g. replicating a legacy pipeline), there is no supported way to
   get it. The old formula was not a convention choice — it was a
   known approximation the project had not yet replaced.

### Reference formula

Same as above, with the influence-function derivation:

```text
β̂ − β = (X*'X*)⁻¹ [ X*' e − β̂_λ · X*' D_δ Z · (γ̂ − γ) ] + o_p(n^{-1/2})
```

The first term gives the heteroskedastic `X*'(I − ρ̂² D_δ) X*`
contribution; the second gives the `ρ̂² F V̂_γ F'` generated-regressor
contribution, since `∂λ / ∂γ' = −λ(λ + Zγ) Z' = −δ · Z'`.

---

## v1.6.4 → v1.6.5 — ⚠️ Standalone LIML correctness fix

**Narrow correctness follow-up to v1.6.4.** If your codebase only uses
`sp.ivreg`, `sp.iv.iv`, `sp.iv.fit`, or `sp.ivreg(method='liml')` you
are **not affected** — those paths were fixed in v1.6.4. This release
closes an orphan copy of the same bug that lived in the standalone
`sp.liml` / `sp.iv.liml` entry point.

### What changed numerically

Anything calling `sp.liml(...)` directly will see both **β̂ and SE
change** compared to ≤ v1.6.4. Two independent bugs were fixed:

1. **κ_LIML solver**: switched from the non-symmetric
   `np.linalg.eigvals(inv(A) @ B)` (which can silently return complex
   eigenvalues and a biased κ) to the proper generalized symmetric
   eigenvalue problem `scipy.linalg.eigh(S_exog, S_full)`. Point
   estimates β̂ shift to the correct κ.
2. **Sandwich meat**: the cluster / robust meat used raw `X` instead of
   the k-class transformed `AX = (I − κ M_Z) X`. Same bug family as
   v1.6.4 for 2SLS; same fix (use the influence-function regressor in
   the meat).

### Post-fix consistency checks

- `sp.liml(...)` now produces **byte-identical** output to
  `sp.ivreg(..., method='liml')`.
- β̂ agrees with `linearmodels.IVLIML` to machine precision.
- Cluster SEs differ from `linearmodels.IVLIML` by ~0.1–0.2% because
  StatsPAI uses the k-class FOC-derived meat `AX = (I − κ M_Z) X`,
  while `linearmodels` uses the 2SLS-style meat `X̂ = P_Z X`
  regardless of κ. Both estimators are asymptotically equivalent and
  coincide exactly at κ = 1 (2SLS). The convention is documented in
  the new test file `tests/reference_parity/test_liml_se_parity.py`.

### What you should do

1. **If you have published LIML results** from a version ≤ v1.6.4 via
   `sp.liml(...)`, re-run and update — the old κ could be materially
   off and the old SE was built from the wrong meat.
2. **If you want LIML and only used `sp.ivreg(method='liml')`**, no
   action needed; v1.6.4 already has the correct formula.
3. **If you pinned SE or coefficient values** against the standalone
   `sp.liml` in your test suite, re-baseline to the v1.6.5 numbers.

### Reference formula (same as v1.6.4 for the k-class meat)

```text
β̂ − β = (X' A X)⁻¹ (AX)' u ,  A = (1 − κ) I + κ P_Z
Meat (cluster):  Σ_c (Σ_{i∈c} (AX)_i u_i)(·)'
Bread         :  (X' A X)⁻¹  = (AX' X)⁻¹
```

For 2SLS (κ = 1) `AX = P_Z X = X̂`; for LIML/Fuller `AX` is the
k-class transformed regressor.

---

## v1.6.3 → v1.6.4 — ⚠️ IV SE correctness fix

**Correctness-fix release.** No API surface changes, no new functions,
no docstring renames. **Numerical output of IV cluster / robust SE
changes** — this is the whole point of the release.

### What changed numerically

`sp.iv`, `sp.ivreg`, and `sp.iv.fit(method='2sls' | 'liml' | 'fuller')`
produce different standard errors when called with `robust={'hc0',
'hc1', 'hc2', 'hc3'}` or `cluster=...`. The fix restores the textbook
Cameron–Miller (2015) / Stata `ivregress` / `linearmodels` formula —
meat uses the projected regressor `X̂ = P_W X` rather than the raw
`X = [X_exog, X_endog]`.

Concretely the sandwich is now

```text
V̂ = (X̂'X̂)⁻¹ · [ Σ_c (X̂_c' û_c)(û_c' X̂_c) ] · (X̂'X̂)⁻¹
```

for the cluster case, and analogously for HC0/HC1/HC2/HC3. Before v1.6.4
the bread used `X̂` but the meat used `X`, which is a strictly incorrect
estimator for 2SLS — it happens to coincide with the correct formula
only when the first stage is a perfect fit (never, in practice).

### Who is affected

- Any IV workflow using `robust=` or `cluster=` with 2SLS, LIML, or Fuller.
- **Not affected**: point estimates (`β̂` is algebraically unchanged by
  the projection in the meat), nonrobust default SE, `method='gmm'`,
  `method='jive'`, and `sp.iv.ujive` / `ijive` / `rjive`.

### What you should do

1. **If you have published results** citing an IV SE / t-stat / p-value
   / CI from StatsPAI ≤ 1.6.3, re-run and update. The bias in the
   reported SE can be several-fold depending on first-stage fit —
   **not a rounding issue**.
2. **If you have pinned SE values in your test suite** against an
   earlier StatsPAI version, expect a mismatch. You can verify the new
   numbers by cross-checking with `linearmodels.IV2SLS(...).fit(
   cov_type='clustered', debiased=True)` — they should now agree to
   machine precision.
3. **If you were intentionally trying to reproduce the old (wrong)
   numbers**, don't. There is no supported way to get the
   pre-v1.6.4 behaviour because it was not a convention choice — it
   was a bug.

### Reference formula

For k-class with parameter κ (2SLS → κ=1, LIML → κ=κ_LIML, Fuller →
κ_LIML − α/(n−K)):

- Bread: `(X' A X)⁻¹` with `A = (1−κ) I + κ P_W`
- Meat: uses `A X` (the k-class transformed regressor); for 2SLS
  `A X = P_W X = X̂`
- FOC: `X' A (y − X β) = 0`, so the influence function is
  `β̂ − β = (X'AX)⁻¹ (AX)' u`, and the cluster/robust variance
  plugs `(AX)_i u_i` into the moment sum.

Pre-v1.6.4 the implementation plugged `X_i u_i` instead of `(AX)_i u_i`.

---

## v1.6.2 → v1.6.3 — DiD frontier sprint

**Strictly additive** plus one docstring / label truth-up. No existing
estimator's numerical path changes.

### User-visible changes worth noting

1. **`sp.continuous_did(method='att_gt')` result labels** —
   - ``result.method`` changed from
     `"Continuous DID (Callaway et al. 2024)"` to
     `"Continuous DID (dose-bin heuristic)"`.
   - ``result.estimand`` changed from
     `"ACRT (Average Causal Response on Treated)"` to
     `"Sample-weighted mean of dose-bin 2x2 DIDs (not CGS 2024 ATT(d|g,t))"`.
   - Why: the previous labels claimed paper fidelity with CGS (2024)
     that the implementation did not deliver. Numerical output is
     unchanged. If you were parsing these strings in a pipeline, update
     the matcher.
   - If you actually want a CGS (2024)-style estimator: the new
     `method='cgs'` is an **MVP** (2-period design, OR only) with
     paper formulas flagged `[待核验]`. See
     `docs/rfc/continuous_did_cgs.md`.

2. **`sp.did_multiplegt(dynamic=H)` semantic clarification** — the
   docstring now states explicitly that this is a pair-rollup
   extension, **not** the dCDH (2024) `did_multiplegt_dyn` estimator.
   Numerical output is unchanged; if you were using `dynamic=H` and
   calling it "dCDH 2024", switch to the new `sp.did_multiplegt_dyn`
   (also MVP — see `docs/rfc/multiplegt_dyn.md`).

### New functions (no migration needed, just additive)

`sp.lp_did`, `sp.ddd_heterogeneous`, `sp.did_timevarying_covariates`,
`sp.did_multiplegt_dyn` (MVP), `sp.continuous_did(method='cgs')` (MVP).

### Bib key updates

`paper.bib` entry `dechaisemartin2022fixed` upgraded from SSRN to the
published *Econometrics Journal* 26(3):C1–C30 (2023) version. Any
downstream uses of the bib key via `[@dechaisemartin2022fixed]` are
unaffected; the expanded citation will now render to the journal
version.

---

## v1.5.x → agent-native infrastructure (Unreleased)

Pure-additive release. **No migration required** for existing code.
New agent-native surface area documented here for adopters.

### 1. Exception taxonomy (new public module)

```python
from statspai.exceptions import (
    AssumptionViolation, IdentificationFailure,
    DataInsufficient, ConvergenceFailure,
    NumericalInstability, MethodIncompatibility,
)
```

Domain errors subclass the right stdlib base (`ValueError` /
`RuntimeError`), so existing `try / except ValueError` blocks still
catch `AssumptionViolation` and `DataInsufficient`, and
`except RuntimeError` still catches `ConvergenceFailure` and
`NumericalInstability`. No call-site changes required.

New code should prefer the specific subclass + attach a
`recovery_hint`:

```python
raise AssumptionViolation(
    "Parallel trends rejected at p=0.003",
    recovery_hint="Run sp.sensitivity_rr for Rambachan-Roth honest CI.",
    diagnostics={"test": "pretrends", "pvalue": 0.003},
    alternative_functions=["sp.sensitivity_rr", "sp.callaway_santanna"],
)
```

### 2. Agent-native result methods

- `result.violations()` — structured list of assumption /
  diagnostic issues with `severity` / `recovery_hint` / `alternatives`.
- `result.to_agent_summary()` — JSON-ready structured payload.
- Complement (do not replace) existing `summary()` / `tidy()` /
  `next_steps()`.

### 3. Registry agent cards

- `sp.agent_card(name)` — full metadata including pre-conditions,
  assumptions, failure modes with recovery hints, ranked
  alternatives, typical minimum N.
- `sp.agent_cards(category=None)` — bulk export of entries that
  have at least one agent-native field populated (currently:
  `regress`, `iv`, `did`, `callaway_santanna`, `rdrobust`, `synth`).

### 4. Guide `## For Agents` blocks

Run `python scripts/sync_agent_blocks.py` after any change to a
registered spec's agent-native fields. The `--check` flag is
CI-friendly and fails non-zero on drift.

---

## v1.4.x → v1.5.0

Minor release.  Only one change requires any migration:

### `sp.mr` is now a dispatcher function, not a module alias

Before v1.5.0, `sp.mr` was a reference to the `statspai.mendelian`
submodule, and `sp.mr.mr_ivw(...)` worked as attribute access on the
module.

In v1.5.0, `sp.mr` is the new **unified dispatcher** for the MR family,
matching the pattern of `sp.synth` / `sp.decompose` / `sp.dml`:

```python
sp.mr("ivw",   beta_exposure=bx, beta_outcome=by,
       se_exposure=sx, se_outcome=sy)
sp.mr("egger", beta_exposure=bx, beta_outcome=by,
       se_exposure=sx, se_outcome=sy)
sp.mr("mvmr",  snp_associations=snp_df,
       outcome="beta_y", outcome_se="se_y",
       exposures=["beta_bmi", "beta_ldl"])
```

| Old (<= v1.4.2) | New (>= v1.5.0) |
| --- | --- |
| `sp.mr.mr_ivw(...)` | `sp.mr_ivw(...)` (already available since v0.9) or `sp.mr("ivw", ...)` |
| `sp.mr.mr_egger(...)` | `sp.mr_egger(...)` or `sp.mr("egger", ...)` |
| `sp.mr.mr_presso(...)` | `sp.mr_presso(...)` or `sp.mr("presso", ...)` |
| `sp.mr` (as module alias) | `sp.mendelian` (module access preserved under this name) |

**Rule of thumb:** if your code uses `sp.mr_*` (underscore form) it
already works unchanged in v1.5.0.  Only the uncommon
`sp.mr.<attribute>` pattern needs rewriting.

### Output numerical differences you may notice after upgrading

- `sp.mr_egger` / `sp.mendelian_randomization(..., methods=["egger"])`
  slope p-values and CIs now use `t(n − 2)` rather than `Normal`, matching
  `sp.mr_pleiotropy_egger` and R's `MendelianRandomization` package.
  Effect is invisible for `n_snps ≥ ~100`.  For very small `n_snps` (say
  5 or 6) CIs widen by ~1.6×.
- `sp.mr_presso` p-values now use the `(k + 1) / (B + 1)` MC convention,
  so they are strictly positive (floor `1 / (B + 1)`).  No change for
  non-extreme cases; fixes `-inf` propagation through `log(p)` downstream.

---

## From PyStataR to StatsPAI

`PyStataR` is deprecated. All of its functionality is now available in
[StatsPAI](https://github.com/brycewang-stanford/StatsPAI), under a
unified `sp.*` namespace.

```bash
pip install statspai
```

```python
import statspai as sp
```

## API mapping

| PyStataR | StatsPAI |
|---|---|
| `pdtab.tab1(df, 'x')` / `tab2(df, 'x', 'y')` | `sp.tab(df, 'x')` / `sp.tab(df, 'x', 'y')` |
| `pywinsor2.winsor2(df, ['x'], cuts=(1,99))` | `sp.winsor(df, ['x'], cuts=(1,99))` |
| `pywinsor2.outlier_indicator(df, ['x'])` | `sp.outlier_indicator(df, ['x'])` |
| `pyoutreg.outreg(models, 'out.xlsx')` | `sp.outreg2(models, filename='out.xlsx')` |
| `pyegen.rowmean(df, ['x1','x2'])` | `sp.rowmean(df, ['x1','x2'])` |
| `pyegen.rowtotal(df, ['x1','x2'])` | `sp.rowtotal(df, ['x1','x2'])` |
| `pyegen.rowmax/rowmin(df, [...])` | `sp.rowmax(df, [...])` / `sp.rowmin(df, [...])` |
| `pyegen.rowsd(df, [...])` | `sp.rowsd(df, [...])` |
| `pyegen.rownonmiss(df, [...])` | `sp.rowcount(df, [...])` |
| `pyegen.rank(df, 'x', by='g')` | `sp.rank(df, 'x', by='g')` |

## Why migrate

- **One package, one namespace.** `sp.*` covers everything PyStataR did,
  plus DID, RD, synthetic control, IV, matching, DML, causal forest,
  meta-learners, and more.
- **Actively maintained.** PyStataR is frozen; new features land only in
  StatsPAI.
- **Cleaner naming.** No "Stata" in the name — StatsPAI is Python-native.

## Questions

Open an issue on
[StatsPAI/issues](https://github.com/brycewang-stanford/StatsPAI/issues).
