# DiD ↔ Stata option-depth campaign — 3-month plan

Branch `wt/did-stata-depth`, worktree `.claude/worktrees/did-stata-depth`.
Origin: `芷涵改进建议/statspai改进建议-zhihan.md` (external review of StatsPAI's
DiD surface vs six Stata commands).

## 0. What the review got right, and what it got wrong

The review was audited item-by-item against the installed Stata ado help files
(`csdid` v1.81, `did_imputation` 2023-11-22, `did_had`, `did_multiplegt_dyn`,
`eventstudyinteract` 0.1) rather than against its own paraphrase. Three classes:

### 0.1 Already closed before the review was written

Commit `1507d30b` (2026-07-26) closed the review's entire P0 block:
`wboot`→`bstrap=`, `reps()`→`biters=`, `wbtype()`→`boot_weight_type=`,
`pointwise`→`cband=False`, `saverif()`→`influence_functions()` +
`aggte_from_influence()`, and the BJS option depth
(`pretrends=/balanced=/min_n=/hetby=/save_weights=/save_residuals=`).
`bacon_decomposition()` already returns `negative_weight_share`. `aggte()`
already covers all four csdid aggregations. **No work needed.**

### 0.2 Mis-transcribed by the review — the real option is something else

| Review's claim | What the ado help actually says |
| --- | --- |
| `asinr` = "as-if random assumption test" | Control-set convention for **pre-treatment** ATT(g,t) under not-yet-treated: R uses all cohorts not yet treated, Stata's default uses cohorts not treated at time G. Not a test at all. |
| `long`/`long2` = "unequal-length panel support" | Pre-treatment **base period**: long gaps vs short gaps. `long2` is explicitly "similar to base universal". This is StatsPAI's existing `base_period=`. |

Both were therefore scoped wrong in the review. `long`/`long2` is largely
already shipped; `asinr` turned into a correctness investigation (§WP-1).

### 0.3 Real gaps the review found, and real gaps it missed

Found and real: `did_had` (absent entirely), `stdipw`, `control_cohort`,
`unitcontrols`/`timecontrols`, `fe()`, `project()`, unified `se_method`.

**Missed by the review** (found by reading the help files):
`pscoretrim(#)` (default 0.995 "as in R"), `rc1` (non-locally-efficient RC
estimator), and the two-way clustering remark in csdid's SE section.

## 1. Work packages

| WP | Title | Kind | Gate |
| --- | --- | --- | --- |
| 0 | Worktree, plan, baseline | infra | DiD suite green at HEAD |
| 1 | CS not-yet-treated control set | ⚠️ correctness | Stata `csdid`/`asinr` agreement |
| 2 | `estimator='stdipw'` + `pscoretrim=` | feature | Stata `method(stdipw)` parity |
| 3 | `asinr=` / `long`/`long2` mapping | feature/docs | Stata parity both conventions |
| 4 | `sun_abraham(control_cohort=)` | feature | `eventstudyinteract` parity |
| 5 | BJS `unit_covariates`/`time_covariates`/`fe=` | feature | `did_imputation` parity |
| 6 | BJS `project()` | feature | `did_imputation` parity |
| 7 | `did_multiplegt` output options | feature | analytic |
| 8 | `sp.did_had` core estimator | feature | Stata `did_had` parity |
| 9 | `did_had` placebo/trends_lin/cumulative/QUG/Yatchew | feature | Stata `did_had` parity |
| 10 | Unified `se_method=` incl. `'auto'` | API | analytic + no numeric drift |
| 11 | `pretest=` in CS / sun_abraham | API | matches `pretrends_test()` |
| 12 | Parity fixtures, registry, schemas, docs, CHANGELOG | closeout | full suite green |

## 1a. `did_had` — verified citations and a corrected architecture

### Architecture correction

An early reading of the help text suggested `did_had` was built on
`rdrobust`, which StatsPAI already ships. **That is wrong.** The ado calls
`lprobust` from the **nprobust** package:

```stata
lprobust y_diff_XX treatment_1_XX, eval(grid_XX) kernel(`kernel') bwselect(`bw_method')
```

`lprobust` is local polynomial regression at an *evaluation point* with
MSE-DPI bandwidth selection and robust bias correction — a different
estimator from `rdrobust`'s boundary RD. `rd/_core.py` is still the right
home for the kernel/local-polynomial primitives, but the `bw_method`
family (`mse-dpi`, `imse-dpi`, `mse-rot`, `imse-rot`, `ce-dpi`, `ce-rot`)
and the RBC variance have to be built, not reused. Scope WP-8 accordingly.

### Citations (§10 — verified, two independent sources each)

| Ref | Status |
| --- | --- |
| de Chaisemartin, C., Ciccia, D., D'Haultfœuille, X., Knau, F. — *Difference-in-Differences Estimators When No Unit Remains Untreated* — **arXiv:2405.04465** | Verified via arXiv abstract page + package docs. **No journal publication**; latest version v7, July 2026. Cite as `arXiv preprint` per §10, NOT as "(2025)" the way the ado does. |
| Calonico, S., Cattaneo, M. D., Farrell, M. H. (2019). *nprobust: Nonparametric Kernel-Based Estimation and Robust Bias-Corrected Inference*. **JSS 91(8), 1–33**, doi:10.18637/jss.v091.i08 | Verified via jstatsoft.org + nppackages. |
| Calonico, S., Cattaneo, M. D., Farrell, M. H. (2018). *On the Effect of Bias Estimation on Coverage Accuracy in Nonparametric Inference*. **JASA 113(522), 767–779**; arXiv:1508.02973 | Verified via arXiv + rdpackages reference copy. DOI still to be pulled from Crossref before it enters `paper.bib`. |

### ⚠️ The ado's own Yatchew citation is wrong — do not copy it

`did_had.sthlp` writes:

> Yatchew, A (1997). … *Economics Letters*, Elsevier, vol. 62(3), pages 271–278.

That conflates two different papers:

- *An elementary estimator of the partial linear model* — Economics Letters
  **57(2)**, 135–143, **1997**.
- *An elementary nonparametric differencing test of equality of regression
  functions* — Economics Letters **62(3)**, 271–278, **1999**.

`did_had`'s `yatchew` option performs a **differencing test of linearity**,
so the intended reference is the **1999** paper; the ado's year is wrong.
Verified via RePEc author listing + ScienceDirect. Cite the 1999 entry and
do not propagate the ado's version.

## 1b. `did_had` — algorithm transcribed from the ado

Read off `did_had.ado` directly (not the help text), so WP-8 can be built
without re-deriving it.

### Setup

Heterogeneous-adoption design: **all** groups are untreated in period 1,
then every treated group receives a strictly positive dose at the *same*
period F. There is no untreated group — only "quasi-untreated" ones whose
dose is close to zero. With variation in timing instead, use
`did_multiplegt_dyn`.

Per event-study effect ℓ:

* `y_diff` = Y_{F-1+ℓ} − Y_{F-1}, the outcome evolution.
* `treatment_1` = dose at F-1+ℓ. With `dynamic`, the *cumulative* dose
  from F to F-1+ℓ instead (static vs dynamic normalization).

### Core estimator

```
lprobust y_diff  treatment_1, eval(0) kernel(...) bwselect(...)

mu_hat     = Result[1,5]              # conventional local-poly fit at D = 0
mu_hat_bc  = Result[1,6]              # bias-corrected
M_hat      = mu_hat - mu_hat_bc       # estimated bias
se_mu      = Result[1,8]              # robust (RBC) SE

beta_qs = (mean(y_diff) - mu_hat) / mean(treatment_1)
B_hat   = -M_hat / mean(treatment_1)          # bias term for the CI
se      = se_mu   / mean(treatment_1)
```

So the whole estimator is: *the average outcome evolution, minus what a
quasi-untreated group would have done (the local-polynomial intercept at
dose 0), rescaled by the average dose.* The only piece StatsPAI lacks is
`lprobust` — local polynomial at an evaluation point with MSE-DPI
bandwidth selection and CCF robust bias correction.

### QUG test (paper §3.3) — closed form, no bandwidth needed

```mata
D_np = sort(select(D, D :> 0), 1)      // positive doses, ascending
T    = D_np[1] / (D_np[2] - D_np[1])   // smallest dose / gap to the next
p    = 1 - (1 + 1/T)^(-1)  ==  1/(1 + T)
```

Both statistics converge to E₁/E₂ with iid Exponential(1), giving
CDF P(T ≤ t) = t/(1+t) and hence the p-value above. This is cheap and can
ship independently of the `lprobust` engine.

### `lprobust`'s `e(Result)` layout — the build target for WP-8a

`nprobust` is now installed locally (`net install nprobust, from(
https://raw.githubusercontent.com/nppackages/nprobust/master/stata)`),
which `did_had` requires and which was missing — so `did_had` could not
have run on this machine either.

`e(Result)` is a 1x10 row, and this is what `did_had.ado`'s hard-coded
indices mean:

| col | name | used by did_had as |
| ---: | --- | --- |
| 1 | `eval` | the evaluation point, always 0 |
| 2 | `h` | `h_star` |
| 3 | `b` | bias bandwidth |
| 4 | `N` | effective sample inside the bandwidth |
| 5 | `tau_us` | `mu_hat_XX_alt` — conventional fit |
| 6 | `tau_bc` | `mu_hat_XX_alt_ub` — bias-corrected |
| 7 | `se_us` | (unused) |
| 8 | `se_rb` | `se_mu_XX` — robust bias-corrected SE |
| 9,10 | `CI_l_rb`, `CI_r_rb` | coverage check |

So `M_hat = tau_us - tau_bc` is the estimated bias, and `did_had` builds
its CI from that rather than from `se_us`.

**Fixed-bandwidth reference to build against first.** Bandwidth
selection is a separate, larger piece; pinning the estimator at a
*supplied* `h` isolates it. On
`tests/stata_parity/option_parity/data_86_lprobust.csv` (n=400, dose from
a Gamma(1.4, 0.6), `y = 0.8 + 1.3d - 0.4d^2 + N(0, 0.5)`):

```stata
lprobust y d, eval(g0) h(0.8) b(0.8) kernel(epanechnikov)
```

| quantity | value |
| --- | ---: |
| `N` | 241 |
| `tau_us` | 0.748178965690396 |
| `tau_bc` | 0.796816090541766 |
| `se_us` | 0.073878309309036 |
| `se_rb` | 0.107051396374158 |

Build `_lprobust_at_point(x, y, x0, h, b, kernel, p)` against these four
numbers before touching bandwidth selection. The open question to settle
first is `lprobust`'s default variance estimator (`vce(nn)` vs the `hc`
family) — read it off the help rather than assuming, since it decides
whether `se_rb` is reproducible at all.

### Remaining WP-8/9 surface

`bw_method` ∈ {mse-dpi (default), mse-rot, imse-dpi, imse-rot, ce-dpi,
ce-rot}; `kernel` ∈ {epanechnikov (default), triangular, uniform,
gaussian}; `placebo(k)`; `trends_lin` (subtract each group's F-2→F-1
evolution, costing one placebo); `dynamic` (cumulative normalization);
`yatchew` (linearity test, Appendix E — **cite the 1999 paper, not the
ado's 1997**).

## 2. Standing rules for this campaign

- Every new estimator path gets a **Stata fixture** under
  `tests/stata_parity/` plus a committed `results/*_Stata.json`, wired into
  `compare.py`. No new "display only" legs — the budget + empty-join gates
  added in the JSS campaign apply.
- Every new public function is registered in `registry.py`, appears in
  `sp.list_functions()`, and carries NumPy docstring with `Examples`.
- No numeric output of an existing estimator moves without a
  ⚠️ correctness entry in `CHANGELOG.md` + `MIGRATION.md`.
- Citations: `did_had` needs de Chaisemartin et al. (2025), Calonico–Cattaneo–
  Farrell (2018, 2019), Yatchew (1997) verified against Crossref/arXiv before
  any of them enters a docstring. Bib keys only.
- Commit gate: nothing is committed or pushed without explicit authorization.
