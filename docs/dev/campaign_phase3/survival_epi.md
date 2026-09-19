# Phase 3 — survival / epidemiology / smoothing family (`survival_epi`)

Worktree `wt/pc-survival-epi` (base `d127e7d8`). Everything below is left
uncommitted. One test file carries every comparison:
`tests/reference_parity/test_survival_epi_R_parity.py` (49 tests, all pass).
It reads `_fixtures/survival_epi_R.json` and `_fixtures/survival_epi_stata.json`.
Those two files are written by `_generate_survival_epi_R.R` and
`_fixtures/_generate_survival_epi_stata.do`. Both scripts read the CSVs that
`_fixtures/_generate_survival_epi_data.py` writes. The data script and the R
script were re-run and produced byte-identical files.

Reference versions: R 4.5.2 (2025-10-31); survival 3.8.3, cmprsk 2.2.12,
epitools 0.5.10.1, DescTools 0.99.60, pROC 1.19.0.1, epiR 2.0.94. Stata 18 MP
(born 07 Jun 2023). SSC packages stcompet 1.0.7 (06nov2012) and diagt 2.032
(diagti 2.053) are installed only in the private ado directory
`tests/reference_parity/_fixtures/_ado_survival_epi/`.

## 1. Per-function outcome

Outcome classes: 1 = aligned; 2 = defect found and fixed, now aligned;
3 = documented convention difference; 6 = no reference computes the same
quantity. "Max rel err" is the largest relative error observed on the
fixture data.

| function | reference | class | max rel err est / SE | test |
| --- | --- | --- | --- | --- |
| `cuminc` (CIF, variances, CIs) | cmprsk::cuminc 2.2.12; Stata stcompet 1.0.7 | 2 | est 6e-16 / var(gray) 3e-13, se(delta) 2e-16, log-log CI 2e-16 | test_survival_epi_R_parity.py |
| `cuminc` Gray's test | cmprsk::cuminc `$Tests` (rho 0 and 1, 2 and 3 groups) | 2 | stat 9e-15, p 2e-14 | same |
| `finegray` | cmprsk::crr 2.2.12; Stata stcrreg | 2 | R: coef 6e-15 / robust SE 6e-15 / model SE 3e-15 / loglik 0; Stata: coef 5e-9 (Stata's ml stop) / SE 1.4e-10 / ll 1e-16 | same |
| `cox_frailty` | survival::coxph(frailty(theta=, sparse=FALSE)) 3.8.3; Stata stcox, shared() | 2 | fixed theta: beta 2e-14 / SE 4e-14 / log-frailties 1e-11 / integrated loglik 6e-16. Theta maximiser: 5e-8 vs R optimize, 7.5e-6 vs Stata e(theta) | same |
| `direct_standardize` | epitools::ageadjust.direct 0.5.10.1; Stata dstdize | 1 with new options; default CI is 6 | rate 0 / CI 2e-16 (epitools gamma), SE 3e-13 and CI 4e-14 (dstdize) | same |
| `indirect_standardize` | Stata istdize (exact CI); epitools::ageadjust.indirect (log-normal CI) | 1 (+ option) | SMR 0 / expected 0 / CI 6e-16 | same |
| `breslow_day_test` | DescTools::BreslowDayTest 0.99.60 (correct = F/T); Stata cc, by() bd tarone | 1 | stat and p 1.3e-14; M-H OR 4e-16 | same |
| `roc_curve` | pROC 1.19.0.1 (auc, var and ci.auc with DeLong); Stata roctab (default and hanley) | 2 | AUC 1e-15; DeLong var 2e-16; empirical Hanley SE 6e-15 | same |
| `auc` | pROC::auc; Stata roctab | 2 | 2e-16 | same |
| `sensitivity_specificity` | epiR::epi.tests 2.0.94 (wilson, exact); Stata diagti 2.053 | 1 (+ exact option) | 2e-15 | same |
| `diagnostic_test` | alias of the above | 1 | 2e-15 | same |
| `kdensity` | Stata kdensity (5 kernels, default width); R bw.nrd0, bw.SJ | 2 | density 9e-16; bw.nrd0 0; Stata width 4e-16; SJ 2e-7 vs bw.SJ(nb=1e7, tol=1e-14) | same |
| `lpoly` | Stata lpoly (fit and se() with pwidth) | 1 (+ Stata SE option) | fit 1.6e-12 / SE 1e-12 | same |
| `power_case_control` | Stata power twoproportions; R stats::power.prop.test(strict=TRUE) | 1 with `test="chi2"`; default Wald is 6 | 4e-13 (Stata normal CDF) | same |
| `power_ols` | pwr::pwr.f2.test is the nearest; not the same quantity | 6 | not compared (see section 5) | none |
| `cox` (`ties="breslow"`; found on the way, not on the list) | survival::coxph(ties=), Stata stcox (breslow, efron) | 2 | coef 1e-9 (sp.cox stops Newton at a 1e-9 absolute step) / SE 1e-9 / loglik 1e-15 | same |

## 2. Defects

Each defect below was found by comparing on identical bytes. None was caught
by the existing tests: the old tests are direction and range checks, or
closed-form checks on tie-free data. The before and after numbers come from
the fixture data. Before = `git archive HEAD`; after = this worktree.

1. **`cuminc` delta-method variance used the wrong cumulative-incidence
   difference.** The docstring cites the Marubini-Valsecchi estimator, whose
   terms use `F(t) - F(t_j)`. The code used `F(t) - F(t_{j-1})`. First
   divergence: already at the first event time, where the variance must be
   the binomial `p(1-p)/n`. There the old value was 0.000228, against
   stcompet's 0.000282. Fix: use `F(t) - F(t_j)`. Cause-1 SE at
   t = 0.25 / 2.0 / 7.0 went from 0.015107 / 0.027944 / 0.032854 to
   0.016795 / 0.030489 / 0.035325 (stcompet agrees to 2e-16). Added
   `variance="gray"` (cmprsk's variance) and `conf_type="log-log"`
   (stcompet's bounds). **Default output changed (SE and CI).**
2. **Gray's test was a different test under the right name.** It was a
   log-rank on unweighted subdistribution risk sets with a hypergeometric
   variance. The unweighted risk set agrees with Gray's `R_k = n_k(1 -
   F_k(t-))/S_k(t-)` only when there is no censoring, and the variance never
   agrees with Gray's. Rewritten to Gray's score and asymptotic covariance,
   with a new `rho=` argument. Cause 1 went from chi2 = 2.3950 (p = 0.1217)
   to 1.8499 (p = 0.1738); cause 2 from 0.5354 to 0.7835. **Default output
   changed.**
3. **`finegray` weights used the right-continuous censoring KM.** The
   weights were `Ĝ(t)/Ĝ(T_i)`; cmprsk and stcrreg both use left limits,
   `Ĝ(t-)/Ĝ(T_i-)`. The two differ whenever a censoring ties an event time,
   which it does here. First divergence: the weight matrix. Coefficients
   went from (0.440740, 0.270487, 0.282621) to (0.440603, 0.270658, 0.281948)
   (crr agrees to 6e-15). **Default output changed.**
4. **`finegray` standard errors were the inverse information.** Fine and
   Gray's sandwich, the one with the censoring-estimation term (cmprsk
   `var`), was not implemented. It is now the default, `vce="robust"`.
   `small_sample=True` applies stcrreg's N/(N-1). `vce="model"` keeps the
   old quantity (crr `invinf`). SEs went from (0.084511, 0.168711, 0.168777)
   to (0.081090, 0.159197, 0.157572). **Default output changed.**
5. **`cox_frailty` returned the ordinary Cox fit.** The beta step optimised
   the plain partial likelihood and ignored the frailty offset. Theta was
   searched on [0.5, 100] of a likelihood that was not the gamma marginal.
   The SEs came from the plain Cox information. Rewritten:
   - For fixed theta, a full Newton fit of the gamma-penalised partial
     likelihood in (beta, w). The derivatives are vectorised by cluster, so
     n = 2000 with G = 200 takes 7 s (a first dense version took 159 s).
   - Theta maximises the integrated likelihood: partial likelihood minus
     the penalty, plus the closed-form gamma correction. This is R's
     `c.loglik` and Stata's `e(ll)`.
   - SEs come from the inverse of the full penalised information.
   - New options: `theta=` (fixed variance, 0 = Cox) and `ties=`. New
     outputs: `lr_theta0` (chibar2), `log_frailties`, `loglik_cox`.
   - `.theta` is now the frailty **variance**, as in R and Stata. It was
     previously documented as a precision. It hit its 0.5 bound here.

   Fixture data: beta went from (0.480666, -0.506041) to (0.528795,
   -0.504217); SE from (0.099537, 0.176958) to (0.105516, 0.187037); theta
   from 0.500004 (precision, at the bound) to 0.240201 (variance). Stata:
   0.240199, chibar2 6.30 on both sides. **Default output changed; `.theta`
   changed meaning.**
6. **`roc_curve` / `auc` were wrong with tied scores.** The curve stepped
   one observation at a time in sort order, so a tie between a case and a
   control counted 0 or 1 depending on an unstable argsort. On one simulated
   draw the result was 0.6694 against the Mann-Whitney 0.6672. The curve is
   now traced over distinct scores, and the AUC is the placement-value
   Mann-Whitney with ties counted as one half. On the fixture's tied score
   the AUC went from 0.68376068 to 0.68389966 (pROC and roctab agree to
   2e-16). `thresholds`, `tpr` and `fpr` now have one entry per distinct
   score (previously one per observation). Added
   `se_method="delong"|"hanley-empirical"`. **Default output changed when
   scores tie.**
7. **`kdensity(bw_method="sheather-jones")` returned the Silverman rule.**
   Its body was literally `0.9*a*n^(-1/5)`. It is now the Sheather-Jones
   solve-the-equation rule without binning: 0.5222 went to 0.5592, which is
   2e-7 from R `bw.SJ(nb=1e7, tol=1e-14)` and 1.8e-3 from R's binned default.
   **Output changed.**
8. **`kdensity`'s default Silverman width matched neither reference it
   claims.** It combined R's quantile type with Stata's 1.349 constant. It is
   now exactly R `bw.nrd0` (1.34). `bw_method="stata"` reproduces Stata
   kdensity's width, including Stata's percentile rule. On the fixture
   0.522176 went to 0.525683 (+0.67%). **Default output changed.** This is a
   convention choice between two constants, not an error. It is changed
   because the default kernel is R's default and the old value matched no
   reference.
9. **`sp.cox(ties="breslow")` silently ran Efron.** The code said so in a
   comment: "Breslow currently reuses the Efron helper". The Efron helpers
   now take `breslow=`. On the fixture's no-competing-risk data, x1 went
   from 0.355720 to 0.335603 (coxph and stcox Breslow agree to 1e-9).
   **Output changed for `ties="breslow"` with tied times.** Found through
   the Fine-Gray identity test (no competing events means the Breslow Cox
   model).
10. Minor fixes with no default change:
    - `power_case_control`'s sample-size search only stepped upwards from
      the Wald closed form. It now also steps down, which matters for the
      new `test="chi2"`.
    - `breslow_day_test` silently dropped strata with a zero fitted cell. It
      now warns.
    - `indirect_standardize`'s comment called the doubled exact tail
      "mid-p". It is not mid-p; the comment is corrected.

**License note for the owner.** The Gray-test covariance, cmprsk's `cinc`
variance and crr's sandwich (`crrvv`) were re-implemented from the formulas
in cmprsk's Fortran source, which is GPL-2. The code is new, vectorised numpy
written from those formulas, not a line-by-line translation. Please judge
whether this needs a provenance note.

## 3. Proposed promotion records (`scripts/build_parity_index.py::_FROZEN_PROMOTIONS`)

```python
_SE_TEST = [
    "tests/reference_parity/test_survival_epi_R_parity.py",
    "tests/reference_parity/_fixtures/survival_epi_R.json",
    "tests/reference_parity/_fixtures/survival_epi_stata.json",
]
_SE_R = {"R": "R version 4.5.2 (2025-10-31)"}
{
    "cuminc": {
        "status": "bit-exact",
        "reference": "R cmprsk::cuminc (estimate, var, Tests); Stata stcompet (ci, se, hi, lo)",
        "reference_versions": {**_SE_R, "cmprsk": "2.2.12", "Stata": "18 MP", "stcompet": "1.0.7 (06nov2012)"},
        "tolerance": "1e-10 rel (observed: CIF 6e-16, Gray variance 3e-13, delta SE 2e-16, Gray test 9e-15)",
        "sides": ["py", "R", "Stata"],
        "test": _SE_TEST,
        "note": (
            "variance='gray' is cmprsk's asymptotic variance; variance='delta' (default) is the "
            "Marubini-Valsecchi delta method stcompet reports, conf_type='log-log' its bounds. "
            "Gray's test is cmprsk's, unstratified, rho 0 and 1, two and three groups. "
            "Tied event and censoring times in the fixture. Regenerate via "
            "_generate_survival_epi_R.R / _generate_survival_epi_stata.do."
        ),
    },
    "finegray": {
        "status": "bit-exact",
        "reference": "R cmprsk::crr (coef, var, invinf, loglik); Stata stcrreg",
        "reference_versions": {**_SE_R, "cmprsk": "2.2.12", "Stata": "18 MP"},
        "tolerance": (
            "R 1e-10 rel (observed 6e-15); Stata coefficients 1e-7 and SEs 1e-8 "
            "(Stata's ml stops 5e-9 away), log likelihood 1e-10"
        ),
        "sides": ["py", "R", "Stata"],
        "test": _SE_TEST,
        "note": (
            "Censoring KM at left limits, as crr and stcrreg. vce='robust' (default) is crr's "
            "var; small_sample=True is stcrreg's N/(N-1) scaling; vce='model' is crr's invinf. "
            "Both causes. Breslow ties."
        ),
    },
    "cox_frailty": {
        "status": "aligned",
        "reference": "R survival::coxph(... + frailty(id, theta=, sparse=FALSE)); Stata stcox, shared()",
        "reference_versions": {**_SE_R, "survival": "3.8.3", "Stata": "18 MP"},
        "tolerance": (
            "fixed theta: beta and SE 1e-9, integrated log likelihood 1e-10 (observed 4e-14); "
            "theta maximiser 1e-6 vs R optimize (observed 5e-8) and 5e-5 vs Stata e(theta) "
            "(observed 7.5e-6)"
        ),
        "sides": ["py", "R", "Stata"],
        "test": _SE_TEST,
        "note": (
            "Bit-exact at a fixed theta, including at Stata's own theta. Aligned rather than "
            "bit-exact because theta is compared as the maximiser of a flat integrated "
            "likelihood; ours attains a log likelihood at least as high as Stata's. "
            "R's default sparse=TRUE / method='em' fit is not the comparison target."
        ),
    },
    "direct_standardize": {
        "status": "bit-exact",
        "reference": "R epitools::ageadjust.direct; Stata dstdize",
        "reference_versions": {**_SE_R, "epitools": "0.5.10.1", "Stata": "18 MP"},
        "tolerance": "1e-10 rel (observed 3e-13)",
        "sides": ["py", "R", "Stata"],
        "test": _SE_TEST,
        "note": (
            "ci_method='gamma' (Poisson variance) is epitools; ci_method='normal' with "
            "variance='binomial' is dstdize. The default lognormal interval has no package "
            "reference; the rate itself is pinned to both."
        ),
    },
    "indirect_standardize": {
        "status": "bit-exact",
        "reference": "Stata istdize (exact CI); R epitools::ageadjust.indirect (log-normal CI)",
        "reference_versions": {**_SE_R, "epitools": "0.5.10.1", "Stata": "18 MP"},
        "tolerance": "1e-10 rel (observed 6e-16)",
        "sides": ["py", "R", "Stata"],
        "test": _SE_TEST,
        "note": "ci_method='exact' (default) is istdize; 'lognormal' is epitools. p_value not referenced.",
    },
    "breslow_day_test": {
        "status": "bit-exact",
        "reference": "R DescTools::BreslowDayTest (correct=FALSE/TRUE); Stata cc, by() bd tarone",
        "reference_versions": {**_SE_R, "DescTools": "0.99.60", "Stata": "18 MP"},
        "tolerance": "1e-10 rel (observed 1.3e-14)",
        "sides": ["py", "R", "Stata"],
        "test": _SE_TEST,
        "note": "Four strata, one with small cells; Mantel-Haenszel common OR on all three sides.",
    },
    "roc_curve": {
        "status": "bit-exact",
        "reference": "R pROC::roc/auc/var/ci.auc (DeLong); Stata roctab (default and hanley)",
        "reference_versions": {**_SE_R, "pROC": "1.19.0.1", "Stata": "18 MP"},
        "tolerance": "1e-10 rel (observed 6e-15)",
        "sides": ["py", "R", "Stata"],
        "test": _SE_TEST,
        "note": (
            "Continuous and heavily tied scores. se_method='delong' is pROC and roctab's "
            "default; 'hanley-empirical' is roctab, hanley. The default 'hanley' (exponential "
            "Q1/Q2 approximation) has no package reference."
        ),
    },
    "auc": {
        "status": "bit-exact",
        "reference": "R pROC::auc; Stata roctab",
        "reference_versions": {**_SE_R, "pROC": "1.19.0.1", "Stata": "18 MP"},
        "tolerance": "1e-10 rel (observed 2e-16)",
        "sides": ["py", "R", "Stata"],
        "test": _SE_TEST,
        "note": "Ties counted one half; also asserted against the mid-rank Mann-Whitney identity.",
    },
    "sensitivity_specificity": {
        "status": "bit-exact",
        "reference": "R epiR::epi.tests (method wilson / exact); Stata diagti",
        "reference_versions": {**_SE_R, "epiR": "2.0.94", "Stata": "18 MP", "diagt": "2.032 (diagti 2.053)"},
        "tolerance": "1e-12 rel on intervals, 1e-10 on ratios (observed 2e-15)",
        "sides": ["py", "R", "Stata"],
        "test": _SE_TEST,
        "note": "Three tables, one with a zero cell. ci_method='wilson' (default) or 'exact'.",
    },
    "diagnostic_test": {
        "status": "bit-exact",
        "reference": "R epiR::epi.tests; Stata diagti",
        "reference_versions": {**_SE_R, "epiR": "2.0.94", "Stata": "18 MP", "diagt": "2.032 (diagti 2.053)"},
        "tolerance": "1e-12 rel (observed 2e-15)",
        "sides": ["py", "R", "Stata"],
        "test": _SE_TEST,
        "note": "Alias of sensitivity_specificity; the Stata block calls sp.diagnostic_test.",
    },
    "kdensity": {
        "status": "bit-exact",
        "reference": "Stata kdensity (at(), bwidth(), kernel()); R bw.nrd0 / bw.SJ",
        "reference_versions": {**_SE_R, "Stata": "18 MP"},
        "tolerance": (
            "density and default widths 1e-10 rel (observed 9e-16); Sheather-Jones 1e-6 vs "
            "bw.SJ(nb=1e7, tol=1e-14) (observed 2e-7, R's pair-count binning)"
        ),
        "sides": ["py", "R", "Stata"],
        "test": _SE_TEST,
        "note": (
            "Kernels epanechnikov (= Stata epan2), biweight, gaussian, uniform (= rectangle), "
            "triangular (= triangle). bw_method='silverman' is bw.nrd0; 'stata' is kdensity's "
            "default width. The cosine kernel is not Stata's and is not compared."
        ),
    },
    "lpoly": {
        "status": "bit-exact",
        "reference": "Stata lpoly (at(), bwidth(), degree(), kernel(), se(), pwidth())",
        "reference_versions": {"Stata": "18 MP"},
        "tolerance": "1e-10 rel (observed 1.6e-12)",
        "sides": ["py", "Stata"],
        "test": _SE_TEST,
        "note": (
            "Degrees 0/1/2, kernels epan2 / gaussian / biweight, fits and se_method='stata' "
            "SEs with pwidth. Stata's rule-of-thumb bandwidth is not implemented; the default "
            "robust SE is not Stata's."
        ),
    },
    "power_case_control": {
        "status": "bit-exact",
        "reference": "Stata power twoproportions; R stats::power.prop.test(strict=TRUE)",
        "reference_versions": {**_SE_R, "Stata": "18 MP"},
        "tolerance": "1e-11 rel vs Stata (observed 4e-13, Stata's normal CDF); 1e-10 vs R (observed 6e-16)",
        "sides": ["py", "R", "Stata"],
        "test": _SE_TEST,
        "note": (
            "test='chi2' (pooled null, both tails), 1:1 to 1:3 allocation, one- and two-sided. "
            "The default test='wald' has no package reference."
        ),
    },
}
```

`cox` is already Track A (24_coxph, Efron). If you want the Breslow path on
record, its note can gain: "ties='breslow' pinned to coxph(ties='breslow')
and stcox (default) at 1e-8 in test_survival_epi_R_parity.py".

## 4. CHANGELOG / MIGRATION

**⚠️ Correctness** (proposed lines):

- `sp.cuminc`: the delta-method CIF variance used `F(t) - F(t_{j-1})`
  instead of `F(t) - F(t_j)`. SEs and CIs are now Stata `stcompet`'s. Gray's
  test is now Gray's test, matching R `cmprsk::cuminc` to 1e-14. It had been
  a log-rank on unweighted subdistribution risk sets.
- `sp.finegray`: the censoring KM is evaluated at left limits, as `cmprsk::crr`
  and `stcrreg` do, which changes coefficients when censorings tie event
  times. Standard errors now default to Fine and Gray's sandwich
  (`vce="robust"`); the old inverse information is `vce="model"`.
- `sp.cox_frailty`: rewritten. It had returned the ordinary Cox fit.
  Coefficients, SEs and the integrated likelihood now match R
  `coxph(frailty(sparse=FALSE))` and Stata `stcox, shared()`. `.theta` is now
  the frailty variance (as in R and Stata); it used to be documented as a
  precision.
- `sp.roc_curve` / `sp.auc`: tied scores are counted one half (Mann-Whitney);
  the AUC used to depend on the sort order of tied cases and controls. The
  curve arrays now have one entry per distinct score.
- `sp.kdensity`: `bw_method="sheather-jones"` returned the Silverman rule; it
  now computes Sheather-Jones. The default `"silverman"` width is now exactly
  R `bw.nrd0` (IQR/1.34, a 0.67% change when the IQR binds).
- `sp.cox(ties="breslow")` ran Efron's approximation; it now runs Breslow's.

**Added:** `cuminc(variance=, conf_type=, rho=)`; `finegray(vce=, small_sample=)`;
`cox_frailty(theta=, ties=)` and `.lr_theta0`, `.log_frailties`, `.loglik_cox`;
`direct_standardize(ci_method=, variance=)` and `.se`;
`indirect_standardize(ci_method=)`;
`roc_curve(se_method="delong"|"hanley-empirical")`;
`sensitivity_specificity(ci_method="exact")`;
`kdensity(bw_method="stata"|"nrd0")`; `lpoly(se_method="stata", pwidth=)`;
`power_case_control(test="chi2")`. The cross-language parity file is
`test_survival_epi_R_parity.py`.

**Fixed:** `breslow_day_test` warns when it drops a stratum;
`power_case_control`'s sample-size search now finds the minimum from above.

**MIGRATION rows:**

| function | change | old behaviour back |
| --- | --- | --- |
| `cuminc` | default SE/CI (delta-method formula fixed); Gray statistic | none (old values were wrong) |
| `finegray` | coefficients with censoring/event ties; default SE robust | `vce="model"` for the old SE; old weights not kept (wrong) |
| `cox_frailty` | all outputs; `.theta` now variance | none (old fit ignored the frailty); for a precision use `1/res.theta` |
| `roc_curve` / `auc` | AUC with ties; `thresholds/tpr/fpr` length = distinct scores | none |
| `kdensity` | default width = `bw.nrd0`; `sheather-jones` real | `bw_method="stata"` is 1.349 with Stata's percentiles |
| `cox` | `ties="breslow"` now Breslow | `ties="efron"` is what it used to run |

## 5. Not closed / not comparable

- **`power_ols` (class 6).** The docstring defines neither "standardised
  effect" nor a test. The code is a normal approximation,
  `Phi(es sqrt(n-k-1)/sqrt(1-R2) - z)`. The nearest canonical quantity is
  `pwr::pwr.f2.test(u=1, v=n-k-2, f2=es^2/(1-R2))`, a noncentral F with
  `lambda = f2 n`. At es = 0.3, k = 2, R2 = 0.2 they give 0.4141 vs 0.4012
  (n = 30), 0.6329 vs 0.6237 (n = 50) and 0.99700 vs 0.99689 (n = 200). The
  gap is the approximation, not a bug. Not edited:
  **`src/statspai/power/power.py` is conflict-prone** (another line is
  editing it).
- **`power_case_control` default `test="wald"` (class 6).** No package
  computes power for the unpooled Wald test. Stata, `power.prop.test` and
  epiR `epi.sscc` all use the pooled null. The `chi2` option is pinned; the
  default is left alone for the owner to decide.
- **Default intervals with no package reference (class 6):**
  - `direct_standardize`'s log-normal CI.
  - `roc_curve`'s exponential Hanley-McNeil SE (Stata's `hanley` estimates
    Q1 and Q2 from the data).
  - `indirect_standardize`'s doubled exact-tail p. R `poisson.test` uses the
    minimum-likelihood rule instead.

  Worth deciding whether the defaults should become DeLong / gamma. Not
  changed here, to avoid a silent default change.
- **`kdensity`:**
  - The `cosine` kernel is the "optimal cosine" on [-1, 1]. Stata's `cosine`
    is `1 + cos(2 pi z)` on |z| < 1/2, and R's is scaled to unit variance.
    It is not compared.
  - Stata's default `epanechnikov` (sqrt(5) support) and `parzen` are not
    offered.
  - R `density()` values are FFT-binned, so they are not a reference for
    density values; only R's bandwidths are compared.
  - `bw_method="stata"` is unweighted only; it raises `NotImplementedError`
    with weights.
- **`lpoly`:**
  - Stata's default bandwidth (Fan-Gijbels rule of thumb) and hence its
    default pilot width (1.5 x ROT) are not implemented. Back-solving the
    kernel constant from Stata's output gave values that did not match the
    textbook constants (Stata's ROT ratio gaussian/epan2 = 0.45157 vs 0.45171
    from the exact constants), so this is left open, not guessed.
  - `se_method="stata"` therefore requires `pwidth=`.
  - `KernSmooth::locpoly` is a binned estimator and was not compared.
- **Lead, not investigated (outside this family):**
  `survival/models.py::_cox_score_individual` computes Breslow score
  residuals regardless of `ties`, and contains dead code. `sp.cox(robust=...)`
  or `cluster=` with Efron ties and tied times may therefore differ from
  `coxph(robust=TRUE)`, which uses Efron residuals. Worth one comparison.
- **Scale note:** `finegray` builds an (event times x n) weight matrix. This
  is the same order of memory as before; n = 4000 takes 2.8 s with the
  robust SE.

## 6. For the integrator

- `.gitignore`: add `tests/reference_parity/_fixtures/_ado_survival_epi/`.
  It holds the private SSC install of stcompet and diagt; do not commit it.
  The Stata `.log` is already covered by `*.log`.
- **`src/statspai/registry.py`** has hand-written specs that now hide
  parameters. `tests/test_registry_param_drift.py` fails until these are
  added:
  - `direct_standardize`:
    `ParamSpec("ci_method", "str", False, "lognormal", enum=["lognormal", "gamma", "normal"])`,
    `ParamSpec("variance", "str", False, "poisson", enum=["poisson", "binomial"])`
  - `indirect_standardize`:
    `ParamSpec("ci_method", "str", False, "exact", enum=["exact", "lognormal"])`
  - `roc_curve`:
    `ParamSpec("se_method", "str", False, "hanley", enum=["hanley", "hanley-empirical", "delong"])`.
    Also change the description from "AUC (trapezoidal)" to "AUC
    (Mann-Whitney, ties = 1/2)".
  - `sensitivity_specificity`:
    `ParamSpec("ci_method", "str", False, "wilson", enum=["wilson", "exact"])`
- **Schemas:** regenerate with `scripts/dump_schemas.py`. `--check`
  currently reports `tools.json`, `functions.json` and `agent_cards.json`
  stale because of the new parameters.
- `tests/reference_parity/REFERENCES.md` / `R_PACKAGE_VERSIONS.md` need the
  new fixture row. New R packages were installed for this family: cmprsk
  2.2-12, coxme 2.2-22 (installed, unused) and epitools 0.5-10.1.
- **Files touched in `src/`:** `survival/competing_risks.py`,
  `survival/frailty.py`, `survival/models.py` (the Cox `ties` fix only),
  `epi/diagnostic.py`, `epi/standardize.py`, `epi/stratified.py`,
  `nonparametric/kdensity.py`, `nonparametric/lpoly.py` and
  `power/study_designs.py`. `power/power.py` is not touched.
- **Docs touched:** `docs/guides/competing_risks.md` (its "not
  parity-certified" note and SE limitation were replaced) and
  `docs/reference/survival.md` (`ties='exact'` was advertised but never
  supported; frailty example).
- **Tests run:**
  - The 35 test files that reference these functions, plus the new file:
    920 passed, 4 skipped.
  - Doctests of the 9 modified modules: 30 passed.
  - `test_registry_param_drift.py` fails only because of the registry
    entries listed above.
