# Causal-Inference Correctness Pass — 2026-06-21

A focused, **inference-correctness** lane, deliberately disjoint from the
concurrent `2026-06-21-statspai-hardening-month/` run (which owns quality
gates, lint/mypy ratchets, network anchors, and the Stata-migration
translator).

## Operating constraints honored

- **Initial handoff was no commit / no push.** This lane was first left in the
  working tree for review. It is now being published because the user later
  explicitly requested "split then changes then commit and push with codex
  signature."
- **No JOSS-review impact.** Did **not** touch `paper.md`, `paper.bib`,
  `CITATION.cff`, `.zenodo.json`, `release/`, `Paper-JSS/`, `docs/index.md`,
  or any GitHub Release. The one new reference (Imbens & Manski 2004) is
  written as a hand-verified canonical string in the source docstring, *not*
  added to `paper.bib`, to avoid perturbing the bibliography during review.
- **No collision with the parallel agent.** Edited only `timeseries/` and
  `bounds/` source (modules the other lane is not in) and added a **new**
  test file rather than editing existing tests that the test-coverage agent
  commits to.

## Files touched

| File | Change |
|---|---|
| `src/statspai/timeseries/structural_break.py` | CUSUM: linear BDE boundary (⚠️ correctness) |
| `src/statspai/bounds/lee_manski.py` | Lee bounds: genuine Imbens–Manski `C_n` CI (⚠️ correctness) + `_imbens_manski_cn` helper |
| `src/statspai/rd/_core.py` | RD HC sandwich: W² kernel weighting (⚠️ correctness) + `weights` param |
| `src/statspai/rd/hte.py`, `rd2d.py`, `bandwidth.py` | pass kernel weights to `_sandwich_variance` |
| `src/statspai/rd/bias_aware.py` | W²-consistent `_local_cov_yd` (keeps fuzzy SE PSD) |
| `src/statspai/matching/match.py` | honest `_ai_se` docstring + **JOSS-safe** default-SE guidance: `UserWarning` on the default `auto`→`'ai'` nearest path steering to `abadie_imbens`, stronger public docstring. **Default number unchanged** (locked `test_default_sp_match_se_is_ai_unchanged` still passes). |
| `tests/test_correctness_inference_fixes.py` | **new** — 10 regression + Monte-Carlo size/coverage tests |

---

## Fix 1 — `sp.cusum_test`: wrong CUSUM boundary → ~32% false-positive rate

**Bug.** The Brown–Durbin–Evans recursive-residual CUSUM compared the CUSUM
path against a **constant** critical value `1.358`. That value is the
`sup|Brownian bridge|` quantile belonging to the *OLS-CUSUM* (Ploberger &
Krämer 1992), a different test. The recursive CUSUM's crossing boundary is
**linear** in the recursion index:

```
|W_t| > a · [√m + 2 s / √m]   ⇔   |cusum_t| > a · [1 + 2 s/m],   s = 1..m,   m = n−k
```

with `a = 0.948` at the 5% level (1.143 @1%, 0.850 @10%). The constant boundary
over-rejected late breaks (true boundary widens to `3a` at the sample end) and
under-rejected early ones.

**Evidence (Monte-Carlo, n=120, B=1500, H0 = stable relation):**

| boundary | empirical size @ nominal 5% |
|---|---|
| old constant 1.358 | **0.318** |
| new linear BDE | **0.037** |

Power against a mean shift at t=60 is **1.000** (the recursive CUSUM's design
alternative). Low power against a *pure slope* change with mean-zero regressor
is the known Krämer–Ploberger–Alt (1988) inconsistency, not a regression — the
old "power" was just its broken 32% null rejection.

**API note.** `result["critical_value"]` is now the per-recursion boundary
*array* (was a scalar); the key set is unchanged
(`cusum, critical_value, max_cusum, n_obs, reject`). `reject` now reflects a
crossing anywhere along the path.

---

## Fix 2 — `sp.lee_bounds`: Horowitz–Manski CI mislabeled "Imbens–Manski"

**Bug.** The CI applied the two-sided `z_{1−α/2}` to *both* bound endpoints.
That is the Horowitz–Manski interval covering the identified **set**, which
**over-covers** the parameter — yet it was labelled "Imbens–Manski CI". The
genuine Imbens & Manski (2004) interval for the *parameter* uses a critical
value `C_n` solving

```
Φ(C_n + Δ/σ_max) − Φ(−C_n) = 1 − α,   Δ = ub − lb,   σ_max = max(se_lb, se_ub)
```

which interpolates between the one-sided `z_{1−α}` (wide bounds) and two-sided
`z_{1−α/2}` (point identification).

**Evidence (`_imbens_manski_cn`, α=0.05):**

| width/σ | C_n |
|---|---|
| 0.0 | 1.9600 (= z₀.₉₇₅) |
| 0.5 | 1.7697 |
| 2.0 | 1.6461 |
| ∞ | 1.6449 (= z₀.₉₅) |

The corrected CI is **narrower** (correct) than the old one whenever the bounds
have positive width, while still bracketing `[lb, ub]`.

**Reference (verified — CLAUDE.md §10).** Imbens, G. W. & Manski, C. F. (2004),
"Confidence Intervals for Partially Identified Parameters," *Econometrica*
72(6), 1845–1857, doi:10.1111/j.1468-0262.2004.00555.x. Verified via Crossref
API **and** RePEc/IDEAS + Google Scholar.

---

## Verification run

```
pytest tests/test_correctness_inference_fixes.py            -> 10 passed
pytest tests/test_new_v06_modules.py tests/test_phase9to14.py \
       tests/test_tierD_p2_bounds_sensitivity_analytic.py \
       tests/test_estimator_provenance_round8.py \
       --doctest-modules <both touched modules>            -> 98 passed
black --check <3 files>                                     -> clean
flake8 <3 files>                                            -> clean
```

---

## Hand-off: CHANGELOG / MIGRATION entries to add (NOT applied — avoid release-lane collision)

Both are **⚠️ correctness fixes** that change *inference output* (not point
estimates). Per CLAUDE.md §12 they need CHANGELOG + MIGRATION notes. I did not
edit those files because the parallel lane is doing release prep; paste these
when convenient:

**CHANGELOG.md → `### ⚠️ Correctness`**
- `sp.cusum_test`: the recursive-residual CUSUM now uses the Brown–Durbin–Evans
  *linear* crossing boundary `a·[1+2s/(n−k)]` (a=0.948 @5%) instead of a
  constant 1.358. The old boundary rejected ≈32% of stable series at a nominal
  5% level. `result["critical_value"]` is now the boundary array.
- `sp.lee_bounds`: the confidence interval is now the genuine Imbens–Manski
  (2004) interval for the partially identified parameter (critical value `C_n`),
  replacing a Horowitz–Manski set interval that over-covered. CIs are narrower.
- `sp.rdrobust` / `sp.rd2d` / RD HTE / `sp.rd_bias_aware_fuzzy`: the
  heteroskedasticity-robust local-polynomial variance now uses the
  Calonico–Cattaneo–Titiunik (2014) kernel weighting `X'W·diag(e²)·W·X`
  (weight **squared**). The previous meat carried only one power of the kernel
  weight, inflating every HC-robust RD standard error (≈1.4× for a uniform
  kernel vs R `rdrobust` vce="hc0"). Point estimates are unchanged; SEs now
  match R. (Cluster-robust RD SEs were already correct.)

**MIGRATION.md**
- `sp.cusum_test`: `critical_value` changed scalar → array; `reject` may differ
  (correct size). If you compared `max_cusum` to a hard-coded 1.358, switch to
  the returned `reject`.
- `sp.lee_bounds`: reported CI endpoints are narrower (correct coverage). No
  change to point bounds / midpoint estimate.
- RD module: HC-robust SEs/CIs/p-values are now smaller (correct) and match R
  `rdrobust`. Point estimates unchanged. Numbers published with prior versions
  using the conventional/robust HC variance were conservative (too wide).

---

## Background correctness audit (running)

Three read-only auditors swept DiD, IV/RD, and panel/synth/matching for *new*
numerical bugs (Monte-Carlo size/coverage + closed-form/R cross-checks). This
seeds the month-long "close the Stata/R gap by correctness, not feature count"
backlog.

### Fix 3 (applied this pass) — RD HC standard errors

The IV/RD auditor found the RD HC variance carried the kernel weight to the
**first** power (W¹) instead of squared (W²); see the CHANGELOG entry above and
the `_core.py` / `bias_aware.py` diffs. Verified against R `rdrobust`:
conventional SE 0.066234 → **0.046835** (R hc0 = 0.046684); the residual ~0.3%
is the documented HC1-vs-HC0 dof convention. All `cov95` RD coverage tests and
the bias-aware fuzzy SE pass after a symmetric W² fix to `_local_cov_yd`.

### Confirmed-but-deferred (ready-to-apply patches for the next sprint)

Ranked by "would a user publish a wrong number". All are reproduced; none are
applied live (each either changes a default/flagship number with a parity-test
blast radius best handled with fixture regeneration, or needs a real variance
reimplementation + R parity — both higher-collision than this pass should take).

| # | Sev | Site | Bug | Evidence | Fix |
|---|---|---|---|---|---|
| 1 | HIGH (interim ✓) | `matching/match.py:887` (`_resolve_se_method`) | `sp.match(method='nearest')` default SE is the naive matched-pair `std/√n` (`'ai'`); ignores control-reuse variance | nominal-95% coverage **0.81**, SE ≈0.68× true SD | **This pass (JOSS-safe):** default number kept; added a `UserWarning` + stronger docstring steering to `abadie_imbens`. **Post-acceptance:** resolve `'auto'`→`'abadie_imbens'` (already implemented & verified at 0.95 coverage); rewrite the `test_default_sp_match_se_is_ai_unchanged` lock; re-check Paper-JSS #12 PSM parity row; ⚠️ CHANGELOG/MIGRATION |
| 2 | HIGH | `did/gardner_2s.py:316` | `gardner_did` two-stage SE ignores Stage-1 estimation uncertainty (docstring overclaims it is "adjusted for first-stage residualisation") | coverage **0.78**, SE/SD 0.65, does not shrink with n | implement Gardner(2021)/`did2s` GMM two-stage variance (IF stacking) **or** bootstrap the two-step; parity vs R `did2s` |
| 3 | HIGH | `did/did_imputation.py:240` → `_cluster_se_imputation` | BJS imputation **overall-ATT** SE under-counts FE-estimation variance | coverage **0.865**, SE/SD 0.76, structural | full BJS(2024) analytic variance or unit-cluster bootstrap; per-horizon SEs already ≈ok |
| 4 | MED | `did/callaway_santanna.py:976` (`_pretrend_test`) | plug-in χ² pre-trend Wald over-rejects in finite samples (pre-cells correlated) | size **0.142** @ nominal 5% (n=60); →0.05 as n grows | multiplier-bootstrap uniform pre-test (as R `did`) or Hotelling-T² df adjustment |
| 5 | MED | `did/gardner_2s.py:332` | `event_study=True` returns an **unweighted** mean of post coefs → different headline ATT than the non-ES path | ES ATT 1.634 vs non-ES 1.815 on same data | weight post coefs by treated-obs counts (`did2s` convention) |
| 6 | MED→LOW | `rd/rdrobust.py` robust path | bias-corrected "Robust" estimate is a standalone local-quadratic, not the CCT `μ̂_p − bias`; diverges when `b≠h` (rho≠1) | BC 0.500 vs R 0.502 at h=0.3,b=0.5 | build the CCT robust point+variance (bias-estimation covariance) for rho≠1 |
| 7 | LOW | `iv.py:1071` | `sp.iv(..., cluster=<Series>)` raises `ValueError: truth value of a Series is ambiguous` (must pass a column-name string) | crash, not wrong number | guard `if cluster is not None and ...`; accept a Series |

### Verified CLEAN by the audit (cross-checked, no bug — valuable to record)

- **DiD:** Callaway–Sant'Anna (point **and** SE; never- & not-yet-treated;
  aggregation), Sun–Abraham, `did_2x2` (HC1), Bacon decomposition (identity
  holds), classic TWFE `event_study`.
- **IV:** just-identified 2SLS = Wald ratio; LIML = 2SLS (8e-15); GMM = 2SLS;
  Olea–Pflueger effective F; Anderson–Rubin size 0.0475 & CI coverage; LIML κ.
  (Homoskedastic/cluster SE differ from R only by Stata's documented dof
  conventions.)
- **RD density:** `sp.rddensity` matches R `rddensity` (T, p, bandwidths).
- **Panel:** one-way & multiway cluster-robust SE (CR1/CGM), within = LSDV.
- **Synth:** simplex weights (exact recovery), California Prop99 ATT & placebo
  p-value calibration, synthdid. **IPW:** unbiased, coverage 0.957.

### Suggested month sequencing toward Stata/R parity

1. **Week 1** — land this pass (CUSUM, IM, RD) + item 1 (matching default SE):
   all "correct estimator already exists / one-line" wins. Add `⚠️` CHANGELOG.
2. **Week 2** — item 2 (`gardner_did` two-stage variance) with a committed R
   `did2s` parity fixture; item 5 (ES ATT weighting) alongside.
3. **Week 3** — item 3 (BJS overall-ATT variance) with a BJS/`did_imputation`
   reference fixture; item 4 (CS multiplier-bootstrap pre-test).
4. **Week 4** — item 6 (CCT robust point/variance for rho≠1) + item 7 (IV
   cluster-Series guard); then a fresh adversarial audit round on the modules
   not yet swept (qte, mediation, survival inference, spatial HAC, gmm).

Guiding principle (matches CLAUDE.md §6): **trust is bounded by the worst SE in
the package.** A user who finds one anti-conservative CI distrusts every
estimate. Closing inference bugs raises the floor faster than adding estimators.

---

## Follow-up pass (roadmap items 1, 2, 7 — implemented & validated)

| # | What shipped | Validation |
|---|---|---|
| 7 | `sp.iv(..., cluster=<Series>)` accepts a Series/array (was a crash); misaligned length fails loudly | Series-cluster SE == string-cluster SE exactly; 97 IV tests green |
| 2a | `gardner_did(vce='bootstrap')` — pairs-cluster bootstrap of the full two-step; analytic default warns | MC 95% coverage **0.78 → 0.90**; point unchanged; module/param docstrings de-overclaimed |
| 2b | `did_imputation(vce='bootstrap')` — cluster bootstrap of the full imputation estimator; analytic default warns | MC 95% coverage **0.87 → 0.94**; point unchanged |
| 1 | `sp.match` default-SE guidance (JOSS-safe): `UserWarning` + docstring, **number unchanged** | locked `test_default_sp_match_se_is_ai_unchanged` still passes |

All new SEs are **opt-in** (`vce='bootstrap'`) so the default published numbers
are unchanged — safe under the Codex commit race and the JOSS-review window.
Regression tests in `tests/test_correctness_inference_fixes.py` (19 total).
CHANGELOG `### ⚠️ Correctness`/`### Added`/`### Changed` and three new
`MIGRATION.md` Unreleased sections (cusum / lee_bounds / RD) now document the
pushed correctness fixes and these additions.

## Follow-up pass 2 (roadmap items 4, 5, 6)

| # | What shipped | Validation |
|---|---|---|
| 4 | `callaway_santanna` pre-trend Wald: Hotelling-T² → `F(k, G−k)` finite-sample correction (was plug-in `χ²(k)`) | MC size **0.155 → 0.070** @ nominal 5%; ATT point/SE unchanged; 86 CS tests green |
| 5 | `gardner_did(event_study=True)` overall ATT: treated-obs-**weighted** mean of post coefs (was unweighted) | ES ATT now **== non-ES ATT** exactly (1.753 vs audit's unweighted 1.63) |
| 6 | `sp.rdrobust` `rho!=1` bias-corrected estimate | **Documented**, not reimplemented — default `rho=1` is exact CCT; a correct `b!=h` point+variance needs R-parity validation. Added a docstring `.. note::` + CHANGELOG "Known limitations". |

Item 4 and item 5 change *inference output* (pre-trend p-value / ES overall
ATT); both now have `⚠️` CHANGELOG entries and `MIGRATION.md` Unreleased
sections (`cs-pretrend-f`, `gardner-es-weighting`). Regression tests added to
`tests/test_correctness_inference_fixes.py` (now 21 tests). Item 6 is the only
remaining roadmap item, deliberately deferred to an R-parity sprint rather than
shipping an under-validated bias-correction on a flagship estimator mid-review.

Item 1's full default switch to `abadie_imbens` waits for JOSS acceptance
(touches the #12 PSM parity row).
