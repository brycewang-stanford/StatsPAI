# Phase 3 — inference / sensitivity family

Worktree `wt/pc-inference-sens`, base `d127e7d8`. Every number below comes
from the committed fixtures
(`tests/reference_parity/_fixtures/inference_sens_R.json`,
`inference_sens_stata.json`) or from a scratch run recorded here. Nothing is
committed.

Generators (run in this order, from the repo root / `_fixtures/`):

1. `python tests/reference_parity/_fixtures/_generate_inference_sens_data.py`
   (CSVs, incl. the full assignment set of each RI design for `ritest`)
2. `Rscript tests/reference_parity/_generate_inference_sens_R.R`
3. `cd tests/reference_parity/_fixtures && stata-mp -b do _generate_inference_sens_stata.do`

Tests: `tests/reference_parity/test_inference_sens_R_parity.py` (49 tests),
`tests/reference_parity/test_inference_sens_stata_parity.py` (24 tests).

## 1. Per-function table

Outcome classes: 1 aligned/bit-exact · 2 defect fixed then 1 · 3 convention ·
4 reference wrong / non-unique · 5 stochastic · 6 no canonical reference.

| function | reference (version) | class | max rel err est / SE (observed) | test |
| --- | --- | --- | --- | --- |
| `cluster_robust_se` | R `sandwich::vcovCL` 3.1.1 (HC1/cadjust one-way, HC0 no adjust, CGM two-way `multi0=FALSE`); Stata 18 `regress, vce(cluster)` | 1 | SE 2.1e-15 (R), 9.8e-16 (Stata) | both |
| `cr3_jackknife_vcov` | R `sandwich::vcovJK(center="estimate")` 3.1.1, `summclust` 0.7.0 (CV3); Stata `regress, vce(jackknife, cluster() mse double)`; identity vs `clubSandwich::vcovCR(type="CR3")` 0.6.2 × (G-1)/G | 1 | V 6.5e-14 (R), SE 4.3e-15 (Stata) | both |
| `jackknife_se` | R `sandwich::vcovJK(center="mean")`; Stata `regress, vce(jackknife, cluster() double)` (SE, t(G-1) p, CI) | 1 (+3: Stata default float replicates) | SE 1.7e-15 (R), 2.0e-15 (Stata); p 1.4e-14; CI 9.2e-12 | both |
| `wild_cluster_bootstrap` | R `fwildclusterboot::boottest` 0.14.3; Stata `boottest` 4.5.3 (WCR, Rademacher, full 2^12 enumeration) | 2 | p exact (k/4096); t 1.6e-14 (R), 1.9e-14 (Stata) | both |
| `wild_cluster_boot` | same | 2 | p exact; t ≤ 1.9e-14 | both |
| `subcluster_wild_bootstrap` | Stata `boottest, bootcluster()` 4.5.3 (fwildclusterboot refuses a bootcluster that is neither a cluster variable nor a regressor) | 2 | p exact; t 1.8e-14 | Stata |
| `wild_cluster_ci_inv` | R `fwildclusterboot::boottest(conf_int)` (uniroot tol 1e-13); Stata `boottest` CI = class 4 | 2 (R) / 4 (Stata) | CI 4.8e-12 (R) | both |
| `fisher_exact` | R `ri2::conduct_ri` 0.5.0 (+randomizr 2.0.1, full enumeration); Stata `ritest` 1.1.7 (`samplingsourcefile()` = full assignment set) | 2 (enumeration added) | p exact; statistic 1e-16 (R), float 8e-9 (Stata `ritest` stores T(obs) in single precision) | both |
| `ri_test` | same | 2 | p exact; statistic ≤ 2.5e-16 (R) | both |
| `rosenbaum_bounds` / `rosenbaum_gamma` | R `DOS2::senWilcox` 0.5.2 (Rosenbaum's own code); Stata `rbounds` 1.1.6; R `rbounds::psens` 2.2 (4-dp rounded, `zero_method="wilcox"`); `stats::binom.test` (sign test) | 2 | 4.6e-15 (DOS2, Stata sig+); sign 2.9e-15; Stata sig- 4.5e-17 abs | both |
| `evalue_rd` | R `EValue::evalues.RD` 4.1.4 (5 cases incl. `true`, `alpha`, `grid`, CI crossing) | 1 | 5.8e-14 (grid built by `seq` vs `np.arange`) | R |
| `bias_factor` | R `EValue::multi_bound(confounding(), RRAUc, RRUcY)` 4.1.4 | 1 | 0 | R |
| `evalue_from_result` | R `EValue::twoXtwoRR` → `evalues.RR` (chain `sp.relative_risk` → `sp.evalue_from_result`) | 1 | 3.3e-16 | R |
| `oster_bounds` | Stata `psacalc` 2.1 (Oster's own); R `robomit::o_delta/o_beta` 1.0.7 (rounded to 6 dp) | 2 | 4.4e-14 (psacalc); 4.3e-7 abs (robomit rounding) | both |
| `oster_delta` | same | 2 | 1e-14 (psacalc); robomit 5e-7 abs | both |
| `pate` | none (see §5) | 6 + defect fixed (AIPW, known-truth) | — | `test_pate_parity.py` |
| `sp.test`, `sp.lincom` | Stata 18 `test` / `lincom` (existing `test_postestimation_stata_parity.py`, 7 model blocks) | 1 | linear models ≤ 2.2e-15; ML fits ≤ 8e-10 on statistics, far-tail p ≤ 4.7e-8 | existing |
| `sp.margins` | Stata 18 `margins, dydx(*)` (5 blocks) | 1 (aligned, mechanism confirmed) | ≤ 2.7e-7 est, 7.5e-8 SE (probit, logit-interaction) | existing |
| `margins_at`, `contrast`, `pwcompare` | not covered by the Stata fixture | — (unchanged, analytical) | — | — |
| `conley` | Stata `acreg` (existing tests; constants embedded) | 1 (promotion only) | SE ~5e-15, asserted 1e-9 | existing |

Counts: **class 1: 10** (`cluster_robust_se`, `cr3_jackknife_vcov`,
`jackknife_se`, `evalue_rd`, `bias_factor`, `evalue_from_result`; promotions
of existing evidence: `test`, `lincom`, `margins`, `conley`); **class 2: 10**
(`wild_cluster_bootstrap`, `wild_cluster_boot`, `subcluster_wild_bootstrap`,
`wild_cluster_ci_inv`, `fisher_exact`, `ri_test`, `rosenbaum_bounds`,
`rosenbaum_gamma`, `oster_bounds`, `oster_delta`); **class 3**: documented
convention notes only (Stata float jackknife replicates, ritest float T(obs),
clubSandwich CR3 without (G-1)/G, psens zero handling); **class 4: 1** (Stata
boottest CI); **class 6: 1** (`pate`, with defect D7 fixed on known truth);
**not attempted: 3** (`margins_at`, `contrast`, `pwcompare`).

## 2. Defects

### D1. Wild cluster bootstrap never enumerated and counted ties (`wild_cluster_bootstrap`, `wild_cluster_boot`, `subcluster_wild_bootstrap`)
*Found:* boottest / fwildclusterboot switch to full enumeration of the
Rademacher grid when `2^G <= reps`; StatsPAI always sampled `n_boot` draws, so
its p-value was a Monte-Carlo estimate of a number both references compute
exactly. It also counted `|t*| >= |t|`; both references count strictly
(`fwildclusterboot:::get_bootstrap_pvalue`: `mean(abs(t_stat) < abs(t_boot))`;
boottest likewise), and under enumeration the identity draw and its negation
tie with |t| exactly, inflating p by 2/2^G.
*Fix:* shared vectorised engine in `inference/wild_bootstrap.py`
(`_wild_weight_matrix`, `_wcr_t_stats`, `_symmetric_boot_pvalue`) used by all
three functions. Enumerates when Rademacher and `2^G <= n_boot` (boottest
rule); ties identified from the weight matrix (w ≡ +1 or w ≡ −1), not by a
tolerance on t (a tolerance moved the CI jumps — see D2).
*Before → after* (G = 12, `n_boot=9999`): `d` p 0.92569 (MC) → 0.9306640625
(= 3812/4096, both references); `x2` 0.054505 → 0.0546875; subcluster 0.90999 →
0.91259765625 (Stata). Default output changes whenever `2^G <= n_boot`
(G ≤ 9 at the default 999) and, by ≤ 2/B, otherwise. Result dicts gain
`enumerated`, `n_boot_requested`; `n_boot` now reports draws used. Same path
changes `sp.panel` feols `wild` p-values (`panel/feols.py` calls
`wild_cluster_bootstrap`).
Not touched: `inference/iv_wild.py` still counts `>=` (line 369) — out of
this family's scope; flag for the IV line.

### D2. `wild_cluster_ci_inv` located endpoints by linear interpolation on a 41-point grid
*Found:* p(h0) is a step function; the endpoint is a jump location. The old
code interpolated p linearly between grid points 0.3 cluster-SEs apart.
*Fix:* grid only brackets; each bracket is bisected to machine precision on
the step function; warns when p never drops below alpha inside the grid
(previously returned the grid edge silently).
*Before → after:* `d` CI (−1.114049, 1.103695) → (−1.121498336078, 1.103681178126);
`x1` (0.622619, 1.461448) → (0.621322810379, 1.454918146702). fwildclusterboot:
(−1.121498336078426, 1.1036811781262224) — agreement 4.8e-12.

### D3. `fisher_exact` / `ri_test` never enumerated small designs
*Found:* ri2 (via `randomizr::obtain_permutation_matrix`) enumerates all
assignments when their number ≤ sims; StatsPAI sampled `n_perm` random
permutations with replacement even when the design had 924 assignments.
*Fix:* `_enumerate_assignments` (complete, cluster, stratified) in
`inference/randomization.py`; exact distribution when count ≤ `n_perm(s)`;
exact Hodges–Lehmann test inversion in that case (vectorised).
*Before → after:* simple design p 0.1443 (MC, seed 1) → 0.142857… = 132/924;
stratified 0.0315 → 0.0302040816 (= 148/4900). `ri_test` gains key `exact`.

### D4. Rosenbaum bounds: continuity correction, zero handling, two-sided bound in the wrong direction
*Found vs DOS2::senWilcox (Rosenbaum's own implementation):* (a) StatsPAI
subtracted a 0.5 continuity correction none of DOS2 / rbounds::psens / Stata
rbounds applies; (b) it dropped zero differences before ranking, whereas DOS2
and Stata rbounds rank them and give them weight 0; (c) the two-sided bound
doubled `sf(|z|)` with z computed at the "greater" worst case, so for a
negative effect it evaluated the wrong tail. (d) sign test two-sided likewise.
*Fix:* `diagnostics/rosenbaum.py`: no continuity correction; new parameter
`zero_method={"pratt" (default, DOS2 / Stata), "wilcox" (rbounds::psens)}`;
"less" = "greater" on negated differences; two-sided = min(1, 2 × smaller
one-sided bound) (DOS2).
*Before → after* (40 pairs, 7 zero differences), upper bound, alternative
"greater", Γ = 1, 1.5, 2, 2.5, 3: (0.0022, 0.0302, 0.1077, 0.2232, 0.3525) →
(0.002835, 0.041828, 0.149128, 0.301256, 0.459566) = DOS2 = Stata sig+.
Two-sided on the **negated** data: (4.2e-3, 7.7e-5, 1e-6, 0, 0) → (0.00567,
0.0837, 0.2983, 0.6025, 0.9191) — the old bound declared a negative effect
insensitive to any hidden bias. Anti-conservative in every case.

### D5. `oster_bounds` used Oster's approximation with the two R² gains swapped
*Found vs psacalc:* β* = β̃ − δ(β̊−β̃)(R̃²−R̊²)/(R_max−R̃²) instead of
… (R_max−R̃²)/(R̃²−R̊²) (the sibling `oster_delta` had the correct
orientation; `tests/test_diagnostics.py` encoded the swapped formula in its
"manual calculation"). Also both functions implemented only the
approximation, while the reference Oster ships (psacalc; robomit ports it)
solves her exact quadratic / cubic.
*Fix:* new `diagnostics/_oster.py` = port of psacalc's Mata (`bound`,
`d1quadsol`, `dnot1cubsol`, root selection included) with psacalc's inputs
(σ_yy, σ_xx after `mcontrol`, t_x). Data path → exact (`method="exact"`);
summary-statistics path → corrected approximation (`method="approximate"`).
`r_max > 1` now raises; `r_max ≤ R̃²` still falls back but warns (was silent).
New keys `beta_adjusted_alternatives`, `method`.
*Before → after* (inference_sens_reg, y ~ t | x1 x2, R_max = 1.3 R̃² =
0.61269): δ* 1.20492 → 0.904199433 (psacalc 0.9041994327987157); β*(δ=1)
0.078546 → −0.071462629 (psacalc −0.07146262869283698). Docstring example
δ* 1.8 → 8.89.

### D6. `oster_delta` default `r_max=1.3` was used as an R² of 1.3
*Found:* the default is documented as Oster's "1.3 × R²" but the code only
replaced `r_max ≤ R_full`, so 1.3 stayed an (impossible) absolute R².
*Fix:* values > 1 are read as a multiplier of R_full (so the default is
min(1, 1.3 R_full), as documented); ≤ 0 means the same; (0, R_full] falls
back with a warning. Point quantities now exact (shared `_oster.py`),
bootstrap bounds use the same exact solution. Further `x_base` entries act as
psacalc's `mcontrol()` (verified: 4.4e-14).
*Before → after* (defaults): δ* 0.16189 → 0.90420; β*(δ=1) −2.39096 →
−0.07146.

### D7. `pate(method="aipw")` was not doubly robust
*Found (known truth, no canonical reference):* correct participation model,
misspecified outcome model (τ(x) = 1 + x²): AIPW bias −0.168 (MC SE 0.013,
16 reps) / −0.355 (another design, 20 reps) while IPW was unbiased. The
augmentation was normalised by the sum of all odds weights and rescaled by
n_exp / n_tgt instead of normalised within each arm.
*Fix:* Hajek per-arm augmentation (`inference/pate.py`, `_pate_aipw`).
*After:* bias −0.001 (MC SE 0.016). Regression test:
`test_pate_parity.py::test_aipw_is_doubly_robust_when_only_the_participation_model_is_right`.

## 3. Proposed promotion records (`_FROZEN_PROMOTIONS`)

```python
    # ---- inference / sensitivity family (phase 3) ----
    "cluster_robust_se": {
        "status": "bit-exact",
        "reference": "R sandwich::vcovCL (HC1, cadjust; HC0; two-way multi0=FALSE); Stata regress, vce(cluster)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "sandwich": "3.1.1", "Stata": "18"},
        "tolerance": "SE 1e-10 rel (observed 2.1e-15 R, 9.8e-16 Stata)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": (
            "One-way CR1 = vcovCL(type='HC1', cadjust=TRUE) = Stata vce(cluster); CR0 with "
            "df_adjust=False; two-way Cameron-Gelbach-Miller with each component's own G. "
            "The two-way matrix is positive definite on the fixture, so the default PSD "
            "projection (sandwich fix=TRUE) is inactive; both fix settings are in the fixture."
        ),
    },
    "cr3_jackknife_vcov": {
        "status": "bit-exact",
        "reference": "R sandwich::vcovJK(center='estimate'), summclust (CV3); Stata regress, vce(jackknife, cluster() mse double)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "sandwich": "3.1.1", "summclust": "0.7.0", "Stata": "18"},
        "tolerance": "vcov 1e-10 rel (observed 6.5e-14 R, SE 4.3e-15 Stata)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": (
            "(G-1)/G * sum (b_(g) - b)(b_(g) - b)', centred at the full-sample estimate. "
            "clubSandwich's CR3 is the same matrix without the (G-1)/G factor, asserted as an "
            "identity. Stata's jackknife prefix stores replicates in float unless `double` "
            "(SEs move ~7e-8); the fixture uses double and records the float default."
        ),
    },
    "jackknife_se": {
        "status": "bit-exact",
        "reference": "R sandwich::vcovJK(center='mean'); Stata regress, vce(jackknife, cluster() double)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "sandwich": "3.1.1", "Stata": "18"},
        "tolerance": "SE / CI 1e-10 rel, p 1e-9 (observed SE 2.0e-15, p 1.4e-14, CI 9.2e-12)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "Replicates centred at their mean; t(G-1) p-values and intervals as Stata reports them.",
    },
    "wild_cluster_bootstrap": {
        "status": "bit-exact",
        "reference": "R fwildclusterboot::boottest; Stata boottest (WCR, Rademacher, full enumeration)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "fwildclusterboot": "0.14.3", "Stata": "18", "boottest": "4.5.3"},
        "tolerance": "p exact (multiple of 1/4096); t 1e-10 rel (observed 1.9e-14)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": (
            "G = 12 and B >= 2^12, so both references and StatsPAI enumerate all 4096 sign "
            "vectors (boottest's rule, adopted in this sweep). p = #{|t*| > |t|}/B, strict; "
            "before the sweep StatsPAI sampled and counted ties. Nonzero null (h0 = 0.2) included."
        ),
    },
    "wild_cluster_boot": {
        "status": "bit-exact",
        "reference": "R fwildclusterboot::boottest; Stata boottest (WCR, Rademacher, full enumeration)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "fwildclusterboot": "0.14.3", "Stata": "18", "boottest": "4.5.3"},
        "tolerance": "p exact (multiple of 1/4096); t 1e-10 rel (observed 1.9e-14)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "Result-object entry point (sp.regress fit) to the same WCR engine as sp.wild_cluster_bootstrap.",
    },
    "subcluster_wild_bootstrap": {
        "status": "bit-exact",
        "reference": "Stata boottest, bootcluster() (CRVE clustered at g6, signs flipped at s12)",
        "reference_versions": {"Stata": "18", "boottest": "4.5.3"},
        "tolerance": "p exact (multiple of 1/4096); t 1e-10 rel (observed 1.8e-14)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": (
            "fwildclusterboot 0.14.3 refuses a bootcluster that is neither a clustering "
            "variable nor a regressor, so there is no R side. Rademacher only; the default "
            "Webb weights are sampled (not enumerable) and are not pinned."
        ),
    },
    "wild_cluster_ci_inv": {
        "status": "bit-exact",
        "reference": "R fwildclusterboot::boottest confidence interval (uniroot tol 1e-13)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "fwildclusterboot": "0.14.3"},
        "tolerance": "CI endpoints 1e-9 rel (observed 4.8e-12)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
        ],
        "note": (
            "Endpoints are the jumps of the enumerated p(h0) step function, found by bisection "
            "(was: linear interpolation on a 41-point grid, 0.7% off). Stata boottest's CI is "
            "T4: its Chandrupatla search returns early on step functions, and both reported "
            "endpoints are values its own test rejects (p = 204/4096, 202/4096); asserted in "
            "the Stata test."
        ),
    },
    "fisher_exact": {
        "status": "bit-exact",
        "reference": "R ri2::conduct_ri (randomizr full enumeration); Stata ritest over the full assignment set",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "ri2": "0.5.0", "Stata": "18", "ritest": "1.1.7"},
        "tolerance": "p exact (k / N_assignments); statistic 1e-12 rel vs R, 2^-23 vs Stata",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": (
            "Complete (924 assignments), cluster (70) and stratified (4900) designs; ATE, KS "
            "and rank-sum statistics. Exact enumeration when the design has <= n_perm "
            "assignments (ri2's rule) was added in this sweep. ritest stores T(obs) in single "
            "precision. The Hodges-Lehmann interval is not pinned."
        ),
    },
    "ri_test": {
        "status": "bit-exact",
        "reference": "R ri2::conduct_ri (randomizr full enumeration); Stata ritest over the full assignment set",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "ri2": "0.5.0", "Stata": "18", "ritest": "1.1.7"},
        "tolerance": "p exact (k / N_assignments); statistic 1e-12 rel vs R, 2^-23 vs Stata",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": "Difference in means, Welch t and KS; complete and cluster randomization.",
    },
    "rosenbaum_bounds": {
        "status": "bit-exact",
        "reference": "R DOS2::senWilcox (Rosenbaum); Stata rbounds; R stats::binom.test (sign test); R rbounds::psens (zero_method='wilcox', 4-dp)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "DOS2": "0.5.2", "rbounds": "2.2", "Stata": "18", "rbounds (Stata)": "1.1.6"},
        "tolerance": "bounding p-values 1e-12 rel (observed 4.6e-15); Stata sig- 1e-15 abs; psens at its own 4-dp rounding",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": (
            "Fixed in this sweep: a continuity correction no reference applies, zeros dropped "
            "before ranking (DOS2 and Stata rank them with weight 0; psens convention kept as "
            "zero_method='wilcox'), and a two-sided bound that evaluated the wrong tail for "
            "negative effects (p = 0 at Gamma = 3 where DOS2 gives 0.919)."
        ),
    },
    "rosenbaum_gamma": {
        "status": "bit-exact",
        "reference": "R DOS2::senWilcox (Rosenbaum); Stata rbounds",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "DOS2": "0.5.2", "Stata": "18", "rbounds (Stata)": "1.1.6"},
        "tolerance": "bounding p-values 1e-12 rel (observed 4.6e-15)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
        ],
        "note": "Alias of sp.rosenbaum_bounds (same object); called directly in test_rosenbaum_gamma_alias_and_long_format_agree.",
    },
    "evalue_rd": {
        "status": "bit-exact",
        "reference": "R EValue::evalues.RD",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "EValue": "4.1.4"},
        "tolerance": "1e-12 rel (observed 5.8e-14: grid built by seq vs np.arange)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
        ],
        "note": "Five tables incl. non-null true, alpha = 0.1, grid = 1e-3 and a CI crossing the null (E-value 1).",
    },
    "bias_factor": {
        "status": "bit-exact",
        "reference": "R EValue::multi_bound(confounding(), RRAUc, RRUcY)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "EValue": "4.1.4"},
        "tolerance": "1e-14 rel (observed 0)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
        ],
        "note": "Plus the identity B(e, e) = RR at the E-value e of RR.",
    },
    "evalue_from_result": {
        "status": "bit-exact",
        "reference": "R EValue::twoXtwoRR -> evalues.RR",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "EValue": "4.1.4"},
        "tolerance": "1e-13 rel (observed 3.3e-16)",
        "sides": ["py", "R"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
        ],
        "note": (
            "End-to-end chain sp.relative_risk -> sp.evalue_from_result(measure='RR') on three "
            "2x2 tables (harmful, protective, CI crossing the null). The SMD path is covered "
            "only through sp.evalue (Track A 23_evalue)."
        ),
    },
    "oster_bounds": {
        "status": "bit-exact",
        "reference": "Stata psacalc (Oster); R robomit::o_delta / o_beta",
        "reference_versions": {"Stata": "18", "psacalc": "2.1", "R": "R version 4.5.2 (2025-10-31)", "robomit": "1.0.7"},
        "tolerance": "1e-12 rel vs psacalc (observed 4.4e-14); 5e-7 abs vs robomit (it rounds to 6 dp)",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": (
            "Data path: Oster's exact solution (quadratic at delta = 1, cubic otherwise, "
            "psacalc's root selection). Until 1.28 the function used her first-order "
            "approximation with the two R-squared gains swapped (delta* 1.205 vs psacalc 0.904). "
            "The summary-statistics path remains the (corrected) approximation and is analytical."
        ),
    },
    "oster_delta": {
        "status": "bit-exact",
        "reference": "Stata psacalc (Oster); R robomit::o_delta / o_beta",
        "reference_versions": {"Stata": "18", "psacalc": "2.1", "R": "R version 4.5.2 (2025-10-31)", "robomit": "1.0.7"},
        "tolerance": "1e-12 rel vs psacalc (observed 4.4e-14); 5e-7 abs vs robomit",
        "sides": ["py", "R", "Stata"],
        "test": [
            "tests/reference_parity/test_inference_sens_R_parity.py",
            "tests/reference_parity/test_inference_sens_stata_parity.py",
            "tests/reference_parity/_fixtures/inference_sens_R.json",
            "tests/reference_parity/_fixtures/inference_sens_stata.json",
        ],
        "note": (
            "delta_star and beta_star_delta1 only (bootstrap SEs of the bounds are not pinned). "
            "The default r_max = 1.3 now means min(1, 1.3 R_full); before it was used as an "
            "R-squared of 1.3. Extra x_base entries = psacalc mcontrol() (verified)."
        ),
    },
    "conley": {
        "status": "bit-exact",
        "reference": "Stata acreg (Colella, Lalive, Sakalli & Thoenig)",
        "provenance": (
            "Stata 18 MP acreg output embedded as constants in "
            "test_conley_acreg_spacetime_parity.py (commands recorded next to each oracle "
            "matrix, synthetic geo-panel regenerated from default_rng(7)); the IV case in "
            "test_iv_hdfe_stata_parity.py records acreg 1.1.0 on _fixtures/iv_hdfe_panel.csv."
        ),
        "reference_versions": {"Stata": "18 MP", "acreg": "1.1.0"},
        "tolerance": "SE 1e-9 rel (observed ~5e-15); absorbed-IV spatial 1e-10",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_conley_acreg_spacetime_parity.py",
            "tests/reference_parity/test_iv_hdfe_stata_parity.py",
        ],
        "note": (
            "Seven spatial / spatio-temporal kernel configurations; the full acreg e(V) is "
            "reproduced entrywise including Mata _makesymmetric. Off-diagonals of acreg depend "
            "on regressor order (asserted); StatsPAI reports the symmetric part. The IV "
            "space-time case is 1e-3 (acreg carries a numerically zero constant column)."
        ),
    },
    "test": {
        "status": "aligned",
        "reference": "Stata 18 test (after regress / ivregress / logit / poisson)",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-6 rel (observed <= 2.3e-15 on linear fits, <= 8e-10 on ML fits, far-tail p <= 4.7e-8)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_postestimation_stata_parity.py",
            "tests/reference_parity/_fixtures/postestimation_stata.json",
        ],
        "note": (
            "Seven model blocks. The ML-block gap is Stata's default ML convergence tolerance: "
            "refitting with nrtolerance/tolerance/ltolerance 1e-14 closes the probit AME gap "
            "from 1e-7 to 2e-12 (scratch check, 2026-09-18)."
        ),
    },
    "lincom": {
        "status": "aligned",
        "reference": "Stata 18 lincom (after regress / ivregress / logit / poisson)",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-6 rel (observed <= 2.3e-15 on linear fits, <= 2.3e-8 overall)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_postestimation_stata_parity.py",
            "tests/reference_parity/_fixtures/postestimation_stata.json",
        ],
        "note": (
            "Seven model blocks, sum and mixed contrasts with the constant. The ML-block gap is "
            "Stata's default ML convergence tolerance (see the sp.test record)."
        ),
    },
    "margins": {
        "status": "aligned",
        "reference": "Stata 18 margins, dydx(*)",
        "reference_versions": {"Stata": "18"},
        "tolerance": "1e-6 rel (observed 2.7e-7 AME / 7.5e-8 SE on probit and logit-with-interaction; <= 1e-10 otherwise)",
        "sides": ["py", "Stata"],
        "test": [
            "tests/reference_parity/test_postestimation_stata_parity.py",
            "tests/reference_parity/_fixtures/postestimation_stata.json",
        ],
        "note": (
            "Average marginal effects with delta-method SEs after logit, probit, poisson, "
            "regress-with-interaction, logit-with-interaction. The 1e-7 gap is Stata's default "
            "ML convergence: with tolerances 1e-14 Stata's AMEs match StatsPAI to 2e-12 "
            "(probit) and 1e-13 (logit interaction). The fixture's glm_cloglog_margins block "
            "is not asserted by the test."
        ),
    },
```

`margins_at`, `contrast`, `pwcompare`: **no promotion** — the Stata fixture
does not exercise them.

## 4. Proposed CHANGELOG / MIGRATION

**Added**
- `sp.rosenbaum_bounds(zero_method=...)` (`"pratt"` default = DOS2 /
  Stata rbounds; `"wilcox"` = rbounds::psens).
- Wild bootstrap results: `enumerated`, `n_boot_requested`; `sp.ri_test`
  result: `exact`; `sp.oster_bounds`: `method`, `beta_adjusted_alternatives`.
- Cross-language fixtures for the inference / sensitivity family vs
  sandwich, clubSandwich, summclust, fwildclusterboot, ri2, DOS2, rbounds,
  EValue, robomit and Stata regress / jackknife / boottest / ritest / rbounds
  / psacalc.

**⚠️ Correctness**
- `sp.wild_cluster_bootstrap`, `sp.wild_cluster_boot`,
  `sp.subcluster_wild_bootstrap` (and `sp.panel` feols `wild` p-values):
  enumerate the Rademacher grid when `2**G <= n_boot` and count `|t*| > |t|`
  strictly, as Stata boottest / R fwildclusterboot. p changes (e.g. 0.9257 →
  0.9307 at G = 12).
- `sp.wild_cluster_ci_inv`: endpoints located by bisection of the p(h0) step
  function instead of linear interpolation between grid points (0.7% shift on
  the fixture); warns when the grid does not bracket an endpoint.
- `sp.fisher_exact`, `sp.ri_test`: exact enumeration when the design has no
  more assignments than `n_perm(s)`.
- `sp.rosenbaum_bounds` / `sp.rosenbaum_gamma`: no continuity correction;
  zeros ranked (Pratt); two-sided / "less" bounds evaluated in the right tail
  (the old two-sided bound reported p ≈ 0 for negative effects).
- `sp.oster_bounds`: exact Oster solution from data (psacalc); the
  summary-statistics approximation had the R² gains swapped.
- `sp.oster_delta`: exact solution; default `r_max=1.3` now means
  min(1, 1.3 R_full) as documented (was an R² of 1.3).
- `sp.pate(method="aipw")`: per-arm normalised augmentation; was not doubly
  robust (bias −0.17 with a correct participation model).

**MIGRATION rows** (all default-output changes):

| function | old | new | reproduce old |
| --- | --- | --- | --- |
| wild bootstrap family | MC p with `>=` | exact (enumerated) p with `>` when 2^G ≤ n_boot | not kept (MC estimate of the same quantity; `>=` was wrong) |
| `wild_cluster_ci_inv` | interpolated endpoint | jump location | not kept |
| `fisher_exact`, `ri_test` | MC p on small designs | exact p | pass `n_perm(s)` below the number of assignments |
| `rosenbaum_bounds` | cc −0.5, zeros dropped, wrong-tail two-sided | DOS2 | `zero_method="wilcox"` reproduces the zero handling only |
| `oster_bounds` | swapped approximation | exact (data) / corrected approximation (summaries) | not kept (wrong formula) |
| `oster_delta` | r_max=1.3 absolute, approximation | min(1, 1.3 R_full), exact | pass an explicit `r_max` in (R_full, 1] |
| `pate(method="aipw")` | mis-scaled augmentation | per-arm Hajek | not kept |

## 5. Not closed

- **`pate`** (class 6): docstring names no command; CRAN has no canonical
  transport-AIPW package (searched `tools::CRAN_package_db()` titles /
  descriptions for transportab/generaliz/target population: only unrelated
  or multi-source packages, e.g. CausalMetaR). Defect D7 fixed on known
  truth. The bootstrap SE is stochastic (T3 at best) and not compared.
- **`margins_at`, `contrast`, `pwcompare`**: not in the Stata postestimation
  fixture; `src/statspai/postestimation/*` is owned by another line, so no new
  comparison was attempted.
- **Webb / Mammen wild weights**: cannot be enumerated by boottest, only
  sampled — T3; not pinned.
- **`fisher_exact` Hodges–Lehmann interval**: no reference computes the same
  grid-based inversion; not pinned.
- **`oster_bounds` summary-statistics path**: the corrected approximation is
  checked analytically only (psacalc needs the data).
- `inference/iv_wild.py` still counts ties (`>=`) — outside this family; flag
  for the IV line.

## 6. Notes for the integrator

- **.gitignore**: `tests/reference_parity/_fixtures/_ado_inference_sens/`
  (private SSC ado dir: boottest, ritest, rbounds, psacalc, summclust).
- Signature change `sp.rosenbaum_bounds(..., zero_method="pratt")` → regenerate
  schemas (`scripts/dump_schemas.py`) and add the parameter to the registry
  entry (registry not edited here).
- R packages installed in the user library: rbounds 2.2, sensitivitymv
  1.4.4, robomit 1.0.7, ri2 0.5.0 (+randomizr, estimatr), DOS2 0.5.2,
  fwildclusterboot 0.14.3 and summclust 0.7.0 from GitHub (built with an empty
  `FLIBS` Makevars because `/opt/gfortran` is absent).
- Existing tests edited because they pinned the defective formulas:
  `tests/test_diagnostics.py` (Oster δ* 1.0 → 4.0, β* 0 → 1.5, new
  vanishing-adjustment test), `tests/test_tierD_bounds_analytic.py`
  (`test_matches_exact_oster_solution`), `tests/reference_parity/test_oster_delta_parity.py`
  (zero-crossing via the exact cubic), `tests/reference_parity/test_pate_parity.py`
  (new DR test).
- Test status in this worktree: the new files (49 R + 24 Stata tests) and
  1443 tests of every module touched pass. `tests/test_schema_export.py::
  test_committed_schemas_dir_is_in_sync` fails until `scripts/dump_schemas.py`
  is rerun (new `zero_method` parameter) -- `schemas/` is integrator-owned.
- Pre-existing, unrelated: the `summary()` doctests in `bounds/partial_id.py`
  fail at HEAD too (they print and expect nothing).
