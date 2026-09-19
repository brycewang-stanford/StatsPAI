# Phase 3, family `rd_iv` — RD extras, rlasso, weak-IV / JIVE / IV-QR

Worktree `wt/pc-rd-iv`, all changes uncommitted. Every number below was
measured on the committed fixture bytes in this worktree
(`PYTHONPATH=src`, `sp.__file__` inside the worktree).

New evidence files:

| file | writes / reads |
| --- | --- |
| `tests/reference_parity/_generate_rd_iv_R.R` | `rd_iv_ivw.csv`, `rd_iv_rueda.csv` (= `ivDiag::rueda`), `rd_iv_ivqr.csv`, `rd_iv_R.json` (ivDiag 1.0.6, ivmodel 1.9.1, quantreg 6.1, lfe 3.1.1) |
| `tests/reference_parity/_fixtures/_generate_rd_iv_stata.do` | `rd_iv_Stata.json` (Stata 18: ivreg2 4.1.12, weakivtest 10/28/2020, weakiv 2.4.07, jive 1.0.2 (st0108), rdrobust 11.1.0) |
| `tests/reference_parity/_generate_rd_iv_rd_R.R` | `rd_iv_kink.csv`, `rd_iv_density.csv`, `rd_iv_hte.csv`, `rd_iv_frd.csv`, `rd_iv_locrand.csv`, `rd_iv_rd_R.json` (rdrobust 4.0.0, rddensity 2.6, lpdensity 2.5, rdd 0.57, rdhte 0.2.0, sandwich 3.1.1, RDHonest 1.0.1.9000, rdlocrand 2.0); reads `rdsenate.csv` |
| `tests/reference_parity/test_rd_iv_R_parity.py` | 27 tests (IV block) |
| `tests/reference_parity/test_rd_iv_rd_R_parity.py` | 60 tests (RD block) |

All generators are deterministic and were re-run end-to-end; the five
pre-existing rlasso generators reproduce their committed fixtures
byte-for-byte (`git status` clean after re-running).

## 1. Per-function outcome

Outcome classes: 1 aligned/bit-exact, 2 defect fixed then 1, 3 convention
difference, 4 reference wrong, 5 stochastic (T3), 6 no canonical reference /
different estimand, open = not closed.

| function | reference | class | max rel err est / SE | test |
| --- | --- | --- | --- | --- |
| `tF_critical_value` | R `ivDiag::tF` 1.0.6 (26 F values, 4 … 1e4) | 2 | 0 / – | `test_rd_iv_R_parity.py` |
| `tF_adjustment` | same (thin alias) | 2 | 0 / – | same |
| `effective_f_test` | Stata `weakivtest` (6 designs, k=1 and 3, HC1 and cluster); R `ivDiag::eff_F` (k=1) | 1 (+4 for ivDiag k>1) | 3.7e-14 (Stata), 1.3e-12 (R) | same |
| `iv_diag` | R `ivDiag::ivDiag` analytic block (6 designs) | 2 | 2.8e-12 / 5.9e-12 (2SLS), 4.3e-13 (OLS SE), tF 1.3e-14, tF CI 3.5e-12 | same |
| `weakrobust` | R `ivmodel::CLR`; Stata `weakiv, md small` | 2 | CLR stat 0 / p 1.2e-11 (ivmodel); CLR set 1.2e-5 (ivmodel `uniroot` tol); CLR/K 1.1e-7, AR 2.1e-8 (Stata) | same |
| `jive` | Stata `jive` (st0108) `ujive1`/`ujive2`, default and `robust` | 2 | 3.0e-13 / 4.4e-13 | same |
| `ivqreg` | root of the IV-QR equation with R `quantreg::rq` (3 quantiles) | 6 (definition-level check) | 1e-13 (asserted 1e-6: Brent `xtol`) | same |
| `zero_first_stage` | none | 6 | – | – |
| `lasso_iv` | none for BIC/AIC/CV-penalised Lasso-IV | 6 (mislabel fixed) | – | – |
| `rlasso` | R `hdm::rlasso` 0.3.2 | 1 (bookkeeping) | 7.3e-14 (β on support), 1.8e-12 (residuals) | `test_rlasso_parity.py` |
| `rlasso_effect` | `hdm::rlassoEffect` | 1 | 3.3e-15 / 1.9e-16 (+ growth/AJR vignettes ≤ 2e-15) | `test_rlasso_parity.py`, `test_rlasso_vignette_parity.py` |
| `rlasso_effects` | `hdm::rlassoEffects` (incl. 15 cps2012 targets) | 1 | 2.8e-12 / 2.0e-15 | same |
| `rlasso_iv` | `hdm::rlassoIV` (4 paths + EminentDomain + AJR) | 1 | 6.1e-9 / 6.2e-10 (EminentDomain, rank-deficient); ≤ 3e-15 elsewhere | same |
| `rlassologit` | `hdm::rlassologit` / `glmnet` 4.1.10 | 1 (aligned) | post: 1.7e-10; `post=False`: 1.3e-6 (glmnet convergence) | `test_rlassologit_parity.py` |
| `rlassologit_effect` | `hdm::rlassologitEffect` | 1 (aligned) | 1.1e-15 / 5.3e-14 (post); SE 3.2e-7 (`post=False`) | `test_rlassologit_effect_parity.py` |
| `rlassologit_effects` | `hdm::rlassologitEffects` | 1 | 8.6e-16 / 2.5e-14 | same |
| `rkd` | R `rdrobust(deriv=1, vce='hc1')` 4.0.0; Stata `rdrobust, deriv(1)` 11.1.0 | 2 | 5.1e-12 / 5.2e-12 (R); 4.5e-11 / 6.0e-11 (Stata) | `test_rd_iv_rd_R_parity.py` |
| `rdrobust` (deriv=1 path) | same, 10 R cells + 6 Stata cells | 2 | 8.3e-12 / 6.0e-12; h 3.9e-12 | same |
| `rdplot` | R `rdrobust::rdplot` (8 `binselect`, manual bins + tri kernel, covs, Senate) | 2 | bins 4.0e-12; J / N exact; p=4 poly coef 1.6e-10 | same |
| `rdplotdensity` | R `rddensity::rdplotdensity` (lpdensity) | 2 | 8.0e-12 (f_p, f_q, se_p, se_q) | same |
| `mccrary_test` | R `rdd::DCdensity` 0.57 (CRAN archive) | 2 | 6.5e-12 | same |
| `rdhte` | R `rdhte` 0.2.0 (11 cells: continuous / 0-1 subgroups, HC0-3, CR1, p=2, 3 kernels, selected h) | 2 | 2.2e-12 / 1.1e-12 | same |
| `rdbwhte` | R `rdhte::rdbwhte` | 2 | < 1e-8 (h) | same |
| `rdhte_lincom` | R `rdhte::rdhte_lincom` | 2 | < 1e-9 (est, z, CI, joint χ²) | same |
| `rd_bias_aware_fuzzy` | R `RDHonest(y | d ~ x)` for every ingredient and RDHonest's interval | 2 (AR set itself: 6) | fixed h/M 1.1e-14 / 3.6e-15; selected h 3.5e-8 | same |
| `rdsensitivity` | R `rdlocrand::rdsensitivity` / `rdrandinf` | 5 (T3) + estimates 1 | estimates exact (via rdrandinf obs.stat); p within 4 pooled MC SEs at 4000 draws | same |
| `rdrbounds` | R `rdlocrand::rdrbounds` | 2 → 5 (T3) | within 4 pooled MC SEs at 4000 draws | same |
| `rdd` | alias of `sp.rdrobust` (Track A 06_rd) | 1 (alias proof) | 0 | `test_track_a_alias_equivalence.py` |
| `geographic_rd` | alias of `sp.rdms` (Track A 89_rdms) | 1 (alias proof) | 0 | same |
| `multi_cutoff_rd` | alias of `sp.rdmc` (frozen R rdmulti) | 1 | identical to `rdmc`; R 1e-9 | `test_rdmulti_parity.py` |
| `rd2d`, `rd2d_bw`, `boundary_rd` | R `rd2d` 1.0.0 | open | – | – |
| `rd_multi_score`, `multi_score_rd` | none (arXiv 2508.15692 method, no package) | 6 | – | – |

Counts: class 1 = 10 (7 rlasso-family + 3 aliases), class 2 = 16, class 5 =
2 (`rdsensitivity` p-values, `rdrbounds`), class 6 = 5 (`ivqreg`,
`zero_first_stage`, `lasso_iv`, `rd_multi_score`, `multi_score_rd`), open =
3 (`rd2d`, `rd2d_bw`, `boundary_rd`). `effective_f_test` also carries a T4
note against ivDiag.

## 2. Defects

Default numeric output changes are marked **(default)**.

**D1 `tF_critical_value` / `tF_adjustment` — fabricated table (default).**
The 24-row "Table 3a" in `diagnostics/weak_iv.py` was not LMMP's: c(10) was
3.16 (LMMP / ivDiag: 3.4353), c(15) 2.54 (2.8662), c(50) 1.98 (2.1529), and
it returned 1.96 from F = 75 on, contradicting its own "104.7" docstring.
Every error anti-conservative. Found by the first ivDiag comparison. Replaced
by ivDiag's 84-point table indexed by √F with ivDiag's interpolation; 0 error
on 26 F values. Below F = 4 we return `inf` where ivDiag clamps to 18.66
(documented).

**D2 `iv_diag` / `anderson_rubin_test` — tF indexed by the wrong F, and
reported for k > 1 (default).** The critical value was looked up at the
homoskedastic first-stage F although the t-ratio uses HC1 / cluster SEs;
ivDiag uses the effective (robust Wald) F. On `rd_iv_ivw` (k=1): 18.66 at
F=3.96 → 12.238 (HC1, F_eff 4.293) / 16.475 (cluster). tF is now NaN (None
in `anderson_rubin_test`) with more than one instrument, as ivDiag omits it.

**D3 `iv_diag` — `se_ols` ignored `vcov` and `cluster` (default).** Always
homoskedastic: 0.030699 → 0.035494 (HC1) / 0.039353 (cluster) on
`rd_iv_ivw`, now = ivDiag.

**D4 CLR / K orthonormalisation (`weakrobust`, `conditional_lr_test`,
`conditional_lr_ci`, `k_test_ci`) (default).** `Zs = solve(L', Z')'` gives
`Z L^-1`, not orthonormal once k ≥ 2 (Zs'Zs − I = 4e-3 on the fixture), so
every S / T statistic was mis-scaled. CLR 0.0094457 → 0.0090722 (ivmodel
0.00907221605522, 4.1%); K 0.0090481 → 0.0086880 (Stata 0.0086879820).
Found by holding `clr_stat` against ivmodel.

**D5 `weakrobust` K "at h0" read off the nearest grid point (default).**
The grid is centred on the 2SLS estimate, so K was evaluated at a different
β (31.889 vs 32.144 on the docstring example; the grid endpoint whenever h0
lies outside the grid). Now exact at h0.

**D6 CLR conditional critical value by Monte Carlo (default).**
`conditional_lr_test`, `conditional_lr_ci` (hence `weakrobust`,
`iv_diag(include_clr_ci=True)`) simulated a quantity ivmodel and weakiv
integrate. At the weakrobust default (5000 draws) the CLR set's lower end
moved −4.43 / −3.79 between seeds 0 and 1 (ivmodel −3.7680). New
`method='exact'` (default; `'simulate'` keeps the old path, `clr_method=` on
`weakrobust`): CLR p 1.2e-11 vs ivmodel; set −3.7680547 (ivmodel −3.7680086,
limited by its `uniroot` default tolerance; Stata display −3.76805). The
integral was checked against 4e6-draw simulation for k = 2, 3, 5.

**D7 `jive` — `variant='jive2'` was not a jackknife estimator, and the
default SE omitted the sandwich (default).** `jive2` used `fitted/(1−h)`
(keeps observation i): 0.18519 → −3.38682 (Stata UJIVE2 −3.386819). Default
SE was `s²(X̂'X)⁻¹`: 3.3019 → 7.0217 (Stata 7.021736). The n×n projection
is no longer formed.

**D8 `rdrobust` — explicit `p ≤ deriv` silently raised to `deriv+1`
(default only for callers passing p=1 with deriv=1).** `rdrobust(deriv=1,
p=1)` returned the p=2 estimate (1.3843 instead of R's 0.9933 on
`rd_iv_kink`, vce nn). `p` now
defaults to `None` → R's rule (1, or deriv+1); explicit p is honoured;
`deriv > p` raises.

**D9 `rdrobust` — no check that the window holds enough observations.**
`h=1e-4` with zero observations returned estimate 5.4e-6, SE 6e-22, p = 0.0.
Now `DataInsufficient` (R errors too).

**D10 `rkd` — ad hoc default bandwidth; fuzzy SE without the kink
covariance (default).** Now `rdrobust(deriv=1, vce='hc1')` on `rd_iv_kink`: default h 0.21760
(rule of thumb) → 0.21801 (R mserd, hc1), estimate 1.03151 → 1.03463; fuzzy
SE at h=0.4 0.225889 → 0.217965 (R 0.217964991849126; 3.6%). Robust bias-corrected row added to
`model_info['robust']`.

**D11 `rdplot` — bin selector not R's; `kernel` ignored (default).** An
"IMSE" rule with a 0.7 fudge factor and caps: esmv J = (17, 16) → (38, 41) on
`rd_iv_kink` (R). The kernel argument never reached the fit (default
'triangular', fit unweighted); default is now R's 'uniform', which keeps the
old picture's curve. Bin CIs now R's t intervals. New
`statspai.rd._rdplot_core` ports rdplot's numerics; numbers on
`fig.rdplot_data`. Fewer than 20 observations now raises (R stops too).

**D12 `rdplotdensity` — per-side ECDF, rule-of-thumb bandwidth,
heuristic SE (default).** Left density at the cutoff on `rd_iv_density`:
0.6045 → 0.2325 (R). Now `lpdensity` port with rddensity bandwidths and R's
`n_side/(n−1)` scale; numbers on `fig.rdplotdensity_data`. A side too sparse
for rddensity now raises instead of plotting.

**D13 `mccrary_test` — a different estimator under McCrary's name
(default).** Bin width, bandwidth, SE and silent fallbacks (density 0.01, SE
0.1) did not follow DCdensity. θ on `rd_iv_density` 1.3036 (SE 0.2756) →
1.2707 (0.1910) = R. Also changes `sp.rdrobust(...).model_info['mccrary']`
(the automatic manipulation check calls it). New `bin_width=` argument.

**D14 `rdhte` / `rdbwhte` / `rdhte_lincom` — conventional inference,
rule-of-thumb bandwidth, `b` ignored, binary z not subgroups (default).**
Point estimates at a given h were already R's. SEs were the order-p fit's
HC1 (0.0697 on T at h=0.4) instead of R's robust bias-corrected HC3 (0.0993);
default h 0.1506 → 0.3893 (R rdbwselect). `bandwidth_h` was rounded to 6
decimals. Degenerate fits returned zeros with `1e10·I` covariance silently.
New `q`, `vce` on `rdhte`; `q`, `bwselect`, `vce`, `cluster` on `rdbwhte`;
`linfct` on `rdhte_lincom`. `rdbwhte` returns a DataFrame for subgroups.

**D15 `rd_bias_aware_fuzzy` — bias bound below the worst case, old M rule,
rule-of-thumb h (default).** Bias `h²M/12` understates the local-linear
Hölder worst case (asymptotic constant 1/10 for the triangular kernel; the
exact bound depends on the realised weights). Rebuilt on the RDHonest port:
exact weights-based bias, MROT for `M_y` and `M_d`, RDHonest's FRD default
bandwidth, nearest-neighbour 2×2 variance (cluster form with `cluster=`).
At fixed h=0.4, M=(2, 0.5): set (1.1549, 2.3799) → (1.1327, 2.3635);
defaults: (0.62, 3.92) with h 0.141, M_y 89.7 → (0.941, 3.143) with h
0.1713, M_y 35.4 (= RDHonest). Endpoints are now root-found, not grid
points. RDHonest's own interval returned in `model_info['bias_aware']['rdhonest']`.

**D16 `rdrbounds` — median split instead of the extremum over thresholds
(default).** Only one Rosenbaum pattern (the median) and fixed-n
assignment: upper bounds at Γ = 1.2 / 1.5 / 2 were 0.012 / 0.056 / 0.159 vs
R 0.021 / 0.094 / 0.346 on `rd_iv_locrand` — anti-conservative. Now R's
algorithm (Bernoulli, all u, R's pattern families); T3 agreement.

**D17 `rlasso_effect(s)` — unidentified targets returned silently.** A
target spanned by the selected controls (cps2012 `female:hsd08`, residual
variance 4e-30 of its own) gives noise (hdm −4.7e13, StatsPAI 1.9e-36). Now a
`RuntimeWarning`; no number changes. The vignette test's claim that "rare
categories diverge" was false: 15 of 16 targets agree to ≤ 3e-12 and are now
pinned.

**Doc / label fixes (no numbers):** `lasso_iv` was attributed to BCCH
(2012) in docstring and `model_info['method']` while using BIC/AIC/CV Lasso
(BCCH is `sp.rlasso_iv`); n×n annihilator removed. `ivqreg` claimed to
match Stata `ivqreg2` (Machado–Santos Silva, a different estimator) and
"R quantreg::ivqreg" (does not exist); over-identified models use identity
weighting, not CH's.

## 3. Proposed `_FROZEN_PROMOTIONS` entries

```python
_RD_IV_R = "R version 4.5.2 (2025-10-31)"
{
    "tF_critical_value": {
        "status": "bit-exact",
        "reference": "R ivDiag::tF 1.0.6 (LMMP 2022 tF table)",
        "reference_versions": {"R": _RD_IV_R, "ivDiag": "1.0.6"},
        "tolerance": "critical value at 26 F values from 4 to 1e4: rel 1e-12 (observed 0)",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_iv_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_R.json"],
        "note": (
            "Through 1.28.0 the table was not LMMP's (c(10) 3.16 vs 3.4353; 1.96 "
            "from F = 75), every error anti-conservative. Now ivDiag's 84-point "
            "sqrt(F) table and interpolation. Below F = 4 StatsPAI returns inf "
            "where ivDiag clamps to 18.66."
        ),
    },
    "tF_adjustment": {  # same record, alias of tF_critical_value
        "status": "bit-exact",
        "reference": "R ivDiag::tF 1.0.6 (LMMP 2022 tF table)",
        "reference_versions": {"R": _RD_IV_R, "ivDiag": "1.0.6"},
        "tolerance": "rel 1e-12 on the same 26 F values (observed 0)",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_iv_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_R.json"],
        "note": "Thin alias of tF_critical_value, asserted directly in the same test.",
    },
    "effective_f_test": {
        "status": "bit-exact",
        "reference": "Stata weakivtest (Montiel Olea & Pflueger) after ivreg2; R ivDiag::eff_F 1.0.6",
        "reference_versions": {"Stata": "18 MP", "weakivtest": "10/28/2020",
                               "ivreg2": "4.1.12", "R": _RD_IV_R, "ivDiag": "1.0.6"},
        "tolerance": "F_eff rel 1e-9 on 6 designs (observed 3.7e-14 vs Stata, 1.3e-12 vs ivDiag for k = 1)",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_rd_iv_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_Stata.json",
                 "tests/reference_parity/_fixtures/rd_iv_R.json"],
        "note": (
            "k = 1 and 3 instruments, HC1 and clustered, on ivDiag::rueda and a "
            "seeded weak-IV design. ivDiag's k > 1 effective F uses the "
            "un-partialled Z'Z and differs by 13-16% from weakivtest and "
            "StatsPAI (asserted as a reference disagreement)."
        ),
    },
    "iv_diag": {
        "status": "bit-exact",
        "reference": "R ivDiag::ivDiag 1.0.6 (analytic block)",
        "reference_versions": {"R": _RD_IV_R, "ivDiag": "1.0.6", "lfe": "3.1.1"},
        "tolerance": (
            "2SLS / OLS coefficients and SEs, classical first-stage F, effective F, "
            "tF critical value and interval: rel 1e-9 on 6 designs (observed <= 6e-12)"
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_iv_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_R.json"],
        "note": (
            "Bootstrap, CLR / K sets and LTZ are not part of this record. Fixed on "
            "the way: tF indexed by the homoskedastic F and reported for k > 1; "
            "se_ols ignored vcov and cluster."
        ),
    },
    "weakrobust": {
        "status": "aligned",
        "reference": "R ivmodel::CLR 1.9.1; Stata weakiv 2.4.07 (md small)",
        "reference_versions": {"R": _RD_IV_R, "ivmodel": "1.9.1", "Stata": "18 MP", "weakiv": "2.4.07"},
        "tolerance": (
            "CLR statistic and p-value vs ivmodel rel 1e-9 (observed 1.2e-11); CLR set "
            "vs ivmodel 5e-5 (observed 1.2e-5, ivmodel's uniroot default tolerance); "
            "CLR / K / AR vs Stata weakiv 1e-6 (observed 1.1e-7 / 1.1e-7 / 2.1e-8, "
            "not bisected further)"
        ),
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_rd_iv_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_R.json",
                 "tests/reference_parity/_fixtures/rd_iv_Stata.json"],
        "note": (
            "Three fixes: the orthonormalised instruments were not orthonormal "
            "for k >= 2 (CLR 4.1% high); K 'at h0' was read off the nearest grid "
            "point; the CLR critical value was simulated (now integrated exactly, "
            "clr_method='simulate' keeps the old path). The set endpoints also "
            "satisfy p_CLR(endpoint) = alpha to 1e-9 (reference-free identity)."
        ),
    },
    "jive": {
        "status": "bit-exact",
        "reference": "Stata jive 1.0.2 (Stata Journal st0108) ujive1 / ujive2",
        "reference_versions": {"Stata": "18 MP", "jive": "1.0.2"},
        "tolerance": "coefficients and SEs (default and robust) rel 1e-9 (observed 3.0e-13 / 4.4e-13)",
        "sides": ["py", "Stata"],
        "test": ["tests/reference_parity/test_rd_iv_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_Stata.json"],
        "note": (
            "variant='jive1' = ujive1, 'jive2' = ujive2. Through 1.28.0 jive2 was "
            "fitted/(1-h) (not a jackknife instrument) and the default SE omitted "
            "the IV sandwich (half of Stata's). A brute-force leave-one-out "
            "reconstruction is asserted alongside."
        ),
    },
    "rlasso": {
        "status": "bit-exact",
        "reference": "R hdm::rlasso 0.3.2",
        "reference_versions": {"R": _RD_IV_R, "hdm": "0.3.2"},
        "tolerance": "support exact; beta / sigma / loadings / residuals atol 1e-6, lambda0 rtol 1e-8 (observed rel <= 1.8e-12)",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rlasso_parity.py",
                 "tests/reference_parity/_fixtures/rlasso_R.json"],
        "note": "Evidence existed since the hdm port; the tests called statspai.rlasso.* rather than sp.*, so no record was built. Four specifications (post / intercept / homoscedastic).",
    },
    "rlasso_effect": {
        "status": "bit-exact",
        "reference": "R hdm::rlassoEffect 0.3.2",
        "reference_versions": {"R": _RD_IV_R, "hdm": "0.3.2"},
        "tolerance": "alpha / se atol 1e-6 (observed rel <= 3.3e-15), incl. hdm's GrowthData vignette",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rlasso_parity.py",
                 "tests/reference_parity/test_rlasso_vignette_parity.py",
                 "tests/reference_parity/_fixtures/rlasso_R.json"],
        "note": "Partialling out and double selection.",
    },
    "rlasso_effects": {
        "status": "bit-exact",
        "reference": "R hdm::rlassoEffects 0.3.2",
        "reference_versions": {"R": _RD_IV_R, "hdm": "0.3.2"},
        "tolerance": "alpha / se rtol 1e-9 on cps2012 (observed 2.8e-12); atol 1e-6 on the synthetic fixture",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rlasso_parity.py",
                 "tests/reference_parity/test_rlasso_vignette_parity.py",
                 "tests/reference_parity/_fixtures/rlasso_vignette_R.json"],
        "note": "15 identified cps2012 targets pinned; the 16th (female:hsd08) is unidentified on both sides and now warns.",
    },
    "rlasso_iv": {
        "status": "bit-exact",
        "reference": "R hdm::rlassoIV 0.3.2",
        "reference_versions": {"R": _RD_IV_R, "hdm": "0.3.2"},
        "tolerance": "coef / se atol 1e-6 (observed <= 3e-15); EminentDomain atol 1e-4 (observed 6.1e-9: pseudo-inverse of a rank-deficient control block)",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rlasso_parity.py",
                 "tests/reference_parity/test_rlasso_vignette_parity.py",
                 "tests/reference_parity/_fixtures/rlasso_R.json"],
        "note": "All four select_Z / select_X paths, BCCH EminentDomain and the AJR vignette.",
    },
    "rlassologit": {
        "status": "aligned",
        "reference": "R hdm::rlassologit 0.3.2 (glmnet 4.1.10 engine)",
        "reference_versions": {"R": _RD_IV_R, "hdm": "0.3.2", "glmnet": "4.1.10"},
        "tolerance": "support exact; post-Lasso atol 1e-5 (observed rel 1.7e-10); post=False atol 1e-4 (observed 1.3e-6: glmnet coordinate-descent convergence)",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rlassologit_parity.py",
                 "tests/reference_parity/_fixtures/rlassologit_R.json"],
        "note": "The non-post fit inherits glmnet's stopping rule; the post-Lasso fit is an unpenalised logit and agrees to 1e-10.",
    },
    "rlassologit_effect": {
        "status": "bit-exact",
        "reference": "R hdm::rlassologitEffect 0.3.2",
        "reference_versions": {"R": "4.5.2", "hdm": "0.3.2"},
        "tolerance": "alpha / se atol 1e-6 (observed rel 1.1e-15 / 5.3e-14 post; se 3.2e-7 with post=False)",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rlassologit_effect_parity.py",
                 "tests/reference_parity/_fixtures/rlassologit_effect_R.json"],
        "note": "",
    },
    "rlassologit_effects": {
        "status": "bit-exact",
        "reference": "R hdm::rlassologitEffects 0.3.2",
        "reference_versions": {"R": "4.5.2", "hdm": "0.3.2"},
        "tolerance": "coef / se atol 1e-6 (observed rel 8.6e-16 / 2.5e-14)",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rlassologit_effect_parity.py",
                 "tests/reference_parity/_fixtures/rlassologit_effect_R.json"],
        "note": "",
    },
    "rkd": {
        "status": "bit-exact",
        "reference": "R rdrobust::rdrobust(deriv = 1, vce = 'hc1') 4.0.0; Stata rdrobust, deriv(1) 11.1.0",
        "reference_versions": {"R": _RD_IV_R, "rdrobust": "4.0.0", "Stata": "18 MP", "rdrobust (Stata)": "11.1.0"},
        "tolerance": "conventional and robust estimate / SE, bandwidth: rel 1e-9 (observed 5.2e-12 vs R, 6.0e-11 vs Stata)",
        "sides": ["py", "R", "Stata"],
        "test": ["tests/reference_parity/test_rd_iv_rd_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_rd_R.json",
                 "tests/reference_parity/_fixtures/rd_iv_Stata.json"],
        "note": (
            "sp.rkd is rdrobust(deriv=1, vce='hc1') with the conventional row as "
            "headline. Through 1.28.0 its default bandwidth was a rule of thumb and "
            "the fuzzy SE dropped the kink covariance (3.6% on the fixture). Sharp, fuzzy, "
            "clustered, fixed and selected bandwidth."
        ),
    },
    "rdplot": {
        "status": "bit-exact",
        "reference": "R rdrobust::rdplot 4.0.0",
        "reference_versions": {"R": _RD_IV_R, "rdrobust": "4.0.0"},
        "tolerance": (
            "J / J_IMSE / J_MV and bin counts exact; bin means, SEs, t intervals and "
            "polynomial values rel 1e-9 (observed 4.0e-12); p = 4 global polynomial "
            "coefficients rel 1e-8 (observed 1.6e-10, raw-scale conditioning)"
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_iv_rd_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_rd_R.json"],
        "note": (
            "All eight binselect rules, manual bins with a triangular kernel, "
            "covariates, and the Senate data (missing outcomes, mass points). "
            "Numbers returned on fig.rdplot_data. Through 1.28.0 the bin count was "
            "a rule of thumb (esmv 17/16 bins vs R's 38/41) and kernel was ignored."
        ),
    },
    "rdplotdensity": {
        "status": "bit-exact",
        "reference": "R rddensity::rdplotdensity 2.6 (lpdensity 2.5)",
        "reference_versions": {"R": _RD_IV_R, "rddensity": "2.6", "lpdensity": "2.5"},
        "tolerance": "f_p, f_q, se_p, se_q at every grid point rel 1e-8 (observed 8.0e-12); nh exact",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_iv_rd_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_rd_R.json"],
        "note": (
            "Mass-point data and the Senate margin. Through 1.28.0 each side used "
            "its own ECDF (density at the cutoff 0.60 vs R's 0.23 on the fixture)."
        ),
    },
    "mccrary_test": {
        "status": "bit-exact",
        "reference": "R rdd::DCdensity 0.57 (CRAN archive)",
        "reference_versions": {"R": _RD_IV_R, "rdd": "0.57"},
        "tolerance": "theta, se, z rel 1e-9; p rel 1e-8; bin width 1e-12 (observed 6.5e-12)",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_iv_rd_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_rd_R.json"],
        "note": "Default and fixed bin/bandwidth, and the Senate data. Through 1.28.0 a different estimator with silent fallbacks.",
    },
    "rdhte": {
        "status": "bit-exact",
        "reference": "R rdhte::rdhte 0.2.0 (sandwich 3.1.1)",
        "reference_versions": {"R": _RD_IV_R, "rdhte": "0.2.0", "sandwich": "3.1.1", "rdrobust": "4.0.0"},
        "tolerance": "coef, coef.bc, se.rb, vcov rel 1e-9 (observed 2.2e-12); bandwidths 1e-8",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_iv_rd_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_rd_R.json"],
        "note": (
            "11 cells: continuous and 0/1-subgroup moderators, HC0-HC3, CR1, p = 2, "
            "three kernels, selected and fixed bandwidths. Through 1.28.0 inference "
            "was conventional, h a rule of thumb and binary z not subgroups."
        ),
    },
    "rdbwhte": {
        "status": "bit-exact",
        "reference": "R rdhte::rdbwhte 0.2.0",
        "reference_versions": {"R": _RD_IV_R, "rdhte": "0.2.0", "rdrobust": "4.0.0"},
        "tolerance": "bandwidths rel 1e-8 (continuous and per-subgroup)",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_iv_rd_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_rd_R.json"],
        "note": "rdrobust::rdbwselect on x (per subgroup for a 0/1 moderator).",
    },
    "rdhte_lincom": {
        "status": "bit-exact",
        "reference": "R rdhte::rdhte_lincom 0.2.0",
        "reference_versions": {"R": _RD_IV_R, "rdhte": "0.2.0"},
        "tolerance": "estimate, z, CI, joint chi-square rel 1e-9; p 1e-8",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_iv_rd_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_rd_R.json"],
        "note": "Subgroup difference and continuous CATE at z = 1 (linfct=).",
    },
    "rd_bias_aware_fuzzy": {
        "status": "aligned",
        "reference": "R RDHonest 1.0.1.9000, RDHonest(y | d ~ x)",
        "reference_versions": {"R": _RD_IV_R, "RDHonest": "1.0.1.9000"},
        "tolerance": (
            "estimate, std.error, maximum.bias, conf.low/high, M, first stage: rel "
            "1e-9 at fixed h and M (observed 1.1e-14), 1e-6 with selected h "
            "(observed 3.5e-8, optimiser)"
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_iv_rd_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_rd_R.json"],
        "note": (
            "The record covers every ingredient and RDHonest's linearised interval "
            "(model_info['bias_aware']['rdhonest']). The headline Anderson-Rubin-"
            "type set has no reference implementation; it is pinned by its "
            "defining identity |T(t)| = cv(b(t)) at both endpoints. Through 1.28.0 "
            "the bias bound was h^2 M / 12 (below the Holder worst case)."
        ),
    },
    "rdsensitivity": {
        "status": "aligned",
        "reference": "R rdlocrand::rdsensitivity / rdrandinf 2.0",
        "reference_versions": {"R": _RD_IV_R, "rdlocrand": "2.0"},
        "tolerance": (
            "per-window estimates rel 1e-9 (rdrandinf observed statistic); "
            "randomization p-values within 4 pooled binomial SEs at 4000 draws "
            "(Monte Carlo on both sides, not parity)"
        ),
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_iv_rd_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_rd_R.json"],
        "note": "T3 for the p-values. R's rdsensitivity reports a p-value surface over tau; StatsPAI's tau = 0 column is compared.",
    },
    "rdrbounds": {
        "status": "aligned",
        "reference": "R rdlocrand::rdrbounds 2.0",
        "reference_versions": {"R": _RD_IV_R, "rdlocrand": "2.0"},
        "tolerance": "upper and lower bounds within 4 pooled binomial SEs at 4000 draws (Monte Carlo on both sides, not parity)",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rd_iv_rd_R_parity.py",
                 "tests/reference_parity/_fixtures/rd_iv_rd_R.json"],
        "note": "T3. Through 1.28.0 only the median threshold was used: upper bounds 0.012 / 0.056 / 0.159 vs R 0.021 / 0.094 / 0.346 (anti-conservative).",
    },
    "multi_cutoff_rd": {  # copy of the rdmc record plus the alias test
        "status": "bit-exact",
        "reference": "rdmulti::rdmc 2.0.0 (Cattaneo, Titiunik, Vazquez-Bare & Keele)",
        "reference_versions": {"R": "R version 4.5.2 (2025-10-31)", "rdmulti": "2.0.0"},
        "tolerance": "identical to sp.rdmc on the fixture (exact); per-cutoff and pooled estimates vs R 1e-9 rel",
        "sides": ["py", "R"],
        "test": ["tests/reference_parity/test_rdmulti_parity.py",
                 "tests/reference_parity/_fixtures/rdmulti_R.json"],
        "note": "Alias of sp.rdmc (returns rdmc(*args, **kwargs)); asserted equal to rdmc and to R in test_multi_cutoff_rd_is_rdmc_on_the_r_fixture.",
    },
}
```

Existing record to update: `conditional_lr_ci` — now `method='exact'` by
default; test `test_clr_confidence_set_matches_ivmodel_exactly` asserts
rel 5e-5 (observed 1.5e-7 / 3.4e-8 on `weakiv_R.json`, limited by ivmodel's
`uniroot` tolerance); the Monte-Carlo convergence test is kept for
`method='simulate'`. Suggested tolerance text: "exact conditional critical
value (numerical integration, as ivmodel): endpoints 1.5e-7 vs ivmodel,
asserted 5e-5 (ivmodel's uniroot default tolerance); method='simulate'
retains the n_sim-convergence assertion". `k_test_ci` could also be
promoted (K set hull agrees with Stata weakiv's display string to 5e-6).

Alias proofs added (`src/statspai/_parity_taxonomy.py::TRACK_A_ALIASES`,
tests in `test_track_a_alias_equivalence.py`): `rdd` → `rdrobust` @ 06_rd
(legs estimates, bandwidths: budget 1e-12, observed 0), `geographic_rd` →
`rdms` @ 89_rdms (estimates: 1e-12, observed 0). **The index must be rebuilt
before `tests/test_parity_index.py::test_track_a_aliases_are_all_proven`
passes** (it looks the aliases up in `_parity_index.json`).

Also note: `rdrobust`'s existing Track A record is unaffected (deriv = 0);
its deriv = 1 path is now pinned by the rkd rows above.

## 4. CHANGELOG / MIGRATION

**⚠️ Correctness fixes**

- `sp.tF_critical_value` / `sp.tF_adjustment`: the tF table was not Lee,
  McCrary, Moreira & Porter's (c(10) 3.16 instead of 3.4353; 1.96 from F = 75
  instead of 106.09); replaced by the table in R `ivDiag::tF`, reproduced
  exactly. All old values were too small. Recompute any tF interval.
- `sp.iv_diag`, `sp.anderson_rubin_test`, `sp.weakrobust`: the tF critical
  value is indexed by the effective (robust / cluster) F, not the classical F,
  and is not reported with more than one instrument (NaN / None).
- `sp.iv_diag`: `se_ols` now honours `vcov` / `cluster`.
- `sp.weakrobust`, `sp.conditional_lr_ci`, `sp.k_test_ci`, `conditional_lr_test`:
  instruments were not orthonormalised with two or more instruments; CLR and K
  statistics were mis-scaled (4% on the reference fixture). The K statistic
  "at h0" is now evaluated at h0. The CLR critical value / p-value is now the
  exact conditional distribution (`method='exact'`; `'simulate'` keeps the
  Monte-Carlo version; `weakrobust(clr_method=)`).
- `sp.jive`: `variant='jive2'` is now the Angrist-Imbens-Krueger JIVE2 (Stata
  `ujive2`); the default standard error is the IV sandwich (it was about half).
- `sp.rdrobust`: `p` defaults to R's rule (1, or deriv + 1) and an explicit
  `p` is no longer raised silently; `deriv > p` raises; an empty kernel window
  raises instead of returning a zero-SE estimate.
- `sp.rkd`: default bandwidth is rdrobust's `mserd` for the first derivative;
  fuzzy SE includes the covariance of the two kinks.
- `sp.rdplot`: bins, bin intervals and the fitted polynomial are R
  `rdplot`'s; `kernel` is honoured and defaults to `'uniform'` (R's). Numbers
  on `fig.rdplot_data`.
- `sp.rdplotdensity`: R `rdplotdensity` / `lpdensity`; the old curves were
  about 2x too high. Numbers on `fig.rdplotdensity_data`.
- `sp.mccrary_test`: now R `rdd::DCdensity`; new `bin_width=`. The McCrary
  p-value in `sp.rdrobust(...).model_info['mccrary']` changes accordingly.
- `sp.rdhte` / `sp.rdbwhte` / `sp.rdhte_lincom`: R `rdhte` (robust
  bias-corrected HC3 inference, rdbwselect bandwidth, 0/1 moderators as
  subgroups; new `q`, `vce`, `linfct`, ...).
- `sp.rd_bias_aware_fuzzy`: worst-case bias from the realised weights (was
  `h^2 M / 12`), RDHonest's rule-of-thumb M and default bandwidth,
  nearest-neighbour variance; RDHonest's interval added to `model_info`.
- `sp.rdrbounds`: maximum / minimum over all Rosenbaum thresholds with
  Bernoulli assignment (R `rdrbounds`); the median-split upper bounds were too
  small.

**Fixed (no numeric change):** `sp.rlasso_effect(s)` warns on unidentified
targets; `sp.lasso_iv` no longer attributed to BCCH (2012) and no longer
builds an n×n matrix; `sp.ivqreg` docstring no longer claims agreement with
Stata `ivqreg2` / "quantreg::ivqreg".

**Added:** reference-parity tests and generators listed at the top.

**MIGRATION rows**

| function | what moves | how to get the old number |
| --- | --- | --- |
| `tF_critical_value`, `tF_adjustment` | every value between F = 4 and 106 | none (old table was wrong) |
| `iv_diag` | `tF_critical_value`, `tF_adjusted_ci`, `se_ols`, `ci_ols`, `p_ols`; tF NaN for k > 1 | none |
| `weakrobust` | `clr_stat`, `clr_pvalue`, `clr_ci`, `k_stat`, `k_pvalue`, `k_ci`, `tF_critical_value` | `clr_method='simulate'` restores simulation only |
| `jive` | `variant='jive2'` estimates; default SEs | none |
| `rdrobust` | results for explicit `p <= deriv` | pass `p=deriv+1` |
| `rkd` | default bandwidth, fuzzy SE | pass `h=` explicitly |
| `rdplot` | bin count / positions, CI bars; default kernel string | `nbins=` to fix bins |
| `rdplotdensity` | every curve | none |
| `mccrary_test` | estimate / SE / p; `model_info['n_bins']` is now the number of histogram cells | none |
| `rdhte`, `rdbwhte`, `rdhte_lincom` | SE / CI / p (robust), default bandwidth, binary-z output, `rdbwhte` return type for subgroups | `vce='hc1'` for the HC1 flavour; conventional SE not exposed |
| `rd_bias_aware_fuzzy` | CI, M defaults, h default, naive SE (nearest-neighbour) | none |
| `rdrbounds` | both bounds | none |

## 5. Not closed

- **`rd2d`, `rd2d_bw`, `boundary_rd` (open).** R `rd2d` 1.0.0 (installed)
  estimates *point-wise* boundary effects (`rd2d` bivariate local
  polynomial; `rd2d.distance` on Euclidean distance to each evaluation
  point) with its own bandwidth selectors (`rdbw2d`, `rdbw2d.distance`).
  `sp.rd2d`'s distance approach runs a univariate RD on the signed distance
  to the boundary *line* and reports one pooled effect; its bandwidth is a
  Silverman / curvature rule of thumb; `model_info['bandwidth']` is rounded
  to 6 decimals. Different estimand and estimator: needs a rebuild on R
  rd2d's construction (as `rdms` was), roughly 1,500 lines of R to port.
  `boundary_rd` is an alias and inherits whatever `rd2d` becomes.
- **`rd_multi_score`, `multi_score_rd` (class 6).** Method of arXiv
  2508.15692; no R / Stata package found (CRAN search for rd*/honest names).
- **`zero_first_stage` (class 6).** The van Kippersluis–Rietveld procedure
  has no package; its component regressions are `sp.regress(robust='hc1')` /
  `sp.hdfe_ols`, already T2 elsewhere; the corrected-estimate SE is a
  bootstrap (T3 at best).
- **`lasso_iv` (class 6).** BIC / AIC / CV-penalised Lasso instrument
  selection + 2SLS has no reference implementation; the BCCH estimator is
  `sp.rlasso_iv` (T2 vs hdm). Mislabel fixed.
- **`ivqreg` (class 6, partial).** Just-identified case reproduces the
  inverse-QR root computed with R `quantreg::rq` (1e-13); `IVQR` 0.1.0, the
  only CH package found, errors on R 4.5 ("the condition has length > 1" in
  `ivqr.vc`); Stata `ivqreg2` is Machado–Santos Silva (different estimator),
  `ssc describe ivqreg` → r(601). Over-identified models use identity
  weighting (CH use the inverse covariance of b̂) — a documented deviation,
  not fixed; porting quantreg's kernel covariance is the route.
- **`rdsensitivity`, `rdrbounds` (T3).** Randomisation p-values cannot be
  pinned; compared within MC error only.
- **`effective_f_test` labels.** `stock_yogo_10pct = 23.1` is Montiel
  Olea–Pflueger's k = 1 worst-case τ = 10% critical value (weakivtest reports
  23.109 for k = 1, 11.570 for the k = 3 fixture), not Stock–Yogo's, and the
  "strength" text uses it for every k. Not changed (key is public API,
  `weak_iv.py` is conflict-prone); porting weakivtest's critical values is the
  fix.
- **Stata `weakiv` residual.** CLR / K agree with ivmodel and StatsPAI to
  1.1e-7 and AR to 2e-8; not bisected (inside budget; ivmodel matches
  StatsPAI to 1e-12). Its CLR p-value under `small` differs by 1.4e-3
  relative and was not compared.

## 6. Integrator checklist

- **.gitignore**: add `tests/reference_parity/_fixtures/_ado_rd_iv/` (private
  Stata ado dir created by `_generate_rd_iv_stata.do`).
- **registry.py**: `mccrary_test` gains `ParamSpec("bin_width", "float", False,
  None, "Histogram bin width (DCdensity's bin); auto if None")` after `alpha`
  (`tests/test_registry.py::test_recent_handwritten_specs_match_callable_signature`
  fails until then); `rdrobust`'s `ParamSpec("p", ...)` default 1 → None
  ("1, or deriv + 1 when deriv > 0 (R's rule)").
- **Signature changes (re-dump schemas)**: `rdrobust(p=None)`,
  `rdplot(kernel='uniform')`, `rdplotdensity(h: float | (float, float))`,
  `mccrary_test(bin_width=)`, `rdhte(q=, vce=)`, `rdbwhte(q=, bwselect=,
  vce=, cluster=)` (return type), `rdhte_lincom(weights=None, linfct=)`,
  `weakrobust(clr_method=)`, `conditional_lr_test(method=)`,
  `conditional_lr_ci(method=)`.
- **Conflict-prone files touched** (another line edits them):
  `src/statspai/diagnostics/weak_iv.py` (tF table, AR tF indexing,
  weakrobust K / clr_method — real defects only). Not touched:
  `rd/diagnostics.py`, `rd/dashboard.py`, `regression/iv.py`.
- **Index rebuild** needed for the two alias proofs and the promotions above;
  `REFERENCES.md` / `R_PACKAGE_VERSIONS.md` rows for rdhte 0.2.0, rdd 0.57
  (CRAN archive), ivDiag 1.0.6, lfe 3.1.1, quantreg 6.1, lpdensity 2.5, and
  the Stata packages above.
- R packages installed into the user library for this work: `rd2d`, `rdhte`,
  `ivDiag` (+ deps), `rdd` 0.57 from the CRAN archive, `IVQR` from GitHub
  (`yuchang0321/IVQR`, unusable on R 4.5, not used by any fixture).

## 7. Test status in this worktree

- `tests/reference_parity/`: 3115 passed, 2 skipped (full suite).
- IV / RD / lasso / diagnostics unit tests: all pass except four that
  wait on integrator-owned files — `test_parity_index.py`
  (`test_snapshot_matches_fresh_regeneration`,
  `test_public_parity_doc_is_in_sync`, `test_track_a_aliases_are_all_proven`:
  index rebuild) and `test_registry.py::test_recent_handwritten_specs_match_callable_signature`
  (`mccrary_test` `bin_width` ParamSpec).
- Existing tests re-pinned to the corrected behaviour (each with a comment):
  `tests/iv/test_iv_diag.py` (tF NaN for k=2), `tests/test_article_aliases.py`
  (`conditional_lr_ci` critical value now exact chi2(1) 3.841459, was a
  simulated 3.7807), `tests/reference_parity/test_weakiv_meta_parity.py`
  (exact CLR set vs ivmodel; MC convergence kept for `method='simulate'`),
  `tests/test_diagnostics.py` (McCrary `n_bins`), `tests/test_cov95_rd_*`
  (rdplot < 20 obs and rdplotdensity sparse side now raise, as R does;
  rdhte `p=0` legal, rdbwhte subgroup frame).
- Pre-existing doctest failures unrelated to this work (also on `main`):
  `_article_aliases.psm`, `_article_aliases.rdd`.
