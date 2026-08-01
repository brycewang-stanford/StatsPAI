# Dynamic Panel GMM — Two-Month Hardening Plan

**Owner:** StatsPAI core · **Window:** 8 weeks · **Status doc:** live, updated as
each work item lands.

The 2026-07 classic-design audit rated dynamic panel GMM **"weakest classic
design"**: 2 exported functions (`sp.xtabond`, `sp.gmm`), one materialised
`plm::pgmm` parity row, and a large surface of unimplemented mainstream
functionality. This document is the remediation plan and its running log.

---

## 0. Baseline established 2026-07-31 (measured, not assumed)

Reference toolchain confirmed available on this machine:

| Reference | Version | Role |
| --- | --- | --- |
| Stata 18 MP `xtabond` / `xtdpd` / `xtdpdsys` | built-in | primary parity anchor |
| Stata `xtabond2` (Roodman) | SSC, installed 2026-07-31 | collapse / system GMM / diff-in-Hansen anchor |
| Stata `xtdpdgmm` (Kripfganz) | SSC, installed 2026-07-31 | modern moment-set anchor |
| R `plm::pgmm` | 2.6.7 | secondary anchor |
| R `pdynmc` | 0.9.12 | third anchor (iterated GMM, nonlinear moments) |

### 0.1 What already works (verified)

`sp.xtabond` on `webuse abdata` (140 firms, unbalanced 1976–1984), spec
`xtabond n, lags(1) noconstant vce(robust)`:

| quantity | StatsPAI | Stata 18 | rel. err |
| --- | --- | --- | --- |
| ρ̂ (L1.n) | 1.02334907 | 1.0233491 | ~1e-8 |
| robust SE | 0.10353203 | 0.10353203 | ~1e-8 |
| N (diff. obs) | 751 | 751 | exact |
| # instruments | 28 | 28 | exact |
| AR(1) z | −2.585866 | −2.585866 | ~1e-7 |
| AR(2) z | −1.108055 | −1.108055 | ~1e-7 |

So the *core one-step difference-GMM arithmetic is correct and Stata-exact*,
including the block-diagonal instrument enumeration, the MA(1) `H` weight, the
robust sandwich, and the Arellano–Bond serial-correlation tests on an
**unbalanced, gapped** real panel. This is the foundation the rest builds on.

### 0.2 What is broken or missing (measured)

**B1 — listwise deletion destroys the instrument set (correctness).**
`xtabond()` starts with `data[[id, time, y] + x].dropna()`. A `NaN` in *any*
covariate at period *t* deletes the whole row, which removes `y_{i,t}` from the
**instrument** pool and from the lag pool, not just from the estimation sample.
Reproducing Arellano–Bond (1991) Table 4 requires user-built lag columns, whose
leading `NaN`s then silently amputate the design:

| | StatsPAI (user-built lags) | Stata `xtabond n l(0/1).w l(0/2).k, lags(2)` |
| --- | --- | --- |
| N | **331** | 611 |
| # instruments | **19** | 32 |
| ρ̂₁ | **0.6597** | 0.8491 |

A 22 % coefficient error on the canonical dataset of the literature. Not a
tolerance issue — a design-matrix bug.

**B2 — no lag operator.** There is no way to write `l(0/2).k`. Users must build
lag columns by hand, which triggers B1. No formula interface either.

**B3 — `method='system'` raises `NotImplementedError`.** Blundell–Bond system
GMM is the *default* choice in applied work for persistent series (ρ near 1 —
exactly the abdata case above, where difference GMM returns ρ̂ = 1.02). The
`sp.panel` dispatcher documents `method='system'` and routes to a hard failure.

**B4 — no instrument-class distinction.** Every `x` is treated as strictly
exogenous and enters as its own single `Δx` standard-instrument column. There is
no `predetermined` or `endogenous` class, i.e. no equivalent of `xtabond2`'s
`gmm(x, lag(1 .))` / `gmm(x, lag(2 .))` / `iv(x)` split. Most applied dynamic
panel specifications are therefore inexpressible.

**B5 — no `collapse`.** Instrument proliferation is *the* practical failure mode
of dynamic panel GMM (Roodman 2009 §5). With `T = 9` and full lag depth the
instrument count grows as O(T²) and overfits the endogenous regressor, biasing
estimates toward the within estimator and inflating the Hansen p-value toward
1.0. `collapse` is the standard remedy and is absent.

**B6 — no constant, no time dummies.** `noconstant` is the only behaviour.
Roodman's standing advice is to always include time dummies in dynamic panel
GMM (they make the no-cross-sectional-correlation assumption more plausible).

**B7 — no forward orthogonal deviations.** Arellano–Bover FOD is the
recommended transform for gapped panels because it does not propagate a gap
into two lost observations. Absent → the ~1 % gapped-panel discrepancy the
docstring currently warns about.

**B8 — no difference-in-Hansen tests.** System GMM's extra level moments are
only credible if their incremental validity is testable (`estat overid`,
`xtabond2`'s "Difference-in-Hansen tests of exogeneity of instrument subsets").
Without it, system GMM cannot be shipped responsibly even once B3 is done.

**B9 — no instrument-count guardrail.** No warning when
`#instruments > #units`, the single most cited diagnostic red flag.

**B10 — AR test variance ignores the Windmeijer correction under
`twostep=True, robust=True`** (docstring admits ~0.1 % deviation from Stata).

**B11 — `sp.gmm` is a thin BFGS wrapper.** Numerical Jacobian by finite
differences; no analytic linear-GMM path; no HAC/cluster weight matrices; no
iterated-GMM convergence reporting; `se='unadjusted'` returns `(D'WD)^{-1}/n`,
which is only valid at the efficient `W = S^{-1}`; CUE re-inverts `S` inside the
objective with a silent identity fallback. No parity test against R `gmm::gmm`
or Stata `gmm`.

**B12 — performance.** Instrument construction is a per-unit Python loop with
dict lookups; the Windmeijer correction loops `k × N_units`. Untested above a
few hundred units.

**B13 — thin API surface.** No `sp.xtdpdsys`, no Anderson–Hsiao baseline, no
bias-corrected LSDV (Kiviet/Bruno) companion, no dynamic-panel `estat`
integration, no chooser guide in `docs/guides/`.

---

## 1. Design decisions taken up front

**D1. One estimator, many moment sets.** Rather than separate `xtabond` /
`xtdpdsys` / `xtabond2` implementations, build a single moment-set compiler:
a specification declares, per variable, (instrument class, lag range, collapse,
which equation). Difference GMM, system GMM, Anderson–Hsiao and level-only GMM
are then *configurations*, not codepaths. This is how `xtdpdgmm` is structured
and it is what makes difference-in-Hansen tests natural (drop a moment subset,
refit, difference the J statistics).

**D2. Backwards compatibility is absolute.** `sp.xtabond(df, y=..., x=[...])`
must keep returning bit-identical numbers. Every new capability is opt-in via
new keyword arguments with defaults reproducing today's behaviour. The one
exception is B1, which is a **⚠️ correctness fix** and gets CHANGELOG +
MIGRATION treatment.

**D3. Reference-first.** No moment set ships without a Stata (and where
possible R) parity fixture committed under
`tests/reference_parity/_fixtures/`. Fixtures are generated by checked-in
scripts so they are reproducible, and the generated JSON is committed so CI does
not need Stata/R.

**D4. Package split.** `gmm/arellano_bond.py` (672 lines) becomes
`gmm/_dynpanel/` with `_spec.py`, `_data.py`, `_moments.py`, `_estimate.py`,
`_inference.py`, `_diagnostics.py`. `gmm/arellano_bond.py` remains as the public
`xtabond` entry point. Per CLAUDE.md §4, ~800 lines/file, split by concern.

**D5. Fail loudly.** Instrument-count overrun, rank-deficient weight matrix,
collapsed-instrument under-identification, and non-stationary ρ̂ ≥ 1 under
difference GMM all warn with actionable text and land in `model_info`.

---

## 2. Schedule

### Week 1 — Harness, correctness fix, refactor

| # | Item | Definition of done |
| --- | --- | --- |
| 1.1 | Reference fixture generators: `_generate_dynpanel_stata.do`, `_generate_dynpanel_R.R` over abdata + 3 simulated panels (balanced, ragged, interior-gapped) | `dynpanel_stata.json`, `dynpanel_R.json` committed; ≥40 reference rows |
| 1.2 | **B1 fix**: separate availability frame from estimation frame | abdata AB(1991) Table-4 spec matches Stata `xtabond` to ≤1e-8 |
| 1.3 | **B2**: lag-operator spec (`"l(0/2).k"`, `"L2.n"`) resolved internally | canonical spec expressible in one call |
| 1.4 | **D4** package split, no numeric change | full test suite green, `git diff` shows pure move + hooks |
| 1.5 | Regression lock: golden-value test on today's outputs before refactor | guards D2 |

### Week 2 — Instrument classes + collapse

| # | Item | Definition of done |
| --- | --- | --- |
| 2.1 | **B4** moment-set compiler: `endogenous=`, `predetermined=`, `exogenous=` with per-variable lag ranges | matches `xtabond2 ... gmm(...) iv(...)` |
| 2.2 | **B5** `collapse=True` | matches `xtabond2, collapse` to ≤1e-8 |
| 2.3 | **B6** `constant=True`, `time_dummies=True` | matches `xtabond2` with `i.year` |
| 2.4 | **B9** instrument-count guardrail + `model_info['n_instruments']` vs `n_units` warning | warning fires on abdata full-depth spec |

### Week 3 — System GMM

| # | Item | Definition of done |
| --- | --- | --- |
| 3.1 | **B3** level-equation moments, stacked system, correct `H` for the system weight | one-step matches `xtdpdsys` and `xtabond2` to ≤1e-8 |
| 3.2 | Two-step + Windmeijer for the system | matches `xtabond2, twostep robust` |
| 3.3 | System AR(1)/AR(2), Sargan, Hansen | matches `xtabond2` output block |
| 3.4 | `sp.xtdpdsys` public alias + registry + dispatcher `method='system'` unblocked | `sp.panel(method='system')` works |

### Week 4 — Transformations & gaps

| # | Item | Definition of done |
| --- | --- | --- |
| 4.1 | **B7** forward orthogonal deviations | matches `xtabond2, orthogonal` (diff and system) to ≤1e-8 |
| 4.2 | Gapped-panel convention audit; align FD path with Stata | interior-gap warning removed or narrowed to a documented, tested residue |
| 4.3 | Gapped/ragged fixtures wired into the parity suite | 3 gap topologies × 4 specs green |

### Week 5 — Inference & diagnostics

| # | Item | Definition of done |
| --- | --- | --- |
| 5.1 | **B8** difference-in-Hansen for arbitrary instrument subsets | matches `xtabond2` diff-in-Hansen block |
| 5.2 | **B10** Windmeijer-corrected AR-test variance | two-step robust AR z matches Stata to ≤1e-8 |
| 5.3 | `estat`-style postestimation surface (`sp.estat(result, 'abond' / 'sargan' / 'overid')`) | registered, schema-exported |
| 5.4 | Cluster-robust VCE with user-supplied cluster; small-sample correction options | matches `xtabond2` under `cluster()` |

### Week 6 — Companion estimators

| # | Item | Definition of done |
| --- | --- | --- |
| 6.1 | Anderson–Hsiao IV baseline (`method='ah'`, level and difference instruments) | matches hand-computed 2SLS + `xtdpdgmm` |
| 6.2 | Bias-corrected LSDV (Bruno 2005 / Kiviet 1995) `sp.xtlsdvc` | matches Stata `xtlsdvc` |
| 6.3 | Iterated GMM + CUE for the dynamic panel | matches `pdynmc` iterated |

### Week 7 — `sp.gmm` overhaul

| # | Item | Definition of done |
| --- | --- | --- |
| 7.1 | **B11a** analytic linear-GMM path (`linear=` moment spec) | closed-form, matches Stata `gmm` linear |
| 7.2 | **B11b** HAC (Newey–West/Bartlett) and one-way cluster weight matrices | matches R `gmm::gmm(vcov="HAC")` |
| 7.3 | **B11c** correct `se='unadjusted'`, honest CUE, iteration diagnostics | no silent identity fallback; `converged` in `model_info` |
| 7.4 | Parity file `test_general_gmm_parity.py` vs R `gmm::gmm` and Stata `gmm` | ≥12 reference rows |

### Week 8 — Performance, integration, docs, release

| # | Item | Definition of done |
| --- | --- | --- |
| 8.1 | **B12** vectorised/sparse instrument construction and Windmeijer | ≥10× on N=5000, T=10; benchmark committed |
| 8.2 | Registry / schema / `sp.help` / `_house_style` entries for every new symbol | `registry_stats.py --check` clean |
| 8.3 | `docs/guides/choosing_dynamic_panel_estimator.md` | published, cross-linked |
| 8.4 | Coverage ≥95 % on `gmm/` | `pytest --cov=statspai.gmm` |
| 8.5 | CHANGELOG + MIGRATION (⚠️ correctness fix for B1), version bump | ready for release; **no GitHub Release while JOSS review is open** |

---

## 3. Risk register

| Risk | Mitigation |
| --- | --- |
| Changing B1 changes existing users' numbers | It is a genuine correctness fix; MIGRATION entry + `⚠️ correctness` CHANGELOG; the old behaviour was never Stata-consistent for specs with covariate `NaN`s |
| System GMM has several conventions in the wild (`xtdpdsys` vs `xtabond2`) | Match `xtabond2` as primary (it is what the literature reports) and pin `xtdpdsys` divergences as documented, tested facts |
| Refactor breaks the bit-exact abdata parity | Week 1.5 golden-value lock lands *before* the refactor |
| JOSS review in flight | Work is additive to a module the paper mentions only in passing; no `paper.md` scope change unless a new exported symbol demands it. **No GitHub Release.** |

---

## 4. Running log

Entries appended as work lands. Format: `[week.item] date — outcome`.

<!-- LOG START -->

### 2026-07-31 — Weeks 1–5 landed

**[1.5] Golden-value lock.** `tests/test_xtabond_golden.py` pins eight
specifications (one/two-step × robust/classical, `lags=2`, capped
`gmm_lags`, gapped panel) at `rtol=1e-8` on coefficients, SEs, sample size,
instrument count, both AR z statistics and Sargan/Hansen. Landed *before*
any refactor. Every subsequent change kept it green; the one deliberate
revision is documented in the file (one-step fits now report a Hansen J
where they previously reported `NaN` — an added field, not a changed
number).

**[1.1] Reference harness.** `_generate_dynpanel_stata.do` (26 specs across
`xtabond`, `xtabond2`, `xtdpdsys`) → `_fold_dynpanel_stata.py` →
`dynpanel_stata.json`; `_generate_dynpanel_R.R` (11 specs across
`plm::pgmm` and `pdynmc`) → `dynpanel_R.json`; `dynpanel_abdata.csv`
exported in `%21.16e` so Python estimates on the same bytes Stata did. R
and Stata agree exactly on difference GMM, which rules out a
shared-convention artefact.

Two Stata-18 facts worth recording: `estat sargan` / `estat abond` no
longer leave `r(chi2)`/`r(df)` behind (the collection framework clears
them), so everything is harvested from `e()`; and `e(sargan)` after
`twostep` is the *Hansen J*, not a Sargan statistic.

**[1.2] B1 fixed — the headline correctness bug.** Availability is now per
variable. abdata Table-4 spec: 331 → 611 observations, 19 → 32 instruments,
ρ̂ 0.660 → 0.849, matching Stata to 1.4e-13. CHANGELOG `⚠️ correctness` +
MIGRATION written.

**[1.3] Lag operators.** `l(0/2).k`, `L2.k`, `L.k` parse in `x=`,
`predetermined=`, `endogenous=`.

**[1.4] Package split.** `gmm/_dynpanel/{_spec,_data,_moments,_estimate,
_inference,_diagnostics,_fit}.py`; `arellano_bond.py` is the documented
surface. Numerically identity-preserving (golden lock green throughout).

**[2] Instrument classes, collapse, time dummies, guardrail.** All
reference-verified: `predetermined_lags=(2, None)` reproduces `xtabond ...,
pre(w, lagstruct(1, .))` exactly, `endogenous_lags=(3, None)` reproduces
`endogenous(w, lagstruct(1, .))`, `gmm_lags=(2, 4)` reproduces
`maxldep(3)`, `collapse=True` reproduces `xtabond2, collapse` (28 → 7
instruments, ρ̂ 1.02 → 1.39), year dummies as regressors reproduce
`xtabond ... yr1979-yr1984`.

Discovered and documented: Stata `xtabond`'s `lagstruct()` counts *further*
lags beyond the deepest regressor lag, while `xtabond2`'s `gmm(x, lag(a b))`
counts absolute lags from the equation period. StatsPAI uses the absolute
convention and pins the mapping in tests.

**[3] System GMM.** `method='system'` / `sp.xtdpdsys`, bit-exact against
`xtabond2` one-step, two-step Windmeijer and collapsed. Layout facts that
had to be got right, all now pinned by tests:

- the level equation is instrumented by a *single* lagged difference
  (`Δy_{t-1}` for `gmm_lags=(2, .)`); deeper ones are redundant;
- `iv()` defaults to `equation(both)` — **one** column carrying the
  transformed value on transformed rows and the level on level rows, which
  is why `xtabond2` counts `iv(w k)` as 2 instruments in a system fit, not 4;
- `H` is Roodman's `h(3)`, `[[MM', M], [M', I]]`;
- σ̂² is estimated from the **transformed rows only** (level residuals still
  contain α_i). `xtabond2` divides by `2N*` where `xtabond` divides by
  `2(N* − k)`, so the two Sargan statistics differ by exactly `N*/(N* − k)` —
  asserted as an identity rather than papered over. The Hansen J has no such
  free scale and matches exactly everywhere.

**[4] Forward orthogonal deviations.** `orthogonal=True`, bit-exact against
`xtabond2, orthogonal` for difference and system, one- and two-step. Two
conventions were discovered empirically and are documented in the code:

1. a FOD row is *labelled* with the period of the first difference it
   replaces, so the instrument grid, the level cross-block and the AR test
   share one period index;
2. `H` is built from the **balanced-grid** FOD operator, not each unit's own
   one ("H always has block diagonal form, with all blocks the same" —
   `xtabond2` help). Using the per-unit operator moves the FOD system-GMM
   coefficients by up to 13% on ragged `abdata`.

**[4.2] Gap convention — localised, not fixed.** On a hole-punched `abdata`
the *design* matches Stata exactly: same sample (613), same per-unit row
counts (min 2 / avg 4.37857143 / max 7), same instrument count (28), and a
*just-identified* fit — where the weight matrix cancels — reproduces
`xtabond2` to 2e-15. The over-identified fit does not, by 2–6%. So the
divergence is confined to the one-step weight matrix: StatsPAI uses
`H = M M'` for the actual differencing operator; Stata's gapped convention
is different and undocumented (period-distance, sequence-distance,
zero-off-diagonal and identity variants were all tried and none reproduce
it). Both estimators stay consistent; only finite-sample efficiency differs.
The warning now says exactly this and points at `orthogonal=True`. Pinned by
`TestGappedPanelConvention`.

**[5.1] Difference-in-Hansen.** Per-subset C tests in
`model_info['difference_in_hansen']`, matching `xtabond2`'s block. The
construction matters: the restricted fit must reuse the **full** model's
moment covariance (`A = (Ω[keep, keep])⁻¹`), not a freshly estimated one.
Re-estimating Ω — the natural-looking mistake — turns the level-instrument
verdict on `abdata` from C = 5.32 (p = 0.62, fine) into 18.63 (p = 0.009,
rejected). Both variants were computed side by side and the correct one
identified against Stata.

### Verification state at end of session

Full suite run in two halves (the machine was also running four other
sessions' pytest processes, hence the wall-clock):

- core (`tests/` less parity): **13,062 passed**, 12 failed, 65 skipped.
- `tests/reference_parity/` + contract suites: **1,430 passed**, 2 skipped,
  3 xfailed.

Of the 12 core failures, **4 were this workstream's and are fixed**:

| test | cause | fix |
| --- | --- | --- |
| `test_cov95_panel_reg::test_system_gmm_not_implemented` | asserted the `NotImplementedError` gate this work removed | rewritten to exercise the dispatcher route and check `_cons` / `n_obs_level` |
| `test_panel_cov_diagnostics::test_system_gmm_not_implemented` | same | same |
| `test_schema_export::test_committed_schemas_dir_is_in_sync` | registry gained `xtdpdsys` | `scripts/dump_schemas.py` re-run |
| `test_parity_gap_boundaries::…` | transient — read a mid-edit file during the 50-min run | passes on re-run |

The remaining 8 are **not attributable to this work** and were confirmed
individually:

- 4 × `test_jss_*` — the JSS documents quote registry counts that were
  *already* stale at `HEAD` (`docs/jss_source_audit_dossier.md` says 1,145
  against a live registry of 1,147 before `xtdpdsys` was added), and the
  `publication-grade` string the lint forbids is also present at `HEAD`.
  JSS manuscript wording is release-prep owned elsewhere; left alone
  deliberately.
- `test_agent_blocks_drift` — drift is entirely in
  `docs/guides/qte_family.md` (another session's QTE method rename).
- `test_estimator_provenance_round7::TestDistIvProvenance` — `dist_iv`,
  another session.
- `test_signature_house_style::test_lint_ratchet_holds` — the `se`
  parameter count moved 58 → 59; enumerating every exported symbol with an
  `se` parameter shows none of them is new here (`sp.xtdpdsys` has no `se`).

All dynamic-panel and dispatcher suites green: 196 tests across
`test_dynpanel_abdata_parity.py` (45), `test_gmm.py`,
`test_xtabond_golden.py`, `test_xtabond_windmeijer.py`,
`test_gmm_dynamic_panel_parity.py`, `test_panel_dispatcher.py`,
`test_cov95_panel_reg.py`, `test_panel_cov_diagnostics.py`,
`test_schema_export.py`. `black` + `flake8` clean on `src/statspai/gmm` and
every touched test; `registry_stats.py --check` clean; `arellano_bond.py`
doctests pass.

One behavioural note surfaced by the fixed dispatcher tests: because the
two-step solve is now always computed (the Hansen J is defined at that
optimum), a rank-deficient design can trip the singular-matrix warning even
in a one-step fit. When the user did not ask for two-step, that warning is
now re-worded to say the *Hansen J* is unreliable while the one-step
estimate and its SEs are unaffected, and the note is recorded in
`model_info['hansen_warning']`.

**Note on the working tree.** Several other Claude Code sessions were editing
this repository concurrently (`dml/`, `qte/`, `matching/`, `forest/`,
`registry.py`, `paper.bib`). A `black src/statspai` run early in this session
reformatted eleven files outside this workstream; those were identified
(by checking that their diff was *exactly* what black produces from the
`HEAD` version) and reverted. `registry.py` and `paper.bib` contain
interleaved edits from several sessions.

### 2026-07-31 (session 2) — Weeks 5–8 complete

Everything the plan listed is now implemented. Since the first session:

**[5.2] Two-step AR variance — ⚠️ correctness.** The Arellano-Bond variance
carries a `(W'q)' Avar(β̂) (W'q)` term that was always evaluated at the
*uncorrected* robust sandwich, even when the reported VCE was the
Windmeijer-corrected or conventional two-step one. On the AB(1991) spec the
two-step AR(1) z was −4.32 against Stata's −3.10 (39% error). Swapping in
the reported VCE makes **every** AR statistic across the whole VCE menu
exact. One-step results are untouched (the two VCEs coincide there, so the
swap is identically zero).

**[5.4] `cluster=`.** Cluster-robust SEs on a coarser unit than the panel
id. Only the sandwich meat re-groups. Finer-than-unit clusters raise;
multi-step GMM with fewer clusters than moments refuses rather than
inverting a rank-deficient weight (Stata proceeds with a warning).

**[6.1] Anderson-Hsiao.** `method='ah'`, levels and differences variants,
all four exact against `xtabond2`. Implementing it exposed a real bug:
IV-style instruments are **not** subject to Stata's `missing=0` convention
— only GMM-style ones are — so an IV term reaching deeper than the equation
must *shrink the sample*, not be zero-filled. Zero-filling moved the
`G4` coefficients by up to 60%.

**[6.3] `steps=`.** Any number of steps, `'iterated'` (to a fixed point) or
`'cue'`. Tested analytically rather than against a fixture: the iterated
estimate is verified to be an actual fixed point by re-running one step by
hand, and CUE is verified to attain a strictly lower value of the criterion
it is defined to minimise.

**[7] `sp.gmm` rebuilt.** Closed-form path for affine moments, analytic
`jacobian=`, `vcov=` over iid/robust/HAC/cluster, `center=`, honest
convergence reporting. Two conventions were pinned by experiment: R's
`sandwich` evaluates the Bartlett kernel at `lag/bw` (3% SE difference if
mis-stated), and `gmm::gmm`'s stock `optim` tolerances stop ~1e-4 from the
optimum — tightening R's control list moved *R* onto StatsPAI's answer, and
an independent closed-form solve agrees to 2e-15.

**[8] Performance.** Per-unit Python loops in the weight matrix, moment
covariance, Windmeijer derivative and AR lag vector replaced by segment sums
and banded row-pair products. 20k units × 15 periods, two-step: 2.56 s →
1.12 s; system GMM 6.24 s → 2.94 s. `benchmarks/bench_dynpanel.py`.

**AR provenance.** A consolidated sweep over all 29 Stata specs (checking
AR statistics, not just coefficients) found two configurations where the AR
test is a reconstruction rather than a match — and one outright bug: under
`orthogonal=True` the statistic was computed on the FOD residuals, giving
**+4.11 where Stata reports −3.25**, an inverted conclusion. It now runs on
first-differenced residuals at the fitted coefficients (what "AR in first
differences" means) and lands within 0.5–6%. Clustered AR differs ~10–14%
on a grouping convention. Both are bounded by tests and announced in
`model_info['ar_note']`.

**Final parity.** Coefficients and standard errors: worst relative error
**1.7e-11** across all 29 specs. AR statistics: exact except the two
documented reconstructions.

### 2026-07-31 (session 3) — conventions settled from xtabond2's source

`xtabond2` ships its Mata source (`xtabond2.mata`, 58 KB). Reading
`_ARTests` settled both outstanding AR reconstructions *exactly*, replacing
inference-from-outputs with the implementation itself:

- **AR variance is grouped by unit, never by cluster.** Both accumulations
  (`wHw = Σ_i s_i²` and `ZHw = Σ_i Z_i'e_i · s_i`) loop `for (i = N; i; i--)`.
  Only the third term, `(W'q)' V (W'q)`, picks up the clustering through the
  reported VCE. StatsPAI had been grouping all three by cluster — worth
  10-14% on the statistic. Now exact (3.6e-14).
- **Under `orthogonal`, the AR test mixes two row spaces.** `_ARTests` takes
  `wl`/`w` and `pX` from `_Difference(...)` — the *differenced* residuals and
  regressors — while `ZHw` uses the estimation (FOD) instruments and
  residuals, aligned by unit, and `m2VZXA` uses the estimator's own weight.
  Implemented as `ar_test_cross_basis`; now exact (1e-13) for one- and
  two-step, difference and system.

The full sweep across all 29 Stata/`xtabond2` specifications — coefficients,
standard errors **and** AR statistics — is now **1.7e-11 worst case**.

`_H` confirmed the balanced-grid construction (`H = M'M` on the full T
grid, one matrix shared by all units) that had been established empirically
for forward orthogonal deviations.

### Remaining

Nothing from the original plan. Open items are new observations, not
backlog:

- The gapped-panel weight convention. Timeboxed: settling it needs
  `xtabond2`'s full row-indexing scheme (how `touse` zeroes interact with a
  ``T x T`` `H` on the level grid), which is a materially larger read than
  `_ARTests` was, for a case where both estimators stay consistent and
  `orthogonal=True` is the recommended answer regardless. The divergence is
  bounded, tested, and warned about.
- `xtdpdgmm`-style nonlinear (Ahn-Schmidt) moment conditions were never in
  scope and remain unimplemented.

<!-- LOG END -->
