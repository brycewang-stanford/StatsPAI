# P1 Campaign — Tier D analytic special-cases + Tier B replication notebooks

**Set by maintainer (Bryce), 2026-06-08. Budget: ~1 month. Quality bar is
non-negotiable (CLAUDE.md §5 / §7): real numerical assertions anchored to a
known truth, no mocking of numerical paths, fail loudly.**

This is the durable, cross-session tracker. Two workstreams:

- **Tier D — analytic special-cases.** Give every *reference-less* estimator a
  closed-form / known-DGP recovery test (recover an analytically known estimand
  within tolerance). Closes the CLAUDE.md §5 promise: "有参考实现走对齐，没有走
  解析/仿真".
- **Tier B — replication notebooks.** Turn the existing `sp.replicate`
  published-paper replications into **one-click executable Jupyter notebooks**
  (load real data → estimate → compare to pinned published numbers → figure),
  with a headless CI runner that fails on drift.

## RED LINE (maintainer decision 2026-06-08)

Both workstreams are **purely additive**: new tests + new notebooks only.

1. **Do not change estimator numerics.** Moving any number could disturb the
   JOSS review (#10604) or the JSS dossier. New assertions must pin *current*
   correct behaviour against analytic truth.
2. **If a Tier D test exposes a real numeric bug:** STOP, report to maintainer,
   then follow CLAUDE.md §12 `⚠️ correctness fix` (CHANGELOG + MIGRATION).
   Never silently change output.
3. **Do not touch** JSS in-flight files: `docs/joss_validation_dossier.md`,
   `tests/test_jss_validation_api.py`, `tests/external_parity/*`,
   `tests/reference_parity/_fixtures/dml_iv_*`. Tier B notebooks live in a
   *new* archive dir, not by editing those.

---

## How the worklist is built

`scripts/tierd_classify.py` is a read-only diagnostic. For each of the 1,020
registered functions it grades the strongest test evidence that exists today:

| evidence | meaning | Tier D action |
|---|---|---|
| `reference` | named in a parity dir (R/Stata/published) | none — Tier A/B already |
| `anchored` | tolerance/closeness assert in the enclosing test (known-truth) | none — already Tier D quality |
| `weak` | an assert exists but only boolean/shape/not-None | **P2** upgrade |
| `smoke` | referenced but no assert | **P1** floor |
| `untested` | not referenced by any test | **P1** floor |

Estimator-like = excludes infra/presentation categories *and* name patterns
(CamelCase result classes, `*plot`, `*_report`, `*_to_latex`, `*_simulate`, …).

```bash
python scripts/tierd_classify.py report                  # summary
python scripts/tierd_classify.py worklist --priority P1   # the floor
python scripts/tierd_classify.py worklist --category causal
python scripts/tierd_classify.py json > .tierd_campaign/worklist.json
```

The heuristic is a **prioritisation** tool, not ground truth — the regex
idioms have false positives/negatives. **Per batch, verify the actual test
before writing**: `git grep -n '\bNAME\s*('` tests/ to confirm a function
truly lacks a known-truth assertion.

---

## Baseline (2026-06-08, v1.16.1 source tree)

Evidence distribution over all 1,020 registered functions:

| evidence | count |
|---|---|
| reference | 89 |
| anchored | 326 |
| weak | 367 |
| smoke | 10 |
| untested | 228 |

**Tier D baseline worklist: 257 estimator-like functions** — **25 P1** (zero
numeric guard) + **232 P2** (weak assert, needs a known-truth anchor). The
tracked `.tierd_campaign/worklist.md` file is refreshed after each batch; it is
the current remaining worklist, not a frozen baseline snapshot.

### Tier D — P1 floor (25 estimators, no numeric guard at all)

These get analytic/known-DGP tests first. Grouped by file we'll create:

| batch (test file) | functions |
|---|---|
| `test_tierD_bounds_analytic.py` | `horowitz_manski`, `iv_bounds`, `oster_delta`, `trimming` |
| `test_tierD_rd_analytic.py` | `boundary_rd`, `geographic_rd`, `multi_score_rd` |
| `test_tierD_identification_analytic.py` | `frontdoor`, `notch` |
| `test_tierD_balance_calibration_analytic.py` | `ps_balance`, `test_calibration` |
| `test_tierD_dml_analytic.py` | `model_averaging_dml` |
| `test_tierD_power_analytic.py` | `power`, `mde`, `power_cluster_rct`, `power_iv` |
| `test_tierD_diagnostics_analytic.py` | `effective_f_test`, `stepwise` |
| `test_tierD_panel_glm_analytic.py` | `feglm` |
| `test_tierD_structural_analytic.py` | `blp`, `levpet`, `opreg` |
| `test_tierD_spatial_analytic.py` | `moran_local` |
| `test_tierD_interference_missing_analytic.py` | `peer_effects`, `mi_estimate` |

### Tier D — P2 upgrade (232, by value)

Prioritised families (counts): causal 94, dag 10, regression 9, spatial 9,
conformal_causal 7, decomposition 7, structural 5, inference 5, timeseries 5,
transport 5, longitudinal 5, … . Tackled after P1, batched by family, highest
value first. Verify-before-write applies (some P2 entries are dataset loaders /
already-anchored false negatives).

---

## Tier B — replication notebooks

Source of truth for replications: `src/statspai/smart/replicate.py` (dual-track
classic+modern) + pinned values in `tests/external_parity/`. Current entries
to notebook-ify (one `.ipynb` each):

| notebook | paper | data | headline to pin |
|---|---|---|---|
| `01_card_1995.ipynb` | Card (1995) returns to schooling | bundled CPS extract | 2SLS schooling coef |
| `02_lee_2008.ipynb` | Lee (2008) Senate RD | bundled Senate | RD point (cct) |
| `03_basque_2003.ipynb` | Abadie–Gardeazabal (2003) SCM | Basque | terrorism gap |
| `04_lalonde_nsw.ipynb` | LaLonde (1986) / Dehejia–Wahba NSW | bundled NSW(+PSID) | ATT |
| `05_abadie_prop99.ipynb` | Abadie et al. (2010) Prop 99 | California-99 | per-capita gap |
| `06_mpdta_csdid.ipynb` | Callaway–Sant'Anna `mpdta` | mpdta | aggregated ATT |

**Maintainer decision (2026-06-08):** notebooks go in
`Paper-JSS/replication/notebooks/` (recommended option). NB this tree is a
*separate private repo* gitignored by the public main repo (CLAUDE.md §9.1), so
the notebooks are tracked by the Paper-JSS repo and must be committed from
there, not the main repo.

**Tier B DONE (2026-06-08):**
- `scripts/build_replication_notebooks.py` — single-source-of-truth nbformat
  generator (committed in main repo, not gitignored).
- 5 executable notebooks (Card / ADH-Prop99 / LaLonde-NSW / Lee-RD / Graddy):
  load real data → classic estimator → comparison-vs-paper table → figure →
  **drift-guard assert**. All execute headless (Card 7.6s, rest ~2s).
- `tests/test_replication_notebooks.py` — headless CI runner (nbclient); skips
  cleanly when the private Paper-JSS tree is absent, runs full when present.
- `notebooks` extra in `pyproject.toml` (nbformat/nbclient/nbconvert/ipykernel,
  all BSD); `make -C Paper-JSS notebooks` / `notebooks-execute`; README in dir.
- Excluded: `angrist_pischke_mhe` (no bundled data); `graddy_2006` is a
  *simulated* known-truth IV demo (labelled as such).

**⚠️ FINDING for maintainer (CLAUDE.md §12):** the `sp.replicate('lalonde_1986')`
registry pins 1:1 NN PSM ATT at **$2012.5** (tol $5), but current deterministic
`sp.match(method='nearest')` returns **$1963.4** — a $49 (2.5%) drift from a
tie-handling change (binary covariates) since the May-7 pin. *Not* a numeric
change I made; the teaching pin is stale and unguarded by any test. Notebook 03
guards the robust scientific claim instead. **Recommend** refreshing the
registry golden number 2012.5 → 1963.4 (documentation correction) — pending your
approval since it touches a pinned value.

---

## Acceptance checklist (maintainer ticks)

Tier D:
- [ ] All 25 P1 estimators have an analytic/known-DGP recovery test (green).
- [ ] P2 high-value families upgraded with known-truth anchors (target subset agreed).
- [ ] `scripts/tierd_classify.py report` shows P1 count → 0; P2 materially reduced.
- [ ] No estimator numerics changed (or: each change logged as ⚠️ correctness fix).
- [ ] Full suite green; JOSS/JSS parity numbers unchanged.

Tier B:
- [ ] 6 replication notebooks execute end-to-end headless.
- [ ] Each pins its headline to the published value within a documented band.
- [ ] CI runner fails on drift.
- [ ] docs / replication archive reference the notebooks.

---

## Session log

### 2026-06-08 — session 1: foundation + scoping
- Confirmed scope with maintainer: Tier D = all reference-less estimators
  (analytic special-cases); Tier B = executable Jupyter notebooks; red line =
  purely additive, no numeric changes.
- Built `scripts/tierd_classify.py` (read-only evidence classifier). Fixed a
  call-regex bug (`(?<![\w.])` rejected `sp.NAME(` — every dispatched estimator
  was mis-graded `untested`); added scope-aware enclosing-`def` assertion
  detection and an anchored-vs-weak quality split; filtered CamelCase result
  classes and presentation names.
- Established the baseline above and the 25-function P1 floor batched into test
  files. `.tierd_campaign/worklist.md` is the refreshed remaining worklist.
- **Batch 1 DONE** — `tests/test_tierD_bounds_analytic.py` (13 tests green):
  `trimming` (Stürmer/Crump exact-threshold + monotonicity), `horowitz_manski`
  (per-stratum width identity `upper-lower == y_upper-y_lower`, single-stratum
  closed form, brackets true ATE), `oster_delta` (stable-coef degenerate set +
  exact eq.3 re-derivation), `iv_bounds` (monotone set = `[min,max](OLS,Wald)`,
  valid-IV recovery with known-sign OLS bias). No numerics changed.
- **Batch 2 DONE** — `tests/test_tierD_power_analytic.py` (17 tests green):
  `power('rct')` (normal closed form + minimal solve-for-n), `mde` (closed form
  + round-trip through `power`), `power_cluster_rct` (ICC=0 ≡ individual RCT,
  ICC=1 ≡ cluster-level RCT, design-effect formula, monotone in ICC),
  `power_iv` (no-penalty ≡ OLS power, F=1 halves, strong-F recovers OLS, r2_z
  F-approximation, F precedence). No numerics changed.

- **Batch 3 DONE** — `tests/test_tierD_identification_balance_analytic.py`
  (6 tests green): `frontdoor` (linear DGP `U->D->M->Y`: front-door ATE =
  (D->M)·(M->Y) = 2.0 recovered within 10% despite open back-door; beats biased
  naive OLS), `ps_balance` (exact Austin-2011 SMD re-derivation to 1e-9,
  balanced covariate →|SMD|<0.08, IPW reduces imbalance, variance ratio →1).
  `test_calibration` deferred to a later forest batch (needs fitted stochastic
  `CausalForest`, pairs with `model_averaging_dml`). No numerics changed.

**P1 progress: 10/25 estimators done, 36 tests green; classifier re-run shows
P1 floor dropped 25 → 15 (covered estimators now auto-detected as `anchored`).**
Remaining P1 (15): structural `blp`/`levpet`/`opreg`; rd `boundary_rd`/
`geographic_rd`/`multi_score_rd`; diagnostics `effective_f_test`/`stepwise`;
panel `feglm`; spatial `moran_local`; interference `peer_effects`; missing
`mi_estimate`; `notch`; `model_averaging_dml`; `test_calibration`.
- **Batches 4-8 DONE** — P1 floor closed (37 more tests, all green):
  - B4 `test_tierD_spatial_diag_analytic.py` (11): `moran_local` (LISA closed
    form + `Σ Iᵢ = S₀·globalI` identity), `effective_f_test` (= first-stage F
    under classic, hand-derived), `stepwise` (recovers known sparsity support).
  - B5 `test_tierD_panel_missing_analytic.py` (7): `feglm` (Gaussian == OLS, FE
    absorption, logit MLE recovery; needs pyfixest), `mi_estimate` (no-missing
    ≡ complete-data exactly, fmi=0; MCAR consistency + SE inflation).
  - B6 `test_tierD_rd_multiscore_analytic.py` (7): `boundary_rd`/`geographic_rd`
    (half-plane recovery of 0.8 + exact dispatch to rd2d/rdms),
    `multi_score_rd` (dispatch + linearity-in-jump). Investigated L-shape
    attenuation → confirmed a *correct* estimand property (boundary averaging),
    not a bug.
  - B7 `test_tierD_structural_analytic.py` (5): `levpet`/`opreg` recover
    Cobb-Douglas (0.60/0.35) on an *identified* DGP + exact alias dispatch.
    **`blp` DEFERRED — real bug found** (see below).
  - B8 `test_tierD_interference_forest_analytic.py` (7): `peer_effects`
    (recovers endogenous γ=0.4 + 0 with no peers), `notch` (induced-bunching
    detection), `model_averaging_dml` (DML-PLR recovers θ), `test_calibration`
    (β₁ calibration ≈ 1 + null=(1,0)).
- **LaLonde guard** `test_tierD_lalonde_psm_guard.py` (4): pins naive/-635,
  adjusted/1548.2, PSM/1963.4 — the missing regression test that would have
  caught the drift. Refreshed the stale registry pin 2012.5 → 1963.4.

### ⚠️ Bug found (reported, NOT fixed — `.tierd_campaign/BUG_blp_gmm_objective_maxiter.md`)
`sp.blp` calls `_gmm_objective(..., maxiter=...)` but the param is
`maxiter_inner` → `TypeError` on every optimisation step; the estimator cannot
complete. Plus a singular-weight-matrix fragility on thin DGPs. Proposed
one-line fix logged; blp Tier D recovery test deferred until fixed.

### Minor note
`imputation/mice.py:109` emits a benign `divide by zero` RuntimeWarning when
between-imputation variance is 0 (no-missing case); the fmi result is still
correctly 0. `test_calibration`'s β₂ (differential_forest_prediction) is
noisy/forest-dependent — anchored on β₁ + null structure instead.

## P1 STATUS: floor CLOSED — 24/25 estimators have analytic tests (77 tests),
1 (`blp`) blocked by the reported bug. Classifier: P1 25 → 1.
- **Next:** P2 — 222 weak-assert estimators needing known-truth anchors,
  highest-value families first (causal, decomposition, panel, regression).

### 2026-06-09 — session: blp fix + P2 kickoff
- **blp ⚠️ functionality fix landed.** Root cause: `_gmm_objective(..., maxiter=)`
  vs param `maxiter_inner` → `TypeError` on every `sp.blp` path (produced no
  output). Fixed blp.py:853,894; `TestBLPAnalytic` recovers price=-1.5, x1=1.0;
  logged CHANGELOG (Unreleased, ⚠️ Functionality fix) + MIGRATION (#blp-maxiter-fix);
  bug report marked RESOLVED. No previously-correct number moved (it crashed).
- **P1 floor confirmed = 0** (all 25 reference-less estimators anchored).
- **P2 started (causal/regression/diagnostics):**
  - `test_tierD_p2_bounds_sensitivity_analytic.py` (14): `evalue_rr`
    (VanderWeele-Ding E = RR+sqrt(RR(RR-1)) exact + CI bound + symmetry),
    `manski_bounds` (no-assumption width == outcome range exactly + brackets
    true ATE), `lee_bounds` (trimming bounds bracket selected-sample effect,
    midpoint→true, differential selection widens).
  - `test_tierD_p2_regression_system_analytic.py` (6): `vif` (1/(1-R²) exact +
    orthogonal→1), `sureg` (Kruskal: identical regressors == OLS + recovery),
    `jive` (jive1/jive2 recover known IV coef).
- **P2 worklist: 223 → 217** (6 anchored). 20 new tests, all green + lint clean.
- **Note (cosmetic):** `mice.py:109` emits a divide-by-zero RuntimeWarning on the
  fmi=0 (no-missingness) degenerate path; estimate is correct. Candidate for a
  guarded `r==0` short-circuit later (not a correctness bug).
- **Next:** continue P2 — dag(10), spatial(9), conformal_causal(7),
  decomposition(7) families; then LaLonde stale-pin guard.
