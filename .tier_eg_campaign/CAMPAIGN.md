# Tier E (不变性 / invariance) + Tier G (鲁棒性 / robustness) campaign

**Goal (set by maintainer, 2026-06-08):** for the six **core estimator
modules** — `did iv rd synth dml panel` — build a *deep, solid* suite of

- **Tier E — invariance / metamorphic tests**: assert each estimator's
  *defining* algebraic & statistical invariances directly from a fitted result,
  with **no external reference** required. These are property-based
  (`hypothesis`) where the property is generic (row permutation, row scaling,
  duplication, column reordering), and hand-rolled seeded-randomized where the
  metamorphic relation is estimator-specific (treatment sign-flip, weight
  identity, FWL residual identity, moment/orthogonality identities, cross-fit
  seed bounds).
- **Tier G — robustness tests**: assert estimators **fail loudly**
  (CLAUDE.md §7 — 失败要响亮) on degenerate / adversarial input — `NaN`/`inf`,
  tiny `n`, single cluster, perfect separation, constant / perfectly-collinear
  columns, extreme magnitudes, wrong dtypes, missing columns, mismatched
  lengths — rather than returning a silent `None` / `NaN` or a wrong number.

Budget: ~1 month. Quality bar is non-negotiable (CLAUDE.md §5): real numerical
assertions, no mocking of numerical paths.

This file is the durable, cross-session tracker.

---

## Decisions locked (2026-06-08, maintainer)

1. **Scope = the six core estimators**, deep (not the 950-fn breadth).
2. **Tier E engine = `hypothesis` + hand-rolled hybrid.** `hypothesis>=6.100`
   added to `[project.optional-dependencies].dev`. Generic invariances use
   `@given` with a fixed `derandomize`/seed and `deadline=None`; estimator-
   specific metamorphic relations use `np.random.default_rng(seed)` parametrized
   tests, matching the existing `tests/test_dml_orthogonality_invariants.py`
   house style.

---

## Hard rules (inherited from the coverage campaign)

1. **Test-only.** Add tests under `tests/tier_eg/`. Do **not** change estimator
   numerics — that could move JOSS #10604 / JSS dossier numbers.
2. **If a test reveals a real bug:** stop, report to maintainer, then follow
   CLAUDE.md §12 `⚠️ correctness fix` (CHANGELOG + MIGRATION). Never silently
   change numerical output during JOSS review.
3. **Do not touch** the JOSS/JSS files in flight (`docs/joss_validation_dossier.md`,
   `tests/test_jss_validation_api.py`, `tests/external_parity/*`,
   `tests/reference_parity/_fixtures/dml_iv_*`).
4. **No tolerance fudging.** When a metamorphic relation is exact (permutation,
   FWL, weight identity) assert it to `rtol≈1e-9`. When it is only
   distributional (seed stability), assert a *documented, justified* band and
   say why in a comment.

---

## The Tier E invariance catalogue (what "做扎实" means per estimator)

A core estimator's Tier E suite should cover, where the relation is *defined*:

| # | Invariance | Exact? | Applies to |
|---|---|---|---|
| E1 | **Row-permutation** — shuffling observation order leaves point estimate, SE, CI bit-identical (up to FP). | exact | all |
| E2 | **Outcome location shift** — `y → y + c` shifts only the intercept; slope/ATT/τ unchanged. | exact | iv panel did(level) |
| E3 | **Outcome scale** — `y → a·y` scales coefficients & SEs by `a`; t-stats / p-values invariant. | exact | iv panel did rd synth |
| E4 | **Regressor scale** — `x → x/s` scales that coefficient by `s`; fitted values invariant. | exact | iv panel rd |
| E5 | **Row duplication** — duplicating every row leaves the point estimate unchanged (SE shrinks predictably under iid weights). | exact (point) | iv panel did |
| E6 | **Weight identity** — uniform `weights` ≡ unweighted run. | exact | did rd panel synth |
| E7 | **Treatment sign-flip / relabel** — swapping treated⇄control flips the ATT sign, |ATT| & SE unchanged. | exact | did |
| E8 | **Column reordering** — permuting covariate columns leaves all estimates unchanged. | exact | all w/ covars |
| E9 | **FWL / partialling-out identity** — residual-regression recovers the same slope. | exact | iv panel |
| E10 | **Moment / orthogonality identity** — empirical score ≈ 0 at the solution; sandwich SE identity. | exact | dml iv-gmm |
| E11 | **Cross-fit / bootstrap seed determinism** — same `random_state` ⇒ identical output; different seeds within a documented band. | mixed | dml did(boot) synth(placebo) |
| E12 | **Cutoff/translation invariance** — shifting running var & cutoff together leaves the RD estimate unchanged. | exact | rd |
| E13 | **Donor/unit-permutation** — relabelling control units leaves the synthetic weights & gap unchanged. | exact | synth |

## The Tier G robustness catalogue

| # | Adversarial input | Required behaviour (§7) |
|---|---|---|
| G1 | `NaN` / `inf` in y / x / running var | explicit error **or** documented drop + recorded diagnostic — never silent `NaN` estimate |
| G2 | tiny `n` (below identification) | clear error, not a crash deep in linalg |
| G3 | constant outcome / constant regressor | loud error or rank-deficiency warning |
| G4 | perfect collinearity / duplicate column | drop-with-warning or explicit error |
| G5 | single cluster / single unit / one period | loud error on the inference path |
| G6 | perfect separation (binary first stage) | warn / error, no silent garbage |
| G7 | extreme magnitudes (1e12) / near-singular design | stable answer or rank warning, no `inf`/overflow silently |
| G8 | wrong dtype / missing column / mismatched length | `KeyError`/`ValueError`/`TypeError` with a usable message |
| G9 | empty treated or empty control group | loud error |
| G10 | all-treated / no-variation-in-treatment | loud error |

> Not every cell applies to every estimator; the per-module file documents which
> invariances/robustness cases are *defined* for that estimator and why any are
> N/A.

---

## Per-module progress

| module | Tier E file | Tier G file | E cases | G cases | status |
|---|---|---|---|---|---|
| iv     | `tests/tier_eg/test_iv_invariance.py`    | `tests/tier_eg/test_iv_robustness.py`    | 11 | 9 | ✅ E1,E2,E3,E4,E5,E8,E9,instr-scale; G1,G2,G3,G4,G6,G8 |
| dml    | `tests/tier_eg/test_dml_invariance.py`   | `tests/tier_eg/test_dml_robustness.py`   | 9 | 7 | ✅ seed-det,E2,E3,E8,treat-scale,cross-fit-band; G missing/folds/irm/constant-d/NaN |
| panel  | `tests/tier_eg/test_panel_invariance.py` | `tests/tier_eg/test_panel_robustness.py` | 11 | 7 | ✅ E1-E6,E8,E9 (fe+twoway); G1,G4,G5,G7,G8 + OBS-1 |
| did    | `tests/tier_eg/test_did_invariance.py`   | `tests/tier_eg/test_did_robustness.py`   | 14 | 9 | ✅ 2x2 E1,E2,E3,E5,E6,E7,E8; CS E1,E2,E3,E5,E8,ctrl-group-agree; G9,G10,G5,G8,G1(NaN/inf); CS missing-id,NaN,no-never-ctrl (FINDING→fixed) |
| rd     | `tests/tier_eg/test_rd_invariance.py`    | `tests/tier_eg/test_rd_robustness.py`    | 11 | 7 | ✅ E1,E3,E12,run-scale,reflection; G1,G2,G3,G8,one-sided,weights-NotImpl |
| synth  | `tests/tier_eg/test_synth_invariance.py` | `tests/tier_eg/test_synth_robustness.py` | 8 | 6 | ✅ E1,E2,E3,E13(donor-relabel); G1,treated-absent,pre/post-period,missing,single-donor |

Shared harness: `tests/tier_eg/_helpers.py` (result accessors, seeded DGPs,
invariance assert utilities, hypothesis strategies + profile).

---

## How to run

```bash
.venv/bin/python -m pytest tests/tier_eg/ -q            # the whole campaign
.venv/bin/python -m pytest tests/tier_eg/test_iv_*.py -q
```

## Bugs surfaced (report to maintainer before any fix — CLAUDE.md §12)

### OBS-1 [robustness/UX gap, NOT a numerical bug] — `sp.panel` silently
reinterprets a globally-constant regressor as the intercept

*Surfaced by:* `tests/tier_eg/test_panel_robustness.py::test_panel_constant_regressor_documented`.

A regressor column that is *globally constant* (e.g. `x ≡ 1.0`, or any `x ≡ c`)
is passed straight to `linearmodels.PanelOLS`, which treats an all-`c` column as
the model **intercept** and returns `coef_x = mean(y)/c` with a large, misleading
t-stat (t≈10.6, p≈0 in the repro). Verified this is **upstream linearmodels
semantics**, not StatsPAI arithmetic — a direct `PanelOLS(y, x_ones,
entity_effects=True).fit()` returns the identical number.

Contrast: a *time-invariant* (between-only) regressor — the more common
"collinear with the FE" case — correctly raises `AbsorbingEffectError`. Only the
zero-variance-everywhere column slips through.

*Severity:* MEDIUM-LOW. A user only hits it by feeding a column that is constant
in the estimation sample (e.g. a dummy that collapsed after filtering); the risk
is a spuriously "significant" coefficient. **Not** a §12 correctness regression
(no published number changes). *Suggested fix (maintainer's call, deferred —
JOSS review in flight):* validate regressor variance in `panel_reg.panel` before
dispatch and `raise`/`warn` on a zero-variance column, mirroring the existing
`AbsorbingEffectError` path. The Tier G test pins both acceptable outcomes so a
future guard is test-visible.

## Full-suite result (2026-06-08)

`pytest tests/ -n 8 --no-cov` (default `-m 'not slow'`): **8644 passed, 2 failed,
89 skipped, 1 xfailed** in 10m40s.

The **2 failures are pre-existing count-drift guards, NOT caused by this work** —
proven by re-running them with the entire `tests/tier_eg/` package removed + the
`pyproject` dev-extra reverted: they fail identically.

1. `test_jss_validation_api.py::test_validation_report_collected_counts_match_jss_headline`
   — `reference_parity: collected 149 but headline 124`. The 25-test drift comes
   from earlier parity commits (HC2/HC3, two-way cluster, CR2/CR3 modules) added
   after the JSS headline was last synced. `tests/tier_eg/` adds **0**
   reference_parity tests.
2. `test_jss_release_manifest.py::test_coverage_findings_track_b1000_artifacts`
   — `len(canonical) == 7` but the B=1000 coverage-MC artifact now has 9
   scenarios.

Both require updating `JSS_HEADLINE_TEST_COUNTS` / `len(canonical)` **and** the
manuscript's `tab:internal-parity` in lockstep — which touches JOSS/JSS files in
flight. Per Hard Rule #3 this is **out of scope** for the test-only campaign and
left for the maintainer to sync (deliberately NOT fixed here).

> ⚠️ Concurrency note: a second window/agent was active in this repo during the
> session (HEAD advanced 7ffd71e→91af3e9; a `other-window-P1-WIP-do-not-touch`
> stash is on the stack). That window committed this campaign as
> `0f7ef2f test(validation): add Tier E/G core estimator suite`. Both stashes are
> preserved untouched.

## Acceptance checklist (maintainer ticks)

- [x] All six modules have a Tier E + a Tier G file, each with the *defined*
      invariances/robustness cases (N/A cases documented).
- [x] `pytest tests/tier_eg/ -q` green — **105 passed** (2026-06-08).
- [x] No regression to wider suite from this work: `pytest tests/ --collect-only`
      clean (**8757 collected, 0 errors**) with the new package present, and the
      adjacent pre-existing `tests/test_dml_orthogonality_invariants.py` still
      green. Changes are purely additive (new `tests/tier_eg/*` + a `[dev]`
      extra); no estimator source touched, so JOSS/JSS dossier numbers are
      unaffected. _Full end-to-end `pytest -q` (~8.7k tests) not run to
      completion here — left for CI._
- [x] One observation surfaced (**OBS-1**, panel constant-regressor quirk),
      logged above; confirmed **upstream linearmodels** semantics, NOT a §12
      numerical regression → no silent fix made.

## Session log

### 2026-07-06 — did module completed + CS gap surfaced
- The `did` Tier E/G files already existed and were green (20 tests) but the
  per-module table still read `☐` — a stale tracker, now corrected to ✅.
- **Deepened the thinner staggered (Callaway–Sant'Anna) half** from 3 E / 2 G
  to 6 E / 3 G: added CS E5 (unit duplication point-invariance), CS E8
  (covariate column reorder), a methodological never-vs-not-yet control-group
  agreement (banded), and a documented no-never-treated-control robustness
  case. Module now 14 E / 9 G; `pytest tests/tier_eg/ -q` → **111 passed**.
- **⚠️ FINDING (reported, NOT fixed — §12):**
  `.tier_eg_campaign/FINDING_cs_nevertreated_no_control.md`.
  `sp.callaway_santanna(control_group="nevertreated")` with zero never-treated
  units returned a silent `ATT = 0.0` (undefined cells aggregate to 0) instead
  of failing loudly. Reproduced on seeds 3/7/11, zero warnings. Guard pins
  both acceptable outcomes so the fix is a deliberate, test-visible change.
  **RESOLVED same session** (maintainer asked to fix): up-front guard raises
  `MethodIncompatibility`; §12 in CHANGELOG (⚠️ Correctness) + MIGRATION.
  No previously-valid estimate moves; 247 callaway/cs tests green.
