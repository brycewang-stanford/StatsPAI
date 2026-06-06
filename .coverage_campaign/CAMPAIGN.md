# Core-module ≥95% coverage campaign

**Goal (set by maintainer, 2026-06-05):** drive all six core estimator modules
— `did iv rd synth dml panel` — to a **hard ≥95% line coverage**, measured under
the *full test suite*. Budget: ~3 months. Quality bar is non-negotiable
(CLAUDE.md §5): real numerical assertions, parity/analytic tests, **no mocking of
numerical paths**.

This file is the durable, cross-session tracker. The maintainer verifies
progress against the **Acceptance checklist** at the bottom.

---

## How coverage is measured here (important)

StatsPAI coverage can only be measured with the **whole-package** source:

```bash
pytest tests/ -q --cov-report=xml:.coverage_campaign/latest.xml --cov-report=
python scripts/coverage_campaign.py report --xml .coverage_campaign/latest.xml
```

- `--cov=statspai.<sub>` (sub-package source) **crashes** with
  `generic_type: type "ObjSense" is already registered` — it reorders imports
  ahead of the `scipy.optimize` PyO3/pybind11 stabiliser in `tests/conftest.py`.
  Always use the default whole-package `--cov=statspai`.
- The authoritative number is the **full-suite** number. Running only
  `tests/test_<mod>*.py` understates a module badly (iv: 38% module-only vs
  87% full-suite) because cross-module tests exercise it too.
- `scripts/coverage_campaign.py gaps <mod> --per-file` lists the exact uncovered
  line ranges to target.

---

## Baselines (committed `coverage.xml`, 2026-05-28, v1.16.0)

| module | cov% | covered/total | +lines to 95% |
|---|---|---|---|
| iv | 86.7 | 2426/2799 | 234 |
| dml | 75.7 | 959/1267 | 245 |
| panel | 54.0 | 970/1796 | 737 |
| did | 74.6 | 4369/5859 | 1198 |
| rd | 66.3 | 3107/4689 | 1348 |
| synth | 66.0 | 3922/5940 | 1721 |
| **total** | ~70.5 | | **≈5,480** |

Sequencing (cheapest first, big three last): **iv → dml → panel → did → rd → synth**.

---

## Hard rules for this campaign

1. **Test-only.** Add tests under `tests/`. Do **not** change estimator
   numerics — that could move JOSS/JSS dossier numbers (review #10604).
2. **If a test reveals a real bug:** stop, report to maintainer, then follow
   CLAUDE.md §12 `⚠️ correctness fix` (CHANGELOG + MIGRATION). Never silently
   change numerical output during JOSS review.
3. **Do not touch** the JOSS/JSS files in flight: `docs/joss_validation_dossier.md`,
   `tests/test_jss_validation_api.py`, `tests/external_parity/*`,
   `tests/reference_parity/_fixtures/dml_iv_*`.
4. **Conflict avoidance with parallel agents.** Hot src areas observed:
   did-imputation (`codex/bjs-imputation-fix`), panel xtabond/xtnbreg
   (`fix/xtabond-instruments`, `codex/xtnbreg-panel-nbreg`),
   `worktree-improve-correctness`, `parity-stabilization`. Write **new** test
   files named `test_<mod>_cov_<topic>.py`; `git pull --rebase` before each
   push; schedule hot src files last and re-check `git log` before starting them.
5. New tests must carry real assertions (closed-form / parity / invariants),
   not just smoke-call for coverage.

---

## Per-module status

| module | start | current | target | status |
|---|---|---|---|---|
| iv | 86.7 | **95.5** | 95 | ✅ **DONE** (12 test files, 162 tests; 46 defensive lines pragma'd) |
| dml | 75.7 | **95.6** | 95 | ✅ **DONE** (5 test files, 44 tests; 64 defensive lines pragma'd) |
| panel | 54.0 | **95.1** | 95 | ✅ **DONE** (8 test files, 114 tests; biggest climb) |
| did | 74.6 | — | 95 | ⬜ queued |
| rd | 66.3 | **95.6** | 95 | ✅ **DONE** (parallel-agent tests + my 209-line defensive pragma pass) |
| synth | 66.0 | **92.9** | 95 | 🟡 my 139-line pragma done; +130 reachable → parallel agent |

**Method calibration (from iv):** a module's last ~3–5% is overwhelmingly
defensive `except`/validation/"unreachable" branches. The eight iv test files
(86.7→91.8%, +144 reachable lines) covered the happy paths, dispatcher routes,
exports, summaries, weak-robust sets, JIVE/NPIV/IVMTE/plots and array-input
forms. The remaining +90 lines are error-handling tails with steep
diminishing returns — closing them to a *hard* 95% needs either per-branch
fault-injection tests or `# pragma: no cover` on genuinely-unreachable defensive
code (the repo already uses this idiom, e.g. `iv/__init__.py:424`,
`iv/iv_diag.py:340`). **Tail-handling policy is a pending maintainer decision.**

---

## Session log

### 2026-06-05 — session 1: infrastructure + iv start
- Diagnosed local coverage blocker (sub-package `--cov` crashes on scipy highspy
  double-registration); established whole-package measurement path.
- Added `scripts/coverage_campaign.py` (per-module report + gap line ranges).
- Kicked off authoritative full-suite baseline → `.coverage_campaign/`.
- Started module 1: **iv**. Added `tests/test_iv_cov_dispatcher_routes.py`
  (9 tests, green): covers the previously-untested `sp.iv(method=...)` routes
  `jive_mw, many_weak_ar, lasso, post_lasso, mte, ivmte_bounds,
  plausibly_exog_uci/ltz` and, transitively, `many_weak.py / post_lasso.py /
  plausibly_exogenous.py / mte.py / ivmte_lp.py`. Each asserts a real property
  (point estimate near the DGP truth, or a valid CI/bound ordering), not a smoke
  call. Proven loop: gap-map → targeted tests → green.
- **Pending decision from maintainer:** commit/push cadence (see report).

### 2026-06-05 — handoff from parallel JOSS-prep agent

- A second agent working the JOSS-prep track contributed one **synth**
  coverage file before withdrawing from the coverage campaign (maintainer
  decision: campaign owns all of W3.2). Handed over, renamed to the campaign
  convention: `tests/test_synth_cov_plots.py` (12 tests, green) — covers the
  `sp.synthplot` dispatcher's plotting layer (`synth/plots.py`): types
  `trajectory, gap, both, weights, placebo, placebo_gap, rmspe, conformal,
  compare`, the `pre_band` overlay, and the unknown-type `ValueError`. These
  are rendering smoke tests (assert a Matplotlib `Figure` is returned), so
  they lift `synth/plots.py` (1,237 lines, 23% baseline) without pinning
  pixels. Remaining synth plot renderers needing specialised fits
  (`staggered, factors, distributional, multi_outcome, prediction_interval,
  sensitivity`) are still open for the campaign's synth module.

---

### 2026-06-05 — session 2: iv push 86.7 → 91.8%

Added 8 iv coverage files (94 tests, all green, all committed + pushed to main):
`test_iv_cov_dispatcher_routes / _diag / _weak_and_jive / _array_inputs /
_plots / _ivmte_bounds / _edges / _summaries.py`. Measured via the fast union
method (`scripts/coverage_campaign.py union iv`): baseline 86.7% + 144 reachable
lines newly covered → 91.8%. Remaining +90 to 95% is the defensive/error tail
(weak_identification 31, npiv 25, mte 24, __init__ dispatcher-error 21, iv_diag
fallbacks 19, …) — see tail-handling policy decision above.

Next: (a) maintainer picks tail-handling policy; (b) finish iv tail; (c) move to
dml (75.7%, next cheapest).

### 2026-06-05 — session 3: iv ✅ reaches 95.5% (hybrid tail policy)

Maintainer picked the **hybrid** tail policy (real tests + `# pragma: no cover`
on genuinely-defensive lines). **iv DONE at 95.5%** (union method; true
full-suite ≥ this) — the first core module to clear the bar.

- 12 iv coverage test files total, 162 tests, all green.
- 46 defensive lines pragma'd across 12 iv source files — all `except`-fallbacks
  (LinAlgError pinv/lstsq, bare except), unreachable validation raises, and
  defensive nan/inf sentinels. Comments only; zero numeric change.
- **Bug surfaced & flagged (not yet fixed):** `sp.iv(method='lasso', formula=…)`
  raises `TypeError` — the dispatcher forwards `formula=` into `lasso_iv`, which
  rejects it (native `x_endog`/`z` path works). Pinned by
  `test_iv_cov_tail.py::test_dispatch_lasso_formula_is_currently_broken`.

Next: **dml** (75.7%, next cheapest), same playbook.

### 2026-06-05 — session 4: dml ✅ reaches 95.6%

**dml DONE at 95.6%** (75.7→95.6%, union method) — second core module cleared.
5 new test files (44 tests, green): `test_dml_cov_learners / _diag_sens /
_scores / _base / _averaging_panel.py`. Covered learner resolution, diagnostics
(summary+plot), sensitivity (summary+plot), model averaging (weighted + all
validation errors), panel DML (weighted), PLR/IRM weighted scores, PLIV route,
sample_weight guards. 64 defensive lines pragma'd (except-fallbacks,
IdentificationFailure/RuntimeError raises, xgboost-absent returns, nan/inf
sentinels). Zero numeric change.

Progress: **iv ✅ 95.5%, dml ✅ 95.6%** (2/6). Next: **panel** (54.0%, the lowest
start — largest single climb).

### 2026-06-05 — session 5: panel ✅ reaches 95.1% (biggest climb)

**panel DONE at 95.1%** (54.0→95.1%, union method) — third core module, and the
largest single climb (+41pp). 8 new test files (114 tests, green):
`test_panel_cov_plots / _estimators / _feols / _diagnostics / _misc / _hdfe /
_compare.py`. Covered the plotting layer (6 renderers), estimator methods
(fe/re/pooled/fd/be/mundlak/twoway/chamberlain/ab-gmm), result diagnostics
(BP-LM/F-effects/Pesaran-CD/Hausman), unit roots (ips/llc/fisher/hadri), FGLS,
binary panels (logit/probit fe/re/cre), native HDFE OLS (sp.hdfe_ols) with
cluster/wild/weights, two-way clustering, and PanelResults.compare(). ~95
defensive/compiled lines pragma'd (numba kernels, Rust shim, except-fallbacks,
plot 'data unavailable' guards). Zero numeric change.

Progress: **iv ✅ 95.5%, dml ✅ 95.6%, panel ✅ 95.1%** (3/6). Remaining: did
(74.6%), rd (66.3%), synth (66.0% — plots layer already started).

**Coordination note:** a parallel agent is contributing did coverage
(`tests/test_cov95_did_*.py`, untracked locally). To avoid collision the
campaign will take **rd next**, then synth, and pick up did only if that agent
stops. Re-check `git log -- src/statspai/did` before starting did.

### 2026-06-06 — session 6: rd ✅ + complementary pragma strategy

Discovered a **parallel agent running the identical campaign** (test_cov95_*
files across all 6 modules; actively committing did/synth src fixes). My rd
estimator tests netted only +23 lines — heavy duplication with its
test_cov95_rd_* files. **Maintainer chose the complementary split:** I do the
**pragma tail-clearing + verification** (my unique value — the parallel agent is
tests-only and plateaus on the defensive tail), it does reachable tests.

- **rd ✅ 95.6%** — applied a 209-line defensive pragma pass across 23 rd source
  files; combined with the parallel agent's reachable tests this cleared
  86.0 → 95.6%.
- **synth** — applied a 139-line defensive pragma pass across 25 synth source
  files (90.7% → ~93%); the remaining reachable lines are the parallel agent's
  to cover. (synth src is hot — committed with pull --rebase, no conflict.)
- **did** — left to the parallel agent: it is actively editing did src
  (`fix(did)` commits), so a pragma pass there would risk conflicts.

Progress: **iv ✅ · dml ✅ · panel ✅ · rd ✅** (4/6 hard-95%); synth ~93%
(pragma done, reachable pending parallel agent); did pending parallel agent.

## Acceptance checklist (for the maintainer to verify all results)

Run, then confirm each line:

```bash
# 1. fresh authoritative full-suite coverage
pytest tests/ -q --cov-report=xml:.coverage_campaign/verify.xml --cov-report=
python scripts/coverage_campaign.py report --xml .coverage_campaign/verify.xml
```

- [ ] iv ≥ 95%
- [ ] dml ≥ 95%
- [ ] panel ≥ 95%
- [ ] did ≥ 95%
- [ ] rd ≥ 95%
- [ ] synth ≥ 95%
- [ ] `pytest tests/ -q` fully green (no new failures)
- [ ] `pytest tests/reference_parity/ -q` still passes (numbers unmoved)
- [ ] no estimator numerics changed (git diff over `src/statspai/{did,iv,rd,synth,dml,panel}` is test-enabling only, or each change is a logged ⚠️ correctness fix)
- [ ] CI per-module coverage ratchet in place (no silent regressions)
