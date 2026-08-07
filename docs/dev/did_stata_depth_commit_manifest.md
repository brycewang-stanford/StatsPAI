# Commit manifest — DiD ↔ Stata option-depth campaign

Branch `wt/did-stata-depth`. **Nothing is committed yet** — this is the
proposed split for review.

Seven themed commits, ordered so each is independently green: the three
correctness fixes land first (they stand alone and are what a reader most
needs to see), then the feature work, then the agent-surface fix, then
docs. Schema regeneration rides with the commit whose parameters it
carries, so no commit leaves `schemas/*.json` inconsistent with the code.

Verification at HEAD of this branch: **3634 passed / 0 failed** across the
DiD, registry, schema and parity-harness suites (post-`black`); every
parity number reproduced by re-running its committed `.do` file.

---

## 1 — `fix(did): ⚠️ sun_abraham SEs dropped the cohort-share variance term`

The IW estimator multiplies two estimated objects, so SA (2021) Prop. 3
gives it a two-part variance. StatsPAI reported only `w'Var(β̂)w`,
**understating** SEs wherever more than one cohort contributes at an
event time. Invisible at single-cohort event times, where `Var(ŵ)` is
degenerate — 0.02% agreement there vs up to 2.0% drift where cohorts
pool.

Deliberate divergence recorded: `fixest::sunab` treats the shares as
fixed; `eventstudyinteract` (the author's own package) does not. StatsPAI
follows the latter.

```
src/statspai/did/sun_abraham.py          (_cohort_share_vcov, variance assembly)
tests/reference_parity/test_sunab_control_cohort_parity.py
tests/reference_parity/test_sunab_did2s_mpdta_parity.py   (tolerance rationale)
tests/stata_parity/option_parity/83_sunab_control_cohort.do (+ results/)
CHANGELOG.md · MIGRATION.md
```

⚠️ **Moves published numbers.** Point estimates unchanged; per-event-time
SEs/CIs/p-values rise at multi-cohort event times (0.6–2.0% on `mpdta`).

---

## 2 — `fix(did): ⚠️ did_multiplegt_dyn bootstrap used the wrong switch date`

The point estimate finds each unit's first treatment *change* in either
direction; the bootstrap re-derived it as `min(time | d == 1)` — correct
only for switch-**on** units. Switch-off units got `_F` = their own first
period, which has no base period, so they dropped out of every replicate.

Effect on non-absorbing panels: NaN SEs for switch-out samples, and a
pooled SE that silently collapsed onto the switch-in-only one
(0.182973 vs the correct 0.119610). Absorbing panels bit-identical —
which is why the existing `DIDmultiplegtDYN` parity suite never caught it.

```
src/statspai/did/did_multiplegt_dyn.py   (bootstrap _first_switch recompute)
tests/reference_parity/test_multiplegt_dyn_options_parity.py::TestBootstrapSwitchDate
CHANGELOG.md
```

⚠️ Moves bootstrap SEs on non-absorbing panels only. Point estimates
unchanged.

---

## 3 — `fix(did): BJS returned a confident number from an unidentified fit`

`unit_covariates` gives each unit its own slope, which needs ≥ k+1
untreated periods per unit. Stata refuses outright (rc 481); StatsPAI
returned `lsqr`'s minimum-norm answer with no warning. Also adds column
equilibration to the sparse Y(0) solve — interacting with raw calendar
years left the fit 1.6e-4 from Stata, and rescaling (exact, span-
preserving) brings it to 5.9e-8.

```
src/statspai/did/did_imputation.py       (identification guard, equilibration)
tests/reference_parity/test_bjs_fe_covariates_parity.py::TestIdentificationGuard
```

Raises where it previously answered. No previously-valid result changes.

---

## 4 — `feat(did): csdid convention and estimator depth for callaway_santanna`

`notyet_cutoff=`, `estimator='stdipw'` (alias) and `'ipw_abadie'` (new),
`pscore_trim=0.995` with a binding-trim tally and warning.

Records that StatsPAI's `'ipw'` is Stata's `method(stdipw)`, verified
against R `did` 2.3.0 dispatching `est_method="ipw"` →
`DRDID::std_ipw_did_panel`. **No existing numbers move.**

```
src/statspai/did/callaway_santanna.py
tests/reference_parity/test_csdid_conventions_stata_parity.py
tests/stata_parity/option_parity/82_csdid_conventions.do  (+ results/)
```

---

## 5 — `feat(did): Y(0)-model depth for did_imputation, control_cohort for sun_abraham`

`unit_covariates` / `time_covariates` / `fe=` / `project=`, and
`sun_abraham(control_cohort=)`.

```
src/statspai/did/did_imputation.py · src/statspai/did/sun_abraham.py
tests/reference_parity/test_bjs_fe_covariates_parity.py
tests/reference_parity/test_bjs_project_parity.py
tests/stata_parity/option_parity/84_bjs_fe_covariates.do  (+ results/)
```

---

## 6 — `feat(did): switcher restrictions, pretests, and one se_method vocabulary`

`did_multiplegt_dyn`: `switchers=` / `same_switchers=` / `effects_equal=`.
`sun_abraham` gains a joint pre-trend test (it had none); both it and
`callaway_santanna` gain `pretest=` / `pretest_periods=`. Shared
`se_method=` normalizer in `did/_core.py`, additive over every native
spelling, with a documented `'auto'` rule.

```
src/statspai/did/_core.py                (normalize_se_method, FEW_CLUSTERS)
src/statspai/did/did_multiplegt_dyn.py · sun_abraham.py · callaway_santanna.py
src/statspai/did/did_imputation.py       (se_method synonym)
tests/reference_parity/test_multiplegt_dyn_options_parity.py
tests/test_did_pretest_integration.py · tests/test_did_se_method_vocabulary.py
tests/stata_parity/option_parity/85_multiplegt_dyn_options.do  (+ data + results/)
```

---

## 7 — `fix(registry): 128 entries hid parameters from the agent schema`

Hand-written `params=` lists never pick up new signature arguments, so
capabilities become invisible to agents silently. Completes the four DiD
entries, freezes the other 124 (501 hidden parameters) as a ratchet.

```
src/statspai/registry.py
scripts/registry_param_drift_baseline.json
tests/test_registry_param_drift.py
schemas/*.json · src/statspai/schemas/*.json   (regenerated)
docs/dev/did_stata_depth_plan.md
```

---

## Not included

`sp.did_had` is **not** built. The plan doc carries the transcribed
algorithm, the corrected architecture (it needs `nprobust`'s `lprobust`,
**not** `rdrobust`), and four verified citations — including the
correction that the ado's own Yatchew reference is wrong (it cites 1997
against the 1999 paper's volume and pages).

## ⚠️ Rebase required — `main` moved during this work

`main` advanced two commits after this branch was cut from `594ab995`:

- `30cce15f fix(inference): ⚠️ vcov= was silently ignored on sp.regress / sp.ivreg`
- `b6f636e6 feat(parity): Stata coverage 61 -> 75 of 81, and make the Stata leg a contract`

Neither conflicts *semantically* with this work — `30cce15f` normalizes
which **sandwich** a regression uses (`core/_vcov_spec.py`), while this
branch's `se_method=` normalizes which **procedure** a DiD estimator uses
(`did/_core.py`). They were deliberately kept in separate modules.

Eight files are touched by both sides and need care on rebase:

| file | resolution |
| --- | --- |
| `schemas/*.json` (3) | **regenerate, do not merge** — `python scripts/dump_schemas.py` after taking `main`'s version |
| `src/statspai/schemas/*.json` (3) | same; `dump_schemas.py` mirrors them |
| `CHANGELOG.md` | manual append — keep both sides' entries |
| `MIGRATION.md` | manual append — keep both sides' entries |

`b6f636e6` also turned the Stata leg into a contract, which is what
forced the layout below.

### Why the fixtures live in `tests/stata_parity/option_parity/`

The new contract globs `tests/stata_parity/results/*_Stata.json` and
`tests/stata_parity/[0-9][0-9]_*.do` to enumerate Track A, then holds
every match to the Track A contract: registered module inventory,
joinable headline row, registered tolerance budget, strictness tier.

These four fixtures are **option-level** — one file carries several fits
of the same command under different switches, with no single headline
row — so they are not Track A modules and were moved into a
subdirectory, which the non-recursive globs do not reach. Putting them
one level up fails `test_parity_json_rows_keep_the_joinable_schema` and
`test_tier_a_parity_fixture_lock_is_current`; both were observed and are
the reason for the layout. See that directory's `README.md`.

## ⚠️ Never push without checking fast-forward first

This nearly went wrong. `main` in this repo is worked by more than one
line at a time, and the other line **rewrote history** mid-campaign: the
two commits this branch was rebased onto (`b6f636e6`, `30cce15f`) ceased
to exist, replaced by same-message commits at different SHAs
(`2157e96e`, `42ba3438`) with nine further commits stacked on top.

Pushing `HEAD:main` at that moment would have **deleted eleven commits
from `origin/main`**, including two ⚠️ correctness fixes:

- `42ba3438 fix(inference): ⚠️ vcov= was silently ignored on sp.regress / sp.ivreg`
- `6bcefece fix(did): ⚠️ aggte SEs dropped the weight-estimation influence term`

Nothing would have stopped it automatically. The check that caught it was
explicit.

**Before every push, in this order:**

```sh
git fetch origin
git merge-base --is-ancestor origin/main HEAD && echo SAFE || echo REBASE-FIRST
git rev-list --count HEAD..origin/main    # must be 0
```

If the second line says REBASE-FIRST, rebase onto `origin/main` — never
onto `main`, which may itself be stale — and never reach for `--force`.

Same-message commits at different SHAs are the signature of a rewritten
history. `git log --oneline HEAD..origin/main` is what makes the damage
visible before it happens.

### What the two lines collided on, and how it resolved

| file | resolution |
| --- | --- |
| `src/statspai/did/_core.py` | pure append on both sides — theirs added `weight_influence` (aggte weight-estimation influence), this branch added `normalize_se_method`; both kept |
| `CHANGELOG.md` | both added `### ⚠️ Correctness` under Unreleased; merged into ONE header, and this branch's `Added`/`Fixed` items folded into the existing sections rather than duplicating them |
| `MIGRATION.md` | append-only, both entries kept |
| `schemas/*.json` | regenerated, never merged |

Worth noting the two lines converged on the same class of bug from
different directions: `6bcefece` fixes aggte treating **estimated
aggregation weights** as fixed; this branch fixes `sun_abraham` treating
**estimated cohort shares** as fixed. Complementary, both retained.

## Pre-push checks

- [ ] Full `pytest -q` (not just the DiD subset)
- [ ] `pytest tests/reference_parity/ -q`
- [ ] `black src tests && flake8 src tests`
- [ ] Confirm no file overlaps the concurrent Paper-JSS line
      (`.do` files renumbered 75/76/77 → 82/83/84 to avoid exactly that)
- [ ] Decide whether the two ⚠️ SE changes need a note to JOSS review
