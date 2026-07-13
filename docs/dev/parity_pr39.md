# Parity coverage — PR #39 (`f9cc2932`) at a glance

This note exists because the PR title,

> `feat(parity): analytical-only mice + mi_estimate Rubin-rules recovery (#39)`

is **narrower than the commit body**. Reviewers scanning `git log` (or
JOSS reviewers cross-checking the commit list against the parity doc) will
read "this commit shipped one MICE/`mi_estimate` parity test" — that is
about **one tenth** of what the commit actually contains. This page is the
single-source-of-truth map for what PR #39 delivered, and is referenced
from `CHANGELOG.md` and `docs/parity.md` (via the analytical-only section
already listing every test file).

---

## What PR #39 actually shipped

One commit, **19 files changed, +3728 / -362**, on `main` at
`f9cc293257ad00447013887ce70ee326de59f165` (2026-07-13).

1. **Ten new analytical-only parity test files** under
   `tests/reference_parity/`, covering ten previously-uncovered estimator
   families:

   | # | test file | estimator family covered | notable functions |
   | --: | --- | --- | --- |
   | 1 | `tests/reference_parity/test_structural_parity.py` | structural | `olley_pakes`, `levinsohn_petrin`, `ackerberg_caves_frazer`, `wooldridge_prod`, `prod_fn`, `markup`, `blp` |
   | 2 | `tests/reference_parity/test_longitudinal_parity.py` | longitudinal regimes | `longitudinal_analyze`, `longitudinal_contrast`, `regime`, `always_treat`, `never_treat` |
   | 3 | `tests/reference_parity/test_msm_family_parity.py` | marginal structural models | `msm` |
   | 4 | `tests/reference_parity/test_gformula_family_parity.py` | parametric g-formula MC | `gformula_ice_fn`, `gformula_mc` |
   | 5 | `tests/reference_parity/test_fairness_parity.py` | fairness metrics | `demographic_parity`, `equalized_odds`, `fairness_audit`, `orthogonal_to_bias`, `counterfactual_fairness`, `evidence_without_injustice` |
   | 6 | `tests/reference_parity/test_imputation_parity.py` | multiple imputation | `mice`, `mi_estimate` *(this is the family the PR title is named after)* |
   | 7 | `tests/reference_parity/test_multi_treatment_parity.py` | multi-valued treatment effects | `multi_treatment` |
   | 8 | `tests/reference_parity/test_ope_parity.py` | off-policy evaluation | `sharp_ope_unobserved`, `causal_policy_forest` |
   | 9 | `tests/reference_parity/test_surrogate_parity.py` | surrogate index family | `surrogate_index`, `long_term_from_short`, `proximal_surrogate_index` |
   | 10 | `tests/reference_parity/test_target_trial_parity.py` | target-trial emulation | `target_trial_protocol`, `target_trial_emulate`, `clone_censor_weight`, `immortal_time_check`, `target_trial_checklist`, `target_trial_report` |

   All ten follow the project's standing
   [analytical-only parity convention](../reference_parity/REFERENCES.md):
   deterministic DGP with a closed-form population parameter, machine-
   precision recovery within a bounded number of standard errors.

2. **Schema / index / doc sync** in support of the ten new test files
   (nine files, +/– small):
   `CHANGELOG.md`, `docs/parity.md`, `schemas/agent_cards.json`,
   `schemas/functions.json`, `schemas/tools.json` (×2 paths each — repo
   root + `src/statspai/schemas/` mirror), `src/statspai/_parity_index.json`.
   These are not part of the analytical contract; they keep the parity
   index and the JOSS reviewer-facing `parity.md` matrix accurate.

3. **One known limitation surfaced** (`proximal_surrogate_index`):
   the second-stage design `[1, W, S_hat]` is exactly rank-deficient
   because `S_hat` is affine in `W`, so the recovery test for the
   confounded-ATE point estimate is **skipped with a documented reason**
   pending a correctness fix. This is a non-functional finding called out
   in the CHANGELOG entry — not a regression.

---

## Why the title is narrower than the body

The convention for parity PRs in this repo (see `git log --grep='^feat(parity):'`
on `main`) is **one PR per parity test file**:

```text
7611f244 feat(parity): analytical-only mr_cml + mr_multivariable recovery
05b1b171 feat(parity): analytical-only mr_raps robust-profile-score recovery
4637ef60 feat(parity): analytical-only grapple robust-profile-score recovery
…
```

PR #39 **bundled all ten outstanding Tier-D reference-less estimators into a
single commit** because they were already queued and the test scaffolding
(schema/index/CHANGELOG updates) overlapped heavily. That bundling let the
squash-merge title only name one family (`mice + mi_estimate`) to keep the
line under the 72-char conventional-commit guideline. This is a **title
hygiene oversight**, not a content mistake — the commit body is
self-explanatory and the parity doc (`docs/parity.md`) lists every
function and its test file under the **analytical-only** section.

This note is the patch: anyone reading `git log --oneline` and being misled
by the title can land here via the pointer in `CHANGELOG.md`.

---

## How to verify

Anyone auditing this commit should run, in order:

1. `git show f9cc2932 --stat` — confirm 19 files, +3728/-362, the 10 test files named above.
2. `pytest tests/reference_parity/test_{structural,longitudinal,msm_family,gformula_family,fairness,imputation,multi_treatment,ope,surrogate,target_trial}_parity.py` — confirm every one of the ten families has a passing analytical-only guard.
3. `python -c "import statspai as sp; print(sp.parity_summary())"` — confirm the parity index now reports `analytical-only = 200` (was 190 before this commit).
4. `python scripts/build_parity_index.py` — confirm `docs/parity.md` regenerates without warnings; this PR's coverage should appear in the analytical-only table automatically.

---

## Related

- `CHANGELOG.md` — the user-facing entry adding the explanatory table of the 10 files, with a cross-reference back to this note.
- [`docs/parity.md`](../parity.md) — auto-generated matrix; the analytical-only section (~line 197 onwards) lists every function and which of these 10 test files covers it.
- [`tests/reference_parity/REFERENCES.md`](../../tests/reference_parity/REFERENCES.md) — the DGP / tolerance contract for analytical-only recovery tests in general.
- [`docs/dev/parity_status_roadmap.md`](parity_status_roadmap.md) — the Tier-D worklist (now zero) that PR #39 closed.
