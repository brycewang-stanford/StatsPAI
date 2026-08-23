# DiD 2025/2026 literature programme — three-month plan

Source corpus: `8.4-5-2days-camp/11.1-因果推断/DiD/DiD-相关论文pdf`, the
2025/2026 entries (#51 Baker et al. JEL practitioner's guide, #52
de Chaisemartin–D'Haultfœuille complex designs, #54 Feng et al. StatMed,
#55 Sutton et al. JECH, #56 Ulloa-Pérez et al. staggered comparison, #57
de Chaisemartin et al. `did_multiplegt_dyn` overview, #58 Roth on
interpreting event studies).

Targets: **StatsPAI**, **Paper-DiD-JAE**, **Paper-JSS**.

Every citation below was verified against two independent sources
(Crossref API, arXiv abstract pages, publisher PDFs) before entering
`paper.bib` / `references/anchors.bib` / `manuscript/jss-bib.bib`, per
CLAUDE.md §10.

---

## Selection rule

The corpus supports far more work than three months allows, so items are
ranked by one criterion: **does the paper identify something a package
can be wrong about, in a way a reference implementation can adjudicate?**
That rule promotes Roth (2026) and Baker et al. (2026) to the front and
demotes the field-specific reviews to citation-level work, because the
first two make falsifiable claims about what software does and the
others summarise what the literature says.

---

## Month 1 — StatsPAI: what the estimators actually construct

### 1.1 Event-study reference conventions (**done**)

Roth (2026) shows that recent DiD methods build the pre-treatment half of
an event study against a different reference period from the
post-treatment half, so a kink or a jump appears at the treatment date
with no treatment effect present. Delivered:

- `src/statspai/did/_bjs_pretrends.py` — the three constructions, with
  the BJS auxiliary lead regression (untreated subsample, leads with all
  earlier relative times pooled as the omitted category) solved by
  Frisch–Waugh against the same Y(0) design the imputation step uses.
- `sp.did_imputation(..., pretrend_method=)` — `"bjs"` (new default),
  `"in-sample"` (previous behaviour; `fect`/`did2s`), `"symmetric"`
  (Roth's `β̂^{BJS,new}`).
- `sp.event_study_convention()` / `sp.compare_event_study_conventions()`
  — the convention registry, and an empirical audit that decomposes each
  path's gap from the dynamic TWFE benchmark into a per-half vertical
  shift and a residual. Warns when the registry disagrees with the data.
- ⚠️ correctness: the previous leads were in-sample residual means,
  attenuated by `N0/N`; documented in CHANGELOG and MIGRATION.
- A rank guard: leads covering every pre-treatment period of a cohort are
  collinear with that cohort's unit effects and now raise.
- `sp.did(method="bjs", ...)` now forwards `pretrends=` and
  `pretrend_method=` instead of dropping them.

Evidence: coefficients and standard errors match Stata `did_imputation,
pretrends(k)` to 1e-12 / 1e-13 relative on two designs; `symmetric`
equals dynamic TWFE up to a common shift to 1e-8; `in-sample` equals
`symmetric` times `N0/N` to 1e-10; CS `base_period="universal"`
reproduces dynamic TWFE to 1e-8.
Tests: `tests/test_did_event_study_conventions.py` (16, including the
R `did2s` pins added in 1.3).

### 1.2 The systematic audit (**done**)

`scripts/audit_reference_claims.py` generalises the discovery method. It
runs each estimator to establish which objects it reports, reads the
committed parity JSONs to establish which of those objects carry a pinned
reference value, and reports the difference. On the nine core staggered
estimators: **61 reported objects, 22 pinned, 39 unpinned.** Output
committed at `docs/parity_object_coverage.md`; `--check N` gates growth.

Two design decisions worth keeping:

- The headline (reported vs pinned) requires no judgement. Whether a
  docstring "claims" parity on a particular object does require
  judgement, so it is reported separately, with the matched string
  attached. An earlier revision folded them together and matched the R
  package `fect` inside the word "effects".
- The audit reads the parity **archive**, not test-file literals. The new
  Stata and R references added by this programme therefore do *not* close
  their rows. That is the correct incentive: closing a row should mean
  packaging a parity module, not widening the definition of a pin.

### 1.3 `sp.gardner_did` (**resolved: not a defect**)

It builds its leads the same in-sample way, and that turned out to be
correct: R `did2s` 1.2.1 is the reference it documents, and its
coefficients reproduce did2s to **3.5e-14** at all 26 horizons.
`did_imputation(pretrend_method="in-sample")` matches the same reference
to 1.4e-12, which upgrades that option's registry label from asserted to
verified. A convention is a defect only when it is not the one the
function claims.

What the run *did* surface is its standard errors: the analytic
event-study SE is a median **0.71x** the did2s value, which prices the
existing runtime warning that stage-one estimation is ignored. The
direction is not uniform (one horizon reaches 1.52x), so an omitted
positive variance term is not the whole story. Pinned as a measurement,
not fixed.

### 1.4 Parity-module packaging (**done**)

Deferred in the previous round because packaging required editing
`compare.py`, `TIER_A_FIXTURE_LOCK.json` and both parity READMEs, which
were then the other line's most active files while this branch was three
commits behind. That reason expired: those files were last touched 12
days ago and a sweep of all eight worktrees found none of them holding
the parity harness. Done now, on a current base.

`84_bjs_pretrends` pins the lead vector against Stata at rel < 1e-14
(estimate and SE) and the horizons against R and Stata at ~1e-8.

Two findings came out of the packaging itself, which is the argument for
doing it rather than leaving the numbers in a test file:

- **The comparator was silently dropping evidence.** `collect()` skipped
  any Python statistic the R side did not emit, discarding nine py-Stata
  rows across four modules (`31_dfl`, `59_liml`, `28_frontier`,
  `84_bjs_pretrends`) — all of which agree at machine precision. Fixed,
  with unmatched rows now reported instead of vanishing.
- **The imputation estimator's horizon standard errors are wrong, and
  this is new.** StatsPAI differs from the references by 4.9-13% with
  non-uniform sign; R `didimputation` and Stata agree with each other to
  ~3e-9. The existing runtime warning (analytic SE under-counts
  fixed-effect estimation variance) predicts a uniform direction and this
  gap does not have one, so the warning does not explain it. Registered
  as an open defect outside the tolerance budget, **not** diagnosed.
  This is the top remaining technical item.

- **Staggered `symmetric`.** Currently refuses outside non-staggered
  balanced designs, because the `N/N0` factor is only derived there. The
  generalisation is Liu (2025)'s block bias; implementing it needs a
  reference to check against.

### 1.5 Baker et al. (2026) forward-engineering contract (**done**)

Their eight-step recipe — define the target parameter, state the
identification assumption formally, choose the estimation method, declare
the uncertainty frame, estimate, run sensitivity, run heterogeneity, keep
learning — maps onto an emittable object. `sp.did_design_contract(result)` returns
the eight slots with what the call determined and what it left implicit.

The stated risk was that it becomes a form to fill in. The mitigation is
that it never fills a slot with a default: a comparison group the result
does not record comes back `undetermined`, not `nevertreated`. Step 8 is
`not-evidenced`, because whether DiD is the right design is a judgement
about the setting and no object can evidence it. Tests pin exactly this —
a bare object scores zero determined slots, and `aggte` is what moves
step 1 from undetermined to determined, since the raw `att_gt` object
genuinely has not chosen a summary.

**Not yet wired into `sp.audit_result` / `sp.preflight`.** Those live in
files the other work line is active in; see the deferral note below.

### 1.6 Few-cluster diagnostics from Ulloa-Pérez et al. (2025) (**done**)

The paper turned out to be enough; the replication materials were not
needed. Their grid is stated in the text: clusters in {30, 50, 100},
N in {500, 1000, 2000, 5000}, five periods, 1,000 replications. So are
the results: at 30 clusters two-way Mundlak covered 60–85% against a
nominal 95%, the doubly-robust and IPW estimators also under-covered, and
"all approaches had lower than nominal coverage with few clusters".

`sp.did_cluster_diagnostics` grades a design against those three cells
and nothing else. Below 30 it returns `below-evidence` — their finding is
the closest evidence available and does not extend downward, so the
honest report is "coverage unknown", not "coverage worse". The grid
boundaries are pinned by a parametrised test, so an edit that quietly
moves the threshold off what they actually ran has to move a test.

---

## Month 2 — Paper-DiD-JAE

### 2.1 A fourth taxonomy class (**done**)

The manuscript's taxonomy classifies gaps between two *numbers*. Roth's
finding is a gap between two *paths*, and the existing classes cannot
express it: the whole post-treatment half agrees to 1e-11 across seven
estimators while the plots support opposite readings.

- New taxonomy row **Reference-period asymmetry** (in
  `scripts/build_assets.py`, so it regenerates).
- New §5 finding, positioned after the three aggregation defects.
- `scripts/build_event_study_conventions.py` →
  `manuscript/tables/event_study_reference_conventions.tex`, registered
  in the exhibit map and wired into `make assets`.
- Abstract, introduction, discussion and conclusion counts moved from
  five findings to six; the "three anti-conservative" count updated.
- §2 now cites the 2025/26 synthesis literature and states plainly that
  estimator choice (their subject) and implementation fidelity (this
  paper's) are separate questions.

### 2.2 Remaining (**not started**)

- `manuscript/tables/object_coverage.tex` (new) reports the audit as a
  paper exhibit, in §4 where the "coverage of an estimator is not
  coverage of its options" claim lives. It is the measured form of that
  claim: 61 reported objects, 22 pinned, and every Section 5 finding in
  an unpinned cell.
- The public-application slots are still the paper's binding constraint,
  and nothing in this programme moves them. The Cengiz raw-data rerun
  blocker is unchanged.
- The convention table is Tier B (a known DGP). Promoting the finding
  would mean showing it changes a reading on a public panel — e.g.
  re-plotting a published event study under both conventions.
- de Chaisemartin et al. (2025) `did_multiplegt_dyn` overview supplies
  four worked real-data examples with matched Stata/R/Python commands.
  That is the most efficient available route to a second public
  application, and it should be the next thing attempted.

---

## Month 3 — Paper-JSS

### 3.1 Done

- New paragraph in §5 (both the compact and long-form parity sections):
  why comparing values is the wrong instrument for an event-study path,
  what the two new functions do, and the first cross-software pin on a
  pre-trend *vector*.
- Two verified bib entries; `make bib-split-check` passes.
- `manuscript/generated_claims.tex` regenerated against the branch.

### 2.3 Ordering constraint (JAE)

`scripts/check_exhibit_map.py` reports `tab:object-coverage missing
source artifact: statspai:scripts/audit_reference_claims.py`. That is
correct, not a bug: the exhibit depends on a script that lives only on
the unmerged StatsPAI branch. It resolves on merge, and until then it is
the check doing its job.

---

## Cross-repo merge order

1. StatsPAI `wt/did-lit2026`
2. Paper-DiD-JAE `wt/lit2026` (its object-coverage exhibit needs 1)
3. Paper-JSS `wt/lit2026` (its regenerated counts need 1)

### 3.2 Ordering constraint

The JSS branch presupposes the StatsPAI branch. Its regenerated claim
counts (1,173 registered functions, 394 agent cards) are only correct
once `wt/did-lit2026` is merged. **Merge StatsPAI first, then JSS.**

### 3.3 Not added to JSS, on purpose

`sp.did_design_contract` and `sp.did_cluster_diagnostics` are
agent-facing design diagnostics, which belongs in the JSS §3 material —
and §3 is precisely where the other line removed the agent-facing
contract subsection during its submission pass. Adding them there now
would re-open a section that line deliberately closed. They are recorded
here and in the CHANGELOG; whether JSS mentions them is that line's call,
not this one's.

### 3.4 Remaining

- `replication/scripts/generate_manuscript_claims.py` computes its repo
  root as `parents[3]`, which is wrong inside a git worktree. It belongs
  to the JSS audit line, so it was left alone; whoever owns that line
  should make it resolve the root from `git rev-parse --show-toplevel`.
- The JSS PDF's per-page human visual check has not been redone for the
  new paragraph.

---

## What this programme deliberately does not do

- **It does not touch the JOSS submission.** No claim in `paper.md`
  changes, and no GitHub Release is cut.
- **It does not chase estimator coverage.** The corpus would justify
  new estimators (dCDH complex designs, distribution-regression DiD);
  the argument for spending the time on conventions instead is that an
  uncovered estimator is a visible gap while a wrong convention is an
  invisible one.
- **It does not adjudicate `fixest` versus `eventstudyinteract` on the
  Sun–Abraham off-diagonal.** That open item from the JAE §5 is
  unchanged.
