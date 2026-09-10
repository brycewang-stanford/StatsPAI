# Parity sweep — running findings log

Working log for the cross-language parity campaign. Every entry is either
fixed in-branch, or carries the reason it is deferred. Nothing here is
allowed to end as "noticed and forgotten".

## Fixed

### F1 — `sp.rdbwselect` ran a retired rule of thumb (⚠️ correctness)
`rd/bandwidth.py` never imported `rd/_cct_bandwidth.py`, so the public
selector used a single-step formula whose exponent 1/5 equals CCT's
1/(2p+3) only at p == 1 and which produced no bias bandwidth `b`.
Measured on the Lee 2008 senate replica: `h` = 4.63 against
`rdrobust::rdbwselect` 17.75 (3.8x too narrow); the docstring meanwhile
advertised Calonico, Cattaneo and Farrell (2020). Rewired onto the CCT
cascade `sp.rdrobust` already used. Pinned by Track A module 88.

### F2 — the four `comb` bandwidth selectors were never implemented (⚠️ correctness)
In `_cct_bandwidth.cct_bandwidth`, `two`/`form` resolved `msecomb1`,
`msecomb2`, `cercomb1`, `cercomb2` to the plain `rd` cascade. R computes
the `rd`, `two` and `sum` cascades to completion and then combines the
finished h and b element-wise per side: `comb1 = min(rd, sum)`,
`comb2 = median(two, rd, sum)`. `comb1` masked the defect wherever `rd`
is already the smaller of the pair; `msecomb2` was 2.8e-4 low and
`cercomb2` 5.5e-3 low. This path is shared with the *already certified*
`sp.rdrobust(bwselect=...)`, so the defect reached a certified surface.

### F3 — `sp.rdbwselect` rounded its output to six decimals
A bandwidth is an input to the next estimator; rounding it capped any
downstream agreement at ~1e-6 relative and silently perturbed
`sp.rdrobust(h=...)` when a user fed one back in. Now returned at full
precision.

### F4 — `msesum` / `cersum` were unreachable from the public API
`_VALID_METHODS` listed eight of the ten selectors `rdrobust` offers,
omitting exactly the two sum-form cascades that `comb1`/`comb2` are
built from. Now `set(BW_SELECTORS)`.

### F5 - local randomization ignored missing data and reported p = 0.000 (⚠️ correctness)
`sp.rdrandinf`, `sp.rdwinselect`, `sp.rdsensitivity` and `sp.rdrbounds`
had no missing-data handling at all, while `sp.rdrobust` in the same
subpackage has always dropped incomplete rows. On the Lee 2008 senate
replica -- 93 missing outcomes, 6 of them inside a +/-2 window --
`sp.rdrandinf` returned `estimate = nan`, and because `abs(nan) >=
abs(nan)` is False for every draw the permutation counter never
incremented, so the reported p-value was exactly **0.000**. A statistic
that failed to compute was being published as the most significant
result the test can produce. All four entry points now drop non-finite
rows through the new shared `rd/_core._complete_cases` and warn with the
count. The estimate becomes 10.1675, which is the difference in means
of the complete cases.

### F6 - `sp.rdsensitivity` hung on `plt.show()` (⚠️ availability)
The function built a matplotlib figure unconditionally and ended with
`plt.show()`. Under an interactive backend (`macosx` is the default
here) that blocks until a human closes the window, so the call never
returned in a script, a test, a CI job or an agent session -- measured
at over ten minutes before being killed, against 0.58s now. Plotting is
now opt-in via `plot=False` and the figure is attached to
`result.attrs["figure"]` instead of being shown. Displaying is the
caller's decision.

`plt.show()` elsewhere in the package was audited at the same time:
`plots/_script_editor.py` is an interactive editor and is correct as is;
`rd/rkd.py` gates on `show=True`, which is opt-out rather than opt-in but
does not trap a caller who reads the signature.

### F7 — RD covariates collinear with the running variable degraded silently (⚠️ correctness)
Was O1. A covariate that is a smooth function of the running variable is
absorbed by the polynomial terms the estimator already fits, so the
augmented design loses rank and the adjustment is unidentified.

The mechanism is worth stating precisely because it is not a logic error
anyone would find by reading the code: StatsPAI and `rdrobust` both try a
Cholesky factorisation and fall back to a pseudo-inverse when it fails,
but NumPy's Cholesky *succeeds* where R's refuses — measured, the
smallest relative singular value is 7e-16 against a `sqrt(eps)` cutoff of
1.5e-8, and our Cholesky accepted it 12 times out of 12. So the shared
fallback never fired on our side. Tightening the pseudo-inverse cutoff to
R's was tried first and changed nothing, which is what located the real
divergence.

Carrying on was not self-consistent either: on one covariate scaling
`sp.rdrobust` went on to report a NaN standard error, and on another the
solve raised a bare `LinAlgError: Singular matrix` naming no covariate.
Now `sp.rdrobust` and `sp.rdbwselect` raise a `ValueError` naming the
offending covariate, via the shared `rd/_core._check_covariate_rank`.
Guarded by `tests/test_rd_covariate_rank.py`, which also asserts the
check stays quiet on merely-correlated covariates — a guard that refused
ordinary designs would be worked around rather than heeded.

### F8 — `sp.rdsampsi` had no data mode, so R's reference was unreachable
Was O2. Implemented from `deparse(rdpower::rdsampsi)` and
`rdpower:::rdpower.powerNR`. All three effect sizes now reproduce R's
required sample sizes exactly. The two details that decide the answer:
the sample size is ceilinged *inside* the Newton-Raphson solve rather
than at the end (rounding at the end gives 1015 where R reports 1016 at
tau=3), and the sides are allocated by `sqrt(variance)` rather than by
observed counts. Exposing the per-side variances needed a `components`
out-parameter on `cct_bias_corrected`; they were already computed and
then summed away.

### F9 - `rd/bandwidth.py`'s dead second implementation was removed
Was O5. The file exported exactly one function, `rdbwselect`, and it now
delegates entirely to `rd/_cct_bandwidth`. An AST reachability pass over
the module found 16 of its 17 top-level functions unreachable from that
entry point - 592 lines of the retired rule of thumb, including duplicate
copies of `_cer_factor` and `_local_residual_var` that `rd/rdrobust.py`
also defines for itself. All 16 are deleted; the file went 1007 to ~300
lines. The module and parameter docstrings went with them: they still
advertised "all eight bandwidth selection methods" and defined `comb1`
as `min(mserd, mseleft, mseright)`, which is not the rule.

`tests/test_cov95_rd_r2_bandwidth.py` lost the four tests that imported
the deleted private helpers and kept the four that exercise the public
`sp.rdbwselect`. Retargeting them was considered and rejected - they
asserted degenerate-bandwidth fallbacks of code that no longer exists, so
keeping them would have meant keeping the code.

### F10 - CLAUDE.md 5.1's Track A checklist now points at the gate
Was O6. The checklist was assembled from "what was missed last time", so
it lagged what the gates actually assert. Following it while adding
module 88 still missed four registrations (`HEADLINE`, the Python-side
reproducibility report, the fixture lock, and the schema bundle), which
the gates caught as 7 failures.

It now names `tests/test_parity_harness_contract.py` (42 assertions,
including `py_modules == set(TOLERANCES) == set(HEADLINE)`),
`scripts/tier_a_fixture_lock.py` and `make -C Paper-JSS audit` as the
authority, and marks the prose list as a non-normative overview. The
instruction changed from "tick the list" to "run the three gates and fix
what they report". The one trap that is *not* discoverable by running a
gate is called out explicitly: passing a single module name to
`verify_reproduce.py` overwrites the whole report with that one row.

The value of the change was demonstrated immediately. A later gate run in
the same session caught two *further* module-88 gaps that neither the old
checklist nor the first gate pass had surfaced: `tests/stata_parity/README.md`
had no row for it (the R-side README did), and
`test_strictness_tier_breakdown_matches_current_artifacts` carries a
hard-coded `machine: 78` that had to become 79. Fixing the Stata README
then invalidated the fixture lock, which is hash-locked over it, so the
lock had to be regenerated again. Each registration triggers the next
gate; that chain is the thing a static list cannot represent.

### F11 - the Stata run's empty-named artifacts are now ignored
Was O4. A full `verify_reproduce_stata.py` run leaves
`tests/stata_parity/.pdf` and `.png` behind: a `graph export` whose
filename macro resolves to the empty string. No `.do` file in this repo
calls `graph export`, so the caller is one of the installed reference
ados, not our code -- which means it cannot be fixed here, only
contained. Both paths are now in `.gitignore`, so they stop showing up as
untracked noise after every Stata run and cannot be swept into a change
set by accident.

### F12 - `sp.rdms` rebuilt on the reference's own construction (⚠️ correctness)
Was O8. The estimator now builds the score `rdmulti::rdms` builds --
Euclidean distance to the boundary point signed by treatment status --
and delegates to `sp.rdrobust`, so it inherits the CCT cascade and robust
bias correction instead of re-deriving them badly. Track A module
`89_rdms` pins three boundary points at 7.6e-12 against R and 3.3e-9
against Stata, with the six effective sample sizes exactly equal on all
three sides.

Two things were needed beyond the arithmetic. `treat=` (R's `zvar`) was
added, because treatment on a two-dimensional boundary is not implied by
the coordinates -- the old code hard-coded `x1 >= cutoff1` behind a
comment reading "Convention", which is an assumption about the design
masquerading as a fact about the data. And the Stata side needed
`rdmulti`, which is **not** on SSC: `ssc describe rdmulti` returns
`r(601)`, tested rather than assumed per CLAUDE.md 5.1, so the `.do` file
`net install`s from the rdpackages GitHub mirror into a local gitignored
ado path rather than the user's PLUS.

Two naming traps were caught by checking rather than by reading. R's
`rdms$B` is the bias-*corrected estimate* while Stata's `e(B)` is the
bias *bandwidth* -- same letter, different object. And R's printed table
shows `Coefs` (conventional) with the robust CI, so the first comparison
"disagreed at 1.5%" purely because a bias-corrected estimate was being
held against a conventional one. Both point estimates are pinned
separately now so neither can be confused for the other again.

### F13 - `sp.rdrobust` rounded its reported bandwidth on the default path
Found while comparing module 89's bandwidths. `bwselect="mserd"` (the
default) reported `h = 0.342397`; `bwselect="cct"` -- the same cascade,
different spelling -- reported `0.34239727634769035`. The estimate is
computed at full precision either way, so the rounding was invisible in
the result and only surfaced when the *reported* `h` was held against R's
`0.34239727634748`.

It matters because `model_info["bandwidth_h"]` is read back out and
re-fitted at: `rd/diagnostics.py`, `rd/dashboard.py` and `rd/rdrobust.py`
all do it. This is F3 one layer down. An existing test already asserted
full precision -- but only for the `cct` spelling, which is exactly why
the other path kept its rounding. The new test covers both spellings and
asserts they agree; reintroducing the rounding makes it fail.

### F15 — `sp.rdbwselect(fuzzy=)` parsed the argument and discarded it (⚠️ correctness)

The function read the treatment column, validated it, dropped its missing
rows — and then called `cct_bandwidth` without it. What came back was the
sharp bandwidth, while the parameter's docstring promised the bandwidth
"accounts for first-stage variance in the Wald / IV estimator".

The reason this is worth stating separately from "we forgot to pass an
argument" is **why it was invisible**. The repository's fuzzy fixture used
one-sided noncompliance (`treat = (margin >= 0) & (i %% 10 != 0)` — nobody
below the cutoff treated). R's `rdbwselect` detects exactly that case:
`var(T_l) == 0` sets `perf_comp`, which drops `T` and returns the sharp
bandwidth rather than dividing by a zero first-stage jump. So the reference
*also* returned the sharp bandwidth, the fixture agreed to 1.2e-08, and the
RFC recorded the conclusion "`fuzzy` already gets the correct bandwidth".
It was the sharp cascade agreeing with itself.

Measured on a two-sided-noncompliance replica of the same data: `h` is
**9%–16%** off. The new fixture
(`_fixtures/_generate_rdrobust_fuzzy_R.R`) is built that way for this
reason, and `test_rdbwselect_passes_fuzzy_through` asserts not only that
the fuzzy bandwidth matches R but that it **differs from the sharp one** —
without that second assertion the test would have passed against the
broken version too.

### F16 — fuzzy bias correction was a ratio of separately corrected parts (⚠️ correctness)

`sp.rdrobust(fuzzy=)` reached the CCT operator for the sharp quantities and
then divided: the sharp bias-corrected estimate by a separately
bias-corrected first stage, and both SEs by `|first stage|`.

R does not do that. It builds `D = [Y, T, Z...]`, applies the
bias-correction operator `Q_q` to every column at once, and forms

    s_Y    = [1/tau_T, -tau_Y/tau_T^2]
    tau_bc = tau_cl - s_Y' (bias_Y, bias_T)

with the same `s_Y` collapsing the residual matrix before the sandwich, so
the covariance between numerator and denominator is carried into both
variances. The old construction drops it. Measured: robust estimate 0.96%
off, robust SE 2.5%, conventional SE 1.1%.

`_vbr` needed the same treatment — the MSE being minimised is the ratio's,
not the reduced form's — which is what makes F15's 9–16% a bandwidth error
rather than a cosmetic one. Both are now pinned across `p` 1–2,
covariates, `hc0`–`hc3`, `mserd`/`cerrd`/`msetwo` and two kernels at
6.3e-13, and the four `xfail(strict=True)` markers that guarded this are
removed. **That was the last cross-language xfail in the suite.**

### F17 — `compare.py` manufactured a phantom `Paper-JSS/` in every worktree

`tests/r_parity/compare.py` unconditionally did
`PAPER_TABLES_DIR.mkdir(parents=True)` to drop one `.tex` into
`Paper-JSS/manuscript/tables/`. `Paper-JSS/` is a git-ignored, local-only
tree that exists in the main checkout and not in a worktree, and three JSS
test modules skip themselves on `Paper-JSS/.exists()`.

So running the parity comparison inside a worktree created a directory with
a single `.tex` in it and no replication scripts underneath, which defeated
the skip guard and turned two designed skips into two failures — failures
that look like a regression in the change under test and are not. The write
is now conditional on `Paper-JSS/manuscript` already existing, and says so
when it skips.

## Open

### O3 - Paper-JSS: regenerate after merge, plus one literal the macros miss

*Still open. The counts below moved again with the fuzzy work: module
count is unchanged (fuzzy is pinned in `reference_parity`, not Track A),
but `RegistryCertified` / `ParityCrossLanguage` should be re-derived
from the main checkout rather than from this table.*
`Paper-JSS` is a separate, gitignored repository living inside the main
tree, and `replication/scripts/generate_manuscript_claims.py` resolves
`statspai` from the **main** tree -- so running it from this worktree
would regenerate the pre-merge numbers. It has to happen after the merge.

What the generator will emit once this branch lands (computed here):

| macro | now | after |
| --- | ---: | ---: |
| `RegistryCertified` | 168 | 176 |
| `RegistryCertifiedValidated` | 394 | 397 |
| `ParityCrossLanguage` | 169 | 177 |
| `ParityInternalEvidence` | 226 | 221 |
| `ParityEstimatorCrossLanguagePct` | 21.9% | 22.9% |
| `RParityModuleCount` | 87 | 89 |
| `StataParityModuleCount` | 81 | 83 |

`RegistryCertified` stays one below `ParityCrossLanguage`, which is the
`did_multiplegt_dyn` exception the manuscript already explains (its API
is experimental while its numbers are parity-backed), so that prose
holds.

**One number will not move on its own.**
`sections/05-parity-compact.tex:298` reads "Of the
`\RParityModuleCount{}` R-joined modules, **86** receive a pass-type
verdict" -- a literal. The tier breakdown confirms the arithmetic: 87
modules were 78 machine + 7 iterative + 1 moderate + 1 T4, i.e. 86
pass-type; with modules 88 and 89 it is 80 + 7 + 1 + 1, so the literal
becomes **88**. Worth replacing with a generated macro rather than editing,
since this is precisely the drift the claims mechanism exists to stop
and it escaped as a literal.

Pre-existing drift found while checking, **not** caused by this branch:
`sections/05-parity.tex` (lines 212, 1022) and
`sections/08-computational-details.tex` (line 38) still say "81 R-joined
modules". Neither file is `\input` by `main.tex` -- both are the
archival long-form versions, so nothing compiled is wrong today -- but
they will mislead whoever reads them next. The compiled sections use
`-compact` variants; the "86"/"87" in `04-examples-compact.tex` are
module *identifiers* (`86_fect`, `87_interflex`), not counts, and are
correct as they stand.

Sequence after merge: `make -C Paper-JSS manuscript-claims`, fix the
literal, then `make -C Paper-JSS audit`.

### O7 - the pinned `black` disagrees with the repo, in both directions
`.pre-commit-config.yaml` pins `black` at **23.9.1**; the venv carries
**26.5.1**. Measured on this branch:

| black | files it would rewrite |
| --- | ---: |
| 23.9.1 (the pin, `stages: [pre-commit]`) | **140** |
| 26.5.1 (the venv) | 12 |

The committed code matches neither, and is far closer to the newer one.
This is not a dormant inconsistency: the hook runs at pre-commit stage,
so the next hook-triggering checkpoint rewrites 140 files as a side
effect. It was hit twice while doing this work and reverted both times.

The 12 files the newer `black` disagrees with are all in `examples/`,
`papers/` and `scripts/` - none touched by this branch, whose own files
are clean under 26.5.1.

Not decided here, deliberately. Either resolution produces a large diff
unrelated to parity (bump the pin and reformat 12 files, or keep the pin
and reformat 140), and choosing the target version is a preference about
the repo rather than a correctness question; CLAUDE.md 9.2 also warns
against casually editing shared config. Recommendation: bump the pin to
the version the code actually reflects and reformat the 12 stragglers as
a change of its own, so the hook stops being a landmine.

### F14 - module 06's golden had recorded a rounded bandwidth
Surfaced by the three-sided reproducibility run after F13 landed: the
Python side reported `06_rd` as **drift**, worst 1.7e-08, on exactly one
row -- `legacy_internal_mserd_bandwidth_h`, which moved from `17.754397`
to `17.7543972960588`.

That is the F13 fix showing up in a committed artifact rather than a new
problem: the golden had been recording a rounding artifact as if it were
the selected bandwidth. Regenerated by re-running the module, as
CLAUDE.md 5.1 requires (goldens are never hand-edited). The parity join
is unaffected -- the `legacy_internal_mserd_*` rows are Python-only
diagnostics with no R or Stata counterpart, so `compare.collect` drops
them -- and the drift stayed inside the 1e-6 budget throughout.

Two stale claims went with it. The module's `bandwidth_parity_note` said
the rows "should not be used as the cross-language default-h parity
claim", and the inline comment called them a "legacy internal-selector
diagnostic". Both describe a separate rule-of-thumb selector that no
longer exists: `sp.rdrobust(...)` with no `bwselect` now reaches the same
CCT cascade as `bwselect='cct'`, and the two spellings agree to 1.7e-12
on the bandwidth and 3e-14 on the robust estimate. The rows are kept --
they are a useful convergence check between the two spellings, which is
the module-level counterpart of the F13 unit test -- but they now say so.

The names `legacy_internal_mserd_*` were deliberately **not** changed.
`tests/test_parity_gap_boundaries.py` already pins the right property
("pin the convergence, not the historical gap") and the name records
where the row came from; renaming would have churned two tests, the
golden, both rendered tables and the fixture lock to no numerical end.

## Self-audit: do my own documented numbers match the code?

Every campaign entry above quotes measurements. A number in a CHANGELOG
that has drifted from the code is the same defect this work exists to
find, so the current-state claims were re-derived from the installed
package rather than trusted:

| claim | source | verified |
| --- | --- | --- |
| module 88: 1.8e-12 vs R, 3.7e-9 vs Stata | CHANGELOG, TOLERANCES, docstring | yes |
| module 89: 7.6e-12 vs R, 3.3e-9 vs Stata | CHANGELOG, MIGRATION, TOLERANCES | yes |
| `sp.rdsampsi` exact at all three effect sizes | CHANGELOG, promotion note | yes |
| `sp.rdwinselect` Nl/Nr exactly equal to R | CHANGELOG, promotion note | yes |
| `sp.rdsensitivity` returns in well under a second | CHANGELOG (0.58s) | yes |
| `sp.rdbwselect` exposes ten selectors, unrounded | CHANGELOG, docstring | yes |
| cross-language 177, internal 221, 22.9% | O3 projection | yes |

12 of 12. The historical figures (the pre-fix 4.63 against 17.75, the
p = 0.000, the 4.804 at the middle boundary point) cannot be re-derived
without reverting the fixes; each was measured at the time and is quoted
with the fixture it was measured on, so a reader can reproduce it by
checking out the parent commit.
