# JSS review period: changes to frozen manuscript artifacts

The JSS manuscript describes one tagged release (the `tag` in
`tests/jss_review_freeze.json`). Its parity, coverage, seed-study and timing
tables are re-tabulated from frozen artifacts, which that manifest hashes.
While the paper is under review, any change to one of those artifacts must be
recorded here; `tests/test_jss_review_freeze.py` fails until it is.

Rules:

1. **Do not regenerate a frozen artifact to make a table look better.** A
   change here is a correctness fix, a reference-environment change, or a
   new module, and it goes through CHANGELOG / MIGRATION as usual
   (⚠️ correctness fix if an output moves).
2. **One entry per change**, newest first, giving the date, the commit, the
   reason, and the effect on the paper: which table, listing or quoted
   number moves, and from what to what. "None" is a valid effect only if it
   was checked (for example, a module the paper does not print).
3. **List every changed path in backticks**, exactly as the test reports it,
   and **the SHA of every commit that changed it** (7+ hex characters, in
   backticks) in the same entry. Since 2026-09-28 the check is per commit:
   naming a path once no longer covers later commits to the same file.
   Merge commits that bring a frozen file in count as commits.
   A commit cannot name its own SHA, so the entry goes in a follow-up
   commit pushed together with it; the check runs at pre-push and in CI,
   never at pre-commit.
4. The manuscript itself is not edited during review. Recorded changes are
   folded into the next revision, which is re-anchored to a new release and
   re-frozen with `python scripts/jss_review_freeze.py --write --release X.Y.Z`.
5. When the paper is decided, set `"active": false` in the manifest.

## Entries

### 2026-10-05 — call traces re-recorded after the weight diagnostic reached logit, probit, glm and iv

Commit `8cdfa8d8`. `sp.logit`, `sp.probit`, `sp.glm` and `sp.iv` now record
the Kish effective sizes of a weighted fit and warn when the weights are
dispersed. The edits are in `regression/logit_probit.py`,
`regression/glm.py` and `regression/iv.py`, which Track A modules 02, 35,
48, 57 and 59 and the original-data module 01 execute, so their source
hashes in the traces changed.

Effect on the paper: none. The six modules were run on the source before
and after the change and their result files are identical; against the
committed files they reproduce within 3e-11, inside the 1e-9
reproducibility tolerance, and the committed files are kept.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after two error classes changed in `sp.fect` and `sp.interflex`

Commit `2df681ff`. Four precondition errors (`method='ife'` without `r=`,
`method='mc'` without `lam=`, `estimator='kernel'` without `bw=`,
`vce='bootstrap'` with another estimator) now raise
`MethodIncompatibility` with a recovery hint instead of a bare
`ValueError`. The edits are in `synth/fect.py` and
`regression/interflex.py`, which Track A modules 86 and 87 execute, so
their source hashes in the trace changed.

Effect on the paper: none. The two modules were run on the source before
and after the change and their result files are byte-identical.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — traces of modules 24, 41 and 62 re-recorded for the second Wooldridge round

Commit `208af7fd`. `regression/tobit.py` and `regression/truncreg.py` gain a
`formula=` argument, and `survival/models.py` accepts the data first in
`sp.survreg`. Those files are on the estimation path of Track A modules
24, 41 and 62, whose traces were re-recorded.

**Effect on the paper.** None. The new argument is not used by the
modules and the fits with `y=` and `x=` are unchanged; only source
digests and `seconds` differ in the ledger, and no implementation
classification moves. No result file changes.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the Wooldridge textbook pass

Commit `85cdb0c6`. The pass edits `core/results.py` (`predict`),
`core/utils.py` (the namespace formulas are evaluated in),
`regression/ols.py`, `regression/iv.py`, `regression/glm.py`,
`regression/logit_probit.py`, `regression/quantile.py`,
`panel/panel_reg.py`, `did/_core.py` and the `estat` modules. Those files
are on the estimation path of 72 Track A modules and of original-data
modules, so both ledgers were re-recorded on the committed tree.

**Effect on the paper.** None. Only `exercised_sources` digests and
`seconds` change; compared field by field with the previous ledgers, no
module's implementation classification moves, so the native / port /
third-party census and the provenance marks of the original-data ledger
are unchanged. No estimate, standard error or table cell is read from
these files. The estimates of the modules themselves are untouched: the
changes are to prediction, diagnostics and formula parsing, and the
Track A result files are byte-identical.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — original-data ledger refreshed; LaLonde PSM row regenerated

Commits `c725ba3d` (the result and the roll-up table) and `dfe32e16` (the trace). Eight original-data modules had traces that no longer
described the tree, after source changes that had re-recorded the R
ledger only: `regression/ols.py` (the few-treated diagnostic, `5ad164e1`),
`regression/iv.py`, `matching/match.py` (`39a8be49`), `synth/scm.py` and
`inference/ipw.py`. Every module was rerun on the current tree. Seven
reproduce their committed result within the 1e-9 reproducibility
tolerance (the largest move is 1.3e-10, the Card IV coefficient) and
keep the committed file. One does not: the tied-propensity-score fix of
`39a8be49` moves `psm_att` in module 04.

Effect on the paper: the original-data table's LaLonde row. `psm_att`
goes from 1963.43 to 1967.94 and its standard error from 1186.31 to
1192.17; the relative gap to R `MatchIt` (2006.86) goes from 0.022 to
0.019 and to the published 1794 from 0.094 to 0.097. No other row moves.

- `tests/orig_parity/results/04_lalonde_original_py.json`
- `tests/orig_parity/results/parity_table_orig.md`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after `sp.iv` accepted `small=False` for GMM

Commit `dfe32e16`. `regression/iv.py` gained the GMM branch of the
large-sample rescaling and a longer docstring; Track A modules 02, 35
and 59 execute that file, so their source hashes in the trace changed.

Effect on the paper: none. The three modules were run on the source
before and after the change and their result files are byte-identical.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — outcome-only classic SCM fits the raw pre-treatment path; module 52 regenerated

Commits `51f569ba` and `81314cbb`. `synth/scm.py`, on the estimation path
of Track A modules 07 and 52, changes a default: without `covariates=` or
`special_predictors=` the pre-treatment outcomes are no longer rescaled
period by period before the donor weights are solved
(`standardize_predictors=None`). `51f569ba` re-records the traces of both
modules. `81314cbb` regenerates the Python result of module 52, whose
fits are outcome-only, with the fixture lock, the parity tables and the
index that depend on it.

**Effect on the paper.** Module 07 uses special predictors and reproduces
its committed file exactly. Module 52: the average post-treatment gap,
the pre-treatment RMSE and the five donor weights are unchanged to 1e-14
(the design has a unique exact fit, which both problems share), so its
parity rows against R `Synth` and Stata `synth` are unchanged at
4.1e-10. Its placebo standard error, which no reference reports and no
parity row uses, moves from 2.159192 to 1.828665, because the placebo
fits of the donor units are not exact and follow the new default. In
`parity_table.md` that changes the StatsPAI SE cell of the
`avg_post_gap` row; in the three-way table the Stata column's worst
relative difference for module 52 reads 2e-15 instead of 9.2e-15. No
tolerance, status or tier moves. If the manuscript quotes the README's
Proposition 99 example, that number is now -19.51 (10.80) instead of
-19.76 (11.23).

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/r_parity/results/52_scm_unique_py.json`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/results/parity_table_3way.tex`

### 2026-10-05 — call traces re-recorded after the weight diagnostic reached the fast and negative-binomial entry points

Commit `d7b67583`. `sp.fast.feols`, `sp.fast.fepois` and `sp.nbreg` now
record the Kish effective sizes of a weighted fit and warn when the
weights are dispersed. The edits are in `core/_agent_summary.py`,
`fast/feols.py`, `fast/fepois.py` and `regression/count.py`, which Track
A modules 03, 14, 15, 37, 42, 47, 53, 58, 63 and 64 execute, so their
source hashes in the trace changed.

Effect on the paper: none. The ten modules were run on the source
before and after the change and their result files are byte-identical;
only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the few-treated diagnostic reached the fixed-effects entry points

Commit `5ad164e1`. `sp.panel`, `sp.hdfe_ols`, `sp.feols` and `sp.regress`
now record a 0/1 regressor that is ever 1 in only a few clusters and
warn. The edits are in `core/_agent_summary.py`, `regression/ols.py`,
`panel/panel_reg.py`, `panel/feols.py` and `fixest/wrapper.py`, which
Track A modules 01, 14, 35, 51, 53, 54, 55, 56, 67 and 69 execute, so
their source hashes in the trace changed.

Effect on the paper: none. The ten modules were run on the source
before and after the change and their result files are byte-identical;
only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded for the third round of the Clarke textbook pass

Commit `ff889a42`. `matching/match.py` and `matching/_ai2016.py` (module 11)
gained the caliper condition of the Abadie-Imbens (2016) ATE variance, a
branch module 11, which estimates the ATT without a caliper, does not
execute. `forest/causal_forest.py` (modules 13 and 24) accepts `treat=` and
`covariates=` as other names for `d=` and `x=`; the decorator line changed
and nothing below it. The traces of modules 11, 13 and 24 were re-recorded.
The rest of the commit is in `agent/_translation/`, which no Track A
module executes.

**Effect on the paper.** None. Only the recorded source hashes move; no
result file changed and no module's implementation classification moved.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after a docstring in `synth/scm.py` changed

Commit `57ccb23f`. `synth/scm.py` is on the estimation path of Track A
modules 07 and 52. Only its docstring changed: the description of the
default without predictors, which called the fit the simplex
least-squares estimator, now says that each pre-treatment period is
rescaled and how to obtain the least-squares weights. No executable line
changed. The traces of modules 07 and 52 were re-recorded on the tree of
that commit.

**Effect on the paper.** None. Only `exercised_sources` digests and
`seconds` changed in the trace, and no module's implementation
classification moved. The Python results of modules 07 and 52 were not
regenerated. No estimate, standard error or table cell is read from this
file.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the last two commits of the Croissant textbook pass

Commits `912e97d9` and `623ca007`. Source files on the estimation path of
Track A modules changed: `regression/tobit.py` (module 41; `method='scls'`
is a new branch, the maximum likelihood path is unchanged),
`regression/logit_probit.py` (modules 48 and 57; fits also report a family
of goodness-of-fit measures), `regression/multinomial.py` (modules 44, 45,
46 and 49; fits store their per-observation log-likelihood), and
`src/statspai/__init__.py` (new exports `sp.etpoisson` and `sp.nlogit`),
which is on the path of modules 03, 13, 15, 24, 25, 26, 27, 53, 65 and 66
and of original-data module 08. The traces of all of these were
re-recorded.

**Effect on the paper.** None. Modules 41, 44, 45, 46, 48, 49 and 57 were
re-run with `tests/r_parity/verify_reproduce_py.py` and are byte-identical
to their committed files. No result file was regenerated and no parity row
changes. No module's implementation classification moved.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`
### 2026-10-04 — call traces re-recorded after a comment in `callaway_santanna.py` changed

Commit `6c00bcce`. One source file on the estimation path of Track A
modules 04 and 79 changed, `did/callaway_santanna.py`, and only in a
comment: the note above the default propensity-score trimming level said
Stata `csdid` inherits 0.995, and `csdid` 1.81 defaults to no trimming.
No executable line of the file changed. The same commit changes
`panel/feols.py` (a base level for a categorical regressor) and
`inference/g_computation.py`, which are on no Track A module's path. The
traces of modules 04 and 79 were re-recorded on the tree of that commit.

**Effect on the paper.** None. The Python results of both modules were
re-run on the new tree: module 04 reproduces its committed file exactly
and module 79 to 1.4e-15. Only `exercised_sources` digests and `seconds`
changed in the trace. No estimate, standard error or table cell is read
from this file.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call trace of module 73 re-recorded after a docstring edit

Commit `75f9a73a`. A paragraph in the docstring of `sp.gardner_did`
(`did/gardner_2s.py`) moved under the `second_stage` parameter so that the
strict docs build parses it. Track A module 73 executes that file.

Effect on the paper: none. No code changed; only the recorded source hash
of module 73 moves.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call trace re-recorded after `sp.rdms` gained the cumulative-cutoff form

Commit `a002611b`. `sp.rdms` without `x2` estimates one score with
cumulative cutoffs. The code is in `rd/rdmulti.py`, which Track A module
89 executes.

Effect on the paper: none. Module 89 calls the two-score form with scalar
cutoffs, which is unchanged; it was re-run and its result file is
byte-identical. Only the recorded source hash moves.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the second round of the Clarke textbook pass

Commit `a4a892ff`. `matching/match.py` and `matching/_ai2016.py` (module
11) gained the Abadie-Imbens (2016) standard error for the ATE, a new
function and a new branch that module 11, which estimates the ATT, does
not execute. `inference/jackknife.py` (module 53) gained
`sp.wild_cluster_boot(confidence_set=)`, off by default. The traces of
modules 11 and 53 were re-recorded.

**Effect on the paper.** None. Only the recorded source hashes move; the
result files of both modules are untouched and no module's implementation
classification changed.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the general form of `sp.gardner_did` and `sp.drdid(est_method='all')`

Commit `87fd76ad`. Two source files on the estimation path of Track A
modules changed. `did/gardner_2s.py` (module 73) gains `treat=`, `fe=`
and `second_stage=`, which hand over to a new module before the existing
code is reached; a call with `first_treat=` runs the lines it ran before.
`did/wooldridge_did.py` (modules 17 and 38) gains
`sp.drdid(est_method='all')`, a loop over the existing estimators that
returns before the body when it is requested. The traces of modules 17,
38 and 73 were re-recorded on the tree of that commit.

**Effect on the paper.** None. The Python results of the three modules
were re-run on the new tree and on the tree before the change. Module 17
reproduces its committed file exactly. Modules 38 and 73 differ from
theirs by 1.5e-11 and 7.3e-11 in relative terms on both trees, so the
difference is the machine's and not the change's, and it is inside the
1e-9 reproducibility tolerance. Only `exercised_sources` digests and
`seconds` changed in the trace, and no module's implementation
classification moved. No estimate, standard error or table cell is read
from this file.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded after `sp.rdmcplot` and `sp.bitest` were exported

Commit `c1af9b25`. Two new public functions are imported in
`src/statspai/__init__.py`, which the modules that reach the package
through a lazy attribute execute: Track A 03, 13, 15, 24, 25, 26, 27, 53,
65, 66 and original-data module 08.

Effect on the paper: none on any number. The eleven modules were re-run
and their result files are byte-identical. The registered-function count
moves from 1,300 to 1,303, which the manuscript's inventory will pick up
at the next re-anchoring.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded after `sp.best_linear_projection` gained `.attrs['vcov']`

Commit `6a2558e3`. The projection of forest scores now attaches the
coefficient covariance to the returned table. The edit is in
`forest/_grf_inference.py` and `forest/forest_tools.py`, which Track A
modules 13 and 24 execute.

Effect on the paper: none. No estimate or standard error is computed
differently; the result files of modules 13 and 24 are untouched, and
only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded after `sp.survreg` was rewritten

Commit `dbad2724`. `survival/models.py` is on the estimation path of
Track A module 24 (`sp.cox`). Only `sp.survreg` in that file changed (it
now honours `robust=` and `cluster=`, iterates to the optimum and supports
gamma frailty); `sp.cox` was not touched. The trace of module 24 was
re-recorded. The original-data trace was re-recorded in the same run
because the gate reported it stale; none of its modules has
`survival/models.py` on its path, so only digests moved there.

**Effect on the paper.** None. Module 24 was re-run with
`tests/r_parity/verify_reproduce_py.py` and is byte-identical to its
committed file. No module calls `sp.survreg`, so no parity row changes.
No module's implementation classification moved.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — original-data call trace of module 02 re-recorded

Commit `424bdd41`. Commit `6c9fd021` edited `did/callaway_santanna.py`,
which the original-data module 02 (mpdta) executes, and left its trace
stale. The same commit gives `RLassoEffectsResult` the agent contract;
that class is on no parity module's path.

Effect on the paper: none. Module 02 was re-run and its result file is
byte-identical; only the recorded source hash moves.

- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — call trace re-recorded after `sp.rdms` gained several boundary points

Commit `7f700ff5`. `sp.rdms` accepts lists for `cutoff1=` / `cutoff2=`
and an `xnorm=` column, returning one row per boundary point and a pooled
row. The code is in `rd/rdmulti.py`, which Track A module 89 executes.

Effect on the paper: none. Module 89 calls `sp.rdms` with scalar cutoffs,
a path that is unchanged; it was re-run and its result file is
byte-identical. Only the recorded source hash moves.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the Remix labs pass (time-varying covariates in Callaway-Sant'Anna, unidentified imputation leads)

Commit `6c9fd021`. Two source files on the estimation path of Track A
modules changed. `did/callaway_santanna.py` (modules 04 and 79) reads a
covariate that varies within unit in the earlier period of each ATT(g,t)
cell instead of in the unit's first row, and gains the improved doubly
robust estimator `estimator='drimp'`. `did/_bjs_pretrends.py` (module 84)
refuses lead coefficients that the design does not identify instead of
inverting a singular matrix. The traces of modules 04, 79 and 84 were
re-recorded on the tree of that commit.

**Effect on the paper.** None. The Python results of modules 04, 79 and 84
were re-run on the new tree, together with 16 and 17, whose estimators
share files with the change. Each reproduces its committed file; the
largest relative difference is 1.4e-15 (module 79). None of these modules
uses a time-varying covariate, and module 84 requests leads that are
identified, so the changed branches are not reached. Only
`exercised_sources` digests and `seconds` changed in the trace, and no
module's implementation classification moved. No estimate, standard error
or table cell is read from this file.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the Croissant textbook pass (zero-modified counts, per-observation log-likelihoods, new exports)

Commits `e799808c`, `725a734a` and `d9bfd690`. Source files on the
estimation path of Track A modules changed: `regression/zeroinflated.py`
(modules 63 and 64; the logistic terms of the likelihood are evaluated in
an overflow-safe form, and the Vuong statistic in `diagnostics` compares
against a model fitted on its own), `regression/count.py` and
`regression/logit_probit.py` (modules 37, 42, 47, 48, 57 and 58; fits
store their per-observation log-likelihood), `regression/tobit.py` (module
41; the result keeps its design for `sp.cmtest`), and
`src/statspai/__init__.py` (new exports `sp.ivprobit`, `sp.ivtobit`,
`sp.ivpoisson`, `sp.cmtest`, `sp.vuong`), which is on the path of modules
03, 13, 15, 24, 25, 26, 27, 53, 65 and 66 and of original-data module 08.
The traces of all of these were re-recorded on the tree of `d9bfd690`.

**Effect on the paper.** None. The Python results of modules 37, 41, 42,
47, 48, 57, 58, 63 and 64 were re-run on the new tree with
`tests/r_parity/verify_reproduce_py.py`. Every one reproduces its
committed file, the largest relative difference being 8.9e-16. No result
file was regenerated and no parity row changes. The Vuong statistic that
changed value (see `CHANGELOG.md`) is not an output of modules 63 or 64.
No module's implementation classification moved.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the Clarke textbook pass (matching scores, `sp.twfe_decomposition`, `sp.wild_cluster_boot`)

Commit `39a8be49`. Three source files on the estimation path of Track A
modules changed: `matching/match.py` and `matching/_ai2016.py` (module 11;
the propensity index is evaluated once per distinct covariate row, and a
probit score is available), `did/wooldridge_did.py` (modules 17 and 38;
only `sp.twfe_decomposition`, which neither module calls, was rewritten)
and `inference/jackknife.py` (module 53; `sp.wild_cluster_boot` gained
`h0=`). The traces of modules 11, 17, 38 and 53 were re-recorded.

**Effect on the paper.** None. The Python results of the four modules were
re-run on the new tree. Modules 17 and 53 are byte-identical. Module 11
moves in the sixteenth digit (621.7932847377471 to 621.7932847377473),
which is the row-wise evaluation of the same index. Module 38 differs from
its committed file in the twelfth digit, and does so identically with the
previous `wooldridge_did.py` put back, so that is this machine's numerical
libraries against the ones the file was produced with, not this commit.
Both are far inside the 1e-9 reproducibility tolerance, so the committed
result files were left as they are and no parity row changes. No module's
implementation classification moved.

- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — module 22 (`sp.sensemakr`) standard error from a QR factor; traces re-recorded for the causal-ML textbook pass

Commit `b24b2777`. `sp.sensemakr` computed the treatment standard error
from `pinv(Z'Z)`, which costs eight digits on unscaled regressors. It now
uses the QR factor of `Z`. The same commit fixes classifier nuisances in
`sp.dml` PLR / PLIV, the IRM branch of `sp.dml_sensitivity`, and adds
`sp.lm_lin`; these touch `dml/_base.py`, `dml/plr.py`, `dml/irm.py`,
`dml/pliv.py`, `dml/double_ml.py`, `forest/forest_tools.py`,
`diagnostics/sensemakr.py` and `statspai/__init__.py`, which Track A
modules 03, 08, 13, 15, 22, 24, 25, 26, 27, 53, 65, 66, 71 and
original-data module 08 execute.

Effect on the paper: module 22 only, and only in the agreement columns.
The StatsPAI standard error of the treatment coefficient moves from
587.0203418 to 587.0203721 (R: 587.0203721), so its relative difference
from R falls from 5.2e-08 to 2.1e-15; `t_treat`, `rv_q` and `rv_qa` move
in the eighth digit toward R for the same reason. The point estimate, the
PASS verdict and the registered tolerance (1e-6) are unchanged. Modules
08, 13, 24 and 71 were rerun and their result files are byte-identical
(regressor nuisances and the forest paths are untouched); for the others
only the recorded source hashes move.

- `tests/r_parity/results/22_sensemakr_py.json`
- `tests/r_parity/results/parity_table.md`
- `tests/r_parity/results/parity_table.tex`
- `tests/r_parity/results/parity_table_3way.md`
- `tests/r_parity/results/parity_table_3way.tex`
- `tests/r_parity/TIER_A_FIXTURE_LOCK.json`

### 2026-10-04 — call traces re-recorded after `sp.rdrobust`'s density check moved to `sp.rddensity`

Commit `ba15e1b9`. The manipulation check that `sp.rdrobust` runs beside
the estimate now calls `sp.rddensity` instead of McCrary's binned test,
and the violation message in `core/_agent_summary.py` names the test.
`rd/rdrobust.py` is on the path of Track A modules 06 and 89 and of the
original-data module 05; `core/_agent_summary.py` on that of module 14.

Effect on the paper: none. The four modules were re-run and their result
files are byte-identical; the check sits beside the estimate and does not
enter it. Only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the `sp.rddensity` binomial windows and `sp.rdmc` inference

Commits `7082264e` and `276b2e33`. `sp.rddensity` gained options for the
windows of its binomial tests (`bino_w=` and four more), and
`sp.rdmc(cutoff_var=)` now reports robust bias-corrected intervals. The
edits are in `diagnostics/rddensity.py` and `rd/rdmulti.py`, which
Track A modules 09 and 89 execute, so their source hashes in the trace
changed. The default binomial table, the density test and `sp.rdms`
are untouched.

Effect on the paper: none. Modules 09 and 89 were re-run and their
result files are byte-identical; only the recorded source hashes move.
`sp.rdmc`, `sp.rdrandinf` and `sp.rdwinselect`, whose numbers did
change (see CHANGELOG, "Correctness: local randomization and multi-cutoff
RD"), have no Track A module. In the manuscript they appear only in the
function inventory, whose one-line descriptions are unchanged.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded after two warning texts changed

Commit `98538177`. The unbalanced-panel warning and docstring of
`sp.callaway_santanna` now state what `allow_unbalanced_panel=True`
assumes (the result records it in `model_info`), and the `sp.rdrobust`
warning for a discrete running variable quotes the mass-points study.
The edits are in `did/callaway_santanna.py` and `rd/rdrobust.py`, which
Track A modules 04, 06, 79, 89 and the original-data modules 02 and 05
execute, so their source hashes in the traces changed.

Effect on the paper: none. The six modules were run on the source
before and after the change and their result files are byte-identical;
only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded after the weight diagnostic reached the fixed-effects and count entry points

Commit `7579f977`. `sp.panel`, `sp.hdfe_ols`, `sp.feols`, `sp.poisson` and
`sp.ppmlhdfe` now record the Kish effective sizes of a weighted fit and
warn when the weights are dispersed. The edits are in
`core/_agent_summary.py`, `panel/panel_reg.py`, `panel/feols.py`,
`regression/count.py` and `fixest/wrapper.py`, which Track A modules 14,
35, 37, 42, 47, 58, 67 and 69 execute, so their source hashes in the
trace changed.

Effect on the paper: none. The eight modules were run on the source
before and after the change and their result files are byte-identical;
only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded for the `sp.rdrobust` refusal paths

Commit `4b3a65d4`. `sp.rdrobust(masspoints='off')` now raises
`NumericalInstability` where it used to crash on a complex number or
return a NaN interval, warns when it falls back to the legacy selector,
and warns when clusters coincide with the support points of the running
variable. The edits are in `rd/rdrobust.py` and `rd/_cct_bandwidth.py`,
which Track A modules 06, 88, 89 and the original-data module 05 execute,
so their source hashes in the traces changed.

Effect on the paper: none. The four modules were run on the source
before and after the change and their result files are byte-identical;
only the recorded source hashes move.

- `tests/r_parity/results/_implementation_trace.json`
- `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — call traces re-recorded after `sp.regress` gained the weight diagnostic

- **Commits.** `712d3bf2`.
- **Reason.** `sp.regress` now records `model_info['n_effective_weights']`
  on a weighted fit and warns when the weights are dispersed and the
  variance is the classical one, or when HC0 / HC1 / HC2 rest on a small
  Kish effective sample. The edits are in `src/statspai/regression/ols.py`
  and `src/statspai/core/_agent_summary.py`, on the estimation path of
  Track A modules 01, 14, 35, 51, 53, 54, 55, 56 and 69 and of
  original-data modules 01, 04, 04b and 09.
- **Effect on the paper.** None. No Track A module is weighted; a field
  may be added and a warning raised on weighted fits, and nothing is
  computed differently. The Python result files of modules 01 and 14 are
  byte-identical on the parent tree and on the changed tree. Only
  `exercised_sources` digests and `seconds` changed in the traces.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after the effective-cluster diagnostic reached the fixed-effects entry points

- **Commits.** `f030cf55`.
- **Reason.** `sp.panel`, `sp.hdfe_ols` and `sp.feols` now record
  `n_clusters_effective` and warn when many clusters are few in effect,
  through a helper shared with `sp.regress`. The edits are in
  `src/statspai/panel/panel_reg.py`, `src/statspai/panel/feols.py`,
  `src/statspai/fixest/wrapper.py`, `src/statspai/regression/ols.py` and
  `src/statspai/core/_agent_summary.py`, on the estimation path of Track A
  modules 01, 14, 35, 51, 53, 54, 55, 56, 67 and 69 and of original-data
  modules 01, 04, 04b and 09. Original-data module 08 was re-traced in
  the same commit: its record was stale against `src/statspai/__init__.py`
  on the main it was rebased onto.
- **Effect on the paper.** None. A field is added and a warning may be
  raised; nothing is computed differently. The Python result files of
  modules 14, 35, 67 and 69 are byte-identical on the parent tree and on
  the changed tree. Only `exercised_sources` digests and `seconds`
  changed in the traces.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-04 — release 1.38.0: parity tables carry the new version string

- **Commits.** `58748eb1` (release 1.38.0) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** None while the manuscript stays anchored where
  it is. If it is re-anchored to this release, the appendix parity tables
  read "StatsPAI 1.38.0". The registry count stays at 1,294.

### 2026-10-04 — release 1.37.0: parity tables carry the new version string

- **Commits.** `db5fc782` (release 1.37.0) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** None while the manuscript stays anchored where
  it is. If it is re-anchored to this release, the appendix parity tables
  read "StatsPAI 1.37.0", the registry count is 1,294, and Python 3.10 is
  the oldest supported version.

### 2026-10-03 — call traces re-recorded after `sp.label_values` and `sp.decode` were exported

- **Commits.** `dbe20e11`.
- **Reason.** Two label helpers were added to the public namespace, so
  `src/statspai/__init__.py` changed. That file is on the recorded
  estimation path of Track A modules 03, 13, 15, 24, 25, 26, 27, 53, 65
  and 66. The rest of the commit is label handling in `sp.read_data`,
  `sp.tab`, `sp.describe`, `sp.regtable(labels=)` and `sp.stata`, none of
  which a parity module calls.
- **Effect on the paper.** None. No estimator was touched and no Python
  result file changed; only the `exercised_sources` digest of
  `__init__.py` and `seconds` changed in the trace. The registered
  function count goes from 1,292 to 1,294, which matters only when the
  manuscript is next re-anchored.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after `sp.regress` gained the few-treated-clusters diagnostic

- **Commits.** `6c2d7e3a`.
- **Reason.** `sp.regress` now warns when a cluster-level 0/1 regressor has
  few clusters on one side and records `model_info['few_treated_clusters']`.
  The edits are in `src/statspai/regression/ols.py` and
  `src/statspai/core/_agent_summary.py`, on the estimation path of Track A
  modules 01, 14, 35, 51, 53, 54, 55, 56 and 69 and of original-data
  modules 01, 04, 04b and 09.
- **Effect on the paper.** None. A field may be added and a warning
  raised; nothing is computed differently. The Python result files of
  modules 01 and 14 are byte-identical on the parent tree and on the
  changed tree. Only `exercised_sources` digests and `seconds` changed in
  the traces.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 13 and 24 after the forest docstring splice was made version-independent

- **Commits.** `6ae1fd6b` re-recorded
  `tests/r_parity/results/_implementation_trace.json` for Track A modules
  13 and 24.
- **Reason.** The traces bind SHA-256 digests of every source file on each
  module's estimation path, and both modules execute
  `src/statspai/forest/regression_forests.py`. The same commit changed how
  that file splices the shared forest options into four docstrings at
  import (clean the docstring, then substitute), because Python 3.13 strips
  docstring indentation at compile time and the old indentation-sensitive
  match never fired there. No estimation code changed and no committed
  result file was regenerated or edited.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed for the two modules; the native / port / third-party
  census is unchanged. No estimate, standard error or table cell is read
  from this file.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after CR2 / CR3 / two-way clustering took the t(G - 1) reference

- **Commits.** `9a922b3d`.
- **Reason.** `sp.regress`, `sp.feols` and `sp.ivreg` with `vce='cr2'` /
  `'cr3'`, and `sp.regress(cluster=[a, b])`, now take p-values and
  intervals from t(G - 1) instead of the normal. The edits are in
  `src/statspai/regression/ols.py`, `src/statspai/regression/iv.py`,
  `src/statspai/fixest/wrapper.py`, `src/statspai/inference/jackknife.py`
  and `src/statspai/core/_agent_summary.py`, on the estimation path of
  Track A modules 01, 02, 14, 35, 51, 53, 54, 55, 56, 59 and 67 and of
  original-data modules 01, 04, 04b and 09.
- **Effect on the paper.** None on the tables. The Track A rows compare
  estimates and standard errors, which this change does not touch, and
  modules 53, 54 and 56 call the standalone helpers (`sp.cr2_se`,
  `sp.twoway_cluster`, `sp.multiway_cluster_vcov`), not the changed
  branches: the Python result files of 02, 14, 53, 54, 56, 59 and 67 are
  byte-identical on the parent tree and on the changed tree. Only
  `exercised_sources` digests and `seconds` changed in the traces. If the
  text states the reference distribution of the CR2 / CR3 options, it is
  now t(G - 1).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after `sp.regress` gained the effective-cluster diagnostic

- **Commits.** `99f78beb`.
- **Reason.** `sp.regress` now records `model_info['n_clusters_effective']`
  and warns when 30 or more clusters are fewer than 30 in effect. The
  edits are in `src/statspai/regression/ols.py` and
  `src/statspai/core/_agent_summary.py`, on the estimation path of Track A
  modules 01, 14, 35, 51, 53, 54, 55, 56 and 69 and of original-data
  modules 01, 04, 04b and 09.
- **Effect on the paper.** None. A field is added and a warning may be
  raised; no estimate or standard error is computed differently. The
  Python result files of the clustered modules 14, 53 and 54 are
  byte-identical on the parent tree and on the changed tree. Only
  `exercised_sources` digests and `seconds` changed in the traces.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after the few-cluster hint was reworded

- **Commits.** `e44aa48e`.
- **Reason.** The recovery hint attached to the few-cluster warning was
  replaced by one shared sentence
  (`statspai.core._agent_summary.FEW_CLUSTERS_HINT`). The edits are in
  `src/statspai/core/_agent_summary.py`, `src/statspai/regression/ols.py`
  and `src/statspai/panel/panel_reg.py`, on the estimation path of Track A
  modules 01, 14, 35, 51, 53, 54, 55, 56 and 69 and of original-data
  modules 01, 04, 04b and 09. Only the text of a warning changed.
- **Effect on the paper.** None on any number: only `exercised_sources`
  digests and `seconds` changed in the traces. One thing to weigh at the
  next re-anchor: the new size study (`tests/reliability/`, not a frozen
  artifact) shows the wild cluster bootstrap over-rejecting with one
  dominant cluster and under-rejecting with two treated clusters. If the
  manuscript recommends it for few clusters without that qualification,
  the sentence should be qualified.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 17 and 38 after a docstring edit in the ETWFE module

- **Commits.** `794e1a9f`.
- **Reason.** The `controls` entry of the `sp.etwfe` docstring was
  rewritten in `src/statspai/did/wooldridge_did.py`, which is on the
  estimation path of modules 17 and 38. No executable line changed.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed in the trace.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 37, 42, 47 and 58 after weighted Poisson fits took their weights into the covariance

- **Commits.** `ddceeee9`.
- **Reason.** `sp.poisson(weights=)` and `sp.ppmlhdfe(weights=)` computed
  the covariance without the weights, and three count models reported an
  unweighted log-likelihood for a weighted fit. The edit is in
  `src/statspai/regression/count.py`, on the estimation path of the four
  modules.
- **Effect on the paper.** None. The four modules are unweighted, and
  without weights the new code multiplies by nothing: their Python
  result files are byte-identical on the parent tree and on the fixed
  tree. Only `exercised_sources` digests and `seconds` changed in the
  trace. The parity index gains one reference test
  (`tests/reference_parity/test_count_weights_stata_parity.py`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `src/statspai/_parity_index.json`

### 2026-10-03 — call traces re-recorded for 37, 42, 47 and 58 after the count models learned to drop missing rows

- **Commits.** `3917e872`.
- **Reason.** `sp.poisson`, `sp.nbreg` and `sp.ppmlhdfe` failed when a
  plain-column formula met a missing value; they now drop those rows.
  The edit is in `src/statspai/regression/count.py`, on the estimation
  path of the four modules.
- **Effect on the paper.** None. The four fixtures have no missing
  values, so the new step returns the frame untouched: the Python result
  files are byte-identical on the parent tree and on the fixed tree.
  Only `exercised_sources` digests and `seconds` changed in the trace.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 03, 15, 47 and 53 after `sp.fast.feols_jax` took the convention of `sp.fast.feols`

- **Commits.** `ca410f77`.
- **Reason.** The JAX backend kept the pre-1.31 small-sample count while
  reporting `ssc='fixest'`. The edit is in
  `src/statspai/fast/jax_feols.py` and `src/statspai/fast/_jax_fallback.py`,
  which the four modules import through `statspai.fast`.
- **Effect on the paper.** None on any table. No Track A module calls
  `sp.fast.feols_jax`; only `exercised_sources` digests and `seconds`
  changed in the trace. The Track C GPU benchmark times this function,
  and the change adds integer arithmetic outside the timed linear
  algebra, so the timings are not affected in any measurable way; they
  are re-measured at the next re-anchor in any case. One sentence to
  check at that point: wherever the manuscript says the JAX backend
  returns the same result as the NumPy backend, that was true of the
  coefficients and is now also true of the standard errors.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after docstring repairs that unblock the strict docs build

- **Commits.** `fd6c43ca` re-recorded
  `tests/r_parity/results/_implementation_trace.json` for Track A modules
  01, 02, 13, 14, 24, 35, 51, 53, 54, 55, 56 and 59, and
  `tests/orig_parity/results/_implementation_trace.json` for original-data
  modules 01, 04, 04b and 09.
- **Reason.** The traces bind SHA-256 digests of every source file on each
  module's estimation path. The same commit edited four of those files,
  docstrings only: `src/statspai/regression/ols.py` and
  `src/statspai/regression/iv.py` (a References separator and the
  `**kwargs` options of `IVRegression.fit`),
  `src/statspai/forest/regression_forests.py` (indentation of the
  placeholder that is replaced by the shared forest options at import; the
  runtime docstrings were hashed before and after and are identical) and
  `src/statspai/forest/causal_forest.py` (a Returns section).
  `mkdocs build --strict` had aborted on these since 2026-09-28. No
  executable line changed and no committed result file was regenerated or
  edited.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed; every other field of all 16 modules was compared with
  the previous file and is identical, so the native / port / third-party
  census and the original-data ledger's provenance marks are unchanged. No
  estimate, standard error or table cell is read from these files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 03, 15, 47 and 53 after `sp.fast.fepois` took fixest's small-sample factors

- **Commits.** `01260115`.
- **Reason.** `sp.fast.fepois` gained `ssc=` (default `'fixest'`) and a
  weighted `hc1` score. The edit is in `src/statspai/fast/fepois.py`,
  which the four modules import through `statspai.fast`.
- **Effect on the paper.** None. No Track A module calls
  `sp.fast.fepois`: the Poisson modules (37, 47, 58, 67) run
  `sp.ppmlhdfe` / `sp.fepois`, which were not touched. The Python result
  files of 03, 15, 17, 37, 47, 58 and 67 are byte-identical on the parent
  tree and on the fixed tree. Only `exercised_sources` digests and
  `seconds` changed in the trace.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `src/statspai/_parity_index.json`

### 2026-10-03 — call traces re-recorded for 03, 15, 47 and 53 after the `sp.fast.feols` degrees-of-freedom count for one absorbed dimension

- **Commits.** `67492d1a`.
- **Reason.** With a single absorbed dimension `sp.fast.feols` used
  `n - p - (G - 1)` residual degrees of freedom where fixest and reghdfe
  use `n - p - G`; two clustered layouts were off by one as well. The fix
  is in `src/statspai/fast/feols.py`, on the estimation path of modules
  03, 15, 47 and 53.
- **Effect on the paper.** None. Those modules fit two-way models
  clustered on a key that nests one of the effects (or are unclustered
  two-way fits), the case that was already exact: their Python result
  files are byte-identical on the parent tree and on the fixed tree.
  Only `exercised_sources` digests and `seconds` changed in the trace.
  The Track C timings of `01_hdfe` were already marked stale by earlier
  commits and are re-measured at the next re-anchor; the fix changes one
  integer subtraction. The parity index gains one reference test
  (`tests/reference_parity/test_fast_feols_weights_fixest_parity.py`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `src/statspai/_parity_index.json`

### 2026-10-03 — call traces re-recorded after `sp.regress` kept its weights and its fitted rows under CR2, CR3 and two-way clustering

- **Commits.** `87a128eb`.
- **Reason.** `sp.regress(weights=)` with `vce='cr2'` / `'cr3'` or two-way
  clustering refit the model without the weights; with missing values in
  a formula variable those variances read their cluster keys from the
  first `n` rows; `sp.twoway_cluster` and `sp.cr2_se` ignored the weights
  of a weighted fit. The fixes are in `src/statspai/regression/ols.py`,
  `src/statspai/inference/jackknife.py` and
  `src/statspai/inference/twoway_cluster.py`, which are on the estimation
  path of Track A modules 01, 14, 51, 53, 54, 55 and 56 and of
  original-data modules 01, 04, 04b and 09.
- **Effect on the paper.** None. Every one of those modules is unweighted
  and runs on data with no missing values, which the fixes leave
  untouched. Checked by running modules 01, 54 and 56 on the source tree
  of the parent commit and on the fixed tree: the three result files are
  byte-identical between the two. (Both differ from the committed files
  in trailing floating-point digits, at most 3e-13, because this session
  runs with `STATSPAI_SKIP_RUST=1` under load; the committed files were
  kept.) Only `exercised_sources` digests and `seconds` changed in the
  traces; no implementation classification moved. The parity index gains
  one reference test
  (`tests/reference_parity/test_regress_vce_weights_stata_parity.py`),
  which is not a Track A module and feeds no table.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`
  - `src/statspai/_parity_index.json`
  - `docs/parity.md`

### 2026-10-03 — release 1.36.0: parity tables carry the new version string

- **Commits.** `0f70d29c` (release 1.36.0) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** None while the manuscript stays anchored where
  it is. If it is re-anchored to this release, the appendix parity tables
  read "StatsPAI 1.36.0", and everything the paper quotes from the registry
  and the parity index should be regenerated against the `v1.36.0` tag at
  the same time (the entries below list what moved since 1.35.0).

### 2026-10-03 — call traces re-recorded for 35 and 69 after `sp.panel` named its default small-sample convention

- **Commits.** `eda627e1`.
- **Reason.** `sp.panel` without `ssc=` now writes
  `model_info['ssc'] = 'linearmodels'` and a description of that scaling,
  and the few-cluster warning mentions `ssc='stata'`. The edit is in
  `src/statspai/panel/panel_reg.py`, which is on the estimation path of
  modules 35 and 69, so their source digests moved.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed; the classification of both modules is the same
  (35 third-party `linearmodels`, 69 native). No estimate or standard
  error moves: the default scaling is untouched, and
  `tests/reference_parity/test_panel_ssc_stata_parity.py` still holds
  the default to `linearmodels` at 1e-14.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — minimum-norm rule for non-unique synthetic-control weights; `52_scm_unique` recovers its truth to 2e-14

- **Commits.** `54ad1443`. The traces, parity index and fixture lock are
  re-recorded in the commit that follows it, which carries this entry.
- **Reason.** When the simplex least-squares minimiser is not unique (a
  treated unit inside the donors' hull, a predictor weighted to zero) the
  weights were SLSQP's choice from a uniform start. They are now the
  minimum-norm weights among the minimisers, computed exactly. The rule
  does not depend on the solver or the starting point, and a nested fit
  with its placebo fits takes seconds.
- **Effect on the paper.** `52_scm_unique` improves: `avg_post_gap`
  2.000000155 to 2.000000000 (relative gap to R 7.78e-08 to 4.14e-10) and
  the three donor weights to within 2e-14 of 0.5, 0.3, 0.2 (gaps to R
  1e-7 to 1e-9, which is R's own error). `07_scm`, `18_augsynth`,
  `19_gsynth`, `12_sdid` and original-data `03_basque_original` were
  rerun and their result files are byte-identical: their solutions are
  unique. No verdict, tier or implementation classification changes.
  Track C synthetic-control timings are now far out of date in the
  favourable direction; they are re-measured at the next re-anchor.
- **Paths.**
  - `tests/r_parity/results/52_scm_unique_py.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table.tex`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/results/parity_table_3way.tex`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after `sp.write_data` was exported and the matched-frame helper changed

- **Commits.** `bed80f3f` re-recorded
  `tests/r_parity/results/_implementation_trace.json` for Track A modules
  03, 13, 15, 24, 25, 26, 27, 53, 65 and 66, and
  `tests/orig_parity/results/_implementation_trace.json` for original-data
  modules 04, 04b and 08.
- **Reason.** The traces bind SHA-256 digests of every source file on each
  module's estimation path. `a6465c4a` edited `src/statspai/__init__.py`
  (the `sp.write_data` export), which those Track A modules and module 08
  execute lazily while estimating, and earlier commits edited
  `src/statspai/matching/_matched_frame.py`, which original-data modules 04
  and 04b execute (`824d202d`, `7fc9ce22`, `ac01445f`; that helper builds
  the matched-sample frame, and the Track A side of those commits was
  re-traced in the entry below). This entry re-records traces only: no
  committed result file was regenerated or edited.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed; every other field of all 13 modules was compared with
  the previous file and is identical, so the native / port / third-party
  census and the original-data ledger's provenance marks are unchanged. No
  estimate, standard error or table cell is read from these files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 11, 35 and 69 after a matching bookkeeping fix and the panel vce check

- **Commits.** `ac01445f` (module 11), `25aeed06` (modules 35 and 69).
- **Reason.** `matching/_matched_frame.py` no longer divides by a zero
  weight sum when a bootstrap replicate of local linear matching has
  cancelling weights (module 11's estimation path), and
  `panel/panel_reg.py` now raises on a `vce=` value it does not
  implement instead of ignoring it (modules 35 and 69). The traces bind
  the SHA-256 of every source file executed.
- **Effect on the paper.** None. Only the `exercised_sources` digests of
  those three modules changed; their committed `_py.json` results are
  byte-identical. Module 11 runs nearest-neighbour matching, not the
  local linear path that was fixed; modules 35 and 69 pass no `vce=`.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — exact inner solver for synthetic-control weights; three Track A result files and the Basque original-data file move below 4e-7

- **Commits.** `95d87260` (`sp.didregress`, optimized SDID covariates,
  `rddensity` conventional statistic and binomial tests, exact inner
  solver for simplex weights). The traces, schemas, parity index and
  fixture lock are re-recorded in the commit that follows it, which
  carries this entry.
- **Reason.** `synth/_core.py::solve_simplex_weights` solved problems
  with fewer rows than donors by SLSQP, which stops about 1e-5 short in
  the weights. When the minimiser is certified unique (optimality and
  affine independence of the tight donors) it is now solved exactly. Where
  the certificate fails SLSQP's choice is kept. A nested fit on the
  Proposition 99 data goes from 55 seconds to 2.
- **Effect on the paper.** Inside every registered budget; no verdict
  changes. `07_scm`: five values move, the largest
  `weight_Asturias` 0.01489185 to 0.01489185 (3.7e-7 relative; the
  printed table cell goes from 0.0148919 to 0.0148918), and the recorded
  winning start of the outer search is `regression` again in place of
  `dirichlet_3` (the two starts tie; the optimum is the same).
  `18_augsynth`:
  three values, at most 1.0e-7 (`att_augmented` relative gap to R 7.91e-06
  to 7.92e-06, `pre_rmspe` 3e-06 to 3.01e-06). `52_scm_unique`: the
  standard error of `avg_post_gap`, 1.1e-7. Original-data
  `03_basque_original`: estimate -0.89458856 to -0.89458854. Modules 12,
  19 and 09 were rerun and did not change. No implementation
  classification moved in either ledger. Track C synthetic-control timings
  were not re-measured; `scripts/trace_perf_path.py --check` decides at
  the next re-anchor whether they are stale.
- **Paths.**
  - `tests/r_parity/results/07_scm_py.json`
  - `tests/r_parity/results/18_augsynth_py.json`
  - `tests/r_parity/results/52_scm_unique_py.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/results/parity_table_3way.tex`
  - `tests/orig_parity/results/03_basque_original_py.json`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded for 16, 73 and 84 after the bootstrap seed was recorded

- **Commit.** `3d9e056d`.
- **Reason.** `sp.did_imputation` and `sp.gardner_did` now write
  `boot_seed` into `model_info` when `vce='bootstrap'`, so the result
  card can report whether the bootstrap SE is reproducible. Both source
  files are on the estimation paths of Track A modules 16, 73 and 84.
- **Effect on the paper.** None. Only the `exercised_sources` digests of
  those three modules changed; the committed `_py.json` results are
  byte-identical (the modules use the analytic variance, where the new
  field is not written).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after per-coefficient degrees of freedom in the result class

- **Commits.** `bcc3070c` (`sp.difference_in_means`, exact CR2 degrees of
  freedom, `sp.panel` formula terms, Bacon summary). The traces, schemas and
  parity index are re-recorded in the commit that follows it, which carries
  this entry.
- **Reason.** `core/results.py` gained `_coefficient_df`, which reads
  `data_info['df_by_coefficient']` when a fit records it. Only `sp.cr2_se`
  records it. The file is on the estimation path of every module, so both
  ledgers were re-recorded in full. `panel/panel_reg.py`, `did/bacon.py`,
  `iv/iv_diag.py` and `registry.py` changed in the same commit.
- **Effect on the paper.** None. Module `53_cr2` is the only one that
  calls `sp.cr2_se`, and it records the standard errors, which did not
  change (the degrees of freedom did). Without the new key the result class
  returns the scalar it returned before. No result file changed. No implementation classification moved
  in either ledger. The registered-function count goes from 1,280 to 1,281,
  which the paper does not track between anchors.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-03 — parity index: `sp.ttest` gained a reference test

- **Commit.** `606fb6c3` added
  `tests/reference_parity/test_ttest_known_values.py` and regenerated
  `docs/parity.md`.
- **Reason.** The stability audit found `sp.ttest` stable with no
  parity-test evidence.
- **Effect on the paper.** None on any table. In `docs/parity.md`, `ttest`
  moves from "unverified" to analytical-only (T1).
- **Paths.**
  - `docs/parity.md`

### 2026-10-02 — pre-trend power integrated to 1e-7; traces re-recorded for 02, 10, 21, 59, 76 and the original-data ledger

- **Commits.** `826295bf` (Montiel Olea-Pflueger critical values, robust
  endogeneity tests, pre-trend power from coefficients and a covariance).
  The traces, schemas and parity index were re-recorded in the commit that follows it, which carries this entry.
- **Reason.** `sp.pretrends_power` integrates a multivariate-normal
  rectangle probability. SciPy's default absolute tolerance of 1e-5 left
  the power moving in the fifth digit between two calls, so the tolerance
  is now 1e-7 and the slope solver stops at that precision. The same commit
  touched `regression/iv.py` (diagnostics only) and `did/honest_did.py`
  (a refactor of the moments path), which are on the estimation path of
  the other four modules. The original-data ledger had also gone stale
  from the earlier textbook commits (`rd/_cct_bandwidth.py`,
  `rd/rdrobust.py`, `synth/_core.py`, `synth/scm.py`, `regression/iv.py`)
  and from `regression/ols.py` and `core/_vcov_spec.py` on `main`.
- **Effect on the paper.** Module 76 only, inside its registered 1e-3
  budget. The six Python values moved by 1e-6 to 1e-5 in absolute terms,
  for example `power_slope_0p02` 0.33164959 to 0.33164981 and
  `slope_for_power_0p5` 0.02791382 to 0.02791357. The headline relative
  gap to R `pretrends` goes from 3.98e-05 to 4.05e-05 at slope 0.02 and
  from 1.05e-05 to 8.57e-06 at slope 0.05. The verdict and the gap note
  ("rel < 1e-4") stand. Modules 02, 10, 21 and 59 were rerun and their
  result files are unchanged in the committed digits (02 and 59 differ
  from the committed files at 1e-12, which predates this commit and is
  below the 1e-9 reproducibility tolerance, so the files were left alone).
  No implementation classification moved in either ledger.
- **Paths.**
  - `tests/r_parity/results/76_pretrends_py.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-02 — call traces re-recorded for 17, 38 and 85 after the scope-map fields were added

- **Commit.** `d3f1b828`.
- **Reason.** `sp.event_study` and the linear `sp.etwfe` now record the
  configuration that `sp.validation_scope` reads (`n_adoption_dates`,
  `covariates`, `intensity`, `absorb`, `cluster_level`; `family`) in
  `model_info`. `did/event_study.py` and `did/wooldridge_did.py` are on the
  estimation paths of Track A modules 17, 38 and 85, whose traces bind the
  SHA-256 of every source file executed.
- **Effect on the paper.** None. Only the `exercised_sources` digests of
  those three modules changed; their committed `_py.json` results are
  byte-identical and no module's implementation classification changed.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-03 — call traces re-recorded after `sp.regress(robust='ewc')`

- **Commit.** `503cb21e` added the `ewc` covariance kind to
  `src/statspai/regression/ols.py` and `src/statspai/core/_vcov_spec.py`,
  a rescaled F for it in `src/statspai/postestimation/hypothesis.py`, and
  re-recorded `tests/r_parity/results/_implementation_trace.json` for
  modules 01 02 07 14 18 37 41 42 43 44 45 46 47 48 49 51 52 53 54 55 56 57
  58 59 61 62 63 64 67.
- **Reason.** Those 29 modules execute one of the two edited files. The
  new kind is opt-in and no existing branch changed:
  `verify_reproduce_py.py --no-report 01_ols 02_iv 14_ols_cluster 51_newey
  53_cr2 54_twoway_cluster 55_hc2_hc3 56_multiway_cluster` reports 8
  reproduce, 0 drift.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed; no module's implementation classification moved.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-02 — call traces re-recorded after the design-based textbook pass; three estimator fixes on Track A paths

- **Commits.** `fd2d0af3` (`sp.etwfe` with no untreated comparison),
  `58169a79` (RD pilot-bandwidth quantile), `d4a47bac`
  (`sp.match(method='nnmatch')`), `55654839`
  (`sp.mediate(inference='robust')`), `27ebf052` (LIML `kappa`, factor
  terms in IV formulas), `47d229a1` (`sp.synth(v_method='regression')`,
  `sp.rdrobust(scalepar=)`). The regenerated artefacts are in `6fdcfa8f`,
  `47d229a1` and `740fa734`: `docs/parity.md`, and
  `tests/r_parity/results/_implementation_trace.json` re-recorded for
  modules 02 03 06 07 13 15 17 18 19 24 25 26 27 35 36 38 52 53 59 65 66
  69 88 89.
- **Reason.** The traces bind the digests of every source file on each
  module's estimation path, and these commits edited
  `did/wooldridge_did.py`, `rd/_cct_bandwidth.py`, `rd/rdrobust.py`,
  `regression/iv.py`, `panel/panel_reg.py`, `mediation/mediate.py`,
  `synth/_core.py`, `synth/scm.py` and `matching/__init__.py`.
- **Effect on the paper.** No Track A number moved. Three of the commits
  are correctness fixes on code the Track A modules execute, so each was
  re-run and compared with the committed Python result:
  - `06_rd`, `88_rdbwselect`, `89_rdms`: byte-identical. The fix changes
    the pilot bandwidth only when `IQR / 1.349` of the running variable is
    below its standard deviation, which is not the case on those fixtures.
    The claim "data-driven bandwidths match `rdrobust`" was false on such
    data before the fix (5.6e-5 on the Lee House data) and holds to 6e-12
    now; if the paper states the scope of the RD rows, heavy-tailed
    running variables are now covered by
    `tests/reference_parity/test_rdbwselect_iqr_pilot.py`.
  - `59_liml`: moves at 1e-15 (not re-committed). The fixture's `kappa`
    is 1.00057, far enough from 1 that the old computation was accurate;
    the error grew as `kappa` approached 1.
  - `17_etwfe`, `38_drdid`: unchanged. The trim applies only when no unit
    is untreated in some period, and both fixtures have never-treated
    units.
  In `docs/parity.md` the "analytical-only (T1)" count goes from 325 to
  327 and "unverified" from 533 to 531: `from_stata` and `stata` are now
  credited to `tests/reference_parity/test_iv_stata_commands_parity.py`,
  which runs Stata command lines through them and compares with Stata 18
  output. Only `exercised_sources` digests and `seconds` changed in the
  trace; no module's implementation classification moved.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `docs/parity.md`

### 2026-10-03 — call traces re-recorded after `sp.ardl` was exported; two functions gained external-replication evidence

- **Commit.** `d6d3eb0b` added `sp.ardl` and `ARDLResult` (new exports in
  `src/statspai/__init__.py`), the `break_vars=` / `vce=` arguments of
  `sp.structural_break`, regenerated `docs/parity.md` and re-recorded
  `tests/r_parity/results/_implementation_trace.json` for modules 03 13 15
  24 25 26 27 53 65 66.
- **Reason.** Those ten modules import the package root, whose digest the
  trace binds. `src/statspai/timeseries/structural_break.py` is not on any
  Track A estimation path, and its new arguments default to the previous
  behaviour.
- **Effect on the paper.** None on any table. Only `exercised_sources`
  digests and `seconds` changed in the trace. In `docs/parity.md` the
  registered total is 1,280 (two new names) and the
  "external-replication (published numbers)" count goes from 3 to 5:
  `ardl` and `unitroot` are credited to
  `tests/external_parity/test_stock_watson_4e_ch15.py`, which checks them
  against the RATS output shipped with a textbook's replication files.
  That test needs the files on disk and is skipped otherwise, so it is
  evidence a reader can rerun only after downloading them. If the paper
  quotes the external-replication count, say what kind of evidence these
  two rows are.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `docs/parity.md`

### 2026-10-02 — call traces re-recorded after `sp.regress` gained HAC options; parity-index denominators moved

- **Commits.** `aad512aa` added `sp.ttest` (new exports in
  `src/statspai/__init__.py`) and regenerated `docs/parity.md`. `dd753900`
  added `sp.unitroot`, the `hac_lags=` / `hac_small=` arguments of
  `sp.regress` (`src/statspai/regression/ols.py`), regenerated
  `docs/parity.md` again and re-recorded
  `tests/r_parity/results/_implementation_trace.json` for modules 01 03 13
  14 15 24 25 26 27 51 53 54 55 56 65 66.
- **Reason.** The traces bind SHA-256 digests of every source file on each
  module's estimation path; `regression/ols.py` and the package root are on
  the path of those sixteen. `aad512aa` was pushed without re-recording, so
  the trace check was stale on `main` between the two commits. The new
  `sp.regress` arguments default to the previous behaviour:
  `verify_reproduce_py.py --no-report 01_ols 14_ols_cluster 51_newey 53_cr2
  54_twoway_cluster 55_hc2_hc3 56_multiway_cluster` reports 7 reproduce,
  0 drift.
- **Effect on the paper.** None on any table. Only `exercised_sources`
  digests and `seconds` changed in the trace; no module's implementation
  classification moved. In `docs/parity.md` the registered-function
  denominators grew by the four new names (`ttest`, `TTestResult`,
  `unitroot`, `UnitRootResult`), which carry no cross-language evidence
  row: "unverified" 529 to 533 and estimator callables 736 to 738. One
  thing worth knowing at the next re-anchor: module `51_newey` passes
  against Stata with `rel_se` 1e-2 because Stata's `newey` scales the
  covariance by `N/(N-K)`. `sp.regress(..., robust="hac", hac_lags=4,
  hac_small=True)` now reproduces that Stata golden to 3e-16
  (`tests/test_regress_hac_options.py`), so the row could be moved to the
  strict budget by adding a Stata-convention row to the module. That is a
  change to a frozen artifact and was not made here.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `docs/parity.md`
### 2026-10-02 — call traces re-recorded for 16, 73 and 84 after the scope-map fields were added

- **Commit.** `f88b6bee`.
- **Reason.** `sp.did_imputation` and `sp.gardner_did` now record the
  configuration that `sp.validation_scope` reads (`y0_fe`, `y0_covariates`,
  `horizon_requested`; `covariates`) in `model_info`. The traces bind the
  SHA-256 of every source file on each module's estimation path, and
  `did/did_imputation.py` and `did/gardner_2s.py` are on the paths of
  Track A modules 16, 73 and 84.
- **Effect on the paper.** None. Only the `exercised_sources` digests of
  those three modules changed; their committed `_py.json` results are
  byte-identical, so no estimate or standard error moved, and no module's
  implementation classification changed. The same commit regenerates the
  six option-level Stata fixtures under `tests/stata_parity/option_parity/`
  in double precision. Those are outside the Track A enumeration and are
  not read by any manuscript table.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-02 — release 1.35.0: parity tables carry the new version string

- **Commits.** `173f2bf8` (release 1.35.0) regenerated
  `tests/r_parity/results/parity_table.tex` and
  `tests/r_parity/results/parity_table_3way.tex` with
  `python tests/r_parity/compare.py`, and refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`.
- **Reason.** The tables print the StatsPAI version in their caption.
  One line changes in each; no row, estimate or standard error moves.
- **Effect on the paper.** The appendix parity tables read "StatsPAI
  1.35.0" once the manuscript is re-anchored to this release. Everything
  the paper quotes from the registry and the parity index should be
  regenerated against the `v1.35.0` tag at the same time (see the entries
  below for what moved since 1.34.2).

### 2026-10-02 — call traces re-recorded for 12 and 76 after the pre-trend and placebo fixes

- **Commits.** `3958ab1c` changed `src/statspai/did/pretrends.py`
  (`sp.sensitivity_rr`), `src/statspai/synth/sdid.py`
  (`sp.synthdid_placebo(kind='time')`) and `src/statspai/registry.py`,
  and re-recorded `tests/r_parity/results/_implementation_trace.json`
  for modules 12 and 76.
- **Reason.** Correctness fix to `sp.sensitivity_rr` and a new option on
  the SDID placebo helper. Neither is called by a Track A module:
  `verify_reproduce_py.py --no-report 12_sdid 76_pretrends` reports 2
  reproduce, 0 drift.
- **Effect on the paper.** None on any table. If the text describes
  `sp.sensitivity_rr` intervals or quotes a breakdown `Mbar` from it, the
  numbers move (wider intervals; see `MIGRATION.md`). The honest-DiD rows
  of the paper use `sp.honest_did`, which is unchanged.

### 2026-10-02 — call traces re-recorded after the example-data loaders were labelled as simulated

- **Commits.** `8b3c53ab` changed `src/statspai/synth/datasets.py` and
  `src/statspai/synth/sdid.py` (docstrings, and `df.attrs['simulated']`
  on the returned frames) and re-recorded
  `tests/r_parity/results/_implementation_trace.json` for modules 07,
  12, 18 and 19, whose entry scripts load those datasets.
- **Reason.** The loaders did not say their rows are simulated. No row
  of any dataset changed: the modules' dumped CSVs and committed results
  are byte-identical, and `verify_reproduce_py.py --no-report` reports 89
  reproduce, 0 drift.
- **Effect on the paper.** None on any number. One thing to check in the
  text: wherever the manuscript calls the module 12 input "California
  Proposition 99", it is the simulated replica
  (`sp.datasets.california_prop99(simulated=True)`), on which both
  StatsPAI and R give -17.9, not the published -15.60. On the real panel
  StatsPAI gives -15.6038279 against R `synthdid` 0.0.9's -15.6038279
  (`tests/reference_parity/test_oct2026_fourth_pass.py`).

### 2026-10-02 — call traces re-recorded twice during the known-truth hardening pass; parity-index denominators moved

- **Commits.** `e364393e` and `06fe553d` re-recorded
  `tests/r_parity/results/_implementation_trace.json` and
  `tests/orig_parity/results/_implementation_trace.json`. The sources
  that staled them: `badaed34` (loud fallbacks in 62 handlers, many on
  Track A estimation paths), `9d7c9854` (`src/statspai/rd/_cct_bandwidth.py`),
  `5adaff59` (`src/statspai/registry.py`), `6c56e516`
  (`src/statspai/dtr/q_learning.py`, a docstring note) and `c5577401`
  (`src/statspai/did/event_study.py`, `src/statspai/did/pretrends.py`).
- **Reason.** Correctness fixes found by probing estimators without
  parity evidence against simulated designs with a known answer. None of
  the fixed code paths is exercised with the options a Track A module
  uses: `python tests/r_parity/verify_reproduce_py.py --no-report`
  reports 89 reproduce, 0 drift, before and after.
- **Effect on the paper, traces.** None. Checked field by field against
  the version before `e364393e`: in 70 of 89 Track A modules and all 12
  original-data modules only `exercised_sources` digests and `seconds`
  differ; no `packages`, boundary-call package set, `rscript_launches` or
  `error` changed.
- **Effect on the paper, quoted counts.** `docs/parity.md` and
  `src/statspai/_parity_index.json` are not frozen artifacts, but the
  manuscript quotes their "honest denominators" through
  `\ParityEstimatorTotal` in `generated_claims.tex`. `27820fec` moved 84
  plots, bundled datasets, exporters, language-model helpers and
  catalogue listings out of the estimator denominator
  (`statspai._parity_taxonomy.NON_NUMERIC_CALLABLES`), and the new
  anchors in `tests/reference_parity/` added evidence. Estimator
  callables: 822 to 738. With any evidence: 572 to 694 (679 at
  `5630df21`, 694 after the anchors added later the same day). Cross-language:
  417, unchanged, so the cross-language share reads 56.5% where it read
  50.7%. Infrastructure: 131 to 215. All registered: 1274, unchanged.
  The next revision should regenerate `generated_claims.tex` and say in
  the text that the denominator was redefined, since the share rose
  without any new cross-language row.
- **Changed estimator defaults a reader could hit.** `sp.deepiv` now
  defaults to the paired-sample loss; `sp.event_study` drops event times
  with no treated observation; `sp.did_few_treated` reports the exact
  inversion interval. None is a Track A module configuration. Full list
  in `MIGRATION.md` under `oct2026-known-truth-fixes`.

### 2026-09-28 — call traces re-recorded after the agent-surface pass (W7–W10)

- **Commits.** `1e702f2` (result contract on every result class, MCP
  hardening, error taxonomy, discovery / schemas) changed files on Track A
  estimation paths — `src/statspai/core/results.py`,
  `src/statspai/_result_serialize.py`, `src/statspai/did/__init__.py`,
  `src/statspai/_input_validation.py` — which staled the traces;
  `71d0df1` re-recorded `tests/r_parity/results/_implementation_trace.json`
  and `tests/orig_parity/results/_implementation_trace.json`.
- **Reason.** No estimator's numbers changed: the edits add agent-facing
  methods (`to_dict(detail=)`, `violations`, `next_steps`, `result_card`),
  stop `next_steps()` printing, and raise `ColumnNotFound` (still a
  `MethodIncompatibility`) with a did-you-mean hint.
- **Effect on the paper.** None. Checked field by field against the
  previous commit: across all 89 Track A modules and the 12 original-data
  modules no `packages`, boundary-call package set, `rscript_launches` or
  `error` changed; only `exercised_sources` digests and `seconds` differ.
  Traced in a fresh `site-packages` venv (Python 3.11.15, pandas 2.3.3; a
  first pass under pandas 3.0.6 was discarded because the `50_xtabond`
  entry script references `pd.errors.SettingWithCopyWarning`, removed in
  pandas 3).

### 2026-09-28 — freeze paused: the paper has not been submitted yet

- **Reason.** The freeze was written at 1.32.0 on 2026-09-27 on the
  assumption that the JSS package had gone in. It has not; submission is
  planned for about two weeks later. A freeze is meant to start at
  submission, so `"active"` is set to `false` in
  `tests/jss_review_freeze.json` and both checks pass vacuously until then.
- **At submission.** Re-anchor to the release actually submitted and
  re-freeze with `python scripts/jss_review_freeze.py --write --release
  X.Y.Z`, which rewrites every hash and sets `"active": true` again. The
  entries below describe changes against the provisional 1.32.0 anchor;
  they stay as history and do not need to be carried into the new freeze.

### 2026-09-28 — call traces re-recorded after the routing commit changed `__init__.py`

- **Commits.** `e53532f1` (machine-readable estimator routing: exports
  `sp.route` / `sp.decision_guide` from `src/statspai/__init__.py`) went
  in without a re-trace; the ten Track A modules whose recorded path
  includes the package `__init__` (`03_hdfe`, `13_causal_forest`,
  `15_hdfe_cluster`, `24_coxph`, `25_lmm`, `26_glmm_logit`,
  `27_glmm_aghq`, `53_cr2`, `65_spatial`, `66_spatial_gmm`) were stale
  until the commit that carries this entry re-recorded
  `tests/r_parity/results/_implementation_trace.json`.
- **Reason.** Only the exported-name table of `src/statspai/__init__.py`
  changed (two new lazy exports); no estimator source moved. The
  original-data ledger (`tests/orig_parity/results/_implementation_trace.json`)
  does not bind `__init__.py` and was not stale.
- **Effect on the paper.** None. Checked field by field against `HEAD`
  with the same script as the entry below: across all 89 Track A modules
  and the 12 original-data modules no `packages`, boundary-call package
  set, `rscript_launches` or `error` changed; only the ten
  `exercised_sources` digests and `seconds` differ. Traced in the same
  `site-packages` venv (Python 3.11.15, pandas 2.3.3) as the entry below.

### 2026-09-28 — call traces re-recorded for sdid(treat=) and an es_inference docstring

- **Commits.** `80abadb7` (re-trace after `sp.sdid(treat=...)`, commit
  `46ec6369`) and `fb8c61ff` (re-trace of modules 10 and 21 after a
  docstring correction in `src/statspai/did/es_inference.py`). Neither
  recorded an entry here at the time; this one covers both.
- **Reason.** Both edited source files on Track A estimation paths, so the
  bound SHA-256 digests went stale. No estimator code path changed in
  `fb8c61ff`: the edit corrects `sp.uniform_bands`' stated Monte Carlo
  error (about 5e-3, not 1e-3).
- **Effect on the paper.** None. Checked field by field against
  `v1.32.0`: across all 89 Track A modules and the 12 original-data modules
  only `exercised_sources` digests and `seconds` differ, no classification
  moved, the 86 / 0 / 3 census and the original-data provenance marks are
  unchanged. No estimate, standard error or table cell is read from these
  files. (`80abadb7` also re-traced the original-data modules; the first
  version of this entry named only the Track A file, which the per-commit
  check added the same day caught.)
- **For the next revision (not frozen artifacts).** The same commits and
  `a089a6f0` add reference tests that pin the full joint event-study
  covariance behind `sp.event_study_vcov` against R `did`, `fixest`,
  `did2s`, `etwfe` and Stata `did_imputation`. The registry grade of
  `sp.event_study_vcov` moves from unverified to bit-exact, so
  `\ParityCrossLanguage` in `generated_claims.tex` (415 at 1.32.0) will
  read higher when regenerated; other functions added since 1.32.0 move
  the totals as well. The HonestDiD passage in Section 4 ("one extractor,
  `sp.event_study_vcov`") can then cite this cross-language evidence for
  the off-diagonal blocks, which Track A never compared.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json` (`80abadb7`,
    `fb8c61ff`)
  - `tests/orig_parity/results/_implementation_trace.json` (`80abadb7`)

### 2026-09-28 — call traces re-recorded after the agent-native audit fixes

- **Commit.** `c18b1183` (the agent-native audit commit; CHANGELOG
  "Unreleased": MCP tool-list profiles and discovery meta-tools,
  `search_functions` ranking, `sp.did` aliases / indicator-to-cohort,
  `sp.rd(cutoff=)` fix, nested diagnostics and `degradations` in the agent
  payload); re-traced every Track A module and every original-data module.
- **Reason.** `src/statspai/_aliases.py` (did-you-mean on `**kwargs`
  functions) and `src/statspai/core/results.py` (`to_dict` diagnostics /
  degradations, `to_json(detail=)`) sit on every module's estimation path,
  so every `exercised_sources` digest moved and both traces went stale
  (`tests/test_parity_implementation_provenance.py`,
  `tests/test_orig_parity_native_contract.py`). Neither change touches a
  numerical path: the alias wrapper only changes the exception raised for an
  unknown keyword, and the result changes only affect serialisation.
- **Effect on the paper.** None. The re-trace was checked field by field
  against the previous record: no module's `packages`, boundary-call
  package set, `rscript_launches` or `error` changed, so the native / port /
  third-party census of Section 5.3 and Appendix A and the original-data
  ledger's provenance marks are unchanged. What did change besides the
  digests: the recorded tracing environment (Python 3.10.20 → 3.11.15,
  scipy 1.15.3 → 1.17.1, scikit-learn 1.6.1 → 1.9.1; numpy 2.2.6 and
  pandas 2.3.3 as in the release trace), `seconds`, and a few boundary
  callee *names* inside the same package (scikit-learn 1.9 renamed
  `_base.predict` to `_base.MultiOutputLinearModel.predict`), because
  this run was made in a fresh venv rather than the release container. A
  first attempt in the container's `dist-packages` layout was discarded:
  the tracer keys third-party boundaries on `site-packages`, so it had
  silently reclassified every sklearn / statsmodels / pyfixest call as
  native. No estimate, standard error or table cell is read from these
  files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-09-27 — call traces re-recorded after the post-release merge

- **Commits.** The merge `4ad3a993` of `main` (upstream `a916c7bc`..`881fdf8c`: sdid
  refusal of staggered cohorts, LaTeX output precision, `did_had` bandwidth
  selectors, forest covariance attrs, and others) into the 1.32.0 release
  line; re-traced Track A modules 03 13 15 24 25 26 27 35 53 65 66 and
  original-data module 08.
- **Reason.** The traces bind SHA-256 digests of every source file on each
  module's estimation path; the merged commits edited some of those files,
  and upstream had recorded its traces before the release's
  `__version__`-line normalisation.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` changed; no module's implementation classification moved
  (checked field by field against `v1.32.0`), so the native / port /
  third-party census of Section 5.3 and Appendix A (86 / 0 / 3) and the
  original-data ledger's provenance marks are unchanged. No estimate,
  standard error or table cell is read from these files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the cluster-Lasso pass (Gaillac and L'Hour)

- **Commits.** `f1fa33c4` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (Track A modules
  01 02 04 05 06 07 08 09 11 12 14 16 17 18 19 20 22 52 53 55 73 88) and
  `tests/orig_parity/results/_implementation_trace.json` (all 12
  modules). The sources that staled them are in the same commit:
  `src/statspai/datasets/_canonical.py` and
  `src/statspai/datasets/__init__.py` (the `nsw_dw` docstring and two
  `attrs` flags), `src/statspai/synth/_core.py` and
  `src/statspai/synth/scm.py` (a permutation interval added to
  `model_info`).
- **Reason.** A pass over the companion code of Gaillac and L'Hour's
  *Machine Learning for Econometrics*
  (`docs/dev/2026-10-05-gaillac-lhour-ml4econometrics-review.md`). None
  of the edits changes a number a Track A module computes:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on 07, 11,
  12, 18, 19, 22 and 52 reported 7 reproduce, 0 drift.
- **Effect on the paper.** None. Checked field by field against the
  version before `f1fa33c4`: in the 22 Track A modules and the 12
  original-data modules only `exercised_sources` digests and `seconds`
  differ; no `packages`, boundary-call package set, `rscript_launches` or
  `error` changed. No estimate, standard error or table cell is read
  from these files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the pass over *The Effect* (Huntington-Klein)

- **Commits.** `119ea880` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (Track A modules
  01 02 04 05 06 11 14 16 17 22 37 42 44 45 46 47 48 49 51 53 54 55 56 57
  58 63 64 67 78 79 83 84 85 89) and
  `tests/orig_parity/results/_implementation_trace.json` (all 12
  modules). The sources that staled them are in `96f33427`:
  `src/statspai/core/utils.py` (narrow integer columns are widened to
  `int64` before a formula is evaluated), `src/statspai/did/_core.py` (the
  same, for covariate formulas), `src/statspai/rd/rdrobust.py` (a check
  that the columns are numeric), `src/statspai/matching/match.py` (the
  coarsening of `method='cem'`) and `src/statspai/diagnostics/sensemakr.py`
  (group benchmarks).
- **Reason.** A replay of the Stata code of Huntington-Klein's *The
  Effect* (2nd ed.) against Stata 18
  (`docs/dev/2026-10-05-the-effect-2e-review.md`). None of the edits
  changes a number a Track A module computes:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on the 34
  modules reported 34 reproduce, 0 drift (fixtures byte-identical, worst
  relative difference 0). No module squares a narrow integer column
  inside a formula, and none uses coarsened exact matching.
- **Effect on the paper.** None. Checked field by field against the
  version before `119ea880`: in the 34 Track A modules and the 12
  original-data modules only `exercised_sources` digests and `seconds`
  differ. No estimate, standard error or table cell is read from these
  files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the pass over the de Chaisemartin-D'Haultfoeuille DiD textbook

- **Commits.** `b7b4166f` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (Track A modules
  01 03 05 13 14 15 17 24 25 26 27 38 51 53 54 55 56 65 66 78 81 86) and
  `tests/orig_parity/results/_implementation_trace.json` (modules 01 04
  04b 08 09). The sources that staled them are in the same commit:
  `src/statspai/regression/ols.py` (`dfadjust=`; the CR2 / CR3 fits store
  their full covariance), `src/statspai/did/did_multiplegt_dyn.py`
  (non-binary treatments, analytic joint tests, `same_switchers`,
  `controls=`), `src/statspai/did/did_multiplegt.py` (non-binary
  treatments), `src/statspai/did/sun_abraham.py` (observation shares,
  `window_rule=`), `src/statspai/did/wooldridge_did.py` (a docstring),
  `src/statspai/synth/fect.py` (large panels, a convergence warning) and
  `src/statspai/__init__.py` (one export).
- **Reason.** A run of the four applications of de Chaisemartin and
  D'Haultfoeuille's DiD textbook against Stata 18
  (`docs/dev/2026-10-05-dcdh-did-textbook-review.md`). None of the edits
  changes a number a Track A module computes:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on the 22
  modules reported 22 reproduce, 0 drift (fixtures byte-identical, worst
  relative difference 1.8e-12). The Track A fixtures of modules 05, 78 and
  81 are balanced panels with a binary treatment, where the corrected
  code paths coincide with the old ones.
- **Effect on the paper.** None. Checked field by field against the
  version before `b7b4166f`: in the 22 Track A modules and the 5
  original-data modules only `exercised_sources` digests and `seconds`
  differ. No estimate, standard error or table cell is read from these
  files.
- **Paths.**

### 2026-10-05 — module 43 (Heckman) standard errors, its budget, and the traces after the syllabus pass (Xu and Lan)

- **Commits.** `aab5c0ff` changed `tests/r_parity/results/43_heckman_py.json`
  (the three standard errors), `tests/r_parity/compare.py` (the
  `43_heckman` `rel_se` budget, `5e-4` to `1e-6`), the rendered
  `tests/r_parity/results/parity_table.md` and
  `tests/r_parity/results/parity_table_3way.md`, and
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`. It re-recorded
  `tests/r_parity/results/_implementation_trace.json` (the entries of
  these Track A modules differ from the previous commit:
  01 03 13 14 15 24 25 26 27 33 39 43 51 53 54 55 56 65 66)
  and `tests/orig_parity/results/_implementation_trace.json` (all 12
  modules). The sources that staled the traces are in the same commit:
  `src/statspai/regression/ols.py` (the collinearity scan works on blocks
  of the cross-product), `src/statspai/regression/heckman.py`,
  `src/statspai/timeseries/arima.py`, `src/statspai/timeseries/var.py`
  and `src/statspai/__init__.py` (six new exports and an import line
  re-sorted by the pre-commit hook).
- **Reason.** A syllabus audit of 徐小君、蓝嘉俊《因果推断计量经济学》
  against Stata 18
  (`docs/dev/2026-10-05-xu-lan-causal-econometrics-review.md`). The
  two-step Heckman variance took the expected information of the
  first-step probit; R `sampleSelection::heckit` and Stata
  `heckman, twostep` take the observed information. That was the whole of
  the gap module 43 carried inside its tolerance.
- **Effect on the paper.** Module 43 only. Its Python standard errors move
  in the sixth digit: intercept 0.14540320 to 0.14540375, slope
  0.05678104 to 0.05678103, inverse Mills ratio 0.21715296 to 0.21717166.
  The worst relative standard-error gap of the module, computed from the
  three committed result files, goes from 1.9e-5 (R) / 8.6e-5 (Stata) to
  7.9e-11 / 4.3e-9, so any table or sentence that quotes the module's SE
  gap or its `5e-4` budget is stale; the module now sits in the default
  `1e-6` tier. Point estimates, the headline metric (`rel_est`) and the
  PASS verdict are unchanged. No other module's numbers changed:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on 01, 02,
  03, 14, 15, 33, 35, 39, 43, 51, 53, 54, 55, 56 and 59 reported 15
  reproduce, 0 drift. In both trace files only `exercised_sources`
  digests, `seconds` and module 43's result digest differ; no
  implementation classification moved. The registry census the paper
  quotes also moved with this commit: 1,306 to 1,312 registered
  functions.
- **Paths.**
  - `tests/r_parity/results/43_heckman_py.json`
  - `tests/r_parity/compare.py`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the second round of the syllabus pass

- **Commits.** `0ec5f60c` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (Track A modules
  01 14 41 51 53 54 55 56) and
  `tests/orig_parity/results/_implementation_trace.json` (all 12
  modules). The sources that staled them are in the same commit:
  `src/statspai/regression/ols.py` (the bread matrix of the centred fit
  is assembled by index arrays instead of a double loop) and
  `src/statspai/regression/tobit.py` (`ll=None` means no lower limit).
- **Reason.** Second round of the audit in
  `docs/dev/2026-10-05-xu-lan-causal-econometrics-review.md`. Neither
  edit changes a number a Track A module computes:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on 01, 14,
  41, 51, 53, 54, 55 and 56 reported 8 reproduce, 0 drift.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` differ; no implementation classification moved. No estimate,
  standard error or table cell is read from these files.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded for every module: the SCM interval and `summary()` of infinite bounds

- **Commits.** `7b279680` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (all 89 Track A
  modules) and `tests/orig_parity/results/_implementation_trace.json`
  (all 12). The sources that staled them are in the same commit:
  `src/statspai/core/results.py` (`CausalResult.summary()` prints an
  infinite interval end, a file on every estimation path),
  `src/statspai/synth/scm.py` (the `ci` of `sp.synth(method='classic')`
  is now the inversion of the placebo rank test) and
  `src/statspai/datasets/_canonical.py` / `src/statspai/datasets/__init__.py`
  (`nsw_dw(simulated=False)`).
- **Reason.** Decisions left open by the pass over Gaillac and L'Hour's
  *Machine Learning for Econometrics*
  (`docs/dev/2026-10-05-gaillac-lhour-ml4econometrics-review.md`).
- **Effect on the paper.** None on any frozen number: Track A modules
  07 and 52 compare weights, the estimate and fit statistics, not the
  interval, and `python tests/r_parity/verify_reproduce_py.py --no-report
  07_scm 52_scm_unique` reports 2 reproduce, 0 drift. Checked field by
  field against the previous traces: only `exercised_sources` digests and
  `seconds` differ in the 89 + 12 modules; no `packages`, boundary-call
  package set, `rscript_launches` or `error` changed. One thing to carry
  into the next revision of the manuscript: if the text describes the
  confidence interval that classic SCM prints, it is now the
  constant-effect inversion of the placebo test (Firpo and Possebom
  2018), not a normal interval around the placebo spread.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the second round of the dCDH textbook pass

- **Commits.** `c9a593ea` re-recorded
  `tests/r_parity/results/_implementation_trace.json` (Track A modules
  10 21 78). The sources that staled them are in the same commit:
  `src/statspai/did/did_multiplegt_dyn.py` (the slope-estimation term of
  the variance under `controls=`; the regression behind that option is
  weighted and fitted within `trends_nonparam` cells) and
  `src/statspai/did/es_inference.py` (`sp.uniform_bands` reports a sup-t
  test and takes `terms=`).
- **Reason.** Open items of
  `docs/dev/2026-10-05-dcdh-did-textbook-review.md`. None of the edits
  changes a number a Track A module computes:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on
  `10_honest_did`, `21_honest_relmags` and `78_multiplegt_dyn` reported
  3 reproduce, 0 drift (worst relative difference 0). Module 78 does not
  use `controls=`, and the event-study covariance modules 10 and 21 read
  is untouched.
- **Effect on the paper.** None. In the three modules only
  `exercised_sources` digests and `seconds` differ from the version
  before `c9a593ea`. No estimate, standard error or table cell is read from
  this file.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces of modules 68 and 75 re-recorded after `absorb_dof`

- **Commits.** `1286f507` re-recorded the entries of Track A modules 68 and 75
  in `tests/r_parity/results/_implementation_trace.json`. The source that
  staled them is in the same commit: `src/statspai/panel/hdfe.py`
  (`sp.absorb_ols` takes `absorb_dof=`; the default, `'reghdfe'`, is
  the behaviour it had).
- **Reason.** `areg` is translated to the absorbing path
  (`docs/dev/2026-10-05-xu-lan-causal-econometrics-review.md`, third
  round). No default changed:
  `python tests/r_parity/verify_reproduce_py.py --no-report` on 03, 15,
  37, 68 and 75 reported 5 reproduce, 0 drift.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` differ; no implementation classification moved. No estimate,
  standard error or table cell is read from this file.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the chapter 1 functions of the syllabus pass

- **Commits.** `23711a16` re-recorded the entries of Track A modules 03 13 15
  24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in the same commit: `src/statspai/__init__.py`
  (four new exports: `prtest`, `sktest`, `swilk`, `ci`).
- **Reason.** Fourth round of the audit in
  `docs/dev/2026-10-05-xu-lan-causal-econometrics-review.md`. No
  estimator on a Track A path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves from
  1,312 to 1,316 registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the *Causal Inference in R* pass

- **Commits.** `57171c15` re-recorded the entries of Track A modules 03 13 15
  24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `d5e62968`: `src/statspai/__init__.py` (six new
  exports), `src/statspai/inference/ipw.py` (estimands `'ATO'` and `'ATM'`
  added; the ATE / ATT / ATC branches are untouched) and
  `src/statspai/inference/g_computation.py` (estimand `'ATC'` added).
- **Reason.** Audit in
  `docs/dev/2026-10-05-barrett-causal-inference-in-r-review.md`. No
  estimator on a Track A path changed. The three correctness fixes of the
  pass (Crump trimming, weighted variances in the balance tables, minimal
  adjustment sets) are on no Track A or original-data path.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. Checked by rerunning the Python side of original-data
  modules 06, 07, 09, 10 and 11 and of Track A modules 03 and 53 on the new
  tree: 06, 11, 03 and 53 rewrite their result files byte for byte; 07, 09
  and 10 differ from the committed files in the 13th significant digit.
  Module 07 rerun with the code as it was before `d5e62968` gives the
  same digits as with it, so the difference is between this machine and
  the run that produced the committed files and does not come from this
  pass. The committed files were kept. The registry census the paper quotes
  moves from 1,316 to 1,322 registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the Facure notebooks pass

- **Commits.** `0a2a5124` re-recorded the entries of Track A modules 02 03
  04 05 07 08 11 13 15 16 17 24 25 26 27 35 36 38 50 52 53 59 65 66 70 71
  72 78 79 83 84 85 in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The sources
  that staled them are in the same commit: the categorical-covariate
  wrapper (`src/statspai/core/_covariates.py`) on `sp.ipw`, `sp.aipw`,
  `sp.match`, `sp.dml`, `sp.tmle`, `sp.g_computation`, `sp.drdid` and
  `sp.callaway_santanna`; the date-typed time wrapper in
  `src/statspai/did/_core.py` on the four staggered estimators; the
  repeated-row check in `src/statspai/did/callaway_santanna.py`; the
  `inference=` validation in `src/statspai/synth/scm.py`; the bracket
  block in `src/statspai/regression/iv.py`; and five new exports in
  `src/statspai/__init__.py`.
- **Reason.** The pass over Facure's *Causal Inference in Python*
  (`docs/dev/2026-10-05-facure-causal-inference-in-python-review.md`).
  Both wrappers pass a call with numeric covariates and numeric time
  straight through.
  `python tests/r_parity/verify_reproduce_py.py --no-report` on the 32
  Track A modules reported 32 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ, and no implementation classification moved. The
  registry census the paper quotes moves from 1,322 to 1,327 registered
  functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the *Linear Model and Extensions* pass

- **Commits.** `9621744d` re-recorded the entries of Track A modules 01 02
  03 04 05 07 13 14 15 16 17 18 24 25 26 27 35 37 40 41 42 43 44 45 46 47
  48 49 51 52 53 54 55 56 57 58 59 61 62 63 64 65 66 67 69 78 79 83 84 85
  in `tests/r_parity/results/_implementation_trace.json` and all 12
  modules of `tests/orig_parity/results/_implementation_trace.json`. The
  source that staled them is in `8d9ba5a3`: `src/statspai/__init__.py`
  (six new exports), `src/statspai/core/utils.py` (`I(x^2)` read as a
  power), `src/statspai/core/_vcov_spec.py` and
  `src/statspai/regression/ols.py` (`robust='hc4'`; the collinearity
  check of models without an intercept), `src/statspai/regression/iv.py`,
  `src/statspai/regression/glm.py`, `src/statspai/regression/quantile.py`,
  `src/statspai/did/_core.py`, `src/statspai/panel/panel_reg.py`,
  `src/statspai/survival/models.py`.
- **Reason.** The audit in
  `docs/dev/2026-10-05-ding-linear-model-review.md`. No default on a
  Track A path changed: `python tests/r_parity/verify_reproduce_py.py
  --no-report` on the 50 modules reported 50 reproduce, 0 drift. The one
  correctness fix that touches a traced estimator, the Efron score
  residuals behind `sp.cox(robust=)`, does not reach module 24, which
  compares model-based standard errors.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves from
  1,327 to 1,333 registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the Kohler-Kreuter pass

- **Commits.** `c4688894` re-recorded the entries of Track A modules 01 02
  03 13 14 15 24 25 26 27 35 37 42 44 45 46 47 48 49 51 53 54 55 56 57 58
  63 64 65 66 67 in `tests/r_parity/results/_implementation_trace.json`
  and all 12 modules of
  `tests/orig_parity/results/_implementation_trace.json`. The sources that
  staled them are in `0a810f36`: in `src/statspai/core/utils.py` the
  formula helper reads the levels of `C(g)` from the rows that are used
  and accepts columns named like Python keywords, and
  `src/statspai/__init__.py` exports thirteen new names.
- **Reason.** The pass over Kohler, Kreuter and Haensch, *Data Analysis
  Using Stata* (`docs/dev/2026-10-05-kohler-kreuter-4e-review.md`). The
  two changes of the formula helper act on a factor level without
  complete rows and on a column named like a keyword.
  `python scripts/trace_parity_provenance.py` re-ran the 31 Track A
  modules and the 12 original-data modules and reported `ok` for each:
  every one reproduces its committed result.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ, and no implementation classification moved. The
  registry census the paper quotes moves from 1,333 to 1,346 registered
  functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the Dogan textbook pass

- **Commits.** `0527cbe7` re-recorded every Track A module in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The sources
  that staled them are in the same commit: `src/statspai/core/results.py`
  (an `AttributeError` message that names where a statsmodels attribute
  lives), `src/statspai/__init__.py` (two new exports),
  `src/statspai/postestimation/hypothesis.py` (comma-separated
  restrictions, names matched without regard to blanks) and
  `src/statspai/timeseries/arima.py`.
- **Reason.** The pass over Dogan, *Introduction to Econometrics with
  Python* (`docs/dev/2026-10-05-dogan-python-econometrics-review.md`).
  The default `sp.arima` method now maximises the exact likelihood.
  Track A module `39_arima` calls `method="innovations_mle"`, which was
  exact before and is unchanged, so its row does not move.
  `python scripts/trace_parity_provenance.py` re-ran both ledgers and
  reported `ok` for each module: every one reproduces its committed
  result.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ, and no implementation classification moved. The
  registry census the paper quotes moves from 1,346 to 1,348 registered
  functions. A sentence that describes the default `sp.arima` as exact
  maximum likelihood is true from this commit on, and was not before.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces of modules 24, 48, 57, 63 and 64 re-recorded after the second round of the *Linear Model and Extensions* pass

- **Commits.** `adcfffbb` re-recorded the entries of Track A modules 24 48 57
  63 64 in `tests/r_parity/results/_implementation_trace.json`. The
  source that staled them is in the same commit:
  `src/statspai/survival/models.py` (`sp.kaplan_meier(conf_type=)`
  deprecation of its default; Wald and score tests in `sp.cox`
  diagnostics), `src/statspai/regression/logit_probit.py`
  (`predict(what='confidence')`) and
  `src/statspai/regression/zeroinflated.py` (three more diagnostics
  keys).
- **Reason.** Second round of the audit in
  `docs/dev/2026-10-05-ding-linear-model-review.md`. No default on a
  Track A path changed: `python tests/r_parity/verify_reproduce_py.py
  --no-report` on the five modules reported 5 reproduce, 0 drift.
- **Effect on the paper.** None. Only `exercised_sources` digests and
  `seconds` differ; no implementation classification moved. No
  estimate, standard error or table cell is read from this file.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-05 — call traces re-recorded after the second round of the Facure notebooks pass

- **Commits.** `218937a9` re-recorded the entries of Track A modules 05 08 11
  16 17 20 36 38 70 71 73 75 81 83 84 85 in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `b4f05a76`: the categorical-covariate and date-typed
  time wrappers were put on the remaining DiD and matching estimators
  (`gardner_did`, `stacked_did`, `wooldridge_did`, `bacon_decomposition`,
  `lp_did`, `did_multiplegt`, `event_study`, `psmatch2`, `policy_tree`
  and others).
- **Reason.** Second round of
  `docs/dev/2026-10-05-facure-causal-inference-in-python-review.md`. Both
  wrappers pass a call with numeric covariates and numeric time straight
  through.
  `python tests/r_parity/verify_reproduce_py.py --no-report` on the 16
  Track A modules reported 16 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census is unchanged by this round.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the third round of the Facure notebooks pass

- **Commits.** `45b56635` re-recorded the entries of Track A modules 13 and
  24 in `tests/r_parity/results/_implementation_trace.json`. The source
  that staled them is in the same commit:
  `src/statspai/forest/_grf_family.py` and
  `src/statspai/forest/causal_forest.py` (categorical covariates become
  one indicator per level; numeric covariates take the path they took
  before).
- **Reason.** Third round of
  `docs/dev/2026-10-05-facure-causal-inference-in-python-review.md`.
  `python tests/r_parity/verify_reproduce_py.py --no-report 13_causal_forest
  24_coxph` reported 2 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; only `exercised_sources` digests and `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the Dogan follow-up

- **Commits.** `e34f417f` re-recorded every Track A module in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The sources
  that staled them are in the same commit: `src/statspai/__init__.py`
  (the message of the package-level `AttributeError`),
  `src/statspai/regression/iv.py` (IV results report formula terms under
  the names `sp.regress` gives them),
  `src/statspai/postestimation/hypothesis.py` and
  `src/statspai/postestimation/_design.py`.
- **Reason.** Follow-up to the pass over Dogan, *Introduction to
  Econometrics with Python*
  (`docs/dev/2026-10-05-dogan-python-econometrics-review.md`). The IV
  change relabels coefficients and leaves every array as it was.
  `python scripts/trace_parity_provenance.py` re-ran both ledgers and
  reported `ok` for each module: every one reproduces its committed
  result.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ, and no implementation classification moved. The
  registry census is unchanged at 1,348. Where the paper prints an IV
  coefficient of a transformed or categorical term, its label is now the
  formula name (`np.log(x)`, `C(g)[T.2]`) and no longer `np.log[x]`,
  `g[2]`.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — Hansen's *Econometrics* pass: conditional logit row, call traces

- **Commits.** `8b2e429b` rewrote `tests/r_parity/results/46_clogit_py.json`.
  `faef6a04` regenerated the rollups that read it
  (`tests/r_parity/results/parity_table.md`, `parity_table.tex`,
  `parity_table_3way.md`, `parity_table_3way.tex`), refreshed
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json`, and re-recorded the entries of
  Track A modules 02 03 13 15 24 25 26 27 33 35 44 45 46 49 53 59 65 66 69
  in `tests/r_parity/results/_implementation_trace.json` and all 12
  modules of `tests/orig_parity/results/_implementation_trace.json`. The
  source that staled them is in `8b2e429b`: `src/statspai/__init__.py` (ten
  new exports), `src/statspai/regression/iv.py`,
  `src/statspai/regression/multinomial.py`, `src/statspai/panel/_cre.py`,
  `src/statspai/panel/xt_tools.py`, `src/statspai/timeseries/var.py`.
- **Reason.** Audit in `docs/dev/2026-10-05-hansen-econometrics-review.md`.
  `sp.clogit` now evaluates its likelihood without a loop over choice sets
  and finishes with Newton steps, so the estimate is at the optimum and no
  longer where BFGS stopped. 2SLS projects by QR and solves the second
  stage by least squares.
- **Effect on the paper.** One row moves. Module 46 (`clogit`,
  `beta_x`): the relative difference from R goes from 1.31e-08 to 2.04e-13
  for the estimate and from 2.68e-09 to 1.01e-13 for the standard error.
  The row was inside the 1e-6 budget before and after, so no evidence tier
  changes. `python tests/r_parity/verify_reproduce_py.py --no-report` on
  the other 18 modules reported 18 reproduce, 0 drift (run before the last
  rebase); module 02 (`iv`) moves by 5.2e-12, inside the 1e-9
  reproducibility tolerance, and its committed file was kept. In both
  trace files only `exercised_sources` digests, `seconds` and the result
  digest of module 46 differ; no implementation classification moved. The
  registry gains ten entries and one submodule (`multivariate`).
- **Paths.**
  - `tests/r_parity/results/46_clogit_py.json`
  - `tests/r_parity/results/parity_table.md`
  - `tests/r_parity/results/parity_table.tex`
  - `tests/r_parity/results/parity_table_3way.md`
  - `tests/r_parity/results/parity_table_3way.tex`
  - `tests/r_parity/TIER_A_FIXTURE_LOCK.json`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.gam` and `sp.conformal_regression`

- **Commits.** `ab3f96b8` re-recorded the entries of Track A modules 03 13 15
  24 25 26 27 53 63 64 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `c41e6bfc`: `src/statspai/__init__.py` (three new
  exports: `gam`, `GAMResult`, `conformal_regression`) and
  `src/statspai/regression/zeroinflated.py` (the fitted-value vectors of
  `sp.zip_model` / `sp.zinb` move from `diagnostics` to `data_info`).
- **Reason.** Third round of the audit in
  `docs/dev/2026-10-05-ding-linear-model-review.md`. No estimator on a
  Track A path changed: `python tests/r_parity/verify_reproduce_py.py
  --no-report` on the 12 modules reported 12 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,361
  registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.xthtaylor`

- **Commits.** `7fa469a8` re-recorded the entries of Track A modules 03 13 15
  24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `d8964f19`: `src/statspai/__init__.py` (one new
  export, `xthtaylor`).
- **Reason.** Second round of the audit in
  `docs/dev/2026-10-05-hansen-econometrics-review.md`. No estimator on a
  Track A or original-data path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry gains one entry.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the discrete choice covariance

- **Commits.** `8149feec` re-recorded the entries of Track A modules 44 45 46 49
  in `tests/r_parity/results/_implementation_trace.json` and rewrote
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `46c3483a`: `src/statspai/regression/multinomial.py`
  (`sp.mlogit`, `sp.ologit`, `sp.oprobit` and `sp.clogit` now keep the
  covariance matrix they already computed).
- **Reason.** Third round of the audit in
  `docs/dev/2026-10-05-hansen-econometrics-review.md`.
- **Effect on the paper.** None. `python tests/r_parity/verify_reproduce_py.py
  --no-report` on 44, 45, 46 and 49 reported 4 reproduce, 0 drift, each
  byte for byte. In the trace files only `exercised_sources` digests and
  `seconds` differ. No registry entry was added.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the Das *Causal Inference in R* pass

- **Commits.** `eb32bfdd` re-recorded the entries of Track A modules
  02 03 11 13 15 24 25 26 27 35 37 42 47 53 58 59 63 64 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and of modules 01, 04,
  04b and 08 in `tests/orig_parity/results/_implementation_trace.json`. The
  source that staled them is in `757bf198`: `src/statspai/__init__.py`
  (three new exports), `src/statspai/regression/iv.py`,
  `src/statspai/regression/count.py`, `src/statspai/matching/match.py` and
  `src/statspai/survival/models.py`.
- **Reason.** Audit in
  `docs/dev/2026-10-06-das-causal-inference-in-r-review.md`. `sp.iv` reads
  the two-part formula of `AER::ivreg`; `sp.poisson` reports a corrected
  overdispersion statistic among its diagnostics; `sp.match` counts tied
  matches and warns; `CoxResult.ph_test()` is the Grambsch-Therneau score
  test. None of these is on the path that produces a coefficient or a
  standard error.
- **Effect on the paper.** None. No committed result file changed. In both
  trace files only `exercised_sources` digests and `seconds` differ; no
  implementation classification moved. The registry gains three entries
  (`power_ttest`, `esize`, `cor_test`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — Maitra's *Static and Dynamic Econometric Modelling* pass: call traces

- **Commits.** `06c2ed66` re-recorded every entry of
  `tests/r_parity/results/_implementation_trace.json` (89 modules) and of
  `tests/orig_parity/results/_implementation_trace.json` (12 modules). The
  source that staled them is in `8e68b894`: `src/statspai/__init__.py` (three
  new exports), `src/statspai/core/results.py` (two pointers for
  statsmodels attribute names), `src/statspai/timeseries/arima.py`,
  `src/statspai/regression/quantile.py`, `src/statspai/inference/ttest.py`
  and `src/statspai/inference/basic_tests.py`.
- **Reason.** Audit in `docs/dev/2026-10-06-maitra-static-dynamic-review.md`.
  `sp.arima` uses the exact diffuse initialisation for differenced models
  and searches a badly scaled series on a standardised copy.
- **Effect on the paper.** None. `python tests/r_parity/verify_reproduce_py.py
  --no-report 39_arima 40_qreg`, the two Track A modules that run the
  changed estimators, reported 2 reproduce, 0 drift, both byte-identical:
  module 39 fits an undifferenced series of order one, which takes the
  unchanged path. No result file, parity table or fixture lock changed. In
  both trace files only `exercised_sources` digests and `seconds` differ;
  no implementation classification moved. The registry gains three entries
  (`chow_test`, `bds`, `rolling`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the Gow and Ding pass

- **Commits.** `7d14ea5c` re-recorded the entries of 45 Track A modules in
  `tests/r_parity/results/_implementation_trace.json` and of 6 of the
  12 modules of `tests/orig_parity/results/_implementation_trace.json`. The
  source that staled them is in `a07851ca`: `src/statspai/core/utils.py` (`factor(x)`
  read as `C(x)`; the formula rewriter is renamed `r_formula_idioms`, and
  its callers in `regression/ols.py`, `glm.py`, `iv.py`, `multinomial.py`,
  `panel/panel_reg.py`, `did/_core.py`, `postestimation/_design.py`,
  `survey/estimators.py` with it), `src/statspai/regression/ols.py`
  (`hac_panel=`; the two-way clustered covariance is stored and adjusted
  when it is not positive semi-definite), `src/statspai/regression/count.py`
  (Poisson starting values and the flat-deviance stop),
  `src/statspai/inference/jackknife.py`, `src/statspai/fixest/wrapper.py`
  (a warning) and `src/statspai/__init__.py` (four new exports).
- **Reason.** The audit in
  `docs/dev/2026-10-06-gow-ding-accounting-review.md`. Two of the changes
  alter numbers, and neither is reached by a Track A module: no module
  clusters in two dimensions on a covariance that is not positive
  semi-definite, and every Poisson fit on a Track A path converged before
  and converges to the same estimate now. `python
  tests/r_parity/verify_reproduce_py.py --no-report` on the 45 modules
  reported 45 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,372
  registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the Ness *Causal AI* pass

- **Commits.** `e305f9bd` re-recorded the entries of Track A modules
  03 04 08 13 14 15 24 25 26 27 50 53 65 66 71 72 79 in
  `tests/r_parity/results/_implementation_trace.json` and of modules 02, 06,
  07, 08, 09, 10 and 11 in
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `159d0cb5`: `src/statspai/__init__.py` (four new
  exports), `src/statspai/core/_validate.py`,
  `src/statspai/core/_agent_summary.py`, `src/statspai/inference/ipw.py`,
  `src/statspai/inference/aipw.py`, `src/statspai/inference/g_computation.py`,
  `src/statspai/tmle/tmle.py` and `src/statspai/dml/_base.py`.
- **Reason.** Audit in `docs/dev/2026-10-06-ness-causal-ai-review.md`. The
  estimators named above convert the treatment column through one shared
  check that names a column of labels; `sp.ipw` also stores each arm's
  effective sample size and warns when it is small, and the violations
  report reads it. None of these is on the path that produces a
  coefficient or a standard error. The substantive fixes of the pass are
  in `dag/` and `causal_discovery/`, which no Track A module exercises.
- **Effect on the paper.** None. No committed result file changed. In both
  trace files only `exercised_sources` digests and `seconds` differ; no
  implementation classification moved. The registry gains four entries
  (`bayes_net`, `BayesNet`, `refute`, `RefutationResult`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the forecasting toolkit and the `sp.arima` optimiser change

- **Commits.** `8358dbf5` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 39 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and of module(s)
  08 in `tests/orig_parity/results/_implementation_trace.json`. The
  source that staled them is in `fdcf0a19`: `src/statspai/__init__.py` (19
  new exports, the forecasting functions and their result classes) and, for
  module 39 only, `src/statspai/timeseries/arima.py`.
- **Reason.** The audit in
  `docs/dev/2026-10-06-hyndman-fpp-pythonic-review.md`. `sp.arima` with the
  default `method='statespace'` now starts its likelihood search twice and
  checks it with simplex searches, and estimates a differenced model on the
  differenced series. Module 39 runs `method='innovations_mle'`, which is
  unchanged: `python tests/r_parity/verify_reproduce_py.py --no-report` on
  Track A modules 03 13 15 24 25 26 27 39 53 65 66 reported 11 reproduce, 0
  drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes. In the trace files `exercised_sources` digests and `seconds`
  differ, and for module 39 the count of attribute reads on the statsmodels
  result object moves from 11 to 8; its classification as served by
  statsmodels is unchanged. The registry census the paper quotes moves to
  1,395 registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the Bayesian econometrics pass

- **Commits.** `02f99f19` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `cefca309`: `src/statspai/__init__.py` (the
  exports of the new `statspai.mcmc` subpackage: `bayes_regress`,
  `bayes_mixed`, `bma`, `bayes_factor`, `savage_dickey`,
  `bayes_bootstrap`, seven chain diagnostics and their result classes).
- **Reason.** The audit in
  `docs/dev/2026-10-06-ramirez-hassan-bayesian-econometrics-review.md`.
  No estimator on a Track A or original-data path changed. The
  correctness fixes of that commit are in `sp.bayes_iv`,
  `sp.bayes_hte_iv` and `sp.bayes_fuzzy_rd`, which no parity module
  calls.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,413
  registered functions across 89 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the dynamic panel fix (Hansen, Table 17.3)

- **Commits.** `9705f342` re-recorded the entry of Track A module 50 in
  `tests/r_parity/results/_implementation_trace.json` and rewrote
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled module 50 is in `1879dce1`:
  `src/statspai/gmm/_dynpanel/_estimate.py`,
  `src/statspai/gmm/_dynpanel/_fit.py` and
  `src/statspai/gmm/arellano_bond.py`.
- **Reason.** Audit in `docs/dev/2026-10-05-hansen-econometrics-review.md`
  (findings 15 to 18). Three changes to dynamic panel GMM: which period
  dummies `time_dummies=True` keeps, a rank check before the weight matrix
  is inverted, and the generalized inverse used when it is singular. Each
  one acts only on a fit that has period dummies or a rank-deficient
  weight. Module 50 has neither.
- **Effect on the paper.** None. No committed result file changed; module
  50 rerun on the new code returns its two coefficients and standard
  errors unchanged to the sixteenth digit. In the Track A trace only the
  three `exercised_sources` digests of module 50 and its `seconds`
  differ. In the original-data trace only `seconds` differ. No
  implementation classification moved.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call trace of module 39 re-recorded after the `sp.arima` likelihood change

- **Commits.** `c3c01c13` re-recorded the entries of Track A modules 39 in
  `tests/r_parity/results/_implementation_trace.json`. The source that
  staled them is in `2d82b3f8`: `src/statspai/timeseries/arima.py`.
- **Reason.** Second round of the audit in
  `docs/dev/2026-10-06-hyndman-fpp-pythonic-review.md`. `sp.arima` with the
  default `method='statespace'` evaluates the likelihood of the
  differenced series by the innovations algorithm
  (`timeseries/_arma_core.py`) instead of statsmodels' Kalman filter; the
  two agree to 1e-8. Module 39 runs `method='innovations_mle'`, which is
  unchanged: `python tests/r_parity/verify_reproduce_py.py 39_arima
  --no-report` reported 1 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in the trace file only `exercised_sources` digests and
  `seconds` differ. The registry census stays at 1,413 registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.abnormal_returns` and the `sp.feols` covariance adjustment

- **Commits.** `ed111e56` re-recorded the entries of Track A modules
  03 13 15 24 25 26 27 53 65 66 67 in
  `tests/r_parity/results/_implementation_trace.json` and of 1 module(s) of
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `8abcf7f1`: `src/statspai/fixest/wrapper.py`
  (`sp.feols` sets the negative eigenvalues of a multiway clustered
  covariance to zero, as R's fixest does) and `src/statspai/__init__.py`
  (two new exports: `abnormal_returns`, `AbnormalReturnsResult`).
- **Reason.** Second round of the audit in
  `docs/dev/2026-10-06-gow-ding-accounting-review.md`. The adjustment
  acts only on a multiway covariance that is not positive semi-definite,
  which no Track A module has: `python
  tests/r_parity/verify_reproduce_py.py --no-report` on the 11 modules
  reported 11 reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,415
  registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — the Lalonde matching row after `sp.match` keeps tied matches by default

- **Commits.** `4d79e7b2` rewrote
  `tests/orig_parity/results/04_lalonde_original_py.json`. `b627761f`
  regenerated the rollup that reads it
  (`tests/orig_parity/results/parity_table_orig.md`) and re-recorded the
  entries of Track A module 11 in
  `tests/r_parity/results/_implementation_trace.json` and of modules 04 and
  04b in `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `4d79e7b2`: `src/statspai/matching/match.py`.
- **Reason.** Decision recorded in
  `docs/dev/2026-10-06-das-causal-inference-in-r-review.md`. With
  replacement, `sp.match` now keeps every control tied at the nearest
  distance (`ties='all'`) instead of the first in data order, so the
  estimate no longer depends on the order of the rows.
- **Effect on the paper.** One row of the original-data table moves.
  Module 04 (`psm_att`, Lalonde, 614 rows): the StatsPAI estimate goes from
  1967.94 to 1968.80 (the table prints 1968, now 1969) and its standard
  error from 1192.17 to 1181.66. The relative gaps to R `MatchIt` (0.019)
  and to the published value (0.097) are unchanged at the precision
  printed. The new value equals Stata 18
  `teffects psmatch (re78) (treat ..., logit), atet` (1968.799715855857) to
  the last digit. Module 04b (NSW with PSID-1) and Track A module 11
  (NSW-DW) have no exact ties and did not move:
  `python tests/r_parity/verify_reproduce_py.py --no-report 11_psm` reports
  the fixture byte-identical. In the Track A trace only the
  `exercised_sources` digest of module 11 differs.
- **Paths.**
  - `tests/orig_parity/results/04_lalonde_original_py.json`
  - `tests/orig_parity/results/parity_table_orig.md`
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.threshold` (Hansen, chapter 23)

- **Commits.** `9eff7366` re-recorded the entries of Track A modules
  03 13 15 24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and rewrote
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `b6a1cdb9`: `src/statspai/__init__.py` (one new export,
  `threshold`).
- **Reason.** Audit in `docs/dev/2026-10-05-hansen-econometrics-review.md`
  (finding 19). `sp.threshold` is a new function in a new file,
  `src/statspai/regression/threshold.py`, which no parity module calls.
  Nothing on the path of an existing estimator changed.
- **Effect on the paper.** None. No committed result file changed. In both
  trace files only the `src/statspai/__init__.py` digest under
  `exercised_sources` and `seconds` differ; no implementation
  classification moved. The registry gains one entry.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.path_analysis`

- **Commits.** `087480e9` re-recorded the entries of Track A modules
  03 13 15 24 25 26 27 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and of module 08 in
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `10eff6e9`: `src/statspai/__init__.py` (two new exports).
- **Reason.** A new function, `sp.path_analysis`, in a module no parity
  script calls.
- **Effect on the paper.** None. No committed result file changed; in both
  trace files only the digest of `src/statspai/__init__.py` and `seconds`
  differ. The registry gains two entries.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.bayes_ivreg`

- **Commits.** `dbbaf462` re-recorded the entries of Track A modules 03 13 15 24 25 26 27 53 65 66
  in `tests/r_parity/results/_implementation_trace.json` and all 12 modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `1e429eef`: `src/statspai/__init__.py` (one new
  export, `bayes_ivreg`).
- **Reason.** Second round of the audit in
  `docs/dev/2026-10-06-ramirez-hassan-bayesian-econometrics-review.md`.
  No estimator on a Track A or original-data path changed.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in both trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,419
  registered functions across 89 submodules.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after `sp.identify_counterfactual`

- **Commits.** `c3f1f752` holds the re-recorded entries of the Track A
  and original-data modules whose import path includes
  `src/statspai/__init__.py`, in
  `tests/r_parity/results/_implementation_trace.json` and
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `81821d0f`: two new exports in
  `src/statspai/__init__.py`.
- **Reason.** `sp.identify_counterfactual` (ID* and IDC*), the open item
  of `docs/dev/2026-10-06-ness-causal-ai-review.md`. The new module is
  `src/statspai/dag/counterfactual_id.py`, which no Track A module
  exercises.
- **Effect on the paper.** None. No committed result file changed. In both
  trace files only `exercised_sources` digests and `seconds` differ; no
  implementation classification moved. The registry gains two entries
  (`identify_counterfactual`, `CounterfactualIdentification`).
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after a typing fix

- **Commits.** `dc858d18` re-recorded the entries of Track A modules
  04 05 06 13 16 17 20 24 44 45 49 50 73 75 77 78 79 80 83 84 85 88 89 in
  `tests/r_parity/results/_implementation_trace.json` and of two modules
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `fd124cce`: 45 return statements in 23 files, each
  wrapped in `np.asarray`.
- **Reason.** The mypy step of CI had been failing on main (848 errors
  against a ceiling of 827), which skipped the test run behind it.
  `np.asarray` is the identity on an array: `python
  tests/r_parity/verify_reproduce_py.py --no-report` on the 23 modules
  reported 23 reproduce, 0 drift.
- **Effect on the paper.** None; in both trace files only
  `exercised_sources` digests and `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the last items of the forecasting audit

- **Commits.** `f8940920` re-recorded the entries of Track A module(s) 03 13 15 24 25 26 27 39 53 65 66 in
  `tests/r_parity/results/_implementation_trace.json` and of original-data
  module(s) 08 in
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `122b028c`: `src/statspai/__init__.py` (five new exports:
  `ts_features`, `bootstrap_series`, `bagged_forecast`,
  `BaggedForecastResult`, `seasonal_dummies`) and, for module 39,
  `src/statspai/timeseries/arima.py` (new `boxcox=` and `biasadj=`
  arguments, unused by the module).
- **Reason.** Fourth round of the audit in
  `docs/dev/2026-10-06-hyndman-fpp-pythonic-review.md`. No estimator on a
  Track A or original-data path changed: `python
  tests/r_parity/verify_reproduce_py.py 39_arima --no-report` reported 1
  reproduce, 0 drift.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in the trace files only `exercised_sources` digests and
  `seconds` differ. The registry census the paper quotes moves to 1,426
  registered functions.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the feols IV diagnostics

- **Commits.** `6ecde941` re-recorded the entries of Track A modules
  [67] in
  `tests/r_parity/results/_implementation_trace.json` and of 0 module(s)
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `e497086c`.
- **Reason.** The audit in
  `docs/dev/2026-10-06-gow-ding-accounting-review.md`. `python
  tests/r_parity/verify_reproduce_py.py --no-report` on those modules:
  Summary: 1 reproduce, 0 drift, 0 skipped/errored.
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in the trace files only `exercised_sources` digests and
  `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the matchit translation fix

- **Commits.** `005ffadd` re-recorded the entries of Track A modules
  [] in
  `tests/r_parity/results/_implementation_trace.json` and of 0 module(s)
  of `tests/orig_parity/results/_implementation_trace.json`. The source
  that staled them is in `d42a34b3`.
- **Reason.** The audit in
  `docs/dev/2026-10-06-gow-ding-accounting-review.md`. `python
  tests/r_parity/verify_reproduce_py.py --no-report` on those modules:
  no Track A module was stale
- **Effect on the paper.** No estimate, standard error or table cell
  changes; in the trace files only `exercised_sources` digests and
  `seconds` differ.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after the FCI, mediation, hill-climbing and forest-tuning batch

- **Commits.** `c2adac47` re-recorded the entries of Track A modules
  03 08 11 13 15 24 25 26 27 36 53 65 66 70 71 in
  `tests/r_parity/results/_implementation_trace.json` and of modules
  04 04b 08 in `tests/orig_parity/results/_implementation_trace.json`.
  The source that staled them is in `284c81a9`.
- **Reason.** Four new functions (`hill_climb`, `bootstrap_edges`, `tune_causal_forest`, and the non-linear outcome models of `sp.mediate`), the completed `sp.fci`, and a higher cap on the MCP tool manifest; notes in `docs/dev/2026-10-06-das-causal-inference-in-r-review.md`. The feature commit is the one before the cited source commit. No parity script calls the changed estimation code.
- **Effect on the paper.** None. No committed result file changed; in both trace files only source digests and `seconds` differ. The registry gains three entries.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`

### 2026-10-06 — call traces re-recorded after backtick-quoted names in formulas

- **Commits.** `badb893c` re-recorded the entries of every Track A and
  original-data module whose estimation path reads a formula, in
  `tests/r_parity/results/_implementation_trace.json` and
  `tests/orig_parity/results/_implementation_trace.json`. The source that
  staled them is in `9c88e2d2`: `src/statspai/core/utils.py`,
  `src/statspai/regression/ols.py`, `src/statspai/regression/iv.py`,
  `src/statspai/regression/count.py` and `src/statspai/regression/gam.py`.
- **Reason.** Third round of
  `docs/dev/2026-10-06-ness-causal-ai-review.md`. A column name written
  between backticks is read as `Q("...")`; the IV parser no longer
  rewrites a hyphen inside a quoted name. A formula without backticks or
  quoted names takes the same path as before: the translation is skipped
  unless the formula contains a backtick.
- **Effect on the paper.** None. No committed result file changed. In both
  trace files only `exercised_sources` digests and `seconds` differ; no
  implementation classification moved.
- **Paths.**
  - `tests/r_parity/results/_implementation_trace.json`
  - `tests/orig_parity/results/_implementation_trace.json`
