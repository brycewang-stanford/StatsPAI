# What specialists require of StatsPAI, and how to prove it delivers

Written 2026-11-05/06 for the owner and for later agent sessions. It answers
three questions the owner asked. Who are the top specialists, and what do they
require? What tools and materials does the work need? How do we prove the
result? Then it records what this round did and what is left.

## 1. What the top specialists require

"Specialist" covers five groups. Each reads a statistics package differently.

| Group | Typical question | What they require |
| --- | --- | --- |
| Applied economists (DiD, IV, RD, SC, panel) | "Will my referee's Stata give the same number?" | Same estimates and SEs as Stata / R on the same data; modern estimators with their diagnostics (pre-trends, weak-IV, density tests); sensitivity (HonestDiD, Oster, sensemakr); tables and figures for the paper |
| Method authors (the people who wrote `did`, `rdrobust`, `synthdid`, `DoubleML`, `grf`) | "Did you implement my estimator, or something near it?" | Faithful defaults and options, the same answer as their package, and written reasons for every place you differ |
| Epidemiologists and biostatisticians | "Does it follow the analysis plan?" | g-methods, TMLE, survival, target-trial emulation; reporting standards; refusal to run on inputs that break the method |
| ML-causal researchers | "Is the uncertainty right, and does it scale?" | Cross-fitting that reproduces; intervals that cover; runtime on large data |
| Reviewers, journal data editors, agent builders | "Can I trust and rerun it without the author?" | Fail loudly on bad input; seeds and provenance; stable schemas and structured errors; an honest list of what is not checked |

Across the groups the requirements collapse to five.

1. **Correctness against the canonical implementation.** The same data must
   give the same estimate and SE as the package the method's authors maintain.
2. **Calibrated inference.** A 95% interval must cover about 95% of the time.
3. **No silent wrong numbers.** Missing values, `inf`, string levels and
   degenerate designs must produce the right answer or a clear refusal.
4. **Reproducibility and speed.** Same seed, same answer; at least as fast as
   the reference on the same task.
5. **Honest limits.** Every function says what evidence it has; gaps are
   listed by name, not averaged away.

The fifth matters most for trust. A package that claims everything is checked
is believed less than one that names its 27 unchecked estimators.

## 2. What tools and materials the work needs

| Need | Status before this round | Where it lives |
| --- | --- | --- |
| Reference engines (Stata 18 MP, R with pinned packages) | Present | `/Applications/Stata`, `renv.lock`, `tests/reference_parity/R_PACKAGE_VERSIONS.md` |
| Same-data parity harness | Present (Track A, frozen fixtures) | `tests/r_parity/`, `tests/stata_parity/`, `tests/reference_parity/` |
| Known-truth DGPs and Monte Carlo coverage | Present (Track B) | `tests/reference_parity/`, `tests/coverage_monte_carlo/` |
| Same-task timing harness | Present (Track C) | `tests/perf/` |
| An external yardstick for "good statistical software" | **Missing** | Added: rOpenSci standards audit |
| Systematic edge-case probing | **Missing** (only per-estimator tests) | Added as probes in the audit; fixes in the input-hygiene line |
| One page with all the proof | **Missing** (spread over parity.md, FINDINGS.md, perf tables) | Added: `docs/scorecard.md` |
| Agent benchmark run | Protocol written, run not done | `tests/agent_bench/`; waits on the OSF deposit (owner action) |
| An idle machine for timings | Owner scheduling | `tests/perf/run_when_idle.sh` |

The external yardstick is the rOpenSci Statistical Software Peer Review
standards (https://stats-devguide.ropensci.org/standards.html, version 0.2.0).
They are written by statisticians who review statistical packages, so they are
a fair stand-in for "what experts check". The audit is
[`2026-11-05-ropensci-standards-audit.md`](2026-11-05-ropensci-standards-audit.md).

## 3. How to prove it

No single number proves one package is the best. Four claims can be measured,
and each can fail.

| Claim | Measure | Where it is reported | Falsified by |
| --- | --- | --- | --- |
| Correct | Share of estimators matched to an R / Stata / author reference on the same data; relative gap distribution | `docs/scorecard.md` section 1, `sp.parity_summary()`, `docs/parity.md` | A reference run that disagrees beyond the registered tolerance |
| Calibrated | Monte Carlo coverage inside the 99% Wilson band | scorecard section 2, `tests/coverage_monte_carlo/FINDINGS.md` | A row outside the band without a mechanism |
| Fast | Median time vs the reference on the same file and task | scorecard section 3, `tests/perf/results/perf_table.md` | A slower row on an idle machine |
| Honest | Unverified estimators and weak families listed by name; standards met / partial / gap | scorecard section 4, the standards audit | A claim that a listed item is checked when it is not |

The scorecard is regenerated from committed artifacts and gated by the release
census (`release_gate.py --fix-census` / `--check`), so it cannot drift from the
evidence the way the README tier table did.

Targets, each checkable on the scorecard. Status after this round in brackets.

- unverified estimators: 0, or each one named with a reason [done: 27 to 0];
- cross-language share of estimators: from 50.2% (1.42.1) to above 55%
  [53.9%];
- coverage rows outside the band: each with a written mechanism [SDID and
  DML-PLR, both explained in FINDINGS.md];
- rOpenSci gaps: from 28 to under 15, with no known silent-wrong-number path
  [the two silent paths and five more input-hygiene gaps closed; the audit's
  status lines say which].

## 4. What this round did

36 commits on `main` after 1.42.1 (`831f154e1` to `4f8e97f37`). Seven lines
ran: one by the coordinator and six by agents in separate worktrees. Every
push passed the pre-push gates. No tag, PyPI upload or GitHub Release was made.

**Evidence (estimators, live `sp.parity_summary()`).**

| | 1.42.1 | Now |
| --- | ---: | ---: |
| Estimators | 1,012 | 1,009 |
| With numerical evidence | 985 (97.3%) | 1,009 (100%) |
| Matched to an R / Stata / author reference on the same data | 508 (50.2%) | 544 (53.9%) |
| Unverified | 27 | 0 |

The three estimators that left the denominator are not estimators
(`dgp_ddc` simulates data, `llm_code_responses` wraps a model call,
`reset_spec_counter` clears session state).

- Panel (`43bbeef44`): ten `xt*` commands graded against Stata 18 and R.
  Seven are bit-exact and three aligned, each documented gap pinned in a test.
- Regression and inference (`42ae79d18`): `intreg`, `heckprobit`,
  `gologit`, `prais`, `cnsreg`, `nls`, `mprobit`, `standardize_coefs` and
  `conley_sensitivity` bit-exact. `iv_wild_bootstrap` is aligned with
  `boottest`, with all 4,096 sign vectors enumerated on both sides.
- Matching (`b2a0671a2`): 16 weighting and balance tools graded against
  WeightIt, cobalt, propensity, halfmoon, lmw, cem and Stata `pscore`.
  Ten are bit-exact, six aligned.
- Known-truth evidence for eleven more estimators (`824fcf2be`). Most of it
  already existed, but the index could not see it.

**Correctness fixes (⚠️ in CHANGELOG and MIGRATION).**

- `sp.iv_wild_bootstrap` and every WRE path with two or more endogenous
  regressors: the restricted refit uses LIML, as `boottest` does
  (`8a1d327b4`).
- `sp.propensity_score(method='probit')` now runs to the maximum
  (`04f4b1e2a`).
- The weighted log likelihood of `intreg`, `heckprobit`, `tobit` and
  `heckman(method='ml')` is now Stata's pseudo log likelihood (`8a1d327b4`,
  `26e36cacc`).
- `sp.bacon_decomposition` refuses an unidentified design instead of
  returning 0 (`7a413980c`).
- Input hygiene (`e9b5890f2` to `a2d4c3263`):
  - `sp.synth` and its siblings refuse a missing treated outcome; it used to
    move the ATT silently.
  - Ordered models refuse unordered string outcomes; they used to sort
    them alphabetically.
  - Every estimator refuses `inf` with the same message.
  - `sp.feols` checks its input before any backend.
  - `Decimal` columns are read as numbers.

**Other fixes.**

- `sp.audit` checks forests as forests, and `sp.regtable` tabulates them.
- Registry evidence notes no longer leak between functions.
- `validation_scope` certifies synth donor-weight uniqueness.
- MCMC samplers record their seed and data hash.
- Exporters check the file suffix.

**Proof layer.**

- Nightly CI was red on every platform for three nights; fixed, and the
  release gate now regenerates the version-stamped tables (`831f154e1`).
- The README tier counts were wrong by a wide margin. They now come from the
  release census (`576063cd0`).
- `docs/scorecard.md` is new (`419906699`, `c7c92766e`).
- The rOpenSci standards audit is new (`20c29b1b6`).

## 5. Open items and owner decisions

**Decisions only the owner can make.**

1. Crump trimming when the sample equation has several roots. StatsPAI takes
   the smallest. The R packages agree with each other but follow no rule we
   could identify. The alternative is the root with the smallest variance
   objective; it is more principled but changes numbers.
   **Decided 2026-11-06:** the root with the smallest sample objective,
   smallest root on a tie (⚠️ correctness fix; the R parity fixture's
   smallest root is already the minimiser, so it is unchanged).
2. Continuous-treatment weights in `sp.ps_weights`. Keep the density ratio,
   or follow the WeightIt / propensity convention?
   **Decided 2026-11-06:** kept. WeightIt 1.7.0 `glm` equals propensity
   (7e-15): a ratio of standard normal densities of standardized values,
   i.e. the density ratio times `sd_ML(A) / sigma_ML`. The density ratio
   is the standard definition; WeightIt's weights are rebuilt from ours.
3. Resolved 2026-11-06, no change needed. The compiled JSS section
   (`05-parity-compact.tex`) uses the exact binomial 99% acceptance region
   `[0.931, 0.967]`, which is correct. The wrong `[0.935, 0.967]` appears only
   in the uncompiled long version `05-parity.tex` and the Chinese translation
   `main-zh.md`; fix those whenever they are next regenerated.
4. CausalAgentBench v2 cannot run until the OSF deposit is made. Until then
   the scorecard makes no agent-performance claim.

**Engineering items left open.**

- `sp.ivqreg` and `sp.lasso_iv` have no usable external reference.
  - `ivqreg`: R `IVQR` fails on R 4.5; Stata `ivqreg` is not on SSC.
  - `lasso_iv` is StatsPAI's own procedure.
- `sp.xtivreg` lacks the `fd`, `small`, `regress` and `first` options.
  **Done 2026-11-06** (`stata_xtivreg_opts.do`, 20 fits and 8 first stages).
- WRE confidence sets can be disjoint under weak IV. `boottest` finds the
  second piece; StatsPAI does not search for it.
  **Done 2026-11-06:** `ci_set` (grid scan plus bisection); boottest's
  default grid cuts that second piece at its grid end.
- The remaining rOpenSci gaps are listed in the audit:
  - a shared `missing="raise"` option;
  - time-series date-gap checks through `read_series`;
  - result accessors that keep the sample index;
  - MCMC start values and resumption.
  **All four done 2026-11-06** (gaps #4, #7, #9, #11; status lines in the
  audit). Still open there: a separate missing-value policy for predictors
  and response (RE2.2), a test that triggers every error message (G5.2b),
  173 validated functions without a method reference, and `inits=` /
  `continue_from=` on the other 15 `statspai.mcmc` samplers.
- Families with no cross-language row: `bayes`, `doe`, `conformal_causal`,
  `interference`, `proximal`, `target_trial` and others (scorecard section 4).
  Several have R references worth trying first: `doe` (`AlgDesign`, `DoE.base`),
  `target_trial` (`TrialEmulation`), `interference` (`inferference`).
  These are leads to verify, not claims.
  - 2026-11-06 (`parity-families` line): 32 estimators gained same-data
    R / Stata rows (`doe` 11 of 14, `proximal` 4 of 9, `interference` 4 of
    12, `conformal_causal` 3 of 12, `target_trial` 2 of 6, `longitudinal`
    2 of 5, `forest` 9 of 27, `experimental` 5 of 15). `TrialEmulation`
    fits sequential trials, not clone-censor-weight, and `inferference`
    estimates allocation-strategy effects no StatsPAI function computes;
    neither was used. `bayes` samplers, `causal_rl`, `fairness` and
    `neural_causal` stay at known truth.
- Track C was timed on 1.40.2; re-time on an idle machine at the next
  re-anchor. `benchmarks/RESULTS.md` is from 1.18.0 and should be rerun, or
  dropped in favour of Track C.
- Worktrees from this round remain under `.claude/worktrees/`. Their work is
  all on `main`. `input-hygiene` holds an uncommitted copy of what was pushed
  from `input-hygiene-commit`.

## 6. Release 1.43.0 (2026-11-06)

Released to PyPI with tag `v1.43.0` = `401de8169`. No GitHub Release was
created, so no Zenodo DOI was minted. It shipped these figures:

- Estimators with numerical evidence: 1,009 of 1,009.
- Matched to an R / Stata / author reference on the same data: 576 (57.1%),
  against 508 (50.2%) at 1.42.1. The target of 55% is met.
- Full local suite on the release tree: 48,429 passed, 0 failed. The six
  push workflows on the tagged commit are green.

The Linux and pandas 3 spot check of the 36 new or changed test files,
run before tagging, found two platform-dependent bugs that macOS could not
show. Both are fixed in the release (`26d546a60`):

- `sp.event_study` relied on `np.linalg.inv` raising for a singular `X'X`.
  OpenBLAS does not raise, so a duplicated covariate gave a different
  estimate on Linux.
- The time-series gap check read `asi8` as nanoseconds. pandas 3 parses
  dates at microsecond resolution.

Day two also fixed two more problems:

- `sp.longitudinal_contrast` on the MSM path returned 0 by construction.
- The g-formula contrast ignored the covariance of the two regimes.

These were found while closing the parity line's open items (`8364d85f9`).
