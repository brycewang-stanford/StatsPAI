# Owner decisions, 2026-10-20

The 2026-10-19 audit left a list of questions that only the owner could
settle (defaults of public functions, deprecations, GPL provenance, which
reference to follow when references disagree). On 2026-10-20 Bryce
delegated them ("you make the best decisions for me then continue"). The
coordinator decided them and the fix agents implemented them in five lines
of work (W5-*). This file records each decision, the items deliberately
kept as they are, and the release recommendation. Decisions are final for
this release cycle; reopening one goes through the normal deprecation
process (CLAUDE.md §3.8).

Rule for the whole wave. No number changes in this release except where
an item is marked ⚠️ (a correctness fix, logged in CHANGELOG and
MIGRATION.md). Behaviour that will change later is announced now with a
`FutureWarning` and switched no earlier than the next minor release.

## W5-defaults (defaults, deprecations, small API)

| # | Decision | This release | Later |
| --- | --- | --- | --- |
| 1 | `sp.causal_forest(n_jobs=)` default `1` to `-1` | Done. `n_jobs` is now an explicit, registered parameter of `sp.causal_forest`; outputs are byte-identical for any thread count (two tests pin it). Core API snapshot updated with the MIGRATION acknowledgement. | None |
| 2 | `sp.hdfe_ols` multi-way cluster small-sample factor moves to `cluster_df='min'` (Stata `reghdfe`, R `fixest`) | `FutureWarning` when two or more cluster variables are given without `cluster_df=`; standard errors unchanged (`'per_term'`, R `sandwich::vcovCL`). | Default becomes `'min'` |
| 3 | `sp.sun_abraham` headline moves to the cell-weighted ATT of `fixest::summary(agg = "att")` | The existing `aggregation=` is the switch (no second parameter). Without it, a `FutureWarning`; `.estimate` unchanged. `sp.did(method='sun_abraham')` now passes a missing `aggregation` through. Track A module 05 already passes `aggregation='fixest_att'`, so it is unaffected. | Default becomes `'fixest_att'` |
| 4 | `sp.bcf` will route to `sp.bart_causal(method='bcf')` | `FutureWarning` on every call; numbers unchanged. `sp.bcf_ordinal` and `sp.bcf_factor_exposure` call the same fit through a private function and do not warn. | Reroute; result type becomes that of `sp.bart_causal` |
| 5 | `matplotlib>=3.5` becomes a core dependency | Done; still imported only when a figure is drawn (`import statspai` does not load it, tested). seaborn and plotly stay in `[plotting]`. README, README_CN, docs index and getting-started updated. | None |
| 6 | `sp.binscatter`: keep `method='classic'` as default | With `controls` and `'classic'`, an `AssumptionWarning` says residualize-then-bin is not consistent for `E[y \| x, controls]` and suggests `method='binsreg'`; recorded in `bin_data.attrs['diagnostics']`. | None |
| 7 | Public reset of the session spec-search counter | `sp.reset_spec_counter(outcome=None)`, registered; callable over MCP by name or through `call_function`; the multiplicity violation's recovery hint names it. | None |
| 8 | MCP text block that duplicates `structuredContent` | Default stays `full` for client compatibility. Opt-in `STATSPAI_MCP_TEXT=summary` / `statspai-mcp --text summary` sends a few summary lines; `structuredContent` is unchanged. Measured on seven tools: 48% fewer bytes per response (102,490 to 53,076), about 12k tokens. Documented in `docs/dev/modules/agent_mcp.md`. | Revisit the default once the major clients are known to read `structuredContent` |
| 9 | ⚠️ `'rlasso'` alias in classifier roles becomes `sp.RlassologitClassifier` (`hdm` fits propensities with `rlassologit`) | Numbers change for that alias only; regressor roles keep `RlassoRegressor`; `'rlasso_lpm'` selects the old linear-probability classifier. CHANGELOG ⚠️ + MIGRATION. | None |
| 10 | `sp.balance_table` SMD of a categorical level follows cobalt's binary convention (`p(1-p)`, pooled) | Unreleased feature, so the existing CHANGELOG entry was updated rather than a ⚠️ added. Equal to `cobalt::bal.tab(binary = "std", s.d.denom = "pooled")` to 1e-12 on a fixture, including cobalt's whole-sample denominator when a level separates the arms (cobalt GPL, used as a black box). Numeric covariates, 0/1 numeric columns included, keep the released sample-SD denominator. | Possibly align numeric 0/1 columns too (would change released numbers, needs its own ⚠️) |
| 11 | This record | This file | |

## Decisions carried out by the other W5 lines

These were decided at the same time; the implementing lines record their
own evidence in CHANGELOG and the module files.

- **W5-cleanroom-pm.** `sp.panel_match` was written by an agent that had
  read R PanelMatch's GPL-3 source. Decision: clean-room rewrite from the
  method paper (`imai2023matching`), the package documentation and the
  committed tests and fixtures only, same public API, old files deleted.
  All existing tests, including the R parity at 1e-10 and the identity
  tests that reproduce R's three behaviours, must pass against the new code.
- **W5-cleanroom-ts.** The local-projection extensions of `612557ee`
  (IV-LP, explicit lag design, state dependence, panel LP) and the
  Dumitrescu-Hurlin panel Granger test were written after reading lpirfs
  and plm (GPL) source. Decision: clean-room rewrite from the papers and
  documentation, same API, parity tests at 1e-12. The multi-unit scpi
  commit `b7c85f1a` (local only, written after reading R scpi GPL source)
  is discarded unread and multi-unit `sp.scest` is reimplemented from the
  scpi paper, its documentation and R scpi outputs as a black box; the
  local branch and worktree are deleted.
- **W5-numeric.** `sp.sac` gets full-information standard errors (as
  `sp.gns` computes them) instead of the conditional diagonal-Hessian ones
  (⚠️, Stata 18 `spregress, ml` parity). `sp.margins` / avg slopes after
  `fepois` compute the delta-method SE the way Stata does after `poisson
  ... i.fe` / `ppmlhdfe`, and keep refusing other links without a Stata
  reference. Many-weak-instrument SEs use the instrument projection with a
  simulation coverage check; the R ivmodel discrepancy is documented as T4
  and not copied. `sp.breakdown_m` on the default C-LF relative-magnitude
  path uses the log-grid scan plus bisection (the sets are not nested in
  M). `sp.causal_impact` also propagates the variance-parameter
  uncertainty, target null coverage at least 93% in the hard design. The
  schema generator exports `Union[str, List[str]]` as string-or-array.
- **W5-new.** A two-stage randomized (saturation) interference estimator
  (Hudgens-Halloran direct, indirect, total and overall effects with the
  paper's variances, citation verified from two sources); `sp.spillover`
  stays as it is. MMQR fixed-effects quantile regression (Stata `xtqreg`)
  with the influence-function SEs, parity if within 1e-6 of `xtqreg`,
  otherwise the gap documented. `sp.from_stata` for `xtregar`, `xttobit`,
  `xtintreg`.
- **W5-upstream.** Issue and email drafts only (StataCorp, SSC authors,
  fect, PanelMatch, CMAverse), each with a minimal reproduction; Bryce
  sends them himself.

## Kept as they are

- **`sp.didregress(aggregate='dlang')` degrees of freedom.** The t
  reference uses the residual degrees of freedom of the cell regression,
  which is exact under normal cell errors. Stata's count, which leaves out
  the absorbed group effects, is available as `dlang_df='stata'` and
  matches Stata 18 to 1e-10. Default unchanged.
- **`sp.sumstats` prints stored variable labels by default.** It is what
  a Stata user sees with `summarize` on labelled data; `use_labels=False`
  prints names. Default unchanged.
- **`sp.bart_causal` priors follow the reference.** `method='bart'` uses
  the `dbarts::bart2` prior that bartCause runs (75 trees, equally spaced
  cut points); the 8-seed screen gap to bartCause is 0.02 posterior SDs.
  This is a stochastic screen (tier S), not parity. `method='bcf'` is
  unchanged.
- **`sp.xtintreg` real-arithmetic fast path rejected.** Expanding the
  complex step in real arithmetic was about 3 times faster but moved the
  standard errors by 1.5e-12 to 5.5e-12, below the numerical resolution of
  the central-difference Hessian but not bit-identical. Kept the current
  path for byte-identical outputs; it can be reconsidered if speed
  matters. The 0.5% `se(sigma_e)` gap to Stata's default fit stays a
  documented reference difference: Stata's analytic Hessian under adaptive
  quadrature is not the curvature of its own likelihood, and with
  `intmethod(ghermite) intpoints(100)` the two agree to 1e-10.
- **Documented T4 differences stay T4.** Where a reference implementation
  disagrees with an independent check (a second reference, an identity,
  or a uniquely identified design), StatsPAI keeps its number, documents
  the difference, and does not copy the reference. Examples from this
  cycle: R PanelMatch's three behaviours (reproduced only by identity
  tests, not adopted), R ivmodel's many-weak SEs, Stata `stteffects
  ipwra, atet` averaging, Stata `xtintreg` under adaptive quadrature, and
  CMAverse's CDE / INTref SEs. None of these is relabelled as parity.

## Release recommendation

Release the wave as **1.40.0**, a minor release, after every W5 line has
landed on main. The release carries new public functions, `FutureWarning`
deprecations and ⚠️ correctness fixes (the `'rlasso'` alias, `sp.sac`
standard errors and whatever W5-numeric changes), so it is not a patch.
The earliest release that may switch the deprecated defaults (items 2, 3
and 4) is 1.41.0, one minor release later.

Before tagging:

1. On a quiet machine, the full `pytest` suite and
   `pytest tests/reference_parity/ -q` on main; regenerate the parity index
   and schemas in that order; `python scripts/registry_stats.py --check`.
2. Both parity ledgers describe the tree (`parity-traces` hook), and
   `python scripts/trace_perf_path.py --check` says whether Track C needs
   re-timing.
3. CHANGELOG coherence: one `## [Unreleased]` section, ⚠️ entries matched
   by MIGRATION.md notes.
4. `python scripts/release_gate.py --fix-census` then `--check` in the
   release window; sdist file list checked against `git ls-files`.

Tagging and the PyPI / TestPyPI uploads need Bryce's authorization in the
session that does them (CLAUDE.md §9). No GitHub Release. The papers stay
on their anchors (JSS 1.39.1, DiD-JAE 1.39.1); after the release, run
`Paper-DiD-JAE/scripts/check_release_compat.py` and record whether the
papers' numbers still reproduce, without re-anchoring.
