# Agent probe, 2026-11-13

What an agent with no knowledge of the source runs into. Two surfaces were
driven as a client would drive them, and every defect was written down
before any source file was opened.

- **MCP over stdio.** `python -m statspai.agent.mcp_server` from the
  worktree, a JSON-RPC client, and the server's own instructions
  (`detect_design`, `preflight`, `recommend`, fit with `as_handle`,
  `audit_result`, every `next_calls` item marked ready, sensitivity, plots,
  `bibtex`). Sixteen tasks: 2x2 DiD, staggered DiD, event study, IV, sharp
  RD, fuzzy RD, synthetic control, SDID, matching, DML, panel FE, a Stata
  do-file, R calls, a cleaning chain with `transform_data`, three broken
  designs (no pre-period, weak instrument, no overlap) and seven wrong tool
  choices.
- **Python self-description.** `sp.search_functions`,
  `sp.describe_function`, `sp.function_schema`, `result.next_steps()` and
  `to_dict()` only, on 29 functions outside the audited entry points. For
  each: required arguments against the signature, defaults, every enum
  value tried in a real call, the result class, the fields the card names.

Data were simulated with known effects, one CSV per design.

## Numbers

| | Round 1 (main at `f4a8f7b87`) | Round 2 (after the fixes) |
| --- | ---: | ---: |
| MCP tasks finished without reading source | 16 of 16 | 16 of 16 |
| Headline estimates and SEs equal to the direct Python call | 24 of 24 | same |
| Tool calls | 176 | 150 |
| Median calls per task | 11.5 | 9.5 |
| Median bytes per response | 4,535 | 4,453 |
| Largest non-image response | 105,415 (an error) | 46,443 (`synth_sensitivity`); largest error 6,372 |
| `next_calls` marked ready and dispatched | 24 | 43 |
| ... of which returned an error | 5 | 0 |
| ... of which ran but were unusable (no image twice, the sharp reduced form on a fuzzy design, a headline on a covariate, a null estimate after 291 s) | 5 | 0 |
| `next_calls` not ready (arguments missing) | 14 | 2 |
| Python functions probed / with a card or error defect | 29 / 8 | 29 / 1 (open: #43) |

Round 1 was exploratory: the calls include the errors made on the way
(wrong argument names, wrong tools). Round 2 replays the same sixteen tasks
along the path the server's instructions describe, so its call count is not
comparable one to one; the follow-up rows are. The two follow-ups still not
ready in round 2 are correct as they stand: `bacon_decomposition` after a
2x2 fit without a unit id, and a `callaway_santanna` card whose cohort
column `recommend` derived and the dataset does not have.

"Finished without reading source" means the analysis reached a correct
estimate using only what the server or the cards said. It does not mean
the surrounding guidance was right: the table below is what was wrong.

## Defects

Surface codes: M = MCP server, P = Python self-description, B = both.

| # | Surface | Defect | Outcome |
| ---: | --- | --- | --- |
| 1 | B | `violations()` after `sp.match` quoted the pre-matching SMD (0.499) as "after matching" (0.037) | fixed |
| 2 | B | `sp.audit`: OLS `feols` audited as IV; Sun-Abraham as generic | fixed |
| 3 | B | `sp.audit`: synth placebo / pre-fit and matching balance always `missing` though the fit holds them | fixed |
| 4 | B | `sp.audit`: two-period DiD told to run `sp.pretrends_test`, `sp.honest_did`, Bacon | fixed (still `missing`, with a reason and no function; Bacon `not_applicable`) |
| 5 | M | `audit_result` unchanged after the suggested checks ran | fixed (`_followup_evidence`, `sp.audit(evidence=)`) |
| 6 | M | `honest_did_from_result` ready on a 2x2 fit and on a fit without leads; fails | fixed (not offered) |
| 7 | M | IV follow-ups not ready although the formula names every role; `exog` never listed, so the filled call dropped the controls (2SLS 2.0295 against the fit's 1.9957) | fixed |
| 8 | M | Fuzzy RD follow-up `rdsensitivity` tabulated the sharp reduced form (2.2 to 2.4 against a LATE of 3.18) | fixed (`rdbwsensitivity`, `rdplacebo` with `fuzzy=`) |
| 9 | M | Synth follow-ups not ready for want of arguments the fit call had; SDID, match, feols, panel fits had no follow-ups | fixed |
| 10 | M | `recommend` synth call marked ready without `treatment_time` (a numpy scalar was dropped) | fixed |
| 11 | B | `recommend`: 2x2 shape routed to unit FE on the group indicator (headline became the coefficient on a covariate) | fixed |
| 12 | B | `recommend`: continuous regressor routed to `did_multiplegt` (291 s, null estimate) | fixed |
| 13 | B | `recommend`: LIML card dropped covariates; Anderson-Rubin card used `d=` / `z=` | fixed |
| 14 | B | `preflight`: unknown method returned PASS; `dml`, `synth`, `sdid`, `feols` unknown; matching PASS under perfect separation | fixed |
| 15 | M | `rdplot`, `overlap_plot`, `balanceplot` returned `"Figure(1000x700)"` | fixed (PNG image block) |
| 16 | B | `anderson_rubin_ci` result had no `to_dict`; MCP returned the method name only | fixed |
| 17 | M | `bibtex(keys=[citation_key])` unknown; suggested another paper | fixed |
| 18 | M | `robust_synth` attributed to the augmented SCM paper (follow-up text and citation map) | fixed |
| 19 | P | `wild_cluster_bootstrap` enum listed `normal` (refused) | fixed |
| 20 | P | `cic` schema: no default for `seed` (42) | fixed |
| 21 | P | `result_schema.fields` named constructor parameters that are not attributes (`power_val`, `tables`) | fixed (80 classes) |
| 22 | P | `KitagawaResult.next_steps()` carried the IV test's failure modes | fixed |
| 23 | P | Search: `Kaplan-Meier survival curve` ranked `kaplan_meier` fourth; `zero-inflated count model` missed `zinb` / `zip_model`; a long query lost `rosenbaum_bounds` | fixed |
| 24 | M | `match` on a continuous treatment: 105 KB error | fixed (message and a 1,500-character cap) |
| 25 | M | `remediation.fix` ended with unrelated category advice | fixed (`category_fix`) |
| 26 | M | `transform_data`: misspelt column and wrong `function` argument gave bare pandas / TypeError text | fixed |
| 27 | B | `sp.stata` on `use` / `import delimited`: "do the step in pandas" | fixed |
| 28 | M | `from_r(matchit(distance="glm"))` emitted a payload `match` refuses; failures had no `error_kind` | fixed |
| 29 | B | 2x2 `sp.did` card said `normal`; the interval uses t(n - 4) | fixed |
| 30 | M | HDFE result: `fitted_within` uncapped (76 KB reply) | fixed |
| 31 | P | `rosenbaum_bounds(data=)` without `pair_id`: bare `ValueError` | fixed |
| 32 | M | `call_function(stata, <several estimation commands>)` returns the last result only | open: `sp.stata` returns one object by design; needs a decision on a multi-result payload |
| 33 | B | `from_stata("xtreg ... i.year, fe")` gives `feols('... + C(year) | id')`, which needs pyfixest | open: the native HDFE engine has no factor terms |
| 34 | B | `feols` / `panel` / `liml` with several regressors: `estimate` is null (no focal term) | open: by design; the coefficients table is complete |
| 35 | B | `next_steps()` lists every registered failure mode as a step whether or not it occurred | open: wording / design of the generic card-derived steps |
| 36 | B | Family-inherited next steps on non-family results (`sp.regress('time ~ ...')` after `sp.cox`; RD steps after `sp.bunching`) | open: needs per-family review of `core/next_steps.py` |
| 37 | M | `plot_from_result` draws a coefficient plot for RD and synth results; `kind='rdplot'` is echoed but the picture is the same | open |
| 38 | M | `sensitivity_from_result` default (`evalue`) needs `sd=` on a mean difference; the hint says so | left: the error is actionable |
| 39 | M | `seed_injected` on calls with no stochastic step (2x2 `did`); card `seed_note` contradicts the replay line | open |
| 40 | M | Successful responses repeat `data_provenance` (about 0.5 KB) and `load_data` carries it twice | open: budgeted by owner decision of 2026-10-20 |
| 41 | M | `detect_design` proposes an RD candidate on an id or outcome column at the median | open: low-confidence candidate, costs bytes only |
| 42 | M | `synth` on a panel whose donor pool contains treated units returns an estimate with no flag | open: needs a treatment column to detect |
| 43 | P | `partial_identification` returns `CausalResult` or `BoundsResult` by method; card has no result class | open: return type should be one class (owner decision) |
| 44 | B | `sp.did_multiplegt` on a continuous treatment: minutes, null estimate, no error | open: outside the agent layer; `recommend` no longer routes there |
| 45 | P | `to_dict()` of `ARIMAResult` / `MEGLMResult` carries `_`-prefixed internals (scrubbed over MCP, not in Python) | `ARIMAResult` fixed the same day by the time-series probe; `MEGLMResult` open |
| 46 | B | `audit` asks for `robust_se` on a clustered `feols` fit (`missing`) | open |
| 47 | M | `synth_sensitivity` (now a ready follow-up of `synth`) answers with 46 KB | open: its donor-sensitivity table is 15 KB; needs a summary at `detail='agent'` |
| 48 | M | `sdid` audit lists pre-fit and placebo checks as `missing`: the SDID result stores neither under a name the audit reads | open |

## What was verified to be right

- All 24 headline numbers compared (estimates and standard errors of the
  fits in tasks 1 to 14) equal the direct `sp.<function>` call to the last
  printed digit.
- Weak instruments are flagged at every step (preflight WARN, `recommend`
  puts LIML first, a `weak_instrument` violation on the fit, audit failed).
- No pre-period: preflight FAIL and a `DataInsufficient` error with a
  correct hint from both `callaway_santanna` and `did`.
- Every wrong tool choice returned a taxonomy error whose hint named the
  right tool.
- Handle misses list the handles that exist.
- `transform_data` lineage is carried into every later result.

## Reproducing

The client, data generator and both rounds' call logs are not committed
(they are a few hundred lines of scratch code). The tests are the durable
part: `tests/test_agent_probe_nov13.py` (44 in-process statements) and
`tests/test_mcp_stdio_followups.py` (one server process, the follow-up
loops of five fits).
