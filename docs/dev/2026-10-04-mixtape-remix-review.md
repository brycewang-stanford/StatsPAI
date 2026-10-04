# Cunningham, *Causal Inference: The Remix* workshop labs (2026): review

Source: `the-remix-tour-2026`, the companion materials of Scott Cunningham's
summer 2026 European workshops (decks, labs, shiny apps). The labs are the
part that runs: about 8,000 lines of do-files (and their R twins) on
difference-in-differences with covariates, staggered adoption, triple
differences, continuous doses and synthetic control. The Pisa and Berlin
folders are supersets of the others.

Method. Each real-data lab was run in Stata 18 MP (drdid 1.91, csdid 1.81,
jwdid, did2s, did_imputation, eventstudyinteract, honestdid) and the same
lines were run through `sp.stata` or the direct `sp.*` call. Simulation
labs draw from Stata's random-number generator and cannot be replayed
number for number; their estimation lines were checked for translation
only. Where StatsPAI and Stata disagreed the first step was to find the
first point of divergence, as §5.1 of `CLAUDE.md` asks.

## What was wrong in StatsPAI

1. **Time-varying covariates in `sp.callaway_santanna`** (Brazil lab, 14
   controls that change every year). The covariate was read from the
   unit's first row for every ATT(g, t). `csdid` and R `did` read it in the
   earlier period of each cell. Simple ATT 0.3985 (0.0613) before, 0.4028
   (0.0598) after, which is `csdid`'s; all 196 cells within 6e-6. R `did`
   2.3.0 agrees to 1e-9 on the committed fixture in the four combinations
   of base period and control group.
2. **Headline SE of `sp.etwfe_emfx` for the group, calendar and event
   aggregations.** The estimate was the mean of the rows and the SE that
   of the simple aggregate. Now the delta-method SE of that mean, equal to
   Stata's through the posted `e(V)` to 1e-9.
3. **`sp.did_imputation(pretrends=k)` with unidentified leads** (Baker lab,
   `pretrends(24)` on a panel with no never-treated units). The existing
   guard caught `k = 24` only. For `k` from 18 to 23 the singular Gram
   matrix was inverted and every lead came back with SEs between 800 and
   5,900. Stata omits `pre18` to `pre24`. StatsPAI now refuses and names
   the longest identified run; `pretrends=17` reproduces Stata's `pre1` to
   `pre17`.

## What was missing

- `csdid ..., method(drimp)`: `sp.callaway_santanna(estimator='drimp')`.
- `drdid`, `jwdid`, `csdid_estat` and the `estat` aggregations in the
  translator; `sp.estat(result, 'event' | ...)` behind them.
- `did2s`. Stata's command takes a treatment dummy, any first stage (the
  Medicaid lab uses region-by-year effects) and any second stage;
  `sp.gardner_did` took the first-treatment period and fitted unit and
  period effects. It now has the general form (`treat=`, `fe=`,
  `second_stage=`). Every `did2s` line of the labs replays: Baker's event
  study (41 coefficients, 8e-12 on the estimate, 4e-9 on the SE; `pre_24`
  is all zero and is reported as omitted on both sides) and the three
  Medicaid specifications (SEs within 6e-8; the estimates differ in the
  seventh digit because Stata demeans within `unit()` in single
  precision).
- `drdid ..., all`, the most common form in the labs (12 lines):
  `sp.drdid(est_method='all')`. Stata's panel table has a sixth row,
  `sipwra`, which StatsPAI does not compute.

## Agreement found, nothing to change

| lab | lines | agreement |
| --- | --- | --- |
| Lalonde, `drdid` five estimators | panel | 1e-14 (`drimp`) to 2e-6 (logit stopping rule) |
| TVA, `drdid` panel and repeated cross-sections, `rc1` | 8 | 1e-12 estimate, 2e-8 SE |
| TVA, `csdid` five `method()` values | 20 cells | 3e-12 estimate, 1e-8 SE |
| Baker, TWFE event study (`reghdfe`) | 41 coefficients | 2e-10 |
| Baker, `csdid, notyet` and four `csdid_estat` | 61 rows | 3e-8 estimate, 3e-7 SE |
| Baker, `jwdid` and `estat event` | 19 rows | 4e-12 |
| Baker, `did2s` static | 1 | 1e-12 |
| Baker, `did_imputation` horizons 0 to 17 and 17 leads | 35 | 5e-8 estimate, 7e-7 SE (the outcome is stored in single precision) |
| Medicaid, `did2s` event study | 17 rows | 1e-7 absolute on the estimate, 6e-8 SE |

## Things the labs do that Stata lets through

These are not StatsPAI findings, but a reader replaying the labs will meet
them, and in each case StatsPAI's answer differs from Stata's on purpose.

- `csdid y, ... ipw long2` and `csdid ..., dripw`. `csdid` takes the
  estimator in `method()`. A bare word falls into the `*` of its syntax
  and is ignored, so these lines run the default. The translation runs the
  default too and says so.
- `csdid` on a panel with no never-treated units and without `notyet`.
  Stata prints "Using Not yet treated data" and continues.
  `sp.callaway_santanna(control_group='nevertreated')` raises.
- `eventstudyinteract` in the Baker lab passes relative-time dummies that
  are not zero for the control cohort. The reported pre-treatment
  coefficients are about -30 on a design whose true pre-trend is zero.
  `sp.sun_abraham` builds the indicators itself and returns zeros.
- `honestdid, pre(1/2) post(3/4)` in the TVA lab counts the omitted base
  column of `e(b)`, so `post` points at the base period and at 1950; the
  intervals printed are `[0, 0]`. The positions should be `post(4/5)`.
- `csdid ..., long2` with `gvar = 1945` on decennial data, where 1945 is
  not a period, stops `csdid_estat event` with a conformability error.

## Replaying whole do-files

After the commands were in place each real-data lab was fed to `sp.stata`
whole, dropping a line whenever it was refused, to see what stops a file
before its estimates. What turned up was general do-file grammar, and
four more things that were wrong.

- `csdid` was translated with StatsPAI's propensity trimming (0.995).
  `csdid` 1.81 defaults to `pscoretrim(1)`. No lab has a control above
  0.995, so a design was built for it: four never-treated units with a
  covariate value deep in the treated range. First post-treatment cell
  1.4400 (0.7104) untrimmed, 1.3164 (0.3862) trimmed; StatsPAI gives both
  to 1e-8. The translation now writes `pscore_trim=1.0`.
- `reghdfe y i.x, absorb()` was translated to `sp.hdfe_ols('y ~ C(x) |
  ...')`, which does not parse. The TVA lab's first regression. Now
  `i.x`; `ib1940.x` needed `sp.hdfe_ols` to accept a base level, added.
- `collapse (first)` skipped missing values (that is `firstnm`).
- Stata's `did2s ..., unit()` (version 0.5) demeans the outcome variable
  in memory and leaves it demeaned. Not a StatsPAI matter, but it cost an
  hour: reference values for `teffects` taken after a `did2s` line in the
  same do-file were on the altered outcome. The fixture do-file runs the
  `teffects` block first and says why. The Medicaid lab calls `did2s`
  with `unit()` twice on the same outcome; the second call is harmless
  only because the unit effect is in the model.

Grammar added: `` `r(mean)' `` and `local m = r(mean)` (in Stata's
`%18.0g` text, checked on eight values), varlist ranges and wildcards in
`keep` / `drop`, `collapse (firstnm)`, `reshape wide` / `long`, `ib0. x`
with a space. `teffects ra` and `teffects ipwra` run through
`sp.g_computation(by_arm=True)` and agree with Stata to 1e-14.

Where each lab stands now:

| lab | runs to the end | what still stops it |
| --- | --- | --- |
| Lalonde | yes | |
| Medicaid | yes | |
| China WTO | no | `makespline` |
| TVA | no | `honestdid`, `matrix` |
| Baker estimators | no | `forvalues` loops, `eventstudyinteract`, `matrix`, `did_imputation, allhorizons` |
| Castle equivalence | all but one line | `reg gdiff post` on the two rows left by a `collapse`: no residual degrees of freedom. Stata prints the coefficient without a standard error; `sp.regress` refuses |
| Castle event study | no | `matrix b = r(table)` and the macros read from it |
| Triple difference | no | a simulation: `set obs`, `expand`, `rnormal` |

## The worked project (`pisa/claude`)

The Pisa and Berlin stops include a complete analysis written during the
course: national suicide-prevention strategies in 16 European countries,
1994 to 2010, with the R code and every output table committed. That is a
set of reference values on real data, so StatsPAI was run on the same
panel.

| estimator | R | agreement |
| --- | --- | --- |
| TWFE and Bacon decomposition | `fixest`, `bacondecomp` | exact; the three component weights and averages to 1e-12 |
| Callaway-Sant'Anna, outcome regression on unemployment (time-varying), universal base | `did` | simple, group and 17 event-time ATTs to 5e-7; also `dr`, unconditional and not-yet-treated variants |
| BJS imputation with a covariate | `didimputation` | 2e-9 estimate, 2e-11 SE |
| Sun-Abraham | `fixest::sunab` | `aggregation='fixest_att'` exact (3.0340, 1.3802); event-time estimates to 4e-9 |
| Synthetic DiD by timing group | `synthdid` | the three ATTs to the printed digit; placebo SEs differ by resampling |
| Ridge-augmented SCM, four treated units | `augsynth` | ATT to 3e-5, weights to 1e-5 |
| Outcome-only SCM, four treated units | `augsynth(progfunc = "None")` | weights to 1e-5 after the default changed, see below |

The Callaway-Sant'Anna rows are a second, independent confirmation of the
time-varying covariate fix above: unemployment changes every year, and
before the fix these numbers did not match.

**Outcome-only classic SCM: the default changed.**
`sp.synth(method='classic')` with no predictors treated each
pre-treatment period as a predictor, rescaled it by its range across
units and used an equal V. `augsynth`'s SCM fits the raw pre-treatment
path. For Ireland the old default had a pre-treatment sum of squares of
51.55 against 45.14 for the least-squares weights, and one donor weight
differed by 0.13.

The rescaling was removed, fourteen tests failed, and the change was
withdrawn: among the failures was
`tests/reference_parity/test_synth_rest_R_parity.py`, whose fixture was
built to reproduce the rescaled fit against R `Synth` (with `custom.v`
chosen to make the two problems identical), so the old default was a
convention with evidence behind it and not a slip. Only the docstring,
which called it the least-squares estimator, was corrected.

Bryce then decided the default should be the least-squares fit, since it
is what `augsynth` and `synthdid`'s `sc` return and what a nested V
converges to. It was done as its own change with a migration note:
`standardize_predictors=None` rescales covariates and special predictors
and leaves pre-treatment outcomes alone; `True` is the earlier
convention, and the R `Synth` fixture now asks for it by name, so that
evidence still stands for that option. The README's Proposition 99
example moved from -19.76 to -19.51 and the outcome-only Texas fit from
21,482 to 21,013.

Two defaults worth knowing when porting such a project, both documented
and neither changed: `sp.sun_abraham`'s overall estimate is the
equal-weighted average over event times unless
`aggregation='fixest_att'`, and its event-time SEs carry the cohort-share
term that `fixest` omits (`share_variance=False` reproduces `fixest`).

The remaining labs of the tour (triple differences, China's WTO entry)
are ordinary regressions on simulated or small data and exercise nothing
beyond `reg` / `feols`.

## Left out on purpose

- Loops (`forvalues`, `foreach`). `CLAUDE.md` rules them out for
  `sp.stata` and another line re-affirmed it with a test this week.
  Unrolling a loop over a literal range is mechanical, but the decision
  is not this pass's to reverse.
- `honestdid`. Its `pre()` and `post()` are positions in `e(b)`, omitted
  base columns included, and the TVA lab shows how easily they point at
  the wrong coefficient. A translation would inherit that. Call
  `sp.honest_did(result, ...)` on the fitted event study.
- `eventstudyinteract`. It takes the user's relative-time dummies as
  given; `sp.sun_abraham` builds them. The Baker lab's dummies are wrong
  for the control cohort, so translating the line faithfully would mean
  reproducing a wrong number.
- `csdid2`: its standard errors differ from `csdid`'s (recorded in an
  earlier review).
- `allsynth`, `makespline`, `matrix`, `expand`, `file`: output handling,
  simulation scaffolding, or commands seen in one lab.
- `did_imputation, allhorizons`: the horizons are in the data;
  `sp.did_imputation(horizon=)` wants the list.
- Stata's sixth `drdid, all` row on a panel, `sipwra`.

## Files

- `tests/reference_parity/test_stata_did_commands_parity.py`, with
  `_fixtures/_generate_did_commands_{data.py,Stata.do,R.R}` and their
  outputs.
- `tests/test_stata_translation_did.py`,
  `tests/test_bjs_pretrend_identification.py`,
  `tests/test_stata_remix_grammar.py`,
  `tests/test_scm_outcome_only_least_squares.py`.
