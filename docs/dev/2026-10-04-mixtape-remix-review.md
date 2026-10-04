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

## Open

- `teffects ra` and `teffects ipwra` have no StatsPAI counterpart with
  Stata's standard errors. Another line of work is rewriting the
  `teffects` translation; this was left to it.
- `egen`, `reshape`, `expand`, `matrix`, loops: most lab do-files stop at
  one of these before the first estimation line. `egen` is the most
  frequent (215 uses). The `bysort` prefix is being added elsewhere.
- `did_imputation, allhorizons` is reported as not translated; the list of
  horizons is in the data and `sp.did_imputation(horizon=)` wants it
  written out.
- `csdid2` is not translated: its standard errors differ from `csdid`'s
  (a divergence recorded in an earlier review).
- `csdid`'s `pscoretrim()` default. The syntax line of csdid 1.81 reads
  `pscoretrim(real 1.0)`, and the guide says both sides default to 0.995.
  No lab or fixture has a control unit with a propensity score above
  0.995, so the two could not be told apart. Needs a design built for it.
- `allsynth` (bias-corrected synthetic control) and `makespline` are not
  translated; `sp.augsynth` covers the ridge-augmented estimator.

## Files

- `tests/reference_parity/test_stata_did_commands_parity.py`, with
  `_fixtures/_generate_did_commands_{data.py,Stata.do,R.R}` and their
  outputs.
- `tests/test_stata_translation_did.py`,
  `tests/test_bjs_pretrend_identification.py`.
