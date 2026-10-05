# de Chaisemartin and D'Haultfoeuille, difference-in-differences textbook: review

Source: `cc_xd_didtextbook`, the SSC package (distribution date 2026-04-28)
with the data and solution do-files of the four applications of Clément de
Chaisemartin and Xavier D'Haultfoeuille's textbook on
difference-in-differences. Wolfers (2006) is a binary staggered design,
Moser and Voena (2012) a single treatment date with 7,248 units, Pierce and
Schott (2016) one period with a continuous treatment, Gentzkow, Shapiro and
Sinkinson (2011) a count treatment that moves in both directions. The
do-files are 763 lines and come without output.

Method. Each do-file was run in Stata 18 MP with the packages current on
2026-10-05 (`twowayfeweights`, `did_multiplegt_dyn` of 17 January 2026,
`did_multiplegt_old`, `did_multiplegt_stat`, `did_had`, `did_imputation`,
`eventstudyinteract`, `csdid`, `honestdid`, `pretrends`, `fect`,
`sdid_event`). The same estimators were then run in StatsPAI on the same
`.dta` bytes. Where the two disagreed the first step was to find the first
point of divergence, as §5.1 of `CLAUDE.md` asks. The book is a few months
old and two of its own lines no longer run: `sotable` stops on an option its
current version rejects, and `did_multiplegt_stat`'s help file is cited
under another name.

## What was wrong in StatsPAI

1. **`sp.did_multiplegt` on a non-binary treatment** (Gentzkow, the count of
   newspapers). The estimator signed every switcher from a baseline other
   than 0 as a switch off, whichever way it moved, and did not divide by the
   size of the change. It returned -0.00082. `did_multiplegt_old` returns
   0.0057790681 and the corrected estimator returns that to 1e-8, with both
   placebos. Binary treatments are unchanged.
2. **Sun and Abraham interaction weights on an unbalanced panel** (Wolfers:
   52 state-years have no divorce rate). The weight of a cohort at a
   relative time was its share of units. It is its share of the
   observations at that relative time, which is what the paper defines and
   what `eventstudyinteract` computes. Unweighted effect at relative time 0
   on the binned specification: -0.1399 before, -0.1054 after, Stata
   -0.105359896. With weights the share is now the share of weight at that
   relative time, so time-varying weights are handled too. Balanced panels
   are unchanged (the two shares coincide).
3. **`sp.sun_abraham(event_window=)` put the left-out relative times in the
   reference.** Restricting the Wolfers window to (-14, 12) moved the
   effect at relative time 0 from 0.246 to 0.502, because relative times 13
   to 19, where the effect is negative, had joined period -1 in the omitted
   category. The window now only selects what is reported
   (`window_rule='report'`). `window_rule='bin'` pools the ends, which is
   the textbook's specification and reproduces `eventstudyinteract` to nine
   digits; `window_rule='reference'` is the old behaviour.
4. **`sp.did_multiplegt_dyn(same_switchers=True)` did not hold the switchers
   fixed.** A switcher only had to be observed at every horizon. One that
   ran out of not-yet-switched controls at a long horizon still entered the
   short ones (518 and 509 switchers at the two horizons of the textbook's
   chapter 8 exercise; Stata has 509 and 509).
5. **`aggregation='switchers'` was not `Av_tot_eff` when the treatment
   change differs across horizons.** The reference divides the summed
   effects by the treatment changes in place at each horizon. For a binary
   treatment that stays switched the divisor is one and nothing changes.
6. **`controls=` in `sp.did_multiplegt_dyn`**, two things. The
   first-difference regression was fitted on the pre-switch periods of the
   groups that switch and never on the groups that do not, whose outcomes
   were then left unadjusted. And on a panel with holes the change between
   two rows several periods apart was used as a one-period change. Found
   while building the test panel, not on the book's data. On the test
   panel the first effect moves from 0.4611 to 0.5354 (Stata 0.535409313).
7. **`vce(hc2 clustvar, dfadjust)` was translated to `robust='hc2'`.** The
   cluster variable and `dfadjust` were dropped without a note. On Pierce
   and Schott the p-value is 0.136 with Stata's adjusted degrees of freedom
   and 0.131 without.
8. **`sp.fect` did not run on Moser and Voena** (7,248 units, 40 periods).
   The initial two-way fit built one dummy column per unit, a 17 GB matrix.
   Large panels now solve the same least-squares fit by alternating
   projections (equal to the dense fit to 1e-11 where both run). The
   fixed-effects counterfactual takes 0.4 seconds on the full panel and
   returns the TWFE coefficient, 0.2882615, as it should with one treatment
   date.
9. **`sp.fect` stopped at the iteration cap without saying so.**
   `model_info['converged']` was `False` and nothing warned. It warns now.
   This matters on exactly this panel; see the next section.

## The interactive fixed effects estimate on Moser and Voena is not a number

The book reports `fect ..., method("ife") r(2) tol(1e-4)`: ATT 0.2967. On
the same panel `sp.fect(method='ife', r=2)` returns

| `tol` | iterations | ATT |
| --- | ---: | ---: |
| 1e-3 | 133 | 0.277 |
| 1e-4 | 2,580 | 0.131 |
| 1e-5 | 12,237 | 0.051 |
| 1e-6 | 20,000 (cap) | 0.036 |

and with one factor it climbs from 0.436 to 0.564 over the same range. The
EM iterations barely move the fit from one step to the next (the outcome
is a patent count, mostly zeros, and the factors are weak), so a relative
tolerance of 1e-4 stops them long before the estimate has settled. Stata's
0.2967 and StatsPAI's 0.131 are two stopping points on the same slow path,
reached by two implementations of the same update with different
arithmetic; neither is the estimator's value. No fix on our side would
"match" here, and matching would mean nothing. What StatsPAI can do, and
now does, is warn when the cap is hit. Anyone using IFE on a sparse count
panel should tighten `tol` until the ATT stops moving and report that.

## What was missing

- **`sp.twowayfeweights`**, the authors' diagnostic and the first thing
  every chapter runs. `type='feTR'` and `'fdTR'`, controls, other
  treatments, the test that the weights are unrelated to a variable,
  observation weights. All nine calls of the do-files agree with Stata
  (counts of positive and negative weights, their sums, the coefficient,
  the random-weights regressions). `sp.twfe_decomposition` covered only a
  binary treatment that turns on once.
- **Non-binary treatments in `sp.did_multiplegt_dyn`**, which is all of
  chapter 8. Periods are ranked (the panel is four-yearly), cells where a
  group has been on both sides of its starting treatment are dropped,
  switchers are matched on the period-one level, the variance centres
  switchers within (baseline, switch date, treatment at the switch).
- **Analytic joint tests.** With `se_method='analytic'` the tests of joint
  nullity and of equal effects are Wald tests on the analytic covariance,
  as in the command. The 500 bootstrap replicates that used to be drawn for
  them are gone, which is where the "hundred times faster" of the docstring
  had been lost.
- **`design=`, `by_path=`, `normalized_weights=`** of `did_multiplegt_dyn`.
- **`sp.regress(dfadjust=True)`** for the Bell-McCaffrey degrees of freedom
  of HC2 and CR2 (Stata 18's `dfadjust`). The CR2 and CR3 fits now also
  carry their full covariance, so `sp.test` works on them.
- **Translations** of `twowayfeweights`, `did_multiplegt_dyn`, `did_had`,
  `did_multiplegt_old`.

## Agreement found, nothing to change

| application | commands | agreement |
| --- | --- | --- |
| Wolfers | weighted TWFE, event-study TWFE | printed digits |
| Wolfers | `did_multiplegt_dyn, effects(13) placebo(13) weight()` | 26 estimates and SEs, `Av_tot_eff`, the condition number 4286.8, joint p 1.226e-06 |
| Wolfers | `did_imputation, horizons(0/12) pre(13) autosample` | all 26 rows |
| Pierce and Schott | `did_had, effects(4) placebo(3)` | estimates, SEs, intervals, bandwidth, QUG test |
| Moser and Voena | static and event-study TWFE, `did_imputation` | printed digits |
| Moser and Voena | `did_multiplegt_dyn, trends_nonparam()` | 39 estimates and SEs, both joint tests |
| Moser and Voena | `honestdid, delta(rm) mvec(1)` | [-0.198, 0.223] against [-0.196, 0.221]; both are grid searches |

## Where StatsPAI departs from `did_multiplegt_dyn`, and why

On an unbalanced panel the command drops, for each period-one treatment, the
periods in which no group with that treatment is still unswitched. It then
drops the groups whose average post-switch treatment, over what is left,
equals their period-one treatment. The second rule is meant for the
`dont_drop_larger_lower` option; the source says it "can only arise" there.
It also fires here. County 30093 starts with three newspapers, has four in
1908 and three again in 1912. No other county with three newspapers is
unswitched in 1908, so that row is dropped; one is in 1912 (it was missing
in 1908), so that row stays. What is left of 30093 after its switch averages
three, and the county is discarded. With it go 1892 to 1904, when it was the
only not-yet-switched control for three counties that switched in 1904.

StatsPAI keeps the county as a control until it switches. 1,122 switchers
contribute to the first effect instead of 1,119, and the effect is 0.014548
against 0.014424. The evidence that this is the only difference: remove the
county from the input and every effect, placebo, standard error, switcher
count and test agrees with Stata (`test_gentzkow_event_study_on_the_command_s_sample`).
In the chapter 8 subsample a second county (53033) goes the same way. The
synthetic test panel keeps never-switchers at every starting level, so the
rule cannot fire there and the two implementations are compared on the same
sample. Worth reporting to the authors; a draft is at the end of this note.

## Left out on purpose

- **`did_multiplegt_stat`** (AS and WAS, chapter 8 section 6). With
  `exact_match` its placebo equals `did_multiplegt_old`'s, which
  `sp.did_multiplegt` reproduces, but its WAS (0.0057148) is a different
  weighting of the same comparisons than DID_M (0.0057791), and its
  standard errors are analytic. A new estimator, not a translation.
- ~~`sdtest`.~~ `sp.sdtest` and its translation landed on main from
  another line the same day; the book's line reproduces (f = 0.7764,
  p = 0.0008).
- ~~`sotable`.~~ Second round: `sp.uniform_bands` reports the sup-t test
  (`attrs['supt_pvalue']`) and takes `terms=`, the hand-built dummies of
  an `sp.regress` fit. Moser and Voena's 18 leads: p = 0.068, critical
  value 2.748 (`sotable`: 0.068 and 2.742 before it stops on an option
  error). Wolfers' 13 placebos: p = 0.608 (`sotable`: 0.607).
- **`csdid [weight=stpop]`** with a weight that varies over time.
  `sp.callaway_santanna` refuses such a weight, on purpose: the weight
  defines the unit's share of the target population. `csdid` runs. What it
  does with the weight in each two-period cell was not established.
- **`sdid_event`** (event-study synthetic control and synthetic DiD on
  7,248 units with 200 bootstrap replicates). The Stata run had not
  finished after two hours on a busy machine, so there is no reference to
  compare with.
- **`pretrends power`.** `sp.pretrends_slope_for_power` exists and is
  pinned against Roth's R package (Track A 76); it was not rerun on the
  book's six leads.
- **`trends_lin`, `predict_het`** of `did_multiplegt_dyn`, as before.

## Open

- ~~Standard errors of `did_multiplegt_dyn` with `controls=`.~~ Closed in
  a second round the same day. The analytic variance now has the term for
  the estimation of the covariate slopes (`U^{var,X}` of the companion
  paper), and the regression behind the option is weighted and fitted
  within `trends_nonparam` cells as the command's is. Six configurations
  agree with Stata to 5e-7 on estimates, standard errors, placebos and
  `Av_tot_eff`.
- **`design=` totals.** Path counts are the command's for the frequent
  paths; the command also counts switchers with an observed path and no
  estimable effect, so its total is larger (1,067 against 1,054) and its
  shares a little smaller.
- **`placebo_sign` of `sp.did_multiplegt`.** The default `'stata'` is the
  sign of `did_multiplegt_old, robust_dynamic`. A plain `did_multiplegt_old`
  run reports the opposite sign, which is `'r'`. The docstring said
  otherwise and is corrected; the default is left alone because changing it
  flips the sign of every placebo existing callers get.
- **`sdid_event` auto-installs.** Running it installed a dependency into
  the user's PLUS directory without asking, despite the local `adopath`.

## Files

- `src/statspai/did/twowayfeweights.py` (new), `did_multiplegt.py`,
  `did_multiplegt_dyn.py`, `sun_abraham.py`
- `src/statspai/regression/ols.py` (`dfadjust=`)
- `src/statspai/synth/fect.py` (large panels, convergence warning) and
  `tests/test_fect_large_panel.py`
- `src/statspai/agent/_translation/_stata_did.py`, `_stata.py`,
  `_stata_options.py`
- `tests/reference_parity/test_dcdh_textbook_stata_parity.py` with
  `_fixtures/dcdh_textbook_data.csv`, `dcdh_textbook_Stata.json` and their
  two generators: a synthetic panel, 162 Stata numbers, runs in CI
- `tests/external_parity/test_dcdh_did_textbook.py`: the book's data,
  skipped unless `STATSPAI_DCDH_TEXTBOOK_DIR` points at it
- `tests/test_stata_translation_dcdh.py`

To rerun the book's side: `net get cc_xd_didtextbook` from SSC, run each
`solution.do` with the `use` line pointed at the local data, and

    STATSPAI_DCDH_TEXTBOOK_DIR=/path/to/cc_xd_didtextbook \
        pytest tests/external_parity/test_dcdh_did_textbook.py

## Reported to the authors

Posted on 2026-10-05 as
<https://github.com/Credible-Answers/did_multiplegt_dyn/issues/179>, with a
43-group synthetic example and the textbook data. Everything in the issue
was rerun in Stata on the repository's version of the command (9 June
2026) and on the SSC version of 17 January 2026.

What Stata itself shows, without any StatsPAI number:

- Synthetic example. Group 41 switches at t=3 and group 42, same
  baseline, is unswitched then. As is, group 41 has no `Effect_1` (20
  switchers). Changing only group 42's treatment at t=5 gives it one (21
  switchers).
- Textbook data. Setting county 30093 to 4 newspapers in 1912 moves
  `Effect_1` from 0.01442443 (1,119 switchers) to 0.01454833 (1,122
  switchers). The second number is StatsPAI's on the data as distributed.

A correction to an earlier version of this note, which said that dropping
county 30093 by hand leaves Stata's output unchanged. It does not: Stata
then returns 0.01451880 with 1,119 switchers, a third number, not traced.
What holds is the statement in the section above: StatsPAI on the data
without the county equals Stata on the data with it.
