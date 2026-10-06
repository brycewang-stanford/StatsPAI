# Rosenbaum, *An Introduction to the Theory of Observational Studies*: what the book needs and what StatsPAI had

*2026-10-06. Worktree `wt/rosenbaum-itos`.*

## What was done

The material for this book is not a set of scripts. It is three R packages
by the author, with the book's data and worked examples in their help pages:
`iTOS` 1.0.3 (matching, balance, small tools), `weightedRank` 0.7.0
(sensitivity analysis in block designs) and `tightenBlock` 0.1.7. Every
exported function and every example was run in R 4.5.2, the data written to
CSV, and the same analysis done with StatsPAI on those bytes.

The user's note that the material might be dated did not apply. `weightedRank`
0.7.0 is from January 2026 and implements a paper that appeared in JRSS-B in
2025. The gap was on our side. Before this pass StatsPAI had Rosenbaum bounds
for matched pairs and one-to-one optimal matching by total distance. The book
works throughout with matched sets of one treated person and several
controls, built with fine balance, and analysed with statistics that
concentrate on the sets where an effect is visible. None of that was
available.

The numbers in this document are from that run on the book's data. The
committed test fixtures are simulated to the same shapes, because the
book's data are not ours to redistribute (the earlier textbook passes made
the same choice). The same test file runs on the book's data when it is
pointed at them, and all 130 comparisons pass there too. See "Rerun".

All three packages are GPL-2 and StatsPAI is MIT. Nothing was translated. The
implementations follow the papers (the two-stage moment formula for a
stratum, the separable and Taylor bounds, the two-criteria network) and are
organised differently from the R code. They were then compared with R as a
black box.

## Results by topic

| Book | R | StatsPAI | Outcome |
| --- | --- | --- | --- |
| Ch. 4, matching for a propensity score with a caliper and fine balance | `iTOS::makematch`, `addcaliper`, `addinteger` | `sp.two_criteria_match` (new) | Same optimal objective (0 and 0; 30 and 0 with two controls). |
| Ch. 5 to 6, the binge-drinking match with two control groups (`bingeM`) | `makematch` with eleven cost terms | `sp.two_criteria_match` | Cost matrices equal to 1e-10. Objective equal on truncated costs, lower on the costs as given. See "Three places where the reference is not the target". |
| Ch. 5, is the match as good as an experiment | `iTOS::evalBal` | `sp.balance_vs_randomization` (new) | The nine p-values of the matched sample equal to 1e-8 under each of five settings. Shares of better-balanced experiments agree within Monte Carlo error (screen, not parity). |
| Ch. 8, Table 8.1, HDL cholesterol in 406 sets of four | `weightedRank::wgtRank` | `sp.weighted_rank` (new) | 28 combinations of weight and Gamma, deviates equal to 1e-9. At Gamma = 5 the p-value bound is 0.943 (Wilcoxon), 0.177 (Quade), 0.015 (`u868`), 0.005 (`u878`). |
| Ch. 8, point estimates and intervals under bias | `wgtRankCI` | `sp.weighted_rank(estimates=True)` | 24 cases, ends equal to the 1e-5 R solves to. |
| Ch. 9, choosing among weights | `wgtRanktt` | `sp.weighted_rank(phi=[...])` | Joint p-value equal to 1e-8. |
| Ch. 10, amplification | `amplify` | `sp.amplify` (new) | Equal. |
| Ch. 11, design sensitivity and power | `estPower`, `noether` | `sp.weighted_rank_power`, `sp.noether_test` (new) | Equal at the observed sample size. `estPower` is wrong for other sizes (below). |
| Ch. 12 to 13, two control groups and evidence factors | `dwgtRank`, `ef2C`, `sensitivitymv::truncatedP` | `sp.weighted_rank(scores=, block_scale=)`, `sp.evidence_factors`, `sp.truncated_product` (new) | Equal. Combined bound 0.0121 at Gamma = 2.3, Upsilon = 1.45. |
| Blocks with two treated people (periodontal data) | `gwgtRank`, `gwgtRankC` | `sp.weighted_rank(treated=, conditional=True)` | Equal to 1e-9 for the conditional test, 1e-6 for the unconditional one (R goes through `BiasedUrn`). |
| Strata of any size; two-sample comparison | `iTOS::ev`, `evall`; `senstrat`, `sen2sample` | `sp.rosenbaum_stratified` (new) | Equal to 1e-9 against the exact method. |
| Tightening a block design | `tightenBlock::tighten` | `sp.tighten_blocks` (new) | Same objective on truncated costs. Drops 0, 6 and 13 pairs at the three subset prices of the package example, as R does. |
| Matched pairs | `DOS2::senWilcox`, `senU` | `sp.rosenbaum_bounds` (existing) | Already equal. Cross-checked against `sp.weighted_rank` on pairs. |
| One-to-one optimal matching | `optmatch::pairmatch` | `sp.optimal_match` (existing) | Same matched pairs and effect estimate on the binge data. |

## New functions

Ten, in two groups. The guide `docs/guides/observational_block_designs.md`
shows them in the order a study uses them.

**Design.** `sp.two_criteria_match`, `sp.tighten_blocks`,
`sp.balance_vs_randomization`.

**Analysis.** `sp.weighted_rank`, `sp.weighted_rank_power`,
`sp.rosenbaum_stratified`, `sp.noether_test`, `sp.evidence_factors`,
`sp.truncated_product`, `sp.amplify`.

Where the R packages have several functions for variants of one analysis,
StatsPAI has one function with arguments. `wgtRank`, `dwgtRank`, `gwgtRank`,
`wgtRankCI`, `wgtRanktt` and `gwgtRankC` are all `sp.weighted_rank`. The
seven cost builders of `iTOS` are term dictionaries in one call, which an
agent can write as JSON.

Three things go beyond the R packages.

1. `gamma_critical` is solved for. R users find it by trying values.
2. The Taylor bound and the exact single-stratum search are available from
   the same call as the separable approximation, and the moments are exact
   for strata of any size. On eight large strata the separable p-value is
   visibly smaller than the bound that does not rely on it.
3. The matcher is a compiled successive-shortest-path solver on the
   four-layer network. The book's largest match takes 0.5 seconds against 11
   in R, and uses the costs without rounding.

## Three places where the reference is not the target

Each is handled by the rule in `CLAUDE.md` section 5.1, step 3. We keep our
value, show the mechanism, rebuild the reference's number from our own
quantities, and give evidence from a third source.

### 1. `estPower` with `ssratio` other than 1

The power of the sensitivity analysis in a study with `s` times as many
blocks compares the statistic, whose standard deviation shrinks by
`sqrt(s)`, with a critical value whose standard deviation also shrinks by
`sqrt(s)`. `weightedRank::estPower` 0.7.0 divides the second by `s`. At
`s = 1` nothing changes and the two agree to 1e-13. At `s = 1000/406` on the
HDL data R reports power 0.217 at Gamma = 7 where the formula gives 0.075.

Evidence. A pilot of 4000 blocks and a target of 150 (`s = 0.0375`).
Simulated studies of 150 blocks reject 90%, 53% and 22% of the time at three
values of Gamma. StatsPAI estimates 90%, 57% and 25%. The R formula gives 0%
at all three. The test `test_power_estimate_tracks_simulated_power` repeats
this, and
`test_weighted_rank_power_sample_ratio_differs_from_estpower_by_one_term`
rebuilds R's numbers from ours by putting the `s` back.

This is worth reporting to the package author. A draft is at the end of this
file for Bryce to send or not.

### 2. `makematch` truncates costs

`rcbalance::callrelax` passes `as.integer(cost)` to the solver. A
Mahalanobis distance of 3.9 becomes 3. Penalties are integers and survive, so
the constraints of the design are honoured, but the choice among controls
that satisfy them is made on the integer part of the distance.

On the costs truncated the same way StatsPAI's optimum equals R's exactly
(153 for the never-binge match, 105,379 for the past-binge match). On the
costs as given StatsPAI finds 268.1 and 105,546.2. R's matches, priced at the
costs as given, cost 326.8 and 105,567.8. HiGHS solves the same network as a
linear program and reaches 105,546.2.

A consequence is that `sp.two_criteria_match` does not select the same 206
controls as the book's `bingeM`. It selects a set that is cheaper on both
distances (209.0 against 232.1 for pairing, 59.0 against 94.6 for balance). 143 of the 206 never-binge controls are common to both.
The parity test therefore compares objectives. It does not compare
membership, which is not unique even under R's own costs.

### 3. Two conditional tests

`wgtRankC` and `gwgtRankC` both condition on the extreme responses of a
block. They are the same test when no outcomes are tied and differ slightly
when some are (p = 0.0373 and 0.0385 on the HDL data at Gamma = 6, which has
30 blocks with tied extremes). `conditional=True` is `gwgtRankC`, the later
and more general of the two, which also handles several treated people per
block. `wgtRankC` is matched on the untied blood-pressure data.

## Things noticed and left alone

1. **Ties created by floating point.** The combined blood-pressure outcome is
   a sum of two ratios. Differences that are equal on paper differ in the
   fifteenth digit, so the signed-rank test sees 207 distinct absolute
   differences where there are 202. The p-value at Gamma = 2 is 0.04196 on
   the values written to CSV and 0.04185 on the values in memory. R behaves
   the same way and so does every rank test in every package. It is why the
   fixtures are read back from CSV on both sides. Whether rank functions
   should round before ranking is a question for a separate decision. It
   would change existing numbers.
2. **`sp.optimal_match(caliper=)` drops pairs after the assignment is
   solved.** That is documented behaviour and is not the same as a caliper
   that shapes the match. `sp.two_criteria_match` has the penalised caliper.
   The older function now points to it.
3. **`senstrat`'s default method is less exact than its documentation
   suggests.** `BiasedUrn` returns moments to 1e-7, and at Gamma = 1, where
   every candidate has the same expectation, that noise decides which
   variance is used. The effect is in the ninth digit of the deviate.

## Not done

1. **M-statistics for matched sets** (`sensitivitymw::senmw`, Huber scores).
   The book does not use them. `sp.rosenbaum_stratified(score="raw")` accepts
   scores computed elsewhere.
2. **Exact null distributions by convolution** (`iTOS::gconv`). The bounds
   here use the normal approximation, as the R functions do. For matched
   pairs `sp.noether_test` is exact.
3. **Nonbipartite matching**, which the periodontal example uses to pair
   pairs into blocks of four (`nbpMatching`). The fixture takes the blocks as
   given.
4. **Datasets.** The book's NHANES extracts are neither in `sp.datasets` nor
   in the test fixtures. They reach us through GPL-licensed packages. The
   underlying numbers are public federal survey data, so shipping them is
   probably defensible, but it is a licensing call that is Bryce's to make.
5. **Sparse networks.** Costs are dense treated-by-control matrices, refused
   above 60 million cells. Larger problems should be split by an exactly
   matched covariate first.

## Evidence

- `tests/reference_parity/test_rosenbaum_itos_parity.py`: 130 comparisons
  with R. Reference generator `_fixtures/_generate_rosenbaum_itos.R`, about
  a minute and a half on the simulated fixtures.
- `tests/test_rosenbaum_itos_pass.py`: 29 tests with no reference
  implementation in the loop. Level under randomization for three weights
  and for the conditional test. No more than 5% rejection when the data
  carry exactly the bias the analysis allows, alongside 90% rejection by the
  randomization test on the same data. Stratum moments against enumeration
  of every assignment, and the Rosenbaum-Krieger maximum against enumeration
  of every binary `u`. Optimal matches against enumeration of every
  selection and both assignments. Power against simulation.

## Rerun

On the committed, simulated fixtures (needs R with `iTOS`, `weightedRank`,
`tightenBlock`, `senstrat`, `sensitivitymv`, `DOS2`, `rcbalance`, `rlemon`,
`jsonlite`):

```bash
Rscript tests/reference_parity/_fixtures/_generate_rosenbaum_itos_data.R
Rscript tests/reference_parity/_fixtures/_generate_rosenbaum_itos.R
pytest tests/reference_parity/test_rosenbaum_itos_parity.py tests/test_rosenbaum_itos_pass.py -q
```

On the book's data, written to a directory outside the repository:

```bash
Rscript tests/reference_parity/_fixtures/_generate_rosenbaum_itos_data.R --book /some/dir
STATSPAI_ITOS_FIXTURES=/some/dir Rscript tests/reference_parity/_fixtures/_generate_rosenbaum_itos.R
STATSPAI_ITOS_FIXTURES=/some/dir pytest tests/reference_parity/test_rosenbaum_itos_parity.py -q
```

## Draft note to the `weightedRank` maintainer

Not sent. For Bryce to use, edit or drop.

> Dear Professor Rosenbaum,
>
> In `estPower()` of weightedRank 0.7.0 the noncentrality is computed as
>
> ```r
> ncp <- ((detail[3] - jackout$delm) + crit * sqrt(detail[4]) / ssratio) /
>   sqrt(jackout$delv / ssratio)
> ```
>
> `detail[4]` is the bounding variance of the mean statistic at the pilot
> sample size. With `ssratio` times as many blocks that variance is divided
> by `ssratio`, so its square root should be divided by `sqrt(ssratio)`. As
> written the critical value moves too far, and power is overstated for
> `ssratio > 1` and understated for `ssratio < 1`. With `ssratio = 1` the
> function is unaffected.
>
> A check that does not rely on the formula: draw 4000 blocks of three with
> Normal errors and a shift of 0.6 as the pilot and ask for the power at 150
> blocks (`ssratio = 0.0375`). Simulated studies of 150 blocks reject at
> Gamma = 1.8, 2.3, 2.8 about 90%, 53% and 22% of the time with `u868`.
> `estPower` returns essentially zero at all three. With `sqrt(ssratio)` it
> returns 90%, 57% and 25%.
>
> Thank you for the package and the book.
