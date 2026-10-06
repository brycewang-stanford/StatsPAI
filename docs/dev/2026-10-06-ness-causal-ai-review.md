# Ness, *Causal AI* (Manning, 2025): review

Source: the book's code and data at <https://github.com/altdeep/causalML>
(MIT licence), `book/` (13 notebooks for chapters 2 to 13) and `datasets/`.
The rest of that repository is material from the author's course and was
used only for one data set.

Method. The notebooks keep their printed outputs, so part of the answer key
is the book itself. The book works in pgmpy, y0, DoWhy and Pyro. For every
step that has an independent implementation in R, the step was also rerun
there (dagitty, bnlearn, pcalg, causaleffect) so that a number is compared
with two sources that do not share code. Each analysis was then redone with
the `sp.*` function a user would reach for. Where the two sides disagreed,
the first step was to find the first point of divergence (section 5.1 of
`CLAUDE.md`).

What kind of book this is. It is about the graph side of causal inference:
build a DAG, test it, fit a model on it, intervene on the model, reason
about counterfactuals, identify an estimand, and only then estimate. That
is a different probe from the econometrics textbooks. They found problems
in estimators. This one found problems in the layer that tells a user (or
an agent) *which* estimator is licensed, and that layer had been checked
far less.

About half of the book is outside StatsPAI's scope and was not adopted:
deep generative models in Pyro (chapters 2, 5, parts of 6 and 9, the second
half of 11) and causal models built from language models (chapter 13). The
book is also dated in places. It pins pgmpy 0.1.19 for one notebook because
later versions have a bug, DoWhy's refuters report p-values whose meaning
the text itself misreads (see below), and one of its DoWhy estimates is a
number that no weighting formula reproduces.

## What was wrong in StatsPAI

1. **`sp.dag` dropped every arrow written right to left.** `"X <- Z"` has
   no `->` in it, so the statement was skipped. `sp.dag("X <- Z; Z -> Y;
   X -> Y")` was a graph without the confounding arrow, and
   `adjustment_sets("X", "Y")` returned the empty set as valid. A chain
   with both directions, `"A -> B <- C"`, created a node named `"B <- C"`.
   A name on its own, an undirected edge and a typo such as `=>` were
   skipped without a word. Arrows now read either way and chain, a name
   alone declares a node, and anything that cannot be read raises. So does
   a cycle, which used to be accepted.

2. **A Graphviz specification kept its quotation marks.** The book writes
   its graphs in DOT for DoWhy (listing 11.1). `sp.dag` read the edges but
   named the nodes `'"Won Items"'`, quotes included, so no node matched a
   column. Quoted names, the `digraph { ... }` and `dag { ... }` wrappers
   and `{A B} -> C` groups are now read.

3. **`adjustment_sets` reported that no set exists when the smallest one
   had seven or more variables.** The search stopped at size six and
   returned an empty list, which means "not identified by adjustment".
   With eight observed confounders of X and Y the summary said "No valid
   backdoor adjustment set exists!" and the recommender answered "Not
   identifiable under the declared DAG". Existence is now decided exactly
   at any size: a valid set exists if and only if the ancestral one is
   valid [@vanderzander2019separators]. Small sets are enumerated as
   before. When none is that small, the ancestral set is pruned one
   variable at a time. dagitty returns the same sets on the two graphs in
   the reference fixture.

4. **`backdoor_paths` returned paths that do not enter the exposure.** It
   returned every path that is not causal. `X -> M -> C <- Y` leaves X
   along an arrow, so it is not a backdoor path, and it was listed as one.
   Three things followed. `classify_variable` called the mediator M a
   confounder. The parents of an M-bias collider were called confounders
   although the path between them is closed. `summary` printed every edge
   as `→` whatever its direction, so `X ← Z → Y` read as `X → Z → Y`.
   `path_status` now has three types (`causal`, `backdoor`, `noncausal`)
   and an `arrows` field with the path as drawn. A confounder is a
   non-collider on a backdoor path that no collider closes.

5. **`sp.identify` treated any node whose name starts with `U_` as
   unobserved.** Nothing documented this. For `U_rate -> X; U_rate -> Y;
   X -> Y`, `adjustment_sets` offered `{U_rate}` and `identify` answered
   "not identifiable". Only declared latents and bidirected edges count
   now.

6. **`sp.front_door` returned a shrunken estimate when the mediator did
   not vary in one arm.** In the book's data (listing 11.12) nobody with
   low engagement won an item. The estimator fits `Y ~ M` separately in
   each arm. In the arm where M is constant that regression has no M
   coefficient, least squares returned the minimum-norm solution, the
   coefficient was zero, and the effect came out as 153.81: the two-arm
   answer scaled by the share of the other arm. No warning. It now raises
   `IdentificationFailure`, because `E[Y | D = 0, M = 1]` has no data and
   the front-door formula needs it. `outcome_model="additive"` is the
   explicit way to extrapolate: one pooled regression without interaction.
   With a binary mediator and no covariates it is the product of two
   coefficients, which is what DoWhy's `frontdoor.two_stage_regression`
   computes, and it reproduces the book's 170.20560581290403 to 1e-12.

7. **`sp.pc_algorithm` lost edges, and searched sets PC does not search.**
   Two separate problems.
   - *Orientation.* Two colliders can claim one edge in opposite
     directions in a finite sample. Each orientation zeroed the reverse
     entry of the adjacency matrix, so the second one deleted the edge.
     The skeleton had it and the CPDAG did not. On a six-variable Gaussian
     design with n = 120 this happened in 85 of 200 draws (156 edges).
     The first collider in node order now keeps the edge, the clash is
     returned in `orientation_conflicts`, and the third Meek rule, which
     was missing, is applied.
   - *Skeleton.* Conditioning sets were drawn from the union of the two
     neighbourhoods, and the neighbourhoods changed as edges fell within
     a level. The docstring described PC and cited PC-stable; the code was
     neither. It ran tests that no version of PC runs and gave results
     that depended on column order. On the book's transportation data it
     removed the edge E - R that pcalg and bnlearn both keep. The search
     is now PC-stable [@colombo2014order], with the pair and subset order
     of `pcalg::skeleton`, so the separating sets are the same ones.

8. **`sp.fci` stopped its skeleton search at the first level that removed
   nothing.** An edge whose separating set was two sizes larger than
   anything found so far was never tested against it. With three common
   causes, `X <- {A, B, C} -> Y`, the spurious `X - Y` edge survived at
   n = 5000. FCI now shares the PC-stable search. Its edge labels were
   also drawn wrong at the left end: a bidirected edge printed as
   `X >-> Y` and a left-pointing arrow as `X >-- Y`. The marks behind the
   labels were right. With both fixed, the partial ancestral graph equals
   `pcalg::fci` edge for edge and mark for mark on three designs, one of
   them with two unobserved common causes.

9. **`sp.pc_algorithm` on categorical data said "At least 2 variables are
   required".** It keeps numeric columns only, found none, and reported
   the count. `sp.ges` and nine treatment-effect estimators answered a
   column of labels with numpy's "could not convert string to float". All
   now name the column and the way out.

10. **`DAG.test_implications` could not test categorical data.** It
    computes a partial correlation. The book's chapter 4 data are labels.

11. **`sp.ipw` said nothing when one arm's weights rested on a tenth of
    its units.** In listing 11.9 the controls have propensities between
    0.77 and 1 and an effective sample size of 106 out of 963. The
    estimate (-832.6 normalised) is far from the regression's 178.1, and
    the share of extreme propensities, which `violations()` did check,
    was under its threshold. The numbers themselves are right: the
    Horvitz-Thompson and Hajek estimates were recomputed by hand from a
    logit and agree. DoWhy prints +437.8 for "ips_weight" on the same
    data, a figure neither formula gives with a logit fitted by
    statsmodels or scikit-learn. `sp.ipw` now stores each arm's Kish
    effective sample size and warns when one falls below a fifth of the
    arm. The threshold was calibrated: on a logit design with one
    covariate it fires for the ATE at a slope of 3 and for the ATT at a
    slope of 2, and stays quiet below.

12. **The recommender wrote a call that could not run.** For names such as
    `In-game Purchases` it produced `sp.regress('In-game Purchases ~ ...')`,
    which the formula parser reads as a subtraction. Terms that are not
    identifiers are now written `Q("...")`. It also mentioned only the
    backdoor strategy when the graph licensed three. The front door and an
    instrument are now listed with their calls, as DoWhy's
    `identify_effect` lists them.

## What was missing

- **`sp.bayes_net`**: a discrete causal Bayesian network, the engine half
  of the book runs on (chapters 3, 6, 7, 9, 12). Tables estimated from data
  with an optional Dirichlet prior, or written by hand, or given as
  functions of the parents. Exact inference by variable elimination.
  `query(..., do=...)` is the truncated factorisation. `expectation` takes
  a utility per state, which covers the decision problems of chapter 12.
  `counterfactual` builds a twin network and is refused unless every
  non-root node is deterministic, because tables alone do not determine a
  counterfactual. A query whose answer depends on a parent configuration
  that never occurs in the data warns, which is positivity checked on the
  query actually asked.
- **`IdentificationResult.estimate(data)`**: the ID estimand evaluated on
  categorical data. It needs only observed columns, so it covers the front
  door and anything else with a latent confounder that `bayes_net` cannot
  fit. Treatment values whose answer depends on an empty cell come back
  missing, with a warning. An optional bootstrap gives intervals.
- **Estimand simplification**. The ID algorithm conditions each factor on
  all its predecessors. For the book's gaming graph it printed
  `P(In-game Purchases | Customization Level, Guild Membership, Player
  Skill Level, Side-quest Engagement, Side-quest Group Assignment, Time
  Spent Playing, Won Items)`. Conditioning variables that the graph
  d-separates from the factor are now dropped, leaving `P(In-game
  Purchases | Guild Membership, Player Skill Level, Time Spent Playing,
  Won Items)`.
- **`sp.refute`**: the four data-side refuters of DoWhy for any estimator
  (placebo treatment, dummy outcome, random common cause, data subset).
- **Chi-square and G tests** in `DAG.test_implications` and
  `sp.pc_algorithm`, one implementation for both (`dag/_ci_tests.py`).
- `DAG.noncausal_paths`, `DAG.format_path`.

## Reproduced

On the book's data (`tests/external_parity/test_ness_causal_ai.py`, 15
tests, skipped without `STATSPAI_NESS_DIR`):

| Listing | Quantity | Reference | StatsPAI |
| --- | --- | --- | --- |
| 3.4 | kernel P(T \| O, R), maximum likelihood | bnlearn | 1e-9 |
| 3.5 | same, Dirichlet prior | book: 0.7007299270072993 ... | 1e-9 |
| 3.7 | P(E \| T = train), P(E \| T = car) | book: 0.6162, 0.5586 | 4 decimals |
| 4.1 | d-separation, four statements | book | equal |
| 4.4 | chi-square, 30 rows | book: 1.1611111111111112, p 0.5595873983053805, df 2 | 1e-9 |
| 4.6 | 11 implications, chi-square and G | dagitty, bnlearn | 1e-9 |
| course | PC on two categorical data sets | pcalg | same CPDAG |
| 7.1 to 7.4 | adjusted observational effect | the book's experiment, -38.39 | -38.88 (`sp.aipw`, `sp.ipw`, `sp.g_computation(by_arm=True)`), inside sampling error |
| 9.9, 9.10 | Monty Hall, two counterfactuals | book: 0.6667, 1.0000, 0.6667 | 2/3, 1, 2/3 |
| 9.11 to 9.18 | femur counterfactual | closed form 169.42 / 169.94 by sex | `sp.SCM`: 169.66 |
| 10.3 to 10.6 | identifiable or not, backdoor and front-door estimands | y0 | same verdicts |
| 11.4 | backdoor set, front-door set, instruments | DoWhy | equal |
| 11.5 | regression estimate and interval | book: 178.0861711575792, [168.68114922, 187.4911931] | 1e-9 |
| 11.12 | front door, two-stage | book: 170.20560581290403 | 1e-9 with `outcome_model="additive"` |
| 12.5 | E(U \| X) and E(U(Y_x)), four values | book: 57000, 37000, 39000, 34000 | equal |
| 12.12 | Newcomb: E(U) under do(choice) given intent, four values | book: 51000, 50000, 951000, 950000 | equal |

On simulated data against R
(`tests/reference_parity/test_ness_causal_ai_parity.py`, in CI):
the implied independencies and their chi-square tests (dagitty), the same
tests and the G test (bnlearn), kernels, a do-query written out as a sum,
adjustment sets and instruments on two graphs (dagitty), identification
verdicts on seven graphs (causaleffect), PC on Gaussian data with its
separating sets and CPDAG (pcalg), PC on categorical data, and the FCI
graph with its edge marks on three designs.

`tests/test_ness_causal_ai_pass.py` pins each behaviour above on data built
in the test.

## Where StatsPAI differs on purpose

- **Refutation p-values.** DoWhy's placebo refuter printed "New effect
  -537.1, p value 0.0" for the weighting estimate in listing 11.17, and
  the text beside it reads the result as a pass. `sp.refute` reports the
  mean of the reruns, their central range, and a two-sided Monte Carlo
  p-value that the reruns are centred where they should be. It refuses a
  number of simulations too small to reject at the chosen level (20
  reruns cannot give p < 0.095). On the same data the placebo mean for
  `sp.ipw` is 0.26.
- **A shuffled outcome is a weak check.** Shuffling removes the
  confounding together with the effect, so a comparison of means passes
  it. The dummy outcome that keeps the confounding in place
  (`outcome_function=`, listing 11.18) is the one that catches a missing
  adjustment, and the documentation says so.
- **Chi-square degrees of freedom.** They are counted from the levels
  present in each stratum, as dagitty and bnlearn's `x2-adf` do. pgmpy
  agrees on the book's example. bnlearn's plain `x2` uses the nominal
  count and is not offered. No continuity correction is applied.
- **An untestable implication has no p-value.** dagitty returns p = 1 when
  no stratum has two levels of both variables. `test_implications`
  returns a missing p-value with `df = 0`. Inside PC the same case is "not
  rejected", as it must be for the search to proceed.
- **Clashing colliders.** On the fixture where two colliders claim one
  edge, pcalg's result has the later collider's orientation and bnlearn
  keeps the earlier one with a warning. StatsPAI keeps the earlier one and
  returns the clash. Skeleton and separating sets do not depend on this
  choice and match pcalg.
- **Front door.** The default still fits each arm separately, which allows
  the mediator's effect to differ by arm. DoWhy's two-stage version is the
  additive model and is available by name.

## Not adopted

- Pyro models, variational inference, normalising-flow SCMs, VAEs and
  transformer-based causal models (chapters 2, 5, 6.11 to 6.20, 9.19 to
  9.30, 11.20 to 11.33, 13). They need a probabilistic programming
  backend and are not estimation of a causal effect from tabular data.
- Parameter learning with a latent node by EM (listing 3.6). The book's
  own output is a table of 0.5 throughout: the latent's states are not
  identified without more structure.
- DoWhy's `add_unobserved_common_cause` as a refuter. `sp.sensemakr` and
  `sp.evalue` answer that question with interpretable parameters.

## Open items

1. **Counterfactual identification: done in a second round, see below.**
2. **Verma constraints.** The second chapter 4 notebook tests a functional
   constraint in a graph with a latent variable. Nothing in StatsPAI
   derives such constraints.
3. **Formula terms in backticks: done in a third round.** Backticks are
   translated to `Q("...")` in `core.utils.r_formula_idioms`, which every
   patsy-based entry point already calls, and the four parsers that read
   names on their own (IV, count models, `ppmlhdfe` fixed effects, `gam`
   smooths) accept the quoted form. The same probe found that `sp.iv`
   turned `Q("x-1")` into `Q("x+ -1")`. Thirteen entry points are pinned
   in `tests/test_formula_backticks.py`, each against the same fit on
   renamed columns.
4. **`sp.sensemakr` returns a dict**, not a result object with
   `.summary()`.
5. **FCI on harder graphs.** Three designs agree with pcalg mark for mark.
   None of them needs the discriminating-path rule or the possible-d-sep
   step, so those are still unchecked.
6. **`sp.notears`, `sp.lingam`, `sp.ges`, `sp.fci`** refuse non-numeric
   columns (now with a message that says so). Only PC learned to use
   them. `sp.icp` still answers a labelled column with numpy's error.
7. `DAG.adjustment_sets` enumerates up to size six and returns one pruned
   set above that. A graph with several minimal sets larger than six gets
   one of them.

## Second round: counterfactual identification

Listing 10.8 identifies the effect of treatment on the treated,
`P(A_{T=-t} = +a | T = +t)`, with y0's `idc_star`. The first round left
this open. `sp.identify_counterfactual(dag, event, given=)` now implements
ID* and IDC* [@shpitser2008complete], written from the paper.

What it returns. A verdict, the formula in interventional distributions,
and for each interventional term its observational formula from
`sp.identify` or the statement that it needs an experiment. When every
term has one, `.estimate(data)` evaluates the whole thing on categorical
observational data; `.evaluate(net)` evaluates it in a fully specified
`sp.bayes_net`.

Evidence.

- *Soundness.* In random structural models (three to five binary
  variables, random bidirected edges, random response functions) every
  formula returned was evaluated and compared with the counterfactual
  probability computed by enumerating the exogenous variables. Four seeds,
  1,092 identified queries, largest difference zero to rounding. A
  220-query version is in `tests/test_counterfactual_identification.py`.
- *The paper and the book.* The paper's worked example
  `P(y_x | x', z_d, d)` comes out as
  `sum_w P_{z,w}(y, x') P_x(w) / P(x')`, as printed there. The book's
  listing 10.8 gives the six factors y0 prints.
- *cfid* (R, GPL, used as a black box). On 1,117 random queries the two
  agree on 1,025. Of the 92 disagreements, 80 are queries cfid identifies
  and StatsPAI does not, 12 the reverse. The 12 are covered by the
  soundness check. For the 80 no general arbiter was built; the ones
  examined by hand are cfid's errors. It answers the probability of
  necessity `P(Y_{X=0} = 0 | X = 1, Y = 1)` on `X -> Y` with "0" (in the
  model `Y = X` it is 1), and `P(V1 = v, V1_{V0=v0} = v)` on `V0 -> V1`
  with `P_{v0}(v1)`, which two models with identical experiments
  contradict. Among StatsPAI's 250 refusals, one had a joint probability
  that was zero in every random model tried, and that one is zero only
  because the variables are binary. Eight textbook verdicts are pinned
  against cfid in the reference-parity file, and the necessity query is
  pinned as a documented difference.

Four things the paper's figure does not spell out, each found by the
soundness check failing:

1. In line 6 the term of a component fixes *everything* outside it. Giving
   each node only its own outside parents put nodes of one component in
   different worlds and the recursion did not terminate.
2. In line 9 a variable that is both set to `x` and observed at `x` leaves
   the subscript (consistency): `P(y_x, x) = P(y, x)`, not `P_x(y, x)`.
3. Line 8's conflict includes a variable set in one world and left natural,
   with no value known, in another inside the same component.
4. In IDC*, the test that moves evidence into the subscripts has to
   condition on the remaining evidence, has to be made on a world graph
   whose merges do not depend on the values in the query, and has to carry
   the moved value into the remaining evidence as well as the outcome.

Completeness is not established. The algorithm is complete in the paper;
this implementation is conservative where a value is a bound symbol, and
the comparison with cfid cannot settle it.

## Rerun

```bash
git clone --depth 1 https://github.com/altdeep/causalML /some/folder/causalML
export STATSPAI_NESS_DIR=/some/folder/causalML/datasets

# R answer keys (dagitty, bnlearn, pcalg, causaleffect, jsonlite)
Rscript tests/external_parity/ness_causal_ai_reference.R
Rscript tests/reference_parity/_fixtures/_generate_ness_causal_ai.R

pytest tests/external_parity/test_ness_causal_ai.py \
       tests/reference_parity/test_ness_causal_ai_parity.py \
       tests/test_ness_causal_ai_pass.py -q
```
