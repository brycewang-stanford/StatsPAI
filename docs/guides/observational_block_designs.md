# Matched designs and sensitivity analysis

This guide follows the order of work in Rosenbaum's *An Introduction to the
Theory of Observational Studies* (2025). Build the matched sample without
looking at outcomes. Check that it resembles an experiment on the covariates
you measured. Analyse the outcomes. Then ask how much bias from covariates you
did not measure it would take to change the conclusion.

| Step | Function |
| --- | --- |
| Build matched sets | `sp.two_criteria_match` |
| Tighten an existing block design | `sp.tighten_blocks` |
| Compare balance with a randomized experiment | `sp.balance_vs_randomization` |
| Test, estimate and bound under hidden bias | `sp.weighted_rank` |
| Plan: which test, how many blocks | `sp.weighted_rank_power` |
| Strata of unequal size, or no matching at all | `sp.rosenbaum_stratified` |
| Two control groups | `sp.evidence_factors` |
| Explain Gamma to a reader | `sp.amplify` |

## 1. Matching on two criteria

A match has two jobs that pull in different directions. Pairs should be close
on the covariates that predict the outcome. The two groups as a whole should
have the same distribution of every covariate, including ones that are too
sparse or too numerous to pair on. `sp.two_criteria_match` takes one distance
for each job.

```python
import statspai as sp

m = sp.two_criteria_match(
    df, "treated",
    ps=["age", "female", "education", "smoker", "bmi"],
    ratio=1,
    pair=[
        {"type": "near_exact", "on": "female", "penalty": 10000},
        {"type": "mahalanobis", "on": ["age", "education", "bmi"]},
    ],
    balance=[
        {"type": "integer", "on": "smoker", "penalty": 1000},
        {"type": "caliper", "on": "pscore", "width": (-1, 0.03), "penalty": 10},
    ],
)
print(m.summary())
matched = m.matched          # long format, one row per person, column `mset`
```

A term in `pair` is paid for each treated person and their own control. A term
in `balance` is paid in a second assignment of the same controls that is free
to ignore the pairing. It is therefore zero whenever the selected controls
*can* be lined up with the treated group on that covariate, whoever they are
paired with. A large `near_exact` or `integer` penalty in `balance` is fine
balance when it can be met and near-fine balance when it cannot.

The five term types are `mahalanobis` (rank based), `near_exact`, `integer`,
`caliper` and `quantile`. A caliper given as a pair such as `(-1, 0.03)` is
directional. It tolerates controls whose score is above the treated person's
and penalises the other direction, which helps when the treated group sits in
the upper tail of the propensity score.

Put a covariate in `pair` when it predicts the outcome and you want it equal
within sets. Put it in `balance` when you only need the groups to agree on it.
Put it in both when exact pairing is not always possible, so that the
mismatches that remain at least cancel.

`subset_cost` lets the match leave a treated person out at a price. Use it
when a few treated people have no acceptable control, and report how many
were dropped.

## 2. Is the match as good as an experiment?

```python
bal = sp.balance_vs_randomization(matched, "treated", covariates, n_sim=1000)
print(bal.table)
```

For each covariate the table gives a two-sample p-value in the matched
sample, and the share of 1000 simulated randomized experiments on the same
people that were better balanced. A good match beats most of them. The
p-values are a yardstick. They are not hypothesis tests, since nobody
believes the matched sample was randomized.

## 3. Outcome analysis and sensitivity

With one treated person and `J - 1` controls in each of `I` matched sets:

```python
res = sp.weighted_rank(
    "outcome", data=matched, treat="treated", block="mset",
    gamma=[1, 1.5, 2, 3], phi="u868", estimates=True,
)
print(res.summary())
res.gamma_critical           # the Gamma at which p reaches alpha
```

`gamma=1` is a randomization test. `gamma=2` allows two people in the same
set to differ by a factor of two in their odds of treatment because of
something unmeasured, and reports the largest p-value such a bias could
produce. `gamma_critical` is the number to report. The study is insensitive
to biases smaller than it.

### Choosing the weights

`phi` decides how much a matched set counts as a function of how spread out
its outcomes are.

| `phi` | What it is | When |
| --- | --- | --- |
| `"wilcoxon"` | every set counts the same | efficient with no bias, the most sensitive to bias |
| `"quade"` | weight proportional to the rank of the range | a mild improvement |
| `"u868"`, `"u878"` | little weight on the quiet sets | a sound default |
| `"u888"`, `"mixed"` | almost all weight on the most dispersed sets | large samples, effects that are not small |

In the book's alcohol and HDL cholesterol example the Wilcoxon analysis is
overturned by a bias of 3.5 and the `u878` analysis only by a bias of 6.1,
on the same 406 matched sets.

Do not try them all and report the best. Either choose before looking, or
pass a list. `phi=["u868", "u878"]` refers the larger of the two deviates to
their joint distribution, which costs very little because the two statistics
are highly correlated.

`conditional=True` uses only the largest and smallest outcome of each set. It
discards sets in which the treated person is neither, and in exchange can be
insensitive to much larger biases.

`sp.weighted_rank_power` estimates from the data in hand how these choices
would perform in a study of a given size. Use it to plan the next study, or
on a planning sample that the final analysis will not reuse.

### Explaining Gamma

`Gamma = 2` describes an unmeasured covariate that doubles the odds of
treatment and also determines the outcome. Few covariates are like that.
`sp.amplify(2, 3)` returns 5. The same analysis therefore covers a covariate
that triples the odds of treatment and multiplies the odds of a higher
outcome by five.

## 4. Other designs

**Sets of unequal size, or coarse strata.** `sp.rosenbaum_stratified` takes
any number of treated people and controls in each stratum. With
`strata=None` it is the sensitivity analysis for an unmatched two-group
comparison.

**Matched pairs.** `sp.rosenbaum_bounds` (Wilcoxon, sign, permutational t)
and `sp.noether_test` (a sign test on the pairs with the largest
differences).

**Two control groups.** If treated people can be matched to controls of two
different kinds, the comparison of treated with the first control and the
comparison of the second control with the other two are nearly independent
tests of the same hypothesis, open to different biases.
`sp.evidence_factors` runs both and combines them.

**Sets of several sizes.** Analyse each size with `sp.weighted_rank` and
combine the p-values with `sp.truncated_product`.

**A block design that needs one more covariate.** `sp.tighten_blocks` keeps
the treated person of each block and the controls that balance the new
covariate, without rematching from scratch.

## For agents

- Build the design first and analyse second. `two_criteria_match` and
  `tighten_blocks` do not accept an outcome.
- `weighted_rank` needs sets of equal size. On `MethodIncompatibility` fall
  back to `rosenbaum_stratified`.
- Report `gamma_critical` together with the p-value at `gamma=1`. A small
  p-value alone says nothing about unmeasured bias.
- A list of `phi` is the supported way to compare weights. Looping over
  `phi` and keeping the smallest p-value is not.
