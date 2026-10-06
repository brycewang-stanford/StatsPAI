# Ness, *Causal AI*, in StatsPAI

Robert Osazuwa Ness's book (Manning, 2025) teaches the graph side of causal
inference: draw the DAG, test it against data, fit a model on it, intervene
on the model, reason about counterfactuals, identify an estimand, and only
then estimate. It works in pgmpy, y0, DoWhy and Pyro. This guide maps the
chapters to the StatsPAI calls that give the same numbers and says where
StatsPAI makes a different choice.

The notebooks and data are at <https://github.com/altdeep/causalML>. They
are not redistributed with StatsPAI. The examples below read the CSV files
from its `datasets/` folder.

## Chapter by chapter

| chapter | the book computes | StatsPAI |
| --- | --- | --- |
| 3 | a DAG as a list of edges | `sp.dag("A -> E; S -> E; ...")`; a Graphviz `digraph { ... }` string is read as is |
| 3 | causal Markov kernels from data (pgmpy `fit`) | `sp.bayes_net(dag, df)`; `.cpt(node)` |
| 3 | the same with a Dirichlet prior | `sp.bayes_net(dag, df, prior=1)` |
| 3 | a conditional query by variable elimination | `net.query("E", evidence={"T": "train"})` |
| 4 | d-separation | `g.d_separated(x, y, {...})` |
| 4 | every independence the graph implies | `g.implied_independencies()` |
| 4 | chi-square tests of those on categorical data | `g.test_implications(df)` (`test="g-test"` for the likelihood ratio) |
| 6 | a structural model with assignment functions | `sp.bayes_net(dag, cpts={node: function})` for discrete models; `sp.SCM` for continuous ones |
| 7 | the effect an experiment would show, from observational data | `g.adjustment_sets(x, y)`, then `sp.aipw` / `sp.ipw` / `sp.g_computation(by_arm=True)` |
| 7 | graph mutilation, `do` | `g.do("X")`; `net.do(X="high")`; `net.query(..., do={...})` |
| 9 | counterfactuals on a parallel-world graph | `net.counterfactual(variables, evidence=, do=)` |
| 9 | abduction, action, prediction with continuous noise | `sp.SCM(...).counterfactual(evidence, intervention)` |
| 10 | is `P(Y \| do(X))` identified, and by what formula | `sp.identify(g, x, y)`: `.identifiable`, `.estimand` |
| 10 | the estimand evaluated on data | `sp.identify(g, x, y).estimate(df)` |
| 10 | a counterfactual query: the effect of treatment on the treated | `sp.identify_counterfactual(g, event, given=)` |
| 11 | backdoor, front-door and instrumental strategies | `g.recommend_estimator(x, y)` lists all the graph licenses |
| 11 | regression, stratification, matching, weighting | `sp.regress`, `sp.match`, `sp.ipw`, `sp.aipw` |
| 11 | double machine learning, T-learner | `sp.dml`, `sp.metalearner(learner="t")` |
| 11 | front door by two regressions | `sp.front_door(..., outcome_model="additive")` |
| 11 | instrument, regression discontinuity | `sp.iv`, `sp.rdrobust` |
| 11 | refutation tests | `sp.refute(estimator, df, y=, treat=, covariates=, method=)` |
| 11 | sensitivity to an unobserved confounder | `sp.sensemakr`, `sp.evalue` |
| 12 | expected utility of an action, seen and done | `net.expectation("U", evidence={...})` and `net.expectation("U", do={...})` |

Chapters 2, 5 and 13, the flow-based models of chapter 6, the image
examples of chapter 9 and the Bayesian neural model of chapter 11 are deep
generative models in Pyro. StatsPAI does not cover them.

## Build the graph, then test it

```python
import pandas as pd
import statspai as sp

df = pd.read_csv("datasets/transportation_survey.csv")   # six categorical columns
g = sp.dag("A -> E; S -> E; E -> O; E -> R; O -> T; R -> T")

g.implied_independencies()      # eleven statements, the list dagitty gives
tests = g.test_implications(df) # chi-square for labels, Fisher z for numbers
tests.sort_values("p_value").head()
```

Each row is one independence the graph commits to. `statistic`, `df` and
`p_value` equal `dagitty::localTests(type = "cis.chisq")`. `p_holm` adjusts
for testing them all. `cramers_v` is there because, as the book shows in
listings 4.8 and 4.9, a p-value shrinks with the sample while the size of
the departure does not: with enough rows every such test rejects a graph
that is only approximately right.

## Fit the graph and intervene on it

```python
net = sp.bayes_net(g, df)
net.cpt("T")                                  # P(T | O, R)
net.query("E", evidence={"T": "train"})       # seeing
net.query("T", do={"E": "uni"})               # doing
```

A table can also be written by hand, or given as a function of the parents.
That is how a decision problem or a structural model is entered:

```python
net = sp.bayes_net(
    "C -> X; C -> Y; X -> Y; Y -> U",
    cpts={
        "C": {"bear": 0.5, "bull": 0.5},
        "X": {"bear": {"debt": 0.8, "equity": 0.2},
              "bull": {"debt": 0.2, "equity": 0.8}},
        "Y": {("bear", "debt"): {"failure": 0.3, "success": 0.7},
              ("bull", "debt"): {"failure": 0.9, "success": 0.1},
              ("bear", "equity"): {"failure": 0.7, "success": 0.3},
              ("bull", "equity"): {"failure": 0.6, "success": 0.4}},
        "U": lambda Y: -1000 if Y == "failure" else 99000,
    },
)
net.expectation("U", evidence={"X": "debt"})   # 57000: what debt-financed firms earn
net.expectation("U", do={"X": "debt"})         # 39000: what choosing debt earns
```

Keys with several parents are tuples in the alphabetical order of the
parents' names (`("C", "X")` above).

Two things the network tells you that a table of numbers would not:

- A parent configuration that never occurs in the data has no frequencies.
  A query whose answer depends on one warns that positivity fails, and
  `net.unseen` counts them. `prior=1` makes the smoothing explicit.
- `net.counterfactual(...)` is refused unless every node other than the
  roots is a deterministic function of its parents. Tables alone do not
  say how one unit's value would change. Move each node's noise into a
  root and give the node a function, as the book does for Monty Hall.

## Identify, then evaluate

When a confounder is unobserved the network cannot be fitted. The estimand
can still be evaluated if the effect is identified:

```python
g = sp.dag("X -> M -> Y; U -> X; U -> Y", latent=["U"])
res = sp.identify(g, "X", "Y")
res.estimand        # 'sum_{M} [P(M | X) * sum_{X'} [P(X') * P(Y | M, X')]]'
res.estimate(df)    # P(Y | do(X)) from the observed columns only
```

Rows come back missing, with a warning, for treatment values whose formula
needs a conditional probability that has no data behind it.

## Counterfactual queries

The effect of treatment on the treated compares what the treated got with
what they would have got untreated, `P(Y_{X=0} = y | X = 1)`. The evidence
and the outcome live in different worlds, so `sp.identify` does not apply.

```python
g = sp.dag("T -> W -> A; B -> V -> A; C -> T; C -> A; C -> B")   # listing 10.7
res = sp.identify_counterfactual(g, [("A", 1, {"T": 0})], given=[("T", 1)])
print(res.summary())
res.estimate(df)        # when res.from_observational_data
```

An event is `(variable, value)` for the world as it is and `(variable,
value, {intervened: value})` for a world in which something was set. The
result gives the formula in interventional distributions, says for each
term whether observational data can replace the experiment, and evaluates
it.

Some counterfactuals no experiment can settle. The probability of
necessity, `P(Y_{X=0} = 0 | X = 1, Y = 1)`, needs the joint behaviour of
`Y` in two worlds:

```python
sp.identify_counterfactual(sp.dag("X -> Y"), [("Y", 0, {"X": 0})],
                           given=[("X", 1), ("Y", 1)]).identifiable   # False
```

For those, state the structural model (`sp.bayes_net(...).counterfactual`,
`sp.SCM`) or bound the quantity.

## One graph, three strategies

The book's online-game example (chapter 11) is identified three ways at
once. Ask the graph:

```python
g = sp.dag(causal_graph, latent=["Prior Experience"])   # the book's DOT string
rec = g.recommend_estimator("Side-quest Engagement", "In-game Purchases")
print(rec.summary())
```

The recommendation is the backdoor adjustment, and the alternatives list
the front door through `Won Items` and the instrument, each with a call
that runs. The strategies rest on different assumptions, so agreement among
them is evidence for the graph and disagreement is evidence against it. On
the book's data:

| strategy | call | estimate |
| --- | --- | --- |
| backdoor, regression | the recommended `sp.regress(...)` | 178.09 |
| backdoor, doubly robust | `sp.aipw(df, y=, treat=, covariates=)` | 177.79 |
| backdoor, weighting | `sp.ipw(...)` | -832.60, with a warning |
| front door | `sp.front_door(..., outcome_model="additive")` | 170.21 |
| instrument | `sp.iv(...)` | first-stage F of 1.3: not usable |

The weighting estimate comes with a warning that the controls have an
effective sample size of 106 out of 963. The front door with the default
`outcome_model="by_arm"` raises, because nobody with low engagement won an
item and `E[Y | D = 0, M = 1]` has no data. The additive model extrapolates
under no interaction and should be reported as resting on that.

## Refute

```python
X = ["Guild Membership", "Player Skill Level", "Time Spent Playing"]
sp.refute(sp.aipw, df, y="In-game Purchases", treat="Side-quest Engagement",
          covariates=X, method="placebo_treatment", seed=0).summary()
```

| `method=` | what is altered | the reruns should centre on |
| --- | --- | --- |
| `"placebo_treatment"` | treatment shuffled across units | zero |
| `"dummy_outcome"` | outcome shuffled, or replaced by `outcome_function(df)` plus noise | zero |
| `"random_common_cause"` | a random covariate added | the original estimate |
| `"data_subset"` | random subsets of the rows | the original estimate |

A shuffled outcome is a weak check, because shuffling removes the
confounding along with the effect. The check that catches a missing
adjustment keeps the confounding in place:

```python
sp.refute(estimator, df, y=..., treat=..., method="dummy_outcome",
          outcome_function=lambda f: 100 * f["Guild Membership"] + 50 * f["Player Skill Level"])
```

None of these detects an unobserved confounder. For that use
`sp.sensemakr` or `sp.evalue`.

## Learn the graph

```python
out = sp.pc_algorithm(df, ci_test="chi-square")
out["edges"], out["undirected_edges"], out["orientation_conflicts"]
```

The search is PC-stable and returns the skeleton and separating sets of
`pcalg::pc`. `orientation_conflicts` lists colliders the tests imply but
that could not be drawn because another collider had already oriented one
of their edges the other way. A non-empty list says the sample contradicts
itself about those edges.

## Where StatsPAI differs from the book's tools

- The refutation p-value asks whether the reruns are centred where they
  should be, and a number of simulations too small to ever reject at the
  chosen level is refused.
- `sp.ipw` reproduces a logit-based Horvitz-Thompson or Hajek estimate
  exactly. On the book's data that is not the figure DoWhy prints.
- The front door fits each treatment arm separately by default. DoWhy's
  two-stage version is `outcome_model="additive"`.
- An implication that no stratum can test has a missing p-value, not a
  p-value of one.

The full account of the comparison is in
`docs/dev/2026-10-06-ness-causal-ai-review.md`.
