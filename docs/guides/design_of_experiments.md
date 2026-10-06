# Design of experiments: factorial, optimal and space-filling designs, sensitivity analysis

`sp.randomize`, `sp.power` and `sp.optimal_design` answer who gets
treated and how many units are needed. The functions here answer a
different question: which combinations of factor levels to run. That
question comes up in a multi-arm field experiment, in a conjoint or
factorial survey, and whenever the "experiment" is a computer model that
is run at chosen parameter values, such as a structural model, a
simulation study or the objective of a simulated-moments estimator.

The functions follow V. Roshan Joseph, *Experimental Design for Data
Science and Engineering* (2025, bib key `joseph2025experimental`), and
were checked against the R packages that book uses. What was compared and
how closely is in the last section.

| Question | Function |
| --- | --- |
| Which runs for several two-level factors, all or a fraction | `sp.factorial_design` |
| Which effects are real in a design without replication | `sp.factorial_effects` |
| How confounded is a design I already have | `sp.design_aberration` |
| Factors are shares that sum to one | `sp.mixture_design` |
| I know the model; where should the runs go | `sp.doe_optimal` |
| I do not know the model; spread the runs evenly | `sp.space_filling` |
| Add a second batch, or validation runs | `sp.design_augment` |
| How good is this design at filling the region | `sp.design_criteria` |
| Which inputs of my model matter | `sp.morris_screening`, `sp.sobol_indices` |
| A few points that stand in for a distribution | `sp.support_points` |
| A test set that looks like the whole data | `sp.split_data` |
| Minimise an expensive function in few evaluations | `sp.sequential_design` |

```python
import numpy as np
import pandas as pd
import statspai as sp
```

## Factorial experiments

Four factors at two levels in eight runs instead of sixteen. The fourth
factor is assigned to the three-way interaction of the others.

```python
d = sp.factorial_design(
    ["price", "ad", "pack", "ship"], generators=["ship = price*ad*pack"]
)
print(d.summary())
```

```text
Factorial design: 2^(4-1) fractional factorial, resolution 4
Runs: 8    Factors: 4
Generators: ship = price:ad:pack
Defining relation: I = price:ad:pack:ship
Resolution: 4
Aliased effects:
  price:ad = pack:ship
  price:pack = ad:ship
  price:ship = ad:pack
```

Resolution 4 means that main effects are clear of two-factor
interactions, and two-factor interactions come in pairs that cannot be
told apart. Without generators, `sp.factorial_design(k, n_runs=...)`
searches for the fraction with minimum aberration. That search is
exhaustive and is refused beyond about nine factors in 32 or 64 runs;
give the generators of a tabulated design there.

`d.design` holds the runs coded -1 / +1. Pass `{name: [low, high]}` to
get them in real units, `randomize=True` for a run order, and
`center_points=` to add centre runs.

Eight runs and seven effects leave nothing to estimate the error
variance with. A regression reports coefficients and no standard errors.
`sp.factorial_effects` judges the effects against each other.

```python
run = d.design.copy()
rng = np.random.default_rng(7)
run["sales"] = (50 - 6 * run["price"] + 4 * run["ad"]
                + 3 * run["price"] * run["ad"] + rng.normal(0, 1, 8))
fit = sp.factorial_effects(run, "sales")
print(fit.summary())
```

```text
            effect    coef  lenth_t   lenth_p  active
price        -11.8  -5.898   -32.01 6.429e-05    True
ad           8.345   4.173    22.65 0.0002214    True
pack         0.106 0.05302   0.2878     0.807   False
ship        0.6828  0.3414    1.853   0.08256   False
price:ad     7.078   3.539    19.21 0.0003429    True
price:pack  0.2655  0.1328   0.7206    0.4169   False
price:ship  0.2258  0.1129   0.6127    0.6105   False
Lenth: PSE = 0.3685, margin of error = 0.8479, simultaneous = 1.807
Aliased: price:ad = pack:ship
```

`effect` is the mean response at the high level minus that at the low
level, twice the regression coefficient. Lenth's pseudo standard error is
a robust scale of the effects themselves, valid when most of them are
noise. The margin of error comes from a simulated null distribution, as
in Lenth's own R package `unrepx`. The t approximation of the 1989 paper
is available with `reference="t"`; with seven effects its margin is 1.6
times as wide. `fit.plot()` draws the half-normal plot.

The `price:ad` row is the sum of two interactions. The design cannot say
which one is real, and `fit.aliases` records that. With replication or
centre points the usual `se`, `t` and `p` columns appear next to Lenth's.

A design from elsewhere, with factors at any number of levels, is
assessed with `sp.design_aberration`. It returns the generalized word
length pattern. `A1 = A2 = 0` means every pair of factors is balanced,
and `A3` then measures how much main effects are confounded with
two-factor interactions.

## Mixture experiments

When the factors are shares (a portfolio, a budget, a blend) the region
is a simplex.

```python
sp.mixture_design(["stocks", "bonds", "cash"], degree=2).design
sp.mixture_design(3, kind="simplex_centroid").design
sp.mixture_design(3, kind="space_filling", n=11, seed=1,
                  constraint=lambda f: f["x1"] + f["x2"] < 0.7).design
```

The first two are the classical designs for a polynomial in the shares.
The third spreads the runs evenly over the part of the simplex that
satisfies a constraint.

## Optimal designs for a known model

When the model to be fitted is known, the runs can be placed where its
parameters are estimated best. For a quadratic demand curve on prices
from 0 to 10 with nine runs:

```python
sp.doe_optimal("p + I(p**2)", {"p": (0, 10)}, n=9, seed=1).design
```

```text
    p  n
0   0  3
1   5  3
2  10  3
```

Without `n` the function returns the approximate design, support points
with weights, and certifies it with the equivalence theorem. An
`efficiency` of 1 means that no design on the candidate set does better.

A nonlinear model is written with its parameters in braces, in the syntax
of `sp.nls`. Its information matrix depends on the unknown parameters, so
a guess is needed.

```python
logistic = "{a} / (1 + exp(-{b} * (p - {c})))"
sp.doe_optimal(logistic, {"p": (0, 10)},
               params={"a": 100, "b": 1.2, "c": 5}).design
```

```text
       p  weight
0  4.114  0.3333
1  5.845  0.3333
2     10  0.3333
```

Three parameters, three points. That design is the best one if the guess
is right, and it leaves no way to check the model. Averaging the
criterion over a range of parameter values spreads the runs.

```python
sp.doe_optimal(logistic, {"p": (0, 10)}, params={"a": 100},
               prior={"b": (0.6, 2.0), "c": (3.5, 6.5)}, seed=1).design
```

```text
       p  weight
0  3.430  0.1661
1  4.538  0.1791
2  5.519  0.1840
3  6.691  0.1780
4     10  0.2928
```

`criterion="A"` and `"I"` minimise the average variance of the estimates
and of the predictions. `family="binomial"` or `"poisson"` treats the
expression as the linear predictor of a logit or Poisson model.
`candidates=` takes a table of admissible runs instead of a box, which is
how qualitative factors (`"C(g) + x"`) and irregular regions are handled.

An optimal design is optimal for the model it was given. If the
functional form is in doubt, add space-filling runs with
`sp.design_augment`.

## Space-filling designs

For a computer model nothing is known in advance about where the response
changes, and there is no noise to average out by repeating a run. The runs
should then cover the region evenly, and keep doing so when only some of
the factors turn out to matter.

```python
sf = sp.space_filling(
    30, {"beta": (0.90, 0.99), "gamma": (1, 5), "rho": (0, 0.95)}, seed=1
)
sf.design.head()
sf.criteria
```

The default is a maximum projection design. It fills the full region and
every projection onto a subset of the factors. Other choices of `method`:

- `"maximin"` maximises the smallest distance between two runs.
- `"uniform"` minimises a discrepancy and suits numerical integration.
- `"lhs"` is a random Latin hypercube, the cheap baseline.
- `"sobol"` and `"halton"` are scrambled low-discrepancy sequences.

On the MaxPro criterion (smaller is better) the 30-run design above
scores 25.6. A random Latin hypercube of the same size scores 98.

A region that is not a box is handled by `constraint=`, a function that
says which candidate runs are feasible. `sp.design_augment(design, n_new)`
adds runs to an existing design and leaves the old ones in place, which is
what a second batch or a set of validation runs needs.

## Which inputs matter

A model `y = f(x1, ..., xp)` with uncertain inputs. The steady-state
capital stock of a growth model serves as an example.

```python
def model(d):
    r = 1 / d["beta"] - 1
    e = 1 / (1 - d["alpha"])
    return (d["alpha"] * d["tfp"] / (r + d["delta"])) ** e

inputs = {"beta": (0.94, 0.99), "delta": (0.05, 0.10),
          "alpha": (0.30, 0.40), "tfp": (0.9, 1.1)}
```

Morris screening is the cheap first pass, 100 evaluations here.

```python
sp.morris_screening(model, inputs, r=20, seed=1).effects
```

```text
          mu  mu_star  sigma   share
beta   6.348    6.348  4.101  0.3462
delta -4.718    4.718  3.564  0.2574
alpha  5.166    5.166  1.819  0.2818
tfp    2.102    2.102  1.066  0.1146
```

`mu_star` ranks the inputs. A `sigma` that is large relative to `mu_star`
says the effect of that input is non-linear or depends on other inputs.

Sobol' indices split the variance of the output among the inputs.

```python
sp.sobol_indices(model, inputs, n=2048, seed=1).indices
```

```text
        first  first_lower  first_upper   total  total_lower  total_upper
beta   0.3129       0.2359       0.3772  0.3848       0.3492       0.4190
delta  0.2817       0.2118       0.3518  0.3433       0.3082       0.3729
alpha  0.2667       0.1866       0.3235  0.3099       0.2823       0.3395
tfp    0.0533      -0.0266       0.1186  0.0563       0.0518       0.0613
```

The first-order index is the share of the variance an input explains on
its own. The total index adds its interactions. An input with a total
index near zero can be fixed at any value. Inputs with other
distributions are given as `scipy.stats` objects in place of the bounds.

Two cautions. The decomposition assumes independent inputs. And this is
sensitivity of a model's output to its inputs, which is a different
question from the sensitivity of a causal estimate to unobserved
confounding (`sp.sensemakr`, `sp.evalue`).

When each evaluation is expensive, fit a surrogate on a space-filling
design with `sp.gp_regress(..., interpolate=True)` and compute the indices
on `lambda d: fit.predict(d)["mean"]`.

## A few points for a whole distribution

To push an input distribution through an expensive model, a random sample
of 40 draws is a poor summary. Support points are the 40 points whose
empirical distribution is closest to the target.

```python
from scipy import stats

rep = sp.support_points(
    {"beta": stats.beta(60, 2), "delta": stats.uniform(0.05, 0.05),
     "alpha": stats.norm(0.35, 0.02), "tfp": stats.lognorm(0.05)},
    40, seed=1,
)
model(rep.points).mean()
```

The mean output over these 40 points is 6.87. The value from 200,000
draws is 6.94. Random samples of 40 miss it by 0.48 on average (root mean
squared error over 200 samples), seven times the error of the support
points.

`sp.support_points(sample, n)` does the same from a sample, for instance
MCMC draws, and `subsample=True` returns actual rows.

The same idea gives a train / test split in which the test set has the
joint distribution of the whole data, outcome included.

```python
train, test = sp.split_data(df, test_size=0.2, seed=1)
```

A measured test error then depends less on the luck of the split. Use a
random split wherever a procedure relies on the two parts being
independent samples, as in cross-fitting or honest estimation.

## Sequential design

When each evaluation is expensive, the next run should use what the
earlier ones showed. `sp.sequential_design` fits a Gaussian process to the
runs so far and adds the run with the largest expected improvement.

```python
obj = lambda d: ((d["a"] - 0.3) ** 2 + (d["b"] - 0.7) ** 2
                 + 0.3 * np.sin(8 * d["a"]) * np.sin(8 * d["b"]))
res = sp.sequential_design(obj, {"a": (0, 1), "b": (0, 1)}, n_new=25, seed=1)
res.best
```

```text
{'a': 0.206, 'b': 0.599, 'y': -0.2791}
```

A grid of 160,000 points puts the minimum at (0.205, 0.600) with value
-0.2791. The search used 35 evaluations. `goal="emulate"` instead adds
runs where the prediction is least certain, to learn the whole surface,
and `res.predict(newdata)` is then a fast stand-in for the function.
`noisy=True` is for functions that return a different value on each call.

Expected improvement finds the neighbourhood of a global optimum in few
runs. It is not a convergence proof, and the optimum is located up to the
spacing of the candidate points. Finish with a local optimiser from
`res.best` when more digits are needed.

## How the functions were checked

| Function | Reference | Result |
| --- | --- | --- |
| `sp.design_criteria` | `SFDesign` | Four criteria equal to 1e-12 on the same runs. |
| `sp.design_augment` | `MaxPro::MaxProAugment` | Same runs chosen from the same candidates. |
| `sp.space_filling` | `SFDesign`, `MaxPro` | Stochastic search. Criterion values on par or better. A screen. |
| `sp.factorial_design` | `FrF2` | Same word length pattern for 17 design sizes. |
| `sp.design_aberration` | `DoE.base::GWLP` | Equal to 1e-10, including unbalanced mixed-level designs. |
| `sp.factorial_effects` | `lm`, `unrepx` | Effects and PSE exact. Simulated margins within Monte Carlo error. |
| `sp.mixture_design` | none | Counts and closed forms. |
| `sp.doe_optimal` | `AlgDesign::optFederov`, closed forms | Same criterion value on two exact designs. Polynomial, exponential and logistic designs match their closed forms. |
| `sp.sobol_indices` | `sensitivity::soboljansen`, analytic indices | Equal to 1e-12 after a documented divisor. Ishigami function recovered. |
| `sp.morris_screening` | `sensitivity::morris` | Equal to 1e-10 from the same trajectories. |
| `sp.support_points` | `support::sp` | Stochastic. Energy distance on par. A screen. |
| `sp.split_data` | `twinning::twin`, `SPlit` | Twinning returns the same rows from the same start. SPlit on par. |
| `sp.gp_regress` | `rkriging` | Predictions equal to 1e-10 at given hyperparameters. Fitted values agree to 1e-4. |
| `sp.sequential_design` | known optima | Recovers the optimum of test functions. |

All of these R packages are GPL or LGPL. Nothing was translated from
them. The implementations follow the papers and were compared with R as a
black box.

Not covered: qualitative factors in space-filling designs, nested and
multi-fidelity designs, minimax designs, minimum energy designs, factor
ranking from data (`huang2025factor`), MOFAT screening designs
(`xiao2023maximum`), and Bayesian analysis of factorial experiments.

## References

Bib keys in `paper.bib`: `joseph2025experimental`, `box2005statistics`,
`wu2021experiments`, `lenth1989quick`, `daniel1959use`, `xu2001generalized`,
`scheffe1958experiments`, `scheffe1963simplex`, `kiefer1960equivalence`,
`fedorov1972theory`, `yu2010monotonic`, `chaloner1995bayesian`,
`joseph2015maximum`, `morris1995exploratory`, `mckay1979comparison`,
`johnson1990minimax`, `hickernell1998generalized`, `morris1991factorial`,
`campolongo2007effective`, `sobol2001global`, `jansen1999analysis`,
`saltelli2010variance`, `harenberg2019uncertainty`, `mak2018support`,
`joseph2022split`, `vakayil2022data`, `jones1998efficient`,
`sacks1989design`, `huang2025factor`, `xiao2023maximum`.
