# Joseph, *Experimental Design for Data Science and Engineering*: what the book needs and what StatsPAI had

*2026-10-07. Worktree `wt/joseph-expdes`.*

## What was done

The material is the author's two code repositories for the book (Chapman
and Hall/CRC, 2025): `rexpdes` (eleven R scripts, about 4,100 lines) and
`pyexpdes`, a Python port that calls the same R packages through
`Rscript`. Every chapter script was read, the R packages it relies on were
installed (R 4.5.2), and the computations were rerun on public test
functions and simulated data, in R and with StatsPAI.

Before this pass StatsPAI had nothing for the subject. `statspai.
experimental` covers the randomised controlled trial: assignment, balance,
attrition, sample size. The book is about which combinations of factor
levels to run, in a physical experiment or on a computer model. The only
overlap was `sp.gp_regress`, `sp.nls` and `sp.regress`.

The note that the book might be dated was right in one specific way. The
book itself is new, but four of the R packages its code loads have been
removed from CRAN and now exist only in the archive: `support`, `mixexp`,
`minimaxdesign` and `mined`. `support` was built from the archived source.
`mixexp` and `minimaxdesign` did not build on this machine and were not
used; the two mixture designs they provide have closed forms. The Python
port does not help, since it shells out to the same packages.

Thirteen functions were added as a new subpackage `statspai.doe`, and one
defect was found and fixed in an existing function.

All reference packages are GPL or LGPL and StatsPAI is MIT. Nothing was
translated. The implementations follow the papers and were compared with R
as a black box.

## Results by chapter

| Book | R | StatsPAI | Outcome |
| --- | --- | --- | --- |
| Ch. 1 to 2, linear regression with prediction bands | `lm`, `predict` | `sp.regress` (existing) | Equal. |
| Ch. 2, kriging at given hyperparameters | `rkriging::Fit.Kriging(fit = FALSE)` | `sp.gp_regress(optimize_hyper=False)` (existing) | Mean and standard deviation equal to 1e-10 once the nugget is matched (below). |
| Ch. 2, Gaussian process fitted by likelihood, no noise | `Fit.Kriging` | `sp.gp_regress(interpolate=True)` (new option) | Length scale and variance equal to 1e-4, predictions to 2e-4. |
| Ch. 2, Gaussian process regression on a replicated design | `Fit.Kriging(interpolation = FALSE)` | `sp.gp_regress` | **Did not agree before this pass.** See "A defect in `sp.gp_regress`". Now equal to 1e-4. |
| Ch. 2 and 9, D-optimal design for a polynomial | `AlgDesign::optFederov` | `sp.doe_optimal` (new) | Same ten runs, criterion equal to 1e-9. |
| Ch. 3, IMSE, MMSE and maximum entropy designs | hand-written `optim` | not added | One-dimensional illustrations of criteria that the space-filling designs approximate. |
| Ch. 4, maximin, MaxPro and uniform criteria | `SFDesign::maximin.crit`, `maxpro.crit`, `uniform.crit` | `sp.design_criteria` (new) | Equal to 1e-12. |
| Ch. 4, maximin / MaxPro / uniform Latin hypercubes | `SFDesign`, `MaxPro` | `sp.space_filling` (new) | Stochastic. Criterion values on par or better (table below). A screen, not parity. |
| Ch. 4, sequential augmentation, constrained regions | `MaxPro::MaxProAugment` | `sp.design_augment`, `sp.space_filling(constraint=)` (new) | Same runs chosen from the same candidates. |
| Ch. 4, qualitative factors, branching, nested, multi-fidelity designs, minimum energy designs | `MaxProQQ`, `SLHD`, `mined`, hand-written | not added | See "Not done". |
| Ch. 5, uniform designs, Sobol' points | `SFDesign::uniformLHD`, `spacefillr` | `sp.space_filling(method='uniform' / 'sobol')` | As above. |
| Ch. 5, support points, uncertainty propagation | `support::sp` | `sp.support_points` (new) | Stochastic. Energy distance 1.72245 to 1.72248 against 1.72250 to 1.72252 for R on the 1,000-row example. A screen. |
| Ch. 6, Sobol' indices | `sensitivity::soboljansen` | `sp.sobol_indices` (new) | Equal to 1e-12 after a divisor is accounted for (below). |
| Ch. 6, Morris screening | `sensitivity::morris` | `sp.morris_screening` (new) | Equal to 1e-10 from the same trajectories. |
| Ch. 6, derivative-based measures, MOFAT designs | `sensitivity::delsa`, `MOFAT` | not added | See "Not done". |
| Ch. 7, active learning (ALM, ALC, ALMV) | hand-written on `rkriging` | `sp.sequential_design(goal='emulate')` (new) | The largest-variance rule only. |
| Ch. 7, expected improvement | hand-written on `rkriging` | `GPResult.expected_improvement`, `sp.sequential_design` (new) | Criterion equal to 2e-4 on the book's example, same next point. The search reaches -0.6034 where the true minimum is -0.6035. |
| Ch. 7, inverse designs | `OSFD` | not added | |
| Ch. 8, two-level fractional factorials | hand-written, `FrF2` | `sp.factorial_design` (new) | Word length patterns equal for 17 design sizes from 8 to 64 runs. |
| Ch. 8, half-normal plot, effects | `lm`, hand-written | `sp.factorial_effects` (new) | Effects and Lenth's pseudo standard error equal. Margins: see below. |
| Ch. 8, generalized word length pattern of L9, L18 | `DoE.base::GWLP` | `sp.design_aberration` (new) | Equal to 1e-10. |
| Ch. 8, Bayesian analysis of a factorial | `HiGarrote` | not added | |
| Ch. 8, simplex-lattice, simplex-centroid, constrained mixtures | `mixexp`, `support::sp` | `sp.mixture_design` (new) | Counts and closed forms; the constrained design is on the simplex and inside the constraint. |
| Ch. 9, nonlinear least squares | `nls` | `sp.nls` (existing) | Coefficients and standard errors equal to 1e-6, the tolerance of `nls`. |
| Ch. 9, locally and Bayesian D-optimal designs | `ICAOD::locally`, `bayes` | `sp.doe_optimal(params=, prior=)` (new) | Same four support points and equal weights as `locally`. `ICAOD` is a 100-iteration metaheuristic and reports points to two or three digits; ours is certified by the equivalence theorem (bound 0.9999). |
| Ch. 10, data splitting | `SPlit` | `sp.split_data(method='support')` (new) | Stochastic, on par. The nearest-row step (`SPlit::subsample`) is exact. |
| Ch. 10, twinning | `twinning::twin` | `sp.split_data(method='twinning')` (new) | The same rows from the same start in five cases, including a factor column and a row count that `r` does not divide. |
| Ch. 10, balanced sampling, supervised compression | `BalancedSampling`, `supercompress` | not added | |
| Ch. 11, factor ranking from data, twin Gaussian process | `first`, `twingp` | not added | |

Quality of the searched designs, best criterion of three seeds against the
five-seed range of `SFDesign` (MaxPro and discrepancy: smaller is better;
maximin: larger is better):

| Design | StatsPAI | SFDesign, min to max |
| --- | --- | --- |
| MaxPro, 20 runs, 2 factors | 24.93 | 25.39 to 25.84 |
| MaxPro, 50 runs, 5 factors | 25.94 | 26.55 to 27.24 |
| Maximin, 50 runs, 5 factors | 0.5596 | 0.5407 to 0.5561 |
| Uniform, 50 runs, 5 factors | 0.0847 | 0.0846 to 0.0857 |

## New functions

Thirteen, in `statspai.doe`. The guide `docs/guides/
design_of_experiments.md` shows them in use.

**Designs.** `sp.factorial_design`, `sp.mixture_design`, `sp.doe_optimal`,
`sp.space_filling`, `sp.design_augment`, `sp.sequential_design`.

**Assessing a design.** `sp.design_criteria`, `sp.design_aberration`.

**Analysis.** `sp.factorial_effects`, `sp.sobol_indices`,
`sp.morris_screening`.

**Representative points.** `sp.support_points`, `sp.split_data`.

Existing function, new options: `sp.gp_regress(interpolate=, likelihood=)`
and `GPResult.expected_improvement()`.

Why these are in a package for causal inference and econometrics. Factorial
and conjoint experiments are run by economists and are analysed with the
tools of chapter 8. A structural model solved at chosen parameter values is
a computer experiment in the book's sense: Sobol' indices of such models
are an established practice (`harenberg2019uncertainty`), a simulated
method of moments objective is the expensive function that chapter 7
optimises, and space-filling designs are how a simulation study over a
parameter region should be laid out.

## A defect in `sp.gp_regress`

The book's Figure 2.9 fits a Gaussian process to ten sites observed twice
with noise. `sp.gp_regress` returned a length scale of 0.0014 where
`rkriging` returns 0.048, and the two predictions differed by up to 0.79
on a function whose range is about 1.5.

This was ours. The likelihood at the R solution is higher (-1.2107) than
at the point we returned (-1.2447). When the length scale is far below the
spacing of the data every site is its own island, the likelihood no longer
depends on the length scale, and its gradient is zero. The optimiser
started at the standard deviation of the regressor, stepped into that
plateau and stopped; the four random restarts landed there as well. The
fit then reverts to the mean between the sites.

Fix: before optimising, the likelihood is evaluated on a coarse grid of
length scales and noise shares, and the two best grid points are added as
starting values. On the example the fit now agrees with `rkriging` to
3e-7 in the prediction. A fit whose length scale is below a quarter of the
smallest gap in the data now carries a note that says what that means.

Who was affected: fits where the default start is far above the right
length scale, most visibly replicated or coarsely spaced designs in one
regressor. The earlier comparison of this function with an exact posterior
(`tests/reference_parity/test_bayes_nonparametric_exact.py`) fixes the
hyperparameters and did not exercise the optimiser.

## Four places where the reference and the paper differ

**Lenth's margins.** The 1989 paper refers `effect / PSE` to a t
distribution with `m / 3` degrees of freedom. Lenth's own package
`unrepx` simulates the null distribution instead. With the seven effects
of the book's example the two margins of error are 101.6 and 61.9, and the
simultaneous ones 243 and 130. The simulated reference is the default
(`reference='simulated'`, seed fixed so that repeated calls agree). In
20,000 simulated experiments with no real effect and seven effects each,
5.1% of the effects fall beyond the simulated margin and 5.3% of the
experiments have an effect beyond the simultaneous one. For the t
reference the figures are 2.1% and 1.3% at a nominal 5% (2.9% and 2.4%
with fifteen effects). `reference='t'` reproduces the paper.

**The divisor in `sensitivity::soboljansen`.** Jansen's estimator divides
the sums of squared differences by `2n`. The R function divides by
`2n - 1`. This was found by evaluating candidate formulas against the
package's output; its value sits exactly midway between the `2n` and the
`2(n - 1)` versions. StatsPAI follows the paper, and the test undoes the
factor before comparing, after which the two agree to 1e-12.

**The nugget in `rkriging`.** For a noise-free fit `rkriging` adds 1e-6
of the process variance to the diagonal. `sp.gp_regress(interpolate=True)`
adds 1e-8 of the variance of the outcome. With the nugget matched the
predictions agree to 1e-10; with the defaults they differ by about 1e-5.

**Which likelihood `rkriging` maximises.** Its estimates coincide with the
restricted likelihood (the constant mean integrated out), not with the
profile likelihood that the book's own hand-written `OKfit` uses. Both are
now available (`likelihood='reml'`, the default, and `'ml'`). The test of
`'ml'` checks the estimating equation directly.

A fifth point is a choice, not a discrepancy. For the first-order Sobol'
index neither Jansen's estimator (the book's) nor that of Saltelli et al.
(2010) dominates: on the borehole function with 512 Sobol' points the root
mean squared errors are 0.005 against 0.012 for the input that explains
83% of the variance, and 0.008 against 0.0001 for the three inert inputs.
Jansen's is the default and the docstring gives these numbers.

## An independent review of the new code

After the functions passed their own tests a second reviewer, with no
knowledge of how they had been checked, was asked to look for wrong
numbers and crashes on inputs the tests did not cover. The formulas held
(Lenth, Jansen and Saltelli, Morris, the word length recursion, the
multiplicative algorithm and its bounds, the exchange updates, the
support-point iteration; the exchange algorithm matched exhaustive search
in nine small cases). What it found, all fixed and each now a test in
`tests/test_doe.py::TestReviewFindings`:

- `sp.doe_optimal` crashed on a first-order model in eight or more
  factors. The optimal design puts weight `2^-p` on each corner, which fell
  under a fixed weight threshold. The thresholds are now relative to the
  largest weight. Boxes of more than ten factors are refused with advice
  to pass `candidates=`, because the approximate design then has thousands
  of support points (twelve factors took five minutes).
- A generator with a minus sign (`D = -AB`) built the right runs but
  printed the defining relation and the aliases without the sign, so the
  summary showed `D = -AB` in one line and `D = AB` in another. Words and
  aliases now carry their sign (`I = -ABD = ACE = -BCDE`, `A = -BD = CE`).
- The I criterion averaged over the candidate set. On a box of four or
  more factors that set is quasi-random points plus all corners, so the
  corners were over-represented (2.6925 for a case whose value is 2.6667),
  and on a coarse grid the I-optimal weights of a quadratic came out as
  0.261 / 0.478 / 0.261 instead of 1/4, 1/2, 1/4. The region is now
  sampled separately and uniformly.
- For a binomial or Poisson response the I criterion mixed the variance
  function into the region average. It is now the average variance of the
  estimated linear predictor, and the docstring says so.
- Smaller ones: generators could not be written for factor names with
  spaces; a nonlinear model silently misread factors called `_n` or
  `price-usd`; `sp.sequential_design` overwrote a factor called `y`;
  `sp.factorial_effects` failed on integer column names and on
  `seed=None`, and its docstring promised support for centre points that
  the code refuses; `sp.design_criteria(target=)` matched columns by
  position; `sp.support_points` gave a numpy error when few draws had
  positive weight; one Morris trajectory gave a silent NaN.

## Not done

Ordered by how much an applied user would miss them.

1. **Qualitative factors in space-filling designs** (`MaxProQQ`, sliced
   Latin hypercubes). A simulation study often has a categorical factor.
   Today the user builds one design per level.
2. **Factor ranking from data** (`first`, Huang and Joseph 2025). Total
   Sobol' indices estimated from a data set by nearest neighbours, with
   forward selection. It would connect to variable importance elsewhere in
   the package. Stochastic, so the evidence would be a screen.
3. **Bayesian analysis of factorial experiments** (`HiGarrote`), which
   uses effect heredity. `sp.factorial_effects` stops at Lenth.
4. **Other active-learning criteria** (ALC, ALMV). The book's own
   comparison does not favour them clearly over the largest-variance rule.
5. **Regular fractions at three levels and orthogonal arrays** (L9, L18).
   `sp.design_aberration` assesses them; nothing builds them.
6. **MOFAT screening designs, derivative-based sensitivity, minimax and
   minimum energy designs, inverse designs, nested and multi-fidelity
   designs, supervised compression, twin Gaussian processes, balanced
   sampling.** Specialised; no request for them yet.

The minimum-aberration search in `sp.factorial_design` is exhaustive and
refuses problems with more than 30,000 generator choices (ten factors in
32 or 64 runs are the first ones refused). A catalogue would lift that.

`sp.doe_optimal` finds exact designs by a multi-start exchange. On a
problem with many local optima (14 runs for a three-factor quadratic on a
5^3 grid) the default 20 starts reach the best known value in 87 to 100%
of seeds, depending on the criterion. The function reports the efficiency
relative to the approximate optimum, so a poor local optimum is visible.

## Rerun

```bash
# R side (needs the packages named at the top of the script; `support`
# is archived on CRAN and has to be installed from the archive)
Rscript tests/reference_parity/_fixtures/_generate_joseph_doe.R

# Python side
pytest tests/reference_parity/test_joseph_doe_parity.py tests/test_doe.py -q
```

The R packages for this pass are in a private library next to the book
material (`改进建议-收集整理/35-Joseph-ExperimentalDesign/_rlib`), with
the probes that were used to pin down conventions in `run/`. On this
machine R's Fortran flags point at a compiler that is not installed;
`support` builds with `R_MAKEVARS_USER` set to a file that points `FLIBS`
at Homebrew's `libgfortran`.
