# StatsPAI Examples

These examples are short, offline scripts for reviewers and new users. They use
the teaching datasets bundled with `statspai`, so they do not download data or
require network access after installation.

From a source checkout:

```bash
python -m pip install -e ".[dev,plotting]"
python examples/card_iv.py
python examples/did_mpdta.py
python examples/rd_lee.py
python examples/synth_prop99.py
python examples/gmethods_timevarying.py
python examples/nhefs_whatif.py
python examples/dml_card.py
python examples/policy_index_hdfe_iv.py
```

Or after installing the released package:

```bash
python -m pip install statspai
python examples/card_iv.py
```

The scripts cover canonical causal-inference designs:

- `card_iv.py` - instrumental variables using Card (1995).
- `did_mpdta.py` - staggered difference-in-differences using `mpdta`.
- `rd_lee.py` - sharp regression discontinuity: Lee's (2008) close-election
  design on the U.S. Senate extract distributed with R `rdrobust`.
- `synth_prop99.py` - synthetic control using California Proposition 99.
- `gmethods_timevarying.py` - g-methods (parametric g-formula + marginal
  structural model) for time-varying confounding, the signature problem of
  modern causal epidemiology. Uses a self-contained simulation, so it needs
  no bundled dataset.
- `nhefs_whatif.py` - reproduces the published g-methods estimates from
  Hernán & Robins, *Causal Inference: What If*, on the real bundled NHEFS
  data: IP weighting, standardization/g-formula, and g-estimation all
  recover the book's ~3.4-3.5 kg effect of quitting smoking on weight, plus
  an E-value sensitivity analysis. Uses `sp.datasets.nhefs()`.
- `dml_card.py` - double/debiased machine learning (`sp.dml`) on Card
  (1995): partially linear and partially linear IV models for the return to
  schooling, recovering the classic pattern that the IV estimate exceeds the
  partialling-out one. The DoubleML-aligned, high-dimensional entry point.

## 5-minute tutorial for Stata users (offline, in Chinese)

- `notebooks/statspai_vs_stata_5min.ipynb` - a first-contact tutorial that puts
  each Stata command (`summarize`, `regress`, `reghdfe`, `ivregress 2sls`,
  `esttab`, `csdid`, `rdrobust`) next to its one-line StatsPAI equivalent on
  the bundled datasets, and ends with `sp.from_stata` (translate one command)
  and `sp.stata` (run a do-file snippet). It runs on the Python kernel and
  points to the method tutorials below. The Stata output shown in the
  notebook was produced by Stata 18 on the same data; running the notebook
  needs no Stata. Prose and code comments are in Chinese. Uses only core
  StatsPAI plus matplotlib for the two figures.

```bash
python -m pip install "statspai[plotting]" jupyter
jupyter notebook examples/notebooks/statspai_vs_stata_5min.ipynb
```

## Method tutorials (offline, in Chinese)

Ten teaching notebooks, one per topic. Each one states the assumption the
method rests on, works an example on a bundled dataset or on a simulation
with a known truth, shows the diagnostics that decide whether the estimate can
be trusted, and closes with a checklist. Prose and code comments are in Chinese; figure labels are in
English. They need only core StatsPAI plus matplotlib, and the committed
notebooks ship their executed outputs.

- `notebooks/tutorial_did_staggered.ipynb` - staggered difference-in-differences.
  A simulated panel with a known truth shows TWFE missing the ATT by a third,
  the Goodman-Bacon decomposition shows why, and `sp.callaway_santanna` /
  `sp.aggte` recover it. Then the full workflow on `mpdta`: event study,
  uniform bands, Sun-Abraham / imputation / ETWFE / two-stage cross-checks, and
  `sp.honest_did` sensitivity to parallel-trends violations.
- `notebooks/tutorial_iv_card.ipynb` - instrumental variables on Card (1995).
  First stage, reduced form and the Wald ratio by hand, why hand-rolled
  two-step standard errors are wrong, a weak-instrument Monte Carlo,
  `sp.iv.iv_diag` (effective F, tF, Anderson-Rubin), over-identification and
  LIML, the LATE interpretation, and a plausibly-exogenous sensitivity check.
- `notebooks/tutorial_rd_senate.ipynb` - regression discontinuity on the U.S.
  Senate close-election data. `sp.rdplot`, a local linear fit by hand,
  `sp.rdrobust` and what the robust bias-corrected row means, bandwidth and
  specification sensitivity, why not global polynomials, the density test,
  placebo cutoffs, `sp.rd_honest`, and a simulated fuzzy design.
- `notebooks/tutorial_synth_prop99.ipynb` - synthetic control on California's
  Proposition 99. Donor weights and pre-treatment fit, placebo inference and
  where the permutation p-value comes from, in-time placebo, leave-one-out,
  and a comparison with demeaned, augmented and synthetic DID estimators.
- `notebooks/tutorial_matching_lalonde.ipynb` - matching and weighting on the
  LaLonde data against the experimental benchmark. Balance tables and SMDs,
  propensity scores and overlap, `sp.match`, `sp.ipw`, `sp.ebalance`,
  `sp.aipw`, effective sample size, all estimates side by side, and
  `sp.sensemakr` for unobserved confounding.

- `notebooks/tutorial_regression_se.ipynb` - regression and standard errors.
  Monte Carlo rejection rates show classical standard errors failing under
  heteroskedasticity, robust ones failing under within-group correlation, and
  cluster-robust ones failing with eight clusters; then HC0-HC3, where to
  cluster, the wild cluster bootstrap and CR2, `sp.test` / `sp.lincom`, and
  `sp.regtable`.
- `notebooks/tutorial_panel_fe.ipynb` - panel data and fixed effects. Pooled
  OLS, fixed effects three ways, random effects, the Hausman test (including
  the negative statistic and what to do about it), the Mundlak regression,
  what fixed effects cannot estimate, and two-way fixed effects on the
  castle-doctrine state panel.
- `notebooks/tutorial_rct_thornton.ipynb` - analysing a randomized experiment
  (Thornton's HIV-results incentive). Balance, the difference in means and
  the Neyman standard error, randomization inference by hand and with
  `sp.fisher_exact`, Lin's covariate adjustment, dose response, subgroup
  interactions with a Holm correction, and power.
- `notebooks/tutorial_hte_causal_forest.ipynb` - heterogeneous treatment
  effects on a simulation with a known CATE. Five meta-learners, the causal
  forest and out-of-bag predictions, the calibration test and sample-split
  RATE (with a homogeneous-effect placebo), best linear projection, group
  effects, and a policy tree under a treatment cost. Takes about three
  minutes to run.
- `notebooks/tutorial_gmethods_nhefs.ipynb` - g-methods on NHEFS, following
  Hernan and Robins. IP weighting and standardization by hand and in one
  call, g-estimation, AIPW and TMLE, the E-value, and a two-visit simulation
  in which every regression specification is wrong while the g-formula and a
  marginal structural model recover the truth.

```bash
python -m pip install "statspai[plotting]" jupyter
jupyter notebook examples/notebooks/tutorial_did_staggered.ipynb
```

## Networked notebook (requires internet + `doubleml`)

One reviewer notebook is **not** offline — it fetches the canonical 401(k)
data from DoubleML's public distribution (StatsPAI bundles no copy):

- `notebooks/reproduce_401k_doubleml.ipynb` - reproduces the DoubleML / `hdm`
  401(k) result with `sp.dml`, side by side with `doubleml-for-py` on the same
  data. The partially linear estimates match to the displayed precision; the
  committed notebook ships its executed outputs. Written as a DML tutorial
  with prose and code comments in Chinese: a hand-rolled cross-fitted PLR,
  the choice of controls and learners, IRM / ATTE and overlap, repeated
  cross-fitting, shared-fold parity with DoubleML, the IIVM LATE of
  participation, and omitted-variable sensitivity. Runs in about two minutes.

```bash
python -m pip install statspai doubleml scikit-learn matplotlib jupyter
jupyter nbconvert --to notebook --execute --inplace examples/notebooks/reproduce_401k_doubleml.ipynb
```
