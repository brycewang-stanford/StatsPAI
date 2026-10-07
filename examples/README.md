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
  each Stata command (`regress`, `reghdfe`, `ivregress 2sls`, `esttab`,
  `csdid`, `rdrobust`) next to its one-line StatsPAI equivalent on the bundled
  datasets, and ends with `sp.from_stata`. The Stata output shown in the
  notebook was produced by Stata 18 on the same data; running the notebook
  needs no Stata. Prose and code comments are in Chinese. Uses only core
  StatsPAI plus matplotlib for the two figures.

```bash
python -m pip install "statspai[plotting]" jupyter
jupyter notebook examples/notebooks/statspai_vs_stata_5min.ipynb
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
