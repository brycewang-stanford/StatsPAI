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
```

Or after installing the released package:

```bash
python -m pip install statspai
python examples/card_iv.py
```

The scripts cover canonical causal-inference designs:

- `card_iv.py` - instrumental variables using Card (1995).
- `did_mpdta.py` - staggered difference-in-differences using `mpdta`.
- `rd_lee.py` - sharp regression discontinuity using Lee (2008).
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
