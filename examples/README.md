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
