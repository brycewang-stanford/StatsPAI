"""Data for 94_bjs_options.do: mpdta plus the columns the options read.

mpdta (``tests/orig_parity/data/02_mpdta_original.csv``) with four
columns added:

* ``st``  the state FIPS code, ``countyreal // 1000``;
* ``region``  ``st % 5 + 1``, five groups of states: a cell effect
          ``region#year`` has untreated observations in every cell, which
          ``st#year`` does not (mpdta's treatment is a state minimum wage,
          so every treated county-year sits in a state-year with no
          untreated county, and no state-year effect can be imputed);
* ``x``   a time-varying control (mpdta's own ``lpop`` does not vary over
          time, so with unit effects it is collinear and ``controls(lpop)``
          is the default fit);
* ``w``   a unit-level estimation weight.

    python tests/stata_parity/option_parity/_generate_data_94_bjs_options.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
d = pd.read_csv(_ROOT / "tests" / "orig_parity" / "data" / "02_mpdta_original.csv")
d = d.sort_values(["countyreal", "year"]).reset_index(drop=True)
rng = np.random.default_rng(20261015)
units = d["countyreal"].unique()
w_unit = dict(zip(units, np.round(np.exp(rng.normal(scale=0.5, size=len(units))), 10)))
d["st"] = d["countyreal"] // 1000
d["region"] = d["st"] % 5 + 1
d["x"] = np.round(
    0.5 * d["lpop"] + 0.1 * (d["year"] - 2003) + rng.normal(size=len(d)), 10
)
d["w"] = d["countyreal"].map(w_unit)
d.to_csv(Path(__file__).with_name("data_94_bjs_options.csv"), index=False)
print(d.shape, d["st"].nunique(), "states")
