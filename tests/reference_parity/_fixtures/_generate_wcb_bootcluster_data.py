"""Data for the crossing-bootcluster case of sp.wild_cluster_boot.

Error clusters ``a`` (3) and bootstrap clusters ``b`` (3) cut across each
other: 9 intersection cells, so boottest enumerates all 2^9 Rademacher
draws and the p-values are exact on both sides.

    python _generate_wcb_bootcluster_data.py   (from this directory)
"""

import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261020)
n = 360
a = rng.integers(1, 4, n)
b = rng.integers(1, 4, n)
cell = rng.normal(size=(4, 4))
x = rng.normal(size=n) + 0.3 * a
w = rng.normal(size=n)
y = 1.0 + 0.4 * x - 0.5 * w + cell[a, b] + rng.normal(size=n)
pd.DataFrame(
    {"y": np.round(y, 12), "x": np.round(x, 12), "w": np.round(w, 12), "a": a, "b": b}
).to_csv(pathlib.Path(__file__).with_name("wcb_bootcluster.csv"), index=False)
