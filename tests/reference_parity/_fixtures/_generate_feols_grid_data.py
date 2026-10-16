"""Write feols_grid_data.csv for test_feols_grid_fixest_parity.py (seeded)."""

import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261017)
n = 1500
firm = rng.integers(0, 60, n)
year = rng.integers(0, 12, n)
g = firm // 4  # clusters coarser than the firm
h = rng.integers(0, 15, n)
a_f = rng.normal(size=60)[firm]
a_y = rng.normal(size=12)[year]
x1 = rng.normal(size=n) + 0.5 * a_f
x2 = rng.normal(size=n)
w = rng.uniform(0.3, 3.0, n)
u = rng.normal(size=n) * (0.5 + np.abs(x2)) + rng.normal(size=16)[g]
y = 1 + 0.8 * x1 - 0.4 * x2 + a_f + 0.5 * a_y + u
pd.DataFrame(
    {"y": y.round(10), "x1": x1.round(10), "x2": x2.round(10), "w": w.round(10),
     "firm": firm, "year": year, "g": g, "h": h}
).to_csv(pathlib.Path(__file__).with_name("feols_grid_data.csv"), index=False)
