"""Two-way fixed-effects Poisson fixture for tests/test_margins_fe_delta.py.

python tests/fixtures/margins_fe/_make_fep2.py  (writes fep2.csv)
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261020)
n = 400
g = rng.integers(1, 26, size=n)
h = rng.integers(1, 9, size=n)
a = rng.normal(scale=0.5, size=26)
b = rng.normal(scale=0.3, size=9)
x1 = rng.normal(size=n) + 0.3 * a[g]
x2 = rng.normal(size=n)
d = (rng.random(n) < 0.4).astype(int)
w = rng.uniform(0.5, 2.0, size=n)
mu = np.exp(0.3 * x1 - 0.2 * x2 + 0.4 * d + a[g] + b[h])
y = rng.poisson(mu)
df = pd.DataFrame({"g": g, "h": h, "x1": x1, "x2": x2, "d": d, "w": w, "y": y})
df.to_csv(Path(__file__).with_name("fep2.csv"), index=False, float_format="%.17g")
