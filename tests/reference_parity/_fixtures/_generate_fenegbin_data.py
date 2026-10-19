"""Write ``fenegbin_data.csv`` for ``test_fenegbin_parity.py``.

Then run ``Rscript _generate_fenegbin_R.R`` (R ``fixest`` and ``MASS``,
both GPL, used as black boxes: only their output is read).

Design: 1500 rows, firms ``g`` (80, five of them with only zero outcomes,
three singletons), periods ``t`` (6), clusters ``cl`` (25) crossing both,
NB2 counts with theta = 1.5, analytic weights ``w``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261019)
n = 1500
g = rng.integers(0, 77, n)
g[:3] = [77, 78, 79]  # three singleton firms
t = rng.integers(0, 6, n)
cl = rng.integers(0, 25, n)
a = 0.6 * rng.normal(size=80)
b = 0.3 * rng.normal(size=6)
x1 = rng.normal(size=n) + 0.3 * a[g]
x2 = rng.normal(size=n)
mu = np.exp(0.2 + 0.5 * x1 - 0.25 * x2 + a[g] + b[t])
theta = 1.5
y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
y[np.isin(g, [10, 11, 12, 13, 14])] = 0.0  # five all-zero firms
w = rng.uniform(0.5, 2.0, n)
pd.DataFrame(dict(y=y, x1=x1, x2=x2, g=g, t=t, cl=cl, w=w)).to_csv(
    Path(__file__).with_name("fenegbin_data.csv"), index=False, float_format="%.17g"
)
