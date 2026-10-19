"""Write ``fixest_postestimation_data.csv`` for
``test_fixest_postestimation_parity.py`` (fixef / predict / margins after
fits with absorbed fixed effects) and the wild-bootstrap dispatch tests.

Then run ``Rscript _generate_fixest_postestimation_R.R`` (R ``fixest`` and
``marginaleffects``, used as black boxes).
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261019)
n = 1200
f1 = rng.integers(0, 30, n)
f2 = rng.integers(0, 8, n)
f3 = rng.integers(0, 5, n)
cl = rng.integers(0, 12, n)
a1 = rng.normal(size=30)
a2 = rng.normal(size=8)
a3 = 0.3 * rng.normal(size=5)
x1 = rng.normal(size=n) + 0.3 * a1[f1]
x2 = rng.normal(size=n)
fe = a1[f1] + a2[f2]
z1 = rng.normal(size=n)
z2 = rng.normal(size=n)
u = rng.normal(size=n)
d1 = 0.6 * z1 + 0.4 * z2 + 0.5 * u + rng.normal(size=n)
y = 1 + 0.5 * x1 - 0.3 * x2 + fe + a3[f3] + rng.normal(size=n)
yiv = 1 + 0.7 * d1 + 0.3 * x1 + a1[f1] + u + rng.normal(size=n)
yp = rng.poisson(np.exp(-0.2 + 0.3 * x1 + 0.2 * x2 + 0.4 * fe))
eta = -0.2 + 0.6 * x1 - 0.4 * x2 + 0.4 * a1[f1]
yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
d = pd.DataFrame(
    {
        "y": y,
        "yiv": yiv,
        "yp": yp,
        "yb": yb,
        "x1": x1,
        "x2": x2,
        "d1": d1,
        "z1": z1,
        "z2": z2,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "cl": cl,
        # a balanced panel of 100 units over 12 periods (HAC variances)
        "id": np.arange(n) // 12,
        "t": np.arange(n) % 12,
    }
)
d.round(8).to_csv(
    Path(__file__).with_name("fixest_postestimation_data.csv"),
    index=False,
    lineterminator="\n",
    float_format="%.8f",
)
