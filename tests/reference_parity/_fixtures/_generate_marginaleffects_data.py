"""Write ``marginaleffects_data.csv`` for the marginaleffects parity test.

Then run ``Rscript _generate_marginaleffects_R.R`` (R ``marginaleffects``,
used as a black box) to write ``marginaleffects_R.json``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261017)
n = 400
x = rng.normal(size=n)
z = rng.normal(size=n)
g = rng.integers(0, 3, n)
b = rng.integers(0, 2, n)
w = rng.uniform(0.5, 2.0, n)
eta = 0.3 + x - 0.5 * z + 0.4 * b + 0.3 * x * b + 0.2 * (g == 2)
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
yc = 1 + x + 0.5 * x**2 + b + 0.2 * g + rng.normal(size=n)
cnt = rng.poisson(np.exp(0.2 + 0.4 * x - 0.3 * z + 0.2 * (g == 1)))
d = pd.DataFrame(
    {"x": x, "z": z, "g": g, "b": b, "w": w, "y": y, "yc": yc, "cnt": cnt}
)
d.round(8).to_csv(
    Path(__file__).with_name("marginaleffects_data.csv"),
    index=False,
    lineterminator="\n",
    float_format="%.8f",
)
