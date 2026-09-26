"""Write the input of tests/reference_parity/test_margins_ext_parity.py.

Run before ``_generate_margins_ext_stata.do``. Deterministic; the CSV is
committed so the Stata and Python sides read identical bytes.

Design (every column exercises a margins convention added in 1.32):

* ``g`` -- three-level factor (1..3): discrete changes vs the base level;
* ``x`` -- continuous, entering as ``x + x^2`` (Stata ``c.x##c.x``,
  StatsPAI ``x + I(x**2)``) so dy/dx must include ``2 b2 x``;
* ``z`` -- continuous, interacted with ``g`` in the probit;
* ``w`` -- analytic weights (Stata ``[aw=w]``), which margins averages with;
* ``expo`` -- exposure for the count models (``exposure(expo)``), which the
  default prediction (number of events) includes;
* ``yl`` linear, ``yb`` binary, ``yc`` count outcomes.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent

rng = np.random.default_rng(20260926)
n = 700
g = rng.choice([1, 2, 3], size=n, p=[0.45, 0.35, 0.20])
x = rng.normal(size=n) + 0.2 * (g - 2)
z = rng.normal(size=n)
w = rng.uniform(0.5, 2.0, size=n)
expo = rng.uniform(1.0, 4.0, size=n)
gd = np.array([0.0, 0.7, -0.5])[g - 1]
yl = 1.0 + 0.5 * x + 0.3 * x**2 - 0.2 * z + gd + 0.4 * (g == 2) * x + rng.normal(size=n)
eta = -0.3 + 0.8 * x - 0.25 * x**2 + 0.5 * z + gd
yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
yc = rng.poisson(expo * np.exp(0.1 + 0.3 * x - 0.2 * z + 0.5 * gd))

df = pd.DataFrame(
    {"yl": yl, "yb": yb, "yc": yc, "x": x, "z": z, "g": g, "w": w, "expo": expo}
)
df.to_csv(OUT / "margins_ext_data.csv", index=False, float_format="%.17g")
print(f"wrote {len(df)} rows")
