"""Generate the fixed DGP for the g-computation (g-formula) parity test.

Run once; commit the output ``gformula_data.csv`` so both the base-R
fixture generator (``_generate_gformula.R``) and the Python test
(``test_gformula_parity.py``) consume the exact same dataset.

DGP — point-exposure with linear confounding (Robins 1986 g-formula
setting; standardization example as in Snowden-Rose-Mortimer 2011):

    x1, x2 ~ N(0, 1),  x3 ~ Bernoulli(0.5)
    P(D=1|X) = logistic(-0.2 + 0.6 x1 - 0.4 x2 + 0.5 x3)
    y = 0.5 + 1.2 d + 0.8 x1 - 0.5 x2 + 0.3 x3 + N(0, 0.7)

True ATE = 1.2 (homogeneous, additive).  The outcome model is exactly
linear-additive, matching sp.g_computation's default OLS Q-model, so
the parametric g-formula is correctly specified on this DGP.

n = 800.  pandas ``to_csv`` writes shortest-roundtrip float repr, so R
``read.csv`` recovers bit-identical doubles.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(20260612)
n = 800

x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
x3 = rng.binomial(1, 0.5, size=n)

lin = -0.2 + 0.6 * x1 - 0.4 * x2 + 0.5 * x3
p = 1.0 / (1.0 + np.exp(-lin))
d = (rng.uniform(size=n) < p).astype(int)

TRUE_ATE = 1.2
y = 0.5 + TRUE_ATE * d + 0.8 * x1 - 0.5 * x2 + 0.3 * x3 + rng.normal(scale=0.7, size=n)

df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "x3": x3})

out = pathlib.Path(__file__).parent / "gformula_data.csv"
df.to_csv(out, index=False)
print(f"Wrote {out}  (n={n}, true ATE={TRUE_ATE}, treated share={d.mean():.3f})")
