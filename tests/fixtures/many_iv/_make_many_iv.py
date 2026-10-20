"""Many-instrument fixture for tests/test_liml_many_weak_se.py.

python tests/fixtures/many_iv/_make_many_iv.py   (writes many_iv.csv)
Rscript tests/fixtures/many_iv/ivmodel_many_iv.R (writes ivmodel_many_iv.json)
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n, K = 200, 20
Z = rng.normal(size=(n, K))
w1 = rng.normal(size=n)
w2 = rng.binomial(1, 0.5, n)
e = rng.standard_t(5, size=n)
v = 0.6 * e + rng.chisquare(3, size=n) - 3
d = Z @ np.full(K, 0.1) + 0.3 * w1 + v
y = 1 + 0.5 * d + 0.4 * w1 - 0.2 * w2 + e
df = pd.DataFrame(Z, columns=[f"z{j}" for j in range(K)])
df["w1"], df["w2"], df["d"], df["y"] = w1, w2, d, y
df.to_csv(Path(__file__).with_name("many_iv.csv"), index=False, float_format="%.17g")
