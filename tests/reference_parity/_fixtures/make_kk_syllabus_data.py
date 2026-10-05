"""Write ``kk_syllabus.csv``: the data behind
``test_kohler_kreuter_stata_parity.py``.

Synthetic, small and awkward on purpose: ties, missing values, unequal
weights, strata with a few sampling units each. Stata 18 read the same
bytes when ``kk_syllabus_reference.do`` produced the reference numbers, so
do not regenerate the file without rerunning that do-file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261005)
n = 420
strata = np.repeat(np.arange(1, 7), n // 6)
psu = strata * 100 + rng.integers(1, 6, size=n)
g = rng.integers(1, 4, size=n)
f = rng.integers(0, 2, size=n)
x1 = np.round(rng.normal(50, 12, size=n))
x2 = np.round(rng.gamma(2.0, 3.0, size=n), 1)
u = rng.normal(size=n)
y = np.round(20 + 0.6 * x1 - 1.5 * x2 + 4 * (g == 3) + 3 * f + 8 * u)
d = (0.04 * (x1 - 50) - 0.1 * x2 + 0.5 * f + rng.logistic(size=n) > -0.3).astype(int)
o = np.digitize(0.03 * (x1 - 50) + rng.normal(size=n), [-0.8, 0.0, 0.9]) + 1
w = np.round(rng.uniform(0.5, 4.0, size=n) * (1 + 0.3 * (strata % 2)), 3)
fw = rng.integers(1, 4, size=n)
z = np.round(y + rng.normal(0, 6, size=n))
df = pd.DataFrame(
    {"id": np.arange(1, n + 1), "strata": strata, "psu": psu, "w": w, "fw": fw,
     "g": g, "f": f, "x1": x1, "x2": x2, "y": y, "z": z, "d": d, "o": o}
)  # fmt: skip
df.loc[rng.choice(n, 25, replace=False), "y"] = np.nan
df.loc[rng.choice(n, 12, replace=False), "x2"] = np.nan
df.loc[rng.choice(n, 8, replace=False), "g"] = np.nan
df.to_csv(Path(__file__).with_name("kk_syllabus.csv"), index=False)
