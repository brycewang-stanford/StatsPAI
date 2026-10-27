"""Data for tests/reference_parity/test_eivreg_stata_parity.py.

Two regressors measured with error (x1, x2), one measured without (z),
analytic weights w. Values are written with 8 decimals so that Stata's
``import delimited`` (after ``set type double``) and pandas read the
same doubles.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261028)
n = 80
t1 = rng.normal(size=n)
t2 = 0.5 * t1 + rng.normal(size=n)
z = rng.normal(size=n)
y = 1.0 + 0.8 * t1 - 0.5 * t2 + 0.3 * z + rng.normal(scale=0.7, size=n)
df = pd.DataFrame(
    {
        "y": y,
        "x1": t1 + rng.normal(scale=0.5, size=n),
        "x2": t2 + rng.normal(scale=0.6, size=n),
        "z": z,
        "w": rng.uniform(0.5, 3.0, size=n),
    }
)
out = Path(__file__).with_name("eivreg_data.csv")
df.to_csv(out, index=False, float_format="%.8f")
