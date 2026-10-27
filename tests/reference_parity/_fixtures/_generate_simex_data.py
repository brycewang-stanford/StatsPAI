"""Data for tests/reference_parity/test_simex_r_screen.py.

``w`` is ``x`` plus normal error with standard deviation 0.5; ``z`` is
measured without error; ``y`` is linear in ``x`` and ``z``, ``d`` a logit
outcome. Written with 8 decimals so R and pandas read the same doubles.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261028)
n = 500
x = rng.normal(size=n)
z = rng.normal(size=n)
p = 1.0 / (1.0 + np.exp(-(-0.5 + 1.0 * x + 0.5 * z)))
df = pd.DataFrame(
    {
        "y": 1.0 + 2.0 * x + 0.5 * z + rng.normal(size=n),
        "d": (rng.uniform(size=n) < p).astype(int),
        "w": x + rng.normal(scale=0.5, size=n),
        "z": z,
    }
)
df.to_csv(Path(__file__).with_name("simex_data.csv"), index=False, float_format="%.8f")
