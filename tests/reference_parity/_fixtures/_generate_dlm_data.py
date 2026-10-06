"""Synthetic series for tests/reference_parity/test_dlm_parity.py.

``dlm.csv``: 140 dates, a regressor ``x``, an outcome ``y`` whose intercept
and slope follow random walks, and ``level``, a local level series.

    python tests/reference_parity/_fixtures/_generate_dlm_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261007)
T = 140
x = rng.normal(1.0, 1.0, size=T)
b0 = 1.0 + np.cumsum(rng.normal(scale=0.25, size=T))
b1 = 0.5 + np.cumsum(rng.normal(scale=0.15, size=T))
y = b0 + b1 * x + rng.normal(scale=0.5, size=T)
level = np.cumsum(rng.normal(scale=0.4, size=T)) + rng.normal(scale=1.0, size=T)
pd.DataFrame({"t": np.arange(1, T + 1), "x": x, "y": y, "level": level}).to_csv(
    Path(__file__).parent / "dlm.csv", index=False, float_format="%.10g"
)
