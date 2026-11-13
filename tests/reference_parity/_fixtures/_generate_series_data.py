"""Data for ``test_series_R_parity.py``.

400 rows in 40 clusters: ``x`` uniform on (0, 3), a linear regressor ``z``,
``y = sin(2 x) + 0.5 z + e`` with a cluster effect and an error whose
spread grows with ``x``. Values have six decimals.

    python tests/reference_parity/_fixtures/_generate_series_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    rng = np.random.default_rng(20261116)
    n = 400
    g = np.repeat(np.arange(1, 41), 10)
    x = rng.uniform(0, 3, n)
    z = rng.normal(size=n) + 0.3 * x
    e = rng.normal(size=n) * (0.2 + 0.15 * x) + 0.2 * rng.normal(size=41)[g]
    y = np.sin(2 * x) + 0.5 * z + e
    pd.DataFrame(
        {"y": np.round(y, 6), "x": np.round(x, 6), "z": np.round(z, 6), "g": g}
    ).to_csv(HERE / "series_data.csv", index=False)
