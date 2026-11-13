"""Data for ``test_zero_first_stage_stata.py``.

1,500 rows in 60 clusters (``g``) and 12 strata (``f``). ``zfs`` marks the
600 rows where the instrument ``z`` does not move the treatment ``d``; it
has a direct effect of 0.25 on ``y`` everywhere (an exclusion violation),
the structural coefficient of ``d`` is 0.8, and ``x`` is a control that
enters both equations. Values have six decimals.

    python tests/reference_parity/_fixtures/_generate_zero_first_stage_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    rng = np.random.default_rng(20261115)
    n = 1500
    g = rng.integers(1, 61, size=n)
    f = rng.integers(1, 13, size=n)
    zfs = (np.arange(n) % 5 < 2).astype(int)
    cl = rng.normal(size=61)[g]
    fe = rng.normal(size=13)[f]
    x = rng.normal(size=n) + 0.5 * fe
    z = rng.normal(size=n) + 0.3 * x + 0.4 * cl
    u = rng.normal(size=n) + 0.6 * cl
    d = np.where(zfs == 1, 0.0, 0.7 * z) + 0.4 * x + 0.5 * u + fe + rng.normal(size=n)
    y = 0.8 * d + 0.25 * z - 0.3 * x + 0.9 * u + 0.5 * fe + rng.normal(size=n)
    df = pd.DataFrame(
        {
            "y": np.round(y, 6),
            "d": np.round(d, 6),
            "z": np.round(z, 6),
            "x": np.round(x, 6),
            "zfs": zfs,
            "g": g,
            "f": f,
        }
    )
    df.to_csv(HERE / "zero_first_stage.csv", index=False)
    print(df.groupby("zfs").size().to_dict())
