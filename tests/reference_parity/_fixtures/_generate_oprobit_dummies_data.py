"""Data for the ordered-model fixture: 4,000 rows, 120 groups (119 dummies),
an unscaled regressor x ~ 1000 +- 50, 40 clusters, four ordered outcomes.

    python tests/reference_parity/_fixtures/_generate_oprobit_dummies_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "oprobit_dummies.csv"


def main() -> None:
    rng = np.random.default_rng(20260930)
    m, groups = 4000, 120
    df = pd.DataFrame(
        dict(
            x=rng.normal(size=m) * 50 + 1000,
            g=rng.integers(0, groups, m),
            z=rng.normal(size=m),
        )
    )
    df["cl"] = df.g // 3
    lat = (
        0.01 * (df.x - 1000)
        + 0.4 * df.z
        + rng.normal(size=groups)[df.g] * 0.3
        + rng.normal(size=m)
    )
    df["y"] = np.digitize(lat, [-0.8, 0.1, 0.9])
    df.to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
