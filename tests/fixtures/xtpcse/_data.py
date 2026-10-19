"""Simulated time-series cross-section behind ``tests/test_xtpcse_stata.py``.

``python tests/fixtures/xtpcse/_data.py <dir>`` writes ``tscs.csv`` (10
panels x 20 periods, AR(1) disturbances with a common shock) and
``tscs_unbal.csv`` (the same with 17 rows removed, including interior
gaps) for ``stata_xtpcse.do``.
"""

import sys

import numpy as np
import pandas as pd


def make_tscs(seed: int = 20261019) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m, T = 10, 20
    common = rng.normal(size=T)
    rows = []
    for i in range(m):
        sd = 0.5 + 0.15 * i
        load = rng.uniform(0.2, 1.2)
        rho = rng.uniform(0.1, 0.7)
        u = 0.0
        for t in range(T):
            u = rho * u + sd * rng.normal()
            x1, x2 = rng.normal(size=2)
            y = 1.0 + 0.8 * x1 - 0.5 * x2 + 0.3 * i + load * common[t] + u
            rows.append((i + 1, t + 1, y, x1, x2))
    return pd.DataFrame(rows, columns=["id", "t", "y", "x1", "x2"])


def make_unbalanced(seed: int = 20261019) -> pd.DataFrame:
    df = make_tscs(seed)
    rng = np.random.default_rng(seed + 1)
    drop = rng.choice(len(df), size=17, replace=False)
    return df.drop(index=drop).reset_index(drop=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    make_tscs().to_csv(f"{out}/tscs.csv", index=False)
    make_unbalanced().to_csv(f"{out}/tscs_unbal.csv", index=False)
