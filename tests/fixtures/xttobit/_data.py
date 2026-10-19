"""Simulated unbalanced panel behind ``tests/test_xttobit_stata.py``.

``python tests/fixtures/xttobit/_data.py <dir>`` writes ``retobit.csv`` for
``stata_xttobit.do``: 120 panels of 2 to 7 periods, a normal panel effect,
the outcome censored at 0 from below and at 4 from above, plus an
interval-coded version (bands of width 1, a fifth exact).
"""

import sys

import numpy as np
import pandas as pd


def make_panel(seed: int = 20261019) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(120):
        nu = 0.8 * rng.normal()
        for t in range(int(rng.integers(2, 8))):
            x1, x2 = rng.normal(size=2)
            ys = 0.7 + 1.0 * x1 - 0.6 * x2 + nu + rng.normal()
            rows.append((i + 1, t + 1, x1, x2, ys))
    df = pd.DataFrame(rows, columns=["id", "t", "x1", "x2", "ys"])
    df["y"] = df["ys"].clip(0, 4)
    lo = np.floor(df["ys"].to_numpy())
    hi = lo + 1.0
    exact = rng.uniform(size=len(df)) < 0.2
    lo = np.where(exact, df["ys"], lo)
    hi = np.where(exact, df["ys"], hi)
    lo = np.where(df["ys"] < -1, np.nan, lo)
    hi = np.where(df["ys"] < -1, -1.0, hi)
    df["lo"], df["hi"] = lo, hi
    return df.drop(columns="ys")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    make_panel().to_csv(f"{out}/retobit.csv", index=False)
