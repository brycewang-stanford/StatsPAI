"""Simulated panel behind ``tests/test_xtregar_stata.py``.

``python tests/fixtures/xtregar/_data.py <dir>`` writes ``ar1panel.csv``
for ``stata_xtregar.do``: 80 panels of 4 to 12 periods with an AR(1)
disturbance (rho = 0.6) whose time index has gaps in about a sixth of the
periods (the first period of a panel is always kept), a panel effect
correlated with ``x1``, a time-invariant regressor ``z``, and a flag
``bal`` marking the panels without gaps.
"""

import sys

import numpy as np
import pandas as pd


def make_panel(seed: int = 20261020) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(80):
        T = int(rng.integers(4, 13))
        nu = rng.normal()
        z = rng.normal()
        e = rng.normal() / np.sqrt(1 - 0.36)
        keep = rng.uniform(size=T) > 0.17
        keep[0] = True
        for t in range(T):
            if t > 0:
                e = 0.6 * e + rng.normal()
            x1 = 0.5 * nu + rng.normal()
            x2 = rng.normal()
            y = 1.0 + 0.8 * x1 - 0.4 * x2 + 0.5 * z + nu + e
            if keep[t]:
                rows.append((i + 1, t + 1, y, x1, x2, z))
    df = pd.DataFrame(rows, columns=["id", "t", "y", "x1", "x2", "z"])
    gaps = df.groupby("id")["t"].transform(lambda s: int((s.diff().fillna(1) > 1).any()))
    df["bal"] = 1 - gaps
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    make_panel().to_csv(f"{out}/ar1panel.csv", index=False)
