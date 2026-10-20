"""Unbalanced simulated panel behind ``tests/test_xtivreg_stata.py``.

``python tests/fixtures/xtivreg/_data.py <dir>`` writes ``panel.csv`` for
``stata_xtivreg.do`` (Stata 18 MP, batch mode, run from that directory).
"""

import sys

import numpy as np
import pandas as pd


def make_panel(seed: int = 20261019) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(90):
        mu = rng.normal()
        w = rng.normal()  # time-invariant regressor
        for t in range(int(rng.integers(2, 9))):
            z1, z2, x1 = rng.normal(size=3)
            v = rng.normal()
            d = 0.8 * z1 + 0.5 * z2 + 0.3 * mu + 0.6 * v + rng.normal()
            y = 1.0 + 0.7 * d + 0.5 * x1 - 0.4 * w + mu + v
            rows.append((i + 1, t + 1, y, d, x1, w, z1, z2, i % 30 + 1))
    return pd.DataFrame(rows, columns=["id", "t", "y", "d", "x1", "w", "z1", "z2", "c"])


def make_balanced_panel(seed: int = 20261020) -> pd.DataFrame:
    """80 panels x 6 periods with a time-invariant regressor ``w``, its
    dyadic rounding ``wd`` (multiples of 1/8, so a panel mean of it is
    exact) and instrument ``zb`` (the panel mean of ``z1``), for the
    balanced-panel conventions of ``xtivreg, re``."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(80):
        mu = rng.normal()
        w = rng.normal()
        z1 = rng.normal(size=6)
        for t in range(6):
            z2, x1, v = rng.normal(size=3)
            d = 0.8 * z1[t] + 0.5 * z2 + 0.3 * mu + 0.6 * v + rng.normal()
            y = 1.0 + 0.7 * d + 0.5 * x1 - 0.4 * w + mu + v
            rows.append((i + 1, t + 1, y, d, x1, w, z1[t], z2, i % 20 + 1))
    df = pd.DataFrame(rows, columns=["id", "t", "y", "d", "x1", "w", "z1", "z2", "c"])
    df["wd"] = np.round(df["w"] * 8) / 8
    return add_panel_mean(df)


def add_panel_mean(df: pd.DataFrame) -> pd.DataFrame:
    """``zb``: the panel mean of ``z1`` (a time-invariant instrument)."""
    return df.assign(zb=df.groupby("id")["z1"].transform("mean"))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    add_panel_mean(make_panel()).to_csv(f"{out}/panel.csv", index=False)
    make_balanced_panel().to_csv(f"{out}/balanced.csv", index=False)
