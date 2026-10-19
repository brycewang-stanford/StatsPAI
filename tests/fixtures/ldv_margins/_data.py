"""Simulated data behind ``tests/test_ldv_margins_stata.py``.

``python tests/fixtures/ldv_margins/_data.py <dir>`` writes ``ldv.csv`` there
for ``stata_margins.do`` (run with Stata 18 MP in batch mode from that
directory; it writes ``stata_margins.tsv``, which ``_to_json`` turns into
``stata_margins.json``).
"""

import sys

import numpy as np
import pandas as pd


def make_ldv_frame(seed: int = 20261019, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "g": rng.integers(1, 4, size=n),
        }
    )
    u = rng.normal(size=n)
    gd = np.where(df.g == 2, 0.4, np.where(df.g == 3, -0.3, 0.0))
    df["ys"] = 0.5 + df.x1 - 0.5 * df.x2 + gd + u
    df["yo"] = np.digitize(df.ys, [0, 1, 2])
    eta1 = 0.5 * df.x1 - 0.3 * df.x2 + gd
    eta2 = -0.4 * df.x1 + 0.6 * df.x2
    E = np.column_stack([np.ones(n), np.exp(eta1), np.exp(eta2)])
    P = E / E.sum(1, keepdims=True)
    cum = P.cumsum(1)
    r = rng.uniform(size=n)
    df["ym"] = (r[:, None] > cum).sum(1)
    df["yt"] = np.maximum(df.ys, 0)
    df["z"] = rng.normal(size=n)
    e2 = 0.6 * u + 0.8 * rng.normal(size=n)
    df["s"] = (0.3 + df.z + 0.5 * df.x1 + e2 > 0).astype(int)
    df["yh"] = np.where(df.s == 1, df.ys, np.nan)
    df["xe"] = df.z + 0.5 * u + rng.normal(size=n)
    df["yb"] = (df.x1 + 0.5 * df.xe + u > 0).astype(int)
    df["w"] = 0.5 + rng.uniform(size=n)
    df["ybh"] = np.where(
        df.s == 1, (df.ys + 0.3 * rng.normal(size=n) > 0.6).astype(float), np.nan
    )
    df["g2"] = (df.g == 2).astype(float)
    df["g3"] = (df.g == 3).astype(float)
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    make_ldv_frame().to_csv(f"{out}/ldv.csv", index=False)
