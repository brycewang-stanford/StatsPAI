"""Data for the summary-index parity test.

Writes ``summary_index_input.csv``: an experiment with 320 units, a binary
``treat`` (missing for three units), an analytic weight ``w`` and five
correlated components ``y1`` ... ``y5`` (``y4`` binary, ``y5`` higher is
worse). Columns ``m1`` ... ``m5`` are the same components with about 8%
of values missing at random, and two units missing every component. Each
value is written as the shortest string that round-trips the double.
Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_summary_index_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "summary_index_input.csv"


def make() -> pd.DataFrame:
    rng = np.random.default_rng(20261028)
    n = 320
    treat = (rng.uniform(size=n) < 0.5).astype(float)
    common = rng.normal(size=n)
    y1 = 2.0 + 0.4 * treat + common + rng.normal(size=n)
    y2 = 10.0 + 3.0 * (0.2 * treat + 0.7 * common + rng.normal(size=n))
    y3 = 0.5 * y1 + rng.normal(scale=2.0, size=n)
    y4 = (common + 0.3 * treat + rng.normal(size=n) > 0).astype(float)
    y5 = 5.0 - 0.3 * treat - 0.5 * common + rng.normal(size=n)
    w = rng.uniform(0.5, 3.0, size=n)
    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "treat": treat,
            "w": w,
            "y1": y1,
            "y2": y2,
            "y3": y3,
            "y4": y4,
            "y5": y5,
        }
    )
    df.loc[[17, 140, 301], "treat"] = np.nan
    for j in range(1, 6):
        m = df[f"y{j}"].copy()
        m[rng.uniform(size=n) < 0.08] = np.nan
        df[f"m{j}"] = m
    df.loc[[55, 200], [f"m{j}" for j in range(1, 6)]] = np.nan
    return df


def main() -> None:
    df = make()
    lines = [",".join(df.columns)]
    for row in df.itertuples(index=False):
        lines.append(
            ",".join(
                "" if pd.isna(v) else repr(float(v)) if isinstance(v, float) else str(v)
                for v in row
            )
        )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
