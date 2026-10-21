"""Individual-level panel for sp.did_multiplegt vs Stata did_multiplegt_old.

Treatment varies at the group x period level (on and off switches, plus a
count variant); several individuals per cell, with a cell size that varies
by group and period, so a cell-size weighting rule is visible in the
estimate. ``da`` / ``ya`` is an absorbing (staggered) version with a
dynamic effect. Writes dcdh_disaggregated_panel.csv next to this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def build(seed: int = 20261021) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(40):
        on = int(rng.integers(3, 7))
        off = int(rng.integers(on + 1, 10))
        never = g < 8
        fx = rng.normal(scale=0.5)
        for t in range(1, 9):
            d = 0 if never else int(on <= t < off)
            d2 = d * int(1 + (g % 3))  # count treatment: 0, 1, 2 or 3
            da = 0 if never else int(t >= on)  # absorbing (staggered)
            n = int(rng.integers(1, 6))  # individuals in this cell
            for _ in range(n):
                e = rng.normal()
                rows.append(
                    {
                        "g": g,
                        "t": t,
                        "d": d,
                        "d2": d2,
                        "da": da,
                        "x": rng.normal(),
                        "y": fx + 0.2 * t + 0.7 * d + e,
                        "y2": fx + 0.2 * t + 0.3 * d2 + e,
                        "ya": fx + 0.2 * t + (0.5 + 0.2 * (t - on)) * da + e,
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).with_name("dcdh_disaggregated_panel.csv")
    build().to_csv(out, index=False, float_format="%.17g")
    print(out)
