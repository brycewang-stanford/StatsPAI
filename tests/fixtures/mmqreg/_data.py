"""Fixture panel for tests/test_mmqreg.py (written to mmqr_panel.csv for
the Stata reference ``stata_mmqreg.do``).

Location-scale DGP with panel effects in both location and scale, 60
unbalanced panels plus one single-observation panel, a period variable for
two-way effects, half the panels with skewed errors.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_panel() -> pd.DataFrame:
    rng = np.random.default_rng(20261020)
    rows = []
    for i in range(60):
        a = rng.normal()
        d = abs(rng.normal()) * 0.5 + 0.5
        for t in range(7 if i % 5 else 5):
            x1 = rng.normal() + 0.3 * a
            x2 = rng.uniform(0, 2)
            sig = d + 0.4 * x2 + 0.2 * abs(x1) + 0.1 * t
            e = rng.chisquare(3) / 3 - 1 if i % 2 else rng.normal()
            y = a + 0.2 * t + 1.0 * x1 - 0.5 * x2 + sig * e
            rows.append((i + 1, t + 1, y, x1, x2, abs(x1)))
    rows.append((61, 3, 0.7, 0.4, 1.1, 0.4))  # a single-observation panel
    return pd.DataFrame(rows, columns=["id", "t", "y", "x1", "x2", "ax1"])


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    make_panel().to_csv(out / "mmqr_panel.csv", index=False, float_format="%.17g")
