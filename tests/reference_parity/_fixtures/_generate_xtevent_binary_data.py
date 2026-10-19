"""Binary staggered-adoption panel with a confound and proxies, for the
``sp.xtevent`` trend and proxy-IV fixtures (Stata ``xtevent``).

A confound ``eta`` drifts before adoption and moves both the outcome and
the proxies ``x`` and ``x2``; ``w`` is an external instrument for ``x``
(correlated with it, excluded from the outcome).

    python tests/reference_parity/_fixtures/_generate_xtevent_binary_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "xtevent_binary.csv"


def main() -> None:
    rng = np.random.default_rng(20261019)
    units, periods = 90, 16
    rows = []
    for i in range(1, units + 1):
        g = int(rng.integers(5, 13)) if rng.random() < 0.75 else 0
        a = rng.normal()
        eta = 0.0
        for t in range(1, periods + 1):
            lead = (g - t) if g else 99
            eta = 0.7 * eta + rng.normal() + (0.25 if 0 < lead <= 3 else 0.0)
            z = int(g > 0 and t >= g)
            w = rng.normal()
            x = eta + 0.5 * w + 0.5 * rng.normal()
            x2 = 0.8 * eta + rng.normal()
            y = a + 0.1 * t + 1.0 * z + 0.6 * eta + rng.normal()
            rows.append(
                (i, t, z, round(y, 6), round(x, 6), round(x2, 6), round(w, 6), i % 9)
            )
    pd.DataFrame(rows, columns=["id", "t", "z", "y", "x", "x2", "w", "cl"]).to_csv(
        OUT, index=False
    )


if __name__ == "__main__":
    main()
