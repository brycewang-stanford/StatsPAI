"""Repeated cross-section for the ``sp.didregress(aggregate=...)`` fixtures.

Run this, then ``_generate_didregress_dlang_Stata.do`` in Stata 18, which
writes ``didregress_dlang_Stata.csv``.

Ten groups observed in six periods, 4 to 30 individuals per group-period
cell; groups 1-2 treated from period 3, groups 3-4 from period 5. ``x``
and ``w`` vary within cells, ``z`` is constant within each group-period
cell (a group-time covariate). ``didregress_dlang_panel.csv`` is a panel
(60 units in 8 groups, 6 periods) for ``xtdidregress``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20261019)
    rows = []
    a = rng.normal(size=10)
    b = rng.normal(size=6)
    # multiples of 1/16: exact in float and double storage alike
    z = np.round(rng.normal(size=(10, 6)) * 16) / 16
    for g in range(1, 11):
        for t in range(1, 7):
            d = int((g <= 2 and t >= 3) or (g in (3, 4) and t >= 5))
            n = int(rng.integers(4, 31))
            for _ in range(n):
                x = rng.normal() + 0.2 * g
                w = rng.uniform()
                y = (
                    a[g - 1]
                    + b[t - 1]
                    + 0.8 * x
                    - 0.5 * w * (1 + 0.1 * g)
                    + 0.3 * z[g - 1, t - 1]
                    + 1.2 * d
                    + rng.normal()
                )
                rows.append(
                    (g, t, d, round(y, 5), round(x, 5), round(w, 5), z[g - 1, t - 1])
                )
    pd.DataFrame(rows, columns=["g", "t", "d", "y", "x", "w", "z"]).to_csv(
        HERE / "didregress_dlang.csv", index=False
    )

    rows = []
    for i in range(1, 61):
        g = (i - 1) % 8 + 1
        a_i = rng.normal()
        for t in range(1, 7):
            d = int((g <= 2 and t >= 3) or (g == 3 and t >= 5))
            x = rng.normal() + 0.1 * t
            y = a_i + 0.2 * t + 0.7 * x + 1.0 * d + rng.normal()
            rows.append((i, g, t, d, round(y, 5), round(x, 5)))
    pd.DataFrame(rows, columns=["id", "g", "t", "d", "y", "x"]).to_csv(
        HERE / "didregress_dlang_panel.csv", index=False
    )


if __name__ == "__main__":
    main()
