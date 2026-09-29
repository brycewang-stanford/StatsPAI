"""Data for the ``sp.suest`` vs Stata ``suest`` fixture.

300 observations in 30 clusters, three outcomes sharing cluster shocks; the
second outcome has three missing values, so the equations use different
samples.

    python tests/reference_parity/_fixtures/_generate_suest_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "suest_data.csv"


def main() -> None:
    rng = np.random.default_rng(20260929)
    n, G = 300, 30
    cl = np.repeat(np.arange(G), n // G)
    a = rng.normal(size=G)[cl]
    x, w = rng.normal(size=n), rng.normal(size=n)
    d = pd.DataFrame(
        dict(
            y1=0.5 * x + 0.3 * w + a + rng.normal(size=n),
            y2=0.4 * x + a + rng.normal(size=n),
            y3=0.1 * x - 0.2 * w + rng.normal(size=n),
            x=x,
            w=w,
            cl=cl,
        )
    )
    d.loc[[3, 17, 40], "y2"] = np.nan
    d.to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
