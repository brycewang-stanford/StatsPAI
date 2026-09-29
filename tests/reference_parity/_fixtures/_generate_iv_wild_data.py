"""Data for the ``sp.ivreg(vce='wild')`` vs Stata ``boottest`` (WRE) fixture.

400 observations in 10 clusters, one endogenous regressor, one instrument,
one exogenous control, cluster-level shocks in both equations.

    python tests/reference_parity/_fixtures/_generate_iv_wild_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "iv_wild_data.csv"


def main() -> None:
    rng = np.random.default_rng(20260929)
    n, G = 400, 10
    cl = np.repeat(np.arange(G), n // G)
    a = rng.normal(size=G)[cl]
    z = rng.normal(size=n) + 0.5 * rng.normal(size=G)[cl]
    w = rng.normal(size=n)
    v = rng.normal(size=n) + 0.5 * a
    x = 0.6 * z + 0.3 * w + v
    y = 0.4 * x + 0.5 * w + 0.7 * v + a + rng.normal(size=n)
    pd.DataFrame(dict(y=y, x=x, z=z, w=w, cl=cl)).to_csv(
        OUT, index=False, float_format="%.17g"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
