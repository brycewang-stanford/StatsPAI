"""Data for ``sp.lee_bounds(covariates=...)`` (Stata ``leebounds, tight()``):
a randomised treatment raising retention, with retention and outcomes that
differ across a three-level covariate and a binary one.

    python tests/reference_parity/_fixtures/_generate_leebounds_tight_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "leebounds_tight.csv"


def main() -> None:
    rng = np.random.default_rng(20260930)
    n = 3000
    d = rng.integers(0, 2, n)
    x = rng.integers(0, 3, n)
    z = rng.integers(0, 2, n)
    p_sel = 0.45 + 0.12 * x + 0.1 * d + 0.05 * z * d
    s = (rng.uniform(size=n) < p_sel).astype(int)
    y = 1.0 + 0.6 * d + 0.8 * x - 0.3 * z + rng.normal(size=n)
    y = np.where(s == 1, np.round(y, 6), np.nan)
    pd.DataFrame(dict(y=y, d=d, s=s, x=x, z=z)).to_csv(
        OUT, index=False, float_format="%.17g"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
