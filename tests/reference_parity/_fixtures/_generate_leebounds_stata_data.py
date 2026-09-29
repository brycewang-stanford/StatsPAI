"""Datasets for the ``sp.lee_bounds(trimming='leebounds')`` vs Stata fixture.

Eight samples (``sample`` 1-8): continuous outcomes stored as float32 -- the
storage that makes ``leebounds``' 16-digit threshold macro land beside the
data value -- with either arm retained more often, plus two integer outcomes
with heavy ties (the tie branch).

    python tests/reference_parity/_fixtures/_generate_leebounds_stata_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "leebounds_samples.csv"


def main() -> None:
    frames = []
    for k in range(1, 9):
        rng = np.random.default_rng(20260929 + k)
        n = 700 + 37 * k
        d = rng.integers(0, 2, n)
        hi = 0.95 if k % 2 else 0.8
        lo = 0.8 if k % 2 else 0.95
        s = (rng.random(n) < np.where(d == 1, hi, lo)).astype(int)
        if k <= 6:
            y = np.float32(np.exp(rng.normal(4 + 0.3 * d, 1.2, n)))
        else:
            y = rng.poisson(2 + d, n).astype(float)
        y = np.where(s == 1, y.astype(float), np.nan)
        frames.append(pd.DataFrame(dict(sample=k, d=d, s=s, y=y)))
    pd.concat(frames).to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
