"""Datasets for the missing-value sample rules of ``sp.lee_bounds``.

Three samples (two continuous outcomes, one integer outcome with ties). Each
carries copies of the outcome, treatment, selection and covariate with
missing values planted: a selected row with no outcome (``y2``), a missing
treatment (``d2``), a missing selection indicator (``s3``) and a missing
covariate (``xm``). Stata ``leebounds`` drops the first three kinds of row
from the whole computation.

    python tests/reference_parity/_fixtures/_generate_leebounds_missing_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "leebounds_missing.csv"


def main() -> None:
    frames = []
    for k, n in ((1, 300), (2, 600), (3, 500)):
        rng = np.random.default_rng(20261019 + k)
        d = rng.integers(0, 2, n)
        p_sel = 0.6 + 0.15 * d if k != 2 else 0.75 - 0.12 * d
        s = (rng.random(n) < p_sel).astype(int)
        if k < 3:
            y = rng.normal(1 + 0.3 * d, 1, n)
        else:
            y = rng.integers(0, 5, n).astype(float)
        x = rng.integers(0, 3, n)
        u = rng.random(n)
        y = np.where(s == 1, y, np.nan)
        frames.append(
            pd.DataFrame(
                {
                    "sample": k,
                    "y": y,
                    "d": d,
                    "s": s,
                    "x": x,
                    "y2": np.where((u < 0.06) & (s == 1), np.nan, y),
                    "d2": np.where(u > 0.96, np.nan, d),
                    "s3": np.where((u > 0.5) & (u < 0.53), np.nan, s),
                    "xm": np.where((u > 0.2) & (u < 0.24), np.nan, x),
                }
            )
        )
    pd.concat(frames).to_csv(OUT, index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
