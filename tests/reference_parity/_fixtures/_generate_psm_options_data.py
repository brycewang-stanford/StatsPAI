"""Write ``psm_options_data.csv``, the data behind the matching option fixtures.

Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_psm_options_data.py

then ``_generate_psm_options_stata.do`` (Stata 18) and
``_generate_psm_options_R.R`` (R ``Matching``), both from this directory.

The covariates are continuous (one binary column aside), so neither the
propensity score nor any covariate distance has ties: the option grid is
about the options, not about how each package breaks ties. Values are
written with 17 significant digits and read by Stata with ``asdouble``,
so all three programs compute on the same doubles.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "psm_options_data.csv"


def make(n: int = 500, seed: int = 20261015) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.uniform(-1.5, 1.5, size=n)
    x4 = rng.binomial(1, 0.4, size=n).astype(float)
    index = -0.8 + 0.7 * x1 - 0.5 * x2 + 0.4 * x3 + 0.5 * x4
    d = rng.binomial(1, 1.0 / (1.0 + np.exp(-index))).astype(float)
    noise = rng.normal(size=n) * (1.0 + 0.5 * np.abs(x1))
    y = 1.0 + 1.5 * d + 2.0 * x1 + x2 - x3 + 0.5 * x4 + 0.5 * d * x1 + noise
    return pd.DataFrame(
        {"id": np.arange(1, n + 1), "y": y, "d": d, "x1": x1, "x2": x2,
         "x3": x3, "x4": x4}
    )  # fmt: skip


if __name__ == "__main__":
    make().to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")
