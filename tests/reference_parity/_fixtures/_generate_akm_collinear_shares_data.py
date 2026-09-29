"""Shift-share data with collinear share columns, for the AKM parity fixture.

300 locations, 40 industries. Column 5 duplicates column 4, column 12 is the
sum of columns 10 and 11, and column 20 is column 19 plus 1e-12 noise -- the
cases R's ``qr()`` (LINPACK dqrdc2) drops before AKM inference. A shift-share
instrument, an endogenous regressor and an outcome follow a just-identified
design with two controls.

    python tests/reference_parity/_fixtures/_generate_akm_collinear_shares_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20260929)
    n, K = 300, 40
    W = rng.dirichlet(np.ones(K) * 0.5, size=n)
    W[:, 5] = W[:, 4]
    W[:, 12] = W[:, 10] + W[:, 11]
    W[:, 20] = W[:, 19] + rng.normal(size=n) * 1e-12
    g = rng.normal(size=K)
    z = W @ g
    c1, c2 = rng.normal(size=n), rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.8 * z + 0.3 * c1 + 0.5 * u + rng.normal(scale=0.3, size=n)
    y = 1.5 * x - 0.2 * c2 + u
    pd.DataFrame(dict(y=y, x=x, z=z, c1=c1, c2=c2)).to_csv(
        HERE / "akm_collinear_loc.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(W).to_csv(
        HERE / "akm_collinear_shares.csv",
        index=False,
        header=False,
        float_format="%.17g",
    )
    pd.DataFrame({"g": g}).to_csv(
        HERE / "akm_collinear_shocks.csv", index=False, float_format="%.17g"
    )
    print("wrote akm_collinear_{loc,shares,shocks}.csv")


if __name__ == "__main__":
    main()
