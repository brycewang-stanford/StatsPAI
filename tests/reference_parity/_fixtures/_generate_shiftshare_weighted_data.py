"""Weighted shift-share data for the ``weights=`` fixtures of
``sp.ssaggregate`` / ``sp.bartik`` / ``sp.shift_share_se``.

400 locations, 25 industries, location weights (population-like), two
controls, a just-identified design with the shift-share instrument.

    python tests/reference_parity/_fixtures/_generate_shiftshare_weighted_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20260929)
    n, K = 400, 25
    W = rng.dirichlet(np.ones(K) * 0.4, size=n)
    g = rng.normal(size=K)
    z = W @ g
    c1, c2 = rng.normal(size=n), rng.normal(size=n)
    w = rng.lognormal(3, 1, size=n)
    u = rng.normal(size=n)
    x = 0.8 * z + 0.3 * c1 + 0.5 * u + rng.normal(scale=0.4, size=n)
    y = -0.6 * x + 0.2 * c2 + u
    pd.DataFrame(dict(y=y, x=x, z=z, c1=c1, c2=c2, w=w)).to_csv(
        HERE / "shiftshare_weighted_loc.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(W).to_csv(
        HERE / "shiftshare_weighted_shares.csv",
        index=False,
        header=False,
        float_format="%.17g",
    )
    pd.DataFrame({"g": g}).to_csv(
        HERE / "shiftshare_weighted_shocks.csv", index=False, float_format="%.17g"
    )
    print("wrote shiftshare_weighted_{loc,shares,shocks}.csv")


if __name__ == "__main__":
    main()
