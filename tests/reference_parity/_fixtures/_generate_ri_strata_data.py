"""Small stratified experiments for the ``sp.ri_test(strata=)`` fixture.

``ri_strata_units.csv``: 3 strata x 6 units, 3 treated per stratum
(20^3 = 8,000 assignments, enumerable). ``ri_strata_clusters.csv``: 4 strata
x 4 clusters of 5 units, 2 treated clusters per stratum (6^4 = 1,296).

    python tests/reference_parity/_fixtures/_generate_ri_strata_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20260929)
    rows = []
    for s in range(3):
        treat = rng.permutation([1, 1, 1, 0, 0, 0])
        for u in range(6):
            x = rng.normal()
            rows.append(
                dict(
                    block=s,
                    Z=treat[u],
                    x=x,
                    y=0.6 * treat[u] + 0.8 * x + s + rng.normal(),
                )
            )
    pd.DataFrame(rows).to_csv(
        HERE / "ri_strata_units.csv", index=False, float_format="%.17g"
    )
    rows = []
    for s in range(4):
        treat = rng.permutation([1, 1, 0, 0])
        for c in range(4):
            shock = rng.normal()
            for u in range(5):
                x = rng.normal()
                rows.append(
                    dict(
                        block=s,
                        clust=4 * s + c,
                        Z=treat[c],
                        x=x,
                        y=0.5 * treat[c] + 0.5 * x + shock + rng.normal(),
                    )
                )
    pd.DataFrame(rows).to_csv(
        HERE / "ri_strata_clusters.csv", index=False, float_format="%.17g"
    )
    print("wrote ri_strata_{units,clusters}.csv")


if __name__ == "__main__":
    main()
