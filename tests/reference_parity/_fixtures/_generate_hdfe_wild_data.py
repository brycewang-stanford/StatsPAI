"""Panel for the ``sp.hdfe_ols(wild=True)`` vs Stata ``boottest`` fixture.

60 units in 12 clusters over 10 years; unit effects nested in the clusters,
a year effect that is not, cluster-level shocks, and a treatment dose that
switches on in year 6 -- the case where the bootstrap must re-absorb the year
effect.

    python tests/reference_parity/_fixtures/_generate_hdfe_wild_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "hdfe_wild_panel.csv"


def main() -> None:
    rng = np.random.default_rng(20260929)
    units, years, G = 60, 10, 12
    cl = np.arange(units) % G
    dose = rng.gamma(2.0, 0.5, units)
    a_u = rng.normal(size=units)
    a_t = rng.normal(size=years)
    shock = rng.normal(scale=0.8, size=(G, years))
    rows = []
    for u in range(units):
        for t in range(years):
            post = float(t >= 5)
            x2 = rng.normal()
            y = (
                0.15 * dose[u] * post
                + 0.3 * x2
                + a_u[u]
                + a_t[t]
                + shock[cl[u], t]
                + rng.normal()
            )
            rows.append(
                dict(unit=u, year=2000 + t, cl=cl[u], d=dose[u] * post, x2=x2, y=y)
            )
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
