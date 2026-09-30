"""Two-period shift-share panel for ``sp.rotemberg_summary`` (the
Goldsmith-Pinkham, Sorkin & Swift summary table): 120 units x 2 periods,
25 industries with period-specific shares and shocks, two controls, analytic
weights. Written long (one row per unit-period, one share column per
industry) and shocks as industry x period.

    python tests/reference_parity/_fixtures/_generate_rotemberg_summary_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20261003)
    units, K = 120, 25
    inds = np.arange(101, 101 + K)
    rows, shocks = [], []
    g = {}
    for year in (1990, 2000):
        base = rng.normal(0.5, 1.0, K) + (0.3 if year == 2000 else 0.0)
        for k, ind in enumerate(inds):
            g[(year, ind)] = base[k]
            shocks.append(dict(year=year, ind=int(ind), g=base[k]))
    conc = rng.gamma(0.6, size=(units, K))
    # Industry-specific first stages, some negative: negative Rotemberg
    # weights, as in the applications GPSS summarise.
    pi = rng.normal(1.0, 1.2, K)
    for u in range(units):
        for year in (1990, 2000):
            s = conc[u] * rng.uniform(0.7, 1.3, K)
            s = 0.4 * s / s.sum()
            zk = np.array([s[k] * g[(year, inds[k])] for k in range(K)])
            c1, c2 = rng.normal(size=2)
            x = float(pi @ zk) + 0.3 * c1 + rng.normal(scale=0.4)
            y = -0.6 * x + 0.2 * c2 + rng.normal(scale=0.5) + 0.3 * (year == 2000)
            row = dict(
                unit=u + 1,
                year=year,
                y=y,
                x=x,
                c1=c1,
                c2=c2,
                w=float(rng.uniform(0.5, 3.0)),
                t2=int(year == 2000),
            )
            row.update({f"sh{ind}": s[k] for k, ind in enumerate(inds)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        HERE / "rotemberg_panel.csv", index=False, float_format="%.17g"
    )
    pd.DataFrame(shocks).to_csv(
        HERE / "rotemberg_shocks.csv", index=False, float_format="%.17g"
    )
    print("wrote rotemberg_panel.csv, rotemberg_shocks.csv")


if __name__ == "__main__":
    main()
