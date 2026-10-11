"""Write the Berry (1994) parity dataset ``_fixtures/berry_logit_data.csv``.

120 markets, three nests, four firms, 6-14 products per market. Shares are
EXACT one-level nested-logit shares (no simulation) of

    delta_jt = 1.0 + 1.0 * x1_jt + 0.5 * x2_jt - 1.5 * p_jt + xi_jt,   rho = 0.6

    s_jt = exp(delta_jt / (1 - rho)) / D_g^rho / (1 + sum_h D_h^(1 - rho)),
    D_g  = sum_{k in g} exp(delta_kt / (1 - rho)),

with price endogenous (it loads on xi) and shifted by a cost variable ``w``
that is excluded from demand. The file carries no instruments: both
StatsPAI and pyblp build them from the characteristics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "_fixtures" / "berry_logit_data.csv"
TRUE = {"const": 1.0, "x1": 1.0, "x2": 0.5, "price": -1.5, "rho": 0.6}


def simulate(n_markets: int, seed: int, rho: float = TRUE["rho"]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_markets):
        J = int(rng.integers(6, 15))
        nest = rng.integers(0, 3, J)
        firm = rng.integers(0, 4, J)
        x1 = rng.normal(0.0, 1.0, J)
        x2 = rng.uniform(0.0, 2.0, J)
        w = rng.uniform(0.0, 1.5, J)
        xi = rng.normal(0.0, 0.4, J)
        price = 1.0 + 0.9 * w + 0.25 * x1 + 0.6 * xi + rng.uniform(0.0, 0.3, J)
        delta = (
            TRUE["const"]
            + TRUE["x1"] * x1
            + TRUE["x2"] * x2
            + TRUE["price"] * price
            + xi
        )
        ev = np.exp(delta / (1.0 - rho))
        D = np.array([ev[nest == g].sum() for g in range(3)])
        denom = 1.0 + (D[D > 0] ** (1.0 - rho)).sum()
        share = ev / D[nest] ** rho / denom
        frames.append(
            pd.DataFrame(
                {
                    "market_id": t,
                    "product_id": np.arange(J),
                    "firm": firm,
                    "nest": nest,
                    "share": share,
                    "price": price,
                    "x1": x1,
                    "x2": x2,
                    "w": w,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    simulate(120, seed=1994).to_csv(OUT, index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
