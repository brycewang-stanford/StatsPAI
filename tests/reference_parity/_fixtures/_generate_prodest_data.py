#!/usr/bin/env python3
"""Write ``prodest_panel.csv`` for ``test_prodest_parity.py``.

A seeded Cobb-Douglas firm panel that satisfies the proxy-variable timing
assumptions (AR(1) productivity, predetermined capital, materials and
investment strictly monotone in productivity given capital). Labour responds
to productivity (the Ackerberg-Caves-Frazer case) and to a persistent,
firm-specific wage shock that does not move materials or investment, so
lagged labour is a relevant instrument for labour once the control function
has absorbed productivity and capital.

The panel is deliberately unbalanced: one firm in eight skips a year in the
middle of its history and one in five enters late. A lag operator that shifts
by position instead of by calendar year pairs the wrong periods across those
gaps, which is one of the things the parity test pins.

Run from this directory (then the R and Stata generators)::

    python3 _generate_prodest_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "prodest_panel.csv"

BETA_L, BETA_K = 0.60, 0.30
RHO, SIGMA_XI, SIGMA_ETA = 0.70, 0.20, 0.10


def simulate(seed: int = 20260913, n_firms: int = 250, n_years: int = 10):
    rng = np.random.default_rng(seed)
    rows = []
    for fid in range(1, n_firms + 1):
        omega = rng.normal(0.0, SIGMA_XI / np.sqrt(1.0 - RHO**2))
        k = rng.normal(1.0, 0.5)
        first = 2000 + (int(rng.integers(1, 4)) if fid % 5 == 0 else 0)
        skip = 2000 + int(rng.integers(4, 7)) if fid % 8 == 0 else None
        wage = rng.normal(0.0, 0.3 / np.sqrt(1.0 - 0.8**2))
        for year in range(2000, 2000 + n_years):
            omega = RHO * omega + rng.normal(0.0, SIGMA_XI)
            wage = 0.8 * wage + rng.normal(0.0, 0.3)
            ell = 0.5 * omega + 0.3 * k - 0.6 * wage + rng.normal(0.5, 0.05)
            m = 0.8 * omega + 0.5 * k + rng.normal(0.0, 0.05)
            inv = np.exp(0.5 + 0.6 * omega + 0.3 * k + rng.normal(0.0, 0.05))
            y = BETA_L * ell + BETA_K * k + omega + rng.normal(0.0, SIGMA_ETA)
            if year >= first and year != skip:
                rows.append(
                    {
                        "id": fid,
                        "year": year,
                        "y": y,
                        "l": ell,
                        "k": k,
                        "m": m,
                        "lninv": np.log(inv),
                    }
                )
            k = 0.9 * k + 0.1 * np.log(inv)
    return pd.DataFrame(rows)


def main() -> None:
    df = simulate()
    df.to_csv(OUT, index=False, float_format="%.10f")
    gaps = (df.groupby("id")["year"].diff() > 1).sum()
    print(
        f"wrote {OUT.name}: {len(df)} rows, {df['id'].nunique()} firms, "
        f"{gaps} within-firm gaps"
    )


if __name__ == "__main__":
    main()
