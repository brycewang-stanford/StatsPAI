"""Geo panel for ``sp.hdfe_ols(vce='conley', conley_time=...)``: 150 units
at fixed coordinates, 8 periods, spatially and serially correlated errors.

    python tests/reference_parity/_fixtures/_generate_hdfe_conley_panel_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "hdfe_conley_panel.csv"


def main() -> None:
    rng = np.random.default_rng(20261004)
    units, periods = 150, 8
    lat = rng.uniform(30, 40, units)
    lon = rng.uniform(-100, -90, units)
    # spatially smooth unit component + AR(1) in time
    field = np.sin(lat / 2.0) + np.cos(lon / 3.0)
    rows = []
    for i in range(units):
        e_prev = 0.0
        for t in range(1, periods + 1):
            e = 0.6 * e_prev + rng.normal()
            e_prev = e
            x1 = field[i] + rng.normal()
            x2 = rng.normal()
            y = 0.5 * x1 - 0.2 * x2 + field[i] * 0.8 + 0.1 * t + e
            rows.append(dict(id=i + 1, t=t, lat=lat[i], lon=lon[i], x1=x1, x2=x2, y=y))
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
