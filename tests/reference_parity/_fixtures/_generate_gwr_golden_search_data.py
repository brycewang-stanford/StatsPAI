"""Simulated point data for the GWR bandwidth-search black-box fixture.

90 points on a 100 x 60 rectangle with two regressors whose effects vary
smoothly in space. ``_generate_gwr_golden_search_R.R`` runs R ``GWmodel``'s
bandwidth search on it as a black box.

    python tests/reference_parity/_fixtures/_generate_gwr_golden_search_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20261020)
    n = 90
    u = rng.uniform(0, 100, n)
    v = rng.uniform(0, 60, n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1 + (0.5 + u / 50) * x1 + np.sin(v / 15) * x2 + rng.normal(0, 0.5, n)
    pd.DataFrame({"u": u, "v": v, "y": y, "x1": x1, "x2": x2}).to_csv(
        HERE / "gwr_golden_search_points.csv", index=False, float_format="%.17g"
    )


if __name__ == "__main__":
    main()
