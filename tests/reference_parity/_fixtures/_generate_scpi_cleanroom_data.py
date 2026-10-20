"""Two synthetic panels for tests/reference_parity/test_scpi_cleanroom_parity.py.

``scpi_cleanroom_long.csv``: 6 donors and a treated unit ``tr``, periods
1-24, treatment from period 18 (T0 = 17, T1 = 7), a one-factor outcome.
``scpi_cleanroom_short.csv``: the same design with 20 periods, treatment from
13 (T0 = 12), so T0 - J = 6 and the ridge radius uses the screened design.
``scpi_cleanroom_tiny.csv``: 3 donors, 6 periods, treatment from 4 (T0 = 3),
where the L1-L2 ball of the simulation is enlarged.

    python tests/reference_parity/_fixtures/_generate_scpi_cleanroom_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def panel(seed: int, n_t: int, t0: int, n_donors: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=n_t))
    rows = []
    for j in range(n_donors + 1):
        lam, a = rng.uniform(0.5, 1.5), rng.normal()
        u = "tr" if j == n_donors else f"d{j:02d}"
        for t in range(n_t):
            eff = 2.0 if (u == "tr" and t >= t0) else 0.0
            rows.append((u, t + 1, a + lam * f[t] + rng.normal(0, 0.3) + eff))
    return pd.DataFrame(rows, columns=["unit", "time", "y"])


panel(0, 24, 17).to_csv(
    HERE / "scpi_cleanroom_long.csv", index=False, float_format="%.17g"
)
panel(5, 20, 12).to_csv(
    HERE / "scpi_cleanroom_short.csv", index=False, float_format="%.17g"
)
panel(306, 6, 3, n_donors=3).to_csv(
    HERE / "scpi_cleanroom_tiny.csv", index=False, float_format="%.17g"
)
