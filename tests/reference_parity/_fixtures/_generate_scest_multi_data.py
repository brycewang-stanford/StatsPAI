"""Synthetic staggered panel for tests/reference_parity/test_scest_multi_parity.py.

12 units, 2001-2020, a two-factor outcome; u01, u02 and u03 adopt in 2012,
2014 and 2015 (treatment column ``treat``), the other nine are never treated.
    python tests/reference_parity/_fixtures/_generate_scest_multi_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(20261020)
N, T = 12, 20
f = np.cumsum(rng.normal(0, 1, size=(T, 2)), axis=0)
load = rng.uniform(0.2, 1.5, size=(N, 2))
alpha = rng.normal(0, 2, size=N)
adopt = {"u01": 12, "u02": 14, "u03": 15}
rows = []
for i in range(N):
    name = f"u{i + 1:02d}"
    for t in range(T):
        y = 10 + alpha[i] + load[i] @ f[t] + rng.normal(0, 0.5)
        d = int(name in adopt and t + 1 >= adopt[name])
        if d:
            y += 1.0 + 0.3 * (t + 1 - adopt[name])
        rows.append((name, 2000 + t + 1, y, d))
pd.DataFrame(rows, columns=["unit", "year", "y", "treat"]).to_csv(
    HERE / "scest_multi_panel.csv", index=False
)
