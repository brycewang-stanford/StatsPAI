"""Fixed panel for the weighted ``sp.fast.feols`` parity test.

Run once; commit ``fast_feols_weights.csv`` so the R generator
(``_generate_fast_feols_weights_R.R``) and the Python test read the same
bytes. Unbalanced two-way panel: 150 firms x up to 8 years, 25 clusters
(firms nested in ``g``), a heteroskedastic error and a weight unrelated
to the regressors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261003)
rows = []
for firm in range(150):
    a = rng.normal()
    for year in range(8):
        if rng.uniform() < 0.12:
            continue  # unbalanced
        x1 = rng.normal() + 0.5 * a
        x2 = rng.normal() + 0.1 * year
        e = rng.normal() * (0.6 + 0.4 * abs(x1))
        y = 1.0 + 0.8 * x1 - 0.5 * x2 + a + 0.2 * year + e
        rows.append((firm, year, firm % 25, y, x1, x2, rng.uniform(0.5, 3.0)))

df = pd.DataFrame(rows, columns=["firm", "year", "g", "y", "x1", "x2", "w"])
df.to_csv("fast_feols_weights.csv", index=False, float_format="%.17g")
print(f"wrote fast_feols_weights.csv with {len(df)} rows")
