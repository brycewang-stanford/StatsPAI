"""Data for the R black-box fixture of sp.lprobust_at_point / sp.lpbwselect_*.

    python tests/reference_parity/_fixtures/_generate_lprobust_selectors_data.py
    Rscript tests/reference_parity/_fixtures/_generate_lprobust_selectors_R.R

Two designs: a smooth one on [-1, 2] and one with the running variable
rounded to 0.1, so nearest-neighbour matching meets ties.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261020)
x1 = rng.uniform(-1.0, 2.0, 300)
y1 = np.sin(2.0 * x1) + x1**2 / 3.0 + rng.normal(0.0, 0.4, 300)
x2 = np.round(rng.normal(0.0, 1.0, 250), 1)
y2 = np.exp(x2 / 2.0) + rng.normal(0.0, 0.3, 250)
out = pd.DataFrame(
    {
        "design": ["smooth"] * 300 + ["ties"] * 250,
        "x": np.r_[x1, x2],
        "y": np.r_[y1, y2],
    }
)
path = Path(__file__).with_name("lprobust_selectors_data.csv")
out.to_csv(path, index=False, float_format="%.17g")
print(f"wrote {path.name}")
