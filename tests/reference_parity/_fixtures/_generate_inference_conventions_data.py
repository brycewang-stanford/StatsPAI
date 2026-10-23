"""Data of the 2026-10-23 external audit of t / F / p conventions vs Stata 18.

The data-generating process is the auditor's (generate_data.py of the
audit bundle, seed 20261025), kept verbatim so that the numbers here can
be set beside theirs: 30 groups x 8 periods, an endogenous regressor with
two instruments, binary / count / panel outcomes, a column collinear with
x1 (dup = 2 * x1), and missing values in an outcome and a cluster column.

    python tests/reference_parity/_fixtures/_generate_inference_conventions_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).with_name("inference_conventions.csv")

rng = np.random.default_rng(20261025)
n = 240
g = np.repeat(np.arange(30), 8)
t = np.tile(np.arange(8), 30)
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
z1 = rng.normal(size=n)
z2 = rng.normal(size=n)
u = rng.normal(size=n)
end = 0.8 * z1 + 0.4 * z2 + 0.5 * u + rng.normal(size=n)
d = pd.DataFrame(
    dict(
        x1=x1,
        x2=x2,
        z1=z1,
        z2=z2,
        xd=end,
        y=1 + 0.3 * x1 - 0.2 * x2 + 0.6 * end + u,
        g=g,
        t=t,
        seq=np.arange(n) + 1,
        w=np.exp(0.3 * x1),
        fw=rng.integers(1, 4, n),
        cat=rng.integers(0, 3, n),
    )
)
d["yb"] = rng.binomial(1, 1 / (1 + np.exp(-(0.2 + 0.5 * x1 - 0.3 * x2))))
d["yc"] = rng.poisson(np.exp(0.3 + 0.2 * x1 - 0.2 * x2))
d["yp"] = 1 + 0.3 * x1 - 0.2 * x2 + rng.normal(size=30)[g] + rng.normal(size=n)
d["dup"] = 2 * x1
d["ym"] = d.y
d.loc[[1, 5, 12, 68], "ym"] = np.nan
d["gm"] = d.g.astype(float)
d.loc[[2, 6], "gm"] = np.nan
for i in range(1, n):
    d.loc[i, "y"] += 0.25 * (d.loc[i - 1, "y"] - 1)
d.to_csv(OUT, index=False)
