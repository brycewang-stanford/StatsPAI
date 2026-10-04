"""Staggered panel with covariates for ``test_stata_did_commands_parity.py``.

Four hundred units over 2001-2006; cohorts first treated in 2003, 2004 and
2005 plus a never-treated group. Two time-invariant covariates drive both
the cohort a unit falls in and its untreated trend, so the unconditional
comparison is biased and the estimators that use the covariates differ from
one another (outcome regression, weighting, doubly robust, improved doubly
robust). ``xt`` is a covariate that varies within unit. Run from this folder: ``python _generate_did_commands_data.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261004)
n, years = 400, np.arange(2001, 2007)
x1 = rng.normal(size=n)
x2 = (rng.uniform(size=n) < 0.4).astype(int)
score = 0.6 * x1 + 0.8 * x2 + rng.logistic(size=n)
cohort = np.select(
    [score > 1.6, score > 0.7, score > -0.1], [2003, 2004, 2005], default=0
)
alpha = rng.normal(size=n) + 0.5 * x1
rows = []
for i in range(n):
    for t in years:
        trend = (0.3 + 0.25 * x1[i] - 0.2 * x2[i]) * (t - 2001)
        on = cohort[i] > 0 and t >= cohort[i]
        effect = (1.0 + 0.3 * (t - cohort[i]) + 0.4 * x2[i]) if on else 0.0
        rows.append(
            {
                "id": i + 1,
                "year": int(t),
                "g": int(cohort[i]),
                "x1": round(float(x1[i]), 6),
                "x2": int(x2[i]),
                "y": round(float(alpha[i] + trend + effect + rng.normal()), 6),
            }
        )
out = pd.DataFrame(rows)
out["d04"] = (out["g"] == 2004).astype(int)
# A covariate that moves over time, from its own stream so the columns above
# are unchanged: csdid conditions each ATT(g,t) on its value in the earlier
# period of the cell.
rng_t = np.random.default_rng(20261005)
out["xt"] = np.round(
    out["x1"] + 0.3 * (out["year"] - 2001) * out["x2"] + rng_t.normal(size=len(out)),
    6,
)
out.to_csv(Path(__file__).with_name("did_commands_data.csv"), index=False)
print(out.groupby("g")["id"].nunique())
