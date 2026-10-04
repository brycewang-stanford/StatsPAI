"""Panel for test_dcdh_textbook_stata_parity.py.

One synthetic panel that exercises what the de Chaisemartin and
D'Haultfoeuille textbook applications need and the earlier fixtures did not
have: a count treatment that moves in both directions from different
starting levels, periods four years apart, a binary staggered treatment with
missing outcomes (an unbalanced panel), and observation weights that vary
over time.

Every starting level of the count treatment keeps groups that never change,
so each switcher has a control at every horizon: the reference command's
handling of control-less interior periods (see the dev note of 2026-10-05)
does not come into play and the two implementations estimate on the same
sample.

Run: python _generate_dcdh_textbook_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261005)
G, T = 150, 8
rows = []
for g in range(1, G + 1):
    base = int(rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15]))
    switch = int(rng.choice([0, 3, 4, 5, 6, 7], p=[0.3, 0.15, 0.15, 0.15, 0.15, 0.1]))
    cohort = int(rng.choice([0, 3, 4, 5, 6], p=[0.3, 0.2, 0.2, 0.15, 0.15]))
    alpha = rng.normal()
    alpha2 = rng.normal()
    w0 = float(np.round(rng.uniform(0.5, 3.0), 3))
    d_prev, d = base, base
    for t in range(1, T + 1):
        if switch and t == switch:
            d = max(base + int(rng.choice([-1, 1, 2], p=[0.25, 0.55, 0.2])), 0)
            if d == base:
                d = base + 1
        elif switch and t > switch:
            d = max(d + int(rng.choice([-1, 0, 1], p=[0.15, 0.7, 0.15])), 0)
        x = 0.3 * t + rng.normal()
        y = alpha + 0.2 * t + 0.5 * d + 0.2 * d_prev + 0.3 * x + rng.normal(scale=0.7)
        d2 = int(cohort > 0 and t >= cohort)
        effect = 0.4 * (t - cohort + 1) if d2 else 0.0
        y2 = alpha2 + 0.1 * t + effect + rng.normal(scale=0.6)
        if rng.uniform() < 0.07:
            y2 = np.nan
        rows.append(
            {
                "g": g,
                "t": t,
                "year": 1996 + 4 * t,
                "state": (g - 1) // 6 + 1,
                "d": d,
                "x": round(x, 6),
                "y": round(y, 6),
                "w": w0,
                "wt": round(w0 * (1 + 0.05 * t), 6),
                "cohort": cohort,
                "d2": d2,
                "y2": None if np.isnan(y2) else round(y2, 6),
            }
        )
        d_prev = d
df = pd.DataFrame(rows)
df["dy"] = df.groupby("g")["y"].diff()
df["dd"] = df.groupby("g")["d"].diff()
out = Path(__file__).with_name("dcdh_textbook_data.csv")
df.to_csv(out, index=False)
print(out, df.shape)
print(df.groupby("g")["d"].first().value_counts().sort_index().to_dict())
