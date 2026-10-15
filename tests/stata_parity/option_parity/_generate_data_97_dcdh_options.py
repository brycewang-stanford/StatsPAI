"""Data for 97_dcdh_options.do: one panel for every did_multiplegt_dyn option.

180 groups over 8 periods. The binary treatment ``d`` moves both ways:
never-switchers at 0 and at 1, groups that switch in, groups that switch out,
and some of each that later return to their period-one level, so
``switchers()`` / ``same_switchers`` / ``only_never_switchers`` each select a
different sample. Columns:

* ``g`` group, ``t`` period (1..8);
* ``d`` / ``y`` the binary treatment and its outcome;
* ``reg`` a time-invariant cell (1..3) with its own linear trend, for
  ``trends_nonparam(reg)``; every (reg, period-one treatment) cell has
  never-switchers, so no cell runs out of controls;
* ``x`` a time-varying covariate that enters the outcome;
* ``w`` a positive, time-varying observation weight;
* ``cl`` a cluster holding six groups (``cluster(cl)`` coarser than ``g``);
* ``keep`` 0 on 4% of the rows after period one: ``drop if keep == 0`` gives
  the unbalanced panel (the period-one row is kept, so every group keeps its
  baseline treatment);
* ``dc`` / ``yc`` a treatment whose period-one level is continuous (no two
  groups share it), changed by +1 or -0.5 at the switch, for
  ``continuous(1)`` / ``continuous(2)``.

    python tests/stata_parity/option_parity/_generate_data_97_dcdh_options.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(97)
G, T = 180, 8
rows = []
for g in range(1, G + 1):
    reg = 1 + (g - 1) % 3
    kind = rng.choice(
        ["never0", "never1", "in", "out", "in_back", "out_back"],
        p=[0.25, 0.15, 0.25, 0.15, 0.12, 0.08],
    )
    F = int(rng.integers(3, 8))
    back = F + int(rng.integers(2, 4))
    fe = rng.normal()
    w_g = np.exp(rng.normal(scale=0.4))
    dc0 = rng.uniform(0.0, 2.0)
    dc_step = 1.0 if rng.random() < 0.6 else -0.5
    dc_switches = rng.random() < 0.55
    Fc = int(rng.integers(3, 8))
    exposure = 0
    for t in range(1, T + 1):
        base = 1 if kind in ("never1", "out", "out_back") else 0
        if kind in ("in", "out"):
            d = base if t < F else 1 - base
        elif kind in ("in_back", "out_back"):
            d = 1 - base if F <= t < back else base
        else:
            d = base
        exposure = exposure + 1 if d != base else 0
        x = 0.3 * t + rng.normal()
        y = (
            fe
            + 0.15 * reg * t
            + (1.0 + 0.25 * exposure) * (d - base)
            + 0.5 * x
            + rng.normal(scale=0.6)
        )
        dc = dc0 + (dc_step if (dc_switches and t >= Fc) else 0.0)
        yc = fe + 0.2 * t + 0.4 * dc0 * t / T + 0.8 * dc + rng.normal(scale=0.5)
        keep = 1 if t == 1 or rng.random() > 0.04 else 0
        rows.append(
            {
                "g": g,
                "t": t,
                "d": d,
                "y": round(y, 10),
                "reg": reg,
                "x": round(x, 10),
                "w": round(w_g * (1 + 0.1 * t), 10),
                "cl": 1 + (g - 1) // 6,
                "keep": keep,
                "dc": round(dc, 10),
                "yc": round(yc, 10),
            }
        )
df = pd.DataFrame(rows)
df.to_csv(Path(__file__).with_name("data_97_dcdh_options.csv"), index=False)
print(df.shape, df.groupby("g")["d"].agg(["first", "max", "min"]).value_counts())
