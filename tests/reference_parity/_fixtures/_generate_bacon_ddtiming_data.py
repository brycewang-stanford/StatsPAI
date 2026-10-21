"""Unbalanced staggered panels for ``bacon_decomposition(balance='observed_rows')``.

Run this, then ``_generate_bacon_ddtiming_Stata.do`` (Stata 18, ``ddtiming``
0.2), which writes ``bacon_ddtiming_Stata.json``. Three panels of 40 units
over 2001-2010 (column ``panel``), timing groups 2001 (treated throughout),
2004, 2006, 2008 and never treated, with about 12% of the rows dropped at
random:

* panel 0 also drops the 2004 row of one 2004 unit, and one 2001 unit loses
  its 2001 row, so it is first observed treated (an already-treated control
  whatever its first period);
* panel 1 has no already-treated group;
* panel 2 has one but no deliberately missing first treated period.

The random drops leave some 2x2 comparisons without treatment variation
after their fixed effects, and some weights negative.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def _panel(seed: int, always: bool, missing_first: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cohorts = ([2001] * 5 if always else []) + [2004] * 8 + [2006] * 9
    cohorts += [2008] * 8 + [0] * 10
    rows = []
    for i, g in enumerate(cohorts, start=1):
        a = rng.normal()
        for t in range(2001, 2011):
            d = int(g > 0 and t >= g)
            effect = d * (0.5 + 0.1 * (t - g if g else 0))
            y = a + 0.1 * (t - 2000) + effect + rng.normal(0, 0.3)
            rows.append((i, t, g, d, round(y, 6)))
    df = pd.DataFrame(rows, columns=["unit", "year", "g", "d", "y"])
    df = df[rng.random(len(df)) > 0.12]
    if missing_first:
        u = df.loc[df["g"] == 2004, "unit"].iloc[0]
        df = df[~((df["unit"] == u) & (df["year"] == 2004))]
    df = df[df.groupby("unit")["year"].transform("size") >= 2]
    return df.reset_index(drop=True)


def main() -> None:
    specs = [(True, True), (False, False), (True, False)]
    frames = [
        _panel(100 + k, always, missing).assign(panel=k)
        for k, (always, missing) in enumerate(specs)
    ]
    out = pd.concat(frames, ignore_index=True)
    out = out[["panel", "unit", "year", "g", "d", "y"]]
    out.to_csv(HERE / "bacon_ddtiming_data.csv", index=False)


if __name__ == "__main__":
    main()
