"""A small staggered panel for ``sp.stacked_did`` vs Stata ``stackedev``.

60 units observed 2011-2018, cohorts 2014 / 2015 / 2016 plus never-treated
units (``first_treat`` missing), a time-varying control ``x`` that differs
between treated and control units, cohort x event-time effects that grow with
exposure, and three missing unit-periods so the number of treated rows varies
across event times (the treated-observation weights then differ from equal
weights).

Calendar-year cohort codes keep Stata ``stackedev``'s cluster variable
``stack * clust_unit`` one-to-one with (unit, stack) for unit ids below 1008.

    python tests/reference_parity/_fixtures/_generate_stacked_did_stackedev_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20261022)
    ids = np.arange(1, 61)
    years = np.arange(2011, 2019)
    cohort = np.full(ids.size, np.nan)
    cohort[:8] = 2014
    cohort[8:18] = 2015
    cohort[18:24] = 2016
    alpha = rng.normal(0.0, 1.0, ids.size)
    lam = rng.normal(0.0, 0.5, years.size)
    rows = []
    for i, uid in enumerate(ids):
        for j, yr in enumerate(years):
            g = cohort[i]
            treated = np.isfinite(g)
            x = rng.normal(0.4 * treated + 0.05 * j, 1.0)
            k = yr - g if treated else np.nan
            effect = 0.0
            if treated and k >= 0:
                effect = (0.3 + 0.15 * k) * (1.0 + 0.2 * (g - 2014))
            y = alpha[i] + lam[j] + 0.5 * x + effect + rng.normal(0.0, 0.7)
            rows.append((uid, yr, g, x, y))
    df = pd.DataFrame(rows, columns=["id", "year", "first_treat", "x", "y"])
    drop = (
        ((df["id"] == 3) & (df["year"] == 2017))
        | ((df["id"] == 12) & (df["year"] == 2015))
        | ((df["id"] == 40) & (df["year"] == 2012))
    )
    df = df.loc[~drop].reset_index(drop=True)
    df.to_csv(HERE / "stacked_did_stackedev.csv", index=False, float_format="%.17g")
    print(f"wrote stacked_did_stackedev.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
