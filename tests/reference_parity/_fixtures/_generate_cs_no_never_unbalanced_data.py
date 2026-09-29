"""Panels for the CS no-never-treated / unbalanced not-yet-treated fixture.

``cs_no_never_data.csv``: four cohorts (5, 7, 9, 11), no never-treated units,
twelve periods, true ATT 1 -- the case where the late cells have no
comparison units (R ``did`` drops the periods from the last cohort's
treatment date on).

``cs_unbalanced_notyet_data.csv``: cohorts 4 and 6 plus never-treated units
over eight periods, dynamic effects and a trend in the never-treated only
(so never-treated and not-yet-treated comparisons differ), 10% of the rows
removed.

    python tests/reference_parity/_fixtures/_generate_cs_no_never_unbalanced_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(20260929)
    rows = []
    for u in range(200):
        g = (5, 7, 9, 11)[u % 4]
        a = rng.normal()
        for t in range(1, 13):
            rows.append(dict(id=u, t=t, g=g, y=a + 0.1 * t + (t >= g) + rng.normal()))
    pd.DataFrame(rows).to_csv(
        HERE / "cs_no_never_data.csv", index=False, float_format="%.17g"
    )

    rng = np.random.default_rng(20260930)
    rows = []
    for u in range(300):
        g = (4, 6, 0)[u % 3]
        a = rng.normal()
        for t in range(1, 9):
            te = 1 + (t - g) if (g > 0 and t >= g) else 0
            trend = 0.3 * t if g == 0 else 0.0
            rows.append(dict(id=u, t=t, g=g, y=a + trend + te + rng.normal(scale=0.5)))
    df = pd.DataFrame(rows)
    df = df.drop(df.sample(frac=0.1, random_state=1).index)
    df.to_csv(HERE / "cs_unbalanced_notyet_data.csv", index=False, float_format="%.17g")
    print("wrote cs_no_never_data.csv, cs_unbalanced_notyet_data.csv")


if __name__ == "__main__":
    main()
