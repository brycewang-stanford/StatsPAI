"""StatsPAI balance_panel parity — Module 69 (Python side).

Constructs a deliberately unbalanced panel (one unit short by one period)
and runs ``sp.balance_panel`` to drop the short unit. The companion
``69_balance_panel.R`` rebuilds the same panel from the CSV, computes the
same ``counts == n_periods`` filter using base R ``aggregate()`` / ``%in%``,
and emits per-row ``y`` / ``id`` / ``time`` for three well-spread rows
spanning the kept units so compare.py joins on (module, statistic).

Tolerance: rel < 1e-6 (machine tier). ``balance_panel`` is a row-filter /
re-sort, so any disagreement would indicate a bug in the unit-count
heuristic or the sort order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "69_balance_panel"


def make_data(seed: int = PARITY_SEED + 2) -> pd.DataFrame:
    """5 entities × 4 periods, with one (id=2, t=1) deliberately missing."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(5):
        for t in range(4):
            if uid == 2 and t == 1:  # unbalanced: unit 2 has 3 obs
                continue
            rows.append({"id": uid, "year": t, "y": float(rng.normal())})
    return pd.DataFrame(rows)


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    balanced = sp.balance_panel(df, entity="id", time="year")
    balanced = balanced.reset_index(drop=True)

    n = len(balanced)
    rows: list[ParityRecord] = []
    rows.append(
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="n_obs_balanced",
            estimate=float(n),
            n=len(df),
        )
    )
    rows.append(
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="n_units_kept",
            estimate=float(balanced["id"].nunique()),
            n=len(df),
        )
    )
    # Spot-check three rows: top, middle, bottom of the balanced output.
    for k in (0, n // 2, n - 1):
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"row{k}_id",
                estimate=float(balanced["id"].iloc[k]),
                n=len(df),
            )
        )
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"row{k}_year",
                estimate=float(balanced["year"].iloc[k]),
                n=len(df),
            )
        )

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "n_obs_in": int(len(df)),
            "n_obs_out": int(n),
            "n_units_kept": int(balanced["id"].nunique()),
            "note": (
                "sp.balance_panel keeps only entities observed in every "
                "period (id=2 dropped here). The R side applies the same "
                "filter using base R counts."
            ),
        },
    )


if __name__ == "__main__":
    main()
