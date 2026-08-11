"""StatsPAI LP-DiD parity (Python side) -- Module 83.

Runs ``sp.lp_did`` on a staggered fixture. The companion ``83_lpdid.R``
transcribes the same estimator directly rather than calling a package,
because no LP-DiD package is installed on the R side; the transcription
follows the definition ``sp.lp_did`` documents and implements:

  * at horizon h, regress  Δy = Y_{t+h} − Y_{t−1}  on  Δd = d_t − d_{t−1};
  * the treated arm is Δd == 1;
  * a clean control must have d == 0 across the whole window
    [t + min(−1, h−1), t + max(0, h)] and Δd == 0;
  * calendar-time fixed effects, cluster-robust SE by unit.

The fixture deliberately contains **only 0→1 transitions**. ``sp.lp_did``
documents that its handling of switch-off events is "not verified against
the published paper", so pinning a reference against that branch would
pin behaviour the maintainer has explicitly flagged as unconfirmed. That
branch stays an open coverage item rather than a silently blessed one.

Tolerance: rel < 1e-10. Both sides solve the same OLS on the same rows,
so only floating-point summation order should separate them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "83_lpdid"
HORIZONS = (-2, 3)
# h = -1 is the LP-DiD base period and is identically zero by construction,
# so it carries no information and is excluded from the comparison.
COMPARED_HORIZONS = (-2, 0, 1, 2, 3)


def make_data(n_units: int = 200, n_periods: int = 10, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cohorts = rng.choice([4, 6, 8, 0], size=n_units, p=[0.25, 0.25, 0.25, 0.25])
    rows = []
    for i in range(n_units):
        g = int(cohorts[i])
        unit_fe = rng.normal(0.0, 1.0)
        for t in range(1, n_periods + 1):
            treated = 1 if (g > 0 and t >= g) else 0
            effect = 0.8 * (t - g + 1) if treated else 0.0
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "d": treated,
                    "y": unit_fe + 0.3 * t + effect + rng.normal(0.0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    fit = sp.lp_did(df, y="y", unit="unit", time="time", treatment="d", horizons=HORIZONS)
    es = pd.DataFrame(fit.model_info["event_study"])

    rows: list[ParityRecord] = []
    for h in COMPARED_HORIZONS:
        row = es.loc[es["relative_time"] == h]
        if row.empty:
            raise SystemExit(f"lp_did returned no horizon {h}")
        row = row.iloc[0]
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"lpdid_h{h}_att",
                estimate=float(row["att"]),
                se=float(row["se"]),
                n=int(row["n_obs"]),
            )
        )

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "clean_controls": "not_yet_treated",
            "time_fe": True,
            "cluster": "unit",
            "horizons": list(HORIZONS),
            "reference_kind": "direct_transcription_no_r_package",
            "note": (
                "Fixture has only 0->1 transitions; sp.lp_did documents its "
                "switch-off handling as not verified against the published paper, "
                "so that branch is deliberately not pinned here."
            ),
        },
    )


if __name__ == "__main__":
    main()
