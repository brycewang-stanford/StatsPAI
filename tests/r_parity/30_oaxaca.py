"""StatsPAI Blinder-Oaxaca decomposition parity (Python side).
Module 30.

Generates a deterministic two-group wage gap dataset and runs
sp.decompose('oaxaca'). The companion 30_oaxaca.R uses
oaxaca::oaxaca with the threefold (Neumark-style) decomposition.

Tolerance: rel < 1e-3 on the gap, explained, and unexplained
components.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "30_oaxaca"


def make_data(n_per_group: int = 500, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    female = np.repeat([0, 1], n_per_group)
    educ = rng.normal(15, 3, 2 * n_per_group)
    exper = rng.normal(10, 5, 2 * n_per_group) - 1.0 * (female == 1)
    log_wage = (
        2.0 + 0.07 * educ + 0.02 * exper - 0.20 * female
        + rng.normal(0, 0.3, 2 * n_per_group)
    )
    return pd.DataFrame({
        "log_wage": log_wage,
        "educ": educ,
        "exper": exper,
        "female": female,
    })


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    res = sp.decompose(
        "oaxaca", data=df, y="log_wage", group="female",
        x=["educ", "exper"],
    )

    overall = res.overall
    gs = res.group_stats

    rows: list[ParityRecord] = [
        ParityRecord(MODULE, "py", "gap",
                     estimate=float(overall["gap"]), n=int(len(df))),
        ParityRecord(MODULE, "py", "explained",
                     estimate=float(overall["explained"]),
                     se=float(overall["explained_se"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "unexplained",
                     estimate=float(overall["unexplained"]),
                     se=float(overall["unexplained_se"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "mean_y_a",
                     estimate=float(gs["mean_a"]), n=int(len(df))),
        ParityRecord(MODULE, "py", "mean_y_b",
                     estimate=float(gs["mean_b"]), n=int(len(df))),
        ParityRecord(MODULE, "py", "beta_a_educ",
                     estimate=float(gs["beta_a"]["educ"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "beta_b_educ",
                     estimate=float(gs["beta_b"]["educ"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "beta_a_exper",
                     estimate=float(gs["beta_a"]["exper"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "beta_b_exper",
                     estimate=float(gs["beta_b"]["exper"]),
                     n=int(len(df))),
    ]

    # Detailed contributions.
    for _, row in res.detailed.iterrows():
        v = row["variable"]
        rows.append(ParityRecord(
            MODULE, "py", f"explained_{v}",
            estimate=float(row["contribution"]),
            se=float(row["se"]), n=int(len(df))))

    write_results(
        MODULE, "py", rows,
        extra={
            "reference": "group_a (male)",
            "decomposition_note": (
                "The total wage gap, the group-conditional means, "
                "all four group-specific coefficients (beta_a_educ, "
                "beta_b_educ, beta_a_exper, beta_b_exper), and the "
                "unexplained component all match oaxaca::oaxaca at "
                "rel < 1e-14. The 'explained' overall component and "
                "the per-variable contributions differ by ~6-9% "
                "because sp.decompose('oaxaca') uses the TWOFOLD "
                "Blinder-Oaxaca decomposition (gap = explained + "
                "unexplained), while oaxaca::oaxaca's threefold "
                "decomposition splits explained = endowments + "
                "interaction. Adding R's coef(endowments) and "
                "coef(interaction) recovers sp's explained to "
                "machine precision: 0.04107 + (-0.00249) = 0.03858."
            ),
        },
    )


if __name__ == "__main__":
    main()
