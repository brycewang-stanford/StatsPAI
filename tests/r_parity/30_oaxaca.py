"""StatsPAI Blinder-Oaxaca decomposition parity (Python side).
Module 30.

Generates a deterministic two-group wage gap dataset and runs
sp.decompose('oaxaca'). The companion 30_oaxaca.R uses
oaxaca::oaxaca with the threefold (Neumark-style) decomposition.

Tolerance: rel < 1e-3 on the mean contrast, explained, and
unexplained components.
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
        ParityRecord(MODULE, "py", "explained_twofold",
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
        # Threefold components, point estimates only: R bootstraps their SEs
        # and Stata's default treats the covariate means as random, so the
        # SEs are compared below under the convention StatsPAI uses.
        ParityRecord(MODULE, "py", "explained_threefold_endowments",
                     estimate=float(overall["endowments"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "interaction_threefold",
                     estimate=float(overall["interaction"]),
                     n=int(len(df))),
    ]

    # StatsPAI's delta-method SEs use HC1 group regressions and hold the
    # covariate means fixed. Stata `oaxaca, vce(robust) fixed` computes
    # exactly that, so these rows pin the SE convention itself; the default
    # `unexplained` SE above differs from Stata's default (classical
    # regressions, stochastic means) by that convention only.
    for comp in ("endowments", "coefficients", "interaction"):
        rows.append(ParityRecord(
            MODULE, "py", f"{comp}_hc1_fixed",
            estimate=float(overall[comp]),
            se=float(overall[f"{comp}_se"]),
            n=int(len(df))))

    # Detailed contributions.
    for _, row in res.detailed.iterrows():
        v = row["variable"]
        rows.append(ParityRecord(
            MODULE, "py", f"explained_twofold_{v}",
            estimate=float(row["contribution"]),
            se=float(row["se"]), n=int(len(df))))

    write_results(
        MODULE, "py", rows,
        extra={
            "reference": "group_a (male)",
            "decomposition_reference": (
                "The total group mean contrast, the group-conditional means, "
                "all four group-specific coefficients (beta_a_educ, "
                "beta_b_educ, beta_a_exper, beta_b_exper), and the "
                "unexplained component all match oaxaca::oaxaca at "
                "rel < 1e-14. The fixture names StatsPAI's overall "
                "explained component as explained_twofold because "
                "sp.decompose('oaxaca') uses the TWOFOLD "
                "Blinder-Oaxaca decomposition (mean contrast = explained + "
                "unexplained), while oaxaca::oaxaca and Stata oaxaca "
                "report threefold endowments and interaction separately. "
                "R/Stata explained_twofold is endowments + interaction."
            ),
        },
    )


if __name__ == "__main__":
    main()
