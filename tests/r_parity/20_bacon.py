"""StatsPAI Goodman-Bacon decomposition parity (Python side) -- Module 20.

Runs sp.bacon_decomposition on the mpdta replica. The companion
20_bacon.R uses bacondecomp::bacon. Tolerance: rel < 1e-3 on the
overall TWFE coefficient.

The decomposition returns per-cohort-pair comparisons; we compare
the overall TWFE coefficient (a scalar that both sides should agree
on) plus the per-cohort-pair point estimates indexed by
(treated_cohort, comparison_cohort).
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "20_bacon"


def main() -> None:
    df = sp.datasets.mpdta()
    dump_csv(df, MODULE)

    res = sp.bacon_decomposition(
        df, y="lemp", treat="treat", time="year", id="countyreal"
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="beta_twfe",
            estimate=float(res["beta_twfe"]),
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="weighted_sum",
            estimate=float(res["weighted_sum"]),
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="negative_weight_share",
            estimate=float(res["negative_weight_share"]),
            n=int(len(df)),
        ),
    ]

    # Per-cohort-pair point estimates.
    decomp = res["decomposition"]
    for _, row in decomp.iterrows():
        treated = int(row["treated"])
        control = row["control"]
        if isinstance(control, str):
            control_str = "never"
        else:
            control_str = str(int(control))
        rows.append(
            ParityRecord(
                module=MODULE, side="py",
                statistic=f"pair_{treated}_vs_{control_str}_est",
                estimate=float(row["estimate"]),
                n=int(len(df)),
            )
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "n_comparisons": int(res["n_comparisons"]),
            "already_treated_control_weight_share": (
                float(res["already_treated_control_weight_share"])
            ),
            "decomposition_note": (
                "StatsPAI follows R/Stata bacondecomp dyad windows "
                "and weighting for the uncontrolled Goodman-Bacon "
                "decomposition. The overall TWFE coefficient, "
                "weighted sum, negative-weight mass, and pairwise "
                "2x2 estimates match the R reference to numerical "
                "precision; Stata's ado exports only the aggregate "
                "rows in the committed golden artifact."
            ),
        },
    )


if __name__ == "__main__":
    main()
