"""StatsPAI BJS imputation parity (Python side) -- Module 16.

Runs sp.did_imputation on the mpdta replica (the Borusyak-Jaravel-
Spiess imputation estimator). The companion 16_bjs.R uses
didimputation::did_imputation.

Tolerance: rel < 1e-3 on the simple ATT.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "16_bjs"


def main() -> None:
    df = sp.datasets.mpdta()
    dump_csv(df, MODULE)

    fit = sp.did_imputation(
        df, y="lemp", group="countyreal", time="year",
        first_treat="first_treat",
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_bjs",
            estimate=float(fit.estimate),
            se=None,
            ci_lo=None,
            ci_hi=None,
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="se_cluster_if",
            estimate=float(fit.se),
            n=int(len(df)),
        ),
    ]

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "did_imputation",
            "n_treated_obs": int(fit.model_info["n_treated_obs"]),
            "n_control_obs": int(fit.model_info["n_control_obs"]),
            "parity_note": (
                "sp.did_imputation fits the untreated-only TWFE "
                "first stage on the unbalanced untreated panel and "
                "matches didimputation::did_imputation's simple ATT "
                "on the mpdta replica."
            ),
            "stata_never_treated_coding": (
                "Stata did_imputation expects missing Ei for never-treated "
                "units; the Stata parity script recodes first_treat==0 "
                "before estimation."
            ),
            "se_reference": (
                "SE rows are side-specific: StatsPAI reports "
                "se_cluster_if, R reports se_didimputation, and Stata "
                "reports se_stata_did_imputation."
            ),
        },
    )


if __name__ == "__main__":
    main()
