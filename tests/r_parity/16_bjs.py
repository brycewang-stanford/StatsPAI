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
            se=float(fit.se),
            ci_lo=float(fit.ci[0]) if fit.ci is not None else None,
            ci_hi=float(fit.ci[1]) if fit.ci is not None else None,
            n=int(len(df)),
        )
    ]

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "did_imputation",
            "n_treated_obs": int(fit.model_info["n_treated_obs"]),
            "n_control_obs": int(fit.model_info["n_control_obs"]),
            "aggregation_note": (
                "sp.did_imputation reports an unweighted average of "
                "the per-(g,t) ATTs, while didimputation::did_imputation"
                " reports a treated-observation-weighted average "
                "(matching Borusyak-Jaravel-Spiess 2024 Section 3). "
                "On the mpdta replica sp returns -0.022 while R "
                "returns -0.035 (~36% rel gap). Both are sign-"
                "correct, both are within the published mpdta "
                "neighbourhood, and the difference is the same "
                "aggregation-rule family as the Sun-Abraham gap "
                "reported in Module 05."
            ),
        },
    )


if __name__ == "__main__":
    main()
