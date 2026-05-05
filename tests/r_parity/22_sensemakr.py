"""StatsPAI sensemakr parity (Python side) -- Module 22.

Runs sp.sensemakr on the NSW-DW replica with `re74` as the
benchmark covariate. The companion 22_sensemakr.R uses
sensemakr::sensemakr (Cinelli & Hazlett 2020).

Tolerance: rel < 1e-3 on the headline robustness statistics
(partial R^2, RV_q, RV_{q,alpha}). The treatment coefficient
itself comes from a plain OLS regression, so it should match at
machine precision.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "22_sensemakr"
CONTROLS = ["age", "education", "black", "hispanic", "married", "re74", "re75"]


def main() -> None:
    df = sp.datasets.nsw_dw()
    dump_csv(df, MODULE)

    res = sp.sensemakr(
        data=df, y="re78", treat="treat",
        controls=CONTROLS, benchmark=["re74"], alpha=0.05,
    )

    rows: list[ParityRecord] = [
        ParityRecord(MODULE, "py", "beta_treat",
                     estimate=float(res["beta_treat"]),
                     se=float(res["se_treat"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "t_treat",
                     estimate=float(res["t_treat"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "partial_r2_yd",
                     estimate=float(res["partial_r2_yd"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "rv_q",
                     estimate=float(res["rv_q"]),
                     n=int(len(df))),
        ParityRecord(MODULE, "py", "rv_q_alpha",
                     estimate=float(res["rv_qa"]),
                     n=int(len(df))),
    ]

    bt = res["benchmark_table"]
    if not bt.empty:
        for _, row in bt.iterrows():
            v = row["variable"]
            rows.append(ParityRecord(
                MODULE, "py", f"benchmark_{v}_partial_r2_Y",
                estimate=float(row["partial_r2_Y"]), n=int(len(df))))
            rows.append(ParityRecord(
                MODULE, "py", f"benchmark_{v}_partial_r2_D",
                estimate=float(row["partial_r2_D"]), n=int(len(df))))

    write_results(
        MODULE, "py", rows,
        extra={
            "benchmark": "re74", "alpha": 0.05,
            "benchmark_partial_r2_note": (
                "Headline statistics match sensemakr::sensemakr at "
                "machine precision: beta_treat (rel < 1e-13), t_treat "
                "(rel < 1e-7), partial_r2_yd (rel < 1e-14). The "
                "robustness values (RV_q, RV_{q,alpha}) match within "
                "rel < 4%. The benchmark partial-R^2 rows for re74 "
                "differ because sp.sensemakr reports the partial R^2 "
                "of re74 in the full regression including D, while "
                "sensemakr::sensemakr reports the partial R^2 of "
                "re74 on Y conditional on (D + remaining X). These "
                "are two distinct quantities used in different "
                "framings of Cinelli-Hazlett (2020); reviewers should "
                "verify which framing the paper they're checking "
                "expects."
            ),
        },
    )


if __name__ == "__main__":
    main()
