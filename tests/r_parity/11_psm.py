"""StatsPAI PSM 1:1 NN matching parity (Python side) -- Module 11.

Runs sp.psm on the NSW-DW replica with logistic propensity score and
1:1 nearest-neighbour matching with replacement (the MatchIt /
psmatch2 default convention). Tolerance: rel < 1e-2 on the ATT.

PSM has many small implementation choices (caliper, ties, replacement,
bias correction); the goal is to verify that the StatsPAI ATT is
within the same neighbourhood as MatchIt::matchit on identical data
with matched options, not bit-equal recovery.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "11_psm"


def main() -> None:
    df = sp.datasets.nsw_dw()
    dump_csv(df, MODULE)

    fit = sp.psm(
        df,
        y="re78",
        d="treat",
        X=["age", "education", "black", "hispanic", "married", "re74", "re75"],
        method="nn",
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_psm",
            estimate=float(fit.estimate),
            se=float(fit.se),
            ci_lo=float(fit.ci[0]) if fit.ci is not None else None,
            ci_hi=float(fit.ci[1]) if fit.ci is not None else None,
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="n_treated",
            estimate=float(fit.model_info["n_treated"]), n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="n_control",
            estimate=float(fit.model_info["n_control"]), n=int(len(df)),
        ),
    ]

    write_results(MODULE, "py", rows,
                  extra={"distance": fit.model_info["distance"],
                         "method": fit.model_info["method"],
                         "replace": fit.model_info["replace"],
                         "bias_correction": fit.model_info["bias_correction"]})


if __name__ == "__main__":
    main()
