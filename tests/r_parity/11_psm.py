"""StatsPAI PSM 1:1 NN matching parity (Python side) -- Module 11.

Runs sp.psm on the NSW-DW replica with logistic propensity score and
1:1 nearest-neighbour matching with replacement (the MatchIt /
psmatch2 default convention). Tolerance: rel < 1e-6 on the ATT; SE
rows are retained as side-specific diagnostics because matching
packages use different variance conventions -- see the ``se_reference``
note in ``extra`` for what each of the three SE rows estimates.

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
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="se_abadie_imbens",
            estimate=float(fit.se),
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="n_treated",
            estimate=float(fit.model_info["n_treated"]), n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="n_control_full",
            estimate=float(fit.model_info["n_control"]), n=int(len(df)),
        ),
    ]

    write_results(MODULE, "py", rows,
                  extra={"distance": fit.model_info["distance"],
                         "method": fit.model_info["method"],
                         "replace": fit.model_info["replace"],
                         "bias_correction": fit.model_info["bias_correction"],
                         "count_note": (
                             "n_control_full is the full untreated sample; "
                             "MatchIt's matched-data control count is a "
                             "post-matching support diagnostic and is not "
                             "the same estimand when matching with "
                             "replacement."
                         ),
                         "se_reference": (
                             "The att_psm row compares point estimates only; "
                             "the three SE rows are side-specific diagnostics "
                             "that never join. StatsPAI's default for "
                             "nearest-neighbour matching is the "
                             "Abadie-Imbens (2006, eq. 14) sample-ATT SE, "
                             "the estimator Stata psmatch2 reports under "
                             "ai(J) (se_abadie_imbens = 643.35 here). "
                             "MatchIt documents no canonical analytic SE for "
                             "matching with replacement, so the R fixture "
                             "records a weighted-lm-on-matched-data "
                             "diagnostic (se_matchit_lm). Stata teffects "
                             "psmatch reports its own Abadie-Imbens variant "
                             "(se_teffects_ai = 621.79), 3.4% below ours; "
                             "the two Stata commands implement different AI "
                             "variances and the gap is NOT yet localised to "
                             "a specific term, so it is recorded as an open "
                             "item rather than claimed as a convention."
                         )})


if __name__ == "__main__":
    main()
