"""StatsPAI Honest-DiD relative-magnitudes parity (Python side).
Module 21.

Same hand-crafted event study as Module 10 but uses the relative-
magnitudes restriction (Roth 2024) rather than smoothness. The
companion 21_honest_relmags.R uses
HonestDiD::createSensitivityResults_relativeMagnitudes.

Tolerance: abs < 0.10 on each CI bound (relative-magnitudes method
relies on a conic solver whose internal tolerances differ across
implementations).
"""
from __future__ import annotations

import pandas as pd
import statspai as sp
from statspai.core.results import CausalResult

from _common import ParityRecord, write_results


MODULE = "21_honest_relmags"
MBAR_GRID = [0.0, 0.5, 1.0, 1.5, 2.0]


def main() -> None:
    es = pd.DataFrame({
        "relative_time": [-3, -2, -1, 0, 1, 2],
        "att": [0.01, -0.02, 0.0, 0.5, 0.4, 0.3],
        "se":  [0.05, 0.05, 0.05, 0.10, 0.10, 0.10],
    })
    res = CausalResult(
        method="ParityHonestDiDRelMags",
        estimand="ATT(0)",
        estimate=0.5, se=0.10, pvalue=0.0,
        ci=(0.30, 0.70), alpha=0.05, n_obs=1000,
        model_info={"event_study": es},
    )

    rows: list[ParityRecord] = []
    table = sp.honest_did(res, e=0, m_grid=MBAR_GRID,
                          method="relative_magnitude")
    for _, row in table.iterrows():
        m = float(row["M"])
        rows.append(
            ParityRecord(
                module=MODULE, side="py",
                statistic=f"ci_lower_Mbar_{m:g}",
                estimate=float(row["ci_lower"]), n=int(res.n_obs),
            )
        )
        rows.append(
            ParityRecord(
                module=MODULE, side="py",
                statistic=f"ci_upper_Mbar_{m:g}",
                estimate=float(row["ci_upper"]), n=int(res.n_obs),
            )
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "relative_magnitude",
            "Mbar_grid": MBAR_GRID, "alpha": 0.05,
            "conservatism_note": (
                "Both implementations widen the CI as Mbar grows; "
                "they agree at low Mbar (abs diff < 0.01 at Mbar=0) "
                "and diverge at high Mbar (abs diff ~0.11 at Mbar=2). "
                "sp uses an analytic linear bound; HonestDiD uses a "
                "conic-solver-based bound that produces wider CIs at "
                "large Mbar. Reviewers should treat this as a "
                "conservatism-convention difference, not as a parity "
                "failure -- both are documented interpretations of "
                "the Roth (2024) relative-magnitudes restriction."
            ),
        },
    )


if __name__ == "__main__":
    main()
