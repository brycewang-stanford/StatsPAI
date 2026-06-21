"""StatsPAI Honest-DiD relative-magnitudes parity (Python side).
Module 21.

Same hand-crafted event study as Module 10 but uses the relative-
magnitudes restriction (Roth 2024) rather than smoothness. The
companion 21_honest_relmags.R uses
HonestDiD::createSensitivityResults_relativeMagnitudes.

Tolerance: abs < 1e-6 on each CI bound when backend='honestdid' is
available. The fixture uses HonestDiD's Conditional method because R
HonestDiD and Stata honestdid share an exact finite-grid bridge for that
reference method. The default StatsPAI analytic path remains
dependency-light; this parity row explicitly exercises the optional
HonestDiD-compatible reference backend.
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
                          method="relative_magnitude",
                          backend="honestdid",
                          honestdid_method="Conditional")
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
            "honestdid_method": "Conditional",
            "backend": "HonestDiD",
            "Mbar_grid": MBAR_GRID, "alpha": 0.05,
            "reference_backend_note": (
                "sp.honest_did(..., backend='honestdid') delegates the "
                "relative-magnitudes CI to the R HonestDiD Conditional "
                "reference implementation, giving exact compatibility with "
                "HonestDiD::createSensitivityResults_relativeMagnitudes(..., "
                "method='Conditional'). "
                "The default backend='native' path is retained as a "
                "dependency-light analytic sensitivity bound and is not "
                "the claim exercised by this parity row."
            ),
        },
    )


if __name__ == "__main__":
    main()
