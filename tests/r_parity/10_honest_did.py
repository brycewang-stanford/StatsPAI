"""StatsPAI Honest-DiD parity (Python side) -- Module 10.

Compares sp.honest_did against HonestDiD::createSensitivityResults
on a hand-crafted event study, evaluating the robust CI bounds
under the smoothness restriction at five values of M. Tolerance:
abs < 0.05 on each CI bound (the difference is the analytic-vs-
conic-solver gap; bit-equal parity is not expected).

The closed-form breakdown_m primitive is independently pinned at
relative tolerance 1e-10 against the Rambachan-Roth Definition 2
formula in tests/external_parity/test_honest_did_paper_parity.py.
"""
from __future__ import annotations

import pandas as pd
import statspai as sp
from statspai.core.results import CausalResult

from _common import ParityRecord, write_results


MODULE = "10_honest_did"
M_GRID = [0.0, 0.05, 0.1, 0.2, 0.5]


def main() -> None:
    # Hand-crafted event study so the test is deterministic and the
    # R side can mirror the inputs without an intermediate CSV. The
    # numbers are similar to the test_honest_did_paper_parity.py
    # fixtures and exercise both the pre-period (relative_time -3,
    # -2, -1) and the post-period (relative_time 0, 1, 2).
    es = pd.DataFrame({
        "relative_time": [-3, -2, -1, 0, 1, 2],
        "att": [0.01, -0.02, 0.0, 0.5, 0.4, 0.3],
        "se":  [0.05, 0.05, 0.05, 0.10, 0.10, 0.10],
    })
    res = CausalResult(
        method="ParityHonestDiDInput",
        estimand="ATT(0)",
        estimate=0.5, se=0.10, pvalue=0.0,
        ci=(0.30, 0.70), alpha=0.05, n_obs=1000,
        model_info={"event_study": es},
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="breakdown_m_e0",
            estimate=float(sp.breakdown_m(res, e=0, method="smoothness", alpha=0.05)),
            n=int(res.n_obs),
        )
    ]

    table = sp.honest_did(res, e=0, m_grid=M_GRID, method="smoothness")
    for _, row in table.iterrows():
        m = float(row["M"])
        rows.append(
            ParityRecord(
                module=MODULE, side="py", statistic=f"ci_lower_M_{m:g}",
                estimate=float(row["ci_lower"]), n=int(res.n_obs),
            )
        )
        rows.append(
            ParityRecord(
                module=MODULE, side="py", statistic=f"ci_upper_M_{m:g}",
                estimate=float(row["ci_upper"]), n=int(res.n_obs),
            )
        )

    write_results(MODULE, "py", rows,
                  extra={"method": "smoothness", "alpha": 0.05,
                         "M_grid": M_GRID})


if __name__ == "__main__":
    main()
