"""StatsPAI doubly-robust DiD parity (Python side) -- Module 38.

Generates a 2x2 DR-DID panel DGP with one covariate and runs sp.drdid
(improved panel DR estimator, Sant'Anna & Zhao 2020). The companion R
script uses DRDID::drdid_imp_panel and Stata uses drdid, drimp.

Tolerance: rel < 1e-6 on ATT/CI and SE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "38_drdid"


def make_data(n: int = 800, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    # Logistic propensity in x
    ps = 1 / (1 + np.exp(-(0.3 + 0.5 * x)))
    G = rng.binomial(1, ps).astype(int)

    # 2x2 panel: one row per unit-period
    rows = []
    for i in range(n):
        Gi = int(G[i])
        xi = float(x[i])
        # Pre-period
        y_pre = 1.0 + 0.5 * xi + 0.2 * Gi + rng.normal(0, 1)
        rows.append({"id": i, "post": 0, "treated": Gi, "x": xi, "y": y_pre})
        # Post-period: true ATT = 2.0
        y_post = 1.0 + 0.5 * xi + 0.2 * Gi + 1.5 + 2.0 * Gi + rng.normal(0, 1)
        rows.append({"id": i, "post": 1, "treated": Gi, "x": xi, "y": y_post})
    return pd.DataFrame(rows)


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    res = sp.drdid(
        data=df, y="y", group="treated", time="post",
        covariates=["x"],
        method="imp",
        id="id",
        random_state=PARITY_SEED,
    )

    rows: list[ParityRecord] = [
        ParityRecord(MODULE, "py", "att",
                     estimate=float(res.estimate),
                     se=float(res.se),
                     n=int(len(df))),
    ]
    # CI bounds
    rows.append(ParityRecord(
        MODULE, "py", "ci_lower",
        estimate=float(res.ci[0]), n=int(len(df))))
    rows.append(ParityRecord(
        MODULE, "py", "ci_upper",
        estimate=float(res.ci[1]), n=int(len(df))))

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "DR-DID (improved, Sant'Anna-Zhao 2020)",
            "covariates": ["x"],
            "id": "id",
            "se_method": "influence_function",
            "reference_method_note": (
                "sp.drdid(..., id='id', method='imp') uses the panel "
                "Sant'Anna-Zhao improved estimator with calibrated propensity "
                "scores, matching DRDID::drdid_imp_panel and Stata drdid, drimp."
            ),
        },
    )


if __name__ == "__main__":
    main()
