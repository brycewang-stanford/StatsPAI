"""StatsPAI PPML+HDFE parity with three-way FE (gravity DGP) -- 47.

Companion to module 37 (single-FE PPML) but with the canonical gravity
specification: origin + destination + year fixed effects. Before the
2026-05-28 IRLS fix (Task 2.2/3.3), sp.ppmlhdfe could not converge in
1000 iterations on this DGP and required falling back to a single FE
to compare against fixest/Stata.

Tolerance: rel < 1e-3 on coefficients; rel < 5e-2 on HC1 SEs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "47_ppmlhdfe_3fe"


def make_data(seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    N, T = 50, 10
    rows = []
    for i in range(N):
        origin = i % 5
        dest = (i // 5) % 5
        for t in range(T):
            dist = rng.uniform(1.0, 5.0)
            contig = rng.binomial(1, 0.3)
            log_mu = (
                -0.8 * np.log(dist) + 0.4 * contig
                + 0.2 * origin + 0.3 * dest + 0.05 * t - 1.0
            )
            rows.append({
                "trade": float(rng.poisson(np.exp(log_mu))),
                "log_dist": float(np.log(dist)),
                "contig": float(contig),
                "origin": int(origin),
                "dest": int(dest),
                "year": int(t),
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    res = sp.ppmlhdfe(
        formula="trade ~ log_dist + contig | origin + dest + year",
        data=df,
        robust="hc1",
    )

    rows: list[ParityRecord] = []
    for name in ["log_dist", "contig"]:
        rows.append(ParityRecord(
            module=MODULE, side="py",
            statistic=f"beta_{name}",
            estimate=float(res.params[name]),
            se=float(res.std_errors[name]),
            n=int(len(df))))

    write_results(
        MODULE, "py", rows,
        extra={"fe": "origin + dest + year", "vcov": "HC1",
               "n_obs": int(len(df))},
    )


if __name__ == "__main__":
    main()
