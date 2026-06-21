"""StatsPAI local projections parity (Python side) -- Module 34.

Generates a deterministic AR(1) + shock series and runs the
lpirfs-compatible Cholesky/unit-shock path in sp.local_projections at
horizons 0..5. The companion 34_lp.R uses lpirfs::lp_lin.

Tolerance: rel < 1e-6 on the impulse-response coefficients and
Newey-West standard errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "34_lp"
H_MAX = 5


def make_data(T: int = 200, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    x = rng.normal(0, 1, T)
    for t in range(1, T):
        y[t] = 0.6 * y[t - 1] + 0.5 * x[t - 1] + rng.normal(0, 0.5)
    return pd.DataFrame({"y": y, "x": x})


def main() -> None:
    df = make_data()
    df["y_lag"] = df["y"].shift(1)
    df = df.iloc[1:].reset_index(drop=True)
    dump_csv(df, MODULE)

    fit = sp.local_projections(
        data=df, outcome="y", shock="x",
        horizons=H_MAX,
        identification="lpirfs_cholesky",
        endog_order=["y", "x"],
    )
    direct_fit = sp.local_projections(
        data=df, outcome="y", shock="x",
        controls=["y_lag"],
        horizons=H_MAX,
        auto_lag=False,
        identification="direct",
    )

    rows: list[ParityRecord] = []
    for h, irf in enumerate(fit.irf):
        rows.append(ParityRecord(
            module=MODULE, side="py",
            statistic=f"irf_h{h}",
            estimate=float(irf),
            se=float(fit.se[h]),
            n=int(fit.n_obs_per_horizon[h])))
    for h, irf in enumerate(direct_fit.irf):
        rows.append(ParityRecord(
            module=MODULE, side="py",
            statistic=f"irf_direct_ols_h{h}",
            estimate=float(irf),
            n=int(direct_fit.n_obs_per_horizon[h])))

    write_results(
        MODULE, "py", rows,
        extra={
            "horizons": H_MAX,
            "identification": "lpirfs_cholesky",
            "endog_order": ["y", "x"],
            "r_identification_reference": (
                "sp.local_projections(..., identification='lpirfs_cholesky') "
                "follows lpirfs::lp_lin lags_endog_lin=1, shock_type=1: "
                "a reduced VAR(1), a unit Cholesky shock using the supplied "
                "endogenous-variable order, and horizon-by-horizon "
                "Newey-West OLS."
            ),
            "stata_direct_ols_reference": (
                "Python also records irf_direct_ols_h* point diagnostics "
                "for the Stata regress y_{t+h} x y_lag fixture."
            ),
        },
    )


if __name__ == "__main__":
    main()
