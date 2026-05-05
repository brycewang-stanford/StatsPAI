"""StatsPAI LMM parity (Python side) -- Module 25.

Random-intercept LMM on a deterministic n=1000, 50-group panel.
The companion 25_lmm.R uses lme4::lmer with REML. Tolerance:
rel < 1e-3 on the fixed effects and variance components (REML
optimisers can land on slightly different local optima but on a
balanced panel like this they should agree to sub-percent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "25_lmm"


def make_panel(n_groups: int = 50, n_per: int = 20, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gid = np.repeat(np.arange(n_groups), n_per)
    x1 = rng.normal(size=n_groups * n_per)
    group_re = rng.normal(scale=1.0, size=n_groups)
    y = (
        2.0 + 1.5 * x1 + group_re[gid]
        + rng.normal(scale=0.5, size=n_groups * n_per)
    )
    return pd.DataFrame({"y": y, "x1": x1, "gid": gid})


def main() -> None:
    df = make_panel()
    dump_csv(df, MODULE)

    fit = sp.mixed(data=df, y="y", x_fixed=["x1"], group="gid",
                   method="reml")

    rows: list[ParityRecord] = [
        ParityRecord(MODULE, "py", "beta_intercept",
                     estimate=float(fit.params["_cons"]),
                     se=float(fit.std_errors["_cons"]),
                     n=int(fit.n_obs)),
        ParityRecord(MODULE, "py", "beta_x1",
                     estimate=float(fit.params["x1"]),
                     se=float(fit.std_errors["x1"]),
                     n=int(fit.n_obs)),
        ParityRecord(MODULE, "py", "logLik",
                     estimate=float(fit.log_likelihood),
                     n=int(fit.n_obs)),
        ParityRecord(MODULE, "py", "icc",
                     estimate=float(fit.icc),
                     n=int(fit.n_obs)),
    ]

    write_results(MODULE, "py", rows,
                  extra={"method": "reml",
                         "n_groups": int(fit.n_groups),
                         "cov_type": "unstructured"})


if __name__ == "__main__":
    main()
