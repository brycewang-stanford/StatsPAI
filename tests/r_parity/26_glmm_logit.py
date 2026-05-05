"""StatsPAI GLMM logit parity (Python side) -- Module 26.

Random-intercept logistic GLMM on a deterministic n=1500, 60-group
panel. The companion 26_glmm_logit.R uses lme4::glmer with family =
binomial. Tolerance: rel < 1e-2 on the fixed effects (Laplace
approximation default; AGHQ in Module 27).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "26_glmm_logit"


def make_data(n_groups: int = 60, n_per: int = 25,
              seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gid = np.repeat(np.arange(n_groups), n_per)
    n = n_groups * n_per
    x1 = rng.normal(size=n)
    group_re = rng.normal(scale=0.8, size=n_groups)
    linpred = -0.5 + 0.7 * x1 + group_re[gid]
    prob = 1 / (1 + np.exp(-linpred))
    y = (rng.uniform(size=n) < prob).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "gid": gid})


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    fit = sp.melogit(data=df, y="y", x_fixed=["x1"], group="gid")

    rows: list[ParityRecord] = [
        ParityRecord(MODULE, "py", "beta_intercept",
                     estimate=float(fit.params["_cons"]),
                     se=float(fit.bse["_cons"]),
                     n=int(fit.n_obs)),
        ParityRecord(MODULE, "py", "beta_x1",
                     estimate=float(fit.params["x1"]),
                     se=float(fit.bse["x1"]),
                     n=int(fit.n_obs)),
        ParityRecord(MODULE, "py", "logLik",
                     estimate=float(fit.log_likelihood),
                     n=int(fit.n_obs)),
    ]

    write_results(MODULE, "py", rows,
                  extra={"family": "binomial",
                         "link": "logit",
                         "n_groups": int(fit.n_groups),
                         "nAGQ": 1})


if __name__ == "__main__":
    main()
