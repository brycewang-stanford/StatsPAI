"""StatsPAI seemingly unrelated regressions parity (Python side) -- Module 60.

Two-equation SUR with cross-correlated errors. sp.sureg's one-step FGLS
(Sigma = E'E / n from equation-by-equation OLS residuals, no df
correction) matches the Stata sureg default; the R side reproduces the
same convention via systemfit(method='SUR',
control=systemfit.control(methodResidCov='noDfCor')).
"""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "60_sureg"


def make_data(n=1000, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    w = rng.normal(0, 1, n)
    e = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.6], [0.6, 1.5]], size=n)
    y1 = 1.0 + 0.8 * x1 + 0.4 * w + e[:, 0]
    y2 = -0.5 + 0.6 * x2 - 0.3 * w + e[:, 1]
    return pd.DataFrame({"y1": y1, "y2": y2, "x1": x1, "x2": x2, "w": w})


def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.sureg({"eq1": ("y1", ["x1", "w"]),
                    "eq2": ("y2", ["x2", "w"])}, data=df, method="fgls")
    labels = ["eq1_intercept", "eq1_x1", "eq1_w",
              "eq2_intercept", "eq2_x2", "eq2_w"]
    rows = []
    for i, lab in enumerate(labels):
        rows.append(ParityRecord(MODULE, "py", f"beta_{lab}",
            estimate=float(res.params_all[i]),
            se=float(res.se_all[i]),
            n=int(res.n_obs)))
    write_results(MODULE, "py", rows,
                  extra={"method": "one-step FGLS", "sigma_divisor": "n"})


if __name__ == "__main__":
    main()
