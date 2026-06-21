"""StatsPAI plain Poisson regression parity (Python side) -- Module 58.

Plain (no-FE) Poisson ML with default observed-information SEs against
R stats::glm(poisson) and Stata poisson. The HDFE robust Poisson is
covered separately by 37_ppmlhdfe / 47_ppmlhdfe_3fe.
"""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "58_poisson"


def make_data(n=1000, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.binomial(1, 0.5, n).astype(float)
    lam = np.exp(0.2 + 0.5 * x1 + 0.3 * x2)
    y = rng.poisson(lam)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.poisson(formula="y ~ x1 + x2", data=df)
    rows = []
    for nm, lab in [("Intercept", "intercept"), ("_cons", "intercept"),
                    ("x1", "x1"), ("x2", "x2")]:
        if nm in res.params.index:
            rows.append(ParityRecord(MODULE, "py", f"beta_{lab}",
                estimate=float(res.params[nm]),
                se=float(res.std_errors[nm]),
                n=int(len(df))))
    write_results(MODULE, "py", rows, extra={"family": "poisson", "engine": "statsmodels"})


if __name__ == "__main__":
    main()
