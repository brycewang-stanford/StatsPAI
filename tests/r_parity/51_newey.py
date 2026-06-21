"""StatsPAI Newey-West HAC parity (Python side) -- Module 51."""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "51_newey"

def make_data(T=200, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, T)
    e = rng.normal(0, 1, T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t-1] + 0.3 * x[t] + e[t]
    return pd.DataFrame({"y": y, "x": x, "t": np.arange(T)})

def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.regress("y ~ x", df, robust="hac", lags=4)
    rows = []
    for nm, lab in [("Intercept", "intercept"), ("x", "x")]:
        if nm in res.params.index:
            rows.append(ParityRecord(MODULE, "py", f"beta_{lab}",
                estimate=float(res.params[nm]),
                se=float(res.std_errors[nm]),
                n=int(len(df))))
    write_results(MODULE, "py", rows, extra={"vcov": "HAC", "lags": 4})

if __name__ == "__main__":
    main()
