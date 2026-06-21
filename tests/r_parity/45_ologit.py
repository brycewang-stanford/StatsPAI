"""StatsPAI ordered logit parity (Python side) -- Module 45."""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "45_ologit"

def make_data(n=1000, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    ystar = 0.7 * x + rng.logistic(0, 1, n)
    y = np.where(ystar < -0.5, 0, np.where(ystar < 0.5, 1, 2))
    return pd.DataFrame({"y": y, "x": x.astype(float)})

def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.ologit(formula="y ~ x", data=df)

    rows = []
    for nm, label in [("x", "x"), ("/cut1", "cut1"), ("/cut2", "cut2")]:
        if nm in res.params.index:
            rows.append(ParityRecord(MODULE, "py", f"beta_{label}",
                estimate=float(res.params[nm]),
                se=float(res.std_errors[nm]),
                n=int(len(df))))
    write_results(MODULE, "py", rows, extra={"engine": "statsmodels"})

if __name__ == "__main__":
    main()
