"""StatsPAI binary probit parity (Python side) -- Module 48."""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "48_probit"

def make_data(n=1000, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n); x2 = rng.binomial(1, 0.5, n).astype(float)
    from scipy import stats
    ystar = 0.3 + 0.6*x1 - 0.4*x2 + rng.normal(0, 1, n)
    y = (ystar > 0).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})

def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.probit(formula="y ~ x1 + x2", data=df)
    rows = []
    for nm, lab in [("_cons","intercept"), ("x1","x1"), ("x2","x2")]:
        if nm in res.params.index:
            rows.append(ParityRecord(MODULE, "py", f"beta_{lab}",
                estimate=float(res.params[nm]),
                se=float(res.std_errors[nm]),
                n=int(len(df))))
    write_results(MODULE, "py", rows, extra={"link": "probit", "engine": "statsmodels"})

if __name__ == "__main__":
    main()
