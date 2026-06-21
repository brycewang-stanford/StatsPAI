"""StatsPAI conditional logit parity (Python side) -- Module 46."""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "46_clogit"

def make_data(n_groups=300, n_per=3, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        x = rng.normal(0, 1, n_per)
        u = 0.8 * x + rng.gumbel(0, 1, n_per)
        chosen = int(np.argmax(u))
        for i in range(n_per):
            rows.append({"group": g, "choice": int(i == chosen), "x": float(x[i])})
    return pd.DataFrame(rows)

def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.clogit(formula="choice ~ x", data=df, group="group")
    rows = [ParityRecord(MODULE, "py", "beta_x",
                          estimate=float(res.params["x"]),
                          se=float(res.std_errors["x"]),
                          n=int(len(df)))]
    write_results(MODULE, "py", rows, extra={"group_var": "group", "engine": "statsmodels"})

if __name__ == "__main__":
    main()
