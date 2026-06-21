"""StatsPAI multinomial logit parity (Python side) -- Module 44."""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "44_mlogit"

def make_data(n=1000, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    u1 = 0.3 + 0.5 * x1 - 0.2 * x2
    u2 = -0.1 - 0.3 * x1 + 0.4 * x2
    expu = np.column_stack([np.zeros(n), u1, u2])
    expu = np.exp(expu - expu.max(axis=1, keepdims=True))
    p = expu / expu.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p[i]) for i in range(n)])
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})

def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.mlogit(formula="y ~ x1 + x2", data=df, base=0)
    rows = []
    # sp parameter naming: [<class>]<varname>, e.g. "[1]x1", "[1]_cons"
    for cls in [1, 2]:
        for nm, label in [("_cons", "intercept"), ("x1", "x1"), ("x2", "x2")]:
            key = f"[{cls}]{nm}"
            if key in res.params.index:
                rows.append(ParityRecord(MODULE, "py",
                    f"class{cls}_{label}",
                    estimate=float(res.params[key]),
                    se=float(res.std_errors[key]),
                    n=int(len(df))))
    write_results(MODULE, "py", rows, extra={"base_class": 0, "engine": "statsmodels"})

if __name__ == "__main__":
    main()
