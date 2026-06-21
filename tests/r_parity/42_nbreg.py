"""StatsPAI negative binomial parity (Python side) -- Module 42."""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "42_nbreg"

def make_data(n=600, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.binomial(1, 0.5, n).astype(float)
    mu = np.exp(0.8 + 0.5 * x1 - 0.4 * x2)
    # NB with dispersion alpha = 1.0
    alpha = 1.0
    p = mu / (mu + 1.0 / alpha)
    n_param = 1.0 / alpha
    y = rng.negative_binomial(n=n_param, p=1 - p)
    return pd.DataFrame({"y": y.astype(float), "x1": x1, "x2": x2})

def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.nbreg(formula="y ~ x1 + x2", data=df)
    rows = []
    for nm, label in [("_cons", "intercept"), ("x1", "x1"), ("x2", "x2")]:
        if nm in res.params.index:
            rows.append(ParityRecord(MODULE, "py",
                f"beta_{label}",
                estimate=float(res.params[nm]),
                se=float(res.std_errors[nm]),
                n=int(len(df))))
    write_results(MODULE, "py", rows, extra={"dispersion": "mean", "engine": "statsmodels"})

if __name__ == "__main__":
    main()
