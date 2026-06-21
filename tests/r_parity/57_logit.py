"""StatsPAI binary logit parity (Python side) -- Module 57.

Sister module of 48_probit: plain binary logit ML with the default
observed-information SEs, against R stats::glm(binomial(logit)) and
Stata logit.
"""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "57_logit"


def make_data(n=1000, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.binomial(1, 0.5, n).astype(float)
    eta = 0.4 + 0.8 * x1 - 0.6 * x2
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(int)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.logit(formula="y ~ x1 + x2", data=df)
    rows = []
    for nm, lab in [("Intercept", "intercept"), ("_cons", "intercept"),
                    ("x1", "x1"), ("x2", "x2")]:
        if nm in res.params.index:
            rows.append(ParityRecord(MODULE, "py", f"beta_{lab}",
                estimate=float(res.params[nm]),
                se=float(res.std_errors[nm]),
                n=int(len(df))))
    write_results(MODULE, "py", rows, extra={"link": "logit", "engine": "statsmodels"})


if __name__ == "__main__":
    main()
