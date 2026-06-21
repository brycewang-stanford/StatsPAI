"""StatsPAI LIML parity (Python side) -- Module 59.

Over-identified linear IV model (one endogenous regressor, two
instruments) estimated by limited-information maximum likelihood,
against R ivmodel::ivmodel()'s LIML row and Stata ivregress liml.
"""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "59_liml"


def make_data(n=1500, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    w = rng.normal(0, 1, n)
    u = rng.normal(0, 1, n)
    x = 0.5 * z1 + 0.4 * z2 + 0.3 * w + 0.6 * u + rng.normal(0, 1, n)
    y = 1.0 + 0.7 * x + 0.5 * w + u
    return pd.DataFrame({"y": y, "x": x, "w": w, "z1": z1, "z2": z2})


def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.liml(data=df, y="y", x_endog=["x"], x_exog=["w"], z=["z1", "z2"])
    rows = []
    for nm, lab in [("x", "x"), ("w", "w"), ("_cons", "intercept")]:
        if nm in res.params.index:
            rows.append(ParityRecord(MODULE, "py", f"beta_{lab}",
                estimate=float(res.params[nm]),
                se=float(res.std_errors[nm]),
                n=int(len(df))))
    write_results(MODULE, "py", rows,
                  extra={"n_instruments": 2, "endog": "x"})


if __name__ == "__main__":
    main()
