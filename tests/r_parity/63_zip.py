"""StatsPAI zero-inflated Poisson parity (Python side) -- Module 63.

ZIP with a logit inflation equation, against R pscl::zeroinfl
(dist='poisson', link='logit') and Stata zip ..., inflate(...).
"""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "63_zip"


def make_data(n=1500, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.binomial(1, 0.5, n).astype(float)
    z = rng.normal(0, 1, n)
    p_inflate = 1.0 / (1.0 + np.exp(-(-0.7 + 0.9 * z)))
    lam = np.exp(0.5 + 0.6 * x1 - 0.4 * x2)
    infl = rng.binomial(1, p_inflate)
    y = np.where(infl == 1, 0, rng.poisson(lam))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "z": z})


def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.zip_model(formula="y ~ x1 + x2", data=df, inflate=["z"], maxiter=2000, tol=1e-12)
    name_map = [
        ("const", "count_intercept"),
        ("x1", "count_x1"),
        ("x2", "count_x2"),
        ("inflate_const", "inflate_intercept"),
        ("inflate_z", "inflate_z"),
    ]
    rows = []
    for nm, lab in name_map:
        if nm in res.params.index:
            rows.append(ParityRecord(MODULE, "py", f"beta_{lab}",
                estimate=float(res.params[nm]),
                se=float(res.std_errors[nm]),
                n=int(len(df))))
    write_results(MODULE, "py", rows,
                  extra={"inflate_link": "logit", "count_dist": "poisson"})


if __name__ == "__main__":
    main()
