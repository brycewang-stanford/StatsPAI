"""StatsPAI truncated regression parity (Python side) -- Module 62.

Left-truncated (at 0) normal regression against R truncreg::truncreg
and Stata truncreg, ll(0). sp.truncreg estimates ln_sigma; the sigma
row is reported on the natural scale with the delta-method SE so it
can be compared to the R/Stata direct-sigma parameterisation.
"""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "62_truncreg"


def make_data(n=1400, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    ystar = 1.0 + 0.8 * x1 - 0.5 * x2 + rng.normal(0, 1.2, n)
    keep = ystar > 0.0
    return pd.DataFrame({"y": ystar[keep], "x1": x1[keep], "x2": x2[keep]})


def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.truncreg(data=df, y="y", x=["x1", "x2"], ll=0.0, maxiter=2000, tol=1e-12)
    rows = []
    for nm, lab in [("_cons", "intercept"), ("x1", "x1"), ("x2", "x2")]:
        if nm in res.params.index:
            rows.append(ParityRecord(MODULE, "py", f"beta_{lab}",
                estimate=float(res.params[nm]),
                se=float(res.std_errors[nm]),
                n=int(len(df))))
    if "ln_sigma" in res.params.index:
        ln_sigma = float(res.params["ln_sigma"])
        se_ln = float(res.std_errors["ln_sigma"])
        sigma = float(np.exp(ln_sigma))
        rows.append(ParityRecord(MODULE, "py", "sigma",
            estimate=sigma, se=sigma * se_ln, n=int(len(df))))
    write_results(MODULE, "py", rows,
                  extra={"truncation": "left at 0",
                         "sigma_note": ("sp.truncreg estimates ln_sigma; sigma row is "
                                        "exp(ln_sigma) with the delta-method SE.")})


if __name__ == "__main__":
    main()
