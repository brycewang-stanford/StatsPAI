"""StatsPAI Newey-West HAC parity (Python side) -- Module 51."""
from __future__ import annotations
import math

import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "51_newey"
# The lag length the companion 51_newey.R pins.
R_SIDE_LAGS = 4

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
    # sp.regress used to take lags=4 explicitly; that keyword was removed and
    # HAC now always uses the Newey-West rule floor(4 * (T/100)^(2/9)). At the
    # T = 200 of this fixture that rule returns exactly 4, so the comparison
    # against the R side (which fixes lags = 4) is still like-for-like. The
    # guard below fails loudly if the fixture size ever changes, because then
    # the two sides would silently be using different lag lengths.
    expected_lags = math.floor(4 * (len(df) / 100) ** (2 / 9))
    if expected_lags != R_SIDE_LAGS:
        raise SystemExit(
            f"HAC lag mismatch: R fixes lags={R_SIDE_LAGS} but the Newey-West "
            f"rule gives {expected_lags} at T={len(df)}. Re-pin the R side or "
            "restore an explicit lag option before trusting this row."
        )
    res = sp.regress("y ~ x", df, robust="hac")
    rows = []
    for nm, lab in [("Intercept", "intercept"), ("x", "x")]:
        if nm in res.params.index:
            rows.append(ParityRecord(MODULE, "py", f"beta_{lab}",
                estimate=float(res.params[nm]),
                se=float(res.std_errors[nm]),
                n=int(len(df))))
    write_results(MODULE, "py", rows, extra={"vcov": "HAC", "lags": R_SIDE_LAGS, "lag_source": "newey_west_rule"})

if __name__ == "__main__":
    main()
