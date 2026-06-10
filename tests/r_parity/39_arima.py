"""StatsPAI ARIMA parity (Python side) -- Module 39.

DGP: AR(2) with phi1=0.6, phi2=-0.2. Fits ARIMA(2,0,0). The
companion R/Stata sides fit the same exact-MLE convention.

sp.arima now exposes ``ARIMAResult.se`` (statsmodels' asymptotic SEs),
so we compare standard errors alongside the point estimates and logLik.

Tolerance: rel < 1e-6 on AR coefficients and log-likelihood.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "39_arima"


def make_data(T: int = 300, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    eps = rng.normal(0, 0.7, T)
    for t in range(2, T):
        y[t] = 0.6 * y[t - 1] - 0.2 * y[t - 2] + eps[t]
    return pd.DataFrame({"y": y})


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    res = sp.arima(df["y"].values, order=(2, 0, 0), method="innovations_mle")

    rows: list[ParityRecord] = [
        ParityRecord(
            MODULE,
            "py",
            "ar1",
            estimate=float(res.params["ar.L1"]),
            se=float(res.se["ar.L1"]),
            n=int(len(df)),
        ),
        ParityRecord(
            MODULE,
            "py",
            "ar2",
            estimate=float(res.params["ar.L2"]),
            se=float(res.se["ar.L2"]),
            n=int(len(df)),
        ),
        ParityRecord(
            MODULE,
            "py",
            "sigma2",
            estimate=float(res.params["sigma2"]),
            se=float(res.se["sigma2"]),
            n=int(len(df)),
        ),
        ParityRecord(
            MODULE, "py", "logLik", estimate=float(res.log_likelihood), n=int(len(df))
        ),
    ]

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "order": "(2,0,0)",
            "engine": "statsmodels",
            "method": "innovations_mle",
            "reference_method_note": (
                "sp.arima(..., method='innovations_mle') uses statsmodels' "
                "stationary/invertible innovations-MLE path to align with "
                "stats::arima(method='ML') and tightly converged Stata arima. "
                "The default sp.arima() method remains exact state-space MLE."
            ),
        },
    )


if __name__ == "__main__":
    main()
