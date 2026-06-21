"""StatsPAI Cox proportional hazards parity (Python side) -- Module 24.

Generates a deterministic survival DGP with two covariates and runs
sp.survival.cox. The companion 24_coxph.R uses survival::coxph.

Tolerance: rel < 1e-3 on the log-hazard-ratio coefficients (Cox PH
likelihood is non-linear, but Efron's tie-handling matches between
implementations on a clean DGP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "24_coxph"


def make_data(n: int = 300, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, n)
    linpred = 0.5 * x1 - 0.3 * x2
    T = -np.log(rng.uniform(0.01, 0.99, n)) / np.exp(linpred)
    C = -np.log(rng.uniform(0.01, 0.99, n)) / 0.3
    time = np.minimum(T, C)
    event = (T <= C).astype(int)
    return pd.DataFrame({"time": time, "event": event, "x1": x1, "x2": x2})


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    fit = sp.survival.cox(
        data=df, duration="time", event="event",
        x=["x1", "x2"], ties="efron",
    )

    rows: list[ParityRecord] = []
    for name in ["x1", "x2"]:
        beta = float(fit.params[name])
        se = float(fit.std_errors[name])
        rows.append(ParityRecord(
            module=MODULE, side="py", statistic=f"beta_{name}",
            estimate=beta, se=se,
            ci_lo=beta - 1.959963984540054 * se,
            ci_hi=beta + 1.959963984540054 * se,
            n=int(len(df)),
        ))

    rows.append(ParityRecord(
        module=MODULE, side="py", statistic="concordance",
        estimate=float(fit.concordance), n=int(len(df)),
    ))

    write_results(MODULE, "py", rows,
                  extra={"ties": "efron",
                         "n_events": int(df["event"].sum())})


if __name__ == "__main__":
    main()
