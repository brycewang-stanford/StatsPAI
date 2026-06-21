"""StatsPAI Tobit parity (Python side) -- Module 41.

DGP: linear latent variable censored at 0 from below. Tests
sp.tobit against R censReg::censReg and Stata's tobit.

Tolerance: rel < 1e-3 on coefficients; rel < 1e-2 on SEs and sigma.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "41_tobit"


def make_data(n: int = 500, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    ystar = 0.5 + 0.8 * x + rng.normal(0, 1, n)
    y = np.maximum(ystar, 0.0)
    return pd.DataFrame({"y": y, "x": x})


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    res = sp.tobit(data=df, y="y", x=["x"], ll=0)

    coef = res.detail["coefficient"]
    se = res.detail["se"]
    var_ = res.detail["variable"]
    dct = {str(v): (float(coef[i]), float(se[i])) for i, v in enumerate(var_)}

    rows: list[ParityRecord] = []
    for name in ["const", "x"]:
        if name in dct:
            est, sev = dct[name]
            label = "intercept" if name == "const" else name
            rows.append(ParityRecord(
                MODULE, "py",
                statistic=f"beta_{label}",
                estimate=est, se=sev, n=int(len(df))))
    # Sigma (residual scale)
    if "sigma" in dct:
        est, sev = dct["sigma"]
        rows.append(ParityRecord(
            MODULE, "py", "sigma",
            estimate=est, se=sev, n=int(len(df))))

    write_results(MODULE, "py", rows,
                  extra={"left_censor": 0.0, "engine": "statsmodels"})


if __name__ == "__main__":
    main()
