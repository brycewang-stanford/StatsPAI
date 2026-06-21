"""StatsPAI quantile regression parity (Python side) -- Module 40.

Generates a heteroskedastic linear DGP and runs sp.qreg at tau=0.5.
The R side uses quantreg::rq; Stata uses qreg.

Tolerance: rel < 1e-3 on coefficients (simplex solver converges to
the same vertex); rel < 5e-2 on SEs (different SE methods).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "40_qreg"


def make_data(n: int = 500, seed: int = PARITY_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eps = rng.normal(0, 1.0 + 0.3 * x1**2, n)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + eps
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    res = sp.qreg(data=df, y="y", x=["x1", "x2"], quantile=0.5)
    # detail is a pandas.DataFrame-like dict with parallel arrays
    coef = res.detail["coefficient"]
    se = res.detail["se"]
    var_ = res.detail["variable"]
    # Build dict by variable name
    dct = {str(v): (float(coef[i]), float(se[i])) for i, v in enumerate(var_)}

    rows: list[ParityRecord] = []
    for name in ["const", "x1", "x2"]:
        if name in dct:
            est, sev = dct[name]
            label = "intercept" if name == "const" else name
            rows.append(ParityRecord(
                MODULE, "py",
                statistic=f"beta_{label}",
                estimate=est, se=sev, n=int(len(df))))

    write_results(MODULE, "py", rows,
                  extra={"quantile": 0.5, "engine": "statsmodels"})


if __name__ == "__main__":
    main()
