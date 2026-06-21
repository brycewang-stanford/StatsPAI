"""StatsPAI Heckman selection parity (Python side) -- Module 43."""
from __future__ import annotations
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "43_heckman"

def make_data(n=600, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    # Selection eqn: sel = 1{0.3 + 0.5*z + u > 0}
    u = rng.normal(0, 1, n)
    sel_lat = 0.3 + 0.5 * z + u
    sel = (sel_lat > 0).astype(int)
    # Outcome eqn (when selected)
    eps = rng.normal(0, 1, n)
    y_lat = 1.0 + 0.5 * x + eps
    y = np.where(sel == 1, y_lat, np.nan)
    return pd.DataFrame({"y": y, "x": x, "z": z, "sel": sel})

def main():
    df = make_data()
    dump_csv(df, MODULE)
    res = sp.heckman(data=df, y="y", x=["x"], select="sel", z=["z"])

    coef = res.detail["coefficient"]
    se = res.detail["se"]
    var_ = res.detail["variable"]
    dct = {str(v): (float(coef[i]), float(se[i])) for i, v in enumerate(var_)}

    rows = []
    for name in ["const", "x"]:
        if name in dct:
            est, sev = dct[name]
            label = "intercept" if name == "const" else name
            rows.append(ParityRecord(MODULE, "py",
                f"beta_{label}", estimate=est, se=sev, n=int(len(df))))
    # lambda (IMR) coefficient
    if "lambda (IMR)" in dct:
        est, sev = dct["lambda (IMR)"]
        rows.append(ParityRecord(MODULE, "py", "lambda_imr",
            estimate=est, se=sev, n=int(len(df))))

    write_results(MODULE, "py", rows,
        extra={"method": "2-step Heckman", "engine": "statspai"})

if __name__ == "__main__":
    main()
