"""StatsPAI Arellano-Bond GMM parity (Python side) -- Module 50."""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd, statspai as sp
from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

MODULE = "50_xtabond"

def make_data(N=100, T=8, seed=PARITY_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    y_prev = np.zeros(N)
    for t in range(T):
        x = rng.normal(0, 1, N)
        y = 0.5 * y_prev + 0.3 * x + rng.normal(0, 1, N)
        for i in range(N):
            rows.append({"id": i, "time": t, "y": float(y[i]), "x": float(x[i])})
        y_prev = y
    return pd.DataFrame(rows)

def main():
    df = make_data()
    dump_csv(df, MODULE)
    # gmm_lags=(2, None) -> all available deeper lags, matching Stata's
    # `xtabond y x, lags(1)` default (GMM-style instruments L(2/.).y).
    res = sp.xtabond(
        df, y="y", x=["x"], id="id", time="time", lags=1,
        gmm_lags=(2, None), method="difference", twostep=False, robust=True,
    )
    coef = res.detail["coefficient"]
    se = res.detail["se"]
    var_ = res.detail["variable"]
    dct = {str(v): (float(coef[i]), float(se[i])) for i, v in enumerate(var_)}

    rows = []
    # sp labels the lagged dependent "L1.y"; Stata reports it as "L.y".
    if "L1.y" in dct:
        est, sev = dct["L1.y"]
        rows.append(ParityRecord(MODULE, "py", "beta_y_lag",
            estimate=est, se=sev, n=int(len(df))))
    if "x" in dct:
        est, sev = dct["x"]
        rows.append(ParityRecord(MODULE, "py", "beta_x",
            estimate=est, se=sev, n=int(len(df))))

    write_results(MODULE, "py", rows,
        extra={"method": "Arellano-Bond difference GMM",
               "step": "one-step", "vcov": "robust"})

if __name__ == "__main__":
    main()
