"""Track C performance -- DML PLR.

Times sp.dml(model='plr') with linear-regression nuisance learners
(closed-form, no learner Monte Carlo) at N in {1000, 5000, 10000}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp
from sklearn.linear_model import LinearRegression

from _common import TimingResult, time_repeat, write_results


N_LIST = [1_000, 5_000, 10_000]
N_REPS = 3


def make_data(n: int, p: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    d = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n)) > 0
    d = d.astype(int)
    y = 0.5 * d + X.sum(axis=1) * 0.1 + rng.normal(size=n)
    cols = {f"x{i+1}": X[:, i] for i in range(p)}
    cols["d"] = d
    cols["y"] = y
    return pd.DataFrame(cols)


def main() -> None:
    rows: list[TimingResult] = []
    for n in N_LIST:
        df = make_data(n=n)

        def run_sp() -> None:
            sp.dml(
                data=df, y="y", d="d",
                X=[f"x{i+1}" for i in range(5)],
                model="plr",
                model_y=LinearRegression(),
                model_d=LinearRegression(),
            )

        med, iqr, mn, mx, peak = time_repeat(run_sp, n_reps=N_REPS, warmup=1)
        rows.append(TimingResult(
            estimator="04_dml", side="py", n=n, n_reps=N_REPS,
            median_time_s=med, iqr_time_s=iqr,
            min_time_s=mn, max_time_s=mx, peak_mem_mb=peak,
            extra={"model": "plr", "ml_g": "LinearRegression",
                   "ml_m": "LinearRegression", "n_folds": 5},
        ))
        print(f"  N={n:>6}  median={med:.3f}s")

    write_results("04_dml", "py", rows)


if __name__ == "__main__":
    main()
