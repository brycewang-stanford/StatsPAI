"""Track C performance benchmark -- HDFE 2-way FE.

Times sp.fast.feols across N in {1e4, 1e5, 1e6} on a deterministic
gravity-style panel with two FE factors (firm, year). Results
written to results/01_hdfe_py.json; companion R-side benchmark
01_hdfe_perf.R times fixest::feols on the same DGP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import TimingResult, time_repeat, write_results


N_LIST = [10_000, 100_000, 1_000_000]
N_REPS = 5


def make_panel(n: int, n_firms: int, n_years: int = 20,
                seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firms, size=n)
    year = rng.integers(0, n_years, size=n)
    firm_fe = rng.normal(0, 1.0, size=n_firms)
    year_fe = rng.normal(0, 0.5, size=n_years)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (
        2.0 * x1 - 1.5 * x2 + firm_fe[firm] + year_fe[year]
        + rng.normal(scale=0.5, size=n)
    )
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2,
                          "firm": firm, "year": year})


def main() -> None:
    rows: list[TimingResult] = []
    for n in N_LIST:
        # Scale firms so cardinality grows but stays << n.
        n_firms = max(50, int(np.sqrt(n) * 2))
        df = make_panel(n=n, n_firms=n_firms)

        def run_sp() -> None:
            sp.fast.feols("y ~ x1 + x2 | firm + year", data=df, vcov="iid")

        med, iqr, mn, mx, peak = time_repeat(run_sp, n_reps=N_REPS, warmup=1)
        rows.append(TimingResult(
            estimator="01_hdfe", side="py", n=n, n_reps=N_REPS,
            median_time_s=med, iqr_time_s=iqr, min_time_s=mn, max_time_s=mx,
            peak_mem_mb=peak,
            extra={"n_firms": n_firms, "n_years": 20,
                   "formula": "y ~ x1 + x2 | firm + year"},
        ))
        print(f"  N={n:>9}  n_firms={n_firms:>5}  median={med:.3f}s  iqr={iqr:.3f}s")

    write_results("01_hdfe", "py", rows)


if __name__ == "__main__":
    main()
