"""Track C performance -- Classical SCM.

Times sp.synth(method='classic') at N_donors in {20, 50, 100} on a
deterministic 30-period panel. Companion R script times Synth::synth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import TimingResult, time_repeat, write_results


N_DONORS_LIST = [20, 50, 100]
T = 30
N_REPS = 3


def make_panel(n_donors: int, t: int = T, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_units = n_donors + 1
    units = np.arange(n_units)
    rows = []
    # Two-factor DGP: y_{it} = lambda_i' f_t + eps
    F = rng.normal(0, 1, size=(t, 2))
    Lambda = rng.normal(0, 1, size=(n_units, 2))
    for i in units:
        for s, year in enumerate(range(1970, 1970 + t)):
            y = Lambda[i] @ F[s] + rng.normal(0, 0.3)
            rows.append({"unit_id": int(i), "year": year, "y": float(y)})
    return pd.DataFrame(rows)


def main() -> None:
    rows: list[TimingResult] = []
    for n_donors in N_DONORS_LIST:
        df = make_panel(n_donors=n_donors)

        def run_sp() -> None:
            sp.synth(
                df, outcome="y", unit="unit_id", time="year",
                treated_unit=0, treatment_time=1985,
                method="classic",
            )

        med, iqr, mn, mx, peak = time_repeat(run_sp, n_reps=N_REPS, warmup=1)
        rows.append(TimingResult(
            estimator="03_scm", side="py",
            n=n_donors, n_reps=N_REPS,
            median_time_s=med, iqr_time_s=iqr,
            min_time_s=mn, max_time_s=mx, peak_mem_mb=peak,
            extra={"n_donors": n_donors, "T": T},
        ))
        print(f"  n_donors={n_donors:>4}  T={T}  median={med:.3f}s")

    write_results("03_scm", "py", rows)


if __name__ == "__main__":
    main()
