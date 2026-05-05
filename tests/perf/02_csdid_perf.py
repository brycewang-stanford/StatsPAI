"""Track C performance -- Callaway-Sant'Anna staggered DiD.

Times sp.callaway_santanna at N_units in {1000, 5000, 25000} with
T = 5 years. The companion R script times did::att_gt + did::aggte.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import TimingResult, time_repeat, write_results


N_UNITS_LIST = [1_000, 5_000, 25_000]
T = 5
N_REPS = 3   # CS-DiD is heavier; fewer reps


def make_panel(n_units: int, t: int = T, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n_units), t)
    years = np.tile(np.arange(t), n_units)
    # 60% never-treated, 40% treated split equally across 3 cohorts.
    cohort = np.zeros(n_units, dtype=int)
    treated_idx = rng.choice(n_units, size=int(0.4 * n_units), replace=False)
    rng.shuffle(treated_idx)
    third = len(treated_idx) // 3
    cohort[treated_idx[:third]] = 2
    cohort[treated_idx[third:2 * third]] = 3
    cohort[treated_idx[2 * third:]] = 4
    first_treat = cohort[units]
    treat = ((first_treat > 0) & (years >= first_treat)).astype(int)
    unit_fe = rng.normal(0, 0.5, size=n_units)
    year_fe = rng.normal(0, 0.3, size=t)
    eps = rng.normal(0, 0.4, size=n_units * t)
    y = (
        unit_fe[units] + year_fe[years] - 0.05 * treat + eps
    )
    return pd.DataFrame({
        "y": y, "unit": units, "year": years,
        "first_treat": first_treat, "treat": treat,
    })


def main() -> None:
    rows: list[TimingResult] = []
    for n_units in N_UNITS_LIST:
        df = make_panel(n_units=n_units)

        def run_sp() -> None:
            sp.callaway_santanna(
                df, y="y", g="first_treat", t="year", i="unit",
                estimator="reg", control_group="nevertreated",
            )

        med, iqr, mn, mx, peak = time_repeat(run_sp, n_reps=N_REPS, warmup=1)
        rows.append(TimingResult(
            estimator="02_csdid", side="py",
            n=int(n_units * T), n_reps=N_REPS,
            median_time_s=med, iqr_time_s=iqr,
            min_time_s=mn, max_time_s=mx, peak_mem_mb=peak,
            extra={"n_units": n_units, "T": T},
        ))
        print(f"  N_units={n_units:>6}  N_obs={n_units*T:>7}  median={med:.3f}s")

    write_results("02_csdid", "py", rows)


if __name__ == "__main__":
    main()
