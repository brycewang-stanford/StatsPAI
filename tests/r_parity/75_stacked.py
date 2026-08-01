"""StatsPAI stacked-DiD parity (Python side) -- Module 75.

Runs ``sp.stacked_did`` under both control-group conventions; the companion R
script builds the same stacks by hand and fits them with ``fixest::feols``.

There is no CRAN package for Cengiz-Dube-Lindner-Zipperer stacking, so the R
side is a hand-written reference rather than a package call. What it pins is
that StatsPAI's stacking construction (one sub-experiment per cohort, clean
controls over the window, cohort-specific unit and time fixed effects,
k = -1 as reference) reproduces the same design when written independently.

Tolerance: rel < 1e-6 on the event-study coefficients and the post mean.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from _common import ParityRecord, dump_csv, write_results

MODULE = "75_stacked"
WINDOW = (-3, 3)


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    rows = []
    for u in range(240):
        g = int(rng.choice([6, 9, 12, 0]))  # 0 = never treated
        fe = rng.normal(0, 0.8)
        for t in range(1, 17):
            on = 1 if (g > 0 and t >= g) else 0
            rows.append(
                {
                    "id": u,
                    "year": t,
                    "first_treat": g,
                    "y": fe + 0.15 * t + 1.3 * on + rng.normal(0, 0.6),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    import statspai as sp

    df = _panel()
    dump_csv(df, MODULE)

    rows: list[ParityRecord] = []
    for spec, never_only in (("never", True), ("nyt", False)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sp.stacked_did(
                df,
                y="y",
                group="id",
                time="year",
                first_treat="first_treat",
                window=WINDOW,
                never_treated_only=never_only,
            )
        n = int(fit.model_info["n_stacked_obs"])
        es = fit.model_info["event_study"]
        for _, row in es.iterrows():
            k = int(row["relative_time"])
            if k == -1:  # reference period, not estimated
                continue
            rows.append(
                ParityRecord(
                    module=MODULE,
                    side="py",
                    statistic=f"{spec}_att_rel_{k}",
                    estimate=float(row["att"]),
                    se=float(row["se"]),
                    n=n,
                )
            )
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"{spec}_ATT_post",
                estimate=float(fit.estimate),
                se=None,
                n=n,
            )
        )

    write_results(MODULE, "py", rows, extra={"estimator": "stacked_did"})


if __name__ == "__main__":
    main()
