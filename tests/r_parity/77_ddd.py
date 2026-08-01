"""StatsPAI triple-differences parity (Python side) -- Module 77.

Runs ``sp.ddd_heterogeneous`` on a staggered DDD panel; the companion R
script runs ``triplediff::ddd`` (Ortiz-Villavicencio & Sant'Anna 2025) on
the same dumped CSV with ``xformla = ~1``, ``est_method = "dr"`` and
never-treated controls. With no covariates the doubly-robust DDD collapses
to the unconditional cell means StatsPAI computes, so the per-``(g, t)``
estimates should agree exactly.

Two things are deliberately NOT compared:

* Standard errors. ``triplediff`` reports analytical influence-function
  SEs; ``sp.ddd_heterogeneous`` has only a cluster bootstrap. Comparing
  them would compare two different variance estimators.
* Pre-treatment placebo cells. ``triplediff`` with ``base_period =
  "varying"`` also reports ``(g, t)`` for ``t < g``; StatsPAI only builds
  post cells.

The overall aggregate is compared under ``weight_by="cohort"``, which is
``triplediff``'s ``pg`` convention (cohort share over all units, both
subgroups). StatsPAI's default weights by treated-eligible units instead;
both are emitted so the harness records the gap rather than hiding it.

Tolerance: rel < 1e-6 on the cell estimates and on the cohort-weighted
aggregate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from _common import ParityRecord, dump_csv, write_results

MODULE = "77_ddd"
COHORTS = [0, 2, 3, 4]  # 0 = never treated
PERIODS = [1, 2, 3, 4]
COVARS = ["cov1", "cov2"]


def _panel() -> pd.DataFrame:
    """Staggered DDD panel; only the affected subgroup carries the effect.

    The affected share varies across cohorts (0.7 / 0.5 / 0.35) on purpose:
    that is exactly the case where the two aggregation conventions differ,
    so the harness pins the gap rather than a coincidence.
    """
    rng = np.random.default_rng(11)
    share = {2: 0.70, 3: 0.50, 4: 0.35, 0: 0.55}
    rows = []
    uid = 0
    for g in COHORTS:
        for _ in range(150):
            uid += 1
            b = int(rng.random() < share[g])
            fe = rng.normal(0.0, 1.0)
            # Covariates that shift both selection and the outcome path, so
            # the conditional and unconditional estimands genuinely differ.
            c1 = rng.normal(0.4 * b, 1.0)
            c2 = rng.normal(-0.2 * (g != 0), 1.0)
            for t in PERIODS:
                on = g != 0 and t >= g
                effect = (2.0 + 0.5 * (t - g)) if (on and b == 1) else 0.0
                # A subgroup-specific trend the DDD is supposed to net out.
                y = fe + 0.30 * t + 0.45 * b * t + effect + rng.normal(0, 0.8)
                y += (0.5 * c1 - 0.3 * c2) * t
                rows.append(
                    {
                        "id": uid,
                        "time": t,
                        "state": g,
                        "partition": b,
                        "y": y,
                        "cov1": c1,
                        "cov2": c2,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    import statspai as sp

    df = _panel()
    dump_csv(df, MODULE)

    rows: list[ParityRecord] = []
    fits = {}
    for weight_by in ("cohort", "eligible"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits[weight_by] = sp.ddd_heterogeneous(
                df,
                y="y",
                unit="id",
                time="time",
                cohort="state",
                subgroup="partition",
                n_boot=0,
                seed=0,
                weight_by=weight_by,
            )

    for _, row in fits["cohort"].detail.iterrows():
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"ddd_g{int(row['cohort'])}_t{int(row['time'])}",
                estimate=float(row["ddd"]),
                n=int(len(df)),
            )
        )

    rows.append(
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="simple_ATT_cohort_weights",
            estimate=float(fits["cohort"].estimate),
            n=int(len(df)),
        )
    )
    rows.append(
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="simple_ATT_eligible_weights",
            estimate=float(fits["eligible"].estimate),
            n=int(len(df)),
        )
    )

    # Conditional DDD: all three nuisance combinations, cells and analytic
    # standard errors. The SEs are compared here (unlike the unconditional
    # block above) because the analytic path IS the reference's variance
    # estimator, not a bootstrap standing in for it.
    for method in ("dr", "ipw", "reg"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cond = sp.ddd_heterogeneous(
                df,
                y="y",
                unit="id",
                time="time",
                cohort="state",
                subgroup="partition",
                n_boot=0,
                seed=0,
                weight_by="cohort",
                x=COVARS,
                est_method=method,
                se="analytic",
            )
        for _, row in cond.detail.iterrows():
            tag = f"{method}_g{int(row['cohort'])}_t{int(row['time'])}"
            rows.append(
                ParityRecord(
                    module=MODULE,
                    side="py",
                    statistic=f"ddd_{tag}",
                    estimate=float(row["ddd"]),
                    se=float(row["se"]),
                    n=int(len(df)),
                )
            )
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"simple_ATT_{method}",
                estimate=float(cond.estimate),
                se=float(cond.se),
                n=int(len(df)),
            )
        )

    write_results(MODULE, "py", rows, extra={"estimator": "ddd_heterogeneous"})


if __name__ == "__main__":
    main()
