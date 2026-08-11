"""StatsPAI design-based staggered-rollout parity (Python side) -- Module 82.

Runs ``sp.staggered_rollout`` / ``sp.staggered_cs`` / ``sp.staggered_sa`` on a
randomised rollout panel; the companion R script runs ``staggered::staggered``
(Roth & Sant'Anna's own package) on the same dumped CSV.

Why this module is not redundant with 04_csdid
----------------------------------------------
Every other DiD module in this harness reconciles a *parallel-trends*
estimator. This one reconciles a **design-based** estimator: identification
comes from random adoption timing, and the standard errors are Neyman rather
than sampling-based. The panel is therefore generated with adoption dates
assigned at random, which is the design the estimator is for -- and which no
other module here supplies.

Both standard errors are emitted. R reports a conservative (Neyman) SE and an
adjusted SE that subtracts the part of the variance random timing identifies;
reconciling only one of the two would leave half the inference path unchecked.

Tolerance: rel < 1e-6 on every estimate and both standard errors.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from _common import ParityRecord, dump_csv, write_results

MODULE = "82_staggered"
EVENT_TIMES = (0, 1, 2)


def _rollout_panel() -> pd.DataFrame:
    """A randomised staggered rollout with no never-treated units.

    Adoption dates are dealt at random, so the design-based assumption holds
    by construction. Leaving out never-treated units is deliberate: it makes
    ``max(g)`` finite, a branch the canonical mpdta panel never reaches.
    """
    rng = np.random.default_rng(20260811)
    n_units, periods, cohorts = 240, range(1, 7), (3, 4, 5, 6)
    unit_g = rng.permutation(np.tile(cohorts, n_units // len(cohorts)))
    unit_fe = rng.normal(0.0, 0.8, size=n_units)

    rows = []
    for u in range(n_units):
        g = int(unit_g[u])
        for t in periods:
            rel = t - g
            y = unit_fe[u] + 0.15 * t
            if rel >= 0:
                y += 0.4 + 0.1 * rel
            y += rng.normal(0.0, 0.5)
            rows.append({"unit": u + 1, "time": t, "first_treat": g, "y": y})
    return pd.DataFrame(rows)


def _record(statistic: str, res, n: int) -> list[ParityRecord]:
    """Point estimate plus both standard errors as three parity rows."""
    return [
        ParityRecord(
            module=MODULE,
            side="py",
            statistic=statistic,
            estimate=float(res.estimate),
            n=n,
        ),
        ParityRecord(
            module=MODULE,
            side="py",
            statistic=f"{statistic}_se_neyman",
            estimate=float(res.model_info["se_neyman"]),
            n=n,
        ),
        ParityRecord(
            module=MODULE,
            side="py",
            statistic=f"{statistic}_se_adjusted",
            estimate=float(res.model_info["se_adjusted"]),
            n=n,
        ),
    ]


def main() -> None:
    import statspai as sp

    frame = _rollout_panel()
    dump_csv(frame, MODULE)
    n_units = int(frame["unit"].nunique())
    keys = dict(y="y", i="unit", t="time", g="first_treat")

    rows: list[ParityRecord] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for estimand in ("simple", "cohort", "calendar"):
            for tag, efficient in (("efficient", True), ("plugin", False)):
                res = sp.staggered_rollout(
                    frame, estimand=estimand, efficient=efficient, **keys
                )
                rows += _record(f"{estimand}_{tag}", res, n_units)

        for e in EVENT_TIMES:
            res = sp.staggered_rollout(
                frame, estimand="eventstudy", event_time=e, **keys
            )
            rows += _record(f"eventstudy_e{e}", res, n_units)

        rows += _record("cs_simple", sp.staggered_cs(frame, **keys), n_units)
        rows += _record("sa_simple", sp.staggered_sa(frame, **keys), n_units)

    write_results(MODULE, "py", rows, extra={"estimator": "staggered_rollout"})


if __name__ == "__main__":
    main()
