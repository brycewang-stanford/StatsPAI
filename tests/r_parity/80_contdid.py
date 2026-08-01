"""StatsPAI continuous-treatment DiD parity (Python side) -- Module 80.

Runs ``sp.cgs_continuous_did`` on a two-period panel with a continuous dose;
the companion R script runs ``contdid::cont_did`` (Callaway, Goodman-Bacon &
Sant'Anna's own package) on the same dumped CSV.

Three spline specifications are emitted -- linear with no interior knots,
cubic with none, cubic with two -- because the whole estimator is a spline
regression in the dose and a single degree would leave the basis mostly
unexercised.

``curve_basis="reference"`` is used for the dose curves. ``contdid`` fits the
spline on the range of the observed treated doses but evaluates the reported
curves on a basis re-anchored to the ends of the dose GRID, so its reported
curves are a rescaled version of the fitted dose response and do not line up
with the overall ACRT it returns alongside them. StatsPAI defaults to one
consistent basis and keeps the reference's behaviour behind that flag. The overall ATT and ACRT
are computed on the fitted basis in both packages and need no flag.

Standard errors are not compared: ``contdid`` routes them through the ``pte``
package's aggregation layer, which StatsPAI does not replicate.

Tolerance: rel < 1e-6 on the curves and both overall quantities.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from _common import ParityRecord, dump_csv, write_results

MODULE = "80_contdid"
SPECS = ((1, 0), (3, 0), (3, 2))
GRID_POINTS = (0, 30, 60, 89)  # where on the 90-point grid to compare


def _panel() -> pd.DataFrame:
    """Two periods, half the units dosed continuously, a quadratic response.

    The quadratic term is what makes ACRT vary with the dose; a purely
    linear design would let a wrong basis pass unnoticed.
    """
    rng = np.random.default_rng(19)
    rows = []
    for uid in range(1, 2001):
        treated = uid <= 1000
        d = float(rng.uniform(0.02, 1.0)) if treated else 0.0
        fe = rng.normal(0.0, 1.0)
        for t in (1, 2):
            eff = (1.6 * d + 0.9 * d**2) if (treated and t == 2) else 0.0
            rows.append(
                {
                    "id": uid,
                    "time_period": t,
                    "G": 2 if treated else 0,
                    "D": d,
                    "Y": fe + 0.5 * t + eff + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    import statspai as sp

    df = _panel()
    dump_csv(df, MODULE)

    rows: list[ParityRecord] = []
    for degree, num_knots in SPECS:
        tag = f"d{degree}k{num_knots}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.cgs_continuous_did(
                df,
                y="Y",
                dose="D",
                time="time_period",
                unit="id",
                cohort="G",
                degree=degree,
                num_knots=num_knots,
                curve_basis="reference",
            )
        for j in GRID_POINTS:
            rows.append(
                ParityRecord(
                    module=MODULE,
                    side="py",
                    statistic=f"{tag}_att_d_{j}",
                    estimate=float(res.att_d[j]),
                    n=int(res.n_units),
                )
            )
            rows.append(
                ParityRecord(
                    module=MODULE,
                    side="py",
                    statistic=f"{tag}_acrt_d_{j}",
                    estimate=float(res.acrt_d[j]),
                    n=int(res.n_units),
                )
            )
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"{tag}_overall_att",
                estimate=float(res.overall_att),
                n=int(res.n_units),
            )
        )
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"{tag}_overall_acrt",
                estimate=float(res.overall_acrt),
                n=int(res.n_units),
            )
        )

    write_results(MODULE, "py", rows, extra={"estimator": "cgs_continuous_did"})


if __name__ == "__main__":
    main()
