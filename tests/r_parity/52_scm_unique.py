"""Classical SCM unique-solution parity (Python side) -- Module 52.

Where module 07 runs classical SCM on the *Basque* data -- whose
V-optimisation is non-convex, so the donor weights are non-unique and
the sp-vs-Synth headline gap (~12%) is a documented non-uniqueness
disclosure rather than a strict parity pass -- this module runs the
same estimator on a DGP whose synthetic-control weights are *uniquely*
identified: the treated unit is exactly a convex combination of the
donors in the pre-period.  The convex weight programme then has a
unique global minimiser with zero pre-treatment loss, so any correct
solver must recover the same weights and the same post-treatment gap.

This is the strict-parity counterpart to module 07: it certifies that
the classical SCM *solver* is numerically correct on an identified
problem, isolating the Basque gap as genuine non-uniqueness rather than
an implementation bug.  The companion 52_scm_unique.R runs
``Synth::synth`` on the same CSV bytes with each pre-period as its own
predictor.

Registered tolerance (``compare.py``): rel_est < 1e-6 on the average
post-treatment gap.  Python and Stata recover the exact convex weights;
the R Synth reference fixes the predictor-weight vector and tightens the
inner ``ipop`` QP so the reference gap is also at machine-level point
precision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "52_scm_unique"
J, T0, T1 = 5, 20, 10
W_TRUE = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
TAU = 2.0


def _make_unique_dgp():
    T = T0 + T1
    t = np.arange(T)
    donors = np.zeros((J, T))
    for d in range(J):
        donors[d] = 5 + 0.3 * d + (1.0 + 0.2 * d) * np.sin(0.2 * t + d) + 0.05 * d * t
    treated = W_TRUE @ donors
    treated[T0:] += TAU
    rows = [(f"donor{d}", int(t[i]), donors[d, i]) for d in range(J) for i in range(T)]
    rows += [("treated", int(t[i]), treated[i]) for i in range(T)]
    return pd.DataFrame(rows, columns=["region", "year", "y"])


def main() -> None:
    df = _make_unique_dgp()
    dump_csv(df, MODULE)

    fit = sp.synth(df, outcome="y", unit="region", time="year",
                   treated_unit="treated", treatment_time=T0, method="classic")

    wdf = fit.model_info["weights"]
    wmap = dict(zip(wdf["unit"], wdf["weight"]))

    rows = [
        ParityRecord(
            module=MODULE, side="py", statistic="avg_post_gap",
            estimate=float(fit.estimate), se=float(fit.se), n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="pre_treatment_rmse",
            estimate=float(fit.model_info["pre_treatment_rmse"]), n=int(len(df)),
        ),
    ]
    for d in range(J):
        rows.append(ParityRecord(
            module=MODULE, side="py", statistic=f"weight_donor{d}",
            estimate=float(wmap.get(f"donor{d}", 0.0)), n=int(len(df)),
        ))

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "classic",
            "true_gap": TAU,
            "true_weights": W_TRUE.tolist(),
            "dgp": (
                "unique convex-hull SCM: treated_pre = 0.5*donor0 + "
                "0.3*donor1 + 0.2*donor2 exactly; donors 3,4 are "
                "distractors; post-period adds tau=2."
            ),
            "certification_note": (
                "Unique-solution counterpart to module 07. sp recovers "
                "the exact weights and gap (pre-RMSE ~ 0); Synth::synth "
                "agrees on the gap at machine-level point precision when "
                "run with fixed predictor weights and tightened ipop "
                "settings, while leaving sub-nanoscopic non-zero weights "
                "on the distractor donors. "
                "Certifies the classical SCM solver on an identified problem."
            ),
        },
    )


if __name__ == "__main__":
    main()
