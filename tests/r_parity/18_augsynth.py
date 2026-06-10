"""StatsPAI Augmented SCM parity (Python side) -- Module 18.

Runs the **native Python** augmented SCM (``sp.augsynth(backend='native')``)
on the Basque-Country replica (Ben-Michael, Feller & Rothstein 2021).
The companion 18_augsynth.R uses ``augsynth::augsynth`` on the same CSV.

Tier: T2 iterative parity.  The native Python path ports the same
control-mean centering, time-holdout ridge-lambda CV, SCM weights, and
ridge-augmented weight formula used by ``augsynth::augsynth`` for this
no-covariate Ridge+SCM specification.  The optional
``backend='augsynth'`` R bridge remains a migration convenience, NOT the
parity comparator.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "18_augsynth"


def main() -> None:
    df = sp.datasets.basque_terrorism()
    dump_csv(df, MODULE)

    fit = sp.augsynth(
        df,
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        backend="native",
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_augmented",
            estimate=float(fit.estimate),
            se=float(fit.se),
            n=int(len(df)),
        ),
    ]
    _pre = fit.model_info.get("pre_treatment_rmse",
                              fit.model_info.get("pre_rmspe"))
    if _pre is not None:
        rows.append(
            ParityRecord(
                module=MODULE, side="py", statistic="pre_rmspe",
                estimate=float(_pre),
                n=int(len(df)),
            )
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "augmented",
            "backend": fit.model_info.get("backend", "native"),
            "n_donors": int(fit.model_info.get("n_donors", 0)),
            "tier": "T2",
            "reference_backend": "augsynth",
            "native_parity_note": (
                "Headline row is the NATIVE Python augmented SCM "
                "(backend='native'). It ports augsynth's centered pre-outcome "
                "SCM plus ridge-augmented weight formula and matches "
                "augsynth::augsynth on the Basque fixture within iterative "
                "solver tolerance. The optional backend='augsynth' R bridge "
                "is a migration convenience, not the parity comparator."
            ),
        },
    )


if __name__ == "__main__":
    main()
