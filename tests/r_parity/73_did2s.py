"""StatsPAI Gardner two-stage DiD parity (Python side) -- Module 73.

Runs ``sp.gardner_did`` on the mpdta replica; the companion R script runs
``did2s::did2s`` with the same two-way FE first stage on the same CSV.

Tolerance: rel < 1e-6 on the point estimate. The standard errors are a
documented convention gap rather than a parity failure -- ``did2s``
propagates first-stage estimation error into the second-stage variance
while ``sp.gardner_did``'s default ``vce='analytic'`` clusters the
stage-2 residuals only, so it lands ~18% low. ``vce='bootstrap'``
recovers R's SE to ~3% and is emitted alongside so the taxonomy can
attribute the gap.
"""

from __future__ import annotations

import warnings

import statspai as sp

from _common import ParityRecord, dump_csv, write_results

MODULE = "73_did2s"


def main() -> None:
    df = sp.datasets.mpdta()
    dump_csv(df, MODULE)

    with warnings.catch_warnings():
        # gardner_did warns that the analytic SE understates; that is
        # precisely the convention this module documents.
        warnings.simplefilter("ignore")
        fit = sp.gardner_did(
            df,
            y="lemp",
            group="countyreal",
            time="year",
            first_treat="first_treat",
        )
        boot = sp.gardner_did(
            df,
            y="lemp",
            group="countyreal",
            time="year",
            first_treat="first_treat",
            vce="bootstrap",
        )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="static_ATT",
            estimate=float(fit.estimate),
            se=float(fit.se),
            ci_lo=float(fit.ci[0]) if fit.ci is not None else None,
            ci_hi=float(fit.ci[1]) if fit.ci is not None else None,
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="static_ATT_bootstrap_se",
            estimate=float(boot.estimate),
            se=float(boot.se),
            n=int(len(df)),
        ),
    ]

    write_results(
        MODULE,
        "py",
        rows,
        extra={"estimator": "gardner_did", "vce": "analytic + bootstrap"},
    )


if __name__ == "__main__":
    main()
