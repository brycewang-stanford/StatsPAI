"""StatsPAI HC2 / HC3 heteroskedasticity-robust SE parity (Python side) -- Module 55.

Companion ``55_hc2_hc3.R`` runs ``lm`` + ``sandwich::vcovHC`` on identical bytes.

Module 01 pins the HC1 heteroskedasticity-robust SE. This module pins the
two small-sample variants referees increasingly prefer for finite samples
(MacKinnon & White 1985):

  * **HC2**: leverage adjustment ``e_i / sqrt(1 - h_ii)``.
  * **HC3**: jackknife-style adjustment ``e_i / (1 - h_ii)`` (more
    conservative; approximates the delete-one jackknife).

``sp.regress(robust="hc2"/"hc3")`` and ``sandwich::vcovHC(type="HC2"/"HC3")``
implement the same MacKinnon-White adjustments, so both rows are a strict
machine-precision headline (rel_se ~ 1e-9). This certifies an additional
``sp.regress`` covariance path; it does not introduce a new certified
symbol (``regress`` is already certified via modules 01/14).
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "55_hc2_hc3"
FORMULA = "lemp ~ treat + year"


def main() -> None:
    df = sp.datasets.mpdta()
    dump_csv(df, MODULE)

    fits = {"hc2": sp.regress(FORMULA, data=df, robust="hc2"),
            "hc3": sp.regress(FORMULA, data=df, robust="hc3")}

    n = int(fits["hc2"].data_info.get("n_obs", len(df)))
    rows: list[ParityRecord] = []
    for kind, fit in fits.items():
        for name in fit.params.index:
            beta = float(fit.params[name])
            canonical = "(Intercept)" if name == "Intercept" else name
            rows.append(
                ParityRecord(
                    module=MODULE, side="py",
                    statistic=f"{kind}_{canonical}",
                    estimate=beta, se=float(fit.std_errors[name]), n=n,
                )
            )

    write_results(
        MODULE, "py", rows,
        extra={
            "formula": FORMULA,
            "vcov": "HC2 + HC3 (MacKinnon-White 1985)",
            "matches": "sandwich::vcovHC(type='HC2'/'HC3')",
        },
    )


if __name__ == "__main__":
    main()
