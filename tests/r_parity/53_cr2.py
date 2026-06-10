"""StatsPAI CR2 / CR3 cluster-robust SE parity (Python side) -- Module 53.

Companion ``53_cr2.R`` runs ``lm`` + ``clubSandwich::vcovCR`` on identical
data so the small-G cluster-robust corrections are pinned cross-language.

Module 14 already pins the CR1 (HC1) cluster-robust SE against
``sandwich::vcovCL``. This module covers the two finite-sample
corrections referees increasingly require for few-cluster inference but
that diverge across implementations in their leverage / df adjustment:

  * **CR2** (Bell & McCaffrey 2002): ``sp.cr2_se`` and
    ``clubSandwich::vcovCR(type="CR2")`` both apply the
    ``(I - H_gg)^{-1/2}`` bias correction, so they agree to machine
    precision. This is the strict headline (``rel_se < 1e-6``) and the
    only symbol the parity-README scan promotes to ``certified``.

  * **CR3**: ``sp.fast.crve(type="cr3")`` applies the same analytic
    ``(I - H_gg)^{-1}`` cluster-leverage adjustment as
    ``clubSandwich::vcovCR(type="CR3")``. The exact delete-one-cluster
    jackknife remains available separately as ``sp.cr3_jackknife_vcov``.
"""
from __future__ import annotations

import numpy as np
import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "53_cr2"
FORMULA = "lemp ~ treat + year"


def main() -> None:
    df = sp.datasets.mpdta()
    dump_csv(df, MODULE)

    fit = sp.regress(FORMULA, data=df)
    cr2 = sp.cr2_se(fit, df, cluster="countyreal")

    # CR3 analytic clubSandwich parity. Build the design matrix in the same
    # column order as fit.params (Intercept, treat, year) so the diagonal
    # of the returned vcov lines up with the coefficient names.
    X = np.column_stack(
        [np.ones(len(df)), df["treat"].to_numpy(float), df["year"].to_numpy(float)]
    )
    y = df["lemp"].to_numpy(float)
    cl = df["countyreal"].to_numpy()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    cr3_se = np.sqrt(np.diag(sp.fast.crve(X, resid, cl, type="cr3")))

    n = int(fit.data_info.get("n_obs", len(df)))
    rows: list[ParityRecord] = []
    for i, name in enumerate(fit.params.index):
        beta = float(fit.params[name])
        canonical = "(Intercept)" if name == "Intercept" else name
        rows.append(
            ParityRecord(
                module=MODULE, side="py",
                statistic=f"cr2_{canonical}",
                estimate=beta, se=float(cr2.std_errors[name]), n=n,
            )
        )
        rows.append(
            ParityRecord(
                module=MODULE, side="py",
                statistic=f"cr3_{canonical}",
                estimate=beta, se=float(cr3_se[i]), n=n,
            )
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "formula": FORMULA,
            "vcov": "CR2 (Bell-McCaffrey) + CR3 (clubSandwich analytic)",
            "cluster_var": "countyreal",
            "cr3_reference": (
                "sp.fast.crve(type='cr3') matches clubSandwich type=CR3 "
                "analytic (I-H)^-1 adjustment. Exact delete-one-cluster "
                "jackknife remains available as sp.cr3_jackknife_vcov."
            ),
        },
    )


if __name__ == "__main__":
    main()
