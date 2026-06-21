"""StatsPAI Generalized SCM parity (Python side) -- Module 19.

Runs the **native Python** generalized SCM (``sp.gsynth(backend='native')``)
on the Basque-Country replica (Xu 2017). The companion 19_gsynth.R uses
``gsynth::gsynth`` on the same CSV.

Tier: T2 native parity. The Python implementation estimates the
``gsynth``/``fect`` two-way fixed-effect factor convention on the full
never-treated control panel, then projects the treated unit on the
pre-treatment periods. The optional ``backend='gsynth'`` R bridge remains
a migration convenience, NOT the parity comparator.
"""
from __future__ import annotations

import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "19_gsynth"


def main() -> None:
    df = sp.datasets.basque_terrorism()
    df = df.copy()
    df["treated_indicator"] = (
        (df["region"] == "Basque Country") & (df["year"] >= 1970)
    ).astype(int)
    dump_csv(df, MODULE)

    fit = sp.gsynth(
        df,
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        backend="native",
        seed=PARITY_SEED,
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_gsynth",
            estimate=float(fit.estimate),
            se=float(fit.se) if fit.se is not None else None,
            n=int(len(df)),
        ),
    ]
    _nf = fit.model_info.get("n_factors")
    if _nf is not None:
        rows.append(ParityRecord(
            module=MODULE, side="py", statistic="n_factors",
            estimate=float(_nf), n=int(len(df))))
    _pre = fit.model_info.get("pre_treatment_rmse",
                              fit.model_info.get("pre_rmse"))
    if _pre is not None:
        rows.append(ParityRecord(
            module=MODULE, side="py", statistic="pre_rmse",
            estimate=float(_pre), n=int(len(df))))

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "gsynth (Xu 2017)",
            "backend": fit.model_info.get("backend", "native"),
            "n_donors": int(fit.model_info.get("n_donors", 0)),
            "tier": "T2",
            "reference_backend": "gsynth",
            "native_parity_note": (
                "Headline row is the NATIVE Python generalized SCM "
                "(backend='native'). It ports gsynth/fect's two-way FE "
                "factor specification on the full never-treated control panel "
                "and matches gsynth::gsynth on the Basque fixture within "
                "machine tolerance. The optional backend='gsynth' R bridge "
                "is a migration convenience, not the parity comparator."
            ),
        },
    )


if __name__ == "__main__":
    main()
