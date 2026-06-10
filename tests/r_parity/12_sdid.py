"""StatsPAI Synthetic DID parity (Python side) -- Module 12.

Runs the **native Python** synthetic-DID estimator
(``sp.sdid(backend='native')``) on the california_prop99 replica
(Arkhangelsky et al. 2021). The companion 12_sdid.R uses
``synthdid::synthdid_estimate`` on the same CSV.

Tier: T2 native reference parity.  The native Python implementation now
mirrors synthdid's collapsed-form Frank-Wolfe weight solver and zeta
scaling, so the ATT matches the R reference on identical CSV bytes.  The
headline ``att_sdid`` row is point-only; backend-native placebo SEs are
reported as explicitly named diagnostic rows.
"""
from __future__ import annotations

import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "12_sdid"


def main() -> None:
    df = sp.datasets.california_prop99()
    dump_csv(df, MODULE)

    fit = sp.sdid(
        df,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        backend="native",
        seed=PARITY_SEED,
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_sdid",
            estimate=float(fit.estimate),
            se=None,
            ci_lo=None,
            ci_hi=None,
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="se_native_placebo",
            estimate=float(fit.se),
            se=None,
            ci_lo=None,
            ci_hi=None,
            n=int(len(df)),
        ),
    ]

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "sdid",
            "n_treated": int(fit.model_info["n_treated"]),
            "n_control": int(fit.model_info["n_control"]),
            "T_pre": int(fit.model_info["T_pre"]),
            "T_post": int(fit.model_info["T_post"]),
            "se_method": fit.model_info["se_method"],
            "backend": fit.model_info.get("backend", "native"),
            "validation_tier": fit.model_info.get("validation_tier"),
            "reference_backend": fit.model_info.get("reference_backend"),
            "tier": "T2",
            "native_parity_note": (
                "Headline row is the NATIVE Python SDID ATT "
                "(backend='native'). The native solver mirrors synthdid's "
                "collapsed-form Frank-Wolfe weights and zeta scaling, so the "
                "ATT is a T2 same-byte reference-parity row. Backend-native "
                "placebo SEs are reported as separately named diagnostic rows."
            ),
            "se_reference": (
                "Python records StatsPAI deterministic all-control placebo "
                "SEs, R records synthdid_se placebo SEs, and Stata records "
                "sdid placebo SEs."
            ),
        },
    )


if __name__ == "__main__":
    main()
