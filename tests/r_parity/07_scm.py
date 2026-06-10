"""StatsPAI classical SCM parity (Python side) -- Module 07.

Runs the native Python classical SCM
(``sp.synth(method='classic', backend='native')``) on the Basque-Country
replica and emits the average post-treatment gap, the pre-treatment RMSE,
and the donor weights. The companion 07_scm.R uses ``Synth::synth`` on
the same CSV with the standard pre-treatment-outcomes-only predictor
spec.

Tier: reference-disagreement disclosure. Under the ADH special-predictor
specification, the native result tracks Stata ``synth`` closely while R
``Synth`` and Stata differ on the same CSV bytes. The row is therefore
not sold as deterministic R/Synth equality. Native correctness is
separately certified on the uniquely identified SCM DGP in module
52_scm_unique; users who need exact R numbers can call the optional
``backend='synth'`` bridge.
"""

from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results

MODULE = "07_scm"


def main() -> None:
    df = sp.datasets.basque_terrorism()
    dump_csv(df, MODULE)

    # Match R Synth's exact ADH(2010) specification: every pre-treatment
    # year as its own outcome special-predictor with nested V-W
    # optimisation. Under this common specification the native Python
    # solver tracks Stata synth to rel ~4.2e-4 on the replica, while
    # R Synth and Stata differ by ~2.3e-2. The residual is therefore a
    # reference-disagreement/local-optimum disclosure, not a hidden native
    # solver failure.
    pre_years = list(range(1955, 1970))
    fit = sp.synth(
        df,
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        method="classic",
        backend="native",
        special_predictors=[("gdppc", yr, "mean") for yr in pre_years],
        v_method="nested",
        placebo=False,
    )

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="avg_post_gap",
            estimate=float(fit.estimate),
            se=float(fit.se),
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE,
            side="py",
            statistic="pre_treatment_rmse",
            estimate=float(fit.model_info["pre_treatment_rmse"]),
            n=int(len(df)),
        ),
    ]

    # weights is a DataFrame with columns ['unit', 'weight'] listing
    # the active donors only; absent donors carry implicit weight 0.
    # Emit a row for every donor in the donor pool so the comparator
    # can match against the R-side weight vector.
    weights_df = fit.model_info["weights"]
    donor_pool = sorted(
        df.loc[df["region"] != "Basque Country", "region"].unique().tolist()
    )
    weight_map = dict(zip(weights_df["unit"], weights_df["weight"]))
    for unit in donor_pool:
        rows.append(
            ParityRecord(
                module=MODULE,
                side="py",
                statistic=f"weight_{unit}",
                estimate=float(weight_map.get(unit, 0.0)),
                n=int(len(df)),
            )
        )

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "method": "classic",
            "backend": fit.model_info.get("backend", "native"),
            "validation_tier": fit.model_info.get("validation_tier"),
            "reference_backend": fit.model_info.get("reference_backend"),
            "treatment_time": 1970,
            "treated_unit": "Basque Country",
            "n_donors": int(fit.model_info["n_donors"]),
            "placebo": False,
            "tier": "T4",
            "solver_best_start": fit.model_info.get("solver_best_start"),
            "solver_near_best_start_count": int(
                fit.model_info.get("solver_near_best_start_count", 0)
            ),
            "solver_near_best_weight_class_count": int(
                fit.model_info.get("solver_near_best_weight_class_count", 0)
            ),
            "solver_near_best_weight_l1_max": float(
                fit.model_info.get("solver_near_best_weight_l1_max", 0.0)
            ),
            "weight_solution_nonunique": bool(
                fit.model_info.get("weight_solution_nonunique", False)
            ),
            "native_note": (
                "Headline row uses backend='native'. The Basque donor-weight "
                "solution is not unique under the ADH special-predictor "
                "nested-V specification: deterministic multi-start diagnostics "
                "find multiple near-best donor-weight classes, native tracks "
                "Stata synth on the same CSV, and R Synth and Stata synth "
                "choose measurably different local optima. Native correctness "
                "is separately certified on a uniquely identified DGP in "
                "module 52_scm_unique; backend='synth' is available when exact "
                "R Synth numbers are required."
            ),
        },
    )


if __name__ == "__main__":
    main()
