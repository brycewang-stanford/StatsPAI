"""StatsPAI PSM 1:1 NN matching parity (Python side) -- Module 11.

Runs sp.psm on the NSW-DW replica with logistic propensity score and
1:1 nearest-neighbour matching with replacement (the MatchIt /
psmatch2 default convention). Tolerance: rel < 1e-6 on the ATT and on
the one SE row that joins (``se_teffects_ai``, the Abadie-Imbens 2016
estimated-score variance that Stata ``teffects psmatch`` reports, run
here as ``se_method='abadie_imbens_2016'``); the other SE rows are
side-specific diagnostics because matching packages use different
variance conventions -- see the ``se_reference`` note in ``extra``.

PSM has many small implementation choices (caliper, ties, replacement,
bias correction); the goal is to verify that the StatsPAI ATT is
within the same neighbourhood as MatchIt::matchit on identical data
with matched options, not bit-equal recovery.
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "11_psm"


def main() -> None:
    df = sp.datasets.nsw_dw()
    dump_csv(df, MODULE)

    fit = sp.psm(
        df,
        y="re78",
        d="treat",
        X=["age", "education", "black", "hispanic", "married", "re74", "re75"],
        method="nn",
    )
    # Same matching, Stata teffects psmatch's variance: the Abadie-Imbens
    # (2016) correction for the estimated logit score. ai_matches=1 is
    # teffects' default vce(robust, nn(2)) (the unit itself counts there).
    fit_teffects = sp.psm(
        df,
        y="re78",
        d="treat",
        X=["age", "education", "black", "hispanic", "married", "re74", "re75"],
        method="nn",
        se_method="abadie_imbens_2016",
    )
    assert abs(fit_teffects.estimate - fit.estimate) == 0.0
    ai2016 = fit_teffects.model_info["ai2016_components"]

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_psm",
            estimate=float(fit.estimate),
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="se_abadie_imbens",
            estimate=float(fit.se),
            n=int(len(df)),
        ),
        # Point-valued like the Stata row it joins (estimate = the SE,
        # se = None), so compare.py scores it under rel_est.
        ParityRecord(
            module=MODULE, side="py", statistic="se_teffects_ai",
            estimate=float(fit_teffects.se),
            n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="n_treated",
            estimate=float(fit.model_info["n_treated"]), n=int(len(df)),
        ),
        ParityRecord(
            module=MODULE, side="py", statistic="n_control_full",
            estimate=float(fit.model_info["n_control"]), n=int(len(df)),
        ),
    ]

    extra = {
        "distance": fit.model_info["distance"],
        "method": fit.model_info["method"],
        "replace": fit.model_info["replace"],
        "bias_correction": fit.model_info["bias_correction"],
        "count_note": (
            "n_control_full is the full untreated sample; "
            "MatchIt's matched-data control count is a "
            "post-matching support diagnostic and is not "
            "the same estimand when matching with "
            "replacement."
        ),
        "se_teffects_ai_method": (
            "sp.psm(se_method='abadie_imbens_2016', "
            "ai_matches=1) == Stata teffects psmatch, atet "
            "vce(robust, nn(2))"
        ),
        "se_teffects_ai_components": {
            k: float(ai2016[k])
            for k in ("base_se", "c_V_c", "d_V_d")
        },
        "se_reference": (
            "The att_psm row compares point estimates only. "
            "se_teffects_ai is the one SE row that joins: "
            "Stata teffects psmatch reports the Abadie-"
            "Imbens (2016) variance for matching on an "
            "*estimated* propensity score -- the AI-2006 "
            "population-ATT variance sigma^2 minus the "
            "estimated-score term c'V_gamma c plus "
            "d'V_gamma d, with the within-arm conditional "
            "variance taken over the unit plus its nearest "
            "same-arm unit (vce(robust, nn(2)), ties "
            "included) -- and se_method='abadie_imbens_2016' "
            "rebuilds it from StatsPAI's matched frame, "
            "within-arm sigma^2, logit design and observed-"
            "information V_gamma (see "
            "se_teffects_ai_components: base_se 673.42, "
            "c'Vc 74905, d'Vd 8036, so the score term is "
            "negative and the corrected SE 621.79 is below "
            "the base). The remaining rel 7e-8 is Stata's ML "
            "stopping rule for the logit (its e(bps)/e(Vps) "
            "plugged into the same formula give e(V) at "
            "1e-9). StatsPAI's default for nearest-neighbour "
            "matching stays the Abadie-Imbens (2006, eq. 14) "
            "sample-ATT SE that Stata psmatch2 reports under "
            "ai(J) ties (se_abadie_imbens = 640.41, 3.0% above): "
            "different estimand (sample vs population ATT) "
            "and no estimated-score term, a documented "
            "convention, not a gap. MatchIt documents no "
            "canonical analytic SE for matching with "
            "replacement, so the R fixture records a "
            "weighted-lm-on-matched-data diagnostic "
            "(se_matchit_lm) that never joins."
        )
    }
    write_results(MODULE, "py", rows, extra=extra)


if __name__ == "__main__":
    main()
