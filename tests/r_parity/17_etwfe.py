"""StatsPAI Wooldridge ETWFE parity (Python side) -- Module 17.

Runs sp.etwfe on the mpdta replica. The companion 17_etwfe.R uses
etwfe::etwfe, the canonical R port of Wooldridge's extended
two-way fixed-effects estimator.

Three aggregations are pinned, not one:

* ``att_etwfe`` -- the treated-observation-weighted simple ATT under the
  default not-yet-treated comparison group (R ``emfx(type='simple')``).
* ``att_group_notyet_<g>`` -- the per-cohort ATT under the same fit
  (R ``emfx(type='group')``).
* ``att_etwfe_never`` / ``att_group_never_<g>`` -- the same two
  aggregations under ``cgroup='nevertreated'``, which is the design
  ``sp.wooldridge_did`` estimates directly.

The group rows exist because the simple row alone did not constrain them:
through 1.26.0 StatsPAI read the cohort-level ATTs off a *separate*,
unsaturated cohort x post regression, so ``att_etwfe`` matched R to 1e-9
while every cohort ATT was off by up to 37%. A headline row cannot police
the aggregation beneath it -- pin both.

Tolerance: rel_est < 1e-6, rel_se < 1e-3 (registered for 17_etwfe).
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, dump_csv, write_results


MODULE = "17_etwfe"


def main() -> None:
    df = sp.datasets.mpdta()
    dump_csv(df, MODULE)

    kw = dict(
        y="lemp",
        group="countyreal",
        time="year",
        first_treat="first_treat",
        cluster="countyreal",
    )

    # --- not-yet-treated comparison group (R etwfe default) -------------
    fit = sp.etwfe(df, panel=False, **kw)
    simple = sp.etwfe_emfx(fit, type="simple", weighting="treated")

    rows: list[ParityRecord] = [
        ParityRecord(
            module=MODULE, side="py", statistic="att_etwfe",
            estimate=float(simple.estimate),
            se=float(simple.se),
            ci_lo=float(simple.ci[0]) if simple.ci is not None else None,
            ci_hi=float(simple.ci[1]) if simple.ci is not None else None,
            n=int(len(df)),
        )
    ]
    for row in fit.detail.itertuples():
        rows.append(
            ParityRecord(
                module=MODULE, side="py",
                statistic=f"att_group_notyet_{int(row.cohort)}",
                estimate=float(row.att), se=float(row.se), n=int(len(df)),
            )
        )

    # --- never-treated comparison group --------------------------------
    # The per-cohort rows are taken from sp.wooldridge_did itself, not from
    # sp.etwfe, so that this module *exercises* wooldridge_did rather than
    # asserting an alias to it. That assertion is what went wrong before
    # 1.27.0: the registry credited sp.wooldridge_did with this module while
    # nothing here ever called it. See _parity_taxonomy.REFUTED_ALIASES.
    never = sp.etwfe(df, cgroup="nevertreated", **kw)
    direct = sp.wooldridge_did(data=df, **kw)
    rows.append(
        ParityRecord(
            module=MODULE, side="py", statistic="att_etwfe_never",
            estimate=float(never.estimate), se=float(never.se),
            ci_lo=float(never.ci[0]) if never.ci is not None else None,
            ci_hi=float(never.ci[1]) if never.ci is not None else None,
            n=int(len(df)),
        )
    )
    for row in direct.detail.itertuples():
        rows.append(
            ParityRecord(
                module=MODULE, side="py",
                statistic=f"att_group_never_{int(row.cohort)}",
                estimate=float(row.att), se=float(row.se), n=int(len(df)),
            )
        )

    write_results(
        MODULE, "py", rows,
        extra={
            "method": "Wooldridge ETWFE + emfx simple / group aggregations",
            "n_cohorts": int(fit.model_info["n_cohorts"]),
            "aggregation_note": (
                "att_etwfe uses the R etwfe default no-ivar fixed-effect "
                "structure (StatsPAI panel=False) and sp.etwfe_emfx("
                "weighting='treated'), which averages cohort-time marginal "
                "effects over treated post-period observations like "
                "etwfe::emfx(type='simple'). Point estimates and clustered "
                "delta-method SEs match the R reference. The att_group_* rows "
                "are the "
                "per-cohort ATTs from the same fits, matching "
                "etwfe::emfx(type='group'). The never-treated rows compare "
                "against R etwfe(cgroup='never', ivar=countyreal) because "
                "sp.etwfe(cgroup='nevertreated') absorbs unit fixed effects "
                "by two-way demeaning; point estimates are invariant to the "
                "ivar choice, the clustered SE is not."
            ),
            "se_convention_note": (
                "Both sides now count the same K in the CR1 finite-sample "
                "factor: unit effects are nested in the countyreal cluster "
                "and dropped, the period effects are not and are counted, "
                "which is fixest's ssc(fixef.K='nested') and reghdfe's "
                "default (StatsPAI: did._core.fe_dof_not_nested). Through "
                "1.26.0 sp.wooldridge_did counted only the explicit design "
                "columns, which put these SE rows near 8e-4 relative; they "
                "are now 1.3e-8 to 5.5e-6. That entire residual is R's, not "
                "ours: marginaleffects differentiates the emfx aggregation "
                "numerically with a forward difference, and re-running "
                "emfx(type='group', numderiv='richardson') moves R by "
                "exactly the observed gap (1.31e-8 / 5.48e-6 / 2.30e-6 for "
                "g=2004/2006/2007) onto 0.0117881770227968 / "
                "0.0131899740310652 / 0.0138580376110556 -- which our "
                "analytic delta-method SEs reproduce to 2.2e-10 / 2.5e-9 / "
                "1.3e-9. Stata jwdid, estat group -- an independent second "
                "reference that differentiates analytically -- agrees with "
                "our three SEs to all 16 committed digits, so R's default is "
                "the outlier and the mechanism is triangulated rather than "
                "merely plausible. The committed golden keeps the "
                "marginaleffects default, because that is the number a user "
                "running the canonical reference actually sees. The "
                "att_etwfe* rows deliberately do NOT take the same nested-K "
                "treatment: sp.etwfe's simple-ATT branch already agrees with "
                "Stata jwdid, estat simple to 7e-15, and applying "
                "fe_dof_not_nested there would scale its SE by 1.001 and "
                "move it 1e-3 away from both references. Measured, not "
                "assumed -- the two branches carry different explicit design "
                "column counts, so the same CR1 factor needs a different K."
            ),
        },
    )


if __name__ == "__main__":
    main()
