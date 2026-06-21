"""StatsPAI original-data parity (Python side) -- Module 11.

E-value sensitivity analysis (VanderWeele & Ding, *Ann Intern Med* 2017)
demonstrated with ``sp.evalue`` on the REAL NHEFS data bundled in
``sp.datasets.nhefs()``.

The E-value answers: *how strong would an unmeasured confounder have to
be* -- on the risk-ratio scale, with both treatment and outcome -- *to
fully explain away an observed association?*  For a risk ratio RR >= 1 it
has the deterministic closed form

    E = RR + sqrt(RR * (RR - 1))                                   (VD-2017)

(and the same formula applied to 1/RR when RR < 1).  Because it is a
fixed function of the input ratio, ``sp.evalue`` must reproduce the
``EValue`` R package and that closed form to ~1e-10 -- this module pins
that.

PRIMARY (clean ratio measure):  the IP-weighted 10-year mortality risk
ratio of quitting smoking (``qsmk``) vs not, on the FULL NHEFS
(n=1629).  Stabilized IP weights from the canonical book confounder
model give RR_adj ~ 0.99 (death risk is essentially unchanged after
adjustment -- the well-known "null after confounding control" result);
its CI straddles 1, so the E-value collapses to 1.00.  We therefore ALSO
report the *crude* mortality RR (~1.33, before adjustment, CI excludes
1), whose E-value (~1.98) is the pedagogically interesting number and
the one whose closed form we anchor.

SECONDARY (continuous outcome):  the Chapter-12 weight effect (book 3.4
kg; our IPW point 3.44 kg).  ``sp.evalue`` natively accepts a continuous
effect via ``measure='SMD'``, implementing the VanderWeele-Ding
standardized-mean-difference approximation ``RR_approx = exp(0.91 * d)``
with ``d = effect / SD(outcome)``.  We pass the standardized effect and
confirm it equals both the manual ``exp(0.91 d)`` route and
``EValue::evalues.MD``.

Anchors per statistic: StatsPAI (sp.evalue) | R gold (EValue package +
hand-rolled stabilized IPW) on the same CSV bytes | the deterministic
closed form / published convention.
"""

from __future__ import annotations

import math

import numpy as np
import statsmodels.api as sm
import statspai as sp

from _common import OrigRecord, write_results
from _nhefs import book_design

MODULE = "11_nhefs_evalue"

# VanderWeele-Ding standardized-mean-difference -> approximate risk ratio.
VD_SMD_COEF = 0.91


def _stabilized_ipw_rr(dd, covs, y: str, treat: str) -> tuple[float, float, float]:
    """Hand-rolled stabilized-IPW risk ratio (book Programs 12.3-12.4 on a
    binary outcome): logistic propensity, stabilized weights, IP-weighted
    risk in each arm, return (RR, risk1, risk0)."""
    X = sm.add_constant(dd[covs].astype(float).values)
    A = dd[treat].astype(float).values
    Y = dd[y].astype(float).values
    den = sm.Logit(A, X).fit(disp=0)
    p_den = den.predict(X)
    p_num = A.mean()  # stabilizing numerator P(A=1)
    sw = np.where(A == 1, p_num / p_den, (1 - p_num) / (1 - p_den))
    r1 = np.sum(sw * A * Y) / np.sum(sw * A)
    r0 = np.sum(sw * (1 - A) * Y) / np.sum(sw * (1 - A))
    return r1 / r0, r1, r0


def _boot_rr_ci(dd, covs, y, treat, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(dd))
    ests = np.empty(n_boot)
    for b in range(n_boot):
        s = dd.iloc[rng.choice(idx, size=len(idx), replace=True)]
        try:
            ests[b] = _stabilized_ipw_rr(s, covs, y, treat)[0]
        except Exception:
            ests[b] = np.nan
    ests = ests[~np.isnan(ests)]
    lo, hi = np.percentile(ests, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> None:
    full = sp.datasets.nhefs()  # n=1629 -- death is complete
    full.to_csv(__import__("_common").DATA_DIR / f"{MODULE}.csv", index=False)
    n_full = len(full)

    dd_full, covs = book_design(full)
    A = dd_full["qsmk"].astype(float).values
    Y = dd_full["death"].astype(float).values

    # ---- PRIMARY: mortality risk ratios + E-values --------------------
    # Crude (unadjusted) RR -- CI excludes 1; the interesting E-value.
    risk1_c, risk0_c = Y[A == 1].mean(), Y[A == 0].mean()
    rr_crude = risk1_c / risk0_c
    # Wald CI on log-RR (closed form -> identical in R).
    n1, n0 = (A == 1).sum(), (A == 0).sum()
    se_log = math.sqrt((1 - risk1_c) / (risk1_c * n1) + (1 - risk0_c) / (risk0_c * n0))
    z = sp_norm_ppf(1 - 0.05 / 2)
    lo_c = rr_crude * math.exp(-z * se_log)
    hi_c = rr_crude * math.exp(z * se_log)

    # IP-weighted (adjusted) RR -- bootstrap CI (straddles 1 here).
    rr_adj, r1_adj, r0_adj = _stabilized_ipw_rr(dd_full, covs, "death", "qsmk")
    lo_a, hi_a = _boot_rr_ci(dd_full, covs, "death", "qsmk", n_boot=1000, seed=42)

    ev_crude = sp.evalue(estimate=rr_crude, ci=(lo_c, hi_c), measure="RR")
    ev_adj = sp.evalue(estimate=rr_adj, ci=(lo_a, hi_a), measure="RR")

    # Closed-form check for the crude point E-value.
    rr_used = rr_crude if rr_crude >= 1 else 1 / rr_crude
    closed_crude = rr_used + math.sqrt(rr_used * (rr_used - 1))

    # ---- SECONDARY: Ch12 weight effect (continuous) -------------------
    cc = sp.datasets.nhefs(complete_case=True)
    dd_cc, covs_cc = book_design(cc)
    wt_res = sp.ipw(
        dd_cc,
        y="wt82_71",
        treat="qsmk",
        covariates=covs_cc,
        estimand="ATE",
        seed=42,
        n_bootstrap=500,
    )
    effect = float(wt_res.estimate)  # ~3.44 kg (book 3.4)
    sd_out = float(cc["wt82_71"].std(ddof=1))
    d_smd = effect / sd_out
    # CI limits standardized too (closest-to-null limit drives evalue_ci).
    lo_d = float(wt_res.ci[0]) / sd_out
    hi_d = float(wt_res.ci[1]) / sd_out

    # Native StatsPAI SMD route (implements exp(0.91 d) internally).
    ev_smd = sp.evalue(estimate=d_smd, ci=(lo_d, hi_d), measure="SMD")
    # Manual VanderWeele-Ding RR_approx, passed as a plain RR, must agree.
    rr_approx = math.exp(VD_SMD_COEF * d_smd)
    rr_approx_lo = math.exp(VD_SMD_COEF * lo_d)
    rr_approx_hi = math.exp(VD_SMD_COEF * hi_d)
    ev_smd_manual = sp.evalue(
        estimate=rr_approx, ci=(rr_approx_lo, rr_approx_hi), measure="RR"
    )

    rows = [
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="evalue_crude_rr_point",
            estimate=float(ev_crude["evalue_estimate"]),
            se=None,
            n=int(n_full),
            published=closed_crude,
            citation="VanderWeele-Ding 2017 E = RR + sqrt(RR(RR-1)) (crude mortality RR)",
            extra={
                "rr": rr_crude,
                "rr_ci": [lo_c, hi_c],
                "evalue_ci": float(ev_crude["evalue_ci"]),
                "risk1": risk1_c,
                "risk0": risk0_c,
            },
        ),
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="evalue_ipw_rr_point",
            estimate=float(ev_adj["evalue_estimate"]),
            se=None,
            n=int(n_full),
            published=1.0,
            citation="VanderWeele-Ding 2017 E-value, IP-weighted mortality RR (~null after adjustment)",
            extra={
                "rr": rr_adj,
                "rr_ci": [lo_a, hi_a],
                "evalue_ci": float(ev_adj["evalue_ci"]),
                "risk1": float(r1_adj),
                "risk0": float(r0_adj),
            },
        ),
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="evalue_smd_point",
            estimate=float(ev_smd["evalue_estimate"]),
            se=None,
            n=int(len(cc)),
            published=float(ev_smd_manual["evalue_estimate"]),
            citation="VanderWeele-Ding SMD approx exp(0.91*d) on Ch12 weight effect",
            extra={
                "effect_kg": effect,
                "sd_outcome": sd_out,
                "d": d_smd,
                "rr_approx": float(ev_smd["rr_estimate"]),
                "rr_approx_manual": rr_approx,
                "evalue_ci": float(ev_smd["evalue_ci"]),
                "evalue_manual_point": float(ev_smd_manual["evalue_estimate"]),
            },
        ),
    ]

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "data_source": "sp.datasets.nhefs (NHEFS / What If)",
            "n_full": int(n_full),
            "n_complete": int(len(cc)),
            "vd_smd_coef": VD_SMD_COEF,
            "closed_form": "E = RR + sqrt(RR*(RR-1))",
            "smd_native_eq_manual": math.isclose(
                ev_smd["evalue_estimate"],
                ev_smd_manual["evalue_estimate"],
                rel_tol=1e-9,
            ),
        },
    )

    print(
        f"[{MODULE}] crude RR={rr_crude:.4f} CI({lo_c:.3f},{hi_c:.3f}) "
        f"E={ev_crude['evalue_estimate']:.4f} (closed {closed_crude:.4f}) "
        f"E_ci={ev_crude['evalue_ci']:.4f} || "
        f"ipw RR={rr_adj:.4f} CI({lo_a:.3f},{hi_a:.3f}) "
        f"E={ev_adj['evalue_estimate']:.4f} || "
        f"SMD d={d_smd:.4f} RR_approx={rr_approx:.4f} "
        f"E={ev_smd['evalue_estimate']:.4f} (manual {ev_smd_manual['evalue_estimate']:.4f})"
    )


def sp_norm_ppf(q: float) -> float:
    """Inverse standard-normal CDF (avoid importing scipy explicitly)."""
    from statistics import NormalDist

    return NormalDist().inv_cdf(q)


if __name__ == "__main__":
    main()
