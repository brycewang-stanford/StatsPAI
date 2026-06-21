"""StatsPAI original-data parity (Python side) -- Module 09.

Reproduces Hernán & Robins, *Causal Inference: What If*, **Chapter 15**
(Outcome regression and propensity scores) on the REAL NHEFS data
bundled in ``sp.datasets.nhefs()``.  Estimand: the average causal effect
of quitting smoking (``qsmk``) on 10-year weight change (``wt82_71``, kg).

Two demonstrations on the same complete-case sample (n=1566):

  (a) **Outcome regression with effect modification** (Program 15.1).
      The linear model ``wt82_71 ~ qsmk + smokeintensity +
      qsmk:smokeintensity + <canonical confounders>`` fit by OLS via
      ``sp.regress``.  The book reports the qsmk *main* coefficient
      2.56 and the interaction 0.0467, so the qsmk effect at
      ``smokeintensity=5`` is 2.79 kg and at ``smokeintensity=40`` is
      4.43 kg.  (The raw main coefficient 2.56 is NOT the marginal
      effect -- it is the conditional effect among never-smokers /
      ``smokeintensity=0``.)

  (b) **Propensity-score adjustment** (Programs 15.2-15.4).
      A logistic propensity score P(qsmk=1|L) on the canonical
      confounder set, then the ATE estimated two ways:
        (i)  PS included as a continuous covariate in the outcome
             model ``wt82_71 ~ qsmk + ps`` -- book Program 15.3, qsmk
             effect ~3.5 kg;
        (ii) PS-decile stratification (cut PS into ten strata, regress
             ``wt82_71 ~ qsmk + C(psdecile)``) -- book Program 15.4,
             ~3.5 kg.
      For cross-reference we also report the ``sp.g_computation``
      standardized marginal ATE, which targets the same estimand.

Compares against:
  (a) the book's published numbers (Chapter 15 programs);
  (b) the R-side base-R reference run on the same CSV bytes
      (``09_nhefs_ch15_outcome.R``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import statspai as sp

from _common import OrigRecord, write_results
from _nhefs import book_design, dump_csv

MODULE = "09_nhefs_ch15_outcome"

# Canonical confounder linear predictor (matches _nhefs.book_design /
# R_CONF_FORMULA): categoricals as indicators, continuous with quadratics.
CONF_RHS = (
    "C(sex) + C(race) + C(education) + C(exercise) + C(active) "
    "+ age + I(age**2) + smokeintensity + I(smokeintensity**2) "
    "+ smokeyrs + I(smokeyrs**2) + wt71 + I(wt71**2)"
)


def main() -> None:
    df = dump_csv(MODULE)  # n=1566 complete case; identical bytes for R
    n = len(df)

    # ------------------------------------------------------------------
    # (a) Outcome regression with effect modification -- Program 15.1.
    # ------------------------------------------------------------------
    # qsmk + smokeintensity + qsmk:smokeintensity, then the canonical
    # confounders (smokeintensity already enters linearly; its square
    # comes from the confounder block).
    a_formula = (
        "wt82_71 ~ qsmk + smokeintensity + qsmk:smokeintensity "
        "+ C(sex) + C(race) + C(education) + C(exercise) + C(active) "
        "+ age + I(age**2) + I(smokeintensity**2) "
        "+ smokeyrs + I(smokeyrs**2) + wt71 + I(wt71**2)"
    )
    a_res = sp.regress(a_formula, data=df)
    b_qsmk = float(a_res.params["qsmk"])
    se_qsmk = float(a_res.std_errors["qsmk"])
    b_int = float(a_res.params["qsmk:smokeintensity"])
    se_int = float(a_res.std_errors["qsmk:smokeintensity"])
    eff_5 = b_qsmk + 5.0 * b_int
    eff_40 = b_qsmk + 40.0 * b_int

    # ------------------------------------------------------------------
    # (b) Propensity-score adjustment -- Programs 15.2-15.4.
    # ------------------------------------------------------------------
    # PS model: logistic P(qsmk=1 | L) on the canonical confounder set.
    ps_formula = f"qsmk ~ {CONF_RHS}"
    psmod = smf.logit(ps_formula, data=df).fit(disp=0)
    df = df.copy()
    df["ps"] = psmod.predict(df)

    # (b)(i) PS as a continuous covariate in the outcome model (Prog 15.3).
    psi_res = sp.regress("wt82_71 ~ qsmk + ps", data=df)
    b_psi = float(psi_res.params["qsmk"])
    se_psi = float(psi_res.std_errors["qsmk"])
    ci_psi = psi_res.conf_int().loc["qsmk"].to_numpy(dtype=float)

    # (b)(ii) PS-decile stratification (Prog 15.4): ten PS strata as FE.
    df["psdecile"] = pd.qcut(df["ps"], 10, labels=False)
    psd_res = sp.regress("wt82_71 ~ qsmk + C(psdecile)", data=df)
    b_psd = float(psd_res.params["qsmk"])
    se_psd = float(psd_res.std_errors["qsmk"])
    ci_psd = psd_res.conf_int().loc["qsmk"].to_numpy(dtype=float)

    # Cross-reference: standardized marginal ATE via g-computation.
    dd, covs = book_design(df)
    gc = sp.g_computation(
        dd,
        y="wt82_71",
        treat="qsmk",
        covariates=covs,
        estimand="ATE",
        seed=42,
        n_boot=500,
    )

    rows = [
        # (a) Program 15.1
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="om_qsmk_main_coef",
            estimate=b_qsmk,
            se=se_qsmk,
            n=n,
            published=2.56,
            citation="Hernán-Robins, What If Program 15.1 (qsmk main coef)",
            extra={"note": "conditional effect at smokeintensity=0; NOT marginal"},
        ),
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="om_qsmk_x_smkint",
            estimate=b_int,
            se=se_int,
            n=n,
            published=0.0467,
            citation="Hernán-Robins, What If Program 15.1 (qsmk:smokeintensity)",
        ),
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="om_effect_smkint5",
            estimate=float(eff_5),
            se=None,
            n=n,
            published=2.79,
            citation="Hernán-Robins, What If Program 15.1 (effect at smokeintensity=5)",
        ),
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="om_effect_smkint40",
            estimate=float(eff_40),
            se=None,
            n=n,
            published=4.43,
            citation="Hernán-Robins, What If Program 15.1 (effect at smokeintensity=40)",
        ),
        # (b)(i) Program 15.3
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="ps_in_outcome_ate",
            estimate=b_psi,
            se=se_psi,
            n=n,
            published=3.5,
            citation="Hernán-Robins, What If Program 15.3 (PS as covariate)",
        ),
        # (b)(ii) Program 15.4
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="ps_decile_ate",
            estimate=b_psd,
            se=se_psd,
            n=n,
            published=3.5,
            citation="Hernán-Robins, What If Program 15.4 (PS-decile stratification)",
        ),
        # cross-reference: standardized marginal ATE
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="gcomp_std_ate",
            estimate=float(gc.estimate),
            se=float(gc.se),
            n=n,
            published=3.5,
            citation="Hernán-Robins, What If Ch.15 (standardized marginal ATE, cross-ref)",
        ),
    ]

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "data_source": "sp.datasets.nhefs (NHEFS / What If)",
            "n_obs": n,
            "ps_min": float(df["ps"].min()),
            "ps_max": float(df["ps"].max()),
            "ps_mean": float(df["ps"].mean()),
            "ci_ps_in_outcome": [float(ci_psi[0]), float(ci_psi[1])],
            "ci_ps_decile": [float(ci_psd[0]), float(ci_psd[1])],
            "ci_gcomp": [float(gc.ci[0]), float(gc.ci[1])],
        },
    )
    print(
        f"[{MODULE}] (a) qsmk_main={b_qsmk:.4f} (book 2.56) "
        f"int={b_int:.4f} (book 0.0467) "
        f"eff@5={eff_5:.3f} (book 2.79) eff@40={eff_40:.3f} (book 4.43)  "
        f"(b) ps_in_outcome={b_psi:.4f} ps_decile={b_psd:.4f} "
        f"gcomp={gc.estimate:.4f} (book ~3.5)"
    )


if __name__ == "__main__":
    main()
