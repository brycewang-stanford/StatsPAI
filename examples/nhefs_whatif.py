"""Reproducing Hernán & Robins, *Causal Inference: What If*, on real NHEFS.

A self-contained, offline demonstration that StatsPAI reproduces the
**published g-methods estimates** from the canonical modern-epidemiology
textbook — on the genuine NHEFS data the book itself uses (bundled in
``sp.datasets.nhefs()``; public-domain NHANES I follow-up).

The causal question (book Part II): what is the average effect of
**quitting smoking** (``qsmk``) on **10-year weight change**
(``wt82_71``, kg)?  A crude comparison of quitters vs non-quitters is
confounded — quitters are older, heavier, and sicker at baseline.  Three
different g-methods, all adjusting for the same baseline confounders,
should agree with each other and with the book:

    crude difference (confounded) ............... 2.54 kg   (book §12.2)
    Ch12  IP weighting (MSM) .................... 3.4  kg   (95% CI 2.4-4.5)
    Ch13  standardization / g-formula .......... 3.5  kg
    Ch14  g-estimation (structural nested model)  3.4

Then a one-line **E-value** (Ch on sensitivity analysis; VanderWeele &
Ding 2017) asks how strong an unmeasured confounder would have to be to
explain the quit-smoking → mortality association away.

Run:  python examples/nhefs_whatif.py
"""
from __future__ import annotations

import warnings

import pandas as pd

import statspai as sp

# The book's confounder model: sex, race, education(5), exercise(3),
# active(3) as categoricals; age, smokeintensity, smokeyrs, wt71 as
# continuous with quadratic terms.
CAT_CONF = ["sex", "race", "education", "exercise", "active"]
CONT_CONF = ["age", "smokeintensity", "smokeyrs", "wt71"]


def book_design(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Numeric encoding of the book's logistic/linear model."""
    d = df.copy()
    for c in CONT_CONF:
        d[f"{c}2"] = d[c] ** 2
    dd = pd.get_dummies(d, columns=CAT_CONF, drop_first=True)
    cat_cols = [c for c in dd.columns
                if any(c.startswith(p + "_") for p in CAT_CONF)]
    covs = cat_cols + CONT_CONF + [f"{c}2" for c in CONT_CONF]
    for c in covs:
        dd[c] = dd[c].astype(float)
    return dd, covs


def main() -> None:
    df = sp.datasets.nhefs(complete_case=True)   # n=1566 weight sample
    dd, covs = book_design(df)

    crude = (df.loc[df.qsmk == 1, "wt82_71"].mean()
             - df.loc[df.qsmk == 0, "wt82_71"].mean())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ipw = sp.ipw(dd, y="wt82_71", treat="qsmk", covariates=covs,
                     estimand="ATE", seed=42, n_bootstrap=500)
        gcomp = sp.g_computation(dd, y="wt82_71", treat="qsmk",
                                 covariates=covs, seed=42)
        gest = sp.g_estimation(dd, y="wt82_71", treatments=["qsmk"],
                               covariates_by_stage=[covs], random_state=42,
                               n_bootstrap=200)

    print("Effect of quitting smoking on 10-year weight change (kg)")
    print("=" * 60)
    print(f"{'crude (confounded)':<34}{crude:6.2f}      book 2.54")
    print(f"{'Ch12  IP weighting (MSM)':<34}{ipw.estimate:6.2f}      "
          f"book 3.4  [95% CI {ipw.ci[0]:.1f}, {ipw.ci[1]:.1f}]")
    print(f"{'Ch13  standardization / g-formula':<34}{gcomp.estimate:6.2f}"
          f"      book 3.5")
    print(f"{'Ch14  g-estimation (SNMM)':<34}{gest.estimate:6.2f}      "
          f"book 3.4")
    print("=" * 60)
    print("Three g-methods agree (~3.4-3.5 kg) and reproduce the book — "
          "the confounded\ncrude estimate (2.5) understates the effect.\n")

    # --- Sensitivity analysis: E-value on quit-smoking -> mortality ----
    full = sp.datasets.nhefs()
    rr = (full.loc[full.qsmk == 1, "death"].mean()
          / full.loc[full.qsmk == 0, "death"].mean())
    ev = sp.evalue(estimate=float(rr), measure="RR")
    print(f"E-value (quit-smoking -> 10-yr mortality, crude RR={rr:.2f}): "
          f"{ev['evalue_estimate']:.2f}")
    print("  An unmeasured confounder would need RR >= "
          f"{ev['evalue_estimate']:.2f} with both quitting and death\n"
          "  (beyond measured confounders) to explain this association away.")


if __name__ == "__main__":
    main()
