"""StatsPAI original-data parity (Python side) -- Module 07.

Reproduces Hernán & Robins, *Causal Inference: What If*, **Chapter 13**
(standardization / the parametric g-formula) on the REAL NHEFS data
bundled in ``sp.datasets.nhefs()``.  Estimand: the average causal effect
of quitting smoking (``qsmk``) on 10-year weight change (``wt82_71``, kg)
obtained by *standardization* -- fit a parametric outcome model, predict
the potential outcome for every subject under ``qsmk=1`` and ``qsmk=0``,
and average the contrast.

Compares against:
  (a) the book's published numbers -- crude difference 2.54 kg
      (§12.2); standardized ATE 3.5 kg, bootstrap 95% CI ≈ (2.6, 4.5)
      (Program 13.3, outcome model wt82_71 ~ qsmk + qsmk:smokeintensity
      + confounders);
  (b) the R-side base-R standardization run on the same CSV bytes
      (``07_nhefs_ch13_gformula.R``), point estimate from the book's
      exact (interaction) model, bootstrap CI via the ``boot`` package.

CONVENTION NOTE.  StatsPAI's ``sp.g_computation`` fits an *additive*
parametric outcome model ``Q(D,X)`` -- it stacks ``[D, X]`` and predicts
by flipping only the treatment column, so it does NOT carry the book's
``qsmk:smokeintensity`` effect-modification term.  That additive
standardization gives 3.463 kg, which matches an additive Python/R gold
to machine precision and rounds to the book's 3.5.  To also anchor the
book's *exact* estimator we add a Python gold (statistic
``gformula_interaction``) that reproduces Program 13.3 verbatim
(3.517 kg) -- the R side hits the same number on the same bytes.  As a
doubly-robust cross-check, ``sp.aipw`` on the same design gives 3.551 kg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statspai as sp

from _common import OrigRecord, write_results
from _nhefs import book_design, dump_csv

MODULE = "07_nhefs_ch13_gformula"


def _book_standardization_ate(d: pd.DataFrame, covs: list[str]) -> float:
    """Book Program 13.3 standardization: fit
    ``wt82_71 ~ qsmk + qsmk:smokeintensity + confounders`` by OLS,
    predict every subject's outcome under qsmk=1 and qsmk=0, average the
    contrast.  Returns the standardized ATE (kg)."""
    work = d.copy()
    work["qsmk_smkint"] = work["qsmk"] * work["smokeintensity"]
    xcols = ["qsmk", "qsmk_smkint"] + covs
    X = sm.add_constant(work[xcols].astype(float))
    fit = sm.OLS(work["wt82_71"].astype(float), X).fit()

    d1 = work.copy()
    d1["qsmk"] = 1.0
    d1["qsmk_smkint"] = d1["smokeintensity"]
    d0 = work.copy()
    d0["qsmk"] = 0.0
    d0["qsmk_smkint"] = 0.0
    p1 = fit.predict(sm.add_constant(d1[xcols].astype(float), has_constant="add"))
    p0 = fit.predict(sm.add_constant(d0[xcols].astype(float), has_constant="add"))
    return float((p1 - p0).mean())


def _bootstrap_ci(
    d: pd.DataFrame, covs: list[str], *, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """Nonparametric bootstrap of the book standardization ATE.
    Returns (se, ci_lo, ci_hi) using the 2.5/97.5 percentile interval."""
    rng = np.random.default_rng(seed)
    n = len(d)
    idx_arr = d.reset_index(drop=True)
    ests = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ests[b] = _book_standardization_ate(idx_arr.iloc[idx], covs)
    se = float(np.std(ests, ddof=1))
    lo, hi = (float(x) for x in np.percentile(ests, [2.5, 97.5]))
    return se, lo, hi


def main() -> None:
    df = dump_csv(MODULE)  # n=1566 complete case; identical bytes for R
    n = len(df)

    # (1) Crude (unadjusted) mean weight-change difference -- book §12.2.
    crude = (
        df.loc[df.qsmk == 1, "wt82_71"].mean() - df.loc[df.qsmk == 0, "wt82_71"].mean()
    )

    # (2) Standardization / parametric g-formula via StatsPAI (additive Q).
    dd, covs = book_design(df)
    res = sp.g_computation(
        dd,
        y="wt82_71",
        treat="qsmk",
        covariates=covs,
        estimand="ATE",
        n_boot=2000,
        seed=42,
    )

    # (3) Python gold reproducing the book's EXACT Program 13.3 outcome
    #     model (with the qsmk:smokeintensity interaction) + bootstrap CI.
    gold_ate = _book_standardization_ate(dd, covs)
    gold_se, gold_lo, gold_hi = _bootstrap_ci(dd, covs, n_boot=2000, seed=42)

    # (4) Doubly-robust cross-check (AIPW on the same design).
    aipw = sp.aipw(
        dd, y="wt82_71", treat="qsmk", covariates=covs, estimand="ATE", seed=42
    )

    rows = [
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="crude_diff",
            estimate=float(crude),
            se=None,
            n=n,
            published=2.54,
            citation="Hernán-Robins, What If §12.2 (crude)",
        ),
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="gformula_ate",
            estimate=float(res.estimate),
            se=float(res.se),
            n=n,
            published=3.5,
            citation="Hernán-Robins, What If Program 13.3 (standardization)",
            extra={
                "model": "additive Q(D,X) (no qsmk:smokeintensity)",
                "ci": [float(res.ci[0]), float(res.ci[1])],
            },
        ),
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="gformula_interaction",
            estimate=float(gold_ate),
            se=float(gold_se),
            n=n,
            published=3.5,
            citation="Hernán-Robins, What If Program 13.3 (qsmk:smokeintensity)",
            extra={
                "model": "wt82_71 ~ qsmk + qsmk:smokeintensity + conf",
                "ci": [gold_lo, gold_hi],
            },
        ),
    ]

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "data_source": "sp.datasets.nhefs (NHEFS / What If)",
            "n_obs": n,
            "ci_gformula_additive": [float(res.ci[0]), float(res.ci[1])],
            "ci_gformula_interaction": [gold_lo, gold_hi],
            "published_gformula_ci": [2.6, 4.5],
            "aipw_crosscheck": float(aipw.estimate),
            "aipw_ci": [float(aipw.ci[0]), float(aipw.ci[1])],
        },
    )
    print(
        f"[{MODULE}] crude={crude:.4f} (book 2.54)  "
        f"g-formula additive={res.estimate:.4f} 95% CI "
        f"({res.ci[0]:.2f},{res.ci[1]:.2f})  "
        f"interaction(book)={gold_ate:.4f} 95% CI ({gold_lo:.2f},{gold_hi:.2f})  "
        f"aipw={aipw.estimate:.4f}  (book 3.5 [2.6,4.5])"
    )


if __name__ == "__main__":
    main()
