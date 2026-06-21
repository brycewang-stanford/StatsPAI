"""StatsPAI original-data parity (Python side) -- Module 06.

Reproduces Hernán & Robins, *Causal Inference: What If*, **Chapter 12**
(IP weighting and marginal structural models) on the REAL NHEFS data
bundled in ``sp.datasets.nhefs()``.  Estimand: the average causal effect
of quitting smoking (``qsmk``) on 10-year weight change (``wt82_71``, kg).

Compares against:
  (a) the book's published numbers -- crude difference 2.54 kg
      (§12.2); IP-weighted ATE 3.4 kg, 95% CI (2.4, 4.5) (Program 12.4);
  (b) the R-side base-R stabilized-IPW MSM run on the same CSV bytes
      (``06_nhefs_ch12_ipw.R``).

StatsPAI exposes point-treatment IP weighting as ``sp.ipw`` (Hájek /
normalized ATE).  This is the package's IP-weighting estimator; it now
uses the same converged logistic propensity score as the R side and
recovers the book's IP-weighted ATE to within rounding (3.44 vs 3.4 ->
both round to the book's one-decimal figure).  The R side reproduces the
book's *exact* stabilized-weight saturated MSM (3.44) for a tight
same-bytes anchor.
"""

from __future__ import annotations

import statspai as sp

from _common import OrigRecord, write_results
from _nhefs import book_design, dump_csv

MODULE = "06_nhefs_ch12_ipw"


def main() -> None:
    df = dump_csv(MODULE)  # n=1566 complete case; identical bytes for R
    n = len(df)

    # (1) Crude (unadjusted) mean weight-change difference -- book §12.2.
    crude = (
        df.loc[df.qsmk == 1, "wt82_71"].mean() - df.loc[df.qsmk == 0, "wt82_71"].mean()
    )

    # (2) IP-weighted ATE via StatsPAI on the book's confounder model.
    dd, covs = book_design(df)
    res = sp.ipw(
        dd,
        y="wt82_71",
        treat="qsmk",
        covariates=covs,
        estimand="ATE",
        seed=42,
        n_bootstrap=500,
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
            statistic="ipw_att",
            estimate=float(res.estimate),
            se=float(res.se),
            n=n,
            published=3.4,
            citation="Hernán-Robins, What If Program 12.4 (IP-weighted ATE)",
        ),
    ]

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "data_source": "sp.datasets.nhefs (NHEFS / What If)",
            "n_obs": n,
            "ci_ipw": [float(res.ci[0]), float(res.ci[1])],
            "published_ipw_ci": [2.4, 4.5],
        },
    )
    print(
        f"[{MODULE}] crude={crude:.4f} (book 2.54)  "
        f"ipw_att={res.estimate:.4f} 95% CI "
        f"({res.ci[0]:.2f},{res.ci[1]:.2f})  (book 3.4 [2.4,4.5])"
    )


if __name__ == "__main__":
    main()
