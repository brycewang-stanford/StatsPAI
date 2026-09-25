"""StatsPAI original-data parity (Python side) -- Module 08.

Reproduces Hernán & Robins, *Causal Inference: What If*, **Chapter 14**
(G-estimation of a structural nested mean model) on the REAL NHEFS data
bundled in ``sp.datasets.nhefs()``.  Estimand: ``psi``, the additive
(rank-preserving) one-parameter SNMM coefficient for the effect of
quitting smoking (``qsmk``) on 10-year weight change (``wt82_71``, kg).

Compares against:
  (a) the book's published number -- Program 14.2: ``psi`` ~ 3.4 kg,
      95% CI (2.5, 4.5);
  (b) the R-side base-R reference run on the same CSV bytes
      (``08_nhefs_ch14_gestimation.R``), which implements BOTH the book's
      exact *logistic* g-estimation (fit ``qsmk ~ H(psi) + L``, solve for
      the ``psi`` at which the coefficient on ``H(psi)`` is zero) and the
      *linear* moment-condition g-estimator that StatsPAI uses.

CONVENTION NOTE.  Since 1.29 ``sp.g_estimation`` defaults to
``propensity_model='logit'``: it solves ``sum (A - A_hat)(Y - psi A) = 0``
jointly with the treatment-free OLS equations, ``A_hat`` a logistic fit on
``L``. Because the logit score makes ``A - A_hat`` orthogonal to ``L``, this
is exactly Program 14.2's closed form -- the ``psi`` at which the
coefficient on ``H(psi) = Y - psi*A`` vanishes in ``qsmk ~ H(psi) + L``
(3.4611 on NHEFS, the R side's ``snmm_psi``). Before 1.29 StatsPAI used the
*linear* moment condition (``psi = cov(Y_res, A_res) / var(A_res)`` with OLS
residuals; 3.4626), which ``propensity_model='linear'`` still reproduces and
the R side still carries as ``snmm_psi_linear``.
"""

from __future__ import annotations

import numpy as np
import statspai as sp

from _common import OrigRecord, write_results
from _nhefs import book_design, dump_csv

MODULE = "08_nhefs_ch14_gestimation"


def main() -> None:
    df = dump_csv(MODULE)  # n=1566 complete case; identical bytes for R
    n = len(df)

    # StatsPAI additive-SNMM g-estimate of psi via sp.g_estimation
    # (single stage: the point-treatment qsmk effect on wt82_71).
    dd, covs = book_design(df)
    res = sp.g_estimation(
        dd,
        y="wt82_71",
        treatments=["qsmk"],
        covariates_by_stage=[covs],
        n_bootstrap=500,
        random_state=42,
    )
    psi = float(res.estimate)
    se = float(res.se)
    ci = (float(res.ci[0]), float(res.ci[1]))

    # Independent in-script reproduction of the linear moment condition
    # (StatsPAI's pre-1.29 default, propensity_model='linear') -- the value
    # the R "linear g-estimation" anchor must match.
    X = dd[covs].values.astype(float)
    A = df["qsmk"].values.astype(float)
    Y = df["wt82_71"].values.astype(float)
    Xa = np.column_stack([np.ones(n), X])
    beta_y, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    beta_a, *_ = np.linalg.lstsq(Xa, A, rcond=None)
    Yres = Y - Xa @ beta_y
    Ares = A - Xa @ beta_a
    psi_linear = float(np.sum(Yres * Ares) / np.sum(Ares**2))

    rows = [
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="snmm_psi",
            estimate=psi,
            se=se,
            n=n,
            published=3.4,
            citation="Hern\u00e1n-Robins, What If Program 14.2 (G-estimation SNMM psi)",
            extra={
                "ci": [ci[0], ci[1]],
                "pvalue": float(res.pvalue),
                "method": res.method,
            },
        ),
        OrigRecord(
            module=MODULE,
            side="py",
            statistic="snmm_psi_linear_check",
            estimate=psi_linear,
            se=None,
            n=n,
            published=3.4,
            citation="Linear moment-condition g-estimate (propensity_model='linear')",
        ),
    ]

    write_results(
        MODULE,
        "py",
        rows,
        extra={
            "data_source": "sp.datasets.nhefs (NHEFS / What If)",
            "n_obs": n,
            "estimand": "additive SNMM psi (qsmk -> wt82_71)",
            "ci_snmm": [ci[0], ci[1]],
            "published_psi": 3.4,
            "published_snmm_ci": [2.5, 4.5],
            "statspai_default_propensity_model": "logit",
        },
    )
    print(
        f"[{MODULE}] g-est psi={psi:.4f} 95% CI "
        f"({ci[0]:.2f},{ci[1]:.2f})  linear_check={psi_linear:.4f}  "
        f"(book 3.4 [2.5,4.5])"
    )


if __name__ == "__main__":
    main()
