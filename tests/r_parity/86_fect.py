"""StatsPAI fect counterfactual-estimator parity (Python side) -- Module 86.

Runs the native ``sp.fect`` port of Liu, Wang and Xu's counterfactual
estimators on a deterministic staggered-adoption factor panel and emits
one block of rows per outcome model:

- ``fe``  : two-way fixed effects (imputation estimator);
- ``ife`` : interactive fixed effects with ``r = 2`` factors, EM on the
  incomplete untreated panel;
- ``mc``  : matrix completion with a fixed nuclear-norm penalty.

Rows per method: ``<m>_att_avg`` (ATT over treated observations),
``<m>_att_avg_unit`` (mean of per-unit ATTs), ``<m>_beta_x1`` /
``<m>_beta_x2`` (covariate coefficients), ``<m>_mu`` (grand mean),
``<m>_rmse`` (pre-treatment fit on treated units), and
``<m>_att_on_<k>`` (ATT by relative period in fect's coding, where 0 is
the last untreated period and 1 the first treated period; ``n`` carries
fect's cell count). The companion 86_fect.R runs ``fect::fect`` with
``se = FALSE``, ``CV = FALSE``, ``force = "two-way"``, ``tol = 1e-12`` and ``max.iteration = 20000``
on the same CSV bytes; the Stata port (``fect_stata``, installed from
GitHub into a local ado path) runs the same three specifications.

Tolerance: rel_est 1e-6 (machine level; every row is a deterministic
fixed point of the same EM map started from the same two-way initial
fit). No SE rows: fect's inference is resampling-based and optional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results


MODULE = "86_fect"
R_FACTORS = 2
LAMBDA_MC = 0.002


def make_data(seed: int = PARITY_SEED) -> pd.DataFrame:
    """Staggered adoption with a two-factor error structure and covariates.

    60 units x 20 periods: 20 never-treated units and four cohorts of 10
    units first treated in periods 8, 11, 14, and 17 (so the earliest
    cohort has 7 untreated periods, above fect's ``min.T0 = 5`` for the
    ife/mc models). The treatment effect grows with time since
    treatment and varies by unit, and both factors load on treatment
    timing so a plain two-way model is biased -- which is what separates
    the three rows.
    """
    rng = np.random.default_rng(seed)
    N, T = 60, 20
    units = np.arange(1, N + 1)
    periods = np.arange(1, T + 1)
    first_treat = np.zeros(N, dtype=int)
    for c, start in enumerate((8, 11, 14, 17)):
        first_treat[20 + 10 * c : 30 + 10 * c] = start
    alpha = rng.normal(0.0, 1.0, N)
    xi = rng.normal(0.0, 1.0, T)
    F = rng.normal(0.0, 1.0, (T, R_FACTORS))
    L = rng.normal(0.0, 1.0, (N, R_FACTORS))
    # Loadings correlate with adoption timing (earlier cohorts load more).
    L[:, 0] += np.where(first_treat > 0, (20 - first_treat) / 6.0, 0.0)
    tau_unit = rng.normal(1.0, 0.3, N)
    rows = []
    for i, u in enumerate(units):
        for t_idx, t in enumerate(periods):
            x1 = rng.normal(0.0, 1.0) + 0.3 * L[i, 1]
            x2 = rng.normal(0.0, 1.0)
            d = int(first_treat[i] > 0 and t >= first_treat[i])
            since = (t - first_treat[i] + 1) if d else 0
            effect = tau_unit[i] * (1.0 + 0.2 * since) if d else 0.0
            y = (
                5.0 + alpha[i] + xi[t_idx] + F[t_idx] @ L[i]
                + 1.0 * x1 - 0.5 * x2 + effect + rng.normal(0.0, 0.5)
            )
            rows.append({"id": int(u), "time": int(t), "Y": float(y), "D": d,
                         "X1": float(x1), "X2": float(x2)})
    return pd.DataFrame(rows)


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)

    specs = {
        "fe": {},
        "ife": {"r": R_FACTORS},
        "mc": {"lam": LAMBDA_MC},
    }
    rows: list[ParityRecord] = []
    extra: dict = {
        "design": "60 units x 20 periods; 20 never-treated; cohorts first "
                  "treated in 8/11/14/17; two latent factors; two covariates",
        "force": "two-way",
        "tol": 1e-12,
        "max_iter": 20000,
        "r_ife": R_FACTORS,
        "lambda_mc": LAMBDA_MC,
        "relative_time_coding": (
            "fect: 0 = last untreated period, 1 = first treated period "
            "(StatsPAI relative_time = fect_time - 1)"
        ),
    }
    n_obs = int(len(df))
    for m, kw in specs.items():
        fit = sp.fect(
            df, y="Y", treat="D", unit="id", time="time",
            covariates=["X1", "X2"], method=m, tol=1e-12, max_iter=20000, **kw,
        )
        mi = fit.model_info
        rows.append(ParityRecord(module=MODULE, side="py", statistic=f"{m}_att_avg",
                                 estimate=float(fit.estimate), n=n_obs))
        rows.append(ParityRecord(module=MODULE, side="py", statistic=f"{m}_att_avg_unit",
                                 estimate=float(mi["att_avg_unit"]), n=n_obs))
        rows.append(ParityRecord(module=MODULE, side="py", statistic=f"{m}_beta_x1",
                                 estimate=float(mi["beta"]["X1"]), n=n_obs))
        rows.append(ParityRecord(module=MODULE, side="py", statistic=f"{m}_beta_x2",
                                 estimate=float(mi["beta"]["X2"]), n=n_obs))
        rows.append(ParityRecord(module=MODULE, side="py", statistic=f"{m}_mu",
                                 estimate=float(mi["mu"]), n=n_obs))
        rows.append(ParityRecord(module=MODULE, side="py", statistic=f"{m}_rmse",
                                 estimate=float(mi["pre_treatment_rmse"]), n=n_obs))
        for _, r in fit.detail.iterrows():
            rows.append(ParityRecord(
                module=MODULE, side="py",
                statistic=f"{m}_att_on_{int(r['fect_time'])}",
                estimate=float(r["att"]), n=int(r["count"]),
            ))
        extra[f"{m}_niter"] = int(mi["niter"])
        extra[f"{m}_converged"] = bool(mi["converged"])
        if m == "mc":
            extra["mc_lambda_norm"] = float(mi["lambda_norm"])

    write_results(MODULE, "py", rows, extra=extra)


if __name__ == "__main__":
    main()
