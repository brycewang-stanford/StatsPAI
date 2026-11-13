"""Data for the survival / epidemiology items of the 2026-11 gap list.

Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_gaps_epi_data.py

then ``Rscript tests/reference_parity/_fixtures/_generate_gaps_epi_R.R`` and
``stata-mp -b do _generate_gaps_epi_Stata.do`` (from this directory).

* ``gaps_surv.csv``: 240 subjects in 30 clinics and 3 sites. Follow-up is
  recorded in whole weeks, so failure times are tied (up to seven at one
  time), with delayed entry strictly before exit. For the exact partial
  likelihood (``sp.cox(ties='exact')``) and the robust and clustered
  variances of ``sp.aft``.
* ``gaps_obs.csv``: 700 subjects of an observational study with a
  confounded binary treatment, a continuous, a binary and a count outcome.
  The propensity score reaches close to 0 and 1, so percentile truncation
  of the weights binds. For ``sp.g_computation`` with logistic and Poisson
  outcome models, ``sp.ipw(truncate=)`` and the sandwich standard error of
  ``sp.overlap_weights``.
* ``gaps_long.csv``: 150 subjects with four visits each, a binary, a count
  and a continuous response. For the QIC of ``sp.gee`` and the standard
  errors of the variance components of ``sp.mixed`` / ``sp.meglm``.
* ``gaps_gee_alpha.csv``: the working-correlation parameters ``sp.gee``
  estimates on ``gaps_long.csv`` (see :func:`gee_alpha`).
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def surv() -> pd.DataFrame:
    rng = np.random.default_rng(2026111301)
    n = 240
    clinic = rng.integers(1, 31, n)
    site = (clinic - 1) % 3
    arm = rng.integers(0, 2, n)
    age = np.round(rng.normal(62, 8, n))
    ecog = rng.integers(0, 3, n)
    frailty = rng.normal(0, 0.35, 30)[clinic - 1]
    haz = 0.012 * np.exp(
        0.025 * (age - 62) + 0.4 * ecog - 0.55 * arm + 0.15 * site + frailty
    )
    t = rng.weibull(1.3, n) / haz ** (1 / 1.3)
    c = rng.uniform(20, 110, n)
    time = np.ceil(np.minimum(t, c))  # whole weeks: ties
    event = (t <= c).astype(int)
    entry = np.floor(rng.uniform(0, 0.4, n) * (time - 1))
    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "clinic": clinic,
            "site": site,
            "arm": arm,
            "age": age,
            "ecog": ecog,
            "entry": entry,
            "time": time,
            "event": event,
        }
    )


def obs() -> pd.DataFrame:
    rng = np.random.default_rng(2026111302)
    n = 700
    x1 = np.round(rng.normal(0, 1, n), 3)
    x2 = np.round(rng.normal(0, 1, n), 3)
    x3 = rng.integers(0, 2, n)
    ps = 1 / (1 + np.exp(-(-0.3 + 1.4 * x1 - 0.9 * x2 + 0.7 * x3)))
    a = (rng.random(n) < ps).astype(int)
    y = np.round(1.0 + 0.8 * a + 0.9 * x1 + 0.5 * x2 - 0.4 * x3 + rng.normal(0, 1, n), 4)
    pb = 1 / (1 + np.exp(-(-0.8 + 0.7 * a + 0.6 * x1 - 0.5 * x2 + 0.3 * x3)))
    ybin = (rng.random(n) < pb).astype(int)
    mu = np.exp(0.2 + 0.35 * a + 0.3 * x1 - 0.2 * x2 + 0.25 * x3)
    ycnt = rng.poisson(mu)
    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "a": a,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "y": y,
            "ybin": ybin,
            "ycnt": ycnt,
        }
    )


def long() -> pd.DataFrame:
    rng = np.random.default_rng(2026111303)
    n, t = 150, 4
    pid = np.repeat(np.arange(1, n + 1), t)
    visit = np.tile(np.arange(t), n)
    trt = np.repeat(rng.integers(0, 2, n), t)
    age = np.repeat(np.round(rng.normal(50, 10, n)), t)
    u = np.repeat(rng.normal(0, 0.8, n), t)
    slope = np.repeat(rng.normal(0, 0.25, n), t)
    eta = -0.6 + 0.5 * trt + 0.25 * visit + 0.02 * (age - 50) + u
    ybin = (rng.random(n * t) < 1 / (1 + np.exp(-eta))).astype(int)
    ycnt = rng.poisson(np.exp(0.3 + 0.3 * trt + 0.15 * visit + 0.5 * u))
    y = np.round(
        2.0
        + 0.6 * trt
        + (0.4 + slope) * visit
        + 0.03 * (age - 50)
        + 1.2 * u
        + rng.normal(0, 0.9, n * t),
        4,
    )
    return pd.DataFrame(
        {
            "pid": pid,
            "visit": visit,
            "trt": trt,
            "age": age,
            "y": y,
            "ybin": ybin,
            "ycnt": ycnt,
        }
    )


def gee_alpha(panel: pd.DataFrame) -> pd.DataFrame:
    """Working-correlation parameters of ``sp.gee`` on ``gaps_long.csv``.

    ``geepack`` estimates the correlation parameter by another moment
    equation than ``sp.gee`` (which follows R ``gee`` and Stata ``xtgee``),
    so the two fits differ in their coefficients. To compare the QIC of the
    same fit, the R script passes these values to ``geeglm(corstr =
    "fixed")``; the coefficients then solve the same estimating equations.
    """
    import statspai as sp

    rows = []
    for family, y in (("binomial", "ybin"), ("poisson", "ycnt"), ("gaussian", "y")):
        for corstr in ("exchangeable", "ar1"):
            fit = sp.gee(
                f"{y} ~ trt + visit + age",
                panel,
                id="pid",
                time="visit",
                family=family,
                corstr=corstr,
                tol=1e-13,
            )
            rows.append(
                {
                    "family": family,
                    "corstr": corstr,
                    "alpha": repr(float(fit.model_info["corr_alpha"])),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    surv().to_csv(HERE / "gaps_surv.csv", index=False)
    obs().to_csv(HERE / "gaps_obs.csv", index=False)
    panel = long()
    panel.to_csv(HERE / "gaps_long.csv", index=False)
    gee_alpha(pd.read_csv(HERE / "gaps_long.csv")).to_csv(
        HERE / "gaps_gee_alpha.csv", index=False
    )
    print("wrote gaps_surv.csv, gaps_obs.csv, gaps_long.csv, gaps_gee_alpha.csv")
