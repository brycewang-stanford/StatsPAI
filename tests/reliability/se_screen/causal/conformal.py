"""SE screen: conformal prediction intervals for causal quantities.

The claim screened here is coverage of a prediction interval at a fresh
draw from the population, not a standard error.

Design constants (fixed before the first run, 2026-11-13):

* Training sample ``N = 1000`` from ``_dgp.obs``; 500 fresh test rows per
  replication; nominal coverage 0.90 (the functions' default
  ``alpha = 0.1``).
* Potential outcomes carry **independent** noise: ``Y(0) = 1 + x1 +
  0.5 x2 + e0`` and ``Y(1) = 1 + x1 + 0.5 x2 + tau(x) + e1``. The
  observed outcome has the same law as in ``_dgp.obs``, and the
  individual effect ``Y(1) - Y(0) = tau(x) + e1 - e0`` is not a function
  of ``x``. An interval that claims to cover the individual effect has
  to cover that; ``coverage_secondary`` records coverage of ``tau(x)``.
* Each replication's ``cover`` is the share of the 500 test rows
  covered. ``B = 400`` replications; the band is the Wilson 99% band for
  a proportion at ``B`` draws, which is conservative here because each
  replication averages 500 test rows.
* Rows marked ``conservative`` use a union bound for the individual
  effect; only under-coverage is a finding there.

Run: ``PYTHONPATH=src python tests/reliability/se_screen/causal/conformal.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _dgp import PS_B1, PS_B2, TAU0, TAU1  # noqa: E402
from _harness import Row, main  # noqa: E402
from scipy.special import expit  # noqa: E402

import statspai as sp  # noqa: E402

N = 1000
N_TEST = 500
X = ["x1", "x2"]


def _draw(rng, n):
    """Observational design with independent potential-outcome noise."""
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    ps = expit(PS_B1 * x1 + PS_B2 * x2)
    d = (rng.uniform(size=n) < ps).astype(int)
    tau = TAU0 + TAU1 * x1
    y0 = 1 + x1 + 0.5 * x2 + rng.normal(size=n)
    y1 = 1 + x1 + 0.5 * x2 + tau + rng.normal(size=n)
    y = np.where(d == 1, y1, y0)
    return pd.DataFrame(
        {"y": y, "d": d, "x1": x1, "x2": x2, "y0": y0, "y1": y1, "tau": tau}
    )


def _cov(lo, hi, target):
    lo, hi, target = np.asarray(lo), np.asarray(hi), np.asarray(target)
    return float(np.mean((lo <= target) & (target <= hi)))


def _counterfactual(arm):
    def fit(rng):
        df, test = _draw(rng, N), _draw(rng, N_TEST)
        r = sp.conformal_counterfactual(
            df,
            "y",
            "d",
            X,
            X_test=test[X].to_numpy(),
            random_state=int(rng.integers(2**31)),
        )
        lo = r.lower_Y1 if arm == 1 else r.lower_Y0
        hi = r.upper_Y1 if arm == 1 else r.upper_Y0
        return {
            "cover": _cov(lo, hi, test[f"y{arm}"]),
            "width": float(np.mean(np.asarray(hi) - np.asarray(lo))),
        }

    return fit


def _ite_interval(rng):
    df, test = _draw(rng, N), _draw(rng, N_TEST)
    r = sp.conformal_ite_interval(
        df,
        "y",
        "d",
        X,
        X_test=test[X].to_numpy(),
        random_state=int(rng.integers(2**31)),
    )
    return {
        "cover": _cov(r.lower, r.upper, test["y1"] - test["y0"]),
        "cover2": _cov(r.lower, r.upper, test["tau"]),
        "width": float(np.mean(np.asarray(r.upper) - np.asarray(r.lower))),
    }


def _cate(rng):
    """Intervals are returned for the rows of the sample itself."""
    df = _draw(rng, N)
    r = sp.conformal_cate(
        df, "y", "d", X, alpha=0.1, random_state=int(rng.integers(2**31))
    )
    lo, hi = r.model_info["cate_lower"], r.model_info["cate_upper"]
    return {
        "cover": _cov(lo, hi, df["y1"] - df["y0"]),
        "cover2": _cov(lo, hi, df["tau"]),
        "width": float(np.mean(np.asarray(hi) - np.asarray(lo))),
    }


def _regression(method):
    def fit(rng):
        n = 300 if method != "split" else N
        df, test = _draw(rng, n), _draw(rng, N_TEST)
        out = sp.conformal_regression(
            "y ~ x1 + x2 + d",
            df,
            newdata=test,
            method=method,
            seed=int(rng.integers(2**31)),
        )
        return {
            "cover": _cov(out["lower"], out["upper"], test["y"]),
            "width": float(np.mean(out["upper"] - out["lower"])),
        }

    return fit


def _weighted(shift):
    """Split conformal; with ``shift`` the test rows have x1 ~ N(0.5, 1)
    and the calibration scores carry the true density ratio."""

    def fit(rng):
        df, cal, test = _draw(rng, 500), _draw(rng, 500), _draw(rng, N_TEST)
        kw = {}
        if shift:
            # accept-reject to draw test rows from the shifted law
            big = _draw(rng, 6 * N_TEST)
            ratio = np.exp(0.5 * big["x1"] - 0.125)
            keep = rng.uniform(size=len(big)) < ratio / ratio.max()
            test = big[keep].head(N_TEST)
            kw["weights_calib"] = np.exp(0.5 * cal["x1"].to_numpy() - 0.125)
            kw["weights_test"] = np.exp(0.5 * test["x1"].to_numpy() - 0.125)
        cols = ["x1", "x2", "d"]
        out = sp.weighted_conformal_prediction(
            df[cols].to_numpy(),
            df["y"].to_numpy(),
            cal[cols].to_numpy(),
            cal["y"].to_numpy(),
            test[cols].to_numpy(),
            **kw,
        )
        lo, hi = np.asarray(out[0]), np.asarray(out[1])
        return {"cover": _cov(lo, hi, test["y"]), "width": float(np.mean(hi - lo))}

    return fit


def _ite_method(fn, **extra):
    def fit(rng):
        df, test = _draw(rng, N), _draw(rng, N_TEST)
        kw = dict(extra)
        if "protected" in kw:
            df["s"] = (df["x2"] > 0).astype(int)
            test["s"] = (test["x2"] > 0).astype(int)
        r = fn(
            df,
            "y",
            "d",
            X,
            *(["s"] if "protected" in kw else []),
            test_data=test,
            seed=int(rng.integers(2**31)),
        )
        iv = np.asarray(r.intervals)
        return {
            "cover": _cov(iv[:, 0], iv[:, 1], test["y1"] - test["y0"]),
            "cover2": _cov(iv[:, 0], iv[:, 1], test["tau"]),
            "width": float(np.mean(iv[:, 1] - iv[:, 0])),
        }

    return fit


def _continuous(rng):
    """Continuous dose; the band is for the outcome at the test row's dose."""

    def draw(n):
        t = rng.uniform(0, 5, n)
        x = rng.normal(size=n)
        y = 2.0 + 0.7 * t + 0.5 * x + rng.normal(0, 0.5, n)
        return pd.DataFrame({"y": y, "t": t, "x": x})

    train, test = draw(N), draw(N_TEST)
    r = sp.conformal_continuous(
        train,
        y="y",
        treatment="t",
        covariates=["x"],
        test_data=test,
        random_state=int(rng.integers(2**31)),
    )
    p = r.predictions
    return {
        "cover": _cov(p["lo"], p["hi"], test["y"]),
        "width": float(np.mean(p["hi"] - p["lo"])),
    }


P = dict(kind="prediction", nominal=0.90, n=N, truth=0.90)
ROWS = [
    Row(
        "conformal_counterfactual.y1",
        "conformal_counterfactual",
        "sp.conformal_counterfactual(df, y, d, X, X_test=...) band for Y(1)",
        fit=_counterfactual(1),
        design="_draw; target Y(1) of fresh rows",
        **P,
    ),
    Row(
        "conformal_counterfactual.y0",
        "conformal_counterfactual",
        "sp.conformal_counterfactual(...) band for Y(0)",
        fit=_counterfactual(0),
        design="_draw; target Y(0) of fresh rows",
        **P,
    ),
    Row(
        "conformal_ite_interval",
        "conformal_ite_interval",
        "sp.conformal_ite_interval(df, y, d, X, X_test=...)",
        fit=_ite_interval,
        conservative=True,
        design="_draw; target Y(1) - Y(0) of fresh rows",
        **P,
    ),
    Row(
        "conformal_cate",
        "conformal_cate",
        "sp.conformal_cate(df, y, d, X, alpha=0.1).model_info['cate_lower/upper']",
        fit=_cate,
        conservative=True,
        design="_draw; target Y(1) - Y(0) of the sample rows",
        **P,
    ),
    Row(
        "conformal_regression.jackknife_plus",
        "conformal_regression",
        "sp.conformal_regression('y ~ x1 + x2 + d', df, newdata=test)",
        fit=_regression("jackknife+"),
        design="_draw with n = 300; target y of fresh rows",
        **{**P, "n": 300},
    ),
    Row(
        "conformal_regression.split",
        "conformal_regression",
        "sp.conformal_regression(..., method='split')",
        fit=_regression("split"),
        design="_draw; target y of fresh rows",
        **P,
    ),
    Row(
        "conformal_regression.full",
        "conformal_regression",
        "sp.conformal_regression(..., method='full')",
        fit=_regression("full"),
        design="_draw with n = 300; target y of fresh rows",
        B=200,
        **{**P, "n": 300},
    ),
    Row(
        "weighted_conformal_prediction",
        "weighted_conformal_prediction",
        "sp.weighted_conformal_prediction(Xtr, ytr, Xcal, ycal, Xtest)",
        fit=_weighted(False),
        design="500 train, 500 calibration rows",
        **P,
    ),
    Row(
        "weighted_conformal_prediction.shift",
        "weighted_conformal_prediction",
        "sp.weighted_conformal_prediction(..., weights_calib=w, weights_test=w)",
        fit=_weighted(True),
        design="test rows from x1 ~ N(0.5, 1), true density ratio as weights",
        **P,
    ),
    Row(
        "conformal_debiased_ml",
        "conformal_debiased_ml",
        "sp.conformal_debiased_ml(df, y, d, X, test_data=test)",
        fit=_ite_method(sp.conformal_debiased_ml),
        conservative=True,
        design="_draw; target Y(1) - Y(0) of fresh rows",
        **P,
    ),
    Row(
        "conformal_density_ite",
        "conformal_density_ite",
        "sp.conformal_density_ite(df, y, d, X, test_data=test)",
        fit=_ite_method(sp.conformal_density_ite),
        design="_draw; target Y(1) - Y(0) of fresh rows",
        **P,
    ),
    Row(
        "conformal_fair_ite",
        "conformal_fair_ite",
        "sp.conformal_fair_ite(df, y, d, X, 's', test_data=test)",
        fit=_ite_method(sp.conformal_fair_ite, protected=True),
        conservative=True,
        design="_draw; protected group s = 1[x2 > 0]; target Y(1) - Y(0)",
        **P,
    ),
    Row(
        "conformal_continuous",
        "conformal_continuous",
        "sp.conformal_continuous(train, y=, treatment=, covariates=, test_data=)",
        fit=_continuous,
        design="dose t ~ U(0, 5), y = 2 + 0.7 t + 0.5 x + N(0, 0.25)",
        **P,
    ),
]

MECHANISMS: dict = {}

if __name__ == "__main__":
    main(__file__, ROWS, MECHANISMS)
