"""SE calibration screen: spatial regression and nonparametric fits.

Spatial rows: a 25 by 25 lattice (n = 625) with rook contiguity,
row-normalised. Spatial parameter 0.4, slope on ``x1`` 0.5.

Nonparametric rows: n = 1,000, x uniform on [0, 1], target the regression
function at x = 0.5. One design is linear (no smoothing bias, so the row
tests the variance alone), one is curved.

    python tests/reliability/se_screen/spatial.py [--only name] [--confirm]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import Row, coef, main, tidy_row, wald  # noqa: E402,F401

FAMILY = "spatial"
SIDE = 25
N = SIDE * SIDE
RHO = 0.4
BETA = 0.5
X0 = 0.5


def _sp():
    import statspai as sp

    return sp


def _rook(side=SIDE):
    n = side * side
    w = np.zeros((n, n))
    for i in range(side):
        for j in range(side):
            k = i * side + j
            if i > 0:
                w[k, k - side] = 1
            if i < side - 1:
                w[k, k + side] = 1
            if j > 0:
                w[k, k - 1] = 1
            if j < side - 1:
                w[k, k + 1] = 1
    return w / w.sum(axis=1, keepdims=True)


W = _rook()
_SOLVE = np.linalg.inv(np.eye(N) - RHO * W)


def d_sar(rng):
    x = rng.normal(size=N)
    y = _SOLVE @ (1 + BETA * x + rng.normal(size=N))
    return pd.DataFrame({"y": y, "x1": x})


def d_sem(rng):
    x = rng.normal(size=N)
    y = 1 + BETA * x + _SOLVE @ rng.normal(size=N)
    return pd.DataFrame({"y": y, "x1": x})


def d_sdm(rng):
    x = rng.normal(size=N)
    y = _SOLVE @ (1 + BETA * x + 0.3 * (W @ x) + rng.normal(size=N))
    return pd.DataFrame({"y": y, "x1": x})


def d_slx(rng):
    x = rng.normal(size=N)
    y = 1 + BETA * x + 0.3 * (W @ x) + rng.normal(size=N)
    return pd.DataFrame({"y": y, "x1": x})


def d_sac(rng):
    x = rng.normal(size=N)
    lam = np.linalg.inv(np.eye(N) - 0.3 * W)
    y = _SOLVE @ (1 + BETA * x + lam @ rng.normal(size=N))
    return pd.DataFrame({"y": y, "x1": x})


def d_np_linear(rng, n=1000):
    x = rng.uniform(size=n)
    return pd.DataFrame({"x": x, "y": 1 + 2 * x + 0.5 * rng.normal(size=n)})


def d_np_curved(rng, n=1000):
    x = rng.uniform(size=n)
    return pd.DataFrame({"x": x, "y": np.sin(3 * x) + 0.5 * rng.normal(size=n)})


# ------------------------------------------------------------------- fits
def _term(r, *needles):
    for t in r.params.index:
        if any(str(t).lower() == n for n in needles):
            return t
    raise KeyError(f"none of {needles} in {list(r.params.index)}")


def fit_sp(est, which="x1"):
    def fit(df):
        r = getattr(_sp(), est)(W, df, "y ~ x1")
        if which == "x1":
            return coef(r, "x1")
        return coef(r, _term(r, "rho", "lambda", "w_y", "wy"))

    return fit


def fit_lpoly(df):
    r = _sp().lpoly(df, y="y", x="x", grid=np.array([X0]))
    return r.fitted[0], r.se[0], r.ci_lower[0], r.ci_upper[0]


def fit_lprobust(h):
    def fit(df):
        r = _sp().lprobust_at_point(df.x.to_numpy(), df.y.to_numpy(), X0, h)
        return wald(r.tau_bc, r.se_rb)  # the robust bias-corrected interval

    return fit


def R(name, estimator, call, dgp, fit, truth=BETA, n=N, B=None, design=""):
    return Row(name, estimator, call, n, truth, dgp, fit, B, design)


ROWS = [
    R("sar/beta", "sar", 'sar(W, data, "y ~ x1")', d_sar, fit_sp("sar")),
    R(
        "sar/rho",
        "sar",
        "sar(...), spatial lag coefficient",
        d_sar,
        fit_sp("sar", "rho"),
        truth=RHO,
    ),
    R("sem/beta", "sem", 'sem(W, data, "y ~ x1")', d_sem, fit_sp("sem")),
    R(
        "sem/lambda",
        "sem",
        "sem(...), spatial error coefficient",
        d_sem,
        fit_sp("sem", "rho"),
        truth=RHO,
    ),
    R("sdm/beta", "sdm", 'sdm(W, data, "y ~ x1")', d_sdm, fit_sp("sdm")),
    R("slx/beta", "slx", 'slx(W, data, "y ~ x1")', d_slx, fit_sp("slx")),
    R(
        "sac/beta",
        "sac",
        'sac(W, data, "y ~ x1")',
        d_sac,
        fit_sp("sac"),
        design="lag 0.4 and error 0.3",
    ),
    R(
        "sar_gmm/beta",
        "sar_gmm",
        'sar_gmm(W, data, "y ~ x1")',
        d_sar,
        fit_sp("sar_gmm"),
    ),
    R(
        "sar_gmm/rho",
        "sar_gmm",
        "sar_gmm(...), spatial lag coefficient",
        d_sar,
        fit_sp("sar_gmm", "rho"),
        truth=RHO,
    ),
    R(
        "lpoly/linear-truth",
        "lpoly",
        "lpoly(y, x) at x = 0.5",
        d_np_linear,
        fit_lpoly,
        truth=2.0,
        n=1000,
        design="linear regression function: no smoothing bias",
    ),
    R(
        "lpoly/curved-truth",
        "lpoly",
        "lpoly(y, x) at x = 0.5",
        d_np_curved,
        fit_lpoly,
        truth=float(np.sin(1.5)),
        n=1000,
        design="sin(3x): the default bandwidth leaves bias",
    ),
    R(
        "lprobust_at_point/curved",
        "lprobust_at_point",
        "lprobust_at_point(x, y, 0.5, h=0.15)",
        d_np_curved,
        fit_lprobust(0.15),
        truth=float(np.sin(1.5)),
        n=1000,
        design="sin(3x); robust bias-corrected interval",
    ),
]

MECHANISMS: dict = {}

if __name__ == "__main__":
    main(__file__, FAMILY, ROWS, MECHANISMS)
