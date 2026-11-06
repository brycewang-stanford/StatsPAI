"""Synthetic data for ``tests/reference_parity/test_regcmd_stata_r_parity.py``.

Two files, both read unchanged by Stata 18 (``import delimited, asdouble``),
R (``read.csv``) and pandas (``float_precision="round_trip"``). Every float
is rounded to 8 decimals so all three parse the same doubles.

``regcmd_cs.csv`` -- 800 rows in 40 clusters ``g``, for the likelihood
models:

* ``x1``, ``x2`` normal, ``x3`` a dummy, ``z`` the selection-only regressor,
  ``pw`` sampling weights in [0.5, 2]; every error carries a cluster shock
  so that clustered and plain standard errors differ.
* ``ilo`` / ``ihi`` interval-coded outcome (``intreg``): bands of width 1,
  open below -1 and above 3, a fifth exact, five rows with no bounds.
* ``s`` selection indicator and ``ybh`` the binary outcome observed when
  ``s == 1`` (``heckprobit``; error correlation 0.6).
* ``yo`` ordinal outcome 1-4 whose latent error scale depends on ``x1``,
  so the parallel-lines restriction fails for ``x1`` only (``gologit2``).

``regcmd_wild.csv`` -- 480 rows in 12 clusters ``cl``, for the WRE wild
bootstrap: two endogenous regressors ``d1`` / ``d2``, instruments ``z1`` /
``z2`` / ``z3``, an exogenous control ``w``. With 12 clusters the 4,096
Rademacher sign vectors are enumerated on both sides, so the bootstrap
p-values are exact numbers, not Monte Carlo estimates.

    python tests/reference_parity/_fixtures/_generate_regcmd_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def _cross_section() -> pd.DataFrame:
    rng = np.random.default_rng(20261105)
    n, G = 800, 40
    g = np.repeat(np.arange(1, G + 1), n // G)
    shock = rng.normal(size=G)[g - 1]
    x1 = rng.normal(size=n) + 0.3 * shock
    x2 = rng.normal(size=n)
    x3 = (rng.uniform(size=n) < 0.4).astype(float)
    z = rng.normal(size=n)
    pw = rng.uniform(0.5, 2.0, size=n)

    # interval regression
    yi = 0.5 + x1 - 0.5 * x2 + 0.4 * x3 + 0.5 * shock + 1.2 * rng.normal(size=n)
    lo = np.floor(yi)
    hi = lo + 1.0
    exact = rng.uniform(size=n) < 0.2
    lo = np.where(exact, yi, lo)
    hi = np.where(exact, yi, hi)
    left, right = yi < -1, yi > 3
    lo = np.where(left, np.nan, lo)
    hi = np.where(left, -1.0, hi)
    hi = np.where(right, np.nan, hi)
    lo = np.where(right, 3.0, lo)
    lo[:5] = np.nan
    hi[:5] = np.nan

    # probit with sample selection, corr(u, e) = 0.6
    e = rng.normal(size=n)
    u = 0.6 * e + 0.8 * rng.normal(size=n)
    s = (0.3 + z + 0.5 * x1 + 0.4 * shock + e > 0).astype(float)
    yb = (0.2 + 0.8 * x1 - 0.5 * x2 + 0.3 * x3 + 0.3 * shock + u > 0).astype(float)
    ybh = np.where(s == 1, yb, np.nan)

    # ordinal outcome, latent logistic error with scale exp(0.35 x1)
    lat = (
        0.9 * x1
        - 0.6 * x2
        + 0.5 * x3
        + 0.4 * shock
        + np.exp(0.35 * x1) * rng.logistic(size=n)
    )
    yo = 1 + np.digitize(lat, [-1.2, 0.3, 1.6])

    out = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "g": g,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "z": z,
            "pw": pw,
            "ilo": lo,
            "ihi": hi,
            "s": s,
            "ybh": ybh,
            "yo": yo,
        }
    )
    return out


def _wild() -> pd.DataFrame:
    rng = np.random.default_rng(20261106)
    n, G = 480, 12
    cl = np.repeat(np.arange(1, G + 1), n // G)
    a = rng.normal(size=G)[cl - 1]
    w = rng.normal(size=n)
    z1, z2, z3 = rng.normal(size=(3, n))
    u = 0.7 * a + rng.normal(size=n) * (1.0 + 0.5 * np.abs(w))
    v1 = 0.5 * u + rng.normal(size=n)
    v2 = -0.4 * u + rng.normal(size=n)
    d1 = 0.5 * z1 + 0.3 * z2 + 0.2 * w + 0.4 * a + v1
    d2 = 0.2 * z1 - 0.4 * z3 + 0.6 * z2 + v2
    y = 1.0 + 0.3 * d1 - 0.2 * d2 + 0.5 * w + u
    return pd.DataFrame(
        {
            "cl": cl,
            "y": y,
            "d1": d1,
            "d2": d2,
            "w": w,
            "z1": z1,
            "z2": z2,
            "z3": z3,
        }
    )


def _write(frame: pd.DataFrame, name: str) -> None:
    floats = frame.select_dtypes("float").columns
    frame = frame.copy()
    frame[floats] = frame[floats].round(8)
    frame.to_csv(HERE / name, index=False)


def main() -> None:
    _write(_cross_section(), "regcmd_cs.csv")
    _write(_wild(), "regcmd_wild.csv")


if __name__ == "__main__":
    main()
