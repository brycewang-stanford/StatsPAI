"""Data for the variance-estimator menu fixtures of ``sp.regress`` / ``sp.iv``.

One cross-section read by both references (``_generate_vce_menu_R.R`` and
``tests/stata_parity/option_parity/92_vce_menu.do``):

* ``g`` -- 12 clusters, few enough that the wild cluster bootstrap can
  enumerate all 2**12 Rademacher sign vectors, which makes its p-value
  deterministic on every side;
* ``yr`` (9 levels) and ``h`` (5 levels) -- second and third clustering
  dimensions, not nested in ``g``;
* rows in time order with AR(1) errors, for the Newey-West variance;
* ``lat`` / ``lon`` in degrees over a few hundred kilometres, for Conley;
* ``w`` -- positive analytic weights;
* ``d`` endogenous, ``z1`` / ``z2`` excluded instruments, ``x1`` / ``x2`` / ``x3``
  exogenous controls; ``x3`` has a zero coefficient (a null for the wild
  bootstrap that is neither trivially rejected nor trivially kept).

Two small files hold cases where a covariance estimator returns negative
variances (``vce_menu_nonpsd_conley.csv``, ``vce_menu_nonpsd_twoway.csv``).

    python tests/reference_parity/_fixtures/_generate_vce_menu_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
OUT = HERE / "vce_menu_data.csv"
OUT_CONLEY = HERE / "vce_menu_nonpsd_conley.csv"
OUT_TWOWAY = HERE / "vce_menu_nonpsd_twoway.csv"


def main() -> None:
    rng = np.random.default_rng(20261015)
    n, G, Y, H = 360, 12, 9, 5
    g = rng.integers(0, G, n)
    yr = rng.integers(0, Y, n)
    h = rng.integers(0, H, n)
    lat = rng.uniform(34.0, 37.0, n)
    lon = rng.uniform(-100.0, -96.0, n)
    w = rng.uniform(0.5, 3.0, n)
    # cluster shocks in g and yr, AR(1) in row order, heteroskedasticity
    a_g = rng.normal(0, 0.6, G)[g]
    a_y = rng.normal(0, 0.4, Y)[yr]
    e = np.empty(n)
    e[0] = rng.normal()
    for i in range(1, n):
        e[i] = 0.5 * e[i - 1] + rng.normal()
    x1 = rng.normal(size=n) + 0.3 * a_g
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n) + 0.5 * rng.normal(0, 1, G)[g]
    z1 = rng.normal(size=n) + 0.2 * rng.normal(0, 1, G)[g]
    z2 = rng.normal(size=n)
    u = a_g + a_y + e * (0.6 + 0.4 * np.abs(x2))
    d = 0.8 * z1 + 0.5 * z2 + 0.3 * x1 + 0.6 * u + rng.normal(size=n)
    y = 1.0 + 0.15 * d + 0.4 * x1 - 0.3 * x2 + 0.0 * x3 + u
    pd.DataFrame(
        dict(
            y=y,
            d=d,
            x1=x1,
            x2=x2,
            x3=x3,
            z1=z1,
            z2=z2,
            w=w,
            g=g + 1,
            yr=yr + 1,
            h=h + 1,
            t=np.arange(1, n + 1),
            lat=lat,
            lon=lon,
        )
    ).to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")
    nonpsd_conley()
    nonpsd_twoway()


def nonpsd_conley() -> None:
    """2SLS with county dummies and one location per county.

    The spatial weights then carry no information within a county, and the
    uniform-kernel Conley matrix has negative variances (acreg reports those
    standard errors as missing).
    """
    rng = np.random.default_rng(0)
    n = 600
    county = np.repeat(np.arange(30), 20)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = 0.8 * z + 0.5 * u + rng.normal(size=n)
    y = 1 + 0.3 * d + 0.5 * x + u
    pd.DataFrame(
        dict(
            y=y,
            d=d,
            x=x,
            z=z,
            county=county,
            lat=30 + county * 0.3,
            lon=110 + county * 0.25,
        )
    ).to_csv(OUT_CONLEY, index=False, float_format="%.17g")
    print(f"wrote {OUT_CONLEY}")


def nonpsd_twoway() -> None:
    """60 rows, 3 x 3 clusters: the two-way inclusion-exclusion matrix of a
    2SLS fit has negative variances (ivreg2 reports them as missing)."""
    rng = np.random.default_rng(0)
    m = 60
    a = rng.integers(0, 3, m)
    b = rng.integers(0, 3, m)
    z = rng.normal(size=m)
    x = rng.normal(size=m)
    u = rng.normal(size=m)
    d = z + u + rng.normal(size=m)
    y = d + x + u
    pd.DataFrame(dict(y=y, d=d, z=z, x=x, a=a, b=b)).to_csv(
        OUT_TWOWAY, index=False, float_format="%.17g"
    )
    print(f"wrote {OUT_TWOWAY}")


if __name__ == "__main__":
    main()
