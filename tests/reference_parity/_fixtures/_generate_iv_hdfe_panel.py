"""Generate the county-month HDFE-IV panel fixture (``iv_hdfe_panel.csv``).

The design mirrors the empirical shape of Zhang et al. (2026, *Science*
393:831-836, doi:10.1126/science.aee0747): a county x year-month panel where a
county-level *policy-intensity index* is instrumented by (historical solar
resource) x (inverse climate-policy uncertainty), with county and year-month
fixed effects and county-clustered inference. Sizes are scaled down so the
fixture stays committable.

Run with the repo venv::

    python tests/reference_parity/_fixtures/_generate_iv_hdfe_panel.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

N_COUNTY, N_MONTH = 120, 24
SEED = 20260823


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    county = np.repeat(np.arange(N_COUNTY), N_MONTH)
    ym = np.tile(np.arange(N_MONTH), N_COUNTY)
    n = county.size

    # County-level primitives (time-invariant).
    sun = rng.normal(size=N_COUNTY)  # historical solar resource
    lat = rng.uniform(22.0, 48.0, size=N_COUNTY)
    lon = rng.uniform(80.0, 128.0, size=N_COUNTY)
    poor = (rng.uniform(size=N_COUNTY) < 0.35).astype(int)
    prov = np.repeat(np.arange(N_COUNTY // 10), 10)
    alpha = rng.normal(scale=1.5, size=N_COUNTY)  # county FE

    # Time-level primitives.
    cpu_inv = rng.normal(size=N_MONTH)  # inverse policy uncertainty
    delta = rng.normal(scale=0.8, size=N_MONTH)  # year-month FE

    z = sun[county] * cpu_inv[ym]  # interaction instrument
    # Second, weaker interaction instrument so the over-identified spec has
    # something to say (Hansen J, LIML vs 2SLS vs GMM2s all differ).
    z2 = sun[county] * (cpu_inv[ym] ** 2 - 1.0)
    u = rng.normal(size=n)  # structural confounder
    temp = rng.normal(size=n)
    wind = rng.normal(size=n)

    policy = (
        1.10 * z
        + 0.45 * z2
        + 0.70 * u
        + 0.20 * temp
        + alpha[county]
        + delta[ym]
        + rng.normal(size=n)
    )
    shannon = (
        2.44
        - 0.0125 * policy
        + 0.60 * u
        + 0.15 * temp
        - 0.05 * wind
        + alpha[county]
        + delta[ym]
        + rng.normal(scale=0.5, size=n)
    )
    ndvi = -0.02 * policy + 0.3 * u + alpha[county] + delta[ym] + rng.normal(size=n)
    lai = 0.03 * policy + 0.3 * u + alpha[county] + delta[ym] + rng.normal(size=n)

    return pd.DataFrame(
        {
            "county": county,
            "ym": ym,
            "prov": prov[county],
            "poor": poor[county],
            "lat": lat[county],
            "lon": lon[county],
            "sun": sun[county],
            "cpu_inv": cpu_inv[ym],
            "z": z,
            "z2": z2,
            "policy": policy,
            "shannon": shannon,
            "ndvi": ndvi,
            "lai": lai,
            "temp": temp,
            "wind": wind,
        }
    )


if __name__ == "__main__":
    out = pathlib.Path(__file__).with_name("iv_hdfe_panel.csv")
    build().to_csv(out, index=False, float_format="%.10f")
    print(f"wrote {out} ({len(build())} rows)")
