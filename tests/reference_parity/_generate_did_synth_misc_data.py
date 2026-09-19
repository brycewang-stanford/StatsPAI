"""Write the fixed-seed CSVs behind ``test_did_synth_misc_parity.py``.

Run once from the repository root::

    python tests/reference_parity/_generate_did_synth_misc_data.py

The R generator ``_generate_did_synth_misc_R.R`` reads the same bytes, so the
two sides never see different data. Values are written with ``%.17g`` so the
CSV round-trips every double exactly.

Files
-----
did_synth_misc_harvest.csv
    Balanced staggered panel (150 units x 8 periods), cohorts 3 / 5 / 6 and
    never-treated (coded 0), cohort-specific and dynamic effects so that the
    cohort weights in every aggregation matter.
did_synth_misc_spill_single.csv
    One treatment cohort (period 3) on a 20 x 20 square, 4 periods, a direct
    effect plus two spillover rings (0, 2] and (2, 4] around the treated
    corner.
did_synth_misc_spill_stag.csv
    Two cohorts (3 and 5) in opposite corners, 6 periods. Each ring's
    spillover starts when the cluster it borders is treated -- the design in
    which the timing of ring exposure matters.
did_synth_misc_impact.csv
    One intervened series and two control series, 100 periods, intervention
    at period 71 (a CausalImpact-style design).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "_fixtures"


def _write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, float_format="%.17g")


def harvest_panel() -> pd.DataFrame:
    rng = np.random.default_rng(20260918)
    n, periods = 150, np.arange(1, 9)
    cohort = rng.choice([3, 5, 6, 0], size=n, p=[0.25, 0.2, 0.25, 0.3])
    alpha = rng.normal(0, 1, n)
    lam = rng.normal(0, 0.5, len(periods))
    base_eff = {3: 1.0, 5: 2.0, 6: 0.5}
    rows = []
    for i in range(n):
        g = cohort[i]
        for k, t in enumerate(periods):
            eff = 0.0
            if g != 0 and t >= g:
                eff = base_eff[g] + 0.3 * (t - g)
            y = alpha[i] + lam[k] + 0.1 * t * (g == 5) + eff + rng.normal(0, 0.7)
            rows.append((i + 1, t, g, y))
    return pd.DataFrame(rows, columns=["id", "t", "g", "y"])


def _spatial(rng, n, centres, radius):
    """Uniform points on [0, 20]^2; unit is in cluster k if within the
    square of half-width ``radius`` around centre k."""
    x = rng.uniform(0, 20, n)
    yc = rng.uniform(0, 20, n)
    member = np.full(n, -1)
    for k, (cx, cy) in enumerate(centres):
        inside = (np.abs(x - cx) < radius) & (np.abs(yc - cy) < radius)
        member[inside & (member == -1)] = k
    return x, yc, member


def spill_single() -> pd.DataFrame:
    rng = np.random.default_rng(20260919)
    n = 400
    x, yc, member = _spatial(rng, n, [(2.5, 2.5)], 2.5)
    treated = member == 0
    d = np.sqrt(
        (x[:, None] - x[treated][None, :]) ** 2
        + (yc[:, None] - yc[treated][None, :]) ** 2
    ).min(axis=1)
    fe = rng.normal(0, 1, n)
    rows = []
    for i in range(n):
        if treated[i]:
            eff = 2.0 + rng.normal(0, 0.3)
        elif d[i] <= 2:
            eff = 1.0
        elif d[i] <= 4:
            eff = 0.4
        else:
            eff = 0.0
        for t in (1, 2, 3, 4):
            val = fe[i] + 0.3 * t + (eff * (1 + 0.2 * (t - 3)) if t >= 3 else 0.0)
            val += rng.normal(0, 0.5)
            rows.append((i + 1, t, 3 if treated[i] else 0, x[i], yc[i], val))
    return pd.DataFrame(rows, columns=["id", "t", "g", "cx", "cy", "y"])


def spill_stag() -> pd.DataFrame:
    rng = np.random.default_rng(20260920)
    n = 600
    centres = [(2.5, 2.5), (17.5, 17.5)]
    cohorts = [3, 5]
    x, yc, member = _spatial(rng, n, centres, 2.5)
    treated = member >= 0
    g_unit = np.where(treated, np.array(cohorts)[np.maximum(member, 0)], 0)
    # Distance to each cluster's nearest treated unit.
    dist_k = []
    for k in range(2):
        tk = member == k
        dist_k.append(
            np.sqrt(
                (x[:, None] - x[tk][None, :]) ** 2
                + (yc[:, None] - yc[tk][None, :]) ** 2
            ).min(axis=1)
        )
    dist_k = np.column_stack(dist_k)
    near_k = dist_k.argmin(axis=1)
    near_d = dist_k.min(axis=1)
    fe = rng.normal(0, 1, n)
    rows = []
    for i in range(n):
        for t in range(1, 7):
            eff = 0.0
            if treated[i] and t >= g_unit[i]:
                eff = (2.0 if g_unit[i] == 3 else 3.0) + 0.25 * (t - g_unit[i])
            elif not treated[i]:
                onset = cohorts[near_k[i]]
                if t >= onset:
                    if near_d[i] <= 2:
                        eff = 1.0 + 0.2 * (t - onset)
                    elif near_d[i] <= 4:
                        eff = 0.4
            val = fe[i] + 0.2 * t + eff + rng.normal(0, 0.5)
            rows.append((i + 1, t, g_unit[i], x[i], yc[i], val))
    return pd.DataFrame(rows, columns=["id", "t", "g", "cx", "cy", "y"])


def impact_series() -> pd.DataFrame:
    rng = np.random.default_rng(20260921)
    n = 100
    x1 = 100 + np.cumsum(rng.normal(0, 1, n))
    x2 = 50 + np.cumsum(rng.normal(0, 0.5, n))
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.6 * ar[t - 1] + rng.normal(0, 1)
    y = 5 + 1.2 * x1 - 0.8 * x2 + ar
    y[70:] += 4.0
    return pd.DataFrame({"t": np.arange(1, n + 1), "y": y, "x1": x1, "x2": x2})


if __name__ == "__main__":
    _write(harvest_panel(), "did_synth_misc_harvest.csv")
    _write(spill_single(), "did_synth_misc_spill_single.csv")
    _write(spill_stag(), "did_synth_misc_spill_stag.csv")
    _write(impact_series(), "did_synth_misc_impact.csv")
