"""Coverage round-2 — ``statspai.rd.locrand`` (local-randomization RD).

Round 1 covered the main rdrandinf / rdwinselect / rdsensitivity / rdrbounds
paths. This file adds:

- ``rdwinselect`` *without* user covariates (pseudo-covariate construction +
  cleanup) and with polynomial adjustment;
- ``rdsensitivity`` with a tiny window in the grid (the <2-obs NaN row);
- ``rdrbounds`` error branches (tiny window, gamma < 1) and gamma=1 path;
- ``rdrandinf`` error branches (too few obs, unbalanced sides, bad statistic).

Real synthetic RD data; assertions check structural properties (p-values in
[0,1], balance flags boolean, errors raise) — never fabricated numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _df(seed=4, n=1200, tau=3.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 0.5 * x + tau * (x >= 0) + rng.normal(0, 0.4, n)
    z = rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "z2": rng.normal(size=n)})


def test_rdwinselect_pseudo_covariates_and_poly():
    df = _df()
    # no covs -> pseudo quantile covariates built and cleaned up afterwards
    out = sp.rdwinselect(df, x="x", c=0, nwindows=5, p=1, seed=1)
    assert {"window_left", "window_right", "p_value", "balanced"}.issubset(out.columns)
    assert out["balanced"].dtype == bool
    # pseudo-cov columns must not leak back into the user's frame
    assert not any(col.startswith("_x_q") for col in df.columns)


def test_rdwinselect_with_covs_baseline():
    df = _df()
    out = sp.rdwinselect(df, x="x", c=0, covs=["z", "z2"], nwindows=4, seed=2)
    assert len(out) == 4
    pv = out["p_value"].dropna()
    assert ((pv >= 0) & (pv <= 1)).all()


def test_rdsensitivity_tiny_window_row():
    df = _df()
    # include a near-zero window so n_left/n_right < 2 -> NaN row branch
    out = sp.rdsensitivity(
        df, y="y", x="x", c=0, wlist=[1e-6, 0.2, 0.4], n_perms=120, seed=3
    )
    assert out["estimate"].isna().any()
    fin = out["estimate"].dropna()
    assert len(fin) >= 1


def test_rdrbounds_gamma_one_and_bounds():
    df = _df()
    out = sp.rdrbounds(
        df,
        y="y",
        x="x",
        c=0,
        wl=-0.3,
        wr=0.3,
        gamma_list=[1.0, 2.0],
        n_perms=120,
        seed=5,
    )
    assert {"gamma", "pvalue_upper", "pvalue_lower"}.issubset(out.columns)
    pu = out["pvalue_upper"]
    assert ((pu >= 0) & (pu <= 1)).all()


def test_rdrbounds_window_too_small_raises():
    df = _df()
    with pytest.raises(ValueError, match="observations"):
        sp.rdrbounds(df, y="y", x="x", c=0, wl=-1e-6, wr=1e-6, n_perms=50)


def test_rdrbounds_gamma_below_one_raises():
    df = _df()
    with pytest.raises(ValueError, match="gamma"):
        sp.rdrbounds(
            df, y="y", x="x", c=0, wl=-0.3, wr=0.3, gamma_list=[0.5], n_perms=50
        )


def test_rdrandinf_too_few_obs_raises():
    df = _df()
    with pytest.raises(ValueError, match="observations"):
        sp.rdrandinf(df, y="y", x="x", c=0, wl=-1e-7, wr=1e-7, n_perms=50)


def test_rdrandinf_bad_statistic_raises():
    df = _df()
    with pytest.raises(ValueError, match="[Uu]nknown statistic"):
        sp.rdrandinf(
            df, y="y", x="x", c=0, wl=-0.3, wr=0.3, statistic="not_a_stat", n_perms=50
        )


def test_rdrandinf_unbalanced_sides_raises():
    rng = np.random.default_rng(0)
    # almost all observations on the right side within the window
    x = np.concatenate([np.array([-0.29]), rng.uniform(0.01, 0.3, 200)])
    y = rng.normal(size=len(x))
    df = pd.DataFrame({"y": y, "x": x})
    with pytest.raises(ValueError, match="each side"):
        sp.rdrandinf(df, y="y", x="x", c=0, wl=-0.3, wr=0.3, n_perms=50)
