"""Standard error of ``sp.match(method='cem')`` and the bootstrap diagnostic.

CEM compares treated and control units inside coarsened cells, so the
outcome variation *between* cells never enters the estimate. Through
1.43.1 the ATT's standard error was built from each arm's variance around
its overall mean, which includes that between-cell variation; in the SE
calibration screen (``tests/reliability/se_screen/causal/matching.py``)
it was 1.42 times the sampling standard deviation. It is now the
within-cell variance the documentation always described.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _cells(seed: int = 0, n: int = 600, noise: float = 1.0) -> pd.DataFrame:
    """Two binary covariates: four cells, matched exactly by CEM."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    d = (rng.uniform(size=n) < 0.3 + 0.2 * a + 0.1 * b).astype(int)
    y = 5.0 * a - 3.0 * b + 2.0 * d + noise * rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "a": a, "b": b})


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.match(df, "y", "d", ["a", "b"], method="cem", n_bins=2, **kw)


def test_cem_att_se_is_the_within_cell_formula():
    df = _cells()
    r = _fit(df)
    n_t_total = int(df["d"].sum())
    ss = {0: 0.0, 1: 0.0}
    dfree = {0: 0, 1: 0}
    counts = []
    for _, cell in df.groupby(["a", "b"]):
        for arm in (0, 1):
            yy = cell.loc[cell["d"] == arm, "y"].to_numpy()
            ss[arm] += float(((yy - yy.mean()) ** 2).sum())
            dfree[arm] += len(yy) - 1
        counts.append(((cell["d"] == 1).sum(), (cell["d"] == 0).sum()))
    sig_t, sig_c = ss[1] / dfree[1], ss[0] / dfree[0]
    var = sum((nt / n_t_total) ** 2 * (sig_t / nt + sig_c / nc) for nt, nc in counts)
    assert r.se == pytest.approx(np.sqrt(var), rel=1e-10)


def test_cem_att_se_vanishes_when_cells_explain_the_outcome():
    """No noise inside a cell: the matched comparison is exact and so is
    the estimate. The old formula reported the between-cell spread."""
    df = _cells(noise=0.0)
    r = _fit(df)
    assert r.estimate == pytest.approx(2.0, abs=1e-10)
    assert r.se <= 1e-10 * np.abs(df["y"]).max()


def test_cem_att_se_tracks_the_sampling_spread():
    """300 draws: mean SE over the SD of the estimate. The old SE also
    counted the cell effects (5 and 3 against unit noise) and was far
    above the sampling spread."""
    est, se = [], []
    for seed in range(300):
        r = _fit(_cells(seed=seed, n=400))
        est.append(r.estimate)
        se.append(r.se)
    ratio = np.mean(se) / np.std(est, ddof=1)
    # Monte Carlo error of the ratio at 300 draws is about 4%.
    assert 0.88 < ratio < 1.12


def test_bootstrap_reports_an_interquartile_scale():
    rng = np.random.default_rng(0)
    n = 600
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x1 + 0.5 * x2))).astype(int)
    y = 1 + x1 + 0.5 * x2 + d + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.match(
            df,
            "y",
            "d",
            ["x1", "x2"],
            method="kernel",
            se_method="bootstrap",
            bootstrap_reps=200,
            bootstrap_seed=1,
        )
    # A smooth estimator: the two scales agree (their ratio on 200 normal
    # draws has a standard error of about 8%).
    assert r.model_info["bootstrap_se_iqr"] == pytest.approx(r.se, rel=0.3)
    assert "bootstrap_heavy_tails" not in r.model_info


def test_llr_with_a_narrow_bandwidth_warns_about_heavy_tails():
    """Local linear weights explode when a window holds few controls; the
    reported SE is then several times the interquartile scale."""
    rng = np.random.default_rng(0)
    n = 150
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x1 + 0.5 * x2))).astype(int)
    y = 1 + x1 + 0.5 * x2 + (1 + 0.5 * x1) * d + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    with pytest.warns(UserWarning, match="heavy tails"):
        r = sp.match(
            df,
            "y",
            "d",
            ["x1", "x2"],
            method="llr",
            kernel="tricube",
            bwidth=0.03,
            bootstrap_reps=60,
            bootstrap_seed=1,
        )
    assert r.model_info["bootstrap_heavy_tails"] is True
    assert r.se > 2.0 * r.model_info["bootstrap_se_iqr"]
