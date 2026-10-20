"""sp.spillover reports the cell contrasts it documents (2026-10-19).

The function was labelled Hudgens-Halloran (2008) and IPW; it computes
differences in means between own-treatment x peer-exposure cells split at
the median positive exposure. These tests pin that definition by hand, and
the loud failures that replaced silent behaviour (ignored covariates,
unknown exposure_fn falling back to 'fraction', an empty cell reported as a
zero effect).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _data(seed=0, n_clusters=150, size=6, tau=1.5, gamma=2.0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        d = rng.binomial(1, 0.5, size)
        for i in range(size):
            frac = (d.sum() - d[i]) / (size - 1)
            y = 1.0 + tau * d[i] + gamma * frac + rng.normal()
            rows.append({"cl": c, "d": int(d[i]), "frac": frac, "y": y})
    return pd.DataFrame(rows)


def test_contrasts_equal_hand_computed_cell_means():
    df = _data()
    res = sp.spillover(df, y="y", treat="d", cluster="cl", n_bootstrap=50)
    pos = df["frac"][df["frac"] > 0]
    c = float(np.median(pos))
    hi = df["frac"] >= c
    m = lambda mask: df.loc[mask, "y"].mean()  # noqa: E731
    d1h, d0h, d0l = (
        m((df.d == 1) & hi),
        m((df.d == 0) & hi),
        m((df.d == 0) & ~hi),
    )
    mi = res.model_info
    assert mi["median_exposure"] == pytest.approx(c, rel=1e-12)
    assert mi["direct_effect"] == pytest.approx(d1h - d0h, rel=1e-12)
    assert mi["spillover_effect"] == pytest.approx(d0h - d0l, rel=1e-12)
    assert mi["total_effect"] == pytest.approx(d1h - d0l, rel=1e-12)
    assert mi["cell_sizes"]["untreated_low"] == int(((df.d == 0) & ~hi).sum())
    assert "Hudgens" not in res.method


def test_direct_contrast_recovers_own_effect_under_bernoulli_assignment():
    """Bernoulli assignment: D independent of exposure, so the direct
    contrast estimates tau = 1.5 (and the spillover contrast gamma times the
    gap in mean exposure between the strata)."""
    df = _data(seed=3, n_clusters=400)
    res = sp.spillover(df, y="y", treat="d", cluster="cl", n_bootstrap=200)
    mi = res.model_info
    assert abs(mi["direct_effect"] - 1.5) < 4 * mi["direct_se"]
    c = mi["median_exposure"]
    u = df[df.d == 0]
    gap = u.frac[u.frac >= c].mean() - u.frac[u.frac < c].mean()
    assert abs(mi["spillover_effect"] - 2.0 * gap) < 4 * mi["spillover_se"]


def test_covariates_and_unknown_exposure_raise():
    df = _data(n_clusters=40).assign(x=0.0)
    with pytest.raises(sp.MethodIncompatibility, match="covariates"):
        sp.spillover(df, y="y", treat="d", cluster="cl", covariates=["x"])
    with pytest.raises(sp.MethodIncompatibility, match="exposure_fn"):
        sp.spillover(df, y="y", treat="d", cluster="cl", exposure_fn="share")


def test_empty_cell_raises_instead_of_reporting_zero():
    # Every untreated unit has all cluster-mates treated: no untreated
    # low-exposure cell, so the spillover contrast does not exist.
    rows = []
    for c in range(30):
        for i in range(4):
            d = int(i > 0)
            rows.append({"cl": c, "d": d, "y": float(d) + 0.1 * i + c * 0.01})
    with pytest.raises(sp.DataInsufficient, match="untreated_low"):
        sp.spillover(pd.DataFrame(rows), y="y", treat="d", cluster="cl")


def test_missing_outcomes_keep_their_treatment_in_peer_exposure():
    # 2026-10-20 probe: rows with a missing outcome were dropped before the
    # exposure was computed, so their treatment vanished from their
    # cluster-mates' exposure. The exposure is the full-cluster ``frac``.
    df = _data(seed=2)
    rng = np.random.default_rng(9)
    holes = rng.choice(len(df), 60, replace=False)
    df.loc[holes, "y"] = np.nan
    with pytest.warns(UserWarning, match="missing outcome"):
        res = sp.spillover(df, y="y", treat="d", cluster="cl", n_bootstrap=50)
    obs = df[df["y"].notna()]
    c = float(np.median(obs["frac"][obs["frac"] > 0]))
    hi = obs["frac"] >= c
    m = lambda mask: obs.loc[mask, "y"].mean()  # noqa: E731
    mi = res.model_info
    assert mi["median_exposure"] == pytest.approx(c, rel=1e-12)
    assert mi["direct_effect"] == pytest.approx(
        m((obs.d == 1) & hi) - m((obs.d == 0) & hi), rel=1e-12
    )
    assert res.n_obs == len(obs)


def test_missing_treatment_is_refused():
    df = _data(seed=3)
    df["d"] = df["d"].astype(float)
    df.loc[5, "d"] = np.nan
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="missing treatment"):
        sp.spillover(df, y="y", treat="d", cluster="cl", n_bootstrap=10)
