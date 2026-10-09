"""Coverage campaign (did, Oct 2026) -- spillover rings, continuous dose, overlap DiD.

``sp.spillover_did``, ``sp.cgs_continuous_did`` and ``sp.overlap_weighted_did``:
a known-truth recovery for each, the options that must reproduce another call
exactly, and every refusal by exception type and message.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

# ══════════════════════════════════════════════════════════════════════
#  sp.spillover_did
# ══════════════════════════════════════════════════════════════════════


def _ring_panel(seed=0, n_side=14, T=4, direct=2.0, spill=0.8, noise=0.1):
    """Units on a grid; a few are treated in period 3, neighbours spill."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    xs, ys = xs.ravel().astype(float), ys.ravel().astype(float)
    n = xs.size
    treated = np.zeros(n, dtype=bool)
    treated[rng.choice(n, size=8, replace=False)] = True
    d = np.sqrt((xs[:, None] - xs[None, :]) ** 2 + (ys[:, None] - ys[None, :]) ** 2)
    near = (d[:, treated].min(axis=1) <= 1.0) & ~treated
    rows = []
    for u in range(n):
        a = rng.normal()
        for t in range(1, T + 1):
            post = t >= 3
            eff = direct * (treated[u] and post) + spill * (near[u] and post)
            rows.append(
                {
                    "id": u,
                    "t": t,
                    "g": 3 if treated[u] else 0,
                    "lon": xs[u],
                    "lat": ys[u],
                    "y": a + 0.2 * t + eff + noise * rng.normal(),
                }
            )
    return pd.DataFrame(rows), d


S_KW = dict(y="y", unit="id", time="t", cohort="g", ring_edges=(0.0, 1.0))


@pytest.fixture(scope="module")
def ring_data():
    return _ring_panel()


def test_spillover_recovers_direct_and_ring_effects(ring_data):
    df, _ = ring_data
    r = sp.spillover_did(df, coords=["lon", "lat"], **S_KW)
    # 8 treated units, about 30 ring units, noise sd 0.1 on a two-period
    # difference: both standard errors are a few hundredths; 0.2 is ample.
    assert r.direct == pytest.approx(2.0, abs=0.2)
    assert r.rings["estimate"].iloc[0] == pytest.approx(0.8, abs=0.2)
    text = r.summary()
    assert "measured against the clean controls" in text
    assert "WARNING" not in text  # more than 30 clean controls
    d = r.to_dict()
    assert d["direct"] == r.direct and d["method"] == r.method


def test_spillover_distance_matrix_equals_coordinates(ring_data):
    df, dist = ring_data
    a = sp.spillover_did(df, coords=["lon", "lat"], **S_KW)
    b = sp.spillover_did(df, distances=dist, **S_KW)
    # the matrix is the Euclidean distance of the same coordinates
    assert b.direct == pytest.approx(a.direct, abs=1e-12)
    assert b.direct_se == pytest.approx(a.direct_se, rel=1e-12)
    pd.testing.assert_frame_equal(a.rings, b.rings)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(alpha=1.5), "alpha must be in"),
        (dict(ring_edges=(1.0, 0.5)), "must be increasing"),
        (dict(ring_edges=(1.0,)), "must be increasing"),
        (dict(y="nope"), "column 'nope' not in data"),
        (dict(coords=["lon"]), "exactly two columns"),
        (dict(coords=["lon", "nope"]), "coordinate column 'nope'"),
    ],
)
def test_spillover_argument_guards(ring_data, kwargs, match):
    df, _ = ring_data
    base = {**S_KW, "coords": ["lon", "lat"]}
    with pytest.raises(MethodIncompatibility, match=match):
        sp.spillover_did(df, **{**base, **kwargs})


def test_spillover_data_guards(ring_data):
    df, dist = ring_data
    with pytest.raises(MethodIncompatibility, match="`distances` must be"):
        sp.spillover_did(df, distances=dist[:5, :5], **S_KW)
    with pytest.raises(DataInsufficient, match="no treated units"):
        sp.spillover_did(df.assign(g=0), coords=["lon", "lat"], **S_KW)
    # adoption in the first period: no base period for any cohort
    with pytest.raises(DataInsufficient, match="no estimable"):
        sp.spillover_did(
            df.assign(g=np.where(df["g"] > 0, 1, 0)), coords=["lon", "lat"], **S_KW
        )


def test_spillover_warns_when_a_later_cohort_moves_a_unit_closer():
    # Three units on a line; the far one is treated first, the near one later.
    rng = np.random.default_rng(1)
    n = 60
    lon = np.arange(n, dtype=float)
    g = np.zeros(n, dtype=int)
    g[10] = 3  # treated first
    g[15] = 4  # treated later, closer to units 13..17
    rows = [
        {"id": u, "t": t, "g": g[u], "lon": lon[u], "lat": 0.0, "y": rng.normal()}
        for u in range(n)
        for t in range(1, 6)
    ]
    df = pd.DataFrame(rows)
    with pytest.warns(UserWarning, match="move to a closer ring"):
        sp.spillover_did(
            df,
            y="y",
            unit="id",
            time="t",
            cohort="g",
            coords=["lon", "lat"],
            ring_edges=(0.0, 2.0, 4.0),
        )


# ══════════════════════════════════════════════════════════════════════
#  sp.cgs_continuous_did
# ══════════════════════════════════════════════════════════════════════


def _dose_panel(seed=0, n_units=300, slope=0.4, noise=0.2, cohorts=(2,)):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        treated = rng.random() < 0.6
        g = int(rng.choice(cohorts)) if treated else 0
        dose = float(rng.uniform(0.5, 3.0)) if treated else 0.0
        a = rng.normal(scale=0.4)
        for t in (1, 2, 3):
            eff = slope * dose if (g and t >= g) else 0.0
            rows.append(
                {
                    "i": u,
                    "t": t,
                    "g": g,
                    "dose": dose,
                    "cl": u % 25,
                    "y": a + 0.15 * t + eff + noise * rng.normal(),
                }
            )
    return pd.DataFrame(rows)


D_KW = dict(y="y", dose="dose", time="t", unit="i", cohort="g", degree=1)


@pytest.fixture(scope="module")
def dose_data():
    return _dose_panel()


def test_cgs_recovers_linear_dose_response(dose_data):
    r = sp.cgs_continuous_did(dose_data, **D_KW)
    frame = r.to_frame()
    assert list(frame.columns) == ["dose", "att_d", "acrt_d"]
    # ATT(d) = 0.4 d, so the average causal response is 0.4 everywhere; with
    # 180 treated units and noise 0.2 the slope's SE is about 0.03.
    np.testing.assert_allclose(frame["acrt_d"], 0.4, atol=0.12)
    np.testing.assert_allclose(frame["att_d"], 0.4 * frame["dose"], atol=0.2)
    d = r.to_dict()
    assert d["overall_att"] == r.overall_att and d["method"] == r.method


def test_cgs_not_yet_treated_equals_never_treated_with_one_cohort(dose_data):
    # A single cohort leaves no not-yet-treated unit: same comparison group.
    a = sp.cgs_continuous_did(dose_data, **D_KW)
    b = sp.cgs_continuous_did(dose_data, control_group="notyettreated", **D_KW)
    assert b.overall_att == pytest.approx(a.overall_att, abs=1e-12)
    np.testing.assert_allclose(b.att_d, a.att_d, atol=1e-12)


def test_cgs_cohort_without_a_base_period_warns_and_contributes_nothing(dose_data):
    df = dose_data.copy()
    movers = df["i"] % 7 == 0
    df.loc[movers & (df["g"] == 2), "g"] = 1  # adopts in the first period
    ref = sp.cgs_continuous_did(df[df["g"] != 1], **D_KW)
    with pytest.warns(
        UserWarning, match="to use as the base period, so it contributes no cells"
    ):
        got = sp.cgs_continuous_did(df, **D_KW)
    # dropping the cohort by hand gives the same curve on the same grid
    assert got.overall_att == pytest.approx(ref.overall_att, abs=1e-10)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(control_group="clean"), "control_group must be"),
        (dict(alpha=0.0), "alpha must be in"),
        (dict(dose="nope"), "column 'nope' not in data"),
        (dict(cluster="nope"), "cluster column 'nope' not found"),
    ],
)
def test_cgs_argument_guards(dose_data, kwargs, match):
    with pytest.raises(MethodIncompatibility, match=match):
        sp.cgs_continuous_did(dose_data, **{**D_KW, **kwargs})


def test_cgs_no_estimable_cell(dose_data):
    df = dose_data.assign(g=np.where(dose_data["g"] > 0, 1, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(DataInsufficient, match="no estimable"):
            sp.cgs_continuous_did(df, **D_KW)


# ══════════════════════════════════════════════════════════════════════
#  sp.overlap_weighted_did
# ══════════════════════════════════════════════════════════════════════


def _two_by_two(seed=0, n=600):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    treat = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x))).astype(int)
    post = rng.integers(0, 2, n)
    y = 1 + 0.5 * x + treat + 0.5 * post + 2.0 * treat * post + rng.normal(size=n)
    return pd.DataFrame({"y": y, "treat": treat, "post": post, "x": x})


O_KW = dict(y="y", treat="treat", time="post", covariates=["x"])


def test_overlap_did_custom_classifier_equals_builtin_logit():
    df = _two_by_two()
    builtin = sp.overlap_weighted_did(df, **O_KW)
    custom = sp.overlap_weighted_did(
        df, ps_model=LogisticRegression(max_iter=1000, solver="lbfgs"), **O_KW
    )
    # the same sklearn estimator with the same settings
    assert custom.estimate == pytest.approx(builtin.estimate, abs=1e-12)
    # homogeneous effect of 2; n = 600 with unit noise gives an SE near 0.17
    assert builtin.estimate == pytest.approx(2.0, abs=0.6)


def test_overlap_did_guards():
    df = _two_by_two()
    with pytest.raises(ValueError, match="Missing columns"):
        sp.overlap_weighted_did(df.drop(columns="x"), **O_KW)
    with pytest.raises(ValueError, match="all 4 \\(treat, time\\) cells"):
        sp.overlap_weighted_did(df[~((df["treat"] == 1) & (df["post"] == 1))], **O_KW)
