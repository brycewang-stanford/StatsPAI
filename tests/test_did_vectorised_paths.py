"""The vectorised paths of event_study, lp_did and did_multiplegt_dyn.

Each of the three used to loop over units in Python or pandas. The loops are
gone from the hot path; these tests state what the replacements must equal --
the per-cluster, per-unit or per-frame definition, written out plainly here or
kept in the module as the fallback -- and bound the run time so a return to
quadratic cost is noticed.

Tolerances: sums are compared at 1e-12 relative. The replacements add in the
same order as the loops, so on one machine they agree bit for bit; the slack
is for BLAS and SIMD differences across platforms.
"""

from __future__ import annotations

import importlib
import time
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did._core import cluster_bootstrap_draw

es = importlib.import_module("statspai.did.event_study")
lp = importlib.import_module("statspai.did.lp_did")
dyn = importlib.import_module("statspai.did.did_multiplegt_dyn")
arr = importlib.import_module("statspai.did._dcdh_arrays")

RTOL = 1e-12


def _panel(n_units=60, n_periods=10, seed=0, drop=0.0, gap=False, switch_off=False):
    """Staggered panel: cohorts 4 / 6 / 8, 40% never treated."""
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(1, n_periods + 1), n_units)
    cohort = rng.choice([4, 6, 8, np.nan], size=n_units, p=[0.2, 0.2, 0.2, 0.4])
    tt = cohort[ids]
    start = np.nan_to_num(tt, nan=1e9)
    treated = (t >= start).astype(float)
    if switch_off:
        back = rng.random(n_units) < 0.3
        treated = np.where(back[ids] & (t >= start + 2), 0.0, treated)
    x1 = rng.normal(size=len(ids))
    y = rng.normal(size=n_units)[ids] + 0.3 * t + 1.5 * treated + 0.5 * x1
    y = y + rng.normal(size=len(ids))
    d = pd.DataFrame(
        {
            "id": ids,
            "t": t,
            "tt": tt,
            "treated": treated,
            "y": y,
            "x1": x1,
            "w": rng.uniform(0.5, 2.0, size=len(ids)),
            "st": ids % 7,
        }
    )
    if drop:
        d = d[rng.random(len(d)) > drop]
    if gap:
        d = d[~((d["id"] % 3 == 0) & (d["t"] == 5))]
    return d.reset_index(drop=True)


# ---------------------------------------------------------------------------
# event_study
# ---------------------------------------------------------------------------


def test_cluster_meat_is_the_sum_of_per_cluster_outer_products():
    rng = np.random.default_rng(1)
    n, k = 137, 4
    X = rng.normal(size=(n, k))
    resid = rng.normal(size=n)
    w = rng.uniform(0.2, 3.0, size=n)
    clusters = rng.integers(0, 11, size=n)  # unequal cluster sizes
    XtX_inv = np.linalg.inv(X.T @ X)

    for weights in (None, w):
        meat = np.zeros((k, k))
        for c in np.unique(clusters):
            m = clusters == c
            u = resid[m] if weights is None else np.sqrt(weights[m]) * resid[m]
            score = (X[m] * u[:, None]).sum(axis=0)
            meat += np.outer(score, score)
        G = len(np.unique(clusters))
        factor = (G / (G - 1)) * ((n - 1) / (n - k - 3))
        want = factor * XtX_inv @ meat @ XtX_inv
        want = 0.5 * (want + want.T)

        se, vcov = es._cluster_se(X, resid, XtX_inv, clusters, w=weights, k_fe=3)
        np.testing.assert_allclose(vcov, want, rtol=RTOL, atol=0)
        np.testing.assert_allclose(se, np.sqrt(np.diag(want)), rtol=RTOL)


def test_a_missing_cluster_label_counts_as_a_cluster_with_no_score():
    # the loop it replaced: np.unique keeps NaN as a group, `ids == nan`
    # selects no row
    mat = np.arange(12.0).reshape(6, 2)
    ids = np.array([1.0, np.nan, 1.0, 2.0, np.nan, 2.0])
    sums = es._group_sums(mat, ids)
    assert sums.shape == (3, 2)
    np.testing.assert_array_equal(sums[0], mat[[0, 2]].sum(axis=0))
    np.testing.assert_array_equal(sums[1], mat[[3, 5]].sum(axis=0))
    np.testing.assert_array_equal(sums[2], [0.0, 0.0])


@pytest.mark.parametrize("weighted", [False, True])
def test_twfe_demeaning_is_one_unit_sweep_then_one_period_sweep(weighted):
    d = _panel(seed=2, drop=0.2)
    w = d["w"].to_numpy().copy() if weighted else None
    if weighted:
        w[d["id"].to_numpy() == 3] = 0.0  # zero-weight unit: plain mean
    cols = ["y", "x1", "treated"]

    want = d[cols].to_numpy(dtype=float)
    for ids in (d["id"].to_numpy(), d["t"].to_numpy()):
        for g in np.unique(ids):
            m = ids == g
            if w is not None and w[m].sum() > 0:
                mean = (w[m][:, None] * want[m]).sum(axis=0) / w[m].sum()
            else:
                mean = want[m].mean(axis=0)
            want[m] -= mean

    Y, X, names = es._demean_twfe(d, "y", ["x1", "treated"], "id", "t", w=w)
    assert names == ["x1", "treated"]
    np.testing.assert_allclose(Y, want[:, 0], rtol=RTOL, atol=1e-13)
    np.testing.assert_allclose(X, want[:, 1:], rtol=RTOL, atol=1e-13)


def test_event_study_matches_dummy_variable_ols_on_an_unbalanced_panel():
    # On an unbalanced panel one unit sweep and one period sweep is not the
    # two-way projection, so the reference here is the estimator's own
    # definition: OLS of the swept outcome on the swept event-time dummies,
    # CR1 by unit with the unit effects nested in the cluster.
    d = _panel(n_units=80, seed=3, drop=0.1)
    r = sp.event_study(d, "y", "tt", "t", "id", window=(-2, 2))
    tab = r.model_info["event_study"]
    est = tab[~tab["is_reference"]]

    rel = (d["t"] - d["tt"]).clip(-2, 2)
    dums = np.column_stack([(rel == k).to_numpy(dtype=float) for k in (-2, 0, 1, 2)])
    Z = np.column_stack([d["y"].to_numpy(), dums])
    for ids in (d["id"].to_numpy(), d["t"].to_numpy()):
        for g in np.unique(ids):
            m = ids == g
            Z[m] -= Z[m].mean(axis=0)
    yv, X = Z[:, 0], Z[:, 1:]
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ yv
    u = yv - X @ beta
    meat = np.zeros((4, 4))
    for g in np.unique(d["id"]):
        m = (d["id"] == g).to_numpy()
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G, n = d["id"].nunique(), len(d)
    # nested rule: the unit effects sit inside the cluster and are not
    # counted, the period effects are, level for level
    k_fe = d["t"].nunique()
    V = (G / (G - 1)) * ((n - 1) / (n - 4 - k_fe)) * bread @ meat @ bread

    np.testing.assert_allclose(est["att"].to_numpy(), beta, rtol=1e-10)
    np.testing.assert_allclose(est["se"].to_numpy(), np.sqrt(np.diag(V)), rtol=1e-10)
    np.testing.assert_allclose(r.model_info["vcov"], V, rtol=1e-9, atol=1e-14)


# ---------------------------------------------------------------------------
# lp_did
# ---------------------------------------------------------------------------


def _sorted_for_lp(d):
    return d.sort_values(["id", "t"]).reset_index(drop=True)


@pytest.mark.parametrize("lo,hi", [(-1, 0), (-1, 3), (-4, 0), (-2, 5), (-1, 9)])
def test_stable_zero_window_equals_the_per_unit_definition(lo, hi):
    d = _sorted_for_lp(_panel(seed=4, drop=0.15, gap=True, switch_off=True))
    d.loc[d.index[::17], "treated"] = np.nan  # a missing treatment is not 0
    got = lp._stable_zero_window(d, "id", "t", "treated", lo, hi)
    want = lp._stable_zero_window_by_unit(d, "id", "t", "treated", lo, hi)
    np.testing.assert_array_equal(got, want)
    assert got.any() or hi >= 9


def test_stable_zero_window_counts_rows_not_calendar_periods():
    # unit 0 misses period 3: its window at t=4 reaches back one ROW, to t=2
    d = pd.DataFrame(
        {
            "id": [0, 0, 0, 0, 1, 1, 1, 1, 1],
            "t": [1, 2, 4, 5, 1, 2, 3, 4, 5],
            "treated": [0, 0, 0, 1, 0, 0, 0, 0, 0],
        }
    )
    got = lp._stable_zero_window(d, "id", "t", "treated", -1, 0)
    np.testing.assert_array_equal(
        got, [False, True, True, False, False, True, True, True, True]
    )


def test_lp_did_sample_equals_a_per_unit_construction():
    d = _sorted_for_lp(_panel(seed=5, drop=0.1, gap=True))
    d["_d_prev"] = d.groupby("id")["treated"].shift(1)
    d["_delta_d"] = d["treated"] - d["_d_prev"]
    h = 2
    got = lp._build_lp_did_sample(
        df=d,
        y="y",
        unit="id",
        time="t",
        treatment="treated",
        h=h,
        controls=["x1"],
        clean_controls="not_yet_treated",
        never_treated_ids=None,
    )

    rows = []
    for uid, u in d.groupby("id"):
        u = u.reset_index(drop=True)
        for k in range(1, len(u) - h):
            window = u["treated"].iloc[k - 1 : k + h + 1].to_numpy()
            switch_on = u["treated"].iloc[k] - u["treated"].iloc[k - 1] == 1
            clean = bool(np.all(window == 0))
            if switch_on or clean:
                rows.append(
                    (
                        uid,
                        u["t"].iloc[k],
                        u["y"].iloc[k + h] - u["y"].iloc[k - 1],
                        1.0 if switch_on else 0.0,
                        u["x1"].iloc[k],
                    )
                )
    want = pd.DataFrame(rows, columns=["id", "t", "_dy", "_delta_d", "x1"])
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), want, check_dtype=False, rtol=RTOL
    )


def test_lp_did_cluster_variance_equals_the_per_cluster_sum():
    rng = np.random.default_rng(6)
    n = 150
    sample = pd.DataFrame(
        {
            "dy": rng.normal(size=n),
            "dd": (rng.random(n) < 0.3).astype(float),
            "x": rng.normal(size=n),
            "t": rng.integers(1, 6, size=n),
            "c": rng.integers(0, 13, size=n),
        }
    )
    scores: dict = {}
    beta, se, n_obs = lp._ols_with_cluster_se(
        sample,
        y_col="dy",
        x_cols=["dd", "x"],
        time_col="t",
        cluster_col="c",
        time_fe=True,
        scores_out=scores,
    )

    times = np.unique(sample["t"])
    X = np.column_stack(
        [sample["dd"], sample["x"]]
        + [(sample["t"] == t).astype(float) for t in times[1:]]
        + [np.ones(n)]
    )
    yv = sample["dy"].to_numpy()
    bread = np.linalg.pinv(X.T @ X)
    b = bread @ X.T @ yv
    u = yv - X @ b
    k = X.shape[1]
    meat = np.zeros((k, k))
    for c in np.unique(sample["c"]):
        m = (sample["c"] == c).to_numpy()
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G = sample["c"].nunique()
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * bread @ meat @ bread

    assert n_obs == n
    np.testing.assert_allclose(beta, b[0], rtol=RTOL)
    np.testing.assert_allclose(se, np.sqrt(V[0, 0]), rtol=RTOL)
    # the per-cluster influences reproduce the variance they are stored for
    np.testing.assert_allclose(sum(v * v for v in scores.values()), V[0, 0], rtol=1e-11)
    assert list(scores) == list(np.unique(sample["c"]))


# ---------------------------------------------------------------------------
# did_multiplegt_dyn
# ---------------------------------------------------------------------------


def _dcdh_frame(d):
    """The frame did_multiplegt_dyn hands to its event code."""
    df = d.copy()
    df["_tidx"] = pd.factorize(df["t"], sort=True)[0] + 1
    df = df.sort_values(["id", "_tidx"]).reset_index(drop=True)
    for part in dyn._first_switch(df, group="id", time="_tidx", treatment="treated"):
        df = df.merge(part, on="id", how="left")
    return df


def _gappy_dcdh_panel(seed=7):
    d = _panel(n_units=70, seed=seed, drop=0.1, gap=True, switch_off=True)
    d.loc[d.index[::11], "w"] = 0.0
    d.loc[d.index[::23], "y"] = np.nan
    return d


def test_first_switch_equals_the_per_unit_definition():
    d = _gappy_dcdh_panel()
    d.loc[d.index[::29], "treated"] = np.nan
    d["_tidx"] = d["t"]
    got = dyn._first_switch(d, group="id", time="_tidx", treatment="treated")
    want = dyn._first_switch_by_unit(d, group="id", time="_tidx", treatment="treated")
    for a, b in zip(got, want):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert len(got[0]) > 10 and set(got[1]["_dir"]) == {1, -1}


@pytest.mark.parametrize("weights", [None, "w"])
def test_array_event_sample_equals_the_frame_definition(weights):
    df = _dcdh_frame(_gappy_dcdh_panel())
    panel = arr.build_panel(
        df,
        y="y",
        group="id",
        time="_tidx",
        treatment="treated",
        weights=weights,
        cluster=None,
    )
    assert panel is not None and panel.ordered
    compared = 0
    for t_pre, t_post, t_anchor in [(3, 4, 4), (3, 7, 7), (1, 3, 5), (5, 6, 6)]:
        for ids in (set(range(0, 70, 2)), set(range(5, 60))):
            frame = dyn._event_sample(
                df, "id", "_tidx", "y", weights, ids, t_pre, t_post, t_anchor
            )
            mask = panel.labels.isin(list(ids))
            got = arr._sample(
                panel, mask, panel.col(t_pre), panel.col(t_post), panel.col(t_anchor)
            )
            assert (frame is None) == (got is None)
            if frame is None:
                continue
            f_ids, f_dy, f_w = frame
            idx, dy, w = got
            assert list(panel.labels[idx]) == list(f_ids)
            np.testing.assert_array_equal(dy, f_dy)
            np.testing.assert_array_equal(w, f_w)
            compared += 1
    assert compared >= 6


def test_array_event_sample_follows_frame_row_order_when_units_are_shuffled():
    # units stacked in a different order than their labels sort in: the
    # sample must list them as the frame does, which fixes the summation order
    df = _dcdh_frame(_gappy_dcdh_panel(seed=8))
    order = np.random.default_rng(0).permutation(df["id"].unique())
    df = pd.concat([df[df["id"] == u] for u in order], ignore_index=True)
    panel = arr.build_panel(
        df,
        y="y",
        group="id",
        time="_tidx",
        treatment="treated",
        weights="w",
        cluster=None,
    )
    assert panel is not None and not panel.ordered
    ids = set(range(0, 70))
    f_ids, f_dy, f_w = dyn._event_sample(df, "id", "_tidx", "y", "w", ids, 3, 6, 6)
    idx, dy, w = arr._sample(
        panel, np.ones(panel.n, dtype=bool), panel.col(3), panel.col(6), panel.col(6)
    )
    assert list(panel.labels[idx]) == list(f_ids)
    assert list(f_ids) != sorted(f_ids)
    np.testing.assert_array_equal(dy, f_dy)
    np.testing.assert_array_equal(w, f_w)


@pytest.mark.parametrize(
    "opts",
    [
        dict(),
        dict(weights="w", cluster="st"),
        dict(same_switchers=True, normalized=True),
        dict(control="never_treated", switchers="in"),
    ],
)
def test_array_estimates_equal_the_frame_definition(opts):
    d = _gappy_dcdh_panel(seed=9)
    df = _dcdh_frame(d)
    kw = dict(
        df=df,
        y="y",
        group="id",
        time="_tidx",
        treatment="treated",
        horizons=[-2, -1, 0, 1, 2],
        control=opts.get("control", "not_yet_treated"),
        switchers=opts.get("switchers"),
        same_switchers=opts.get("same_switchers", False),
        weights=opts.get("weights"),
        cluster=opts.get("cluster", "id"),
        normalized=opts.get("normalized", False),
    )
    got = dyn._estimate_all_horizons(**kw)
    want = dyn._estimate_all_horizons_frame(**kw)
    assert any(c["n_events"] for c in want["cell_estimates"])
    np.testing.assert_array_equal(got["cluster_codes"], want["cluster_codes"])
    for a, b in zip(got["cell_estimates"], want["cell_estimates"]):
        assert a["horizon"] == b["horizon"]
        assert a["n_switchers"] == b["n_switchers"] and a["n_events"] == b["n_events"]
        for key in ("delta_l", "w_switchers", "_se_analytic", "_delta_raw", "dose_now"):
            np.testing.assert_allclose(a[key], b[key], rtol=RTOL, equal_nan=True)
        for key in ("_influence", "_influence_raw", "_lag_dose"):
            np.testing.assert_allclose(
                a[key], b[key], rtol=RTOL, atol=1e-13, equal_nan=True
            )
    for h, effects in want["group_effects"].items():
        pd.testing.assert_series_equal(got["group_effects"][h], effects, rtol=RTOL)


@pytest.mark.parametrize("cluster", ["id", "st"])
def test_array_bootstrap_replicate_is_the_resampled_frame(cluster):
    d = _gappy_dcdh_panel(seed=10)
    df = _dcdh_frame(d)
    horizons = [-1, 0, 1, 2]
    panel = arr.build_panel(
        df,
        y="y",
        group="id",
        time="_tidx",
        treatment="treated",
        weights="w",
        cluster=cluster,
    )
    resampler = arr.Resampler(panel, df, cluster)
    rng_a, rng_b = np.random.default_rng(11), np.random.default_rng(11)
    for _ in range(4):
        got = arr.point_estimates(
            resampler.draw(rng_a),
            horizons=horizons,
            control="not_yet_treated",
            directions=(1, -1),
            normalized=True,
            match_baseline=True,
        )["delta"]

        bdf = cluster_bootstrap_draw(
            df, cluster_col=cluster, rng=rng_b, relabel_cols=["id"]
        ).drop(columns=["_F", "_dir", "_base"])
        for part in dyn._first_switch(
            bdf, group="id", time="_tidx", treatment="treated"
        ):
            bdf = bdf.merge(part, on="id", how="left")
        want = dyn._estimate_all_horizons_frame(
            df=bdf,
            y="y",
            group="id",
            time="_tidx",
            treatment="treated",
            horizons=horizons,
            control="not_yet_treated",
            weights="w",
            cluster=cluster,
            normalized=True,
        )
        want = [c["delta_l"] for c in want["cell_estimates"]]
        assert np.isfinite(want).any()
        np.testing.assert_allclose(got, want, rtol=RTOL, equal_nan=True)
    # both consumed the generator identically
    assert rng_a.integers(1 << 30) == rng_b.integers(1 << 30)


def test_frames_the_matrices_cannot_represent_go_to_the_frame_code():
    df = _dcdh_frame(_panel(seed=12))
    kw = dict(
        y="y", group="id", time="_tidx", treatment="treated", weights=None, cluster=None
    )
    assert arr.build_panel(df, **kw) is not None
    dup = pd.concat([df, df.iloc[:3]], ignore_index=True)
    assert arr.build_panel(dup, **kw) is None
    reversed_unit = pd.concat(
        [df.iloc[:10].iloc[::-1], df.iloc[10:]], ignore_index=True
    )
    assert arr.build_panel(reversed_unit, **kw) is None
    varying = df.assign(_tcell=np.where(df.index % 2 == 0, "a", "b"))
    assert arr.build_panel(varying, **kw) is None


# ---------------------------------------------------------------------------
# run time: generous bounds, far above the vectorised cost and far below the
# per-unit loops they replaced
# ---------------------------------------------------------------------------


def _timed(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        out = fn()
    return out, time.perf_counter() - t0


def test_event_study_scales_to_twenty_thousand_units():
    d = _panel(n_units=20_000, seed=13)
    r, seconds = _timed(lambda: sp.event_study(d, "y", "tt", "t", "id", window=(-3, 3)))
    assert np.isfinite(r.se) and r.se > 0
    assert seconds < 20, f"event_study took {seconds:.1f}s (was 14s as a loop)"


def test_lp_did_scales_to_twenty_thousand_units():
    d = _panel(n_units=20_000, seed=14)
    r, seconds = _timed(lambda: sp.lp_did(d, "y", "id", "t", "treated"))
    assert np.isfinite(r.se) and r.se > 0
    assert seconds < 30, f"lp_did took {seconds:.1f}s (was 74s as a loop)"


def test_did_multiplegt_dyn_bootstrap_scales_to_two_thousand_units():
    d = _panel(n_units=2_000, seed=15)
    r, seconds = _timed(
        lambda: sp.did_multiplegt_dyn(
            d, "y", group="id", time="t", treatment="treated", seed=1
        )
    )
    assert np.isfinite(r.se) and r.se > 0
    assert r.model_info["n_boot"] == 500
    assert seconds < 30, f"did_multiplegt_dyn took {seconds:.1f}s (was 79s)"
