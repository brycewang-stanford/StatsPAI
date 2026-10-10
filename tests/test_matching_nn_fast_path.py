"""The vectorised nearest-neighbour search == the row-by-row definition.

With-replacement matching used to read one full distance row per treated
unit. The search now finds each unit's matches from a sort of the scores
(or from blocks of the distance matrix) and only then applies the rule.
That is admissible only if every treated unit gets exactly the controls the
definition gives it: the full row of distances, the ``k`` smallest by
``(distance, data order)``, and under ``ties='all'`` every control whose
squared distance equals the ``k``-th. Equality of distances is decided on
the bits, so the tests are built on designs where most distances are tied.

The definition is restated here, not imported, and applied to distances
from ``scipy.spatial.distance.cdist`` for the propensity score.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

import statspai as sp
from statspai.matching.match import MatchEstimator

COVARIATES = ["x0", "x1", "x2", "x3", "x4"]


def _design(kind: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if kind == "continuous":
        X = rng.normal(size=(n, 5))
    elif kind == "discrete":
        X = np.column_stack([rng.integers(0, m, n) for m in (2, 2, 3, 2, 4)]).astype(
            float
        )
    elif kind == "duplicated_rows":
        base = np.column_stack(
            [rng.normal(size=(n // 5, 3)), rng.integers(0, 3, (n // 5, 2))]
        )
        X = base[rng.integers(0, len(base), n)]
    else:  # grid: few cells, and equal distances between different cells
        X = np.column_stack(
            [
                rng.integers(0, 3, n),
                rng.integers(0, 3, n),
                np.clip(np.round(rng.normal(size=n)), -1, 1),
                rng.integers(0, 2, n),
                rng.integers(0, 2, n),
            ]
        ).astype(float)
    index = (X - X.mean(axis=0)) @ np.array([0.5, -0.4, 0.3, 0.2, -0.1]) - 0.2
    treat = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-index))).astype(int)
    df = pd.DataFrame(X, columns=COVARIATES)
    df["t"] = treat
    df["y"] = np.round(treat + X.sum(axis=1) + rng.normal(size=n), 2)
    return df


def _fit(df: pd.DataFrame, **kwargs):
    est = MatchEstimator(data=df, y="y", treat="t", covariates=COVARIATES, **kwargs)
    # The covariate matrix as the estimator holds it: the scaling of a
    # covariate distance is a sum over its rows, which rounds differently
    # on a copy laid out another way.
    seen = {}
    block = est._distance_block

    def spy(X, idx_from, idx_to, pscore=None):
        seen["X"] = X
        return block(X, idx_from, idx_to, pscore)

    est._distance_block = spy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = est.fit()
    est._distance_block = block
    est._X_seen = seen.get("X")
    return est, result


def _brute_force(dist, pool_order, *, k, ties, caliper, tol, scale):
    """Matches of every target from its full distance row."""
    out, n_tied, left_out = [], 0, 0
    for row in dist:
        d = row.astype(float).copy()
        if caliper is not None:
            d[d > caliper] = np.inf
        finite = np.flatnonzero(np.isfinite(d))
        kk = min(k, finite.size)
        if kk == 0:
            out.append(np.array([], dtype=int))
            continue
        ranked = sorted(finite.tolist(), key=lambda j: (d[j], pool_order[j]))
        nearest = np.array(ranked[:kk], dtype=int)
        d2 = d**2 / scale
        tied = np.flatnonzero(np.isfinite(d) & (d2 <= d2[nearest[-1]] + tol))
        if tied.size > kk:
            n_tied += 1
            if ties == "all":
                nearest = tied
            else:
                left_out += int(tied.size - kk)
        out.append(nearest)
    return out, n_tied, left_out


def _distances(est):
    """(treated x control) distances the estimator matched on, and the scale."""
    a = est._assignment
    idx_t, idx_c, pscore = a["idx_t"], a["idx_c"], a["pscore"]
    if est.distance == "propensity":
        dist = cdist(pscore[idx_t][:, None], pscore[idx_c][:, None])
        return dist, float(np.var(pscore, ddof=1))
    return est._distance_block(est._X_seen, idx_t, idx_c, pscore), 1.0


def _assert_matches_definition(df, **kwargs):
    est, result = _fit(df, **kwargs)
    a = est._assignment
    idx_t, idx_c = a["idx_t"], a["idx_c"]
    dist, scale = _distances(est)
    labels = np.asarray(df.index)
    pool_order = np.argsort(np.argsort(labels, kind="stable"))[idx_c]
    want, n_tied, left_out = _brute_force(
        dist,
        pool_order,
        k=est.n_matches,
        ties=est.ties,
        caliper=est.caliper,
        tol=est.tie_tolerance,
        scale=scale,
    )

    assert len(a["matches"]) == len(want)
    for got, exp, w in zip(a["matches"], want, a["weights"]):
        if est.ties == "all":
            np.testing.assert_array_equal(np.sort(got), np.sort(exp))
        else:
            # nearest first, the earlier row among equals
            np.testing.assert_array_equal(got, exp)
        np.testing.assert_array_equal(w, np.full(len(exp), 1.0 / max(len(exp), 1)))
    assert result.model_info["n_units_with_tied_matches"] == n_tied
    assert result.model_info["n_tied_matches_left_out"] == left_out

    # What is reported is the definition applied to those sets: the effect
    # of each matched treated unit, and each control's accumulated share.
    y = df["y"].to_numpy(dtype=float)
    effects = [
        y[idx_t[i]] - np.average(y[idx_c[m]], weights=np.full(len(m), 1.0 / len(m)))
        for i, m in enumerate(a["matches"])
        if len(m)
    ]
    assert result.estimate == float(np.mean(effects))
    share = np.zeros(len(df))
    used = np.zeros(len(df), dtype=bool)
    for m in a["matches"]:
        for j in idx_c[m]:
            share[j] += 1.0 / len(m)
            used[j] = True
    frame = result.model_info["matched_data"]
    weight = frame["_weight"].to_numpy(dtype=float)
    np.testing.assert_array_equal(weight[idx_c][used[idx_c]], share[idx_c][used[idx_c]])
    assert np.all(np.isnan(weight[idx_c][~used[idx_c]]))
    size = np.array([len(m) for m in a["matches"]], dtype=float)
    np.testing.assert_array_equal(frame["_nn"].to_numpy()[idx_t], size)
    return est


@pytest.mark.parametrize("kind", ["discrete", "duplicated_rows", "grid"])
@pytest.mark.parametrize("distance", ["propensity", "mahalanobis", "euclidean"])
@pytest.mark.parametrize("n_matches", [1, 3])
@pytest.mark.parametrize("ties", ["all", "first"])
def test_matches_equal_the_row_by_row_definition_under_heavy_ties(
    kind, distance, n_matches, ties
):
    df = _design(kind, 320, seed=11)
    est = _assert_matches_definition(
        df, distance=distance, n_matches=n_matches, ties=ties
    )
    # the design does what it is for: tied controls are the rule
    assert est._tie_stats["targets"] > 0.25 * len(est._assignment["idx_t"])


@pytest.mark.parametrize("kind", ["continuous", "discrete", "grid"])
@pytest.mark.parametrize(
    "distance, caliper", [("propensity", 0.004), ("mahalanobis", 0.7)]
)
@pytest.mark.parametrize("ties", ["all", "first"])
def test_matches_equal_the_definition_inside_a_caliper(kind, distance, caliper, ties):
    df = _design(kind, 300, seed=5)
    est = _assert_matches_definition(
        df, distance=distance, n_matches=4, ties=ties, caliper=caliper
    )
    if kind == "continuous" and ties == "first":
        # the caliper binds: treated units with one to three matches, and none
        sizes = np.array([len(m) for m in est._assignment["matches"]])
        assert np.any((sizes > 0) & (sizes < 4)) and np.any(sizes == 0)


@pytest.mark.parametrize("kind", ["continuous", "grid", "duplicated_rows"])
@pytest.mark.parametrize("distance", ["propensity", "mahalanobis"])
@pytest.mark.parametrize("tol", [1e-12, 1e-4, 5e-2])
def test_matches_equal_the_definition_with_a_tie_tolerance(kind, distance, tol):
    df = _design(kind, 300, seed=3)
    _assert_matches_definition(df, distance=distance, n_matches=2, tie_tolerance=tol)


def test_ties_first_follows_the_index_labels_not_the_row_position():
    df = _design("discrete", 300, seed=8)
    df.index = np.random.default_rng(1).permutation(len(df)) * 3
    _assert_matches_definition(df, distance="propensity", ties="first", n_matches=2)
    _assert_matches_definition(df, distance="mahalanobis", ties="first", n_matches=2)


def test_more_matches_requested_than_controls():
    df = _design("continuous", 60, seed=2)
    df.loc[df.index[:54], "t"] = 1
    df.loc[df.index[54:], "t"] = 0
    est = _assert_matches_definition(df, distance="propensity", n_matches=10)
    assert all(len(m) == 6 for m in est._assignment["matches"])


@pytest.mark.parametrize("ties", ["all", "first"])
def test_scores_closer_than_a_representable_square_are_tied(ties):
    # cdist forms sqrt((a - b)^2); for scores 1e-170 apart the square
    # underflows and the distance is exactly 0, so such controls are tied
    # with an exact duplicate, and ties='first' keeps the earliest row,
    # not the control whose score is nearest. A search on |a - b| would
    # rank them.
    rng = np.random.default_rng(0)
    n = 80
    df = _design("continuous", n, seed=4)
    score = np.round(rng.uniform(0.2, 0.8, n), 1)
    score[::7] = 3e-170 * rng.permutation(len(score[::7]) + 1)[1:]
    df["score"] = score
    df["t"] = (np.arange(n) % 2 == 0).astype(int)
    est = _assert_matches_definition(df, pscore="score", n_matches=1, ties=ties)
    a = est._assignment
    tiny_t = np.flatnonzero(a["pscore"][a["idx_t"]] < 1e-100)
    tiny_c = np.flatnonzero(a["pscore"][a["idx_c"]] < 1e-100)
    assert tiny_t.size and tiny_c.size > 1
    for i in tiny_t:
        want = tiny_c if ties == "all" else tiny_c[:1]
        np.testing.assert_array_equal(np.sort(a["matches"][i]), want)


@pytest.mark.parametrize("ties", ["all", "first"])
@pytest.mark.parametrize("caliper", [None, 1.5])
def test_a_distance_matrix_with_missing_entries_takes_the_row_definition(ties, caliper):
    rng = np.random.default_rng(6)
    dist = np.round(np.abs(rng.normal(size=(40, 25))) * 2, 0)
    dist[rng.uniform(size=dist.shape) < 0.2] = np.inf
    dist[3, :] = np.inf
    dist[5, 4] = np.nan
    est = MatchEstimator(
        data=_design("continuous", 50, seed=1),
        y="y",
        treat="t",
        covariates=COVARIATES,
        distance="mahalanobis",
        n_matches=2,
        ties=ties,
    )
    pool_order = rng.permutation(dist.shape[1]).astype(float)
    for block in (dist, np.where(np.isfinite(dist), dist, 9.0)):
        est._tie_stats = {"targets": 0, "left_out": 0}
        got, weights = est._nn_match_from_dist(block, caliper, pool_order=pool_order)
        want, n_tied, left_out = _brute_force(
            block, pool_order, k=2, ties=ties, caliper=caliper, tol=0.0, scale=1.0
        )
        for g, e, w in zip(got, want, weights):
            np.testing.assert_array_equal(g, e)
            assert g.dtype == np.dtype(int) and len(w) == len(e)
        assert est._tie_stats == {"targets": n_tied, "left_out": left_out}


def test_matching_twenty_thousand_rows_is_not_quadratic():
    rng = np.random.default_rng(0)
    n = 20_000
    X = rng.normal(size=(n, 5))
    index = X @ np.array([0.5, -0.4, 0.3, 0.2, -0.1]) - 0.2
    df = pd.DataFrame(X, columns=COVARIATES)
    df["t"] = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-index))).astype(int)
    df["y"] = df["t"] + X.sum(axis=1) + rng.normal(size=n)

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sp.match(df, y="y", treat="t", covariates=COVARIATES)
        ps = sp.psmatch2(df, treat="t", covariates=COVARIATES, y="y")
    elapsed = time.perf_counter() - start
    assert elapsed < 20.0, f"matching n={n} took {elapsed:.1f}s"

    # each treated unit's reported gap is the smallest gap to any control
    frame = result.model_info["matched_data"]
    score = frame["_pscore"].to_numpy()
    treated = frame["_treated"].to_numpy() == 1
    controls = np.sort(score[~treated])
    at = np.searchsorted(controls, score[treated])
    below = np.abs(score[treated] - controls[np.clip(at - 1, 0, len(controls) - 1)])
    above = np.abs(controls[np.clip(at, 0, len(controls) - 1)] - score[treated])
    np.testing.assert_array_equal(
        frame["_pdif"].to_numpy()[treated], np.minimum(below, above)
    )
    assert ps.att == result.estimate
