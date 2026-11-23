"""Matching without replacement: bounded memory, same matches (issue #56).

The static processing orders used to build the full ``n_treated x
n_control`` distance matrix first (10.1 GB for 16,845 treated and 74,998
controls). On a propensity score they now search the sorted pool; on other
distances they read one distance row at a time. The matches are defined by
the greedy rule, which the reference below applies to a dense matrix.
"""

from __future__ import annotations

import importlib
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_match_mod = importlib.import_module("statspai.matching.match")


def _frame(seed: int, n_t: int, n_c: int, ties: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_t + n_c
    score = rng.uniform(0.05, 0.95, n)
    if ties:
        score = np.round(score, 2)
    df = pd.DataFrame(
        {
            "treat": np.repeat([1, 0], [n_t, n_c]),
            "pscore": score,
            "x2": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _greedy_att(df: pd.DataFrame, k: int, caliper, m_order: str) -> float:
    """ATT of greedy score matching without replacement, from a dense matrix.

    Each treated unit, in the ``m_order`` order (ties by row), takes its
    ``k`` nearest unused controls inside the caliper (ties by row).
    """
    t = np.flatnonzero(df["treat"].to_numpy() == 1)
    c = np.flatnonzero(df["treat"].to_numpy() == 0)
    ps = df["pscore"].to_numpy()
    y = df["y"].to_numpy()
    dist = np.sqrt((ps[t][:, None] - ps[c][None, :]) ** 2)
    if m_order == "data":
        order = np.arange(len(t))
    elif m_order == "largest":
        order = np.lexsort((t, -ps[t]))
    elif m_order == "smallest":
        order = np.lexsort((t, ps[t]))
    else:
        order = np.lexsort((t, dist.min(axis=1)))
    used = np.zeros(len(c), dtype=bool)
    effects = []
    for i in order:
        d = dist[i].copy()
        if caliper is not None:
            d[d > caliper] = np.inf
        d[used] = np.inf
        pick = [j for j in np.lexsort((c, d))[:k] if np.isfinite(d[j])]
        if not pick:
            continue
        used[pick] = True
        effects.append(y[t[i]] - y[c[pick]].mean())
    return float(np.mean(effects))


@pytest.mark.parametrize(
    "seed,sizes,k,caliper,m_order,ties",
    list(
        itertools.product(
            [0, 1],
            [(60, 200), (150, 120)],
            [1, 3],
            [None, 0.02],
            ["largest", "smallest", "data", "smallest_min_dist"],
            [False, True],
        )
    ),
)
def test_score_search_reproduces_the_greedy_rule(
    seed, sizes, k, caliper, m_order, ties
):
    df = _frame(seed, *sizes, ties)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.match(
            df,
            y="y",
            treat="treat",
            covariates=["pscore"],
            pscore="pscore",
            replace=False,
            n_matches=k,
            caliper=caliper,
            m_order=m_order,
        )
    # same pairs, so the same mean of the same differences up to summation
    # order (the reference sums in processing order)
    assert res.estimate == pytest.approx(
        _greedy_att(df, k, caliper, m_order), rel=1e-12, abs=1e-14
    )


def _cdist_shapes(monkeypatch, **kwargs) -> list:
    """Shapes of the distance blocks one ``sp.match`` call computes."""
    shapes = []
    real = _match_mod.cdist

    def recording(*args, **kw):
        out = real(*args, **kw)
        shapes.append(out.shape)
        return out

    monkeypatch.setattr(_match_mod, "cdist", recording)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp.match(**kwargs)
    return shapes


@pytest.mark.parametrize("m_order", ["largest", "data", "smallest_min_dist"])
def test_score_matching_builds_no_distance_matrix(monkeypatch, m_order):
    df = _frame(3, 400, 1600, ties=False)
    shapes = _cdist_shapes(
        monkeypatch,
        data=df,
        y="y",
        treat="treat",
        covariates=["pscore"],
        pscore="pscore",
        replace=False,
        caliper=0.03,
        m_order=m_order,
    )
    # the issue's reproduction: this call computed a (400, 1600) matrix
    assert shapes == []


def test_mahalanobis_reads_one_row_at_a_time(monkeypatch):
    df = _frame(4, 300, 900, ties=False)
    shapes = _cdist_shapes(
        monkeypatch,
        data=df,
        y="y",
        treat="treat",
        covariates=["pscore", "x2"],
        distance="mahalanobis",
        replace=False,
        m_order="data",
    )
    # one target's distances to the 900 controls at a time, never (300, 900)
    assert shapes.count((1, 900)) == 300
    assert (300, 900) not in shapes


def test_dynamic_order_announces_the_dense_matrix(monkeypatch):
    df = _frame(5, 40, 80, ties=False)
    cls = _match_mod.MatchEstimator
    monkeypatch.setattr(cls, "_DENSE_WARN_BYTES", 8 * 40 * 80)
    with pytest.warns(UserWarning, match=r"full 40 x 80 distance matrix"):
        sp.match(
            df,
            y="y",
            treat="treat",
            covariates=["pscore"],
            pscore="pscore",
            replace=False,
            m_order="closest",
        )


def test_caliper_excluding_every_control_matches_nobody():
    df = _frame(6, 30, 60, ties=False)
    df.loc[df["treat"] == 1, "pscore"] += 5.0
    with pytest.raises(Exception) as err, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp.match(
            df,
            y="y",
            treat="treat",
            covariates=["pscore"],
            pscore="pscore",
            replace=False,
            caliper=0.01,
            m_order="largest",
        )
    assert "match" in str(err.value).lower()
