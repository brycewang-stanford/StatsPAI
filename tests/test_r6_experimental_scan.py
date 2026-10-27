"""Option-scan round 6: ``statspai.experimental``.

Each test states the correct behaviour. Ground truth is an invariance
(assignment must not depend on index labels), an exact enumeration of a
randomization distribution, a sibling estimator (``sp.lee_bounds``), the
randomizr 2.0.1 rules (``block_ra`` / ``cluster_ra`` refuse missing blocks
and clusters; ``block_and_cluster_ra`` randomizes whole clusters within
blocks and refuses clusters that span blocks), or R ``lm``'s F statistic.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _strata_frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "s": np.repeat([0, 1, 2, 3], n // 4),
            "c": np.repeat(np.arange(8), n // 8),
            "x": rng.normal(size=n),
        }
    )


# --------------------------------------------------------------------------
# sp.randomize
# --------------------------------------------------------------------------


def test_randomize_strata_ignores_duplicate_index_labels():
    # The assignment is a function of row positions and the seed only.
    # Duplicate labels used to collapse the label->position map: only the
    # last row of each label was assigned, the rest kept uninitialised
    # memory (every unit "control" in the probe).
    df = _strata_frame()
    dup = df.copy()
    dup.index = np.repeat([0, 1], len(df) // 2)
    for seed in range(5):
        ref = sp.randomize(df, strata="s", seed=seed).data["treatment"].to_numpy()
        got = sp.randomize(dup, strata="s", seed=seed).data["treatment"].to_numpy()
        np.testing.assert_array_equal(got, ref)
        assert dup.assign(t=got).groupby("s")["t"].sum().tolist() == [5, 5, 5, 5]


def test_randomize_categorical_strata_with_unused_level():
    # An unused category used to crash (an empty float index array).
    df = _strata_frame()
    df["s"] = pd.Categorical(df["s"], categories=[3, 2, 1, 0, 9])
    r = sp.randomize(df, strata="s", seed=1)
    assert r.data.groupby("s", observed=True)["treatment"].sum().tolist() == [5] * 4


def test_randomize_missing_strata_raises():
    # randomizr::block_ra: "`blocks` must not contain NA". We used to return
    # uninitialised integers (9.2e18) as the arm of those rows.
    df = _strata_frame()
    df.loc[:4, "s"] = np.nan
    with pytest.raises(DataInsufficient, match="strata"):
        sp.randomize(df, strata="s", seed=1)


def test_randomize_missing_cluster_raises():
    # randomizr::cluster_ra: "`clusters` must not contain NA". We used to give
    # the rows with a missing cluster the arm of the last cluster.
    df = _strata_frame().astype({"c": float})
    df.loc[:4, "c"] = np.nan
    with pytest.raises(DataInsufficient, match="cluster"):
        sp.randomize(df, cluster="c", seed=1)


def test_randomize_strata_and_cluster_assigns_whole_clusters_within_strata():
    # randomizr::block_and_cluster_ra: whole clusters, complete randomization
    # of the clusters of each block. strata= used to win and cluster= was
    # dropped: every cluster of the probe was split between arms.
    df = _strata_frame()
    for seed in range(20):
        a = sp.randomize(df, strata="s", cluster="c", seed=seed).data
        assert (a.groupby("c")["treatment"].nunique() == 1).all()
        # two clusters per stratum, half treated: one treated cluster each
        per = a.groupby("s")["treatment"].sum().tolist()
        assert per == [5, 5, 5, 5]


def test_randomize_strata_and_cluster_odd_clusters_misfit_is_random():
    # Three clusters per stratum: 1 or 2 treated clusters, both occur.
    df = pd.DataFrame({"s": np.repeat([0, 1], 6), "c": np.repeat(np.arange(6), 2)})
    seen = set()
    for seed in range(40):
        a = sp.randomize(df, strata="s", cluster="c", seed=seed).data
        assert (a.groupby("c")["treatment"].nunique() == 1).all()
        seen |= set((a.groupby("s")["treatment"].sum() // 2).tolist())
    assert seen == {1, 2}


def test_randomize_cluster_spanning_strata_raises():
    # randomizr: "All units within a cluster must be in the same block."
    df = pd.DataFrame({"s": np.tile([0, 1], 20), "c": np.repeat(np.arange(8), 5)})
    with pytest.raises(MethodIncompatibility, match="cluster"):
        sp.randomize(df, strata="s", cluster="c", seed=1)


# --------------------------------------------------------------------------
# sp.switchback: randomization p-value
# --------------------------------------------------------------------------


def test_switchback_randomization_pvalue_is_valid_by_enumeration():
    # The optimal design with T=12, m=2 flips 4 coins: 16 equally likely
    # paths. Under the sharp null (fixed outcomes) a valid p-value has
    # P(p <= alpha) <= alpha. Counting only strictly larger |estimates|
    # gave p = 0 on the two extreme paths: P(p <= 0.05) = 2/16 = 0.125.
    rng = np.random.default_rng(5)
    T, m = 12, 2
    y = rng.normal(size=T) + 5
    pts = sp.switchback_design(T, m=m, seed=0)["randomize"].to_numpy()
    block = np.cumsum(pts) - 1
    ps = []
    for coins in itertools.product([0, 1], repeat=int(pts.sum())):
        d = np.asarray(coins)[block]
        r = sp.switchback(
            pd.DataFrame({"y": y, "d": d}), "y", "d", m=m, n_draws=4000, seed=1
        )
        ps.append(r.model_info["randomization_pvalue"])
    ps = np.asarray(ps)
    for alpha in (0.05, 0.10):
        assert np.mean(ps <= alpha) <= alpha
    # each path's own estimate counts: the smallest p is about 2/16
    assert ps.min() > 0.1


# --------------------------------------------------------------------------
# sp.attrition_bounds / sp.attrition_test
# --------------------------------------------------------------------------


def _attrition_frame(seed=0, n=400):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n).astype(float)
    y = 1.0 * t + rng.normal(size=n)
    o = (rng.random(n) < np.where(t == 1, 0.9, 0.75)).astype(int)
    return pd.DataFrame({"y": np.where(o == 1, y, np.nan), "t": t, "o": o})


def test_attrition_bounds_drops_missing_treatment():
    # Rows with a missing treatment used to count as control
    # (lower 0.6625 vs 0.7304). sp.lee_bounds drops them with a warning.
    df = _attrition_frame()
    df.loc[:19, "t"] = np.nan
    with pytest.warns(UserWarning, match="treat"):
        got = sp.attrition_bounds(df, "y", "t", "o")
    ref = sp.attrition_bounds(df.iloc[20:], "y", "t", "o")
    lb = sp.lee_bounds(df.iloc[20:], "y", "t", selection="o", n_bootstrap=10)
    assert got["lower_bound"] == pytest.approx(ref["lower_bound"], abs=1e-12)
    assert got["upper_bound"] == pytest.approx(ref["upper_bound"], abs=1e-12)
    assert got["naive_ate"] == pytest.approx(ref["naive_ate"], abs=1e-12)
    assert got["lower_bound"] == pytest.approx(lb.model_info["lower_bound"], abs=1e-12)
    assert got["n_total"] == 380


def test_attrition_bounds_drops_selected_rows_with_missing_outcome():
    # An observed row with a missing outcome made naive_ate NaN. sp.lee_bounds
    # drops such rows with a warning: 0.7725943633430432, 1.3965735387754223.
    df = _attrition_frame()
    rows = df.index[df["o"] == 1][:3]
    df.loc[rows, "y"] = np.nan
    with pytest.warns(UserWarning, match="missing outcome"):
        got = sp.attrition_bounds(df, "y", "t", "o")
    lb = sp.lee_bounds(df, "y", "t", selection="o", n_bootstrap=10)
    assert np.isfinite(got["naive_ate"])
    assert got["lower_bound"] == pytest.approx(lb.model_info["lower_bound"], abs=1e-12)
    assert got["upper_bound"] == pytest.approx(lb.model_info["upper_bound"], abs=1e-12)


@pytest.mark.parametrize("fn", ["bounds", "test"])
def test_attrition_non_binary_treatment_raises(fn):
    # Codes 1/2: arm 1 was taken as treated and arm 2 as control, so the
    # bounds came out with the sign flipped (-1.395, -0.768) and
    # attrition_test reported a NaN control rate.
    df = _attrition_frame()
    df["t"] = df["t"] + 1
    with pytest.raises(MethodIncompatibility, match="0/1"):
        if fn == "bounds":
            sp.attrition_bounds(df, "y", "t", "o")
        else:
            sp.attrition_test(df, "t", "o")


# --------------------------------------------------------------------------
# sp.balance_check: omnibus F with collinear covariates
# --------------------------------------------------------------------------


def test_balance_check_omnibus_f_uses_rank_with_collinear_covariates():
    # A covariate that is a linear combination of others adds nothing to the
    # regression of treatment on covariates. R lm(t ~ x + x2 + z) with
    # x2 = 2 x reports the same F(2, n - 3) as lm(t ~ x + z)
    # (0.35703305388324 on R-generated data); the numerator df used to count
    # the redundant column (F 0.2372 instead of 0.3570 there).
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame(
        {"t": rng.integers(0, 2, n), "x": rng.normal(size=n), "z": rng.normal(size=n)}
    )
    df["x2"] = 2 * df["x"]
    full = sp.balance_check(df, "t", ["x", "z"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coll = sp.balance_check(df, "t", ["x", "x2", "z"])
    assert coll.omnibus_f == pytest.approx(full.omnibus_f, rel=1e-9)
    assert coll.omnibus_p == pytest.approx(full.omnibus_p, rel=1e-9)
