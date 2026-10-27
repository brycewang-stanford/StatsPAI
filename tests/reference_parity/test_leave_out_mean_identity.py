"""``sp.leave_out_mean``: exact arithmetic on small cases and a brute-force
loop on random data (the result is an identity, so tolerances are
rounding only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import ColumnNotFound, MethodIncompatibility, StatsPAIWarning


def _brute(d, value, groups, excl, w):
    out = np.full(len(d), np.nan)
    keys = list(zip(*[d[g].to_numpy() for g in groups]))
    v = d[value].to_numpy(dtype=float)
    for i in range(len(d)):
        if any(pd.isna(k) for k in keys[i]):
            continue
        num = den = 0.0
        for j in range(len(d)):
            if j == i or excl[j] or keys[j] != keys[i] or not np.isfinite(v[j]):
                continue
            num += w[j] * v[j]
            den += w[j]
        if den > 0:
            out[i] = num / den
    return out


def test_small_example_with_duplicate_labels_and_excluded_rows():
    d = pd.DataFrame(
        {
            "firm": [1, 1, 1, 2, 2],
            "fe": [1.0, 2.0, 6.0, 3.0, 5.0],
            "entrant": [False, False, True, False, True],
        },
        index=[0, 0, 1, 1, 2],  # duplicate labels
    )
    with pytest.warns(StatsPAIWarning, match="no peer"):
        got = sp.leave_out_mean(d, "fe", "firm", exclude="entrant")
    # row 0: peer row 1 (2.0); row 1: peer row 0 (1.0); row 2 (excluded):
    # rows 0-1 (1.5); row 3: its only group-mate is excluded -> NaN;
    # row 4 (excluded): row 3 (3.0)
    np.testing.assert_array_equal(got.to_numpy(), [2.0, 1.0, 1.5, np.nan, 3.0])
    assert list(got.index) == [0, 0, 1, 1, 2]
    assert got.attrs["n_no_peers"] == 1 and got.name == "fe_loo"
    # assignable back onto the frame despite the duplicate labels
    d["peer"] = got
    np.testing.assert_array_equal(d["peer"].to_numpy(), got.to_numpy())


def test_matches_a_brute_force_loop_with_weights_and_two_keys():
    rng = np.random.default_rng(0)
    n = 300
    d = pd.DataFrame(
        {
            "firm": rng.integers(0, 15, n),
            "occ": rng.integers(0, 3, n),
            "v": rng.normal(size=n),
            "w": rng.uniform(0.5, 2.0, n),
            "entrant": rng.uniform(size=n) < 0.2,
        }
    )
    d.loc[rng.choice(n, 10, replace=False), "v"] = np.nan
    with pytest.warns(StatsPAIWarning):
        got = sp.leave_out_mean(d, "v", ["firm", "occ"], exclude="entrant", weights="w")
    want = _brute(d, "v", ["firm", "occ"], d.entrant.to_numpy(), d.w.to_numpy())
    np.testing.assert_allclose(got.to_numpy(), want, rtol=1e-12, equal_nan=True)


def test_exclude_as_array_equals_exclude_as_column():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"g": rng.integers(0, 5, 50), "v": rng.normal(size=50)})
    flag = rng.uniform(size=50) < 0.3
    a = sp.leave_out_mean(d.assign(f=flag), "v", "g", exclude="f")
    b = sp.leave_out_mean(d, "v", "g", exclude=flag)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    # no exclusion: the plain jackknife mean (S - v_i) / (n_g - 1)
    c = sp.leave_out_mean(d, "v", "g")
    S = d.groupby("g").v.transform("sum")
    N = d.groupby("g").v.transform("size")
    np.testing.assert_allclose(
        c.to_numpy(), ((S - d.v) / (N - 1)).to_numpy(), rtol=1e-12
    )


def test_missing_group_key_is_nan_and_reported():
    d = pd.DataFrame({"g": [1, 1, None, 1], "v": [1.0, 2.0, 3.0, 4.0]})
    with pytest.warns(StatsPAIWarning, match="missing group key"):
        got = sp.leave_out_mean(d, "v", "g")
    assert np.isnan(got.iloc[2]) and got.attrs["n_missing_group"] == 1
    assert got.iloc[0] == pytest.approx(3.0)


def test_refusals():
    d = pd.DataFrame({"g": [1, 1], "v": [1.0, 2.0], "f": [0, 2]})
    with pytest.raises(ColumnNotFound):
        sp.leave_out_mean(d, "v", "nope")
    with pytest.raises(MethodIncompatibility, match="boolean"):
        sp.leave_out_mean(d, "v", "g", exclude="f")
    with pytest.raises(MethodIncompatibility, match="entries"):
        sp.leave_out_mean(d, "v", "g", exclude=[True])
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.leave_out_mean(d.assign(w=[-1.0, 1.0]), "v", "g", weights="w")
