"""``sp.ri_test`` returns ``RITestResult``: still a dict, now with a summary."""

import pickle

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(7)
    d = rng.integers(0, 2, 60)
    return pd.DataFrame({"y": 0.5 * d + rng.normal(size=60), "d": d})


def test_keys_and_dict_access_unchanged(df):
    r = sp.ri_test(df, y="y", treat="d", n_perms=400, seed=3)
    assert isinstance(r, dict)
    assert sorted(r) == [
        "exact",
        "n_perms",
        "observed",
        "p_one_sided",
        "p_value",
        "perm_distribution",
    ]
    # attribute access reads the same values
    assert r.p_value == r["p_value"]
    assert r.observed == r["observed"]
    # observed statistic is the difference in means
    assert r.observed == pytest.approx(
        df.y[df.d == 1].mean() - df.y[df.d == 0].mean(), rel=1e-12
    )


def test_repr_is_short_and_summary_names_the_test(df):
    r = sp.ri_test(df, y="y", treat="d", n_perms=400, seed=3)
    assert len(repr(r)) < 200
    s = r.summary()
    assert "Randomization Inference" in s and "diff_means" in s


def test_exact_enumeration_is_reported(df):
    small = df.head(10)
    r = sp.ri_test(small, y="y", treat="d", n_perms=10_000, seed=0)
    assert r.exact is True
    assert "exact" in r.summary()


def test_pickle_roundtrip_keeps_metadata(df):
    r = sp.ri_test(df, y="y", treat="d", n_perms=200, seed=1)
    r2 = pickle.loads(pickle.dumps(r))
    assert type(r2).__name__ == "RITestResult"
    assert r2.stat == "diff_means" and r2.p_value == r.p_value


def test_missing_attribute_raises_attribute_error(df):
    r = sp.ri_test(df, y="y", treat="d", n_perms=200, seed=1)
    with pytest.raises(AttributeError):
        r.not_a_field
