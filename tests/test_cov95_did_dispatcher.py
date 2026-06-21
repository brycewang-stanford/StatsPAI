"""Coverage tests for the statspai.did.did() dispatcher (__init__.py)."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _data_2x2(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    treat = rng.integers(0, 2, n)
    post = rng.integers(0, 2, n)
    y = 1 + 2 * treat + 3 * post + 5 * treat * post + rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "y": y,
            "treat": treat,
            "post": post,
            "sub": rng.integers(0, 2, n),
            "x1": rng.normal(0, 1, n),
        }
    )


def _staggered(seed=0, n_units=90, n_periods=8):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 4, 6][u % 3]
        fe = rng.normal()
        for t in range(1, n_periods + 1):
            te = 1.0 * (t - g + 1) if (g > 0 and t >= g) else 0.0
            rows.append(
                {"i": u, "t": t, "y": fe + 0.5 * t + te + rng.normal(0, 0.4), "g": g}
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------


def test_did_not_dataframe():
    with pytest.raises((TypeError, MethodIncompatibility)):
        sp.did([1, 2, 3], y="y", treat="t", time="p")


def test_did_empty_dataframe():
    with pytest.raises(DataInsufficient):
        sp.did(
            pd.DataFrame({"y": [], "treat": [], "post": []}),
            y="y",
            treat="treat",
            time="post",
        )


def test_did_missing_columns():
    df = _data_2x2()
    with pytest.raises(MethodIncompatibility, match="Column"):
        sp.did(df, y="nope", treat="treat", time="post")


def test_did_auto_nonbinary_no_id_raises():
    df = _data_2x2()
    df["treat"] = np.arange(len(df)) % 5
    with pytest.raises(MethodIncompatibility, match="auto-detect"):
        sp.did(df, y="y", treat="treat", time="post")


def test_did_scalar_covariate_and_invalid_controls():
    df = _data_2x2()
    r = sp.did(df, y="y", treat="treat", time="post", covariates="x1")
    assert r.estimate is not None
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.did(df, y="y", treat="treat", time="post", alpha=1.0)
    with pytest.raises(MethodIncompatibility, match="aggregation"):
        sp.did(
            df,
            y="y",
            treat="treat",
            time="post",
            method="2x2",
            aggregation="simple",
        )
    with pytest.raises(MethodIncompatibility, match="panel=False"):
        sp.did(
            df,
            y="y",
            treat="treat",
            time="post",
            method="2x2",
            panel=False,
        )


# ----------------------------------------------------------------------
# 2x2 + aliases
# ----------------------------------------------------------------------


def test_did_auto_2x2():
    df = _data_2x2()
    r = sp.did(df, y="y", treat="treat", time="post")
    assert abs(r.estimate - 5.0) < 0.5


def test_did_classic_collapses_time():
    # method='classic' with >2 time values triggers median split.
    df = _staggered()
    df2 = df.rename(columns={})
    # make a binary treat for 2x2 collapse
    df2["treat_bin"] = (df2["g"] > 0).astype(int)
    r = sp.did(df2, y="y", treat="treat_bin", time="t", method="classic")
    assert r.estimate is not None


def test_did_did2s_alias():
    df = _data_2x2()
    r = sp.did(df, y="y", treat="treat", time="post", method="did2s")
    assert r.estimate is not None


# ----------------------------------------------------------------------
# ddd
# ----------------------------------------------------------------------


def test_did_ddd_autodetect_subgroup():
    df = _data_2x2()
    r = sp.did(df, y="y", treat="treat", time="post", subgroup="sub")
    assert r.method.startswith("Triple") or "DDD" in r.method


def test_did_ddd_method_without_subgroup_raises():
    df = _data_2x2()
    with pytest.raises(ValueError):
        sp.did(df, y="y", treat="treat", time="post", method="ddd")


# ----------------------------------------------------------------------
# callaway_santanna + aggregation forwarding
# ----------------------------------------------------------------------


def test_did_cs_autodetect_with_id():
    df = _staggered()
    r = sp.did(df, y="y", treat="g", time="t", id="i")
    assert r.estimate is not None


def test_did_cs_missing_id_raises():
    df = _staggered()
    with pytest.raises(ValueError):
        sp.did(df, y="y", treat="g", time="t", method="cs")


def test_did_aggregation_forces_cs():
    df = _staggered()
    r = sp.did(
        df,
        y="y",
        treat="g",
        time="t",
        id="i",
        aggregation="dynamic",
        n_boot=50,
        random_state=1,
    )
    assert r is not None


def test_did_aggregation_bad_value_raises():
    df = _staggered()
    with pytest.raises(ValueError):
        sp.did(df, y="y", treat="g", time="t", id="i", aggregation="bogus")


def test_did_aggregation_with_non_cs_method_raises():
    df = _data_2x2()
    with pytest.raises(ValueError):
        sp.did(
            df, y="y", treat="treat", time="post", method="2x2", aggregation="dynamic"
        )


def test_did_panel_false_non_cs_raises():
    df = _data_2x2()
    with pytest.raises(ValueError):
        sp.did(df, y="y", treat="treat", time="post", method="2x2", panel=False)


def test_did_anticipation_non_cs_raises():
    df = _data_2x2()
    with pytest.raises(ValueError):
        sp.did(df, y="y", treat="treat", time="post", method="2x2", anticipation=1)


# ----------------------------------------------------------------------
# sun_abraham / bjs / sdid
# ----------------------------------------------------------------------


def test_did_sun_abraham():
    df = _staggered()
    r = sp.did(df, y="y", treat="g", time="t", id="i", method="sa")
    assert r.estimate is not None


def test_did_sun_abraham_missing_id_raises():
    df = _staggered()
    with pytest.raises(ValueError):
        sp.did(df, y="y", treat="g", time="t", method="sa")


def test_did_bjs():
    df = _staggered()
    r = sp.did(
        df, y="y", treat="g", time="t", id="i", method="bjs", event_window=(-3, 3)
    )
    assert r.estimate is not None


def test_did_bjs_missing_id_raises():
    df = _staggered()
    with pytest.raises(ValueError):
        sp.did(df, y="y", treat="g", time="t", method="bjs")


def test_did_sdid():
    df = _staggered()
    r = sp.did(df, y="y", treat="g", time="t", id="i", method="sdid")
    assert r.estimate is not None


def test_did_sdid_missing_id_raises():
    df = _staggered()
    with pytest.raises(ValueError):
        sp.did(df, y="y", treat="g", time="t", method="sdid")


def test_did_sdid_invalid_se_method_raises_taxonomy():
    df = _staggered()
    with pytest.raises(MethodIncompatibility, match="se_method"):
        sp.did(
            df,
            y="y",
            treat="g",
            time="t",
            id="i",
            method="sdid",
            se_method="bogus",
        )


def test_did_unknown_method_raises():
    df = _data_2x2()
    with pytest.raises(ValueError):
        sp.did(df, y="y", treat="treat", time="post", method="frobnicate")
