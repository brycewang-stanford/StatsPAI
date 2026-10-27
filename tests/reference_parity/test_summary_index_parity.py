"""``sp.summary_index`` vs Stata ``swindex`` and a by-hand Stata reference.

Fixture: ``_fixtures/_generate_summary_index_Stata.do`` (Stata 18) on
``_fixtures/summary_index_input.csv`` (320 units, five components, one
reversed, about 8% missing in the ``m*`` copies, three units with a missing
``treat``).

* ``A_*``: ``swindex`` (SSC, distribution date 2021-02-24) with
  ``normby(control) flip()`` under its rescaling options, run under
  ``set type double``. Observed max abs gap on the index 2.8e-15, on the
  raw weights r(wt) 2.1e-15 (summation order). Tolerance atol 1e-12,
  rtol 1e-10: deterministic, same algorithm.
* ``A_*_float``: the same ado under Stata's default ``set type float``,
  where it stores the z-scores and weights as float. Not a parity row; the
  test pins that the gap is float rounding (below 1e-6 absolute on the
  index, observed 3e-7).
* ``K_*`` and the weighted / imputed / listwise ``A_*`` rows: built by hand
  from ``summarize [aweight=]``, ``egen rowmean``, ``bysort egen mean``,
  ``correlate, covariance`` and Mata ``invsym``. Same tolerance.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    ColumnNotFound,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)

_FIX = pathlib.Path(__file__).parent / "_fixtures"
DATA = pd.read_csv(_FIX / "summary_index_input.csv", float_precision="round_trip")
REF = pd.read_csv(_FIX / "summary_index_Stata.csv", dtype=str)
WREF = pd.read_csv(_FIX / "summary_index_Stata_weights.csv", dtype=str)

Y = [f"y{j}" for j in range(1, 6)]
M = [f"m{j}" for j in range(1, 6)]


def _hex(s) -> float:
    if not isinstance(s, str) or s.strip() in ("", "."):
        return float("nan")
    m = re.fullmatch(r"([+-])1\.([0-9a-f]+)X([+-][0-9a-f]+)", s.strip())
    if m is None:
        if s.strip().startswith(("+0.", "-0.")):
            return 0.0
        raise ValueError(s)
    sign, mant, exp = m.groups()
    if exp == "+3ff":
        return float("nan")
    return float.fromhex(f"{sign}0x1.{mant}p{int(exp, 16)}")


def _ref_index(config: str) -> np.ndarray:
    r = REF[REF["config"] == config]
    assert len(r) == len(DATA), config
    v = pd.Series([_hex(s) for s in r["index_hex"]], index=r["id"].astype(int))
    return v.reindex(DATA["id"]).to_numpy()


def _ref_col(config: str, col: str) -> np.ndarray:
    r = WREF[WREF["config"] == config]
    return np.array([_hex(s) for s in r[col]])


CONFIGS = {
    # swindex
    "A_ref": dict(outcomes=Y, treat="treat", method="anderson", reverse=["y5"]),
    "A_norescale": dict(
        outcomes=Y, treat="treat", method="anderson", reverse=["y5"], rescale="none"
    ),
    "A_full": dict(
        outcomes=Y, treat="treat", method="anderson", reverse=["y5"], rescale="full"
    ),
    "A_nonormby": dict(outcomes=Y, method="anderson", reverse=["y5"]),
    "A_miss_ref": dict(outcomes=M, treat="treat", method="anderson", reverse=["m5"]),
    "A_miss_norescale": dict(
        outcomes=M, treat="treat", method="anderson", reverse=["m5"], rescale=False
    ),
    # by hand
    "K_none": dict(outcomes=Y, treat="treat", reverse=["y5"]),
    "K_ref": dict(outcomes=Y, treat="treat", reverse=["y5"], rescale=True),
    "K_fullsample": dict(outcomes=Y, reverse=["y5"]),
    "K_miss_avail": dict(
        outcomes=M, treat="treat", reverse=["m5"], missing="available"
    ),
    "K_miss_listwise": dict(
        outcomes=M, treat="treat", reverse=["m5"], missing="listwise"
    ),
    "K_miss_impute": dict(outcomes=M, treat="treat", reverse=["m5"], missing="impute"),
    "K_w_miss_ref": dict(
        outcomes=M,
        treat="treat",
        reverse=["m5"],
        weights="w",
        rescale="reference",
        missing="available",
    ),
    "A_w_miss_ref": dict(
        outcomes=M,
        treat="treat",
        method="anderson",
        reverse=["m5"],
        weights="w",
        rescale="reference",
    ),
    "A_miss_listwise": dict(
        outcomes=M,
        treat="treat",
        method="anderson",
        reverse=["m5"],
        missing="listwise",
        rescale="none",
    ),
    "A_miss_impute": dict(
        outcomes=M,
        treat="treat",
        method="anderson",
        reverse=["m5"],
        missing="impute",
        rescale="none",
    ),
}


@pytest.mark.parametrize("config", list(CONFIGS))
def test_index_matches_stata(config):
    res = sp.summary_index(DATA, **CONFIGS[config])
    ref = _ref_index(config)
    ours = res.index.to_numpy()
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(ref))
    np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-12, equal_nan=True)
    assert res.index.index.equals(DATA.index)


@pytest.mark.parametrize(
    "config",
    ["A_ref", "A_norescale", "A_full", "A_nonormby", "A_miss_ref", "A_miss_norescale"],
)
def test_weights_match_swindex_r_wt(config):
    res = sp.summary_index(DATA, **CONFIGS[config])
    np.testing.assert_allclose(
        res.components["weight"].to_numpy(),
        _ref_col(config, "weight_hex"),
        rtol=1e-10,
        atol=1e-12,
    )


def test_weighted_anderson_weights_and_moments():
    res = sp.summary_index(DATA, **CONFIGS["A_w_miss_ref"])
    np.testing.assert_allclose(
        res.components["weight"].to_numpy(),
        _ref_col("A_w_miss_ref_w", "weight_hex"),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        res.components["mean"].to_numpy(),
        _ref_col("A_w_miss_ref", "mean_hex"),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        res.components["sd"].to_numpy(), _ref_col("A_w_miss_ref", "sd_hex"), rtol=1e-12
    )


@pytest.mark.parametrize("config", ["K_none", "K_fullsample", "K_miss_avail"])
def test_means_and_sds_match_summarize(config):
    res = sp.summary_index(DATA, **CONFIGS[config])
    # the reversed component is reported after its sign flip, as summarize saw it
    np.testing.assert_allclose(
        res.components["mean"].to_numpy(), _ref_col(config, "mean_hex"), rtol=1e-12
    )
    np.testing.assert_allclose(
        res.components["sd"].to_numpy(), _ref_col(config, "sd_hex"), rtol=1e-12
    )


@pytest.mark.parametrize(
    "config", ["A_ref", "A_norescale", "A_full", "A_miss_ref", "A_miss_norescale"]
)
def test_swindex_default_float_storage_is_the_only_gap(config):
    ours = sp.summary_index(DATA, **CONFIGS[config]).index.to_numpy()
    ref = _ref_index(config + "_float")
    gap = np.nanmax(np.abs(ours - ref))
    assert 1e-9 < gap < 1e-6  # float rounding, not an algorithmic difference


# ----------------------------------------------------------------- analytic


def _toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "d": [0, 0, 0, 0, 1, 1, 1, 1],
            "a": [1.0, 2.0, 4.0, 3.0, 3.0, 5.0, 4.0, 6.0],
            "b": [2.0, 1.0, 3.0, 5.0, 4.0, 3.0, 6.0, 5.0],
            "c": [0.5, 0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6],
        },
        index=[10, 11, 12, 13, 20, 21, 22, 23],
    )


def test_klk_is_mean_of_control_standardized_zscores():
    df = _toy()
    ctrl = df[df.d == 0]
    z = (df[["a", "b"]] - ctrl[["a", "b"]].mean()) / ctrl[["a", "b"]].std()
    res = sp.summary_index(df, ["a", "b"], treat="d")
    np.testing.assert_allclose(res.index, z.mean(axis=1), rtol=1e-14)
    assert res.index.index.equals(df.index)
    assert res.rescale == "none" and np.isnan(res.index_sd)


def test_anderson_weights_are_inverse_covariance_row_sums():
    df = _toy()
    ctrl = df[df.d == 0]
    cols = ["a", "b", "c"]
    z = (df[cols] - ctrl[cols].mean()) / ctrl[cols].std()
    w = np.linalg.inv(np.cov(z.to_numpy(), rowvar=False)).sum(axis=1)
    res = sp.summary_index(df, cols, treat="d", method="anderson", rescale="none")
    np.testing.assert_allclose(res.components["weight"], w, rtol=1e-12)
    np.testing.assert_allclose(res.index, z.to_numpy() @ w / w.sum(), rtol=1e-12)
    # rescaled: mean 0 and SD 1 in the control group
    r2 = sp.summary_index(df, cols, treat="d", method="anderson")
    ctl = r2.index[df.d == 0]
    assert ctl.mean() == pytest.approx(0, abs=1e-14)
    assert ctl.std() == pytest.approx(1, rel=1e-13)


def test_invariances():
    df = _toy()
    base = sp.summary_index(df, ["a", "b", "c"], treat="d", method="anderson")
    scaled = df.assign(a=3 * df.a + 7, c=-df.c)
    res = sp.summary_index(
        scaled, ["a", "b", "c"], treat="d", method="anderson", reverse=["c"]
    )
    np.testing.assert_allclose(res.index, base.index, rtol=1e-12, atol=1e-14)
    # control_value of another type selects the same reference group
    lab = df.assign(d=df.d.map({0: "ctl", 1: "trt"}))
    res2 = sp.summary_index(lab, ["a", "b"], treat="d", control_value="ctl")
    np.testing.assert_allclose(
        res2.index, sp.summary_index(df, ["a", "b"], treat="d").index
    )


def test_single_outcome_is_its_zscore():
    df = _toy()
    ctrl = df[df.d == 0]
    z = (df.a - ctrl.a.mean()) / ctrl.a.std()
    for method in ("klk", "anderson"):
        res = sp.summary_index(df, ["a"], treat="d", method=method, rescale="none")
        np.testing.assert_allclose(res.index, z, rtol=1e-13)


def test_missing_rules_differ_where_components_are_missing():
    df = _toy()
    df.loc[11, "a"] = np.nan
    df.loc[22, ["a", "b"]] = np.nan
    avail = sp.summary_index(df, ["a", "b"], treat="d", missing="available")
    imp = sp.summary_index(df, ["a", "b"], treat="d", missing="impute")
    lw = sp.summary_index(df, ["a", "b"], treat="d", missing="listwise")
    assert np.isnan(avail.index[22]) and np.isnan(imp.index[22])
    assert np.isnan(lw.index[11]) and not np.isnan(avail.index[11])
    assert avail.index[11] != imp.index[11]
    full = ~df[["a", "b"]].isna().any(axis=1)
    np.testing.assert_allclose(avail.index[full], lw.index[full])


def test_errors_are_typed():
    df = _toy()
    with pytest.raises(ColumnNotFound):
        sp.summary_index(df, ["a", "zz"], treat="d")
    with pytest.raises(MethodIncompatibility):
        sp.summary_index(df, ["a", "b"], treat="d", reverse=["c"])
    with pytest.raises(MethodIncompatibility):
        sp.summary_index(df, ["a"], treat="d", method="pca")
    with pytest.raises(MethodIncompatibility):
        sp.summary_index(df, ["a"], treat="d", rescale="sometimes")
    with pytest.raises(DataInsufficient):
        sp.summary_index(df, ["a"], treat="d", control_value=5)
    const = df.assign(k=np.where(df.d == 0, 1.0, df.a))
    with pytest.raises(DataInsufficient):
        sp.summary_index(const, ["a", "k"], treat="d")
    with pytest.raises(NumericalInstability):
        sp.summary_index(
            df.assign(e=df.a + df.b), ["a", "b", "e"], treat="d", method="anderson"
        )
    with pytest.raises(MethodIncompatibility):
        sp.summary_index(df.assign(w=-1.0), ["a"], treat="d", weights="w")


def test_result_object():
    res = sp.summary_index(_toy(), ["a", "b"], treat="d", method="anderson")
    assert "Anderson" in res.summary()
    assert list(res.to_frame().index) == ["a", "b"]
    assert res.covariance is not None and res.covariance.shape == (2, 2)
    assert set(res.cite().split()) >= {"anderson2008multiple"}


def test_klk_default_imputes_and_anderson_default_uses_available():
    # Kling, Liebman and Katz impute a missing component at the mean of the
    # row's group; swindex averages the components the row has.
    df = _toy()
    df.loc[11, "a"] = np.nan
    for method, rule in (("klk", "impute"), ("anderson", "available")):
        default = sp.summary_index(df, ["a", "b"], treat="d", method=method)
        explicit = sp.summary_index(
            df, ["a", "b"], treat="d", method=method, missing=rule
        )
        np.testing.assert_array_equal(default.index, explicit.index)
        assert default.missing == rule
