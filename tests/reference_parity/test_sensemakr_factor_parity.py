"""``sp.sensemakr`` with a factor control, against R ``sensemakr`` 0.1.6.

The textbook example of Cinelli and Hazlett (2020), repeated in the
sensitivity chapter of Chernozhukov et al.'s *Applied Causal Inference
Powered by ML and AI*, has village fixed effects among the controls and
asks what a confounder one, two and three times as strong as ``female``
would do. Before this test existed ``sp.sensemakr`` failed on a string
control with a NumPy casting error and reported no adjusted estimate.

The fixture is simulated (``_generate_sensemakr_factor.R``); the
reference values are ``sensemakr::sensemakr`` and ``ovb_bounds`` output
at full precision. Tolerance 1e-9 relative: both sides are closed-form
functions of the same least-squares fits.

References
----------
[@cinelli2020making]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
CONTROLS = ["x1", "x2", "region"]
COLUMNS = [
    "r2dz_x",
    "r2yz_dx",
    "adjusted_estimate",
    "adjusted_se",
    "adjusted_lower_CI",
    "adjusted_upper_CI",
]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(FIX / "sensemakr_factor.csv")


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads((FIX / "sensemakr_factor_R.json").read_text(encoding="utf-8"))


def _compare(table: pd.DataFrame, rows: list) -> None:
    assert len(table) == len(rows)
    for (_, ours), theirs in zip(table.iterrows(), rows):
        for col in COLUMNS:
            ref = theirs[col.replace("r2dz_x", "r2dz.x").replace("r2yz_dx", "r2yz.dx")]
            assert ours[col] == pytest.approx(ref, rel=1e-9, abs=1e-12), col
        if "adjusted_t" in theirs:
            assert ours["adjusted_t"] == pytest.approx(theirs["adjusted_t"], rel=1e-9)
        else:  # R reports no t statistic when the adjusted se is zero
            assert np.isnan(ours["adjusted_t"])


def test_headline_statistics_with_a_factor_control(data, reference):
    res = sp.sensemakr(data, y="y", treat="d", controls=CONTROLS, benchmark=["x2"])
    stats = reference["stats"]
    assert res["beta_treat"] == pytest.approx(stats["estimate"], rel=1e-9)
    assert res["se_treat"] == pytest.approx(stats["se"], rel=1e-9)
    assert res["partial_r2_yd"] == pytest.approx(stats["r2yd.x"], rel=1e-9)
    assert res["rv_q"] == pytest.approx(stats["rv_q"], rel=1e-9)
    assert res["rv_qa"] == pytest.approx(stats["rv_qa"], rel=1e-9)


def test_benchmark_multiples_and_adjusted_estimates(data, reference):
    """kd = 1:3 on a strong benchmark, including the bound that hits 1."""
    with pytest.warns(UserWarning, match="above 1"):
        res = sp.sensemakr(
            data, y="y", treat="d", controls=CONTROLS, benchmark=["x1"], kd=[1, 2, 3]
        )
    _compare(res["benchmark_table"], reference["bounds_x1"])


def test_factor_is_benchmarked_as_a_group(data, reference):
    res = sp.sensemakr(
        data, y="y", treat="d", controls=CONTROLS, benchmark=["region"], kd=[1, 2]
    )
    _compare(res["benchmark_table"], reference["bounds_region"])


def test_factor_equals_hand_built_indicators(data):
    dummies = pd.get_dummies(data["region"], drop_first=True, prefix="r").astype(float)
    wide = pd.concat([data.drop(columns="region"), dummies], axis=1)
    a = sp.sensemakr(data, y="y", treat="d", controls=CONTROLS, benchmark=["x2"])
    b = sp.sensemakr(
        wide,
        y="y",
        treat="d",
        controls=["x1", "x2", *dummies.columns],
        benchmark=["x2"],
    )
    for key in ("beta_treat", "se_treat", "rv_q", "rv_qa"):
        assert a[key] == pytest.approx(b[key], rel=1e-12)


def test_kd_and_ky_must_pair_up(data):
    with pytest.raises(ValueError, match="same length"):
        sp.sensemakr(data, y="y", treat="d", controls=CONTROLS, kd=[1, 2], ky=[1])
