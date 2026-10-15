"""Result-contract defects found by the 2026-10-16 agent-native audit.

Each test pins one way an agent reading a result used to fail.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.core.results import _to_jsonable


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def mpdta():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.datasets.mpdta()


def test_numpy_bool_and_missing_times_are_json_values():
    assert _to_jsonable(np.bool_(True)) is True
    assert _to_jsonable(pd.NaT) is None
    assert _to_jsonable(np.datetime64("NaT")) is None
    assert _to_jsonable(np.datetime64("2004-01-01")) == "2004-01-01T00:00:00"


def test_gardner_summary_without_event_study(mpdta):
    # model_info['event_study'] is present but None without an event window.
    r = sp.gardner_did(
        mpdta, y="lemp", group="countyreal", time="year", first_treat="first_treat"
    )
    assert r.model_info.get("event_study") is None
    text = r.summary()
    assert f"{r.estimate:.4f}" in text


def test_etwfe_tidy_reads_its_event_study_convention(mpdta):
    r = sp.etwfe(
        mpdta, y="lemp", group="countyreal", time="year", first_treat="first_treat"
    )
    t = r.tidy()
    assert t["term"].is_unique
    es = r.model_info["event_study"]
    rows = t[t["type"] == "event_study"]
    assert len(rows) == len(es)
    first = es.iloc[0]
    row = rows.iloc[0]
    assert row["estimate"] == first["estimate"]
    z = first["estimate"] / first["se"]
    assert row["p_value"] == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-12)


def test_legacy_to_dict_is_strict_json_at_the_standard_level():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"y": rng.normal(size=200), "x1": rng.normal(size=200)})
    r = sp.ridge("y ~ x1", data=X)
    # The legacy dict carries NaN tuning diagnostics (lambda_hkb on a
    # one-regressor fit); the standard level must still be strict JSON.
    for level in ("minimal", "standard", "agent"):
        json.dumps(r.to_dict(detail=level), allow_nan=False)


def test_result_card_on_a_fit_without_standard_errors():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"y": rng.normal(size=150), "x1": rng.normal(size=150)})
    r = sp.ridge("y ~ x1", data=X)
    card = r.result_card()
    assert card["inference"]["full_covariance"] is None
    json.dumps(dict(card), default=str)


def test_rddensity_says_which_scale_each_field_is_on():
    # estimate / pvalue are the CJM t-test; se / ci belong to the density
    # difference. The identity ties them together.
    r = sp.rddensity(sp.datasets.lee_2008_senate(), x="x", c=0)
    assert r.model_info["scale"] == {
        "estimate": "t_statistic",
        "pvalue": "t_statistic",
        "se": "density_diff",
        "ci": "density_diff",
    }
    diff = r.model_info["density_diff"]
    assert diff / r.se == pytest.approx(r.estimate, rel=1e-12)
    assert r.ci[0] < diff < r.ci[1]
    assert "density-difference scale" in r.estimand
