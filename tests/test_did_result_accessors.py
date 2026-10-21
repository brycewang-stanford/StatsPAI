"""Uniform access to DiD results (2026-10-21 replication report, item 8).

* ``result.attgt`` / ``.event_study`` / ``.aggregations`` read the tables
  each estimator stores and return one layout; a missing table raises an
  ``AttributeError`` subclass with a recovery hint.
* ``tidy()`` has the same leading columns on ``CausalResult`` and
  ``EconometricResults``.
* ``docs/reference/did_result_contents.md`` lists ``model_info`` keys that
  really exist on a default fit of CS, SA, BJS and drdid.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._result_tidy import TIDY_COLUMNS
from statspai.core._result_access import ResultAccessorUnavailable

DOC = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "reference"
    / "did_result_contents.md"
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    df = sp.dgp_did(n_units=120, n_periods=7, staggered=True, seed=4)
    df["g"] = df["first_treat"].fillna(0).astype(int)
    return df


@pytest.fixture(scope="module")
def fits(panel):
    d2 = panel[panel["time"].isin([3, 4]) & panel["g"].isin([0, 4])].copy()
    d2["treated"] = (d2["g"] > 0).astype(int)
    d2["post"] = (d2["time"] == 4).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = sp.callaway_santanna(panel, y="y", cohort="g", time="time", id="unit")
        out = {
            "cs": cs,
            "dynamic": sp.aggte(cs, type="dynamic"),
            "group": sp.aggte(cs, type="group"),
            "calendar": sp.aggte(cs, type="calendar"),
            "sa": sp.sun_abraham(
                panel,
                y="y",
                cohort="g",
                time="time",
                id="unit",
                aggregation="event_time",
            ),
            "bjs": sp.did_imputation(
                panel, y="y", id="unit", time="time", cohort="g", horizon=[0, 1, 2]
            ),
            "bjs_plain": sp.did_imputation(
                panel, y="y", id="unit", time="time", cohort="g"
            ),
            "gardner": sp.gardner_did(
                panel, y="y", id="unit", time="time", cohort="g", event_study=True
            ),
            "etwfe": sp.etwfe(panel, y="y", id="unit", time="time", cohort="g"),
            "drdid": sp.drdid(d2, y="y", treat="treated", time="post", id="unit"),
        }
    return out


# ------------------------------------------------------------------ accessors
def test_attgt_is_the_cs_cell_table(fits):
    cs = fits["cs"]
    t = cs.attgt
    assert list(t.columns[:8]) == [
        "group",
        "time",
        "relative_time",
        "att",
        "se",
        "ci_lower",
        "ci_upper",
        "pvalue",
    ]
    d = cs.detail.reset_index(drop=True)
    np.testing.assert_array_equal(t["att"].to_numpy(), d["att"].to_numpy())
    np.testing.assert_array_equal(t["se"].to_numpy(), d["se"].to_numpy())
    np.testing.assert_array_equal(
        t["relative_time"].to_numpy(), (d["time"] - d["group"]).to_numpy()
    )


def test_event_study_is_the_same_table_on_the_fit_and_on_aggte(fits):
    a, b = fits["cs"].event_study, fits["dynamic"].event_study
    np.testing.assert_array_equal(a["relative_time"], b["relative_time"])
    np.testing.assert_allclose(a["att"], b["att"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(a["se"], b["se"], rtol=0, atol=1e-12)


def test_gardner_dict_becomes_the_standard_table(fits):
    g = fits["gardner"]
    es = g.event_study
    raw = g.model_info["event_study"]
    labels = list(raw["horizon"])
    assert list(es.columns[:6]) == [
        "relative_time",
        "att",
        "se",
        "ci_lower",
        "ci_upper",
        "pvalue",
    ]
    assert [f"D_k{int(e):+d}" for e in es["relative_time"]] == labels
    np.testing.assert_array_equal(es["att"], [raw["coef"][k] for k in labels])
    np.testing.assert_array_equal(es["ci_lower"], [raw["ci"][k][0] for k in labels])
    # p-value is the two-sided normal one, the convention of gardner's CIs.
    from scipy import stats

    z = es["att"] / es["se"]
    np.testing.assert_allclose(es["pvalue"], 2 * stats.norm.sf(np.abs(z)), rtol=1e-12)


def test_etwfe_cells_are_attgt_not_an_event_study(fits):
    e = fits["etwfe"]
    t = e.attgt
    np.testing.assert_array_equal(t["time"] - t["group"], t["relative_time"])
    with pytest.raises(ResultAccessorUnavailable, match="event-study") as err:
        e.event_study
    assert "etwfe_emfx" in err.value.recovery_hint
    assert set(e.aggregations) == {"overall", "group"}


def test_aggregations_keys(fits):
    assert set(fits["cs"].aggregations) == {"overall", "dynamic"}
    assert set(fits["group"].aggregations) == {"overall", "group"}
    assert set(fits["calendar"].aggregations) == {"overall", "calendar"}
    ov = fits["cs"].aggregations["overall"].iloc[0]
    assert ov["att"] == fits["cs"].estimate and ov["se"] == fits["cs"].se


@pytest.mark.parametrize(
    "key, missing",
    [
        ("drdid", "attgt"),
        ("drdid", "event_study"),
        ("drdid", "aggregations"),
        ("bjs_plain", "event_study"),
        ("sa", "attgt"),
        ("dynamic", "attgt"),
    ],
)
def test_missing_table_is_an_attribute_error_with_a_hint(fits, key, missing):
    r = fits[key]
    assert not hasattr(r, missing)
    assert getattr(r, missing, None) is None
    with pytest.raises(AttributeError) as err:
        getattr(r, missing)
    assert isinstance(err.value, sp.MethodIncompatibility)
    assert err.value.recovery_hint


def test_accessor_does_not_shadow_event_study_plot(fits):
    assert callable(fits["cs"].event_study_plot)


# ---------------------------------------------------------------------- tidy()
def test_tidy_contract_is_shared(fits):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=80)})
    df["y"] = 1 + df["x"] + rng.normal(size=80)
    econ = sp.regress("y ~ x", data=df).tidy()
    causal = fits["cs"].tidy()
    assert list(econ.columns) == TIDY_COLUMNS
    assert list(causal.columns[: len(TIDY_COLUMNS)]) == TIDY_COLUMNS
    assert list(causal.columns[len(TIDY_COLUMNS) :]) == ["type"]


# ----------------------------------------------------------------------- docs
def _documented_keys(section: str) -> set:
    text = DOC.read_text(encoding="utf-8")
    m = re.search(rf"^### `{re.escape(section)}`\n(.*?)(?=^### |\Z)", text, re.M | re.S)
    assert m, f"no section for {section} in {DOC.name}"
    keys = re.findall(r"^\| `([A-Za-z_0-9]+)` \|", m.group(1), re.M)
    assert keys, f"no key table in section {section}"
    return set(keys)


@pytest.mark.parametrize(
    "section, key",
    [
        ("sp.callaway_santanna", "cs"),
        ("sp.sun_abraham", "sa"),
        ("sp.did_imputation", "bjs_plain"),
        ("sp.drdid", "drdid"),
    ],
)
def test_documented_model_info_keys_exist(fits, section, key):
    documented = _documented_keys(section)
    missing = documented - set(fits[key].model_info)
    assert not missing, f"{section}: documented but absent {sorted(missing)}"
