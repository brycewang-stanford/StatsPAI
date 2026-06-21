"""Round-2 coverage for statspai.did.did_multiplegt and did_multiplegt_dyn:
binary switching treatment, placebo / dynamic horizons, controls, cluster,
control-group options, and input-validation error paths. Real switch panels."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _switch_panel(seed=0, n_units=80, n_periods=7):
    """Treatment switches on (and sometimes off) across units/time."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        start = rng.integers(2, n_periods)  # switch-on period
        fe = rng.normal()
        d = 0
        for t in range(1, n_periods + 1):
            if t == start:
                d = 1
            te = 1.0 * d
            y = fe + 0.3 * t + te + rng.normal(0, 0.4)
            rows.append(
                {"i": u, "t": t, "y": y, "d": d, "x1": rng.normal(), "st": u % 8}
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def sw():
    return _switch_panel()


# ---------------------------------------------------------------- did_multiplegt
def test_multiplegt_basic(sw):
    r = sp.did_multiplegt(
        sw, y="y", group="i", time="t", treatment="d", n_boot=40, seed=1
    )
    assert r.se >= 0


def test_multiplegt_placebo_dynamic(sw):
    r = sp.did_multiplegt(
        sw,
        y="y",
        group="i",
        time="t",
        treatment="d",
        placebo=2,
        dynamic=2,
        cluster="st",
        n_boot=40,
        seed=2,
    )
    assert isinstance(r.model_info, dict)
    assert r.detail is not None


def test_multiplegt_controls(sw):
    r = sp.did_multiplegt(
        sw,
        y="y",
        group="i",
        time="t",
        treatment="d",
        controls=["x1"],
        n_boot=30,
        seed=3,
    )
    assert r.se >= 0


def test_multiplegt_missing_column_raises(sw):
    with pytest.raises(ValueError):
        sp.did_multiplegt(sw, y="nope", group="i", time="t", treatment="d", n_boot=10)


def test_multiplegt_missing_control_raises(sw):
    with pytest.raises(ValueError):
        sp.did_multiplegt(
            sw, y="y", group="i", time="t", treatment="d", controls=["nope"], n_boot=10
        )


# ------------------------------------------------------------ did_multiplegt_dyn
def test_multiplegt_dyn_basic(sw):
    r = sp.did_multiplegt_dyn(
        sw, y="y", group="i", time="t", treatment="d", dynamic=3, n_boot=40, seed=1
    )
    assert isinstance(r.model_info, dict)
    assert "event_study" in r.model_info


def test_multiplegt_dyn_placebo_never(sw):
    r = sp.did_multiplegt_dyn(
        sw,
        y="y",
        group="i",
        time="t",
        treatment="d",
        placebo=1,
        dynamic=2,
        control="never_treated",
        cluster="st",
        n_boot=40,
        seed=2,
    )
    es = r.model_info["event_study"]
    assert len(es) >= 1


def test_multiplegt_dyn_bad_control_raises(sw):
    with pytest.raises(ValueError):
        sp.did_multiplegt_dyn(
            sw, y="y", group="i", time="t", treatment="d", control="bogus"
        )


def test_multiplegt_dyn_negative_dynamic_raises(sw):
    with pytest.raises(ValueError):
        sp.did_multiplegt_dyn(sw, y="y", group="i", time="t", treatment="d", dynamic=-1)


def test_multiplegt_dyn_nonbinary_raises(sw):
    bad = sw.copy()
    bad["d"] = bad["d"] * 2
    with pytest.raises(ValueError):
        sp.did_multiplegt_dyn(bad, y="y", group="i", time="t", treatment="d")


def test_multiplegt_dyn_missing_col_raises(sw):
    with pytest.raises(ValueError):
        sp.did_multiplegt_dyn(sw, y="nope", group="i", time="t", treatment="d")
