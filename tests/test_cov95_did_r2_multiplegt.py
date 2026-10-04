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


def test_multiplegt_dyn_never_treated_controls_need_never_treated_units(sw):
    """Every unit in this panel switches on, so there is no control set.

    This used to return an all-NaN event study that the assertion here
    (``len(es) >= 1``) accepted, because it checked the shape rather than
    the contents. The estimator now says so.
    """
    assert int((sw.groupby("i")["d"].max() == 0).sum()) == 0
    with pytest.raises(sp.DataInsufficient, match="No switcher could be matched"):
        sp.did_multiplegt_dyn(
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


def test_multiplegt_dyn_placebo_never(sw):
    """The same call on a panel that does have never-treated units."""
    panel = pd.concat(
        [sw, sw[sw["i"] < 20].assign(i=lambda d: d["i"] + 1000, d=0.0)],
        ignore_index=True,
    )
    r = sp.did_multiplegt_dyn(
        panel,
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
    assert np.isfinite(r.estimate)


def test_multiplegt_dyn_bad_control_raises(sw):
    with pytest.raises(ValueError):
        sp.did_multiplegt_dyn(
            sw, y="y", group="i", time="t", treatment="d", control="bogus"
        )


def test_multiplegt_dyn_negative_dynamic_raises(sw):
    with pytest.raises(ValueError):
        sp.did_multiplegt_dyn(sw, y="y", group="i", time="t", treatment="d", dynamic=-1)


def test_multiplegt_dyn_doubled_treatment_halves_the_normalized_effect(sw):
    """A 0/2 treatment is accepted (1.39.0) and is the 0/1 one in other units.

    The non-normalized effects compare the same switchers with the same
    controls, so they are unchanged; per unit of treatment they are halved.
    """
    kwargs = dict(y="y", group="i", time="t", treatment="d", se_method="analytic")
    one = sp.did_multiplegt_dyn(sw, **kwargs).model_info["event_study"]
    doubled = sw.copy()
    doubled["d"] = doubled["d"] * 2
    two = sp.did_multiplegt_dyn(doubled, **kwargs).model_info["event_study"]
    assert two["att"].to_numpy() == pytest.approx(one["att"].to_numpy(), abs=1e-12)
    per_unit_one = sp.did_multiplegt_dyn(sw, normalized=True, **kwargs)
    per_unit_two = sp.did_multiplegt_dyn(doubled, normalized=True, **kwargs)
    a = per_unit_one.model_info["event_study"]["att"].to_numpy()
    b = per_unit_two.model_info["event_study"]["att"].to_numpy()
    assert b == pytest.approx(a / 2, abs=1e-12)


def test_multiplegt_dyn_non_numeric_treatment_raises(sw):
    bad = sw.copy()
    bad["d"] = bad["d"].astype(str)
    with pytest.raises(ValueError):
        sp.did_multiplegt_dyn(bad, y="y", group="i", time="t", treatment="d")


def test_multiplegt_dyn_missing_col_raises(sw):
    with pytest.raises(ValueError):
        sp.did_multiplegt_dyn(sw, y="nope", group="i", time="t", treatment="d")
