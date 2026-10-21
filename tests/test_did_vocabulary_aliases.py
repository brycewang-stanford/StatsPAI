"""One vocabulary across the DiD family (2026-10-21 replication report, item 6).

Every staggered estimator accepts the house-style spellings ``id`` /
``time`` / ``cohort`` (plus ``first_treat`` / ``g``) / ``covariates`` /
``cluster`` / ``weights`` / ``seed`` / ``n_boot`` for the matching argument,
whatever its signature calls it. The assertion is the identity that the
canonical call and the legacy call are the same call: estimate and standard
error agree exactly (``==``, not a tolerance), including the seeded
bootstrap paths, where a seed that was not forwarded would change the SE.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    df = sp.dgp_did(n_units=60, n_periods=6, staggered=True, seed=3)
    df["g"] = df["first_treat"].fillna(0).astype(int)
    df["d"] = ((df["g"] > 0) & (df["time"] >= df["g"])).astype(int)
    df["x"] = np.random.default_rng(0).normal(size=len(df))
    df["w"] = 1.0 + (df["unit"] % 3)
    return df


@pytest.fixture(scope="module")
def two_by_two(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel[panel["time"].isin([2, 3]) & panel["g"].isin([0, 3])].copy()
    d["treated"] = (d["g"] > 0).astype(int)
    d["post"] = (d["time"] == 3).astype(int)
    return d


# (function, canonical kwargs, legacy kwargs, common kwargs)
CASES = [
    (
        "did_imputation",
        dict(id="unit", cohort="g", covariates=["x"], seed=7),
        dict(group="unit", first_treat="g", controls=["x"], boot_seed=7),
        dict(y="y", time="time", vce="bootstrap", n_boot=15),
    ),
    (
        "did_imputation",
        dict(id="unit", g="g", weights="w"),
        dict(group="unit", first_treat="g", weights="w"),
        dict(y="y", time="time"),
    ),
    (
        "gardner_did",
        dict(id="unit", cohort="g", seed=7),
        dict(group="unit", first_treat="g", boot_seed=7),
        dict(y="y", time="time", vce="bootstrap", n_boot=15),
    ),
    (
        "stacked_did",
        dict(unit="unit", cohort="g", vce="cluster unit"),
        dict(group="unit", first_treat="g", cluster="unit"),
        dict(y="y", time="time"),
    ),
    (
        "etwfe",
        dict(id="unit", cohort="g", covariates=["x"]),
        dict(group="unit", first_treat="g", controls=["x"]),
        dict(y="y", time="time"),
    ),
    (
        "wooldridge_did",
        dict(id="unit", g="g"),
        dict(group="unit", first_treat="g"),
        dict(y="y", time="time"),
    ),
    (
        "twfe_decomposition",
        dict(id="unit", cohort="g"),
        dict(group="unit", first_treat="g"),
        dict(y="y", time="time"),
    ),
    (
        "did_multiplegt",
        dict(id="unit", treat="d", seed=11),
        dict(group="unit", treatment="d", seed=11),
        dict(y="y", time="time", n_boot=10),
    ),
    (
        "callaway_santanna",
        dict(id="unit", time="time", cohort="g", seed=5),
        dict(i="unit", t="time", g="g", random_state=5),
        dict(y="y", bstrap=True, biters=50),
    ),
    (
        "sun_abraham",
        dict(id="unit", time="time", cohort="g", cluster="unit"),
        dict(i="unit", t="time", g="g", cluster="unit"),
        dict(y="y", aggregation="event_time"),
    ),
]


def _fit(name, data, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(sp, name)(data, **kwargs)


@pytest.mark.parametrize(
    "name, canonical, legacy, common",
    CASES,
    ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)],
)
def test_canonical_and_legacy_spellings_are_the_same_call(
    panel, name, canonical, legacy, common
):
    a = _fit(name, panel, {**common, **canonical})
    b = _fit(name, panel, {**common, **legacy})
    assert a.estimate == b.estimate
    assert a.se == b.se or (np.isnan(a.se) and np.isnan(b.se))


def test_drdid_canonical_spellings(two_by_two):
    a = _fit(
        "drdid",
        two_by_two,
        dict(y="y", treat="treated", time="post", unit="unit", controls=["x"]),
    )
    b = _fit(
        "drdid",
        two_by_two,
        dict(y="y", group="treated", time="post", id="unit", covariates=["x"]),
    )
    assert (a.estimate, a.se) == (b.estimate, b.se)


def test_bacon_decomposition_accepts_unit_spellings(panel):
    base = sp.bacon_decomposition(panel, y="y", treat="d", time="time", id="unit")
    for alias in ("unit", "i"):
        other = sp.bacon_decomposition(
            panel, y="y", treat="d", time="time", **{alias: "unit"}
        )
        assert other["beta_twfe"] == base["beta_twfe"]
        pd.testing.assert_frame_equal(other["decomposition"], base["decomposition"])


STAGGERED_COHORT_ESTIMATORS = [
    "callaway_santanna",
    "sun_abraham",
    "did_imputation",
    "gardner_did",
    "stacked_did",
    "etwfe",
    "wooldridge_did",
    "twfe_decomposition",
]


@pytest.mark.parametrize("name", STAGGERED_COHORT_ESTIMATORS)
def test_every_staggered_estimator_accepts_the_cohort_vocabulary(name):
    fn = getattr(sp, name)
    import inspect

    accepted = set(inspect.signature(fn).parameters) | set(
        getattr(fn, "__statspai_aliases__", {})
    )
    assert {"id", "time", "cohort", "first_treat", "g"} <= accepted


def test_both_spellings_of_one_argument_is_refused(panel):
    with pytest.raises(TypeError, match="cohort"):
        sp.did_imputation(
            panel, y="y", id="unit", time="time", cohort="g", first_treat="g"
        )


def test_drdid_seed_and_random_state_cannot_disagree(two_by_two):
    with pytest.raises(sp.MethodIncompatibility, match="disagree") as err:
        sp.drdid(
            two_by_two, y="y", treat="treated", time="post", seed=1, random_state=2
        )
    assert "seed=" in err.value.recovery_hint
    # Equal values are one seed, spelled twice; both are inert and say so.
    with pytest.warns(UserWarning, match="no effect"):
        r = sp.drdid(
            two_by_two, y="y", treat="treated", time="post", seed=1, random_state=1
        )
    ref = sp.drdid(two_by_two, y="y", treat="treated", time="post")
    assert (r.estimate, r.se) == (ref.estimate, ref.se)


def test_house_style_cohort_theme():
    from statspai._house_style import alias_index, canonical_for

    assert canonical_for("first_treat") == "cohort"
    assert canonical_for("g") == "cohort"
    assert alias_index()["boot_seed"] == "seed"
