"""``group=`` means the unit id in ``did_imputation`` and the treatment group
in ``drdid`` (2026-10-21 replication report, item 5). A swapped call must
not run silently; a correct call must not be turned down.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did._arg_roles import ArgumentRoleWarning


@pytest.fixture(scope="module")
def staggered() -> pd.DataFrame:
    df = sp.dgp_did(n_units=100, n_periods=8, staggered=True, seed=2)
    df["g"] = df["first_treat"].fillna(0).astype(int)
    df["ever"] = (df["g"] > 0).astype(int)
    return df


@pytest.fixture(scope="module")
def single_cohort(staggered: pd.DataFrame) -> pd.DataFrame:
    df = staggered.copy()
    df.loc[df["g"] > 0, "g"] = 5
    df["ever"] = (df["g"] > 0).astype(int)
    return df


@pytest.fixture(scope="module")
def two_period() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for u in range(80):
        d = int(u < 40)
        x = rng.normal()
        for t in (0, 1):
            rows.append(
                dict(unit=u, post=t, treated=d, x=x, y=x + t + 2 * d * t + rng.normal())
            )
    return pd.DataFrame(rows)


def _no_role_warning(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    assert not [w for w in rec if issubclass(w.category, ArgumentRoleWarning)]
    return out


# ---------------------------------------------------------------- did_imputation
@pytest.mark.parametrize("column", ["ever", "treated"])
def test_did_imputation_refuses_a_treatment_dummy_as_unit_id(staggered, column):
    # Cohort varies within each 0/1 "unit": the cohort check would refuse the
    # call anyway; the message now names the swap.
    with pytest.raises(sp.MethodIncompatibility, match="treatment indicator") as e:
        sp.did_imputation(staggered, y="y", group=column, time="time", first_treat="g")
    assert "id=" in e.value.recovery_hint
    assert isinstance(e.value, ValueError)


def test_did_imputation_flags_two_group_fixed_effects(single_cohort):
    # One cohort: a 0/1 'unit' column cannot be told apart from a two-group
    # repeated cross-section with group fixed effects, so it runs and warns.
    with pytest.warns(ArgumentRoleWarning, match="two clusters"):
        flagged = sp.did_imputation(
            single_cohort, y="y", group="ever", time="time", first_treat="g"
        )
    right = sp.did_imputation(single_cohort, y="y", id="unit", time="time", cohort="g")
    # Identity: on a balanced single-adoption panel the imputation ATT with
    # group fixed effects equals the one with unit fixed effects (both are
    # the difference in means of the treated-minus-control change); only
    # the standard error, clustered on two groups, is wrong.
    assert flagged.estimate == pytest.approx(right.estimate, rel=1e-10, abs=1e-12)


def test_did_imputation_legitimate_binary_unit_ids_are_left_alone():
    # Two units coded 0 and 1, one row per period: a real panel, no warning.
    rows = []
    for u, g in ((0, 0), (1, 3)):
        for t in range(1, 6):
            rows.append(
                dict(unit=u, time=t, g=g, y=float(u + t + (2 if g and t >= g else 0)))
            )
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # few-cluster warning is expected
        r = _no_role_warning(
            sp.did_imputation, df, y="y", id="unit", time="time", cohort="g"
        )
    assert r.estimate == pytest.approx(2.0, abs=1e-12)


def test_did_imputation_normal_call_has_no_role_warning(staggered):
    _no_role_warning(
        sp.did_imputation, staggered, y="y", id="unit", time="time", cohort="g"
    )


# ------------------------------------------------------------------------- drdid
def test_drdid_refuses_a_unit_id_as_treatment(two_period):
    with pytest.raises(sp.MethodIncompatibility, match="must be binary") as e:
        sp.drdid(two_period, y="y", group="unit", time="post")
    assert "treat=" in e.value.recovery_hint and "id=" in e.value.recovery_hint
    assert "unit identifier" in str(e.value)


def test_drdid_refuses_unit_id_in_both_roles(two_period):
    with pytest.raises(sp.MethodIncompatibility, match="also the id= column"):
        sp.drdid(two_period, y="y", group="unit", time="post", id="unit")


def test_drdid_non_01_coding_runs_with_the_arm_named(two_period):
    df = two_period.assign(arm=two_period["treated"] + 1)
    with pytest.warns(ArgumentRoleWarning, match="2 .*taken as the treated"):
        coded = sp.drdid(df, y="y", treat="arm", time="post", id="unit")
    ref = sp.drdid(df, y="y", treat="treated", time="post", id="unit")
    # Same arm assignment (2 <-> 1), so the same estimator on the same data.
    assert (coded.estimate, coded.se) == (ref.estimate, ref.se)


def test_drdid_boolean_coding_is_binary(two_period):
    df = two_period.assign(arm=two_period["treated"].astype(bool))
    coded = _no_role_warning(
        sp.drdid, df, y="y", treat="arm", time="post", covariates=["x"]
    )
    ref = sp.drdid(df, y="y", treat="treated", time="post", covariates=["x"])
    assert (coded.estimate, coded.se) == (ref.estimate, ref.se)
