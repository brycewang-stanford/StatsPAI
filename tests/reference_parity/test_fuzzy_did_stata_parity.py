"""``sp.fuzzy_did`` against Stata ``fuzzydid`` (SSC, 2019-05-07) on fixtures.

Reference: ``_fixtures/fuzzydid_Stata.json``, written by
``_fixtures/_generate_fuzzydid_Stata.do`` (Stata 18 MP) on the CSVs written
by ``_fixtures/_generate_fuzzydid_data.py``. fuzzydid was run as a black
box (installed into a private ado directory; its code was not read).

Three designs: a 2x2 with a binary treatment whose rate also moves in the
control group (Wald-DID, -TC and -CIC all differ), a 2x2 with a treatment
in {0, 1, 2, 3}, and 40 groups over three periods with increasing,
decreasing and stable groups (two group variables).

Tolerances
----------
* Point estimates built from OLS / cell means / quantiles: 1e-11 absolute.
  Observed agreement is 1e-14 to 1e-15 (summation order).
* Rows with a logit / probit conditional-mean model: 1e-6 relative.
  Observed gaps are 6e-11 to 4e-8 and come from Stata's ``logit`` /
  ``probit`` stopping rule: on one cell, ``logit`` with default tolerances
  and with 1e-14 tolerances differ by the same order, and ``sp.fuzzy_did``'s
  Newton solution equals the tight one (test below). Registered as
  reference numerics.
* The Wald-CIC of the multi-period design with ``newcateg(0 2)`` differs
  by 3.7e-5 under ``cic_quantile='exact'``: fuzzydid computes
  ``ceil((count / n0) * n1)`` in floating point and, where ``count * n1 /
  n0`` is an integer, can land on the next order statistic (two control
  cells there have 284 rows each, so every point is such a case).
  ``cic_quantile='stata'`` reproduces it to 1e-15 (pinned below).
* Bootstrap standard errors: stochastic screen (S), not parity. Both sides
  use 500 replications of their own random numbers; the Monte Carlo
  relative SD of a 500-draw bootstrap SE is about 1/sqrt(2 * 500) = 3.2%,
  so two independent ones differ by about 4.5% (SD); the screen allows
  15% (3.3 SD).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FX = Path(__file__).resolve().parent / "_fixtures"
REF = json.loads((FX / "fuzzydid_Stata.json").read_text(encoding="utf-8"))
core = importlib.import_module("statspai.did._fuzzy_did_core")


@pytest.fixture(scope="module")
def d2():
    return pd.read_csv(FX / "fuzzydid_2x2.csv")


@pytest.fixture(scope="module")
def dord():
    return pd.read_csv(FX / "fuzzydid_ordered.csv")


@pytest.fixture(scope="module")
def dmulti():
    return pd.read_csv(FX / "fuzzydid_multi.csv")


def _point(res):
    return dict(zip(res.detail["estimator"], res.detail["estimate"]))


def _check(res, key, *, rtol=None, atol=1e-11):
    ref = REF[key]
    got = _point(res)
    assert set(got) == set(ref)
    for k, v in ref.items():
        if rtol is None:
            assert got[k] == pytest.approx(v, abs=atol), (key, k)
        else:
            assert got[k] == pytest.approx(v, rel=rtol), (key, k)


NOSE = {"se_method": "none", "cic_quantile": "stata"}
DT = ("did", "tc")


# ---------------------------------------------------------------- 2x2


def test_2x2_three_estimators(d2):
    res = sp.fuzzy_did(d2, "y", "g", "t", "d", **NOSE)
    _check(res, "2x2_base")
    assert res.n_obs == REF["N_2x2"]
    # the design is one where the three estimators genuinely differ
    est = _point(res)
    assert abs(est["W_DID"] - est["W_TC"]) > 0.05


def test_2x2_numerators(d2):
    res = sp.fuzzy_did(d2, "y", "g", "t", "d", numerator=True, **NOSE)
    _check(res, "2x2_numerator")


@pytest.mark.parametrize(
    "key, covs",
    [("2x2_x1", ["x1"]), ("2x2_x1_x2", ["x1", "C(x2)"])],
)
def test_2x2_covariates_ols(d2, key, covs):
    res = sp.fuzzy_did(d2, "y", "g", "t", "d", covariates=covs, estimators=DT, **NOSE)
    _check(res, key)


@pytest.mark.parametrize(
    "key, y, modelx",
    [
        ("2x2_x1_ols_logit", "y", ("ols", "logit")),
        ("2x2_x1_ols_probit", "y", ("ols", "probit")),
        ("2x2_yb_x1_logit_logit", "yb", ("logit", "logit")),
    ],
)
def test_2x2_covariates_glm(d2, key, y, modelx):
    res = sp.fuzzy_did(
        d2, y, "g", "t", "d", covariates=["x1"], modelx=modelx, estimators=DT, **NOSE
    )
    _check(res, key, rtol=1e-6)


def test_logit_gap_is_statas_stopping_rule(d2):
    """The logit rows' 1e-8 gaps: Stata's default ``logit`` stops early."""
    cell = d2[(d2.g == 1) & (d2.t == 0)]
    x = np.column_stack([np.ones(len(cell)), cell["x1"].to_numpy()])
    beta = core._glm_fit(x, cell["d"].to_numpy(dtype=float), "logit")
    tight = REF["logit_cell10_tight"]
    default = REF["logit_cell10_default"]
    ours = {"x1": beta[1], "_cons": beta[0]}
    for k in ("x1", "_cons"):
        # our Newton solution is Stata's with 1e-14 tolerances
        assert ours[k] == pytest.approx(tight[k], rel=1e-10, abs=1e-12)
    # and Stata's default solution is off by the order of the gaps above
    gap = max(abs(default[k] - tight[k]) for k in ("x1", "_cons"))
    assert 1e-11 < gap < 1e-5


def test_2x2_eqtest_point_differences(d2):
    res = sp.fuzzy_did(d2, "y", "g", "t", "d", eqtest=True, n_boot=50, seed=3)
    eq = res.model_info["eqtest"].set_index("comparison")["difference"]
    ref = REF["2x2_eqtest_b"]
    assert eq["W_DID-W_TC"] == pytest.approx(ref["DID_TC"], abs=1e-11)
    assert eq["W_DID-W_CIC"] == pytest.approx(ref["DID_CIC"], abs=1e-11)
    assert eq["W_TC-W_CIC"] == pytest.approx(ref["TC_CIC"], abs=1e-11)


# ---------------------------------------------------------------- ordered


def test_ordered_treatment(dord):
    _check(sp.fuzzy_did(dord, "y", "g", "t", "d", **NOSE), "ordered_base")
    _check(
        sp.fuzzy_did(dord, "y", "g", "t", "d", numerator=True, **NOSE),
        "ordered_numerator",
    )
    _check(
        sp.fuzzy_did(dord, "y", "g", "t", "d", newcateg=[1, 3], **NOSE),
        "ordered_newcateg_1_3",
    )
    assert REF["N_ordered"] == len(dord)


def test_ordered_covariates(dord):
    kw = dict(covariates=["x1"], estimators=DT, **NOSE)
    _check(sp.fuzzy_did(dord, "y", "g", "t", "d", **kw), "ordered_x1")
    for key, mx in (
        ("ordered_x1_ols_ols_logit", ("ols", "ols", "logit")),
        ("ordered_x1_ols_ols_probit", ("ols", "ols", "probit")),
    ):
        _check(sp.fuzzy_did(dord, "y", "g", "t", "d", modelx=mx, **kw), key, rtol=1e-6)


# ---------------------------------------------------------------- multi


def test_multi_groups_and_periods(dmulti):
    res = sp.fuzzy_did(dmulti, "y", "gt", "t", "d", group_change_next="gf", **NOSE)
    _check(res, "multi_base")
    assert res.n_obs == REF["N_multi"]
    comp = res.model_info["comparisons"]
    # two period pairs, each with increasing and decreasing groups
    assert sorted(zip(comp.period_from, comp.direction)) == [
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 1),
    ]


def test_multi_newcateg_and_the_cic_quantile_rule(dmulti):
    kw = dict(group_change_next="gf", newcateg=[0, 2], se_method="none")
    stata_rule = sp.fuzzy_did(dmulti, "y", "gt", "t", "d", cic_quantile="stata", **kw)
    _check(stata_rule, "multi_newcateg_0_2")
    exact = _point(sp.fuzzy_did(dmulti, "y", "gt", "t", "d", **kw))
    ref = REF["multi_newcateg_0_2"]
    # DID / TC do not use quantiles: identical under either rule
    assert exact["W_DID"] == pytest.approx(ref["W_DID"], abs=1e-11)
    assert exact["W_TC"] == pytest.approx(ref["W_TC"], abs=1e-11)
    # the documented convention gap of the CIC
    assert 1e-6 < abs(exact["W_CIC"] - ref["W_CIC"]) < 1e-4


def test_cic_quantile_rules_agree_off_the_integer_points():
    """Where count * n1 / n0 is never an integer the two rules coincide."""
    rng = np.random.default_rng(0)
    c0, c1, y = rng.normal(size=7), rng.normal(size=11), rng.normal(size=50)
    # 7 and 11 are coprime: count * 11 / 7 is an integer only for count 0, 7
    y = y[(y > c0.min()) & (y < c0.max())]
    np.testing.assert_array_equal(
        core._qq(c0, c1, y, "exact"), core._qq(c0, c1, y, "stata")
    )


def test_multi_covariates(dmulti):
    res = sp.fuzzy_did(
        dmulti,
        "y",
        "gt",
        "t",
        "d",
        group_change_next="gf",
        covariates=["x1"],
        estimators=DT,
        se_method="none",
    )
    _check(res, "multi_x1")


# ---------------------------------------------------------------- bootstrap (S)


@pytest.mark.parametrize(
    "key, kwargs",
    [
        ("2x2_boot_se", {}),
        ("2x2_boot_cluster_se", {"cluster": "cl"}),
    ],
)
def test_bootstrap_se_screen_2x2(d2, key, kwargs):
    res = sp.fuzzy_did(d2, "y", "g", "t", "d", n_boot=500, seed=11, **kwargs)
    got = dict(zip(res.detail["estimator"], res.detail["se"]))
    for k, v in REF[key].items():
        assert got[k] == pytest.approx(v, rel=0.15), (key, k)
    assert res.model_info["bootstrap_failures"] == 0


def test_bootstrap_se_screen_multi(dmulti):
    res = sp.fuzzy_did(
        dmulti, "y", "gt", "t", "d", group_change_next="gf", n_boot=500, seed=11
    )
    got = dict(zip(res.detail["estimator"], res.detail["se"]))
    for k, v in REF["multi_boot_se"].items():
        assert got[k] == pytest.approx(v, rel=0.15), k
