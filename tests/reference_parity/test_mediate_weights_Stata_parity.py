"""``sp.mediate(weights=...)`` against Stata 18 ``mediate [pw=]``.

Reference: Stata 18 ``mediate (y x) (m x) (d) [pw=w]`` on
``_fixtures/mediate_weights.csv`` from
``_fixtures/_generate_mediate_weights_stata.do``: linear outcome and
mediator with and without the interaction, logit outcome with a logit
mediator, linear outcome with a probit mediator; effects with robust
(sandwich) standard errors. Estimates are held to 1e-8 and SEs to 1e-6
(observed 1e-10), except with the probit mediator, where Stata's default
convergence leaves the estimates 6e-10 and the SEs up to 7e-6 from ours
(held to 1e-8 and 2e-5).

Two identities check the weighting itself: unit weights reproduce the
unweighted estimator, and integer weights equal duplicated rows (point
estimates; the sandwich differs because duplicated rows are counted as
independent units).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"

_CASES = {
    "1": dict(y="y", mediator="m", interaction=True),
    "2": dict(y="y", mediator="m", interaction=False),
    "3": dict(
        y="yb",
        mediator="mb",
        interaction=True,
        outcome_model="logit",
        mediator_model="logit",
    ),
    "4": dict(y="y", mediator="mb", interaction=True, mediator_model="probit"),
}


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "mediate_weights_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "mediate_weights.csv", float_precision="round_trip")


@pytest.mark.parametrize("case", sorted(_CASES))
def test_weighted_effects_match_stata(ref, data, case):
    r = sp.mediate(
        data,
        treat="d",
        covariates=["x"],
        weights="w",
        inference="robust",
        **_CASES[case],
    )
    det = r.detail.set_index("effect")
    probit = case == "4"
    for name, (est, se) in ref[case]["b"].items():
        assert det.loc[name, "estimate"] == pytest.approx(est, rel=1e-8), name
        assert det.loc[name, "se"] == pytest.approx(
            se, rel=2e-5 if probit else 1e-6
        ), name
    assert r.n_obs == ref[case]["N"]


def test_unit_weights_reproduce_unweighted(data):
    kw = dict(y="y", treat="d", mediator="m", covariates=["x"], interaction=True)
    a = sp.mediate(data.assign(one=1.0), weights="one", inference="robust", **kw)
    b = sp.mediate(data, inference="robust", **kw)
    da, db = a.detail.set_index("effect"), b.detail.set_index("effect")
    for name in ("NIE", "NDE", "TE"):
        assert da.loc[name, "estimate"] == pytest.approx(
            db.loc[name, "estimate"], rel=1e-10
        )
        assert da.loc[name, "se"] == pytest.approx(db.loc[name, "se"], rel=1e-6)


def test_integer_weights_equal_duplicated_rows(data):
    d = data.iloc[:300].copy()
    d["k"] = np.where(np.arange(len(d)) % 4 == 0, 3.0, 1.0)
    kw = dict(
        y="yb",
        treat="d",
        mediator="mb",
        covariates=["x"],
        outcome_model="logit",
        mediator_model="logit",
        inference="robust",
    )
    a = sp.mediate(d, weights="k", **kw).detail.set_index("effect")
    b = sp.mediate(d.loc[d.index.repeat(d["k"].astype(int))], **kw)
    b = b.detail.set_index("effect")
    for name in ("NIE", "NDE", "PNIE", "TNDE", "TE"):
        assert a.loc[name, "estimate"] == pytest.approx(
            b.loc[name, "estimate"], rel=1e-9
        )


def test_weighted_bootstrap_runs_and_delta_refuses(data):
    r = sp.mediate(
        data,
        y="y",
        treat="d",
        mediator="m",
        covariates=["x"],
        weights="w",
        n_boot=40,
        seed=1,
    )
    robust = sp.mediate(
        data,
        y="y",
        treat="d",
        mediator="m",
        covariates=["x"],
        weights="w",
        inference="robust",
    )
    assert r.estimate == pytest.approx(robust.estimate, rel=1e-12)
    with pytest.raises(sp.MethodIncompatibility, match="weights="):
        sp.mediate(data, y="y", treat="d", mediator="m", weights="w", inference="delta")


def test_negative_weight_is_refused(data):
    d = data.copy()
    d.loc[0, "w"] = -1.0
    with pytest.raises(sp.MethodIncompatibility, match="non-negative"):
        sp.mediate(d, y="y", treat="d", mediator="m", weights="w", inference="robust")
