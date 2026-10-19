"""``sp.match(sampling_weights=)`` vs R ``MatchIt`` 4.7.2 ``s.weights``.

Fixture: ``_fixtures/match_sweights_R.json`` from
``_fixtures/_generate_match_sweights_R.R`` on ``matchit_constraints.csv``
with sampling weights ``sw = 0.4 + (row % 5) * 0.3`` (rows from 1). The
reference is the recipe of MatchIt's vignettes: ``s.weights`` in the
propensity model, ``lm(y ~ d, weights = match_data weights)``,
``sandwich::vcovCL(cluster = ~subclass)`` without replacement and
``vcovHC(type = "HC3")`` with replacement. MatchIt and sandwich are GPL
and were used as black boxes (outputs only).

The weighted score, the matched pairs, the estimate and the standard error
are all compared; the pairs are discrete, so they are the real test.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "match_sweights_R.json").read_text(encoding="utf-8"))
CASES = [k for k in REF if k != "versions"]


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(_FIX / "matchit_constraints.csv")
    df["sw"] = 0.4 + (np.arange(1, len(df) + 1) % 5) * 0.3
    return df


def _fit(data, name):
    c = REF[name]
    kw = dict(replace=c["replace"], n_matches=c["ratio"], m_order="largest")
    if "caliper" in c:
        kw.update(caliper=c["caliper"], caliper_scale="sd")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return c, sp.match(
            data,
            y="y",
            treat="d",
            covariates=["x1", "x2", "g"],
            sampling_weights="sw",
            **kw,
        )


@pytest.mark.parametrize("name", CASES)
def test_weighted_score_and_pairs_match_matchit(data, name):
    c, r = _fit(data, name)
    md = r.model_info["matched_data"]
    # R glm stops at its epsilon = 1e-8 deviance criterion: scores 1e-9 apart
    np.testing.assert_allclose(md["_pscore"], c["distance"], rtol=0, atol=5e-9)
    ncols = [f"_n{j + 1}" for j in range(c["ratio"])]
    row_of_id = dict(zip(md["_id"].to_numpy(), md.index.to_numpy() + 1))
    ours = {}
    for i, row in md.loc[md["_treated"] == 1, ncols].iterrows():
        got = sorted(int(row_of_id[v]) for v in row if np.isfinite(v))
        if got:
            ours[int(i) + 1] = got
    theirs = {}
    for t, cs in zip(c["treated_row"], c["control_rows"]):
        cs = cs if isinstance(cs, list) else [cs]
        theirs[int(t)] = sorted(int(v) for v in cs if v is not None and v != "NA")
    theirs = {t: v for t, v in theirs.items() if v}
    assert ours == theirs


@pytest.mark.parametrize("name", CASES)
def test_estimate_and_se_match_matchit(data, name):
    c, r = _fit(data, name)
    assert abs(r.estimate - c["est"]) < 1e-9 * max(1.0, abs(c["est"]))
    assert abs(r.se - c["se"]) < 1e-9 * c["se"]


def test_unit_sampling_weights_reproduce_the_unweighted_point_estimate(data):
    df = data.assign(one=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.match(
            df,
            "y",
            "d",
            ["x1", "x2", "g"],
            replace=False,
            m_order="largest",
            sampling_weights="one",
        )
        b = sp.match(df, "y", "d", ["x1", "x2", "g"], replace=False, m_order="largest")
    assert abs(a.estimate - b.estimate) < 1e-12


def test_refusals(data):
    with pytest.raises(MethodIncompatibility):
        sp.match(data, "y", "d", ["x1"], method="cem", sampling_weights="sw")
    with pytest.raises(MethodIncompatibility):
        sp.match(data, "y", "d", ["x1"], estimand="ATE", sampling_weights="sw")
    bad = data.assign(sw=data["sw"] - 1.0)
    with pytest.raises(MethodIncompatibility):
        sp.match(bad, "y", "d", ["x1"], sampling_weights="sw")
