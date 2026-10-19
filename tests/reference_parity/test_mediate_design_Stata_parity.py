"""``sp.mediate(design=...)`` against Stata 18 ``mediate [pw=], vce(cluster)``.

Stata's ``svy`` prefix does not support ``mediate``. A design with PSUs and
no strata or FPC gives the linearisation variance ``G / (G - 1)`` times the
sum of the PSU totals' outer products (R ``survey::svyrecvar``); Stata's
clustered sandwich for ``mediate`` (``vce(cluster psu)``) is the same sum
without the ``G / (G - 1)`` factor. On ``_fixtures/mediate_design.csv``
(from ``_fixtures/_generate_mediate_design_stata.do``; 48 PSUs) the effects
agree to 1e-8 and our standard errors times ``sqrt((G - 1) / G)`` equal
Stata's to 1e-6. With strata the
design-based SEs are checked against an identity: one stratum holding all
PSUs gives the cluster SEs.
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
    "2": dict(
        y="yb",
        mediator="mb",
        interaction=True,
        outcome_model="logit",
        mediator_model="logit",
    ),
}


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "mediate_design_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "mediate_design.csv", float_precision="round_trip")


@pytest.mark.parametrize("case", sorted(_CASES))
def test_cluster_design_matches_stata(ref, data, case):
    des = sp.svydesign(data, weights="w", cluster="psu")
    r = sp.mediate(
        data,
        treat="d",
        covariates=["x"],
        design=des,
        inference="robust",
        **_CASES[case],
    )
    det = r.detail.set_index("effect")
    G = ref[case]["N_clust"]
    for name, (est, se) in ref[case]["b"].items():
        assert det.loc[name, "estimate"] == pytest.approx(est, rel=1e-8), name
        rescaled = det.loc[name, "se"] * np.sqrt((G - 1) / G)
        assert rescaled == pytest.approx(se, rel=1e-6), name
    assert r.model_info["survey_design"]["n_psu"] == ref[case]["N_clust"]


def test_strata_change_the_se_but_not_the_estimate(data):
    kw = dict(y="y", treat="d", mediator="m", covariates=["x"], inference="robust")
    flat = sp.mediate(data, design=sp.svydesign(data, weights="w", cluster="psu"), **kw)
    strat = sp.mediate(
        data, design=sp.svydesign(data, weights="w", cluster="psu", strata="st"), **kw
    )
    weighted = sp.mediate(data, weights="w", **kw)
    a, b = flat.detail.set_index("effect"), strat.detail.set_index("effect")
    w = weighted.detail.set_index("effect")
    for name in ("NIE", "NDE", "TE"):
        assert a.loc[name, "estimate"] == pytest.approx(
            w.loc[name, "estimate"], rel=1e-12
        )
        assert b.loc[name, "estimate"] == pytest.approx(
            w.loc[name, "estimate"], rel=1e-12
        )
    # degrees of freedom: #PSU - #strata
    assert strat.model_info["survey_design"]["df"] == 48 - 6
    assert np.isfinite(b.loc["NIE", "se"]) and b.loc["NIE", "se"] > 0


def test_design_needs_robust_inference_and_no_weights(data):
    des = sp.svydesign(data, weights="w", cluster="psu")
    with pytest.raises(sp.MethodIncompatibility, match="robust"):
        sp.mediate(data, y="y", treat="d", mediator="m", design=des, n_boot=20)
    with pytest.raises(sp.MethodIncompatibility, match="weights"):
        sp.mediate(
            data,
            y="y",
            treat="d",
            mediator="m",
            design=des,
            weights="w",
            inference="robust",
        )
