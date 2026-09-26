"""``CausalQuestion.estimate`` keeps the declaration binding (review §6.3).

A declared estimand the design cannot deliver used to be "coerced" with a
note inside the plan; the returned estimate carried only the delivered
label. It now records the departure, warns, and can refuse (``strict``).
The estimation sample size is the estimator's own (it was ``len(data)``).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 400
    d = pd.DataFrame({"x": rng.normal(size=n)})
    d["treat"] = (rng.random(n) < 0.5).astype(int)
    d["outcome"] = 1 + 0.5 * d["treat"] + d["x"] + rng.normal(size=n)
    return d


def test_estimation_sample_is_reported_and_recorded(df):
    df.loc[:9, "x"] = np.nan
    q = sp.causal_question("treat", "outcome", data=df, design="rct", covariates=["x"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = q.estimate()
    assert r.n == 390
    assert {
        "kind": "sample",
        "declared_rows": 400,
        "estimation_rows": 390,
    } in r.deviations


def test_estimand_fallback_is_recorded_warned_and_refusable(df):
    q = sp.causal_question(
        "treat",
        "outcome",
        data=df,
        design="selection_on_observables",
        estimand="CATE",
        covariates=["x"],
    )
    with pytest.warns(sp.exceptions.StatsPAIWarning, match="declared CATE"):
        r = q.estimate()
    assert r.declared_estimand == "CATE"
    dev = [d for d in r.deviations if d["kind"] == "estimand"]
    assert dev and dev[0]["delivered"] == r.estimand != "CATE"
    assert "Declared CATE" in r.summary()
    with pytest.raises(
        sp.exceptions.MethodIncompatibility, match="cannot be delivered"
    ):
        q.estimate(strict=True)


def test_matching_declaration_has_no_deviation(df):
    q = sp.causal_question("treat", "outcome", data=df, design="rct")
    with warnings.catch_warnings():
        warnings.simplefilter("error", sp.exceptions.StatsPAIWarning)
        r = q.estimate(strict=True)
    assert r.deviations == []
    assert r.declared_estimand == r.estimand == "ATE"
