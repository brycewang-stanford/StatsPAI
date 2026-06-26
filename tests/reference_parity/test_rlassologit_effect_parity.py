"""Reference parity: ``sp.rlassologit_effect`` / ``rlassologit_effects`` vs hdm.

Pins StatsPAI's logistic high-dimensional treatment-effect estimators against
``hdm::rlassologitEffect(s)`` (Belloni-Chernozhukov-Wei post-double-selection
for GLMs). ``hdm`` is deterministic on the default penalty path, so the bar is
near machine precision; the assertions use ``atol=1e-6`` (observed: post path
~1e-14, no-post path ~1e-7 from the non-post shooting tolerance).

Fixtures + reference numbers are produced by ``_generate_rlassologit_effect.R``
(re-run only on a contract change); no R is needed at test time.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from statspai.rlasso import rlassologit_effect, rlassologit_effects

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def ref():
    path = _FIXTURE_DIR / "rlassologit_effect_R.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(_FIXTURE_DIR / "hdm_rlassologit_effect_data.csv")
    y = df["y"].to_numpy()
    d = df["d"].to_numpy()
    X = df[[c for c in df.columns if c.startswith("x")]].to_numpy()
    return X, y, d


@pytest.mark.parametrize("post, key", [(True, "single_post"), (False, "single_nopost")])
def test_rlassologit_effect_matches_hdm(ref, data, post, key):
    X, y, d = data
    res = rlassologit_effect(X, y, d, post=post)
    exp = ref[key]
    np.testing.assert_allclose(res.alpha, exp["alpha"], atol=1e-6)
    np.testing.assert_allclose(res.se, exp["se"], atol=1e-6)
    np.testing.assert_allclose(res.tstat, exp["t"], atol=1e-5)
    np.testing.assert_allclose(res.pvalue, exp["pval"], atol=1e-6)


def test_rlassologit_effects_multitarget_matches_hdm(ref, data):
    X, y, _ = data
    out = rlassologit_effects(X, y, index=[0, 1])
    results = list(out.values())
    targets = ref["multi"]["targets"]
    assert len(results) == len(targets) == 2
    for res, exp in zip(results, targets):
        np.testing.assert_allclose(res.alpha, exp["coef"], atol=1e-6)
        np.testing.assert_allclose(res.se, exp["se"], atol=1e-6)


def test_rlassologit_effect_result_shape(data):
    X, y, d = data
    res = rlassologit_effect(X, y, d)
    assert res.se > 0
    assert np.isfinite(res.alpha)
    assert res.n_selected >= 0
    lo, hi = res.conf_int()
    assert lo < res.alpha < hi
