"""``CausalResult.summary()`` with and without a stored pre-trend test.

``model_info['pretrend_test']`` is ``None`` when no test was run
(``sp.sun_abraham(..., pretest='none')``, or an event window with no leads);
``summary()`` used to raise ``TypeError: 'NoneType' object is not
subscriptable`` on it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def panel():
    r = np.random.default_rng(1)
    cohorts = r.choice([0, 4, 6], size=60, p=[0.4, 0.3, 0.3])
    a = r.normal(size=60)
    rows = []
    for i in range(60):
        for t in range(1, 9):
            g = cohorts[i]
            y = a[i] + 0.2 * t + 2.0 * int(g > 0 and t >= g) + r.normal(scale=0.5)
            rows.append(dict(id=i, t=t, g=g, y=y))
    return pd.DataFrame(rows)


def _sa(panel, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.sun_abraham(panel, y="y", g="g", t="t", i="id", **kw)


def test_summary_without_pretrend_test(panel):
    res = _sa(panel, pretest="none")
    assert res.model_info.get("pretrend_test") is None
    text = res.summary()
    assert "Pre-trend Test" not in text
    assert f"{res.estimate:.4f}" in text


def test_summary_reports_the_stored_pretrend_test(panel):
    res = _sa(panel)
    pt = res.model_info["pretrend_test"]
    assert isinstance(pt, dict)
    line = next(ln for ln in res.summary().splitlines() if "Pre-trend Test" in ln)
    assert f"{float(pt['statistic']):.4f}" in line
    assert f"{float(pt['pvalue']):.4f}" in line
