"""Hausman FE-vs-RE test when ``V_FE - V_RE`` is not positive semi-definite.

Both implementations (``sp.hausman_test`` and ``PanelResults.hausman_test``)
clamped a negative statistic to 0, reporting p = 1 and "use RE" -- on a panel
where the unit effect loads on the regressor and FE is required. A negative
statistic now yields ``recommendation="inconclusive"``, ``pvalue=nan`` and an
``AssumptionWarning`` pointing to ``sigmamore=True`` and the Mundlak test,
which these tests also check works end to end through ``sp.test``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning


def _panel(seed: int, loading: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_id, n_t = 100, 8
    unit = np.repeat(np.arange(n_id), n_t)
    effect = rng.normal(0, 1, n_id)[unit]
    x1 = rng.normal(size=n_id * n_t) + loading * effect
    x2 = rng.normal(size=n_id * n_t)
    y = 1 + 0.5 * x1 - 0.3 * x2 + effect + rng.normal(0, 0.5, n_id * n_t)
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "id": unit, "year": np.tile(np.arange(n_t), n_id)}
    )


def test_negative_statistic_is_inconclusive_in_both_implementations():
    df = _panel(seed=0, loading=0.4)
    with pytest.warns(AssumptionWarning, match="sigmamore=True"):
        free = sp.hausman_test(df, y="y", x=["x1", "x2"], id="id", time="year")
    fe = sp.panel(df, "y ~ x1 + x2", entity="id", time="year", method="fe")
    with pytest.warns(AssumptionWarning, match="sigmamore=True"):
        method = fe.hausman_test()
    for out in (free, method):
        assert out["statistic"] < 0
        assert np.isnan(out["pvalue"])
        assert out["recommendation"] == "inconclusive"
    # sigmamore keeps the variance difference usable and rejects RE.
    assert fe.hausman_test(sigmamore=True)["recommendation"] == "FE"


def test_valid_case_is_unchanged():
    df = _panel(seed=2, loading=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        out = sp.panel(
            df, "y ~ x1 + x2", entity="id", time="year", method="fe"
        ).hausman_test()
    assert out["statistic"] >= 0 and not out["psd_violation"]
    assert out["recommendation"] == "RE"


@pytest.mark.parametrize("seed, loading, reject", [(1, 0.0, False), (0, 0.4, True)])
def test_mundlak_alternative_runs_through_sp_test(seed, loading, reject):
    df = _panel(seed=seed, loading=loading)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.panel(
            df,
            "y ~ x1 + x2",
            entity="id",
            time="year",
            method="mundlak",
            cluster="entity",
        )
    out = sp.test(res, "_mean_x1 _mean_x2")
    assert (out["pvalue"] < 0.05) is reject


@pytest.mark.parametrize("method", ["fe", "re", "mundlak", "pooled"])
def test_panel_single_coefficient_test_matches_reported_pvalue(method):
    df = _panel(seed=3, loading=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.panel(
            df,
            "y ~ x1 + x2",
            entity="id",
            time="year",
            method=method,
            cluster="entity",
        )
    i = list(res.params.index).index("x1")
    reported = float(np.asarray(res.pvalues)[i])
    assert sp.test(res, "x1")["pvalue"] == pytest.approx(reported, rel=1e-10)
