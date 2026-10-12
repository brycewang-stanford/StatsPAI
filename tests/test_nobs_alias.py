"""A fit that records its sample size as ``n_obs`` still reports ``nobs``.

About twenty-five estimators (production functions, frontier models,
selection models, ...) put the sample size in ``data_info["n_obs"]``.
``EconometricResults`` reads ``data_info["nobs"]``, so ``result.nobs`` was
None, ``result.n_obs`` raised AttributeError and ``to_dict()["n_obs"]``
was None for all of them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import statspai as sp
from statspai.core.results import EconometricResults


def _result(data_info):
    params = pd.Series({"x": 1.0})
    return EconometricResults(
        params=params,
        std_errors=pd.Series({"x": 0.1}),
        model_info={"model_type": "test"},
        data_info=data_info,
    )


def test_n_obs_key_feeds_every_reader():
    res = _result({"n_obs": 137})
    assert res.nobs == 137
    assert res.n_obs == 137
    assert res.data_info["n_obs"] == 137
    assert res.to_dict()["n_obs"] == 137


def test_nobs_key_wins_when_both_are_recorded():
    res = _result({"nobs": 50, "n_obs": 137})
    assert res.nobs == 50
    assert res.data_info["n_obs"] == 137


def test_no_sample_size_is_still_reported_as_unknown():
    res = _result({})
    assert res.nobs is None
    assert not hasattr(res, "n_obs")
    assert "nobs" not in res.data_info


def test_levinsohn_petrin_reports_its_sample_size():
    rng = np.random.default_rng(0)
    rows = []
    for fid in range(120):
        omega = rng.normal(0.0, 0.3)
        k = rng.normal(2.0, 0.5)
        for yr in range(8):
            omega = 0.7 * omega + rng.normal(0.0, 0.2)
            k = 0.9 * k + 0.3 * rng.normal(1.0, 0.3)
            lab = 0.6 * k + omega + rng.normal(1.0, 0.2)
            m = 0.5 * k + omega + rng.normal(0.5, 0.05)
            y = 0.6 * lab + 0.3 * k + omega + rng.normal(0.0, 0.1)
            rows.append(dict(id=fid, year=yr, y=y, l=lab, k=k, m=m))
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.levinsohn_petrin(df, output="y", free="l", state="k", proxy="m")
    assert res.nobs == len(res.tfp)
    assert res.n_obs == res.nobs
    assert res.nobs == res.data_info["n_obs"]
    assert 0 < res.nobs <= len(df)
