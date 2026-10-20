"""Ordinal outcomes given as an ordered pandas Categorical keep their order.

2026-10-20 probe: ``sp.gologit`` / ``sp.ologit`` / ``sp.oprobit`` sorted
the labels of ``pd.Categorical(..., categories=["lo", "mid", "hi", "top"],
ordered=True)`` alphabetically (hi < lo < mid < top), so the x1 slope came
out 0.12 instead of 0.81 (truth 0.8). The declared order must give exactly
the fit on the integer codes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 2000
    x1, x2 = rng.normal(size=(2, n))
    latent = 0.8 * x1 + 0.5 * x2 + rng.logistic(size=n)
    k = np.digitize(latent, [-1.0, 0.5, 2.0])
    labels = np.array(["lo", "mid", "hi", "top"])[k]
    y = pd.Categorical(labels, categories=["lo", "mid", "hi", "top"], ordered=True)
    return pd.DataFrame({"x1": x1, "x2": x2, "k": k, "y": y})


def test_gologit_uses_the_declared_order(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.gologit(data, y="y", x=["x1", "x2"], pl=True)
        b = sp.gologit(data, y="k", x=["x1", "x2"], pl=True)
    assert a.model_info["categories"] == ["lo", "mid", "hi", "top"]
    np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(
        a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=1e-8
    )
    assert a.params.iloc[0] == pytest.approx(0.8, abs=0.1)


@pytest.mark.parametrize("fn", ["ologit", "oprobit"])
def test_ordered_models_use_the_declared_order(data, fn):
    a = getattr(sp, fn)("y ~ x1 + x2", data=data)
    b = getattr(sp, fn)("k ~ x1 + x2", data=data)
    np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(
        a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=1e-8
    )


def test_unused_categories_are_skipped(data):
    d = data.copy()
    d["y"] = (
        d["y"]
        .cat.add_categories(["none"])
        .cat.reorder_categories(["none", "lo", "mid", "hi", "top"], ordered=True)
    )
    a = sp.ologit("y ~ x1 + x2", data=d)
    b = sp.ologit("k ~ x1 + x2", data=d)
    np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-10)
