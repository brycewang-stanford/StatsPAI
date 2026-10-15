"""Agent-native serialization (`.to_dict()`) for domain result dataclasses.

Design principle 3 wants every result object serialisable through one entry
point. The flagship `CausalResult` / `EconometricResults` already had
`.to_dict()`; the lighter domain result dataclasses (negative controls,
proximal, four-way mediation, network spillover, ITS, Rosenbaum bounds, BCF
extensions, …) only had `.summary()`. They now expose a JSON-safe `.to_dict()`
backed by the shared `statspai._result_serialize.result_to_dict` helper.

This suite pins that every such result serialises to **strict** JSON (no
``NaN`` / ``Infinity`` tokens, numpy/pandas converted) and that the shared
helper rejects non-dataclasses.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._result_serialize import result_to_dict


@pytest.fixture(scope="module")
def base():
    rng = np.random.default_rng(0)
    n = 500
    U = rng.normal(size=n)
    X = rng.normal(size=n)
    D = (0.6 * U + 0.3 * X + rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame(
        dict(
            Y=1.5 * D + U + 0.5 * X + rng.normal(size=n),
            D=D,
            X=X,
            NCE=0.8 * U + rng.normal(size=n) * 0.5,
            NCO=0.8 * U + rng.normal(size=n) * 0.5,
            Z=0.7 * U + rng.normal(size=n) * 0.5,
            W=0.7 * U + rng.normal(size=n) * 0.5,
            M=0.5 * D + 0.3 * X + rng.normal(size=n) * 0.5,
        )
    )


def _make(name, base):
    rng = np.random.default_rng(1)
    n = len(base)
    if name == "negative_control":
        return sp.negative_control_outcome(base, nco="NCO", treat="D")
    if name == "proximal":
        return sp.proximal_regression(
            base, y="Y", treat="D", z_proxy="Z", w_proxy="W", covariates=["X"]
        )
    if name == "four_way":
        return sp.four_way_decomposition(
            base, y="Y", treat="D", mediator="M", covariates=["X"]
        )
    if name == "network_hte":
        e = rng.uniform(size=n)
        df = base.assign(e=e, d=base["D"].astype(float), x2=rng.normal(size=n))
        return sp.network_hte(
            df,
            y="Y",
            treatment="d",
            neighbor_exposure="e",
            covariates=["X", "x2"],
            n_folds=3,
        )
    if name == "inward_outward":
        df = base.assign(
            d=base["D"].astype(float),
            ei=rng.uniform(size=n),
            eo=rng.uniform(size=n),
        )
        return sp.inward_outward_spillover(
            df, y="Y", treatment="d", inward_exposure="ei", outward_exposure="eo"
        )
    if name == "its":
        t = np.arange(80)
        df = pd.DataFrame(
            dict(y=1 + 0.1 * t + (t >= 40) * 3 + rng.normal(size=80), p=t)
        )
        return sp.its(df, y="y", time="p", intervention=40)
    if name == "rosenbaum":
        return sp.rosenbaum_bounds(rng.normal(1, 1, 80), rng.normal(0, 1, 80))
    if name == "bcf_ordinal":
        df = pd.DataFrame(
            dict(y=rng.normal(size=n), t=rng.integers(0, 3, n), x=rng.normal(size=n))
        )
        return sp.bcf_ordinal(df, y="y", treat="t", covariates=["x"])
    raise AssertionError(name)


RESULTS = [
    "negative_control",
    "proximal",
    "four_way",
    "network_hte",
    "inward_outward",
    "its",
    "rosenbaum",
    "bcf_ordinal",
]


class TestToDict:
    @pytest.mark.parametrize("name", RESULTS)
    def test_to_dict_is_strict_json(self, name, base):
        r = _make(name, base)
        d = r.to_dict()
        assert isinstance(d, dict) and d
        payload = json.dumps(d)  # strict: raises on NaN/Inf/numpy
        assert "NaN" not in payload and "Infinity" not in payload

    @pytest.mark.parametrize("name", RESULTS)
    def test_to_dict_covers_all_fields(self, name, base):
        from dataclasses import fields

        r = _make(name, base)
        assert set(r.to_dict()) == {f.name for f in fields(r)}


class TestSharedHelper:
    def test_rejects_non_dataclass(self):
        with pytest.raises(TypeError):
            result_to_dict({"not": "a dataclass"})

    def test_dict_subclass_result_exports_its_items(self):
        # PolicyTreeResult-style results subclass dict: the fields are
        # items, and __dict__ is empty. The export used to be {}.
        class _R(dict):
            pass

        r = _R(value=1.5, flag=np.bool_(True), _private=1)
        out = result_to_dict(r)
        assert out == {"value": 1.5, "flag": True}

    def test_bcf_factor_and_longitudinal_expose_method(self):
        # Avoid the slow MCMC happy path for these two — just pin the method.
        assert callable(sp.BCFFactorExposureResult.to_dict)
        assert callable(sp.BCFLongResult.to_dict)
