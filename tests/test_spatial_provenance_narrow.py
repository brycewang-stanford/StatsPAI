"""Provenance wrappers in spatial_iv / spatial_did / bartik / the legacy
spatial fits catch only the failures of a result that cannot carry the
record (2026-10-27). Through 1.40.2 they caught ``Exception``, so a bug in
the provenance code vanished without a trace (CLAUDE.md section 3.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
import statspai.output._lineage as lineage


def _ring(n: int) -> np.ndarray:
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i - 1) % n] = W[i, (i + 1) % n] = 1.0
    return W / W.sum(1, keepdims=True)


def _iv_call():
    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(size=n)
    d = 0.5 * x + rng.normal(size=n)
    df = pd.DataFrame({"y": 1 + d + x + rng.normal(size=n), "d": d, "x": x})
    return sp.spatial_iv(df, y="y", endog=["d"], exog=["x"], W=_ring(n))


def _bartik_call():
    rng = np.random.default_rng(0)
    n, K = 60, 5
    shares = pd.DataFrame(rng.dirichlet(np.ones(K), size=n))
    shocks = pd.Series(rng.normal(size=K))
    d = 0.8 * shares.to_numpy() @ shocks.to_numpy() + rng.normal(size=n)
    data = pd.DataFrame({"y": 1.5 * d + rng.normal(size=n), "d": d})
    return sp.bartik(data, y="y", endog="d", shares=shares, shocks=shocks)


def _legacy_call():
    from statspai.spatial.models._legacy import sar

    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": 1 + x + rng.normal(size=n), "x": x})
    return sar(_ring(n), df, "y ~ x")


@pytest.mark.parametrize("call", [_iv_call, _bartik_call, _legacy_call])
def test_provenance_bug_is_not_swallowed(monkeypatch, call):
    def boom(*args, **kwargs):
        raise RuntimeError("bug in attach_provenance")

    monkeypatch.setattr(lineage, "attach_provenance", boom)
    with pytest.raises(RuntimeError, match="bug in attach_provenance"):
        call()


@pytest.mark.parametrize("call", [_iv_call, _bartik_call, _legacy_call])
def test_result_that_cannot_carry_provenance_is_still_returned(monkeypatch, call):
    def refuse(*args, **kwargs):
        raise AttributeError("cannot set _provenance")

    monkeypatch.setattr(lineage, "attach_provenance", refuse)
    assert call() is not None
