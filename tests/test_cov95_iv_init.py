"""Coverage tests for statspai.iv.__init__ dispatcher branches NOT covered by
the dispatcher-routing suite: ivqreg, shift_share->bartik routing, the
explicit-args resolution error, and augmented-diagnostics edge cases.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp

_ivmod = importlib.import_module("statspai.iv")


def _iv_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    eps = rng.normal(size=n)
    d = 0.8 * z + 0.5 * eps + rng.normal(0, 0.5, n)
    y = 1.0 + 2.0 * d + 0.4 * x + eps
    return pd.DataFrame({"y": y, "d": d, "z": z, "x": x})


def test_dispatch_ivqreg_explicit_args():
    df = _iv_df(seed=1)
    r = sp.iv(data=df, method="ivqreg", y="y", endog="d", instruments="z",
              tau=0.5, n_grid=11)
    assert hasattr(r, "params")


def test_dispatch_method_must_be_str():
    with pytest.raises(TypeError):
        sp.iv(method=123)


def test_dispatch_unknown_method():
    df = _iv_df(seed=2)
    with pytest.raises(ValueError, match="Unknown method"):
        sp.iv("y ~ (d ~ z)", data=df, method="totally_not_a_method")


def test_resolve_iv_args_error_without_formula_or_kwargs():
    # jive1 requires explicit y/endog/instruments; supplying none -> ValueError
    with pytest.raises(ValueError, match="explicit"):
        sp.iv(method="jive1")


def test_dispatch_shift_share_routes_to_bartik():
    # shift_share routes to bartik; bartik then complains about its own
    # required arguments. Either way the routing branch (413-422) executes.
    df = _iv_df(seed=3)
    with pytest.raises(TypeError):
        sp.iv(data=df, method="shift_share")  # missing shares/shocks


def test_fit_alias_matches_dispatch():
    df = _iv_df(seed=4)
    r1 = sp.iv.fit("y ~ (d ~ z) + x", data=df, method="2sls")
    r2 = sp.iv("y ~ (d ~ z) + x", data=df, method="2sls")
    np.testing.assert_allclose(np.asarray(r1.params), np.asarray(r2.params),
                               rtol=1e-8)


def test_rename_helper_conflict_raises():
    kwargs = {"endog": "d", "treatment": "d2"}
    with pytest.raises(TypeError):
        _ivmod._rename(kwargs, {"endog": "treatment"})


def test_unwrap_singleton_str_multiple_raises():
    kwargs = {"instrument": ["z1", "z2"]}
    with pytest.raises(ValueError):
        _ivmod._unwrap_singleton_str(kwargs, "instrument")


def test_unwrap_singleton_str_single():
    kwargs = {"instrument": ["z1"]}
    _ivmod._unwrap_singleton_str(kwargs, "instrument")
    assert kwargs["instrument"] == "z1"


def test_augmented_diagnostics_attached_for_2sls():
    df = _iv_df(seed=5)
    r = sp.iv("y ~ (d ~ z) + x", data=df, method="2sls",
              augmented_diagnostics=True)
    diag = getattr(r, "diagnostics", {})
    # KP rk + Olea-Pflueger effective F should be attached
    assert any("KP rk" in str(k) for k in diag) or \
        any("effective F" in str(k) for k in diag)


def test_getattr_unknown_raises_attributeerror():
    with pytest.raises(AttributeError):
        _ivmod.__getattr__("definitely_not_a_symbol")
