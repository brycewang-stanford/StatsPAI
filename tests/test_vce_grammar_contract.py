"""Every public estimator that takes ``vce=`` through ``@accepts_vce`` honours it.

Two layers:

* a contract over **every** decorated function, with the wrapped estimator
  swapped for a probe so no data is needed: ``vce="cluster <c>"`` must reach
  the estimator's own cluster argument, the declared native kinds must write
  their arguments, and an unimplemented kind must raise before anything runs;
* identities on real estimators: ``f(vce="cluster c")`` returns exactly what
  ``f(cluster="c")`` returns, and the declared default kind changes nothing.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _decorated():
    out = {}
    for name in sp.list_functions():
        obj = getattr(sp, name, None)
        info = getattr(obj, "__statspai_vce__", None)
        if info:
            out[name] = (obj, info)
    return out


DECORATED = _decorated()


def test_the_grammar_reaches_the_estimators_it_was_written_for():
    # A floor, not an inventory: the decorator was applied to 49 functions.
    assert len(DECORATED) >= 49
    for name in ("callaway_santanna", "sun_abraham", "tmle", "aipw", "ipw", "rkd"):
        assert name in DECORATED


def _accepts_vce_wrapper(obj):
    """Walk down ``__wrapped__`` to the wrapper that ``accepts_vce`` built."""
    while obj is not None:
        code = getattr(obj, "__code__", None)
        # Outer decorators copy ``__statspai_vce__`` through functools.wraps;
        # the layer that parses vce= is the one whose code is accepts_vce's.
        if (
            code is not None
            and code.co_name == "wrapper"
            and code.co_filename.endswith("_vcov_spec.py")
            and obj.__closure__
        ):
            cells = [
                c
                for c in obj.__closure__
                if inspect.isfunction(getattr(c, "cell_contents", None))
                and c.cell_contents is getattr(obj, "__wrapped__", None)
            ]
            if cells:
                return obj, cells[0]
        obj = getattr(obj, "__wrapped__", None)
    raise AssertionError("accepts_vce wrapper not found")


def _probe(obj):
    wrapper, cell = _accepts_vce_wrapper(obj)
    original = cell.cell_contents
    sig = inspect.signature(original)
    seen = {}

    def spy(*args, **kwargs):
        seen.update(sig.bind_partial(*args, **kwargs).arguments)
        return seen

    return wrapper, cell, original, spy, seen


@pytest.mark.parametrize("name", sorted(DECORATED))
def test_contract(name):
    obj, info = DECORATED[name]
    wrapper, cell, original, spy, seen = _probe(obj)
    cell.cell_contents = spy
    try:
        cluster_arg = info["cluster"]
        if cluster_arg is not None:
            seen.clear()
            wrapper(vce="cluster probe_col")
            assert seen.get(cluster_arg) == "probe_col"
        # Never implemented anywhere through this decorator.
        seen.clear()
        with pytest.raises(MethodIncompatibility):
            wrapper(vce="conley")
        assert not seen, "the estimator ran before the request was refused"
        with pytest.raises(MethodIncompatibility):
            wrapper(vce="not_a_kind")
        # A default kind leaves the call as it was.
        for kind in info["default"]:
            if kind == "cluster" and cluster_arg is None:
                continue
            seen.clear()
            wrapper(vce=kind if kind != "cluster" else "cluster")
            assert set(seen) <= {"robust", "se_method", "bstrap"}
    finally:
        cell.cell_contents = original


# ---------------------------------------------------------------------------
# Identities on real estimators
# ---------------------------------------------------------------------------


def _same(a, b):
    for attr in ("estimate", "se", "params", "std_errors"):
        x, y = getattr(a, attr, None), getattr(b, attr, None)
        if x is None and y is None:
            continue
        np.testing.assert_allclose(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float), rtol=0, atol=0
        )


@pytest.fixture(scope="module")
def mpdta():
    df = sp.datasets.mpdta()
    # A cluster coarser than the county: 20 blocks of 25 counties.
    df["state"] = df["countyreal"] // 25
    return df


def _quiet(fn, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(**kw)


def test_callaway_santanna_cluster_identity(mpdta):
    base = dict(data=mpdta, y="lemp", g="first_treat", t="year", i="countyreal")
    a = _quiet(sp.callaway_santanna, vce="cluster state", **base)
    b = _quiet(sp.callaway_santanna, clustervars="state", **base)
    _same(a, b)
    c = _quiet(sp.callaway_santanna, **base)
    assert not np.isclose(a.se, c.se)


def test_sun_abraham_cluster_identity_and_default(mpdta):
    base = dict(data=mpdta, y="lemp", g="first_treat", t="year", i="countyreal")
    _same(
        _quiet(sp.sun_abraham, vce="cluster state", **base),
        _quiet(sp.sun_abraham, cluster="state", **base),
    )
    _same(_quiet(sp.sun_abraham, vce="cluster", **base), _quiet(sp.sun_abraham, **base))
    with pytest.raises(MethodIncompatibility):
        sp.sun_abraham(vce="hc3", **base)


@pytest.fixture(scope="module")
def obs():
    rng = np.random.default_rng(7)
    n = 600
    x = rng.normal(size=n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x))).astype(int)
    y = 1 + d + x + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x": x, "g": rng.integers(0, 40, n)})


def test_aipw_cluster_identity(obs):
    base = dict(data=obs, y="y", treat="d", covariates=["x"])
    _same(
        _quiet(sp.aipw, vce="cluster g", **base),
        _quiet(sp.aipw, cluster="g", **base),
    )
    _same(_quiet(sp.aipw, vce="robust", **base), _quiet(sp.aipw, **base))


def test_ipw_follows_stata_teffects(obs):
    base = dict(data=obs, y="y", treat="d", covariates=["x"])
    # vce(robust) is the sandwich, vce(cluster g) the clustered sandwich.
    _same(
        _quiet(sp.ipw, vce="robust", **base),
        _quiet(sp.ipw, se_method="sandwich", **base),
    )
    _same(
        _quiet(sp.ipw, vce="cluster g", **base),
        _quiet(sp.ipw, se_method="sandwich", cluster="g", **base),
    )


def test_tmle_bootstrap_kind(obs):
    base = dict(data=obs, y="y", treat="d", covariates=["x"], n_boot=20)
    kw = (
        dict(random_state=0)
        if "random_state" in inspect.signature(sp.tmle).parameters
        else {}
    )
    _same(
        _quiet(sp.tmle, vce="bootstrap", **base, **kw),
        _quiet(sp.tmle, se_method="bootstrap", **base, **kw),
    )


def test_rkd_cluster_identity():
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.uniform(-1, 1, n)
    y = 0.5 * x + 0.8 * np.maximum(x, 0) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x": x, "g": rng.integers(0, 60, n)})
    _same(
        _quiet(sp.rkd, data=df, y="y", x="x", vce="cluster g"),
        _quiet(sp.rkd, data=df, y="y", x="x", cluster="g"),
    )
    _same(
        _quiet(sp.rkd, data=df, y="y", x="x", vce="hc1"),
        _quiet(sp.rkd, data=df, y="y", x="x"),
    )
