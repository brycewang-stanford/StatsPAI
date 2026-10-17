"""The optional Rust kernels (``statspai_hdfe``) and their loader.

Without the extension only the loader tests and the NumPy-path checks run.
CI's ``rust-hdfe`` workflow builds the extension and sets
``STATSPAI_REQUIRE_RUST=1``, which turns a missing extension into a failure
instead of a skip.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai import _rust_backend
from statspai.exceptions import (
    DataInsufficient,
    MethodIncompatibility,
    MissingDependencyError,
    NumericalInstability,
)

REPO = Path(__file__).resolve().parents[1]
needs_rust = pytest.mark.skipif(
    not _rust_backend.HAS_RUST,
    reason=f"Rust kernels unavailable: {_rust_backend.info()['reason']}",
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_required_extension_is_present_when_ci_asks_for_it():
    if os.environ.get("STATSPAI_REQUIRE_RUST", "") != "1":
        pytest.skip("STATSPAI_REQUIRE_RUST is not set")
    info = sp.fast.backend_info()
    assert info["available"], info


def test_backend_info_is_consistent():
    info = sp.fast.backend_info()
    assert info["available"] is _rust_backend.HAS_RUST
    assert info["min_api_version"] == _rust_backend.MIN_API_VERSION
    if info["available"]:
        assert info["reason"] is None
        assert info["api_version"] >= info["min_api_version"]
        assert info["threads"] >= 1
        assert "install" not in info
    else:
        assert info["reason"]
        assert info["threads"] is None
        assert info["install"] == 'pip install "statspai[rust]"'


def test_skip_switch_disables_the_extension():
    code = "import statspai as sp, json; " "print(json.dumps(sp.fast.backend_info()))"
    env = dict(os.environ, STATSPAI_SKIP_RUST="1")
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    import json

    info = json.loads(out.stdout.strip().splitlines()[-1])
    assert info["available"] is False
    assert info["reason"] == "disabled by STATSPAI_SKIP_RUST"


def test_set_num_threads_rejects_bad_counts():
    for bad in (-1, 1.5, True, "2"):
        with pytest.raises(MethodIncompatibility):
            sp.fast.set_num_threads(bad)


def test_set_num_threads_without_extension_names_the_install():
    if _rust_backend.HAS_RUST:
        pytest.skip("extension present")
    with pytest.raises(MissingDependencyError) as ei:
        sp.fast.set_num_threads(2)
    assert "statspai[rust]" in ei.value.diagnostics["install"]
    assert sp.fast.get_num_threads() is None


@needs_rust
def test_extension_version_matches_the_crate():
    cargo = (REPO / "rust" / "statspai_hdfe" / "Cargo.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"', cargo, re.M).group(1)
    assert _rust_backend.rust.__version__ == version


@needs_rust
def test_thread_count_round_trip():
    before = sp.fast.get_num_threads()
    try:
        sp.fast.set_num_threads(3)
        assert sp.fast.get_num_threads() == 3
    finally:
        sp.fast.set_num_threads(before)


# ---------------------------------------------------------------------------
# Input validation: bad inputs raise ValueError instead of aborting
# ---------------------------------------------------------------------------


@needs_rust
def test_out_of_range_codes_raise_instead_of_crashing():
    r = _rust_backend.rust
    n = 50
    x = np.asfortranarray(np.ones((n, 2)))
    good = np.arange(n, dtype=np.int64) % 5
    for bad in (good - 1, good + 1):  # a -1 code, and a code equal to G
        with pytest.raises(ValueError, match="codes must lie in"):
            r.demean_2d(
                x, [bad, good], [np.ones(5), np.ones(5)], 100, 0.0, 1e-8, True, 5
            )
    with pytest.raises(ValueError):
        r.separation_mask(np.ones(n), [good + 1], np.array([5], dtype=np.int64))
    with pytest.raises(ValueError):
        r.singleton_mask([good - 1])
    with pytest.raises(ValueError):
        r.group_demean(good + 10, np.ones(n), np.zeros(5), np.ones(5, dtype=np.int64))


@needs_rust
def test_malformed_arguments_raise_value_error():
    r = _rust_backend.rust
    n = 40
    codes = np.arange(n, dtype=np.int64) % 4
    x = np.asfortranarray(np.ones((n, 1)))
    with pytest.raises(ValueError, match="accel_period"):
        r.demean_2d(x, [codes, codes], [np.ones(4), np.ones(4)], 10, 0.0, 1e-8, True, 0)
    with pytest.raises(ValueError, match="length"):
        r.demean_2d(
            x, [codes[:-1], codes], [np.ones(4), np.ones(4)], 10, 0.0, 1e-8, True, 5
        )
    with pytest.raises(ValueError, match="Fortran"):
        r.demean_2d(
            np.ones((n, 2)),
            [codes, codes],
            [np.ones(4), np.ones(4)],
            10,
            0.0,
            1e-8,
            True,
            5,
        )
    strided = np.repeat(codes, 2)[::2]  # a non-contiguous view
    assert not strided.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="contiguous"):
        r.demean_2d(
            x, [strided, codes], [np.ones(4), np.ones(4)], 10, 0.0, 1e-8, True, 5
        )
    with pytest.raises(ValueError, match="non-negative"):
        r.demean_2d_weighted(
            x,
            [codes, codes],
            [np.ones(4), np.ones(4)],
            -np.ones(n),
            10,
            0.0,
            1e-8,
            True,
            5,
        )
    with pytest.raises(ValueError, match="primary_starts"):
        r.demean_2d_weighted_sorted(
            x,
            np.array([0, 30, 20, 40], dtype=np.int64),
            np.ones(3),
            [],
            [],
            np.ones(n),
            10,
            0.0,
            1e-8,
            True,
            5,
        )


@needs_rust
def test_empty_inputs_are_no_ops():
    r = _rust_backend.rust
    empty = np.asfortranarray(np.zeros((0, 3)))
    codes = np.zeros(0, dtype=np.int64)
    infos = r.demean_2d(
        empty, [codes, codes], [np.ones(1), np.ones(1)], 10, 0.0, 1e-8, True, 5
    )
    assert [d["converged"] for d in infos] == [True, True, True]
    infos = r.demean_2d_weighted(
        empty,
        [codes, codes],
        [np.ones(1), np.ones(1)],
        np.zeros(0),
        10,
        0.0,
        1e-8,
        True,
        5,
    )
    assert len(infos) == 3


def test_weighted_feols_with_only_singletons_raises():
    # Each id appears once: singleton pruning drops every row. With the
    # API-1 extension this reached Rust with n = 0 and aborted Python.
    df = pd.DataFrame(
        {
            "y": np.arange(10.0),
            "x": np.linspace(0, 1, 10),
            "id": np.arange(10),
            "year": np.repeat([0, 1], 5),
            "w": np.ones(10),
        }
    )
    with pytest.raises(DataInsufficient, match="singleton"):
        sp.fast.feols("y ~ x | id + year", data=df, weights="w")


# ---------------------------------------------------------------------------
# Correctness: the projection, against dense least squares
# ---------------------------------------------------------------------------


def _exact_residual(X, codes_list, weights=None):
    D = np.column_stack([np.eye(int(c.max()) + 1)[c] for c in codes_list])
    sw = np.ones(len(X)) if weights is None else np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(D * sw[:, None], X * sw[:, None], rcond=None)
    return X - D @ beta


def _panel(seed, n, groups, mobility=None):
    rng = np.random.default_rng(seed)
    codes = [rng.integers(0, g, n) for g in groups]
    if mobility is not None:
        # Low mobility between the first two FEs: the second mostly follows
        # the first, which makes alternating projections slow.
        home = rng.integers(0, groups[1], groups[0])
        move = rng.random(n) < mobility
        codes[1] = np.where(move, codes[1], home[codes[0]])
    codes = [pd.factorize(c)[0].astype(np.int64) for c in codes]
    X = np.column_stack(
        [rng.standard_normal(n) + 0.5 * codes[0] % 3, rng.standard_normal(n) * 10 + 5]
    )
    return X, codes


CASES = [
    dict(seed=1, n=3000, groups=(150, 20)),
    dict(seed=2, n=3000, groups=(20, 150)),
    dict(seed=3, n=2500, groups=(300, 60), mobility=0.05),
    dict(seed=4, n=3000, groups=(200, 30, 8)),
    dict(seed=5, n=3000, groups=(60, 50, 40, 5)),
    dict(seed=6, n=2500, groups=(250, 40, 12), mobility=0.1),
]


@needs_rust
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"seed{c['seed']}")
@pytest.mark.parametrize("weighted", [False, True])
def test_rust_demean_matches_the_exact_projection(case, weighted):
    r = _rust_backend.rust
    X, codes = _panel(**case)
    w = (
        np.random.default_rng(case["seed"] + 100).uniform(0.3, 3.0, len(X))
        if weighted
        else None
    )
    counts = [np.bincount(c).astype(float) for c in codes]
    A = np.asfortranarray(X.copy())
    if weighted:
        infos = r.demean_2d_weighted(A, codes, counts, w, 100_000, 0.0, 1e-13, True, 5)
    else:
        infos = r.demean_2d(A, codes, counts, 100_000, 0.0, 1e-13, True, 5)
    assert all(d["converged"] for d in infos), infos
    exact = _exact_residual(X, codes, w)
    # Stopping threshold 1e-13 * max|x| (about 5e-12 here) on the change
    # between sweeps; the distance to the fixed point is a small multiple.
    np.testing.assert_allclose(A, exact, rtol=0, atol=1e-9)


@needs_rust
@pytest.mark.parametrize("case", CASES[:4], ids=lambda c: f"seed{c['seed']}")
def test_rust_and_numpy_backends_agree(case):
    X, codes = _panel(**case)
    fe = pd.DataFrame({f"f{k}": c for k, c in enumerate(codes)})
    out_r, info_r = sp.fast.demean(X, fe, backend="rust", tol=1e-13, max_iter=100_000)
    out_n, info_n = sp.fast.demean(X, fe, backend="numpy", tol=1e-13, max_iter=100_000)
    assert info_r.backend == "rust" and info_n.backend == "numpy"
    np.testing.assert_allclose(out_r, out_n, rtol=0, atol=1e-9)


@needs_rust
def test_results_do_not_depend_on_the_thread_count():
    r = _rust_backend.rust
    X, codes = _panel(seed=7, n=20_000, groups=(800, 50, 9), mobility=0.1)
    X = np.column_stack([X] * 3)
    counts = [np.bincount(c).astype(float) for c in codes]
    rng = np.random.default_rng(8)
    order = np.argsort(rng.integers(0, 700, len(X)), kind="stable")
    Xs = np.ascontiguousarray(X[order])
    resid = rng.standard_normal(len(X))
    cl = np.sort(rng.integers(0, 700, len(X)))
    starts = np.flatnonzero(np.r_[True, cl[1:] != cl[:-1]]).astype(np.int64)
    ends = np.r_[starts[1:], len(cl)].astype(np.int64)
    before = sp.fast.get_num_threads()
    demeaned, meats = [], []
    try:
        for threads in (1, 4):
            sp.fast.set_num_threads(threads)
            A = np.asfortranarray(X.copy())
            r.demean_2d(A, codes, counts, 10_000, 0.0, 1e-12, True, 5)
            demeaned.append(A)
            meats.append(r.cluster_meat(Xs, resid, starts, ends))
    finally:
        sp.fast.set_num_threads(before)
    assert np.array_equal(demeaned[0], demeaned[1])
    assert np.array_equal(meats[0], meats[1])


@needs_rust
def test_cluster_meat_matches_the_definition():
    rng = np.random.default_rng(9)
    n, k, g = 5000, 3, 120
    X = rng.standard_normal((n, k))
    u = rng.standard_normal(n)
    cl = rng.integers(0, g, n)
    from statspai.core._numba_kernels import cluster_meat

    got = cluster_meat(X, u, cl)
    scores = np.zeros((g, k))
    np.add.at(scores, cl, X * u[:, None])
    np.testing.assert_allclose(got, scores.T @ scores, rtol=1e-12, atol=1e-10)


# ---------------------------------------------------------------------------
# Estimators: the two paths agree, and absorbed regressors are caught
# ---------------------------------------------------------------------------


def _poisson_panel(seed=0, n=20_000):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 400, n)
    year = rng.integers(0, 15, n)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    eta = (
        0.4 * x1
        - 0.2 * x2
        + rng.normal(0, 0.5, 400)[firm]
        + rng.normal(0, 0.3, 15)[year]
    )
    return pd.DataFrame(
        {
            "y": rng.poisson(np.exp(eta)),
            "x1": x1,
            "x2": x2,
            "firm": firm,
            "year": year,
            "w": rng.uniform(0.5, 2.0, n),
            "firm_size": rng.standard_normal(400)[firm],
        }
    )


@needs_rust
@pytest.mark.parametrize("weights", [None, "w"])
def test_fepois_rust_irls_matches_the_python_irls(monkeypatch, weights):
    import importlib

    fp = importlib.import_module("statspai.fast.fepois")

    df = _poisson_panel()
    kw = dict(data=df, weights=weights, tol=1e-12, fe_tol=1e-12)
    rust_fit = sp.fast.fepois("y ~ x1 + x2 | firm + year", **kw)
    monkeypatch.setattr(fp, "_HAS_RUST_HDFE", False)
    py_fit = sp.fast.fepois("y ~ x1 + x2 | firm + year", **kw)
    assert rust_fit.fe_kernel == "rust" and py_fit.fe_kernel == "numpy"
    np.testing.assert_allclose(rust_fit.coef_vec, py_fit.coef_vec, rtol=0, atol=1e-9)
    np.testing.assert_allclose(rust_fit.se().values, py_fit.se().values, rtol=1e-7)


@pytest.mark.parametrize("estimator", ["feols", "fepois"])
def test_regressor_absorbed_by_the_fixed_effects_raises(estimator):
    df = _poisson_panel(n=5000)
    fit = getattr(sp.fast, estimator)
    with pytest.raises(NumericalInstability, match="perfectly collinear"):
        fit("y ~ x1 + firm_size | firm + year", data=df)


def test_duplicated_regressor_raises():
    df = _poisson_panel(n=5000)
    df["x1_copy"] = 2.0 * df["x1"]
    with pytest.raises(NumericalInstability, match="collinear"):
        sp.fast.feols("y ~ x1 + x1_copy | firm + year", data=df)


def test_fe_kernel_is_reported():
    df = _poisson_panel(n=3000)
    expected = "rust" if _rust_backend.HAS_RUST else "numpy"
    assert sp.fast.feols("y ~ x1 | firm + year", data=df).fe_kernel == expected
    assert sp.fast.feols("y ~ x1", data=df).fe_kernel == "none"
    assert sp.fast.fepois("y ~ x1 | firm + year", data=df).fe_kernel == expected
    assert (
        sp.fast.feols("y ~ x1 | firm + year", data=df).to_dict()["fe_kernel"]
        == expected
    )


@needs_rust
@pytest.mark.skipif(
    sys.platform != "linux", reason="fork start method is Linux-only here"
)
def test_kernels_work_in_a_forked_child():
    # Rayon's global pool has no threads in a forked child and would hang;
    # the crate's own pool is rebuilt there.
    code = """
import multiprocessing as mp, numpy as np, statspai_hdfe as r
def run(_):
    n = 2000
    c = (np.arange(n) % 50).astype(np.int64); d = (np.arange(n) % 7).astype(np.int64)
    a = np.asfortranarray(np.random.default_rng(0).standard_normal((n, 2)))
    return r.demean_2d(a, [c, d], [np.ones(50), np.ones(7)], 1000, 0.0, 1e-10, True, 5)[0]["converged"]
run(0)
with mp.get_context("fork").Pool(2) as p:
    print(all(p.map(run, range(4))))
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("True")


def test_identification_check_has_no_false_positives():
    rng = np.random.default_rng(31)
    n = 4000
    df = pd.DataFrame(
        {
            "y": rng.standard_normal(n),
            "big": 1e6 + rng.standard_normal(n),  # large mean, unit spread
            "tiny": 1e-11 * rng.standard_normal(n),  # small scale
            "firm": rng.integers(0, 50, n),
            "year": rng.integers(0, 10, n),
        }
    )
    sp.fast.feols("y ~ big", data=df)  # intercept + big: not collinear
    sp.fast.feols("y ~ tiny | firm", data=df)  # small but identified
    fit = sp.fast.feols("y ~ big + tiny | firm + year", data=df)
    assert np.isfinite(fit.coef_vec).all()


def test_two_constant_regressors_are_collinear():
    df = pd.DataFrame({"y": np.arange(20.0), "c": np.full(20, 3.0)})
    with pytest.raises(NumericalInstability, match="collinear"):
        sp.fast.feols("y ~ 1 + c", data=df)
