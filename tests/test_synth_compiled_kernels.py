"""The compiled inner solvers of the synthetic-control family.

``sp.sdid`` (Frank-Wolfe), ``sp.synth(method='sparse')`` (coordinate
descent) and the matrix-completion solver run loops that were moved out of
the interpreter. Each test below states what must still hold: the kernel
returns what a plain-Python transcription of the same iteration returns,
the weights satisfy their constraints, and the interpreted fallback used
without numba is the same function.

Tolerance: on one machine the kernels reproduce the interpreted loops bit
for bit (they call the same BLAS routines in the same order). The tests
allow 1e-12 absolute so that a platform where NumPy and SciPy link
different BLAS builds, which may round an inner product differently in the
last bit, does not fail them.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

import statspai as sp

pytest.importorskip("numba")

sdid_mod = importlib.import_module("statspai.synth.sdid")
sparse_mod = importlib.import_module("statspai.synth.sparse")
mc_core = importlib.import_module("statspai.matrix_completion._core")

ATOL = 1e-12  # see module docstring


# ----------------------------------------------------------------------
# references: the iterations, written out in plain Python
# ----------------------------------------------------------------------


def _fw_reference(Y, zeta, intercept, weights, min_decrease, max_iter):
    """synthdid's ``sc.weight.fw`` with ``A w`` carried between steps."""
    Y = np.array(Y, dtype=float)
    n, k = Y.shape[0], Y.shape[1] - 1
    w = np.full(k, 1.0 / k) if weights is None else np.array(weights, dtype=float)
    if intercept:
        Y = Y - Y.mean(axis=0, keepdims=True)
    A, b = Y[:, :k].copy(), Y[:, k].copy()
    eta = n * zeta**2
    ax = A @ w
    prev = None
    for _ in range(max_iter):
        resid = ax - b
        half_grad = A.T.copy() @ resid + eta * w
        i = int(np.argmin(half_grad))
        ww = float(w @ w)
        dir_sq = ww - 2.0 * float(w[i]) + 1.0
        if dir_sq != 0.0:
            err_dir = A[:, i] - ax
            denom = float(err_dir @ err_dir) + eta * dir_sq
            num = float(half_grad[i]) - float(half_grad @ w)
            step = 0.0 if denom <= 0 else min(1.0, max(0.0, -num / denom))
            w = w * (1.0 - step)
            w[i] += step
            ax = ax + step * err_dir
            resid = ax - b
        val = zeta**2 * float(w @ w) + float(resid @ resid) / n
        if prev is not None and prev - val <= min_decrease**2:
            break
        prev = val
    return w


def _cd_reference(X, y, lam, max_iter=1000, tol=1e-8):
    """Cyclic coordinate descent on ``0.5 ||y - X w||^2 + lam ||w||_1``."""
    J = X.shape[1]
    w = np.zeros(J)
    r = y.copy()
    norms = np.sum(X**2, axis=0)
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(J):
            if norms[j] < 1e-12:
                continue
            r += X[:, j] * w[j]
            x = float(X[:, j] @ r) / norms[j]
            w[j] = np.sign(x) * max(abs(x) - lam / norms[j], 0.0)
            r -= X[:, j] * w[j]
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w


def _fw_problem(seed, n, k):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, 2)) @ rng.normal(size=(2, k + 1))
    Y += 0.3 * rng.normal(size=(n, k + 1))
    return Y, rng


def _panel(J, T=40, T0=30, seed=0):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(T, 2)).cumsum(axis=0)
    Y = rng.normal(size=(J, 2)) @ factors.T + rng.normal(size=(J, 1))
    Y += rng.normal(scale=0.5, size=(J, T))
    Y[0, T0:] += 2.0
    return pd.DataFrame(
        {
            "unit": np.repeat(np.arange(J), T),
            "time": np.tile(np.arange(T), J),
            "y": Y.ravel(),
        }
    )


# ----------------------------------------------------------------------
# Frank-Wolfe
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("shape", [(12, 5), (30, 19), (7, 25), (1, 4), (9, 1)])
def test_fw_kernel_matches_the_plain_python_iteration(seed, shape):
    n, k = shape
    Y, rng = _fw_problem(seed, n, k)
    zeta = 10 ** rng.uniform(-6, 0)
    start = [None, rng.dirichlet(np.ones(k)), np.eye(k)[rng.integers(k)]][seed % 3]
    for intercept in (True, False):
        for max_iter in (100, 10000):
            kw = dict(zeta=zeta, intercept=intercept, min_decrease=1e-5)
            got = sdid_mod._sc_weight_fw(Y, weights=start, max_iter=max_iter, **kw)
            want = _fw_reference(Y, zeta, intercept, start, 1e-5, max_iter)
            np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)


@pytest.mark.parametrize("seed", range(6))
def test_fw_weights_stay_on_the_simplex(seed):
    Y, rng = _fw_problem(seed, 25, 15)
    for zeta in (1e-6, 0.3):
        w = sdid_mod._solve_synthdid_simplex(
            Y, zeta=zeta, intercept=True, min_decrease=1e-5
        )
        assert w.shape == (15,)
        assert np.all(w >= 0.0)
        # every step is a convex combination with a vertex
        assert abs(w.sum() - 1.0) < 1e-10


def test_fw_does_not_modify_its_inputs():
    Y, rng = _fw_problem(3, 10, 6)
    start = rng.dirichlet(np.ones(6))
    Y0, s0 = Y.copy(), start.copy()
    sdid_mod._sc_weight_fw(
        Y, zeta=0.1, intercept=True, weights=start, min_decrease=1e-5, max_iter=500
    )
    np.testing.assert_array_equal(Y, Y0)
    np.testing.assert_array_equal(start, s0)


def test_fw_interpreted_fallback_is_the_same_function(monkeypatch):
    Y, rng = _fw_problem(11, 20, 12)
    kw = dict(zeta=1e-3, intercept=True, min_decrease=1e-6, max_iter=10000)
    compiled = sdid_mod._sc_weight_fw(Y, **kw)
    # a None entry makes the import inside the solver raise ImportError
    monkeypatch.setitem(sys.modules, "statspai.synth._sdid_kernels", None)
    interpreted = sdid_mod._sc_weight_fw(Y, **kw)
    np.testing.assert_allclose(compiled, interpreted, rtol=0, atol=ATOL)


def test_sdid_estimate_is_the_weighted_double_difference():
    """End to end: tau = (-omega, 1/N1)' Y (-lambda, 1/T1) at the weights."""
    T0 = 14
    d = _panel(12, T=20, T0=T0, seed=5)
    res = sp.sdid(d, "y", "unit", "time", 0, T0, n_reps=5, seed=0)
    Y = d.pivot(index="unit", columns="time", values="y")
    info = res.model_info
    omega = info["unit_weights"].set_index("unit")["weight"]
    lam = np.asarray(info["time_weights"], dtype=float)
    assert omega.size == 11 and lam.size == T0
    assert abs(omega.sum() - 1.0) < 1e-10 and (omega >= 0).all()
    assert abs(lam.sum() - 1.0) < 1e-10 and (lam >= 0).all()
    controls = Y.loc[omega.index].to_numpy()
    treated = Y.loc[0].to_numpy()
    w = omega.to_numpy()
    tau = (treated[T0:].mean() - w @ controls[:, T0:].mean(axis=1)) - (
        treated[:T0] @ lam - w @ controls[:, :T0] @ lam
    )
    assert res.estimate == pytest.approx(tau, abs=1e-10)


# ----------------------------------------------------------------------
# coordinate descent
# ----------------------------------------------------------------------


def _layouts(X):
    """The same matrix in the memory layouts the callers produce."""
    wide = np.zeros((2 * X.shape[0], 2 * X.shape[1]))
    wide[::2, ::2] = X
    return {
        "C": np.ascontiguousarray(X),
        "F": np.asfortranarray(X),
        "transpose-view": np.ascontiguousarray(X.T).T,
        "strided": wide[::2, ::2],
    }


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("shape", [(14, 6), (30, 19), (5, 12), (1, 3)])
def test_cd_kernel_matches_the_plain_python_iteration(seed, shape):
    T, J = shape
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(T, 2)) @ rng.normal(size=(2, J))
    base += 0.3 * rng.normal(size=(T, J))
    y = base @ rng.dirichlet(np.ones(J)) + 0.1 * rng.normal(size=T)
    for lam in (1e-3, 0.3, 50.0):
        for name, X in _layouts(base).items():
            got = sparse_mod._coordinate_descent(X, y, lam)
            want = _cd_reference(X, y, lam)
            np.testing.assert_allclose(got, want, rtol=0, atol=ATOL, err_msg=name)


def test_cd_solution_satisfies_the_lasso_optimality_conditions():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 9))
    y = X[:, :3] @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=40)
    lam = 2.0
    w = sparse_mod._coordinate_descent(X, y, lam, max_iter=5000, tol=1e-12)
    grad = X.T @ (y - X @ w)
    active = w != 0
    # subgradient conditions of 0.5 ||y - X w||^2 + lam ||w||_1
    np.testing.assert_allclose(grad[active], lam * np.sign(w[active]), atol=1e-8)
    assert np.all(np.abs(grad[~active]) <= lam + 1e-8)
    assert active.sum() < 9  # the penalty does select


def test_cd_a_large_penalty_gives_the_zero_vector():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(15, 5))
    y = rng.normal(size=15)
    lam = float(np.max(np.abs(X.T @ y))) * 1.01
    np.testing.assert_array_equal(sparse_mod._coordinate_descent(X, y, lam), 0.0)


def test_cd_interpreted_fallback_is_the_same_function(monkeypatch):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(20, 8))
    y = rng.normal(size=20)
    compiled = sparse_mod._coordinate_descent(X, y, 0.4)
    monkeypatch.setitem(sys.modules, "statspai.synth._sparse_kernels", None)
    interpreted = sparse_mod._coordinate_descent(X, y, 0.4)
    np.testing.assert_allclose(compiled, interpreted, rtol=0, atol=ATOL)


def test_cd_without_donors_still_raises():
    # leave-one-out CV with a single donor relies on this failing loudly
    with pytest.raises(ValueError):
        sparse_mod._coordinate_descent(np.empty((5, 0)), np.ones(5), 0.1)


# ----------------------------------------------------------------------
# matrix completion
# ----------------------------------------------------------------------


def _soft_impute_reference(Y, obs, theta, fixed_effects, max_iter, tol, max_rank):
    """Soft-impute as the module docstring states it, with textbook calls."""
    Y0 = np.where(obs, Y, 0.0)
    F = np.full(Y.shape, Y0[obs].mean())
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        Z = np.where(obs, Y0, F)
        E = Z
        if fixed_effects == "two-way":
            E = Z - Z.mean(axis=1, keepdims=True) - Z.mean(axis=0, keepdims=True)
            E = E + Z.mean()
        elif fixed_effects == "unit":
            E = Z - Z.mean(axis=1, keepdims=True)
        elif fixed_effects == "time":
            E = Z - Z.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(E, full_matrices=False)
        s = np.maximum(s - theta, 0.0)
        if max_rank is not None:
            s[max_rank:] = 0.0
        F_new = (Z - E) + (U * s) @ Vt
        rel = np.linalg.norm(F_new - F) / (np.linalg.norm(F) + 1e-300)
        F = F_new
        if rel < tol:
            break
    return F, s, n_iter


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("fixed_effects", ["two-way", "unit", "time", "none"])
@pytest.mark.parametrize("max_rank", [None, 2])
def test_mc_solver_matches_textbook_soft_impute(seed, fixed_effects, max_rank):
    rng = np.random.default_rng(seed)
    N, T = 25, 9
    Y = rng.normal(size=(N, 2)) @ rng.normal(size=(2, T)) + rng.normal(size=(N, 1))
    Y += 0.3 * rng.normal(size=(N, T))
    obs = rng.uniform(size=(N, T)) < 0.8
    sol = mc_core.mc_nnm_fit(
        Y, obs, 0.7, fixed_effects=fixed_effects, max_rank=max_rank, max_iter=400
    )
    F, s, n_iter = _soft_impute_reference(
        Y, obs, 0.7, fixed_effects, 400, 1e-10, max_rank
    )
    assert sol["n_iter"] == n_iter
    np.testing.assert_allclose(sol["fit"], F, rtol=0, atol=ATOL)
    np.testing.assert_allclose(sol["singular_values"], s, rtol=0, atol=ATOL)


def test_mc_centring_removes_exactly_the_fixed_effects():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(30, 8))
    E = mc_core._center(Z, "two-way")
    np.testing.assert_allclose(E.mean(axis=0), 0.0, atol=1e-13)
    np.testing.assert_allclose(E.mean(axis=1), 0.0, atol=1e-13)
    np.testing.assert_allclose(mc_core._center(Z, "unit").mean(axis=1), 0, atol=1e-13)
    np.testing.assert_allclose(mc_core._center(Z, "time").mean(axis=0), 0, atol=1e-13)
    assert mc_core._center(Z, "none") is Z
    # additive effects are annihilated: only the interaction survives
    a, b = rng.normal(size=(30, 1)), rng.normal(size=(1, 8))
    np.testing.assert_allclose(mc_core._center(a + b + 3.0, "two-way"), 0, atol=1e-12)


# ----------------------------------------------------------------------
# packaging and speed
# ----------------------------------------------------------------------


def test_kernels_are_not_imported_with_the_package():
    code = (
        "import sys, statspai\n"
        "bad = [m for m in ('numba', 'statspai.synth._sdid_kernels',"
        " 'statspai.synth._sparse_kernels') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


def test_sdid_placebo_inference_wall_clock():
    """200 placebo refits on a 20 x 40 panel took 17 s in the interpreter."""
    d = _panel(20)
    # warm-up: absorbs JIT compilation when there is no on-disk cache yet
    sp.sdid(d, "y", "unit", "time", 0, 30, n_reps=2, seed=0)
    start = time.perf_counter()
    res = sp.sdid(d, "y", "unit", "time", 0, 30, seed=0)
    elapsed = time.perf_counter() - start
    assert np.isfinite(res.estimate) and res.se > 0
    assert elapsed < 20.0, f"sp.sdid took {elapsed:.1f}s"


def test_sparse_synth_cross_validation_wall_clock():
    """Leave-one-donor-out CV over 20 penalties took about 30 s interpreted."""
    from statspai.synth import sparse_synth

    d = _panel(20)
    sparse_mod._coordinate_descent(np.eye(3), np.ones(3), 0.1)  # warm-up
    start = time.perf_counter()
    res = sparse_synth(d, "y", "unit", "time", 0, 30)
    elapsed = time.perf_counter() - start
    assert np.isfinite(res.estimate)
    assert elapsed < 20.0, f"sparse_synth took {elapsed:.1f}s"
