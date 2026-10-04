"""``sp.fect`` on panels too large for a dense dummy design.

The initial two-way fit used to build one dummy column per unit, so a panel
of 7,248 units and 40 periods (Moser and Voena 2012, an application of the
de Chaisemartin and D'Haultfoeuille textbook) needed a 17 GB matrix and did
not run. Above ``_DENSE_INITIAL_FIT_CELLS`` the same least-squares fit is
now solved by alternating projections.
"""

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# ``statspai.synth.fect`` the attribute is the function; the module is here
FECT = sys.modules["statspai.synth.fect"]


@pytest.mark.parametrize("force", [0, 1, 2, 3])
@pytest.mark.parametrize("n_cov", [0, 2])
def test_iterative_initial_fit_is_the_dense_least_squares_fit(force, n_cov):
    rng = np.random.default_rng(10 * force + n_cov)
    T, N = 12, 40
    Y = rng.normal(size=(T, N))
    X = rng.normal(size=(T, N, n_cov)) if n_cov else None
    untreated = (rng.uniform(size=(T, N)) > 0.25).astype(int)
    # every unit and every period keeps an untreated cell
    untreated[0, :] = 1
    untreated[:, 0] = 1
    dense, beta_dense = FECT._initial_fit(Y, X, untreated, force)
    iterative, beta_iter = FECT._initial_fit_iterative(Y, X, untreated, force)
    np.testing.assert_allclose(iterative, dense, rtol=0, atol=1e-11)
    np.testing.assert_allclose(beta_iter, beta_dense, rtol=0, atol=1e-11)


def _panel(seed=1, n=60):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        g = int(rng.choice([0, 6, 8]))
        a, lam = rng.normal(), rng.normal()
        for t in range(1, 13):
            d = int(g > 0 and t >= g)
            y = a + 0.2 * t + lam * np.sin(t / 2.0) + d + rng.normal(scale=0.3)
            rows.append((i, t, d, y))
    return pd.DataFrame(rows, columns=["i", "t", "d", "y"])


@pytest.mark.parametrize("method, extra", [("fe", {}), ("ife", {"r": 1})])
def test_estimates_do_not_depend_on_the_path(monkeypatch, method, extra):
    df = _panel()
    kwargs = dict(treat="d", unit="i", time="t", method=method, tol=1e-9, **extra)
    dense = sp.fect(df, "y", **kwargs)
    monkeypatch.setattr(FECT, "_DENSE_INITIAL_FIT_CELLS", 0)
    iterative = sp.fect(df, "y", **kwargs)
    assert iterative.estimate == pytest.approx(dense.estimate, abs=1e-7)
    np.testing.assert_allclose(
        iterative.detail["att"], dense.detail["att"], rtol=0, atol=1e-6
    )


def test_a_large_panel_runs_and_equals_two_way_fixed_effects():
    """2,000 units by 30 periods: 60,000 rows and 2,029 dummy columns."""
    rng = np.random.default_rng(0)
    n, t_max = 2000, 30
    unit = np.repeat(np.arange(n), t_max)
    time = np.tile(np.arange(1, t_max + 1), n)
    treated = unit < 100
    d = (treated & (time >= 20)).astype(int)
    y = (
        rng.normal(size=n)[unit]
        + 0.1 * time
        + 0.5 * d
        + rng.normal(scale=0.5, size=n * t_max)
    )
    df = pd.DataFrame({"i": unit, "t": time, "d": d, "y": y})
    assert n * t_max * (n + t_max) > FECT._DENSE_INITIAL_FIT_CELLS
    r = sp.fect(df, "y", treat="d", unit="i", time="t", method="fe")
    # one treatment date on a balanced panel: imputation is two-way FE
    twfe = sp.feols("y ~ d | i + t", data=df)
    assert r.estimate == pytest.approx(float(twfe.params["d"]), abs=1e-6)


def test_stopping_at_the_iteration_cap_warns():
    df = _panel()
    with pytest.warns(sp.exceptions.ConvergenceWarning, match="max_iter"):
        r = sp.fect(
            df, "y", treat="d", unit="i", time="t", method="ife", r=1,
            tol=1e-12, max_iter=5,
        )  # fmt: skip
    assert r.model_info["converged"] is False
    with warnings.catch_warnings():
        warnings.simplefilter("error", sp.exceptions.ConvergenceWarning)
        ok = sp.fect(df, "y", treat="d", unit="i", time="t", method="ife", r=1)
    assert ok.model_info["converged"] is True
