"""Reference parity: weighted HC1-robust SEs in ``sp.did_2x2`` / ``sp.ddd``.

Pins the 2026-07 ⚠️ correctness fix against **Stata 18 MP** values captured
live on 2026-07-23 from an identical deterministic dataset (seed
``20260723``, regenerated in :func:`_make_data` below).

The bug: with analytic weights the ``robust=True`` branch built the sandwich
meat as ``X' diag(w e^2) X``, but the WLS score is ``w_i x_i e_i``, so the
correct meat is ``sum_i w_i^2 e_i^2 x_i x_i'`` (Stata aweight-robust / R
``sandwich`` convention).  The unsquared-``w`` meat produced SEs ~9% away
from ``regress ..., [aw=w] robust`` on dispersed weights, while the cluster
branch in the same functions squared the score correctly (and matched Stata
all along).

Stata reference commands::

    import delimited did2x2_w.csv, clear asdouble
    gen double tp  = treat*post
    gen double ts  = treat*sub
    gen double ps  = post*sub
    gen double tps = treat*post*sub
    regress y treat post tp [aw=w], robust                 // 2x2
    regress y treat post sub tp ts ps tps [aw=w], robust   // DDD

Stata reference values are hard-coded constants (the audited reference
baseline); the data is regenerated deterministically so the test is hermetic
(no Stata needed at run time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# Stata 18 MP, captured 2026-07-23 (display %21.17f, asdouble import)
STATA_2X2_B = 1.31672389689084768
STATA_2X2_SE = 0.27826909422959262
STATA_DDD_B = 1.31207350632160025
STATA_DDD_SE = 0.53513022394411458


def _make_data() -> pd.DataFrame:
    """Recreate the exact dataset used to capture the Stata 18 references."""
    rng = np.random.default_rng(20260723)
    n = 500
    treat = rng.integers(0, 2, n)
    post = rng.integers(0, 2, n)
    sub = rng.integers(0, 2, n)
    w = rng.uniform(0.2, 5.0, n)  # deliberately dispersed weights
    y = (
        1.0
        + 0.5 * treat
        + 0.3 * post
        + 0.4 * sub
        + 1.2 * treat * post
        + 0.2 * treat * sub
        + 0.1 * post * sub
        + 0.7 * treat * post * sub
        + rng.normal(0, 1 + 0.5 * treat, n)
    )
    return pd.DataFrame({"y": y, "treat": treat, "post": post, "sub": sub, "w": w})


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return _make_data()


def test_did_2x2_weighted_robust_matches_stata(data: pd.DataFrame) -> None:
    res = sp.did_2x2(data, y="y", treat="treat", time="post", weights="w", robust=True)
    # Machine precision: observed agreement ~2e-16 (b) / ~3e-16 (se)
    np.testing.assert_allclose(res.estimate, STATA_2X2_B, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.se, STATA_2X2_SE, rtol=0, atol=1e-12)


def test_ddd_weighted_robust_matches_stata(data: pd.DataFrame) -> None:
    res = sp.ddd(
        data,
        y="y",
        treat="treat",
        time="post",
        subgroup="sub",
        weights="w",
        robust=True,
    )
    np.testing.assert_allclose(res.estimate, STATA_DDD_B, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.se, STATA_DDD_SE, rtol=0, atol=1e-12)


def test_weighted_robust_differs_from_unsquared_w_meat(data: pd.DataFrame) -> None:
    """Tripwire: the correct w^2 meat must NOT equal the old unsquared-w meat.

    Recomputes the historical (buggy) SE directly from the WLS algebra; if a
    refactor ever reverts the meat to ``w`` the parity tests above fail on
    the Stata anchor AND this test documents exactly which formula came back.
    """
    n = len(data)
    treat = data["treat"].to_numpy(float)
    post = data["post"].to_numpy(float)
    X = np.column_stack([np.ones(n), treat, post, treat * post])
    w = data["w"].to_numpy(float)
    w = w * (n / w.sum())
    y = data["y"].to_numpy(float)
    XtWX_inv = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = XtWX_inv @ (X * w[:, None]).T @ y
    resid = y - X @ beta
    k = X.shape[1]
    meat_old = X.T @ (X * ((n / (n - k)) * w * resid**2)[:, None])
    se_old = float(np.sqrt((XtWX_inv @ meat_old @ XtWX_inv)[3, 3]))
    assert abs(se_old - STATA_2X2_SE) > 1e-3, (
        "unsquared-w SE unexpectedly equals the Stata anchor; "
        "tripwire no longer discriminates"
    )
