"""Option scan (2026-10-27): variable selection (stepwise, best_subset, lasso_select).

Each test states the correct behaviour; every one failed before the fix
landed with it (the defect is named in the test's comment or docstring).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp


def _orthogonal_noise(M: np.ndarray, rng: np.random.Generator, scale: float = 1.0):
    """Noise exactly orthogonal to the columns of M, with sd ``scale``."""
    e = rng.normal(size=M.shape[0])
    e = e - M @ np.linalg.lstsq(M, e, rcond=None)[0]
    return scale * e / e.std()


# ----------------------------------------------------------------------
# 5. best_subset must be able to choose the empty model
# ----------------------------------------------------------------------
@pytest.mark.parametrize("criterion", ["bic", "aic", "adjr2"])
def test_best_subset_selects_nothing_when_no_candidate_reduces_rss(criterion):
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 4))
    y = 3.0 + _orthogonal_noise(np.column_stack([np.ones(n), X]), rng)
    df = pd.DataFrame(X, columns=list("abcd")).assign(y=y)
    res = sp.best_subset(df, "y", list("abcd"), criterion=criterion)
    assert res.selected == []
    assert res.dropped == list("abcd")


# ----------------------------------------------------------------------
# 6. stepwise(criterion='pvalue') enters on p < alpha_in alone
# ----------------------------------------------------------------------
def _pvalues(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    s2 = r @ r / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return 2 * stats.t.sf(np.abs(beta / se), n - k)


def _forward_pvalue(df: pd.DataFrame, y: str, cands: list, alpha_in: float) -> list:
    """Textbook forward selection: add the smallest p-value while p < alpha_in
    (Stata ``stepwise, pe()``)."""
    chosen: list = []
    yv = df[y].to_numpy(float)
    while True:
        best, best_p = None, alpha_in
        for c in cands:
            if c in chosen:
                continue
            X = np.column_stack([np.ones(len(df))] + [df[v] for v in chosen + [c]])
            p = _pvalues(X, yv)[-1]
            if p < best_p:
                best, best_p = c, p
        if best is None:
            return chosen
        chosen.append(best)


def test_stepwise_pvalue_forward_matches_textbook_algorithm():
    rng = np.random.default_rng(1)
    n = 2000
    x1, x2, x3 = rng.normal(size=(3, n))
    M = np.column_stack([np.ones(n), x1, x2])
    e = _orthogonal_noise(M, rng)
    # t-statistic of x2 set to exactly 2.3 (p ~ 0.021): with e orthogonal to
    # [1, x1, x2] the fitted slopes are exactly (1, c) and se(c) does not
    # depend on c
    r2 = (
        x2
        - np.column_stack([np.ones(n), x1])
        @ np.linalg.lstsq(np.column_stack([np.ones(n), x1]), x2, rcond=None)[0]
    )
    s = np.sqrt(e @ e / (n - 3))
    c = 2.3 * s / np.sqrt(r2 @ r2)
    y = x1 + c * x2 + e
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    ref = _forward_pvalue(df, "y", ["x1", "x2", "x3"], 0.05)
    assert ref[:2] == ["x1", "x2"]  # the design does what it says
    res = sp.stepwise(
        df,
        y="y",
        x=["x1", "x2", "x3"],
        method="forward",
        criterion="pvalue",
        alpha_in=0.05,
        verbose=False,
    )
    assert res.selected == ref


# ----------------------------------------------------------------------
# 12. lasso_select rejects an unknown method
# ----------------------------------------------------------------------
def test_lasso_select_rejects_unknown_method():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    df["y"] = df["a"] + rng.normal(size=80)
    with pytest.raises(ValueError):
        sp.lasso_select(df, y="y", x=["a", "b", "c"], method="BIC", verbose=False)


@pytest.mark.parametrize(
    "method,start",
    [("forward", None), ("backward", None), ("both", None), ("both", "full")],
)
def test_stepwise_pvalue_matches_stata(method, start):
    """Stata 18 on this data (written with %.17g, read ``asdouble``):
    ``stepwise, pe(.05)``, ``pr(.10)``, ``pr(.10) pe(.05) forward`` and
    ``pr(.10) pe(.05)`` all keep x1 and x3 (p = 0.0333); 2026-10-27 run."""
    rng = np.random.default_rng(7)
    n = 2000
    X = rng.normal(size=(n, 6))
    cols = [f"x{j}" for j in range(1, 7)]
    y = X[:, 0] + 0.06 * X[:, 1] + 0.045 * X[:, 2] + 0.03 * X[:, 3] + rng.normal(size=n)
    df = pd.DataFrame(X, columns=cols).assign(y=y)
    kw = {} if start is None else {"start": start}
    res = sp.stepwise(
        df,
        "y",
        cols,
        method=method,
        criterion="pvalue",
        alpha_in=0.05,
        alpha_out=0.10,
        verbose=False,
        **kw,
    )
    assert sorted(res.selected) == ["x1", "x3"]
