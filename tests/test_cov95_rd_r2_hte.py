"""Coverage round-2 — ``statspai.rd.hte`` (heterogeneous RD effects).

Round 1 covered the scalar-moderator happy path. This file adds:

- clustered SE path;
- multivariate moderator Z (2-dim) with grid evaluation;
- explicit ``eval_points``;
- ``result.plot()`` (the ``_rdhte_plot`` helper, incl. the dz>1 guard);
- ``rdhte_lincom`` covariance assembly + length-mismatch error;
- input-validation error branches (bad kernel, bad p, missing columns,
  missing cluster column);
- ``rdbwhte`` with a covariate list.

Real synthetic RD data with a moderator-driven jump; assertions check the
recovered level / spread, positive bandwidth, ordered CIs — never fabricated
numbers.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402

JUMP_Z0 = 2.0
JUMP_Z1 = 5.0


def _hte_df(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    treat = (x >= 0).astype(float)
    z = rng.integers(0, 2, n).astype(float)
    z2 = rng.normal(size=n)
    eff = JUMP_Z0 + (JUMP_Z1 - JUMP_Z0) * z
    y = 0.5 * x + eff * treat + 0.2 * z2 + rng.normal(0, 0.4, n)
    g = (np.arange(n) // 20).astype(int)
    return pd.DataFrame({"y": y, "x": x, "z": z, "z2": z2, "g": g})


def test_rdhte_clustered_se():
    df = _hte_df()
    r = sp.rdhte(df, y="y", x="x", z="z", c=0, cluster="g")
    assert (r.detail["se"] > 0).all()
    assert r.se > 0


def test_rdhte_multivariate_z():
    df = _hte_df()
    r = sp.rdhte(df, y="y", x="x", z=["z", "z2"], c=0, n_eval=12)
    det = r.detail
    assert "cate" in det.columns
    assert r.model_info["n_z"] == 2
    assert len(det) >= 1


def test_rdhte_explicit_eval_points():
    df = _hte_df()
    pts = np.array([0.0, 1.0])
    r = sp.rdhte(df, y="y", x="x", z="z", c=0, eval_points=pts)
    by_z = r.detail.groupby("z_value")["cate"].first()
    assert abs(by_z.loc[by_z.index.min()] - JUMP_Z0) < 1.0


def test_rdhte_plot_scalar():
    df = _hte_df()
    r = sp.rdhte(df, y="y", x="x", z="z", c=0)
    ax = r.plot()
    assert ax is not None
    plt.close("all")


def test_rdhte_plot_provided_axes():
    df = _hte_df()
    r = sp.rdhte(df, y="y", x="x", z="z", c=0)
    fig0, ax0 = plt.subplots()
    ax = r.plot(ax=ax0, title="HTE", xlabel="z")
    assert ax is ax0
    plt.close("all")


def test_rdhte_plot_multivariate_raises():
    df = _hte_df()
    r = sp.rdhte(df, y="y", x="x", z=["z", "z2"], c=0, n_eval=9)
    with pytest.raises(NotImplementedError):
        r.plot()
    plt.close("all")


def test_rdhte_lincom_average():
    df = _hte_df()
    r = sp.rdhte(df, y="y", x="x", z="z", c=0)
    n = len(r.detail)
    w = np.ones(n) / n
    out = sp.rdhte_lincom(r, weights=w)
    assert {"estimate", "se", "ci", "pvalue"}.issubset(out)
    assert out["se"] > 0
    assert out["ci"][0] < out["ci"][1]


def test_rdhte_lincom_length_mismatch():
    df = _hte_df()
    r = sp.rdhte(df, y="y", x="x", z="z", c=0)
    with pytest.raises(ValueError, match="length"):
        sp.rdhte_lincom(r, weights=np.ones(len(r.detail) + 3))


def test_rdhte_validation_errors():
    df = _hte_df()
    with pytest.raises(ValueError, match="kernel"):
        sp.rdhte(df, y="y", x="x", z="z", c=0, kernel="bad")
    with pytest.raises(ValueError, match="p must"):
        sp.rdhte(df, y="y", x="x", z="z", c=0, p=0)
    with pytest.raises(ValueError, match="not found"):
        sp.rdhte(df, y="nope", x="x", z="z", c=0)
    with pytest.raises(ValueError, match="[Cc]luster"):
        sp.rdhte(df, y="y", x="x", z="z", c=0, cluster="missing_col")


def test_rdbwhte_with_cov_list():
    df = _hte_df()
    h = sp.rdbwhte(df, y="y", x="x", z=["z", "z2"], c=0)
    assert np.isfinite(h) and h > 0
