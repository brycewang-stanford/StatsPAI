"""Coverage round-2 — ``statspai.rd.rdml`` (ML + RD).

Round 1 covered happy paths of ``rd_forest`` / ``rd_boost`` / ``rd_lasso`` /
``rd_cate_summary``. This file adds the error / option branches:

- ``honesty=False`` forest;
- input-validation errors (no covs, running var in covs, too few obs per
  side, missing covariate column);
- ``rd_cate_summary`` with a method subset and the unknown-method error;
- ``rd_cate_summary`` collecting per-method errors into ``*_error`` keys;
- the variable-importance plot helper (``_importance_plot``) incl. its
  empty-importance guard.

sklearn is installed in this environment, so these paths are reachable. Real
synthetic RD data with a moderator-driven jump; assertions check recovered
magnitude, positive SE, and structural keys — never fabricated numbers.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import MethodIncompatibility  # noqa: E402
from statspai.rd.rdml import _importance_plot  # noqa: E402

JUMP_Z0 = 2.0
JUMP_Z1 = 5.0
ATE = 3.5


def _rd_df(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    treat = (x >= 0).astype(float)
    z = rng.integers(0, 2, n).astype(float)
    cov1 = rng.normal(size=n)
    eff = JUMP_Z0 + (JUMP_Z1 - JUMP_Z0) * z
    y = 0.5 * x + eff * treat + 0.3 * cov1 + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "cov1": cov1})


def test_rd_forest_no_honesty():
    df = _rd_df()
    r = sp.rd_forest(df, y="y", x="x", c=0, covs=["cov1"], n_trees=50, honesty=False)
    est = float(r.estimate)
    assert np.isfinite(est)
    # Planted jump is positive everywhere (eff in {2.0, 5.0}); average ATE=3.5.
    # Recovery band is generous: spans the two subgroup jumps with slack and
    # tolerates forest boundary bias, but still pins sign + magnitude.
    assert est > 0
    assert JUMP_Z0 - 0.5 <= est <= JUMP_Z1 + 1.0
    # SE finite, strictly positive, and not absurdly large for this DGP.
    assert 0 < r.se < 10
    # CI is a proper ordered interval that brackets the point estimate.
    lo, hi = r.ci
    assert lo < hi
    assert lo <= est <= hi
    # P-value is a probability; jump is highly significant here.
    assert 0.0 <= r.pvalue <= 1.0
    # Per-obs CATE detail: one row per estimation-sample obs, all finite, and
    # every subgroup CATE lies within the planted-effect range (sanity bound).
    detail = r.detail
    assert len(detail) == r.model_info["n_estimation"]
    cate = detail["cate"].to_numpy()
    assert np.all(np.isfinite(cate))
    assert cate.min() > 0
    assert JUMP_Z0 - 1.0 <= cate.min() <= cate.max() <= JUMP_Z1 + 1.0
    # ATE is the mean of the per-obs CATEs (internal consistency).
    assert abs(est - cate.mean()) < 1e-6
    # Per-obs CIs are ordered and bracket their own CATE.
    assert np.all(detail["ci_lower"].to_numpy() <= cate)
    assert np.all(cate <= detail["ci_upper"].to_numpy())


def test_rd_forest_requires_covs():
    df = _rd_df()
    with pytest.raises(MethodIncompatibility, match="covariate"):
        sp.rd_forest(df, y="y", x="x", c=0, covs=None, n_trees=50)


def test_rd_forest_running_var_in_covs_errors():
    df = _rd_df()
    with pytest.raises(MethodIncompatibility, match="must not be in covs"):
        sp.rd_forest(df, y="y", x="x", c=0, covs=["x", "cov1"], n_trees=50)


def test_rd_forest_missing_covariate_column():
    df = _rd_df()
    with pytest.raises(MethodIncompatibility, match="not found"):
        sp.rd_forest(df, y="y", x="x", c=0, covs=["does_not_exist"], n_trees=50)


def test_rd_lasso_requires_covs():
    df = _rd_df()
    with pytest.raises(MethodIncompatibility, match="covariates"):
        sp.rd_lasso(df, y="y", x="x", c=0, covs=None)


def test_rd_cate_summary_method_subset():
    df = _rd_df()
    out = sp.rd_cate_summary(df, y="y", x="x", c=0, covs="cov1", methods="lasso")
    assert "lasso" in out
    assert "comparison" in out
    # Only the requested method ran -> exactly one comparison row, no others.
    assert len(out["comparison"]) == 1
    assert "forest" not in out and "boost" not in out
    assert "forest_error" not in out and "boost_error" not in out
    # The lasso result recovers the planted average jump (ATE=3.5) with a
    # generous band, has a positive finite SE, and an ordered bracketing CI.
    res = out["lasso"]
    est = float(res.estimate)
    assert np.isfinite(est)
    assert est > 0
    assert JUMP_Z0 - 1.0 <= est <= JUMP_Z1 + 1.0
    assert 0 < res.se < 10
    lo, hi = res.ci
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
    assert lo <= est <= hi
    assert 0.0 <= res.pvalue <= 1.0
    # The comparison row mirrors the underlying CausalResult.
    row = out["comparison"].iloc[0]
    assert row["method"] == "LASSO RD"
    assert abs(float(row["estimate"]) - est) < 1e-9
    assert abs(float(row["se"]) - res.se) < 1e-9
    assert int(row["n_obs"]) == res.n_obs


def test_rd_cate_summary_unknown_method():
    df = _rd_df()
    with pytest.raises(MethodIncompatibility, match="Unknown methods"):
        sp.rd_cate_summary(
            df, y="y", x="x", c=0, covs=["cov1"], methods=["forest", "nope"]
        )


def test_rd_cate_summary_collects_errors():
    # too few covs/obs forces the per-method try/except error capture
    df = _rd_df(n=1500)
    out = sp.rd_cate_summary(
        df, y="y", x="x", c=0, covs=["cov1"], h=0.001, methods=["forest"]
    )
    # forest should fail (too few obs in tiny bandwidth) -> forest_error key,
    # captured as a string rather than swallowed, and no successful result.
    assert "forest_error" in out
    assert "forest" not in out
    assert isinstance(out["forest_error"], str) and out["forest_error"]
    # With no method succeeding, the comparison table exists but is empty.
    assert "comparison" in out
    assert len(out["comparison"]) == 0
    # No forest result -> heterogeneity drivers fall back to empty dict.
    assert out.get("heterogeneity_drivers") == {}


def test_importance_plot_from_forest():
    df = _rd_df()
    r = sp.rd_forest(df, y="y", x="x", c=0, covs=["cov1", "z"], n_trees=50)
    fig0, ax0 = plt.subplots()
    ax = _importance_plot(r, top_k=5, ax=ax0)
    assert ax is not None
    # Plot reuses the supplied axes (no fresh one created).
    assert ax is ax0
    # Two covariates -> two horizontal bars; widths are the (non-negative)
    # importances, which sum to ~1 across all features.
    assert len(ax.patches) == 2
    widths = [p.get_width() for p in ax.patches]
    assert all(np.isfinite(w) and w >= 0 for w in widths)
    assert abs(sum(widths) - 1.0) < 1e-6
    assert ax.get_xlabel() == "Variable Importance"
    plt.close("all")


def test_importance_plot_creates_axes():
    df = _rd_df()
    r = sp.rd_forest(df, y="y", x="x", c=0, covs=["cov1", "z"], n_trees=50)
    ax = _importance_plot(r)
    assert ax is not None
    # One bar per covariate, drawn from the result's variable_importance.
    vi = r.model_info["variable_importance"]
    assert len(ax.patches) == len(vi)
    widths = [p.get_width() for p in ax.patches]
    assert all(w >= 0 for w in widths)
    assert abs(sum(widths) - sum(vi.values())) < 1e-9
    assert ax.get_xlabel() == "Variable Importance"
    plt.close("all")


def test_importance_plot_empty_raises():
    df = _rd_df()
    r = sp.rd_lasso(df, y="y", x="x", c=0, covs=["cov1"])
    # rd_lasso result has no variable_importance -> guard raises
    with pytest.raises(MethodIncompatibility, match="variable_importance"):
        _importance_plot(r)
