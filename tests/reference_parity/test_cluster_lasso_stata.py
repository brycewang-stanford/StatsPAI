"""Cluster-Lasso against Stata's lassopack / pdslasso.

``sp.rlasso(cluster=)`` and the clustered ``sp.rlasso_effect`` /
``sp.rlasso_iv`` are compared with ``rlasso, cluster()``, ``pdslasso,
cluster()`` and ``ivlasso, cluster()`` (Ahrens, Hansen and Schaffer) on the
same CSV bytes. The reference numbers are in
``_fixtures/cluster_lasso_Stata.json``, produced by
``_fixtures/_generate_cluster_lasso_Stata.do``; the data are a simulated
within-transformed panel (60 units, 6 periods, AR(1) regressors and errors)
rounded to six decimals so both sides read identical values.

Tolerances. Point estimates are post-selection OLS / IV on the same support,
so they agree to solver precision; ``rtol=1e-9`` leaves room for Stata
printing 12-14 significant digits. Variances likewise.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.rlasso._core import _cluster_codes, _lasso_shooting, _score_norm

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "cluster_lasso_Stata.json").read_text(encoding="utf-8"))
XS = [f"x{j + 1}" for j in range(24)]
ZS = [f"z{j + 1}" for j in range(16)]
RTOL = 1e-9


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(FIX / "cluster_lasso_panel.csv")


def test_first_stage_matches_lassopack(panel):
    ref = REF["panel"]
    cols = ZS + XS
    fit = sp.rlasso(
        panel[cols].to_numpy(),
        panel["d"].to_numpy(),
        intercept=False,
        colnames=cols,
        cluster=panel["id"].to_numpy(),
    )
    assert fit.lambda0 == pytest.approx(ref["lambda0"], rel=1e-12)
    assert [c for c, s in zip(cols, fit.index) if s] == ref["selected"]
    np.testing.assert_allclose(fit.beta[fit.index], ref["betaOLS"], rtol=RTOL)


def test_clustering_changes_the_selection(panel):
    """Without ``cluster=`` both packages keep one more variable (x21)."""
    cols = ZS + XS
    fit = sp.rlasso(
        panel[cols].to_numpy(), panel["d"].to_numpy(), intercept=False, colnames=cols
    )
    assert [c for c, s in zip(cols, fit.index) if s] == REF["panel"]["selected_robust"]


@pytest.mark.parametrize(
    "method, key", [("double selection", "pds"), ("partialling out", "plasso")]
)
def test_effect_matches_pdslasso(panel, method, key):
    ref = REF["panel"][key]
    res = sp.rlasso_effect(XS, "y", "d", method=method, data=panel, cluster="id")
    assert res.alpha == pytest.approx(ref["beta"], rel=RTOL)
    assert res.se**2 == pytest.approx(ref["V"], rel=RTOL)


def test_iv_matches_ivlasso(panel):
    both = sp.rlasso_iv("y", "d", ZS, XS, data=panel, cluster="id", intercept=False)
    ref = REF["panel"]["iv_plasso"]
    assert both.coef[0] == pytest.approx(ref["beta"], rel=RTOL)
    assert both.se[0] ** 2 == pytest.approx(ref["V"], rel=RTOL)

    only_z = sp.rlasso_iv(
        "y", "d", ZS, None, data=panel, cluster="id", select_X=False, intercept=False
    )
    ref = REF["panel"]["ivZ_plasso"]
    assert only_z.coef[0] == pytest.approx(ref["beta"], rel=RTOL)
    assert only_z.se[0] ** 2 == pytest.approx(ref["V"], rel=RTOL)


def test_singleton_clusters_reduce_to_heteroskedastic_lasso(panel):
    """One observation per cluster: the cluster loading is the White loading."""
    cols = ZS + XS
    X, d = panel[cols].to_numpy(), panel["d"].to_numpy()
    plain = sp.rlasso(X, d)
    single = sp.rlasso(
        X, d, cluster=np.arange(len(d)), penalty={"gamma": 0.1 / np.log(len(d))}
    )
    np.testing.assert_array_equal(plain.index, single.index)
    np.testing.assert_allclose(plain.beta, single.beta, rtol=0, atol=1e-12)
    np.testing.assert_allclose(plain.loadings, single.loadings, rtol=1e-12)


def _is_fixed_point(X, y, codes, support, lambda0):
    """Does the loading iteration map ``support`` to itself?"""
    coef, *_ = np.linalg.lstsq(X[:, support], y, rcond=None)
    resid = y - X[:, support] @ coef
    lam = lambda0 * _score_norm(X, resid, codes) / np.sqrt(len(y))
    beta = _lasso_shooting(X, y, lam, X.T @ X, X.T @ y)
    return np.array_equal(np.abs(beta) > 0, support)


def test_path_dependence_is_a_second_fixed_point():
    """Where the two packages disagree, both supports are self-consistent.

    The loadings depend on the residuals and the residuals on the selected
    support, so the iteration can have more than one fixed point. On this
    panel lassopack stops at {z1 z2 z3 x2 x3} and hdm's iteration (which
    StatsPAI follows) at {z1 z2 z3 x1 x2 x3}. Each support reproduces itself
    under StatsPAI's cluster loadings, and on lassopack's support the
    post-Lasso coefficients agree: the difference is the path, not the
    penalty.
    """
    df = pd.read_csv(FIX / "cluster_lasso_panel_path.csv")
    cols = ZS + XS
    X, d = df[cols].to_numpy(), df["d"].to_numpy()
    codes, _ = _cluster_codes(df["id"].to_numpy(), len(d))
    fit = sp.rlasso(X, d, intercept=False, colnames=cols, cluster=codes)
    ours = fit.index
    theirs = np.isin(cols, REF["path"]["selected"])
    assert not np.array_equal(ours, theirs)
    assert _is_fixed_point(X, d, codes, ours, fit.lambda0)
    assert _is_fixed_point(X, d, codes, theirs, fit.lambda0)
    coef, *_ = np.linalg.lstsq(X[:, theirs], d, rcond=None)
    np.testing.assert_allclose(coef, REF["path"]["betaOLS"], rtol=RTOL)


def test_cluster_argument_is_validated(panel):
    X, d = panel[XS].to_numpy(), panel["d"].to_numpy()
    with pytest.raises(ValueError, match="one entry per row"):
        sp.rlasso(X, d, cluster=np.arange(5))
    with pytest.raises(ValueError, match="at least two clusters"):
        sp.rlasso(X, d, cluster=np.zeros(len(d)))
    with pytest.raises(ValueError, match="heteroskedastic loadings"):
        sp.rlasso(X, d, cluster=panel["id"], penalty={"homoscedastic": True})
    with pytest.raises(ValueError, match="needs `data`"):
        sp.rlasso_effect(X, panel["y"].to_numpy(), d, cluster="id")
