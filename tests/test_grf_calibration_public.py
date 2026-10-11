"""``sp.grf_calibration``: the calibration test on predictions made elsewhere.

The regression itself is tied to ``grf::test_calibration`` in
``tests/reference_parity/test_ml_causal_R_parity.py``. These tests pin the
public entry point: it is the regression ``sp.calibration_test`` runs on a
fitted forest, it is the weighted cluster-robust least squares one would
write by hand, and it refuses inputs it cannot grade.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

COLS = ["coef", "se", "t", "p", "ci_low", "ci_high"]


@pytest.fixture(scope="module")
def clustered():
    rng = np.random.default_rng(7)
    n, n_clusters = 500, 25
    cluster = rng.integers(0, n_clusters, size=n)
    X = rng.normal(size=(n, 3))
    W = rng.binomial(1, 0.4 + 0.2 * (X[:, 2] > 0)).astype(float)
    shock = rng.normal(scale=0.5, size=n_clusters)[cluster]
    Y = X[:, 1] + (1.0 + X[:, 0]) * W + shock + rng.normal(size=n)
    kw = dict(
        clusters=cluster,
        equalize_cluster_weights=True,
        n_estimators=200,
        random_state=3,
    )
    Y_hat = np.ravel(sp.regression_forest(y=Y, covariates=X, **kw).predictions)
    W_hat = np.ravel(sp.regression_forest(y=W, covariates=X, **kw).predictions)
    size = pd.Series(cluster).map(pd.Series(cluster).value_counts()).to_numpy()
    return dict(
        Y=Y, W=W, X=X, cluster=cluster, Y_hat=Y_hat, W_hat=W_hat, weights=1.0 / size
    )


def test_is_exported_and_registered():
    assert "grf_calibration" in sp.list_functions()
    schema = sp.function_schema("grf_calibration")
    props = schema["parameters"]["properties"]
    assert {"tau_hat", "clusters", "weights", "vce"} <= set(props)


@pytest.mark.parametrize("with_clusters", [True, False])
def test_reproduces_calibration_test_from_the_forest_outputs(clustered, with_clusters):
    """Fed a forest's own OOB predictions, weights and clusters, the public
    operator is the regression ``sp.calibration_test`` runs on that forest."""
    d = clustered
    kw = dict(n_estimators=200, random_state=5, Y_hat=d["Y_hat"], W_hat=d["W_hat"])
    extra = {}
    if with_clusters:
        kw.update(clusters=d["cluster"], equalize_cluster_weights=True)
        extra = dict(clusters=d["cluster"], weights=d["weights"])
    cf = sp.causal_forest(Y=d["Y"], T=d["W"], X=d["X"], **kw)
    want = sp.calibration_test(cf)
    got = sp.grf_calibration(
        Y=d["Y"],
        W=d["W"],
        Y_hat=d["Y_hat"],
        W_hat=d["W_hat"],
        tau_hat=cf.predict(),
        **extra,
    )
    # Same arithmetic on the same numbers; 1e-12 allows for summation order.
    np.testing.assert_allclose(got[COLS], want[COLS], rtol=1e-12, atol=1e-12)


def test_hc1_is_weighted_cluster_robust_least_squares(clustered):
    """With ``vce='HC1'`` the test is ``sp.regress`` without an intercept,
    weighted and clustered: the regression written out by hand."""
    d = clustered
    rng = np.random.default_rng(11)
    tau = 1.0 + d["X"][:, 0] + rng.normal(scale=0.5, size=len(d["Y"]))
    w = d["weights"]
    got = sp.grf_calibration(
        Y=d["Y"],
        W=d["W"],
        Y_hat=d["Y_hat"],
        W_hat=d["W_hat"],
        tau_hat=tau,
        weights=w,
        clusters=d["cluster"],
        vce="HC1",
    )
    tau_bar = np.sum(w * tau) / np.sum(w)
    frame = pd.DataFrame(
        {
            "r": d["Y"] - d["Y_hat"],
            "C": tau_bar * (d["W"] - d["W_hat"]),
            "D": (tau - tau_bar) * (d["W"] - d["W_hat"]),
            "g": d["cluster"],
            "w": w,
        }
    )
    fit = sp.regress("r ~ C + D - 1", data=frame, cluster="g", weights="w")
    # Two implementations of the same closed form; 1e-10 is rounding.
    np.testing.assert_allclose(
        got["coef"], [fit.params["C"], fit.params["D"]], rtol=1e-10
    )
    np.testing.assert_allclose(
        got["se"], [fit.std_errors["C"], fit.std_errors["D"]], rtol=1e-10
    )


def test_weights_enter_only_up_to_scale_and_zero_weight_rows_drop(clustered):
    d = clustered
    tau = 1.0 + d["X"][:, 0]
    base = dict(Y=d["Y"], W=d["W"], Y_hat=d["Y_hat"], W_hat=d["W_hat"])
    one = sp.grf_calibration(tau_hat=tau, weights=d["weights"], **base)
    scaled = sp.grf_calibration(tau_hat=tau, weights=40.0 * d["weights"], **base)
    np.testing.assert_allclose(scaled[COLS], one[COLS], rtol=1e-10)

    keep = np.arange(len(tau)) % 4 != 0
    zeroed = sp.grf_calibration(tau_hat=tau, weights=keep.astype(float), **base)
    subset = sp.grf_calibration(
        tau_hat=tau[keep], **{k: v[keep] for k, v in base.items()}
    )
    np.testing.assert_allclose(zeroed["coef"], subset["coef"], rtol=1e-10)


def test_earlier_keyword_spelling_still_works(clustered):
    d = clustered
    base = dict(
        Y=d["Y"],
        W=d["W"],
        Y_hat=d["Y_hat"],
        W_hat=d["W_hat"],
        tau_hat=1.0 + d["X"][:, 0],
    )
    old = sp.grf_calibration(vcov_type="HC0", **base)
    new = sp.grf_calibration(vce="HC0", **base)
    pd.testing.assert_frame_equal(old, new)


def test_inputs_it_cannot_grade_are_refused(clustered):
    d = clustered
    n = len(d["Y"])
    tau = 1.0 + d["X"][:, 0]
    base = dict(Y=d["Y"], W=d["W"], Y_hat=d["Y_hat"], W_hat=d["W_hat"])
    with pytest.raises(MethodIncompatibility, match="clusters"):
        sp.grf_calibration(tau_hat=tau, clusters=d["cluster"][:-1], **base)
    missing = tau.copy()
    missing[3] = np.nan
    with pytest.raises(MethodIncompatibility, match="finite"):
        sp.grf_calibration(tau_hat=missing, **base)
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.grf_calibration(tau_hat=tau, weights=-np.ones(n), **base)
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.grf_calibration(tau_hat=tau, alpha=1.5, **base)
    with pytest.raises(DataInsufficient, match="rank deficient"):
        sp.grf_calibration(tau_hat=np.full(n, 0.3), **base)
