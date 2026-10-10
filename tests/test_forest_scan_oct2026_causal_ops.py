"""Post-fit operators of ``sp.causal_forest`` against hand computations.

Given a fitted forest, everything downstream of it is a deterministic
function of four stored arrays: the out-of-bag CATE (``cf.predict()``), the
nuisances (``cf.get_nuisances()``), the outcome and the treatment.  Each
test below rebuilds one reported number from those arrays with plain numpy
and compares exactly, so an option that is read and then dropped (weights,
clusters, ``subset``, ``clip``, ``vce``, ``alpha``) shows up as a mismatch.

The forests are small (a few hundred rows, 150 trees) and fitted once per
module.  Nothing here asserts that a forest estimate is close to a truth:
only that the arithmetic after the forest is the documented arithmetic.

Tolerances: ``RTOL = 1e-10`` throughout.  The two sides differ only in the
order of floating-point sums, which is good for 1e-13 on these sizes; 1e-10
leaves three digits of slack without admitting a wrong formula.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.forest.forest_inference import (
    _rate_half_sample_se,
    _rate_influence_se,
    _weighted_rate_from_scores,
    _weighted_rate_influence_se,
    aipw_scores,
    grf_att_atc,
    grf_calibration,
    grf_overlap_ate,
    rate_from_scores,
    rate_rank_weights,
)

RTOL = 1e-10
TREES = 150


# --------------------------------------------------------------------------- #
#  Hand-written reference formulas
# --------------------------------------------------------------------------- #


def _aipw(y, w, m, e, tau):
    """The AIPW score of the ``aipw_scores`` docstring."""
    return tau + (w - e) / (e * (1.0 - e)) * (y - m - (w - e) * tau)


def _sandwich(D, resid, kind, w=None, clusters=None):
    """``sandwich::vcovCL`` conventions, written from their definitions.

    Without clusters every row is its own cluster, so HC0 carries the
    ``n / (n - 1)`` cluster adjustment and HC1 ``n / (n - k)``; HC2 / HC3
    divide the residual by ``sqrt(1 - h)`` / ``1 - h``.  With clusters,
    HC0 / HC1 sum the estimating functions within cluster.
    """
    n, k = D.shape
    w = np.ones(n) if w is None else w
    bread = np.linalg.inv(D.T @ (D * w[:, None]))
    ef = D * (w * resid)[:, None]
    if clusters is None:
        if kind in ("HC0", "HC1"):
            meat = ef.T @ ef * n / (n - 1)
            if kind == "HC1":
                meat = meat * (n - 1) / (n - k)
        else:
            h = w * np.einsum("ij,jk,ik->i", D, bread, D)
            adj = ef / (np.sqrt(1 - h) if kind == "HC2" else (1 - h))[:, None]
            meat = adj.T @ adj
    else:
        assert kind in ("HC0", "HC1")
        labels = np.unique(clusters)
        sums = np.vstack([ef[clusters == g].sum(axis=0) for g in labels])
        G = len(labels)
        meat = sums.T @ sums * G / (G - 1)
        if kind == "HC1":
            meat = meat * (n - 1) / (n - k)
    return bread @ meat @ bread


def _wls(D, y, w=None):
    w = np.ones(len(y)) if w is None else w
    beta = np.linalg.solve(D.T @ (D * w[:, None]), D.T @ (w * y))
    return beta, y - D @ beta


def _mean_se(values, w=None, clusters=None):
    """Weighted mean and grf's standard error of it (docstring of
    ``_grf_family.score_average``)."""
    n = len(values)
    w = np.ones(n) if w is None else w
    est = np.sum(w * values) / np.sum(w)
    if clusters is None:
        var = np.sum(w**2 * (values - est) ** 2) / np.sum(w) ** 2 * n / (n - 1)
    else:
        labels = np.unique(clusters)
        sums = np.array([np.sum((w * (values - est))[clusters == g]) for g in labels])
        G = len(labels)
        var = np.sum(sums**2) / np.sum(w) ** 2 * G / (G - 1)
    return est, np.sqrt(var)


def _att_atc(y, t, m, e, tau, target, w=None, clusters=None):
    """ATT / ATC as documented in ``grf_att_atc`` (weights added)."""
    n = len(y)
    w = np.ones(n) if w is None else w
    treated = t == 1
    control = ~treated
    idx = treated if target == "treated" else control
    tau_raw = np.sum(w[idx] * tau[idx]) / np.sum(w[idx])
    tau_var = np.sum(w[idx] ** 2 * (tau[idx] - tau_raw) ** 2) / np.sum(w[idx]) ** 2
    gamma = np.zeros(n)
    if target == "treated":
        gamma[control] = e[control] / (1 - e[control])
        gamma[treated] = 1.0
    else:
        gamma[control] = 1.0
        gamma[treated] = (1 - e[treated]) / e[treated]
    for arm in (treated, control):
        gamma[arm] = gamma[arm] / np.sum(w[arm] * gamma[arm]) * np.sum(w)
    mu0 = m - e * tau
    mu1 = m + (1 - e) * tau
    corr = t * gamma * (y - mu1) - (1 - t) * gamma * (y - mu0)
    dr = np.sum(w * corr) / np.sum(w)
    if clusters is None:
        var = np.sum(w**2 * corr**2) / np.sum(w) ** 2 * n / (n - 1)
    else:
        labels = np.unique(clusters)
        sums = np.array([np.sum((w * corr)[clusters == g]) for g in labels])
        G = len(labels)
        var = np.sum(sums**2) / np.sum(w) ** 2 * G / (G - 1)
    return tau_raw + dr, np.sqrt(tau_var + var)


def _rate(scores, prio, target):
    """Unweighted RATE as rate_from_scores() documents it.

    Scores are averaged within tied priorities and sorted by decreasing
    priority; TOC_k is the mean of the first k minus the overall mean,
    AUTOC = mean_k TOC_k and QINI = mean_k (k / n) TOC_k.
    """
    s = np.asarray(scores, dtype=float)
    pr = np.asarray(prio, dtype=float)
    s = np.array([s[pr == v].mean() for v in pr])
    s = s[np.argsort(-pr, kind="stable")]
    n = len(s)
    k = np.arange(1, n + 1)
    toc = np.cumsum(s) / k - s.mean()
    return float(np.mean(toc) if target == "AUTOC" else np.mean(k / n * toc))


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #


def _arrays(cf):
    nu = cf.get_nuisances()
    return (
        np.asarray(cf._Y_original, dtype=float),
        np.asarray(cf._T_original, dtype=float),
        np.asarray(nu["Y_hat"], dtype=float),
        np.asarray(nu["W_hat"], dtype=float),
        cf.predict(),
    )


@pytest.fixture(scope="module")
def plain():
    rng = np.random.default_rng(101)
    n = 500
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 1 / (1 + np.exp(-0.6 * X[:, 0]))).astype(float)
    Y = X[:, 1] + (1 + X[:, 0]) * T + rng.normal(size=n)
    cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=TREES, random_state=11)
    return cf, X


@pytest.fixture(scope="module")
def extreme():
    """Forest with user-supplied propensities that reach 0.004 and 0.996,
    so that ``clip`` has something to clip."""
    rng = np.random.default_rng(102)
    n = 500
    X = rng.normal(size=(n, 2))
    e = 1 / (1 + np.exp(-2.5 * X[:, 0]))
    T = rng.binomial(1, e).astype(float)
    T[:3], T[3:6] = 1.0, 0.0
    Y = X[:, 1] + T + rng.normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = sp.causal_forest(
            Y=Y, T=T, X=X, W_hat=e, n_estimators=TREES, random_state=12
        )
    assert e.min() < 0.01 and e.max() > 0.99  # the design premise
    return cf, X, e


@pytest.fixture(scope="module")
def clustered():
    """Clusters of very different sizes with ``equalize_cluster_weights``:
    weights and clusters both enter every average."""
    rng = np.random.default_rng(103)
    sizes = rng.integers(2, 22, 45)
    cl = np.repeat(np.arange(45), sizes)
    n = cl.size
    X = rng.normal(size=(n, 3)) + 0.5 * rng.normal(size=45)[cl, None]
    T = rng.binomial(1, 0.5, n).astype(float)
    # The effect grows with cluster size, so equal-cluster weighting moves
    # every weighted statistic away from its unweighted value.
    Y = X[:, 1] + (1 + X[:, 0] + 0.08 * sizes[cl]) * T + rng.normal(size=n)
    cf = sp.causal_forest(
        Y=Y,
        T=T,
        X=X,
        clusters=cl,
        equalize_cluster_weights=True,
        n_estimators=TREES,
        random_state=13,
    )
    w = 1.0 / sizes[cl]
    return cf, X, cl, w


@pytest.fixture(scope="module")
def continuous():
    rng = np.random.default_rng(104)
    n = 500
    X = rng.normal(size=(n, 2))
    T = 0.5 * X[:, 0] + rng.normal(size=n)
    Y = X[:, 1] + (1 + 0.5 * X[:, 0]) * T + rng.normal(size=n)
    cf = sp.causal_forest(
        Y=Y,
        T=T,
        X=X,
        discrete_treatment=False,
        n_estimators=TREES,
        random_state=14,
    )
    return cf, X


# --------------------------------------------------------------------------- #
#  Scores and the average effect
# --------------------------------------------------------------------------- #


def test_get_scores_is_the_unclipped_aipw_score_on_oob_predictions(plain, extreme):
    for cf in (plain[0], extreme[0]):
        y, t, m, e, tau = _arrays(cf)
        hand = _aipw(y, t, m, e, tau)
        np.testing.assert_allclose(sp.get_scores(cf), hand, rtol=RTOL)
        np.testing.assert_allclose(
            aipw_scores(tau=tau, T=t, e_hat=e, m_hat=m, Y=y), hand, rtol=RTOL
        )


def test_predict_without_data_is_out_of_bag_not_in_bag(plain):
    cf, X = plain
    oob = cf.predict()
    np.testing.assert_array_equal(oob, cf.oob_effect())
    # A row predicted by trees that saw it is not the same number: if the
    # two coincided the "out-of-bag" predictions would be in-sample ones.
    inbag = cf.effect(X)
    assert np.mean(np.abs(oob - inbag)) > 1e-3
    # Passing the training rows back to the averages still uses OOB.
    np.testing.assert_allclose(
        cf.average_treatment_effect(X=X)["estimate"],
        cf.average_treatment_effect()["estimate"],
        rtol=RTOL,
    )


def test_ate_is_the_mean_of_the_scores_with_their_standard_error(plain):
    cf, _ = plain
    y, t, m, e, tau = _arrays(cf)
    est, se = _mean_se(_aipw(y, t, m, e, tau))
    out = cf.average_treatment_effect(clip=0.0)
    assert out["estimate"] == pytest.approx(est, rel=RTOL)
    assert out["se"] == pytest.approx(se, rel=RTOL)
    z = stats.norm.ppf(0.975)
    assert out["ci_low"] == pytest.approx(est - z * se, rel=RTOL)
    assert out["ci_high"] == pytest.approx(est + z * se, rel=RTOL)
    assert out["estimand"] == "ATE" and out["cate_source"] == "out_of_bag"
    # No propensity is outside [0.01, 0.99] here, so the default clip is
    # inactive and gives the same number.
    assert e.min() > 0.01 and e.max() < 0.99
    assert cf.average_treatment_effect()["estimate"] == pytest.approx(est, rel=RTOL)


def test_alpha_sets_the_interval_width_not_the_estimate(plain):
    cf, _ = plain
    a05 = cf.average_treatment_effect()
    a10 = cf.average_treatment_effect(alpha=0.10)
    assert a10["estimate"] == a05["estimate"] and a10["se"] == a05["se"]
    z = stats.norm.ppf(0.95)
    assert a10["ci_high"] - a10["ci_low"] == pytest.approx(2 * z * a05["se"], rel=RTOL)
    assert a10["alpha"] == 0.10


@pytest.mark.parametrize("clip", [0.0, 0.05, 0.2])
def test_clip_bounds_the_propensity_inside_the_score(extreme, clip):
    cf, _, _ = extreme
    y, t, m, e, tau = _arrays(cf)
    e_used = np.clip(e, clip, 1 - clip) if clip > 0 else e
    est, se = _mean_se(_aipw(y, t, m, e_used, tau))
    out = cf.average_treatment_effect(clip=clip)
    assert out["estimate"] == pytest.approx(est, rel=RTOL)
    assert out["se"] == pytest.approx(se, rel=RTOL)
    assert out["pscore_min"] == pytest.approx(e_used.min(), rel=RTOL)
    assert out["pscore_max"] == pytest.approx(e_used.max(), rel=RTOL)


def test_subset_equals_restricting_every_sum_by_hand(plain):
    cf, X = plain
    y, t, m, e, tau = _arrays(cf)
    mask = X[:, 0] > 0.2
    est, se = _mean_se(_aipw(y, t, m, e, tau)[mask])
    by_mask = cf.average_treatment_effect(clip=0.0, subset=mask)
    by_pos = cf.average_treatment_effect(clip=0.0, subset=np.flatnonzero(mask))
    by_series = cf.average_treatment_effect(clip=0.0, subset=pd.Series(mask))
    for out in (by_mask, by_pos, by_series):
        assert out["estimate"] == pytest.approx(est, rel=RTOL)
        assert out["se"] == pytest.approx(se, rel=RTOL)
        assert out["n"] == int(mask.sum()) and out["n_fit"] == len(y)
        assert out["subset"] is True
    # ATT on the subset: every sum of the ATT formula runs over the subset.
    att, att_se = _att_atc(y[mask], t[mask], m[mask], e[mask], tau[mask], "treated")
    out = cf.average_treatment_effect("treated", clip=0.0, subset=mask)
    assert out["estimate"] == pytest.approx(att, rel=RTOL)
    assert out["se"] == pytest.approx(att_se, rel=RTOL)


@pytest.mark.parametrize(
    "subset, exc",
    [
        (np.zeros(500, dtype=bool), DataInsufficient),  # selects nothing
        (np.ones(7, dtype=bool), MethodIncompatibility),  # wrong length
        ([0, 0, 1], MethodIncompatibility),  # repeated position
        ([0, 500], MethodIncompatibility),  # out of range
        ([-1, 2], MethodIncompatibility),  # negative position
        ([0.5, 2.0], MethodIncompatibility),  # not integers
        (["a", "b"], MethodIncompatibility),  # not numeric
        ([3], DataInsufficient),  # one row: no standard error
    ],
)
def test_subset_rejects_what_it_cannot_interpret(plain, subset, exc):
    with pytest.raises(exc):
        plain[0].average_treatment_effect(subset=subset)


def test_subset_cannot_be_combined_with_other_rows(plain):
    cf, X = plain
    with pytest.raises(MethodIncompatibility, match="subset"):
        cf.average_treatment_effect(X=X, subset=np.arange(50))


@pytest.mark.parametrize("target", ["treated", "control"])
def test_att_and_atc_follow_the_documented_two_part_formula(plain, target):
    cf, _ = plain
    y, t, m, e, tau = _arrays(cf)
    est, se = _att_atc(y, t, m, e, tau, target)
    out = cf.average_treatment_effect(target_sample=target, clip=0.0)
    assert out["estimate"] == pytest.approx(est, rel=RTOL)
    assert out["se"] == pytest.approx(se, rel=RTOL)
    assert out["estimand"] == ("ATT" if target == "treated" else "ATC")
    n_arm = int(np.sum(t == (1 if target == "treated" else 0)))
    assert out["effective_sample_size"] == n_arm
    # The pure operator gives the same pair.
    op_est, op_se, corr = grf_att_atc(
        tau=tau, T=t, e_hat=e, m_hat=m, Y=y, target=target
    )
    assert op_est == pytest.approx(est, rel=RTOL)
    assert op_se == pytest.approx(se, rel=RTOL)
    assert corr.shape == y.shape


def test_att_weights_controls_by_the_odds_of_treatment(extreme):
    """With propensities near 0 and 1 the ATT and ATC corrections differ
    by orders of magnitude; a formula that swapped the two arms' weights
    would not survive this design."""
    cf, _, _ = extreme
    y, t, m, e, tau = _arrays(cf)
    for target, clip in (("treated", 0.05), ("control", 0.05), ("treated", 0.0)):
        e_used = np.clip(e, clip, 1 - clip) if clip > 0 else e
        est, se = _att_atc(y, t, m, e_used, tau, target)
        out = cf.average_treatment_effect(target, clip=clip)
        assert out["estimate"] == pytest.approx(est, rel=RTOL)
        assert out["se"] == pytest.approx(se, rel=RTOL)


def test_target_aliases_and_positional_target(plain):
    cf, _ = plain
    ref = cf.average_treatment_effect(target_sample="treated")
    for spelling in ("ATT", " treated ", "att"):
        assert cf.average_treatment_effect(target_sample=spelling) == ref
    assert cf.average_treatment_effect("treated") == ref
    assert sp.average_treatment_effect(cf, target_sample="atc")["estimand"] == "ATC"
    assert sp.average_treatment_effect(cf, target_sample="ato")["estimand"] == "ATO"
    with pytest.raises(MethodIncompatibility, match="string first argument"):
        cf.average_treatment_effect("treated", target_sample="control")
    with pytest.raises(MethodIncompatibility, match="target_sample"):
        cf.average_treatment_effect(target_sample="everyone")
    with pytest.raises(MethodIncompatibility, match="must be a string"):
        sp.average_treatment_effect(cf, target_sample=1)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, float("nan"), "x", None])
def test_average_effect_rejects_bad_alpha(plain, bad):
    with pytest.raises(MethodIncompatibility, match="alpha"):
        plain[0].average_treatment_effect(alpha=bad)


@pytest.mark.parametrize("bad", [0.5, -0.01, float("inf"), "x", None])
def test_average_effect_rejects_bad_clip(plain, bad):
    with pytest.raises(MethodIncompatibility, match="clip"):
        plain[0].average_treatment_effect(clip=bad)


def test_overlap_target_is_the_residual_on_residual_regression_with_hc3(plain):
    cf, _ = plain
    y, t, m, e, _ = _arrays(cf)
    D = np.column_stack([np.ones(len(y)), t - e])
    beta, resid = _wls(D, y - m)
    V = _sandwich(D, resid, "HC3")
    out = cf.average_treatment_effect(target_sample="overlap")
    assert out["estimate"] == pytest.approx(beta[1], rel=RTOL)
    assert out["se"] == pytest.approx(np.sqrt(V[1, 1]), rel=RTOL)
    assert out["estimand"] == "ATO" and out["method"] == "partially_linear"
    op = grf_overlap_ate(Y=y, W=t, Y_hat=m, W_hat=e)
    assert op[0] == pytest.approx(beta[1], rel=RTOL)
    assert op[1] == pytest.approx(np.sqrt(V[1, 1]), rel=RTOL)


def test_overlap_operator_refuses_degenerate_inputs():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(MethodIncompatibility, match="same length"):
        grf_overlap_ate(Y=y, W=y[:3], Y_hat=y, W_hat=y)
    with pytest.raises(DataInsufficient, match="three observations"):
        grf_overlap_ate(Y=y[:2], W=y[:2], Y_hat=y[:2], W_hat=y[:2])
    # W - W_hat constant: the slope is not identified.
    with pytest.raises(sp.exceptions.NumericalInstability, match="constant"):
        grf_overlap_ate(Y=y, W=np.ones(4), Y_hat=np.zeros(4), W_hat=np.full(4, 0.5))


def test_att_operator_refuses_degenerate_inputs():
    one = np.ones(4)
    kw = dict(tau=one, e_hat=one * 0.5, m_hat=one, Y=one)
    with pytest.raises(DataInsufficient, match="both arms"):
        grf_att_atc(T=one, target="treated", **kw)
    with pytest.raises(MethodIncompatibility, match="treated.*control"):
        grf_att_atc(T=np.array([0, 1, 0, 1.0]), target="all", **kw)
    with pytest.raises(DataInsufficient, match="two observations"):
        grf_att_atc(
            tau=one[:1], T=one[:1], e_hat=one[:1], m_hat=one[:1], Y=one[:1], target="x"
        )
    with pytest.raises(MethodIncompatibility, match="same length"):
        aipw_scores(tau=one, T=one[:3], e_hat=one, m_hat=one, Y=one)
    with pytest.raises(MethodIncompatibility, match="unsupported target"):
        aipw_scores(tau=one, T=one, e_hat=one * 0.5, m_hat=one, Y=one, target="treated")


# --------------------------------------------------------------------------- #
#  Clusters and observation weights
# --------------------------------------------------------------------------- #


def test_clustered_ate_uses_weights_and_the_cluster_sum_variance(clustered):
    cf, _, cl, w = clustered
    y, t, m, e, tau = _arrays(cf)
    scores = _aipw(y, t, m, e, tau)
    est, se = _mean_se(scores, w, cl)
    out = cf.average_treatment_effect(clip=0.0)
    assert out["estimate"] == pytest.approx(est, rel=RTOL)
    assert out["se"] == pytest.approx(se, rel=RTOL)
    assert out["n_clusters"] == 45
    assert out["effective_sample_size"] == pytest.approx(
        w.sum() ** 2 / np.sum(w**2), rel=RTOL
    )
    # The design makes the weighted and the unweighted mean differ by far
    # more than rounding, so agreement above is not vacuous.
    assert abs(est - scores.mean()) > 1e-3


@pytest.mark.parametrize("target", ["treated", "control"])
def test_clustered_att_uses_weights_in_every_term(clustered, target):
    cf, _, cl, w = clustered
    y, t, m, e, tau = _arrays(cf)
    est, se = _att_atc(y, t, m, e, tau, target, w=w, clusters=cl)
    out = cf.average_treatment_effect(target, clip=0.0)
    assert out["estimate"] == pytest.approx(est, rel=RTOL)
    assert out["se"] == pytest.approx(se, rel=RTOL)


def test_clustered_overlap_is_weighted_with_a_cluster_hc1_covariance(clustered):
    cf, _, cl, w = clustered
    y, t, m, e, _ = _arrays(cf)
    D = np.column_stack([np.ones(len(y)), t - e])
    beta, resid = _wls(D, y - m, w)
    V = _sandwich(D, resid, "HC1", w=w, clusters=cl)
    out = cf.average_treatment_effect("overlap")
    assert out["estimate"] == pytest.approx(beta[1], rel=RTOL)
    assert out["se"] == pytest.approx(np.sqrt(V[1, 1]), rel=RTOL)


def test_clustered_subset_recounts_the_clusters_it_keeps(clustered):
    cf, _, cl, w = clustered
    y, t, m, e, tau = _arrays(cf)
    mask = cl < 20
    est, se = _mean_se(_aipw(y, t, m, e, tau)[mask], w[mask], cl[mask])
    out = cf.average_treatment_effect(clip=0.0, subset=mask)
    assert out["estimate"] == pytest.approx(est, rel=RTOL)
    assert out["se"] == pytest.approx(se, rel=RTOL)
    assert out["n_clusters"] == 20


@pytest.mark.parametrize("target", ["all", "treated"])
def test_subset_inside_one_cluster_is_refused_not_given_an_infinite_se(
    clustered, target
):
    cf, _, cl, _ = clustered
    one_cluster = cl == int(np.argmax(np.bincount(cl)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(DataInsufficient):
            cf.average_treatment_effect(target, subset=one_cluster)


def test_subset_inside_one_cluster_is_refused_for_the_overlap_target(clustered):
    cf, _, cl, _ = clustered
    one_cluster = cl == int(np.argmax(np.bincount(cl)))
    with pytest.raises(DataInsufficient, match="two clusters"):
        cf.average_treatment_effect("overlap", subset=one_cluster)


# --------------------------------------------------------------------------- #
#  Best linear projection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("vce", ["HC0", "HC1", "HC2", "HC3"])
def test_blp_is_ols_of_the_scores_with_the_requested_covariance(plain, vce):
    cf, X = plain
    y, t, m, e, tau = _arrays(cf)
    scores = _aipw(y, t, m, e, tau)
    A = X[:, :2]
    D = np.column_stack([np.ones(len(y)), A])
    beta, resid = _wls(D, scores)
    V = _sandwich(D, resid, vce)
    tab = sp.best_linear_projection(cf, A=A, vce=vce, alpha=0.10)
    np.testing.assert_allclose(tab["coef"], beta, rtol=RTOL)
    np.testing.assert_allclose(tab["se"], np.sqrt(np.diag(V)), rtol=RTOL)
    np.testing.assert_allclose(tab.attrs["vcov"].to_numpy(), V, rtol=1e-8, atol=1e-14)
    z = stats.norm.ppf(0.95)
    np.testing.assert_allclose(tab["ci_upper"], beta + z * tab["se"], rtol=RTOL)
    np.testing.assert_allclose(
        tab["p"], 2 * stats.norm.sf(np.abs(beta / np.sqrt(np.diag(V)))), rtol=1e-8
    )
    assert list(tab.index) == ["Intercept", "A1", "A2"]


def test_blp_without_covariates_is_the_average_effect(plain):
    cf, _ = plain
    tab = sp.best_linear_projection(cf)
    ate = cf.average_treatment_effect(clip=0.0)
    assert list(tab.index) == ["Intercept"]
    assert tab.loc["Intercept", "coef"] == pytest.approx(ate["estimate"], rel=RTOL)


def test_blp_method_defaults_to_hc3_on_the_effect_modifiers(plain):
    cf, X = plain
    y, t, m, e, tau = _arrays(cf)
    D = np.column_stack([np.ones(len(y)), X])
    beta, resid = _wls(D, _aipw(y, t, m, e, tau))
    tab = cf.best_linear_projection()
    np.testing.assert_allclose(tab["coef"], beta, rtol=RTOL)
    np.testing.assert_allclose(
        tab["se"], np.sqrt(np.diag(_sandwich(D, resid, "HC3"))), rtol=RTOL
    )
    assert list(tab.index) == ["Intercept", "X0", "X1", "X2"]
    assert cf.diagnostics["blp_n_clipped_propensities"] == 0
    # The training matrix passed explicitly is the same projection.
    pd.testing.assert_frame_equal(cf.best_linear_projection(X), tab)


def test_blp_method_clips_and_counts_extreme_propensities(extreme):
    cf, X, _ = extreme
    y, t, m, e, tau = _arrays(cf)
    clip = 0.05
    e_used = np.clip(e, clip, 1 - clip)
    D = np.column_stack([np.ones(len(y)), X])
    beta, _ = _wls(D, _aipw(y, t, m, e_used, tau))
    tab = cf.best_linear_projection(clip=clip)
    np.testing.assert_allclose(tab["coef"], beta, rtol=RTOL)
    assert cf.diagnostics["blp_n_clipped_propensities"] == int(
        np.sum((e < clip) | (e > 1 - clip))
    )


def test_blp_method_refuses_rows_other_than_the_training_rows(plain):
    cf, X = plain
    with pytest.raises(MethodIncompatibility, match="aligned with the training rows"):
        cf.best_linear_projection(X[:40])
    for bad in (0.0, 1.5, "x"):
        with pytest.raises(MethodIncompatibility, match="alpha"):
            cf.best_linear_projection(alpha=bad)
    for bad in (0.5, -1.0, "x"):
        with pytest.raises(MethodIncompatibility, match="clip"):
            cf.best_linear_projection(clip=bad)


@pytest.mark.parametrize("vce", ["HC0", "HC1"])
def test_clustered_blp_is_weighted_and_cluster_robust(clustered, vce):
    cf, X, cl, w = clustered
    y, t, m, e, tau = _arrays(cf)
    D = np.column_stack([np.ones(len(y)), X[:, 0]])
    beta, resid = _wls(D, _aipw(y, t, m, e, tau), w)
    V = _sandwich(D, resid, vce, w=w, clusters=cl)
    tab = sp.best_linear_projection(cf, A=X[:, 0], vce=vce)
    np.testing.assert_allclose(tab["coef"], beta, rtol=RTOL)
    np.testing.assert_allclose(tab["se"], np.sqrt(np.diag(V)), rtol=RTOL)


def test_blp_design_validation(plain):
    cf, X = plain
    with pytest.raises(MethodIncompatibility, match="one row per training"):
        sp.best_linear_projection(cf, A=X[:10])
    bad = X[:, 0].copy()
    bad[3] = np.nan
    with pytest.raises(MethodIncompatibility, match="non-finite"):
        sp.best_linear_projection(cf, A=bad)
    with pytest.raises(MethodIncompatibility, match="vcov_type"):
        sp.best_linear_projection(cf, vce="HC4")
    named = sp.best_linear_projection(cf, A=pd.Series(X[:, 0], name="age"))
    assert list(named.index) == ["Intercept", "age"]
    frame = sp.best_linear_projection(cf, A=pd.DataFrame(X[:, :2], columns=["a", "b"]))
    assert list(frame.index) == ["Intercept", "a", "b"]
    with pytest.raises(MethodIncompatibility, match="unsupported object"):
        sp.best_linear_projection(object())


# --------------------------------------------------------------------------- #
#  Calibration test
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("vce", ["HC0", "HC1", "HC2", "HC3"])
def test_calibration_is_the_documented_no_intercept_regression(plain, vce):
    cf, _ = plain
    y, t, m, e, tau = _arrays(cf)
    n = len(y)
    D = np.column_stack([(t - e) * tau.mean(), (t - e) * (tau - tau.mean())])
    beta, resid = _wls(D, y - m)
    se = np.sqrt(np.diag(_sandwich(D, resid, vce)))
    tab = sp.calibration_test(cf, vce=vce, alpha=0.10)
    np.testing.assert_allclose(tab["coef"], beta, rtol=RTOL)
    np.testing.assert_allclose(tab["se"], se, rtol=RTOL)
    # grf's reporting rule: t against 0, one-sided p from Student t(n - 2).
    np.testing.assert_allclose(tab["t"], beta / se, rtol=RTOL)
    np.testing.assert_allclose(tab["p"], stats.t.sf(beta / se, n - 2), rtol=1e-8)
    crit = stats.t.ppf(0.95, n - 2)
    np.testing.assert_allclose(tab["ci_high"], beta + crit * se, rtol=RTOL)
    np.testing.assert_allclose(tab["ci_low"], beta - crit * se, rtol=RTOL)
    np.testing.assert_array_equal(tab["t_vs_zero"], tab["t"])
    np.testing.assert_array_equal(tab["p_one_sided"], tab["p"])


@pytest.mark.parametrize("vce", ["HC0", "HC1"])
def test_clustered_calibration_uses_weights_and_clusters(clustered, vce):
    cf, _, cl, w = clustered
    y, t, m, e, tau = _arrays(cf)
    tau_bar = np.sum(w * tau) / np.sum(w)
    D = np.column_stack([(t - e) * tau_bar, (t - e) * (tau - tau_bar)])
    # The forest stores weights that sum to one; the regression and the
    # sandwich are invariant to that scale only if both use the same one.
    wn = w / w.sum()
    beta, resid = _wls(D, y - m, wn)
    se = np.sqrt(np.diag(_sandwich(D, resid, vce, w=wn, clusters=cl)))
    tab = sp.calibration_test(cf, vce=vce)
    np.testing.assert_allclose(tab["coef"], beta, rtol=RTOL)
    np.testing.assert_allclose(tab["se"], se, rtol=RTOL)
    assert "cluster-robust" in tab.attrs["method"]


def test_calibration_requires_the_training_sample(plain):
    cf, X = plain
    y, t, *_ = _arrays(cf)
    ref = sp.calibration_test(cf)
    pd.testing.assert_frame_equal(sp.calibration_test(cf, X=X, Y=y, T=t), ref)
    with pytest.raises(MethodIncompatibility, match="training sample"):
        sp.calibration_test(cf, Y=y + 1.0)
    with pytest.raises(MethodIncompatibility, match="training sample"):
        sp.calibration_test(cf, T=1.0 - t)
    with pytest.raises(MethodIncompatibility, match="same row count"):
        sp.calibration_test(cf, Y=y[:10])
    with pytest.raises(MethodIncompatibility, match="NaN"):
        sp.calibration_test(cf, Y=np.where(np.arange(len(y)) == 0, np.nan, y))
    with pytest.raises(MethodIncompatibility, match="numeric"):
        sp.calibration_test(cf, Y=np.array(["a"] * len(y)))
    with pytest.raises(MethodIncompatibility, match="fixed"):
        sp.calibration_test(cf, method="imputation")
    with pytest.raises(MethodIncompatibility, match="method must be"):
        sp.calibration_test(cf, method="other")
    with pytest.raises(MethodIncompatibility, match="vce"):
        sp.calibration_test(cf, vce="HC7")
    for bad in (0.0, "x"):
        with pytest.raises(MethodIncompatibility, match="alpha"):
            sp.calibration_test(cf, alpha=bad)


def test_calibration_operator_refuses_unidentified_designs():
    rng = np.random.default_rng(0)
    n = 30
    y, w = rng.normal(size=n), rng.binomial(1, 0.5, n).astype(float)
    base = dict(Y=y, W=w, Y_hat=np.zeros(n), W_hat=np.full(n, 0.5))
    # A constant prediction leaves the differential column identically 0.
    with pytest.raises(DataInsufficient, match="rank deficient"):
        grf_calibration(tau_hat=np.ones(n), **base)
    with pytest.raises(MethodIncompatibility, match="same length"):
        grf_calibration(tau_hat=np.ones(n - 1), **base)
    with pytest.raises(MethodIncompatibility, match="weights"):
        grf_calibration(tau_hat=rng.normal(size=n), weights=np.ones(3), **base)
    with pytest.raises(DataInsufficient, match="at least 3 rows"):
        grf_calibration(
            tau_hat=rng.normal(size=n),
            weights=np.r_[1.0, 1.0, np.zeros(n - 2)],
            **base,
        )
    with pytest.raises(MethodIncompatibility, match="vce"):
        grf_calibration(tau_hat=rng.normal(size=n), vcov_type="HAC", **base)


def test_calibrate_cate_applies_the_calibration_line(plain):
    cf, X = plain
    tab = sp.calibration_test(cf)
    b1 = tab.loc["mean_forest_prediction", "coef"]
    b2 = tab.loc["differential_forest_prediction", "coef"]
    oob = cf.predict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = sp.calibrate_cate(cf)
        new = X[:25] + 0.1
        cal_new = sp.calibrate_cate(cf, newdata=new)
    np.testing.assert_allclose(
        cal["cate"], b1 * oob.mean() + b2 * (oob - oob.mean()), rtol=RTOL
    )
    np.testing.assert_array_equal(cal["raw_cate"], oob)
    # New rows are predicted by the whole forest, then put on the same line.
    raw = cf.effect(new)
    np.testing.assert_allclose(cal_new["raw_cate"], raw, rtol=RTOL)
    np.testing.assert_allclose(
        cal_new["cate"], b1 * oob.mean() + b2 * (raw - oob.mean()), rtol=RTOL
    )
    assert cal["beta_differential_se"] == pytest.approx(
        tab.loc["differential_forest_prediction", "se"], rel=RTOL
    )
    assert cal["method"] == "blp_oob"


def test_calibrate_cate_centres_on_the_weighted_mean_prediction(clustered):
    cf, _, _, w = clustered
    oob = cf.predict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = sp.calibrate_cate(cf)
    tau_bar = np.sum(w * oob) / np.sum(w)
    assert cal["mean_oob_prediction"] == pytest.approx(tau_bar, rel=RTOL)
    np.testing.assert_allclose(
        cal["cate"],
        cal["beta_mean"] * tau_bar + cal["beta_differential"] * (oob - tau_bar),
        rtol=RTOL,
    )


# --------------------------------------------------------------------------- #
#  Pointwise intervals and variable importance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [0.05, 0.20])
def test_effect_interval_is_the_normal_interval_on_the_little_bag_variance(
    plain, alpha
):
    cf, X = plain
    new = X[:30] * 0.5
    tau = cf.effect(new)
    var = cf.effect_variance(new)
    assert np.all(var > 0)
    lo, hi = cf.effect_interval(new, alpha=alpha)
    z = stats.norm.ppf(1 - alpha / 2)
    np.testing.assert_allclose(lo, tau - z * np.sqrt(var), rtol=RTOL)
    np.testing.assert_allclose(hi, tau + z * np.sqrt(var), rtol=RTOL)


def test_effect_variance_without_rows_is_the_out_of_bag_variance(plain):
    cf, _ = plain
    var = cf.effect_variance()
    assert var.shape == cf.predict().shape and np.all(var > 0)


@pytest.mark.parametrize("bad", [0.0, 1.0, float("nan"), [0.05, 0.1]])
def test_effect_interval_rejects_bad_alpha(plain, bad):
    with pytest.raises(MethodIncompatibility, match="alpha"):
        plain[0].effect_interval(plain[1][:3], alpha=bad)


@pytest.mark.parametrize("decay, depth", [(2.0, 4), (1.0, 2), (0.0, 3)])
def test_variable_importance_is_the_depth_weighted_split_share(plain, decay, depth):
    cf, _ = plain
    counts = cf.split_frequencies(max_depth=depth).to_numpy(dtype=float)
    assert counts.shape == (depth, 3)
    assert np.all(counts.sum(axis=1) > 0)  # every counted depth has splits
    shares = counts / counts.sum(axis=1, keepdims=True)
    weights = np.arange(1, depth + 1, dtype=float) ** (-decay)
    hand = weights @ shares / weights.sum()
    imp = sp.variable_importance(cf, decay_exponent=decay, max_depth=depth)
    np.testing.assert_allclose(imp.to_numpy(), hand, rtol=RTOL)
    assert imp.sum() == pytest.approx(1.0, rel=RTOL)
    assert list(imp.index) == ["X0", "X1", "X2"]
    method = cf.variable_importance(decay_exponent=decay, max_depth=depth)
    np.testing.assert_allclose(method.to_numpy(), hand, rtol=RTOL)


def test_the_first_depth_counts_one_split_per_tree_at_most(plain):
    cf, _ = plain
    counts = cf.split_frequencies(max_depth=1).to_numpy()
    assert 0 < counts.sum() <= cf.diagnostics["n_estimators"]


def test_permutation_importance_is_normalised_and_named(plain):
    cf, _ = plain
    imp = cf.variable_importance(method="permutation")
    assert imp.sum() == pytest.approx(1.0, rel=RTOL)
    assert set(imp.index) == {"X0", "X1", "X2"}
    assert list(imp) == sorted(imp, reverse=True)
    with pytest.raises(MethodIncompatibility, match="unknown method"):
        cf.variable_importance(method="gini")
    with pytest.raises(MethodIncompatibility, match="max_depth"):
        sp.variable_importance(cf, max_depth=0)


# --------------------------------------------------------------------------- #
#  ScalarEffect wrappers
# --------------------------------------------------------------------------- #


def test_ate_and_att_return_the_doubly_robust_number_with_its_interval(plain):
    cf, X = plain
    _, t, _, _, tau = _arrays(cf)
    for effect, target, plug in (
        (cf.ate(), "all", tau.mean()),
        (cf.att(), "treated", tau[t == 1].mean()),
    ):
        ref = cf.average_treatment_effect(target)
        assert float(effect) == pytest.approx(ref["estimate"], rel=RTOL)
        assert effect.se == pytest.approx(ref["se"], rel=RTOL)
        assert effect.ci == pytest.approx((ref["ci_low"], ref["ci_high"]), rel=RTOL)
        assert effect.pvalue == pytest.approx(
            2 * stats.norm.sf(abs(ref["estimate"] / ref["se"])), rel=1e-8
        )
        # The plug-in mean of the OOB predictions is kept beside it.
        assert effect.detail["plug_in_estimate"] == pytest.approx(plug, rel=RTOL)
        assert effect.detail["plug_in_minus_aipw"] == pytest.approx(
            plug - ref["estimate"], rel=1e-8, abs=1e-12
        )
    # The training matrix passed back in is recognised as the training rows.
    assert float(cf.ate(X)) == pytest.approx(float(cf.ate()), rel=RTOL)
    assert float(cf.att(X, t)) == pytest.approx(float(cf.att()), rel=RTOL)


def test_att_validates_its_treatment_vector(plain):
    cf, _ = plain
    n = len(cf._Y_original)
    with pytest.raises(MethodIncompatibility, match="same row count"):
        cf.att(T=np.ones(5))
    with pytest.raises(MethodIncompatibility, match="finite"):
        cf.att(T=np.full(n, np.nan))
    with pytest.raises(MethodIncompatibility, match="numeric"):
        cf.att(T=np.array(["a"] * n))
    with pytest.raises(MethodIncompatibility, match="binary"):
        cf.att(T=np.linspace(0, 2, n))
    with pytest.raises(DataInsufficient, match="no treated"):
        cf.att(T=np.zeros(n))


def test_other_rows_get_the_plug_in_value_with_the_reason_attached(plain):
    cf, X = plain
    new = X[:60] + 0.3
    effect = cf.ate(new)
    assert float(effect) == pytest.approx(cf.effect(new).mean(), rel=RTOL)
    assert effect.se is None
    assert "MethodIncompatibility" in effect.inference_error


# --------------------------------------------------------------------------- #
#  Continuous treatment
# --------------------------------------------------------------------------- #


def test_continuous_scores_use_the_stored_debiasing_weights(continuous):
    cf, _ = continuous
    y, t, m, e, tau = _arrays(cf)
    scores = sp.get_scores(cf)
    g = np.asarray(cf._continuous_debias_weights)
    np.testing.assert_allclose(scores, tau + g * (y - m - tau * (t - e)), rtol=RTOL)
    # The weights are (W - W_hat) / V_hat(X) with a positive variance, so
    # they carry the sign of the treatment residual.
    np.testing.assert_array_equal(np.sign(g), np.sign(t - e))
    est, se = _mean_se(scores)
    out = cf.average_treatment_effect()
    assert out["estimate"] == pytest.approx(est, rel=RTOL)
    assert out["se"] == pytest.approx(se, rel=RTOL)
    assert out["method"] == "aipw_continuous"
    # clip is a propensity bound; it must not touch E[T | X] here.
    assert cf.average_treatment_effect(clip=0.3)["estimate"] == pytest.approx(
        est, rel=RTOL
    )


def test_continuous_overlap_and_subset(continuous):
    cf, X = continuous
    y, t, m, e, _ = _arrays(cf)
    D = np.column_stack([np.ones(len(y)), t - e])
    beta, resid = _wls(D, y - m)
    out = cf.average_treatment_effect("overlap")
    assert out["estimate"] == pytest.approx(beta[1], rel=RTOL)
    assert out["se"] == pytest.approx(
        np.sqrt(_sandwich(D, resid, "HC3")[1, 1]), rel=RTOL
    )
    mask = X[:, 1] > 0
    est, se = _mean_se(sp.get_scores(cf)[mask])
    sub = cf.average_treatment_effect(subset=mask)
    assert sub["estimate"] == pytest.approx(est, rel=RTOL)
    assert sub["se"] == pytest.approx(se, rel=RTOL)


@pytest.mark.parametrize("target", ["treated", "control"])
def test_continuous_treatment_has_no_treated_group(continuous, target):
    cf, _ = continuous
    with pytest.raises(MethodIncompatibility, match="binary treatment"):
        cf.average_treatment_effect(target)
    with pytest.raises(MethodIncompatibility, match="binary treatment"):
        cf.att()
    with pytest.raises(MethodIncompatibility, match="binary treatment"):
        sp.rate(cf)


def test_overlap_diagnostic_is_not_applied_to_a_continuous_treatment(continuous):
    info = continuous[0].diagnostics["nuisance_overlap"]
    assert info["applicable"] is False and "not binary" in info["reason"]


# --------------------------------------------------------------------------- #
#  RATE
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("target", ["AUTOC", "QINI"])
def test_rate_follows_its_definition_on_scores_and_oob_priorities(plain, target):
    cf, _ = plain
    scores = sp.get_scores(cf)
    prio = cf.predict()
    hand = _rate(scores, prio, target)
    out = sp.rate(cf, target=target.lower())
    assert out["estimate"] == pytest.approx(hand, rel=1e-9)
    assert out["target"] == target and out["priority_source"] == "out_of_bag"
    assert out["n_clusters"] is None
    # The estimate is linear in the scores given the ranking.
    a = rate_rank_weights(prio, target)
    assert float(a @ scores) == pytest.approx(hand, rel=1e-9)
    assert abs(a.sum()) < 1e-12  # a contrast: a constant score gives zero
    z = stats.norm.ppf(0.975)
    assert out["ci_high"] == pytest.approx(out["estimate"] + z * out["se"], rel=RTOL)


def test_rate_with_supplied_priorities_and_reversed_ranking(plain):
    cf, X = plain
    scores = sp.get_scores(cf)
    prio = X[:, 0]
    out = sp.rate(cf, priorities=prio)
    assert out["estimate"] == pytest.approx(_rate(scores, prio, "AUTOC"), rel=1e-9)
    assert out["priority_source"] == "supplied"
    # A constant priority is one tie group: every unit gets the mean score
    # and the curve is flat at zero.
    flat = sp.rate(cf, priorities=np.zeros(len(prio)), target="QINI")
    assert abs(flat["estimate"]) < 1e-12
    with pytest.raises(MethodIncompatibility, match="same row count"):
        sp.rate(cf, priorities=prio[:10])


def test_toc_curve_grid_and_endpoints(plain):
    cf, _ = plain
    scores = sp.get_scores(cf)
    prio = cf.predict()
    out = sp.rate(cf, q_grid=20)
    curve = out["toc_curve"]
    assert curve.shape == (20, 2)
    np.testing.assert_allclose(curve[:, 0], np.arange(1, 21) / 20, rtol=1e-12)
    # At q = 1 everyone is treated: the TOC is zero.
    assert abs(curve[-1, 1]) < 1e-12
    # n = 500 and q = k/20 land on whole units, where the curve is the mean
    # score of the top 25k units minus the overall mean.
    order = np.argsort(-prio, kind="stable")
    for j in (0, 4, 9):
        k = 25 * (j + 1)
        assert curve[j, 1] == pytest.approx(
            scores[order][:k].mean() - scores.mean(), rel=1e-9, abs=1e-12
        )
    # The grid does not change the estimate.
    assert sp.rate(cf, q_grid=7)["estimate"] == pytest.approx(out["estimate"], rel=RTOL)


def test_half_sample_se_is_the_spread_of_half_sample_rates(plain):
    cf, _ = plain
    scores = sp.get_scores(cf)
    prio = cf.predict()
    n = len(scores)
    rng = np.random.default_rng(7)
    draws = []
    for _ in range(30):
        idx = rng.choice(n, size=n // 2, replace=False)
        draws.append(_rate(scores[idx], prio[idx], "AUTOC"))
    out = sp.rate(cf, se_method="half_sample", n_bootstrap=30, seed=7)
    assert out["se"] == pytest.approx(np.std(draws, ddof=1), rel=1e-8)
    assert "R=30" in out["method"]
    # The analytic SE estimates the same quantity; 30 draws pin a standard
    # deviation to about 1 / sqrt(2 * 29) = 13%, so a factor of two is a
    # bound that only a wrong scale (sqrt(n), sqrt(R)) would break.
    infl = sp.rate(cf)["se"]
    assert 0.5 < out["se"] / infl < 2.0


def test_rate_operator_and_argument_validation(plain):
    cf, _ = plain
    s = np.arange(6, dtype=float)
    with pytest.raises(MethodIncompatibility, match="same length"):
        rate_from_scores(s, s[:5])
    with pytest.raises(DataInsufficient, match="finite"):
        rate_from_scores(np.r_[s[:5], np.nan], s)
    with pytest.raises(MethodIncompatibility, match="AUTOC"):
        rate_from_scores(s, s, target="AUC")
    for q in ([], [0.5, 0.5, 1.0], [0.0, 1.0], [0.2, 0.9]):
        with pytest.raises(MethodIncompatibility, match="strictly increasing"):
            rate_from_scores(s, s, q=np.array(q))
    for bad in ("AUC", 3):
        with pytest.raises(MethodIncompatibility, match="target"):
            sp.rate(cf, target=bad)
    for bad in (0, 2.5, True):
        with pytest.raises(MethodIncompatibility, match="q_grid"):
            sp.rate(cf, q_grid=bad)
    with pytest.raises(MethodIncompatibility, match="se_method"):
        sp.rate(cf, se_method="jackknife")
    with pytest.raises(MethodIncompatibility, match="fixed effects"):
        sp.rate(cf, se_method="imputation")
    with pytest.raises(MethodIncompatibility, match="n_bootstrap"):
        sp.rate(cf, se_method="half_sample", n_bootstrap=1)


def test_clustered_rate_reports_clusters_and_draws_whole_clusters(clustered):
    cf, _, cl, _ = clustered
    out = sp.rate(cf)
    assert out["n_clusters"] == 45 and out["method"].endswith("clustered")
    half = sp.rate(cf, se_method="half_sample", n_bootstrap=12, seed=3)
    assert half["method"].endswith("clustered") and np.isfinite(half["se"])
    assert half["estimate"] == out["estimate"]


def _weighted_rate(scores, prio, target, w):
    """Weighted RATE written out from its definition, independently of the
    package: every count becomes a weight total. Rows are taken in
    decreasing priority (input order inside a tie), a tie group shares its
    weighted mean score, TOC_k = S_k / C_k - S_n / C_n with C and S the
    running weight and weighted-score totals, AUTOC = sum_k w_k TOC_k / C_n
    and QINI = sum_k w_k (C_k / C_n) TOC_k / C_n."""
    s = np.asarray(scores, dtype=float)
    pr = np.asarray(prio, dtype=float)
    w = np.asarray(w, dtype=float)
    s = np.array([np.average(s[pr == v], weights=w[pr == v]) for v in pr])
    order = np.argsort(-pr, kind="stable")
    s, w = s[order], w[order]
    C, S = np.cumsum(w), np.cumsum(w * s)
    toc = S / C - S[-1] / C[-1]
    share = w / C[-1]
    return float(
        np.sum(share * toc) if target == "AUTOC" else np.sum(share * C / C[-1] * toc)
    )


@pytest.fixture(scope="module")
def grf_weighted_rate():
    """grf::rank_average_treatment_effect.fit(..., sample.weights=) run as a
    black box on fixed inputs (_generate_grf_rate_weighted_R.R)."""
    path = (
        Path(__file__).parent
        / "reference_parity"
        / "_fixtures"
        / "grf_rate_weighted_R.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


RATE_CASES = ["continuous", "tied", "tied_permuted", "tied_clustered", "tied_unit"]


@pytest.mark.parametrize("target", ["AUTOC", "QINI"])
@pytest.mark.parametrize("case", RATE_CASES)
def test_weighted_rate_operator_matches_grf_on_fixed_inputs(
    grf_weighted_rate, case, target
):
    ref = grf_weighted_rate["cases"][case]
    s, pr, w = (np.asarray(ref[k], dtype=float) for k in _RATE_INPUTS)
    q = np.asarray(grf_weighted_rate["q"], dtype=float)
    out = _weighted_rate_from_scores(s, pr, w, target, q)
    # Deterministic on both sides: the floating-point floor, not a
    # statistical tolerance (observed 6e-15).
    assert out["estimate"] == pytest.approx(ref[target]["estimate"], rel=1e-12)
    np.testing.assert_allclose(out["toc"], ref[target]["toc"], rtol=1e-11, atol=1e-12)
    # The hand-written definition used on the forest below is the same map.
    assert _weighted_rate(s, pr, target, w) == pytest.approx(
        ref[target]["estimate"], rel=1e-12
    )


_RATE_INPUTS = ("scores", "priorities", "weights")


@pytest.mark.parametrize("target", ["AUTOC", "QINI"])
@pytest.mark.parametrize("case", ["continuous", "tied_clustered"])
def test_weighted_half_sample_se_is_close_to_the_grf_bootstrap(
    grf_weighted_rate, case, target
):
    ref = grf_weighted_rate["cases"][case]
    s, pr, w = (np.asarray(ref[k], dtype=float) for k in _RATE_INPUTS)
    cl = ref.get("clusters")
    if cl is not None:
        cl = np.unique(cl, return_inverse=True)[1]
    q = np.asarray(grf_weighted_rate["q"], dtype=float)
    se = _rate_half_sample_se(s, pr, target, q, 1000, 0, clusters=cl, weights=w)
    # Two bootstraps with different random numbers (R = 1000 here, 2000 in
    # R): each has a Monte Carlo sd of about 1 / sqrt(2 R), so 10% is some
    # three sds of their difference. A screen, not parity.
    assert se == pytest.approx(ref[target]["std_err"], rel=0.10)


@pytest.mark.parametrize("target", ["AUTOC", "QINI"])
def test_weighted_rate_reduces_to_the_unweighted_one_for_equal_weights(
    grf_weighted_rate, target
):
    ref = grf_weighted_rate["cases"]["continuous"]
    s, pr = (np.asarray(ref[k], dtype=float) for k in ("scores", "priorities"))
    q = np.asarray(grf_weighted_rate["q"], dtype=float)
    w = np.full(len(s), 0.37)
    plain_core = rate_from_scores(s, pr, target, q)
    core = _weighted_rate_from_scores(s, pr, w, target, q)
    assert core["estimate"] == pytest.approx(plain_core["estimate"], rel=1e-12)
    np.testing.assert_allclose(core["toc"], plain_core["toc"], rtol=1e-11, atol=1e-12)
    assert _weighted_rate_influence_se(s, pr, w, target) == pytest.approx(
        _rate_influence_se(s, pr, target), rel=1e-12
    )
    cl = np.arange(len(s)) // 4
    assert _weighted_rate_influence_se(s, pr, w, target, cl) == pytest.approx(
        _rate_influence_se(s, pr, target, cl), rel=1e-12
    )


def test_weighted_rate_refuses_weights_it_cannot_rank_with():
    s, pr = np.arange(6.0), np.arange(6.0)[::-1].copy()
    q = np.array([0.5, 1.0])
    for bad in (np.array([1, 1, 0, 1, 1, 1.0]), np.ones(5), np.full(6, np.nan)):
        with pytest.raises(MethodIncompatibility, match="weights"):
            _weighted_rate_from_scores(s, pr, bad, "AUTOC", q)


@pytest.mark.parametrize("target", ["AUTOC", "QINI"])
def test_rate_uses_the_forest_observation_weights(clustered, target):
    cf, _, cl, w = clustered
    scores = sp.get_scores(cf)
    prio = cf.predict()
    weighted = _weighted_rate(scores, prio, target, w)
    # Premise: on this design the weighted and unweighted RATE differ by
    # several percent, far beyond the tolerance below.
    assert abs(weighted - _rate(scores, prio, target)) > 1e-3
    out = sp.rate(cf, target=target)
    assert out["estimate"] == pytest.approx(weighted, rel=1e-9)
    # The curve ends at zero and the half-sample draws are weighted too.
    assert out["toc_curve"][-1, 1] == pytest.approx(0.0, abs=1e-12)
    assert out["se"] == pytest.approx(
        _weighted_rate_influence_se(scores, prio, w, target, cl), rel=1e-9
    )
    half = sp.rate(cf, target=target, se_method="half_sample", n_bootstrap=400, seed=1)
    assert half["estimate"] == out["estimate"]
    # Bootstrap against the analytic standard error: a screen at 25%.
    assert half["se"] == pytest.approx(out["se"], rel=0.25)


# --------------------------------------------------------------------------- #
#  Group effects of a pooled forest
# --------------------------------------------------------------------------- #


def test_pooled_group_effects_are_group_means_of_the_scores(plain):
    cf, X = plain
    scores = sp.get_scores(cf)
    tau = cf.predict()
    n = len(scores)
    labels = np.where(X[:, 0] > 0, "high", "low")
    tab = sp.forest_group_effects(cf, by=labels)
    assert list(tab.index) == ["high", "low"]
    for g in ("high", "low"):
        rows = labels == g
        est = scores[rows].mean()
        # Heteroskedasticity-robust variance of a group mean with the
        # n / (n - 1) factor of the full sample (docstring of the method).
        se = np.sqrt(np.sum((scores[rows] - est) ** 2) / rows.sum() ** 2 * n / (n - 1))
        assert tab.loc[g, "estimate"] == pytest.approx(est, rel=RTOL)
        assert tab.loc[g, "se"] == pytest.approx(se, rel=RTOL)
        assert tab.loc[g, "n_rows"] == rows.sum()
        assert tab.loc[g, "forest_mean"] == pytest.approx(tau[rows].mean(), rel=RTOL)
    assert tab.attrs["estimand"] == "ATE (all rows)"
    diff = tab.attrs["tests"]["last_minus_first"]
    assert diff["estimate"] == pytest.approx(
        tab.loc["low", "estimate"] - tab.loc["high", "estimate"], rel=RTOL
    )
    # Disjoint groups of independent rows: the difference's variance is the
    # sum of the two variances.
    assert diff["se"] == pytest.approx(np.hypot(*tab["se"].to_numpy()), rel=1e-8)


def test_pooled_group_effects_default_group_is_the_average_effect(plain, clustered):
    for cf in (plain[0], clustered[0]):
        ate = cf.average_treatment_effect(clip=0.0)
        tab = sp.forest_group_effects(cf)
        assert list(tab.index) == ["all"]
        assert tab.loc["all", "estimate"] == pytest.approx(ate["estimate"], rel=RTOL)
        assert tab.loc["all", "se"] == pytest.approx(ate["se"], rel=RTOL)


def test_quantile_groups_partition_the_rows_by_oob_prediction(plain):
    cf, _ = plain
    scores = sp.get_scores(cf)
    tau = cf.predict()
    tab = sp.forest_group_effects(cf, by="cate_quantile", n_groups=5)
    assert list(tab.index) == [f"Q{g}" for g in range(1, 6)]
    assert tab["n_rows"].sum() == len(tau) and set(tab["n_rows"]) == {100}
    order = np.argsort(tau, kind="stable")
    for g in range(5):
        rows = order[100 * g : 100 * (g + 1)]
        assert tab["estimate"].iloc[g] == pytest.approx(scores[rows].mean(), rel=RTOL)
    # Groups are sorted by prediction, so the mean prediction increases.
    assert np.all(np.diff(tab["forest_mean"]) > 0)
    with pytest.raises(MethodIncompatibility, match="at least 2"):
        sp.forest_group_effects(cf, by="cate_quantile", n_groups=1)


def test_group_effects_validation(plain):
    cf, X = plain
    with pytest.raises(MethodIncompatibility, match="one label per training row"):
        sp.forest_group_effects(cf, by=np.ones(4))
    labels = np.where(X[:, 0] > 0, "a", None)
    with pytest.raises(MethodIncompatibility, match="missing labels"):
        sp.forest_group_effects(cf, by=labels)
    with pytest.raises(MethodIncompatibility, match="scale"):
        sp.forest_group_effects(cf, scale="log")
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.forest_group_effects(cf, alpha=1.0)
    with pytest.raises(DataInsufficient, match="no group has"):
        sp.forest_group_effects(cf, min_rows=10_000)
    pct = sp.forest_group_effects(cf, scale="percent")
    assert pct["estimate_pct"].iloc[0] == pytest.approx(
        100 * np.expm1(pct["estimate"].iloc[0]), rel=RTOL
    )
    # The method on the forest is the same function.
    pd.testing.assert_frame_equal(cf.group_effects(), sp.forest_group_effects(cf))


# --------------------------------------------------------------------------- #
#  Support of new rows
# --------------------------------------------------------------------------- #


def test_forest_support_distances_match_a_brute_force_search(plain):
    cf, X = plain
    k = 5
    new = np.vstack([X[:6] * 0.3, [[9.0, 0.0, 0.0]]])
    out = sp.forest_support(cf, new, k=k, quantile=0.9, alpha=0.10)
    mean, sd = X.mean(axis=0), X.std(axis=0)
    ref = (X - mean) / sd
    q = (new - mean) / sd
    d_new = np.sort(np.linalg.norm(q[:, None, :] - ref[None, :, :], axis=2), axis=1)
    np.testing.assert_allclose(out["knn_distance"], d_new[:, k - 1], rtol=1e-9)
    # Reference benchmark: each training row's k-th nearest *other* row.
    d_ref = np.sort(np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=2), axis=1)
    cutoff = np.quantile(d_ref[:, k], 0.9)  # column 0 is the row itself
    assert out.attrs["summary"]["knn_cutoff"] == pytest.approx(cutoff, rel=1e-9)
    np.testing.assert_allclose(out["knn_ratio"], d_new[:, k - 1] / cutoff, rtol=1e-9)
    # The last row is nine standard deviations out on the first covariate.
    assert out["n_outside_range"].iloc[-1] == 1 and not out["supported"].iloc[-1]
    np.testing.assert_array_equal(
        out["supported"], (out["n_outside_range"] == 0) & (out["knn_ratio"] <= 1.0)
    )
    z = stats.norm.ppf(0.95)
    np.testing.assert_allclose(out["cate"], cf.effect(new), rtol=RTOL)
    np.testing.assert_allclose(
        out["ci_high"], out["cate"] + z * out["se"], rtol=RTOL, atol=1e-12
    )
    np.testing.assert_allclose(
        out["se"], np.sqrt(cf.effect_variance(new)), rtol=RTOL, atol=1e-12
    )


def test_forest_support_validation(plain):
    cf, X = plain
    for bad in (0, 2.0, True):
        with pytest.raises(MethodIncompatibility, match="k must be"):
            sp.forest_support(cf, X[:3], k=bad)
    with pytest.raises(MethodIncompatibility, match="quantile and alpha"):
        sp.forest_support(cf, X[:3], quantile=1.0)
    with pytest.raises(MethodIncompatibility, match="quantile and alpha"):
        sp.forest_support(cf, X[:3], alpha=0.0)
    with pytest.raises(DataInsufficient, match="fewer reference rows"):
        sp.forest_support(cf, X[:3], k=len(X))
    frame = pd.DataFrame(X[:4], columns=["X0", "X1", "X2"], index=list("abcd"))
    assert list(sp.forest_support(cf, frame).index) == list("abcd")


def test_forest_diagnostics_reports_overlap_from_the_stored_propensity(plain):
    cf, X = plain
    _, t, _, e, tau = _arrays(cf)
    diag = sp.forest_diagnostics(cf, propensity_bounds=(0.3, 0.7))
    assert diag["n_treated"] == int(t.sum()) and diag["n_control"] == int((1 - t).sum())
    assert diag["cate_mean"] == pytest.approx(tau.mean(), rel=RTOL)
    assert diag["cate_sd"] == pytest.approx(tau.std(ddof=1), rel=RTOL)
    assert diag["overlap_share"] == pytest.approx(
        np.mean((e >= 0.3) & (e <= 0.7)), rel=RTOL
    )
    assert diag["n_low_pscore"] == int(np.sum(e < 0.3))
    assert diag["n_high_pscore"] == int(np.sum(e > 0.7))
    assert any("outside requested overlap bounds" in w for w in diag["warnings"])
    assert cf.forest_diagnostics()["warnings"] == []
    # Rows that are not the training sample have no propensity.
    other = sp.forest_diagnostics(cf, X=X[:50] + 1.0, T=t[:50])
    assert np.isnan(other["overlap_share"]) and np.isnan(other["pscore_min"])
    assert other["n"] == 50
    with pytest.raises(MethodIncompatibility, match="T is required"):
        sp.forest_diagnostics(cf, X=X[:50] + 1.0)
    for bad in ((0.9, 0.1), (0.1,), ("a", "b"), (-0.1, 0.5), 0.5):
        with pytest.raises(MethodIncompatibility, match="propensity_bounds"):
            sp.forest_diagnostics(cf, propensity_bounds=bad)
