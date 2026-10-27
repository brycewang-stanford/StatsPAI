"""Option-scan round 5 (treatment effects / ML): defects left open in round 4.

Every test asserts a true statement: a known truth (coverage of a simulated
interval), an identity with a computation written out here, or an
invariance. Covered: ``sp.aipw(cluster=)`` cross-fitting folds and its
arm-mean fallback, ``sp.pate`` optimiser convergence, ``sp.metalearner``
weighted continuous R-learner and ``fold_indices`` with missing design
columns, ``sp.tmle(fold_indices=, cluster=)``, the test-point weight of
``sp.weighted_conformal_prediction``, ``sp.subgroup_analysis`` subgroups
without an estimate and the ``sp.policy_tree`` gain note.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)

# ---------------------------------------------------------------------------
# sp.aipw(cluster=): folds over whole clusters
# ---------------------------------------------------------------------------


def _clustered(seed: int = 0, G: int = 40, m: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), m)
    x = rng.normal(size=G * m)
    d = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x)))
    y = 1.0 * d + x + rng.normal(size=G)[g] + rng.normal(size=G * m)
    return pd.DataFrame({"y": y, "d": d, "x": x, "cid": g.astype(float), "g": g})


class _FoldRecorder(BaseEstimator, RegressorMixin):
    """Outcome learner that logs (training cluster ids, predicted ids)."""

    log: list = []

    def fit(self, X, y, sample_weight=None):
        self.train_ids_ = set(np.asarray(X)[:, 0].tolist())
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        _FoldRecorder.log.append((self.train_ids_, set(np.asarray(X)[:, 0].tolist())))
        return np.full(len(X), self.mean_)


def test_aipw_cluster_folds_never_train_on_the_predicted_cluster():
    df = _clustered()
    _FoldRecorder.log = []
    res = sp.aipw(
        df,
        y="y",
        treat="d",
        covariates=["cid", "x"],
        cluster="g",
        outcome_learner=_FoldRecorder(),
    )
    assert len(_FoldRecorder.log) == 2 * 5  # two arms x five folds
    for train_ids, test_ids in _FoldRecorder.log:
        assert not (train_ids & test_ids)
    assert res.model_info["fold_unit"] == "cluster"


def test_aipw_cluster_folds_do_not_depend_on_row_order():
    df = _clustered(seed=3)
    shuffled = df.sample(frac=1.0, random_state=11).reset_index(drop=True)
    a = sp.aipw(df, y="y", treat="d", covariates=["x"], cluster="g")
    b = sp.aipw(shuffled, y="y", treat="d", covariates=["x"], cluster="g")
    # Same folds (drawn per sorted cluster label); only the summation order
    # of the OLS fits differs.
    assert b.estimate == pytest.approx(a.estimate, rel=1e-10, abs=1e-12)
    assert b.se == pytest.approx(a.se, rel=1e-10)


def test_aipw_singleton_clusters_give_the_row_level_split():
    # One cluster per row, labels sorted in row order: one draw per cluster
    # is the row-level draw, so the folds and hence the estimate coincide.
    df = _clustered(seed=5)
    df["row"] = np.arange(len(df))
    a = sp.aipw(df, y="y", treat="d", covariates=["x"])
    b = sp.aipw(df, y="y", treat="d", covariates=["x"], cluster="row")
    assert b.estimate == a.estimate


def _cluster_trial(r: int, G: int = 50, m: int = 12) -> pd.DataFrame:
    """Cluster-randomised trial with a cluster-level covariate and shock."""
    rng = np.random.default_rng(r)
    g = np.repeat(np.arange(G), m)
    z = rng.normal(size=G)[g]
    alpha = 2.0 * rng.normal(size=G)[g]
    x = z + 0.3 * rng.normal(size=G * m)
    d = rng.binomial(1, 0.5, size=G)[g]
    y = 1.0 * d + x + alpha + 0.5 * rng.normal(size=G * m)
    return pd.DataFrame({"y": y, "d": d, "z": z, "x": x, "g": g})


def test_aipw_cluster_randomised_trial_interval_covers():
    # Known truth: ATE = 1. With row-level folds a row's outcome model was
    # learned from its own cluster's rows, absorbing the cluster shock; the
    # 95% interval covered 85.8% in 1000 replications (mean SE 0.49 against
    # a sampling SD of 0.63; 87.3% in the first 300). Cluster folds: 94.6%
    # (mean SE 0.74, SD 0.76; 94.0% in the first 300).
    cover = []
    for r in range(300):
        res = sp.aipw(
            _cluster_trial(r),
            y="y",
            treat="d",
            covariates=["z", "x"],
            cluster="g",
            outcome_learner=KNeighborsRegressor(n_neighbors=5),
        )
        cover.append(res.ci[0] <= 1.0 <= res.ci[1])
    # MC SE at 0.95 with 300 draws is 0.0126.
    assert 0.92 <= np.mean(cover) <= 0.98


def test_aipw_without_cluster_is_unchanged():
    # Values from the release before the cluster-fold change.
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "g": rng.integers(0, 40, n),
        }
    )
    df["d"] = rng.binomial(1, 1 / (1 + np.exp(-df.x1)))
    df["y"] = 1 + df.d * 2 + df.x1 + df.x2 + rng.normal(size=n)
    res = sp.aipw(df, y="y", treat="d", covariates=["x1", "x2"])
    assert res.estimate == pytest.approx(1.8824957559789137, rel=1e-10)
    assert res.se == pytest.approx(0.11829284496558841, rel=1e-10)
    assert res.model_info["fold_unit"] == "row"
    nox = sp.aipw(
        df, y="y", treat="d", covariates=["x1", "x2"], cross_fit=False, cluster="g"
    )
    assert nox.estimate == pytest.approx(1.8989041880942872, rel=1e-10)
    assert nox.se == pytest.approx(0.1093495156360968, rel=1e-10)


# ---------------------------------------------------------------------------
# sp.aipw: arm with too few training rows
# ---------------------------------------------------------------------------


def _two_treated(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    x = rng.normal(size=n)
    d = np.zeros(n)
    d[[4, 17]] = 1.0
    return pd.DataFrame(
        {"y": x + d + rng.normal(size=n), "d": d, "x": x, "g": np.arange(n) // 4}
    )


def test_aipw_arm_mean_fallback_warns_and_is_recorded():
    df = _two_treated()
    with pytest.warns(RuntimeWarning, match="training mean"):
        res = sp.aipw(df, y="y", treat="d", covariates=["x"], cross_fit=False, trim=0.0)
    assert res.model_info["n_outcome_arm_mean_fallback"] == 1
    assert res.model_info["outcome_arm_mean_fallback"] == [
        {"fold": 0, "arm": 1, "n_train": 2}
    ]
    # The treated outcome model is the treated mean, so the plug-in part of
    # the ATE is mean(y | d=1) - mean(mu0(x)).
    y1_mean = df.loc[df.d == 1, "y"].mean()
    assert res.model_info["potential_outcome_means"][1] == pytest.approx(
        y1_mean + np.mean(df.d * (df.y - y1_mean) / _logit_ps(df)),
        rel=1e-6,
    )


def _logit_ps(df: pd.DataFrame) -> np.ndarray:
    fit = sm.Logit(df.d.to_numpy(), sm.add_constant(df[["x"]].to_numpy())).fit(
        disp=0, maxiter=300, warn_convergence=False
    )
    return np.asarray(fit.predict())


def test_aipw_fold_without_an_arm_raises():
    # Both treated rows sit in cluster 1 (rows 4..7) and 17 // 4 = 4; put
    # them in one cluster so they share a fold and that fold's training
    # set has no treated row.
    df = _two_treated()
    df.loc[17, "g"] = 1
    with pytest.raises(DataInsufficient) as exc:
        sp.aipw(df, y="y", treat="d", covariates=["x"], cluster="g")
    assert exc.value.recovery_hint


# ---------------------------------------------------------------------------
# sp.pate: optimiser convergence
# ---------------------------------------------------------------------------


def _pate_data(target_shift: float):
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 2))
    d = rng.binomial(1, 0.5, n)
    y = X[:, 0] + d * (1 + X[:, 1]) + rng.normal(size=n)
    exp = pd.DataFrame({"y": y, "d": d, "x1": X[:, 0], "x2": X[:, 1]})
    tg = pd.DataFrame(
        {
            "x1": rng.normal(target_shift, 1.0 if target_shift < 5 else 0.1, 600),
            "x2": rng.normal(0.5, 1, 600),
        }
    )
    return exp, tg


def test_pate_calibration_reports_balance_it_cannot_reach():
    # Target mean of x1 = 8 lies outside the experiment's range (max ~3):
    # entropy balancing cannot match it.
    exp, tg = _pate_data(8.0)
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        res = sp.pate(
            exp,
            tg,
            y="y",
            treatment="d",
            covariates=["x1", "x2"],
            method="calibration",
            n_boot=10,
            seed=1,
        )
    assert res.model_info["converged"] is False
    rec = res.model_info["convergence"][0]
    assert rec["step"] == "entropy_balancing"
    assert rec["max_std_imbalance"] > 1.0
    assert res.model_info["n_boot_not_converged"] == 10


@pytest.mark.parametrize(
    "method, est",
    [("ipw", 1.3286329935755310), ("calibration", 1.3375150318623752)],
)
def test_pate_converged_fit_is_unchanged(method, est):
    exp, tg = _pate_data(0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        res = sp.pate(
            exp,
            tg,
            y="y",
            treatment="d",
            covariates=["x1", "x2"],
            method=method,
            n_boot=40,
            seed=1,
        )
    assert res.model_info["converged"] is True
    assert res.model_info["n_boot_not_converged"] == 0
    # The balanced weights match the target means (what convergence means).
    if method == "calibration":
        assert res.model_info["convergence"][0]["max_std_imbalance"] <= 1e-4
    # Values from the release before the convergence check.
    assert res.estimate == pytest.approx(est, rel=1e-10)


# ---------------------------------------------------------------------------
# sp.metalearner: weighted continuous R-learner; fold_indices + missing design
# ---------------------------------------------------------------------------


def _continuous(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = 0.8 * x1 + x1**2 + rng.normal(size=n)
    y = (1 + 0.5 * x1) * d + np.sin(2 * x1) + x2 + rng.normal(size=n)
    w = np.exp(0.8 * x1)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "w": w / w.mean()})


def test_metalearner_continuous_r_uses_weighted_nuisance_fits():
    df = _continuous()
    n = len(df)
    folds = np.arange(n) % 5
    res = sp.metalearner(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        learner="r",
        weights="w",
        outcome_model=LinearRegression(),
        propensity_model=LinearRegression(),
        fold_indices=folds,
    )
    # Hand-rolled weighted cross-fit of the partially linear coefficient.
    X = df[["x1", "x2"]].to_numpy()
    y, d, w = df.y.to_numpy(), df.d.to_numpy(), df.w.to_numpy()
    yr, dr = np.empty(n), np.empty(n)
    for k in range(5):
        tr, te = folds != k, folds == k
        yr[te] = y[te] - LinearRegression().fit(
            X[tr], y[tr], sample_weight=w[tr]
        ).predict(X[te])
        dr[te] = d[te] - LinearRegression().fit(
            X[tr], d[tr], sample_weight=w[tr]
        ).predict(X[te])
    theta = np.sum(w * dr * yr) / np.sum(w * dr**2)
    assert res.estimate == pytest.approx(theta, rel=1e-10)
    # Unweighted fits gave 1.4743 here; the weighted cross-fit is 1.5455.
    assert theta == pytest.approx(1.5455105070488706, rel=1e-10)


def test_metalearner_fold_indices_drop_rows_with_missing_weights():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["d"] = rng.binomial(1, 0.5, n)
    df["y"] = df.x1 + df.d * (1 + df.x2) + rng.normal(size=n)
    df["w"] = rng.uniform(0.5, 2, n)
    df.loc[[3, 10, 50], "w"] = np.nan
    folds = np.arange(n) % 4
    kw = dict(
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        learner="t",
        weights="w",
        outcome_model=LinearRegression(),
        n_folds=4,
    )
    got = sp.metalearner(df, fold_indices=folds, **kw)
    keep = df.w.notna().to_numpy()
    ref = sp.metalearner(
        df.loc[keep].reset_index(drop=True), fold_indices=folds[keep], **kw
    )
    assert got.n_obs == n - 3
    assert got.estimate == ref.estimate
    assert got.se == ref.se


# ---------------------------------------------------------------------------
# sp.tmle(fold_indices=, cluster=)
# ---------------------------------------------------------------------------


def test_tmle_refuses_folds_that_split_a_cluster():
    df = _clustered(G=20, m=6)
    folds = np.arange(len(df)) % 3  # rows of one cluster land in 3 folds
    with pytest.raises(MethodIncompatibility, match="split") as exc:
        sp.tmle(df, y="y", treat="d", covariates=["x"], cluster="g", fold_indices=folds)
    assert exc.value.recovery_hint
    assert exc.value.diagnostics["n_clusters_split"] == 20


def test_tmle_accepts_cluster_level_folds():
    df = _clustered(G=20, m=6)
    folds = df.g.to_numpy() % 3
    res = sp.tmle(
        df,
        y="y",
        treat="d",
        covariates=["x"],
        cluster="g",
        fold_indices=folds,
        outcome_library=[LinearRegression()],
        propensity_library=[LogisticRegression()],
    )
    assert res.model_info["cross_fitted"] is True
    assert np.isfinite(res.estimate)


# ---------------------------------------------------------------------------
# sp.weighted_conformal_prediction: the test point's own weight
# ---------------------------------------------------------------------------


def _shift_draw(r: int, s: float = 1.5, n_tr: int = 200, n_cal: int = 30):
    rng = np.random.default_rng(r)
    xtr, xc, xt = (
        rng.normal(size=n_tr),
        rng.normal(size=n_cal),
        rng.normal(s, 1, size=1),
    )

    def f(x):
        return x + (0.2 + np.abs(x)) * rng.normal(size=len(x))

    return xtr, f(xtr), xc, f(xc), xt, f(xt)


def test_weighted_conformal_covers_under_covariate_shift():
    # Known truth: TBCR weighted split conformal covers a test point from
    # the shifted law with probability >= 1 - alpha when every point,
    # the test point included, carries the density ratio w(x). Training
    # X ~ N(0, 1), test X ~ N(1.5, 1): w(x) = exp(1.5 x - 1.125). With the
    # test point weighted 1 the intervals covered 81%.
    s, alpha, R = 1.5, 0.1, 1500

    def w(x):
        return np.exp(s * x - s * s / 2)

    hit = 0
    for r in range(R):
        xtr, ytr, xc, yc, xt, yt = _shift_draw(r, s)
        lo, hi, _ = sp.weighted_conformal_prediction(
            xtr[:, None],
            ytr,
            xc[:, None],
            yc,
            xt[:, None],
            weights_calib=w(xc),
            weights_test=w(xt),
            model=LinearRegression(),
            alpha=alpha,
        )
        hit += lo[0] <= yt[0] <= hi[0]
    # MC SE at 0.9 with 1500 draws is 0.0077; 0.877 is three below.
    assert hit / R >= 0.877


def test_weighted_conformal_without_test_weights_warns():
    xtr, ytr, xc, yc, xt, _ = _shift_draw(0)
    with pytest.warns(AssumptionWarning, match="weights_test"):
        sp.weighted_conformal_prediction(
            xtr[:, None],
            ytr,
            xc[:, None],
            yc,
            xt[:, None],
            weights_calib=np.exp(xc),
            model=LinearRegression(),
        )


def test_weighted_conformal_per_point_quantile_identity():
    from statspai.conformal_causal.counterfactual import _weighted_quantile

    rng = np.random.default_rng(4)
    xtr, ytr, xc, yc, _, _ = _shift_draw(1, n_cal=60)
    xt = rng.normal(1.0, 1.0, size=7)
    wc, wt = np.exp(xc), np.exp(xt)
    m = LinearRegression().fit(xtr[:, None], ytr)
    resid = np.abs(yc - m.predict(xc[:, None]))
    lo, hi, pt = sp.weighted_conformal_prediction(
        xtr[:, None],
        ytr,
        xc[:, None],
        yc,
        xt[:, None],
        weights_calib=wc,
        weights_test=wt,
        model=LinearRegression(),
        alpha=0.2,
    )
    scores = np.append(resid, np.inf)
    for j in range(7):
        q = _weighted_quantile(scores, np.append(wc, wt[j]), 0.8)
        assert hi[j] - pt[j] == pytest.approx(q, rel=1e-12)
        assert pt[j] - lo[j] == pytest.approx(q, rel=1e-12)
    # Unweighted call unchanged: equals explicit unit weights everywhere.
    a = sp.weighted_conformal_prediction(
        xtr[:, None], ytr, xc[:, None], yc, xt[:, None], model=LinearRegression()
    )
    b = sp.weighted_conformal_prediction(
        xtr[:, None],
        ytr,
        xc[:, None],
        yc,
        xt[:, None],
        model=LinearRegression(),
        weights_calib=np.ones(60),
        weights_test=np.ones(7),
    )
    np.testing.assert_array_equal(a[0], b[0])


def test_conformal_counterfactual_passes_test_point_weights(monkeypatch):
    import statspai.conformal_causal.counterfactual as cc

    calls = []
    real = cc.weighted_conformal_prediction

    def spy(**kw):
        calls.append(kw)
        return real(**kw)

    monkeypatch.setattr(cc, "weighted_conformal_prediction", spy)
    rng = np.random.default_rng(0)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    t = rng.binomial(1, 1 / (1 + np.exp(-x1)))
    df = pd.DataFrame({"y": x1 + t + rng.normal(size=n), "t": t, "x1": x1, "x2": x2})
    X_test = rng.normal(size=(5, 2))
    cc.conformal_counterfactual(
        df,
        y="y",
        treat="t",
        covariates=["x1", "x2"],
        X_test=X_test,
        model=LinearRegression(),
        random_state=0,
    )
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(
        df[["x1", "x2"]].to_numpy(), t
    )
    g = np.clip(lr.predict_proba(X_test)[:, 1], 1e-3, 1 - 1e-3)
    np.testing.assert_allclose(calls[0]["weights_test"], 1 / g, rtol=1e-12)
    np.testing.assert_allclose(calls[1]["weights_test"], 1 / (1 - g), rtol=1e-12)


# ---------------------------------------------------------------------------
# sp.subgroup_analysis: subgroups without an estimate
# ---------------------------------------------------------------------------


def _subgroup_data(c_in_g2: float) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    d = pd.DataFrame(
        {"g": rng.integers(0, 3, n), "x": rng.normal(size=n), "c": rng.normal(size=n)}
    )
    d.loc[d.g == 2, "c"] = c_in_g2
    d["y"] = 1 + 0.5 * d.x + 0.2 * d.c + rng.normal(size=n)
    return d


@pytest.mark.parametrize("const", [0.0, 0.3, 5.0])
def test_subgroup_with_constant_control_is_kept_with_a_reason(const):
    # Before: the row vanished (c = 0, 5) or came back with an SE from a
    # numerically singular inverse (c = 0.3: 0.0946, against 0.0881 from
    # the regression without the constant control).
    df = _subgroup_data(const)
    with pytest.warns(UserWarning, match="constant within the subgroup"):
        res = sp.subgroup_analysis(df, "y ~ x + c", x="x", by={"G": "g"})
    out = res.results_df.set_index("group_val")
    assert list(out.index) == ["0", "1", "2"]
    assert np.isnan(out.loc["2", "estimate"]) and np.isnan(out.loc["2", "se"])
    assert "'c'" in out.loc["2", "note"]
    assert out.loc["2", "nobs"] == int((df.g == 2).sum())
    # The other subgroups are the per-group regress, vce(robust).
    for v in (0, 1):
        sub = df[df.g == v]
        ref = sm.OLS(sub.y, sm.add_constant(sub[["x", "c"]])).fit(cov_type="HC1")
        assert out.loc[str(v), "estimate"] == pytest.approx(ref.params["x"], rel=1e-10)
        assert out.loc[str(v), "se"] == pytest.approx(ref.bse["x"], rel=1e-10)
        assert out.loc[str(v), "note"] == ""
    assert "no estimate" in res.summary()
    assert "--" in res.to_latex()


def test_subgroup_with_too_few_rows_is_kept_with_a_reason():
    df = _subgroup_data(0.0)
    df["c"] = np.random.default_rng(9).normal(size=len(df))
    df.loc[df.index[:3], "g"] = 7  # a three-row subgroup
    with pytest.warns(UserWarning, match="complete rows"):
        res = sp.subgroup_analysis(df, "y ~ x + c", x="x", by={"G": "g"})
    row = res.results_df.set_index("group_val").loc["7"]
    assert np.isnan(row["estimate"]) and row["nobs"] == 3


# ---------------------------------------------------------------------------
# sp.policy_tree: note on a gain within its own SE of zero
# ---------------------------------------------------------------------------


def _policy_df(n: int = 400):
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    t = rng.integers(0, 2, n)
    return pd.DataFrame({"y": rng.normal(size=n), "t": t, "x1": x1, "x2": x2}), rng


def test_policy_tree_notes_a_gain_near_zero():
    df, rng = _policy_df()
    null_scores = rng.normal(size=len(df))  # no heterogeneity, no effect
    res = sp.policy_tree(
        df,
        y="y",
        treat="t",
        covariates=["x1", "x2"],
        scores=null_scores,
        max_depth=1,
        random_state=0,
    )
    assert abs(res["value_gain"]) <= res["value_gain_se"]
    assert "gain_interval_note" in res["diagnostics"]
    assert "undercovers" in res.diagnostics["gain_interval_note"]


def test_policy_tree_no_note_with_real_heterogeneity():
    df, rng = _policy_df()
    het = 2.0 * (df.x1.to_numpy() > 0) - 1.0 + rng.normal(size=len(df))
    res = sp.policy_tree(
        df,
        y="y",
        treat="t",
        covariates=["x1", "x2"],
        scores=het,
        max_depth=1,
        random_state=0,
    )
    assert res["value_gain"] > 3 * res["value_gain_se"]
    assert res["diagnostics"] == {}
