"""External parity: classifier nuisances in PLR, and IRM sensitivity bounds.

Two things the DML notebooks of Chernozhukov, Hansen, Kallus, Spindler and
Syrgkanis (*Applied Causal Inference Powered by ML and AI*) do that
``sp.dml`` got wrong before this file existed. Both are pinned against
``doubleml-for-py`` on a shared fold partition and on simulated data.

1. **A classifier for a binary treatment in PLR.** The conditional mean of
   a 0/1 variable is ``predict_proba``; ``predict`` returns the hard label.
   ``sp.dml(model='plr', ml_m=<classifier>)`` used the hard label, so the
   treatment residual was ``D - 1{p > 0.5}``. On the book's 401(k) example
   that moved the estimate from 8,754 to 6,044 without any warning.
2. **Sensitivity bounds for IRM.** ``sp.dml_sensitivity`` applied the PLR
   scaling ``sd(psi) / sd(D - m)`` to an IRM fit. The IRM bound needs
   ``sigma^2 = E[(Y - g(D, X))^2]`` and the Riesz representer
   ``alpha = D/m - (1-D)/(1-m)`` with ``nu^2 = E[2 m(W, alpha) - alpha^2]``.

Tolerances: 1e-10 relative for everything that is the same arithmetic on
the same cross-fitted predictions; 1e-6 for robustness values, which
``doubleml`` finds with a bounded scalar minimiser and StatsPAI with a
root finder. The standard errors of the two bounds match crosswise, for
the reason documented in ``test_dml_sensitivity_parity.py``. With clusters
and PLR the crosswise match is to 5e-4 only: ``doubleml`` scales the score
by the plain mean of the score derivative there and by the fold-weighted
one for the point estimate, so its bound standard error at zero
confounding is not its own standard error; StatsPAI's is.

References
----------
[@chernozhukov2018double], [@chernozhukov2022long], [@bach2022doubleml]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import statspai as sp

doubleml = pytest.importorskip("doubleml")

N = 1200
N_FOLDS = 4
COVARIATES = ["x1", "x2", "x3"]
CF_Y, CF_D = 0.04, 0.03


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(N, 3))
    p = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1])))
    d = (rng.uniform(size=N) < p).astype(float)
    z = (rng.uniform(size=N) < 1.0 / (1.0 + np.exp(-X[:, 2]))).astype(float)
    y = (1.0 + 0.5 * X[:, 0]) * d + X[:, 0] + 0.5 * X[:, 2] ** 2 + rng.normal(size=N)
    df = pd.DataFrame(X, columns=COVARIATES)
    df["d"], df["y"], df["z"] = d, y, z
    df["g"] = np.arange(N) // 4  # 300 clusters of four
    df["cont"] = X[:, 0] + rng.normal(size=N)
    return df


@pytest.fixture(scope="module")
def folds() -> np.ndarray:
    return np.arange(N) % N_FOLDS


def _splits(folds):
    return [
        [
            (np.flatnonzero(folds != f), np.flatnonzero(folds == f))
            for f in range(N_FOLDS)
        ]
    ]


def _rf():
    return RandomForestRegressor(n_estimators=40, min_samples_leaf=10, random_state=1)


def _rfc():
    return RandomForestClassifier(n_estimators=40, min_samples_leaf=10, random_state=1)


# ---------------------------------------------------------------- classifiers
def test_plr_with_classifier_for_binary_treatment_matches_doubleml(data, folds):
    fit = sp.dml(
        data,
        y="y",
        treat="d",
        covariates=COVARIATES,
        model="plr",
        ml_g=_rf(),
        ml_m=_rfc(),
        n_folds=N_FOLDS,
        fold_indices=folds,
    )
    dd = doubleml.DoubleMLData(data, y_col="y", d_cols="d", x_cols=COVARIATES)
    ref = doubleml.DoubleMLPLR(
        dd, ml_l=_rf(), ml_m=_rfc(), n_folds=N_FOLDS, draw_sample_splitting=False
    )
    ref.set_sample_splitting(_splits(folds))
    ref.fit()
    assert fit.estimate == pytest.approx(float(ref.coef[0]), rel=1e-10)
    assert fit.se == pytest.approx(float(ref.se[0]), rel=1e-10)


def test_plr_classifier_is_not_the_hard_label_estimate(data, folds):
    """The pre-fix estimate, rebuilt by hand, is a different number."""
    X, D, Y = data[COVARIATES].values, data["d"].values, data["y"].values
    ry, rd = np.zeros(N), np.zeros(N)
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        ry[te] = Y[te] - _rf().fit(X[tr], Y[tr]).predict(X[te])
        rd[te] = D[te] - _rfc().fit(X[tr], D[tr]).predict(X[te])
    hard_label = float(np.sum(ry * rd) / np.sum(rd * rd))
    fit = sp.dml(
        data,
        y="y",
        treat="d",
        covariates=COVARIATES,
        model="plr",
        ml_g=_rf(),
        ml_m=_rfc(),
        n_folds=N_FOLDS,
        fold_indices=folds,
    )
    assert abs(fit.estimate - hard_label) > 0.5 * fit.se


def test_pliv_with_classifiers_uses_probabilities(data, folds):
    """PLIV with a binary treatment and instrument, against the moment.

    doubleml rejects classifiers in PLIV, so the reference is the
    estimator written out: residualise with ``predict_proba`` and solve
    ``E[(ry - theta * rd) * rz] = 0``.
    """
    X, Y = data[COVARIATES].values, data["y"].values
    D, Z = data["d"].values, data["z"].values
    ry, rd, rz = np.zeros(N), np.zeros(N), np.zeros(N)
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        ry[te] = Y[te] - _rf().fit(X[tr], Y[tr]).predict(X[te])
        rd[te] = D[te] - _rfc().fit(X[tr], D[tr]).predict_proba(X[te])[:, 1]
        rz[te] = Z[te] - _rfc().fit(X[tr], Z[tr]).predict_proba(X[te])[:, 1]
    theta = np.mean(ry * rz) / np.mean(rd * rz)
    eps = ry - theta * rd
    se = np.sqrt(np.mean(eps**2 * rz**2) / np.mean(rd * rz) ** 2 / N)
    fit = sp.dml(
        data,
        y="y",
        treat="d",
        instrument="z",
        covariates=COVARIATES,
        model="pliv",
        ml_g=_rf(),
        ml_m=_rfc(),
        ml_r=_rfc(),
        n_folds=N_FOLDS,
        fold_indices=folds,
    )
    assert fit.estimate == pytest.approx(theta, rel=1e-10)
    assert fit.se == pytest.approx(se, rel=1e-10)


def test_classifier_for_a_continuous_target_is_refused(data):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="classifier"):
        sp.dml(
            data.assign(dc=(data["cont"] > 0).astype(int) + 1),  # values 1/2
            y="y",
            treat="dc",
            covariates=COVARIATES,
            model="plr",
            ml_g=LinearRegression(),
            ml_m=LogisticRegression(),
            n_folds=2,
        )


# ---------------------------------------------------------------- sensitivity
def _irm_pair(data, folds, score, cluster=None):
    fit = sp.dml(
        data,
        y="y",
        treat="d",
        covariates=COVARIATES,
        model="irm",
        score=score,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(),
        n_folds=N_FOLDS,
        fold_indices=None if cluster else folds,
        cluster=cluster,
        random_state=3,
    )
    return fit


def _doubleml_irm(data, splits, score, cluster=None):
    kwargs = {"cluster_cols": cluster} if cluster else {}
    dd = doubleml.DoubleMLData(
        data, y_col="y", d_cols="d", x_cols=COVARIATES, **kwargs
    )
    ref = doubleml.DoubleMLIRM(
        dd,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(),
        score=score,
        n_folds=N_FOLDS,
        draw_sample_splitting=False,
    )
    ref.set_sample_splitting(splits)
    ref.fit()
    ref.sensitivity_analysis(cf_y=CF_Y, cf_d=CF_D, rho=1.0, level=0.95)
    return ref


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_irm_sensitivity_matches_doubleml(data, folds, score):
    fit = _irm_pair(data, folds, score)
    ref = _doubleml_irm(data, _splits(folds), score)
    assert fit.estimate == pytest.approx(float(ref.coef[0]), rel=1e-10)
    sens = sp.dml_sensitivity(fit, cf_y=CF_Y, cf_d=CF_D)
    params = ref.sensitivity_params
    assert sens.adjusted_estimate_low == pytest.approx(
        float(params["theta"]["lower"][0]), rel=1e-10
    )
    assert sens.adjusted_estimate_high == pytest.approx(
        float(params["theta"]["upper"][0]), rel=1e-10
    )
    assert sens.rv_q == pytest.approx(float(params["rv"][0]), rel=1e-5)
    # crosswise: see the module docstring
    assert sens.se_low == pytest.approx(float(params["se"]["upper"][0]), rel=1e-10)
    assert sens.se_high == pytest.approx(float(params["se"]["lower"][0]), rel=1e-10)


def test_irm_bound_is_not_the_plr_scaling(data, folds):
    """The pre-fix scaling sd(psi)/sd(D - m) gives a different bound."""
    fit = _irm_pair(data, folds, "ATE")
    info = fit.model_info
    old_s = float(
        np.sqrt(np.mean(info["_y_resid"] ** 2)) / np.sqrt(np.mean(info["_d_resid"] ** 2))
    )
    sens = sp.dml_sensitivity(fit, cf_y=CF_Y, cf_d=CF_D)
    assert abs(sens.s - old_s) / sens.s > 0.05


def test_sensitivity_is_refused_for_iv_models(data, folds):
    fit = sp.dml(
        data,
        y="y",
        treat="d",
        instrument="z",
        covariates=COVARIATES,
        model="pliv",
        ml_g=LinearRegression(),
        ml_m=LinearRegression(),
        ml_r=LinearRegression(),
        n_folds=N_FOLDS,
        fold_indices=folds,
    )
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="pliv"):
        sp.dml_sensitivity(fit, cf_y=CF_Y, cf_d=CF_D)


def _cluster_folds(data):
    """Cluster-level folds, expressed as row splits for doubleml."""
    cl = data["g"].values
    fold_of_cluster = np.arange(cl.max() + 1) % N_FOLDS
    return fold_of_cluster[cl]


@pytest.mark.parametrize("model", ["irm", "plr"])
def test_clustered_sensitivity_matches_doubleml(data, model):
    row_folds = _cluster_folds(data)
    if model == "irm":
        fit = sp.dml(
            data, y="y", treat="d", covariates=COVARIATES, model="irm",
            ml_g=LinearRegression(), ml_m=LogisticRegression(),
            n_folds=N_FOLDS, fold_indices=row_folds, cluster="g",
        )  # fmt: skip
    else:
        fit = sp.dml(
            data, y="y", treat="d", covariates=COVARIATES, model="plr",
            ml_g=LinearRegression(), ml_m=LinearRegression(),
            n_folds=N_FOLDS, fold_indices=row_folds, cluster="g",
        )  # fmt: skip
    dd = doubleml.DoubleMLData(
        data, y_col="y", d_cols="d", x_cols=COVARIATES, cluster_cols="g"
    )
    if model == "irm":
        ref = doubleml.DoubleMLIRM(
            dd, ml_g=LinearRegression(), ml_m=LogisticRegression(), n_folds=N_FOLDS
        )
    else:
        ref = doubleml.DoubleMLPLR(
            dd, ml_l=LinearRegression(), ml_m=LinearRegression(), n_folds=N_FOLDS
        )
    # Overwrite doubleml's own draw with the shared cluster-level partition.
    row_splits = _splits(row_folds)[0]
    n_cl = int(data["g"].max()) + 1
    cl_fold = np.arange(n_cl) % N_FOLDS
    ref._smpls = [row_splits]
    ref._smpls_cluster = [
        [
            ([np.flatnonzero(cl_fold != f)], [np.flatnonzero(cl_fold == f)])
            for f in range(N_FOLDS)
        ]
    ]
    ref.fit()
    ref.sensitivity_analysis(cf_y=CF_Y, cf_d=CF_D, rho=1.0, level=0.95)
    assert fit.estimate == pytest.approx(float(ref.coef[0]), rel=1e-10)
    assert fit.se == pytest.approx(float(ref.se[0]), rel=1e-10)
    sens = sp.dml_sensitivity(fit, cf_y=CF_Y, cf_d=CF_D)
    params = ref.sensitivity_params
    assert sens.adjusted_estimate_low == pytest.approx(
        float(params["theta"]["lower"][0]), rel=1e-10
    )
    tol = 1e-10 if model == "irm" else 5e-4
    assert sens.se_low == pytest.approx(float(params["se"]["upper"][0]), rel=tol)
    assert sens.se_high == pytest.approx(float(params["se"]["lower"][0]), rel=tol)
    zero = sp.dml_sensitivity(fit, cf_y=0.0, cf_d=0.0)
    assert zero.se_low == pytest.approx(fit.se, rel=1e-12)


def test_repeated_cross_fitting_aggregates_by_the_median(data):
    fit = sp.dml(
        data, y="y", treat="d", covariates=COVARIATES, model="irm",
        ml_g=LinearRegression(), ml_m=LogisticRegression(),
        n_folds=N_FOLDS, n_rep=3, random_state=5,
    )  # fmt: skip
    reps = fit.model_info["_sens"]
    assert len(reps) == 3
    sens = sp.dml_sensitivity(fit, cf_y=CF_Y, cf_d=CF_D)
    strength = np.sqrt(CF_Y * CF_D / (1 - CF_D))
    lows = [r["theta"] - strength * np.sqrt(r["sigma2"] * r["nu2"]) for r in reps]
    assert sens.adjusted_estimate_low == pytest.approx(np.median(lows), rel=1e-12)
    assert sens.ci_low < sens.adjusted_estimate_low < sens.adjusted_estimate_high
