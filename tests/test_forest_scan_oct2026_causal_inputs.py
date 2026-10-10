"""Input handling of ``sp.causal_forest``: what is accepted, what is refused.

Three families of checks:

* every documented way of passing the data (formula, column names, arrays;
  categorical effect modifiers; clusters by name) reaches the same fit;
* an input the estimator cannot use is refused with a StatsPAI taxonomy
  error before anything is estimated, and a forest that cannot support
  in-sample inference (no out-of-bag prediction, no little bags) says so
  instead of returning a number;
* the legacy engine (``split_rule="legacy"``, deprecated) still computes
  what its docstrings describe.

Forests here are tiny (300 rows, 60 trees); nothing asserts closeness to a
truth.  ``RTOL = 1e-10`` compares two floating-point routes to one formula.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)
from statspai.forest import CausalForest

RTOL = 1e-10
N = 300
KW = dict(n_estimators=60, random_state=5)


def _data(seed=201, n=N):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    T = rng.binomial(1, 0.5, n).astype(float)
    Y = X[:, 1] + (1 + X[:, 0]) * T + rng.normal(size=n)
    return Y, T, X


@pytest.fixture(scope="module")
def frame():
    Y, T, X = _data()
    rng = np.random.default_rng(202)
    return pd.DataFrame(
        {
            "y": Y,
            "d": T,
            "x0": X[:, 0],
            "x1": X[:, 1],
            "x2": X[:, 2],
            "g": rng.choice(["a", "b", "c"], size=N),
            "cl": rng.integers(0, 30, N),
        }
    )


@pytest.fixture(scope="module")
def fitted(frame):
    return sp.causal_forest(
        Y=frame["y"].to_numpy(),
        T=frame["d"].to_numpy(),
        X=frame[["x0", "x1", "x2"]].to_numpy(),
        **KW,
    )


# --------------------------------------------------------------------------- #
#  Three interfaces, one fit
# --------------------------------------------------------------------------- #


def test_formula_columns_and_arrays_give_the_same_forest(frame, fitted):
    by_formula = sp.causal_forest("y ~ d | x0 + x1 + x2", data=frame, **KW)
    by_columns = sp.causal_forest(data=frame, y="y", d="d", x=["x0", "x1", "x2"], **KW)
    by_house_names = sp.causal_forest(
        data=frame, y="y", treat="d", covariates=["x0", "x1", "x2"], **KW
    )
    ref = fitted.predict()
    for cf in (by_formula, by_columns, by_house_names):
        np.testing.assert_array_equal(cf.predict(), ref)
        assert cf._feature_names == ["x0", "x1", "x2"]
    assert fitted._feature_names == ["X0", "X1", "X2"]
    # Prediction from a frame picks the fitted columns by name, in the
    # fitted order, whatever else the frame holds.
    shuffled = frame[["x2", "g", "x0", "y", "x1"]].head(7)
    np.testing.assert_array_equal(
        by_columns.predict(shuffled), by_columns.effect(frame[["x0", "x1", "x2"]][:7])
    )


def test_n_trees_and_unit_are_aliases(frame):
    a = sp.causal_forest("y ~ d | x0 + x1", data=frame, n_trees=20, random_state=1)
    assert a.diagnostics["n_estimators"] == 20
    with pytest.raises(TypeError):
        sp.causal_forest("y ~ d | x0", data=frame, n_trees=20, n_estimators=20)


def test_controls_enter_the_nuisances_but_not_the_effect_modifiers(frame):
    cf = sp.causal_forest("y ~ d | x0 + x1 | x2", data=frame, **KW)
    assert cf.data_info["n_features"] == 2 and cf.data_info["n_controls"] == 1
    assert cf._feature_names == ["x0", "x1"]
    assert cf.effect(frame[["x0", "x1"]].head(3)).shape == (3,)
    by_cols = sp.causal_forest(data=frame, y="y", d="d", x=["x0", "x1"], w="x2", **KW)
    np.testing.assert_array_equal(by_cols.predict(), cf.predict())
    assert by_cols._control_names == ["x2"]
    # One-dimensional W and X arrays are single columns.
    arr = sp.causal_forest(
        Y=frame["y"].to_numpy(),
        T=frame["d"].to_numpy(),
        X=frame["x0"].to_numpy(),
        W=frame["x2"].to_numpy(),
        **KW,
    )
    assert arr.data_info["n_features"] == 1 and arr.data_info["n_controls"] == 1
    assert arr.effect(np.array([0.1, 0.2])).shape == (2,)
    assert "Number of covariates:       1" in arr.summary()


def test_clusters_may_be_a_column_name(frame):
    by_name = sp.causal_forest("y ~ d | x0 + x1", data=frame, clusters="cl", **KW)
    by_array = sp.causal_forest(
        "y ~ d | x0 + x1", data=frame, clusters=frame["cl"].to_numpy(), **KW
    )
    np.testing.assert_array_equal(by_name.predict(), by_array.predict())
    assert by_name.diagnostics["n_clusters"] == frame["cl"].nunique()
    assert "Clusters:                 30" in by_name.summary()
    with pytest.raises(MethodIncompatibility, match="not a column of data"):
        sp.causal_forest("y ~ d | x0", data=frame, clusters="nope", **KW)
    with pytest.raises(MethodIncompatibility, match="not a column of data"):
        sp.causal_forest(Y=[1.0], T=[1.0], X=[1.0], clusters="cl")


@pytest.mark.parametrize(
    "formula, match",
    [
        ("y ~ d", "Formula must have format"),
        ("y d | x0", "must contain '~'"),
        ("y ~ nope | x0", "not found in data"),
        ("y ~ d | nope", r"Missing columns: \['nope'\]"),
        ("y ~ d | x0 | nope", "not found in data"),
    ],
)
def test_malformed_formulas_are_refused(frame, formula, match):
    with pytest.raises(ValueError, match=match):
        sp.causal_forest(formula, data=frame, **KW)


def test_column_interface_argument_checks(frame):
    with pytest.raises(MethodIncompatibility, match="require data"):
        sp.causal_forest(y="y", d="d", x=["x0"])
    with pytest.raises(MethodIncompatibility, match="not a mixture"):
        sp.causal_forest("y ~ d | x0", data=frame, y="y", d="d", x=["x0"])
    with pytest.raises(MethodIncompatibility, match="needs all of"):
        sp.causal_forest(data=frame, y="y", x=["x0"])
    with pytest.raises(MethodIncompatibility, match=r"not in data: \['nope'\]"):
        sp.causal_forest(data=frame, y="y", d="d", x=["x0", "nope"])
    with pytest.raises(ValueError, match="Must provide either"):
        sp.causal_forest(data=frame)
    with pytest.raises(ValueError, match="Must provide either"):
        CausalForest().fit(formula="y ~ d | x0")


# --------------------------------------------------------------------------- #
#  Categorical effect modifiers
# --------------------------------------------------------------------------- #


def test_text_columns_become_one_indicator_per_level(frame):
    cf = sp.causal_forest(data=frame, y="y", d="d", x=["x0", "g"], **KW)
    assert cf._feature_names == ["x0", "g[a]", "g[b]", "g[c]"]
    by_formula = sp.causal_forest("y ~ d | x0 + g", data=frame, **KW)
    np.testing.assert_array_equal(by_formula.predict(), cf.predict())
    # The indicators are what a hand-built design would hold.
    design = np.column_stack(
        [frame["x0"]] + [(frame["g"] == lv).astype(float) for lv in "abc"]
    )
    np.testing.assert_array_equal(cf._X_original, design)
    # New rows are encoded the same way, from the raw column.
    new = frame.head(6)
    np.testing.assert_array_equal(cf.predict(new), cf.effect(design[:6]))
    # category dtype is treated like text.
    as_cat = frame.assign(g=frame["g"].astype("category"))
    cat = sp.causal_forest(data=as_cat, y="y", d="d", x=["x0", "g"], **KW)
    np.testing.assert_array_equal(cat.predict(), cf.predict())


def test_a_level_not_seen_in_training_is_an_error_at_prediction(frame):
    cf = sp.causal_forest(data=frame, y="y", d="d", x=["x0", "g"], **KW)
    new = frame.head(4).copy()
    new.loc[new.index[0], "g"] = "zzz"
    for call in (cf.predict, cf.effect, lambda d: cf.effect_interval(d)):
        with pytest.raises(MethodIncompatibility, match="levels that were not"):
            call(new)
    # A missing level in a new row is a missing covariate, also refused.
    gap = frame.head(4).copy()
    gap["g"] = gap["g"].astype(object)
    gap.loc[gap.index[1], "g"] = None
    with pytest.raises(MethodIncompatibility, match="NaN or infinite"):
        cf.predict(gap)
    with pytest.raises(MethodIncompatibility, match="missing effect-modifier"):
        cf.predict(frame[["x0"]].head(3))


def test_a_missing_level_in_the_fit_data_is_refused(frame):
    holes = frame.copy()
    holes["g"] = holes["g"].astype(object)
    holes.loc[holes.index[:3], "g"] = None
    with pytest.raises(MethodIncompatibility, match="X contains NaN"):
        sp.causal_forest(data=holes, y="y", d="d", x=["x0", "g"], **KW)


def test_formula_accepts_an_explicit_factor_term(frame):
    coded = frame.assign(k=frame["g"].map({"a": 1, "b": 2, "c": 3}))
    cf = sp.causal_forest("y ~ d | x0 + C(k)", data=coded, **KW)
    assert cf._feature_names == ["x0", "k[1]", "k[2]", "k[3]"]
    # Without C() an integer column is one numeric feature.
    numeric = sp.causal_forest("y ~ d | x0 + k", data=coded, **KW)
    assert numeric._feature_names == ["x0", "k"]


def test_column_interface_accepts_an_explicit_factor_term(frame):
    coded = frame.assign(k=frame["g"].map({"a": 1, "b": 2, "c": 3}))
    by_formula = sp.causal_forest("y ~ d | x0 + C(k)", data=coded, **KW)
    by_columns = sp.causal_forest(data=coded, y="y", d="d", x=["x0", "C(k)"], **KW)
    assert by_columns._feature_names == ["x0", "k[1]", "k[2]", "k[3]"]
    np.testing.assert_array_equal(by_columns.predict(), by_formula.predict())


# --------------------------------------------------------------------------- #
#  Constructor options
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "option, value, match",
    [
        ("n_estimators", 0, "n_estimators"),
        ("n_estimators", 2.5, "n_estimators"),
        ("n_estimators", True, "n_estimators"),
        ("min_samples_leaf", 0, "min_samples_leaf"),
        ("min_samples_leaf", True, "min_samples_leaf"),
        ("max_depth", 0, "max_depth"),
        ("max_depth", 1.5, "max_depth"),
        ("max_samples", "half", "max_samples"),
        ("max_samples", 0.0, "max_samples"),
        ("max_samples", 1.5, "max_samples"),
        ("max_samples", float("nan"), "max_samples"),
        ("split_rule", "cart", "split_rule"),
        ("split_rule", "cffe", "fixed effects"),
        ("bootstrap", True, "bootstrap"),
        ("fe", "oneway", "fe must be"),
        ("ci_group_size", 0, "ci_group_size"),
        ("ci_group_size", True, "ci_group_size"),
        ("nuisance_folds", 0, "nuisance_folds"),
        ("honesty_fraction", 0.0, "honesty_fraction"),
        ("honesty_fraction", 1.0, "honesty_fraction"),
        ("alpha", 0.25, "alpha"),
        ("alpha", -0.1, "alpha"),
        ("imbalance_penalty", -1.0, "imbalance_penalty"),
        ("mtry", 0, "mtry"),
        ("mtry", 1.5, "mtry"),
        ("mtry", True, "mtry"),
    ],
)
def test_bad_options_are_refused_before_fitting(option, value, match):
    Y, T, X = _data(n=60)
    with pytest.raises(MethodIncompatibility, match=match):
        sp.causal_forest(Y=Y, T=T, X=X, **{option: value})


def test_little_bags_need_half_samples():
    Y, T, X = _data(n=60)
    with pytest.raises(MethodIncompatibility, match="at most 0.5"):
        sp.causal_forest(Y=Y, T=T, X=X, max_samples=0.8, ci_group_size=2)
    with pytest.raises(MethodIncompatibility, match="user-supplied nuisance"):
        sp.causal_forest(
            Y=Y, T=T, X=X, model_y=LinearRegression(), nuisance_folds=1, **KW
        )


def test_large_subsamples_switch_variance_estimates_off_with_a_warning():
    Y, T, X = _data()
    with pytest.warns(AssumptionWarning, match="variance estimates are disabled"):
        cf = sp.causal_forest(Y=Y, T=T, X=X, max_samples=0.8, **KW)
    assert cf.ci_group_size == 1
    for call in (lambda: cf.effect_interval(X[:3]), lambda: cf.effect_variance()):
        with pytest.raises(MethodIncompatibility, match="ci_group_size >= 2"):
            call()
    # Averages do not need little bags.
    assert np.isfinite(cf.average_treatment_effect()["se"])
    # forest_support() reports the effect without a standard error.
    sup = sp.forest_support(cf, X[:4])
    assert sup["se"].isna().all() and np.isfinite(sup["cate"]).all()


def test_tree_count_is_rounded_up_to_whole_little_bags():
    Y, T, X = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=7, random_state=1)
    assert cf.diagnostics["n_estimators"] == 8 and cf._engine.num_trees == 8


# --------------------------------------------------------------------------- #
#  Data the forest cannot be fitted on
# --------------------------------------------------------------------------- #


def _fit_with(**override):
    Y, T, X = _data(n=60)
    args = dict(Y=Y, T=T, X=X)
    args.update(override)
    return sp.causal_forest(**args, **KW)


def test_non_finite_inputs_are_refused_by_name():
    Y, T, X = _data(n=60)
    for name, arr in (("Y", Y), ("T", T), ("X", X), ("W", X[:, :1])):
        for bad_value in (np.nan, np.inf):
            bad = arr.copy()
            bad.flat[0] = bad_value
            label = {"Y": "outcome", "T": "treatment"}.get(name, name)
            with pytest.raises(MethodIncompatibility, match=f"{label} contains NaN"):
                _fit_with(**{name: bad})


def test_shape_mismatches_are_refused():
    Y, T, X = _data(n=60)
    with pytest.raises(MethodIncompatibility, match="same row count"):
        _fit_with(T=T[:50])
    with pytest.raises(MethodIncompatibility, match="same row count"):
        _fit_with(X=X[:50])
    with pytest.raises(MethodIncompatibility, match="W must have the same row"):
        _fit_with(W=X[:50])
    with pytest.raises(MethodIncompatibility, match="1D or 2D"):
        _fit_with(X=np.zeros((60, 2, 2)))
    with pytest.raises(MethodIncompatibility, match="W must be a 1D or 2D"):
        _fit_with(W=np.zeros((60, 2, 2)))
    with pytest.raises(MethodIncompatibility, match="at least one feature"):
        _fit_with(X=np.zeros((60, 0)))
    with pytest.raises(MethodIncompatibility, match="numeric"):
        _fit_with(X=np.array([["a", "b"]] * 60))
    with pytest.raises(DataInsufficient, match="at least 3 rows"):
        sp.causal_forest(Y=Y[:2], T=T[:2], X=X[:2])
    with pytest.raises(DataInsufficient, match="too few rows per tree"):
        sp.causal_forest(Y=Y[:6], T=T[:6], X=X[:6], max_samples=0.2)


def test_treatment_coding_is_checked():
    Y, T, X = _data(n=60)
    with pytest.raises(DataInsufficient, match="two treatment values"):
        _fit_with(T=np.ones(60))
    with pytest.raises(MethodIncompatibility, match="binary treatment only"):
        _fit_with(T=np.arange(60) % 3)
    with pytest.raises(MethodIncompatibility, match="coded as 0/1"):
        _fit_with(T=T + 1.0)
    lonely = np.zeros(60)
    lonely[:2] = 1.0
    with pytest.raises(DataInsufficient, match="at least 3 observations per"):
        _fit_with(T=lonely)
    with pytest.raises(DataInsufficient, match="treatment variation"):
        sp.causal_forest(Y=Y, T=np.full(60, 2.0), X=X, discrete_treatment=False)
    # Booleans are a valid 0/1 coding.
    cf = _fit_with(T=T.astype(bool))
    assert set(cf.data_info["treatment_values"]) == {0.0, 1.0}


def test_clusters_and_precomputed_nuisances_are_checked():
    Y, T, X = _data(n=60)
    with pytest.raises(MethodIncompatibility, match="one label per row"):
        _fit_with(clusters=np.arange(10))
    with pytest.raises(MethodIncompatibility, match="missing values"):
        _fit_with(clusters=np.r_[np.nan, np.arange(59.0)])
    with pytest.raises(MethodIncompatibility, match="missing values"):
        _fit_with(clusters=np.array([None] + ["a", "b"] * 29 + ["a"], dtype=object))
    with pytest.raises(DataInsufficient, match="at least two clusters"):
        _fit_with(clusters=np.zeros(60))
    with pytest.raises(MethodIncompatibility, match="Y_hat must have one value"):
        _fit_with(Y_hat=np.zeros(7))
    with pytest.raises(MethodIncompatibility, match="W_hat contains NaN"):
        _fit_with(W_hat=np.r_[np.nan, np.full(59, 0.5)])
    with pytest.raises(MethodIncompatibility, match="only used with fe"):
        _fit_with(id=np.arange(60))
    with pytest.raises(MethodIncompatibility, match="only used with fe"):
        _fit_with(time=np.arange(60))


def test_scalar_nuisances_are_broadcast_and_reported():
    cf = _fit_with(Y_hat=0.25, W_hat=0.5)
    nu = cf.get_nuisances()
    np.testing.assert_array_equal(nu["Y_hat"], np.full(60, 0.25))
    np.testing.assert_array_equal(nu["W_hat"], np.full(60, 0.5))
    assert nu["source"] == {"Y_hat": "user-supplied", "W_hat": "user-supplied"}
    for arr in (nu["Y_hat"], nu["W_hat"]):
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 9.0
    assert nu["overlap"]["applicable"] is True
    with pytest.raises(MethodIncompatibility, match="requires a fitted forest"):
        CausalForest().get_nuisances()


def test_thin_overlap_in_the_stored_propensity_is_reported_at_fit_time():
    Y, T, X = _data(n=200)
    e = np.full(200, 0.5)
    e[:30] = 0.001
    e[30:45] = 0.9995
    T = T.copy()
    T[:30], T[30:45] = 0.0, 1.0
    with pytest.warns(AssumptionWarning, match=r"45/200 \(22\.5%\)"):
        cf = sp.causal_forest(Y=Y, T=T, X=X, W_hat=e, **KW)
    info = cf.diagnostics["nuisance_overlap"]
    assert (info["n_below"], info["n_above"]) == (30, 15)
    assert info["share_outside"] == pytest.approx(0.225)
    assert info["min"] == 0.001 and info["max"] == 0.9995
    assert info["n_at_zero"] == 0 and info["n_at_one"] == 0 and info["warning"]
    # Exactly-zero propensities make the unclipped score undefined.
    e0 = e.copy()
    e0[:30] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf0 = sp.causal_forest(Y=Y, T=T, X=X, W_hat=e0, **KW)
    assert cf0.diagnostics["nuisance_overlap"]["n_at_zero"] == 30
    with pytest.raises(DataInsufficient, match="strictly between 0 and 1"):
        cf0.average_treatment_effect(clip=0.0)
    with pytest.raises(DataInsufficient, match="strictly between 0 and 1"):
        sp.get_scores(cf0)
    assert np.isfinite(cf0.average_treatment_effect()["estimate"])  # clipped
    assert np.isfinite(cf0.average_treatment_effect("overlap", clip=0.0)["estimate"])


# --------------------------------------------------------------------------- #
#  User-supplied nuisance models
# --------------------------------------------------------------------------- #


def test_user_models_are_cross_fitted_in_the_requested_folds():
    Y, T, X = _data()
    folds, seed = 4, 9
    cf = sp.causal_forest(
        Y=Y,
        T=T,
        X=X,
        model_y=LinearRegression(),
        model_t=LogisticRegression(),
        nuisance_folds=folds,
        n_estimators=60,
        random_state=seed,
    )
    y_hat, w_hat = np.empty(N), np.empty(N)
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(X):
        y_hat[test] = LinearRegression().fit(X[train], Y[train]).predict(X[test])
        w_hat[test] = (
            LogisticRegression().fit(X[train], T[train]).predict_proba(X[test])[:, 1]
        )
    nu = cf.get_nuisances()
    np.testing.assert_allclose(nu["Y_hat"], y_hat, rtol=RTOL)
    np.testing.assert_allclose(nu["W_hat"], w_hat, rtol=RTOL)
    assert nu["source"] == {
        "Y_hat": "cross-fitted LinearRegression",
        "W_hat": "cross-fitted LogisticRegression",
    }
    # A treatment model without predict_proba is used through predict().
    reg = sp.causal_forest(
        Y=Y, T=T, X=X, model_t=LinearRegression(), n_estimators=60, random_state=seed
    )
    w_lin = np.empty(N)
    for train, test in KFold(3, shuffle=True, random_state=seed).split(X):
        w_lin[test] = LinearRegression().fit(X[train], T[train]).predict(X[test])
    np.testing.assert_allclose(reg.get_nuisances()["W_hat"], w_lin, rtol=RTOL)
    assert reg.get_nuisances()["source"]["Y_hat"] == "grf regression forest (OOB)"


def test_cross_fitting_never_splits_a_cluster():
    """Rows of a cluster share their covariates exactly, so a 1-nearest-
    neighbour outcome model reproduces a same-cluster outcome whenever the
    cluster is split across folds.  With cluster folds it cannot."""
    rng = np.random.default_rng(203)
    G, m = 40, 6
    cl = np.repeat(np.arange(G), m)
    X = rng.normal(size=(G, 2))[cl]
    T = rng.binomial(1, 0.5, G * m).astype(float)
    Y = rng.normal(size=G * m) + 3.0 * np.arange(G)[cl]
    cf = sp.causal_forest(
        Y=Y,
        T=T,
        X=X,
        clusters=cl,
        model_y=KNeighborsRegressor(n_neighbors=1),
        nuisance_folds=5,
        **KW,
    )
    y_hat = cf.get_nuisances()["Y_hat"]
    # Every prediction is an observed outcome of *another* cluster.
    donor = np.array([np.flatnonzero(Y == v)[0] for v in y_hat])
    assert np.all(cl[donor] != cl)


# --------------------------------------------------------------------------- #
#  Reproducibility and random streams
# --------------------------------------------------------------------------- #


def test_results_do_not_depend_on_the_number_of_threads(fitted, frame):
    X = frame[["x0", "x1", "x2"]].to_numpy()
    other = sp.causal_forest(
        Y=frame["y"].to_numpy(), T=frame["d"].to_numpy(), X=X, n_jobs=2, **KW
    )
    np.testing.assert_array_equal(other.predict(), fitted.predict())
    np.testing.assert_array_equal(other.effect(X[:9]), fitted.effect(X[:9]))
    np.testing.assert_array_equal(
        other.get_nuisances()["W_hat"], fitted.get_nuisances()["W_hat"]
    )


def _bags(cf):
    """The set of subsamples the forest's trees were grown on."""
    return {row.tobytes() for row in cf._engine.drawn_bitmap}


def test_adjacent_seeds_do_not_share_subsamples(fitted, frame):
    X = frame[["x0", "x1", "x2"]].to_numpy()
    nxt = sp.causal_forest(
        Y=frame["y"].to_numpy(),
        T=frame["d"].to_numpy(),
        X=X,
        n_estimators=60,
        random_state=KW["random_state"] + 1,
    )
    # Two trees per little bag share one half-sample by construction.
    assert len(_bags(fitted)) == 30 and len(_bags(nxt)) == 30
    assert _bags(fitted).isdisjoint(_bags(nxt))


def test_nuisance_forests_use_their_own_random_streams():
    """With Y identical to T the two nuisance forests fit the same target.
    If they shared the main seed's subsamples their out-of-bag predictions
    would be identical; independent streams give different ones."""
    _, T, X = _data()
    nu = sp.causal_forest(Y=T.copy(), T=T, X=X, W_hat=None, **KW).get_nuisances()
    assert np.mean(np.abs(nu["Y_hat"] - nu["W_hat"])) > 1e-3


# --------------------------------------------------------------------------- #
#  Prediction inputs
# --------------------------------------------------------------------------- #


def test_prediction_rows_are_validated_against_the_fitted_schema(fitted, frame):
    X = frame[["x0", "x1", "x2"]].to_numpy()
    # One vector with as many entries as features is one row.
    np.testing.assert_array_equal(fitted.effect(X[4]), fitted.effect(X[4:5]))
    with pytest.raises(MethodIncompatibility, match="one-dimensional input has 2"):
        fitted.effect(np.zeros(2))
    with pytest.raises(MethodIncompatibility, match="1D or 2D"):
        fitted.effect(np.zeros((2, 3, 1)))
    with pytest.raises(MethodIncompatibility, match="expected 3"):
        fitted.effect(np.zeros((2, 5)))
    with pytest.raises(DataInsufficient, match="no prediction rows"):
        fitted.effect(np.zeros((0, 3)))
    with pytest.raises(MethodIncompatibility, match="NaN or infinite"):
        fitted.effect(np.array([[0.0, np.nan, 0.0]]))
    with pytest.raises(MethodIncompatibility, match="must be numeric"):
        fitted.effect([["a", "b", "c"]])
    as_text = pd.DataFrame({"X0": ["a"], "X1": ["b"], "X2": ["c"]})
    with pytest.raises(MethodIncompatibility, match="must be numeric"):
        fitted.predict(as_text)
    named = pd.DataFrame(X[:3], columns=["X0", "X1", "X2"])
    np.testing.assert_array_equal(fitted.predict(named), fitted.effect(X[:3]))


def test_unfitted_forests_refuse_every_query():
    cf = CausalForest(n_estimators=10)
    assert cf.summary() == "Causal Forest (not fitted)"
    assert repr(cf) == "CausalForest(fitted=False, n_estimators=10)"
    X = np.zeros((2, 2))
    calls = [
        lambda: cf.effect(X),
        lambda: cf.predict(),
        lambda: cf.effect_interval(X),
        lambda: cf.effect_variance(),
        lambda: cf.oob_effect(),
        lambda: cf.variable_importance(),
        lambda: cf.best_linear_projection(),
        lambda: cf.ate(),
        lambda: cf.att(),
        lambda: cf.average_treatment_effect(),
        lambda: sp.calibration_test(cf),
        lambda: sp.rate(cf),
        lambda: sp.forest_diagnostics(cf),
        lambda: sp.calibrate_cate(cf),
    ]
    for call in calls:
        with pytest.raises(MethodIncompatibility):
            call()
    with pytest.raises(MethodIncompatibility, match="GRF engine"):
        sp.forest_group_effects(cf)


# --------------------------------------------------------------------------- #
#  Too few trees for in-sample inference
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def two_trees():
    Y, T, X = _data()
    with pytest.warns(AssumptionWarning, match="no out-of-bag prediction"):
        cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=2, random_state=1)
    return cf, X


def test_rows_in_every_tree_have_no_out_of_bag_prediction(two_trees):
    cf, X = two_trees
    oob = cf.predict()
    # One little bag draws half the rows; those rows are in both trees.
    assert int(np.isnan(oob).sum()) == N // 2
    assert cf.diagnostics["n_rows_without_oob_prediction"] == N // 2
    # Predictions for explicit rows use every tree and are always defined.
    assert np.isfinite(cf.effect(X)).all()


def test_in_sample_inference_refuses_without_out_of_bag_predictions(two_trees):
    cf, _ = two_trees
    calls = [
        lambda: cf.average_treatment_effect(),
        lambda: cf.average_treatment_effect("overlap"),
        lambda: cf.best_linear_projection(),
        lambda: sp.best_linear_projection(cf),
        lambda: sp.calibration_test(cf),
        lambda: sp.rate(cf),
        lambda: sp.get_scores(cf),
        lambda: sp.forest_group_effects(cf),
    ]
    for call in calls:
        with pytest.raises(DataInsufficient, match="no out-of-bag CATE"):
            call()
    with pytest.raises(NumericalInstability, match="no little bag"):
        cf.effect_variance()
    # The scalar wrapper reports the failure instead of hiding it.
    assert "DataInsufficient" in cf.ate().inference_error


# --------------------------------------------------------------------------- #
#  Legacy engine
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def legacy():
    Y, T, X = _data()
    with pytest.warns(DeprecationWarning, match="split_rule='legacy'"):
        cf = sp.causal_forest(
            Y=Y, T=T, X=X, split_rule="legacy", n_estimators=20, random_state=3
        )
    return cf, Y, T, X


def test_legacy_forest_reports_its_engine_and_refuses_grf_only_tools(legacy):
    cf, _, _, X = legacy
    assert cf.diagnostics["engine"] == "legacy" and cf._engine is None
    assert cf.diagnostics["average_treatment_effect"] == pytest.approx(
        cf.effect(X).mean(), rel=RTOL
    )
    np.testing.assert_array_equal(cf.predict(), cf.effect(X))
    assert cf.get_nuisances()["source"]["Y_hat"] == "legacy cross_val_predict"
    for call in (
        cf.oob_effect,
        cf.effect_variance,
        cf.split_frequencies,
        lambda: cf.variable_importance(method="split"),
        lambda: sp.get_scores(cf),
        lambda: sp.best_linear_projection(cf),
        lambda: sp.variable_importance(cf),
        lambda: sp.calibrate_cate(cf),
        lambda: sp.forest_group_effects(cf),
        lambda: cf.average_treatment_effect(subset=[0, 1, 2]),
        lambda: sp.calibration_test(cf, method="within"),
    ):
        with pytest.raises(MethodIncompatibility):
            call()
    # Its default importance is the permutation measure.
    imp = cf.variable_importance()
    assert imp.sum() == pytest.approx(1.0) and len(imp) == 3


def test_legacy_options_that_need_the_grf_engine_are_refused():
    Y, T, X = _data(n=60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for extra in (
            dict(clusters=np.arange(60) % 5),
            dict(Y_hat=np.zeros(60)),
        ):
            with pytest.raises(MethodIncompatibility, match="require split_rule"):
                sp.causal_forest(Y=Y, T=T, X=X, split_rule="legacy", **extra)
        with pytest.raises(MethodIncompatibility, match="fe= requires"):
            sp.causal_forest(Y=Y, T=T, X=X, split_rule="legacy", fe="unit")


def test_legacy_averages_are_the_documented_scores(legacy):
    cf, Y, T, X = legacy
    nu = cf.get_nuisances()
    m, e = nu["Y_hat"], np.clip(nu["W_hat"], 0.01, 0.99)
    tau = cf.effect(X)
    psi = tau + (T - e) / (e * (1 - e)) * (Y - m - (T - e) * tau)
    out = cf.average_treatment_effect()
    assert out["estimate"] == pytest.approx(psi.mean(), rel=RTOL)
    assert out["se"] == pytest.approx(psi.std(ddof=1) / np.sqrt(N), rel=RTOL)
    assert out["method"] == "aipw"
    assert float(cf.ate()) == pytest.approx(psi.mean(), rel=RTOL)
    # Overlap: residual-on-residual OLS with an intercept (unclipped e).
    r = T - nu["W_hat"]
    D = np.column_stack([np.ones(N), r])
    beta = np.linalg.lstsq(D, Y - m, rcond=None)[0]
    ato = cf.average_treatment_effect("overlap")
    assert ato["estimate"] == pytest.approx(beta[1], rel=1e-9)
    assert ato["method"] == "r_learner_hc3"
    assert ato["effective_sample_size"] == pytest.approx(
        np.sum(r**2) ** 2 / np.sum(r**4), rel=RTOL
    )


def test_legacy_rows_without_nuisances_get_a_labelled_plug_in(legacy):
    cf, _, T, X = legacy
    new = X[:40] + 0.2
    with pytest.warns(AssumptionWarning, match="no cross-fitted nuisances"):
        out = cf.average_treatment_effect(X=new, T=T[:40])
    tau = cf.effect(new)
    assert out["method"] == "plug_in"
    assert out["plug_in_reason"] == "nuisances_unavailable"
    assert out["estimate"] == pytest.approx(tau.mean(), rel=RTOL)
    assert out["se"] == pytest.approx(
        np.sqrt(np.sum((tau - tau.mean()) ** 2)) / 40, rel=RTOL
    )
    with pytest.warns(AssumptionWarning):
        att = cf.average_treatment_effect(X=new, T=T[:40], target_sample="treated")
    assert att["estimate"] == pytest.approx(tau[T[:40] == 1].mean(), rel=RTOL)
    with pytest.warns(AssumptionWarning):
        atc = cf.average_treatment_effect(X=new, T=T[:40], target_sample="control")
    assert atc["estimate"] == pytest.approx(tau[T[:40] == 0].mean(), rel=RTOL)
    with pytest.warns(AssumptionWarning):
        with pytest.raises(MethodIncompatibility, match="needs propensity scores"):
            cf.average_treatment_effect(X=new, T=T[:40], target_sample="overlap")
    with pytest.raises(MethodIncompatibility, match="must match the number"):
        cf.average_treatment_effect(X=new, T=T[:10])
    with pytest.raises(MethodIncompatibility, match="NaN"):
        cf.average_treatment_effect(X=new, T=np.full(40, np.nan))
    with pytest.raises(DataInsufficient, match="no treated"):
        cf.average_treatment_effect(X=new, T=np.zeros(40), target_sample="treated")
    with pytest.raises(DataInsufficient, match="no control"):
        cf.average_treatment_effect(X=new, T=np.ones(40), target_sample="control")


def test_legacy_blp_is_hc1_on_standardised_covariates(legacy):
    cf, Y, T, X = legacy
    nu = cf.get_nuisances()
    m, e_raw = nu["Y_hat"], nu["W_hat"]
    e = np.clip(e_raw, 0.01, 0.99)
    tau = cf.effect(X)
    gamma = tau + (T - e) / (e * (1 - e)) * (Y - m - (T - e_raw) * tau)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    D = np.column_stack([np.ones(N), Z])
    bread = np.linalg.inv(D.T @ D)
    beta = bread @ D.T @ gamma
    u = gamma - D @ beta
    V = bread @ (D * u[:, None]).T @ (D * u[:, None]) @ bread * N / (N - 4)
    tab = cf.best_linear_projection()
    # rtol 1e-7: the implementation adds 1e-12 to each standard deviation.
    np.testing.assert_allclose(tab["coef"], beta, rtol=1e-7)
    np.testing.assert_allclose(tab["se"], np.sqrt(np.diag(V)), rtol=1e-7)
    crit = stats.t.ppf(0.975, N - 4)
    np.testing.assert_allclose(tab["ci_upper"], tab["coef"] + crit * tab["se"])
    assert cf.diagnostics["blp_n_clipped_propensities"] == int(
        np.sum((e_raw < 0.01) | (e_raw > 0.99))
    )
    # Other rows have no scores: a labelled plug-in regression.
    with pytest.warns(UserWarning, match="falling back to the plug-in"):
        other = cf.best_linear_projection(X[:50])
    assert other.shape == (4, 6)


def test_legacy_interval_is_flagged_as_not_a_confidence_interval(legacy):
    cf, _, _, X = legacy
    with pytest.warns(DeprecationWarning, match="not confidence intervals"):
        lo, hi = cf.effect_interval(X[:5], alpha=0.2)
    per_tree = np.array([tree.predict(X[:5]) for tree in cf._forest])
    np.testing.assert_allclose(lo, np.percentile(per_tree, 10, axis=0))
    np.testing.assert_allclose(hi, np.percentile(per_tree, 90, axis=0))


def test_legacy_calibration_and_rate_run_on_the_training_sample(legacy):
    cf, Y, T, X = legacy
    nu = cf.get_nuisances()
    tau = cf.effect(X)
    r = T - nu["W_hat"]
    D = np.column_stack([r * tau.mean(), r * (tau - tau.mean())])
    beta = np.linalg.lstsq(D, Y - nu["Y_hat"], rcond=None)[0]
    tab = sp.calibration_test(cf)
    np.testing.assert_allclose(tab["coef"], beta, rtol=1e-8)
    out = sp.rate(cf)
    assert out["priority_source"] == "in_sample" and np.isfinite(out["se"])
    with pytest.warns(DeprecationWarning, match="honest_variance"):
        hv = sp.honest_variance(cf, n_splits=40, seed=1)
    assert hv["ate"] == pytest.approx(tau.mean(), rel=RTOL)
    rng = np.random.default_rng(1)
    means = [tau[rng.permutation(N)[: N // 2]].mean() for _ in range(40)]
    assert hv["se"] == pytest.approx(np.std(means, ddof=1), rel=RTOL)
    for bad in (1, 2.5, True):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(MethodIncompatibility, match="n_splits"):
                sp.honest_variance(cf, n_splits=bad)


def test_honest_variance_on_a_grf_forest_is_the_doubly_robust_ate(fitted):
    with pytest.warns(DeprecationWarning, match="honest_variance"):
        hv = sp.honest_variance(fitted)
    ref = fitted.average_treatment_effect()
    assert hv["ate"] == ref["estimate"] and hv["se"] == ref["se"]
    assert hv["plug_in_mean"] == pytest.approx(fitted.predict().mean(), rel=RTOL)


def test_legacy_continuous_treatment_is_a_labelled_plug_in():
    rng = np.random.default_rng(204)
    X = rng.normal(size=(N, 2))
    T = X[:, 0] + rng.normal(size=N)
    Y = X[:, 1] + 0.5 * T + rng.normal(size=N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cf = sp.causal_forest(
            Y=Y,
            T=T,
            X=X,
            split_rule="legacy",
            discrete_treatment=False,
            n_estimators=15,
            random_state=1,
        )
    tau = cf.effect(X)
    with pytest.warns(AssumptionWarning, match="requires a binary treatment"):
        out = cf.average_treatment_effect()
    assert out["method"] == "plug_in"
    assert out["plug_in_reason"] == "non_binary_treatment"
    assert out["estimate"] == pytest.approx(tau.mean(), rel=RTOL)
    assert cf.average_treatment_effect("overlap")["method"] == "r_learner_hc3"
    with pytest.raises(MethodIncompatibility, match="binary treatment"):
        cf.average_treatment_effect("treated")
    tab = cf.best_linear_projection()
    assert cf.diagnostics["blp_n_clipped_propensities"] == 0
    # Continuous-treatment score of the docstring: weight (T - e) / Var(T - e).
    nu = cf.get_nuisances()
    r = T - nu["W_hat"]
    gamma = tau + r / np.mean(r**2) * (Y - nu["Y_hat"] - r * tau)
    assert tab.loc["Intercept", "coef"] == pytest.approx(gamma.mean(), rel=1e-7)


def test_legacy_verbose_fit_reports_its_stages(capsys):
    Y, T, X = _data(n=90)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # deprecation, thin overlap
        cf = sp.causal_forest(
            Y=Y,
            T=T,
            X=X[:, 0],
            split_rule="legacy",
            bootstrap=False,
            honest=False,
            n_estimators=3,
            verbose=2,
            random_state=1,
        )
    printed = capsys.readouterr().out
    for stage in ("first stage", "outcome model", "treatment model", "tree 1/3"):
        assert stage in printed
    # Without honesty every tree estimates its leaves on the rows it was
    # grown on, drawn without replacement: 45 of the 90 rows per tree.
    assert len(cf._forest) == 3
    assert all(tree.tree_.n_node_samples[0] == 45 for tree in cf._forest)
    assert (
        "forest was fitted with honest=False" in sp.forest_diagnostics(cf)["warnings"]
    )
