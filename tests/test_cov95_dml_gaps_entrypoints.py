"""Coverage tests for input validation in the DML entry points.

``sp.dml``, ``sp.dml_did``, ``sp.dml_panel``, ``sp.dynamic_dml`` and
``sp.dml_model_averaging`` each reject malformed arguments loudly
(CLAUDE.md section 3.7). These tests pin the exception type and the
message of the branches the main suites leave uncovered, plus a few
numeric branches (default learners, explicit folds, the stacking
fallback) checked against an independent computation.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import statspai as sp
from statspai.dml import did as _did
from statspai.dml import model_averaging as _ma
from statspai.dml._base import _DoubleMLBase
from statspai.exceptions import DataInsufficient, MethodIncompatibility

# ---------------------------------------------------------------------
# sp.dml (shared base class)
# ---------------------------------------------------------------------

XS = ["x0", "x1"]


def _plr_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    d = 0.5 * X[:, 0] + rng.normal(size=n)
    y = 1.5 * d + X[:, 1] + rng.normal(size=n)
    return pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "d": d, "y": y})


def _dml(df, **kw):
    args = dict(y="y", treat="d", covariates=XS, ml_g="linear", ml_m="linear")
    args.update(kw)
    return sp.dml(df, **args)


@pytest.mark.parametrize(
    "kw, fragment",
    [
        (dict(n_folds=2.5), "n_folds must be a positive integer"),
        (dict(alpha="five"), r"alpha must be finite and in the open interval"),
        (dict(random_state="seed"), "random_state must be integer-like"),
        (dict(normalize_ipw="yes"), "normalize_ipw must be a bool"),
        (dict(trimming_threshold="tight"), "trimming_threshold must be a float"),
        (dict(fold_indices="fold"), "fold_indices column 'fold' not in data"),
        (dict(sample_weight=["a"] * 120), "sample_weight must be numeric"),
    ],
)
def test_dml_rejects_malformed_arguments(kw, fragment):
    with pytest.raises(MethodIncompatibility, match=fragment):
        _dml(_plr_data(), **kw)


def test_dml_sklearn_splits_reproduce_the_equivalent_fold_labels():
    df = _plr_data()
    n = len(df)
    labels = np.arange(n) % 3
    splits = [
        (np.flatnonzero(labels != k), np.flatnonzero(labels == k)) for k in (0, 1, 2)
    ]
    by_label = _dml(df, n_folds=3, fold_indices=labels)
    by_split = _dml(df, n_folds=3, fold_indices=splits)
    assert by_split.estimate == pytest.approx(by_label.estimate, rel=0, abs=1e-12)
    assert by_split.se == pytest.approx(by_label.se, rel=0, abs=1e-12)


def test_dml_rejects_splits_with_test_rows_out_of_range():
    df = _plr_data()
    n = len(df)
    half = n // 2
    splits = [
        (np.arange(half, n), np.arange(half)),
        (np.arange(half), np.arange(half, n + 1)),
    ]
    with pytest.raises(MethodIncompatibility, match="split 1 has test rows outside"):
        _dml(df, n_folds=2, fold_indices=splits)


def test_dml_rejects_splits_whose_test_sets_miss_rows():
    df = _plr_data()
    n = len(df)
    labels = np.arange(n) % 3
    # Only two of the three complement splits: fold 2 is never held out.
    splits = [
        (np.flatnonzero(labels != k), np.flatnonzero(labels == k)) for k in (0, 1)
    ]
    with pytest.raises(MethodIncompatibility, match=r"40 row\(s\) never held out"):
        _dml(df, n_folds=2, fold_indices=splits)


def test_dml_non_numeric_pairs_are_not_read_as_splits():
    # Two pairs of labels that cannot be row numbers: not a split, and not
    # one label per row either, so the length check rejects it.
    df = _plr_data()
    with pytest.raises(MethodIncompatibility, match="must be 1-D of length 120"):
        _dml(df, n_folds=2, fold_indices=[(["a"], ["b"]), (["c"], ["d"])])


def test_dml_n_rows_of_two_labels_are_not_read_as_splits():
    df = _plr_data(n=6)
    pairs = [(i, (i + 1) % 6) for i in range(6)]
    with pytest.raises(MethodIncompatibility, match=r"got shape \(6, 2\)"):
        _dml(df, n_folds=2, fold_indices=pairs)


def test_dml_fold_column_with_missing_labels_is_rejected_or_dropped_loudly():
    labels = np.array([0.0, 1.0, np.nan, 1.0, 0.0])
    with pytest.raises(MethodIncompatibility, match="contain missing values"):
        _DoubleMLBase._validate_fold_indices(labels, 5, 2)
    with pytest.raises(MethodIncompatibility, match="must be length 5"):
        _DoubleMLBase._validate_fold_indices(labels[:4], 5, 2)
    codes = _DoubleMLBase._validate_fold_indices(np.array(list("abbab")), 5, 2)
    np.testing.assert_array_equal(codes, [0, 1, 1, 0, 1])


def test_dml_classifier_fitted_on_one_class_predicts_zero_probability():
    X = np.arange(8, dtype=float).reshape(-1, 1)
    target = np.array([0, 0, 0, 0, 1, 1, 0, 0])
    fitted = DecisionTreeClassifier(random_state=0).fit(X[:4], target[:4])
    assert list(fitted.classes_) == [0]
    out = _DoubleMLBase._predict_nuisance(fitted, X[4:], target, "ml_m")
    np.testing.assert_array_equal(out, np.zeros(4))


def test_dml_class_rejects_malformed_column_lists():
    # sp.dml normalises `covariates` before the class sees it; the class
    # is also public (sp.DoubleMLPLR) and must hold the same contract.
    df = _plr_data()
    with pytest.raises(MethodIncompatibility, match="column name or a list"):
        sp.DoubleMLPLR(df, y="y", treat="d", covariates=5)
    with pytest.raises(MethodIncompatibility, match="only column-name strings"):
        sp.DoubleMLPLR(df, y="y", treat="d", covariates=["x0", 1])
    with pytest.raises(MethodIncompatibility, match="instrument must contain only"):
        sp.DoubleMLPLR(df, y="y", treat="d", covariates=XS, instrument=["x0", 1])


def test_dml_base_class_guards_models_without_scores_or_fold_support():
    # The abstract base declares no score and is not in the fold-aware
    # list: a future model class that forgets either must fail loudly
    # instead of silently ignoring the caller's score / folds.
    df = _plr_data()
    kw = dict(y="y", treat="d", covariates=XS)
    with pytest.raises(MethodIncompatibility, match="does not accept"):
        _DoubleMLBase(df, score="ATE", **kw)
    with pytest.raises(
        MethodIncompatibility, match="explicit fold_indices are supported"
    ):
        _DoubleMLBase(df, fold_indices=np.arange(len(df)) % 2, **kw)


def test_dml_with_no_complete_rows_raises_data_insufficient():
    df = _plr_data()
    df["x1"] = np.nan
    with pytest.raises(DataInsufficient, match="no complete rows"):
        _dml(df)


# ---------------------------------------------------------------------
# sp.dml_did
# ---------------------------------------------------------------------


def _did_data(n=240, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    d = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x[:, 0])))
    dy = 1.0 + x[:, 0] + 2.0 * d + rng.normal(size=n)
    return pd.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "d": d, "dy": dy})


def _did_fit(df, **kw):
    args = dict(y="dy", treat="d", covariates=XS, ml_g=LinearRegression(), n_folds=3)
    args.setdefault("ml_m", LogisticRegression())
    args.update(kw)
    return sp.dml_did(df, **args)


@pytest.mark.parametrize(
    "kw, fragment",
    [
        (dict(n_folds=1), "n_folds must be an integer >= 2"),
        (dict(n_rep=0), "n_rep must be a positive integer"),
        (dict(trimming_threshold=0.5), r"trimming_threshold must lie in \[0, 0.5\)"),
        (dict(alpha=1.0), r"alpha must lie in \(0, 1\)"),
        (dict(id="unit"), "id= needs time="),
        (dict(covariates=["x0", "nope"]), r"column\(s\) not in data: \['nope'\]"),
        (dict(fold_indices="fold"), "fold_indices column 'fold' not in data"),
        (dict(fold_indices=np.zeros(5)), "one label per analysis row"),
        (dict(fold_indices=np.arange(240) % 2), "define 2 folds"),
        (dict(ml_m=LinearSVC()), "LinearSVC has no predict_proba"),
    ],
)
def test_dml_did_rejects_malformed_arguments(kw, fragment):
    with pytest.raises(MethodIncompatibility, match=fragment):
        _did_fit(_did_data(), **kw)


def test_dml_did_rejects_missing_fold_labels():
    df = _did_data()
    df["fold"] = (np.arange(len(df)) % 3).astype(float)
    labels = df["fold"].to_numpy().copy()
    labels[0] = np.nan
    with pytest.raises(MethodIncompatibility, match="contain missing values"):
        _did_fit(df, fold_indices=labels)


def test_dml_did_explicit_fold_array_equals_fold_column():
    df = _did_data()
    df["fold"] = np.arange(len(df)) % 3
    by_col = _did_fit(df, fold_indices="fold")
    by_arr = _did_fit(df, fold_indices=df["fold"].to_numpy())
    assert by_arr.estimate == by_col.estimate
    assert by_arr.se == by_col.se
    assert abs(by_col.estimate - 2.0) < 4 * by_col.se


def test_dml_did_default_learners_recover_the_effect():
    df = _did_data(n=400, seed=3)
    fit = sp.dml_did(df, "dy", "d", XS, n_folds=3)
    assert abs(fit.estimate - 2.0) < 4 * fit.se
    assert 0 < fit.se < 0.5
    assert np.isfinite(fit.model_info["logloss_m"])


def test_dml_did_warns_about_rows_with_missing_values():
    df = _did_data()
    df.loc[[0, 1, 2], "x1"] = np.nan
    with pytest.warns(RuntimeWarning, match=r"dropped 3 row\(s\) with missing"):
        fit = _did_fit(df)
    assert fit.n_obs == len(df) - 3


def _long_panel(n=150, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x)))
    y0 = x + rng.normal(size=n)
    y1 = y0 + 1.0 + 0.5 * x + 2.0 * d + rng.normal(size=n)
    unit = np.arange(n)
    pre = pd.DataFrame({"unit": unit, "t": 0, "x0": x, "x1": x**2, "d": d, "y": y0})
    post = pre.assign(t=1, y=y1)
    return pd.concat([pre, post], ignore_index=True)


def _did_long(df, **kw):
    args = dict(
        y="y",
        treat="d",
        covariates=XS,
        time="t",
        id="unit",
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(),
        n_folds=3,
    )
    args.update(kw)
    return sp.dml_did(df, **args)


def test_dml_did_rejects_duplicate_unit_period_rows():
    df = _long_panel()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(MethodIncompatibility, match=r"duplicate \(id, time\) rows"):
        _did_long(df)


def test_dml_did_drops_units_seen_in_one_period_with_a_warning():
    df = _long_panel()
    full = _did_long(df[~df["unit"].isin([0, 1])])
    incomplete = df[~((df["unit"].isin([0, 1])) & (df["t"] == 1))]
    with pytest.warns(RuntimeWarning, match=r"dropped 2 unit\(s\) not observed"):
        fit = _did_long(incomplete)
    # Dropping the two units up front gives the same estimate.
    assert fit.estimate == pytest.approx(full.estimate, rel=0, abs=1e-12)
    assert fit.n_obs == full.n_obs


def test_dml_did_rejects_non_numeric_covariates():
    df = _did_data()
    df["x1"] = np.where(df["x1"] > 0, "hi", "lo")
    with pytest.raises(MethodIncompatibility, match="covariates must be numeric"):
        _did_fit(df)


def test_dml_did_needs_n_folds_units_in_each_group():
    df = _did_data()
    df["d"] = 0
    df.loc[[0, 1], "d"] = 1
    with pytest.raises(DataInsufficient, match="2 treated and 238 comparison"):
        _did_fit(df)


def test_dml_did_repeated_cross_sections_need_n_folds_rows_per_cell():
    rng = np.random.default_rng(5)
    n = 80
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "d": np.tile([0, 1], n // 2),
            "t": 1,
            "y": rng.normal(size=n),
        }
    )
    # Only two treated and two comparison rows in the pre-period.
    df.loc[[0, 1, 2, 3], "t"] = 0
    with pytest.raises(DataInsufficient, match="smallest group x period cell has 2"):
        sp.dml_did(
            df,
            "y",
            "d",
            XS,
            time="t",
            ml_g=LinearRegression(),
            ml_m=LogisticRegression(),
            n_folds=3,
        )


def test_dml_did_rejects_a_training_fold_with_one_group():
    df = _did_data()
    d = df["d"].to_numpy()
    # Fold 0 holds every treated unit, so its complement has none.
    folds = np.where(d == 1, 0, 1 + np.arange(len(df)) % 2)
    with pytest.raises(DataInsufficient, match="training fold holds one group"):
        _did_fit(df, fold_indices=folds)


def test_dml_did_rejects_a_single_row_training_fold():
    df = _did_data(n=40)
    folds = np.zeros(len(df), dtype=int)
    folds[0] = 1  # the complement of fold 0 is one row
    with pytest.raises(DataInsufficient, match=r"training fold has [01] row"):
        _did_fit(df, n_folds=2, score="experimental", fold_indices=folds)


def test_dml_did_untrimmed_propensity_of_one_is_rejected():
    df = _did_data()
    with pytest.raises(DataInsufficient, match="propensity score equals 1"):
        _did_fit(
            df, ml_m=DecisionTreeClassifier(random_state=0), trimming_threshold=0.0
        )


def test_dml_did_predict_mean_is_zero_when_class_one_was_never_seen():
    X = np.arange(6, dtype=float).reshape(-1, 1)
    fitted = DecisionTreeClassifier(random_state=0).fit(X[:3], np.zeros(3, dtype=int))
    np.testing.assert_array_equal(_did._predict_mean(fitted, X[3:]), np.zeros(3))


# ---------------------------------------------------------------------
# sp.dml_panel
# ---------------------------------------------------------------------


def _panel(n_units=30, n_t=4, seed=6):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), n_t)
    time = np.tile(np.arange(n_t), n_units)
    x = rng.normal(size=n_units * n_t)
    d = 0.5 * x + rng.normal(size=n_units * n_t)
    y = 1.0 * d + x + np.repeat(rng.normal(size=n_units), n_t)
    y = y + rng.normal(size=n_units * n_t)
    return pd.DataFrame({"unit": unit, "time": time, "x0": x, "d": d, "y": y})


def _panel_fit(df, **kw):
    args = dict(
        y="y",
        treat="d",
        covariates=["x0"],
        unit="unit",
        ml_g=LinearRegression(),
        ml_m=LinearRegression(),
        n_folds=3,
    )
    args.update(kw)
    return sp.dml_panel(df, **args)


@pytest.mark.parametrize(
    "kw, fragment",
    [
        (dict(covariates=5), "`covariates` must be a column name or a sequence"),
        (dict(covariates=["x0", 3]), "`covariates` must contain non-empty"),
        (dict(y=None), "`y` must be a non-empty column-name string"),
        (dict(alpha=1.5), "alpha must be between 0 and 1"),
        (dict(include_time_fe="yes"), "include_time_fe must be True or False"),
        (dict(binary_treatment=1), "binary_treatment must be True or False"),
        (dict(fold_indices="fold"), "fold_indices column 'fold' not in data"),
        (dict(fold_indices=np.zeros(7)), "fold_indices must be 1-D of length 120"),
        (dict(fold_indices=np.zeros(120)), "must define at least two folds"),
    ],
)
def test_dml_panel_rejects_malformed_arguments(kw, fragment):
    with pytest.raises(MethodIncompatibility, match=fragment):
        _panel_fit(_panel(), **kw)


def test_dml_panel_rejects_non_dataframe():
    with pytest.raises(MethodIncompatibility, match="`data` must be a pandas"):
        _panel_fit(_panel().to_numpy())


def test_dml_panel_with_no_complete_rows_raises():
    df = _panel()
    df["x0"] = np.nan
    with pytest.raises(DataInsufficient, match="No complete observations remain"):
        _panel_fit(df)


@pytest.mark.parametrize(
    "column, fragment",
    [
        ("y", "Outcome contains non-finite values"),
        ("d", "Treatment contains non-finite values"),
        ("x0", "Covariates contain non-finite values"),
    ],
)
def test_dml_panel_rejects_infinite_values(column, fragment):
    df = _panel()
    df.loc[3, column] = np.inf
    with pytest.raises(DataInsufficient, match=fragment):
        _panel_fit(df)


# ---------------------------------------------------------------------
# sp.dynamic_dml
# ---------------------------------------------------------------------


def _dyn_panel(n=200, seed=7):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    t0 = 0.5 * x0 + rng.normal(size=n)
    x1 = 0.5 * x0 + 0.3 * t0 + rng.normal(size=n)
    t1 = 0.5 * x1 + 0.2 * t0 + rng.normal(size=n)
    y = 1.0 * t1 + 0.5 * t0 + x1 + rng.normal(size=n)
    unit = np.arange(n)
    first = pd.DataFrame({"unit": unit, "t": 0, "x": x0, "d": t0, "y": 0.0, "z": x0})
    second = pd.DataFrame({"unit": unit, "t": 1, "x": x1, "d": t1, "y": y, "z": x0})
    return pd.concat([first, second], ignore_index=True)


def _dyn(df, **kw):
    args = dict(y="y", treat="d", id="unit", time="t", covariates="x", n_folds=3)
    args.update(kw)
    return sp.dynamic_dml(df, **args)


def test_dynamic_dml_string_covariate_equals_one_element_list():
    df = _dyn_panel()
    as_str = _dyn(df)  # default RidgeCV learner, covariates="x"
    as_list = _dyn(df, covariates=["x"])
    assert as_str.estimate == as_list.estimate
    assert as_str.se == as_list.se
    assert abs(as_str.estimate - 1.5) < 5 * as_str.se


def test_dynamic_dml_summary_reports_the_modifier_projection():
    fit = _dyn(_dyn_panel(), modifiers=["z"])
    text = fit.summary()
    assert "Heterogeneity (linear projection on modifiers):" in text
    assert fit.coef is not None
    assert fit.coef.round(4).to_string() in text


@pytest.mark.parametrize(
    "kw, fragment",
    [
        (dict(covariates=["x", 2]), "covariates must be a column name or a list"),
        (dict(alpha=0.0), r"alpha must be in \(0, 1\)"),
        (dict(fold_ids=np.arange(7) % 2), r"one entry per complete unit \(200\)"),
    ],
)
def test_dynamic_dml_rejects_malformed_arguments(kw, fragment):
    with pytest.raises(MethodIncompatibility, match=fragment):
        _dyn(_dyn_panel(), **kw)


def test_dynamic_dml_every_unit_missing_somewhere_raises():
    df = _dyn_panel()
    df.loc[df["t"] == 1, "x"] = np.nan
    with pytest.raises(DataInsufficient, match="every unit has a missing value"):
        _dyn(df)


def test_dynamic_dml_singular_moment_system_is_reported():
    df = _dyn_panel()
    df["d"] = 2.0 * df["x"]  # the state predicts the treatment exactly
    with pytest.raises(DataInsufficient, match="moment system is singular"):
        _dyn(df, model_y=LinearRegression(), model_t=LinearRegression())


# ---------------------------------------------------------------------
# sp.dml_model_averaging
# ---------------------------------------------------------------------


def _candidates():
    from sklearn.linear_model import Ridge

    return [
        (LinearRegression(), LinearRegression(), "ols"),
        (Ridge(alpha=5.0), Ridge(alpha=5.0), "ridge"),
    ]


def _avg(df, **kw):
    args = dict(y="y", treat="d", covariates=XS, candidates=_candidates(), n_folds=3)
    args.update(kw)
    return sp.dml_model_averaging(df, **args)


@pytest.mark.parametrize(
    "kw, fragment",
    [
        (dict(y=""), "`y` must be a non-empty column name"),
        (dict(covariates=5), "`covariates` must be a column name or list"),
        (dict(covariates=["x0", 1]), "must contain only non-empty column names"),
        (dict(weight_rule=3), "`weight_rule` must be a string option"),
        (dict(weight_rule="  "), "`weight_rule` must be a non-empty string"),
        (dict(n_folds=True), "`n_folds` must be an integer >= 2"),
        (dict(n_folds="three"), "`n_folds` must be an integer >= 2"),
        (dict(alpha=True), r"`alpha` must be a number in \(0, 1\)"),
        (dict(alpha="five"), r"`alpha` must be a number in \(0, 1\)"),
        (dict(fold_indices=np.zeros(7)), "fold_indices must be 1-D of length 120"),
        (dict(fold_indices=np.zeros(120)), "must define at least two folds"),
    ],
)
def test_model_averaging_rejects_malformed_arguments(kw, fragment):
    with pytest.raises(MethodIncompatibility, match=fragment):
        _avg(_plr_data(), **kw)


def test_model_averaging_rejects_non_dataframe_and_empty_frame():
    df = _plr_data()
    with pytest.raises(MethodIncompatibility, match="`data` must be a pandas"):
        _avg(df.to_numpy())
    with pytest.raises(DataInsufficient, match="`data` is empty"):
        _avg(df.iloc[:0])


def test_model_averaging_fold_array_equals_fold_column():
    df = _plr_data()
    df["fold"] = np.arange(len(df)) % 3
    by_col = _avg(df, fold_indices="fold")
    by_arr = _avg(df, fold_indices=df["fold"].to_numpy())
    assert by_arr.estimate == by_col.estimate
    assert by_arr.se == by_col.se
    assert abs(by_col.estimate - 1.5) < 4 * by_col.se


def test_cls_weights_reject_malformed_inputs():
    target = np.arange(5.0)
    preds = np.column_stack([target, target[::-1]])
    with pytest.raises(MethodIncompatibility, match="`target` must be numeric"):
        _ma._solve_cls_weights(np.array(list("abcde")), preds)
    with pytest.raises(MethodIncompatibility, match="non-finite values"):
        _ma._solve_cls_weights(np.array([0.0, 1.0, np.nan, 3.0, 4.0]), preds)
    with pytest.raises(MethodIncompatibility, match=r"shape \(n, K\) with K > 0"):
        _ma._solve_cls_weights(target, preds[:4])
    with pytest.raises(MethodIncompatibility, match="must match target length"):
        _ma._solve_cls_weights(target, preds, sample_weight=np.ones(4))


def test_cls_weights_beyond_exact_limit_match_the_exact_solution():
    # K = 13 is past the support-enumeration limit, so the SLSQP path
    # runs. The target is a convex combination of two candidates, so the
    # constrained minimum is known exactly.
    rng = np.random.default_rng(8)
    preds = rng.normal(size=(200, 13))
    truth = np.zeros(13)
    truth[2], truth[9] = 0.3, 0.7
    target = preds @ truth
    assert _ma._cls_exact(target, preds, np.ones(200)) is None
    w = _ma._solve_cls_weights(target, preds)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(w >= 0)
    np.testing.assert_allclose(w, truth, rtol=0, atol=1e-5)


def test_cls_weights_fall_back_to_single_best_when_the_solver_fails():
    # One candidate is so badly scaled that the squared loss overflows:
    # the solver cannot converge and the best single candidate is used.
    rng = np.random.default_rng(9)
    target = rng.normal(size=50)
    preds = rng.normal(size=(50, 13))
    preds[:, 4] = target + 0.01 * rng.normal(size=50)
    preds[:, 0] = 1e170
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = _ma._solve_cls_weights(target, preds)
    messages = [str(c.message) for c in caught if c.category is RuntimeWarning]
    assert any("did not converge" in m and "single best" in m for m in messages)
    expected = np.zeros(13)
    expected[4] = 1.0
    np.testing.assert_array_equal(w, expected)
