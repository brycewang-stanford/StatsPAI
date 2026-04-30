"""String-alias resolver for DML nuisance learners.

Users frequently pass ``ml_g='rf'`` / ``ml_m='gbm'`` instead of constructing
a scikit-learn estimator. The underlying estimator classes call
``sklearn.base.clone`` on the learner and crash with a cryptic
``TypeError: Cannot clone object 'rf' (type str) ...`` message.

This module turns common string aliases into properly configured
scikit-learn estimators, dispatched on whether the role is a
regressor or a classifier.

Aliases (case-insensitive)
--------------------------
* ``'rf'``, ``'random_forest'``       → RandomForest{Regressor,Classifier}
* ``'gbm'``, ``'gbr'``, ``'gbc'``,
  ``'gboost'``, ``'gradient_boosting'`` → GradientBoosting{Regressor,Classifier}
* ``'lasso'``                         → LassoCV / LogisticRegressionCV(L1)
* ``'ridge'``                         → RidgeCV  / LogisticRegressionCV(L2)
* ``'linear'``, ``'ols'``             → LinearRegression / LogisticRegression
* ``'logistic'``, ``'logit'``         → classifier only (LogisticRegressionCV)
* ``'xgb'``, ``'xgboost'``            → xgboost.XGB{Regressor,Classifier} (optional)
* ``'lgbm'``, ``'lightgbm'``          → lightgbm.LGBM{Regressor,Classifier} (optional)

Anything that already looks like a scikit-learn estimator (i.e. exposes
``get_params`` and ``fit``) is returned unchanged.
"""

from __future__ import annotations

from typing import Any


_REGRESSOR_ALIASES = {
    "rf", "random_forest", "randomforest",
    "gbm", "gbr", "gboost", "gradient_boosting", "gradientboosting",
    "lasso", "ridge", "linear", "ols",
    "xgb", "xgboost",
    "lgbm", "lightgbm",
}
_CLASSIFIER_ALIASES = {
    "rf", "random_forest", "randomforest",
    "gbm", "gbc", "gboost", "gradient_boosting", "gradientboosting",
    "lasso", "ridge", "linear", "logistic", "logit",
    "xgb", "xgboost",
    "lgbm", "lightgbm",
}


def _is_estimator_like(obj: Any) -> bool:
    """Duck-typing check: scikit-learn-compatible estimator."""
    return hasattr(obj, "fit") and hasattr(obj, "get_params")


def _build_regressor(alias: str) -> Any:
    a = alias.lower()
    if a in {"rf", "random_forest", "randomforest"}:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=1,
        )
    if a in {"gbm", "gbr", "gboost", "gradient_boosting", "gradientboosting"}:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
        )
    if a == "lasso":
        from sklearn.linear_model import LassoCV
        return LassoCV(cv=5, random_state=42)
    if a == "ridge":
        from sklearn.linear_model import RidgeCV
        return RidgeCV()
    if a in {"linear", "ols"}:
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if a in {"xgb", "xgboost"}:
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError(
                "ml_g/ml_m='xgb' requires the optional 'xgboost' package — "
                "install it via `pip install xgboost`."
            ) from e
        return XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            random_state=42, n_jobs=1, verbosity=0,
        )
    if a in {"lgbm", "lightgbm"}:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ImportError(
                "ml_g/ml_m='lgbm' requires the optional 'lightgbm' package — "
                "install it via `pip install lightgbm`."
            ) from e
        return LGBMRegressor(
            n_estimators=200, max_depth=-1, learning_rate=0.1,
            random_state=42, n_jobs=1, verbosity=-1,
        )
    raise ValueError(_alias_error_message(alias, kind="regressor"))


def _build_classifier(alias: str) -> Any:
    a = alias.lower()
    if a in {"rf", "random_forest", "randomforest"}:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=1,
        )
    if a in {"gbm", "gbc", "gboost", "gradient_boosting", "gradientboosting"}:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
        )
    if a == "lasso":
        from sklearn.linear_model import LogisticRegressionCV
        return LogisticRegressionCV(
            cv=5, penalty="l1", solver="liblinear",
            max_iter=2000, random_state=42,
        )
    if a == "ridge":
        from sklearn.linear_model import LogisticRegressionCV
        return LogisticRegressionCV(
            cv=5, penalty="l2", max_iter=2000, random_state=42,
        )
    if a in {"linear", "logistic", "logit"}:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, random_state=42)
    if a == "ols":
        # OLS-as-classifier doesn't make sense — surface a clear error
        # rather than silently substituting LogisticRegression.
        raise ValueError(
            "ml_m='ols' is not valid for a classifier role (binary "
            "treatment / instrument). Use 'logistic' / 'linear' / 'rf' / "
            "'gbm' / 'lasso' / 'ridge' instead."
        )
    if a in {"xgb", "xgboost"}:
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "ml_m='xgb' requires the optional 'xgboost' package — "
                "install it via `pip install xgboost`."
            ) from e
        return XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            random_state=42, n_jobs=1, verbosity=0,
            use_label_encoder=False, eval_metric="logloss",
        )
    if a in {"lgbm", "lightgbm"}:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise ImportError(
                "ml_m='lgbm' requires the optional 'lightgbm' package — "
                "install it via `pip install lightgbm`."
            ) from e
        return LGBMClassifier(
            n_estimators=200, max_depth=-1, learning_rate=0.1,
            random_state=42, n_jobs=1, verbosity=-1,
        )
    raise ValueError(_alias_error_message(alias, kind="classifier"))


def _alias_error_message(alias: str, *, kind: str) -> str:
    universe = _REGRESSOR_ALIASES if kind == "regressor" else _CLASSIFIER_ALIASES
    return (
        f"Unknown {kind} alias {alias!r}. "
        f"Supported string aliases: {sorted(universe)}. "
        f"Or pass any scikit-learn-compatible estimator object."
    )


def resolve_learner(
    spec: Any,
    *,
    kind: str,
    role: str,
) -> Any:
    """Turn ``spec`` into a scikit-learn estimator.

    Parameters
    ----------
    spec : str | estimator | None
        User-supplied learner. ``None`` is a sentinel meaning "use the
        caller's default" — the caller handles that case before this
        function is invoked.
    kind : {'regressor', 'classifier'}
        Required output type. The PLR ml_g is always 'regressor'; IRM
        ml_m is 'classifier'; etc.
    role : str
        Argument name (``'ml_g'`` / ``'ml_m'`` / ``'ml_r'``). Used in
        error messages so the user knows which argument to fix.
    """
    if spec is None:
        raise ValueError(
            f"resolve_learner({role}=None): callers must supply their "
            f"own default before reaching the resolver."
        )
    if isinstance(spec, str):
        if kind == "regressor":
            return _build_regressor(spec)
        if kind == "classifier":
            return _build_classifier(spec)
        raise ValueError(
            f"resolve_learner: kind must be 'regressor' or 'classifier', "
            f"got {kind!r}."
        )
    if _is_estimator_like(spec):
        return spec
    raise TypeError(
        f"{role}={spec!r} (type {type(spec).__name__}) is not a "
        f"scikit-learn estimator and not a recognised string alias. "
        f"Pass either an estimator object (with .fit / .get_params) or "
        f"one of the supported aliases: "
        f"{sorted(_REGRESSOR_ALIASES if kind == 'regressor' else _CLASSIFIER_ALIASES)}."
    )
