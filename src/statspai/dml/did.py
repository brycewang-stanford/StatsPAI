"""Double machine learning for two-period difference-in-differences.

The ATT under conditional parallel trends, with the outcome-trend regression
and the propensity score fitted by any learner and cross-fitted. The score is
the doubly robust one of Sant'Anna and Zhao (2020); with
``in_sample_normalization=False`` it is the score of Chang (2020).

Two data layouts:

* panel: the outcome change of each unit is regressed out, one row per unit;
* repeated cross-sections: four outcome regressions, one per group x period
  cell, as in ``DoubleMLDIDCS``.

Given the same folds and learners the estimates and standard errors equal
those of the ``DoubleML`` Python package (``DoubleMLDID`` / ``DoubleMLDIDCS``)
to floating point (``tests/reference_parity/test_dml_did_doubleml_parity.py``).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._learners import resolve_learner

__all__ = ["dml_did"]

_SCORES = ("observational", "experimental")


def _default_regressor() -> Any:
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )


def _default_classifier() -> Any:
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )


def _predict_mean(fitted: Any, X: np.ndarray) -> np.ndarray:
    """Conditional mean from a fitted learner (class-1 probability for a
    classifier)."""
    from sklearn.base import is_classifier

    if is_classifier(fitted):
        if not hasattr(fitted, "predict_proba"):
            raise MethodIncompatibility(
                f"dml_did: the classifier {type(fitted).__name__} has no "
                "predict_proba.",
                recovery_hint="Pass a probabilistic classifier for ml_m.",
            )
        proba = np.asarray(fitted.predict_proba(X), dtype=float)
        classes = list(getattr(fitted, "classes_", (0, 1)))
        if 1 not in classes:
            return np.zeros(X.shape[0], dtype=float)
        return proba[:, classes.index(1)]
    return np.asarray(fitted.predict(X), dtype=float).ravel()


def _fit_predict(
    learner: Any,
    X: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    role: str,
) -> np.ndarray:
    from sklearn.base import clone

    if len(train) < 2:
        raise DataInsufficient(
            f"dml_did: a training fold has {len(train)} row(s) for the "
            f"{role} nuisance.",
            recovery_hint="Use fewer folds, or check that every group x "
            "period cell has enough observations.",
        )
    model = clone(learner)
    model.fit(X[train], target[train])
    return _predict_mean(model, X[test])


def _splits(
    n: int,
    strata: np.ndarray,
    n_folds: int,
    seed: int,
    fold_labels: Optional[np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if fold_labels is not None:
        return [
            (np.flatnonzero(fold_labels != k), np.flatnonzero(fold_labels == k))
            for k in range(n_folds)
        ]
    from sklearn.model_selection import StratifiedKFold

    counts = np.bincount(strata)
    if counts[counts > 0].min() < n_folds:
        raise DataInsufficient(
            f"dml_did: the smallest group x period cell has "
            f"{int(counts[counts > 0].min())} rows, fewer than "
            f"n_folds={n_folds}.",
            recovery_hint="Reduce n_folds.",
        )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(n), strata))


def _panel_scores(
    dy: np.ndarray,
    d: np.ndarray,
    g0: np.ndarray,
    g1: Optional[np.ndarray],
    m: Optional[np.ndarray],
    score: str,
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(psi_a, psi_b)`` with ``theta = -mean(psi_b) / mean(psi_a)``."""
    p = float(np.mean(d))
    resid0 = dy - g0
    if score == "observational":
        assert m is not None
        if normalize:
            w_a = d / p
            odds = (1.0 - d) * m / (1.0 - m)
            w_resid = d / p - odds / np.mean(odds)
        else:
            w_a = d / p
            w_resid = (d - m) / (p * (1.0 - m))
        psi_b = w_resid * resid0
    else:
        assert g1 is not None
        w_a = np.ones_like(dy)
        if normalize:
            w_resid = d / p - (1.0 - d) / np.mean(1.0 - d)
        else:
            w_resid = (d - p) / (p * (1.0 - p))
        psi_b = (d / p - 1.0) * g0 + (1.0 - d / p) * g1 + w_resid * resid0
    return -w_a, psi_b


def _rcs_scores(
    y: np.ndarray,
    d: np.ndarray,
    t: np.ndarray,
    g: Dict[Tuple[int, int], np.ndarray],
    m: Optional[np.ndarray],
    score: str,
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    p = float(np.mean(d))
    lam = float(np.mean(t))
    d1t1, d1t0 = d * t, d * (1.0 - t)
    d0t1, d0t0 = (1.0 - d) * t, (1.0 - d) * (1.0 - t)
    if score == "observational":
        assert m is not None
        w_a = d / p
        odds = m / (1.0 - m)
        if normalize:
            w11 = d1t1 / np.mean(d1t1)
            w10 = -d1t0 / np.mean(d1t0)
            w01 = -(d0t1 * odds) / np.mean(d0t1 * odds)
            w00 = (d0t0 * odds) / np.mean(d0t0 * odds)
        else:
            w11 = d1t1 / (p * lam)
            w10 = -d1t0 / (p * (1.0 - lam))
            w01 = -d0t1 / (p * lam) * odds
            w00 = d0t0 / (p * (1.0 - lam)) * odds
    else:
        w_a = np.ones_like(y)
        if normalize:
            w11 = d1t1 / np.mean(d1t1)
            w10 = -d1t0 / np.mean(d1t0)
            w01 = -d0t1 / np.mean(d0t1)
            w00 = d0t0 / np.mean(d0t0)
        else:
            w11 = d1t1 / (p * lam)
            w10 = -d1t0 / (p * (1.0 - lam))
            w01 = -d0t1 / ((1.0 - p) * lam)
            w00 = d0t0 / ((1.0 - p) * (1.0 - lam))
    psi_b = (
        w_a * (g[(1, 1)] - g[(1, 0)] - g[(0, 1)] + g[(0, 0)])
        + w11 * (y - g[(1, 1)])
        + w10 * (y - g[(1, 0)])
        + w01 * (y - g[(0, 1)])
        + w00 * (y - g[(0, 0)])
    )
    return -w_a, psi_b


def _binary(values: pd.Series, name: str) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isin(np.unique(arr[~np.isnan(arr)]), (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"dml_did: {name} must be coded 0/1.",
            recovery_hint=f"Recode {name} to a 0/1 indicator.",
        )
    return np.asarray(arr, dtype=float)


def _fold_labels(
    fold_indices: Any, n: int, n_folds: int, index: pd.Index, data: pd.DataFrame
) -> Optional[np.ndarray]:
    if fold_indices is None:
        return None
    if isinstance(fold_indices, str):
        if fold_indices not in data.columns:
            raise MethodIncompatibility(
                f"dml_did: fold_indices column {fold_indices!r} not in data."
            )
        raw = data.loc[index, fold_indices].to_numpy()
    else:
        raw = np.asarray(fold_indices)
        if raw.ndim != 1 or len(raw) != n:
            raise MethodIncompatibility(
                f"dml_did: fold_indices must have one label per analysis row "
                f"({n}); got shape {raw.shape}.",
                recovery_hint="On a long panel pass one label per unit, in "
                "the order of first appearance, or a column name that is "
                "constant within unit.",
            )
    codes, uniques = pd.factorize(raw, sort=True)
    if (codes < 0).any():
        raise MethodIncompatibility("dml_did: fold_indices contain missing values.")
    if len(uniques) != n_folds:
        raise MethodIncompatibility(
            f"dml_did: fold_indices define {len(uniques)} folds but "
            f"n_folds={n_folds}."
        )
    return np.asarray(codes, dtype=np.int64)


def dml_did(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str],
    *,
    time: Optional[str] = None,
    id: Optional[str] = None,
    ml_g: Any = None,
    ml_m: Any = None,
    score: str = "observational",
    in_sample_normalization: bool = True,
    n_folds: int = 5,
    n_rep: int = 1,
    trimming_threshold: float = 0.01,
    fold_indices: Any = None,
    random_state: int = 42,
    alpha: float = 0.05,
) -> CausalResult:
    """Double machine learning difference-in-differences (two periods).

    Estimates the average treatment effect on the treated under parallel
    trends that hold conditionally on covariates, with the nuisance
    functions fitted by machine learners and cross-fitted.

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
        Outcome. With ``time=None`` this is the **change** in the outcome
        between the two periods, one row per unit. Otherwise it is the
        outcome level.
    treat : str
        0/1 indicator of the treated group. On a long panel it may equal 1
        only in the post-period rows of treated units; a unit is treated
        if it is ever 1.
    covariates : sequence of str
        Pre-treatment covariates. On a long panel the first-period values
        are used.
    time : str, optional
        Period column with two values; the larger is the post-period.
        Omit it when ``y`` is already the outcome change.
    id : str, optional
        Unit identifier. With ``time`` and ``id`` the data are a
        two-period panel in long form and the outcome is differenced
        within unit. With ``time`` alone the rows are repeated
        cross-sections.
    ml_g : estimator or str, optional
        Regressor for the outcome nuisance: ``E[dY | X, D=0]`` on a panel,
        ``E[Y | X, D=d, T=t]`` in each of the four cells for repeated
        cross-sections. A scikit-learn estimator or an alias of
        :func:`statspai.dml` (``'lasso'``, ``'rf'``, ``'linear'`` ...).
        Default: gradient boosting, as in :func:`statspai.dml`.
    ml_m : estimator or str, optional
        Classifier for the propensity score ``P(D=1 | X)``. Not used with
        ``score='experimental'``.
    score : {'observational', 'experimental'}, default 'observational'
        ``'observational'`` is the doubly robust score for conditional
        parallel trends. ``'experimental'`` assumes treatment is
        independent of the covariates (a randomised design), uses no
        propensity score, and uses the covariates only to reduce variance.
    in_sample_normalization : bool, default True
        Normalise the weights to average one in the sample (Hajek-type,
        the ``DoubleML`` default). ``False`` gives the score of Chang
        (2020): ``(D - m(X)) / (p (1 - m(X))) * (dY - g0(X))``.
    n_folds : int, default 5
    n_rep : int, default 1
        Repetitions of the sample split; estimates are aggregated by the
        median, and the variance adds the spread across splits.
    trimming_threshold : float, default 0.01
        Propensity scores are clipped to ``[c, 1 - c]``. The number
        clipped is reported and a warning is issued when it is positive.
    fold_indices : array-like or str, optional
        One fold label per analysis row (per unit on a panel), or a column
        name. Requires ``n_rep=1``.
    random_state : int, default 42
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        ``estimate`` is the ATT. ``model_info`` holds the score, the
        nuisance fit (``rmse_g0``, ``logloss_m``), the propensity range,
        the number of clipped scores, the per-repetition estimates and
        ``_psi``, the influence values of the last repetition.

    Notes
    -----
    The estimand is the ATT in the post-period. The identifying assumption
    is that, given ``X``, the untreated outcome of the treated group would
    have changed as that of the comparison group did; ``X`` must be
    measured before treatment.

    The standard error is ``sqrt(mean(psi ** 2) / n)`` for the normalised
    influence function ``psi``. On a panel each row is one unit, so this
    is robust to serial correlation within unit.

    With a correctly specified parametric model the estimator is
    :func:`statspai.drdid`; this function is for when the outcome trend or
    the propensity score needs a flexible learner. Staggered adoption is
    :func:`statspai.callaway_santanna`.

    Cross-fitting removes overfitting bias, not the bias of a learner that
    cannot fit the nuisance at the sample size at hand. Compare
    ``model_info['rmse_g0']`` and ``model_info['logloss_m']`` across
    learners, as one would for any double machine learning estimate. On a
    design with a quadratic trend and propensity and 600 units, a random
    forest leaves the estimate 0.7 above a true effect of 2 (0.1 above at
    20,000 units), while a learner with quadratic terms is centred on it
    (``tests/reference_parity/test_dml_did_doubleml_parity.py``).

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> x = rng.normal(size=(n, 3))
    >>> d = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x[:, 0])))
    >>> dy = 1.0 + x[:, 0] ** 2 + 2.0 * d + rng.normal(size=n)
    >>> df = pd.DataFrame(x, columns=["x1", "x2", "x3"]).assign(d=d, dy=dy)
    >>> fit = sp.dml_did(df, "dy", "d", ["x1", "x2", "x3"])
    >>> bool(abs(fit.estimate - 2.0) < 0.3)
    True

    References
    ----------
    [@chang2020double],
    [@santanna2020doubly],
    [@chernozhukov2018double]
    """
    if score not in _SCORES:
        raise MethodIncompatibility(
            f"dml_did: score must be one of {_SCORES}, got {score!r}.",
            recovery_hint="Use score='observational'.",
        )
    if not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise MethodIncompatibility("dml_did: n_folds must be an integer >= 2.")
    if not isinstance(n_rep, (int, np.integer)) or n_rep < 1:
        raise MethodIncompatibility("dml_did: n_rep must be a positive integer.")
    if not 0.0 <= float(trimming_threshold) < 0.5:
        raise MethodIncompatibility("dml_did: trimming_threshold must lie in [0, 0.5).")
    if not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility("dml_did: alpha must lie in (0, 1).")
    if fold_indices is not None and n_rep != 1:
        raise MethodIncompatibility(
            "dml_did: explicit fold_indices require n_rep=1.",
            recovery_hint="Drop fold_indices or set n_rep=1.",
        )
    if id is not None and time is None:
        raise MethodIncompatibility(
            "dml_did: id= needs time=.",
            recovery_hint="Pass the period column, or pass one row per unit "
            "with y the outcome change.",
        )
    covariates = [covariates] if isinstance(covariates, str) else list(covariates)
    if not covariates:
        raise MethodIncompatibility(
            "dml_did: at least one covariate is required.",
            recovery_hint="Without covariates use sp.did.",
        )
    cols = [y, treat, *covariates] + [c for c in (time, id) if c is not None]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"dml_did: column(s) not in data: {missing}.",
            diagnostics={"missing_columns": missing},
        )
    df = data[list(dict.fromkeys(cols))].dropna()
    if len(df) < len(data):
        warnings.warn(
            f"dml_did: dropped {len(data) - len(df)} row(s) with missing values.",
            RuntimeWarning,
            stacklevel=2,
        )

    t_arr: Optional[np.ndarray] = None
    if time is None:
        layout = "panel"
        frame = df
        d = _binary(frame[treat], treat)
        y_arr = frame[y].to_numpy(dtype=float)
    else:
        periods = np.sort(df[time].unique())
        if len(periods) != 2:
            raise MethodIncompatibility(
                f"dml_did: time must take exactly two values, found "
                f"{len(periods)}.",
                recovery_hint="Keep one pre- and one post-period, or use "
                "sp.callaway_santanna for more periods.",
            )
        post = (df[time] == periods[1]).to_numpy()
        if id is not None:
            layout = "panel"
            if df.duplicated([id, time]).any():
                raise MethodIncompatibility(
                    "dml_did: duplicate (id, time) rows.",
                    recovery_hint="Keep one row per unit and period.",
                )
            counts = df.groupby(id, sort=False)[time].nunique()
            complete = counts.index[counts == 2]
            if len(complete) < len(counts):
                warnings.warn(
                    f"dml_did: dropped {len(counts) - len(complete)} unit(s) "
                    "not observed in both periods.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            df = df[df[id].isin(complete)]
            post = (df[time] == periods[1]).to_numpy()
            pre_rows = df[~post].set_index(id)
            post_rows = df[post].set_index(id).loc[pre_rows.index]
            d_unit = np.maximum(
                _binary(pre_rows[treat], treat), _binary(post_rows[treat], treat)
            )
            frame = pre_rows
            d = d_unit
            y_arr = post_rows[y].to_numpy(dtype=float) - pre_rows[y].to_numpy(
                dtype=float
            )
        else:
            layout = "repeated cross-sections"
            frame = df
            d = _binary(frame[treat], treat)
            t_arr = post.astype(float)
            y_arr = frame[y].to_numpy(dtype=float)
            if not np.isin([0.0, 1.0], np.unique(d[t_arr == 0])).all():
                raise DataInsufficient(
                    "dml_did: the pre-period holds no rows of one group. In "
                    "repeated cross-sections treat= must mark the treated "
                    "group in both periods.",
                    recovery_hint="Code treat as group membership, not as "
                    "group x post.",
                )
    try:
        X = frame[covariates].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "dml_did: covariates must be numeric.",
            recovery_hint="Encode categorical covariates as indicators.",
        ) from exc
    n = len(y_arr)
    n1, n0 = int(d.sum()), int(n - d.sum())
    if n1 < n_folds or n0 < n_folds:
        raise DataInsufficient(
            f"dml_did: {n1} treated and {n0} comparison units; each group "
            f"needs at least n_folds={n_folds}.",
            recovery_hint="Reduce n_folds or check the treat column.",
        )

    reg = (
        _default_regressor()
        if ml_g is None
        else resolve_learner(ml_g, kind="regressor", role="ml_g")
    )
    clf = None
    if score == "observational":
        clf = (
            _default_classifier()
            if ml_m is None
            else resolve_learner(ml_m, kind="classifier", role="ml_m")
        )
    labels = _fold_labels(fold_indices, n, int(n_folds), frame.index, data)
    strata = (d if t_arr is None else 2 * d + t_arr).astype(np.int64)

    thetas: List[float] = []
    variances: List[float] = []
    psi = np.zeros(n)
    info: Dict[str, Any] = {}
    c = float(trimming_threshold)
    for rep in range(int(n_rep)):
        splits = _splits(n, strata, int(n_folds), int(random_state) + rep, labels)
        m_hat: Optional[np.ndarray] = None
        if clf is not None:
            m_hat = np.empty(n)
            for train, test in splits:
                if len(np.unique(d[train])) < 2:
                    raise DataInsufficient(
                        "dml_did: a training fold holds one group only.",
                        recovery_hint="Supply folds stratified by treat.",
                    )
                m_hat[test] = _fit_predict(clf, X, d, train, test, "propensity")
            n_clip = int(np.sum((m_hat < c) | (m_hat > 1.0 - c)))
            raw_range = (float(m_hat.min()), float(m_hat.max()))
            if c > 0:
                m_hat = np.clip(m_hat, c, 1.0 - c)
            if np.any(m_hat >= 1.0):
                raise DataInsufficient(
                    "dml_did: a fitted propensity score equals 1; the "
                    "comparison group cannot be reweighted to it.",
                    recovery_hint="Use trimming_threshold > 0 or restrict "
                    "the sample to the region of overlap.",
                )
            info.update(n_propensity_clipped=n_clip, pscore_range=raw_range)
            eps = 1e-12
            info["logloss_m"] = float(
                -np.mean(
                    d * np.log(np.clip(m_hat, eps, 1))
                    + (1 - d) * np.log(np.clip(1 - m_hat, eps, 1))
                )
            )
        if t_arr is None:
            g0 = np.empty(n)
            g1 = np.empty(n) if score == "experimental" else None
            for train, test in splits:
                g0[test] = _fit_predict(
                    reg, X, y_arr, train[d[train] == 0], test, "outcome"
                )
                if g1 is not None:
                    g1[test] = _fit_predict(
                        reg, X, y_arr, train[d[train] == 1], test, "outcome"
                    )
            psi_a, psi_b = _panel_scores(
                y_arr, d, g0, g1, m_hat, score, bool(in_sample_normalization)
            )
            info["rmse_g0"] = float(np.sqrt(np.mean((y_arr - g0)[d == 0] ** 2)))
        else:
            g: Dict[Tuple[int, int], np.ndarray] = {}
            for dv in (0, 1):
                for tv in (0, 1):
                    cell = np.empty(n)
                    for train, test in splits:
                        rows = train[(d[train] == dv) & (t_arr[train] == tv)]
                        cell[test] = _fit_predict(reg, X, y_arr, rows, test, "outcome")
                    g[(dv, tv)] = cell
            psi_a, psi_b = _rcs_scores(
                y_arr, d, t_arr, g, m_hat, score, bool(in_sample_normalization)
            )
            fitted = sum(
                g[(dv, tv)] * (d == dv) * (t_arr == tv)
                for dv in (0, 1)
                for tv in (0, 1)
            )
            info["rmse_g"] = float(np.sqrt(np.mean((y_arr - fitted) ** 2)))
        j = float(np.mean(psi_a))
        theta = -float(np.mean(psi_b)) / j
        psi = -(psi_a * theta + psi_b) / j
        thetas.append(theta)
        variances.append(float(np.mean(psi**2)) / n)

    estimate = float(np.median(thetas))
    var = float(np.median(np.asarray(variances) + (np.asarray(thetas) - estimate) ** 2))
    se = float(np.sqrt(var))
    z = float(stats.norm.ppf(1 - alpha / 2))
    pvalue = float(2 * stats.norm.sf(abs(estimate / se))) if se > 0 else float("nan")
    if info.get("n_propensity_clipped"):
        warnings.warn(
            f"dml_did: {info['n_propensity_clipped']} of {n} propensity "
            f"scores lie outside [{c:g}, {1 - c:g}] and were clipped. "
            "Comparison units that look almost surely treated carry very "
            "large weights; check overlap.",
            RuntimeWarning,
            stacklevel=2,
        )
    model_info: Dict[str, Any] = {
        "estimator": "DML-DiD",
        "layout": layout,
        "score": score,
        "in_sample_normalization": bool(in_sample_normalization),
        "n_folds": int(n_folds),
        "n_rep": int(n_rep),
        "ml_g": type(reg).__name__,
        "ml_m": None if clf is None else type(clf).__name__,
        "default_learners": ml_g is None and ml_m is None,
        "trimming_threshold": c,
        "n_treated": n1,
        "n_control": n0,
        "treated_share": float(np.mean(d)),
        "post_share": None if t_arr is None else float(np.mean(t_arr)),
        "fold_source": "user" if labels is not None else "internal",
        "theta_reps": [float(v) for v in thetas],
        "se_reps": [float(np.sqrt(v)) for v in variances],
        "covariates": list(covariates),
        "_psi": psi,
        **info,
    }
    return CausalResult(
        method=f"Double ML DiD ({layout})",
        estimand="ATT",
        estimate=estimate,
        se=se,
        pvalue=pvalue,
        ci=(estimate - z * se, estimate + z * se),
        alpha=float(alpha),
        n_obs=n,
        detail=None,
        model_info=model_info,
        _citation_key="dml_did",
    )


CausalResult._CITATIONS["dml_did"] = (
    "@article{chang2020double,\n"
    "  title={Double/debiased machine learning for "
    "difference-in-differences models},\n"
    "  author={Chang, Neng-Chieh},\n"
    "  journal={The Econometrics Journal},\n"
    "  volume={23},\n"
    "  number={2},\n"
    "  pages={177--191},\n"
    "  year={2020},\n"
    "  doi={10.1093/ectj/utaa001}\n"
    "}"
)
