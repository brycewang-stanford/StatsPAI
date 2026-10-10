"""
GRF-style inference add-ons for the StatsPAI CausalForest.

Implements:

- :func:`calibration_test` (alias ``test_calibration``): the best linear
  predictor calibration test of CATE predictions [@chernozhukov2025generic],
  in the form of ``grf::test_calibration``.  Under correct average
  calibration the "mean forest prediction" coefficient is 1; under real
  heterogeneity the "differential forest prediction" coefficient is > 0.
- :func:`calibrate_cate`: CATE predictions rescaled by that calibration.
- :func:`rate`: rank-weighted average treatment effect (AUTOC / QINI)
  [@yadlowsky2025evaluating], ``grf::rank_average_treatment_effect``.
- :func:`honest_variance`: deprecated; see its docstring.
- :func:`average_treatment_effect`: GRF-style ATE/ATT/ATC/ATO aggregation
  of CATE predictions with effective sample size and normal CIs.
- :func:`forest_diagnostics`: overlap and CATE-distribution diagnostics.

Two layers
----------
The public functions above take a fitted :class:`CausalForest` object
(plus, for some, outcome / treatment / feature arrays), never mutate it,
and decide *which* forest outputs to use: out-of-bag CATE predictions,
observation weights and clusters for forests fitted with the GRF engine,
and the training rows only.

Underneath them sits a layer of pure *post-fit operators* -- deterministic
maps from forest outputs (``tau_hat``, the cross-fitted nuisances
``Y_hat = E[Y|X]`` and ``W_hat = E[W|X]``) and the data to a reported
number: :func:`grf_calibration`, :func:`rate_from_scores`,
:func:`aipw_scores`, :func:`grf_att_atc` and :func:`grf_overlap_ate`.  The
forest itself is stochastic and not pinnable across implementations; the
operators are, and each one is pinned to ``grf`` 2.6.1 fed the same forest
outputs (``tests/reference_parity/test_ml_causal_R_parity.py``,
``test_grf_aipw_operator_parity.py``, ``test_grf_cluster_operator_parity.py``).

References
----------
[@chernozhukov2025generic] -- best linear predictor calibration test
(NBER Working Paper 24678 in its 2018 version).

[@yadlowsky2025evaluating] -- rank-weighted average treatment effects.

[@athey2019generalized] -- generalized random forests.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)

if TYPE_CHECKING:
    from .causal_forest import CausalForest


def _require_fitted_forest(forest: "CausalForest", context: str) -> None:
    """Raise a StatsPAI taxonomy error when an inference helper is unfitted."""
    if not getattr(forest, "fitted_", False):
        raise MethodIncompatibility(
            f"{context} requires a fitted forest.",
            recovery_hint="Call fit() before running forest inference.",
        )


def _validate_alpha(alpha: float, context: str) -> float:
    """Validate a confidence/significance level."""
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: alpha must be a finite scalar.",
            recovery_hint="Use an alpha value in the open interval (0, 1).",
            diagnostics={"alpha": alpha},
        ) from exc
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise MethodIncompatibility(
            f"{context}: alpha must be in the open interval (0, 1).",
            recovery_hint="Use an alpha value such as 0.05.",
            diagnostics={"alpha": alpha},
        )
    return alpha_value


def _prepare_forest_features(
    forest: "CausalForest",
    X: Optional[np.ndarray],
    context: str,
) -> np.ndarray:
    """Validate inference features with the forest's fitted schema."""
    if X is None:
        return np.asarray(forest._X_original, dtype=np.float64)
    if callable(getattr(forest, "_prepare_effect_matrix", None)):
        return np.asarray(
            forest._prepare_effect_matrix(X, context=context),
            dtype=np.float64,
        )
    try:
        return np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: X must be numeric.",
            recovery_hint="Pass X shaped (n_samples, n_features).",
        ) from exc


def _prepare_forest_vector(
    values: Any,
    name: str,
    expected_rows: int,
    context: str,
) -> np.ndarray:
    """Validate a numeric inference vector aligned with X."""
    try:
        arr = np.asarray(values, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be numeric.",
            recovery_hint=f"Pass a numeric {name} vector aligned with X.",
        ) from exc
    if arr.shape[0] != expected_rows:
        raise MethodIncompatibility(
            f"{context}: {name} must have the same row count as X.",
            recovery_hint="Align X, Y, and T before running forest inference.",
            diagnostics={
                f"n_{name.lower()}": int(arr.shape[0]),
                "n_x": int(expected_rows),
            },
        )
    if expected_rows == 0:
        raise DataInsufficient(
            f"{context}: no rows were supplied.",
            recovery_hint="Pass at least one inference row.",
        )
    if not np.isfinite(arr).all():
        raise MethodIncompatibility(
            f"{context}: {name} contains NaN or infinite values.",
            recovery_hint=f"Drop or impute non-finite {name} rows.",
        )
    return arr


# ======================================================================
# Training-sample inputs
# ======================================================================


def _require_training_rows(
    forest: "CausalForest",
    X: Optional[np.ndarray],
    Y: Optional[np.ndarray],
    T: Optional[np.ndarray],
    context: str,
) -> None:
    """Refuse ``X`` / ``Y`` / ``T`` that are not the forest's training sample.

    The calibration test, RATE and the doubly-robust averages are defined
    on the rows whose out-of-bag predictions and cross-fitted nuisances the
    forest stored -- exactly as ``grf`` computes them from
    ``forest$predictions``, ``forest$Y.hat`` and ``forest$W.hat``.  Any
    other rows would be silently paired with another sample's nuisances.
    """
    for name, given, stored in (
        ("X", X, forest._X_original),
        ("Y", Y, forest._Y_original),
        ("T", T, forest._T_original),
    ):
        if given is None:
            continue
        arr = np.asarray(given, dtype=np.float64)
        ref = np.asarray(stored, dtype=np.float64)
        if name != "X":
            arr = arr.ravel()
            ref = ref.ravel()
        if arr.shape != ref.shape or not np.array_equal(arr, ref):
            raise MethodIncompatibility(
                f"{context}: {name} must be omitted or equal the forest's "
                "training sample. The test uses the out-of-bag predictions "
                "and cross-fitted nuisances (Y_hat, W_hat), which exist only "
                "for those rows.",
                recovery_hint=(
                    "Omit X / Y / T, or fit a separate forest on the evaluation "
                    "sample (rate() also accepts priorities= from another model)."
                ),
                diagnostics={
                    "argument": name,
                    "n_given": int(arr.shape[0]) if arr.ndim else 0,
                    "n_train": int(ref.shape[0]) if ref.ndim else 0,
                },
            )


def _stored_training_nuisances(
    forest: "CausalForest", context: str
) -> Tuple[np.ndarray, np.ndarray]:
    """``(Y_hat, W_hat)`` of the training sample; raise when absent."""
    m_raw = getattr(forest, "_m_insample", None)
    e_raw = getattr(forest, "_e_insample", None)
    n = int(np.asarray(forest._X_original).shape[0])
    if m_raw is None or e_raw is None:
        raise MethodIncompatibility(
            f"{context}: the forest carries no cross-fitted nuisance "
            "predictions (Y_hat, W_hat).",
            recovery_hint="Refit with sp.causal_forest(); it stores them.",
        )
    Y_hat = np.asarray(m_raw, dtype=np.float64).ravel()
    W_hat = np.asarray(e_raw, dtype=np.float64).ravel()
    if len(Y_hat) != n or len(W_hat) != n:
        raise MethodIncompatibility(
            f"{context}: the stored nuisances do not match the training sample.",
            recovery_hint="Refit the forest.",
            diagnostics={
                "n_train": n,
                "n_y_hat": int(len(Y_hat)),
                "n_w_hat": int(len(W_hat)),
            },
        )
    return Y_hat, W_hat


# ======================================================================
# Calibration test [@chernozhukov2025generic] (grf::test_calibration)
# ======================================================================


def grf_calibration(
    *,
    Y: np.ndarray,
    W: np.ndarray,
    Y_hat: np.ndarray,
    W_hat: np.ndarray,
    tau_hat: np.ndarray,
    vcov_type: str = "HC3",
    alpha: float = 0.05,
    weights: Optional[np.ndarray] = None,
    clusters: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    r"""The ``grf::test_calibration`` regression, given forest outputs.

    Fits, by (weighted) least squares without an intercept,

    .. math::
        Y_i - \hat m(X_i) = \beta_1\,(W_i - \hat e(X_i))\,\bar\tau
            + \beta_2\,(W_i - \hat e(X_i))\,(\hat\tau(X_i) - \bar\tau)
            + \varepsilon_i,

    with :math:`\bar\tau` the (observation-weighted) mean CATE prediction.
    ``beta_1 = 1`` says the mean prediction is calibrated; ``beta_2 = 1``
    says the heterogeneity is calibrated, and ``beta_2 > 0`` significantly
    is evidence that the forest found heterogeneity at all.

    ``t`` tests each coefficient against **0** and ``p`` is the
    **one-sided** p-value for the alternative ``beta > 0`` from Student's
    t with ``n - 2`` degrees of freedom -- ``grf``'s reporting rule
    (``lmtest::coeftest`` then halving). ``vcov_type`` is ``grf``'s
    ``vcov.type`` (default ``"HC3"``); ``grf`` passes it to
    ``sandwich::vcovCL`` with ``clusters`` (one cluster per observation
    when there are none), which is what the covariance here reproduces
    (:func:`statspai.forest._grf_inference.cluster_robust_vcov`).
    ``ci_low`` / ``ci_high`` use the same t quantile. ``t_vs_zero`` and
    ``p_one_sided`` repeat ``t`` and ``p`` under their earlier names.

    Returns
    -------
    DataFrame
        Index ``mean_forest_prediction`` / ``differential_forest_prediction``;
        columns ``coef``, ``se``, ``t``, ``p``, ``ci_low``, ``ci_high``,
        ``t_vs_zero``, ``p_one_sided``.
    """
    from ._grf_inference import cluster_robust_vcov

    vcov_key = str(vcov_type).upper()
    if vcov_key not in ("HC0", "HC1", "HC2", "HC3"):
        raise MethodIncompatibility(
            f"calibration_test(): vce must be HC0, HC1, HC2 or HC3; "
            f"got {vcov_type!r}.",
            recovery_hint="Use grf's default vce='HC3'.",
        )
    arrs = [
        np.asarray(a, dtype=np.float64).ravel() for a in (Y, W, Y_hat, W_hat, tau_hat)
    ]
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        raise MethodIncompatibility(
            "calibration_test(): Y, W, Y_hat, W_hat and tau_hat must be the "
            "same length.",
            recovery_hint="Pass aligned per-observation vectors.",
        )
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if len(w) != n:
        raise MethodIncompatibility(
            "calibration_test(): weights must have one entry per observation.",
            recovery_hint="Pass aligned per-observation weights.",
        )
    n_pos = int(np.sum(w > 0))
    if n_pos < 3:
        raise DataInsufficient(
            "calibration_test() requires at least 3 rows.",
            recovery_hint="Use a larger sample for the calibration regression.",
        )
    Y_, W_, Y_hat_, W_hat_, tau = arrs
    tau_bar = float(np.sum(w * tau) / np.sum(w))
    w_res = W_ - W_hat_
    target = Y_ - Y_hat_
    D = np.column_stack([w_res * tau_bar, w_res * (tau - tau_bar)])
    DtWD = D.T @ (D * w[:, None])
    if np.linalg.matrix_rank(DtWD) < 2:
        raise DataInsufficient(
            "calibration_test(): the calibration design is rank deficient "
            "(constant CATE predictions or a zero mean prediction), so the "
            "differential prediction is not identified.",
            recovery_hint="The forest found no heterogeneity to test.",
        )
    beta = np.linalg.solve(DtWD, D.T @ (w * target))
    resid = target - D @ beta
    V = cluster_robust_vcov(D, resid, weights=w, clusters=clusters, vcov_type=vcov_key)
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    df_resid = n_pos - 2
    p = stats.t.sf(t, df_resid)
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_resid))
    return pd.DataFrame(
        {
            "coef": beta,
            "se": se,
            "t": t,
            "p": p,
            "ci_low": beta - crit * se,
            "ci_high": beta + crit * se,
            "t_vs_zero": t,
            "p_one_sided": p,
        },
        index=["mean_forest_prediction", "differential_forest_prediction"],
    )


def _resolve_calibration_method(forest: "CausalForest", method: str) -> str:
    """``'imputation'`` or ``'within'`` for an FE forest, ``'grf'`` otherwise."""
    from . import _grf_inference as _gi

    if method not in ("auto", "imputation", "within"):
        raise MethodIncompatibility(
            f"method must be 'auto', 'imputation' or 'within', got {method!r}.",
            recovery_hint="Use method='auto'.",
        )
    if not _gi.is_fe_forest(forest):
        if method != "auto":
            raise MethodIncompatibility(
                f"method={method!r} applies to causal forests with fixed "
                "effects (fe=) only.",
                recovery_hint="Use method='auto' for other forests.",
            )
        return "grf"
    if method != "auto":
        return method
    binary = bool(np.all(np.isin(np.unique(forest._T_original), (0.0, 1.0))))
    # Imputation needs period ids (fe='twoway') and an untreated state.
    twoway = getattr(forest, "fe", None) == "twoway"
    return "imputation" if binary and twoway else "within"


@accepts_aliases(_strict=True, controls="covariates")
def calibration_test(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    vce: str = "HC3",
    method: str = "auto",
    covariates: Any = "none",
) -> pd.DataFrame:
    """Best-linear-predictor calibration test of CATEs [@chernozhukov2025generic].

    The form of ``grf::test_calibration``: the outcome residual
    ``Y - Y_hat`` is regressed (no intercept) on the treatment residual
    scaled by the mean forest prediction and on the treatment residual
    times the demeaned prediction,

        Y - Y_hat = b1 (W - W_hat) mean(tau_hat)
                    + b2 (W - W_hat) (tau_hat - mean(tau_hat)) + e.

    ``b1 = 1`` indicates a calibrated mean prediction; ``b2 = 1`` a
    calibrated heterogeneity signal, and a significantly positive ``b2``
    is evidence of heterogeneity -- the headline finding, showing the
    forest captures *real* heterogeneity rather than noise.  Following
    ``grf``, ``t`` tests each coefficient against 0 and ``p`` is
    **one-sided** (alternative > 0) from Student's t with ``n - 2``
    degrees of freedom; the default ``vce='HC3'`` is ``grf``'s.
    ``beta_differential`` is also the slope by which the predictions
    should be rescaled around their mean (see :func:`calibrate_cate`).

    For forests fitted with the GRF engine (the default) ``tau_hat`` is the
    **out-of-bag** prediction, the regression uses the forest's observation
    weights and a cluster-robust covariance when the forest has clusters,
    and the test is defined on the training sample, so ``X`` / ``Y`` / ``T``
    may be omitted (the stored arrays are used) and, if given, must equal
    them.  The regression itself is the pure operator
    :func:`grf_calibration`.

    Forests with two-way fixed effects (``fe='twoway'``) and a binary
    treatment use ``method='imputation'`` by default: the imputation score
    ``Y - alpha_hat_i - gamma_hat_t`` of each treated cell (unit and period
    effects fitted on untreated cells, [borusyak2024revisiting]) is
    regressed on ``mean(tau_oob)`` and ``tau_oob - mean(tau_oob)`` over the
    treated cells.  The score is unbiased for each cell's effect, so
    ``beta_differential`` is the best-linear-predictor slope of the true
    effect on the forest prediction (it corrects the forest's shrinkage)
    and the one-sided test of it is a heterogeneity test.  Standard errors
    use the exact linear weights of the imputation estimator, clustered by
    the forest's clusters, with a normal reference; ``vce`` does not apply.
    ``method='within'`` keeps the earlier regression on globally
    within-transformed variables [aytug2026attenuated], which tests for
    heterogeneity but whose slope is not a de-attenuation factor.

    .. versionchanged:: 1.30.0
       ``method=`` / ``covariates=`` added; forests with two-way fixed
       effects and a binary treatment default to the imputation regression.
       ``t`` / ``p`` are ``grf``'s: each coefficient against 0, one-sided,
       Student t (the ``null`` column is gone; ``t_vs_zero`` /
       ``p_one_sided`` are kept as synonyms).  Legacy-engine forests use
       ``grf``'s regression too: earlier releases regressed an
       inverse-propensity pseudo-outcome on ``[tau_hat, tau_hat -
       mean(tau_hat)]``, so ``differential_forest_prediction`` estimated
       ``b2 - b1`` rather than ``b2``, and paired rows other than the
       training sample with mean-of-Y / mean-of-T stand-in nuisances.

    Parameters
    ----------
    forest : fitted CausalForest
    X, Y, T : optional arrays
        The training sample (defaults to the stored arrays).
    alpha : float
        Level for ``ci_low`` / ``ci_high``.
    vce : {'HC3', 'HC2', 'HC1', 'HC0'}
        Heteroskedasticity-robust covariance (``grf``'s ``vcov.type``).
    method : {'auto', 'imputation', 'within'}, default 'auto'
        Forests with fixed effects only: ``'auto'`` is ``'imputation'`` for
        ``fe='twoway'`` with a binary treatment and ``'within'`` otherwise.
        Other forests accept only ``'auto'``.
    covariates : 'none', 'auto', list of str or array, default 'none'
        Covariates in the untreated outcome model of the imputation
        regression (see :func:`average_treatment_effect`).

    Returns
    -------
    DataFrame
        Rows ``mean_forest_prediction`` and ``differential_forest_prediction``
        with ``coef``, ``se``, ``t``, ``p``, ``ci_low``, ``ci_high``,
        ``t_vs_zero``, ``p_one_sided``.

    Examples
    --------
    ``sp.test_calibration`` is an alias of this function.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.binomial(1, 0.5, size=n)
    >>> tau = 1.0 + X[:, 0]  # heterogeneous effect
    >>> Y = X[:, 1] + tau * T + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({
    ...     "y": Y, "d": T,
    ...     "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
    ... })
    >>> cf = sp.causal_forest(
    ...     data=df, formula="y ~ d | x0 + x1 + x2",
    ...     n_estimators=50, random_state=0,
    ... )
    >>> ct = sp.calibration_test(cf)
    >>> ct.index.tolist()
    ['mean_forest_prediction', 'differential_forest_prediction']
    >>> sp.test_calibration is sp.calibration_test
    True

    References
    ----------
    [@chernozhukov2025generic], [@athey2019generalized]
    """
    _require_fitted_forest(forest, "calibration_test()")
    alpha_value = _validate_alpha(alpha, "calibration_test()")
    X_ = _prepare_forest_features(forest, X, "calibration_test()")
    Y_ = _prepare_forest_vector(
        Y if Y is not None else getattr(forest, "_Y_original", None),
        "Y",
        X_.shape[0],
        "calibration_test()",
    )
    T_ = _prepare_forest_vector(
        T if T is not None else getattr(forest, "_T_original", None),
        "T",
        X_.shape[0],
        "calibration_test()",
    )
    if X_.shape[0] < 3:
        raise DataInsufficient(
            "calibration_test() requires at least 3 rows.",
            recovery_hint="Use a larger sample for the calibration regression.",
        )
    _require_training_rows(forest, X, Y, T, "calibration_test()")
    from . import _grf_inference as _gi

    if _gi.is_grf_forest(forest):
        if _resolve_calibration_method(forest, method) == "imputation":
            from ._fe_imputation import calibration_fe

            return calibration_fe(forest, alpha=alpha_value, covariates=covariates)
        return _gi.calibration_blp(forest, alpha=alpha_value, vcov_type=vce)

    if method != "auto":
        raise MethodIncompatibility(
            f"calibration_test(): method={method!r} needs a GRF-engine forest.",
            recovery_hint="Refit with the default split_rule='grf'.",
        )
    Y_hat, W_hat = _stored_training_nuisances(forest, "calibration_test()")
    tau_hat = np.asarray(forest.effect(X_), dtype=np.float64).ravel()
    return grf_calibration(
        Y=Y_,
        W=T_,
        Y_hat=Y_hat,
        W_hat=W_hat,
        tau_hat=tau_hat,
        vcov_type=vce,
        alpha=alpha_value,
    )


# ======================================================================
# RATE [@yadlowsky2025evaluating] (grf::rank_average_treatment_effect)
# ======================================================================


def _priority_order(priorities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Descending priority order and tie-group codes, as ``grf`` forms them.

    ``grf`` converts priorities with ``as.integer(as.factor(x))``, and
    ``factor`` matches values through ``as.character`` -- 15 significant
    digits -- so values equal to 15 digits are one tie group. Returns the
    stable descending order and the per-unit group code.
    """
    keyed = np.array([float(f"{v:.15g}") for v in priorities], dtype=np.float64)
    _, codes = np.unique(keyed, return_inverse=True)
    order = np.argsort(-codes, kind="stable")
    return order, codes


def _tie_averaged_sorted(
    scores: np.ndarray, priorities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Scores sorted by descending priority, averaged within tie groups."""
    order, codes = _priority_order(priorities)
    sums = np.bincount(codes, weights=scores)
    counts = np.bincount(codes)
    avg = sums / counts
    return avg[codes[order]], order


def rate_rank_weights(priorities: np.ndarray, target: str) -> np.ndarray:
    r"""Weights ``a`` with ``RATE = a' scores``, given the prioritisation.

    Conditional on the ranking, AUTOC and QINI are *linear* in the scores:
    with :math:`\Gamma_{(j)}` the tie-averaged scores in descending priority
    order and :math:`m` their number,

    .. math::
        \mathrm{AUTOC} = \tfrac1m \sum_j \Gamma_{(j)}\,(H_m - H_{j-1} - 1),
        \qquad
        \mathrm{QINI} = \tfrac1m \sum_j \Gamma_{(j)}
                        \bigl(1 - \tfrac jm + \tfrac 1m
                              - \tfrac{m+1}{2m}\bigr),

    which :func:`rate_from_scores` computes by cumulative sums. Averaging
    within a tie group is a symmetric operation, so a group's weight is the
    mean of its members' sorted weights. Returning the weights lets a
    caller whose scores are themselves a linear functional of the data --
    the imputation scores of a fixed-effects forest -- compose the two and
    get an exact standard error instead of treating the scores as data.
    """
    pr = np.asarray(priorities, dtype=np.float64).ravel()
    order, codes = _priority_order(pr)
    m = len(pr)
    r = np.arange(1, m + 1, dtype=np.float64)
    if target == "AUTOC":
        harm = np.concatenate([[0.0], np.cumsum(1.0 / r)])
        w = harm[m] - harm[r.astype(np.int64) - 1]
        c = 1.0
    else:
        w = 1.0 - r / m + 1.0 / m
        c = (m + 1.0) / (2.0 * m)
    a_sorted = (w - c) / m
    n_groups = int(codes.max()) + 1
    sums = np.bincount(codes[order], weights=a_sorted, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups)
    return np.asarray((sums / counts)[codes])


def rate_from_scores(
    scores: np.ndarray,
    priorities: np.ndarray,
    target: str = "AUTOC",
    q: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    r"""RATE point estimate and TOC curve from doubly-robust scores.

    The deterministic core of ``grf::rank_average_treatment_effect`` (and
    of ``grf::rank_average_treatment_effect.fit``) with unit sample
    weights. Units are sorted by decreasing priority; scores are averaged
    within tied priorities; with :math:`S_k` the cumulative score sum of
    the top :math:`k` units,

    .. math::
        \mathrm{TOC}_k = S_k / k - \bar\Gamma, \qquad
        \mathrm{AUTOC} = \tfrac1n \sum_k \mathrm{TOC}_k, \qquad
        \mathrm{QINI} = \tfrac1n \sum_k \tfrac kn\,\mathrm{TOC}_k .

    The TOC curve on a grid ``q`` interpolates linearly between the
    discrete cut points exactly as ``grf`` does.

    Returns
    -------
    dict
        ``estimate``, ``toc_q``, ``toc`` (curve on ``q``), ``n``.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    pr = np.asarray(priorities, dtype=np.float64).ravel()
    n = len(s)
    if len(pr) != n:
        raise MethodIncompatibility(
            "rate(): scores and priorities must be the same length.",
            recovery_hint="Pass one priority per unit.",
            diagnostics={"n_scores": int(n), "n_priorities": int(len(pr))},
        )
    if n < 2 or not np.isfinite(s).all() or not np.isfinite(pr).all():
        raise DataInsufficient(
            "rate(): need at least two units with finite scores and priorities.",
            recovery_hint="Drop non-finite rows before evaluating RATE.",
        )
    key = str(target).upper().strip()
    if key not in ("AUTOC", "QINI"):
        raise MethodIncompatibility(
            "rate(): target must be 'AUTOC' or 'QINI'.",
            recovery_hint="Use a supported RATE summary target.",
            diagnostics={"target": target},
        )
    q_arr = (
        np.round(np.arange(1, 11) / 10.0, 10)
        if q is None
        else np.asarray(q, dtype=np.float64).ravel()
    )
    if (
        q_arr.size == 0
        or np.any(np.diff(q_arr) <= 0)
        or q_arr.min() <= 0
        or q_arr.max() != 1.0
    ):
        raise MethodIncompatibility(
            "rate(): q must be a strictly increasing grid on (0, 1] ending at 1.",
            recovery_hint="Use e.g. q=np.linspace(0.1, 1, 10).",
        )

    s_sorted, _ = _tie_averaged_sorted(s, pr)
    k_all = np.arange(1, n + 1, dtype=np.float64)
    cum = np.cumsum(s_sorted)
    ate = float(cum[-1] / n)
    toc = cum / k_all - ate
    if key == "AUTOC":
        estimate = float(toc.sum() / n)
    else:
        estimate = float(np.sum(k_all / n * toc) / n)

    nw = q_arr * n
    k = np.minimum(np.floor(nw + 1e-15), n).astype(np.int64)
    k_max = int(k.max())
    k_eff = np.maximum(k, 1)
    denom_adj = nw - k_eff
    nxt = np.minimum(k + 1, k_max)  # 1-based, as grf's pmin(idx + 1, max(idx))
    num_adj = denom_adj * s_sorted[nxt - 1]
    toc_grid = (cum[k_eff - 1] + num_adj) / (k_eff + denom_adj) - ate
    return {"estimate": estimate, "toc_q": q_arr, "toc": toc_grid, "n": n}


def _validated_rate_weights(weights: np.ndarray, n: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).ravel()
    if len(w) != n:
        raise MethodIncompatibility(
            "rate(): weights must have one entry per unit.",
            recovery_hint="Pass one weight per score.",
            diagnostics={"n_scores": int(n), "n_weights": int(len(w))},
        )
    if not np.isfinite(w).all() or np.any(w <= 0.0):
        raise MethodIncompatibility(
            "rate(): observation weights must be finite and strictly "
            "positive (a unit of zero weight has no place in the ranking).",
            recovery_hint="Drop zero-weight rows before evaluating RATE.",
        )
    return w


def _weighted_sorted(
    scores: np.ndarray, priorities: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tie-averaged scores and weights in descending priority order.

    Within a tie group the score is the *weighted* mean of the group, so
    the group's weighted total is unchanged. Returns ``(scores, weights,
    order)``; rows of equal priority keep their input order.
    """
    order, codes = _priority_order(priorities)
    avg = np.bincount(codes, weights=weights * scores) / np.bincount(
        codes, weights=weights
    )
    return avg[codes[order]], weights[order], order


def _weighted_rate_from_scores(
    scores: np.ndarray,
    priorities: np.ndarray,
    weights: np.ndarray,
    target: str,
    q: np.ndarray,
) -> Dict[str, Any]:
    r"""RATE and TOC curve with observation weights.

    :func:`rate_from_scores` with every count replaced by a weight total.
    With units in descending priority order, :math:`w_k` their weights,
    :math:`C_k = \sum_{j \le k} w_j`, :math:`S_k = \sum_{j \le k} w_j
    \Gamma_{(j)}` and :math:`W = C_n`,

    .. math::
        \mathrm{TOC}_k = S_k / C_k - S_n / W, \qquad
        \mathrm{AUTOC} = \tfrac1W \sum_k w_k \mathrm{TOC}_k, \qquad
        \mathrm{QINI} = \tfrac1W \sum_k w_k \tfrac{C_k}{W} \mathrm{TOC}_k ,

    which is the unweighted estimator when the weights are equal. The curve
    at ``q`` is the weighted mean score of the top ``q`` *share of weight*
    (the unit straddling the cut enters fractionally) minus the overall
    weighted mean. Agrees with ``grf::rank_average_treatment_effect.fit(
    ..., sample.weights=)`` run on the same inputs to 1e-14 (committed
    fixture ``grf_rate_weighted_R.json``).

    With unequal weights *and* tied priorities AUTOC depends, in the fifth
    digit on the fixture, on the order of the rows inside a tie group (the
    partial sums :math:`C_k` inside the group do); rows keep their input
    order, as the reference does. QINI does not depend on it.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    pr = np.asarray(priorities, dtype=np.float64).ravel()
    n = len(s)
    if n < 2 or len(pr) != n or not (np.isfinite(s).all() and np.isfinite(pr).all()):
        raise DataInsufficient(
            "rate(): need at least two units with finite scores and priorities.",
            recovery_hint="Drop non-finite rows before evaluating RATE.",
        )
    w = _validated_rate_weights(weights, n)
    s_sorted, w_sorted, _ = _weighted_sorted(s, pr, w)
    cum_w = np.cumsum(w_sorted)
    cum_s = np.cumsum(w_sorted * s_sorted)
    total = float(cum_w[-1])
    ate = float(cum_s[-1] / total)
    toc = cum_s / cum_w - ate
    if target == "AUTOC":
        estimate = float(np.sum(w_sorted * toc) / total)
    else:
        estimate = float(np.sum(w_sorted * (cum_w / total) * toc) / total)

    q_arr = np.asarray(q, dtype=np.float64).ravel()
    mass = q_arr * total
    k = np.minimum(np.searchsorted(cum_w, mass, side="right"), n)  # C_k <= mass
    k_eff = np.maximum(k, 1)
    nxt = np.minimum(k, n - 1)  # 0-based index of the unit straddling the cut
    partial = (mass - cum_w[k_eff - 1]) * s_sorted[nxt]
    toc_grid = (cum_s[k_eff - 1] + partial) / mass - ate
    return {"estimate": estimate, "toc_q": q_arr, "toc": toc_grid, "n": n}


def _weighted_rate_influence_se(
    scores: np.ndarray,
    priorities: np.ndarray,
    weights: np.ndarray,
    target: str,
    clusters: Optional[np.ndarray] = None,
) -> float:
    r"""Weighted counterpart of :func:`_rate_influence_se`.

    With :math:`p_k = w_k / W` and :math:`u_k = C_k / W` the weight share
    ranked at or above unit :math:`k`, the weighted estimator is exactly
    :math:`\hat\theta = \sum_k p_k \Gamma_{(k)} (\omega_k - c)` with

    - AUTOC: :math:`\omega_k = \sum_{j \ge k} p_j / u_j`, :math:`c = 1`;
    - QINI: :math:`\omega_k = 1 - u_k + p_k`, :math:`c = \sum_j p_j u_j`.

    Its influence value adds the term from estimating the ranks, as in
    :func:`_rate_influence_phi` with weighted expectations:
    :math:`\phi_k = \Gamma_{(k)}(\omega_k - c) + \sum_j p_j a_j
    (\mathbf 1\{j \le k\} - F_j)`, :math:`F_j = 1 - u_j + p_j` the weight
    share at or below :math:`j` and :math:`a_j = \Gamma_{(j)} / u_j` (AUTOC)
    or :math:`\Gamma_{(j)}` (QINI). The variance is that of the weighted
    mean :math:`\sum_k p_k \phi_k` with the weights held fixed:
    :math:`\sum_k p_k^2 (\phi_k - \hat\theta)^2 \, n / (n - 1)`, or, with
    clusters, the squared cluster sums of :math:`p_k (\phi_k -
    \hat\theta)` times :math:`G / (G - 1)`. Equal weights give
    :func:`_rate_influence_se`.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    w = _validated_rate_weights(weights, len(s))
    s_sorted, w_sorted, order = _weighted_sorted(
        s, np.asarray(priorities, dtype=np.float64).ravel(), w
    )
    n = len(s_sorted)
    p = w_sorted / w_sorted.sum()
    u = np.cumsum(p)
    if target == "AUTOC":
        omega = np.cumsum((p / u)[::-1])[::-1]
        c = 1.0
        a = s_sorted / u
    else:
        omega = 1.0 - u + p
        c = float(np.sum(p * u))
        a = s_sorted.copy()
    share_at_or_below = 1.0 - u + p
    rank_term = np.cumsum(p * a) - float(np.sum(p * a * share_at_or_below))
    phi = s_sorted * (omega - c) + rank_term
    centred = p * (phi - float(np.sum(p * phi)))
    if clusters is None:
        return float(np.sqrt(np.sum(centred**2) * n / (n - 1)))
    codes = np.asarray(clusters, dtype=np.int64)[order]
    n_groups = int(codes.max()) + 1
    summed = np.bincount(codes, weights=centred, minlength=n_groups)
    return float(np.sqrt(float(summed @ summed) * n_groups / (n_groups - 1)))


def _rate_influence_phi(
    scores: np.ndarray, priorities: np.ndarray, target: str
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Per-unit influence function of AUTOC / QINI, rank term included.

    Writing the estimator as :math:`\theta = E[\Gamma\{w(U) - c\}]` with
    :math:`U = 1 - F(S)` the fractional rank from the top
    (:math:`w(u) = -\log u` for AUTOC, :math:`1 - u` for QINI), its
    influence function has two parts: the score term
    :math:`\Gamma_i\{w(U_i) - c\} - \theta` and the rank term
    :math:`E_j[\Gamma_j\,(-w'(U_j))\,(\mathbf 1\{S_i \le S_j\} - F(S_j))]`
    from estimating :math:`F`.

    Returns ``(phi, order)``: ``phi`` in descending-priority order and the
    permutation ``order`` mapping it back to units (``phi[k]`` belongs to
    unit ``order[k]``).
    """
    s_sorted, order = _tie_averaged_sorted(
        np.asarray(scores, dtype=np.float64), np.asarray(priorities, dtype=np.float64)
    )
    n = len(s_sorted)
    r = np.arange(1, n + 1, dtype=np.float64)  # descending rank of s_sorted
    u = r / n
    if target == "AUTOC":
        harm = np.concatenate([[0.0], np.cumsum(1.0 / r)])
        w = harm[n] - harm[r.astype(np.int64) - 1]
        c = 1.0
        a = s_sorted / u  # Gamma_j * (-w'(U_j))
    else:
        w = 1.0 - u + 1.0 / n
        c = (n + 1.0) / (2.0 * n)
        a = s_sorted.copy()
    phi_score = s_sorted * (w - c)
    share_at_or_below = (n - r + 1.0) / n  # empirical F(S_j), ties broken by order
    rank_term = (np.cumsum(a) - float(np.sum(a * share_at_or_below))) / n
    return phi_score + rank_term, order


def _rate_influence_se(
    scores: np.ndarray,
    priorities: np.ndarray,
    target: str,
    clusters: Optional[np.ndarray] = None,
) -> float:
    r"""Analytic standard error of AUTOC / QINI with the rank term included.

    See :func:`_rate_influence_phi`.  Omitting the rank term -- as StatsPAI
    did before 1.30.0 -- overstates the standard error (by 30% on the
    committed fixture).  The result agrees with ``grf``'s half-sample
    bootstrap to within its Monte Carlo error (tested).  With ``clusters``
    (integer codes ``0..G-1`` per unit) the centred influence values are
    summed within clusters before squaring, with a ``G / (G - 1)``
    small-sample factor.
    """
    phi, order = _rate_influence_phi(scores, priorities, target)
    n = len(phi)
    if clusters is None:
        return float(np.std(phi, ddof=1) / np.sqrt(n))
    codes = np.asarray(clusters, dtype=np.int64)[order]
    n_groups = int(codes.max()) + 1
    summed = np.bincount(codes, weights=phi - phi.mean(), minlength=n_groups)
    var_est = float(summed @ summed) / n**2 * n_groups / (n_groups - 1)
    return float(np.sqrt(max(var_est, 0.0)))


def _rate_half_sample_se(
    scores: np.ndarray,
    priorities: np.ndarray,
    target: str,
    q: np.ndarray,
    R: int,
    seed: Optional[int],
    clusters: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> float:
    """``grf``'s half-sample bootstrap SE (``boot_grf(half.sample=TRUE)``).

    With ``clusters`` whole clusters are drawn, as ``grf`` does. With
    ``weights`` every draw is the weighted estimator on its rows.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    draws = np.empty(R)

    def _estimate(idx: np.ndarray) -> float:
        if weights is None:
            core = rate_from_scores(scores[idx], priorities[idx], target, q)
        else:
            core = _weighted_rate_from_scores(
                scores[idx], priorities[idx], weights[idx], target, q
            )
        return float(core["estimate"])

    if clusters is None:
        for b in range(R):
            idx = rng.choice(n, size=n // 2, replace=False)
            draws[b] = _estimate(idx)
    else:
        codes = np.asarray(clusters, dtype=np.int64)
        n_groups = int(codes.max()) + 1
        for b in range(R):
            chosen = rng.choice(n_groups, size=n_groups // 2, replace=False)
            idx = np.flatnonzero(np.isin(codes, chosen))
            draws[b] = _estimate(idx)
    return float(np.std(draws, ddof=1))


def rate(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    target: str = "AUTOC",
    q_grid: int = 100,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    priorities: Optional[np.ndarray] = None,
    se_method: str = "auto",
    n_bootstrap: int = 200,
    variance: str = "bjs",
    cluster: Any = None,
    members: Any = None,
    covariates: Any = "none",
) -> Dict[str, Any]:
    r"""Rank-weighted average treatment effect (RATE) [@yadlowsky2025evaluating].

    ``grf::rank_average_treatment_effect``: evaluates a prioritisation
    rule -- by default the forest's own CATE predictions (out-of-bag for
    GRF-engine forests), or any ``priorities`` vector (e.g. from a model
    trained on another sample) -- on the forest's doubly-robust scores
    (:func:`aipw_scores`, ``grf::get_scores``).  Units are sorted by
    decreasing priority, scores are averaged within tied priorities, and

    - **AUTOC** = mean over ``k`` of ``TOC_k``,
    - **QINI** = mean over ``k`` of ``(k/n) TOC_k``,

    with ``TOC_k`` the mean score of the top ``k`` units minus the overall
    mean -- ``grf``'s estimator, reproduced exactly given the same scores
    and priorities (the pure operator :func:`rate_from_scores`).

    Standard error: ``se_method='influence'`` (default) is the analytic
    influence-function SE including the term from estimating the priority
    ranks, summed within clusters when the forest has clusters;
    ``se_method='half_sample'`` is ``grf``'s half-sample bootstrap with
    ``n_bootstrap`` draws (``R`` in ``grf``, default 200; whole clusters
    are drawn), seeded by ``seed``.  The two agree to Monte Carlo error.

    Because priorities and evaluation share one sample when
    ``priorities`` is omitted, RATE is then a *diagnostic*; for a
    pre-registered test of targeting value, fit the forest on a training
    split and evaluate priorities on a held-out split.

    **Forests with fixed effects** have no propensity, so the curve is read
    on the population they identify -- the treated cells -- through the
    imputation scores :math:`\Gamma_{it} = Y_{it} - \hat\alpha_i -
    \hat\gamma_t` that :func:`statspai.average_treatment_effect` already
    uses there:  ``TOC(q) = ATT(top q by priority) - ATT``, a
    *retrospective* targeting curve ("the effect was this much larger among
    the cells the rule would have prioritised"), not the population RATE a
    randomised design gives.  ``variance=``, ``cluster=``, ``members=`` and
    ``covariates=`` are then passed through exactly as in
    :func:`statspai.forest_group_effects`, and ``se_method`` becomes:

    - ``'imputation'`` (what ``'auto'`` selects): the rank weights and the
      imputation weights are both linear, so their composition makes RATE
      one more linear functional of ``y`` with the same exact, cluster- or
      dyad-robust variance as the ATT.  The weights annihilate the fixed
      effects exactly (``max |Z'V| < 1e-15`` in the committed fixture) and
      sum to zero over treated cells, RATE being a contrast.
    - ``'influence'``: the rank-corrected influence function applied to the
      imputation scores -- it pays for estimating the ranking but treats
      the fixed effects as known.

    ``variance`` defaults to ``'bjs'`` here, not to the ``'forest'`` that
    :func:`statspai.average_treatment_effect` uses, and the reason is
    measured.  ``'forest'`` centres treated residuals on the out-of-bag
    prediction, which nets out the effect heterogeneity and so estimates
    the variance of the RATE *of this sample*; the ATT converges to its
    population value fast enough that this does not show, but RATE loads on
    the tail of the effect distribution and it does.  Over 120 replications
    (N = 150 units, T = 8, staggered adoption selected on the unit effect,
    tau = 0.3 + 0.5 z, priorities held out) nominal 95% intervals covered
    the *population* AUTOC 97.5% of the time with ``'bjs'`` and 90.8% with
    ``'forest'``; against the RATE of the realised sample it was the other
    way round, 99.2% and 95.8%.  So ``'bjs'`` is conservative for the
    population statement most readers want, ``'forest'`` is exact for the
    sample one.  Bias was -0.006 on 0.452 (AUTOC) and -0.0002 on 0.141
    (QINI).

    ``se_method='half_sample'`` raises for these forests: resampling units
    would have to refit the untreated two-way model in every draw, and the
    exact variance supersedes it.

    **Do not read the default ranking as a test.** With the forest's own
    out-of-bag predictions the ranking and the scores share
    ``gamma_hat_t``, so they stay correlated even though no unit predicts
    itself.  On the null design above (400 replications, no heterogeneity
    at all) AUTOC averaged -0.028 rather than 0 and a nominal 5% test
    rejected 16.0% of the time, QINI 12.2%.  :func:`statspai.rate_split`,
    which refits the rule on one half of the units and scores it on the
    other, averaged -0.0002 and rejected 0.5% and 0.0%, with 100% power
    against ``tau = 0.3 + 0.5 z``
    (``tests/reliability/forest_split_evaluation.py``).  A warning fires on
    this path;
    passing ``priorities=`` from a rule fitted elsewhere also silences it.

    .. versionchanged:: 1.31.0
       Forests with fixed effects are supported (they previously raised);
       the default ``se_method`` is ``'auto'``, which is ``'influence'`` for
       every forest that has a propensity, as before.

    .. versionchanged:: 1.30.0
       Scores are ``grf``'s (unclipped) AIPW scores; tied priorities are
       averaged as in ``grf``; QINI uses ``grf``'s ``k/n`` weights, so a
       flat TOC now gives exactly 0 (previously ``-mean(score) / (2n)``);
       the TOC curve interpolates between cut points as ``grf`` does; the
       influence-function SE includes the rank-estimation term
       (previously ~30% too large).

    Parameters
    ----------
    forest : fitted CausalForest
    X, Y, T : arrays, optional
        The training sample (defaults to the stored arrays).
    target : {'AUTOC', 'QINI'}
    q_grid : int
        Number of equally spaced points in ``(0, 1]`` at which the TOC
        curve is reported. Does not affect the estimate.
    alpha : float
    seed : int, optional
        Seed for ``se_method='half_sample'``.
    priorities : array, optional
        Priority score per training unit (higher = treat first). Defaults
        to the forest's CATE predictions.
    se_method : {'auto', 'influence', 'half_sample', 'imputation'}
        ``'auto'`` is ``'influence'``, or ``'imputation'`` for a forest with
        fixed effects. ``'imputation'`` is defined only for those.
    n_bootstrap : int
        Half-sample draws for ``se_method='half_sample'``.
    variance : {'bjs', 'forest'}
        Fixed-effects forests only: how treated residuals are centred
        before the cohort x event-time blocks. ``'bjs'`` (default here) is
        conservative for the population RATE, ``'forest'`` exact for the
        sample's -- see above.
    cluster : array-like or 'dyadic', optional
        Fixed-effects forests only; ``'dyadic'`` needs ``members=``.
    members : array-like, optional
        Fixed-effects forests only: the two members of each dyadic row.
    covariates : {'none', 'auto'}, list of str or array, default 'none'
        Fixed-effects forests only: time-varying covariates for the
        untreated model behind the imputation scores.

    Returns
    -------
    dict with keys ``estimate``, ``se``, ``ci_low``, ``ci_high``,
    ``target``, ``toc_curve`` (``(q_grid, 2)``), ``n``, ``method``,
    ``priority_source``, ``n_clusters``.

    References
    ----------
    [@yadlowsky2025evaluating]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.binomial(1, 0.5, size=n)
    >>> tau = 1.0 + X[:, 0]  # heterogeneous effect
    >>> Y = X[:, 1] + tau * T + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({
    ...     "y": Y, "d": T,
    ...     "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
    ... })
    >>> cf = sp.causal_forest(
    ...     data=df, formula="y ~ d | x0 + x1 + x2",
    ...     n_estimators=50, random_state=0,
    ... )
    >>> res = sp.rate(cf, target="AUTOC")
    >>> sorted(res.keys())
    ['ci_high', 'ci_low', 'estimate', 'method', 'n', 'n_clusters', \
'priority_source', 'se', 'target', 'toc_curve']
    >>> res["toc_curve"].shape
    (100, 2)
    """
    _require_fitted_forest(forest, "rate()")
    try:
        target_key = target.upper().strip()
    except AttributeError as exc:
        raise MethodIncompatibility(
            "rate(): target must be a string.",
            recovery_hint="Use target='AUTOC' or target='QINI'.",
            diagnostics={"target": target},
        ) from exc
    if target_key not in ("AUTOC", "QINI"):
        raise MethodIncompatibility(
            "rate(): target must be 'AUTOC' or 'QINI'.",
            recovery_hint="Use a supported RATE summary target.",
            diagnostics={"target": target},
        )
    if (
        isinstance(q_grid, bool)
        or not isinstance(q_grid, (int, np.integer))
        or int(q_grid) < 1
    ):
        raise MethodIncompatibility(
            "rate(): q_grid must be a positive integer.",
            recovery_hint="Use q_grid >= 1.",
            diagnostics={"q_grid": q_grid},
        )
    if se_method not in ("auto", "influence", "half_sample", "imputation"):
        raise MethodIncompatibility(
            "rate(): se_method must be 'auto', 'influence', 'half_sample' or "
            "'imputation'.",
            recovery_hint="Use the analytic default or grf's half-sample bootstrap.",
            diagnostics={"se_method": se_method},
        )
    q_grid_value = int(q_grid)
    alpha_value = _validate_alpha(alpha, "rate()")

    X_ = _prepare_forest_features(forest, X, "rate()")
    Y_ = _prepare_forest_vector(
        Y if Y is not None else getattr(forest, "_Y_original", None),
        "Y",
        X_.shape[0],
        "rate()",
    )
    T_ = _prepare_forest_vector(
        T if T is not None else getattr(forest, "_T_original", None),
        "T",
        X_.shape[0],
        "rate()",
    )
    n = len(Y_)
    if n < 2:
        raise DataInsufficient(
            "rate() requires at least 2 rows.",
            recovery_hint="Use a larger sample for RATE inference.",
        )

    from . import _grf_inference as _gi

    grf_forest = _gi.is_grf_forest(forest)
    if grf_forest and _gi.is_fe_forest(forest):
        from ._fe_imputation import rate_fe

        return rate_fe(
            forest,
            target_key,
            priorities=priorities,
            q_grid=q_grid_value,
            alpha=alpha_value,
            variance=variance,
            cluster=cluster,
            members=members,
            covariates=covariates,
            se_method=("imputation" if se_method == "auto" else se_method),
        )
    if se_method == "imputation":
        raise MethodIncompatibility(
            "rate(): se_method='imputation' is only defined for a forest with "
            "fixed effects, whose scores are a linear functional of the data.",
            recovery_hint="Use se_method='influence' or 'half_sample'.",
            diagnostics={"se_method": se_method},
        )
    if se_method == "auto":
        se_method = "influence"
    if grf_forest:
        _gi._require_dr(forest, "rate()")
    _require_training_rows(forest, X, Y, T, "rate()")
    if not np.all(np.isin(T_, (0.0, 1.0))):
        raise MethodIncompatibility(
            "rate(): RATE requires a binary treatment.",
            recovery_hint="grf's rank_average_treatment_effect is binary-only too.",
        )
    cluster_codes: Optional[np.ndarray] = None
    if grf_forest:
        _gi.require_finite_oob(forest, "rate()")
        tau_hat = np.asarray(forest._oob_tau, dtype=np.float64)
        # grf::get_scores does not clip the propensity.
        psi, *_ = _gi.dr_scores(forest, tau_hat, clip=0.0)
        if getattr(forest, "_clusters", None) is not None:
            cluster_codes = np.asarray(forest._clusters, dtype=np.int64)
    else:
        Y_hat, W_hat = _stored_training_nuisances(forest, "rate()")
        if np.any((W_hat <= 0.0) | (W_hat >= 1.0)):
            raise NumericalInstability(
                "rate(): some propensities are exactly 0 or 1; the doubly-robust "
                "score is undefined.",
                recovery_hint="Restrict the sample to units with overlap.",
            )
        tau_hat = np.asarray(forest.effect(X_), dtype=np.float64).ravel()
        psi = aipw_scores(
            tau=tau_hat, T=T_, e_hat=W_hat, m_hat=Y_hat, Y=Y_, target="all"
        )
    prio = (
        tau_hat
        if priorities is None
        else _prepare_forest_vector(priorities, "priorities", n, "rate()")
    )

    q_targets = np.linspace(1.0 / q_grid_value, 1.0, q_grid_value)
    q_targets[-1] = 1.0
    # The forest's observation weights (unequal only with
    # equalize_cluster_weights=True and clusters of different sizes), the
    # ones its ATE, best linear projection and calibration test use.
    obs_w: Optional[np.ndarray] = None
    if grf_forest:
        w_all = _gi._weights(forest, n)
        if float(np.ptp(w_all)) > 0.0:
            obs_w = w_all
    if obs_w is None:
        core = rate_from_scores(psi, prio, target_key, q_targets)
    else:
        core = _weighted_rate_from_scores(psi, prio, obs_w, target_key, q_targets)
    estimate = core["estimate"]
    if se_method == "influence":
        if obs_w is None:
            se = _rate_influence_se(psi, prio, target_key, clusters=cluster_codes)
        else:
            se = _weighted_rate_influence_se(
                psi, prio, obs_w, target_key, clusters=cluster_codes
            )
        method = "Rank-corrected influence-function SE"
    else:
        R = int(n_bootstrap)
        if R < 2:
            raise MethodIncompatibility(
                "rate(): n_bootstrap must be >= 2.",
                recovery_hint="Use grf's default of 200.",
            )
        se = _rate_half_sample_se(
            psi,
            prio,
            target_key,
            q_targets,
            R,
            seed,
            clusters=cluster_codes,
            weights=obs_w,
        )
        method = f"Half-sample bootstrap SE (grf), R={R}"
    if cluster_codes is not None:
        method += ", clustered"
    z = stats.norm.ppf(1 - alpha_value / 2)
    if priorities is not None:
        priority_source = "supplied"
    elif grf_forest:
        priority_source = "out_of_bag"
    else:
        priority_source = "in_sample"
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "target": target_key,
        "toc_curve": np.column_stack([core["toc_q"], core["toc"]]),
        "n": n,
        "method": method,
        "priority_source": priority_source,
        "n_clusters": (None if cluster_codes is None else int(cluster_codes.max()) + 1),
    }


# ======================================================================
# Honest subsample variance for aggregate quantities
# ======================================================================


def honest_variance(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    n_splits: int = 25,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Half-sample spread of the mean CATE prediction (deprecated).

    Draws ``n_splits`` random halves (without replacement) of the CATE
    predictions and reports the standard deviation of the half-sample
    means as ``se``. For a half-sample drawn without replacement the
    finite-population correction is exactly one half, so this spread
    estimates the sampling standard error of the full-sample mean,
    ``sd(tau_hat) / sqrt(n)``.

    It is **descriptive**: the predictions are held fixed, so the forest's
    own estimation error is not propagated. ``grf`` has no counterpart to
    this function.  For GRF-engine forests with ``X`` omitted it returns
    the doubly-robust ATE of :func:`average_treatment_effect` and its
    influence-function SE instead; use that function directly.

    .. versionchanged:: 1.30.0
       ``se`` was the standard deviation of the half-sample means divided
       by ``sqrt(n_splits)`` -- the Monte Carlo error of their average,
       which shrinks to zero as ``n_splits`` grows. It is now the spread
       itself.

    Parameters
    ----------
    forest : fitted CausalForest
    X : ndarray, optional
    n_splits : int
        Number of random half-sample draws.
    seed : int, optional

    Returns
    -------
    dict with ``ate``, ``se``, ``ci_low``, ``ci_high`` (95 %).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.binomial(1, 0.5, size=n)
    >>> tau = 1.0 + X[:, 0]  # heterogeneous effect
    >>> Y = X[:, 1] + tau * T + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({
    ...     "y": Y, "d": T,
    ...     "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
    ... })
    >>> cf = sp.causal_forest(
    ...     data=df, formula="y ~ d | x0 + x1 + x2",
    ...     n_estimators=50, random_state=0,
    ... )
    >>> hv = sp.honest_variance(cf, n_splits=25, seed=0)
    >>> bool(hv["se"] >= 0 and hv["ci_low"] <= hv["ate"] <= hv["ci_high"])
    True
    """
    _require_fitted_forest(forest, "honest_variance()")
    warnings.warn(
        "honest_variance() is deprecated: its half-sample spread of fixed "
        "CATE predictions is descriptive and does not propagate the forest's "
        "estimation error. For GRF-engine forests (X omitted) it returns the "
        "doubly-robust ATE and its influence-function SE; use "
        "average_treatment_effect() directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    from . import _grf_inference as _gi

    if _gi.is_grf_forest(forest) and X is None:
        res = _gi.average_effect(forest, "all", alpha=0.05, clip=0.01)
        return {
            "ate": res["estimate"],
            "se": res["se"],
            "ci_low": res["ci_low"],
            "ci_high": res["ci_high"],
            "plug_in_mean": float(np.mean(forest._oob_tau)),
            "method": "aipw",
        }
    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, (int, np.integer))
        or int(n_splits) < 2
    ):
        raise MethodIncompatibility(
            "honest_variance(): n_splits must be an integer >= 2.",
            recovery_hint="Use at least two half-sample splits.",
            diagnostics={"n_splits": n_splits},
        )
    n_splits_value = int(n_splits)
    X_ = _prepare_forest_features(forest, X, "honest_variance()")
    tau = np.asarray(forest.effect(X_)).ravel()
    n = len(tau)
    if n < 2:
        raise DataInsufficient(
            "honest_variance() requires at least 2 CATE rows.",
            recovery_hint="Pass at least two rows of effect modifiers.",
        )
    rng = np.random.default_rng(seed)

    means = np.empty(n_splits_value)
    for s in range(n_splits_value):
        perm = rng.permutation(n)
        half = perm[: n // 2]
        means[s] = float(tau[half].mean())

    ate = float(tau.mean())
    se = float(np.std(means, ddof=1))
    z = stats.norm.ppf(0.975)
    return {
        "ate": ate,
        "se": se,
        "ci_low": ate - z * se,
        "ci_high": ate + z * se,
    }


def aipw_scores(
    *,
    tau: np.ndarray,
    T: np.ndarray,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    Y: np.ndarray,
    target: str = "all",
) -> np.ndarray:
    r"""Per-unit doubly-robust (AIPW) scores :math:`\Gamma_i`.

    This is the estimator's *operator*: the deterministic map from the
    forest's outputs -- CATE predictions :math:`\hat\tau`, cross-fitted
    nuisances :math:`\hat m(X) = \hat E[Y \mid X]` and
    :math:`\hat e(X) = \hat E[T \mid X]` -- to the influence-function
    scores whose sample mean is the point estimate and whose
    :math:`\mathrm{sd}/\sqrt n` is the standard error.

    Separating it out matters for verification. Two independently grown
    forests never share :math:`\hat\tau`, so an end-to-end comparison
    against ``grf`` can only be statistical: both engines are refitted
    under many seeds on fixed data and their seed distributions compared
    (``tests/reference_parity/test_grf_seed_mc_equivalence.py``). The
    operator, by contrast, is a closed form and is pinned
    *exactly* to ``grf::get_scores`` given the same forest outputs -- see
    ``tests/reference_parity/test_grf_aipw_operator_parity.py``.

    For ``target="all"`` this is the identity ``grf`` uses,

    .. math::
        \Gamma_i = \hat\tau(X_i)
            + \frac{T_i-\hat e(X_i)}{\hat e(X_i)(1-\hat e(X_i))}
              \bigl(Y_i-\hat m(X_i)-(T_i-\hat e(X_i))\hat\tau(X_i)\bigr),

    which expands arm-by-arm to ``grf``'s published form
    :math:`\hat\tau + \frac{T}{\hat e}(Y-\hat m-(1-\hat e)\hat\tau)
    - \frac{1-T}{1-\hat e}(Y-\hat m+\hat e\hat\tau)`.

    Parameters
    ----------
    tau, T, e_hat, m_hat, Y : ndarray (n,)
        CATE predictions, treatment indicator, propensity, outcome
        regression, and outcome. ``e_hat`` is used as supplied; clip it
        before calling if overlap is a concern.
    target : {'all'}
        Only the ATE score is a single influence function. ``grf``
        estimates ATT/ATC as a plug-in CATE average over the target arm
        *plus* a Hajek-normalised doubly-robust correction, and adds the
        two variance components rather than taking the dispersion of one
        score vector — see :func:`grf_att_atc`.

    Returns
    -------
    ndarray (n,)
        Scores with ``mean(scores)`` the point estimate.

    References
    ----------
    [@athey2019generalized], [@robins1994estimation]
    """
    tau = np.asarray(tau, dtype=np.float64).ravel()
    T_ = np.asarray(T, dtype=np.float64).ravel()
    e_hat = np.asarray(e_hat, dtype=np.float64).ravel()
    m_hat = np.asarray(m_hat, dtype=np.float64).ravel()
    Y_ = np.asarray(Y, dtype=np.float64).ravel()
    if not (len(tau) == len(T_) == len(e_hat) == len(m_hat) == len(Y_)):
        raise MethodIncompatibility(
            "aipw_scores(): tau, T, e_hat, m_hat and Y must be the same length.",
            recovery_hint="Pass aligned per-observation vectors.",
            diagnostics={
                "n_tau": int(len(tau)),
                "n_T": int(len(T_)),
                "n_e": int(len(e_hat)),
                "n_m": int(len(m_hat)),
                "n_Y": int(len(Y_)),
            },
        )

    # Per-arm outcome regressions implied by (m̂, ê, τ̂):
    #   m = e·μ1 + (1-e)·μ0,  μ1 - μ0 = τ  ⇒  μ0 = m - e·τ,  μ1 = m + (1-e)·τ.
    if target == "all":
        m_full = m_hat + (T_ - e_hat) * tau  # = E[Y|X,T] under the model
        return tau + (T_ - e_hat) / (e_hat * (1.0 - e_hat)) * (Y_ - m_full)
    raise MethodIncompatibility(
        f"aipw_scores(): unsupported target={target!r}.",
        recovery_hint=(
            "Only target='all' is a single influence function. Use "
            "grf_att_atc() for the ATT/ATC decomposition."
        ),
    )


def grf_att_atc(
    *,
    tau: np.ndarray,
    T: np.ndarray,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    Y: np.ndarray,
    target: str,
) -> Tuple[float, float, np.ndarray]:
    r"""ATT / ATC exactly as ``grf::average_treatment_effect`` computes them.

    ``grf`` does **not** estimate ATT as the mean of one doubly-robust
    score. It reports

    .. math::
        \widehat{\mathrm{ATT}}
          = \underbrace{\frac{1}{n_1}\sum_{i:T_i=1}\hat\tau(X_i)}_{\text{plug-in}}
          + \underbrace{\frac1n\sum_i \Delta_i}_{\text{DR correction}},

    where the correction uses **Hajek-normalised** arm weights
    :math:`\gamma` — control units enter with
    :math:`\hat e/(1-\hat e)` renormalised to sum to :math:`n`, treated
    units with a flat weight renormalised the same way — giving

    .. math::
        \Delta_i = T_i\gamma_i(Y_i-\hat\mu_1(X_i))
                 - (1-T_i)\gamma_i(Y_i-\hat\mu_0(X_i)).

    The reported variance is the **sum of the two components'
    variances**, the plug-in dispersion
    :math:`\sum_{i:T_i=1}(\hat\tau_i-\bar\tau)^2/n_1^2` plus
    :math:`\frac{n}{n-1}\sum_i\Delta_i^2/n^2`; the cross-covariance is
    not subtracted.

    This differs materially from dividing a single Robins score by
    :math:`\hat p_1`, which is what StatsPAI reported before v1.21: on
    the committed ``grf`` fixture that route agreed on the point
    estimate to 9.3e-5 but produced a standard error **12% larger**
    than ``grf``'s, given *identical* forest outputs.

    Returns
    -------
    (estimate, se, dr_correction)
        ``dr_correction`` is the per-unit :math:`\Delta_i` vector, kept
        so callers can inspect or re-aggregate the correction term.

    References
    ----------
    [@athey2019generalized], [@robins1994estimation]
    """
    tau = np.asarray(tau, dtype=np.float64).ravel()
    T_ = np.asarray(T, dtype=np.float64).ravel()
    e_hat = np.asarray(e_hat, dtype=np.float64).ravel()
    m_hat = np.asarray(m_hat, dtype=np.float64).ravel()
    Y_ = np.asarray(Y, dtype=np.float64).ravel()
    n = int(len(tau))
    if n < 2:
        raise DataInsufficient(
            "grf_att_atc(): at least two observations are required.",
            recovery_hint="Pass a sample with n >= 2.",
        )

    mu0 = m_hat - e_hat * tau
    mu1 = m_hat + (1.0 - e_hat) * tau
    treated = T_ == 1
    control = ~treated
    n1, n0 = int(treated.sum()), int(control.sum())
    if n1 == 0 or n0 == 0:
        raise DataInsufficient(
            "grf_att_atc(): both arms must be non-empty.",
            recovery_hint="Pass a sample containing treated and control units.",
            diagnostics={"n_treated": n1, "n_control": n0},
        )

    if target == "treated":
        idx = treated
        g_control = e_hat[control] / (1.0 - e_hat[control])
        g_treated = np.ones(n1)
    elif target == "control":
        idx = control
        g_control = np.ones(n0)
        g_treated = (1.0 - e_hat[treated]) / e_hat[treated]
    else:
        raise MethodIncompatibility(
            f"grf_att_atc(): target must be 'treated' or 'control', got {target!r}.",
            recovery_hint="Use average_treatment_effect(target_sample='all') for ATE.",
        )

    gamma = np.zeros(n)
    gamma[control] = g_control / g_control.sum() * n
    gamma[treated] = g_treated / g_treated.sum() * n

    n_target = int(idx.sum())
    tau_raw = float(tau[idx].mean())
    tau_var = float(((tau[idx] - tau_raw) ** 2).sum() / n_target**2)

    dr_correction = T_ * gamma * (Y_ - mu1) - (1.0 - T_) * gamma * (Y_ - mu0)
    dr_mean = float(dr_correction.mean())
    dr_var = float((dr_correction**2).sum() / n**2 * n / (n - 1))

    return tau_raw + dr_mean, float(np.sqrt(tau_var + dr_var)), dr_correction


def _stored_nuisances(
    forest: "CausalForest",
    X_: np.ndarray,
    T_: np.ndarray,
    use_insample: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(Y_hat, W_hat)`` when the rows are the training sample, else None."""
    m_raw = getattr(forest, "_m_insample", None)
    e_raw = getattr(forest, "_e_insample", None)
    if m_raw is None or e_raw is None:
        return None
    m_hat = np.asarray(m_raw, dtype=np.float64).ravel()
    e_hat = np.asarray(e_raw, dtype=np.float64).ravel()
    if use_insample:
        return m_hat, e_hat
    X_train = np.asarray(getattr(forest, "_X_original", np.empty(0)), dtype=np.float64)
    T_train = np.asarray(getattr(forest, "_T_original", np.empty(0)), dtype=np.float64)
    if (
        X_.shape == X_train.shape
        and np.array_equal(X_, X_train)
        and np.array_equal(np.asarray(T_, dtype=np.float64).ravel(), T_train.ravel())
    ):
        return m_hat, e_hat
    return None


@accepts_aliases(_strict=True, controls="covariates")
def average_treatment_effect(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    target_sample: str = "all",
    alpha: float = 0.05,
    clip: float = 0.01,
    variance: str = "forest",
    covariates: Any = "none",
    subset: Any = None,
) -> Dict[str, Any]:
    """Aggregate CATE predictions into ATE/ATT/ATC/ATO targets.

    This mirrors the most-used ``grf::average_treatment_effect`` targets:
    ``"all"`` (ATE), ``"treated"`` (ATT), ``"control"`` (ATC), and
    ``"overlap"`` (ATO, weighted by ``e(X)(1-e(X))``).

    The estimate is the **doubly-robust AIPW influence-function mean**
    (the estimator grf reports), not a plug-in average of the CATE
    predictions.  Using the forest's own cross-fitted nuisances
    :math:`\\hat m(X)=\\hat E[Y\\mid X]` and :math:`\\hat e(X)=\\hat
    E[T\\mid X]`, the ATE score is

    .. math::
        \\Gamma_i = \\hat\\tau(X_i)
            + \\frac{T_i-\\hat e(X_i)}{\\hat e(X_i)(1-\\hat e(X_i))}
              \\bigl(Y_i-\\hat m(X_i)-(T_i-\\hat e(X_i))\\hat\\tau(X_i)\\bigr),

    (:func:`aipw_scores`).  ATT / ATC follow ``grf``: the plug-in CATE
    mean over the target arm plus a Hajek-normalised doubly-robust
    correction, with the two variance components added
    (:func:`grf_att_atc`).  ``se`` for the ATE is the influence-function
    standard error :math:`\\mathrm{sd}(\\Gamma)/\\sqrt n`.  The overlap
    target (ATO) is ``grf``'s Robinson / R-learner regression of
    ``Y - Y_hat`` on ``W - W_hat`` with an intercept and an HC3 standard
    error (:func:`grf_overlap_ate`); it needs no propensity inversion and
    is reported for continuous treatments as well.  Given the same forest
    outputs all four targets reproduce ``grf`` exactly
    (``tests/reference_parity/test_ml_causal_R_parity.py``).

    For forests fitted with the GRF engine (the default) the CATE entering
    the score is the **out-of-bag** prediction, observation weights follow
    ``equalize_cluster_weights``, standard errors are cluster-robust when
    the forest has clusters, and continuous treatments use grf's
    debiasing weights -- the definitions used by
    ``grf::average_treatment_effect``.  ``X`` / ``T`` may only restate the
    training data.

    Forests with fixed effects (``fe=``) have no propensity.  For them only
    ``target_sample='treated'`` is identified, and it is estimated by
    imputation [borusyak2024revisiting]: unit and period effects are fitted
    on the untreated cells, each treated cell's effect is imputed as
    ``Y - alpha_hat_i - gamma_hat_t``, and the ATT is their mean.  The
    standard error uses the exact linear weights of that estimator,
    clustered by the forest's clusters; ``variance`` chooses how treated
    residuals are centred (``'forest'``: out-of-bag forest prediction, then
    cohort x event-time means; ``'bjs'``: cohort x event-time means only,
    the conservative Borusyak-Jaravel-Spiess convention, identical to
    :func:`statspai.did_imputation`).  The payload also reports
    ``forest_plug_in``, the mean OOB prediction over the same cells, which
    is shrunk toward zero when effects are heterogeneous.

    .. versionchanged:: 1.30.0
       Forests with fixed effects return the imputation ATT for
       ``target_sample='treated'`` instead of raising.

    For other forests the doubly-robust scores exist only for the training
    sample; for other rows the function falls back to the plug-in CATE
    average, sets ``method='plug_in'`` and warns.

    .. versionchanged:: 1.30.0
       ``target_sample='overlap'`` is ``grf``'s R-learner estimator for
       every forest. Earlier releases averaged the AIPW scores with
       ``e(1-e)`` weights for legacy-engine forests, a different (also
       consistent) estimator of the same estimand.

    Parameters
    ----------
    forest : fitted CausalForest
    X, T : arrays, optional
        Rows to aggregate over (default: the training sample).
    target_sample : {'all', 'treated', 'control', 'overlap'}
    alpha : float, default 0.05
    clip : float, default 0.01
        Propensity scores are clipped to ``[clip, 1-clip]`` before the
        inverse-propensity terms of the ATE / ATT / ATC scores. ``grf``
        does not clip; ``clip=0`` reproduces it, and the clip is inert
        whenever all propensities already lie inside the band.
    variance : {'forest', 'bjs'}, default 'forest'
        Forests with fixed effects only (see above); ignored otherwise.
    covariates : 'none', 'auto', list of str or array, default 'none'
        Forests with fixed effects only: covariates in the untreated outcome
        model ``Y(0) = alpha_i + gamma_t + C' beta``.  ``'none'`` is
        :func:`statspai.did_imputation` without covariates (same estimate;
        same standard error with ``variance='bjs'``).  ``'auto'`` adds the
        effect modifiers and covariates that vary within units; use it when a
        time-varying covariate drives the untreated outcome and is not
        itself affected by the treatment.
    subset : boolean mask or row positions, optional
        Average over some of the training rows only, as
        ``grf::average_treatment_effect(subset=)`` does. The forest, its
        out-of-bag predictions and its nuisances stay those of the full
        fit; the target is the effect among the selected rows. Choose the
        rows from covariates. A subset built from the forest's own
        predictions on these rows (``oob > median``) describes the fit but
        its standard error does not account for the selection; see
        :func:`statspai.rate_split` for a test of that kind. GRF-engine
        forests without fixed effects only; for fixed-effects forests use
        :func:`statspai.forest_group_effects`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.binomial(1, 0.5, size=n)
    >>> tau = 1.0 + X[:, 0]  # heterogeneous effect
    >>> Y = X[:, 1] + tau * T + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({
    ...     "y": Y, "d": T,
    ...     "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
    ... })
    >>> cf = sp.causal_forest(
    ...     data=df, formula="y ~ d | x0 + x1 + x2",
    ...     n_estimators=50, random_state=0,
    ... )
    >>> ate = sp.average_treatment_effect(cf, target_sample="all")
    >>> ate["estimand"]
    'ATE'
    >>> att = sp.average_treatment_effect(cf, target_sample="treated")
    >>> att["estimand"]
    'ATT'
    """
    if not getattr(forest, "fitted_", False):
        raise MethodIncompatibility(
            "average_treatment_effect() requires a fitted forest.",
            recovery_hint="Call fit() before aggregating treatment effects.",
        )

    try:
        target_key = target_sample.lower().strip()
    except AttributeError as exc:
        raise MethodIncompatibility(
            "average_treatment_effect(): target_sample must be a string.",
            recovery_hint=("Use one of 'all', 'treated', 'control', or 'overlap'."),
            diagnostics={"target_sample": target_sample},
        ) from exc
    aliases = {
        "ate": "all",
        "all": "all",
        "att": "treated",
        "treated": "treated",
        "atc": "control",
        "control": "control",
        "ato": "overlap",
        "overlap": "overlap",
    }
    target = aliases.get(target_key)
    if target is None:
        raise MethodIncompatibility(
            "target_sample must be one of 'all', 'treated', 'control', " "or 'overlap'",
            recovery_hint="Use a supported GRF aggregation target.",
            diagnostics={"target_sample": target_sample},
        )
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "average_treatment_effect(): alpha must be a finite scalar.",
            recovery_hint="Use an alpha value in the open interval (0, 1).",
            diagnostics={"alpha": alpha},
        ) from exc
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise MethodIncompatibility(
            "average_treatment_effect(): alpha must be in the open interval (0, 1).",
            recovery_hint="Use an alpha value such as 0.05.",
            diagnostics={"alpha": alpha},
        )
    try:
        clip_value = float(clip)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "average_treatment_effect(): clip must be a finite scalar.",
            recovery_hint="Use a propensity clip in the interval [0, 0.5).",
            diagnostics={"clip": clip},
        ) from exc
    if not np.isfinite(clip_value) or not 0.0 <= clip_value < 0.5:
        raise MethodIncompatibility(
            "average_treatment_effect(): clip must be in the interval [0, 0.5).",
            recovery_hint="Use a small propensity clip such as 0.01.",
            diagnostics={"clip": clip},
        )

    from . import _grf_inference as _gi

    if subset is not None and (
        _gi.is_fe_forest(forest) or not _gi.is_grf_forest(forest)
    ):
        raise MethodIncompatibility(
            "average_treatment_effect(): subset= is available for GRF-engine "
            "forests without fixed effects.",
            recovery_hint="Use sp.forest_group_effects(forest, by=...) for "
            "averages within groups of a fixed-effects forest.",
        )
    if subset is not None and (X is not None or T is not None):
        raise MethodIncompatibility(
            "average_treatment_effect(): subset= selects training rows; do "
            "not also pass X= or T=.",
            recovery_hint="Drop X= / T=.",
        )
    if _gi.is_fe_forest(forest):
        if X is not None or T is not None:
            _require_training_rows(forest, X, None, T, "average_treatment_effect()")
        from ._fe_imputation import average_effect_fe

        return average_effect_fe(
            forest,
            target_sample=target,
            alpha=alpha_value,
            variance=variance,
            covariates=covariates,
        )

    grf_forest = _gi.is_grf_forest(forest)
    use_insample = X is None and T is None
    if X is None:
        X_ = np.asarray(forest._X_original, dtype=np.float64)
    elif callable(getattr(forest, "_prepare_effect_matrix", None)):
        X_ = np.asarray(
            forest._prepare_effect_matrix(
                X,
                context="average_treatment_effect()",
            ),
            dtype=np.float64,
        )
    else:
        try:
            X_ = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "average_treatment_effect(): X must be numeric.",
                recovery_hint="Pass X shaped (n_samples, n_features).",
            ) from exc
    if grf_forest:
        tau, _ = _gi.training_cate(forest, X_ if X is not None else None)
    else:
        tau = np.asarray(forest.effect(X_), dtype=np.float64).ravel()
    try:
        T_ = np.asarray(
            T if T is not None else getattr(forest, "_T_original", None),
            dtype=np.float64,
        ).ravel()
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "average_treatment_effect(): T must be numeric.",
            recovery_hint="Pass a numeric treatment vector aligned with X.",
        ) from exc
    if len(T_) != len(tau):
        raise MethodIncompatibility(
            "average_treatment_effect(): T must match the number of effect rows.",
            recovery_hint="Pass treatment values aligned with the CATE rows.",
            diagnostics={"n_t": int(len(T_)), "n_effects": int(len(tau))},
        )
    if not np.isfinite(T_).all():
        raise MethodIncompatibility(
            "average_treatment_effect(): T contains NaN or infinite values.",
            recovery_hint="Drop or impute non-finite treatment rows.",
        )
    # ATT and ATC condition on a treated / untreated group, which a
    # continuous treatment does not define: ``T == 1`` there selects the
    # rows whose dose happens to equal one (one year of schooling on the
    # Card design), not a treated arm. grf refuses these targets for a
    # non-binary W; so does StatsPAI, on both engines, rather than
    # returning an average over an arbitrary slice labelled "ATT".
    if target in ("treated", "control") and not np.all(
        np.isin(np.unique(T_), (0.0, 1.0))
    ):
        raise MethodIncompatibility(
            f"average_treatment_effect(): target_sample={target!r} needs a "
            "binary treatment; with a continuous treatment there is no "
            "treated or control group to average over.",
            recovery_hint="Use target_sample='all' (average partial effect) "
            "or 'overlap', or refit with a binary treatment.",
        )
    if target == "treated" and not np.any(T_ == 1):
        raise DataInsufficient(
            "average_treatment_effect(): no treated observations for ATT.",
            recovery_hint="Use target_sample='all' or pass at least one T == 1 row.",
        )
    if target == "control" and not np.any(T_ == 0):
        raise DataInsufficient(
            "average_treatment_effect(): no control observations for ATC.",
            recovery_hint="Use target_sample='all' or pass at least one T == 0 row.",
        )
    try:
        Y_ = np.asarray(
            getattr(forest, "_Y_original", None),
            dtype=np.float64,
        ).ravel()
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "average_treatment_effect(): stored outcomes must be numeric.",
            recovery_hint="Refit the forest with numeric outcomes.",
        ) from exc

    # The AIPW score divides by e(1-e), so it is defined only when e is a
    # propensity -- that is, only when T is binary. With a continuous
    # treatment the same nuisance slot holds E[T | X], a conditional mean
    # on the treatment's own scale, and the clip below silently maps it
    # into [clip, 1-clip]: on the Card design E[educ | X] is about 13
    # years, which clips to 0.99 and makes (T - e) / (e(1 - e)) about
    # 1{,}200. The score then returned an "ATE" of -1266 against a mean
    # CATE of 0.086, with a p-value of 0.0000 attached. The GRF engine
    # uses grf's continuous-treatment score for the 'all' and 'overlap'
    # targets, as grf does; the legacy engine has no such score and
    # reports the average of the fitted tau(x), labelled as descriptive.
    binary_treatment = bool(np.all(np.isin(np.unique(T_), (0.0, 1.0))))
    if grf_forest:
        _require_training_rows(forest, X, None, T, "average_treatment_effect()")
        if binary_treatment or target in ("all", "overlap"):
            return _gi.average_effect(
                forest, target, alpha_value, clip_value, subset=subset
            )
        if subset is not None:
            raise MethodIncompatibility(
                "average_treatment_effect(): subset= with a continuous "
                "treatment supports target_sample='all' or 'overlap'.",
                recovery_hint="Use target_sample='all'.",
            )

    # Outcome and propensity nuisances.  When aggregating on the training
    # sample we reuse the forest's own cross-fitted (cv=3) out-of-fold
    # nuisances m̂ = Ê[Y|X] and ê = Ê[T|X] -- the same quantities grf
    # uses for ``average_treatment_effect`` -- so the ATE/ATT scores are
    # honest doubly-robust influence functions rather than a plug-in mean
    # of the (regularisation-shrunk) CATE predictions.
    # Rows other than the training sample have no cross-fitted nuisances;
    # pairing them with the training sample's would be silently wrong, so
    # they take the documented plug-in route below.
    stored = _stored_nuisances(forest, X_, T_, use_insample)
    m_hat: np.ndarray
    e_hat: np.ndarray
    if stored is not None:
        m_hat, e_hat = stored
    else:
        m_hat = np.empty(0)
        e_hat = np.full(len(tau), float("nan"))
    if target == "overlap" and stored is not None and len(Y_) == len(tau):
        # grf's overlap target is the R-learner regression of the outcome
        # residual on the treatment residual; it needs no propensity
        # inversion and is defined for continuous treatments too.
        return _overlap_result(
            grf_overlap_ate(Y=Y_, W=T_, Y_hat=m_hat, W_hat=e_hat),
            n=int(len(tau)),
            alpha=alpha_value,
            W_res=T_ - e_hat,
            e_hat=e_hat,
        )
    if not binary_treatment:
        warnings.warn(
            "average_treatment_effect: the doubly-robust (AIPW) score "
            "requires a binary treatment, because it divides by "
            "e(1-e) for a propensity e. This forest's treatment takes "
            f"{len(np.unique(T_))} distinct values, so the reported "
            "estimate is the plug-in average of the fitted CATE "
            "predictions and its standard error is descriptive, not a "
            "doubly-robust influence-function SE. Fit with a binary "
            "treatment for AIPW inference.",
            AssumptionWarning,
            stacklevel=3,
        )
        return _plug_in_average(
            tau,
            T_,
            e_hat,
            target,
            alpha,
            reason="non_binary_treatment",
        )

    e_hat = np.clip(e_hat, clip_value, 1.0 - clip_value)
    if len(e_hat) != len(tau) or len(m_hat) != len(tau) or len(Y_) != len(tau):
        # AIPW score is unavailable (out-of-sample without nuisances or a
        # length mismatch); fall back to the plug-in CATE average and flag it.
        warnings.warn(
            "average_treatment_effect: no cross-fitted nuisances exist for "
            "these rows (they are not the forest's training sample), so the "
            "reported estimate is the plug-in average of the CATE "
            "predictions with a descriptive standard error, not the "
            "doubly-robust AIPW estimate.",
            AssumptionWarning,
            stacklevel=2,
        )
        return _plug_in_average(
            tau, T_, e_hat, target, alpha, reason="nuisances_unavailable"
        )

    n = int(len(tau))
    z = float(stats.norm.ppf(1 - alpha_value / 2))

    if target == "all":
        estimand = "ATE"
        psi = aipw_scores(tau=tau, T=T_, e_hat=e_hat, m_hat=m_hat, Y=Y_, target="all")
        estimate = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(n))
        ess = float(n)
    elif target in ("treated", "control"):
        estimand = "ATT" if target == "treated" else "ATC"
        estimate, se, _ = grf_att_atc(
            tau=tau, T=T_, e_hat=e_hat, m_hat=m_hat, Y=Y_, target=target
        )
        ess = float(T_.sum()) if target == "treated" else float((1.0 - T_).sum())
    else:  # overlap is returned above whenever nuisances exist
        raise MethodIncompatibility(  # pragma: no cover - guarded above
            "average_treatment_effect(): target_sample='overlap' needs the "
            "forest's cross-fitted nuisances.",
            recovery_hint="Aggregate on the training sample.",
        )

    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "target_sample": target,
        "estimand": estimand,
        "method": "aipw",
        "effective_sample_size": ess,
        "n": n,
        "alpha": alpha_value,
        "pscore_min": float(e_hat.min()),
        "pscore_max": float(e_hat.max()),
    }


def grf_overlap_ate(
    *,
    Y: np.ndarray,
    W: np.ndarray,
    Y_hat: np.ndarray,
    W_hat: np.ndarray,
) -> Tuple[float, float]:
    r"""``grf::average_treatment_effect(target.sample = "overlap")``.

    OLS of the outcome residual on the treatment residual *with* an
    intercept,

    .. math::
        Y_i - \hat m(X_i) = a + \tau\,(W_i - \hat e(X_i)) + \varepsilon_i,

    reporting :math:`\hat\tau` and its HC3 standard error
    (``sandwich::vcovHC`` default). This is the Robinson / R-learner
    estimator of the overlap-weighted effect
    :math:`E[e(1-e)\tau(X)] / E[e(1-e)]`; it uses no propensity inversion.
    (GRF-engine forests run the weighted, cluster-robust version of the
    same regression in :func:`statspai.forest._grf_inference.average_effect`.)

    Returns
    -------
    (estimate, se)
    """
    from ._grf_inference import cluster_robust_vcov

    arrs = [np.asarray(a, dtype=np.float64).ravel() for a in (Y, W, Y_hat, W_hat)]
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        raise MethodIncompatibility(
            "grf_overlap_ate(): Y, W, Y_hat and W_hat must be the same length.",
            recovery_hint="Pass aligned per-observation vectors.",
        )
    if n < 3:
        raise DataInsufficient(
            "grf_overlap_ate(): at least three observations are required.",
            recovery_hint="Pass a larger sample.",
        )
    y_res = arrs[0] - arrs[2]
    w_res = arrs[1] - arrs[3]
    D = np.column_stack([np.ones(n), w_res])
    if np.linalg.matrix_rank(D) < 2:
        raise NumericalInstability(
            "grf_overlap_ate(): the treatment residual is constant.",
            recovery_hint="The overlap estimator needs residual treatment variation.",
        )
    beta = np.linalg.solve(D.T @ D, D.T @ y_res)
    V = cluster_robust_vcov(D, y_res - D @ beta, vcov_type="HC3")
    return float(beta[1]), float(np.sqrt(V[1, 1]))


def _overlap_result(
    est_se: Tuple[float, float],
    *,
    n: int,
    alpha: float,
    W_res: np.ndarray,
    e_hat: np.ndarray,
) -> Dict[str, Any]:
    estimate, se = est_se
    z = float(stats.norm.ppf(1 - alpha / 2))
    w2 = W_res**2
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "target_sample": "overlap",
        "estimand": "ATO",
        "method": "r_learner_hc3",
        "effective_sample_size": float(w2.sum() ** 2 / np.sum(w2**2)),
        "n": n,
        "alpha": alpha,
        "pscore_min": float(np.min(e_hat)),
        "pscore_max": float(np.max(e_hat)),
    }


def _plug_in_average(
    tau: np.ndarray,
    T_: np.ndarray,
    e_hat: np.ndarray,
    target: str,
    alpha: float,
    reason: str = "unspecified",
) -> Dict[str, Any]:
    """Fallback weighted average of CATE predictions (no AIPW score).

    Used when the doubly-robust influence function cannot be formed: a
    non-binary treatment (no propensity exists), out-of-sample ``X`` with
    no stored nuisances, or a length mismatch. ``reason`` is carried into
    the payload so a caller can tell *which* of those it got.
    """
    if target == "all":
        weights = np.ones_like(tau)
        estimand = "ATE"
    elif target == "treated":
        weights = (T_ == 1).astype(float)
        estimand = "ATT"
    elif target == "control":
        weights = (T_ == 0).astype(float)
        estimand = "ATC"
    else:
        if len(e_hat) != len(tau) or not np.isfinite(e_hat).all():
            raise MethodIncompatibility(
                "average_treatment_effect(): target_sample='overlap' needs "
                "propensity scores, which exist only for the forest's "
                "training sample.",
                recovery_hint="Aggregate on the training sample (omit X and T).",
            )
        weights = e_hat * (1.0 - e_hat)
        estimand = "ATO"
    if float(weights.sum()) <= 0:
        raise DataInsufficient(
            "No observations contribute to the requested target_sample.",
            recovery_hint="Choose a target with support in the supplied sample.",
        )
    estimate = float(np.average(tau, weights=weights))
    norm_w = weights / weights.sum()
    se = float(np.sqrt(np.sum((norm_w**2) * (tau - estimate) ** 2)))
    z = float(stats.norm.ppf(1 - alpha / 2))
    ess = float((weights.sum() ** 2) / np.sum(weights**2))
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "target_sample": target,
        "estimand": estimand,
        "method": "plug_in",
        "plug_in_reason": reason,
        "effective_sample_size": ess,
        "n": int(len(tau)),
        "alpha": float(alpha),
        "pscore_min": float(e_hat.min()) if len(e_hat) else float("nan"),
        "pscore_max": float(e_hat.max()) if len(e_hat) else float("nan"),
    }


def forest_diagnostics(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    propensity_bounds: Tuple[float, float] = (0.05, 0.95),
) -> Dict[str, object]:
    """Return overlap and CATE-distribution diagnostics for a fitted forest.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.binomial(1, 0.5, size=n)
    >>> tau = 1.0 + X[:, 0]  # heterogeneous effect
    >>> Y = X[:, 1] + tau * T + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({
    ...     "y": Y, "d": T,
    ...     "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
    ... })
    >>> cf = sp.causal_forest(
    ...     data=df, formula="y ~ d | x0 + x1 + x2",
    ...     n_estimators=50, random_state=0,
    ... )
    >>> diag = sp.forest_diagnostics(cf)
    >>> bool(diag["n_treated"] + diag["n_control"] == diag["n"])
    True
    """
    _require_fitted_forest(forest, "forest_diagnostics()")
    try:
        low_raw, high_raw = propensity_bounds
        low = float(low_raw)
        high = float(high_raw)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "forest_diagnostics(): propensity_bounds must contain two scalars.",
            recovery_hint="Use bounds such as (0.05, 0.95).",
            diagnostics={"propensity_bounds": propensity_bounds},
        ) from exc
    if not np.isfinite(low) or not np.isfinite(high) or not 0 <= low < high <= 1:
        raise MethodIncompatibility(
            "propensity_bounds must satisfy 0 <= low < high <= 1.",
            recovery_hint="Use bounds such as (0.05, 0.95).",
            diagnostics={"propensity_bounds": propensity_bounds},
        )

    X_ = _prepare_forest_features(forest, X, "forest_diagnostics()")
    from . import _grf_inference as _gi

    if _gi.is_grf_forest(forest):
        tau, _ = _gi.training_cate(forest, X_ if X is not None else None)
    else:
        tau = np.asarray(forest.effect(X_), dtype=np.float64).ravel()
    if X is not None and T is None:
        raise MethodIncompatibility(
            "forest_diagnostics(): T is required when X is supplied.",
            recovery_hint="Pass treatment values aligned with the diagnostic X rows.",
        )
    T_ = _prepare_forest_vector(
        T if T is not None else getattr(forest, "_T_original", np.zeros(len(tau))),
        "T",
        len(tau),
        "forest_diagnostics()",
    )
    stored = _stored_nuisances(forest, X_, T_, use_insample=X is None and T is None)
    warnings = []
    if stored is not None:
        e_hat = np.asarray(stored[1], dtype=np.float64).ravel()
        overlap = (e_hat >= low) & (e_hat <= high)
        if e_hat.min() < low or e_hat.max() > high:
            warnings.append(
                "propensity scores outside requested overlap bounds; report "
                "ATE/ATT with caution or use target_sample='overlap'"
            )
    else:
        # Propensities exist only for the training sample. The overlap
        # fields are reported as missing rather than filled with the
        # treated share, which would read as perfect overlap.
        e_hat = np.full(len(tau), np.nan)
        overlap = np.zeros(len(tau), dtype=bool)
        warnings.append(
            "no propensity scores for these rows (not the forest's training "
            "sample); overlap fields are NaN"
        )
    if float(np.std(tau)) < 1e-8:
        warnings.append("predicted CATE is nearly constant; heterogeneity is weak")
    if not getattr(forest, "honest", True):
        warnings.append("forest was fitted with honest=False")

    return {
        "n": int(len(tau)),
        "n_treated": int(np.sum(T_ == 1)),
        "n_control": int(np.sum(T_ == 0)),
        "cate_mean": float(np.mean(tau)),
        "cate_sd": float(np.std(tau, ddof=1)) if len(tau) > 1 else 0.0,
        "cate_min": float(np.min(tau)),
        "cate_max": float(np.max(tau)),
        "cate_iqr": float(np.subtract(*np.percentile(tau, [75, 25]))),
        "pscore_min": float(np.min(e_hat)) if stored is not None else float("nan"),
        "pscore_max": float(np.max(e_hat)) if stored is not None else float("nan"),
        "overlap_low": float(low),
        "overlap_high": float(high),
        "overlap_share": (
            float(np.mean(overlap)) if stored is not None else float("nan")
        ),
        "n_low_pscore": int(np.sum(e_hat < low)),
        "n_high_pscore": int(np.sum(e_hat > high)),
        "warnings": warnings,
    }


@accepts_aliases(X="newdata")
@accepts_aliases(_strict=True, controls="covariates")
def calibrate_cate(
    forest: "CausalForest",
    newdata: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    method: str = "auto",
    covariates: Any = "none",
) -> Dict[str, Any]:
    """Rescale CATE predictions by their best-linear-predictor calibration.

    Forest CATE predictions are shrunk toward their mean: leaf averaging and
    honest subsampling regularise, so the spread of ``tau_hat`` understates
    the spread of ``tau``.  The calibration regression of
    :func:`calibration_test` estimates the slope ``beta_differential`` of
    the effect signal on the (out-of-bag) predictions; this function returns

        tau_cal(x) = beta_mean * mean(tau_oob)
                     + beta_differential * (tau_hat(x) - mean(tau_oob))

    for the training rows (``newdata=None``, out-of-bag) or for new rows.
    Because the slope is estimated on out-of-bag predictions it tends to 0
    when the effect is homogeneous, so the correction does not manufacture
    heterogeneity.

    Forests with two-way fixed effects and a binary treatment
    (``method='auto'`` or ``'imputation'``): the slope comes from
    regressing imputation scores
    on the out-of-bag predictions over the treated cells (see
    :func:`calibration_test`).  The scores are unbiased for each cell's
    effect, so the slope is a genuine de-attenuation factor, and
    ``tau_bar`` is the mean prediction over treated cells, so
    ``beta_mean * tau_bar`` is the imputation ATT.  In StatsPAI's
    simulations (N = 300 units, T = 8, staggered adoption selected on the
    unit effect, 200 replications) this reduced the RMSE of the treated
    cells' predictions from 0.634 to 0.570 with effects ``(1 + x1)(1 +
    0.2 e)`` and from 0.213 to 0.118 with a constant effect.  The map is
    estimated on treated cells; applying it to other rows assumes it
    carries over.

    ``method='within'`` uses globally within-transformed variables, as
    proposed by [@aytug2026attenuated].  Global two-way demeaning is exact
    only when the effect is homogeneous, so the heterogeneity *test* keeps
    its size, but the slope is not a de-attenuation factor: in StatsPAI's
    simulations (N = 300, T = 6) the OOB predictions had a slope of
    0.65-0.91 on the true effect while ``beta_differential`` stayed at
    0.88-1.07, and calibrated predictions did not reduce RMSE.  A warning
    is emitted in that case.

    .. versionchanged:: 1.30.0
       Forests with fixed effects and a binary treatment calibrate against
       imputation scores by default (``method='within'`` restores the
       earlier regression).

    Parameters
    ----------
    forest : CausalForest
        A forest fitted with the GRF engine (``split_rule="grf"``).
    newdata : array-like, optional
        Effect-modifier rows to calibrate.  ``None`` calibrates the
        out-of-bag predictions of the training rows.  ``X=`` is accepted as
        an alias.
    alpha : float, default 0.05
        Level for the reported slope confidence intervals.
    method : {'auto', 'imputation', 'within'}, default 'auto'
        Forests with fixed effects only; see :func:`calibration_test`.
    covariates : 'none', 'auto', list of str or array, default 'none'
        Imputation regression only; see :func:`average_treatment_effect`.

    Returns
    -------
    dict
        ``cate`` (calibrated predictions), ``raw_cate``, ``beta_mean``,
        ``beta_differential``, their ``*_se``, ``calibration`` (the full
        :func:`calibration_test` frame), ``heterogeneity_detected``
        (one-sided 5% test of ``beta_differential > 0``) and ``warnings``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 1000
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.integers(0, 2, n)
    >>> Y = X[:, 1] + (1 + X[:, 0]) * T + rng.normal(size=n)
    >>> cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=400, random_state=0)
    >>> cal = sp.calibrate_cate(cf)
    >>> cal["cate"].shape
    (1000,)

    References
    ----------
    [@chernozhukov2025generic], [@aytug2026attenuated]
    """
    _require_fitted_forest(forest, "calibrate_cate()")
    alpha_value = _validate_alpha(alpha, "calibrate_cate()")
    from . import _grf_inference as _gi

    if not _gi.is_grf_forest(forest):
        raise MethodIncompatibility(
            "calibrate_cate() needs out-of-bag predictions, which only "
            "GRF-engine forests provide.",
            recovery_hint="Refit with the default split_rule='grf'.",
        )
    resolved = _resolve_calibration_method(forest, method)
    oob = np.asarray(forest._oob_tau, dtype=np.float64)
    if resolved == "imputation":
        from ._fe_imputation import calibration_fe

        table = calibration_fe(forest, alpha=alpha_value, covariates=covariates)
        tau_bar = float(table.attrs["tau_bar"])
    else:
        table = _gi.calibration_blp(forest, alpha=alpha_value)
        w = _gi._weights(forest, oob.size)
        tau_bar = float(np.sum(w * oob) / np.sum(w))
    b_mean = float(table.loc["mean_forest_prediction", "coef"])
    b_diff = float(table.loc["differential_forest_prediction", "coef"])
    if newdata is None:
        raw = oob.copy()
    else:
        raw = np.asarray(
            forest.effect(
                _prepare_forest_features(forest, newdata, "calibrate_cate()")
            ),
            dtype=np.float64,
        ).ravel()
    calibrated = b_mean * tau_bar + b_diff * (raw - tau_bar)
    detected = bool(table.loc["differential_forest_prediction", "p_one_sided"] < 0.05)
    notes = []
    if resolved == "within":
        notes.append(
            "for forests with fixed effects the calibration slope comes from "
            "a globally within-transformed regression, which is misspecified "
            "when effects are heterogeneous; it tests for heterogeneity but "
            "does not correct the shrinkage of the CATE predictions."
        )
    if not detected:
        notes.append(
            "beta_differential is not significantly positive: the forest's "
            "ranking carries no detectable heterogeneity, so calibrated "
            "predictions are close to a constant effect."
        )
    if b_diff < 0:
        notes.append(
            "beta_differential is negative: predictions are anti-correlated "
            "with the effect signal; do not interpret the CATE ranking."
        )
    for note in notes:
        warnings.warn(f"calibrate_cate(): {note}", AssumptionWarning, stacklevel=2)
    return {
        "cate": calibrated,
        "raw_cate": raw,
        "beta_mean": b_mean,
        "beta_mean_se": float(table.loc["mean_forest_prediction", "se"]),
        "beta_differential": b_diff,
        "beta_differential_se": float(
            table.loc["differential_forest_prediction", "se"]
        ),
        "mean_oob_prediction": tau_bar,
        "calibration": table,
        "heterogeneity_detected": detected,
        "warnings": notes,
        "method": "blp_oob" if resolved != "imputation" else "blp_imputation",
    }


# GRF-compatible alias: the R ``grf`` package exposes this test as
# ``test_calibration``. We keep ``test_calibration`` as an alias so
# users familiar with GRF can reach for the same name, while the
# canonical Python name avoids pytest's ``test_*`` auto-discovery.
# ``__test__ = False`` stops pytest from collecting the module-level alias
# itself (e.g. under ``--doctest-modules`` on this file).
test_calibration = calibration_test
test_calibration.__test__ = False  # type: ignore[attr-defined]
