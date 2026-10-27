"""
Augmented Inverse Probability Weighting (AIPW) estimator.

Doubly robust estimator for average treatment effects that combines
outcome regression and inverse probability weighting. Consistent if
EITHER the outcome model OR the propensity score model is correctly
specified (but not necessarily both).

References
----------
Robins, J.M., Rotnitzky, A. and Zhao, L.P. (1994).
"Estimation of Regression Coefficients When Some Regressors Are Not
Always Observed."
*Journal of the American Statistical Association*, 89(427), 846-866.
[@robins1994estimation]

Glynn, A.N. and Quinn, K.M. (2010). "An Introduction to the Augmented Inverse
Propensity Weighted Estimator." *Political Analysis*, 18(1), 36-56.
[@glynn2010introduction]

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
Hansen, C., Newey, W. and Robins, J. (2018).
"Double/Debiased Machine Learning for Treatment and Structural Parameters."
*The Econometrics Journal*, 21(1), C1-C68. [@chernozhukov2018double]
"""

import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from .._input_validation import missing_named_columns
from ..core._call_record import records_call
from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core._validate import require_binary_treatment
from ..core._validate import treatment_as_float as _treatment_as_float
from ..core._vcov_spec import accepts_vce
from ..core.results import CausalResult
from ..exceptions import (
    ColumnNotFound,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from .basic_tests import _check_alpha

_OUTCOME_MODELS = ("linear", "logit", "probit", "poisson")
#: Fewest training rows of an arm for which the outcome regression is fitted;
#: below it the arm's training mean is used (recorded and warned about).
_MIN_ARM_FIT = 3


def _cluster_fold_ids(
    labels: pd.Series, n_folds: int, rng: np.random.Generator
) -> np.ndarray:
    """Cross-fitting fold of every row, drawn once per cluster.

    One ``rng.choice(n_folds)`` draw per cluster, in sorted-label order
    (first-appearance order when the labels do not sort), so the split
    depends on the seed and the labels but not on the row order. With one
    row per cluster and labels sorted in row order it is the row-level
    split ``rng.choice(n_folds, size=n)``.
    """
    try:
        codes, uniques = pd.factorize(labels, sort=True)
    except TypeError:  # mixed label types that do not sort
        codes, uniques = pd.factorize(labels)
    cluster_fold: np.ndarray = np.asarray(rng.choice(n_folds, size=len(uniques)))
    return np.asarray(cluster_fold[codes])


@records_call("aipw")
@accepts_aliases(_strict=True, controls="covariates")
@accepts_vce(default="robust", native={"robust": {"se_method": "influence"}})
@_expands_categorical("covariates")
def aipw(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    estimand: str = "ATE",
    n_folds: int = 5,
    alpha: float = 0.05,
    seed: Optional[int] = 42,
    cross_fit: bool = True,
    se_method: str = "influence",
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    propensity: Optional[Union[float, str]] = None,
    trim: float = 0.01,
    outcome_model: str = "linear",
    design: Optional[Any] = None,
    ps_learner: Optional[Any] = None,
    outcome_learner: Optional[Any] = None,
) -> CausalResult:
    """
    Augmented Inverse Probability Weighting (AIPW) estimator.

    Doubly robust estimator: consistent if either the outcome model
    or the propensity score model is correctly specified.

    Uses cross-fitting (Chernozhukov et al. 2018) to avoid overfitting
    bias from flexible first-stage models.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Binary treatment (0/1).
    covariates : list of str
        Covariates for both propensity score and outcome models.
    estimand : str, default 'ATE'
        Target: 'ATE' (average treatment effect) or
        'ATT' (average treatment effect on the treated).
    n_folds : int, default 5
        Number of cross-fitting folds.
    alpha : float, default 0.05
        Significance level.
    seed : int or None, default 42
        Seed for the cross-fitting fold assignment.  The default is a
        fixed integer so repeated calls on the same data return the same
        estimate — matching ``sp.tmle``, ``sp.bcf`` and
        ``sp.super_learner``, which already default to 42.

        Before 1.21.0 this defaulted to ``None``, which seeds
        ``np.random.default_rng`` from OS entropy: the point estimate
        moved by hundreds of dollars between identical calls and could
        not be pinned even with ``np.random.seed(...)``, because
        ``default_rng(None)`` ignores the legacy global RNG.

        Pass ``seed=None`` explicitly to opt back into a fresh random
        fold split per call (useful for Monte-Carlo studies of
        fold-assignment sensitivity); the seed actually used is always
        recorded in ``result.model_info['seed']``.
    cross_fit : bool, default True
        Cross-fit the nuisance models over ``n_folds`` random folds
        (Chernozhukov et al. 2018). ``False`` fits the logit propensity
        and the two per-arm OLS outcome regressions once on the full
        sample -- the classical parametric AIPW estimator, which is
        what Stata's ``teffects aipw`` and R ``AIPW`` with
        ``k_split = 1`` compute. ``n_folds`` and ``seed`` are then
        unused.
    se_method : {'influence', 'sandwich'}, default 'influence'
        ``'influence'`` reports ``sd(psi) / sqrt(n)`` from the
        estimated efficient influence function, treating the nuisance
        fits as known (the cross-fitting / DML convention; R ``AIPW``
        reports the same quantity). ``'sandwich'`` stacks the logit
        score, the two OLS normal equations and the AIPW moment and
        reports the M-estimation sandwich (divisor ``n``), which
        accounts for the estimated nuisance parameters at finite ``n``.
        This is the robust standard error of Stata's
        ``teffects aipw``. Requires ``cross_fit=False`` and
        ``estimand='ATE'``. The two agree asymptotically when both
        nuisance models are correctly specified.
    weights : str, optional
        Column of sampling (probability) weights, Stata ``[pw=]``. The
        propensity logit and outcome regressions are weighted fits, the
        effect is the weighted mean of the AIPW scores, and every
        standard error is the weighted (robust) one. Weights must be
        strictly positive; their scale does not matter.
    cluster : str, optional
        Column identifying clusters. Standard errors sum the influence
        function within clusters (Stata ``vce(cluster c)``). With
        ``cross_fit=True`` the folds are drawn over whole clusters (one
        seeded draw per cluster, in sorted-label order), so no nuisance
        model is trained on rows of the cluster it predicts;
        ``model_info['fold_unit']`` is then ``'cluster'``.

        .. versionchanged:: 1.41.0
           Folds were drawn row by row, so a cluster's own rows trained
           the nuisance models that predicted it.
    propensity : float or str, optional
        Known probability of treatment: one number (a trial randomised
        with that probability) or the name of a column holding each
        unit's design probability (stratified or covariate-adaptive
        randomisation). No propensity model is fitted and nothing is
        clipped; only the outcome regressions are estimated. Knowing the
        treatment mechanism does not change the efficiency bound of the
        ATE, so the estimator has the same influence function as with an
        estimated propensity, but it is then consistent whatever the
        outcome regressions are, and the influence-function standard
        error needs no correction for a first-stage propensity fit. Not
        combinable with ``se_method='sandwich'``.
    trim : float, default 0.01
        Fitted propensity scores are clipped to ``[trim, 1 - trim]`` before
        they enter the inverse-probability terms. A known ``propensity`` is
        used as given and never clipped. Rows are kept, so the
        target population does not change, but the estimate is no longer
        the textbook AIPW formula on the rows that were clipped. ``trim=0``
        applies no clipping and gives that formula exactly (what a
        hand-written ``mean(psi)``, Stata ``teffects aipw`` and R ``AIPW``
        without truncation compute). The number of clipped scores is in
        ``model_info['n_propensity_clipped']`` and a warning is issued
        when it is not zero. To drop rows with extreme scores instead,
        subset the data first (:func:`statspai.trimming`).

        .. versionchanged:: 1.39.0
           The bound was fixed at 0.01 and could not be changed, and
           ``n_propensity_clipped`` always read 0.
    outcome_model : {'linear', 'logit', 'probit', 'poisson'}, default 'linear'
        The per-arm outcome regression: least squares, a logit or probit
        for a binary (or fractional) outcome, or a Poisson regression with
        a log link for a non-negative one, each fitted by maximum
        likelihood. These are the ``logit``, ``probit`` and ``poisson``
        outcome models of Stata ``teffects aipw (y x, logit) (d x)``. With
        a binary outcome the linear model can predict outside [0, 1]; the
        logit cannot, though
        the AIPW estimate itself is still not a plug-in and is not
        confined to the range (``sp.tmle`` is). ``se_method='sandwich'``
        stacks the score of the chosen model.
    design : SurveyDesign, optional
        A complex survey design from :func:`sp.svydesign` declared on
        ``data`` (same rows). The nuisance models and the AIPW mean are
        fitted with the design weights, and every standard error is the
        Taylor-linearisation variance of the influence rows the chosen
        ``se_method`` uses (the efficient influence function, or the
        stacked sandwich rows with ``se_method='sandwich'``) over strata,
        PSUs and the finite population correction -- R
        ``survey::svyrecvar`` on those rows. The interval and p-value use
        the t distribution with the design degrees of freedom
        (#PSU - #strata), as Stata ``svy``. ``weights=`` and ``cluster=``
        must not be given; ``design=None`` changes nothing. With
        ``cross_fit=True`` and PSUs declared (``sp.svydesign(cluster=)``)
        the folds are drawn over whole PSUs, as with ``cluster=``, and
        ``model_info['fold_unit']`` is ``'psu'``.

        .. versionchanged:: 1.42
           Folds were drawn row by row, so a PSU's own rows trained the
           nuisance models that predicted it.
        A replicate-weight design (:func:`sp.svrepdesign`) is accepted
        too: the estimator is refitted with every replicate's weights
        and the replicate variance replaces the linearisation (R
        ``svrepdesign`` / Stata ``svy brr`` / ``svy jackknife``).

    ps_learner : classifier, optional
        Any estimator with ``fit`` / ``predict_proba`` (scikit-learn style,
        e.g. ``GradientBoostingClassifier()`` or
        ``sp.SuperLearner(task='classification')``) for the propensity
        score in place of the logit, fitted on the covariates in each
        cross-fitting fold (with ``sample_weight`` when ``weights`` is
        given).
    outcome_learner : regressor, optional
        Any regressor for the two per-arm outcome regressions in place of
        ``outcome_model``. With either learner the standard error is the
        influence-function one, valid under cross-fitting
        (``cross_fit=False`` warns: flexible learners fitted and evaluated
        on the same rows bias the estimate); ``se_method='sandwich'`` is
        refused.

    Returns
    -------
    CausalResult
        Doubly robust treatment effect estimate.

    References
    ----------
    Robins, J. M., Rotnitzky, A. and Zhao, L. P. (1994). Estimation of
    Regression Coefficients When Some Regressors are not Always Observed.
    *Journal of the American Statistical Association*.
    doi:10.1080/01621459.1994.10476818 [@robins1994estimation]

    Glynn, A. N. and Quinn, K. M. (2010). An Introduction to the
    Augmented Inverse Propensity Weighted Estimator. *Political
    Analysis*. doi:10.1093/pan/mpp036 [@glynn2010introduction]

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> age = rng.normal(40, 10, n)
    >>> income = rng.normal(50, 15, n)
    >>> education = rng.integers(10, 18, n).astype(float)
    >>> ps = 1 / (1 + np.exp(-(0.03 * (age - 40) + 0.02 * (income - 50))))
    >>> treatment = rng.binomial(1, ps)
    >>> outcome = (2.0 * treatment + 0.05 * age + 0.03 * income
    ...            + 0.1 * education + rng.normal(0, 1, n))
    >>> df = pd.DataFrame({'outcome': outcome, 'treatment': treatment,
    ...                    'age': age, 'income': income,
    ...                    'education': education})
    >>> result = sp.aipw(df, y='outcome', treat='treatment',
    ...                  covariates=['age', 'income', 'education'], seed=0)
    >>> bool(result.estimate > 0)
    True

    Notes
    -----
    The AIPW estimator is:

    .. math::
        \\hat{\\tau}_{AIPW} = \\frac{1}{n} \\sum_i \\left[
            \\hat{\\mu}_1(X_i) - \\hat{\\mu}_0(X_i)
            + \\frac{D_i (Y_i - \\hat{\\mu}_1(X_i))}{\\hat{e}(X_i)}
            - \\frac{(1-D_i)(Y_i - \\hat{\\mu}_0(X_i))}{1-\\hat{e}(X_i)}
        \\right]

    where :math:`\\hat{e}(X)` is the propensity score and
    :math:`\\hat{\\mu}_d(X)` is the outcome regression for treatment
    status :math:`d`.

    See Glynn & Quinn (2010) for an introduction, and
    Chernozhukov et al. (2018) for the cross-fitting procedure.
    """
    if design is not None and getattr(design, "is_replicate_design", False):
        _args = dict(locals())
        from ..survey._replicate_refit import replicate_refit

        return replicate_refit(aipw, _args, design, function="sp.aipw")
    if (
        ps_learner is not None or outcome_learner is not None
    ) and se_method != "influence":
        raise MethodIncompatibility(
            "aipw: se_method='sandwich' stacks parametric nuisance scores; "
            "with ps_learner / outcome_learner use se_method='influence'.",
            recovery_hint="Use se_method='influence' (with cross_fit=True).",
        )
    if (ps_learner is not None or outcome_learner is not None) and not cross_fit:
        warnings.warn(
            "aipw: learners fitted and evaluated on the same rows "
            "(cross_fit=False) overfit the nuisances and bias the estimate; "
            "the influence-function standard error assumes cross-fitting.",
            RuntimeWarning,
            stacklevel=2,
        )
    if se_method not in ("influence", "sandwich"):
        raise MethodIncompatibility(
            f"se_method must be 'influence' or 'sandwich', got {se_method!r}"
        )
    if se_method == "sandwich" and (cross_fit or estimand != "ATE"):
        raise MethodIncompatibility(
            "se_method='sandwich' is the stacked M-estimation variance of "
            "the full-sample parametric AIPW ATE; it requires "
            "cross_fit=False and estimand='ATE'.",
            recovery_hint="Use se_method='influence'.",
        )
    if outcome_model not in _OUTCOME_MODELS:
        raise MethodIncompatibility(
            f"aipw: outcome_model must be one of {_OUTCOME_MODELS}, got "
            f"{outcome_model!r}"
        )
    if propensity is not None and se_method == "sandwich":
        raise MethodIncompatibility(
            "aipw: se_method='sandwich' stacks the score of a fitted "
            "propensity logit; with a known propensity there is none.",
            recovery_hint="Use se_method='influence'.",
        )
    if not (isinstance(trim, (int, float)) and 0.0 <= float(trim) < 0.5):
        raise MethodIncompatibility(
            f"aipw: trim must lie in [0, 0.5), got {trim!r}.",
            recovery_hint="Use trim=0.01 (the default) or trim=0 for no clipping.",
        )
    trim = float(trim)
    _check_alpha(alpha, "aipw")
    if cross_fit and not n_folds >= 2:
        raise MethodIncompatibility(
            f"aipw: cross-fitting needs n_folds >= 2, got {n_folds!r}; with "
            "one fold the nuisance models would be trained on no data.",
            recovery_hint="Use n_folds=5 (the default), or cross_fit=False "
            "to fit the nuisance models on the full sample.",
        )
    rng = np.random.default_rng(seed)
    svy: Any = None
    if design is not None:
        from ..survey.linearization import check_design

        svy = check_design(
            design, data, function="aipw", weights=weights, cluster=cluster
        )

    extra = [c for c in (weights, cluster) if c is not None]
    if isinstance(propensity, str):
        extra.append(propensity)
    named_cols = {
        "y": y,
        "treat": treat,
        "covariates": list(covariates),
        "weights": weights,
        "cluster": cluster,
        "propensity": propensity if isinstance(propensity, str) else None,
    }
    missing_cols = missing_named_columns(named_cols, data.columns)
    if missing_cols:
        raise ColumnNotFound.from_missing(
            missing_cols,
            data.columns,
            context="aipw",
            extra_diagnostics={"missing": list(missing_cols.values())},
        )
    used_cols = list(dict.fromkeys([y, treat] + list(covariates) + extra))
    df = data[used_cols].dropna()
    Y = df[y].values.astype(float)
    D = _treatment_as_float(df[treat], function="aipw")
    X = df[covariates].values.astype(float)
    n = len(Y)

    require_binary_treatment(D, function="aipw", argument=treat)
    n_arm1 = int(D.sum())
    if n_arm1 == 0 or n_arm1 == n:
        raise DataInsufficient(
            f"aipw: {treat!r} has {n_arm1} treated and {n - n_arm1} control "
            "units on the complete rows; the estimator needs an outcome "
            "model for each arm.",
            diagnostics={"n_treated": n_arm1, "n_control": n - n_arm1},
        )
    if outcome_model in ("logit", "probit") and (Y.min() < 0 or Y.max() > 1):
        raise MethodIncompatibility(
            f"aipw: outcome_model={outcome_model!r} needs an outcome in [0, 1].",
            diagnostics={"min": float(Y.min()), "max": float(Y.max())},
        )
    if outcome_model == "poisson" and Y.min() < 0:
        raise MethodIncompatibility(
            "aipw: outcome_model='poisson' needs a non-negative outcome.",
            diagnostics={"min": float(Y.min())},
        )

    # Sampling weights normalised to mean one (their scale is irrelevant);
    # ``None`` keeps the unweighted path byte-identical to earlier releases.
    omega: Optional[np.ndarray] = None
    svy_rows = None
    if svy is not None:
        from ..survey.linearization import design_rows

        svy_rows = design_rows(svy, data, used_cols)
        omega = svy_rows.weights * (n / svy_rows.weights.sum())
    elif weights is not None:
        wv = df[weights].to_numpy(dtype=float)
        if not np.all(np.isfinite(wv)) or np.any(wv <= 0):
            raise MethodIncompatibility(
                "aipw: weights must be finite and strictly positive.",
                diagnostics={"weights": weights},
            )
        omega = wv * (n / wv.sum())
    groups: Optional[np.ndarray] = None
    if cluster is not None:
        groups = pd.factorize(df[cluster])[0]
        if groups.max() + 1 < 2:
            raise MethodIncompatibility(
                "aipw: cluster= needs at least two clusters.",
                diagnostics={"cluster": cluster},
            )
    e_known: Optional[np.ndarray] = None
    if propensity is not None:
        if isinstance(propensity, str):
            e_known = df[propensity].to_numpy(dtype=float)
        else:
            e_known = np.full(n, float(propensity))
        if not np.all(np.isfinite(e_known)) or np.any((e_known <= 0) | (e_known >= 1)):
            raise MethodIncompatibility(
                "aipw: a known propensity must lie strictly between 0 and 1.",
                diagnostics={
                    "min": float(np.nanmin(e_known)),
                    "max": float(np.nanmax(e_known)),
                },
            )
    design_based = omega is not None or groups is not None
    om = np.ones(n) if omega is None else omega

    # Cross-fitted predictions
    mu1_hat = np.zeros(n)  # E[Y|X, D=1]
    mu0_hat = np.zeros(n)  # E[Y|X, D=0]
    e_hat = np.zeros(n)  # P(D=1|X)

    # A survey design with declared PSUs folds by PSU, as cluster= folds by
    # cluster: the PSU is the sampling cluster, and its shared shock would
    # leak into its own out-of-fold predictions. PSU codes identify a PSU
    # within its stratum (Stata svyset; R nest=TRUE).
    psu_codes: Optional[np.ndarray] = None
    if svy is not None and svy_rows is not None and svy.cluster_col:
        psu_codes = np.asarray(svy._psu_codes)[svy_rows.rows]

    if cross_fit:
        if psu_codes is not None:
            fold_ids = _cluster_fold_ids(pd.Series(psu_codes), n_folds, rng)
        elif groups is None:
            fold_ids = rng.choice(n_folds, size=n)
        else:
            # Whole clusters go to one fold: a nuisance fit must not see
            # rows of the cluster it predicts, or the cluster's shared
            # shock leaks into its own out-of-fold predictions. Clusters
            # are drawn in sorted-label order so the split does not depend
            # on row order.
            fold_ids = _cluster_fold_ids(df[cluster], n_folds, rng)
        splits = [(fold_ids != f, fold_ids == f) for f in range(n_folds)]
    else:
        everyone = np.ones(n, dtype=bool)
        splits = [(everyone, everyone)]

    # Folds in which an arm had too few training rows for its outcome
    # regression: (fold, arm, n training rows of that arm).
    arm_mean_fallbacks: List[Tuple[int, int, int]] = []
    for fold, (train_mask, test_mask) in enumerate(splits):
        if not test_mask.any():
            continue  # an empty fold predicts nothing
        X_tr, Y_tr, D_tr = X[train_mask], Y[train_mask], D[train_mask]
        X_te = X[test_mask]
        w_tr = None if omega is None else omega[train_mask]
        for arm in (1, 0):
            n_arm_tr = int(np.sum(D_tr == arm))
            if n_arm_tr == 0:
                raise DataInsufficient(
                    f"aipw: cross-fitting fold {fold} has no "
                    f"{'treated' if arm == 1 else 'control'} rows in its "
                    "training set, so the outcome regression of that arm "
                    "cannot be fitted for the rows it predicts.",
                    recovery_hint="Use fewer folds (n_folds=), or "
                    "cross_fit=False; with cluster= make sure each arm "
                    "occurs in more clusters than there are folds.",
                    diagnostics={
                        "fold": fold,
                        "arm": arm,
                        "n_folds": n_folds,
                        "n_train": int(train_mask.sum()),
                    },
                )
            if outcome_learner is None and n_arm_tr < _MIN_ARM_FIT:
                arm_mean_fallbacks.append((fold, arm, n_arm_tr))

        # Propensity score (logistic regression), unless it is known
        if e_known is None:
            if ps_learner is None:
                e_hat[test_mask] = _fit_propensity(X_tr, D_tr, X_te, w_tr)
            else:
                e_hat[test_mask] = _learner_predict(
                    ps_learner, X_tr, D_tr, X_te, w_tr, proba=True
                )

        # Outcome regressions (OLS on treated and control separately)
        t1, t0 = D_tr == 1, D_tr == 0
        for arm, out in ((t1, mu1_hat), (t0, mu0_hat)):
            w_arm = None if w_tr is None else w_tr[arm]
            if outcome_learner is None:
                out[test_mask] = _fit_outcome(
                    X_tr[arm], Y_tr[arm], X_te, w_arm, outcome_model
                )
            else:
                out[test_mask] = _learner_predict(
                    outcome_learner, X_tr[arm], Y_tr[arm], X_te, w_arm, proba=False
                )

    if arm_mean_fallbacks:
        where = ", ".join(
            f"fold {f} {'treated' if a == 1 else 'control'} ({k} rows)"
            for f, a, k in arm_mean_fallbacks
        )
        warnings.warn(
            f"aipw: the outcome regression of an arm was replaced by that "
            f"arm's training mean where it had fewer than {_MIN_ARM_FIT} "
            f"training rows: {where}. The outcome model of those rows "
            "ignores the covariates; use fewer folds or cross_fit=False.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Clip propensity scores (a design probability is used as given)
    if e_known is not None:
        e_hat = e_known
        n_clipped = 0
    else:
        # Kept unclipped so result.violations() can judge overlap from the
        # fitted scores themselves, as it does for sp.ipw.
        e_raw = e_hat.copy()
        n_clipped = int(np.sum((e_hat < trim) | (e_hat > 1.0 - trim)))
        if trim > 0:
            np.clip(e_hat, trim, 1.0 - trim, out=e_hat)
        elif np.any((e_hat <= 0.0) | (e_hat >= 1.0)):
            raise MethodIncompatibility(
                "aipw: a fitted propensity score is exactly 0 or 1, so the "
                "inverse-probability term is undefined with trim=0.",
                recovery_hint="Use trim > 0, or restrict the sample to the "
                "region of overlap with sp.trimming.",
            )
        if n_clipped:
            warnings.warn(
                f"aipw: {n_clipped} of {n} fitted propensity scores "
                f"({100.0 * n_clipped / n:.1f}%) lie outside "
                f"[{trim:g}, {1.0 - trim:g}] and were clipped to it. The "
                "estimate differs from the unclipped AIPW formula on those "
                "rows; pass trim=0 for no clipping, or restrict the sample "
                "with sp.trimming.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _se_of(u: np.ndarray) -> float:
        """SE of a mean from its (already weight-scaled) influence rows.

        ``sqrt(sum u_i^2) / n``, or with clusters the sum over clusters of
        the within-cluster totals squared (Stata ``vce(cluster)``).
        """
        if svy_rows is not None:
            from ..survey.linearization import design_vcov

            return float(np.sqrt(design_vcov(u / n, svy, svy_rows.rows)[0, 0]))
        if groups is not None:
            u = np.bincount(groups, weights=u)
        return float(np.sqrt(np.sum(u**2)) / n)

    po_means = None
    po_means_se = None
    # AIPW influence function
    if estimand == "ATE":
        phi1 = mu1_hat + D * (Y - mu1_hat) / e_hat
        phi0 = mu0_hat + (1 - D) * (Y - mu0_hat) / (1 - e_hat)
        psi = phi1 - phi0
        if design_based:
            tau = float(np.mean(om * psi))
            m1, m0 = float(np.mean(om * phi1)), float(np.mean(om * phi0))
        else:
            tau = float(np.mean(psi))
            m1, m0 = float(np.mean(phi1)), float(np.mean(phi0))
        po_means = {1: m1, 0: m0}
        if se_method == "sandwich":
            if n_clipped:
                warnings.warn(
                    f"aipw: {n_clipped} propensity score(s) were clipped to "
                    f"[{trim:g}, {1.0 - trim:g}]; the stacked sandwich "
                    "differentiates the "
                    "unclipped logit and is only approximate for those rows.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if_1, if_0 = _aipw_stacked_if(
                X,
                D,
                Y,
                e_hat,
                mu1_hat,
                mu0_hat,
                phi1 - m1,
                phi0 - m0,
                omega,
                outcome_model,
            )
            if design_based:
                se = _se_of(if_1 - if_0)
                po_means_se = {1: _se_of(if_1), 0: _se_of(if_0)}
            else:
                # M-estimation sandwich, divisor n (Stata teffects convention).
                se = float(np.sqrt(np.mean((if_1 - if_0) ** 2) / n))
                po_means_se = {
                    1: float(np.sqrt(np.mean(if_1**2) / n)),
                    0: float(np.sqrt(np.mean(if_0**2) / n)),
                }
        elif design_based:
            se = _se_of(om * (psi - tau))
            po_means_se = {1: _se_of(om * (phi1 - m1)), 0: _se_of(om * (phi0 - m0))}
        else:
            se = float(np.sqrt(np.var(psi, ddof=1) / n))
            po_means_se = {
                1: float(np.sqrt(np.var(phi1, ddof=1) / n)),
                0: float(np.sqrt(np.var(phi0, ddof=1) / n)),
            }
    elif estimand == "ATT":
        p_treat = float(np.mean(om * D)) if design_based else np.mean(D)
        psi = D * (Y - mu0_hat) / p_treat - (1 - D) * e_hat * (Y - mu0_hat) / (
            (1 - e_hat) * p_treat
        )
        # ATT = mean(N_i) / mean(D_i) is a ratio, so its influence function
        # is (N_i - tau * D_i) / p, i.e. psi_i - tau * D_i / p. Before
        # 1.29 the ``- tau * D / p`` centring term was missing and the SE
        # was the standard deviation of psi itself, which is wrong unless
        # tau = 0 (it overstated the SE on every non-null ATT).
        if design_based:
            tau = float(np.mean(om * psi))
            se = _se_of(om * (psi - tau * D / p_treat))
        else:
            tau = float(np.mean(psi))
            se = float(np.sqrt(np.var(psi - tau * D / p_treat, ddof=1) / n))
    else:
        raise ValueError(f"estimand must be 'ATE' or 'ATT', got '{estimand}'")

    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = tau / se if se > 0 else 0
    pvalue = float(2 * stats.norm.sf(abs(z)))
    ci = (tau - z_crit * se, tau + z_crit * se)
    svy_df: Optional[float] = None
    if svy_rows is not None:
        from ..survey.linearization import design_df, design_inference

        svy_df = design_df(svy, svy_rows.rows)
        ci, pvalue = design_inference(tau, se, svy_df, alpha)

    model_info = {
        "estimator": "AIPW (Doubly Robust)",
        "n_folds": n_folds,
        "n_treated": int(D.sum()),
        "n_control": int((1 - D).sum()),
        "mean_propensity": float(e_hat.mean()),
        "cross_fit": bool(cross_fit),
        "se_method": se_method,
        "n_propensity_clipped": n_clipped,
        "trim": float(trim),
        # Unclipped fitted propensity and the bound it is judged against,
        # read by result.violations() (the overlap check sp.ipw already
        # gets). A known design probability is not a fitted score.
        "_pscore": None if e_known is not None else e_raw,
        "trimming_threshold": float(trim),
        "potential_outcome_means": po_means,
        "potential_outcome_means_se": po_means_se,
        "weights": weights,
        "cluster": cluster,
        "propensity": (
            "known"
            if e_known is not None
            else ("logit" if ps_learner is None else type(ps_learner).__name__)
        ),
        "outcome_model": (
            outcome_model if outcome_learner is None else type(outcome_learner).__name__
        ),
        "n_clusters": None if groups is None else int(groups.max() + 1),
        # Unit the cross-fitting folds were drawn over ('cluster' keeps
        # every cluster in one fold).
        "fold_unit": (
            None
            if not cross_fit
            else (
                "psu"
                if psu_codes is not None
                else ("row" if groups is None else "cluster")
            )
        ),
        "n_outcome_arm_mean_fallback": len(arm_mean_fallbacks),
        "outcome_arm_mean_fallback": [
            {"fold": f, "arm": a, "n_train": k} for f, a, k in arm_mean_fallbacks
        ],
        # Provenance for the cross-fitting split: None means the caller
        # explicitly opted into a fresh entropy-seeded split, so this
        # estimate is not reproducible by re-running.
        "seed": seed,
    }
    if svy is not None:
        from ..survey.linearization import design_info

        assert svy_df is not None
        model_info["survey_design"] = design_info(svy, svy_df)

    _result = CausalResult(
        method="AIPW (Doubly Robust)",
        estimand=estimand,
        estimate=tau,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
        _citation_key="aipw",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.aipw",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(covariates),
                "estimand": estimand,
                "n_folds": n_folds,
                "alpha": alpha,
                "seed": seed,
                "cross_fit": cross_fit,
                "se_method": se_method,
                "weights": weights,
                "cluster": cluster,
                "propensity": propensity,
                "outcome_model": outcome_model,
                "design": None if design is None else repr(design),
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _aipw_stacked_if(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    e: np.ndarray,
    mu1: np.ndarray,
    mu0: np.ndarray,
    phi1_c: np.ndarray,
    phi0_c: np.ndarray,
    omega: Optional[np.ndarray] = None,
    outcome_model: str = "linear",
) -> tuple:
    """Influence functions of the two AIPW potential-outcome means.

    Stacks the full-sample logit score ``X (D - e)``, the per-arm OLS
    normal equations ``D X (Y - X b1)`` / ``(1-D) X (Y - X b0)`` and the
    AIPW moments, and returns row ``i`` of ``-A^{-1} psi_i`` for the two
    mean parameters. ``phi1_c`` / ``phi0_c`` are the centred AIPW
    moments. The correction terms are ``E[d psi_mu / d gamma] IF_gamma``
    and ``E[d psi_mu / d beta] IF_beta``; they vanish in expectation only
    when both nuisance models are correct, so at finite ``n`` they move
    the variance.

    With sampling weights ``omega`` (mean one) every estimating equation
    is ``mean(omega_i psi_i) = 0``: each Jacobian becomes an
    ``omega``-weighted mean and each influence row carries its
    ``omega_i`` (the pweight sandwich of Stata ``teffects ... [pw=]``).

    For a logit or Poisson outcome model the link is canonical, so the
    score is still ``x (Y - mu)``; the Hessian and the derivative of the
    AIPW moment with respect to the coefficients pick up the variance
    function ``d mu / d eta`` (``mu (1 - mu)`` and ``mu``). A probit is
    not canonical: its own score and observed Hessian are used, and
    ``d mu / d eta`` is the normal density at the index.
    """
    n = len(Y)
    om = np.ones(n) if omega is None else omega
    Xc = np.column_stack([np.ones(n), X])
    r1, r0 = Y - mu1, Y - mu0
    w = e * (1 - e)
    # Per arm: s is the score multiplier (score = s x), h the weight of
    # minus its derivative (Hessian = mean(h x x')), v = d mu / d eta.
    s1, s0 = r1, r0
    if outcome_model == "logit":
        v1, v0 = mu1 * (1 - mu1), mu0 * (1 - mu0)
        h1, h0 = v1, v0
    elif outcome_model == "poisson":
        v1, v0 = mu1, mu0
        h1, h0 = v1, v0
    elif outcome_model == "probit":
        # Not a canonical link: the score is lam x with
        # lam = (Y - mu) phi / (mu (1 - mu)) and the observed Hessian
        # weight is lam (lam + eta).
        eta1, eta0 = stats.norm.ppf(mu1), stats.norm.ppf(mu0)
        v1, v0 = stats.norm.pdf(eta1), stats.norm.pdf(eta0)
        s1, s0 = r1 * v1 / (mu1 * (1 - mu1)), r0 * v0 / (mu0 * (1 - mu0))
        h1, h0 = s1 * (s1 + eta1), s0 * (s0 + eta0)
    else:
        v1 = v0 = h1 = h0 = np.ones(n)
    if_gamma = np.linalg.solve(
        (Xc * (om * w)[:, None]).T @ Xc / n, (Xc * (om * (D - e))[:, None]).T
    ).T
    if_b1 = np.linalg.solve(
        (Xc * (om * D * h1)[:, None]).T @ Xc / n, (Xc * (om * D * s1)[:, None]).T
    ).T
    if_b0 = np.linalg.solve(
        (Xc * (om * (1 - D) * h0)[:, None]).T @ Xc / n,
        (Xc * (om * (1 - D) * s0)[:, None]).T,
    ).T
    # d/d gamma of D r1 / e is -D r1 (1 - e) / e * x; of (1-D) r0 / (1-e)
    # it is (1-D) r0 e / (1 - e) * x. d/d beta of the AIPW moments is
    # x (1 - D / e) and x (1 - (1-D) / (1-e)).
    g1 = (Xc * (om * -D * r1 * (1 - e) / e)[:, None]).mean(axis=0)
    g0 = (Xc * (om * (1 - D) * r0 * e / (1 - e))[:, None]).mean(axis=0)
    b1 = (Xc * (om * v1 * (1 - D / e))[:, None]).mean(axis=0)
    b0 = (Xc * (om * v0 * (1 - (1 - D) / (1 - e)))[:, None]).mean(axis=0)
    if_1 = om * phi1_c + if_gamma @ g1 + if_b1 @ b1
    if_0 = om * phi0_c + if_gamma @ g0 + if_b0 @ b0
    return if_1, if_0


def _nuisance_designs(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Training and prediction designs with the same columns.

    An intercept is prepended to both. ``sm.add_constant`` on each matrix
    separately skips the intercept wherever a covariate happens to be
    constant, which need not be the same on the two samples (a rare dummy
    is constant in a small test fold): the prediction design then has one
    column fewer than the fit.

    A covariate that is constant on the training rows is collinear with
    the intercept there, so its coefficient is not identified and it is
    left out of both designs. The one exception keeps a case that already
    worked unchanged: a single non-zero constant covariate that takes the
    same value on the prediction rows serves as the intercept itself.
    """
    import statsmodels.api as sm

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    if X_train.ndim != 2 or X_train.shape[0] == 0 or X_train.shape[1] == 0:
        return (
            sm.add_constant(X_train, has_constant="add"),
            sm.add_constant(X_test, has_constant="add"),
        )
    fixed = np.all(X_train == X_train[0], axis=0)
    if not fixed.any():
        return (
            sm.add_constant(X_train, has_constant="add"),
            sm.add_constant(X_test, has_constant="add"),
        )
    j = np.flatnonzero(fixed)
    if (
        j.size == 1
        and X_train[0, j[0]] != 0
        and X_test.shape[0] > 0
        and np.all(X_test[:, j[0]] == X_train[0, j[0]])
    ):
        return X_train, X_test
    return (
        sm.add_constant(X_train[:, ~fixed], has_constant="add"),
        sm.add_constant(X_test[:, ~fixed], has_constant="add"),
    )


def _learner_predict(
    learner: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    w_train: Optional[np.ndarray],
    *,
    proba: bool,
) -> np.ndarray:
    """Fit a clone of a user learner and predict (probability of 1 for a
    classifier)."""
    from ..core._learners import clone_learner, positive_class_proba

    m = clone_learner(learner)
    if w_train is None:
        m.fit(X_train, y_train)
    else:
        m.fit(X_train, y_train, sample_weight=w_train)
    if proba:
        return positive_class_proba(m, X_test, context="ps_learner")
    return np.asarray(m.predict(X_test), dtype=float)


def _fit_propensity(
    X_train: np.ndarray,
    D_train: np.ndarray,
    X_test: np.ndarray,
    w_train: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Logistic regression propensity score (weighted MLE if ``w_train``)."""
    try:
        import statsmodels.api as sm

        X_tr, X_te = _nuisance_designs(X_train, X_test)
        if w_train is None:
            logit = sm.Logit(D_train, X_tr)
            res = logit.fit(disp=0, maxiter=300, warn_convergence=False)
        else:
            # Weighted logit MLE; freq_weights leaves the point estimates
            # of the pweighted likelihood unchanged (only its model-based
            # variance, which is not used, would differ).
            res = sm.GLM(
                D_train, X_tr, family=sm.families.Binomial(), freq_weights=w_train
            ).fit(tol=1e-12, maxiter=300)
        return np.asarray(res.predict(X_te), dtype=float)
    except Exception as exc:
        # A constant propensity turns AIPW into regression-adjustment-only,
        # and the influence-function SE (which assumes both nuisances were
        # fit) is then wrong. Do not degrade silently (CLAUDE.md §3.7).
        warnings.warn(
            "aipw: the propensity-score logit failed "
            f"({type(exc).__name__}: {exc}); falling back to a CONSTANT "
            "propensity. The AIPW estimate reduces to outcome-regression "
            "adjustment and its influence-function standard error is no "
            "longer valid. Check for separated or collinear covariates.",
            ConvergenceWarning,
            stacklevel=2,
        )
        return np.full(len(X_test), np.mean(D_train))


def _fit_outcome(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    w_train: Optional[np.ndarray] = None,
    model: str = "linear",
) -> np.ndarray:
    """Outcome regression: OLS, or a logit / Poisson GLM (weighted if given).

    With fewer than ``_MIN_ARM_FIT`` training rows the arm's (weighted)
    mean is returned; the caller records and warns about it, and refuses
    an arm with no training rows before getting here.
    """
    if len(X_train) < _MIN_ARM_FIT:
        return np.full(len(X_test), float(np.average(Y_train, weights=w_train)))
    try:
        import statsmodels.api as sm

        X_tr, X_te = _nuisance_designs(X_train, X_test)
        if model != "linear":
            family = {
                "logit": sm.families.Binomial,
                "probit": lambda: sm.families.Binomial(link=sm.families.links.Probit()),
                "poisson": sm.families.Poisson,
            }[model]()
            glm = sm.GLM(Y_train, X_tr, family=family, freq_weights=w_train).fit(
                tol=1e-13, maxiter=300
            )
            return np.asarray(glm.predict(X_te), dtype=float)
        ols = (
            sm.OLS(Y_train, X_tr)
            if w_train is None
            else sm.WLS(Y_train, X_tr, weights=w_train)
        )
        res = ols.fit()
        return np.asarray(res.predict(X_te), dtype=float)
    except Exception as exc:
        # A constant outcome model turns AIPW into IPW-only; the
        # influence-function SE is then wrong. Warn (CLAUDE.md §3.7).
        warnings.warn(
            "aipw: the outcome regression failed "
            f"({type(exc).__name__}: {exc}); falling back to a CONSTANT "
            "outcome model. The AIPW estimate reduces to inverse-propensity "
            "weighting and its influence-function standard error is no "
            "longer valid. Check for collinear or degenerate covariates.",
            ConvergenceWarning,
            stacklevel=2,
        )
        return np.full(len(X_test), np.mean(Y_train))


# Citation
CausalResult._CITATIONS["aipw"] = (
    "@article{glynn2010introduction,\n"
    "  title={An Introduction to the Augmented Inverse Propensity "
    "Weighted Estimator},\n"
    "  author={Glynn, Adam N. and Quinn, Kevin M.},\n"
    "  journal={Political Analysis},\n"
    "  volume={18},\n"
    "  number={1},\n"
    "  pages={36--56},\n"
    "  year={2010},\n"
    "  publisher={Cambridge University Press}\n"
    "}"
)
