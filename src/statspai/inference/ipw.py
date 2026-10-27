"""
Inverse Probability Weighting (IPW) estimator for ATE / ATT / ATC.

Standalone IPW without outcome-model augmentation (for augmented/doubly-robust,
see :func:`statspai.inference.aipw`).

Implements the Horvitz-Thompson estimator with logistic propensity scores
and optional trimming, normalized weights, and bootstrapped standard errors.

References
----------
Horvitz, D.G. and Thompson, D.J. (1952).
"A Generalization of Sampling Without Replacement From a Finite Universe."
*Journal of the American Statistical Association*, 47(260), 663-685. [@horvitz1952generalization]

Hirano, K., Imbens, G.W. and Ridder, G. (2003).
"Efficient Estimation of Average Treatment Effects Using the Estimated
Propensity Score."
*Econometrica*, 71(4), 1161-1189. [@hirano2003efficient]

Crump, R.K., Hotz, V.J., Imbens, G.W. and Mitnik, O.A. (2009).
"Dealing with Limited Overlap in Estimation of Average Treatment Effects."
*Biometrika*, 96(1), 187-199. [@crump2009dealing]
"""

from __future__ import annotations

import numbers
import warnings
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._aliases import accepts_aliases
from .._input_validation import missing_named_columns
from ..core._call_record import records_call
from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core._validate import require_binary_treatment
from ..core._validate import treatment_as_float as _treatment_as_float
from ..core._vcov_spec import accepts_vce
from ..core.results import CausalResult
from ..exceptions import (
    AssumptionWarning,
    ColumnNotFound,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from .basic_tests import _check_alpha


@records_call("ipw")
@accepts_aliases(_strict=True, controls="covariates", random_state="seed")
@accepts_vce(
    default=None,
    native={
        "robust": {"se_method": "sandwich"},
        "cluster": {"se_method": "sandwich"},
        "bootstrap": {"se_method": "bootstrap"},
    },
)
@_expands_categorical("covariates")
def ipw(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    estimand: str = "ATE",
    trim: float = 0.0,
    normalize: bool = True,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    se_method: str = "bootstrap",
    ps_model: str = "logit",
    propensity: Optional[Union[float, str]] = None,
    design: Optional[Any] = None,
    ps_learner: Optional[Any] = None,
    cross_fit: bool = True,
    n_folds: int = 5,
) -> CausalResult:
    """
    Inverse Probability Weighting estimator for treatment effects.

    Estimates ATE, ATT, or ATC by weighting observations by the inverse
    of their propensity to receive the treatment they actually received.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment indicator (0/1).
    covariates : list of str
        Variables for the propensity score model (logistic regression).
        A covariate that is an exact linear combination of the constant
        and the covariates listed before it is omitted from the
        propensity model with a warning and recorded in
        ``model_info['omitted']`` (``[{variable, reason}]``), as
        ``sp.logit`` does; the fitted scores do not depend on it.
    estimand : {'ATE', 'ATT', 'ATC', 'ATO', 'ATM'}, default 'ATE'
        The population the weights target. With propensity score ``e``
        and tilting function ``h(e)``, treated units get ``h / e`` and
        controls ``h / (1 - e)``:

        - ``'ATE'``: ``h = 1``, the whole sample.
        - ``'ATT'``: ``h = e``, the treated.
        - ``'ATC'`` (also ``'ATU'``): ``h = 1 - e``, the untreated.
        - ``'ATO'``: ``h = e (1 - e)``, the overlap population of Li,
          Morgan and Zaslavsky (2018) [@li2018balancing]. Weights are bounded by one and the
          covariates of the logit are balanced exactly.
        - ``'ATM'``: ``h = min(e, 1 - e)``, the matching weights of Li and
          Greene (2013) [@li2013weighting], the population a 1:1 caliper match would keep.

        ``'ATO'`` and ``'ATM'`` are defined through normalised weights and
        require ``normalize=True``.
    trim : float, default 0.0
        Trim propensity scores to [trim, 1 - trim]. Common choices: 0.01, 0.05, 0.1.
        Crump et al. (2009) recommend dropping units with p outside [0.1, 0.9].
        Must lie in ``[0, 0.5)``.
    normalize : bool, default True
        If True, use Hajek (normalised) weights. Generally recommended
        for finite-sample stability, and the estimate does not change when
        a constant is added to the outcome. ``False`` is Horvitz-Thompson:
        the weighted sums are divided by the size of the target population
        (``n`` for the ATE, the number of treated for the ATT, of controls
        for the ATC).
    n_bootstrap : int, default 500
        Number of bootstrap iterations for standard error estimation
        (at least 2). ``n_bootstrap=0`` asks for the point estimate alone:
        no resampling is done and the standard error, interval and p-value
        are missing. A resample that holds a single treatment arm has no
        contrast: it is excluded with a warning and counted in
        ``model_info['n_bootstrap_failed']``, and the call is refused when
        fewer than 10 usable resamples remain.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    seed : int, optional
        Seed of the bootstrap resampling (and of the cross-fitting folds
        with ``ps_learner``); ``random_state=`` is accepted as an alias.
        The same seed reproduces the standard error exactly. The default
        ``None`` draws fresh resamples on every call, so the bootstrap
        standard error then varies from run to run by its Monte Carlo
        error; pass a seed for a reproducible result.
    weights : str, optional
        Column of sampling (probability) weights, Stata ``[pw=]``. The
        propensity logit is a weighted fit and each unit's IPW weight is
        multiplied by its sampling weight. Strictly positive; scale-free.
    cluster : str, optional
        Column identifying clusters. The bootstrap resamples whole clusters;
        the sandwich sums the influence function within clusters (Stata
        ``vce(cluster c)``).
    se_method : {'bootstrap', 'sandwich'}, default 'bootstrap'
        ``'sandwich'`` is the stacked M-estimation variance of the logit
        score and the normalised IPW means (divisor ``n``) -- the robust
        standard error of Stata ``teffects ipw``, deterministic and
        bootstrap-free. With ``normalize=False`` it is the variance of the
        Horvitz-Thompson estimator; with ``trim > 0`` the clipped scores
        are held fixed, as they are in the estimator.
    ps_model : {'logit', 'probit'}, default 'logit'
        The binary model for the propensity score. ``'probit'`` is the
        treatment model of Stata ``teffects ipw (y) (d x, probit)``. The
        sandwich variance uses the score and the observed Hessian of the
        chosen model.
    propensity : float or str, optional
        Known probability of treatment: one number, or a column of
        per-unit design probabilities (a randomised trial). No propensity
        model is fitted, the bootstrap resamples the design probabilities
        with their rows, and the sandwich drops the first-stage term. An
        IPW estimator that *estimates* the propensity is the more precise
        of the two even when the truth is known, because the fitted
        propensity absorbs chance imbalance in the covariates; pass the
        known one when the design-based estimator is what is wanted, or
        use ``sp.aipw(propensity=)``, whose variance does not depend on
        the choice.
    design : SurveyDesign, optional
        A complex survey design from :func:`sp.svydesign` declared on
        ``data`` (same rows). The propensity model and the IPW means are
        fitted with the design weights, and the standard error is the
        Taylor-linearisation variance of the stacked influence function
        (the ``'sandwich'`` rows) over strata, PSUs and the finite
        population correction -- R ``survey::svyrecvar`` applied to
        ``-A^{-1} w_i psi_i`` (Binder's method). The interval and p-value
        use the t distribution with the design degrees of freedom
        (#PSU - #strata), as Stata ``svy``. ``se_method`` and
        ``n_bootstrap`` are then not used (a bootstrap of a complex design
        needs replicate weights), and ``weights=`` / ``cluster=`` must not
        be given. ``design=None`` leaves every other path unchanged.
        A replicate-weight design (:func:`sp.svrepdesign`) is accepted
        too: the estimator is refitted with every replicate's weights
        and the replicate variance replaces the linearisation (R
        ``svrepdesign`` / Stata ``svy brr`` / ``svy jackknife``).
    ps_learner : classifier, optional
        Any estimator with ``fit`` / ``predict_proba`` (scikit-learn style,
        e.g. ``GradientBoostingClassifier()`` or
        ``sp.SuperLearner(task='classification')``) for the propensity
        score in place of ``ps_model``, fitted with ``sample_weight`` when
        ``weights`` is given. The standard error is the bootstrap one, and
        every replicate refits the learner. ``se_method='sandwich'`` (it
        stacks the logit score), ``design=`` and ``propensity=`` are
        refused. Weighting alone is not doubly robust and a flexible score
        converges more slowly than ``1/sqrt(n)``, so the estimate carries
        the learner's bias into the contrast; ``sp.aipw(ps_learner=)`` is
        the estimator whose error is second order in the nuisance errors.
    cross_fit : bool, default True
        Used with ``ps_learner`` only: predict each unit's score from a
        learner trained on the other folds. ``False`` trains and predicts
        on the same rows and warns (an overfitted score pushes the weights
        of the training units towards one and the contrast towards the raw
        difference in means). In a bootstrap replicate every copy of a unit
        stays in that unit's fold.
    n_folds : int, default 5
        Number of cross-fitting folds (at least 2), drawn from ``seed``.

    Returns
    -------
    CausalResult
        With `.estimate`, `.se`, `.ci`, `.pvalue`, and propensity score
        diagnostics in `.model_info`, among them ``ess_treated`` and
        ``ess_control``: Kish's effective sample size of each arm's
        weights. When either falls below a fifth of the arm's size an
        :class:`~statspai.exceptions.AssumptionWarning` is raised: the
        estimate then rests on a few heavily weighted units.

        A fitted propensity model that separates the arms completely (its
        index classifies every unit correctly) is refused with
        :class:`~statspai.exceptions.DataInsufficient`: there is no
        overlap to weight on. Fitted scores at the numerical bound without
        complete separation raise a
        :class:`~statspai.exceptions.ConvergenceWarning` and set
        ``model_info['separation']`` to ``'quasi-complete'``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> age = rng.normal(40, 10, n)
    >>> education = rng.normal(13, 3, n)
    >>> experience = rng.normal(15, 7, n)
    >>> p = 1 / (1 + np.exp(-(-2 + 0.03 * (age - 40) + 0.15 * (education - 13))))
    >>> training = (rng.uniform(size=n) < p).astype(int)
    >>> wage = 20 + 2.5 * training + 0.1 * age + 0.5 * education + rng.normal(0, 2, n)
    >>> df = pd.DataFrame({"wage": wage, "training": training, "age": age,
    ...                    "education": education, "experience": experience})
    >>> result = sp.ipw(df, y='wage', treat='training',
    ...                 covariates=['age', 'education', 'experience'],
    ...                 n_bootstrap=100, seed=0)
    >>> result.estimand
    'ATE'

    >>> # ATT with trimming
    >>> result = sp.ipw(df, y='wage', treat='training',
    ...                 covariates=['age', 'education'],
    ...                 estimand='ATT', trim=0.05, n_bootstrap=100, seed=0)
    >>> result.estimand
    'ATT'
    """
    if design is not None and getattr(design, "is_replicate_design", False):
        _args = dict(locals())
        from ..survey._replicate_refit import replicate_refit

        # point estimates only on the refits (n_bootstrap=0)
        return replicate_refit(
            ipw,
            _args,
            design,
            function="sp.ipw",
            overrides={"se_method": "bootstrap", "n_bootstrap": 0},
            # cross-fitting folds would be redrawn on every replicate
            refuse=("cluster", "ps_learner"),
        )
    estimand = estimand.upper()
    if estimand == "ATU":
        estimand = "ATC"
    if estimand not in ("ATE", "ATT", "ATC", "ATO", "ATM"):
        raise ValueError(
            "estimand must be 'ATE', 'ATT', 'ATC' (or 'ATU'), 'ATO' or 'ATM', "
            f"got '{estimand}'"
        )
    if estimand in ("ATO", "ATM") and not normalize:
        raise MethodIncompatibility(
            f"estimand={estimand!r} has no Horvitz-Thompson form: the overlap "
            "and matching-weight populations are defined by the weights "
            "themselves.",
            recovery_hint="Use normalize=True.",
        )

    if se_method not in ("bootstrap", "sandwich"):
        raise MethodIncompatibility(
            f"se_method must be 'bootstrap' or 'sandwich', got {se_method!r}"
        )
    if ps_model not in ("logit", "probit"):
        raise MethodIncompatibility(
            f"ps_model must be 'logit' or 'probit', got {ps_model!r}"
        )
    _check_alpha(alpha, "ipw")
    if ps_learner is not None:
        from ._ps_learner import check_learner_options

        check_learner_options(
            se_method, design, propensity, ps_model, cross_fit, n_folds
        )
    svy: Any = None
    if design is not None:
        from ..survey.linearization import check_design

        svy = check_design(
            design, data, function="ipw", weights=weights, cluster=cluster
        )
        se_method = "sandwich"
    trim_is_number = isinstance(trim, numbers.Real) and not isinstance(trim, bool)
    if not (trim_is_number and 0.0 <= float(trim) < 0.5):
        # at trim >= 0.5 the interval [trim, 1 - trim] is empty or a point
        raise MethodIncompatibility(
            f"ipw: trim must lie in [0, 0.5), got {trim!r}.",
            recovery_hint="Common choices are 0.01, 0.05 and 0.1; trim=0 "
            "leaves the scores as fitted.",
        )
    # n_bootstrap=0 is an explicit request for the point estimate alone
    point_only = se_method == "bootstrap" and n_bootstrap == 0
    if se_method == "bootstrap" and not point_only and not n_bootstrap >= 2:
        raise MethodIncompatibility(
            f"ipw: n_bootstrap must be at least 2, got {n_bootstrap!r}; a "
            "standard deviation needs two replicates.",
            recovery_hint="Use a few hundred replicates, se_method='sandwich', "
            "or n_bootstrap=0 for the point estimate without a standard error.",
        )
    rng = np.random.RandomState(seed)

    # --- Prepare data ---
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
            context="ipw",
            extra_diagnostics={"missing": list(missing_cols.values())},
        )
    used_cols = list(dict.fromkeys([y, treat] + list(covariates) + extra))
    df = data[used_cols].dropna()
    Y = df[y].values.astype(np.float64)
    T = _treatment_as_float(df[treat], function="ipw")
    # A covariate that is a linear combination of the others (and the
    # constant) leaves the propensity likelihood flat in one direction:
    # the sandwich then met a singular Hessian (bare LinAlgError) and each
    # bootstrap refit stopped wherever Newton left it. Omit the later
    # member of each dependent set, as the likelihood estimators do
    # (core/_collinear.py); the fitted scores are unchanged by the drop.
    ps_covariates = list(covariates)
    collinear_omitted: List[Dict[str, str]] = []
    if propensity is None and ps_learner is None:
        from ..core._collinear import drop_collinear_names

        ps_covariates, collinear_omitted = drop_collinear_names(
            df, ps_covariates, "ipw", stacklevel=4
        )
    X = df[ps_covariates].values.astype(np.float64)
    n = len(Y)
    # Sampling weights normalised to mean one; None keeps the unweighted
    # path byte-identical to earlier releases.
    sw: Optional[np.ndarray] = None
    svy_rows = None
    if svy is not None:
        from ..survey.linearization import design_rows

        svy_rows = design_rows(svy, data, used_cols)
        sw = svy_rows.weights * (n / svy_rows.weights.sum())
    elif weights is not None:
        wv = df[weights].to_numpy(dtype=float)
        if not np.all(np.isfinite(wv)) or np.any(wv <= 0):
            raise MethodIncompatibility(
                "ipw: weights must be finite and strictly positive.",
                diagnostics={"weights": weights},
            )
        sw = wv * (n / wv.sum())
    groups: Optional[np.ndarray] = None
    if cluster is not None:
        groups = pd.factorize(df[cluster])[0]
        if groups.max() + 1 < 2:
            raise MethodIncompatibility(
                "ipw: cluster= needs at least two clusters.",
                diagnostics={"cluster": cluster},
            )

    require_binary_treatment(T, function="ipw", argument=treat)
    if T.sum() == 0 or T.sum() == n:
        raise ValueError(
            f"Treatment variable '{treat}' must contain both treated and "
            "control observations"
        )

    # --- Estimate propensity scores (or take the design's) ---
    ps_known = propensity is not None
    fold_ids: Optional[np.ndarray] = None
    if ps_known:
        if isinstance(propensity, str):
            pscore = df[propensity].to_numpy(dtype=float)
        else:
            pscore = np.full(n, float(cast(float, propensity)))
        if not np.all(np.isfinite(pscore)) or np.any((pscore <= 0) | (pscore >= 1)):
            raise MethodIncompatibility(
                "ipw: a known propensity must lie strictly between 0 and 1.",
                diagnostics={
                    "min": float(np.nanmin(pscore)),
                    "max": float(np.nanmax(pscore)),
                },
            )
    elif ps_learner is not None:
        from ._ps_learner import learner_pscore

        fold_ids = rng.choice(n_folds, size=n) if cross_fit else None
        pscore = learner_pscore(ps_learner, X, T, sw, fold_ids)
    else:
        pscore = _estimate_propensity(X, T, sw, ps_model)
    # in-sample learner scores separate the arms by overfitting, which is
    # not the absence of overlap the refusal speaks of
    separation = (
        None
        if ps_known or (ps_learner is not None and not cross_fit)
        else _check_separation(pscore, T, ps_covariates)
    )
    pscore_raw = np.asarray(pscore, dtype=float).copy()  # pre-trim, for overlap

    # --- Trim ---
    if trim > 0:
        pscore = np.clip(pscore, trim, 1 - trim)

    # --- Compute weights ---
    weights_1, weights_0 = _compute_weights(T, pscore, estimand, normalize, sw)

    # --- Point estimate ---
    estimate = float(np.sum(weights_1 * Y) - np.sum(weights_0 * Y))

    # --- How many observations the weights leave each arm with ---
    ess_treated = _kish_ess(weights_1[T == 1])
    ess_control = _kish_ess(weights_0[T == 0])
    n1_arm, n0_arm = int(T.sum()), int(n - T.sum())
    ess_ratio_min = min(ess_treated / n1_arm, ess_control / n0_arm)
    if ess_ratio_min < _ESS_RATIO_WARN:
        thin = "control" if ess_control / n0_arm <= ess_treated / n1_arm else "treated"
        thin_ess, thin_n = (
            (ess_control, n0_arm) if thin == "control" else (ess_treated, n1_arm)
        )
        warnings.warn(
            AssumptionWarning(
                f"ipw: the weights leave the {thin} arm an effective sample "
                f"of {thin_ess:.0f} out of {thin_n} ({thin_ess / thin_n:.0%}); "
                f"propensity scores run from {float(pscore.min()):.3f} to "
                f"{float(pscore.max()):.3f}. The estimate rests on a few "
                "heavily weighted units and can move far from estimators "
                "that model the outcome.",
                recovery_hint=(
                    "Compare with sp.aipw; restrict to the overlap region "
                    "with trim= or sp.trimming; or target the overlap "
                    "population with estimand='ATO'."
                ),
                diagnostics={
                    "ess_treated": ess_treated,
                    "ess_control": ess_control,
                    "n_treated": n1_arm,
                    "n_control": n0_arm,
                    "pscore_min": float(pscore.min()),
                    "pscore_max": float(pscore.max()),
                },
                alternative_functions=["sp.aipw", "sp.trimming", "sp.overlap_weights"],
            ),
            stacklevel=3,
        )

    svy_df: Optional[float] = None
    if svy_rows is not None:
        from ..survey.linearization import design_df, design_vcov

        u_rows = _ipw_sandwich_se(
            X,
            T,
            Y,
            pscore,
            estimand,
            sw,
            None,
            ps_model,
            normalize=normalize,
            e_fit=pscore_raw if trim > 0 else None,
            ps_known=ps_known,
            rows_only=True,
        )
        se = float(np.sqrt(design_vcov(u_rows / n, svy, svy_rows.rows)[0, 0]))
        svy_df = design_df(svy, svy_rows.rows)
        boot_estimates = None
        n_boot_failed = 0
    elif se_method == "sandwich":
        se = _ipw_sandwich_se(
            X,
            T,
            Y,
            pscore,
            estimand,
            sw,
            groups,
            ps_model,
            normalize=normalize,
            e_fit=pscore_raw if trim > 0 else None,
            ps_known=ps_known,
        )
        boot_estimates = None
        n_boot_failed = 0
    else:
        # --- Bootstrap SE (whole clusters when cluster= is given) ---
        boot_estimates = np.empty(n_bootstrap)
        if groups is not None:
            members = [np.flatnonzero(groups == g) for g in range(groups.max() + 1)]
        for b in range(n_bootstrap):
            if groups is None:
                idx = rng.choice(n, size=n, replace=True)
            else:
                pick = rng.choice(len(members), size=len(members), replace=True)
                idx = np.concatenate([members[g] for g in pick])
            Y_b, T_b, X_b = Y[idx], T[idx], X[idx]
            if T_b.min() == T_b.max():
                # A resample with one arm has no contrast: a failed
                # replicate, not ``0 - mean(Y0)``.
                boot_estimates[b] = np.nan
                continue
            sw_b = None if sw is None else sw[idx]
            if ps_known:
                ps_b = pscore[idx]
            elif ps_learner is not None:
                ps_b = learner_pscore(
                    ps_learner,
                    X_b,
                    T_b,
                    sw_b,
                    None if fold_ids is None else fold_ids[idx],
                )
            else:
                ps_b = _estimate_propensity(X_b, T_b, sw_b, ps_model)
            if trim > 0:
                ps_b = np.clip(ps_b, trim, 1 - trim)
            w1, w0 = _compute_weights(T_b, ps_b, estimand, normalize, sw_b)
            boot_estimates[b] = np.sum(w1 * Y_b) - np.sum(w0 * Y_b)

        # SE = sd of the replicates that have both arms (Stata's bootstrap
        # likewise drops the replicates it cannot compute and reports them).
        n_boot_failed = int(np.isnan(boot_estimates).sum())
        if n_boot_failed:
            boot_estimates = boot_estimates[~np.isnan(boot_estimates)]
            # the floor sp.bootstrap applies to its usable replicates
            if len(boot_estimates) < 10:
                raise DataInsufficient(
                    f"ipw: {n_boot_failed} of {n_bootstrap} bootstrap "
                    "resamples contain a single treatment arm and only "
                    f"{len(boot_estimates)} are usable; at least 10 are "
                    "needed for a standard error.",
                    recovery_hint="Raise n_bootstrap, or use "
                    "se_method='sandwich'; with this few units in an arm "
                    "consider a randomization test (sp.fisher_exact).",
                    diagnostics={
                        "n_boot_failed": n_boot_failed,
                        "n_treated": n1_arm,
                        "n_control": n0_arm,
                    },
                )
            warnings.warn(
                f"ipw: {n_boot_failed} of {n_bootstrap} bootstrap resamples "
                "contain a single treatment arm and were excluded; the "
                f"standard error is the spread of the other "
                f"{len(boot_estimates)} and is conditional on both arms "
                "being drawn.",
                RuntimeWarning,
                stacklevel=3,
            )
        se = float("nan") if point_only else float(np.std(boot_estimates, ddof=1))
    t_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (estimate - t_crit * se, estimate + t_crit * se)
    if svy_df is not None:
        from ..survey.linearization import design_inference

        ci, pvalue = design_inference(estimate, se, svy_df, alpha)
    elif point_only:
        pvalue = float("nan")
    else:
        pvalue = float(2 * sp_stats.norm.sf(abs(estimate / se))) if se > 0 else 1.0

    # --- Diagnostics ---
    n_treated = int(T.sum())
    n_control = int(n - n_treated)

    model_info = {
        "model_type": "IPW",
        "estimand": estimand,
        "n_treated": n_treated,
        "n_control": n_control,
        "pscore_mean_treated": float(pscore[T == 1].mean()),
        "pscore_mean_control": float(pscore[T == 0].mean()),
        "pscore_min": float(pscore.min()),
        "pscore_max": float(pscore.max()),
        "ess_treated": ess_treated,
        "ess_control": ess_control,
        "ess_ratio_min": float(ess_ratio_min),
        "trim": trim,
        # Raw propensity distribution so result.violations() can assess overlap
        # (IPW is the most overlap-sensitive estimator). Read via the shared
        # _propensity_extreme_share helper; the trimming bound defaults to 0.01.
        "_pscore": pscore_raw,
        "trimming_threshold": trim,
        "normalized": normalize,
        "n_bootstrap": n_bootstrap if se_method == "bootstrap" else None,
        "se_method": "linearization" if svy is not None else se_method,
        "ps_model": (
            None
            if ps_known
            else (ps_model if ps_learner is None else type(ps_learner).__name__)
        ),
        "propensity": "known" if ps_known else "estimated",
        "weights": weights,
        "cluster": cluster,
        "n_clusters": None if groups is None else int(groups.max() + 1),
    }
    # Recorded only when they occur, so an ordinary fit's model_info is
    # unchanged.
    if n_boot_failed:
        model_info["n_bootstrap_failed"] = n_boot_failed
    if collinear_omitted:
        # same shape as sp.logit / sp.regress: [{variable, reason}]
        model_info["omitted"] = collinear_omitted
    if ps_learner is not None:
        model_info["cross_fit"] = bool(cross_fit)
        model_info["n_folds"] = n_folds if cross_fit else None
    if separation is not None:
        model_info["separation"] = separation
    if svy is not None:
        from ..survey.linearization import design_info

        assert svy_df is not None
        model_info["survey_design"] = design_info(svy, svy_df)

    _result = CausalResult(
        method=f"IPW ({estimand})",
        estimand=estimand,
        estimate=estimate,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.ipw",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(covariates),
                "estimand": estimand,
                "trim": trim,
                "normalize": normalize,
                "n_bootstrap": n_bootstrap,
                "alpha": alpha,
                "seed": seed,
                "weights": weights,
                "cluster": cluster,
                "se_method": se_method,
                "ps_model": ps_model,
                "propensity": propensity,
                "design": None if design is None else repr(design),
                "ps_learner": (
                    None if ps_learner is None else type(ps_learner).__name__
                ),
                "cross_fit": cross_fit,
                "n_folds": n_folds,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

#: An arm whose Kish effective sample size is below this share of its
#: size draws a warning. Under a constant propensity the share is one; at
#: 0.2 four fifths of the arm's information is gone to weight dispersion.
_ESS_RATIO_WARN = 0.2

#: Fitted scores are clipped to ``[_PS_BOUND, 1 - _PS_BOUND]``; a score at
#: the bound is 0 or 1 to the precision the weights can use.
_PS_BOUND = 1e-8


def _kish_ess(w: np.ndarray) -> float:
    """Kish's effective sample size of a weight vector, ``(sum w)^2 / sum w^2``."""
    total = float(np.sum(w))
    squares = float(np.sum(w**2))
    return total * total / squares if squares > 0 else 0.0


def _check_separation(
    pscore: np.ndarray, T: np.ndarray, covariates: List[str]
) -> Optional[str]:
    """Refuse complete separation, warn on quasi-complete separation.

    The two signatures are those of ``regression.logit_probit``: the fitted
    index classifies every unit correctly (complete separation: the
    likelihood has no maximum and the fitted scores are 0 / 1 up to the
    optimiser's stopping rule), or some fitted scores sit at the numerical
    bound although the two arms are not separated (quasi-complete, e.g. a
    category in which only one arm occurs). Under complete separation no
    unit has a counterpart with a comparable score, so the weighted
    contrast is the raw difference in means and is refused; the weaker
    case leaves an estimate that rests on clipped weights and is flagged.
    """
    if np.array_equal(pscore >= 0.5, T == 1):
        raise DataInsufficient(
            "ipw: the covariates separate the treated from the control units "
            "completely (the fitted propensity index classifies every unit "
            "correctly). The propensity model has no finite maximum-"
            "likelihood fit and there is no overlap to weight on.",
            recovery_hint="Drop or coarsen the separating covariate, restrict "
            "the sample to a region where both arms occur, or pass a known "
            "design probability with propensity=.",
            diagnostics={"covariates": list(covariates)},
            alternative_functions=["sp.overlap_plot", "sp.trimming"],
        )
    at_bound = int(np.sum((pscore <= _PS_BOUND) | (pscore >= 1 - _PS_BOUND)))
    if at_bound == 0:
        return None
    warnings.warn(
        ConvergenceWarning(
            f"ipw: quasi-complete separation suspected: {at_bound} fitted "
            "propensity scores are numerically 0 or 1 (clipped to "
            f"{_PS_BOUND:g} / 1 - {_PS_BOUND:g}). Those units have no "
            "counterpart in the other arm and their weights are set by the "
            "clipping bound, not by the data.",
            recovery_hint="Inspect the covariates for a category or range in "
            "which only one arm occurs; drop those units (sp.trimming) or "
            "the covariate.",
            diagnostics={"n_pscore_at_bound": at_bound},
        ),
        stacklevel=4,
    )
    return "quasi-complete"


def _binomial_family(ps_model: str) -> Any:
    import statsmodels.api as sm

    if ps_model == "probit":
        return sm.families.Binomial(link=sm.families.links.Probit())
    return sm.families.Binomial()


def _estimate_propensity(
    X: np.ndarray,
    T: np.ndarray,
    sw: Optional[np.ndarray] = None,
    ps_model: str = "logit",
) -> np.ndarray:
    """Logit or probit propensity score (weighted MLE if ``sw``)."""
    if sw is not None:
        import statsmodels.api as sm

        X_const = sm.add_constant(X, has_constant="add")
        # freq_weights gives the pweighted likelihood's point estimates.
        res = sm.GLM(
            T, X_const, family=_binomial_family(ps_model), freq_weights=sw
        ).fit(tol=1e-12, maxiter=300)
        return np.clip(np.asarray(res.predict(X_const), dtype=float), 1e-8, 1 - 1e-8)
    if ps_model == "probit":
        import statsmodels.api as sm

        X_const = sm.add_constant(X, has_constant="add")
        res = sm.GLM(T, X_const, family=_binomial_family(ps_model)).fit(
            tol=1e-12, maxiter=300
        )
        if getattr(res, "converged", True) is False:
            raise MethodIncompatibility(
                "ipw: the probit propensity model did not converge.",
                recovery_hint="Check for separation or badly scaled "
                "covariates, or use ps_model='logit'.",
            )
        return np.clip(np.asarray(res.predict(X_const), dtype=float), 1e-8, 1 - 1e-8)
    try:
        import statsmodels.api as sm

        X_const = sm.add_constant(X, has_constant="add")
        model = sm.GLM(T, X_const, family=sm.families.Binomial())
        res = model.fit(maxiter=300, disp=0)
        if getattr(res, "converged", True) is False:
            raise RuntimeError("statsmodels GLM did not converge")
        ps = np.asarray(res.predict(X_const), dtype=float)
    except Exception as exc:
        from ..core._fallback import warn_fallback

        warn_fallback(
            "IPW propensity logit (statsmodels GLM)",
            exc,
            "refitting with scikit-learn's unpenalised logistic "
            "regression; check for separation or collinear covariates",
        )
        from sklearn.linear_model import LogisticRegression

        try:
            model = LogisticRegression(
                max_iter=5000,
                solver="newton-cg",
                penalty=None,
                tol=1e-10,
            )
        except TypeError:  # pragma: no cover - old scikit-learn compatibility
            model = LogisticRegression(
                max_iter=5000,
                solver="newton-cg",
                penalty="none",
                tol=1e-10,
            )
        model.fit(X, T)
        ps = model.predict_proba(X)[:, 1]
    # Safety clip to avoid division by zero
    return np.clip(ps, 1e-8, 1 - 1e-8)


def _ipw_sandwich_se(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    e: np.ndarray,
    estimand: str,
    sw: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    ps_model: str = "logit",
    ps_known: bool = False,
    normalize: bool = True,
    e_fit: Optional[np.ndarray] = None,
    rows_only: bool = False,
) -> Any:
    """M-estimation SE of the IPW contrast (Stata ``teffects ipw``).

    ``normalize=False`` is the Horvitz-Thompson estimator
    ``sum(w (a_1 - a_0) Y) / sum(w t)`` with ``t`` the indicator of the
    target population; row ``i`` of its influence function is
    ``[w (a_1 - a_0) Y - tau w t + G' IF_gamma] / mean(w t)``,
    ``G = mean(w Y d(a_1 - a_0) / d gamma)``.

    ``e_fit`` holds the fitted scores when ``e`` has been clipped to
    ``[trim, 1 - trim]``. The propensity score equations are those of the
    fitted model, so their score and Hessian use ``e_fit``; a clipped
    score does not move with the coefficients, so ``d a_k / d gamma`` is
    zero on the clipped rows.

    Stacks the (weighted) logit score ``w (T - e) x`` with the two Hajek
    means ``w a_k (Y - mu_k) = 0``, where ``a_1 = T/e, a_0 = (1-T)/(1-e)``
    (ATE), ``T, (1-T) e/(1-e)`` (ATT), ``T (1-e)/e, 1-T`` (ATC),
    ``T (1-e), (1-T) e`` (ATO) or ``T min(1, (1-e)/e), (1-T) min(1, e/(1-e))``
    (ATM; its kink at ``e = 1/2`` has probability zero). Row ``i``
    of the influence function of ``mu_k`` is
    ``[w a_k (Y - mu_k) + G_k' IF_gamma] / mean(w a_k)`` with
    ``G_k = mean(w (Y - mu_k) d a_k / d gamma)`` and
    ``IF_gamma = H^{-1} w (T - e) x``, ``H = mean(w e (1-e) x x')``.
    Divisor ``n``; with clusters the rows are summed within clusters first.

    For a probit the score is ``w lam x`` with ``lam = (T - e) f / (e (1-e))``
    and ``f`` the normal density at the index, ``H`` is the observed Hessian
    ``mean(w lam (lam + index) x x')`` and ``e (1-e)`` in ``d a_k / d gamma``
    becomes ``f``.

    With ``ps_known`` the propensity is a design quantity, there is no
    first-stage score to stack, and the ``G_k' IF_gamma`` term is dropped.

    ``rows_only=True`` returns the influence rows ``u`` themselves
    (``tau_hat - tau ~= sum(u) / n``, sampling weight inside) instead of
    the SE, for the design-based variance of ``design=``.
    """
    n = len(Y)
    w = np.ones(n) if sw is None else sw
    Xc = np.column_stack([np.ones(n), X])
    odds = e / (1 - e)
    ef = e if e_fit is None else e_fit  # the fitted, unclipped scores
    free = np.ones(n) if e_fit is None else (e_fit == e).astype(float)
    if ps_model == "probit":
        index = sp_stats.norm.ppf(ef)
        dens = sp_stats.norm.pdf(index)  # d e / d index
        score = (T - ef) * dens / (ef * (1 - ef))
        curv = score * (score + index)  # minus the observed second derivative
    else:
        dens = ef * (1 - ef)
        score = T - ef
        curv = dens
    # d(1/e) and d(e/(1-e)) with respect to the index; the logit forms are
    # written out so that path keeps its earlier floating-point result
    if ps_model == "probit":
        d_inv, d_odds = -T * dens / e**2, (1 - T) * dens / (1 - e) ** 2
    else:
        d_inv, d_odds = -T * (1 - e) / e, (1 - T) * odds
    if estimand == "ATE":
        a1, a0 = T / e, (1 - T) / (1 - e)
        da1, da0 = d_inv, d_odds
    elif estimand == "ATT":
        a1, a0 = T, (1 - T) * odds
        da1, da0 = np.zeros(n), d_odds
    elif estimand == "ATC":
        a1, a0 = T * (1 - e) / e, 1 - T
        da1, da0 = d_inv, np.zeros(n)
    elif estimand == "ATO":
        a1, a0 = T * (1 - e), (1 - T) * e
        da1, da0 = -T * dens, (1 - T) * dens
    else:  # ATM
        low = e < 0.5
        a1 = T * np.where(low, 1.0, (1 - e) / e)
        a0 = (1 - T) * np.where(low, odds, 1.0)
        da1 = np.where(low, 0.0, d_inv)
        da0 = np.where(low, d_odds, 0.0)
    if e_fit is not None:
        da1, da0 = da1 * free, da0 * free
    H = (Xc * (w * curv)[:, None]).T @ Xc / n
    if_gamma = np.linalg.solve(H, (Xc * (w * score)[:, None]).T).T

    if not normalize:
        target = {"ATT": T, "ATC": 1 - T}.get(estimand, np.ones(n))
        size = float(np.mean(w * target))
        tau = float(np.mean(w * (a1 - a0) * Y)) / size
        u_ht = w * (a1 - a0) * Y - tau * w * target
        if not ps_known:  # a design probability has no estimation term
            G_ht = (Xc * (w * Y * (da1 - da0))[:, None]).mean(axis=0)
            u_ht = u_ht + if_gamma @ G_ht
        u_ht = u_ht / size
        if rows_only:
            return u_ht
        if groups is not None:
            u_ht = np.bincount(groups, weights=u_ht)
        return float(np.sqrt(np.sum(u_ht**2)) / n)

    def _if(a: np.ndarray, da: np.ndarray) -> np.ndarray:
        mu = np.sum(w * a * Y) / np.sum(w * a)
        resid = Y - mu
        if ps_known:
            return np.asarray(w * a * resid / np.mean(w * a))
        G = (Xc * (w * resid * da)[:, None]).mean(axis=0)
        rows: np.ndarray = (w * a * resid + if_gamma @ G) / np.mean(w * a)
        return rows

    u = _if(a1, da1) - _if(a0, da0)
    if rows_only:
        return u
    if groups is not None:
        u = np.bincount(groups, weights=u)
    return float(np.sqrt(np.sum(u**2)) / n)


def _compute_weights(
    T: np.ndarray,
    pscore: np.ndarray,
    estimand: str,
    normalize: bool,
    sw: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute IPW weights for treated (w1) and control (w0) groups.

    Returns (weights_1, weights_0) such that:
        estimate = sum(w1 * Y) - sum(w0 * Y)

    Sampling weights ``sw`` (mean one) multiply each unit's IPW weight.
    """
    if sw is not None:
        w1, w0 = _raw_weights(T, pscore, estimand)
        w1, w0 = w1 * sw, w0 * sw
        if normalize:
            return w1 / w1.sum(), w0 / w0.sum()
        denom = _target_size(T, estimand, sw)
        return w1 / denom, w0 / denom

    w1, w0 = _raw_weights(T, pscore, estimand)
    if normalize:
        s1 = w1.sum()
        s0 = w0.sum()
        if s1 > 0:
            w1 = w1 / s1
        if s0 > 0:
            w0 = w0 / s0
    else:
        # Horvitz-Thompson: divide by the size of the target population --
        # n for the ATE, the number of treated for the ATT and of controls
        # for the ATC. Dividing the ATT / ATC weights by n (as was done
        # through 1.38.0) returned P(T=1) x ATT and P(T=0) x ATC.
        denom = _target_size(T, estimand, None)
        w1 = w1 / denom
        w0 = w0 / denom

    return w1, w0


def _target_size(T: np.ndarray, estimand: str, sw: Optional[np.ndarray]) -> float:
    """(Weighted) number of units in the population the estimand averages over."""
    one = np.ones(len(T)) if sw is None else sw
    if estimand == "ATT":
        return float(np.sum(one * T))
    if estimand == "ATC":
        return float(np.sum(one * (1 - T)))
    return float(np.sum(one))


def _raw_weights(
    T: np.ndarray, pscore: np.ndarray, estimand: str
) -> tuple[np.ndarray, np.ndarray]:
    """Unscaled IPW weights of the treated and control terms."""

    if estimand == "ATE":
        # Horvitz-Thompson: w1 = T/p, w0 = (1-T)/(1-p)
        w1 = T / pscore
        w0 = (1 - T) / (1 - pscore)
    elif estimand == "ATT":
        # ATT: treated get weight 1, controls get weight p/(1-p)
        w1 = T.copy()
        w0 = (1 - T) * pscore / (1 - pscore)
    elif estimand == "ATC":
        # ATC: controls get weight 1, treated get weight (1-p)/p
        w1 = T * (1 - pscore) / pscore
        w0 = (1 - T).copy()
    elif estimand == "ATO":
        # overlap weights: the probability of the other arm
        w1 = T * (1 - pscore)
        w0 = (1 - T) * pscore
    elif estimand == "ATM":
        # matching weights: min(p, 1-p) over the probability of the own arm
        tilt = np.minimum(pscore, 1 - pscore)
        w1 = T * tilt / pscore
        w0 = (1 - T) * tilt / (1 - pscore)
    return w1, w0
