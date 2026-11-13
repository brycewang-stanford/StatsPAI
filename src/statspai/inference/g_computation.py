"""
G-computation (standardization) estimator for causal effects.

Parametric g-formula for a point-exposure setting: fits an outcome
model :math:`Q(D, X) = E[Y | D, X]`, then standardizes by averaging
predictions under counterfactual treatment values.

    ATE = E[ Q(1, X) - Q(0, X) ]                     (binary D)
    ATT = E[ Q(1, X) - Q(0, X) | D = 1 ]
    dose-response(d) = E[ Q(d, X) ]                  (continuous D)

This is the simplest member of Robins' g-methods family
(g-computation / g-standardization / g-formula). It is fully parametric
and is consistent only when the outcome model is correctly specified —
see AIPW / TMLE for doubly-robust alternatives.

Uses nonparametric bootstrap for inference by default.

References
----------
Robins, J. (1986). "A new approach to causal inference in mortality
studies with a sustained exposure period — application to control of
the healthy worker survivor effect." *Mathematical Modelling*, 7(9-12),
1393-1512. [@robins1986new]

Snowden, J.M., Rose, S. and Mortimer, K.M. (2011). "Implementation of
G-computation on a Simulated Data Set: Demonstration of a Causal
Inference Technique." *American Journal of Epidemiology*, 173(7),
731-738. [@snowden2011implementation]

Hernán, M.A. and Robins, J.M. (2020). *Causal Inference: What If.*
Chapman & Hall/CRC. Chapter 13.
"""

import warnings
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core._validate import treatment_as_float as _treatment_as_float
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from .basic_tests import _check_alpha


@_expands_categorical("covariates")
def g_computation(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    estimand: str = "ATE",
    treat_values: Optional[Sequence[float]] = None,
    ml_Q: Optional[Any] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    *,
    by_arm: bool = False,
    ps_covariates: Optional[List[str]] = None,
    se_method: str = "bootstrap",
    outcome_model: str = "linear",
    contrast: str = "difference",
) -> CausalResult:
    """
    G-computation (parametric g-formula) estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable name.
    treat : str
        Treatment variable. Binary (0/1), discrete, or continuous.
    covariates : list of str
        Baseline covariates to adjust for.
    estimand : {'ATE', 'ATT', 'ATC', 'dose_response'}, default 'ATE'
        - ``'ATE'``: E[Q(1,X) - Q(0,X)] (requires binary D)
        - ``'ATT'``: E[Q(1,X) - Q(0,X) | D=1]
        - ``'ATC'``: E[Q(1,X) - Q(0,X) | D=0], the effect on the untreated
          (``'ATU'`` accepted; not with ``by_arm=True``). The three
          coincide when the outcome model has no treatment-covariate
          interaction and is linear.
        - ``'dose_response'``: grid of E[Q(d,X)] over ``treat_values``
    treat_values : sequence of float, optional
        Treatment levels at which to compute the dose-response curve.
        Required when ``estimand='dose_response'``. Ignored otherwise.
    ml_Q : sklearn-compatible estimator, optional
        Outcome model. Defaults to OLS via statsmodels for interpretability;
        pass any estimator with ``.fit(X, y)`` and ``.predict(X)`` for
        flexible fits (gradient boosting, random forests, etc.).
    n_boot : int, default 500
        Nonparametric bootstrap replications for SE/CI.
    alpha : float, default 0.05
        Significance level.
    seed : int, optional
        Random seed.

    Returns
    -------
    CausalResult
        For ATE/ATT, ``estimate`` is the contrast; for dose-response,
        ``estimate`` is E[Q(d,X)] at the first ``treat_values`` level
        and the full curve lives in ``detail`` and ``model_info['curve']``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> age = rng.normal(40, 10, n)
    >>> edu = rng.normal(12, 3, n)
    >>> trained = (rng.random(n) < 0.5).astype(int)
    >>> wage = 20 + 2.0 * trained + 0.3 * age + 0.5 * edu + rng.normal(0, 3, n)
    >>> df = pd.DataFrame({'wage': wage, 'trained': trained,
    ...                    'age': age, 'edu': edu})

    >>> # Binary treatment ATE
    >>> res = sp.g_computation(df, y='wage', treat='trained',
    ...                        covariates=['age', 'edu'], n_boot=100, seed=0)
    >>> bool(res.estimate is not None)
    True

    >>> # ATT
    >>> res_att = sp.g_computation(df, y='wage', treat='trained',
    ...                            covariates=['age', 'edu'], estimand='ATT',
    ...                            n_boot=100, seed=0)
    >>> bool(res_att.estimate is not None)
    True

    >>> # Dose-response over a continuous dose
    >>> dose = rng.uniform(0, 30, n)
    >>> bp = 120 - 0.4 * dose + 0.2 * age + rng.normal(0, 5, n)
    >>> ddf = pd.DataFrame({'bp': bp, 'dose': dose, 'age': age})
    >>> curve = sp.g_computation(ddf, y='bp', treat='dose', covariates=['age'],
    ...                          estimand='dose_response',
    ...                          treat_values=[0, 10, 20, 30], n_boot=100, seed=0)
    >>> bool('curve' in curve.model_info)
    True

    by_arm : bool, default False
        Fit a linear outcome regression in each treatment arm instead of
        one regression with the treatment as a regressor, and average the
        difference of the two predictions. This is regression adjustment
        as Stata's ``teffects ra`` defines it; it lets every coefficient
        differ between the arms. Binary treatment, ``estimand`` ``'ATE'``
        or ``'ATT'``, no ``ml_Q``.
    ps_covariates : list of str, optional
        With ``by_arm=True``: fit the two regressions by weighted least
        squares, with inverse-probability weights from a logit of the
        treatment on these covariates. This is Stata's ``teffects ipwra``,
        inverse-probability-weighted regression adjustment, which is
        doubly robust: consistent when either the outcome regressions or
        the logit is right.
    se_method : {'bootstrap', 'analytic'}, default 'bootstrap'
        ``'analytic'`` (with ``by_arm=True``) is the sandwich variance of
        the stacked estimating equations, so the sampling error of the
        regression and logit coefficients is carried into the effect; it
        is the standard error ``teffects ra`` / ``teffects ipwra`` report,
        and the confidence interval is the normal one. ``'bootstrap'``
        resamples rows ``n_boot`` times.
    outcome_model : {'linear', 'logit', 'poisson'}, default 'linear'
        Outcome regression when ``ml_Q`` is not given: least squares, a
        logistic regression for a 0/1 outcome, or a Poisson regression
        (log link) for a count or other non-negative outcome. Predictions
        are on the scale of the mean (a risk, an expected count). With
        ``by_arm=True`` this is Stata's ``teffects ra (y x, logit)`` /
        ``(y x, poisson)``, and ``se_method='analytic'`` its standard
        error; without it, it is the standardisation of one model that has
        the treatment as a regressor (R
        ``marginaleffects::avg_comparisons`` of a ``glm``).
    contrast : {'difference', 'ratio', 'odds_ratio'}, default 'difference'
        How the two potential-outcome means are compared: ``E[Y(1)] -
        E[Y(0)]`` (a risk difference for a 0/1 outcome), their ratio (risk
        ratio, rate ratio) or the ratio of their odds (a marginal odds
        ratio, for means inside (0, 1)). For the ratio measures the
        estimate and its standard error are on the natural scale, the
        analytic interval is built on the log scale and exponentiated, and
        the p-value tests a ratio of one; ``model_info`` holds
        ``log_estimate`` and ``log_se``. Not for
        ``estimand='dose_response'``.

    Notes
    -----
    Consistent when :math:`Q(d, X) = E[Y|D=d, X]` is correctly specified
    and positivity holds over the treatment levels of interest. Not
    doubly robust — see :func:`sp.aipw` or :func:`sp.tmle` for DR
    alternatives, or :func:`sp.dml` with ``model='plr'/'irm'`` for
    ML-based orthogonalization.
    """
    if estimand == "ATU":
        estimand = "ATC"
    if estimand not in ("ATE", "ATT", "ATC", "dose_response"):
        raise ValueError(
            "estimand must be 'ATE', 'ATT', 'ATC', or 'dose_response'; "
            f"got '{estimand}'"
        )

    if se_method not in ("bootstrap", "analytic"):
        raise MethodIncompatibility(
            f"g_computation: se_method must be 'bootstrap' or 'analytic'; "
            f"got {se_method!r}."
        )
    _check_alpha(alpha, "g_computation")
    from ._glm_adjustment import CONTRASTS, OUTCOME_MODELS

    if outcome_model not in OUTCOME_MODELS:
        raise MethodIncompatibility(
            f"g_computation: outcome_model must be one of {list(OUTCOME_MODELS)}; "
            f"got {outcome_model!r}.",
            recovery_hint="Use 'linear', 'logit' (0/1 outcome) or 'poisson' "
            "(counts); pass ml_Q= for any other learner.",
        )
    if contrast not in CONTRASTS:
        raise MethodIncompatibility(
            f"g_computation: contrast must be one of {list(CONTRASTS)}; "
            f"got {contrast!r}.",
            recovery_hint="Use 'difference', 'ratio' or 'odds_ratio'.",
        )
    if outcome_model != "linear" and ml_Q is not None:
        raise MethodIncompatibility(
            "g_computation: outcome_model= and ml_Q= both name the outcome " "model.",
            recovery_hint="Pass one of them.",
        )
    if contrast != "difference" and estimand == "dose_response":
        raise MethodIncompatibility(
            "g_computation: contrast= compares the two arms of a binary "
            "treatment; a dose-response curve reports E[Y(d)] at each dose.",
            recovery_hint="Drop contrast=, or use estimand 'ATE' / 'ATT' / 'ATC'.",
        )
    if by_arm:
        return _by_arm(
            data,
            y,
            treat,
            list(covariates),
            estimand=estimand,
            ml_Q=ml_Q,
            outcome_model=outcome_model,
            contrast=contrast,
            ps_covariates=ps_covariates,
            se_method=se_method,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )
    if ps_covariates is not None or se_method == "analytic":
        raise MethodIncompatibility(
            "g_computation: ps_covariates= and se_method='analytic' belong to "
            "the regression fitted in each treatment arm.",
            recovery_hint="Add by_arm=True.",
        )

    df = data[[y, treat] + list(covariates)].dropna().reset_index(drop=True)
    Y = df[y].values.astype(float)
    D = _treatment_as_float(df[treat], function="g_computation")
    X = df[covariates].values.astype(float)
    n = len(Y)
    if outcome_model != "linear":
        from ._glm_adjustment import check_outcome, fit_glm, glm_mean

        check_outcome(Y, outcome_model)

    if estimand in ("ATE", "ATT", "ATC"):
        uniq = set(np.unique(D))
        if not uniq.issubset({0, 1}):
            # Truncate the value dump: a continuous treatment can have
            # hundreds of unique values and the full list turns the
            # error message into a ~20 KB wall of floats — this
            # explodes warning-message size when a caller (e.g. the
            # compare_estimators batch loop) wraps the ValueError in
            # a UserWarning. Filter NaNs before sorting (a lone NaN
            # would sit in a non-deterministic position in Python's
            # sort and omit real values from the preview).
            finite = [v for v in uniq if not (isinstance(v, float) and np.isnan(v))]
            has_nan = len(finite) < len(uniq)
            vals = sorted(finite)
            preview = ", ".join(f"{v:.4g}" for v in vals[:5])
            tail = f", ... ({len(vals) - 5} more)" if len(vals) > 5 else ""
            nan_note = " (plus NaN values present)" if has_nan else ""
            raise ValueError(
                f"estimand='{estimand}' requires binary treatment (0/1); "
                f"treatment has {len(vals)} unique values "
                f"[{preview}{tail}]{nan_note}. For multi-valued or "
                f"continuous D, use estimand='dose_response'."
            )
        n_arm1 = int(np.sum(D == 1))
        if n_arm1 == 0 or n_arm1 == n:
            raise DataInsufficient(
                f"g_computation: {treat!r} has {n_arm1} treated and "
                f"{n - n_arm1} control units on the complete rows; "
                f"estimand={estimand!r} contrasts the two arms and needs both.",
                diagnostics={"n_treated": n_arm1, "n_control": n - n_arm1},
            )
        grid = np.array([0.0, 1.0])
    else:
        if treat_values is None or len(treat_values) == 0:
            raise ValueError(
                "estimand='dose_response' requires 'treat_values' "
                "(sequence of dose levels to evaluate)."
            )
        grid = np.asarray(treat_values, dtype=float)

    def _fit_and_predict(
        Y_: np.ndarray,
        D_: np.ndarray,
        X_: np.ndarray,
        grid_: np.ndarray,
    ) -> np.ndarray:
        """Fit Q(D,X) on (Y_, D_, X_), return n_obs × len(grid_) predictions."""
        design = np.column_stack([D_.reshape(-1, 1), X_])
        if outcome_model != "linear":
            ones = np.ones((X_.shape[0], 1))
            coef = fit_glm(
                Y_, np.column_stack([ones, design]), outcome_model, "g_computation"
            )
            preds = np.empty((X_.shape[0], len(grid_)))
            for k, d_val in enumerate(grid_):
                eta = np.column_stack([ones, np.full_like(ones, d_val), X_]) @ coef
                preds[:, k] = glm_mean(eta, outcome_model)[0]
            return preds
        if ml_Q is None:
            import statsmodels.api as sm

            # has_constant="add": a constant column must not stand in for
            # the intercept that the prediction design below always gets
            fit = sm.OLS(Y_, sm.add_constant(design, has_constant="add")).fit()
            preds = np.empty((X_.shape[0], len(grid_)))
            for k, d_val in enumerate(grid_):
                d_col = np.full((X_.shape[0], 1), d_val)
                new_design = np.column_stack([d_col, X_])
                preds[:, k] = fit.predict(
                    sm.add_constant(new_design, has_constant="add")
                )
            return preds
        else:
            from sklearn.base import clone

            model = clone(ml_Q)
            model.fit(design, Y_)
            preds = np.empty((X_.shape[0], len(grid_)))
            for k, d_val in enumerate(grid_):
                d_col = np.full((X_.shape[0], 1), d_val)
                new_design = np.column_stack([d_col, X_])
                preds[:, k] = model.predict(new_design)
            return preds

    def _point_estimates(
        Y_: np.ndarray,
        D_: np.ndarray,
        X_: np.ndarray,
    ) -> np.ndarray:
        preds = _fit_and_predict(Y_, D_, X_, grid)
        if contrast != "difference":
            if estimand == "ATE":
                target = np.ones(len(D_), dtype=bool)
            else:
                target = D_ == (1 if estimand == "ATT" else 0)
            if target.sum() == 0:
                return np.array([np.nan])
            m0 = float(np.mean(preds[target, 0]))
            m1 = float(np.mean(preds[target, 1]))
            if contrast == "ratio":
                return np.array([m1 / m0 if m0 > 0 and m1 > 0 else np.nan])
            if not (0 < m0 < 1 and 0 < m1 < 1):
                return np.array([np.nan])
            return np.array([(m1 / (1 - m1)) / (m0 / (1 - m0))])
        if estimand == "ATE":
            return np.array([float(np.mean(preds[:, 1] - preds[:, 0]))])
        if estimand in ("ATT", "ATC"):
            treated_mask = D_ == (1 if estimand == "ATT" else 0)
            if treated_mask.sum() == 0:
                return np.array([np.nan])
            return np.array(
                [float(np.mean(preds[treated_mask, 1] - preds[treated_mask, 0]))]
            )
        # dose_response
        return np.asarray(preds.mean(axis=0), dtype=float)

    point = _point_estimates(Y, D, X)
    binary_contrast = estimand in ("ATE", "ATT", "ATC")
    if contrast != "difference" and not np.isfinite(point[0]):
        raise MethodIncompatibility(
            f"g_computation: contrast={contrast!r} is not defined for these "
            "standardised means (a ratio needs positive means, an odds ratio "
            "means strictly between 0 and 1).",
            recovery_hint="Use contrast='difference', or outcome_model='logit' "
            "/ 'poisson' so that the predictions stay in range.",
        )

    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, len(point)), np.nan)
    n_failed = 0
    first_err: Optional[str] = None
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            if binary_contrast and np.unique(D[idx]).size < 2:
                # no contrast of arms can be fitted on one arm
                raise DataInsufficient("the resample contains a single treatment arm")
            boot[b] = _point_estimates(Y[idx], D[idx], X[idx])
        except Exception as e:
            n_failed += 1
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
            # Leave row as NaN so variance is computed only over successes

    n_success = n_boot - n_failed
    if n_success < 2:
        raise RuntimeError(
            f"G-computation bootstrap failed on {n_failed}/{n_boot} replications "
            f"(only {n_success} succeeded; need ≥2 for SE). "
            f"First error: {first_err}. "
            f"Check for multicollinearity, small-cell issues, or treatment "
            f"support problems in your data."
        )
    if n_failed > 0:
        frac = n_failed / n_boot
        warnings.warn(
            f"G-computation: {n_failed}/{n_boot} bootstrap replications "
            f"failed ({frac:.1%}). SE/CI computed over {n_success} successes. "
            f"First error: {first_err}.",
            RuntimeWarning,
            stacklevel=2,
        )

    se = np.nanstd(boot, axis=0, ddof=1)
    lo_q = 100 * (alpha / 2)
    hi_q = 100 * (1 - alpha / 2)
    ci_lo = np.nanpercentile(boot, lo_q, axis=0)
    ci_hi = np.nanpercentile(boot, hi_q, axis=0)

    # Wald p-value on each grid point (dose-response reports a curve)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, point / se, 0.0)
    pvalue = 2 * stats.norm.sf(np.abs(z))

    model_info = {
        "estimator": "G-computation (parametric g-formula)",
        "estimand": estimand,
        "n_boot": n_boot,
        "n_boot_failed": n_failed,
        "n_boot_success": n_success,
        "ml_Q": type(ml_Q).__name__ if ml_Q is not None else "OLS",
        "grid": grid.tolist(),
    }
    if outcome_model != "linear":
        model_info["outcome_model"] = outcome_model
        model_info["ml_Q"] = {"logit": "Logit", "poisson": "Poisson"}[outcome_model]
    if contrast != "difference":
        # a ratio is tested against one, on the log scale
        with np.errstate(divide="ignore", invalid="ignore"):
            log_se = float(np.nanstd(np.log(boot[:, 0]), ddof=1))
        log_est = float(np.log(point[0]))
        pvalue = np.array([2 * stats.norm.sf(abs(log_est / log_se))])
        model_info.update(contrast=contrast, log_estimate=log_est, log_se=log_se)
    if n_failed > 0:
        model_info["first_bootstrap_error"] = first_err

    if estimand == "dose_response":
        detail = pd.DataFrame(
            {
                "dose": grid,
                "estimate": point,
                "se": se,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "pvalue": pvalue,
            }
        )
        model_info["curve"] = detail.copy()
        # Report first-dose point as the summary estimate slot
        summary_est = float(point[0])
        summary_se = float(se[0])
        summary_pv = float(pvalue[0])
        summary_ci = (float(ci_lo[0]), float(ci_hi[0]))
        label = "E[Y(d)]"
    else:
        detail = None
        summary_est = float(point[0])
        summary_se = float(se[0])
        summary_pv = float(pvalue[0])
        summary_ci = (float(ci_lo[0]), float(ci_hi[0]))
        label = estimand + _CONTRAST_LABEL[contrast]

    _result = CausalResult(
        method="G-computation",
        estimand=label,
        estimate=summary_est,
        se=summary_se,
        pvalue=summary_pv,
        ci=summary_ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key="g_computation",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.g_computation",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(covariates),
                "estimand": estimand,
                "treat_values": (list(treat_values) if treat_values else None),
                "ml_Q": type(ml_Q).__name__ if ml_Q is not None else None,
                "n_boot": n_boot,
                "alpha": alpha,
                "seed": seed,
                "outcome_model": outcome_model,
                "contrast": contrast,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


#: Suffix of the estimand label for each contrast.
_CONTRAST_LABEL = {"difference": "", "ratio": " (ratio)", "odds_ratio": " (odds ratio)"}


def _by_arm_glm(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    *,
    estimand: str,
    outcome_model: str,
    contrast: str,
    se_method: str,
    n_boot: int,
    alpha: float,
    seed: Optional[int],
) -> CausalResult:
    """Per-arm logistic / Poisson (or linear, for a ratio) outcome models."""
    from ._glm_adjustment import check_outcome, contrast_of_means, glm_by_arm

    check_outcome(Y, outcome_model)
    n = len(Y)
    fit = glm_by_arm(Y, D, X, estimand=estimand, model=outcome_model)
    pom, psi = fit["pom"], fit["psi"]
    est, se, log_est, log_se = contrast_of_means(pom, psi, contrast)
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    info: dict = {
        "estimator": "Regression adjustment (outcome regression per arm)",
        "estimand": estimand,
        "by_arm": True,
        "ps_covariates": None,
        "se_method": se_method,
        "outcome_model": outcome_model,
        "contrast": contrast,
        "pomean0": float(pom[0]),
        "pomean1": float(pom[1]),
        "pomean0_se": float(np.sqrt(np.mean(psi[:, 0] ** 2) / n)),
        "pomean1_se": float(np.sqrt(np.mean(psi[:, 1] ** 2) / n)),
        "ml_Q": {"linear": "OLS", "logit": "Logit", "poisson": "Poisson"}[
            outcome_model
        ],
    }
    ratio = contrast != "difference"
    if ratio:
        info.update(log_estimate=log_est, log_se=log_se)
    if se_method == "bootstrap":
        rng = np.random.default_rng(seed)
        draws = np.full(n_boot, np.nan)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                bfit = glm_by_arm(
                    Y[idx], D[idx], X[idx], estimand=estimand, model=outcome_model
                )
                draws[b] = contrast_of_means(bfit["pom"], bfit["psi"], contrast)[0]
            except (MethodIncompatibility, RuntimeError, np.linalg.LinAlgError):
                continue  # a resample without support; counted below
        ok = int(np.isfinite(draws).sum())
        if ok < 2:
            raise RuntimeError(
                "g_computation(by_arm=True): the bootstrap failed on "
                f"{n_boot - ok} of {n_boot} resamples; use se_method='analytic'."
            )
        se = float(np.nanstd(draws, ddof=1))
        ci = (
            float(np.nanpercentile(draws, 100 * alpha / 2)),
            float(np.nanpercentile(draws, 100 * (1 - alpha / 2))),
        )
        info.update(n_boot=n_boot, n_boot_success=ok, n_boot_failed=n_boot - ok)
        info["pomean0_se"] = info["pomean1_se"] = float("nan")
        if ratio:
            log_se = float(np.nanstd(np.log(draws), ddof=1))
            info["log_se"] = log_se
    elif ratio:
        ci = (
            float(np.exp(log_est - z_crit * log_se)),
            float(np.exp(log_est + z_crit * log_se)),
        )
    else:
        ci = (est - z_crit * se, est + z_crit * se)
    stat = (log_est / log_se) if ratio else (est / se if se > 0 else np.nan)
    return CausalResult(
        method="Regression adjustment",
        estimand=estimand + _CONTRAST_LABEL[contrast],
        estimate=est,
        se=se,
        pvalue=float(2 * stats.norm.sf(abs(stat))),
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=None,
        model_info=info,
        _citation_key="g_computation",
    )


def _by_arm(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    *,
    estimand: str,
    ml_Q: Optional[Any],
    outcome_model: str = "linear",
    contrast: str = "difference",
    ps_covariates: Optional[List[str]],
    se_method: str,
    n_boot: int,
    alpha: float,
    seed: Optional[int],
) -> CausalResult:
    """``g_computation(by_arm=True)``: an outcome regression per arm."""
    from ._regression_adjustment import regression_adjustment

    if estimand not in ("ATE", "ATT") or ml_Q is not None:
        raise MethodIncompatibility(
            "g_computation(by_arm=True) is the linear regression adjustment "
            "for a binary treatment: estimand 'ATE' or 'ATT', no ml_Q.",
            recovery_hint="Drop by_arm=True for a dose-response curve or a "
            "machine-learning outcome model.",
        )
    ps = list(ps_covariates) if ps_covariates is not None else None
    columns = list(dict.fromkeys([y, treat, *covariates, *(ps or [])]))
    df = data[columns].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(dtype=float)
    D = _treatment_as_float(df[treat], function="g_computation")
    if not set(np.unique(D)).issubset({0.0, 1.0}):
        raise ValueError(
            f"estimand='{estimand}' requires binary treatment (0/1) in " f"'{treat}'."
        )
    n = len(Y)
    ones = np.ones((n, 1))
    X = np.column_stack([ones, df[covariates].to_numpy(dtype=float)])
    Z = None if ps is None else np.column_stack([ones, df[ps].to_numpy(dtype=float)])

    if outcome_model != "linear" or contrast != "difference":
        if ps is not None:
            raise MethodIncompatibility(
                "g_computation(by_arm=True): ps_covariates= (inverse-"
                "probability-weighted regression adjustment) is implemented "
                "for the linear outcome model and the difference of means.",
                recovery_hint="Drop ps_covariates=, or use sp.aipw for a "
                "doubly robust estimate with a binary outcome.",
            )
        return _by_arm_glm(
            Y,
            D,
            X,
            estimand=estimand,
            outcome_model=outcome_model,
            contrast=contrast,
            se_method=se_method,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )
    tau, se, extra = regression_adjustment(Y, D, X, estimand=estimand, Z=Z)
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    info = {
        "estimator": (
            "Inverse-probability-weighted regression adjustment"
            if ps is not None
            else "Regression adjustment (outcome regression per arm)"
        ),
        "estimand": estimand,
        "by_arm": True,
        "ps_covariates": ps,
        "se_method": se_method,
        "pomean0": extra["pomean0"],
        "pomean0_se": extra["pomean0_se"],
        "ml_Q": "OLS",
    }
    if se_method == "bootstrap":
        rng = np.random.default_rng(seed)
        draws = np.full(n_boot, np.nan)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                draws[b] = regression_adjustment(
                    Y[idx],
                    D[idx],
                    X[idx],
                    estimand=estimand,
                    Z=None if Z is None else Z[idx],
                )[0]
            except (MethodIncompatibility, np.linalg.LinAlgError, ValueError):
                continue  # a resample without support; counted below
        ok = int(np.isfinite(draws).sum())
        if ok < 2:
            raise RuntimeError(
                "g_computation(by_arm=True): the bootstrap failed on "
                f"{n_boot - ok} of {n_boot} resamples; use se_method='analytic'."
            )
        se = float(np.nanstd(draws, ddof=1))
        ci = (
            float(np.nanpercentile(draws, 100 * alpha / 2)),
            float(np.nanpercentile(draws, 100 * (1 - alpha / 2))),
        )
        info.update(n_boot=n_boot, n_boot_success=ok, n_boot_failed=n_boot - ok)
        info["pomean0_se"] = float("nan")
    else:
        ci = (tau - z_crit * se, tau + z_crit * se)
    pvalue = float(2 * stats.norm.sf(abs(tau / se))) if se > 0 else float("nan")
    return CausalResult(
        method=(
            "Regression adjustment (IPW-weighted)"
            if ps is not None
            else "Regression adjustment"
        ),
        estimand=estimand,
        estimate=tau,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=None,
        model_info=info,
        _citation_key="g_computation",
    )


# Citation
CausalResult._CITATIONS["g_computation"] = (
    "@article{robins1986new,\n"
    "  title={A new approach to causal inference in mortality studies "
    "with a sustained exposure period: application to control of the "
    "healthy worker survivor effect},\n"
    "  author={Robins, James},\n"
    "  journal={Mathematical Modelling},\n"
    "  volume={7},\n"
    "  number={9-12},\n"
    "  pages={1393--1512},\n"
    "  year={1986}\n"
    "}"
)
