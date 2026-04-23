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
1393-1512. [@robins1986approach]

Snowden, J.M., Rose, S. and Mortimer, K.M. (2011). "Implementation of
G-computation on a Simulated Data Set: Demonstration of a Causal
Inference Technique." *American Journal of Epidemiology*, 173(7),
731-738. [@snowden2011implementation]

Hernán, M.A. and Robins, J.M. (2020). *Causal Inference: What If.*
Chapman & Hall/CRC. Chapter 13.
"""

import warnings
from typing import Optional, List, Any, Sequence
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def g_computation(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    estimand: str = 'ATE',
    treat_values: Optional[Sequence[float]] = None,
    ml_Q: Optional[Any] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
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
    estimand : {'ATE', 'ATT', 'dose_response'}, default 'ATE'
        - ``'ATE'``: E[Q(1,X) - Q(0,X)] (requires binary D)
        - ``'ATT'``: E[Q(1,X) - Q(0,X) | D=1]
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
    >>> # Binary treatment ATE
    >>> sp.g_computation(df, y='wage', treat='trained',
    ...                  covariates=['age', 'edu', 'exp'])

    >>> # ATT
    >>> sp.g_computation(df, y='wage', treat='trained',
    ...                  covariates=['age', 'edu'], estimand='ATT')

    >>> # Dose-response at specified doses
    >>> sp.g_computation(df, y='bp', treat='dose',
    ...                  covariates=['age', 'bmi'],
    ...                  estimand='dose_response',
    ...                  treat_values=[0, 10, 20, 30])

    Notes
    -----
    Consistent when :math:`Q(d, X) = E[Y|D=d, X]` is correctly specified
    and positivity holds over the treatment levels of interest. Not
    doubly robust — see :func:`sp.aipw` or :func:`sp.tmle` for DR
    alternatives, or :func:`sp.dml` with ``model='plr'/'irm'`` for
    ML-based orthogonalization.
    """
    if estimand not in ('ATE', 'ATT', 'dose_response'):
        raise ValueError(
            f"estimand must be 'ATE', 'ATT', or 'dose_response'; "
            f"got '{estimand}'"
        )

    df = data[[y, treat] + list(covariates)].dropna().reset_index(drop=True)
    Y = df[y].values.astype(float)
    D = df[treat].values.astype(float)
    X = df[covariates].values.astype(float)
    n = len(Y)

    if estimand in ('ATE', 'ATT'):
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
            finite = [v for v in uniq
                      if not (isinstance(v, float) and np.isnan(v))]
            has_nan = len(finite) < len(uniq)
            vals = sorted(finite)
            preview = ', '.join(f'{v:.4g}' for v in vals[:5])
            tail = f', ... ({len(vals) - 5} more)' if len(vals) > 5 else ''
            nan_note = ' (plus NaN values present)' if has_nan else ''
            raise ValueError(
                f"estimand='{estimand}' requires binary treatment (0/1); "
                f"treatment has {len(vals)} unique values "
                f"[{preview}{tail}]{nan_note}. For multi-valued or "
                f"continuous D, use estimand='dose_response'."
            )
        grid = np.array([0.0, 1.0])
    else:
        if treat_values is None or len(treat_values) == 0:
            raise ValueError(
                "estimand='dose_response' requires 'treat_values' "
                "(sequence of dose levels to evaluate)."
            )
        grid = np.asarray(treat_values, dtype=float)

    def _fit_and_predict(Y_, D_, X_, grid_):
        """Fit Q(D,X) on (Y_, D_, X_), return n_obs × len(grid_) predictions."""
        design = np.column_stack([D_.reshape(-1, 1), X_])
        if ml_Q is None:
            import statsmodels.api as sm
            fit = sm.OLS(Y_, sm.add_constant(design)).fit()
            preds = np.empty((X_.shape[0], len(grid_)))
            for k, d_val in enumerate(grid_):
                d_col = np.full((X_.shape[0], 1), d_val)
                new_design = np.column_stack([d_col, X_])
                preds[:, k] = fit.predict(sm.add_constant(new_design, has_constant='add'))
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

    def _point_estimates(Y_, D_, X_):
        preds = _fit_and_predict(Y_, D_, X_, grid)
        if estimand == 'ATE':
            return np.array([float(np.mean(preds[:, 1] - preds[:, 0]))])
        if estimand == 'ATT':
            treated_mask = D_ == 1
            if treated_mask.sum() == 0:
                return np.array([np.nan])
            return np.array([
                float(np.mean(preds[treated_mask, 1] - preds[treated_mask, 0]))
            ])
        # dose_response
        return preds.mean(axis=0)

    point = _point_estimates(Y, D, X)

    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, len(point)), np.nan)
    n_failed = 0
    first_err: Optional[str] = None
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
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
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(se > 0, point / se, 0.0)
    pvalue = 2 * (1 - stats.norm.cdf(np.abs(z)))

    model_info = {
        'estimator': 'G-computation (parametric g-formula)',
        'estimand': estimand,
        'n_boot': n_boot,
        'n_boot_failed': n_failed,
        'n_boot_success': n_success,
        'ml_Q': type(ml_Q).__name__ if ml_Q is not None else 'OLS',
        'grid': grid.tolist(),
    }
    if n_failed > 0:
        model_info['first_bootstrap_error'] = first_err

    if estimand == 'dose_response':
        detail = pd.DataFrame({
            'dose': grid,
            'estimate': point,
            'se': se,
            'ci_lower': ci_lo,
            'ci_upper': ci_hi,
            'pvalue': pvalue,
        })
        model_info['curve'] = detail.copy()
        # Report first-dose point as the summary estimate slot
        summary_est = float(point[0])
        summary_se = float(se[0])
        summary_pv = float(pvalue[0])
        summary_ci = (float(ci_lo[0]), float(ci_hi[0]))
        label = 'E[Y(d)]'
    else:
        detail = None
        summary_est = float(point[0])
        summary_se = float(se[0])
        summary_pv = float(pvalue[0])
        summary_ci = (float(ci_lo[0]), float(ci_hi[0]))
        label = estimand

    return CausalResult(
        method='G-computation',
        estimand=label,
        estimate=summary_est,
        se=summary_se,
        pvalue=summary_pv,
        ci=summary_ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='g_computation',
    )


# Citation
CausalResult._CITATIONS['g_computation'] = (
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
