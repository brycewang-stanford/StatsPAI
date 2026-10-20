"""
Causal Impact: structural time-series intervention analysis.

Given a time series that experienced an intervention, estimates the causal
effect by forecasting a counterfactual from covariates (control series)
with a regression + AR(1) state-space model fit on the pre-intervention
period.

The model (frequentist, plug-in parameters):

    y_t = x_t' beta + s_t + eps_t,   eps_t ~ N(0, sigma_obs^2)
    s_t = rho * s_{t-1} + eta_t,     eta_t ~ N(0, sigma_state^2)

``beta`` is OLS on the pre-period, ``rho`` the lag-1 autocorrelation of the
OLS residuals, ``sigma_obs`` their standard deviation and
``sigma_state = sigma_obs * sqrt(1 - rho^2)``; the Kalman filter then
forecasts the post period. The filter step is pinned to ``KFAS::KFS`` in
``tests/reference_parity/test_did_synth_misc_parity.py``. The standard
errors add parameter uncertainty (``_uncertainty.py``).

This is NOT the model of R ``CausalImpact`` (Brodersen et al., 2015), which
fits a Bayesian local-level + spike-and-slab regression by MCMC on
standardised data. The estimand -- the average post-period gap between
the observed and the counterfactual series -- is the same, but the
counterfactual, and above all the uncertainty, come from a different model,
so the two do not agree numerically and are not expected to.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility


def causal_impact(
    data: pd.DataFrame,
    y: str,
    time: str,
    intervention_time: Any,
    covariates: Optional[List[str]] = None,
    alpha: float = 0.05,
    n_seasons: Optional[int] = None,
    weights: Optional[str] = None,
) -> CausalResult:
    """
    Estimate the causal impact of an intervention on a time series.

    Parameters
    ----------
    data : pd.DataFrame
        Time-series data (one row per time period).
    y : str
        Outcome variable (the series that was intervened upon).
    time : str
        Time column (used for ordering; can be int, date, etc.).
    intervention_time : any
        First period of the intervention (inclusive).
    covariates : list of str, optional
        Control time series (not affected by the intervention).
        If None, uses a local-level model without covariates.
    alpha : float, default 0.05
        Significance level for the (frequentist) prediction intervals.
    n_seasons : int, optional
        Accepted for API compatibility but currently not used: the model
        has no seasonal component.
    weights : None
        Not supported, and refused with ``MethodIncompatibility`` rather
        than ignored: the data are one series, one row per period, so
        there is no sample of units for sampling weights to describe, and
        the forecast-error variance has no weighted form. To weight the
        units that make up an aggregate series, build the series as a
        weighted mean first.

    Returns
    -------
    CausalResult

    Notes
    -----
    ``estimate`` is the average post-period effect and ``se`` the standard
    deviation of the error of the forecast actually made,
    ``u_post - L u_pre``, which accounts for estimating the regression
    coefficients exactly; the error covariance is an AR(1)-plus-noise model
    fitted by REML on the pre-period, and intervals and p-values use a t
    distribution with ``n_pre - p - 3`` degrees of freedom
    (``model_info['df']``). Uncertainty in that covariance is not
    propagated, so intervals stay somewhat short with few pre-periods and
    a persistent error (89.5% for a nominal 95% at n_pre = 40, rho = 0.9;
    92%-94% in the other designs tried). Until 2026-10-19 the SE took all
    parameters as known and covered 57.5%-90.5% in the same designs; that
    number is kept as ``model_info['se_avg_conditional']``. The
    ``detail`` column ``effect_se`` carries the pointwise version
    (``predicted_se`` stays the filter's conditional one-step value).
    This is a frequentist regression +
    AR(1) state-space model, not the Bayesian local-level + spike-and-slab
    model of R ``CausalImpact``; the numbers are not comparable one for one.

    Examples
    --------
    Effect of an intervention introduced at week 40, recovered against a
    counterfactual built from an ad-spend control series:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 60
    >>> week = np.arange(n)
    >>> ad_spend = rng.normal(100, 5, size=n)
    >>> sales = 2.0 * ad_spend + rng.normal(0, 3, size=n)
    >>> sales[40:] += 15.0          # intervention lifts sales from week 40
    >>> df = pd.DataFrame({'week': week, 'sales': sales,
    ...                    'ad_spend': ad_spend})
    >>> result = sp.causal_impact(df, y='sales', time='week',
    ...                           intervention_time=40,
    ...                           covariates=['ad_spend'])
    >>> result.n_obs
    60
    >>> bool(result.estimate > 0)        # positive average impact recovered
    True

    References
    ----------
    brodersen2015inferring
    """
    if weights is not None:
        raise MethodIncompatibility(
            "causal_impact: weights= is not supported -- the data are a single "
            "time series (one row per period), so sampling weights have "
            "nothing to weight and the forecast-error variance has no "
            "weighted form.",
            recovery_hint="Aggregate the units into the outcome series with "
            "their weights first (a weighted mean per period), then call "
            "causal_impact on that series.",
            diagnostics={"weights": weights},
        )
    estimator = CausalImpactEstimator(
        data=data,
        y=y,
        time=time,
        intervention_time=intervention_time,
        covariates=covariates,
        alpha=alpha,
        n_seasons=n_seasons,
    )
    _result = estimator.fit()
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.causal_impact",
            params={
                "y": y,
                "time": time,
                "intervention_time": intervention_time,
                "covariates": list(covariates) if covariates else None,
                "alpha": alpha,
                "n_seasons": n_seasons,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


class CausalImpactEstimator:
    """
    Causal Impact estimator using a structural time-series model.

    Fits a regression + AR(1) state-space model on the
    pre-intervention period, then forecasts the counterfactual into the
    post-intervention period. The high-level :func:`causal_impact`
    wrapper is the usual entry point; instantiate this class directly
    only when you want to drive ``.fit()`` yourself.

    Parameters
    ----------
    data : pandas.DataFrame
        Time-series data, one row per period.
    y : str
        Outcome (intervened) series.
    time : str
        Time column used for ordering.
    intervention_time : Any
        First post-intervention period (inclusive).
    covariates : list of str, optional
        Control series unaffected by the intervention.
    alpha : float, default 0.05
        Significance level for the credible intervals.
    n_seasons : int, optional
        Not used (no seasonal component); a value > 1 warns.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 60
    >>> week = np.arange(n)
    >>> ad_spend = rng.normal(100, 5, size=n)
    >>> sales = 2.0 * ad_spend + rng.normal(0, 3, size=n)
    >>> sales[40:] += 15.0
    >>> df = pd.DataFrame({'week': week, 'sales': sales,
    ...                    'ad_spend': ad_spend})
    >>> est = sp.CausalImpactEstimator(df, y='sales', time='week',
    ...                                intervention_time=40,
    ...                                covariates=['ad_spend'])
    >>> result = est.fit()
    >>> (est.n_pre, est.n_post)
    (40, 20)
    >>> bool(result.estimate > 0)
    True
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        time: str,
        intervention_time: Any,
        covariates: Optional[List[str]] = None,
        alpha: float = 0.05,
        n_seasons: Optional[int] = None,
    ) -> None:
        self.data = data.sort_values(time).copy()
        self.y = y
        self.time = time
        self.intervention_time = intervention_time
        self.covariates = covariates or []
        self.alpha = alpha
        self.n_seasons = n_seasons
        if n_seasons is not None and int(n_seasons) > 1:
            # Fail loudly: the argument used to be accepted and silently
            # ignored.
            warnings.warn(
                f"causal_impact: n_seasons={n_seasons} is ignored -- the "
                "model has no seasonal component. Deseasonalise the series "
                "or include seasonal dummies / control series as covariates.",
                UserWarning,
                stacklevel=3,
            )

        self._validate()
        self._prepare()

    def _validate(self) -> None:
        check_columns(
            self.data,
            {"y": self.y, "time": self.time, "covariates": list(self.covariates)},
            function="sp.causal_impact",
        )

        times = self.data[self.time].values
        if self.data[self.time].isna().any() or self.data[self.time].duplicated().any():
            raise MethodIncompatibility(
                "sp.causal_impact: the time column must be unique and complete "
                "(one row per period).",
                recovery_hint="Aggregate to one row per period first.",
            )
        cols = [self.y] + list(self.covariates)
        bad = [c for c in cols if not pd.api.types.is_numeric_dtype(self.data[c])]
        if bad:
            raise MethodIncompatibility(
                f"sp.causal_impact: {bad} must be numeric.",
                recovery_hint="Convert or drop those columns.",
            )
        miss = [c for c in cols if self.data[c].isna().any()]
        if miss:
            # NaN in the outcome made the effect NaN without a word; in a
            # covariate it surfaced as LinAlgError (2026-10-20 probe).
            raise MethodIncompatibility(
                f"sp.causal_impact: missing values in {miss}.",
                recovery_hint="Interpolate or drop the periods before calling; "
                "the state-space fit here needs a complete series.",
            )
        try:
            inside = times.min() < self.intervention_time <= times.max()
        except TypeError as exc:
            raise MethodIncompatibility(
                f"sp.causal_impact: intervention_time={self.intervention_time!r} "
                f"cannot be compared with the time column ({times.dtype}).",
                recovery_hint="Pass a value of the time column's type.",
            ) from exc
        if self.intervention_time not in times:
            # Check if intervention_time is between min and max
            if not inside:
                raise ValueError(
                    f"intervention_time={self.intervention_time} outside data range "
                    f"[{times.min()}, {times.max()}]"
                )

    def _prepare(self) -> None:
        """Split data into pre and post periods."""
        self.times = self.data[self.time].values
        self.Y = self.data[self.y].values.astype(float)

        self.pre_mask = self.times < self.intervention_time
        self.post_mask = self.times >= self.intervention_time

        self.n_pre = int(self.pre_mask.sum())
        self.n_post = int(self.post_mask.sum())

        if self.n_pre < 3:
            raise ValueError(
                f"Need at least 3 pre-intervention periods, got {self.n_pre}"
            )
        if self.n_post < 1:
            raise ValueError("Need at least 1 post-intervention period")

        if self.covariates:
            self.X = self.data[self.covariates].values.astype(float)
            Xa = np.column_stack([np.ones(self.n_pre), self.X[self.pre_mask]])
            if np.linalg.matrix_rank(Xa) < Xa.shape[1]:
                raise MethodIncompatibility(
                    "sp.causal_impact: the covariates are collinear (with each "
                    "other or the constant) over the pre-period.",
                    recovery_hint="Drop the redundant covariate(s).",
                )
        else:
            self.X = None

    def fit(self) -> CausalResult:
        """Fit the structural time-series model and estimate causal impact."""
        Y_pre = self.Y[self.pre_mask]
        X_pre = self.X[self.pre_mask] if self.X is not None else None

        # Fit model on pre-intervention period
        model = self._fit_structural_model(Y_pre, X_pre)

        # Generate counterfactual predictions for full period
        Y_pred, Y_pred_se = self._predict(model, self.X)

        # Point estimates
        Y_actual = self.Y
        pointwise_effect = Y_actual - Y_pred

        # Cumulative effect (post-period only)
        post_effect = pointwise_effect[self.post_mask]
        cumulative_effect = np.cumsum(post_effect)

        # Summary statistics
        avg_effect = float(np.mean(post_effect))
        total_effect = float(np.sum(post_effect))

        # Standard error of the average / cumulative effect. The post-period
        # forecast errors share the latent state, so they are correlated:
        # for post periods t < u, Cov(e_t, e_u) = rho^(u - t) * P_t, with
        # P_t the forecast variance of the state at t; the diagonal adds
        # the observation noise. (Up to StatsPAI 1.28.0 the errors were
        # treated as independent, which understates the SE when rho > 0.)
        cov_post_cond = self._post_forecast_cov(model)
        se_total_cond = float(np.sqrt(max(cov_post_cond.sum(), 0.0)))

        # Parameter uncertainty (2026-10-19): the conditional covariance
        # above treats beta and the plug-in variances as known, and its 95%
        # intervals covered 57%-90% of null effects. The error of the
        # forecast actually made is u_post - L u_pre (beta cancels); its
        # variance under a REML-fitted AR(1)+noise error covariance is what
        # the interval now uses. See _uncertainty.py.
        cov_post, error_model, se_method = self._post_error_cov(model, Y_pre, X_pre)
        se_total = float(np.sqrt(max(cov_post.sum(), 0.0)))
        se_avg = se_total / self.n_post
        n_beta = 1 + (self.X.shape[1] if self.X is not None else 0)
        df_t = max(self.n_pre - n_beta - 3, 1)
        crit = float(stats.t.ppf(1 - self.alpha / 2, df_t))

        point_se = Y_pred_se.copy()
        point_se[self.post_mask] = np.sqrt(np.clip(np.diag(cov_post), 0.0, None))
        pointwise_lower = pointwise_effect - crit * point_se
        pointwise_upper = pointwise_effect + crit * point_se

        # Relative effect
        post_pred_mean = float(np.mean(Y_pred[self.post_mask]))
        relative_effect = (
            avg_effect / post_pred_mean if abs(post_pred_mean) > 1e-10 else np.nan
        )

        # P-value (two-sided t test against zero effect)
        t_stat = avg_effect / se_avg if se_avg > 0 else 0.0
        pvalue = float(2 * stats.t.sf(abs(t_stat), df_t))
        ci = (avg_effect - crit * se_avg, avg_effect + crit * se_avg)

        # Build detail table (time series)
        detail = pd.DataFrame(
            {
                "time": self.times,
                "actual": Y_actual,
                "predicted": Y_pred,
                "predicted_se": Y_pred_se,
                "effect": pointwise_effect,
                "effect_se": point_se,
                "effect_lower": pointwise_lower,
                "effect_upper": pointwise_upper,
                "post_intervention": self.post_mask,
            }
        )

        model_info: Dict[str, Any] = {
            "intervention_time": self.intervention_time,
            "n_pre": self.n_pre,
            "n_post": self.n_post,
            "avg_effect": avg_effect,
            "total_effect": total_effect,
            "se_avg": se_avg,
            "se_total": se_total,
            "se_avg_conditional": se_total_cond / self.n_post,
            "se_total_conditional": se_total_cond,
            "se_method": se_method,
            "error_model": error_model,
            "df": df_t,
            "relative_effect": relative_effect,
            "relative_effect_pct": (
                relative_effect * 100 if not np.isnan(relative_effect) else np.nan
            ),
            "n_covariates": len(self.covariates),
            "model_params": model,
            "Y_pred": Y_pred,
            "Y_actual": Y_actual,
            "times": self.times,
            "pre_mask": self.pre_mask,
            "post_mask": self.post_mask,
            "cumulative_effect": cumulative_effect,
        }

        return CausalResult(
            method="Causal Impact (Structural Time Series)",
            estimand="Average Causal Effect",
            estimate=avg_effect,
            se=se_avg,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=len(self.Y),
            detail=detail,
            model_info=model_info,
            _citation_key="causal_impact",
        )

    # ------------------------------------------------------------------
    # Structural time-series model
    # ------------------------------------------------------------------

    def _fit_structural_model(
        self,
        Y_pre: np.ndarray,
        X_pre: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Fit the regression + AR(1)-state model on pre-intervention data.

        Model: Y_t = beta' * X_t + s_t + eps_t
               s_t = rho * s_{t-1} + eta_t

        Returns dict with fitted parameters.
        """
        n = len(Y_pre)

        if X_pre is not None and X_pre.shape[1] > 0:
            # Add intercept
            X_aug = np.column_stack([np.ones(n), X_pre])
            # OLS for regression coefficients
            try:
                beta = np.linalg.lstsq(X_aug, Y_pre, rcond=None)[0]
            except np.linalg.LinAlgError:
                beta = np.zeros(X_aug.shape[1])
            residuals = Y_pre - X_aug @ beta
        else:
            X_aug = np.ones((n, 1))
            beta = np.array([np.mean(Y_pre)])
            residuals = Y_pre - beta[0]

        # Estimate AR(1) on residuals for local level dynamics
        if n > 2:
            r_lag = residuals[:-1]
            r_lead = residuals[1:]
            if np.var(r_lag) > 1e-10:
                rho = float(np.corrcoef(r_lag, r_lead)[0, 1])
                rho = float(np.clip(rho, -0.99, 0.99))
            else:
                rho = 0.0
        else:
            rho = 0.0

        # Variance decomposition
        sigma_obs = float(np.std(residuals, ddof=1)) if n > 1 else 1.0
        sigma_state = (
            sigma_obs * np.sqrt(1 - rho**2) if abs(rho) < 1 else sigma_obs * 0.1
        )

        return {
            "beta": beta,
            "rho": rho,
            "sigma_obs": max(sigma_obs, 1e-10),
            "sigma_state": max(sigma_state, 1e-10),
            "last_residual": residuals[-1] if len(residuals) > 0 else 0.0,
            "n_covariates": X_pre.shape[1] if X_pre is not None else 0,
        }

    #: above this many periods the dense n x n error covariance is skipped
    _DENSE_MAX = 5000

    def _post_error_cov(
        self,
        model: Dict[str, Any],
        Y_pre: np.ndarray,
        X_pre: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[Dict[str, float]], str]:
        """Post-period forecast-error covariance with parameter uncertainty."""
        from ._uncertainty import (
            error_covariance,
            forecast_error_cov,
            predictor_map,
            reml_ar1_noise,
        )

        n = len(self.Y)
        if n > self._DENSE_MAX:
            from ..exceptions import AssumptionWarning

            warnings.warn(
                AssumptionWarning(
                    f"causal_impact: {n} periods exceed the {self._DENSE_MAX} "
                    "for which the forecast-error covariance with parameter "
                    "uncertainty is formed; the interval is conditional on "
                    "the fitted parameters and will be too narrow.",
                    recovery_hint="Aggregate to a coarser frequency.",
                    diagnostics={"n_periods": n},
                ),
                stacklevel=4,
            )
            return self._post_forecast_cov(model), None, "conditional"
        Xo = (
            np.column_stack([np.ones(len(Y_pre)), X_pre])
            if X_pre is not None and X_pre.shape[1] > 0
            else np.ones((len(Y_pre), 1))
        )
        X_aug = (
            np.column_stack([np.ones(n), self.X])
            if self.X is not None and self.X.shape[1] > 0
            else np.ones((n, 1))
        )
        err = reml_ar1_noise(Y_pre, Xo)
        L = predictor_map(X_aug, self.pre_mask, model)
        V = forecast_error_cov(L, error_covariance(n, err), self.pre_mask)
        return V, err, "reml_error_model"

    def _post_forecast_cov(self, model: Dict[str, Any]) -> np.ndarray:
        """Joint covariance of the post-period forecast errors.

        Under the fitted state space (state ``s_{t+1} = rho s_t + eta``,
        observation ``y_t = x_t'beta + s_t + eps``) with no updates after
        the intervention, ``s_u - a_u = rho^(u-t) (s_t - a_t) + noise`` for
        ``u > t``, so ``Cov(e_t, e_u) = rho^(u-t) P_t`` and
        ``Var(e_t) = P_t + sigma_obs^2``. Parameter uncertainty (beta, rho,
        the variances) is not propagated; this conditional variance is kept
        for reference (``se_total_conditional``).
        """
        P = np.asarray(model["_post_state_var"], dtype=float)
        rho = float(model["rho"])
        sigma_obs = float(model["sigma_obs"])
        m = len(P)
        idx = np.arange(m)
        lag = np.abs(idx[:, None] - idx[None, :])
        first = np.minimum(idx[:, None], idx[None, :])
        cov = (rho**lag) * P[first]
        cov[idx, idx] += sigma_obs**2
        return cov

    def _predict(
        self,
        model: Dict[str, Any],
        X_full: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate counterfactual predictions for all time periods.

        Returns (Y_pred, Y_pred_se) arrays.
        """
        n = len(self.Y)
        beta = np.asarray(model["beta"], dtype=float)
        rho = float(model["rho"])
        sigma_obs = float(model["sigma_obs"])
        sigma_state = float(model["sigma_state"])

        if X_full is not None and X_full.shape[1] > 0:
            X_aug = np.column_stack([np.ones(n), X_full])
        else:
            X_aug = np.ones((n, 1))

        # Regression component
        Y_reg = X_aug @ beta

        # Local level (state) component: AR(1) on residuals
        Y_pred = np.zeros(n)
        Y_pred_se = np.zeros(n)

        state = 0.0
        state_var = sigma_state**2
        post_state_var = []

        for t in range(n):
            Y_pred[t] = Y_reg[t] + state
            Y_pred_se[t] = np.sqrt(sigma_obs**2 + state_var)
            if not self.pre_mask[t]:
                post_state_var.append(state_var)

            if self.pre_mask[t]:
                # Update state with observed data (Kalman-like)
                innovation = self.Y[t] - Y_pred[t]
                K = state_var / (state_var + sigma_obs**2)  # Kalman gain
                state = rho * (state + K * innovation)
                state_var = rho**2 * (1 - K) * state_var + sigma_state**2
            else:
                # Post-period: propagate uncertainty (no update)
                state = rho * state
                state_var = rho**2 * state_var + sigma_state**2

        model["_post_state_var"] = np.asarray(post_state_var, dtype=float)
        return Y_pred, Y_pred_se


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------


def impactplot(
    result: CausalResult,
    type: str = "all",
    ax: Any = None,
    figsize: tuple[float, float] = (12, 9),
    title: Optional[str] = None,
) -> Any:
    """
    Causal Impact visualization (Google-style 3-panel plot).

    Parameters
    ----------
    result : CausalResult
        Result from ``causal_impact()``.
    type : str, default 'all'
        'all': 3-panel (original + pointwise + cumulative).
        'original': actual vs counterfactual.
        'pointwise': pointwise causal effect.
        'cumulative': cumulative effect.
    ax : matplotlib Axes, optional
        Only for single-panel types.
    figsize : tuple
    title : str, optional

    Returns
    -------
    (fig, ax) or (fig, axes)

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib
    >>> matplotlib.use('Agg')
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 60
    >>> week = np.arange(n)
    >>> ad_spend = rng.normal(100, 5, size=n)
    >>> sales = 2.0 * ad_spend + rng.normal(0, 3, size=n)
    >>> sales[40:] += 15.0
    >>> df = pd.DataFrame({'week': week, 'sales': sales,
    ...                    'ad_spend': ad_spend})
    >>> result = sp.causal_impact(df, y='sales', time='week',
    ...                           intervention_time=40,
    ...                           covariates=['ad_spend'])
    >>> fig, axes = sp.impactplot(result, type='all')
    >>> len(axes)
    3
    >>> fig.savefig('impact.png')  # doctest: +SKIP
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    mi = result.model_info
    detail = result.detail
    if detail is None or "actual" not in detail.columns:
        raise ValueError("No detail table. Use causal_impact() result.")

    times = detail["time"].values
    actual = detail["actual"].values
    predicted = detail["predicted"].values
    effect = detail["effect"].values
    eff_lo = detail["effect_lower"].values
    eff_hi = detail["effect_upper"].values
    post = detail["post_intervention"].values
    intervention_time = mi.get("intervention_time")
    cum_raw = mi.get("cumulative_effect")
    if cum_raw is not None and len(cum_raw) < len(times):
        # cumulative_effect only covers post-period; pad pre with zeros
        cumulative = np.zeros(len(times))
        cumulative[post] = cum_raw
    elif cum_raw is not None:
        cumulative = cum_raw
    else:
        cumulative = np.cumsum(effect * post)

    if type == "all":
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        _original_panel(
            axes[0], times, actual, predicted, intervention_time, result.alpha
        )
        _pointwise_panel(
            axes[1], times, effect, eff_lo, eff_hi, post, intervention_time
        )
        _cumulative_panel(axes[2], times, cumulative, post, intervention_time)
        fig.suptitle(title or "Causal Impact Analysis", fontsize=14, y=1.01)
        fig.tight_layout()
        return fig, axes

    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] // 3 + 1))
    else:
        fig = ax.get_figure()

    if type == "pointwise":
        _pointwise_panel(ax, times, effect, eff_lo, eff_hi, post, intervention_time)
    elif type == "cumulative":
        _cumulative_panel(ax, times, cumulative, post, intervention_time)
    else:
        _original_panel(ax, times, actual, predicted, intervention_time, result.alpha)

    ax.set_title(title or f"Causal Impact: {type}", fontsize=13)
    fig.tight_layout()
    return fig, ax


def _original_panel(
    ax: Any,
    times: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    t0: Any,
    alpha: float,
) -> None:
    ax.plot(times, actual, color="#2C3E50", linewidth=1.5, label="Observed")
    ax.plot(
        times,
        predicted,
        color="#3498DB",
        linewidth=1.5,
        linestyle="--",
        label="Counterfactual",
    )
    if t0 is not None:
        ax.axvline(
            x=t0,
            color="#E74C3C",
            linestyle=":",
            linewidth=1,
            alpha=0.7,
            label="Intervention",
        )
    ax.set_ylabel("Value", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _pointwise_panel(
    ax: Any,
    times: np.ndarray,
    effect: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    post: np.ndarray,
    t0: Any,
) -> None:
    ax.plot(times, effect, color="#2C3E50", linewidth=1.5)
    ax.fill_between(times, lo, hi, alpha=0.15, color="#3498DB")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    if t0 is not None:
        ax.axvline(x=t0, color="#E74C3C", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_ylabel("Pointwise Effect", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _cumulative_panel(
    ax: Any,
    times: np.ndarray,
    cumulative: np.ndarray,
    post: np.ndarray,
    t0: Any,
) -> None:
    ax.plot(times, cumulative, color="#2C3E50", linewidth=1.5)
    ax.fill_between(times, 0, cumulative, alpha=0.15, color="#3498DB")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    if t0 is not None:
        ax.axvline(x=t0, color="#E74C3C", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Cumulative Effect", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Citation
CausalResult._CITATIONS["causal_impact"] = (
    "@article{brodersen2015inferring,\n"
    "  title={Inferring Causal Impact Using Bayesian Structural "
    "Time-Series Models},\n"
    "  author={Brodersen, Kay H and Gallusser, Fabian and "
    "Koehler, Jim and Remy, Nicolas and Scott, Steven L},\n"
    "  journal={The Annals of Applied Statistics},\n"
    "  volume={9},\n"
    "  number={1},\n"
    "  pages={247--274},\n"
    "  year={2015},\n"
    "  publisher={Institute of Mathematical Statistics}\n"
    "}"
)
